"""Run one (model, dataset, mode, seed) evaluation and write one parquet file.

    python -m mcqbias.run_eval --config configs/<run>.yaml
    python -m mcqbias.run_eval --model Qwen/Qwen2.5-3B --dataset mmlu \
        --mode full16 --n-questions 100 --seed 0

One row per (question, version), with the schema CLAUDE.md specifies. Anything
over ten minutes belongs in tmux, tee'd to a log on the volume (README).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from mcqbias import data, prompts, rotate, score

REPO_ROOT = Path(__file__).resolve().parents[2]
LABELS = "ABCD"


@dataclass
class Config:
    model: str
    dataset: str = "mmlu"
    mode: str = "balanced1"
    seed: int = 0
    n_questions: int | None = None  # None = the whole filtered population
    batch_size: int = 8
    dtype: str = "bfloat16"  # upstream picks bf16 on an A100; recorded per run
    device_map: str = "auto"
    out_dir: str = "results/raw"
    evict_after: bool = False  # delete this model's HF snapshot when done


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        dirty = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def select_questions(questions, n: int | None, seed: int):
    """A seeded sample, kept in dataset order so qids stay comparable across seeds."""
    if n is None or n >= len(questions):
        return questions
    import random

    chosen = sorted(random.Random(seed).sample(range(len(questions)), n))
    return [questions[i] for i in chosen]


def run(cfg: Config) -> Path:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    questions, dropped = data.load(cfg.dataset)
    print(f"[run] {cfg.dataset}: {len(questions)} questions kept, {dropped} dropped by the filter")
    questions = select_questions(questions, cfg.n_questions, cfg.seed)

    plan = rotate.versions_for(cfg.mode, len(questions), cfg.seed)
    work = [
        (q, rotate.render(q.options, q.gold_index, *rotate.shifts(v)))
        for q, versions in zip(questions, plan)
        for v in versions
    ]
    print(f"[run] {len(questions)} questions x {cfg.mode} = {len(work)} forward passes")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    tokens = score.label_tokens(tokenizer)
    print(f"[run] label ids {tokens.ids}, space-prefixed variants distinct: {tokens.has_space_prefix}")

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model, torch_dtype=getattr(torch, cfg.dtype), device_map=cfg.device_map
    )
    model.eval()
    revision = getattr(model.config, "_commit_hash", None) or "unknown"
    print(f"[run] {cfg.model} @ {revision}, {cfg.dtype}, device {model.device}")

    rows = []
    started = time.time()
    for start in range(0, len(work), cfg.batch_size):
        chunk = work[start : start + cfg.batch_size]
        texts = [
            prompts.build(cfg.dataset, q.subject, q.question, r.contents, r.label_order)
            for q, r in chunk
        ]
        probs = score.score(model, tokenizer, texts, tokens)
        for (q, r), p in zip(chunk, probs):
            pred_label = LABELS[int(p.argmax())]
            rows.append(
                {
                    "model": cfg.model,
                    "dataset": cfg.dataset,
                    "subject": q.subject,
                    "qid": q.qid,
                    "version": r.version,
                    "label_shift": r.label_shift,
                    "content_shift": r.content_shift,
                    "label_order": r.label_order,
                    "gold_pos": r.gold_pos,
                    "gold_label": r.gold_label,
                    "pred_pos": r.label_order.index(pred_label),
                    "pred_label": pred_label,
                    "correct": pred_label == r.gold_label,
                    **{f"p_{lab}": float(p[i]) for i, lab in enumerate(LABELS)},
                    "seed": cfg.seed,
                    "mode": cfg.mode,
                }
            )
        if start % (cfg.batch_size * 50) == 0 and start:
            rate = start / (time.time() - started)
            print(f"[run] {start}/{len(work)}  {rate:.1f} passes/s  "
                  f"eta {(len(work) - start) / rate / 60:.1f} min", flush=True)

    elapsed = time.time() - started
    run_id = f"{Path(cfg.model).name}_{cfg.dataset}_{cfg.mode}_s{cfg.seed}"
    frame = pd.DataFrame(rows).assign(
        run_id=run_id,
        git_commit=git_commit(),
        model_revision=revision,
        dtype=cfg.dtype,
        batch_size=cfg.batch_size,
    )

    out_dir = REPO_ROOT / cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.parquet"
    frame.to_parquet(out_path, index=False)

    print(f"[run] accuracy {frame.correct.mean() * 100:.2f} over {len(frame)} rows")
    print(f"[run] {elapsed / 60:.1f} min, {len(work) / elapsed:.1f} passes/s, "
          f"{elapsed / len(work) * 1000:.1f} ms/pass")
    print(f"[run] wrote {out_path}")

    if cfg.evict_after:
        evict(cfg.model)
    return out_path


def evict(model_id: str) -> None:
    """Drop this model's snapshot. mcq-team-vol is 100 GB; twelve 7B models are 168 GB."""
    from huggingface_hub.constants import HF_HUB_CACHE

    folder = Path(HF_HUB_CACHE) / f"models--{model_id.replace('/', '--')}"
    if folder.is_dir():
        shutil.rmtree(folder)
        print(f"[run] evicted {folder}")


def parse_args(argv=None) -> Config:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, help="YAML with any of the fields below")
    p.add_argument("--model")
    p.add_argument("--dataset")
    p.add_argument("--mode", choices=("full16", "balanced1", "default"))
    p.add_argument("--seed", type=int)
    p.add_argument("--n-questions", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--dtype")
    p.add_argument("--device-map")
    p.add_argument("--out-dir")
    p.add_argument("--evict-after", action="store_true", default=None)
    args = p.parse_args(argv)

    fields = {}
    if args.config:
        import yaml

        fields.update(yaml.safe_load(args.config.read_text()) or {})
    fields.update(
        {
            k: v
            for k, v in vars(args).items()
            if k != "config" and v is not None
        }
    )
    if "model" not in fields:
        p.error("--model is required (or set it in --config)")
    return Config(**fields)


if __name__ == "__main__":
    run(parse_args())
