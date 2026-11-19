"""Run one (model, dataset, mode, seed) evaluation and write one parquet file.

    python -m mcqbias.run_eval --config configs/<run>.yaml
    python -m mcqbias.run_eval --model Qwen/Qwen2.5-3B --dataset mmlu \
        --mode full16 --n-questions 100 --seed 0

One row per (question, version), with the schema CLAUDE.md specifies.

Three things here exist because the instance is billed by the hour and a long
run that dies has to not cost a second one:

* Results are flushed to shard files as they are produced, not held in memory
  until the end. A 68k-pass run that dies at 60k keeps its 60k.
* Resume is keyed on (qid, version) read back from those shards, not on a row
  count, so it stays correct even if the work order ever changes.
* An out-of-memory error halves the batch and retries rather than killing the
  run, which is what upstream does too (utils.py:185-192).

For the whole 12-model sweep use scripts/run_sweep.py, which loads each model
once and reuses it across that model's runs.
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
    flush_every: int = 2000  # rows per shard; ~2-3 min of work at 7B
    dtype: str = "bfloat16"  # upstream picks bf16 on an A100; recorded per run
    device_map: str = "auto"
    out_dir: str = "results/raw"
    evict_after: bool = False  # delete this model's HF snapshot when done

    @property
    def run_id(self) -> str:
        return f"{Path(self.model).name}_{self.dataset}_{self.mode}_s{self.seed}"


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


def plan_work(cfg: Config):
    """Every (question, version) this run must score, in a deterministic order."""
    questions, dropped = data.load(cfg.dataset)
    questions = select_questions(questions, cfg.n_questions, cfg.seed)
    versions = rotate.versions_for(cfg.mode, len(questions), cfg.seed)
    work = [
        (q, rotate.render(q.options, q.gold_index, *rotate.shifts(v)))
        for q, vs in zip(questions, versions)
        for v in vs
    ]
    return work, len(questions), dropped


# ---------------------------------------------------------------- shards

def _shard_paths(out_dir: Path, run_id: str) -> list[Path]:
    return sorted(out_dir.glob(f"{run_id}.part-*.parquet"))


def completed_keys(out_dir: Path, run_id: str) -> set[tuple[str, int]]:
    """(qid, version) pairs already on disk, so a resumed run redoes nothing."""
    done: set[tuple[str, int]] = set()
    for path in _shard_paths(out_dir, run_id):
        try:
            shard = pd.read_parquet(path, columns=["qid", "version"])
        except Exception as exc:  # a shard truncated by a hard kill
            print(f"[run] discarding unreadable shard {path.name}: {exc}")
            path.unlink()
            continue
        done.update(zip(shard.qid, shard.version))
    return done


def _next_shard_index(out_dir: Path, run_id: str) -> int:
    """One past the highest index present — NOT the count.

    Counting breaks as soon as the indices are not contiguous, which happens
    whenever a shard is discarded as unreadable or a resumed run starts from an
    existing set: the count would point back at a shard that already exists and
    every later flush would overwrite the same file, silently discarding most of
    the run's output.
    """
    indices = [int(p.name.split(".part-")[1].split(".")[0]) for p in _shard_paths(out_dir, run_id)]
    return max(indices) + 1 if indices else 0


def _flush(rows: list[dict], out_dir: Path, run_id: str, meta: dict) -> None:
    if not rows:
        return
    index = _next_shard_index(out_dir, run_id)
    path = out_dir / f"{run_id}.part-{index:05d}.parquet"
    tmp = path.with_suffix(".parquet.tmp")
    # Write then rename: a kill mid-write leaves the .tmp, never a half shard
    # that completed_keys would trust.
    pd.DataFrame(rows).assign(**meta).to_parquet(tmp, index=False)
    tmp.rename(path)
    rows.clear()


def consolidate(out_dir: Path, run_id: str) -> Path:
    """Merge shards into the single run file and drop them."""
    shards = _shard_paths(out_dir, run_id)
    out_path = out_dir / f"{run_id}.parquet"
    frame = pd.concat([pd.read_parquet(p) for p in shards], ignore_index=True)
    frame = frame.drop_duplicates(["qid", "version"]).sort_values(["qid", "version"])
    frame.to_parquet(out_path, index=False)
    for shard in shards:
        shard.unlink()
    return out_path


# ---------------------------------------------------------------- model

def load_model(cfg: Config):
    """Returns (model, tokenizer, label tokens, revision). Separated so the sweep
    driver can pay the load cost once per model instead of once per run."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model)
    tokens = score.label_tokens(tokenizer)
    print(f"[run] label ids {tokens.ids}, "
          f"space-prefixed variants distinct: {tokens.has_space_prefix}")

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model, torch_dtype=getattr(torch, cfg.dtype), device_map=cfg.device_map
    )
    model.eval()
    revision = getattr(model.config, "_commit_hash", None) or "unknown"
    print(f"[run] {cfg.model} @ {revision}, {cfg.dtype}, device {model.device}")
    return model, tokenizer, tokens, revision


def _is_oom(exc: BaseException) -> bool:
    return isinstance(exc, (RuntimeError, MemoryError)) and "out of memory" in str(exc).lower()


# ---------------------------------------------------------------- run

def run(cfg: Config, loaded=None) -> Path:
    out_dir = REPO_ROOT / cfg.out_dir if not Path(cfg.out_dir).is_absolute() else Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / f"{cfg.run_id}.parquet"
    if final.exists():
        print(f"[run] {cfg.run_id}: already complete, skipping")
        return final

    work, n_questions, dropped = plan_work(cfg)
    done = completed_keys(out_dir, cfg.run_id)
    pending = [(q, r) for q, r in work if (q.qid, r.version) not in done]
    print(f"[run] {cfg.run_id}: {n_questions} questions, {dropped} dropped by the filter, "
          f"{len(work)} passes, {len(done)} already done, {len(pending)} to go")
    if not pending and not done:
        raise SystemExit(f"{cfg.run_id}: no work planned; check dataset and mode")

    model, tokenizer, tokens, revision = loaded or load_model(cfg)
    meta = {
        "run_id": cfg.run_id,
        "git_commit": git_commit(),
        "model_revision": revision,
        "dtype": cfg.dtype,
    }

    rows: list[dict] = []
    batch_size = cfg.batch_size
    i = 0
    started = time.time()
    last_log = 0.0
    while i < len(pending):
        chunk = pending[i : i + batch_size]
        texts = [
            prompts.build(cfg.dataset, q.subject, q.question, r.contents, r.label_order)
            for q, r in chunk
        ]
        try:
            probs = score.score(model, tokenizer, texts, tokens)
        except Exception as exc:
            if not _is_oom(exc) or batch_size == 1:
                _flush(rows, out_dir, cfg.run_id, {**meta, "batch_size": batch_size})
                raise
            # Upstream halves and retries the same way (utils.py:185-192).
            batch_size = max(1, batch_size // 2)
            print(f"[run] out of memory; batch size -> {batch_size}", flush=True)
            _empty_cache()
            continue

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
                    **{f"p_{lab}": float(p[i_]) for i_, lab in enumerate(LABELS)},
                    "seed": cfg.seed,
                    "mode": cfg.mode,
                }
            )
        i += len(chunk)

        if len(rows) >= cfg.flush_every:
            _flush(rows, out_dir, cfg.run_id, {**meta, "batch_size": batch_size})
        elapsed = time.time() - started
        if elapsed - last_log > 60:
            last_log = elapsed
            rate = i / elapsed
            print(f"[run] {i}/{len(pending)}  {rate:.1f} passes/s  "
                  f"eta {(len(pending) - i) / rate / 60:.1f} min", flush=True)

    elapsed = time.time() - started
    _flush(rows, out_dir, cfg.run_id, {**meta, "batch_size": batch_size})
    out_path = consolidate(out_dir, cfg.run_id)
    frame = pd.read_parquet(out_path, columns=["correct"])

    print(f"[run] accuracy {frame.correct.mean() * 100:.2f} over {len(frame)} rows")
    if pending:
        print(f"[run] {elapsed / 60:.1f} min for {len(pending)} passes, "
              f"{len(pending) / elapsed:.1f} passes/s, "
              f"{elapsed / len(pending) * 1000:.1f} ms/pass, final batch {batch_size}")
    print(f"[run] wrote {out_path}")

    if cfg.evict_after:
        evict(cfg.model)
    return out_path


def _empty_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


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
    p.add_argument("--flush-every", type=int)
    p.add_argument("--dtype")
    p.add_argument("--device-map")
    p.add_argument("--out-dir")
    p.add_argument("--evict-after", action="store_true", default=None)
    args = p.parse_args(argv)

    fields = {}
    if args.config:
        import yaml

        fields.update(yaml.safe_load(args.config.read_text()) or {})
    fields.update({k: v for k, v in vars(args).items() if k != "config" and v is not None})
    if "model" not in fields:
        p.error("--model is required (or set it in --config)")
    return Config(**fields)


if __name__ == "__main__":
    run(parse_args())
