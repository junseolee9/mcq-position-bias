"""Recompute accuracy and RStd from the authors' released result files.

No GPU needed. This settles which model Table 3's "RStd 23.0 / Acc 35.8" row
refers to, and it validates our RStd recipe against the paper's own outputs
before we spend A100 hours (docs/reproduction.md).

Uses upstream's exact definitions:
  - accuracy        utils.get_accuracy
  - invalid items   utils.BAD_OPTIONS / REFER_OPTIONS, as applied in
                    utils.load_results (post-hoc, after inference)
  - RStd            debias_pride.py:152-154 -- per-gold-label recall from
                    sklearn classification_report, x100, np.std (ddof=0)

Usage:
    .venv/bin/python scripts/check_released_results.py
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASED = REPO_ROOT / "data/LLM-MCQ-Bias_data"

# Both helpers live in src/ so there is exactly one copy of each. Importing
# upstream's utils at runtime would need a sys.path insert into third_party and
# would make CPython drop __pycache__ there, which ground rule 1 forbids;
# mcqbias.data holds the filter lists, pinned equal to upstream's by
# tests/test_upstream_facts.py. And because test_released_results.py checks this
# script against the paper's own 22.95, importing rstd from the analysis module
# validates that copy too.
sys.path.insert(0, str(REPO_ROOT / "src"))
from mcqbias.analyze import rstd  # noqa: E402
from mcqbias.data import is_invalid  # noqa: E402


def summarize(result_dir, label_ids="ABCD"):
    rows = []
    for path in sorted(result_dir.glob("*.jsonl")):
        for line in open(path):
            record = json.loads(line)
            if record["type"] != "result":
                continue
            d = record["data"]
            rows.append(
                (
                    label_ids.index(d["ideal"]),
                    label_ids.index(d["sampled"][0]),
                    is_invalid([str(o) for o in d["options"]]),
                )
            )

    out = {"n_total": len(rows), "n_invalid": sum(r[2] for r in rows)}
    for name, keep in (("unfiltered", False), ("filtered", True)):
        subset = [r for r in rows if not (keep and r[2])]
        gold = [r[0] for r in subset]
        pred = [r[1] for r in subset]
        acc = float(np.mean([g == p for g, p in zip(gold, pred)]) * 100)
        std, recalls = rstd(gold, pred, len(label_ids))
        picks = [100 * pred.count(i) / len(pred) for i in range(len(label_ids))]
        out[name] = {"n": len(subset), "acc": acc, "rstd": std,
                     "recalls": recalls, "pick_rate": picks}
    return out


def main():
    if not RELEASED.exists():
        sys.exit(f"missing {RELEASED}; see README for the sparse checkout")

    for model in ("0s_llama-7b", "0s_Llama-2-7b-hf"):
        d = RELEASED / "results_mmlu" / model / "mmlu"
        if not d.exists():
            print(f"{model}: not in the sparse checkout, skipping")
            continue
        r = summarize(d)
        print(f"\n=== MMLU 0-shot, default order, {model} ===")
        print(f"items: {r['n_total']} total, {r['n_invalid']} invalid "
              f"({100 * r['n_invalid'] / r['n_total']:.2f}%)")
        for name in ("unfiltered", "filtered"):
            s = r[name]
            print(f"  {name:10s} n={s['n']:5d}  acc={s['acc']:5.2f}  rstd={s['rstd']:5.2f}")
            print(f"             recall/label  " +
                  "  ".join(f"{k}={v:5.2f}" for k, v in zip("ABCD", s["recalls"])))
            print(f"             pick rate     " +
                  "  ".join(f"{k}={v:5.2f}" for k, v in zip("ABCD", s["pick_rate"])))


if __name__ == "__main__":
    main()
