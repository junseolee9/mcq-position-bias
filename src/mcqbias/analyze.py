"""Turn raw run output into the tables and figures the writeup needs.

    python -m mcqbias.analyze results/raw/*.parquet

The estimators are the ones already exercised on the authors' released data in
scripts/explore_released_bias.py (docs/pilot_released_data.md); this module is
where they live for our own runs. `rstd` is the canonical copy of upstream's
recipe (debias_pride.py:152-154) and is pinned against the paper's own 22.95 by
tests/test_released_results.py.

The point of the whole design is the pair of spans in `coefficient_spans`: how
far apart the label coefficients are, versus how far apart the position
coefficients are, estimated at the same time from the same choice sets.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from statsmodels.discrete.conditional_models import ConditionalLogit

LABELS = "ABCD"
N = 4
REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------- metrics

def rstd(gold, pred, n_labels: int = N) -> tuple[float, list[float]]:
    """Upstream's RStd: per-gold-class recall, x100, population std.

    # adapted from LLM-MCQ-Bias/code/debias_pride.py@2ae2f40 :152-154

    Classes are integers 0..n-1, so this serves both the per-label and the
    per-position version: the arithmetic is the same, only the grouping differs.
    """
    report = classification_report(
        list(gold), list(pred), output_dict=True, zero_division=0
    )
    recalls = [report.get(str(e), {"recall": 0.0})["recall"] * 100 for e in range(n_labels)]
    return float(np.std(recalls)), recalls


def to_long(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per (question, version, alternative) — the conditional logit's input.

    `p_A..p_D` are indexed by label, so an alternative's display position comes
    from `label_order`.
    """
    pos_of_label = np.array(
        [[order.index(lab) for lab in LABELS] for order in frame.label_order],
        dtype=np.int8,
    )
    repeated = frame.loc[frame.index.repeat(N)].reset_index(drop=True)
    label_idx = np.tile(np.arange(N, dtype=np.int8), len(frame))
    long = pd.DataFrame(
        {
            "qid": repeated.qid,
            "version": repeated.version,
            "label_idx": label_idx,
            "pos": pos_of_label.reshape(-1),
            "prob": frame[[f"p_{lab}" for lab in LABELS]].to_numpy().reshape(-1),
        }
    )
    long["label"] = [LABELS[i] for i in long.label_idx]
    long["is_gold"] = (long.label.to_numpy() == repeated.gold_label.to_numpy()).astype(float)
    long["chosen"] = (long.label.to_numpy() == repeated.pred_label.to_numpy()).astype(float)
    return long


def marginals(frame: pd.DataFrame) -> dict:
    """Accuracy, and pick rate and recall by label and by position, each with RStd."""
    gold_label_idx = [LABELS.index(e) for e in frame.gold_label]
    pred_label_idx = [LABELS.index(e) for e in frame.pred_label]
    label_rstd, label_recall = rstd(gold_label_idx, pred_label_idx)
    pos_rstd, pos_recall = rstd(frame.gold_pos, frame.pred_pos)

    long = to_long(frame)
    return {
        "n_rows": len(frame),
        "n_questions": frame.qid.nunique(),
        "accuracy": float(frame.correct.mean() * 100),
        "recall_by_label": dict(zip(LABELS, label_recall)),
        "recall_by_position": {p: r for p, r in enumerate(pos_recall)},
        "rstd_label": label_rstd,
        "rstd_position": pos_rstd,
        "pick_by_label": (long.groupby("label").chosen.mean() * 100).to_dict(),
        "pick_by_position": (long.groupby("pos").chosen.mean() * 100).to_dict(),
    }


# ---------------------------------------------------------------- regression

def _dummies(df: pd.DataFrame, col: str, prefix: str, levels=(1, 2, 3)) -> list[str]:
    """# adapted from scripts/explore_released_bias.py:225 (add_dummies)"""
    for level in levels:
        df[f"{prefix}{level}"] = (df[col] == level).astype(float)
    return [f"{prefix}{level}" for level in levels]


def conditional_logit(long: pd.DataFrame) -> pd.DataFrame:
    """Label and position effects from the same choice sets.

    Each (qid, version) is one choice set of four alternatives. Label A and
    position 0 are the references, so question difficulty and option content
    drop out within the set and `is_gold` absorbs "being the right answer".

    # adapted from scripts/explore_released_bias.py:199 (fit_conditional_logit)
    """
    df = long.copy()
    features = (
        _dummies(df, "label_idx", "label_")
        + _dummies(df, "pos", "pos_")
        + ["is_gold"]
    )
    groups = df.groupby(["qid", "version"], sort=False).ngroup()
    result = ConditionalLogit(
        df.chosen.to_numpy(), df[features].to_numpy(), groups=groups.to_numpy()
    ).fit(disp=False)
    ci = result.conf_int()
    table = pd.DataFrame(
        {
            "coef": result.params,
            "se": result.bse,
            "ci_lo": ci[:, 0],
            "ci_hi": ci[:, 1],
            "p": result.pvalues,
        },
        index=features,
    )
    table.attrs["n_choice_sets"] = int(groups.nunique())
    return table


def coefficient_spans(table: pd.DataFrame) -> dict:
    """max - min coefficient, including the omitted reference level at 0.

    This is the headline comparison: a label span of 2.8 against a position span
    of 0.7 says one thing about the paper's claim, two comparable spans another.
    """
    def span(prefix):
        coefs = [0.0] + [table.coef[k] for k in table.index if k.startswith(prefix)]
        return float(max(coefs) - min(coefs))

    return {
        "label_span": span("label_"),
        "position_span": span("pos_"),
        "is_gold": float(table.coef["is_gold"]),
    }


# ---------------------------------------------------------------- bootstrap

def bootstrap_spread(long: pd.DataFrame, by: str, n_boot: int = 1000, seed: int = 0):
    """CI for the max-min pick-rate spread, resampling QUESTIONS, not rows.

    Rows within a question are not independent — in full16 they are sixteen views
    of the same item — so resampling rows would understate the interval.

    # adapted from scripts/explore_released_bias.py:140 (bootstrap_slot_range)
    """
    per_q = long.groupby(["qid", by]).chosen.mean().unstack().to_numpy()
    per_q = per_q[~np.isnan(per_q).any(axis=1)]
    rng = np.random.default_rng(seed)
    n = len(per_q)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        means = per_q[rng.integers(0, n, n)].mean(axis=0) * 100
        stats[i] = means.max() - means.min()
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(stats.mean()), (float(lo), float(hi))


# ---------------------------------------------------------------- reporting

def summarize(frame: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> dict:
    out = marginals(frame)
    long = to_long(frame)
    table = conditional_logit(long)
    out["logit"] = table
    out["spans"] = coefficient_spans(table)
    out["n_choice_sets"] = table.attrs["n_choice_sets"]
    out["spread_label"] = bootstrap_spread(long, "label", n_boot, seed)
    out["spread_position"] = bootstrap_spread(long, "pos", n_boot, seed)
    return out


def print_report(name: str, s: dict) -> None:
    print(f"\n=== {name} ===")
    print(f"{s['n_rows']} rows over {s['n_questions']} questions, "
          f"{s['n_choice_sets']} choice sets")
    print(f"accuracy {s['accuracy']:.2f}   "
          f"RStd(label) {s['rstd_label']:.2f}   RStd(position) {s['rstd_position']:.2f}")

    print("\n            " + "".join(f"{k:>9}" for k in LABELS) + "   |" +
          "".join(f"{p:>9}" for p in range(N)))
    for what, by_label, by_pos in (
        ("recall", s["recall_by_label"], s["recall_by_position"]),
        ("pick rate", s["pick_by_label"], s["pick_by_position"]),
    ):
        row = "".join(f"{by_label.get(k, float('nan')):9.2f}" for k in LABELS)
        row += "   |" + "".join(f"{by_pos.get(p, float('nan')):9.2f}" for p in range(N))
        print(f"{what:<12}{row}")
    print("            (by label)                  | (by position)")

    print("\nconditional logit")
    print(s["logit"].to_string(float_format=lambda v: f"{v:9.4f}"))
    spans = s["spans"]
    print(f"\nspans: label {spans['label_span']:.3f}   "
          f"position {spans['position_span']:.3f}   is_gold {spans['is_gold']:.3f}")
    for what in ("label", "position"):
        mean, (lo, hi) = s[f"spread_{what}"]
        print(f"pick-rate spread by {what:<9} {mean:6.2f} pp   95% CI {lo:.2f} - {hi:.2f}")


def figure(summaries: dict, out_path: Path) -> Path:
    """Label span against position span per run, with bootstrap-free logit SEs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(summaries)
    x = np.arange(len(names))
    label = [summaries[n]["spans"]["label_span"] for n in names]
    position = [summaries[n]["spans"]["position_span"] for n in names]
    gold = [summaries[n]["spans"]["is_gold"] for n in names]

    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(names)), 4.2))
    ax.bar(x - 0.25, label, 0.25, label="label span")
    ax.bar(x, position, 0.25, label="position span")
    ax.bar(x + 0.25, gold, 0.25, label="is_gold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("logit coefficient range")
    ax.set_title("Label effect vs position effect, same choice sets")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("paths", nargs="+", type=Path, help="results/raw/*.parquet")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results/summary")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    summaries, rows = {}, []
    for path in args.paths:
        frame = pd.read_parquet(path)
        s = summarize(frame, args.n_boot, args.seed)
        summaries[path.stem] = s
        print_report(path.stem, s)
        rows.append(
            {
                "run_id": path.stem,
                "model": frame.model.iloc[0],
                "dataset": frame.dataset.iloc[0],
                "mode": frame["mode"].iloc[0],
                "seed": int(frame.seed.iloc[0]),
                "n_questions": s["n_questions"],
                "accuracy": s["accuracy"],
                "rstd_label": s["rstd_label"],
                "rstd_position": s["rstd_position"],
                **s["spans"],
                "spread_label": s["spread_label"][0],
                "spread_position": s["spread_position"][0],
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_dir / "summary.csv", index=False)
    for name, s in summaries.items():
        s["logit"].to_csv(args.out_dir / f"logit_{name}.csv")
    print(f"\nwrote {args.out_dir / 'summary.csv'}")
    print(f"wrote {figure(summaries, args.out_dir / 'label_vs_position.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
