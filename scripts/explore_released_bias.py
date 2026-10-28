"""Pilot analysis of the authors' released results. No GPU, no new inference.

Four questions, answered from data already on disk (docs/pilot_released_data.md):

  1. Which released conditions can identify a position effect at all?
  2. How large is the total slot bias, measured cleanly within question (`perm`)?
  3. Does a position effect survive controlling for label (`shuffle_both`)?
  4. Does a position effect survive with the label tokens gone (`noid`)?

Plus a design question that sets the Phase 6 budget: at a fixed number of
forward passes, is it better to run many versions of few questions or few
versions of many questions?

Usage:
    .venv/bin/python scripts/explore_released_bias.py [model_dir ...]
"""

import json
import re
import sys
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.discrete.conditional_models import ConditionalLogit

REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASED = REPO_ROOT / "data/LLM-MCQ-Bias_data/results_mmlu"
UPSTREAM_CODE = REPO_ROOT / "third_party/LLM-MCQ-Bias/code"

sys.path.insert(0, str(UPSTREAM_CODE))
from utils import (  # noqa: E402  upstream, never edited
    BAD_OPTIONS,
    REFER_OPTIONS,
    shuffle_options_with_ids,
)

LABELS = "ABCD"
PERMS = sorted(permutations(range(4)))  # utils.permute_options order
LABEL_LINE = re.compile(r"^([A-E])\. ", re.M)
RNG_SEED = 20231101


def is_invalid(options):
    """utils.load_results:80-85."""
    return any(e in o for e in BAD_OPTIONS for o in options) or any(
        e in o.lower() for e in REFER_OPTIONS for o in options
    )


def iter_records(setting_dir):
    """Yield (subject, data) for valid items only, matching the paper's population."""
    for path in sorted(setting_dir.glob("*.jsonl")):
        subject = path.stem
        for line in open(path):
            record = json.loads(line)
            if record["type"] != "result":
                continue
            data = record["data"]
            if is_invalid([str(o) for o in data["options"]]):
                continue
            yield subject, data


# --------------------------------------------------------------------------
# 1. which conditions could identify position separately from label
# --------------------------------------------------------------------------

def confound_map(model_dir):
    rows = []
    for setting_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
        orders = set()
        n = 0
        for _, data in iter_records(setting_dir):
            prompt = data.get("prompt")
            if prompt is None:
                orders.add("(prompt not stored)")
            else:
                orders.add("".join(LABEL_LINE.findall(prompt.split("Options:")[1])))
            n += 1
            if n >= 2000:
                break
        if orders == {""}:
            verdict, order = "no labels shown", "-"
        elif orders == {"ABCD"}:
            verdict, order = "label == position (confounded)", "ABCD only"
        elif orders == {"(prompt not stored)"}:
            verdict, order = "prompt not stored", "-"
        else:
            verdict = "label decoupled from position"
            order = f"{len(orders)} distinct orders"
        rows.append({"setting": setting_dir.name, "label_order": order, "identifies_position": verdict})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 2. perm: total slot bias, content marginalised out within question
# --------------------------------------------------------------------------

def load_perm(setting_dir):
    """One row per (question, version, slot).

    Labels are fixed A B C D top to bottom, so slot == label here. Every original
    option appears at every slot across the 24 permutations, so content quality is
    marginalised out exactly, within question.
    """
    rows = []
    for qid, (subject, data) in enumerate(iter_records(setting_dir)):
        probs = np.asarray(data["probs"])
        if probs.shape != (24, 4):
            continue
        gold_orig = LABELS.index(data["ideal"])
        picked = probs.argmax(axis=1)
        for version, perm in enumerate(PERMS):
            for slot in range(4):
                rows.append(
                    (
                        qid,
                        subject,
                        version,
                        slot,
                        int(perm[slot] == gold_orig),
                        int(picked[version] == slot),
                        probs[version, slot],
                    )
                )
    return pd.DataFrame(
        rows, columns=["qid", "subject", "version", "slot", "is_gold", "chosen", "prob"]
    )


def perm_summary(df):
    pick = df.groupby("slot")["chosen"].mean() * 100
    recall = df[df.is_gold == 1].groupby("slot")["chosen"].mean() * 100
    acc = df[df.is_gold == 1]["chosen"].mean() * 100
    return pick, recall, acc


def bootstrap_slot_range(df, n_boot=300, seed=RNG_SEED):
    """CI for max-min pick rate across slots, resampling questions."""
    rng = np.random.default_rng(seed)
    per_q = (
        df.groupby(["qid", "slot"])["chosen"].mean().unstack().to_numpy()
    )  # (n_questions, 4)
    qs = per_q.shape[0]
    stats = []
    for _ in range(n_boot):
        sample = per_q[rng.integers(0, qs, qs)]
        means = sample.mean(axis=0) * 100
        stats.append(means.max() - means.min())
    return float(np.mean(stats)), tuple(np.percentile(stats, [2.5, 97.5]))


# --------------------------------------------------------------------------
# 3. shuffle_both: position effect controlling for label
# --------------------------------------------------------------------------

def load_shuffle_both(setting_dir):
    """One row per (question, alternative).

    `utils.shuffle_options_with_ids` shuffles (label, content) pairs together, so
    the label keeps its original content and the gold label never changes -- but
    each label's display POSITION varies across questions. That is the one
    released condition where position is not collinear with label.

    `probs` from `prepare_eval_fn_base` is indexed by label in fixed A B C D token
    order, not by display position.
    """
    rows = []
    for qid, (subject, data) in enumerate(iter_records(setting_dir)):
        options = [str(o) for o in data["options"]]
        shown_ids, _ = shuffle_options_with_ids(list(LABELS), options)
        prompt = data.get("prompt")
        if prompt is not None:
            shown = "".join(LABEL_LINE.findall(prompt.split("Options:")[1]))
            assert "".join(shown_ids) == shown, (shown_ids, shown)
        pos_of_label = {label: j for j, label in enumerate(shown_ids)}
        probs = np.asarray(data["probs"])
        picked_label = LABELS[int(probs.argmax())]
        for i, label in enumerate(LABELS):
            rows.append(
                (
                    qid,
                    subject,
                    i,
                    pos_of_label[label],
                    int(label == data["ideal"]),
                    int(label == picked_label),
                    probs[i],
                )
            )
    return pd.DataFrame(
        rows,
        columns=["qid", "subject", "label_idx", "pos", "is_gold", "chosen", "prob"],
    )


def fit_conditional_logit(df, group_cols, features, max_groups=None, seed=RNG_SEED):
    """Phase 5's estimator, exercised on real data.

    Each group is one choice set of 4 alternatives; `chosen` is the picked one.
    """
    groups = df.groupby(group_cols, sort=False).ngroup()
    if max_groups is not None and groups.nunique() > max_groups:
        rng = np.random.default_rng(seed)
        keep = set(rng.choice(groups.unique(), max_groups, replace=False))
        mask = groups.isin(keep)
        df, groups = df[mask], groups[mask]
    model = ConditionalLogit(df["chosen"].to_numpy(), df[features].to_numpy(), groups=groups.to_numpy())
    res = model.fit(disp=False)
    ci = res.conf_int()
    return pd.DataFrame(
        {
            "coef": res.params,
            "se": res.bse,
            "ci_lo": ci[:, 0],
            "ci_hi": ci[:, 1],
            "p": res.pvalues,
        },
        index=features,
    ), groups.nunique()


def add_dummies(df, col, prefix, levels=(1, 2, 3)):
    for level in levels:
        df[f"{prefix}{level}"] = (df[col] == level).astype(float)
    return [f"{prefix}{level}" for level in levels]


# --------------------------------------------------------------------------
# 4. noid: position effect with no label tokens at all
# --------------------------------------------------------------------------

def load_noid(setting_dir):
    """`probs` is indexed by display position; content is in original order.

    Content is not permuted here, so overall pick rate by position is confounded
    with whatever sits at each position. Recall by GOLD position is the usable
    quantity: the item scored is the gold either way, and MMLU's gold position is
    arbitrary, so averaging over questions matches content across positions.
    """
    rows = []
    for qid, (subject, data) in enumerate(iter_records(setting_dir)):
        probs = np.asarray(data["probs"])
        gold_pos = LABELS.index(data["ideal"])
        picked = int(probs.argmax())
        for pos in range(4):
            rows.append((qid, subject, pos, int(pos == gold_pos), int(pos == picked), probs[pos]))
    return pd.DataFrame(
        rows, columns=["qid", "subject", "pos", "is_gold", "chosen", "prob"]
    )


# --------------------------------------------------------------------------
# 5. design: versions per question vs questions, at a fixed forward-pass budget
# --------------------------------------------------------------------------

def design_tradeoff(perm_df, budget=12000, n_boot=120, seed=RNG_SEED):
    """At a fixed number of forward passes, which split estimates slot bias best?

    `perm` has 24 versions per question, so we can emulate k versions x m questions
    with k*m == budget and compare the sampling error of the slot-bias estimate.
    This is the full16-vs-balanced1 question (CLAUDE.md Phase 6) answered with the
    authors' data instead of our GPU hours.
    """
    rng = np.random.default_rng(seed)
    per_q_version = (
        perm_df.groupby(["qid", "version", "slot"])["chosen"].mean().unstack().to_numpy()
    )
    n_q = perm_df["qid"].nunique()
    cube = per_q_version.reshape(n_q, 24, 4)

    rows = []
    for k in (1, 2, 4, 8, 16, 24):
        m = budget // k
        if m > n_q:
            continue
        stats = []
        for _ in range(n_boot):
            qs = rng.integers(0, n_q, m)
            vs = rng.integers(0, 24, (m, k))
            picked = cube[qs[:, None], vs]  # (m, k, 4)
            means = picked.reshape(-1, 4).mean(axis=0) * 100
            stats.append(means.max() - means.min())
        rows.append(
            {
                "versions_per_q": k,
                "questions": m,
                "forward_passes": k * m,
                "slot_range_mean": float(np.mean(stats)),
                "slot_range_sd": float(np.std(stats)),
            }
        )
    return pd.DataFrame(rows)


def design_tradeoff_logit(perm_df, budget=12000, seed=RNG_SEED):
    """Same question as design_tradeoff, but on the estimator we actually report.

    The marginal pick-rate version above could mislead, because the conditional
    logit identifies slot effects *within* a choice set. So refit it at a fixed
    budget for several versions-per-question splits and compare the standard
    errors of the slot coefficients.
    """
    rng = np.random.default_rng(seed)
    all_q = perm_df["qid"].unique()
    slot_cols = add_dummies(perm_df, "slot", "slot_")
    features = slot_cols + ["is_gold"]

    rows = []
    for k in (1, 4, 16, 24):
        m = budget // k
        if m > len(all_q):
            continue
        qs = rng.choice(all_q, m, replace=False)
        chosen_versions = {q: rng.choice(24, k, replace=False) for q in qs}
        mask = perm_df["qid"].isin(qs) & [
            v in chosen_versions.get(q, ()) for q, v in zip(perm_df["qid"], perm_df["version"])
        ]
        subset = perm_df[mask]
        res, n_groups = fit_conditional_logit(subset, ["qid", "version"], features)
        row = {"versions_per_q": k, "questions": m, "choice_sets": n_groups}
        for name in features:
            row[f"se[{name}]"] = res.loc[name, "se"]
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------

def report(model_dir):
    name = model_dir.name
    print("=" * 78)
    print(f"MODEL {name}   (MMLU 0-shot, filtered population)")
    print("=" * 78)

    print("\n[1] Can each released condition identify position separately from label?\n")
    print(confound_map(model_dir).to_string(index=False))

    perm_dir = model_dir / "mmlu_perm"
    if perm_dir.exists():
        print("\n[2] perm -- total slot bias (labels fixed ABCD, so slot == label)")
        perm_df = load_perm(perm_dir)
        pick, recall, acc = perm_summary(perm_df)
        n_q = perm_df["qid"].nunique()
        print(f"    {n_q} questions x 24 permutations, accuracy {acc:.2f}%")
        print("    pick rate  " + "  ".join(f"slot{s}={v:5.2f}" for s, v in pick.items()))
        print("    recall     " + "  ".join(f"slot{s}={v:5.2f}" for s, v in recall.items()))
        mean, (lo, hi) = bootstrap_slot_range(perm_df)
        print(f"    max-min pick rate {mean:.2f} pp  (95% CI {lo:.2f} to {hi:.2f}, questions resampled)")

        print("\n[5] Design: fixed 12000 forward passes, how to split them")
        print("    (a) sampling error of the marginal pick-rate range")
        print(design_tradeoff(perm_df).to_string(index=False))
        print("    (b) standard errors from the conditional logit we report")
        print(design_tradeoff_logit(perm_df).to_string(index=False))
    else:
        perm_df = None

    sb_dir = model_dir / "mmlu_shuffle_both"
    if sb_dir.exists():
        print("\n[3] shuffle_both -- position effect CONTROLLING for label")
        sb = load_shuffle_both(sb_dir)
        label_cols = add_dummies(sb, "label_idx", "label_")
        pos_cols = add_dummies(sb, "pos", "pos_")
        res, n_groups = fit_conditional_logit(
            sb, ["qid"], label_cols + pos_cols + ["is_gold"]
        )
        print(f"    conditional logit, {n_groups} choice sets of 4")
        print("    reference: label A, position 0")
        print(res.to_string(float_format=lambda v: f"{v:8.4f}"))

    noid_dir = model_dir / "mmlu_noid"
    if noid_dir.exists():
        print("\n[4] noid -- position effect with NO label tokens")
        noid = load_noid(noid_dir)
        pick = noid.groupby("pos")["chosen"].mean() * 100
        recall = noid[noid.is_gold == 1].groupby("pos")["chosen"].mean() * 100
        acc = noid[noid.is_gold == 1]["chosen"].mean() * 100
        print(f"    {noid['qid'].nunique()} questions, accuracy {acc:.2f}%")
        print("    pick rate  " + "  ".join(f"pos{p}={v:5.2f}" for p, v in pick.items()))
        print("    recall     " + "  ".join(f"pos{p}={v:5.2f}" for p, v in recall.items()))
        print(f"    recall range {recall.max() - recall.min():.2f} pp")


def main():
    if not RELEASED.exists():
        sys.exit(f"missing {RELEASED}; see README for the sparse checkout")
    wanted = sys.argv[1:] or ["0s_Llama-2-7b-hf", "0s_llama-7b"]
    for name in wanted:
        d = RELEASED / name
        if not d.exists():
            print(f"{name}: not in the sparse checkout, skipping")
            continue
        report(d)
        print()


if __name__ == "__main__":
    main()
