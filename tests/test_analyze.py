"""Phase 5's acceptance test: the estimator must separate label from position.

A pure position bias has to come back as position coefficients with label
coefficients near zero, and a pure label bias the other way round. If the
regression cannot do that on data where we know the answer, nothing it says
about a real model means anything.

The synthetic runs are generated through the real rotate/prompts path, so the
design under test is the one that will actually be run.
"""

import numpy as np
import pandas as pd
import pytest

from mcqbias.analyze import (
    coefficient_spans,
    conditional_logit,
    marginals,
    rstd,
    summarize,
    to_long,
)
from mcqbias.rotate import LABELS, N_VERSIONS, render, shifts

N = 4


def synthetic_run(label_logits, position_logits, gold_logit=0.0, n_questions=250, seed=0):
    """A full16 run whose choices follow exactly the model we want recovered.

    Utility of an alternative = label effect + position effect + gold effect, and
    the choice is drawn from the softmax of that. Drawing, not taking the argmax,
    is what makes this a test of the estimator: a deterministic argmax over a
    dominant label picks that label in every single choice set, which is perfect
    separation — the label coefficients run off to the optimizer's bound and the
    position dimension is not identified at all. Real models vary their choice
    across questions because option content varies; sampling stands in for that,
    and it makes the conditional logit the exact MLE for this design, so the
    recovered coefficients can be compared against the injected ones directly.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for q in range(n_questions):
        options = tuple(f"q{q}opt{i}" for i in range(N))
        gold = int(rng.integers(0, N))
        for v in range(N_VERSIONS):
            r = render(options, gold, *shifts(v))
            utility = np.array(
                [
                    label_logits[LABELS.index(lab)]
                    + position_logits[r.label_order.index(lab)]
                    + (gold_logit if lab == r.gold_label else 0.0)
                    for lab in LABELS
                ]
            )
            probs = np.exp(utility - utility.max())
            probs /= probs.sum()
            pred_label = LABELS[int(rng.choice(N, p=probs))]
            rows.append(
                {
                    "model": "synthetic",
                    "dataset": "synthetic",
                    "subject": "synthetic",
                    "qid": f"synthetic:{q}",
                    "version": r.version,
                    "label_shift": r.label_shift,
                    "content_shift": r.content_shift,
                    "label_order": r.label_order,
                    "gold_pos": r.gold_pos,
                    "gold_label": r.gold_label,
                    "pred_pos": r.label_order.index(pred_label),
                    "pred_label": pred_label,
                    "correct": pred_label == r.gold_label,
                    **{f"p_{lab}": float(probs[i]) for i, lab in enumerate(LABELS)},
                    "seed": seed,
                    "mode": "full16",
                }
            )
    return pd.DataFrame(rows)


FLAT = [0.0] * N
LADDER = [0.0, -1.0, -2.0, -3.0]
PURE_POSITION = FLAT, LADDER
PURE_LABEL = LADDER, FLAT

TOL = 0.3  # coefficient tolerance at 250 questions x 16 versions = 4000 choice sets
NEAR_ZERO = 0.4  # a truly flat dimension still wanders this much from sampling noise


def _coefs(table, prefix):
    return [table.coef[f"{prefix}{i}"] for i in (1, 2, 3)]


def test_pure_position_bias_is_recovered_as_position():
    table = conditional_logit(to_long(synthetic_run(*PURE_POSITION, gold_logit=0.8)))
    spans = coefficient_spans(table)

    np.testing.assert_allclose(_coefs(table, "pos_"), LADDER[1:], atol=TOL)
    np.testing.assert_allclose(_coefs(table, "label_"), FLAT[1:], atol=NEAR_ZERO)
    assert spans["position_span"] > 2.0
    assert spans["label_span"] < NEAR_ZERO, table
    assert table.coef["is_gold"] == pytest.approx(0.8, abs=TOL)


def test_pure_label_bias_is_recovered_as_label():
    table = conditional_logit(to_long(synthetic_run(*PURE_LABEL, gold_logit=0.8)))
    spans = coefficient_spans(table)

    np.testing.assert_allclose(_coefs(table, "label_"), LADDER[1:], atol=TOL)
    np.testing.assert_allclose(_coefs(table, "pos_"), FLAT[1:], atol=NEAR_ZERO)
    assert spans["label_span"] > 2.0
    assert spans["position_span"] < NEAR_ZERO, table
    assert table.coef["is_gold"] == pytest.approx(0.8, abs=TOL)


def test_a_mixture_separates_into_both():
    """The case the paper's own settings cannot do: both effects, one fit."""
    labels, positions = [0.0, -1.5, -1.5, -1.5], [0.0, 0.0, 1.0, 1.0]
    table = conditional_logit(to_long(synthetic_run(labels, positions, gold_logit=0.5)))

    np.testing.assert_allclose(_coefs(table, "label_"), labels[1:], atol=TOL)
    np.testing.assert_allclose(_coefs(table, "pos_"), positions[1:], atol=TOL)
    spans = coefficient_spans(table)
    assert spans["label_span"] > 1.0
    assert spans["position_span"] > 0.5


def test_no_bias_gives_no_spans():
    frame = synthetic_run(FLAT, FLAT, gold_logit=2.0)
    table = conditional_logit(to_long(frame))
    spans = coefficient_spans(table)

    assert spans["label_span"] < NEAR_ZERO
    assert spans["position_span"] < NEAR_ZERO
    assert table.coef["is_gold"] == pytest.approx(2.0, abs=TOL)
    # only a gold effect: often right, never biased toward a label or a slot
    assert 0.5 < frame.correct.mean() < 1.0


def test_to_long_keeps_label_and_position_consistent():
    frame = synthetic_run(*PURE_POSITION)
    long = to_long(frame)
    assert len(long) == len(frame) * N
    # every choice set has exactly one chosen alternative and one gold
    grouped = long.groupby(["qid", "version"])
    assert grouped.chosen.sum().eq(1).all()
    assert grouped.is_gold.sum().eq(1).all()
    assert grouped.pos.nunique().eq(N).all()
    np.testing.assert_allclose(grouped.prob.sum().to_numpy(), 1.0, atol=1e-6)


def test_marginals_and_rstd_agree_with_a_hand_count():
    frame = synthetic_run(*PURE_LABEL, gold_logit=0.0)
    m = marginals(frame)
    assert m["n_questions"] == frame.qid.nunique()
    assert m["n_rows"] == m["n_questions"] * N_VERSIONS
    assert m["accuracy"] == pytest.approx(frame.correct.mean() * 100)

    # A pure label ladder [0, -1, -2, -3]: softmax puts ~64% on A, and the pick
    # rate must fall monotonically from A to D.
    picks = [m["pick_by_label"][lab] for lab in LABELS]
    assert picks[0] > 55
    assert picks == sorted(picks, reverse=True), picks
    assert m["rstd_label"] > m["rstd_position"]
    # by position the design is balanced, so pick rate is flat
    assert max(m["pick_by_position"].values()) - min(m["pick_by_position"].values()) < 2

    # rstd is the std of the four recalls, nothing more
    value, recalls = rstd([0, 1, 2, 3], [0, 1, 2, 2])
    assert recalls == [100.0, 100.0, 100.0, 0.0]
    assert value == pytest.approx(np.std([100, 100, 100, 0]))


def test_bootstrap_ci_brackets_the_point_estimate():
    frame = synthetic_run(*PURE_POSITION)
    s = summarize(frame, n_boot=200, seed=1)
    mean, (lo, hi) = s["spread_position"]
    assert lo <= mean <= hi
    observed = max(s["pick_by_position"].values()) - min(s["pick_by_position"].values())
    assert lo <= observed <= hi
    # a flat label dimension must not produce a large spread
    assert s["spread_label"][0] < s["spread_position"][0]
