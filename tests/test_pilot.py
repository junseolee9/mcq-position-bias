"""Pins the pilot findings (docs/pilot_released_data.md).

The reconstruction checks are the important ones: the shuffle_both analysis only
means anything if our recomputed per-question shuffle really is the one the model
saw, so we verify it against the stored prompt, the stored prediction and the
stored correctness for every item.

Skipped when the released-data sparse checkout is absent (data/ is gitignored).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

RELEASED = REPO_ROOT / "data/LLM-MCQ-Bias_data/results_mmlu"
pytestmark = pytest.mark.skipif(
    not RELEASED.exists(), reason="released-data sparse checkout not present"
)


@pytest.mark.parametrize("model", ["0s_Llama-2-7b-hf", "0s_llama-7b"])
def test_shuffle_reconstruction_is_exact(model):
    """Our recomputed shuffle must match what the model actually saw."""
    from explore_released_bias import LABELS, iter_records

    sys.path.insert(0, str(REPO_ROOT / "third_party/LLM-MCQ-Bias/code"))
    from utils import shuffle_options_with_ids

    import re

    label_line = re.compile(r"^([A-E])\. ", re.M)
    n = 0
    for _, data in iter_records(RELEASED / model / "mmlu_shuffle_both"):
        ids, _ = shuffle_options_with_ids(list(LABELS), [str(o) for o in data["options"]])
        shown = "".join(label_line.findall(data["prompt"].split("Options:")[1]))
        assert "".join(ids) == shown
        picked = LABELS[int(np.argmax(data["probs"]))]
        assert picked == data["sampled"]
        assert (picked == data["ideal"]) == data["correct"]
        n += 1
    assert n == 13592


def test_only_shuffle_both_decouples_label_from_position():
    """The structural claim our critique rests on."""
    from explore_released_bias import confound_map

    df = confound_map(RELEASED / "0s_Llama-2-7b-hf").set_index("setting")
    decoupled = df.index[df["identifies_position"] == "label decoupled from position"]
    assert list(decoupled) == ["mmlu_shuffle_both"]
    for setting in ("mmlu_perm", "mmlu_move_a", "mmlu_move_d"):
        assert df.loc[setting, "identifies_position"] == "label == position (confounded)"
    assert df.loc["mmlu_noid", "identifies_position"] == "no labels shown"


@pytest.mark.parametrize(
    "model,pos_span_min,gold_coef",
    [
        # Position span exceeds the is_gold coefficient for both models.
        ("0s_Llama-2-7b-hf", 0.60, 0.43),
        ("0s_llama-7b", 0.75, 0.38),
    ],
)
def test_position_effect_exceeds_gold_effect(model, pos_span_min, gold_coef):
    from explore_released_bias import (
        add_dummies,
        fit_conditional_logit,
        load_shuffle_both,
    )

    df = load_shuffle_both(RELEASED / model / "mmlu_shuffle_both")
    label_cols = add_dummies(df, "label_idx", "label_")
    pos_cols = add_dummies(df, "pos", "pos_")
    res, n_groups = fit_conditional_logit(df, ["qid"], label_cols + pos_cols + ["is_gold"])
    assert n_groups == 13592

    pos_coefs = [0.0] + [res.loc[c, "coef"] for c in pos_cols]
    pos_span = max(pos_coefs) - min(pos_coefs)
    assert pos_span > pos_span_min
    assert res.loc["is_gold", "coef"] == pytest.approx(gold_coef, abs=0.02)
    assert pos_span > res.loc["is_gold", "coef"]

    # Position effects are not a null result.
    significant = [c for c in pos_cols if res.loc[c, "p"] < 1e-6]
    assert len(significant) >= 2
