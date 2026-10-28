"""Pins our RStd/accuracy recipe against the authors' released results.

If this passes, our metric code reproduces Table 3 exactly, so a difference in a
run of ours is a difference in inference, not in scoring arithmetic.

Skipped when the released-data sparse checkout is absent (it lives under
data/, which is gitignored) -- see README.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASED = REPO_ROOT / "data/LLM-MCQ-Bias_data/results_mmlu"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
pytestmark = pytest.mark.skipif(
    not RELEASED.exists(), reason="released-data sparse checkout not present"
)


@pytest.mark.parametrize(
    "model,population,n,acc,rstd",
    [
        # Table 3's "RStd 23.0 / Acc 35.8" row, identified as Llama-2-7b-hf on
        # the filtered population (docs/reproduction.md).
        ("0s_Llama-2-7b-hf", "filtered", 13592, 35.80, 22.95),
        ("0s_Llama-2-7b-hf", "unfiltered", 14042, 36.06, 22.51),
        ("0s_llama-7b", "filtered", 13592, 33.75, 18.52),
        ("0s_llama-7b", "unfiltered", 14042, 33.84, 18.34),
    ],
)
def test_released_baseline(model, population, n, acc, rstd):
    from check_released_results import summarize

    result = summarize(RELEASED / model / "mmlu")[population]
    assert result["n"] == n
    assert result["acc"] == pytest.approx(acc, abs=0.01)
    assert result["rstd"] == pytest.approx(rstd, abs=0.01)


def test_invalid_fraction_is_the_papers_3_2_percent():
    from check_released_results import summarize

    result = summarize(RELEASED / "0s_Llama-2-7b-hf" / "mmlu")
    assert result["n_invalid"] == 450
    assert result["n_invalid"] / result["n_total"] == pytest.approx(0.032, abs=0.0005)
