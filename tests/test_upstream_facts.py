"""Pins the Phase 1 facts our pipeline is built on (docs/upstream_notes.md).

These read upstream's shipped CSVs and its own filter lists, so they fail if the
submodule moves or if we later paraphrase the filter instead of copying it.
"""

import glob
import sys
from pathlib import Path

import pandas as pd
import pytest

UPSTREAM_CODE = Path(__file__).resolve().parents[1] / "third_party/LLM-MCQ-Bias/code"


@pytest.fixture(scope="module")
def upstream_filters():
    sys.path.insert(0, str(UPSTREAM_CODE))
    try:
        from utils import BAD_OPTIONS, REFER_OPTIONS

        return BAD_OPTIONS, REFER_OPTIONS
    finally:
        sys.path.remove(str(UPSTREAM_CODE))


def count_filtered(task, option_ids, bad_options, refer_options):
    total = filtered = 0
    for path in sorted(glob.glob(str(UPSTREAM_CODE / f"data_{task}/test/*_test.csv"))):
        df = pd.read_csv(path, names=("Question", *option_ids, "Answer"), dtype=str)
        for _, row in df.iterrows():
            options = [str(row[e]) for e in option_ids]
            total += 1
            # Exactly upstream's semantics (utils.py:80-85): BAD_OPTIONS is a
            # case-sensitive substring test, REFER_OPTIONS a lowercased one.
            if any(e in o for e in bad_options for o in options) or any(
                e in o.lower() for e in refer_options for o in options
            ):
                filtered += 1
    return total, filtered


@pytest.mark.parametrize(
    "task,option_ids,expect_total,expect_filtered",
    [
        ("mmlu", "ABCD", 14042, 450),  # 3.20%, matches the paper's stated 3.2%
        ("arc", "ABCD", 1165, 0),
        ("csqa", "ABCDE", 1216, 0),
    ],
)
def test_invalid_item_counts(
    task, option_ids, expect_total, expect_filtered, upstream_filters
):
    total, filtered = count_filtered(task, option_ids, *upstream_filters)
    assert (total, filtered) == (expect_total, expect_filtered)


def test_mmlu_has_57_subjects():
    assert len(glob.glob(str(UPSTREAM_CODE / "data_mmlu/test/*_test.csv"))) == 57
