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


def test_our_copy_of_the_filter_lists_matches_upstream(upstream_filters):
    """src/mcqbias/data.py copies these lists instead of importing them."""
    from mcqbias.data import BAD_OPTIONS, REFER_OPTIONS

    assert (BAD_OPTIONS, REFER_OPTIONS) == upstream_filters


def test_our_loader_reproduces_the_paper_s_filtered_population():
    from mcqbias.data import load

    for task, expect_total, expect_dropped in (("mmlu", 14042, 450), ("arc", 1165, 0)):
        kept, dropped = load(task)
        assert (len(kept) + dropped, dropped) == (expect_total, expect_dropped)
        # gold always points at a real option, and qids are unique
        assert all(0 <= q.gold_index < 4 for q in kept)
        assert len({q.qid for q in kept}) == len(kept)
