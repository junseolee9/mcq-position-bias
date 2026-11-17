"""Phase 3 tests (a), (b) and (d) from CLAUDE.md.

(c) — byte-for-byte agreement with upstream's prompt — lives in test_prompts.py,
where it can call upstream's own prepare_eval.
"""

from collections import Counter

import pytest

from mcqbias.rotate import LABELS, N, N_VERSIONS, render, shifts, versions_for

OPTIONS = ("alpha", "beta", "gamma", "delta")


@pytest.mark.parametrize("gold", range(N))
def test_full16_puts_each_label_at_each_position_four_times(gold):
    """(a)"""
    seen = Counter()
    for v in range(N_VERSIONS):
        r = render(OPTIONS, gold, *shifts(v))
        for position, label in enumerate(r.label_order):
            seen[(position, label)] += 1
    assert len(seen) == N * N
    assert set(seen.values()) == {N}


@pytest.mark.parametrize("gold", range(N))
def test_full16_balances_gold_over_positions_and_labels(gold):
    """(b)"""
    positions, labels = Counter(), Counter()
    for v in range(N_VERSIONS):
        r = render(OPTIONS, gold, *shifts(v))
        # the gold slot must still hold the gold content
        assert r.contents[r.gold_pos] == OPTIONS[gold]
        assert r.label_order[r.gold_pos] == r.gold_label
        positions[r.gold_pos] += 1
        labels[r.gold_label] += 1
    assert set(positions) == set(range(N)) and set(positions.values()) == {N}
    assert set(labels) == set(LABELS) and set(labels.values()) == {N}


@pytest.mark.parametrize("gold", range(N))
def test_default_version_is_the_untouched_order(gold):
    r = render(OPTIONS, gold, 0, 0)
    assert r.version == 0
    assert r.label_order == LABELS
    assert r.contents == OPTIONS
    assert (r.gold_pos, r.gold_label) == (gold, LABELS[gold])


def test_version_id_round_trips():
    assert [shifts(N * k + j) == (k, j) for k in range(N) for j in range(N)]
    assert [render(OPTIONS, 0, k, j).version == N * k + j for k in range(N) for j in range(N)]


def test_balanced1_is_balanced_over_the_dataset():
    """(d)"""
    n = N_VERSIONS * 37
    assigned = [vs[0] for vs in versions_for("balanced1", n, seed=3)]
    assert len(assigned) == n
    assert set(Counter(assigned).values()) == {37}


def test_balanced1_is_seed_dependent_but_reproducible():
    n = N_VERSIONS * 10
    a = [vs[0] for vs in versions_for("balanced1", n, seed=3)]
    assert a == [vs[0] for vs in versions_for("balanced1", n, seed=3)]
    assert a != [vs[0] for vs in versions_for("balanced1", n, seed=4)]


def test_balanced1_spreads_versions_when_n_is_not_a_multiple_of_16():
    assigned = [vs[0] for vs in versions_for("balanced1", N_VERSIONS * 2 + 5, seed=0)]
    counts = Counter(assigned)
    assert max(counts.values()) - min(counts.values()) <= 1


def test_mode_shapes():
    assert versions_for("full16", 3) == [list(range(N_VERSIONS))] * 3
    assert versions_for("default", 3) == [[0]] * 3
    with pytest.raises(ValueError):
        versions_for("full4", 3)
