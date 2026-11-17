"""Crossed label x content rotations.

The paper's own settings never cross these two. `perm` and `cyclic` move option
content while the label column stays A B C D top to bottom, so "label A" and
"position 0" are the same regressor; `shuffle_both` does permute labels but as a
single unbalanced content-seeded draw (docs/upstream_notes.md §4,
docs/pilot_released_data.md §1). Rotating the two independently is what lets a
label effect and a position effect be estimated separately.

Options sit at display positions p in 0..3, top to bottom.

  content shift j   content shown at position p is options[(p + j) % 4]
  label shift k     label shown at position p is "ABCD"[(p + k) % 4]
  version           v = 4 * k + j, so 16 versions per question

Four versions per question cannot balance label x position, gold x position and
gold x label simultaneously — with 4 options that is arithmetically impossible —
so the balanced designs are `full16` within question, or `balanced1` across the
dataset.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

LABELS = "ABCD"
N = 4
N_VERSIONS = N * N


@dataclass(frozen=True)
class Version:
    version: int
    label_shift: int  # k
    content_shift: int  # j
    contents: tuple[str, ...]  # what is printed, in display order
    label_order: str  # labels printed, in display order, e.g. "CDAB"
    gold_pos: int
    gold_label: str


def render(options, gold_index: int, k: int, j: int) -> Version:
    contents = tuple(options[(p + j) % N] for p in range(N))
    label_order = "".join(LABELS[(p + k) % N] for p in range(N))
    gold_pos = (gold_index - j) % N
    return Version(
        version=N * k + j,
        label_shift=k,
        content_shift=j,
        contents=contents,
        label_order=label_order,
        gold_pos=gold_pos,
        gold_label=label_order[gold_pos],
    )


def shifts(version: int) -> tuple[int, int]:
    """Inverse of v = 4k + j."""
    return divmod(version, N)


def versions_for(mode: str, n_questions: int, seed: int = 0) -> list[list[int]]:
    """Which version ids each question gets, in dataset order.

    full16     every question gets all 16
    balanced1  one version per question, round-robin over a seeded shuffle, so
               the design is balanced at dataset level for 1/16 the cost
    default    k = j = 0 only, the paper's original setting; our control
    """
    if mode == "full16":
        return [list(range(N_VERSIONS))] * n_questions
    if mode == "default":
        return [[0]] * n_questions
    if mode == "balanced1":
        order = list(range(n_questions))
        random.Random(seed).shuffle(order)
        assigned = [0] * n_questions
        for rank, qi in enumerate(order):
            assigned[qi] = rank % N_VERSIONS
        return [[v] for v in assigned]
    raise ValueError(f"unknown mode {mode!r}; expected full16, balanced1 or default")


def demo() -> None:
    options = ("alpha", "beta", "gamma", "delta")
    gold = 2

    # (a) in full16 each label appears at each position exactly 4 times
    # (b) gold lands at each position 4 times and carries each label 4 times
    at_position = {(p, lab): 0 for p in range(N) for lab in LABELS}
    gold_positions = {p: 0 for p in range(N)}
    gold_labels = {lab: 0 for lab in LABELS}
    for v in range(N_VERSIONS):
        k, j = shifts(v)
        r = render(options, gold, k, j)
        assert r.version == v
        assert r.contents[r.gold_pos] == options[gold], "gold content moved to the wrong slot"
        for p, lab in enumerate(r.label_order):
            at_position[(p, lab)] += 1
        gold_positions[r.gold_pos] += 1
        gold_labels[r.gold_label] += 1
    assert set(at_position.values()) == {N}, at_position
    assert set(gold_positions.values()) == {N}, gold_positions
    assert set(gold_labels.values()) == {N}, gold_labels

    # k = j = 0 is the paper's untouched order
    base = render(options, gold, 0, 0)
    assert base.label_order == LABELS and base.contents == options
    assert base.gold_pos == gold and base.gold_label == LABELS[gold]

    # (d) balanced1 is balanced over the dataset when n is a multiple of 16
    assigned = [vs[0] for vs in versions_for("balanced1", 16 * 50, seed=1)]
    counts = {v: assigned.count(v) for v in range(N_VERSIONS)}
    assert set(counts.values()) == {50}, counts
    # and the same seed must give the same assignment
    assert assigned == [vs[0] for vs in versions_for("balanced1", 16 * 50, seed=1)]
    print("ok")


if __name__ == "__main__":
    demo()
