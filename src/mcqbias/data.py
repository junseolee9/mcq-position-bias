"""Load the paper's preprocessed MCQ data and drop the items rotation cannot touch.

# adapted from LLM-MCQ-Bias/code/eval_clm_utils.py@2ae2f40 (prepare_eval_samples, :136-163)
# and LLM-MCQ-Bias/code/utils.py@2ae2f40 (load_results' skip_bad_options branch, :80-85)

Upstream drops the "A and B" / "all of the above" items only when it re-reads
results, after inference. We drop them here instead, before inference: rotating
an option that refers to another option is meaningless, and paying for 16
versions of it is waste. The lists and their matching semantics are upstream's,
unchanged, and the paper reports on this same filtered population
(docs/reproduction.md). The deviation in *when* the filter runs is logged in
docs/deviations.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_CODE = REPO_ROOT / "third_party/LLM-MCQ-Bias/code"

# Copied rather than imported: importing from third_party at runtime needs a
# sys.path insert and makes CPython drop __pycache__ into that tree, which ground
# rule 1 forbids. tests/test_upstream_facts.py asserts these stay identical to
# upstream's own lists, so drift fails loudly instead of silently.
# adapted from LLM-MCQ-Bias/code/utils.py@2ae2f40 :17-32
BAD_OPTIONS = [
    "A and B",
    "A and C",
    "A and D",
    "A nor B",
    "A or B",
    "B and C",
    "B and D",
    "C and D",
]
REFER_OPTIONS = [
    "all of the above",
    "all of above",
    "none of the above",
    "none of above",
]

LABELS = "ABCD"


def is_invalid(options) -> bool:
    """utils.load_results:80-85 — BAD_OPTIONS case-sensitive, REFER_OPTIONS lowered."""
    return any(e in o for e in BAD_OPTIONS for o in options) or any(
        e in o.lower() for e in REFER_OPTIONS for o in options
    )


@dataclass(frozen=True)
class Question:
    qid: str  # f"{subject}:{row}", stable across runs and modes
    subject: str
    question: str
    options: tuple[str, str, str, str]  # in the CSV's original order
    gold_index: int  # index into options, 0-3


def _subjects(task_dir: Path) -> list[str]:
    """eval_clm_utils.py:89-90."""
    return sorted(
        f.name.split("_test.csv")[0] for f in task_dir.glob("*_test.csv")
    )


def load(task: str, data_root: Path | None = None) -> tuple[list[Question], int]:
    """Return (kept questions, number dropped by the invalid-option filter)."""
    root = (data_root or UPSTREAM_CODE) / f"data_{task}"
    if not root.is_dir():
        raise FileNotFoundError(
            f"{root} missing. The task CSVs ship in the upstream submodule: "
            "git submodule update --init --recursive"
        )

    kept: list[Question] = []
    dropped = 0
    for subject in _subjects(root / "test"):
        df = pd.read_csv(
            root / "test" / f"{subject}_test.csv",
            names=("Question", *LABELS, "Answer"),
            dtype=str,
        )
        for row, record in df.iterrows():
            options = tuple(str(record[e]) for e in LABELS)
            if is_invalid(options):
                dropped += 1
                continue
            kept.append(
                Question(
                    qid=f"{subject}:{row}",
                    subject=subject,
                    question=record["Question"],
                    options=options,
                    gold_index=LABELS.index(record["Answer"]),
                )
            )
    return kept, dropped
