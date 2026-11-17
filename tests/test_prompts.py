"""Phase 3 test (c): with label order ABCD our prompt is upstream's, byte for byte.

Generated, not pasted — this calls upstream's own `prepare_eval` and compares
every row of several MMLU subjects plus ARC. If upstream's template, system
message, spacing or `.strip()` placement differ from ours by one character, this
fails.

`prepare_eval` does `os.makedirs(f'results_{task}/...')` relative to the working
directory, so it runs in a tmp directory with the data directories symlinked in.
Ground rule 1: nothing is written inside third_party/.
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from mcqbias.prompts import build, system_message, user_prompt
from mcqbias.rotate import render, shifts

UPSTREAM_CODE = Path(__file__).resolve().parents[1] / "third_party/LLM-MCQ-Bias/code"
SUBJECTS = ("abstract_algebra", "anatomy", "us_foreign_policy")


@pytest.fixture(scope="module")
def upstream_prepare_eval(tmp_path_factory):
    """Upstream's prepare_eval, running in a sandbox with the data symlinked in."""
    sandbox = tmp_path_factory.mktemp("upstream_cwd")
    for task in ("mmlu", "arc"):
        (sandbox / f"data_{task}").symlink_to(UPSTREAM_CODE / f"data_{task}")

    sys.path.insert(0, str(UPSTREAM_CODE))
    cwd = os.getcwd()
    os.chdir(sandbox)
    try:
        from eval_clm_utils import prepare_eval

        yield prepare_eval
    finally:
        os.chdir(cwd)
        sys.path.remove(str(UPSTREAM_CODE))


def _upstream_samples(prepare_eval, task, subject):
    args = SimpleNamespace(model_name="prompt-equivalence-test")
    _subjects, _few_shot, prepare_eval_samples, _fn = prepare_eval(args, f"{task},0")
    return prepare_eval_samples(subject)


def _rows(task, subject):
    return pd.read_csv(
        UPSTREAM_CODE / f"data_{task}/test/{subject}_test.csv",
        names=("Question", *"ABCD", "Answer"),
        dtype=str,
    )


@pytest.mark.parametrize("task,subject", [("mmlu", s) for s in SUBJECTS] + [("arc", "arc")])
def test_our_prompt_matches_upstream_byte_for_byte(task, subject, upstream_prepare_eval):
    samples = _upstream_samples(upstream_prepare_eval, task, subject)
    rows = _rows(task, subject)
    assert len(samples) == len(rows) > 0

    for (upstream_input, upstream_options, _ideal), (_i, row) in zip(samples, rows.iterrows()):
        upstream_sys, upstream_user = upstream_input
        # prepare_eval_fn_base:182,186 assembles the 0-shot input as sys + '\n\n' + prompt
        expected = upstream_sys + "\n\n" + upstream_user

        ours = build(
            task,
            subject,
            row["Question"],
            tuple(str(row[e]) for e in "ABCD"),
            "ABCD",
        )
        assert ours == expected
        assert system_message(task, subject) == upstream_sys
        assert user_prompt(row["Question"], upstream_options, "ABCD") == upstream_user


def test_mmlu_system_message_names_the_subject_with_spaces():
    assert system_message("mmlu", "abstract_algebra").startswith(
        "The following are multiple choice questions about abstract algebra."
    )
    assert system_message("arc", "arc") == (
        "The following are multiple choice questions."
        " You should directly answer the question by choosing the correct option."
    )


def test_rotated_prompt_changes_only_the_label_column():
    question = "Find the degree for the given field extension."
    options = ("0", "4", "2", "6")
    base = build("mmlu", "abstract_algebra", question, options, "ABCD")
    base_contents = [line.split(". ", 1)[1] for line in base.splitlines()[-5:-1]]

    for v in range(16):
        r = render(options, 1, *shifts(v))
        text = build("mmlu", "abstract_algebra", question, r.contents, r.label_order)
        lines = text.splitlines()
        assert lines[0] == base.splitlines()[0]  # system message untouched
        assert lines[-1] == "Answer:"
        option_lines = lines[-5:-1]
        assert [line.split(". ", 1)[0] for line in option_lines] == list(r.label_order)
        assert sorted(line.split(". ", 1)[1] for line in option_lines) == sorted(base_contents)
        # the gold line carries the gold label
        assert option_lines[r.gold_pos] == f"{r.gold_label}. {options[1]}"
