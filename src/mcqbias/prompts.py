"""The paper's Figure 6 prompt, with the label order made a parameter.

# adapted from LLM-MCQ-Bias/code/eval_clm_utils.py@2ae2f40
#   create_user_prompt :101-118, sys_msg :93-98, assembly in prepare_eval_fn_base :181-186

Byte-for-byte identical to upstream when the label order is "ABCD"; the only
substitution is which label letter is printed on which line. Verified against
upstream's own `prepare_eval` output (docs/upstream_notes.md §3) by
tests/test_prompts.py.

The trailing-space rule (`bpe_has_space_prefix`, eval_clm_utils.py:177, :187-188)
is tokenizer-dependent and lives in score.py, not here.
"""

from __future__ import annotations

MMLU_SYS = "The following are multiple choice questions about {}."
OTHER_SYS = "The following are multiple choice questions."
SUFFIX = " You should directly answer the question by choosing the correct option."


def system_message(task: str, subject: str) -> str:
    """eval_clm_utils.py:93-98, :145. MMLU names the subject; ARC and CSQA do not."""
    template = (MMLU_SYS if "mmlu" in task else OTHER_SYS) + SUFFIX
    return template.format(subject.replace("_", " "))


def user_prompt(question: str, contents, label_order: str) -> str:
    """eval_clm_utils.py:113-117, with `label_order` replacing the fixed option_ids.

    `contents` and `label_order` are both in display order, top to bottom.
    """
    lines = "\n".join(
        f"{label}. {content}".strip()
        for label, content in zip(label_order, contents)
    )
    return f"Question: {question.strip()}\nOptions:\n" + lines + "\nAnswer:"


def build(task: str, subject: str, question: str, contents, label_order: str) -> str:
    """The full 0-shot input text, exactly as prepare_eval_fn_base assembles it."""
    return system_message(task, subject) + "\n\n" + user_prompt(question, contents, label_order)
