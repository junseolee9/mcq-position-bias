"""Scoring checks that do not need the GPU.

The batched path is a deviation from upstream's batch size 1 (padding shifts
absolute positions and changes reduction order), so it is measured, not assumed.
A tiny random model is enough: the question is whether batching changes the
numbers, not whether the numbers are good.

The model tests skip without network or a cache; the tokenizer-contract tests
run against the real LLaMA tokenizer that upstream used.
"""

import numpy as np
import pytest

from mcqbias.score import LABELS, label_tokens, score

TINY = "hf-internal-testing/tiny-random-gpt2"
LLAMA = "huggyllama/llama-7b"  # tokenizer only, ungated, the model in upstream's run script

PROMPTS = [
    "The following are multiple choice questions about anatomy."
    " You should directly answer the question by choosing the correct option.\n\n"
    "Question: Which bone is the longest?\nOptions:\nA. femur\nB. rib\nC. ulna\nD. stapes\nAnswer:",
    # deliberately much shorter, so left padding actually has to pad
    "Question: 1 + 1?\nOptions:\nA. 1\nB. 2\nC. 3\nD. 4\nAnswer:",
    "Question: Pick C.\nOptions:\nA. no\nB. no\nC. yes\nD. no\nAnswer:",
]


def _load(model_id, model=False):
    transformers = pytest.importorskip("transformers")
    try:
        toker = transformers.AutoTokenizer.from_pretrained(model_id)
        if not model:
            return toker
        import torch

        m = transformers.AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float32
        )
        m.eval()
        return toker, m
    except Exception as exc:  # offline, gated, or no cache
        pytest.skip(f"{model_id} unavailable: {type(exc).__name__}: {exc}")


def test_llama_label_tokens_match_the_recorded_ids():
    """docs/upstream_notes.md §6 — pinned so a tokenizer change cannot pass silently."""
    toker = _load(LLAMA)
    tokens = label_tokens(toker)
    assert tokens.has_space_prefix is True
    assert tokens.needs_trailing_space is False
    assert tokens.ids == [319, 350, 315, 360, 29909, 29933, 29907, 29928]


def test_label_tokens_rejects_a_label_that_is_not_one_token():
    class SplittingTokenizer:
        """One token per character: ':' is 1, so ': A' must be 2, but it is 3."""

        def __call__(self, text):
            return type("Enc", (), {"input_ids": [1] * len(text)})()

        def convert_ids_to_tokens(self, ids):
            return ["?"] * len(ids)

    with pytest.raises(ValueError, match="not a single token"):
        label_tokens(SplittingTokenizer())


def test_batched_scoring_matches_batch_size_one():
    toker, model = _load(TINY, model=True)
    tokens = label_tokens(toker)

    batched = score(model, toker, PROMPTS, tokens)
    one_at_a_time = np.vstack([score(model, toker, [p], tokens) for p in PROMPTS])

    assert batched.shape == (len(PROMPTS), len(LABELS))
    # float32 on CPU: padding changes reduction order, not the answer
    np.testing.assert_allclose(batched, one_at_a_time, atol=1e-5)
    assert (batched.argmax(axis=1) == one_at_a_time.argmax(axis=1)).all()


def test_probabilities_are_a_distribution_over_the_four_labels():
    toker, model = _load(TINY, model=True)
    probs = score(model, toker, PROMPTS, label_tokens(toker))
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    assert (probs >= 0).all()
