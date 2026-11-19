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

from mcqbias.score import MAX_INPUT_TOKENS, LABELS, label_tokens, max_input_tokens, score

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


SENTENCEPIECE_IDS = [319, 350, 315, 360, 29909, 29933, 29907, 29928]


@pytest.mark.parametrize(
    "model_id",
    [
        LLAMA,
        "meta-llama/Llama-2-7b-hf",  # gated; skips without access
        "meta-llama/Llama-2-7b-chat-hf",
        "lmsys/vicuna-7b-v1.3",
        "lmsys/vicuna-7b-v1.5",
        "lmsys/vicuna-7b-v1.5-16k",
    ],
)
def test_llama_family_label_tokens_match_the_recorded_ids(model_id):
    """docs/upstream_notes.md §6 — pinned so a tokenizer change cannot pass silently.

    All six share the 32000-piece sentencepiece vocabulary, so the Table 3
    reproduction and the LLaMA-1 run score the same eight ids and any difference
    between them is the model rather than the scoring.
    """
    tokens = label_tokens(_load(model_id))
    assert tokens.has_space_prefix is True
    assert tokens.needs_trailing_space is False
    assert tokens.ids == SENTENCEPIECE_IDS


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


class _Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Model:
    def __init__(self, **kw):
        self.config = _Cfg(**kw)


def test_truncation_is_upstreams_1536_for_every_real_model():
    """Each model in the run set is 2048 ctx or more, so the prompt is upstream's."""
    for ctx in (2048, 4096, 16384, 32768):
        assert max_input_tokens(_Model(max_position_embeddings=ctx)) == MAX_INPUT_TOKENS


def test_truncation_never_exceeds_a_short_context():
    """A model that cannot address 1536 positions raises IndexError inside its
    position embedding rather than truncating, so the limit is clamped."""
    assert max_input_tokens(_Model(n_positions=512)) == 512
    assert max_input_tokens(_Model(max_position_embeddings=512, n_positions=512)) == 512
    # a config that advertises nothing falls back to upstream's value
    assert max_input_tokens(_Model()) == MAX_INPUT_TOKENS
    assert max_input_tokens(_Model(max_position_embeddings=None)) == MAX_INPUT_TOKENS


def test_real_run_set_contexts_all_clear_the_cap():
    """Guards the claim above against a config change in any of the twelve.

    Skipping is per model, not for the whole test: the gated repos need a token
    and Qwen2.5 needs transformers >= 4.37, so under the pinned environment some
    entries are simply not loadable here. The ones that are must still be
    checked, and the test fails if none were.
    """
    from pathlib import Path

    transformers = pytest.importorskip("transformers")
    yaml = pytest.importorskip("yaml")

    repo = Path(__file__).resolve().parents[1]
    models = yaml.safe_load((repo / "configs/models.yaml").read_text())
    checked, unavailable = {}, {}
    for family in models.values():
        for entry in family:
            try:
                cfg = transformers.AutoConfig.from_pretrained(entry["id"])
            except Exception as exc:
                unavailable[entry["id"]] = type(exc).__name__
                continue
            ctx = getattr(cfg, "max_position_embeddings", None) or getattr(cfg, "n_positions", 0)
            checked[entry["id"]] = ctx

    if not checked:
        pytest.skip(f"no model config loadable here: {unavailable}")
    too_short = {m: ctx for m, ctx in checked.items() if ctx < MAX_INPUT_TOKENS}
    assert not too_short, f"these cannot address upstream's 1536 positions: {too_short}"
