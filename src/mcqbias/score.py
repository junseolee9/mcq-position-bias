"""Next-token scoring over the label letters, exactly as the paper does it.

# adapted from LLM-MCQ-Bias/code/eval_clm_utils.py@2ae2f40 (prepare_eval_fn_base, :176-219)

Upstream softmaxes over **eight** token ids — the space-prefixed and the bare
variant of each label — and sums the two variants per label. CLAUDE.md specifies
a 4-id softmax; that is a different estimator and would change both accuracy and
RStd, so we follow upstream (docs/deviations.md, 2026-10-28).

The returned probabilities are indexed by *label* A..D, not by display position.
With a rotated label order the caller maps the predicted label back to a position
through `label_order`.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

LABELS = "ABCD"
MAX_INPUT_TOKENS = 1536  # eval_clm_utils.py:191, keeps the LAST 1536 tokens


@dataclass(frozen=True)
class LabelTokens:
    ids: list[int]  # 8 ids: ': A'..': D' then ':A'..':D'
    has_space_prefix: bool
    labels: str = LABELS

    @property
    def needs_trailing_space(self) -> bool:
        """eval_clm_utils.py:187-188 — only when the tokenizer cannot tell the two apart."""
        return not self.has_space_prefix


def label_tokens(tokenizer, labels: str = LABELS) -> LabelTokens:
    """Derive and validate the 8 ids for one tokenizer.

    Required per model by CLAUDE.md: every label must be a single token in both
    variants, otherwise `input_ids[-1]` is a fragment and the 8-id scheme is
    scoring something other than the label.
    """
    spaced = [tokenizer(f": {e}").input_ids for e in labels]
    bare = [tokenizer(f":{e}").input_ids for e in labels]
    colon = tokenizer(":").input_ids

    for label, probe in zip(labels + labels, spaced + bare):
        # The probe is ':' plus the label, so exactly one token beyond the colon's
        # own encoding. Longer means the label split; shorter means it merged into
        # the colon, and then input_ids[-1] is not the label at all. Comparing
        # against the colon's own length keeps this valid whether or not the
        # tokenizer prepends BOS. Upstream would silently score the wrong id here;
        # we refuse instead, which only differs on tokenizers where the 8-id
        # scheme is undefined anyway.
        if len(probe) != len(colon) + 1:
            raise ValueError(
                f"label {label!r} is not a single token for this tokenizer: "
                f"{tokenizer.convert_ids_to_tokens(probe)}"
            )

    ids = [p[-1] for p in spaced] + [p[-1] for p in bare]
    has_space_prefix = ids[: len(labels)] != ids[len(labels) :]
    if not has_space_prefix and len(set(ids)) != len(labels):
        raise ValueError(f"label ids collide: {ids}")
    return LabelTokens(ids=ids, has_space_prefix=has_space_prefix, labels=labels)


def _accepts_position_ids(model) -> bool:
    """OPT and a few others derive positions from the mask and take no such argument."""
    import inspect

    return "position_ids" in inspect.signature(model.forward).parameters


@torch.no_grad()
def score(model, tokenizer, prompts: list[str], tokens: LabelTokens):
    """Return an (n_prompts, n_labels) array of label probabilities.

    Left padding, so that `logits[:, -1]` is the position after 'Answer:' for
    every row in the batch. Upstream runs batch size 1 and needs no padding, so
    batching is a deviation in numerics and needs care in two places.

    `position_ids` are derived from the attention mask rather than left to
    default to `arange(seq_len)`. Without this, left padding shifts every real
    token right by the pad count, which a model with learned absolute position
    embeddings reads as a different prompt — measured at up to 10% on the label
    probabilities with a GPT-2 tokenizer/model, while the unpadded row of the
    same batch matched exactly. The twelve models we run use RoPE or ALiBi, where
    a constant offset cancels, so they would have been fine; passing the ids
    removes the question instead of relying on that.

    The remaining difference is reduction order, which is why the equivalence is
    measured and not assumed: `tests/test_score.py` checks the batched path
    against batch size 1, and the same check is required on the real model before
    any long run (docs/reproduction.md).
    """
    if tokens.needs_trailing_space:
        prompts = [p + " " for p in prompts]

    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    batch = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = batch.input_ids[..., -MAX_INPUT_TOKENS:].to(model.device)
    attention_mask = batch.attention_mask[..., -MAX_INPUT_TOKENS:].to(model.device)

    kwargs = {}
    if _accepts_position_ids(model):
        position_ids = attention_mask.cumsum(-1) - 1
        kwargs["position_ids"] = position_ids.masked_fill(attention_mask == 0, 0)

    logits = model(input_ids=input_ids, attention_mask=attention_mask, **kwargs).logits[:, -1]
    probs = F.softmax(logits[..., tokens.ids], dim=-1).to(torch.float32).cpu().numpy()
    n = len(tokens.labels)
    return probs.reshape(-1, 2, n).sum(axis=1)  # sum the spaced and bare variant
