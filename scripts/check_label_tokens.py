"""Verify the 8 label-token ids for every model in configs/models.yaml.

CLAUDE.md requires this per model before any run. Tokenizers are a few hundred
kB, so this runs on a laptop in seconds and costs no GPU time — a model whose
labels are not single tokens cannot be scored the paper's way at all, and it is
far cheaper to learn that here than after a weight download on a billed instance.

    .venv/bin/python scripts/check_label_tokens.py            # all of them
    .venv/bin/python scripts/check_label_tokens.py --ungated  # skip gated repos

Paste the table it prints into docs/upstream_notes.md §6.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mcqbias.score import LABELS, label_tokens  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ungated", action="store_true", help="skip models needing HF_TOKEN")
    parser.add_argument("--models", nargs="*", help="check these ids instead of the config")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    if args.models:
        entries = [{"id": m} for m in args.models]
    else:
        config = yaml.safe_load((REPO_ROOT / "configs/models.yaml").read_text())
        entries = [m for family in config.values() for m in family]

    import transformers

    print(f"transformers {transformers.__version__}\n")
    print(f"{'model':38} {'tokenizer class':26} {'sp':6} {'ids (: X then :X)'}")

    failures, unsupported = [], []
    for entry in entries:
        model_id = entry["id"]
        if args.ungated and entry.get("gated"):
            print(f"{model_id:38} {'-':26} {'-':6} skipped (gated)")
            continue

        # Upstream passes use_fast=False (eval_clm.py:51-54). Not every repo ships a
        # slow tokenizer, so record the class we actually got rather than assume.
        toker = None
        for kwargs in ({"use_fast": False}, {}):
            try:
                toker = AutoTokenizer.from_pretrained(model_id, **kwargs)
                break
            except Exception as exc:
                last = exc
        if toker is None:
            # "Tokenizer class X does not exist" means this transformers is too old
            # for the architecture, not that the model is unusable. Qwen2.5 needs
            # >= 4.37 (docs/deviations.md 2026-11-16).
            note = f"{type(last).__name__}: {last}"
            if "does not exist or is not currently imported" in str(last):
                unsupported.append((model_id, note))
                print(f"{model_id:38} {'unsupported here':26} {'-':6} needs a newer transformers")
            else:
                failures.append((model_id, note))
                print(f"{model_id:38} {'LOAD FAILED':26} {'-':6} {note}")
            continue

        try:
            tokens = label_tokens(toker)
        except ValueError as exc:
            failures.append((model_id, str(exc)))
            print(f"{model_id:38} {type(toker).__name__:26} {'-':6} NOT SINGLE TOKENS: {exc}")
            continue

        print(f"{model_id:38} {type(toker).__name__:26} "
              f"{str(tokens.has_space_prefix):6} {tokens.ids}")

        if not tokens.has_space_prefix:
            # Then ': X' and ':X' give the same id, and score.py must append the
            # trailing space upstream adds (eval_clm_utils.py:187-188).
            print(f"{'':38} {'':26} {'':6} -> prompt needs a trailing space")

        spaced, bare = tokens.ids[: len(LABELS)], tokens.ids[len(LABELS):]
        for variant, ids in ((": X", spaced), (":X", bare)):
            if len(set(ids)) != len(LABELS):
                failures.append((model_id, f"{variant} ids collide: {ids}"))

    if unsupported:
        print("\nnot checkable under this transformers (run again in the modern venv):")
        for model_id, why in unsupported:
            print(f"  {model_id}: {why}")
    if failures:
        print("\nFAILURES — these models cannot be scored the paper's way:")
        for model_id, why in failures:
            print(f"  {model_id}: {why}")
        return 1
    print("\nevery checked model scores the four labels as single tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
