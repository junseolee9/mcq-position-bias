# Deviations

Every intentional departure from the paper's setup, and every post-hoc change to
a run, gets a dated entry here. Newest last.

## 2026-10-28 — Phase 0: compute environment

The spec assumes one A100 40 GB. Actual compute is an A100 through an NSF ACCESS
allocation accessed via JupyterLab, so exact GPU model and memory are recorded
per run instead of assumed. Development happens on a macOS arm64 machine with no
NVIDIA GPU; no reported number comes from it. Not a deviation from the paper's
method — recorded because it changes the timing estimates in `docs/timing.md`.

## 2026-10-28 — Phase 0: library versions pinned to upstream's era

Upstream ships no requirements file, so versions were inferred from the APIs its
code calls and pinned to the paper's era (`transformers==4.36.2`, `torch==2.1.2`,
`numpy==1.26.4`; full list in `requirements.txt`). Current releases were installed
first and rejected: under `transformers` 5.x, `eval_clm.py` breaks in two ways
that would force edits to code we are supposed to run unmodified.

- `AutoModelForCausalLM.from_pretrained(..., torch_dtype=...)` — the argument was
  renamed `dtype`.
- `AutoTokenizer.from_pretrained(..., use_fast=False)` — slow sentencepiece
  tokenizers were dropped, and the fast LLaMA tokenizer is not guaranteed to
  produce the same label-token ids, which is exactly what our scoring depends on.

`numpy` is held below 2.0 because `torch` 2.1.2 is built against the 1.x ABI.

Verified under these pins: upstream's `utils`, `eval_clm_utils`, `debias_utils`,
and `mmlu_categories` all import cleanly (`tests/test_env.py`). `eval_clm.py`
itself imports only on an NVIDIA host — it calls `pynvml.nvmlInit()` at module
level — so it is checked on the A100, not locally.

Two upstream dependencies are deliberately not installed: `openai`, needed only
by the `eval_ichat*` GPT-3.5 path we do not run, and `bitsandbytes`, since
`BitsAndBytesConfig` is imported but unused at bf16.

Exact remote versions are captured in `docs/requirements.remote.lock.txt` by
`scripts/setup_remote.sh`.

## 2026-10-28 — Open questions raised by reading upstream's run script

Not deviations yet; flagged for Phase 1/2 to resolve, since each could change
what "reproducing the baseline" means.

- `scripts/run_llama-7b.sh` evaluates `huggyllama/llama-7b` (LLaMA-**1** 7B), not
  the `meta-llama/Llama-2-7b-hf` named in `CLAUDE.md` Phase 2. Before claiming a
  reproduction gap against Table 3's RStd 23.0 / Acc 35.8, confirm which model
  that row reports.
- Upstream already implements `cyclic` and `perm` settings, i.e. option-content
  permutation with labels left in place. Our contribution is the *crossed* label
  rotation x content rotation; Phase 3 should state the difference explicitly and
  reuse upstream's setting where it coincides.
- Upstream's tokenizer is constructed with `add_bos_token=False`, which confirms
  the no-BOS convention assumed in `CLAUDE.md`. To be recorded with file and line
  in `docs/upstream_notes.md`.
