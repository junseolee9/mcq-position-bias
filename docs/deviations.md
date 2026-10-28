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

`datasets` is also dropped, though `CLAUDE.md` Phase 0 lists it. Nothing needs
it: the preprocessed MMLU / ARC / CSQA CSVs ship inside the upstream submodule
(8 MB of data, 13 MB submodule total) and upstream reads them with
`pandas.read_csv`, so no task data is ever fetched from the Hub. Only model
weights are downloaded, into `$SCRATCH/hf_cache`.

Exact remote versions are captured in `docs/requirements.remote.lock.txt` by
`scripts/setup_remote.sh`.

## 2026-10-28 — Phase 1: `CLAUDE.md` scoring spec corrected to match upstream

`CLAUDE.md` specifies "logits of the 4 label tokens, softmax over those 4 only".
Upstream does not do this. `eval_clm_utils.py:197-202` builds **8** ids — the
space-prefixed and bare variant of each label — softmaxes over all 8, then sums
the two variants per label:

```python
option_indices = [toker(f': {e}').input_ids[-1] for e in option_ids] + \
    [toker(f':{e}').input_ids[-1] for e in option_ids]
probs = F.softmax(logits[..., option_indices], dim=-1)...
probs = probs.reshape(2, len(option_ids)).sum(axis=0)
```

We follow **upstream**, not `CLAUDE.md`, since ground rule 2 requires our scoring
to match the paper's. `score.py` therefore softmaxes over 8 ids and sums pairs;
`p_A..p_D` in the output schema are the summed values. A 4-id softmax is a
different estimator and would change both accuracy and RStd. For LLaMA all eight
ids are single tokens (`docs/upstream_notes.md` §6), so the scheme is well
defined; this must be re-checked per model in Phase 4.

## 2026-10-28 — Phase 1: invalid items are filtered after inference, not before

`CLAUDE.md` has `data.py` dropping the "all of the above" / "A and B" items at
load time, and asks whether the preprocessed data already excludes them. It does
not: the shipped CSVs contain them, inference runs over all of them, and upstream
drops them only when re-reading results (`utils.load_results`, `utils.py:80-85`).
Applying upstream's own lists to the shipped MMLU CSVs filters 450 / 14042 =
**3.20%**, matching the paper's stated 3.2%.

We keep `CLAUDE.md`'s design — filter in `data.py`, before inference — because
rotating an "all of the above" item is meaningless and paying for 16 versions of
it is waste. This is a deviation in *when* the filter applies, not *what* it
filters; the lists and their matching semantics are copied verbatim.

Resolved from the released results (`docs/reproduction.md`): the paper reports on
the **filtered** 13592 rows — `Llama-2-7b-hf` gives 35.80 / 22.95 filtered versus
36.06 / 22.51 unfiltered, and Table 3 says 35.8 / 23.0. So filtering before
inference costs us nothing we compare against; we simply never produce the
unfiltered number that `eval_clm.py` prints.

## 2026-10-28 — Pilot: proposed main design changed from `full16` to `balanced1`

`CLAUDE.md` makes `full16` the default for main results. The pilot on the
authors' released `perm` data (`docs/pilot_released_data.md` §5) shows this buys
nothing for 16x the cost:

- The conditional logit identifies label and position *within* a choice set of 4
  alternatives, so its standard errors track the number of choice sets — i.e.
  forward passes — not versions per question. At a fixed 12000 forward passes,
  SEs are flat from 1 to 24 versions per question (0.0322 vs 0.0342 for slot 1).
- For the marginal tables (pick rate, recall, RStd), spreading the same budget
  over more questions is strictly better, since question-level variance
  dominates: sd of the max-min pick rate is 0.68 at 1 version/question against
  2.36 at 24.

Proposed: `balanced1` over all 13592 items x 5 seeds for the headline numbers,
and `full16` on a ~1000-question subset as the within-question robustness check
(it remains the only design that puts every label at every position for the
*same* question). Not yet acted on — `CLAUDE.md` is unchanged and the decision is
to be confirmed in Phase 5 against our own smoke-test output before Phase 6
launches.

## 2026-10-28 — Open questions raised by reading upstream's run script

Not deviations yet; flagged for Phase 1/2 to resolve, since each could change
what "reproducing the baseline" means.

- ~~`scripts/run_llama-7b.sh` evaluates `huggyllama/llama-7b` (LLaMA-1 7B), not
  `meta-llama/Llama-2-7b-hf`. Which model is Table 3's RStd 23.0 / Acc 35.8?~~
  **Resolved** from the released results: it is `Llama-2-7b-hf` on the filtered
  population, 35.80 / 22.95 (`docs/reproduction.md`). `CLAUDE.md` was right. The
  intermediate guess that it might be LLaMA-1 — whose PriDe-*debiased* accuracy
  is 35.34, coincidentally near 35.8 — was wrong; LLaMA-1's baseline is
  33.75 / 18.52.
- ~~Upstream already implements `cyclic` and `perm`.~~ Resolved in Phase 1
  (`docs/upstream_notes.md` §4): both move option content with the labels left in
  `A B C D`, so they vary gold position but never which label sits at which
  position. `shuffle_both` does permute labels, but as a single content-seeded
  draw per question, unbalanced, with the label column out of order. The crossed
  label x position design is genuinely new; Phase 3 states this in `rotate.py`.
- ~~Tokenizer sets `add_bos_token=False`.~~ Resolved: confirmed at
  `eval_clm.py:51-54` and verified empirically against the LLaMA tokenizer
  (`docs/upstream_notes.md` §6). No BOS, every label a single token.
- New: upstream never computes RStd for a default-order run — its RStd lives
  inside the PriDe pipeline and consumes `perm`/`cyclic` probabilities
  (`debias_pride.py:152-154`). Phase 2 must compute the baseline RStd itself with
  upstream's recipe (per-gold-label recall, x100, `np.std`, `ddof=0`).
