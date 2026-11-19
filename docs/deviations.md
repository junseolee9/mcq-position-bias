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

## 2026-11-16 — Compute is Jetstream2, not an ACCESS JupyterLab node

The Phase 0 entry above recorded "an A100 through an NSF ACCESS allocation accessed via
JupyterLab". The actual machine is a Jetstream2 (OpenStack) instance named `mcq-team`, driven
by a `js2` CLI wrapper, shared by five teammates:

- Flavor `g3.medium` = **A100 10 GB vGPU slice** (8 vCPU, 30 GB RAM), 16 SU/h. Not a
  dedicated A100 40 GB. `~1.3` GB of the slice is reserved, leaving ~8.7 GB usable.
- The 60 GB root disk has ~11 GB free, and only `/workspace` survives shelving: it is the
  volume `mcq-team-vol` (100 GB). Everything — venvs, HF cache, results — lives there.
- Billing is per hour ACTIVE, so the instance is shelved (`js2 stop`) between runs. An idle
  ACTIVE instance costs more than the runs do.
- The allocation exposes `g3.medium` / `g3.large` / `g3.xl` only. There is no H100 flavor,
  so nothing above 13B is reachable at bf16 on this project at all.

Team budget is a hard 10,000 SU with no top-up, shared five ways.

`scripts/setup_remote.sh` and the README's remote section were written for the JupyterLab
assumption (`$SCRATCH`, a `.bashrc` export, an `ipykernel` registration) and are replaced.
No reported number changes; what changes is the timing and SU accounting, and the model set
below.

## 2026-11-16 — Model set: 12 models on a 20 GB slice, varied by variant rather than size

`CLAUDE.md` specifies `Llama-2-7b-hf`, `Llama-3.2-3B` and `Qwen2.5-7B` on one A100 40 GB. Two
constraints replace that:

1. bf16 on the 10 GB slice caps out near 3.5B parameters, so no 7B model loads at all. The
   instance is resized once to `g3.large` (A100 20 GB slice, 18.7 GB usable, 32 SU/h), which
   fits 7B but not 13B (26 GB).
2. **Every family the paper evaluates has 7B as its smallest member.** From
   `debias_pride.py:25-46`: LLaMA-1 7/13/30/65B, Llama-2 7/13/70B (+chat), Vicuna v1.3
   7/13/33B, Vicuna v1.5 7/13B, Falcon 7/40B (+instruct), plus GPT-3.5 via API. There is no
   sub-7B member anywhere in the paper's set, so a per-family *size* ladder is impossible
   here — and Falcon's second size is 40B, so it stays single-size even on a 40 GB card.

So the design varies **variant** within family instead of size, 4 families x 3 members:

| Family | Members |
|---|---|
| Meta Llama | `huggyllama/llama-7b`, `meta-llama/Llama-2-7b-hf`, `meta-llama/Llama-2-7b-chat-hf` |
| Vicuna | `lmsys/vicuna-7b-v1.3`, `lmsys/vicuna-7b-v1.5`, `lmsys/vicuna-7b-v1.5-16k` |
| Falcon | `tiiuae/falcon-7b`, `tiiuae/falcon-7b-instruct`, `tiiuae/falcon-rw-7b` |
| Qwen2.5 | `Qwen/Qwen2.5-7B`, `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-3B` |

Ten of the twelve are the paper's own models. Two consequences worth stating:

- The base-vs-instruction-tuned contrast is now a first-class axis. The paper evaluates
  `Llama-2-chat` and `vicuna` but never separates what instruction tuning does to selection
  bias, so this is a question our design answers and theirs does not.
- Phase 2's reproduction target is inside the main run set, so no separate baseline run is
  needed: `Llama-2-7b-hf` MMLU 0-shot `default` mode should land on the 35.80 / 22.95 we
  already recovered from the released results (`docs/reproduction.md`).

`CLAUDE.md` is left unedited, per ground rule 2; this entry is the record.

## 2026-11-16 — Two venvs on the instance, and `use_fast=False` cannot hold for all models

`transformers==4.36.2` (pinned above to the paper's era) cannot load Qwen2.5 — `Qwen2ForCausalLM`
landed in 4.37. Falcon and Qwen also ship fast-only tokenizers (`tokenizer.json`, no
sentencepiece model), so upstream's `AutoTokenizer.from_pretrained(..., use_fast=False)`
(`eval_clm.py:51-54`) cannot be applied to all twelve models.

Resolution, with the split kept explicit:

- `/workspace/venv-upstream`, pinned at `requirements.txt`, exists only to run upstream's
  `eval_clm.py` unmodified for the pipeline-equivalence check. `huggyllama/llama-7b` loads in
  both environments, so it is the model used for that comparison.
- `/workspace/venv`, modern `transformers`, runs our own `src/mcqbias/` code for all twelve
  models.

Where a model has both a slow and a fast tokenizer (the LLaMA and Vicuna checkpoints), the
eight label-token ids are compared between the two and must agree; the result goes in
`docs/upstream_notes.md` §6 per model. Where only a fast tokenizer exists (Falcon, Qwen), the
single-token requirement for `' A'`/`'A'` is still checked before the model is run, since the
8-id scoring scheme is undefined otherwise.

## 2026-11-16 — Phase 4: batched inference, and `position_ids` under left padding

Upstream runs batch size 1 under a thread pool (`eval_clm.py:82`,
`eval_clm_utils.py:190-195`) and never pads. We batch, because the crossed design
multiplies the number of forward passes. That is a numerical deviation in two
distinct ways, and one of them was a real bug caught by the test rather than by
reasoning:

1. **Absolute positions.** With left padding and no explicit `position_ids`,
   `forward()` defaults to `arange(seq_len)`, so every real token in a padded row
   is shifted right by the pad count. On a GPT-2 model — learned absolute
   position embeddings — this moved the label probabilities by up to 10% relative,
   while the single unpadded row of the same batch matched batch size 1 exactly.
   `score.py` now derives `position_ids` from the attention mask
   (`cumsum - 1`, masked to 0 on pads) for any model whose `forward` accepts them;
   OPT-style models compute positions from the mask themselves and take no such
   argument. The twelve models we actually run use RoPE or ALiBi, where a constant
   offset cancels, so they were never affected — but the fix costs two lines and
   removes the question.
2. **Reduction order.** Padding changes the order of the summations inside the
   kernels, so bitwise equality is not expected. `tests/test_score.py` asserts
   agreement with batch size 1 to `atol=1e-5` in float32 on CPU, and
   `docs/reproduction.md` requires the same comparison on the real model and dtype
   before any long run.

Also recorded per run, beyond CLAUDE.md's schema: `model_revision` (the Hub
snapshot hash actually loaded), `dtype` and `batch_size`. `docs/reproduction.md`
names all three as candidate sources of difference from the released numbers, so
they have to be in the data, not in a notebook.

One stricter-than-upstream check: `score.label_tokens` refuses a tokenizer where
`': X'` or `':X'` is not exactly one token past the colon's own encoding. Upstream
would take `input_ids[-1]` regardless and silently score a fragment or a merged
`':X'` token. This only differs on tokenizers where the 8-id scheme is undefined
anyway, and all twelve models pass it.

## 2026-11-18 — Phase 6 plumbing: truncation clamp, OOM retry, resume

None of this changes what the model is shown; it is recorded because two of the
three were found by a failure rather than by reading, and because the first
touches the prompt.

**Truncation is clamped to what the model can address.** Upstream hardcodes
`input_ids[..., -1536:]` (`eval_clm_utils.py:191`), which is safe for its own
models since LLaMA-1 addresses 2048 positions. `score.max_input_tokens` takes
`min(1536, max_position_embeddings)` instead. For all twelve models in our run
set this returns exactly **1536**, so the prompt is upstream's and nothing is
deviating; `tests/test_score.py` pins that by reading every config in
`configs/models.yaml`. The clamp exists because a model with a shorter context
does not truncate, it raises `IndexError: index out of range in self` inside its
position embedding — which is how it was found, running the sweep end to end on
a 512-position test model.

**Out of memory halves the batch and retries**, which is what upstream does on
`RuntimeError` (`utils.py:185-192`). The per-model `batch_size` in
`configs/models.yaml` is a starting point sized for the 20 GB slice: logits are
computed at every position, so the cost scales with vocabulary, and Qwen2.5's
152k vocabulary makes an item roughly five times more expensive than LLaMA's 32k
one. The batch size actually in force is recorded per row, so a run that halved
mid-way is visible in the data rather than only in the log.

**Results are flushed to shards and runs resume on `(qid, version)`.** Holding
68k rows in memory and writing once at the end meant a dropped SSH session cost
the whole run, and the instance is billed by the hour. One bug here was real and
silent: shard files were numbered by *counting* the existing shards, so as soon
as the indices were not contiguous — after discarding an unreadable shard, or on
any resume — every later flush overwrote the same file and most of the run's
output was lost. Numbering is now `max(index) + 1`.

## 2026-11-19 — Flavor g3.xl, and the run set regains a size axis

Two findings forced this, one practical and one about what the design can ask.

**The 10 GB slice could not be brought back up at all.** `js2 start mcq-team`
hung, and the instance's own event log says why: `schedule_instances` succeeded,
then `compute_unshelve_instance` returned `Error` after three minutes and the
instance fell back to `SHELVED_OFFLOADED`. The flavor's `extra_specs` carry
`pci_passthrough:alias: A100X_10C:1`, so the scheduler found a host advertising
the 10 GB vGPU profile and the compute node then failed to claim the device.
Jetstream2 hides the traceback from tenants. The instance history shows
back-to-back unshelve attempts on 2026-11-17 as well, so this is not new.
The resize path (`openstack server resize`) requires an unshelved instance and
was therefore unreachable; the instance is recreated instead, which schedules
fresh rather than restoring a shelved image.

**Moving to `g3.xl` changes the experiment, not just the hardware.** It is a full
A100 40 GB (~38.7 GB usable, 32 vCPU, 117 GB RAM, 64 SU/h) rather than a vGPU
slice, so the 13B tier fits. The 20 GB plan recorded above had to vary *variant*
within a family because every family in the paper has 7B as its smallest member;
at 40 GB three of the four families carry a size step, and "does the bias shrink
with scale" becomes a question we can ask instead of one we had to drop.

The set is still 4 families x 3 members:

| Family | Members | Within-family axis |
|---|---|---|
| Meta Llama | `huggyllama/llama-7b`, `Llama-2-7b-hf`, `Llama-2-13b-hf` | generation + size |
| Vicuna | `vicuna-7b-v1.3`, `vicuna-7b-v1.5`, `vicuna-13b-v1.5` | base generation + size |
| Falcon | `falcon-7b`, `falcon-7b-instruct`, `falcon-rw-7b` | variant + pretraining data |
| Qwen2.5 | `Qwen2.5-3B`, `-7B`, `-14B` | size, three points |

LLaMA-1 cannot be its own family because its next size is 30B (60 GB), so Meta
Llama spans both generations, as in the 20 GB plan.

Three properties worth stating, since they are why these twelve and not others:

- **The instruction-tuning axis costs no slot.** `vicuna-7b-v1.3` is a fine-tune
  of `llama-7b` and `vicuna-7b-v1.5` of `Llama-2-7b`, so families 1 and 2 pair
  into two matched base/tuned comparisons at identical size, and Falcon carries a
  third. Spending slots on `Llama-2-*-chat-hf` would have bought a more direct
  RLHF contrast at the cost of the size axis; that is the first thing to add if
  the set ever grows past twelve.
- **Tokenizers differ on purpose**: 32k sentencepiece (6 models), 65k BPE (3),
  152k BPE (3). The paper's claim is about token bias, and a set sharing one
  tokenizer could not separate a tokenizer artefact from a general label
  preference.
- **Both load-bearing models from earlier work survive**: `llama-7b`, where the
  pilot found a position span of 0.81 against a label span of 0.91, and
  `Llama-2-7b-hf`, whose `default` condition is the Table 3 reproduction.

**Cost.** Parameter sum 100.4B against 80.6B for the 20 GB set, so 1.25x the
compute at the same 75,028 passes per model: 3,314 PFLOP, about 1,055 SU
realistically and 1,685 SU worst case against the team's 10,000 cap. The flavor
itself is close to cost-neutral — twice the rate, roughly twice the throughput —
so the increase comes from running 13B and 14B models. Weights now total 201 GB
against a 100 GB volume, so `run_sweep.py`'s eviction is mandatory rather than
tidy; the largest single checkpoint, `Qwen2.5-14B` at 29.6 GB, fits alone.

**Access.** Only the two `meta-llama` repos are gated (`gated: manual`, reviewed
per account). One Llama-2 approval covers the collection: verified on
2026-11-19 that the same token reaches `Llama-2-7b-hf`, `-7b-chat-hf`,
`-13b-hf` and `-13b-chat-hf`. No further requests are needed for this set.
