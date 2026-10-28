# Upstream notes — LLM-MCQ-Bias @ 2ae2f40

Phase 1 deliverable. Read-only study of `third_party/LLM-MCQ-Bias/code`; nothing
there was edited. All line numbers are at commit `2ae2f40`.

Entry point for local causal LMs is `eval_clm.py`; `eval_ichat.py` is the
GPT-3.5 path and we do not run it.

---

## 1. Data loading

| What | Where |
|---|---|
| Subject list | `eval_clm_utils.py:89` — `sorted(f.split("_test.csv")[0] for f in os.listdir(f'{data_path}/test'))` |
| Test split | `eval_clm_utils.py:137` in `prepare_eval_samples` |
| Few-shot split | `eval_clm_utils.py:122` in `prepare_few_shot_samples` |
| Paths | `data_mmlu/`, `data_arc/`, `data_csqa/`, each with `dev/` + `test/` |

```python
pd.read_csv(f'{data_path}/test/{subject}_test.csv',
            names=("Question", *option_ids_header, "Answer"), dtype=str)
```

- Headerless CSV, columns `Question, A, B, C, D, Answer`; `dtype=str` throughout,
  so options are never coerced to numbers.
- `Answer` holds the gold **label letter**, not an index.
- MMLU: 57 subjects, 14042 test rows. It is the original MMLU release — there is
  no `data_mmlu/process.py`, unlike `data_arc/` and `data_csqa/` which have one.
- `option_ids = list('ABCD')`, except CSQA which uses `'ABCDE'`
  (`eval_clm_utils.py:82-86`).
- Evaluation order: `utils._index_samples` (`utils.py:198`) shuffles indices with
  `random.Random(123)` then re-sorts, and `max_num_samples` is never passed from
  `eval_clm.py`, so **every** row is evaluated and the seed is inert in practice.

## 2. Invalid-item filtering — happens at ANALYSIS time, not load time

This is the most consequential finding of Phase 1, and it contradicts the guess
in `CLAUDE.md` that the preprocessed data might already exclude these items.

- Lists: `utils.py:17-32` — `BAD_OPTIONS` (`'A and B'`, `'A and C'`, `'A and D'`,
  `'A nor B'`, `'A or B'`, `'B and C'`, `'B and D'`, `'C and D'`) and
  `REFER_OPTIONS` (`'all of the above'`, `'all of above'`, `'none of the above'`,
  `'none of above'`).
- Applied in `utils.load_results` (`utils.py:80-85`) under `skip_bad_options=True`.
- Matching semantics, which we must copy exactly: `BAD_OPTIONS` is a
  **case-sensitive substring** test (`e in option`), `REFER_OPTIONS` a
  **lowercased substring** test (`e in option.lower()`). An item is dropped if
  *any* of its options matches.
- The CSVs on disk still contain these rows. Inference runs over all of them;
  they are skipped only when results are re-read.

Reproduced the paper's figure exactly by applying these lists to the shipped CSVs:

| Task | Total | Filtered | % | Kept |
|---|---|---|---|---|
| MMLU | 14042 | 450 | **3.20%** | 13592 |
| ARC-Challenge | 1165 | 0 | 0.00% | 1165 |
| CSQA | 1216 | 0 | 0.00% | 1216 |

3.20% matches the paper's "about 3.2% of MMLU was excluded", so these lists are
the paper's filter. `data.py` must reproduce them verbatim.

**Consequence for Phase 2.** `load_results` is called only from
`debias_pride.py:90,103`, and only with `only_probs_and_ideals=True`. So the
accuracy that `eval_clm.py` prints (`utils.get_accuracy`, `utils.py:211`) is over
**all 14042 unfiltered** MMLU rows, while the PriDe/RStd path works on the
**13592 filtered** rows. When comparing to Table 3 we must try both and say which
one we matched. The `predictions, labels` branch of `load_results`
(`utils.py:108-110`) is never called anywhere in this repo.

## 3. Prompt construction

- Template: `eval_clm_utils.create_user_prompt` (`eval_clm_utils.py:101-118`).
- System message: `eval_clm_utils.py:93-98`. MMLU uses
  `'The following are multiple choice questions about {}.'` with
  `subject.replace('_', ' ')` (`eval_clm_utils.py:145`); ARC/CSQA use
  `'The following are multiple choice questions.'`. Both then get
  `' You should directly answer the question by choosing the correct option.'`
  appended — same string, one space, no newline.
- Assembly: `prepare_eval_fn_base` (`eval_clm_utils.py:181-188`):
  `sys_msg + '\n\n'`, then each few-shot demo + `'\n\n'`, then the eval prompt.
- Few-shot demos carry **no** system message of their own and end with
  `'Answer: ' + label` (`eval_clm_utils.py:129-132`) — note the space before the
  letter.
- Truncation: `input_ids[..., -1536:]` (`eval_clm_utils.py:191`), left-truncated
  to the last 1536 tokens.

Verified byte-exact 0-shot output by calling upstream's own `prepare_eval`
(MMLU, `abstract_algebra`, row 0). This is the reference string for the Phase 3
test (c):

```python
'The following are multiple choice questions about abstract algebra. You should '
'directly answer the question by choosing the correct option.\n\n'
'Question: Find the degree for the given field extension Q(sqrt(2), sqrt(3), '
'sqrt(18)) over Q.\nOptions:\nA. 0\nB. 4\nC. 2\nD. 6\nAnswer:'
```

Rendered:

```
The following are multiple choice questions about abstract algebra. You should directly answer the question by choosing the correct option.

Question: Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.
Options:
A. 0
B. 4
C. 2
D. 6
Answer:
```

Each option line is `f"{option_id}. {answer}".strip()`, joined with `'\n'`;
the block is preceded by `'Options:\n'` and followed by `'\nAnswer:'`.

## 4. Scoring — 8 token ids, not 4

`prepare_eval_fn_base`, `eval_clm_utils.py:176-219`. One forward pass, logits at
the last position, then:

```python
option_indices = [toker(f': {e}').input_ids[-1] for e in option_ids] + \
    [toker(f':{e}').input_ids[-1] for e in option_ids]
probs = F.softmax(logits[..., option_indices], dim=-1)...
probs = probs.reshape(2, len(option_ids)).sum(axis=0)
sampled = option_ids[np.argmax(probs)]
```

So the softmax runs over **8** ids — the space-prefixed and bare variant of each
label — and the two variants are **summed** to give the 4 reported
probabilities. `CLAUDE.md`'s spec ("softmax over those 4 only") does not match
upstream; we follow upstream. Logged in `docs/deviations.md`.

Other details of the same function:

- `bpe_has_space_prefix = toker(': A').input_ids[-1] != toker(':A').input_ids[-1]`
  (`eval_clm_utils.py:177`). If the tokenizer does **not** distinguish them, a
  trailing `' '` is appended to the prompt (`:187-188`). For LLaMA it does
  distinguish, so the prompt ends at `'Answer:'` with no trailing space.
- `torch.no_grad()` (`:192`), batch size **1** per forward pass.
- Parallelism is a `ThreadPool` over samples, not batching:
  `threads=torch.cuda.device_count()` (`eval_clm.py:82`), halved and retried on
  `RuntimeError` (`utils.py:185-192`).
- Model load: `device_map='auto'`, `use_safetensors=True`,
  `torch_dtype=bfloat16 if torch.cuda.is_bf16_supported() else float16`
  (`eval_clm.py:56-61`).

### Other settings (`eval_clm_utils.py:166-171`)

| Setting | Scoring function | What it changes |
|---|---|---|
| *(none)* | `prepare_eval_fn_base` | default A B C D order |
| `noid` | `prepare_eval_fn_noid` (`:222`) | labels removed **and** scoring switched to per-option LM loss (`model(labels=...).loss`, `argmin`) — the conflated ablation our project criticises, confirmed in code |
| `perm` | `prepare_eval_fn_perm` (`:280`) | all 24 content permutations, labels fixed |
| `cyclic` | `prepare_eval_fn_perm` | 4 cyclic content shifts, labels fixed (`utils.cycle_options`, `utils.py:118`) |
| `shuffle_both` | `prepare_eval_fn_base` | one content-seeded random shuffle of ids *and* options (`utils.shuffle_options_with_ids`, `utils.py:132`) |
| `move_a`..`move_d` | `prepare_eval_fn_base` | swaps the gold option to a fixed label (`utils.move_answer`, `utils.py:142`) |

`perm`/`cyclic` move content with labels left in place, so they vary gold
position but never which label sits at which position. `shuffle_both` does
permute labels, but it is a single content-seeded draw per question, is not
balanced, and reorders the label column out of `A B C D`. None of them give the
crossed label x position design, which is what our `rotate.py` adds.

## 5. RStd

`debias_pride.py:152-154`, reported at `:161-164`.

```python
report = classification_report(labels, predictions, output_dict=True)
recalls = [report[str(e)]['recall'] * 100 for e in range(prior.shape[-1])]
recall_stds.append(np.std(recalls))
```

- Recall is **per gold label** class: of the items whose gold is label L, the
  fraction predicted L.
- Scaled to percent **before** the std, so RStd is in recall-percentage points.
- `np.std` default `ddof=0` — population std over the 4 (or 5) label recalls.
- `rstd` in the output is `np.mean(recall_stds)` over `n_iters` PriDe iterations
  (`n_iters = 5`, or 1 when `RATIO_PREFIX_SAMPLES == 1.0`, `:105-107`).

Note that this lives inside the PriDe debiasing pipeline and consumes
`perm`/`cyclic` probabilities. **Nothing in the repo computes RStd for a plain
default-order run**, so our Phase 2 baseline RStd has to be computed by us using
exactly the recipe above. `get_bootstrap_accuracy_std` (`utils.py:221`) is a
different quantity: std of accuracy over 1000 half-size resamples **without**
replacement, `random.Random(123)`.

## 6. BOS and label-token conventions

Tokenizer construction, `eval_clm.py:51-54`:

```python
AutoTokenizer.from_pretrained(args.pretrained_model_path, use_fast=False,
                              add_bos_token=False, add_eos_token=False)
```

**No BOS is prepended** — confirms the assumption in `CLAUDE.md`. `use_fast=False`
means the slow sentencepiece tokenizer, which is why `sentencepiece` and
`protobuf` are pinned in `requirements.txt`.

Verified against `huggyllama/llama-7b` (the model in
`scripts/run_llama-7b.sh`; ungated, and LLaMA-1/2 share this 32000-piece
sentencepiece vocab):

| | id | token |
|---|---|---|
| `" A"` / `"A"` | 319 / 29909 | `▁A` / `A` |
| `" B"` / `"B"` | 350 / 29933 | `▁B` / `B` |
| `" C"` / `"C"` | 315 / 29907 | `▁C` / `C` |
| `" D"` / `"D"` | 360 / 29928 | `▁D` / `D` |

- Every label is a single token in both variants, so the 8-id scheme is well
  defined.
- `': A'` -> `['▁:', '▁A']` vs `':A'` -> `['▁:', 'A']`, so
  `bpe_has_space_prefix = True` and no trailing space is added.
- `add_bos_token=False` is honoured: `'Answer:'` -> `[673, 29901]`
  (`['▁Answer', ':']`), with no `bos_token_id=1` in front.
- `LlamaTokenizer` warns that `legacy=True` is in effect; upstream does not set
  `legacy`, so we leave the default.
- Minor: standalone `':'` is `▁:` (584), while the `':'` ending `'Answer:'` is
  29901. Harmless — only `input_ids[-1]` of the probe is used.

Per-model label ids for Llama-3.2, Qwen2.5 etc. must be re-derived and added
here in Phase 4, with the `tests/` check required by `CLAUDE.md`.

---

## Carried into later phases

1. **Models.** `debias_pride.py:25-46` lists the paper's full model set and it
   includes both `llama-7b` and `Llama-2-7b-hf`, so the Phase 2 target is in the
   paper even though `scripts/run_llama-7b.sh` only demonstrates LLaMA-1. Which
   model Table 3's RStd 23.0 / Acc 35.8 refers to still needs confirming against
   the paper or the released `LLM-MCQ-Bias_data` repo.
2. **Filtered vs unfiltered accuracy** — see the consequence note in §2.
3. **Batching.** Upstream runs batch size 1 with a thread pool. Batching is the
   obvious speedup for 16 versions/question but changes padding and numerics, so
   it needs a Phase 4 equivalence check against batch size 1 and an entry in
   `docs/deviations.md`.
4. **PriDe prior** (`debias_utils.simple`, `debias_utils.py:19`) estimates a
   label prior as `softmax(mean(log p))` across cyclic versions. Worth comparing
   with our conditional-logit label coefficients in Phase 5.
