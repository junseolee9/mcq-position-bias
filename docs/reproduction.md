# Reproduction

## Reference numbers, recovered from the authors' released results

Before spending any A100 hours, we recomputed the Table 3 baseline from the
authors' own released result files (`data/LLM-MCQ-Bias_data`, commit `29fcbb4`,
sparse checkout) using upstream's exact definitions. Regenerate with:

```bash
.venv/bin/python scripts/check_released_results.py
```

MMLU, 0-shot, default `A B C D` order:

| Model | Population | n | Acc | RStd |
|---|---|---|---|---|
| `llama-7b` (LLaMA-1) | unfiltered | 14042 | 33.84 | 18.34 |
| `llama-7b` (LLaMA-1) | filtered | 13592 | 33.75 | 18.52 |
| `Llama-2-7b-hf` | unfiltered | 14042 | 36.06 | 22.51 |
| **`Llama-2-7b-hf`** | **filtered** | **13592** | **35.80** | **22.95** |

**Table 3's "RStd 23.0 / Acc 35.8" is `Llama-2-7b-hf` on the filtered 13592
rows.** 35.80 / 22.95 matches to the reported precision; no other row does.

Two things this settles, both of which were open after Phase 1:

1. The Phase 2 target model is `meta-llama/Llama-2-7b-hf` as `CLAUDE.md` says.
   The earlier guess that Table 3 might be LLaMA-1 — because
   `scripts/run_llama-7b.sh` uses `huggyllama/llama-7b` and its PriDe-debiased
   accuracy is 35.34, close to 35.8 — is **wrong**. LLaMA-1's baseline is
   33.75 / 18.52.
2. The paper reports on the **filtered** population. Unfiltered gives
   36.06 / 22.51, which does not match. So the `BAD_OPTIONS` / `REFER_OPTIONS`
   filter is applied to every number we compare against, and our decision to
   filter in `data.py` before inference is consistent with the paper's reporting
   (see `docs/deviations.md`).

It also validates our RStd implementation: per-gold-label recall from
`sklearn.classification_report`, x100, `np.std` with `ddof=0`, reproduces the
paper's figure exactly. That recipe is now pinned by `tests/test_released_results.py`.

### The bias being measured

`Llama-2-7b-hf`, filtered, for context on what RStd 22.95 means:

| | A | B | C | D |
|---|---|---|---|---|
| recall | 76.37 | 21.63 | 22.02 | 27.17 |
| pick rate | 61.86 | 11.92 | 11.15 | 15.07 |

The model picks A on 61.9% of questions when the correct answer is A only 25% of
the time. This is the effect the paper attributes mainly to token bias and that
our rotation design is meant to decompose.

## Our own run

Pending — requires the GPU. Target: land on **35.80 / 22.95** for
`meta-llama/Llama-2-7b-hf`, MMLU 0-shot, default order, filtered population,
using our `default` mode.

This is no longer a separate baseline run. `Llama-2-7b-hf` is one of the twelve
models in the main run set (`configs/models.yaml`), so its `default` condition —
which every model gets as the control — *is* the reproduction. If it lands on
35.80 / 22.95 the pipeline is validated on the paper's own number; if it does
not, the gap is recorded here with its cause and nothing is tuned to close it
(ground rule 3).

Known candidate sources of difference, written down before the run so they
cannot be invented afterwards. The first three are now recorded as columns in
every parquet file rather than left to memory:

- Weights: the released run used whatever `${HF_MODELS}/Llama-2-7b-hf` snapshot
  the authors had in 2023; we get the current Hub revision. → `model_revision`.
- dtype: upstream picks bf16 when `torch.cuda.is_bf16_supported()`, else fp16.
  On an A100 slice that is bf16. → `dtype`.
- Batching: upstream runs batch size 1 under a thread pool. → `batch_size`.
  Padding and reduction order change the logits slightly, and left padding also
  shifts absolute positions, which was a real bug before `score.py` started
  deriving `position_ids` from the attention mask (`docs/deviations.md`,
  2026-11-16). Batched output must still be verified against batch size 1 on the
  same questions, on the real model and dtype, before any full run.
- Thread pool and GPU count: upstream sets `threads=torch.cuda.device_count()`
  and `device_map='auto'` shards across visible GPUs, which can change reduction
  order. The Jetstream2 flavors here expose a single vGPU slice, so there is one
  device and no sharding — a difference from the authors' multi-GPU setup that
  removes a source of variation rather than adding one.

The separate check that our port has not drifted from upstream is the
pipeline-equivalence run: upstream's unmodified `eval_clm.py` against our
`default` mode on the same questions with `huggyllama/llama-7b`, which loads
under both the pinned `transformers==4.36.2` and current releases.
