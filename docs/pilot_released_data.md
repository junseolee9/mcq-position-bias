# Pilot: what the authors' own released data already shows

Zero GPU hours. Everything here comes from `data/LLM-MCQ-Bias_data` (commit
`29fcbb4`) and upstream's own helper functions. MMLU, 0-shot, filtered
population (13592 items), models `Llama-2-7b-hf` and `llama-7b` (LLaMA-1).

Regenerate with:

```bash
.venv/bin/python scripts/explore_released_bias.py
```

Written **before** any run of our own, so the predictions in "Consequences"
below are on the record ahead of time (ground rule 3).

---

## 1. Only one released condition can identify position at all

Parsed the stored `prompt` of every released setting and read off the label
order actually shown to the model:

| Setting | Label order shown | Can it identify position? |
|---|---|---|
| `mmlu` (default) | `ABCD` | no — label == position |
| `mmlu_perm` (24 permutations) | `ABCD` only | **no — label == position** |
| `mmlu_move_a` … `move_d` | `ABCD` only | no — label == position |
| `mmlu_noid` | no labels shown | position only, labels gone |
| `mmlu_shuffle_both` | 24 distinct orders | **yes — label decoupled from position** |

This is a structural fact, not a sampling issue. In `perm` and `cyclic` the
option *content* moves but the label column stays `A B C D` top to bottom, so
"label A" and "position 0" are the same regressor. The paper's position-related
evidence rests on exactly these conditions, so it cannot separate the two.

`shuffle_both` is the exception. `utils.shuffle_options_with_ids`
(`utils.py:132`) shuffles `(label, content)` *pairs* together, so:

- the label keeps its original content, and the gold **label** never changes;
- each label's display **position** varies across questions.

That makes position estimable while holding label fixed. The paper has this
condition but uses it only as another "bias is unstable" demonstration, never to
decompose label from position.

Our reconstruction of the per-question shuffle was verified against the stored
prompts and the stored predictions: 13592/13592 exact agreement on the shuffled
label order, on `sampled`, and on `correct`, for both models.

## 2. Total slot bias is huge (`perm`, label and position together)

24 permutations per question, so every option appears at every slot — content
quality is marginalised out exactly, within question.

| Model | Acc | pick rate by slot (0,1,2,3) | max−min | 95% CI |
|---|---|---|---|---|
| `Llama-2-7b-hf` | 36.33 | 56.79 / 16.21 / 11.21 / 15.79 | **45.58 pp** | 44.87 – 46.33 |
| `llama-7b` | 32.93 | 52.10 / 9.54 / 25.64 / 12.72 | **42.56 pp** | 41.73 – 43.47 |

CIs resample questions, 300 reps. This is the effect the paper attributes mainly
to token bias — but in this condition it cannot be anything else, because label
and position are the same thing.

## 3. A position effect survives controlling for label (`shuffle_both`)

Conditional logit, 13592 choice sets of 4 alternatives, grouped by question.
Features: label dummies (A reference), position dummies (position 0 reference),
`is_gold`. This is the estimator `CLAUDE.md` specifies for Phase 5, run here on
real data.

**`Llama-2-7b-hf`**

| term | coef | se | 95% CI | p |
|---|---|---|---|---|
| label B | −1.2901 | 0.0230 | −1.335, −1.245 | <1e-300 |
| label C | −2.3876 | 0.0354 | −2.457, −2.318 | <1e-300 |
| label D | −2.7793 | 0.0412 | −2.860, −2.699 | <1e-300 |
| position 1 | −0.0028 | 0.0327 | −0.067, 0.061 | 0.93 |
| position 2 | **+0.6569** | 0.0318 | 0.595, 0.719 | <1e-300 |
| position 3 | **+0.5649** | 0.0316 | 0.503, 0.627 | <1e-300 |
| `is_gold` | +0.4327 | 0.0243 | 0.385, 0.480 | <1e-300 |

**`llama-7b`**

| term | coef | se | 95% CI | p |
|---|---|---|---|---|
| label B | −0.7792 | 0.0245 | −0.827, −0.731 | <1e-300 |
| label C | −0.5950 | 0.0229 | −0.640, −0.550 | <1e-300 |
| label D | −0.9092 | 0.0254 | −0.959, −0.860 | <1e-300 |
| position 1 | **−0.4646** | 0.0238 | −0.511, −0.418 | <1e-300 |
| position 2 | **−0.2907** | 0.0231 | −0.336, −0.245 | <1e-300 |
| position 3 | **−0.8146** | 0.0268 | −0.867, −0.762 | <1e-300 |
| `is_gold` | +0.3752 | 0.0193 | 0.337, 0.413 | <1e-300 |

Effect spans (max − min coefficient, including the 0 reference):

| Model | label span | position span | `is_gold` |
|---|---|---|---|
| `Llama-2-7b-hf` | 2.78 | 0.66 | 0.43 |
| `llama-7b` | 0.91 | **0.81** | 0.38 |

Two things worth stating plainly:

- The position effect is **not** negligible. For LLaMA-1 it is 0.81 against a
  label span of 0.91 — comparable magnitude, not a rounding error.
- For **both** models the position span **exceeds the `is_gold` coefficient**.
  Where an option sits on the page moves the model's choice more than whether
  the option is correct. That is a sharper framing of the problem than RStd.

Direction differs by model: Llama-2 prefers later positions (2 and 3), LLaMA-1
prefers position 0. So this is not one universal "primacy" artefact, which also
means it will not cancel out by averaging across models.

### Limitation, stated up front

In `shuffle_both` each label keeps its original content, so the label
coefficients absorb the mean quality of whatever content sat at that original
index, not just the label token. Position, in contrast, is assigned by the
shuffle and is clean. So the *position* estimates above are trustworthy while the
*label* estimates are an upper bound on true token bias. Only a crossed design
with label and content rotated independently — our Phase 3 `rotate.py` — removes
this. The pilot motivates the experiment; it does not replace it.

## 4. A position effect survives with the label tokens gone (`noid`)

`noid` removes the labels entirely. Recall by gold position is the usable
quantity here: the scored item is the gold either way, and MMLU's gold position
is arbitrary, so averaging over questions matches content across positions.

| Model | Acc | recall by gold position (0,1,2,3) | range |
|---|---|---|---|
| `Llama-2-7b-hf` | 37.01 | 56.42 / 27.74 / 32.11 / 33.58 | **28.68 pp** |
| `llama-7b` | 33.36 | 42.56 / 31.04 / 30.45 / 30.33 | **12.23 pp** |

With no label tokens present at all, a 12–29 pp recall spread by position
remains. Note that `noid` also switches scoring to per-option LM loss
(`prepare_eval_fn_noid`, `eval_clm_utils.py:222`) — the very conflation our
project criticises — so this is not a clean position estimate either. It is
still hard to read a 28.68 pp label-free position spread as "bias is mainly
token bias".

## 5. Design and budget: `balanced1` is as good as `full16`, at 1/16 the cost

`perm` gives 24 versions per question, so we can emulate "k versions x m
questions" at a fixed forward-pass budget and compare precision directly. Budget
fixed at 12000 forward passes.

Standard errors of the conditional-logit slot coefficients (`llama-7b`):

| versions/question | questions | choice sets | se slot1 | se slot2 | se slot3 | se `is_gold` |
|---|---|---|---|---|---|---|
| 1 | 12000 | 12000 | 0.0322 | 0.0225 | 0.0287 | 0.0215 |
| 4 | 3000 | 12000 | 0.0318 | 0.0228 | 0.0286 | 0.0214 |
| 16 | 750 | 12000 | 0.0332 | 0.0221 | 0.0283 | 0.0213 |
| 24 | 500 | 12000 | 0.0342 | 0.0215 | 0.0299 | 0.0215 |

**Flat.** The conditional logit identifies label and position *within* a choice
set of 4, so its precision tracks the number of choice sets — that is, the number
of forward passes — and not how they are distributed over questions.

For the marginal statistics (pick rate / recall / RStd by label or position),
spreading the same budget over more questions is strictly better, because
question-level variance dominates:

| versions/question | questions | sd of max−min pick rate |
|---|---|---|
| 1 | 12000 | 0.68 |
| 4 | 3000 | 0.92 |
| 8 | 1500 | 1.55 |
| 24 | 500 | 2.36 |

### Consequence for Phase 6

`CLAUDE.md` makes `full16` the default for main results, at 16x cost
(13592 x 16 = 217k forward passes per model-dataset, versus 13.6k for
`balanced1`). The pilot says that for both the conditional logit and the
marginal tables, `balanced1` over the full dataset is as precise or better at
1/16 the cost. Recommendation, to be confirmed in Phase 5 on our own smoke-test
output:

- **main results: `balanced1`** over all 13592 items, 5 seeds. Cost per
  model-dataset ≈ 68k forward passes for all five seeds together, still under
  one third of a single `full16` run.
- **`full16` on a subset** (~1000 questions) as the within-question check, since
  it is the only design that puts every label at every position *for the same
  question*. Keep it for the robustness section, not the headline.

This is the main budget lever against the ~100 A100-hour allowance and it was
decided without spending any of it.

## Predictions for our own runs, recorded in advance

1. Our `default` mode on `Llama-2-7b-hf` reproduces 35.80 / 22.95
   (`docs/reproduction.md`).
2. Our crossed design finds a position effect significantly different from zero
   on both models, with span of order 0.3–0.9 logits — smaller than the label
   effect for Llama-2, comparable to it for LLaMA-1.
3. The label span under our design comes out **smaller** than the
   `shuffle_both` estimate above, because rotating content independently stops
   the label coefficients absorbing content quality.
4. Position-effect direction stays model-specific rather than a universal
   primacy effect.

If 2 fails, the paper's conclusion is upheld on our stronger design and that is
the result we report.
