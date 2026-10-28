# CLAUDE.md — CSCI 567 Project: Token Bias vs. Position Bias in LLM MCQ Selection

## What this project is

We critique Zheng et al., *Large Language Models Are Not Robust Multiple Choice Selectors* (ICLR 2024, arXiv:2309.03882).

- **Paper's claim:** LLM selection bias on MCQs comes mainly from *token bias* (preference for label tokens A/B/C/D), not *position bias* (preference for option positions).
- **Our alternative hypothesis:** a substantial part of the bias comes from *position*. The paper's "removing IDs" ablation changes two things at once (labels removed + scoring switched to option likelihood), so it may underestimate position effects.
- **Our experiment:** rotate label order AND option-content order so every label appears at every position, keep the paper's scoring unchanged, then separate a *label effect* from a *position effect*.

Upstream code: https://github.com/chujiezheng/LLM-MCQ-Bias (archived, read-only; experiment code is in its `code/` folder). Released results: https://github.com/chujiezheng/LLM-MCQ-Bias_data

## Ground rules (always follow)

1. **Never edit files under `third_party/`.** Import from it, or copy a function into `src/` with a comment `# adapted from LLM-MCQ-Bias/<path>@<commit>`.
2. **Change one thing at a time.** Our setup must match the paper's (prompt template, scoring, 0-shot, data filtering) except for label/content order. If you must deviate, log it in `docs/deviations.md` with the reason.
3. **No silent fixes to results.** Never drop runs, change seeds, or tweak prompts after seeing results without logging it in `docs/deviations.md`.
4. **Debug small first:** 1B model + 100 questions before any full run.
5. Ask before installing large dependencies, downloading models over 10 GB, or deleting files.
6. Commit after each phase with a clear message. Never commit `data/`, `results/raw/`, model weights, or HF tokens.

## Target layout

```
mcq-position-bias/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── third_party/LLM-MCQ-Bias/   # upstream clone, untouched; record commit hash in README
├── src/mcqbias/
│   ├── data.py        # load paper's preprocessed data, filter invalid items
│   ├── rotate.py      # label x content rotations
│   ├── prompts.py     # paper's Figure 6 template with arbitrary label order
│   ├── score.py       # next-token probs over the 4 displayed labels
│   ├── run_eval.py    # CLI: model, dataset, mode, n_questions, seed
│   └── analyze.py     # recall tables, RStd, conditional logit, bootstrap CIs
├── configs/           # one YAML per run
├── tests/             # pytest
├── docs/              # upstream_notes.md, deviations.md, timing.md
├── data/              # gitignored
└── results/
    ├── raw/           # gitignored, one row per (question, version)
    └── summary/       # small CSVs + figures, committed
```

## Design spec

### Rotation (`rotate.py`)

Options are indexed by display position `p ∈ {0,1,2,3}` (top to bottom). Original contents `o[0..3]`, original gold index `g`.

- **Content shift `j`:** content shown at position `p` is `o[(p + j) % 4]`. Gold position = `(g - j) % 4`.
- **Label shift `k`:** label shown at position `p` is `"ABCD"[(p + k) % 4]`. So `k=0` → A B C D, `k=1` → B C D A, etc.
- **Version id** `v = 4*k + j`, 16 versions per question.
- Gold label = label shown at the gold position.

Modes:
- `full16`: every question gets all 16 versions (default for main results).
- `balanced1`: each question gets ONE version, assigned round-robin `v = (rank of question in a seeded shuffle) % 16`. Balance holds at dataset level at 1x cost. Use for large-scale or extra seeds.
- `default`: `k=0, j=0` only (paper's original setting; our control).

Do NOT try to get balance with only 4 versions per question using pure rotations: for 4 options it is mathematically impossible to balance label×position, gold×position, and gold×label at the same time.

Rotation is invalid for options that refer to other options ("all of the above", "A and B", "none of the above", "both"). `data.py` must drop these and log the count. Check first whether the paper's preprocessed data already excludes them (paper says about 3.2% of MMLU was excluded).

### Prompt (`prompts.py`)

Reproduce the paper's Figure 6 template exactly, only substituting the label order. Match the paper's choice of NOT prepending a BOS token; confirm against upstream code and note it in `docs/upstream_notes.md`. Options are printed in display order, e.g.:

```
C. <content at pos 0>
D. <content at pos 1>
A. <content at pos 2>
B. <content at pos 3>
Answer:
```

### Scoring (`score.py`)

- One forward pass, logits at the last position.
- Take logits of the 4 label tokens, softmax over those 4 only, argmax = prediction.
- **Tokenizer check (required):** for each model, verify which token the model predicts after `Answer:` ("A" vs " A") and that each label is a single token. Follow upstream's convention; write the result per model in `docs/upstream_notes.md`. Add a test.
- bf16, batched, `torch.no_grad()`.

### Output schema (`results/raw/*.parquet`, one row per question-version)

`model, dataset, subject, qid, version, label_shift, content_shift, label_order, gold_pos, gold_label, pred_pos, pred_label, correct, p_A, p_B, p_C, p_D, seed, mode, run_id, git_commit`

### Analysis (`analyze.py`)

1. Accuracy per condition (default vs rotated).
2. Recall per label and per position (+ RStd for each). Paper's RStd = std of per-label recall.
3. Pick rate per label and per position.
4. **Conditional logistic regression:** each (qid, version) is a choice set of 4 alternatives. Features per alternative: one-hot label (A = reference), one-hot position (pos 0 = reference), `is_gold`. Fit with `statsmodels` `ConditionalLogit` grouped by (qid, version). Report label and position coefficients with CIs.
5. **Bootstrap CIs:** resample questions (not rows), 1000 reps.
6. Figures: bar charts of label effect vs. position effect per model, with error bars.

### Seeds

Inference is deterministic, so a seed controls (a) which questions are sampled and (b) the `balanced1` version assignment. Use 5 seeds.

### Models (all fit on one A100 40 GB in bf16)

Start: `meta-llama/Llama-3.2-1B` (debug). Main: `meta-llama/Llama-2-7b-hf` (overlaps with paper, used for reproduction), `meta-llama/Llama-3.2-3B`, `Qwen/Qwen2.5-7B`. Gated models need `huggingface-cli login`; never write the token to a file in the repo.

### Datasets

MMLU 0-shot first; ARC-Challenge as second check. Use the paper's preprocessed data.

## Phases (do one at a time; stop and report after each)

**Phase 0 — Setup.** Create the repo with the layout above, `git init`, `.gitignore`, venv, `requirements.txt` (torch, transformers, accelerate, datasets, pandas, pyarrow, statsmodels, scipy, matplotlib, pyyaml, pytest, tqdm). Clone upstream into `third_party/` and record its commit hash. Done when: `pytest` runs (even with zero tests) and GPU is visible.

**Phase 1 — Read upstream (no edits).** Find where upstream loads data, builds prompts, scores, computes RStd, and filters invalid items. Write `docs/upstream_notes.md` with file paths and function names for each, plus the BOS and label-token conventions. Done when: notes cover all five items.

**Phase 2 — Reproduce the paper's baseline.** Run upstream's own code (or a faithful port) on `Llama-2-7b-hf`, MMLU 0-shot, default order. Paper Table 3 reports RStd 23.0 / Acc 35.8. Done when: our numbers are written in `docs/reproduction.md` with the gap and possible reasons. Do not tune anything to close the gap.

**Phase 3 — Rotation + prompts.** Implement `data.py`, `rotate.py`, `prompts.py`. Tests must check: (a) in `full16`, each label appears at each position exactly 4 times per question; (b) gold lands at each position exactly 4 times and gets each label exactly 4 times; (c) `k=0, j=0` output matches upstream's prompt byte for byte; (d) `balanced1` is balanced over the dataset.

**Phase 4 — Scoring + run loop.** Implement `score.py` and `run_eval.py` writing the schema above. Smoke test: Llama-3.2-1B, 100 questions, `full16`. Record time per 1k forward passes in `docs/timing.md` and estimate full-run GPU hours. Check that our `default` mode matches Phase 2 numbers on the same questions.

**Phase 5 — Analysis.** Implement `analyze.py` on the smoke-test output. Sanity check with a synthetic run: inject a fake pure position bias and a fake pure label bias into probabilities and confirm the regression recovers each one. Done when: both synthetic cases are recovered correctly.

**Phase 6 — Full runs.** Run configs for each model × dataset × seed. Stop and report the GPU-hour estimate before launching. Budget: about 100 A100-hours total for the whole project.
