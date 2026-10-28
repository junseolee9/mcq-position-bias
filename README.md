# Token Bias vs. Position Bias in LLM MCQ Selection

CSCI 567 project. We critique Zheng et al., *Large Language Models Are Not Robust
Multiple Choice Selectors* (ICLR 2024, [arXiv:2309.03882](https://arxiv.org/abs/2309.03882)).

The paper attributes LLM selection bias on multiple-choice questions mainly to
**token bias** (preference for the label tokens A/B/C/D) rather than **position
bias**. Its "removing IDs" ablation changes two things at once — labels are
removed *and* scoring switches to option likelihood — so it may understate the
position effect. We rotate label order and option-content order independently so
every label appears at every position, keep the paper's scoring unchanged, and
separate a label effect from a position effect.

See `CLAUDE.md` for the full design spec and phase plan.

## Upstream

Upstream code is a git submodule at `third_party/LLM-MCQ-Bias`
(https://github.com/chujiezheng/LLM-MCQ-Bias, archived), so clones need
`--recurse-submodules`.

- Commit: `2ae2f40c77006f7a00a4675bdfb30151c2406691` (pinned by the submodule)
- Never edited. Code we reuse is imported, or copied into `src/` with a
  `# adapted from LLM-MCQ-Bias/<path>@2ae2f40` comment.
- The clone ships the preprocessed MMLU / ARC / CSQA data under `code/data_*`.

Released result files: https://github.com/chujiezheng/LLM-MCQ-Bias_data

## Environments

Two machines, deliberately split:

| | Local (dev) | Remote (runs) |
|---|---|---|
| Hardware | macOS arm64, no NVIDIA GPU | A100, NSF ACCESS allocation via JupyterLab |
| Python | 3.11 (Homebrew) | 3.11+ from the node |
| Used for | code, tests, analysis of downloaded results | all model inference |

All reported numbers come from the remote A100. The local machine runs `pytest`
and `analyze.py` only; `torch` resolves to a CPU/MPS wheel there and must not be
used to produce results.

### Local setup

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest
```

### Remote setup (A100 / JupyterLab)

From a JupyterLab **terminal** on the GPU node:

```bash
git clone --recurse-submodules <this repo> && cd mcq-position-bias
bash scripts/setup_remote.sh
.venv/bin/huggingface-cli login    # gated Llama models
```

The script keeps the Hugging Face cache on `$SCRATCH` (ACCESS home quotas are
too small for model weights), writes the exact installed versions to
`docs/requirements.remote.lock.txt`, registers a `mcqbias (venv)` Jupyter
kernel, and prints the visible GPUs. Set `SCRATCH` yourself if your site names
the large filesystem differently.

Long runs should be started from a JupyterLab terminal with `nohup`, not from a
notebook cell — the kernel is reclaimed when the browser session ends:

```bash
nohup .venv/bin/python -m mcqbias.run_eval --config configs/<run>.yaml \
  > results/raw/<run>.log 2>&1 &
```

## Status

Phase 0 (setup) complete. Next: Phase 1, read upstream and write
`docs/upstream_notes.md`.
