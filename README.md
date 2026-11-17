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
- Upstream ships no requirements file, so `requirements.txt` is pinned to the
  APIs its code calls (`transformers==4.36.2`, `torch==2.1.2`, `numpy<2`) and
  upstream runs unmodified. Current releases break it — see `docs/deviations.md`
  before bumping anything.

## Released results (optional, for reference numbers)

The authors also publish every result file:
https://github.com/chujiezheng/LLM-MCQ-Bias_data. The full repo is 12.45 GB; we
take a ~424 MB sparse slice into `data/` (gitignored):

```bash
git clone --filter=blob:none --no-checkout --depth 1 \
  https://github.com/chujiezheng/LLM-MCQ-Bias_data.git data/LLM-MCQ-Bias_data
cd data/LLM-MCQ-Bias_data
git sparse-checkout init --cone
git sparse-checkout set debias_pride results_mmlu/0s_llama-7b results_mmlu/0s_Llama-2-7b-hf
git checkout main
```

With it, `scripts/check_released_results.py` recomputes the Table 3 baseline on a
laptop — which is how we identified the 35.8 / 23.0 row and validated our RStd
recipe without using the GPU. See `docs/reproduction.md`. Tests that need it skip
cleanly when it is absent.

## Environments

Two machines, deliberately split:

| | Local (dev) | Remote (runs) |
|---|---|---|
| Hardware | macOS arm64, no NVIDIA GPU | Jetstream2 instance `mcq-team`, A100 vGPU slice |
| Python | 3.11 (Homebrew) | 3.11 via `uv`, on the volume |
| Used for | code, tests, analysis of downloaded results | all model inference |

All reported numbers come from `mcq-team`. The local machine runs `pytest` and
`analyze.py` only; `torch` resolves to a CPU/MPS wheel there and must not be
used to produce results.

### Local setup

```bash
/opt/homebrew/bin/python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest
```

### Remote: the `mcq-team` instance

Jetstream2 (NSF ACCESS OpenStack), driven by the `js2` helper. The instance is
**shared by five teammates** and the team has a hard cap of 10,000 SU with no
top-up, so:

| | |
|---|---|
| Instance | `mcq-team` (never launch another; `kill` / `finish` / `vol delete` need the team's say-so) |
| Flavor | `g3.large` = A100 20 GB vGPU slice, 18.7 GB usable, **32 SU/h** (was `g3.medium`, 10 GB / 16 SU/h — too small for 7B) |
| Volume | `mcq-team-vol`, 100 GB, mounted at `/workspace`; the only storage that survives shelving |
| Repo path | `/workspace/mcq-position-bias` (pinned in `.js2.conf`) |

Billing runs while the instance is ACTIVE, not while it is computing, so an idle
instance is the most expensive thing in the project. Shelve it the moment a run
finishes — after checking that nobody else's job is live:

```bash
js2 ssh mcq-team 'tmux ls'     # someone else's session? leave it running
js2 stop mcq-team              # shelve: free, ~30 s
```

Bootstrap, once per volume:

```bash
js2 start mcq-team                      # unshelve, remounts /workspace
js2 push-code mcq-team                  # source files only, verified by checksum
js2 ssh mcq-team 'bash /workspace/mcq-position-bias/scripts/setup_remote.sh'
```

`setup_remote.sh` clones the upstream submodule on the instance (`push-code`
sends source extensions only, so the task CSVs never arrive that way), keeps the
Hugging Face cache at `/workspace/hf`, and builds **two** environments:
`/workspace/venv-upstream` pinned to `requirements.txt` for running upstream's
`eval_clm.py` unmodified, and `/workspace/venv` on modern `transformers` for our
own code, because `transformers==4.36.2` cannot load Qwen2.5. Both lock files are
written back into `docs/`.

Gated `meta-llama/*` checkpoints need a token in `/workspace/env.sh`
(`echo 'export HF_TOKEN=hf_...' >> /workspace/env.sh`) — outside the repo, never
committed.

Anything over ten minutes runs in `tmux`, tee'd to a log on the volume:

```bash
js2 ssh mcq-team
source /workspace/env.sh
cd /workspace/mcq-position-bias
tmux new -s run
/workspace/venv/bin/python -m mcqbias.run_eval --config configs/<run>.yaml \
  2>&1 | tee results/raw/<run>.log
```

Then `js2 pull-results mcq-team` brings the parquet and logs back. Code goes up,
results come down, never the reverse.

## Status

Phases 0–1 complete, plus a zero-GPU pilot on the authors' released results
(`docs/pilot_released_data.md`) that already finds a position effect larger than
the effect of an option being correct. Table 3's baseline is identified and our
RStd recipe is validated against it (`docs/reproduction.md`).

Next: Phase 3 — `data.py`, `rotate.py`, `prompts.py` and their tests, all
local. The GPU is only needed from Phase 4 onward.
