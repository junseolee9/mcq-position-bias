#!/usr/bin/env bash
# Bootstrap this repo on an NSF ACCESS A100 node, run from a JupyterLab terminal.
#
#   git clone <this repo> && bash scripts/setup_remote.sh
#
# Idempotent: safe to re-run. Does not touch third_party/.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Upstream is a submodule; a plain `git clone` leaves it empty.
git -C "$REPO_ROOT" submodule update --init --recursive

# Home directories on ACCESS allocations are quota-limited (often ~25 GB) and
# model weights are tens of GB, so caches go on the large filesystem. Most
# sites export $SCRATCH; override it explicitly if yours uses another name
# (e.g. $PROJECT, $WORK, /anvil/scratch/$USER).
SCRATCH="${SCRATCH:-$HOME/scratch}"
HF_CACHE="$SCRATCH/hf_cache"
mkdir -p "$HF_CACHE"
export HF_HOME="$HF_CACHE"

# HF_HOME must be set in every future session too, not only this one.
BASHRC="$HOME/.bashrc"
if ! grep -q 'HF_HOME.*hf_cache' "$BASHRC" 2>/dev/null; then
    printf '\n# mcq-position-bias: keep HF model cache off the home quota\nexport HF_HOME="%s"\n' "$HF_CACHE" >> "$BASHRC"
    echo "[setup] appended HF_HOME export to $BASHRC"
else
    echo "[setup] HF_HOME export already in $BASHRC"
fi

python3 -m venv "$REPO_ROOT/.venv"
VPY="$REPO_ROOT/.venv/bin/python"
"$VPY" -m pip install --upgrade pip
"$VPY" -m pip install -r "$REPO_ROOT/requirements.txt"

# Record the versions that actually produced the results. The macOS dev
# machine resolves different wheels (CPU/MPS torch), so the cluster lock is
# the reproducibility record that matters.
"$VPY" -m pip freeze > "$REPO_ROOT/docs/requirements.remote.lock.txt"

# Make the venv selectable as a JupyterLab kernel.
"$VPY" -m ipykernel install --user --name mcqbias --display-name "mcqbias (venv)"

echo "[setup] HF_HOME=$HF_HOME"
"$VPY" - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available:", torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  gpu{i}: {p.name}  {p.total_memory / 1e9:.1f} GB")
PY

echo "[setup] done. Gated models still need: .venv/bin/huggingface-cli login"
