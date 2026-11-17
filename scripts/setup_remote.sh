#!/usr/bin/env bash
# Bootstrap this repo on the Jetstream2 instance `mcq-team`, run over `js2 ssh`.
#
#   js2 push-code mcq-team
#   js2 ssh mcq-team 'bash /workspace/mcq-position-bias/scripts/setup_remote.sh'
#
# Idempotent: safe to re-run. Does not touch third_party/ once cloned.
#
# Everything lands on /workspace (the volume). The 60 GB root disk has ~11 GB
# free and is wiped by a kill/relaunch, so nothing we need may live outside it.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="${WORKSPACE:-/workspace}"

if [[ ! -d "$WORKSPACE" ]]; then
    echo "[setup] ERROR: $WORKSPACE does not exist. The volume is not mounted;" >&2
    echo "[setup] run 'js2 start mcq-team' (it remounts) before this script." >&2
    exit 1
fi
case "$REPO_ROOT" in
    "$WORKSPACE"/*) ;;
    *)  echo "[setup] ERROR: repo is at $REPO_ROOT, outside $WORKSPACE." >&2
        echo "[setup] Work there is lost on kill/relaunch. Move the clone first." >&2
        exit 1 ;;
esac

# `js2 push-code` transfers source extensions only (*.py|sh|md|txt|toml|cfg|yaml|yml),
# so the submodule's MMLU/ARC/CSQA CSVs never arrive that way. Clone them instead;
# the instance has network and the submodule is 13 MB.
if [[ ! -f "$REPO_ROOT/third_party/LLM-MCQ-Bias/code/utils.py" ]]; then
    echo "[setup] fetching the upstream submodule (task CSVs live there)"
    git -C "$REPO_ROOT" submodule update --init --recursive
fi

# env.sh is created by `js2 setup` and never overwritten by it, so it is the right
# place for HF_TOKEN. Source it if present; otherwise set the cache path ourselves.
if [[ -f "$WORKSPACE/env.sh" ]]; then
    # shellcheck disable=SC1091
    source "$WORKSPACE/env.sh"
fi
export HF_HOME="${HF_HOME:-$WORKSPACE/hf}"
mkdir -p "$HF_HOME"

# uv ships no Python by itself on this image; `uv python install` fetches a
# standalone build. The image has 3.12, and torch==2.1.2 publishes no 3.12 wheel,
# so the pinned environment needs 3.11 regardless of what the node provides.
if ! command -v uv >/dev/null 2>&1; then
    echo "[setup] installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# Two environments, deliberately split (docs/deviations.md 2026-11-16):
#
#   venv-upstream  pinned to the paper's era; exists only to run upstream's
#                  eval_clm.py unmodified for the pipeline-equivalence check.
#   venv           modern transformers; runs src/mcqbias/ over all 12 models.
#                  Qwen2.5 needs >=4.37, which the pinned env cannot load.
setup_venv() {
    local path="$1" python="$2"
    shift 2
    echo "[setup] $path (python $python)"
    uv venv --python "$python" "$path"
    VIRTUAL_ENV="$path" uv pip install "$@"
}

setup_venv "$WORKSPACE/venv-upstream" 3.11 -r "$REPO_ROOT/requirements.txt"
"$WORKSPACE/venv-upstream/bin/python" -m pip freeze \
    > "$REPO_ROOT/docs/requirements.remote.lock.txt"

setup_venv "$WORKSPACE/venv" 3.11 \
    torch transformers accelerate sentencepiece protobuf \
    pandas pyarrow statsmodels scipy scikit-learn matplotlib pyyaml pytest tqdm
"$WORKSPACE/venv/bin/python" -m pip freeze \
    > "$REPO_ROOT/docs/requirements.remote.main.lock.txt"

echo "[setup] HF_HOME=$HF_HOME"
df -h "$WORKSPACE" | tail -1
"$WORKSPACE/venv/bin/python" - <<'PY'
import torch, transformers
print("torch", torch.__version__, "| transformers", transformers.__version__)
print("cuda available:", torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  gpu{i}: {p.name}  {p.total_memory / 1e9:.1f} GB")
PY

cat <<'EOF'
[setup] done.

Gated models (meta-llama/*) need a token. Put it in /workspace/env.sh, which
`js2 setup` never overwrites and which is outside the repo:

    echo 'export HF_TOKEN=hf_...' >> /workspace/env.sh

Never write the token into the repo. Before any run:  source /workspace/env.sh
EOF
