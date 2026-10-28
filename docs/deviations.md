# Deviations

Every intentional departure from the paper's setup, and every post-hoc change to
a run, gets a dated entry here. Newest last.

## 2026-10-28 — Phase 0: compute environment

The spec assumes one A100 40 GB. Actual compute is an A100 through an NSF ACCESS
allocation accessed via JupyterLab, so exact GPU model and memory are recorded
per run instead of assumed. Development happens on a macOS arm64 machine with no
NVIDIA GPU; no reported number comes from it. Not a deviation from the paper's
method — recorded because it changes the timing estimates in `docs/timing.md`.

## 2026-10-28 — Phase 0: Python and library versions

Python 3.11 locally. The paper's upstream code predates the installed
`transformers` major version, so a faithful port may need small API fixes
(see `docs/upstream_notes.md` once Phase 1 is written). Exact remote versions
are captured in `docs/requirements.remote.lock.txt` by `scripts/setup_remote.sh`.
