"""Phase 0 sanity: package imports, upstream clone usable, versions still pinned.

The version test is the point of this file: upstream (LLM-MCQ-Bias@2ae2f40) targets
the transformers 4.3x API, and newer majors renamed `torch_dtype` and dropped the
slow tokenizers that upstream asks for with `use_fast=False`. If someone upgrades,
this fails here rather than silently changing results.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_CODE = REPO_ROOT / "third_party/LLM-MCQ-Bias/code"


def test_package_importable():
    import mcqbias  # noqa: F401


def test_core_deps():
    import pandas, pyarrow, pynvml, sklearn, statsmodels, torch, transformers  # noqa: F401


def test_upstream_clone_present():
    assert (UPSTREAM_CODE / "eval_clm.py").is_file(), (
        "upstream submodule missing; run: git submodule update --init --recursive"
    )


def test_versions_pinned_to_upstream_era():
    import numpy
    import torch
    import transformers

    assert transformers.__version__.startswith("4.36."), transformers.__version__
    assert torch.__version__.startswith("2.1."), torch.__version__
    assert numpy.__version__.startswith("1."), numpy.__version__


@pytest.mark.parametrize(
    "module", ["mmlu_categories", "utils", "eval_clm_utils", "debias_utils"]
)
def test_upstream_modules_import(module):
    """Upstream is imported, never edited, so it must load under our pins.

    eval_clm itself is excluded: it calls pynvml.nvmlInit() at import time and so
    only loads on an NVIDIA host.
    """
    sys.path.insert(0, str(UPSTREAM_CODE))
    try:
        __import__(module)
    finally:
        sys.path.remove(str(UPSTREAM_CODE))
