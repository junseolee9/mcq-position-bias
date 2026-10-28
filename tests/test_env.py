"""Phase 0 sanity: package importable, core deps present."""


def test_package_importable():
    import mcqbias  # noqa: F401


def test_core_deps():
    import pandas, pyarrow, statsmodels, torch, transformers  # noqa: F401


def test_upstream_clone_present():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    assert (root / "third_party/LLM-MCQ-Bias/code/eval_clm.py").is_file()
