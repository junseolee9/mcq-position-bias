"""The sweep driver: what it plans, what it skips, and when it deletes weights.

Eviction granularity is the one with a price tag. Evicting per run instead of
per model re-downloads a 14 GB checkpoint eight more times than necessary, and
not evicting at all overruns a 100 GB volume with 168 GB of weights.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_sweep  # noqa: E402

MODELS_YAML = """\
alpha:
  - id: org/alpha-a
    batch_size: 16
  - id: org/alpha-b
beta:
  - id: org/beta-a
    batch_size: 8
"""

SWEEP_YAML = """\
runs:
  - dataset: mmlu
    mode: default
    seeds: [0]
  - dataset: mmlu
    mode: balanced1
    seeds: [0, 1]
"""


@pytest.fixture
def configs(tmp_path):
    (tmp_path / "models.yaml").write_text(MODELS_YAML)
    (tmp_path / "sweep.yaml").write_text(SWEEP_YAML)
    return tmp_path / "models.yaml", tmp_path / "sweep.yaml"


def test_specs_expand_seeds_and_carry_the_model_batch_size(configs):
    specs = run_sweep.load_specs(*configs)
    assert [family for family, _, _ in specs] == ["alpha", "alpha", "beta"]

    _family, entry, cfgs = specs[0]
    assert entry["id"] == "org/alpha-a"
    assert len(cfgs) == 3  # default x1 + balanced1 x2
    assert {c.batch_size for c in cfgs} == {16}
    assert [(c.mode, c.seed) for c in cfgs] == [
        ("default", 0), ("balanced1", 0), ("balanced1", 1)
    ]
    # a model with no batch_size falls back to the Config default
    assert {c.batch_size for c in specs[1][2]} == {8}


def test_filters_select_families_and_models(configs):
    assert len(run_sweep.load_specs(*configs, families=["beta"])) == 1
    assert len(run_sweep.load_specs(*configs, models=["org/alpha-b"])) == 1
    assert run_sweep.load_specs(*configs, families=["beta"], models=["org/alpha-b"]) == []


def _fake_run_and_load(monkeypatch, out_dir, log):
    """Record load/run/evict order without touching a model or the GPU."""
    def fake_load_model(cfg):
        log.append(("load", cfg.model))
        return ("model", "toker", "tokens", "rev")

    def fake_run(cfg, loaded=None):
        log.append(("run", cfg.run_id))
        path = out_dir / f"{cfg.run_id}.parquet"
        pd.DataFrame({"qid": ["a:0"], "version": [0], "correct": [True]}).to_parquet(path)
        return path

    monkeypatch.setattr(run_sweep, "load_model", fake_load_model)
    monkeypatch.setattr(run_sweep, "run", fake_run)
    monkeypatch.setattr(run_sweep, "evict", lambda m: log.append(("evict", m)))
    monkeypatch.setattr(run_sweep, "plan_work", lambda cfg: ([None] * 10, 10, 0))


def test_weights_are_evicted_once_per_model_after_its_last_run(configs, tmp_path, monkeypatch):
    out_dir = tmp_path / "raw"
    out_dir.mkdir()
    log = []
    _fake_run_and_load(monkeypatch, out_dir, log)

    assert run_sweep.main([
        "--models-config", str(configs[0]), "--sweep-config", str(configs[1]),
        "--out-dir", str(out_dir),
    ]) == 0

    # one load and one evict per model, with that model's three runs between them
    assert [m for kind, m in log if kind == "load"] == [
        "org/alpha-a", "org/alpha-b", "org/beta-a"
    ]
    assert [m for kind, m in log if kind == "evict"] == [
        "org/alpha-a", "org/alpha-b", "org/beta-a"
    ]
    runs_between = [e for e in log[log.index(("load", "org/alpha-a")):log.index(("evict", "org/alpha-a"))]
                    if e[0] == "run"]
    assert len(runs_between) == 3


def test_no_evict_keeps_the_weights(configs, tmp_path, monkeypatch):
    out_dir = tmp_path / "raw"
    out_dir.mkdir()
    log = []
    _fake_run_and_load(monkeypatch, out_dir, log)

    run_sweep.main([
        "--models-config", str(configs[0]), "--sweep-config", str(configs[1]),
        "--out-dir", str(out_dir), "--no-evict",
    ])
    assert not [e for e in log if e[0] == "evict"]


def test_a_finished_model_is_never_loaded(configs, tmp_path, monkeypatch):
    """The point of this: not downloading 14 GB to discover there is nothing to do."""
    out_dir = tmp_path / "raw"
    out_dir.mkdir()
    log = []
    _fake_run_and_load(monkeypatch, out_dir, log)

    # pre-create every output for alpha-a
    for mode, seed in (("default", 0), ("balanced1", 0), ("balanced1", 1)):
        pd.DataFrame({"qid": ["a:0"]}).to_parquet(out_dir / f"alpha-a_mmlu_{mode}_s{seed}.parquet")

    run_sweep.main([
        "--models-config", str(configs[0]), "--sweep-config", str(configs[1]),
        "--out-dir", str(out_dir),
    ])
    loaded = [m for kind, m in log if kind == "load"]
    assert "org/alpha-a" not in loaded
    assert loaded == ["org/alpha-b", "org/beta-a"]
    # and nothing was evicted for it either, since it was never fetched this time
    assert ("evict", "org/alpha-a") not in log


def test_plan_runs_nothing(configs, tmp_path, monkeypatch, capsys):
    out_dir = tmp_path / "raw"
    out_dir.mkdir()
    log = []
    _fake_run_and_load(monkeypatch, out_dir, log)

    assert run_sweep.main([
        "--models-config", str(configs[0]), "--sweep-config", str(configs[1]),
        "--out-dir", str(out_dir), "--plan",
    ]) == 0
    assert log == []
    assert "forward passes planned" in capsys.readouterr().out


def test_keep_going_reports_failures_and_exits_nonzero(configs, tmp_path, monkeypatch):
    out_dir = tmp_path / "raw"
    out_dir.mkdir()
    log = []
    _fake_run_and_load(monkeypatch, out_dir, log)

    def exploding_run(cfg, loaded=None):
        log.append(("run", cfg.run_id))
        if cfg.model == "org/alpha-a":
            raise RuntimeError("boom")
        return out_dir / f"{cfg.run_id}.parquet"

    monkeypatch.setattr(run_sweep, "run", exploding_run)

    code = run_sweep.main([
        "--models-config", str(configs[0]), "--sweep-config", str(configs[1]),
        "--out-dir", str(out_dir), "--keep-going",
    ])
    assert code == 1
    # it moved on to the other models rather than stopping at the first failure
    assert [m for kind, m in log if kind == "load"] == [
        "org/alpha-a", "org/alpha-b", "org/beta-a"
    ]
    # and the failing model's weights were still released
    assert ("evict", "org/alpha-a") in log
