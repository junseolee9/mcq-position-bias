"""Checkpointing, resume and the out-of-memory retry.

These are the parts that only matter when something goes wrong on a billed
instance, which is exactly why they need a test that makes things go wrong on
purpose. A fake model keeps it fast and lets the failure be scripted.
"""

import numpy as np
import pandas as pd
import pytest

from mcqbias.run_eval import Config, completed_keys, consolidate, plan_work, run
from mcqbias.rotate import N_VERSIONS


class FakeTokens:
    ids = list(range(8))
    labels = "ABCD"
    has_space_prefix = True
    needs_trailing_space = False


class FakeModel:
    """Scores deterministically, and can be told to fail on a schedule.

    `oom_until` makes the first N calls raise the CUDA message torch uses, so the
    halving path is exercised. `die_after` raises a non-OOM error once enough
    rows exist to have been flushed, which is the "run died midway" case.
    """

    def __init__(self, oom_until=0, die_after=None, oom_above_batch=None):
        self.calls = 0
        self.scored = 0
        self.oom_until = oom_until
        self.die_after = die_after
        self.oom_above_batch = oom_above_batch
        self.batch_sizes = []

    def __call__(self, prompts):
        self.calls += 1
        if self.calls <= self.oom_until:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        if self.oom_above_batch is not None and len(prompts) > self.oom_above_batch:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        if self.die_after is not None and self.scored >= self.die_after:
            raise KeyboardInterrupt("simulated kill")
        self.batch_sizes.append(len(prompts))
        self.scored += len(prompts)
        # one-hot on a label that depends on the prompt, so predictions vary
        probs = np.zeros((len(prompts), 4))
        for i, text in enumerate(prompts):
            probs[i, len(text) % 4] = 1.0
        return probs


@pytest.fixture
def fake_score(monkeypatch):
    """Route score.score() at the fake model carried in `loaded`."""
    import mcqbias.run_eval as run_eval

    def fake(model, tokenizer, prompts, tokens):
        return model(prompts)

    monkeypatch.setattr(run_eval.score, "score", fake)
    monkeypatch.setattr(run_eval, "_empty_cache", lambda: None)
    return fake


def cfg_for(tmp_path, **kw):
    defaults = dict(
        model="fake/model", dataset="mmlu", mode="full16", n_questions=8, seed=0,
        batch_size=4, flush_every=8, out_dir=str(tmp_path),
    )
    return Config(**{**defaults, **kw})


def loaded_with(model):
    return (model, object(), FakeTokens(), "rev0")


def test_rows_are_flushed_to_shards_during_the_run(tmp_path, fake_score):
    cfg = cfg_for(tmp_path)
    total = 8 * N_VERSIONS

    model = FakeModel(die_after=40)
    with pytest.raises(KeyboardInterrupt):
        run(cfg, loaded=loaded_with(model))

    # no final file, but the work done before the kill survived
    assert not (tmp_path / f"{cfg.run_id}.parquet").exists()
    shards = sorted(tmp_path.glob(f"{cfg.run_id}.part-*.parquet"))
    assert shards, "nothing was flushed"
    done = completed_keys(tmp_path, cfg.run_id)
    assert 0 < len(done) < total
    # every flushed row is a real (qid, version) from the plan
    planned = {(q.qid, r.version) for q, r in plan_work(cfg)[0]}
    assert done <= planned


def test_resume_scores_only_what_is_missing(tmp_path, fake_score):
    cfg = cfg_for(tmp_path)
    total = 8 * N_VERSIONS

    first = FakeModel(die_after=40)
    with pytest.raises(KeyboardInterrupt):
        run(cfg, loaded=loaded_with(first))
    done_before = len(completed_keys(tmp_path, cfg.run_id))

    second = FakeModel()
    out = run(cfg, loaded=loaded_with(second))

    # the resumed run scored only the remainder, not the whole thing again
    assert second.scored == total - done_before
    frame = pd.read_parquet(out)
    assert len(frame) == total
    assert not frame.duplicated(["qid", "version"]).any()
    # shards are cleaned up once consolidated
    assert not list(tmp_path.glob(f"{cfg.run_id}.part-*.parquet"))


def test_a_completed_run_is_not_redone(tmp_path, fake_score):
    cfg = cfg_for(tmp_path)
    run(cfg, loaded=loaded_with(FakeModel()))

    again = FakeModel()
    run(cfg, loaded=loaded_with(again))
    assert again.scored == 0, "a finished run was scored a second time"


def test_out_of_memory_halves_the_batch_and_continues(tmp_path, fake_score):
    cfg = cfg_for(tmp_path, batch_size=16)
    model = FakeModel(oom_above_batch=4)
    out = run(cfg, loaded=loaded_with(model))

    assert max(model.batch_sizes) <= 4, model.batch_sizes
    assert len(pd.read_parquet(out)) == 8 * N_VERSIONS
    # 16 -> 8 -> 4, so the recorded batch size is the one that actually ran
    assert set(pd.read_parquet(out).batch_size) == {4}


def test_out_of_memory_at_batch_one_is_not_swallowed(tmp_path, fake_score):
    cfg = cfg_for(tmp_path, batch_size=1)
    with pytest.raises(RuntimeError, match="out of memory"):
        run(cfg, loaded=loaded_with(FakeModel(oom_above_batch=0)))
    # whatever had been scored is still flushed before the raise
    assert not (tmp_path / f"{cfg.run_id}.parquet").exists()


def test_a_truncated_shard_is_discarded_not_trusted(tmp_path, fake_score):
    cfg = cfg_for(tmp_path)
    model = FakeModel(die_after=40)
    with pytest.raises(KeyboardInterrupt):
        run(cfg, loaded=loaded_with(model))

    shards = sorted(tmp_path.glob(f"{cfg.run_id}.part-*.parquet"))
    good = completed_keys(tmp_path, cfg.run_id)
    shards[0].write_bytes(b"PAR1 truncated garbage")
    assert len(completed_keys(tmp_path, cfg.run_id)) < len(good)
    assert not shards[0].exists(), "the unreadable shard should have been removed"

    out = run(cfg, loaded=loaded_with(FakeModel()))
    assert len(pd.read_parquet(out)) == 8 * N_VERSIONS


def test_consolidate_deduplicates_overlapping_shards(tmp_path):
    rows = pd.DataFrame(
        {"qid": ["a:0", "a:0", "a:1"], "version": [0, 0, 1], "p_A": [0.1, 0.1, 0.2]}
    )
    rows.iloc[:2].to_parquet(tmp_path / "r.part-00000.parquet", index=False)
    rows.iloc[1:].to_parquet(tmp_path / "r.part-00001.parquet", index=False)
    out = consolidate(tmp_path, "r")
    assert len(pd.read_parquet(out)) == 2
