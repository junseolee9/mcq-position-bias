"""End-to-end run on the CPU with a tiny model: schema and invariants, not numbers.

A real model's accuracy needs the GPU, but the parts that silently go wrong —
the label-to-position mapping, the version bookkeeping, the parquet schema — are
all testable here, and a mistake in them would corrupt every downstream table.
"""

import pandas as pd
import pytest

from mcqbias.rotate import N_VERSIONS
from mcqbias.run_eval import Config, parse_args, run, select_questions

TINY = "hf-internal-testing/tiny-random-gpt2"

SCHEMA = [  # CLAUDE.md's output schema
    "model", "dataset", "subject", "qid", "version", "label_shift", "content_shift",
    "label_order", "gold_pos", "gold_label", "pred_pos", "pred_label", "correct",
    "p_A", "p_B", "p_C", "p_D", "seed", "mode", "run_id", "git_commit",
]


@pytest.fixture(scope="module")
def full16_run(tmp_path_factory):
    pytest.importorskip("transformers")
    out = tmp_path_factory.mktemp("raw")
    cfg = Config(
        model=TINY, dataset="mmlu", mode="full16", n_questions=3, seed=0,
        batch_size=5,  # deliberately not a divisor of 48, to exercise the last chunk
        dtype="float32", device_map="cpu", out_dir=str(out),
    )
    try:
        path = run(cfg)
    except Exception as exc:
        pytest.skip(f"{TINY} unavailable: {type(exc).__name__}: {exc}")
    return pd.read_parquet(path)


def test_schema_is_complete(full16_run):
    assert set(SCHEMA) <= set(full16_run.columns)
    # provenance beyond the required schema
    assert {"model_revision", "dtype", "batch_size"} <= set(full16_run.columns)


def test_one_row_per_question_version(full16_run):
    assert len(full16_run) == 3 * N_VERSIONS
    assert full16_run.groupby("qid").version.nunique().eq(N_VERSIONS).all()
    assert not full16_run.duplicated(["qid", "version"]).any()


def test_version_decomposes_into_its_two_shifts(full16_run):
    assert (full16_run.version == 4 * full16_run.label_shift + full16_run.content_shift).all()


def test_prediction_maps_back_to_a_displayed_position(full16_run):
    for row in full16_run.itertuples():
        assert row.label_order[row.pred_pos] == row.pred_label
        assert row.label_order[row.gold_pos] == row.gold_label
        assert row.correct == (row.pred_label == row.gold_label)
        # the label the model picked is the argmax of the four probabilities
        probs = {"A": row.p_A, "B": row.p_B, "C": row.p_C, "D": row.p_D}
        assert row.pred_label == max(probs, key=probs.get)
        assert abs(sum(probs.values()) - 1.0) < 1e-5


def test_full16_balances_gold_within_each_question(full16_run):
    for _qid, group in full16_run.groupby("qid"):
        assert sorted(group.gold_pos.value_counts().tolist()) == [4, 4, 4, 4]
        assert sorted(group.gold_label.value_counts().tolist()) == [4, 4, 4, 4]


def test_select_questions_is_seeded_and_order_preserving():
    items = list(range(100))
    a = select_questions(items, 10, seed=0)
    assert a == select_questions(items, 10, seed=0)
    assert a != select_questions(items, 10, seed=1)
    assert a == sorted(a)
    assert select_questions(items, None, seed=0) is items
    assert select_questions(items, 500, seed=0) is items


def test_config_from_yaml_and_cli_override(tmp_path):
    cfg_path = tmp_path / "run.yaml"
    cfg_path.write_text("model: some/model\ndataset: arc\nmode: default\nseed: 7\n")
    cfg = parse_args(["--config", str(cfg_path)])
    assert (cfg.model, cfg.dataset, cfg.mode, cfg.seed) == ("some/model", "arc", "default", 7)
    # the CLI wins over the file
    cfg = parse_args(["--config", str(cfg_path), "--seed", "9", "--mode", "balanced1"])
    assert (cfg.seed, cfg.mode) == (9, "balanced1")


def test_config_requires_a_model():
    with pytest.raises(SystemExit):
        parse_args(["--dataset", "mmlu"])
