"""Drive the whole sweep: every model in configs/models.yaml over configs/sweep.yaml.

    # on the instance, inside tmux, with the log on the volume
    source /workspace/env.sh
    /workspace/venv/bin/python scripts/run_sweep.py --plan          # costs nothing
    /workspace/venv/bin/python scripts/run_sweep.py 2>&1 | tee results/raw/sweep.log

    --families qwen2.5 --models Qwen/Qwen2.5-3B     # subsets
    --n-questions 100                               # smoke test over the real sweep
    --no-evict                                      # keep weights (needs disk)

Three properties this has that a shell loop would not:

* **One model load per model, not per run.** Nine runs share a 14 GB load.
* **Resume.** Completed runs are skipped by their output file, and a model whose
  runs are all done is never downloaded at all. Partial runs resume from their
  shards (mcqbias.run_eval).
* **Eviction at the right granularity.** Weights are deleted after a model's
  *last* run, not after each one, so no model is downloaded twice. Twelve 7B
  checkpoints are 168 GB and mcq-team-vol is 100 GB.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mcqbias.run_eval import Config, evict, load_model, plan_work, run  # noqa: E402


def load_specs(models_path: Path, sweep_path: Path, families=None, models=None):
    """(family, model entry, [Config, ...]) for everything that should run."""
    families_cfg = yaml.safe_load(models_path.read_text())
    runs = yaml.safe_load(sweep_path.read_text())["runs"]

    out = []
    for family, entries in families_cfg.items():
        if families and family not in families:
            continue
        for entry in entries:
            if models and entry["id"] not in models:
                continue
            configs = [
                Config(
                    model=entry["id"],
                    dataset=spec["dataset"],
                    mode=spec["mode"],
                    seed=seed,
                    n_questions=spec.get("n_questions"),
                    batch_size=entry.get("batch_size", 8),
                )
                for spec in runs
                for seed in spec["seeds"]
            ]
            out.append((family, entry, configs))
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models-config", type=Path, default=REPO_ROOT / "configs/models.yaml")
    p.add_argument("--sweep-config", type=Path, default=REPO_ROOT / "configs/sweep.yaml")
    p.add_argument("--families", nargs="*", help="only these families")
    p.add_argument("--models", nargs="*", help="only these model ids")
    p.add_argument("--out-dir", default="results/raw")
    p.add_argument("--n-questions", type=int,
                   help="cap every run at this many questions (smoke test)")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--device-map", default="auto")
    p.add_argument("--no-evict", action="store_true",
                   help="keep weights after each model; needs room for all of them")
    p.add_argument("--plan", action="store_true",
                   help="print the plan and forward-pass counts, run nothing")
    p.add_argument("--keep-going", action="store_true",
                   help="on a failure, log it and move to the next model")
    args = p.parse_args(argv)

    specs = load_specs(args.models_config, args.sweep_config, args.families, args.models)
    if not specs:
        p.error("no models selected")

    out_dir = REPO_ROOT / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    overrides = {"out_dir": str(out_dir), "dtype": args.dtype, "device_map": args.device_map}
    if args.n_questions:
        overrides["n_questions"] = args.n_questions

    # --- plan -------------------------------------------------------------
    total_passes = total_todo = 0
    planned = []
    for family, entry, configs in specs:
        configs = [Config(**{**cfg.__dict__, **overrides}) for cfg in configs]
        model_passes = model_todo = 0
        pending_cfgs = []
        for cfg in configs:
            n_passes = len(plan_work(cfg)[0])
            model_passes += n_passes
            if (out_dir / f"{cfg.run_id}.parquet").exists():
                continue
            pending_cfgs.append(cfg)
            model_todo += n_passes
        planned.append((family, entry, pending_cfgs, model_passes, model_todo))
        total_passes += model_passes
        total_todo += model_todo
        state = "done" if not pending_cfgs else f"{len(pending_cfgs)} runs"
        print(f"{family:9} {entry['id']:34} batch {entry.get('batch_size', 8):>3}  "
              f"{model_passes:>7} passes  {state:>9}  {model_todo:>7} to go")

    print(f"\n{total_passes} forward passes planned, {total_todo} still to do "
          f"({total_passes - total_todo} already on disk)")
    if args.plan:
        return 0

    # --- run --------------------------------------------------------------
    failures = []
    started = time.time()
    for family, entry, pending_cfgs, _passes, todo in planned:
        if not pending_cfgs:
            print(f"\n=== {entry['id']}: all runs complete, not loading weights ===")
            continue

        print(f"\n=== {family} / {entry['id']}: {len(pending_cfgs)} runs, {todo} passes ===",
              flush=True)
        try:
            loaded = load_model(pending_cfgs[0])
        except Exception as exc:
            failures.append((entry["id"], f"load: {type(exc).__name__}: {exc}"))
            print(f"[sweep] FAILED to load {entry['id']}: {exc}", flush=True)
            if args.keep_going:
                continue
            raise

        try:
            for cfg in pending_cfgs:
                try:
                    run(cfg, loaded=loaded)
                except Exception as exc:
                    failures.append((cfg.run_id, f"{type(exc).__name__}: {exc}"))
                    print(f"[sweep] FAILED {cfg.run_id}: {exc}", flush=True)
                    traceback.print_exc()
                    if not args.keep_going:
                        raise
        finally:
            del loaded
            if not args.no_evict:
                evict(entry["id"])

    print(f"\n[sweep] {(time.time() - started) / 3600:.2f} h")
    if failures:
        print(f"[sweep] {len(failures)} failures:")
        for what, why in failures:
            print(f"  {what}: {why}")
        return 1
    print("[sweep] all runs complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
