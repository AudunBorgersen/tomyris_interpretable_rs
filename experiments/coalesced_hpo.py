"""Optuna HPO wrapper for coalesced_baseline.

Imports main() and parse_args() directly from coalesced_baseline so that any
changes to the experiment (new metrics, config keys, logging) are automatically
picked up without editing this file.
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import for side effect: caps BLAS/OpenMP threads BEFORE optuna (which imports
# numpy). For a worker spawned by _orchestrate the cap is already in the inherited
# env and is honoured here; for a direct single-GPU run this applies
# TOMYRIS_NUM_THREADS if set. See utils/cpu_threads.
import utils.blas_cap  # noqa: F401

import optuna
import yaml

from coalesced_baseline import main as run_trial
from coalesced_baseline import parse_args

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent
_DEFAULT_HPO_CONFIG = _PROJECT_ROOT / "configs" / "coalesced_hpo" / "default.yml"


def _suggest(trial: optuna.Trial, name: str, spec: dict):
    t = spec["type"]
    kwargs = {k: spec[k] for k in ("low", "high", "step", "log") if k in spec}
    if t == "int":
        return trial.suggest_int(name, **kwargs)
    if t == "float":
        return trial.suggest_float(name, **kwargs)
    if t == "categorical":
        return trial.suggest_categorical(name, spec["choices"])
    raise ValueError(f"Unknown search space type: {t!r}")


# Restricted namespace for evaluating `derived` expressions in the HPO config.
# No builtins; only these helpers are exposed alongside the sampled/fixed values.
_DERIVED_NS = {"round": round, "min": min, "max": max, "int": int, "float": float}


def make_objective(hpo_cfg: dict):
    fixed = hpo_cfg.get("fixed", {})
    search_space = hpo_cfg["search_space"]
    derived_specs = hpo_cfg.get("derived", {})
    drop = set(hpo_cfg.get("drop_from_args", []))
    metric = hpo_cfg["study"]["metric"]

    # Optimize the best value seen across epochs, not the final one. The rank
    # metrics peak early then decay, so final-epoch selection systematically
    # down-ranks the very configs we care about. coalesced_baseline.main()
    # returns `best_<metric>` measured on the VALIDATION split (leak-free model
    # selection); fall back to the raw metric if absent.
    best_metric = f"best_{metric}"
    test_metric = f"test_{metric}"

    def objective(trial: optuna.Trial) -> float:
        sampled = {
            name: _suggest(trial, name, spec) for name, spec in search_space.items()
        }
        ctx = {**fixed, **sampled}
        derived = {
            name: eval(expr, {"__builtins__": {}}, {**_DERIVED_NS, **ctx})
            for name, expr in derived_specs.items()
        }
        params = {**fixed, **sampled, **derived}
        for k in drop:
            params.pop(k, None)
        args = parse_args(argv=[], **params)
        result = run_trial(args)
        # Record the held-out test metric for visibility, but never select on it.
        if test_metric in result:
            trial.set_user_attr(test_metric, result[test_metric])
        # Select on the validation-best metric. (Avoid result.get(best_metric,
        # result[metric]) — the default arg is eagerly evaluated and KeyErrors
        # now that main() no longer returns the bare `<metric>` key.)
        if best_metric in result:
            return result[best_metric]
        if metric in result:
            return result[metric]
        raise KeyError(
            f"Objective metric not found: neither {best_metric!r} nor {metric!r} "
            f"in trial result. Available keys: {sorted(result)}"
        )

    return objective


def _sqlite_path(storage: str | None) -> Path | None:
    """Filesystem path of a sqlite storage URL, or None for other backends."""
    if storage and storage.startswith("sqlite:///"):
        return Path(storage[len("sqlite:///") :])
    return None


def _prepare_storage(storage: str | None):
    """Build an Optuna storage that tolerates several concurrent worker processes.

    For SQLite: ensure the parent dir exists, set a generous busy timeout, and
    enable WAL so multiple workers can write trial results without tripping
    'database is locked'. Non-sqlite URLs (and None) pass through unchanged.
    """
    if not storage:
        return None
    if not storage.startswith("sqlite"):
        return storage

    db_path = _sqlite_path(storage)
    if db_path:
        db_path.parent.mkdir(parents=True, exist_ok=True)

    rdb = optuna.storages.RDBStorage(
        url=storage,
        engine_kwargs={"connect_args": {"timeout": 60}},  # busy timeout (seconds)
    )

    # WAL lets one writer proceed alongside readers; persisted in the db header.
    from sqlalchemy import event

    @event.listens_for(rdb.engine, "connect")
    def _set_wal(dbapi_con, _record):  # noqa: ANN001
        cur = dbapi_con.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.close()

    return rdb


def _log_best(study: optuna.Study, metric: str) -> None:
    log.info("\nBest trial:")
    log.info(f"  {metric}: {study.best_trial.value:.4f}")
    log.info("  Params:")
    for k, v in study.best_trial.params.items():
        log.info(f"    {k}: {v}")


def _orchestrate(hpo_cfg_path: Path, hpo_cfg: dict, gpus: list, n_trials: int) -> None:
    """Spawn one worker subprocess per GPU, all sharing one Optuna study.

    Each GPU needs its own process: coalesced_baseline pins CUDA per-process
    (pycuda.autoinit binds the visible device on import and cannot be switched
    afterwards), so multi-GPU parallelism must be multi-process.
    """
    import math
    import os
    import subprocess

    from utils.cpu_threads import thread_env, threads_per_worker

    study_cfg = hpo_cfg["study"]
    storage = study_cfg.get("storage")
    if not storage:
        raise ValueError(
            "Parallel GPU sweeps require a shared `study.storage` (e.g. "
            "sqlite:///runs/coalesced_hpo/<name>.db) so workers coordinate one "
            "study. Add it to the HPO config or run with a single GPU."
        )

    # Materialise the schema + study row ONCE, before any worker starts. Without
    # this, every worker races to create the study on a fresh SQLite file and all
    # but the one that happens to arrive last crash with 'database is locked' /
    # IntegrityError during concurrent schema creation. After this, workers only
    # ever load an existing study.
    storage_obj = _prepare_storage(storage)
    optuna.create_study(
        study_name=study_cfg["study_name"],
        direction=study_cfg["direction"],
        storage=storage_obj,
        load_if_exists=True,
    )

    per_worker = math.ceil(n_trials / len(gpus))

    # Cap BLAS/OpenMP threads per worker. The coalesced update() does a per-example
    # float64 (n_classes x n_clauses) matvec that numpy farms out to a multithreaded
    # BLAS, which grabs ALL cores by default. With one worker per GPU each grabbing
    # every core, the processes oversubscribe the CPU and thrash — turning a 3-way
    # split into a ~10x per-epoch slowdown. Give each worker an even slice of cores
    # (shared with the standalone entry points via utils.cpu_threads).
    n_cores = os.cpu_count() or len(gpus)
    per_worker_threads = threads_per_worker(len(gpus))
    worker_thread_env = thread_env(per_worker_threads)

    log_dir = (
        Path(storage_path).parent
        if (storage_path := _sqlite_path(storage))
        else (_PROJECT_ROOT / "runs" / "coalesced_hpo")
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        f"Launching {len(gpus)} workers on GPUs {gpus} — "
        f"{per_worker} trials each (~{per_worker * len(gpus)} total), "
        f"{per_worker_threads} threads/worker ({n_cores} cores / {len(gpus)} workers), "
        f"shared study {study_cfg['study_name']!r} at {storage}"
    )

    procs, log_files = [], []
    for g in gpus:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--config",
            str(hpo_cfg_path),
            "--device_id",
            str(g),
            "--n_trials",
            str(per_worker),
        ]
        log_path = log_dir / f"worker_gpu{g}.log"
        log.info(f"  [GPU {g}] -> {log_path}")
        lf = open(log_path, "w")  # noqa: SIM115 — kept open for the worker's lifetime
        log_files.append(lf)
        procs.append(
            subprocess.Popen(
                cmd,
                env={**os.environ, **worker_thread_env},
                stdout=lf,
                stderr=subprocess.STDOUT,
            )
        )

    try:
        failures = [g for g, p in zip(gpus, procs) if p.wait() != 0]
    finally:
        for lf in log_files:
            lf.close()
    if failures:
        log.error(
            f"Worker(s) on GPU(s) {failures} exited non-zero — see {log_dir}/worker_gpu*.log"
        )

    # All workers wrote to the shared study; report the global best.
    study = optuna.load_study(study_name=study_cfg["study_name"], storage=storage_obj)
    _log_best(study, study_cfg["metric"])


def main(
    hpo_cfg_path: Path,
    device_id: int | None = None,
    n_trials: int | None = None,
) -> None:
    with open(hpo_cfg_path) as f:
        hpo_cfg = yaml.safe_load(f)

    study_cfg = hpo_cfg["study"]
    gpus = study_cfg.get("gpus") or []
    storage = study_cfg.get("storage")
    total_trials = n_trials if n_trials is not None else study_cfg["n_trials"]

    # Orchestrator mode: multiple GPUs configured and no specific device assigned
    # to this process yet -> fan out to one worker per GPU.
    if device_id is None and len(gpus) > 1:
        _orchestrate(hpo_cfg_path, hpo_cfg, gpus, total_trials)
        return

    # Worker / single-GPU mode. Resolve which device this process trains on and
    # inject it so every trial pins the right GPU.
    dev = device_id if device_id is not None else (gpus[0] if len(gpus) == 1 else None)
    if dev is not None:
        hpo_cfg.setdefault("fixed", {})["device_id"] = dev

    study = optuna.create_study(
        study_name=study_cfg["study_name"],
        direction=study_cfg["direction"],
        storage=_prepare_storage(storage),
        load_if_exists=bool(storage),
    )
    study.optimize(make_objective(hpo_cfg), n_trials=total_trials)
    _log_best(study, study_cfg["metric"])


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    p = argparse.ArgumentParser(description="Optuna HPO for coalesced_baseline")
    p.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_HPO_CONFIG,
        help="Path to HPO YAML config.",
    )
    p.add_argument(
        "--device_id",
        type=int,
        default=None,
        help="Pin this worker to a single GPU (internal: set by the orchestrator).",
    )
    p.add_argument(
        "--n_trials",
        type=int,
        default=None,
        help="Override study.n_trials (internal: per-worker share).",
    )
    args = p.parse_args()
    main(args.config, device_id=args.device_id, n_trials=args.n_trials)
