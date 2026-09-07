"""Centralized BLAS/OpenMP thread capping.

The coalesced TM update (``tmu/tmu/models/classification/coalesced_classifier.py``)
runs a per-example float64 ``(n_classes x n_clauses)`` matvec that NumPy hands to
a multithreaded BLAS. By default BLAS grabs *every* core, so two training
processes on one host oversubscribe the CPU and thrash — a ~10x per-epoch
slowdown (see commit 5ca53c5, which first fixed this for the HPO orchestrator
only). Every entry point that trains a model routes through here so that mixing
baseline / grid-search / HPO runs on one machine can never oversubscribe.

IMPORTANT: call :func:`configure_blas_threads` at the very top of an entry point,
*before* importing numpy (or anything that imports numpy, e.g. optuna). BLAS
reads these env vars once when its shared library is first loaded; setting them
afterwards has no effect. This module deliberately imports nothing that pulls in
numpy so it is safe to import first.
"""

from __future__ import annotations

import os

# Every common BLAS/OpenMP backend, so all agree on the thread count.
THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

# Single knob to cap threads for every entry point at once, e.g. when launching a
# baseline alongside a running HPO study: `TOMYRIS_NUM_THREADS=8 python ...`.
ENV_OVERRIDE = "TOMYRIS_NUM_THREADS"


def threads_per_worker(n_workers: int) -> int:
    """Even slice of cores per concurrent worker (at least 1)."""
    n_workers = max(1, n_workers)
    n_cores = os.cpu_count() or n_workers
    return max(1, n_cores // n_workers)


def thread_env(threads: int) -> dict[str, str]:
    """Env mapping that pins every BLAS backend to ``threads`` threads.

    Use this to build the environment for spawned worker subprocesses (the cap
    must be in the child's env *before* it starts, since BLAS reads it on load).
    """
    return {var: str(max(1, int(threads))) for var in THREAD_VARS}


# Set default cap to 8 threads, for now. None of the environs are configured in the default environment.
def configure_blas_threads(threads: int | None = 8) -> int | None:
    """Cap this process's BLAS/OpenMP threads. Returns the cap applied, or None.

    Precedence (first that applies wins):
      1. A thread var already in the environment (exported by the user, or set in
         a spawned worker's env by the orchestrator) — respected and mirrored
         across the other backends so they all agree. Never overwritten.
      2. The ``TOMYRIS_NUM_THREADS`` override.
      3. The ``threads`` argument.
      4. No cap — BLAS keeps its default (all cores), which is fastest for a run
         that has the host to itself.

    Must run before numpy/BLAS import to take effect.
    """
    existing = next((os.environ[v] for v in THREAD_VARS if v in os.environ), None)
    if existing is not None:
        for var in THREAD_VARS:
            os.environ.setdefault(var, existing)
        return int(existing) if existing.isdigit() else None

    override = os.environ.get(ENV_OVERRIDE)
    if override is not None:
        threads = int(override)

    if threads is None:
        return None

    threads = max(1, int(threads))
    for var in THREAD_VARS:
        os.environ[var] = str(threads)
    return threads
