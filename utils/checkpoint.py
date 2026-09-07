"""Utilities for loading TM checkpoints."""

import pickle
from pathlib import Path

import numpy as np


def load_tm_cpu(path: str | Path):
    """Load a TM checkpoint (including CUDA-trained ones) onto CPU.

    Bypasses CUDA device initialization so the model can be used for inference
    and analysis without needing MSVC / nvcc.  Training on this returned model
    will not work (CPU fallback only sets up the CPU CFFI inference path).
    """
    from tmu.clause_bank.clause_bank_cuda import ClauseBankCuda

    # Patch __setstate__ so unpickling a ClauseBankCuda skips device init.
    original_setstate = ClauseBankCuda.__setstate__

    def _cpu_setstate(self, state):
        self.__dict__.update(state)
        # device stays None — no GPU init

    ClauseBankCuda.__setstate__ = _cpu_setstate
    try:
        with open(path, "rb") as f:
            tm = pickle.load(f)
    finally:
        ClauseBankCuda.__setstate__ = original_setstate

    if isinstance(tm.clause_bank, ClauseBankCuda):
        tm.clause_bank = _cuda_bank_to_cpu(tm)
        tm.platform = "CPU"

    return tm


def _cuda_bank_to_cpu(tm):
    """Build a CPU ClauseBank from the host-side state of a CUDA clause bank."""
    from tmu.clause_bank.clause_bank import ClauseBank

    # During GPU training the trained TA states live on the device; the host
    # buffer is only a stale copy until an explicit memcpy_dtoh. Sync first,
    # otherwise we serialize the freshly-initialized (0-include) host state.
    if getattr(tm.clause_bank, "device", None) is not None:
        tm.clause_bank.synchronize_clause_bank()

    cuda_host = tm.clause_bank.host
    n_features = cuda_host.number_of_features

    cpu_bank = ClauseBank(
        seed=tm.seed,
        d=tm.d,
        number_of_state_bits_ind=tm.number_of_state_bits_ind,
        number_of_state_bits_ta=tm.number_of_state_bits_ta,
        batch_size=tm.batch_size,
        incremental=tm.incremental,
        X_shape=(1, n_features),
        s=tm.s,
        boost_true_positive_feedback=tm.boost_true_positive_feedback,
        reuse_random_feedback=tm.reuse_random_feedback,
        type_ia_ii_feedback_ratio=tm.type_ia_ii_feedback_ratio,
        number_of_clauses=tm.number_of_clauses,
        max_included_literals=tm.max_included_literals,
        patch_dim=tm.patch_dim,
    )

    # Overwrite the freshly-initialized clause_bank with the trained TA states,
    # then re-run _cffi_init so ptr_ta_state points to the new array.
    cpu_bank.clause_bank = np.ascontiguousarray(cuda_host.clause_bank.copy())
    cpu_bank._cffi_init()

    return cpu_bank
