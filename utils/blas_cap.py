"""Import for side effect: cap BLAS/OpenMP threads before numpy is loaded.

Put ``import utils.blas_cap`` at the top of a training entry point — after the
``sys.path`` setup, before ``import numpy`` (or optuna, which imports numpy). BLAS
reads its thread env vars once when the shared library first loads, so the cap
has to be in place before that import.

This is kept as a side-effecting import (rather than a ``configure_blas_threads()``
call in each entry point) so the cap is a plain import statement and does not push
the following imports out of the module's import section. See utils/cpu_threads
for the policy and the single ``TOMYRIS_NUM_THREADS`` override knob.
"""

from utils.cpu_threads import configure_blas_threads

configure_blas_threads()
