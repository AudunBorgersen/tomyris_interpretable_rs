import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from sklearn.utils.extmath import randomized_svd

from base import BaseRecommender


class SVDRecommender(BaseRecommender):
    """Latent-factor model via randomised SVD with folding-in for unseen users.

    Decomposes x_train ≈ U diag(S) Vt; new (test) users are projected into
    the latent space and back: scores = x @ Vt.T @ Vt.
    """

    def __init__(self, n_factors: int = 64, random_state: int = 42):
        self.n_factors = n_factors
        self.random_state = random_state

    def fit(self, x_train: np.ndarray) -> "SVDRecommender":
        _, _, Vt = randomized_svd(
            x_train.astype(np.float32),
            n_components=self.n_factors,
            random_state=self.random_state,
        )
        self.Vt_: np.ndarray = Vt.astype(np.float32)  # (n_factors, n_items)
        return self

    def _score(self, X: np.ndarray) -> np.ndarray:
        user_latent = X.astype(np.float32) @ self.Vt_.T  # (n_samples, n_factors)
        return user_latent @ self.Vt_  # (n_samples, n_items)
