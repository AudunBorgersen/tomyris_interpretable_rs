import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from base import BaseRecommender


class EASERecommender(BaseRecommender):
    """Embarrassingly Shallow Autoencoder (Steck, WWW 2019).

    Closed-form item-item model: learns a weight matrix B with zero diagonal
    minimising ||X - X B||^2 + l2 * ||B||^2.

    score(x) = x @ B
    """

    def __init__(self, l2: float = 250.0):
        self.l2 = l2

    def fit(self, x_train: np.ndarray) -> "EASERecommender":
        X = x_train.astype(np.float32)
        G = X.T @ X  # (n_items, n_items) item-item gram
        diag = np.diag_indices(G.shape[0])
        G[diag] += self.l2
        P = np.linalg.inv(G)
        B = -P / np.diag(P)  # closed-form solution with the diag constraint
        B[diag] = 0.0
        self.B_: np.ndarray = B.astype(np.float32)
        return self

    def _score(self, X: np.ndarray) -> np.ndarray:
        return X.astype(np.float32) @ self.B_
