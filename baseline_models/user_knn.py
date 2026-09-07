import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from base import BaseRecommender


class UserKNNRecommender(BaseRecommender):
    """User-based collaborative filtering via cosine similarity.

    score(item j | test user x) = sum_{u in top-k neighbours} sim(x, u) * x_train[u, j]
    """

    def __init__(self, k: int = 50):
        self.k = k

    def fit(self, x_train: np.ndarray) -> "UserKNNRecommender":
        self.x_train_: np.ndarray = x_train.astype(np.float32)
        norms = np.linalg.norm(self.x_train_, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.x_train_norm_: np.ndarray = self.x_train_ / norms
        return self

    def _score(self, X: np.ndarray) -> np.ndarray:
        X_f = X.astype(np.float32)
        norms = np.linalg.norm(X_f, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X_norm = X_f / norms

        sims = X_norm @ self.x_train_norm_.T  # (n_samples, n_train_users)

        n_train = sims.shape[1]
        if self.k < n_train:
            kth_pos = n_train - self.k - 1
            pivots = np.partition(sims, kth_pos, axis=1)[:, kth_pos : kth_pos + 1]
            sims = np.where(sims >= pivots, sims, 0.0)

        sims = np.maximum(sims, 0.0)
        return sims @ self.x_train_
