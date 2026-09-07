import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from base import BaseRecommender


class ItemKNNRecommender(BaseRecommender):
    """Item-based collaborative filtering via cosine similarity.

    score(item j | profile x) = sum_{i in x} sim(i, j)

    The item-item similarity matrix is pre-computed at fit time and truncated
    to the top-k most similar neighbours per item.
    """

    def __init__(self, k: int = 50):
        self.k = k

    def fit(self, x_train: np.ndarray) -> "ItemKNNRecommender":
        item_vecs = x_train.T.astype(np.float32)  # (n_items, n_users)
        norms = np.linalg.norm(item_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        item_vecs_norm = item_vecs / norms

        sim = item_vecs_norm @ item_vecs_norm.T  # (n_items, n_items)
        np.fill_diagonal(sim, 0.0)

        n_items = sim.shape[0]
        if self.k < n_items - 1:
            # For each item, keep only the top-k most similar neighbours.
            # kth_pos = index of the (k+1)-th largest value in a row (ascending order).
            # Everything at positions >= kth_pos is in the top-k.
            kth_pos = n_items - self.k - 1
            pivots = np.partition(sim, kth_pos, axis=1)[:, kth_pos : kth_pos + 1]
            sim[sim < pivots] = 0.0

        self.sim_: np.ndarray = sim
        return self

    def _score(self, X: np.ndarray) -> np.ndarray:
        return X.astype(np.float32) @ self.sim_
