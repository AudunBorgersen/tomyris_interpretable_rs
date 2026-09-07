import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from base import BaseRecommender


class RandomRecommender(BaseRecommender):
    """Random ranking — theoretical Hit@k = k / (n_items - |seen|)."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        self.n_items: int = 0

    def fit(self, x_train: np.ndarray) -> "RandomRecommender":
        self.n_items = x_train.shape[1]
        return self

    def _score(self, X: np.ndarray) -> np.ndarray:
        return self._rng.random((len(X), self.n_items)).astype(np.float32)
