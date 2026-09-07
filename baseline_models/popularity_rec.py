import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from base import BaseRecommender


class PopularityRecommender(BaseRecommender):
    """Recommends the same global popularity ranking to every user."""

    def fit(self, x_train: np.ndarray) -> "PopularityRecommender":
        self.item_scores_: np.ndarray = x_train.sum(axis=0).astype(np.float32)
        return self

    def _score(self, X: np.ndarray) -> np.ndarray:
        return np.tile(self.item_scores_, (len(X), 1))
