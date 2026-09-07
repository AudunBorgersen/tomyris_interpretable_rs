import numpy as np


class BaseRecommender:
    """Shared predict() interface matching TMU models so metrics.evaluate() works directly."""

    def fit(self, x_train: np.ndarray) -> "BaseRecommender":
        return self

    def predict(self, X: np.ndarray, return_class_sums: bool = False):
        scores = self._score(X)
        preds = np.argmax(scores, axis=1)
        if return_class_sums:
            return preds, scores
        return preds

    def _score(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
