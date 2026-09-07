import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from sklearn.tree import DecisionTreeClassifier

from base import BaseRecommender


class DecisionTreeRecommender(BaseRecommender):
    """Single multiclass decision tree for masked-item prediction.

    Training samples mirror the evaluation set: for each liked item in a user's
    profile, that item is masked out of the input vector and becomes the
    prediction target. A scikit-learn DecisionTreeClassifier then learns to
    predict the held-out item index from the remaining profile, and
    predict_proba over the items gives the ranking scores.
    """

    def __init__(
        self,
        max_depth: int | None = 20,
        min_samples_leaf: int = 5,
        max_samples_per_user: int | None = 20,
        random_state: int = 42,
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_samples_per_user = max_samples_per_user
        self.random_state = random_state

    def fit(self, x_train: np.ndarray) -> "DecisionTreeRecommender":
        self.n_items_: int = x_train.shape[1]
        X, y = self._build_samples(x_train)
        self.tree_ = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )
        self.tree_.fit(X, y)
        return self

    def _build_samples(self, x_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """One masked sample per liked item (capped per user), matching run_all."""
        rng = np.random.default_rng(self.random_state)
        rows_x, rows_y = [], []
        for user_vec in x_matrix:
            liked = np.where(user_vec == 1)[0]
            if (
                self.max_samples_per_user is not None
                and len(liked) > self.max_samples_per_user
            ):
                liked = rng.choice(liked, self.max_samples_per_user, replace=False)
            for col in liked:
                masked = user_vec.copy()
                masked[col] = 0
                rows_x.append(masked)
                rows_y.append(col)
        return np.array(rows_x, dtype=np.uint8), np.array(rows_y, dtype=np.int64)

    def _score(self, X: np.ndarray) -> np.ndarray:
        # predict_proba only covers classes seen in training; scatter back to
        # the full item catalogue so ranking metrics see every item.
        proba = self.tree_.predict_proba(X.astype(np.uint8))
        scores = np.zeros((len(X), self.n_items_), dtype=np.float32)
        scores[:, self.tree_.classes_] = proba
        return scores
