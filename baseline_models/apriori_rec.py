import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

from base import BaseRecommender


class AprioriRecommender(BaseRecommender):
    """Association-rule recommender built on mlxtend's Apriori.

    Mines frequent itemsets with ``mlxtend.frequent_patterns.apriori`` and forms
    rules with ``association_rules``, then keeps those with a single-item
    consequent. A user is scored by summing the confidence of every rule whose
    antecedent is contained in their profile onto the consequent item.

    This is the classical, fully interpretable "users who liked A and B also
    liked C" model — the closest off-the-shelf analogue to a TM clause, which
    makes it a natural interpretability baseline. Mining and rule generation are
    delegated to mlxtend for reproducibility; only the profile -> score
    aggregation is implemented here.
    """

    def __init__(
        self,
        min_support: float = 0.01,
        min_confidence: float = 0.1,
        max_len: int = 3,
    ):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.max_len = max_len

    def fit(self, x_train: np.ndarray) -> "AprioriRecommender":
        self.n_items_: int = x_train.shape[1]
        df = pd.DataFrame(x_train.astype(bool), columns=list(range(self.n_items_)))

        frequent = apriori(
            df,
            min_support=self.min_support,
            max_len=self.max_len,
            use_colnames=True,
        )

        # Rules are list of (antecedent frozenset[int], consequent int, confidence).
        self.rules_: list[tuple[frozenset, int, float]] = []
        if len(frequent) and (frequent["itemsets"].map(len) >= 2).any():
            rules = association_rules(
                frequent, metric="confidence", min_threshold=self.min_confidence
            )
            single = rules[rules["consequents"].map(len) == 1]
            self.rules_ = [
                (frozenset(ant), next(iter(cons)), float(conf))
                for ant, cons, conf in zip(
                    single["antecedents"],
                    single["consequents"],
                    single["confidence"],
                )
            ]
        return self

    def _score(self, X: np.ndarray) -> np.ndarray:
        scores = np.zeros((len(X), self.n_items_), dtype=np.float32)
        for i, row in enumerate(X):
            liked = frozenset(np.where(row == 1)[0].tolist())
            for antecedent, cons, conf in self.rules_:
                if antecedent <= liked:
                    scores[i, cons] += conf
        return scores
