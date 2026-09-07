"""Shared dataset helpers used by the implicit-feedback loaders.

Kept separate from ``movielens.py`` so the (explicit-rating) MovieLens path stays
byte-identical; only the new implicit-feedback datasets depend on this module.
"""

import numpy as np
import pandas as pd


def k_core_filter(
    interactions: pd.DataFrame,
    k: int,
    user_col: str = "user_id",
    item_col: str = "item_id",
) -> pd.DataFrame:
    """Iteratively prune a long-form interaction table to its k-core.

    Repeatedly drops every user with fewer than ``k`` distinct items and every
    item with fewer than ``k`` distinct users until the table is stable, i.e.
    every surviving user has >= k items and every surviving item has >= k users.

    ``interactions`` is expected to be deduplicated on (user, item) beforehand so
    the per-side counts are over *distinct* partners.

    Args:
        interactions: long-form table with ``user_col`` / ``item_col`` columns.
        k: minimum distinct-partner count required on both sides.
        user_col: name of the user column.
        item_col: name of the item column.

    Returns:
        The k-core subset (a copy). Empty if nothing satisfies the constraint.
    """
    df = interactions
    while True:
        user_counts = df[user_col].value_counts()
        item_counts = df[item_col].value_counts()
        keep_users = user_counts[user_counts >= k].index
        keep_items = item_counts[item_counts >= k].index
        if len(keep_users) == len(user_counts) and len(keep_items) == len(item_counts):
            return df.copy()
        df = df[df[user_col].isin(keep_users) & df[item_col].isin(keep_items)]
        if df.empty:
            return df.copy()


class InteractionBinarizer:
    """Builds a binary user-item matrix from long-form implicit interactions.

    Mirrors the fit / transform / fit_transform interface of TMU's
    StandardBinarizer (and ``datasets.movielens.UserItemBinarizer``), but for
    implicit feedback: any (user, item) interaction counts as a positive 1, so
    there is no rating threshold.

    fit() learns the item vocabulary from the training interactions; transform()
    applies the same item columns to any user subset (unknown items are silently
    dropped). Rows are always ordered by ascending ``user_col`` value, matching
    the ``y_*`` arrays the dataset loaders return.
    """

    def __init__(
        self,
        user_col: str = "user_id",
        item_col: str = "item_id",
        dtype=np.uint32,
    ):
        self.user_col = user_col
        self.item_col = item_col
        self.dtype = dtype
        self.item_index: dict = {}
        self.n_items: int = 0

    def fit(self, interactions: pd.DataFrame) -> "InteractionBinarizer":
        """Learn the item vocabulary (sorted) from an interaction table."""
        items = sorted(interactions[self.item_col].unique())
        self.item_index = {iid: i for i, iid in enumerate(items)}
        self.n_items = len(items)
        return self

    def transform(self, interactions: pd.DataFrame) -> np.ndarray:
        """Build the (n_users, n_items) binary matrix for a set of users.

        Users are ordered by ascending ``user_col`` value; items not seen during
        fit() are ignored.
        """
        users = sorted(interactions[self.user_col].unique())
        user_index = {uid: i for i, uid in enumerate(users)}
        matrix = np.zeros((len(users), self.n_items), dtype=self.dtype)

        rows = interactions[self.user_col].map(user_index)
        cols = interactions[self.item_col].map(self.item_index)
        valid = rows.notna() & cols.notna()
        matrix[rows[valid].astype(int).values, cols[valid].astype(int).values] = 1
        return matrix

    def fit_transform(self, interactions: pd.DataFrame) -> np.ndarray:
        return self.fit(interactions).transform(interactions)
