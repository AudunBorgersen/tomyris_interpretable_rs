"""Vibrent rental dataset loader and binarizer (implicit feedback).

Parallel to ``datasets.movielens.MovieLens1M`` but for the Vibrent outfit-rental
dataset, which is implicit feedback (a rental == a positive interaction; there is
no 1-5 rating). The public API mirrors MovieLens1M so the experiment / baseline /
figure scripts can consume either dataset through ``datasets.get_dataset``.

Differences from the MovieLens loader:
  * No ``rating_threshold`` — any rental is a positive 1, and the 5,987 re-rental
    duplicates collapse to a single interaction.
  * Population is restricted by iterative **k-core** (default k=5) rather than the
    one-sided ``min_ratings`` filter: every surviving user has >= k distinct
    outfits *and* every surviving outfit has >= k distinct users.
  * Item metadata is the outfit catalogue (``outfits.csv``); the genre analog used
    by the interpretability figures is the ``Occasion`` tag family
    (``get_occasions``).
"""

import ast
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from datasets._common import InteractionBinarizer, k_core_filter

_DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data" / "vibrent_rental_dataset"


class VibrentRental:
    """Loader for the Vibrent outfit-rental dataset.

    Raw data is loaded lazily and cached. Call get() to obtain the binary
    user-item interaction matrices used for TM-based modelling.

    Example::

        ds = VibrentRental(k_core=5, test_ratio=0.2, val_ratio=0.1)
        data = ds.get()
        # data["x_train"]: (n_train_users, n_items) binary matrix
        # data["x_val"]:   (n_val_users, n_items)   binary matrix
        # data["x_test"]:  (n_test_users, n_items)  binary matrix
        # data["y_train"]: (n_train_users,) sorted customer IDs
        # data["y_val"]:   (n_val_users,)   sorted customer IDs
        # data["y_test"]:  (n_test_users,)  sorted customer IDs

        ds.item_id_to_name("outfit.5c08...")   # human-readable outfit name
        occasions = ds.get_occasions()         # one-hot Occasion-tag frame
        col_map = ds.col_to_item_id()          # {col_index: outfit_id, ...}
    """

    def __init__(
        self,
        data_dir: Union[str, Path, None] = None,
        test_ratio: float = 0.2,
        val_ratio: float = 0.1,
        k_core: int = 5,
        random_state: int = 42,
        max_items: int = 0,
        unit: str = "group",
    ):
        """
        Args:
            data_dir: Path to the vibrent_rental_dataset/ directory. Defaults to
                      <project_root>/data/vibrent_rental_dataset/.
            test_ratio: Fraction of all users held out for testing.
            val_ratio: Fraction of all users held out for validation. Carved out
                       of the training portion so the test split is identical to
                       the val_ratio=0 case. Set to 0 to disable (no x_val/y_val).
            k_core: Iterative k-core threshold applied before splitting — every
                    surviving user has >= k_core distinct items and every
                    surviving item has >= k_core distinct users.
            random_state: Seed for the user splits.
            max_items: If > 0, restrict to the N items with the most training
                       interactions. Columns are ordered by descending interaction
                       count (most popular = column 0).
            unit: Recommendation granularity. ``"group"`` (default) collapses each
                  physical outfit to its catalogue ``group`` (interchangeable
                  copies of the same garment), which is denser and the intended
                  recommendation target; ``"outfit"`` keeps individual physical
                  items as in the raw rental log.
        """
        if not 0 <= val_ratio < 1 - test_ratio:
            raise ValueError(
                f"val_ratio must be in [0, 1 - test_ratio) = [0, {1 - test_ratio}); "
                f"got {val_ratio}"
            )
        if unit not in ("group", "outfit"):
            raise ValueError(f"unit must be 'group' or 'outfit'; got {unit!r}")
        self.data_dir = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.k_core = k_core
        self.random_state = random_state
        self.max_items = max_items
        self.unit = unit

        self._interactions: pd.DataFrame | None = None
        self._outfits: pd.DataFrame | None = None
        self._outfit_to_group: dict[str, str] | None = None
        self._binarizer: InteractionBinarizer | None = None
        self._split: dict | None = None

    # ------------------------------------------------------------------
    # Raw file loaders
    # ------------------------------------------------------------------

    def _load_interactions(self) -> pd.DataFrame:
        """Long-form (user_id, item_id) rentals, deduplicated on the pair.

        ``item_id`` is the catalogue ``group`` when ``unit="group"`` (the default)
        and the physical ``outfit.id`` when ``unit="outfit"``. Re-rentals — and,
        under group mode, rentals of different physical copies of the same garment
        — collapse to a single positive interaction per (user, item).
        """
        df = pd.read_csv(
            self.data_dir / "user_activity_triplets.csv",
            sep=";",
            usecols=["customer.id", "outfit.id"],
            dtype={"customer.id": str, "outfit.id": str},
        )
        df = df.rename(columns={"customer.id": "user_id", "outfit.id": "item_id"})
        if self.unit == "group":
            df["item_id"] = df["item_id"].map(self.outfit_to_group)
            # Drop rentals whose outfit is absent from the catalogue (no group).
            df = df.dropna(subset=["item_id"])
        return df.drop_duplicates(["user_id", "item_id"], ignore_index=True)

    def _load_outfits(self) -> pd.DataFrame:
        return pd.read_csv(
            self.data_dir / "outfits.csv",
            sep=";",
            dtype=str,
        )

    # ------------------------------------------------------------------
    # Cached raw data
    # ------------------------------------------------------------------

    @property
    def interactions(self) -> pd.DataFrame:
        if self._interactions is None:
            self._interactions = self._load_interactions()
        return self._interactions

    @property
    def outfits(self) -> pd.DataFrame:
        if self._outfits is None:
            self._outfits = self._load_outfits()
        return self._outfits

    @property
    def outfit_to_group(self) -> dict[str, str]:
        """Mapping from physical ``outfit.id`` to its catalogue ``group``.

        Outfits with a missing group are omitted (their rentals are dropped under
        ``unit="group"``).
        """
        if self._outfit_to_group is None:
            cat = self.outfits[["id", "group"]].dropna(subset=["group"])
            self._outfit_to_group = dict(zip(cat["id"], cat["group"]))
        return self._outfit_to_group

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self) -> dict[str, np.ndarray]:
        """Build binary user-item matrices and split into train / val / test sets.

        The full interaction graph is reduced to its k-core, then users are split
        randomly; the item vocabulary is fixed to outfits seen in the training
        split so the column space is consistent across splits. When val_ratio=0
        the validation matrices are empty (shape (0, n_items)).

        Returns:
            dict with keys x_train / x_val / x_test (uint32 matrices) and
            y_train / y_val / y_test (sorted customer-ID arrays, row i = y[i]).
        """
        if self._split is not None:
            return self._split

        interactions = k_core_filter(self.interactions, self.k_core)
        if interactions.empty:
            raise ValueError(
                f"k_core={self.k_core} eliminated all interactions; lower k_core."
            )

        all_users = np.array(sorted(interactions["user_id"].unique()))
        train_users, test_users = train_test_split(
            all_users,
            test_size=self.test_ratio,
            random_state=self.random_state,
        )

        # Carve the validation users out of the training portion. The test split
        # above is computed first and left untouched, so x_test is identical
        # whether or not validation is enabled.
        val_users = np.array([], dtype=train_users.dtype)
        if self.val_ratio > 0:
            val_fraction = self.val_ratio / (1.0 - self.test_ratio)
            train_users, val_users = train_test_split(
                train_users,
                test_size=val_fraction,
                random_state=self.random_state,
            )

        train_int = interactions[interactions["user_id"].isin(train_users)]
        val_int = interactions[interactions["user_id"].isin(val_users)]
        test_int = interactions[interactions["user_id"].isin(test_users)]

        self._binarizer = InteractionBinarizer()
        x_train = self._binarizer.fit_transform(train_int)
        x_val = self._binarizer.transform(val_int)
        x_test = self._binarizer.transform(test_int)

        if self.max_items:
            item_counts = x_train.sum(axis=0).astype(np.int64)
            top_cols = np.argsort(-item_counts)[: self.max_items]
            x_train = x_train[:, top_cols]
            x_val = x_val[:, top_cols]
            x_test = x_test[:, top_cols]
            old_to_item = {col: iid for iid, col in self._binarizer.item_index.items()}
            self._binarizer.item_index = {
                old_to_item[int(old_col)]: new_col
                for new_col, old_col in enumerate(top_cols)
            }
            self._binarizer.n_items = len(top_cols)

        self._split = dict(
            x_train=x_train,
            x_val=x_val,
            x_test=x_test,
            y_train=np.array(sorted(train_users)),
            y_val=np.array(sorted(val_users)),
            y_test=np.array(sorted(test_users)),
        )
        return self._split

    def get_outfits(self) -> pd.DataFrame:
        """Return the outfit catalogue DataFrame (all columns, as strings)."""
        return self.outfits.copy()

    def get_occasions(self) -> pd.DataFrame:
        """Return a one-hot Occasion-tag DataFrame indexed by item_id.

        ``outfits.csv`` stores, per outfit, two parallel Python-list literals:
        ``outfit_tags`` (the tag strings) and ``tag_categories`` (the category of
        each tag). This extracts the tags whose category is ``Occasion`` and
        one-hot encodes them — the Vibrent analog of MovieLens' genre frame used
        by the interpretability figures. Columns are sorted Occasion tag names.

        The index is the matrix's item granularity: catalogue ``group`` under
        ``unit="group"`` (occasion tags unioned over the group's outfits) or
        ``outfit.id`` under ``unit="outfit"``.
        """
        per_item: dict[str, set[str]] = {}
        vocab: set[str] = set()
        for _, row in self.outfits.iterrows():
            occ = self._occasions_for_row(row["outfit_tags"], row["tag_categories"])
            if not occ:
                continue
            key = row["id"] if self.unit == "outfit" else row["group"]
            if key is None or (isinstance(key, float) and np.isnan(key)):
                continue
            per_item.setdefault(key, set()).update(occ)
            vocab.update(occ)

        columns = sorted(vocab)
        frame = pd.DataFrame(0, index=sorted(per_item), columns=columns, dtype=np.uint8)
        for iid, occ in per_item.items():
            for tag in occ:
                frame.at[iid, tag] = 1
        frame.index.name = "item_id"
        return frame

    @staticmethod
    def _occasions_for_row(tags_str: str, cats_str: str) -> set[str]:
        """Tags whose parallel category is 'Occasion' for one outfit row."""
        try:
            tags = ast.literal_eval(tags_str) if isinstance(tags_str, str) else []
            cats = ast.literal_eval(cats_str) if isinstance(cats_str, str) else []
        except (ValueError, SyntaxError):
            return set()
        return {t for t, c in zip(tags, cats) if c == "Occasion"}

    def item_id_to_name(self, item_id: Union[str, list]) -> Union[str, None, list]:
        """Translate one or more item IDs to a human-readable name.

        Accepts an ``outfit.id`` under ``unit="outfit"`` or a catalogue ``group``
        under ``unit="group"`` (returning a representative member outfit's name,
        since grouped copies are otherwise identical).
        """
        if self.unit == "outfit":
            lookup = self.outfits.set_index("id")["name"]
        else:
            lookup = (
                self.outfits.dropna(subset=["group"]).groupby("group")["name"].first()
            )
        if isinstance(item_id, (list, np.ndarray)):
            return [lookup.get(str(iid)) for iid in item_id]
        return lookup.get(str(item_id))

    def col_to_item_id(self) -> dict[int, str]:
        """Return a mapping from item matrix column index to item_id.

        The item_id is a catalogue ``group`` under ``unit="group"`` or an
        ``outfit.id`` under ``unit="outfit"``. Requires get() to have been called
        first.
        """
        if self._binarizer is None:
            raise RuntimeError("Call get() first to fit the binarizer.")
        return {col: iid for iid, col in self._binarizer.item_index.items()}

    @property
    def binarizer(self) -> InteractionBinarizer:
        """The fitted InteractionBinarizer (available after calling get())."""
        if self._binarizer is None:
            raise RuntimeError("Call get() first to fit the binarizer.")
        return self._binarizer
