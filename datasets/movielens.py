"""MovieLens 1M dataset loader and binarizer."""

from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

_DEFAULT_DATA_DIR = Path(__file__).parent.parent / "data" / "movielens1m"

ALL_GENRES = [
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]

AGE_MAP = {
    1: "Under 18",
    18: "18-24",
    25: "25-34",
    35: "35-44",
    45: "45-49",
    50: "50-55",
    56: "56+",
}

OCCUPATION_MAP = {
    0: "other",
    1: "academic/educator",
    2: "artist",
    3: "clerical/admin",
    4: "college/grad student",
    5: "customer service",
    6: "doctor/health care",
    7: "executive/managerial",
    8: "farmer",
    9: "homemaker",
    10: "K-12 student",
    11: "lawyer",
    12: "programmer",
    13: "retired",
    14: "sales/marketing",
    15: "scientist",
    16: "self-employed",
    17: "technician/engineer",
    18: "tradesman/craftsman",
    19: "unemployed",
    20: "writer",
}


class UserItemBinarizer:
    """Converts a ratings DataFrame into a binary user-item interaction matrix.

    Follows the fit / transform / fit_transform interface of TMU's StandardBinarizer.
    fit() learns the item vocabulary from training data; transform() applies the same
    item columns to any user subset (unknown items are silently dropped).

    Row ordering in the output matrix is always sorted ascending by user_id, matching
    the y_train / y_test arrays returned by MovieLens1M.get().
    """

    def __init__(self, threshold: int = 4, dtype=np.uint32):
        """
        Args:
            threshold: Ratings >= threshold are treated as positive interactions.
            dtype: NumPy dtype for the output matrix.
        """
        self.threshold = threshold
        self.dtype = dtype
        self.item_index: dict[int, int] = {}
        self.n_items: int = 0

    def fit(self, ratings: pd.DataFrame) -> "UserItemBinarizer":
        """Learn the item vocabulary from a ratings DataFrame.

        Args:
            ratings: DataFrame with at least a movie_id column.

        Returns:
            self
        """
        items = sorted(ratings["movie_id"].unique())
        self.item_index = {mid: i for i, mid in enumerate(items)}
        self.n_items = len(items)
        return self

    def transform(self, ratings: pd.DataFrame) -> np.ndarray:
        """Build the binary user-item matrix for a set of users.

        Users are ordered by ascending user_id (matching y arrays from get()).
        Items not seen during fit() are ignored.

        Args:
            ratings: DataFrame with columns user_id, movie_id, rating.

        Returns:
            Binary matrix of shape (n_users, n_items).
        """
        users = sorted(ratings["user_id"].unique())
        user_index = {uid: i for i, uid in enumerate(users)}
        matrix = np.zeros((len(users), self.n_items), dtype=self.dtype)

        pos = ratings[ratings["rating"] >= self.threshold]
        rows = pos["user_id"].map(user_index)
        cols = pos["movie_id"].map(self.item_index)

        valid = rows.notna() & cols.notna()
        matrix[rows[valid].astype(int).values, cols[valid].astype(int).values] = 1
        return matrix

    def fit_transform(self, ratings: pd.DataFrame) -> np.ndarray:
        """Fit on ratings then return the binary matrix."""
        return self.fit(ratings).transform(ratings)


class MovieLens1M:
    """Loader for the MovieLens 1M dataset.

    Raw data is loaded lazily and cached. Call get() to obtain the binary
    user-item interaction matrices used for TM-based modelling.

    Example::

        ds = MovieLens1M(rating_threshold=4, test_ratio=0.2, val_ratio=0.1)
        data = ds.get()
        # data["x_train"]: (n_train_users, n_items) binary matrix
        # data["x_val"]:   (n_val_users, n_items)   binary matrix
        # data["x_test"]:  (n_test_users, n_items)  binary matrix
        # data["y_train"]: (n_train_users,) sorted user IDs
        # data["y_val"]:   (n_val_users,)   sorted user IDs
        # data["y_test"]:  (n_test_users,)  sorted user IDs

        ds.movie_id_to_title(1)          # "Toy Story (1995)"
        ds.movie_id_to_title([1, 2, 3])  # ["Toy Story (1995)", ...]

        genres = ds.get_genres()         # DataFrame indexed by movie_id
        col_map = ds.col_to_movie_id()   # {col_index: movie_id, ...}
    """

    def __init__(
        self,
        data_dir: Union[str, Path, None] = None,
        rating_threshold: int = 4,
        test_ratio: float = 0.2,
        val_ratio: float = 0.1,
        min_ratings: int = 20,
        random_state: int = 42,
        max_items: int = 0,
    ):
        """
        Args:
            data_dir: Path to the movielens1m/ directory. Defaults to
                      <project_root>/data/movielens1m/.
            rating_threshold: Ratings >= this value count as positive interactions.
            test_ratio: Fraction of all users held out for testing.
            val_ratio: Fraction of all users held out for validation (for
                       hyperparameter selection). Carved out of the training
                       portion so the test split is identical to the
                       val_ratio=0 case. Set to 0 to disable (no x_val/y_val).
            min_ratings: Users with fewer positive interactions are excluded.
            random_state: Seed for the user splits.
            max_items: If > 0, restrict to the N items with the most positive
                       interactions in the training split. Columns are ordered by
                       descending interaction count (most popular = column 0).
        """
        if not 0 <= val_ratio < 1 - test_ratio:
            raise ValueError(
                f"val_ratio must be in [0, 1 - test_ratio) = [0, {1 - test_ratio}); "
                f"got {val_ratio}"
            )
        self.data_dir = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR
        self.rating_threshold = rating_threshold
        self.test_ratio = test_ratio
        self.val_ratio = val_ratio
        self.min_ratings = min_ratings
        self.random_state = random_state
        self.max_items = max_items

        self._ratings: pd.DataFrame | None = None
        self._movies: pd.DataFrame | None = None
        self._users: pd.DataFrame | None = None
        self._binarizer: UserItemBinarizer | None = None
        self._split: dict | None = None

    # ------------------------------------------------------------------
    # Raw file loaders
    # ------------------------------------------------------------------

    def _load_ratings(self) -> pd.DataFrame:
        return pd.read_csv(
            self.data_dir / "ratings.dat",
            sep="::",
            engine="python",
            names=["user_id", "movie_id", "rating", "timestamp"],
            dtype={
                "user_id": np.int32,
                "movie_id": np.int32,
                "rating": np.int8,
                "timestamp": np.int64,
            },
        )

    def _load_movies(self) -> pd.DataFrame:
        return pd.read_csv(
            self.data_dir / "movies.dat",
            sep="::",
            engine="python",
            names=["movie_id", "title", "genres"],
            dtype={"movie_id": np.int32, "title": str, "genres": str},
            encoding="latin-1",
        )

    def _load_users(self) -> pd.DataFrame:
        df = pd.read_csv(
            self.data_dir / "users.dat",
            sep="::",
            engine="python",
            names=["user_id", "gender", "age", "occupation", "zip_code"],
            dtype={
                "user_id": np.int32,
                "gender": str,
                "age": np.int32,
                "occupation": np.int32,
                "zip_code": str,
            },
        )
        df["age_label"] = df["age"].map(AGE_MAP)
        df["occupation_label"] = df["occupation"].map(OCCUPATION_MAP)
        return df

    # ------------------------------------------------------------------
    # Cached raw data
    # ------------------------------------------------------------------

    @property
    def ratings(self) -> pd.DataFrame:
        if self._ratings is None:
            self._ratings = self._load_ratings()
        return self._ratings

    @property
    def movies(self) -> pd.DataFrame:
        if self._movies is None:
            self._movies = self._load_movies()
        return self._movies

    @property
    def users(self) -> pd.DataFrame:
        if self._users is None:
            self._users = self._load_users()
        return self._users

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self) -> dict[str, np.ndarray]:
        """Build binary user-item matrices and split into train / val / test sets.

        Users are split randomly; the item vocabulary is fixed to items seen
        in the training split so the column space is consistent across splits.
        When val_ratio=0 the validation matrices are empty (shape (0, n_items)).

        Returns:
            dict with keys:
                x_train  ndarray (n_train_users, n_items) uint32
                x_val    ndarray (n_val_users, n_items) uint32
                x_test   ndarray (n_test_users, n_items) uint32
                y_train  ndarray (n_train_users,) int32 — user IDs, row i = y_train[i]
                y_val    ndarray (n_val_users,) int32   — user IDs, row i = y_val[i]
                y_test   ndarray (n_test_users,) int32  — user IDs, row i = y_test[i]
        """
        if self._split is not None:
            return self._split

        ratings = self.ratings

        # Remove users below the minimum positive-interaction threshold
        pos_counts = (
            ratings[ratings["rating"] >= self.rating_threshold]
            .groupby("user_id")
            .size()
        )
        active_users = pos_counts[pos_counts >= self.min_ratings].index
        ratings = ratings[ratings["user_id"].isin(active_users)]

        all_users = np.array(sorted(ratings["user_id"].unique()))
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

        train_ratings = ratings[ratings["user_id"].isin(train_users)]
        val_ratings = ratings[ratings["user_id"].isin(val_users)]
        test_ratings = ratings[ratings["user_id"].isin(test_users)]

        self._binarizer = UserItemBinarizer(threshold=self.rating_threshold)
        x_train = self._binarizer.fit_transform(train_ratings)
        x_val = self._binarizer.transform(val_ratings)
        x_test = self._binarizer.transform(test_ratings)

        if self.max_items:
            item_counts = x_train.sum(axis=0).astype(np.int64)
            top_cols = np.argsort(-item_counts)[: self.max_items]
            x_train = x_train[:, top_cols]
            x_val = x_val[:, top_cols]
            x_test = x_test[:, top_cols]
            old_to_movie = {col: mid for mid, col in self._binarizer.item_index.items()}
            self._binarizer.item_index = {
                old_to_movie[int(old_col)]: new_col
                for new_col, old_col in enumerate(top_cols)
            }
            self._binarizer.n_items = len(top_cols)

        self._split = dict(
            x_train=x_train,
            x_val=x_val,
            x_test=x_test,
            y_train=np.array(sorted(train_users), dtype=np.int32),
            y_val=np.array(sorted(val_users), dtype=np.int32),
            y_test=np.array(sorted(test_users), dtype=np.int32),
        )
        return self._split

    def get_movies(self) -> pd.DataFrame:
        """Return the movies DataFrame with columns: movie_id, title, genres."""
        return self.movies.copy()

    def get_genres(self) -> pd.DataFrame:
        """Return a one-hot genre DataFrame indexed by movie_id.

        Columns are the 18 ML-1M genre strings. Value is 1 if the movie
        belongs to that genre, 0 otherwise.
        """
        df = self.movies.copy()
        for genre in ALL_GENRES:
            df[genre] = df["genres"].str.contains(genre, regex=False).astype(np.uint8)
        return df[["movie_id", "title", *ALL_GENRES]].set_index("movie_id")

    def get_users(self) -> pd.DataFrame:
        """Return the users DataFrame with demographic label columns."""
        return self.users.copy()

    def movie_id_to_title(self, movie_id: Union[int, list]) -> Union[str, None, list]:
        """Translate one or more MovieLens movie IDs to their titles.

        Args:
            movie_id: A single integer ID or a list / array of IDs.

        Returns:
            A title string (or None if not found), or a list thereof.
        """
        lookup = self.movies.set_index("movie_id")["title"]
        if isinstance(movie_id, (list, np.ndarray)):
            return [lookup.get(int(mid)) for mid in movie_id]
        return lookup.get(int(movie_id))

    def col_to_movie_id(self) -> dict[int, int]:
        """Return a mapping from item matrix column index to movie_id.

        Useful for interpreting which column in x_train / x_test corresponds
        to which movie. Requires get() to have been called first.
        """
        if self._binarizer is None:
            raise RuntimeError("Call get() first to fit the binarizer.")
        return {col: mid for mid, col in self._binarizer.item_index.items()}

    # Dataset-neutral aliases so dataset-agnostic consumers (clause analysis,
    # the explorer notebooks) can treat MovieLens1M and VibrentRental uniformly.
    def item_id_to_name(self, item_id: Union[int, list]) -> Union[str, None, list]:
        """Alias of :meth:`movie_id_to_title` for the shared dataset interface."""
        return self.movie_id_to_title(item_id)

    def col_to_item_id(self) -> dict[int, int]:
        """Alias of :meth:`col_to_movie_id` for the shared dataset interface."""
        return self.col_to_movie_id()

    @property
    def binarizer(self) -> UserItemBinarizer:
        """The fitted UserItemBinarizer (available after calling get())."""
        if self._binarizer is None:
            raise RuntimeError("Call get() first to fit the binarizer.")
        return self._binarizer
