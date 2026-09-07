import inspect

from .movielens import (
    ALL_GENRES,
    AGE_MAP,
    OCCUPATION_MAP,
    MovieLens1M,
    UserItemBinarizer,
)
from .vibrent import VibrentRental

__all__ = [
    "MovieLens1M",
    "VibrentRental",
    "UserItemBinarizer",
    "ALL_GENRES",
    "AGE_MAP",
    "OCCUPATION_MAP",
    "get_dataset",
    "DATASETS",
]

# Registry of selectable datasets. Aliases map to the canonical loader class.
DATASETS = {
    "movielens1m": MovieLens1M,
    "ml-1m": MovieLens1M,
    "ml1m": MovieLens1M,
    "vibrent": VibrentRental,
    "vibrent_rental": VibrentRental,
}


def get_dataset(name: str, **cfg):
    """Construct a dataset loader by name, passing only the kwargs it accepts.

    Callers (experiment / baseline configs) may pass a superset of dataset
    arguments — e.g. ``rating_threshold`` (MovieLens-only) alongside ``k_core``
    (Vibrent-only). Each loader receives only the subset its ``__init__`` declares,
    so a single call site works for every dataset.

    Args:
        name: dataset key (case-insensitive); see ``DATASETS`` for valid aliases.
        **cfg: candidate constructor arguments; unknown-to-this-dataset keys are
               silently dropped.

    Returns:
        An instantiated dataset loader exposing the shared ``.get()`` contract.
    """
    key = name.lower()
    if key not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}. Valid options: {sorted(DATASETS)}")
    cls = DATASETS[key]
    accepted = set(inspect.signature(cls.__init__).parameters) - {"self"}
    kwargs = {k: v for k, v in cfg.items() if k in accepted}
    return cls(**kwargs)
