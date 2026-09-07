"""Evaluate all baseline recommenders on MovieLens 1M and print a comparison table."""

import argparse
import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

_PROJ_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJ_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

DEFAULT_DIRECTORY = _PROJ_ROOT / "configs" / "baselines"

from datasets import get_dataset
from utils.metrics import evaluate

from utils.latex_table import results_to_latex

from decision_tree_rec import DecisionTreeRecommender
from item_knn import ItemKNNRecommender
from popularity_rec import PopularityRecommender
from random_rec import RandomRecommender
from svd_rec import SVDRecommender
from user_knn import UserKNNRecommender
from ease_rec import EASERecommender
from apriori_rec import AprioriRecommender

log = logging.getLogger(__name__)


def build_samples(
    x_matrix: np.ndarray,
    max_per_user: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build masked item prediction samples from a binary user-item matrix.

    For each positively-rated item per user, produces one sample:
      X[i] = user rating vector with the target item column zeroed out
      Y[i] = column index of the target item
    """
    if rng is None:
        rng = np.random.default_rng(42)
    rows_x, rows_y = [], []
    for user_vec in x_matrix:
        liked = np.where(user_vec == 1)[0]
        if max_per_user is not None and len(liked) > max_per_user:
            liked = rng.choice(liked, max_per_user, replace=False)
        for col in liked:
            masked = user_vec.copy()
            masked[col] = 0
            rows_x.append(masked)
            rows_y.append(col)
    return np.array(rows_x, dtype=np.uint32), np.array(rows_y, dtype=np.uint32)


def run_baseline(
    name: str,
    model,
    x_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    item_popularity: np.ndarray,
    n_items: int,
    ks: tuple,
) -> dict:
    log.info(f"Fitting {name}...")
    t0 = time.time()
    model.fit(x_train)
    fit_time = time.time() - t0

    log.info(f"Evaluating {name}...")
    t0 = time.time()
    metrics = evaluate(
        model, X_test, Y_test, ks=ks, item_popularity=item_popularity, n_items=n_items
    )
    eval_time = time.time() - t0

    metric_str = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
    log.info(f"  {name}: {metric_str}  [fit={fit_time:.1f}s eval={eval_time:.1f}s]")
    return metrics


def main(args: argparse.Namespace) -> None:
    dataset_name = getattr(args, "dataset", "movielens1m")
    log.info(f"Loading dataset: {dataset_name}...")
    ds = get_dataset(
        dataset_name,
        rating_threshold=getattr(args, "rating_threshold", 4),
        k_core=getattr(args, "k_core", 5),
        unit=getattr(args, "unit", "group"),
        max_items=args.max_items,
    )
    data = ds.get()
    n_items = data["x_train"].shape[1]
    log.info(
        f"Train users: {len(data['y_train'])}  "
        f"Test users: {len(data['y_test'])}  "
        f"Items: {n_items}"
    )

    log.info("Building test samples (every positive held out once per user)...")
    X_test, Y_test = build_samples(data["x_test"], max_per_user=None)
    log.info(f"  {len(X_test)} test samples")

    item_popularity = data["x_train"].sum(axis=0) / len(data["y_train"])
    ks = tuple(args.ks)

    baselines = [
        ("Random", RandomRecommender(seed=42)),
        ("Popularity", PopularityRecommender()),
        (f"Item-kNN (k={args.item_knn})", ItemKNNRecommender(k=args.item_knn)),
        (f"User-kNN (k={args.user_knn})", UserKNNRecommender(k=args.user_knn)),
        (f"SVD (d={args.n_factors})", SVDRecommender(n_factors=args.n_factors)),
        (
            f"DecisionTree (depth={args.dt_max_depth}, min_samples_leaf={args.dt_min_samples_leaf})",
            DecisionTreeRecommender(max_depth=args.dt_max_depth, min_samples_leaf=args.dt_min_samples_leaf),
        ),
        (f"EASE (l2={args.ease_l2:g})", EASERecommender(l2=args.ease_l2)),
        (
            f"Apriori (s={args.apriori_min_support:g}, c={args.apriori_min_confidence:g})",
            AprioriRecommender(
                min_support=args.apriori_min_support,
                min_confidence=args.apriori_min_confidence,
                max_len=args.apriori_max_len,
            ),
        ),
    ]

    all_results: dict[str, dict] = {}
    for name, model in baselines:
        all_results[name] = run_baseline(
            name, model, data["x_train"], X_test, Y_test, item_popularity, n_items, ks
        )

    sample_metrics = next(iter(all_results.values()))
    ordered_cols = (
        [f"hit@{k}" for k in ks]
        + [f"ndcg@{k}" for k in ks]
        + [f"novelty@{k}" for k in ks]
        + [f"coverage@{k}" for k in ks]
    )
    cols = [c for c in ordered_cols if c in sample_metrics]

    col_w = 9
    name_w = max(len(n) for n in all_results) + 2
    header = f"{'Model':<{name_w}}" + "".join(f"{c:>{col_w}}" for c in cols)
    sep = "=" * len(header)

    print(f"\n{sep}")
    print("BASELINE COMPARISON")
    print(sep)
    print(header)
    print("-" * len(header))
    for name, m in all_results.items():
        row = f"{name:<{name_w}}" + "".join(
            f"{m.get(c, float('nan')):>{col_w}.4f}" for c in cols
        )
        print(row)
    print(sep)

    if args.latex:
        latex = results_to_latex(
            all_results,
            ks=list(ks),
            caption="Baseline recommender results on MovieLens 1M.",
            label="tab:baselines",
        )
        print(f"\n{latex}")
        if args.latex_out:
            out_path = Path(args.latex_out)
            if not out_path.is_absolute():
                out_path = _PROJ_ROOT / out_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(latex, encoding="utf-8")
            log.info(f"Wrote LaTeX table to {out_path}")


def load_config(config_path) -> SimpleNamespace:
    """Load a YAML config and flatten its sections into a SimpleNamespace.

    Accepts a direct file path or a bare config name relative to
    configs/baselines/ (e.g. --config default).
    """
    config_path = Path(config_path)
    if not config_path.is_file():
        config_path = DEFAULT_DIRECTORY / f"{config_path.name}.yml"

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    flat: dict = {}
    for section in raw.values():
        flat.update(section)
    return SimpleNamespace(**flat)


def parse_args(argv=None) -> SimpleNamespace:
    """Load --config, then apply any explicitly-set CLI flags on top."""
    p = argparse.ArgumentParser(
        description="Run all baseline recommenders on MovieLens 1M"
    )
    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_DIRECTORY / "movielens.yml",
        help="Path to YAML config (or a bare name under configs/baselines/).",
    )
    # CLI overrides default to None so config values win unless explicitly set.
    p.add_argument(
        "--max_items",
        type=int,
        default=None,
        help="Restrict to top-N items by popularity (0 = full catalogue)",
    )
    p.add_argument(
        "--dataset", type=str, default=None, help="Dataset key (see datasets.DATASETS)"
    )
    p.add_argument("--rating_threshold", type=int, default=None)
    p.add_argument(
        "--k_core", type=int, default=None, help="k-core threshold (implicit datasets)"
    )
    p.add_argument(
        "--unit",
        type=str,
        default=None,
        help="Vibrent item granularity: group | outfit",
    )
    p.add_argument(
        "--item_knn", type=int, default=None, help="Number of neighbours for Item-kNN"
    )
    p.add_argument(
        "--user_knn", type=int, default=None, help="Number of neighbours for User-kNN"
    )
    p.add_argument("--n_factors", type=int, default=None, help="Latent factors for SVD")
    p.add_argument(
        "--dt_max_depth", type=int, default=None, help="Max depth for the decision tree"
    )
    p.add_argument(
        "--dt_min_samples_leaf", type=int, default=None, help="Min samples per leaf for the decision tree"
    )
    p.add_argument("--ease_l2", type=float, default=None, help="EASE L2 regularisation")
    p.add_argument(
        "--apriori_min_support",
        type=float,
        default=None,
        help="Apriori minimum itemset support (fraction of users)",
    )
    p.add_argument(
        "--apriori_min_confidence",
        type=float,
        default=None,
        help="Apriori minimum rule confidence",
    )
    p.add_argument(
        "--apriori_max_len",
        type=int,
        default=None,
        help="Apriori maximum frequent-itemset size",
    )
    p.add_argument(
        "--ks",
        type=int,
        nargs="+",
        default=None,
        help="Cutoff values for metrics (e.g. --ks 1 10 50)",
    )
    p.add_argument(
        "--latex",
        action="store_true",
        default=None,
        help="Also print a LaTeX results table",
    )
    p.add_argument(
        "--latex_out",
        type=str,
        default=None,
        help="Optional path to write the LaTeX table to",
    )

    cli = p.parse_args(argv)
    cfg = load_config(cli.config)

    # Explicitly-set CLI flags (non-None) override the config file.
    for key, val in vars(cli).items():
        if key == "config" or val is None:
            continue
        setattr(cfg, key, val)

    log.info(f"Config: {cli.config}")
    for key, val in vars(cfg).items():
        log.info(f"  {key}: {val}")
    return cfg


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    main(parse_args())
