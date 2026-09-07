"""Grid-search the baseline recommenders on MovieLens 1M.

Each model is fit on x_train and its hyperparameters selected on x_val by a
single validation metric; the held-out x_test metrics are then reported at the
selected configuration. This matches the validation-based selection used for the
TM (experiments/coalesced_hpo.py), so the baselines are tuned under the same
leak-free protocol rather than left at fixed defaults.
"""

import argparse
import itertools
import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

# Import for side effect: caps BLAS/OpenMP threads BEFORE numpy loads (see
# utils/cpu_threads).
import utils.blas_cap  # noqa: F401

import numpy as np
import yaml

_PROJ_ROOT = Path(__file__).parent.parent
DEFAULT_DIRECTORY = _PROJ_ROOT / "configs" / "baselines_hpo"

from datasets import get_dataset
from utils.metrics import evaluate

from utils.latex_table import results_to_latex

from run_all import build_samples

from apriori_rec import AprioriRecommender
from decision_tree_rec import DecisionTreeRecommender
from ease_rec import EASERecommender
from item_knn import ItemKNNRecommender
from popularity_rec import PopularityRecommender
from random_rec import RandomRecommender
from svd_rec import SVDRecommender
from user_knn import UserKNNRecommender

log = logging.getLogger(__name__)

# Model name (as used in the config) -> class. Constructor kwargs come from the
# config grid, so the names here are the only coupling to the config file.
REGISTRY = {
    "Random": RandomRecommender,
    "Popularity": PopularityRecommender,
    "Item-kNN": ItemKNNRecommender,
    "User-kNN": UserKNNRecommender,
    "SVD": SVDRecommender,
    "DecisionTree": DecisionTreeRecommender,
    "EASE": EASERecommender,
    "Apriori": AprioriRecommender,
}


def grid_product(grid: dict) -> list[dict]:
    """Expand a {param: [values]} grid into a list of concrete kwarg dicts."""
    if not grid:
        return [{}]
    keys = list(grid)
    return [
        dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))
    ]


def _display(name: str, params: dict) -> str:
    if not params:
        return name
    inner = ", ".join(f"{k}={v}" for k, v in params.items())
    return f"{name} ({inner})"


def tune_model(
    name: str,
    cls,
    grid: dict,
    x_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    item_popularity: np.ndarray,
    n_items: int,
    ks: tuple,
    select_metric: str,
) -> tuple[dict, dict, dict]:
    """Fit every grid point, select on validation, return (params, val_m, test_m).

    The selected model is kept fitted, so the test metrics are computed on the
    same trained model without refitting.
    """
    eval_kwargs = dict(ks=ks, item_popularity=item_popularity, n_items=n_items)
    combos = grid_product(grid)

    best_score, best_params, best_val, best_model = -np.inf, None, None, None
    for params in combos:
        t0 = time.time()
        model = cls(**params).fit(x_train)
        val_m = evaluate(model, X_val, Y_val, **eval_kwargs)
        score = val_m[select_metric]
        log.info(
            f"  {name} {params or '(no params)'}: "
            f"val {select_metric}={score:.4f}  [{time.time() - t0:.1f}s]"
        )
        if score > best_score:
            best_score, best_params, best_val, best_model = score, params, val_m, model

    test_m = evaluate(best_model, X_test, Y_test, **eval_kwargs)
    log.info(
        f"  -> {name} selected {best_params or '(none)'}: "
        f"val {select_metric}={best_score:.4f}  test {select_metric}={test_m[select_metric]:.4f}"
    )
    return best_params, best_val, test_m


def main(args: SimpleNamespace) -> None:
    cfg = args.cfg
    data_cfg = cfg["data"]
    eval_cfg = cfg["evaluation"]
    models_cfg = cfg["models"]
    out_cfg = cfg.get("output", {})

    ks = tuple(eval_cfg["ks"])
    select_metric = eval_cfg["select_metric"]
    if select_metric not in {
        f"{m}@{k}" for m in ("hit", "ndcg", "novelty", "coverage") for k in ks
    }:
        raise ValueError(
            f"select_metric={select_metric!r} is not computable from ks={ks}"
        )

    val_ratio = data_cfg.get("val_ratio", 0.1)
    if val_ratio <= 0:
        raise ValueError("val_ratio must be > 0 for validation-based selection")

    dataset_name = data_cfg.get("dataset", "movielens1m")
    log.info(f"Loading dataset: {dataset_name}...")
    ds = get_dataset(
        dataset_name,
        rating_threshold=data_cfg.get("rating_threshold", 4),
        k_core=data_cfg.get("k_core", 5),
        unit=data_cfg.get("unit", "group"),
        val_ratio=val_ratio,
        max_items=data_cfg["max_items"],
    )
    data = ds.get()
    n_items = data["x_train"].shape[1]
    log.info(
        f"Train/Val/Test users: {len(data['y_train'])}/{len(data['y_val'])}/"
        f"{len(data['y_test'])}  Items: {n_items}"
    )

    X_test, Y_test = build_samples(data["x_test"], max_per_user=None)
    X_val, Y_val = build_samples(data["x_val"], max_per_user=None)
    if len(X_val) == 0:
        raise RuntimeError("No validation samples — increase val_ratio.")
    log.info(f"  {len(X_val)} val samples, {len(X_test)} test samples")

    item_popularity = data["x_train"].sum(axis=0) / len(data["y_train"])

    test_results: dict[str, dict] = {}
    selected: dict[str, dict] = {}
    for name, grid in models_cfg.items():
        if name not in REGISTRY:
            raise ValueError(f"Unknown model {name!r}; known: {sorted(REGISTRY)}")
        grid = grid or {}
        log.info(f"Tuning {name} ({len(grid_product(grid))} configs)...")
        params, _, test_m = tune_model(
            name,
            REGISTRY[name],
            grid,
            data["x_train"],
            X_val,
            Y_val,
            X_test,
            Y_test,
            item_popularity,
            n_items,
            ks,
            select_metric,
        )
        test_results[_display(name, params)] = test_m
        selected[name] = params

    ordered_cols = (
        [f"hit@{k}" for k in ks]
        + [f"ndcg@{k}" for k in ks]
        + [f"novelty@{k}" for k in ks]
        + [f"coverage@{k}" for k in ks]
    )
    sample = next(iter(test_results.values()))
    cols = [c for c in ordered_cols if c in sample]

    col_w = 9
    name_w = max(len(n) for n in test_results) + 2
    header = f"{'Model':<{name_w}}" + "".join(f"{c:>{col_w}}" for c in cols)
    sep = "=" * len(header)
    print(f"\n{sep}")
    print(f"BASELINE GRID SEARCH -- held-out TEST (selected on val {select_metric})")
    print(sep)
    print(header)
    print("-" * len(header))
    for name, m in test_results.items():
        print(
            f"{name:<{name_w}}"
            + "".join(f"{m.get(c, float('nan')):>{col_w}.4f}" for c in cols)
        )
    print(sep)

    print("\nSelected configurations:")
    for name, params in selected.items():
        print(f"  {name}: {params or '(nothing to tune)'}")

    if out_cfg.get("latex"):
        latex = results_to_latex(
            test_results,
            ks=list(ks),
            caption=(
                "Baseline recommender results on MovieLens 1M, tuned on a "
                f"validation split (selected on {select_metric})."
            ),
            label="tab:baselines_tuned",
        )
        print(f"\n{latex}")
        latex_out = out_cfg.get("latex_out")
        if latex_out:
            out_path = Path(latex_out)
            if not out_path.is_absolute():
                out_path = _PROJ_ROOT / out_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(latex, encoding="utf-8")
            log.info(f"Wrote LaTeX table to {out_path}")


def parse_args(argv=None) -> SimpleNamespace:
    p = argparse.ArgumentParser(description="Grid search the MovieLens 1M baselines")
    p.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_DIRECTORY / "movielens.yml",
        help="Path to YAML config (or a bare name under configs/baselines_hpo/).",
    )
    p.add_argument(
        "--max_items",
        type=int,
        default=None,
        help="Override data.max_items (handy for fast smoke runs).",
    )
    p.add_argument(
        "--select_metric",
        type=str,
        default=None,
        help="Override evaluation.select_metric.",
    )
    cli = p.parse_args(argv)

    config_path = cli.config
    if not config_path.is_file():
        config_path = DEFAULT_DIRECTORY / f"{config_path.name}.yml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if cli.max_items is not None:
        cfg["data"]["max_items"] = cli.max_items
    if cli.select_metric is not None:
        cfg["evaluation"]["select_metric"] = cli.select_metric

    log.info(f"Config: {config_path}")
    return SimpleNamespace(cfg=cfg)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    main(parse_args())
