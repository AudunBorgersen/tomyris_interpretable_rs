"""
Converted from notebook

Load a saved coalesced-TM run and inspect it.

Reports the run's config and per-epoch metrics, re-evaluates the selected
checkpoint on held-out test users, and dumps interpretability views: the top
FOR/AGAINST clauses for the most frequent classes, class-level clause
top-heaviness, literal inclusion frequency, and the item-weight map of a single
clause. The plotly views are written to standalone HTML files.

Usage:
    python experiments/model_explorer.py [--run_dir <run>] \
        [--checkpoint_index -1] [--clause_idx 59] [--out_dir <dir>]
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

_PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJ))

from datasets import get_dataset  # noqa: E402
from experiments.coalesced_baseline import build_samples  # noqa: E402
from utils.checkpoint import load_tm_cpu  # noqa: E402
from utils.clause_analysis import (  # noqa: E402
    analyze_clauses_for_movie,
    literal_inclusion_frequency,
)
from utils.metrics import clause_weight_stats, evaluate  # noqa: E402

DEFAULT_RUN = (
    _PROJ
    / "runs"
    / "coalesced_baseline"
    / "Movielens_final_runs"
    / "20260721T130000_movielens4"
)
DEFAULT_OUT = _PROJ / "figures" / "model_explorer"


def load_metadata(run_dir: Path):
    with open(run_dir / "metadata.json") as f:
        metadata = json.load(f)
    cfg = SimpleNamespace(**metadata["config"])
    print("Config:")
    for k, v in metadata["config"].items():
        print(f"  {k}: {v}")
    epoch_df = pd.DataFrame(metadata["epochs"]).set_index("epoch")
    return cfg, epoch_df


def rebuild_dataset(cfg):
    """Rebuild the dataset the run was trained on, routing by its saved config so
    the splits match exactly. Vibrent runs carry dataset/k_core/unit; MovieLens
    runs fall back to the defaults."""
    ds = get_dataset(
        getattr(cfg, "dataset", "movielens1m"),
        rating_threshold=getattr(cfg, "rating_threshold", 4),
        k_core=getattr(cfg, "k_core", 5),
        unit=getattr(cfg, "unit", "group"),
        test_ratio=0.2,
        min_ratings=20,
        random_state=42,
        max_items=cfg.max_classes,
    )
    data = ds.get()
    col_map = ds.col_to_item_id()  # column index -> item id (movie_id / outfit group)
    n_items = data["x_train"].shape[1]
    print(
        f"Items: {n_items}  Train users: {len(data['y_train'])}  Test users: {len(data['y_test'])}"
    )

    rng = np.random.default_rng(cfg.seed)
    max_pu = cfg.max_per_user if cfg.max_per_user else None
    X_train, Y_train = build_samples(data["x_train"], max_pu, rng)
    X_test, Y_test = build_samples(data["x_test"], max_per_user=1, rng=rng)
    item_popularity = data["x_train"].sum(axis=0) / len(data["y_train"])
    print(f"Train samples: {len(X_train):,}   Test samples: {len(X_test):,}")
    return ds, data, col_map, n_items, X_train, Y_train, X_test, Y_test, item_popularity


def report_metrics(tm, X_test, Y_test, item_popularity, n_items):
    m = evaluate(
        tm,
        X_test,
        Y_test,
        ks=(1, 10, 50),
        item_popularity=item_popularity,
        n_items=n_items,
    )
    ws = clause_weight_stats(tm)
    print("\nRanking metrics:")
    for k, v in m.items():
        print(f"  {k}: {v:.4f}")
    print("\nClause weight stats:")
    for k, v in ws.items():
        print(f"  {k}: {v}")


def report_top_class_clauses(tm, ds, col_map, Y_train, n_items):
    """Top FOR/AGAINST clauses for the most frequently targeted classes."""
    freq_per_class = np.bincount(Y_train, minlength=n_items)
    for col in np.argsort(-freq_per_class)[:10]:
        if freq_per_class[col] > 0:
            analyze_clauses_for_movie(
                tm, int(col), col_map, ds, n_top=3, ignore_self_inclusion=True
            )


def plot_top_heaviness(tm, titles, n_items, out_path):
    """Number of positive-weight clauses per class -- how top-heavy each class is."""
    n_pos_per_class = np.array(
        [(tm.get_weights(i) > 0).sum() for i in range(tm.number_of_classes)]
    )
    order = np.argsort(-n_pos_per_class)
    fig = go.Figure(
        go.Bar(
            x=[titles[i] for i in order],
            y=n_pos_per_class[order],
            hovertemplate="%{x}<br>%{y} positive clauses<extra></extra>",
        )
    )
    fig.update_layout(
        title="Clause top-heaviness by class",
        xaxis_title="Item",
        yaxis_title="# positive-weight clauses",
        xaxis=dict(tickangle=45),
        height=500,
    )
    fig.write_html(out_path)
    print(f"  wrote {out_path}")
    top = pd.Series(n_pos_per_class, index=titles, name="pos_clauses").sort_values(
        ascending=False
    )
    print("\nMost top-heavy classes:")
    print(top.head(10).to_string())


def plot_literal_frequency(tm, titles, out_path):
    """How often each item appears as a positive (likes) / negative (not_likes) literal."""
    freq = literal_inclusion_frequency(tm)
    n_feat = tm.clause_bank.number_of_features
    fig = make_subplots(rows=1, cols=2, subplot_titles=["likes", "not_likes"])
    for col, (f, label) in enumerate(
        [(freq[:n_feat], "likes"), (freq[n_feat:], "not_likes")], start=1
    ):
        order = np.argsort(-f)
        fig.add_trace(
            go.Bar(
                x=[titles[i] for i in order],
                y=f[order],
                name=label,
                hovertemplate="%{x}<br>%{y} clauses<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=col,
        )
    fig.update_layout(title="Literal inclusion frequency", height=500)
    fig.update_xaxes(tickangle=45)
    fig.write_html(out_path)
    print(f"  wrote {out_path}")


def plot_clause_weight_map(tm, ds, col_map, titles, n_items, clause_idx, out_path):
    """Weight of a single clause across every item -- which items it votes for/against."""
    clause_weights = np.array([tm.get_weights(i)[clause_idx] for i in range(n_items)])
    order = np.argsort(clause_weights)
    sorted_weights = clause_weights[order]
    sorted_titles = [titles[i] for i in order]

    fig = go.Figure(
        go.Bar(
            x=sorted_titles,
            y=sorted_weights,
            marker_color=["tomato" if w < 0 else "steelblue" for w in sorted_weights],
            hovertemplate="%{x}<br>weight: %{y}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Clause {clause_idx} - weight across all items  (blue = FOR, red = AGAINST)",
        xaxis_title="Item",
        yaxis_title="Weight",
        xaxis=dict(tickangle=45),
        height=500,
    )
    fig.add_hline(y=0, line_color="black", line_width=0.8)
    fig.write_html(out_path)
    print(f"  wrote {out_path}")

    included = [
        i
        for i in range(tm.clause_bank.number_of_features)
        if tm.get_ta_action(clause_idx, i)
    ]
    print(f"\nClause {clause_idx} - {len(included)} included literals:")
    for lit in included:
        title = ds.item_id_to_name(col_map[lit]) if lit in col_map else f"feature {lit}"
        print(f"  likes: {title}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--run_dir",
        type=Path,
        default=DEFAULT_RUN,
        help="coalesced_baseline run directory",
    )
    p.add_argument(
        "--checkpoint_index",
        type=int,
        default=-1,
        help="index into sorted epoch checkpoints",
    )
    p.add_argument(
        "--clause_idx", type=int, default=59, help="clause to inspect in the weight map"
    )
    p.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_OUT,
        help="directory for saved HTML figures",
    )
    args = p.parse_args()

    assert args.run_dir.exists(), f"Not found: {args.run_dir}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {args.run_dir}")

    # --- Metadata ---
    cfg, epoch_df = load_metadata(args.run_dir)
    print("\nPer-epoch metrics:")
    print(epoch_df.to_string())

    # --- Checkpoint ---
    checkpoints = sorted(args.run_dir.glob("epoch_*.pkl"))
    tm = load_tm_cpu(checkpoints[args.checkpoint_index])
    print(
        f"\nClauses: {tm.clause_bank.number_of_clauses}  "
        f"Features: {tm.clause_bank.number_of_features}  Classes: {tm.number_of_classes}"
    )

    # --- Dataset (rebuilt from the saved config) ---
    ds, data, col_map, n_items, X_train, Y_train, X_test, Y_test, item_popularity = (
        rebuild_dataset(cfg)
    )
    titles = [ds.item_id_to_name(col_map[i]) or f"col{i}" for i in range(n_items)]

    # --- Evaluate ---
    report_metrics(tm, X_test, Y_test, item_popularity, n_items)

    # --- Clause analysis ---
    report_top_class_clauses(tm, ds, col_map, Y_train, n_items)

    # --- Plotly views ---
    print("\nSaving figures:")
    plot_top_heaviness(tm, titles, n_items, args.out_dir / "clause_top_heaviness.html")
    plot_literal_frequency(
        tm, titles, args.out_dir / "literal_inclusion_frequency.html"
    )
    plot_clause_weight_map(
        tm,
        ds,
        col_map,
        titles,
        n_items,
        args.clause_idx,
        args.out_dir / f"clause_{args.clause_idx}_weight_map.html",
    )


if __name__ == "__main__":
    main()
