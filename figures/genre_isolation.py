"""Genre co-taste lift heatmap from a trained coalesced-TM run.

Builds the genre x genre co-taste flow from the model's clauses -- trigger genres
from each clause's positive literals, recommendation genres from its positive
output weights -- normalises to a lift over the base recommendation rate, and
renders a log2-lift heatmap. Genres are ordered by hierarchical clustering of the
lift matrix so the learned blocks (the family-film cluster of Animation /
Children's / Musical, the noir/mystery cluster, ...) appear as contiguous diagonal
blocks. Documentary appears as a blank row -- it never triggers a clause.

Usage:
    python figures/genre_isolation.py [--run_dir runs/coalesced_baseline/<run>] [--no_idf]
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt

_PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJ))

from datasets import MovieLens1M
from datasets.movielens import ALL_GENRES
from utils.checkpoint import load_tm_cpu
from utils.clause_analysis import decode_ta_states, host_clause_bank

OUT_STEM = _PROJ / "figures" / "interpretability" / "genre_lift"
DEFAULT_RUN = _PROJ / "runs" / "coalesced_baseline" / "20260702T133243_feedback_masking"


def rownorm(A):
    s = A.sum(axis=1, keepdims=True)
    return np.divide(A, s, out=np.zeros_like(A), where=s > 0)


def compute_lift(run_dir: Path, use_idf: bool = True) -> np.ndarray:
    """Return the (n_genres, n_genres) trigger->recommendation lift matrix."""
    with open(run_dir / "metadata.json") as f:
        cfg = SimpleNamespace(**json.load(f)["config"])
    # Last epoch, matching experiments/clause_cf_eval.py, so every number quoted
    # in the paper comes from the same model state as the results tables.
    tm = load_tm_cpu(sorted(run_dir.glob("epoch_*.pkl"))[-1])
    n_items = tm.number_of_classes
    n_feat = tm.clause_bank.number_of_features

    ds = MovieLens1M(
        rating_threshold=cfg.rating_threshold,
        test_ratio=0.2,
        min_ratings=20,
        random_state=42,
        max_items=cfg.max_classes,
    )
    ds.get()
    col_map = ds.col_to_movie_id()
    genres_df = ds.get_genres()

    G = np.zeros((n_items, len(ALL_GENRES)), dtype=np.float32)
    for col in range(n_items):
        mid = col_map[col]
        if mid in genres_df.index:
            G[col] = genres_df.loc[mid, ALL_GENRES].to_numpy(dtype=np.float32)
    prevalence = G.mean(axis=0)
    Gw = G * np.log(1.0 / np.clip(prevalence, 1e-9, None))[None, :] if use_idf else G

    states = decode_ta_states(host_clause_bank(tm))
    L_pos = (states[:, :n_feat] >= 128).astype(np.float32)
    W_pos = np.maximum(
        0.0,
        np.stack([tm.weight_banks[i].get_weights() for i in range(n_items)]).astype(
            np.float32
        ),
    )

    P = rownorm(L_pos @ Gw)
    R = rownorm(W_pos.T @ Gw)
    w_c = W_pos.sum(axis=0)
    M = rownorm((P * w_c[:, None]).T @ R)
    pi = (P * w_c[:, None]).sum(axis=0)
    pi = pi / pi.sum()
    base = pi @ M
    return M / np.where(base > 0, base, 1.0)


def cluster_order(M_lift: np.ndarray) -> np.ndarray:
    """Hierarchical-clustering leaf order on the symmetrised log2-lift."""
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import pdist

        L2 = np.log2(np.where(M_lift > 0, M_lift, 1.0))
        S = (L2 + L2.T) / 2
        return leaves_list(linkage(pdist(S), method="average"))
    except Exception as exc:  # scipy missing or degenerate -> keep ALL_GENRES order
        print(f"[warn] clustering unavailable ({exc}); using alphabetical order")
        return np.arange(len(ALL_GENRES))


def main(args: argparse.Namespace) -> None:
    M_lift = compute_lift(Path(args.run_dir), use_idf=not args.no_idf)
    order = cluster_order(M_lift)
    labels = [ALL_GENRES[i] for i in order]

    L2 = np.log2(np.where(M_lift > 0, M_lift, np.nan))[np.ix_(order, order)]
    vmax = float(np.nanmax(np.abs(L2)))

    plt.rcParams.update({"font.size": 8})
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    im = ax.imshow(L2, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Recommended genre")
    ax.set_ylabel("Trigger genre")
    cb = fig.colorbar(im, fraction=0.046, pad=0.04)
    cb.set_label("co-taste lift ($\\log_2$)")

    fig.tight_layout()
    OUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    for ext, kwargs in (("pdf", {}), ("png", {"dpi": 200})):
        out = OUT_STEM.with_suffix(f".{ext}")
        fig.savefig(out, bbox_inches="tight", **kwargs)
        print(f"wrote {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir", default=str(DEFAULT_RUN))
    p.add_argument("--no_idf", action="store_true", help="disable IDF genre weighting")
    main(p.parse_args())
