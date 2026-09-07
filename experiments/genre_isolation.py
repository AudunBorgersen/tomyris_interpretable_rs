"""
Converted from notebook

Genre isolation in clause space: a global interpretability probe.

Do the TM's clauses organise taste into genre-coherent structure, and is the
Children's genre as isolated as we would expect? Each clause is a learned
co-taste rule; we characterise clauses by the genres of the movies they *trigger*
on (positive literals) and the genres of the movies they *recommend* (positive
clause-output weights), then aggregate to a genre x genre co-taste flow matrix.

Findings:
  * The isolated unit is not Children's alone but a *family-film block*
    (Animation / Children's / Musical). Most kids' films carry several genre tags
    (Toy Story = Animation|Children's|Comedy), so the single-genre signal is
    diluted; isolation is read from a *lift* matrix and from the block rather than
    from one diagonal cell.
  * Documentaries never appear on the trigger side. Clause literals are drawn from
    a small popular core of items, and no documentary is watched enough to enter
    it -- a popularity-driven coverage limitation, not a sign that documentary
    taste lacks structure (their within-genre audience co-taste lift is in fact
    high).

Caveat: MovieLens raters are predominantly adults, so this reflects the co-taste
of (largely adult) raters of children's films -- parents, nostalgic viewers --
rather than of children themselves.

Usage:
    python experiments/genre_isolation.py [--run_dir <run>] [--no_idf] [--out_dir <dir>]
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # save figures without requiring a display
import matplotlib.pyplot as plt  # noqa: E402

_PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJ))

from datasets import MovieLens1M  # noqa: E402
from datasets.movielens import ALL_GENRES  # noqa: E402
from utils.checkpoint import load_tm_cpu  # noqa: E402
from utils.clause_analysis import (  # noqa: E402
    clause_pattern,
    decode_ta_states,
    host_clause_bank,
)

DEFAULT_RUN = _PROJ / "runs" / "coalesced_baseline" / "20260702T133243_feedback_masking"
DEFAULT_OUT = _PROJ / "figures" / "interpretability"

# Genres that form the mutually-reinforcing "family-film" block in the lift matrix.
FAMILY = ["Animation", "Children's", "Musical"]


def rownorm(A: np.ndarray) -> np.ndarray:
    s = A.sum(axis=1, keepdims=True)
    return np.divide(A, s, out=np.zeros_like(A), where=s > 0)


def load_run(run_dir: Path):
    """Load the run's config, final-epoch checkpoint, and MovieLens dataset."""
    with open(run_dir / "metadata.json") as f:
        cfg = SimpleNamespace(**json.load(f)["config"])
    # Final-epoch checkpoint, matching figures/genre_isolation.py and the paper's
    # results tables, so every number quoted comes from the same model state.
    tm = load_tm_cpu(sorted(run_dir.glob("epoch_*.pkl"))[-1])

    ds = MovieLens1M(
        rating_threshold=cfg.rating_threshold,
        test_ratio=0.2,
        min_ratings=20,
        random_state=42,
        max_items=cfg.max_classes,
    )
    data = ds.get()
    print(
        f"Run: {run_dir.name} | clauses: {cfg.clauses} | feature_negation: {cfg.feature_negation}"
    )
    return cfg, tm, ds, data


def genre_matrix(tm, ds, use_idf: bool):
    """Return (G, Gw, prevalence): item x genre one-hot, IDF-weighted, prevalence."""
    n_items = tm.number_of_classes
    col_map = ds.col_to_movie_id()
    genres_df = ds.get_genres()

    G = np.zeros((n_items, len(ALL_GENRES)), dtype=np.float32)
    for col in range(n_items):
        mid = col_map[col]
        if mid in genres_df.index:
            G[col] = genres_df.loc[mid, ALL_GENRES].to_numpy(dtype=np.float32)

    prevalence = G.mean(axis=0)  # fraction of the catalogue tagged with each genre
    # Optional IDF weighting down-weights ubiquitous genres (Comedy, Drama) so a
    # multi-label film counts mostly toward its rarer, more informative tags.
    genre_idf = np.log(1.0 / np.clip(prevalence, 1e-9, None))
    Gw = G * genre_idf[None, :] if use_idf else G
    return G, Gw, prevalence


def clause_profiles(tm):
    """Return (L_pos, W_pos): positive-literal inclusion and positive output weights."""
    n_feat = tm.clause_bank.number_of_features
    n_items = tm.number_of_classes
    assert n_feat == n_items, (
        n_feat,
        n_items,
    )  # masked-item: features == items == classes

    states = decode_ta_states(host_clause_bank(tm))  # (n_clauses, n_literals) uint8
    incl = states >= 128  # include action = top state bit set
    L_pos = incl[:, :n_feat].astype(np.float32)  # positive-literal inclusion only

    W = np.stack([tm.weight_banks[i].get_weights() for i in range(n_items)]).astype(
        np.float32
    )
    W_pos = np.maximum(0.0, W)  # (n_items, n_clauses)
    return L_pos, W_pos


def cotaste_flow(L_pos, W_pos, Gw):
    """Genre x genre co-taste flow: for each trigger genre, the distribution over
    recommended genres, weighted by each clause's positive recommendation mass."""
    trig = L_pos @ Gw  # (n_clauses, n_genres) trigger genre mass
    rec = W_pos.T @ Gw  # (n_clauses, n_genres) weighted recommendation genre mass

    P = rownorm(trig)  # per-clause trigger genre distribution
    R = rownorm(rec)  # per-clause recommendation genre distribution
    w_c = W_pos.sum(axis=0)  # clause importance = total positive vote mass

    M = (P * w_c[:, None]).T @ R  # (n_genres, n_genres): trigger g -> rec h
    M = rownorm(M)
    return M, P, R, w_c


def lift_matrix(M, P, w_c):
    """Lift of each trigger genre's recommended-genre distribution over the global
    (unconditional) recommended-genre distribution -- a PMI-style over/under signal."""
    pi = (P * w_c[:, None]).sum(0)  # importance-weighted trigger mass per genre
    pi = pi / pi.sum()
    base = pi @ M  # unconditional recommended-genre distribution
    return M / np.where(base > 0, base, 1.0)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def heatmap(values, title, out_path, *, cmap, vmin, vmax, cbar_label=None):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(ALL_GENRES)))
    ax.set_xticklabels(ALL_GENRES, rotation=90)
    ax.set_yticks(range(len(ALL_GENRES)))
    ax.set_yticklabels(ALL_GENRES)
    ax.set_xlabel("Recommended genre")
    ax.set_ylabel("Trigger genre")
    ax.set_title(title)
    fig.colorbar(im, fraction=0.046, pad=0.04, label=cbar_label)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def self_lift_bar(iso, out_path):
    order = iso.index.tolist()
    colors = ["tomato" if g in FAMILY else "steelblue" for g in order]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(order)), iso["self_lift"].to_numpy(), color=colors)
    ax.axhline(1.0, color="gray", ls="--", lw=0.9)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=90)
    ax.set_ylabel("Self-lift  P(rec=g | trig=g) / P(rec=g)")
    ax.set_title("Genre self-recommendation lift (family-film block highlighted)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------


def report_isolation(M_lift, prevalence):
    """Rank genres by diagonal (self-recommendation) lift and summarise the block."""
    iso = pd.DataFrame(
        {"prevalence": prevalence, "self_lift": np.diag(M_lift)},
        index=ALL_GENRES,
    ).sort_values("self_lift", ascending=False)
    print("\nGenre self-recommendation lift (isolation ranking):")
    print(iso.round(3).to_string())

    fam_idx = [ALL_GENRES.index(g) for g in FAMILY]
    rest_idx = [j for j in range(len(ALL_GENRES)) if j not in fam_idx]
    within = M_lift[np.ix_(fam_idx, fam_idx)].mean()
    to_rest = M_lift[np.ix_(fam_idx, rest_idx)].mean()
    print(f"\nFamily block {FAMILY}:")
    print(f"  mean within-block lift : {within:.2f}")
    print(f"  mean block -> rest lift: {to_rest:.2f}")
    return iso


def report_children_clauses(tm, ds, P, R, w_c, n_top=15):
    """Sanity-check the global metric against readable rules: the clauses whose
    triggers are most dominated by Children's films, and what they recommend."""
    col_map = ds.col_to_movie_id()
    ch = ALL_GENRES.index("Children's")
    ch_score = P[:, ch] * w_c  # children's trigger share x importance
    print(f"\nTop {n_top} most Children's-driven clauses:")
    for c in np.argsort(-ch_score)[:n_top]:
        likes, _ = clause_pattern(tm, int(c), col_map, ds, max_per_polarity=10)
        rec_genre = ALL_GENRES[int(np.argmax(R[c]))] if R[c].sum() > 0 else "-"
        print(
            f"  clause {c:4d} | w_mass={w_c[c]:6.0f} | top-rec-genre={rec_genre:11s} | triggers: {likes}"
        )


def report_documentary_coverage(tm, ds, data, G, L_pos, W_pos):
    """Why documentaries never trigger: popularity, not incoherence.

    No clause triggers on a documentary, yet documentaries are still recommended.
    Clause literals come from a small popular core of items and no documentary is
    watched enough to enter it -- documentary *audiences* are in fact coherent.
    """
    col_map = ds.col_to_movie_id()
    n_items = tm.number_of_classes
    doc = ALL_GENRES.index("Documentary")
    doc_cols = np.where(G[:, doc] == 1)[0]
    pop = (
        data["x_train"].sum(axis=0).astype(np.int64)
    )  # int64: avoid uint-negation sort bug

    print("\n--- Documentary coverage ---")
    print(f"documentary movies in catalogue: {len(doc_cols)}")
    print(
        f"clauses with >=1 documentary literal: {int((L_pos[:, doc_cols].sum(1) > 0).sum())} / {L_pos.shape[0]}"
    )
    print(
        f"documentary items with >=1 positive weight: {int((W_pos[doc_cols].sum(1) > 0).sum())} / {len(doc_cols)}"
    )

    def genre_stats(name):
        cols = np.where(G[:, ALL_GENRES.index(name)] == 1)[0]
        return {
            "n_movies": len(cols),
            "mean_interactions": round(float(pop[cols].mean()), 1),
            "median_interactions": int(np.median(pop[cols])),
            "trigger_clauses": int((L_pos[:, cols].sum(1) > 0).sum()),
        }

    print(
        "\npopularity, not rarity -- rarer genres trigger because their films are more watched:"
    )
    print(
        pd.DataFrame(
            {
                g: genre_stats(g)
                for g in [
                    "Documentary",
                    "Film-Noir",
                    "Western",
                    "Animation",
                    "Children's",
                    "Musical",
                ]
            }
        ).T.to_string()
    )

    print("\nmost-rated documentaries -- popular yet never (or barely) a literal:")
    for c in doc_cols[np.argsort(pop[doc_cols])[::-1][:12]]:
        title = (ds.movie_id_to_title(col_map[c]) or f"col{c}")[:44]
        print(
            f"  {title:45s} pop={int(pop[c]):5d}  "
            f"as_literal={int(L_pos[:, c].sum())}  rec_mass={W_pos[c].sum():.0f}"
        )

    # Only a small, popular subset of items is ever used as a literal.
    incl_any = L_pos.sum(0) > 0
    is_doc = G[:, doc] == 1
    thr = np.median(pop[incl_any & ~is_doc])
    print(f"\nitems used as a literal in >=1 clause: {int(incl_any.sum())} / {n_items}")
    print(f"median interactions of included (non-doc) items: {thr:.0f}")
    print(
        f"documentaries above that threshold: {int(((pop >= thr) & is_doc).sum())}  "
        f"(most-watched documentary has {int(pop[is_doc].max())} interactions)"
    )

    # Audience coherence on the raw interactions, independent of the TM:
    # within-genre co-taste lift P(like j | like i) / P(like j) over same-genre pairs.
    X = data["x_train"].astype(np.float32)
    n_users = X.shape[0]
    C = X.T @ X
    diag = np.diag(C).copy()
    b = diag / n_users
    with np.errstate(divide="ignore", invalid="ignore"):
        lift_ij = (C / np.where(diag[:, None] > 0, diag[:, None], 1)) / np.where(
            b[None, :] > 0, b[None, :], 1
        )
    np.fill_diagonal(lift_ij, np.nan)

    def audience_lift(name):
        cols = np.where(G[:, ALL_GENRES.index(name)] == 1)[0]
        return (
            float(np.nanmean(lift_ij[np.ix_(cols, cols)]))
            if len(cols) > 1
            else float("nan")
        )

    rows = {
        g: {
            "audience_lift": round(audience_lift(g), 2),
            "mean_interactions": round(
                float(pop[np.where(G[:, ALL_GENRES.index(g)] == 1)[0]].mean()), 0
            ),
            "triggers": int(
                (L_pos[:, np.where(G[:, ALL_GENRES.index(g)] == 1)[0]].sum(1) > 0).sum()
            ),
        }
        for g in [
            "Documentary",
            "Film-Noir",
            "Western",
            "Animation",
            "Children's",
            "Musical",
            "Horror",
            "Mystery",
            "Drama",
            "Comedy",
        ]
    }
    print("\nwithin-genre audience co-taste lift (higher = more coherent audience):")
    print(
        pd.DataFrame(rows).T.sort_values("audience_lift", ascending=False).to_string()
    )
    print(
        "\nDocumentary audiences are coherent (high lift) but under-watched -> never a literal."
    )


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
        "--no_idf",
        action="store_true",
        help="use raw genre counts instead of IDF weighting",
    )
    p.add_argument(
        "--out_dir", type=Path, default=DEFAULT_OUT, help="directory for saved figures"
    )
    args = p.parse_args()

    assert args.run_dir.exists(), args.run_dir
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg, tm, ds, data = load_run(args.run_dir)
    G, Gw, prevalence = genre_matrix(tm, ds, use_idf=not args.no_idf)
    L_pos, W_pos = clause_profiles(tm)
    print(
        f"mean positive literals / clause: {L_pos.sum(1).mean():.1f}  "
        f"| clauses with >=1 positive literal: {(L_pos.sum(1) > 0).sum()} / {L_pos.shape[0]}"
    )

    print("\nGenre prevalence (fraction of catalogue):")
    print(
        pd.Series(prevalence, index=ALL_GENRES)
        .sort_values(ascending=False)
        .round(3)
        .to_string()
    )

    M, P, R, w_c = cotaste_flow(L_pos, W_pos, Gw)
    M_lift = lift_matrix(M, P, w_c)

    # Figures: raw flow is swamped by Drama/Comedy prevalence; the lift matrix
    # divides out the base rate and exposes the block structure.
    print("\nSaving figures:")
    heatmap(
        M,
        "Clause co-taste flow (row-normalised): trigger -> recommendation",
        args.out_dir / "genre_flow_raw.png",
        cmap="coolwarm",
        vmin=0,
        vmax=float(M.max()),
    )
    L2 = np.log2(np.where(M_lift > 0, M_lift, np.nan))
    vmax = float(np.nanmax(np.abs(L2)))
    heatmap(
        L2,
        "Co-taste lift (log2): over- / under-recommendation vs base rate",
        args.out_dir / "genre_flow_lift.png",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        cbar_label="log2 lift",
    )

    iso = report_isolation(M_lift, prevalence)
    self_lift_bar(iso, args.out_dir / "genre_self_lift.png")

    report_children_clauses(tm, ds, P, R, w_c)
    report_documentary_coverage(tm, ds, data, G, L_pos, W_pos)


if __name__ == "__main__":
    main()
