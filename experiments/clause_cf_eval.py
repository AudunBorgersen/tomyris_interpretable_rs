"""Evaluate the TM and its clause-based CF blends on the unified ranking protocol.

Loads a trained coalesced-TM run, derives clause-activation embeddings via
``tm.transform``, builds the user-user / item-item collaborative filters of
the paper, Section IV-C, and scores every variant through
``metrics.evaluate`` (``exclude_seen=True``) — i.e. the exact protocol the
baseline table uses. For each CF approach the blend weight ``alpha`` is selected
on the validation split (by ``--select_metric``) and the chosen operating point
is reported on the test split, so the emitted rows are directly comparable to
the baselines.

Outputs:
  * a console table,
  * spliceable LaTeX rows for the results table (``--latex_out``),
  * a per-alpha test sweep CSV consumed by figures/cf_alpha_sweep.py
    (``--sweep_out``).

Example:
    python experiments/clause_cf_eval.py \
        --run_dir runs/coalesced_baseline/20260624T040526_high_val_hpo
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

_PROJ_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJ_ROOT))

from datasets import get_dataset
from experiments.coalesced_baseline import build_samples
from utils.metrics import evaluate
from utils.checkpoint import load_tm_cpu
from utils.latex_table import latex_rows

_FAMILIES = ("hit", "ndcg", "novelty", "coverage")


def minmax_norm(scores: np.ndarray) -> np.ndarray:
    """Per-row min-max normalisation to [0, 1]."""
    lo = scores.min(axis=1, keepdims=True)
    hi = scores.max(axis=1, keepdims=True)
    rng = np.where(hi - lo == 0, 1.0, hi - lo)
    return (scores - lo) / rng


def user_user_knn(
    U_eval: np.ndarray,
    Utr_n: np.ndarray,
    x_train_f: np.ndarray,
    idf: np.ndarray,
    k: int,
    batch: int = 8192,
) -> np.ndarray:
    """Top-k neighbour user-user CF in IDF-weighted cosine space (batched).

    Keeps only each evaluation user's ``k`` most similar training users before the
    weighted rating sum, rather than averaging over the whole population. Batched
    so the (n_eval, n_train) similarity block is never fully materialised.
    """
    Ue_n = normalize(U_eval * idf)
    out = np.zeros((Ue_n.shape[0], x_train_f.shape[1]), dtype=np.float32)
    for s in range(0, Ue_n.shape[0], batch):
        e = min(s + batch, Ue_n.shape[0])
        sims = Ue_n[s:e] @ Utr_n.T
        if k < sims.shape[1]:
            idx = np.argpartition(sims, -k, axis=1)[:, -k:]
            rows = np.arange(e - s)[:, None]
            masked = np.zeros_like(sims)
            masked[rows, idx] = sims[rows, idx]
            sims = masked
        out[s:e] = sims @ x_train_f
    return out


def build_cf_scorers(tm, x_train: np.ndarray, uu_k: int = 50):
    """Precompute the train-derived structures and return per-approach scorers.

    Each scorer maps an evaluation embedding matrix ``U_eval`` (n_eval, n_clauses)
    to a raw (n_eval, n_items) CF score matrix.
    """
    n_items = tm.number_of_classes
    n_clauses = tm.clause_bank.number_of_clauses

    U_train = tm.transform(x_train).astype(np.float32)
    x_train_f = x_train.astype(np.float32)

    # Clause IDF: down-weight clauses that fire for nearly every user (no
    # discriminative power). Used by the top-k user-user neighbourhood.
    act_rate = U_train.mean(axis=0)
    idf = np.zeros_like(act_rate)
    nz = act_rate > 0
    idf[nz] = np.log(1.0 / act_rate[nz])
    Utr_idf_n = normalize(U_train * idf)

    # Mean clause profile of each item's fans (training users who interacted).
    item_profiles = np.zeros((n_items, n_clauses), dtype=np.float32)
    for j in range(n_items):
        fans = x_train[:, j] == 1
        if fans.any():
            item_profiles[j] = U_train[fans].mean(axis=0)

    # Coalesced weight matrix: (n_items, n_clauses), positive = pro-clause.
    W = np.stack([tm.weight_banks[i].get_weights() for i in range(n_items)]).astype(
        np.float32
    )
    item_profiles_wmod = item_profiles * np.abs(W)
    W_pos = np.maximum(0.0, W)

    return {
        "user-user": lambda U: cosine_similarity(U, U_train) @ x_train_f,
        "uu-knn": lambda U: user_user_knn(U, Utr_idf_n, x_train_f, idf, uu_k),
        "item-item": lambda U: U @ item_profiles.T,
        "ii-wmod": lambda U: U @ item_profiles_wmod.T,
        "ii-pos-weight": lambda U: U @ W_pos.T,
    }


def sweep(tm_norm, cf_norm, X, Y, alphas, item_popularity, n_items, ks):
    """Blend normalised TM and CF scores across alphas; return {alpha: metrics}."""
    out = {}
    for a in alphas:
        blended = a * tm_norm + (1 - a) * cf_norm
        out[a] = evaluate(
            None,
            X,
            Y,
            scores=blended,
            ks=ks,
            item_popularity=item_popularity,
            n_items=n_items,
        )
    return out


# Row labels for the CF blend variants, shared by the CLI table and any caller
# that aggregates across runs (kept module-level so both agree on the names).
_CF_PRETTY = {
    "user-user": "\\quad + User--User CF",
    "uu-knn": "\\quad + User--User CF (top-$k$)",
    "item-item": "\\quad + Item--Item CF",
    "ii-wmod": "\\quad + Item--Item CF (weight-mod.)",
    "ii-pos-weight": "\\quad + Item--Item CF (pos-weight)",
}
_TM_ROW = "ToMyRiS (\\ac{TM})"


def evaluate_run(
    run_dir,
    ks=(1, 10, 50),
    alphas=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    select_metric="ndcg@10",
    uu_k=50,
    verbose=True,
):
    """Evaluate one coalesced-TM run: pure TM plus each CF blend, with alpha
    tuned on validation and reported on test.

    This is the reusable core of the CLI ``main()`` — it performs no file writes
    and (beyond optional progress prints) no console output, so both the CLI and
    the cross-run variance aggregator can format the numbers as they need.

    Returns:
        results        : ``{row_label: {metric_key: value}}`` — pure-TM row plus
                         one row per CF approach, all on the test split (alpha
                         selected per this run).
        alpha_by_name  : selected blend weight per CF row label.
        sweep_records  : ``[(approach_key, alpha, metrics), ...]`` full test sweep.
        val_records    : ``[(approach_key, alpha, metrics), ...]`` full validation
                         sweep — lets a caller pool the validation curve across
                         runs and fix one shared alpha per approach.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_absolute():
        run_dir = _PROJ_ROOT / run_dir
    assert run_dir.exists(), f"run_dir not found: {run_dir}"

    with open(run_dir / "metadata.json") as f:
        cfg = SimpleNamespace(**json.load(f)["config"])

    ks = tuple(ks)
    alphas = list(alphas)
    checkpoints = sorted(run_dir.glob("epoch_*.pkl"))
    tm = load_tm_cpu(checkpoints[-1])
    if verbose:
        print(
            f"[{run_dir.name}] loaded {checkpoints[-1].name} | "
            f"clauses: {tm.clause_bank.number_of_clauses} | items: {tm.number_of_classes}"
        )

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
    if len(data["x_val"]) == 0:
        raise SystemExit(
            "No validation split available (val_ratio=0); alpha cannot be tuned. "
            "Re-run the dataset with val_ratio > 0."
        )

    # Hold out every positive once (max_per_user=None): deterministic, no RNG
    # dependence, averaged over all (user, item) pairs to remove single-target
    # variance. Identical protocol to grid_search.py / run_all.py, so the TM/CF
    # rows sit on the same evaluation set as the baselines. The split itself is
    # seeded (random_state=42) independently of the run's training seed, so all
    # runs are scored on one fixed val/test set — the only thing that varies is
    # the trained model.
    X_test, Y_test = build_samples(data["x_test"], max_per_user=None)
    X_val, Y_val = build_samples(data["x_val"], max_per_user=None)

    n_items = data["x_train"].shape[1]
    item_popularity = data["x_train"].sum(axis=0) / len(data["y_train"])
    if verbose:
        print(
            f"  val samples: {len(X_val)} | test samples: {len(X_test)} | items: {n_items}"
        )

    # Embeddings + raw TM scores for both splits.
    U_val = tm.transform(X_val).astype(np.float32)
    U_test = tm.transform(X_test).astype(np.float32)
    _, tm_val = tm.predict(X_val, return_class_sums=True)
    _, tm_test = tm.predict(X_test, return_class_sums=True)
    tm_val_norm = minmax_norm(tm_val.astype(np.float32))
    tm_test_norm = minmax_norm(tm_test.astype(np.float32))

    scorers = build_cf_scorers(tm, data["x_train"], uu_k=uu_k)

    # Pure TM baseline (alpha = 1).
    results: dict[str, dict] = {
        _TM_ROW: evaluate(
            None,
            X_test,
            Y_test,
            scores=tm_test,
            ks=ks,
            item_popularity=item_popularity,
            n_items=n_items,
        )
    }

    sweep_records = []  # for the figure CSV (test split)
    val_records = []  # full validation sweep — for cross-run pooled alpha selection
    alpha_by_name: dict[str, float] = {}  # selected blend weight per CF row
    for key, scorer in scorers.items():
        cf_val_norm = minmax_norm(scorer(U_val))
        cf_test_norm = minmax_norm(scorer(U_test))

        # Select alpha on validation by the chosen metric, excluding alpha = 1.0:
        # that operating point collapses the blend onto the pure-TM baseline (an
        # identical, redundant row), so every CF row reports its best *genuine*
        # blend (alpha < 1.0). Fall back to the full grid only if it has no such
        # value.
        val_sweep = sweep(
            tm_val_norm, cf_val_norm, X_val, Y_val, alphas, item_popularity, n_items, ks
        )
        blend_alphas = [a for a in alphas if a < 1.0] or alphas
        best_alpha = max(blend_alphas, key=lambda a: val_sweep[a][select_metric])

        # Report the chosen operating point on test; record the full test sweep.
        test_sweep = sweep(
            tm_test_norm,
            cf_test_norm,
            X_test,
            Y_test,
            alphas,
            item_popularity,
            n_items,
            ks,
        )
        results[_CF_PRETTY[key]] = test_sweep[best_alpha]
        alpha_by_name[_CF_PRETTY[key]] = best_alpha
        if verbose:
            print(f"  {key:<14} best alpha (val {select_metric}) = {best_alpha}")
        for a in alphas:
            sweep_records.append((key, a, test_sweep[a]))
            val_records.append((key, a, val_sweep[a]))

    return results, alpha_by_name, sweep_records, val_records


def main(args: argparse.Namespace) -> None:
    ks = list(args.ks)
    results, alpha_by_name, sweep_records, _val_records = evaluate_run(
        args.run_dir,
        ks=ks,
        alphas=list(args.alphas),
        select_metric=args.select_metric,
        uu_k=args.uu_k,
    )

    # ---- console table ----
    cols = [f"{fam}@{k}" for fam in _FAMILIES for k in ks]
    name_w = max(len(n) for n in results) + 2
    print("\n" + "=" * 80)
    header = f"{'Model':<{name_w}}" + "".join(f"{c:>11}" for c in cols)
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(f"{name:<{name_w}}" + "".join(f"{m[c]:>11.4f}" for c in cols))
    print("=" * 80)

    # ---- LaTeX rows ----
    rows = latex_rows(results, list(ks), alpha_by_name)
    print("\n% --- splice into tab:baselines_tuned (before \\bottomrule) ---")
    print(rows)
    if args.latex_out:
        out = Path(args.latex_out)
        out = out if out.is_absolute() else _PROJ_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(rows + "\n", encoding="utf-8")
        print(f"\nWrote LaTeX rows to {out}")

    # ---- figure sweep CSV ----
    sweep_out = Path(args.sweep_out)
    sweep_out = sweep_out if sweep_out.is_absolute() else _PROJ_ROOT / sweep_out
    sweep_out.parent.mkdir(parents=True, exist_ok=True)
    with open(sweep_out, "w", encoding="utf-8") as f:
        f.write("approach,alpha,ndcg@10,novelty@10,coverage@10\n")
        for key, a, m in sweep_records:
            f.write(f"{key},{a},{m['ndcg@10']},{m['novelty@10']},{m['coverage@10']}\n")
    print(f"Wrote sweep CSV to {sweep_out}")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run_dir", required=True, help="Path to a coalesced_baseline run dir."
    )
    p.add_argument("--ks", type=int, nargs="+", default=[1, 10, 50])
    p.add_argument(
        "--alphas", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    )
    p.add_argument(
        "--select_metric",
        default="ndcg@10",
        help="Validation metric used to pick alpha per CF approach.",
    )
    p.add_argument(
        "--uu_k",
        type=int,
        default=50,
        help="Neighbourhood size for the top-k user-user CF approach.",
    )
    p.add_argument(
        "--latex_out",
        default="results/tex/latex_rows_cf_eval.tex",
        help="Optional path to write the spliceable LaTeX rows.",
    )
    p.add_argument(
        "--sweep_out",
        default="results/cf_alpha_sweep.csv",
        help="Path for the per-alpha test sweep CSV (figure input).",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
