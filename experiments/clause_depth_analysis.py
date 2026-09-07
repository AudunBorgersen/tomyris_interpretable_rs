"""Clause depth vs. contribution: are short clauses weaker, and were long ones redundant?

Motivated by the popularity-subsampling runs: feedback-masked training roughly
halves mean clause size relative to the unsubsampled baseline while improving
ranking accuracy. This script quantifies, per checkpoint, (a) what the clause
inventory looks like structurally, (b) how redundant the multi-literal clauses
are (were the baseline's long clauses "the same popular movies reshuffled"?),
and (c) how much of the actual recommendation vote is cast by clauses of each
size — so "mean clause length" can be judged against what the clauses do.

Vote decomposition uses prediction semantics (empty clauses output 0, so they
cast no votes) on the top-K recommended items of held-out test users.

Example:
    python experiments/clause_depth_analysis.py \
        --run_dirs runs/coalesced_baseline/20260626T012048_solid_hpo_result \
                   runs/coalesced_baseline/20260702T133243_feedback_masking
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_PROJ_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJ_ROOT))

from datasets import get_dataset
from experiments.coalesced_baseline import build_samples
from utils.checkpoint import load_tm_cpu
from utils.clause_analysis import INCLUDE_THRESHOLD, decode_ta_states, host_clause_bank


# ---------------------------------------------------------------------------
# Structure / redundancy helpers (pure, tested in tests/test_clause_depth.py)
# ---------------------------------------------------------------------------


def size_histogram(incl: np.ndarray) -> dict:
    """Counts of clauses with 0, 1, 2 and >=3 included positive literals."""
    sizes = incl.sum(axis=1)
    return {
        "n_clauses": int(len(sizes)),
        "size0": int((sizes == 0).sum()),
        "size1": int((sizes == 1).sum()),
        "size2": int((sizes == 2).sum()),
        "size3plus": int((sizes >= 3).sum()),
        "size_mean": float(sizes.mean()),
        "size_mean_nonempty": float(sizes[sizes > 0].mean())
        if (sizes > 0).any()
        else 0.0,
    }


def duplicate_fraction(incl: np.ndarray) -> float:
    """Fraction of non-empty clauses whose literal set equals another clause's."""
    nonempty = incl[incl.sum(axis=1) > 0]
    if len(nonempty) == 0:
        return 0.0
    packed = np.packbits(nonempty, axis=1)
    _, inverse, counts = np.unique(
        packed, axis=0, return_inverse=True, return_counts=True
    )
    return float((counts[inverse] > 1).mean())


def max_jaccard(incl: np.ndarray, min_size: int = 2) -> np.ndarray:
    """For each clause with >= min_size literals, its max Jaccard similarity to
    any other non-empty clause. 1.0 = an exact duplicate exists."""
    sizes = incl.sum(axis=1)
    ref = incl[sizes > 0].astype(np.float32)
    query_idx = np.flatnonzero(sizes >= min_size)
    if len(query_idx) == 0 or len(ref) < 2:
        return np.zeros(0)
    q = incl[query_idx].astype(np.float32)
    inter = q @ ref.T
    union = q.sum(1, keepdims=True) + ref.sum(1) - inter
    jac = inter / np.maximum(union, 1)
    # A clause is identical to itself in ref; null that out by dropping the
    # top match when it is the self-match (jaccard exactly 1 with same size &
    # same row). Simplest robust approach: for each query, zero its self row.
    ref_positions = np.flatnonzero(sizes > 0)
    self_col = {pos: c for c, pos in enumerate(ref_positions)}
    for r, qi in enumerate(query_idx):
        jac[r, self_col[qi]] = 0.0
    return jac.max(axis=1)


def literal_concentration(incl: np.ndarray, top_ns=(10, 25)) -> dict:
    """Share of all included-literal slots taken by the most-used items."""
    slot_counts = incl.sum(axis=0)
    total = slot_counts.sum()
    out = {"distinct_literals": int((slot_counts > 0).sum())}
    if total == 0:
        out.update({f"top{n}_slot_share": 0.0 for n in top_ns})
        return out
    ranked = np.sort(slot_counts)[::-1]
    for n in top_ns:
        out[f"top{n}_slot_share"] = float(ranked[:n].sum() / total)
    return out


def vote_share_by_size(
    clause_outputs: np.ndarray,
    W: np.ndarray,
    top_items: np.ndarray,
    sizes: np.ndarray,
) -> dict:
    """Decompose the positive vote mass behind each recommendation by the
    voting clause's literal count.

    Args:
        clause_outputs: (n_samples, n_clauses) 0/1 prediction-semantics outputs.
        W: (n_items, n_clauses) coalesced weights.
        top_items: (n_samples, K) recommended item columns per sample.
        sizes: (n_clauses,) included-literal counts.

    Returns dict with the share of summed positive vote mass cast by clauses
    of size 1, 2 and >=3 (empty clauses never fire at prediction), plus the
    fraction of recommendations whose single largest positive vote comes from
    a multi-literal (>=2) clause.
    """
    masks = {
        "share_size1": sizes == 1,
        "share_size2": sizes == 2,
        "share_size3plus": sizes >= 3,
    }
    totals = dict.fromkeys(masks, 0.0)
    grand_total = 0.0
    top_vote_multi = 0
    n_recs = 0
    for s in range(clause_outputs.shape[0]):
        active = clause_outputs[s].astype(np.float32)
        w_top = W[top_items[s]] * active  # (K, n_clauses) votes actually cast
        pos = np.maximum(w_top, 0.0)
        for key, m in masks.items():
            totals[key] += float(pos[:, m].sum())
        grand_total += float(pos.sum())
        top_clause = pos.argmax(axis=1)  # strongest positive vote per rec
        top_vote_multi += int((sizes[top_clause] >= 2).sum())
        n_recs += pos.shape[0]
    if grand_total == 0:
        return {k: 0.0 for k in masks} | {"top_vote_multi_frac": 0.0}
    out = {k: v / grand_total for k, v in totals.items()}
    out["top_vote_multi_frac"] = top_vote_multi / n_recs
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def analyze_run(run_dir: Path, top_k: int, max_users: int, rng: np.random.Generator):
    with open(run_dir / "metadata.json") as f:
        cfg = SimpleNamespace(**json.load(f)["config"])

    # Last epoch, matching experiments/clause_cf_eval.py, so the depth figures
    # describe the same model state as the reported results tables.
    ckpt = sorted(run_dir.glob("epoch_*.pkl"))[-1]
    tm = load_tm_cpu(ckpt)
    n_feat = tm.clause_bank.number_of_features
    incl = (decode_ta_states(host_clause_bank(tm)) >= INCLUDE_THRESHOLD)[:, :n_feat]
    sizes = incl.sum(axis=1)

    stats = {"run": run_dir.name, "checkpoint": ckpt.name}
    stats.update(size_histogram(incl))
    stats["duplicate_frac"] = duplicate_fraction(incl)
    mj = max_jaccard(incl, min_size=2)
    stats["maxjac_mean"] = float(mj.mean()) if mj.size else float("nan")
    stats["maxjac_ge05_frac"] = float((mj >= 0.5).mean()) if mj.size else float("nan")
    stats.update(literal_concentration(incl))

    ds = get_dataset(
        getattr(cfg, "dataset", "movielens1m"),
        rating_threshold=getattr(cfg, "rating_threshold", 4),
        k_core=getattr(cfg, "k_core", 5),
        unit=getattr(cfg, "unit", "group"),
        test_ratio=0.2,
        val_ratio=getattr(cfg, "val_ratio", 0.1),
        min_ratings=20,
        random_state=42,
        max_items=cfg.max_classes,
    )
    data = ds.get()
    x_test = data["x_test"]
    if max_users and max_users > 0 and len(x_test) > max_users:
        x_test = x_test[rng.choice(len(x_test), max_users, replace=False)]
    X_test, _ = build_samples(x_test, max_per_user=None)

    clause_outputs = tm.transform(X_test).astype(np.uint8)
    W = np.stack(
        [tm.weight_banks[i].get_weights() for i in range(tm.number_of_classes)]
    ).astype(np.float32)
    scores = clause_outputs.astype(np.float32) @ W.T
    scores[X_test.astype(bool)] = -np.inf  # standard seen-item exclusion
    top_items = np.argpartition(-scores, top_k, axis=1)[:, :top_k]

    stats.update(vote_share_by_size(clause_outputs, W, top_items, sizes))
    stats["n_eval_samples"] = int(len(X_test))
    return stats


def main(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    results = []
    for rd in args.run_dirs:
        run_dir = Path(rd)
        if not run_dir.is_absolute():
            run_dir = _PROJ_ROOT / run_dir
        assert run_dir.exists(), f"run_dir not found: {run_dir}"
        print(f"Analyzing {run_dir.name}...")
        results.append(analyze_run(run_dir, args.top_k, args.max_users, rng))

    cols = [
        ("size_mean", "{:.2f}"),
        ("size0", "{:d}"),
        ("size1", "{:d}"),
        ("size2", "{:d}"),
        ("size3plus", "{:d}"),
        ("duplicate_frac", "{:.3f}"),
        ("maxjac_mean", "{:.3f}"),
        ("maxjac_ge05_frac", "{:.3f}"),
        ("distinct_literals", "{:d}"),
        ("top10_slot_share", "{:.3f}"),
        ("top25_slot_share", "{:.3f}"),
        ("share_size1", "{:.3f}"),
        ("share_size2", "{:.3f}"),
        ("share_size3plus", "{:.3f}"),
        ("top_vote_multi_frac", "{:.3f}"),
    ]
    name_w = max(len(r["run"]) for r in results) + 2
    print("\n" + "=" * 100)
    print(f"{'run':<{name_w}}" + "".join(f"{c:>18}" for c, _ in cols))
    for r in results:
        print(
            f"{r['run']:<{name_w}}"
            + "".join(fmt.format(r[c]).rjust(18) for c, fmt in cols)
        )
    print("=" * 100)

    out_csv = Path(args.out_csv)
    out_csv = out_csv if out_csv.is_absolute() else _PROJ_ROOT / out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for r in results for k in r})
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in results:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
    print(f"Wrote {out_csv}")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dirs", nargs="+", required=True)
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument(
        "--max_users",
        type=int,
        default=1000,
        help="Subsample this many test users (one held-out target each). Use -1 to keep all.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_csv", default="results/clause_depth/clause_depth.csv")
    return p.parse_args(argv)


if __name__ == "__main__":
    main(parse_args())
