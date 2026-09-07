"""Clause inspection / reporting helpers for the coalesced TM rec-sys model.

These functions read a trained ``TMCoalescedClassifier``'s clause bank and log
human-readable summaries (per-movie top clauses, global literal frequencies).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from datasets import MovieLens1M, VibrentRental

    # Any dataset exposing the col->item-id map and item-id->label lookup.
    Dataset = MovieLens1M | VibrentRental

log = logging.getLogger(__name__)


def clause_pattern(
    tm,
    clause_idx: int,
    col_map: dict[int, int],
    ds: "Dataset",
    max_per_polarity: int = 8,
) -> tuple[list[str], list[str]]:
    """Return (likes_titles, not_likes_titles) for one clause."""
    n_feat = tm.clause_bank.number_of_features
    likes, not_likes = [], []
    for ta in range(tm.clause_bank.number_of_literals):
        if tm.get_ta_action(clause_idx, ta) != 1:
            continue
        col = ta if ta < n_feat else ta - n_feat
        title = ds.item_id_to_name(col_map[col]) or f"col{col}"
        if ta < n_feat:
            likes.append(title)
        else:
            not_likes.append(title)
    return likes[:max_per_polarity], not_likes[:max_per_polarity]


def analyze_clauses_for_movie(
    tm,
    movie_col: int,
    col_map: dict[int, int],
    ds: "Dataset",
    n_top: int = 5,
    ignore_self_inclusion: bool = False,
) -> None:
    """Log the top positive and negative clauses for one target movie."""
    weights = tm.get_weights(movie_col)
    title = ds.item_id_to_name(col_map[movie_col]) or f"col{movie_col}"

    log.info("")
    log.info("=" * 70)
    log.info(f"Item: {title}  (col={movie_col})")
    log.info(
        f"  w>0: {(weights > 0).sum()}  w<0: {(weights < 0).sum()}  "
        f"range [{weights.min()}, {weights.max()}]"
    )

    n_clauses = tm.clause_bank.number_of_clauses
    candidates = np.arange(n_clauses)
    if ignore_self_inclusion:
        self_included = np.array(
            [bool(tm.get_ta_action(c, movie_col)) for c in range(n_clauses)]
        )
        candidates = candidates[~self_included]
        if self_included.any():
            log.info(
                f"  (skipping {self_included.sum()} clauses that include self-literal {movie_col})"
            )

    for label, order in [
        ("FOR  ", candidates[np.argsort(-weights[candidates])[:n_top]]),
        ("AGAINST", candidates[np.argsort(weights[candidates])[:n_top]]),
    ]:
        log.info(f"  Top {label} clauses:")
        for c in order:
            likes, not_likes = clause_pattern(tm, int(c), col_map, ds)
            log.info(f"    [{label.strip()} w={weights[c]:+4d}] clause {c}")
            if likes:
                log.info(f"      likes     : {likes}")
            if not_likes:
                log.info(f"      not_likes : {not_likes}")


def literal_inclusion_frequency(tm) -> np.ndarray:
    """Count how many clauses include each literal (by scanning TA actions).

    Returns an array of shape (number_of_literals,).
    Note: literal_clause_frequency() in TMU counts activations on the last
    seen batch, not structural inclusion — this function counts inclusion.
    """
    n_clauses = tm.clause_bank.number_of_clauses
    n_lit = tm.clause_bank.number_of_literals
    freq = np.zeros(n_lit, dtype=np.int32)
    for c in range(n_clauses):
        for ta in range(n_lit):
            if tm.get_ta_action(c, ta):
                freq[ta] += 1
    return freq


def global_literal_report(
    tm,
    col_map: dict[int, int],
    ds: "Dataset",
    n_top: int = 20,
) -> None:
    """Log the most frequently *included* literals across all clauses."""
    log.info("Computing literal inclusion frequencies...")
    freq = literal_inclusion_frequency(tm)
    n_feat = tm.clause_bank.number_of_features
    top = np.argsort(-freq)[:n_top]

    log.info("")
    log.info("=" * 70)
    log.info("Most frequently included literals across all clauses:")
    log.info(f"  {'polarity':<10} {'title':<50} clauses")
    log.info(f"  {'-' * 10} {'-' * 50} -------")
    for ta in top:
        col = ta if ta < n_feat else ta - n_feat
        polarity = "likes" if ta < n_feat else "not_likes"
        title = ds.item_id_to_name(col_map[col]) or f"col{col}"
        log.info(f"  {polarity:<10} {title:<50} {freq[ta]}")


# ---------------------------------------------------------------------------
# TA-state decoding + structural stats (shared by training logging + analysis)
# ---------------------------------------------------------------------------

# An 8-bit TA counter includes its literal when state >= 128 (top bit set).
INCLUDE_THRESHOLD = 128


def host_clause_bank(tm):
    """Return a host-side clause bank whose ``.clause_bank`` ta_state is current.

    For a CUDA model this syncs device->host first (cheap memcpy) and returns
    the host bank; for a CPU model it returns the clause bank as-is.
    """
    cb = tm.clause_bank
    if hasattr(cb, "host"):  # ClauseBankCuda
        cb.synchronize_clause_bank()
        return cb.host
    return cb


def decode_ta_states(hb) -> np.ndarray:
    """Decode full 8-bit TA states: (n_clauses, n_literals) uint8.

    The counter for literal k of clause c is bit-sliced across the
    ``number_of_state_bits_ta`` words of chunk ``k // 32``; bit ``k % 32`` of
    slice b is bit b of the counter.

    Each bit-plane is expanded with ``np.unpackbits`` (a contiguous, C-level
    bit expansion) rather than fancy-indexing a length-n_literals column array,
    which is ~8x faster for the per-epoch logging path.
    """
    nC = hb.number_of_clauses
    nChunks = hb.number_of_ta_chunks
    nBits = hb.number_of_state_bits_ta
    nLit = hb.number_of_literals
    ta = np.asarray(hb.clause_bank).reshape(nC, nChunks, nBits)
    states = np.zeros((nC, nLit), dtype=np.uint8)
    for b in range(nBits):
        # View plane b as little-endian bytes, then unpack: column (chunk*32 + p)
        # becomes bit p of word `chunk`, i.e. literal (chunk*32 + p).
        plane = np.ascontiguousarray(ta[:, :, b]).view(np.uint8)
        bits = np.unpackbits(plane, axis=1, bitorder="little")
        states |= bits[:, :nLit].astype(np.uint8) << b
    return states


def topn_counts(score: np.ndarray, n: int) -> np.ndarray:
    """For score (n_classes, n_clauses), count per clause how many classes rank
    it in their top-n (largest score). Returns (n_clauses,) int."""
    K = score.shape[1]
    n = min(n, K - 1)
    idx = np.argpartition(-score, n, axis=1)[:, :n]  # (n_classes, n)
    return np.bincount(idx.ravel(), minlength=K)


def literal_core_stats(pos_include: np.ndarray, profiles: np.ndarray) -> dict:
    """How much of the users' interaction signal the clause antecedents can read.

    The "literal core" is the set of items included as a positive literal in at
    least one clause; any interaction outside it is invisible to every clause
    antecedent (though the item can still be recommended via its weights).

    Args:
        pos_include: (n_clauses, n_features) bool — positive-literal inclusion.
        profiles: (n_users, n_features) binary user-item interaction matrix.

    Returns:
        literal_core_size        : number of items appearing as a positive literal
        literal_core_frac        : fraction of the catalogue in the core
        profile_visibility_mean  : mean over users of the fraction of their
                                   interactions that fall inside the core
        blind_user_frac          : fraction of users with *no* interaction in the
                                   core — the clauses carry zero antecedent
                                   signal for them
    """
    core = pos_include.any(axis=0)
    prof = profiles.astype(bool)
    sizes = prof.sum(axis=1)
    visible = (prof & core).sum(axis=1)
    nonempty = sizes > 0
    frac = visible[nonempty] / sizes[nonempty]
    return {
        "literal_core_size": int(core.sum()),
        "literal_core_frac": float(core.mean()),
        "profile_visibility_mean": float(frac.mean()) if nonempty.any() else 0.0,
        "blind_user_frac": float((visible[nonempty] == 0).mean())
        if nonempty.any()
        else 1.0,
    }


def clause_structure_stats(
    tm, deep_margin: int = 32, top_n: int = 10, profiles: np.ndarray | None = None
) -> dict:
    """Per-epoch clause structure + importance summary for training logs.

    Literal structure:
      clause_size_mean : mean number of included literals per clause
      clause_size_std  : std of included literals per clause
      deep_commit_frac : fraction of *included* literals committed beyond the
                         include threshold by >= deep_margin (state >= 160),
                         i.e. decisively included rather than hovering.

    Weight-based importance (W = n_classes x n_clauses vote weights):
      eff_clauses_per_class : mean participation ratio (Sum|w|)^2 / Sum(w^2) per
                              class — the effective number of clauses carrying a
                              class's predictive weight (low = sparse/readable).
      rank_dead_frac        : fraction of clauses never in any class's top-`top_n`
                              by |weight| — clauses that are never decisive.

    When `profiles` (a binary user-item matrix) is given, the literal-core /
    profile-visibility stats of literal_core_stats() are included as well.
    """
    states = decode_ta_states(host_clause_bank(tm))
    incl = states >= INCLUDE_THRESHOLD
    n_incl = int(incl.sum())
    sizes = incl.sum(1)
    deep = int((states >= INCLUDE_THRESHOLD + deep_margin).sum())

    # Negated-literal accounting. Literals [0:n_feat] are the positive (item
    # present) literals; [n_feat:2*n_feat] are their negations (item absent).
    # feature_negation=False keeps the negated half permanently excluded, so
    # these metrics are identically 0 for the no-negation arm — a cheap per-epoch
    # sanity check as well as the negated-literal signal for the ablation. Logged
    # every epoch so the negation share can be read straight from wandb /
    # metadata.json without re-decoding checkpoints.
    n_feat = tm.clause_bank.number_of_features
    neg_per_clause = incl[:, n_feat:].sum(1)
    n_neg = int(neg_per_clause.sum())
    nonempty = sizes > 0

    # Weight-based importance structure. Weights live host-side, so no sync.
    absW = np.abs(
        np.array(
            [tm.weight_banks[i].get_weights() for i in range(tm.number_of_classes)],
            dtype=np.float64,
        )
    )
    s1, s2 = absW.sum(1), (absW**2).sum(1)
    eff = np.where(s1 > 0, s1**2 / np.where(s2 > 0, s2, 1), 0.0)
    rank_dead_frac = float((topn_counts(absW, top_n) == 0).mean())

    out = {
        "clause_size_mean": float(sizes.mean()),
        "clause_size_std": float(sizes.std()),
        "deep_commit_frac": float(deep / n_incl) if n_incl else 0.0,
        "eff_clauses_per_class": float(eff.mean()),
        "rank_dead_frac": rank_dead_frac,
        # Share of all included-literal slots that are negations (item-absent):
        # the headline negated-literal metric for the ablation.
        "neg_literal_frac": float(n_neg / n_incl) if n_incl else 0.0,
        # Mean negated literals per clause, and the fraction of clauses that use
        # any negation at all.
        "clause_neg_size_mean": float(neg_per_clause.mean()),
        "clause_with_neg_frac": float((neg_per_clause > 0).mean()),
        # Mean over non-empty clauses of the within-clause negation proportion.
        "neg_frac_per_clause_mean": float(
            (neg_per_clause[nonempty] / sizes[nonempty]).mean()
        )
        if nonempty.any()
        else 0.0,
    }
    if profiles is not None:
        out.update(literal_core_stats(incl[:, :n_feat], profiles))
    return out
