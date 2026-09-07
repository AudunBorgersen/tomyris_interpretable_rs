"""Per-item score calibration for popularity-bias correction at inference time.

The TM's class sums run systematically hotter for popular items: their weight
vectors receive more Type Ia reinforcement, so a high raw score for a head item
is unremarkable while the same score for a tail item is highly informative.
These helpers re-express a user's score for item i relative to how item i
scores across a calibration population (training users), so ranking rewards
*anomalously* high scores instead of absolutely high ones.

All functions operate on plain (n_users, n_items) score matrices and are
model-agnostic; `experiments/score_calibration_eval.py` drives them.
"""

import numpy as np


def item_score_stats(
    calib_scores: np.ndarray,
    seen: np.ndarray | None = None,
    sigma_floor: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-item mean and std of scores across a calibration population.

    Args:
        calib_scores: (n_calib, n_items) raw score matrix.
        seen: optional (n_calib, n_items) bool mask of interactions. When
            given, a user's score for an item they have interacted with is
            excluded from that item's stats: at ranking time seen items are
            excluded anyway, so the relevant reference distribution is the
            scores of *candidate* receivers. This also sidesteps inflation
            from self-including clauses firing on the item's own fans.
        sigma_floor: lower bound on the returned std. Class sums are integers
            on the [-T, T] scale, so 1.0 is a conservative floor that keeps
            near-constant score columns from exploding under z-scoring.

    Returns:
        (mu, sigma), each (n_items,) float64. Items with no unseen calibration
        users fall back to mu=0, sigma=sigma_floor (scores pass through).
    """
    s = calib_scores.astype(np.float64)
    if seen is None:
        mu = s.mean(axis=0)
        sd = s.std(axis=0)
    else:
        valid = ~seen
        cnt = valid.sum(axis=0)
        safe = np.maximum(cnt, 1)
        mu = (s * valid).sum(axis=0) / safe
        sd = np.sqrt((((s - mu) ** 2) * valid).sum(axis=0) / safe)
        mu = np.where(cnt == 0, 0.0, mu)
        sd = np.where(cnt == 0, sigma_floor, sd)
    return mu, np.maximum(sd, sigma_floor)


def center_scores(scores: np.ndarray, mu: np.ndarray, lam: float = 1.0) -> np.ndarray:
    """Subtract lam * per-item mean: lam=0 is the raw ranking, lam=1 full centering."""
    return scores.astype(np.float32) - np.float32(lam) * mu.astype(np.float32)


def zscore_scores(scores: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Standardise each item's scores by its calibration mean and std."""
    return (scores.astype(np.float32) - mu.astype(np.float32)) / sigma.astype(
        np.float32
    )


def percentile_scores(
    eval_scores: np.ndarray,
    calib_scores: np.ndarray,
    seen: np.ndarray | None = None,
) -> np.ndarray:
    """Rank each score within its item's calibration distribution (mid-rank).

    Distribution-free alternative to z-scoring: the score for (user, item)
    becomes the fraction of calibration users scoring below it for that item,
    with ties counted half (mid-rank), so every item's scores land on a common
    [0, 1] scale regardless of the shape of its score distribution.

    Args:
        eval_scores: (n_eval, n_items) scores to calibrate.
        calib_scores: (n_calib, n_items) reference scores.
        seen: optional (n_calib, n_items) bool interaction mask; a user's score
            is excluded from the reference distribution of items they have
            seen (same rationale as in item_score_stats).

    Returns:
        (n_eval, n_items) float32 in [0, 1]. Items with an empty reference
        distribution get a constant 0.5.
    """
    out = np.empty(eval_scores.shape, dtype=np.float32)
    for j in range(eval_scores.shape[1]):
        col = calib_scores[:, j] if seen is None else calib_scores[~seen[:, j], j]
        if col.size == 0:
            out[:, j] = 0.5
            continue
        col = np.sort(col)
        left = np.searchsorted(col, eval_scores[:, j], side="left")
        right = np.searchsorted(col, eval_scores[:, j], side="right")
        out[:, j] = (left + right) / (2.0 * col.size)
    return out
