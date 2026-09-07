import numpy as np

# Popularity stratum names, ordered by the integer labels of popularity_strata().
STRATA_NAMES = ("head", "mid", "tail")


def gini(counts: np.ndarray) -> float:
    """Gini coefficient of a non-negative count vector.

    Applied to per-item recommendation counts (zeros included), it measures how
    unevenly the top-K slots are allocated across the catalogue: 0 = every item
    recommended equally often, ->1 = all slots go to a single item.
    """
    x = np.sort(np.asarray(counts, dtype=np.float64))
    n = x.size
    total = x.sum()
    if n == 0 or total == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float(((2 * idx - n - 1) * x).sum() / (n * total))


def popularity_strata(
    item_popularity: np.ndarray, boundaries: tuple[float, float] = (1 / 3, 2 / 3)
) -> np.ndarray:
    """Label each item head (0), mid (1), or tail (2) by popularity rank.

    Items are sorted by descending training popularity and split at the given
    fractions of the catalogue, so the default yields equal-size tertiles by
    item count (not by interaction mass — the head tertile still holds most of
    the interactions).
    """
    order = np.argsort(-item_popularity, kind="stable")
    n = len(order)
    cut1, cut2 = int(n * boundaries[0]), int(n * boundaries[1])
    labels = np.empty(n, dtype=np.int8)
    labels[order[:cut1]] = 0
    labels[order[cut1:cut2]] = 1
    labels[order[cut2:]] = 2
    return labels


def evaluate(
    tm,
    X: np.ndarray,
    Y: np.ndarray,
    ks: tuple = (1, 10, 50),
    batch_size: int = 128,
    item_popularity: np.ndarray | None = None,
    n_items: int | None = None,
    exclude_seen: bool = True,
    scores: np.ndarray | None = None,
) -> dict:
    """Compute ranking metrics for a single-relevant-item evaluation set.

    Always computed: Hit@k, NDCG@k.
    Computed when item_popularity is provided: Novelty@k, ARP@k (mean training
        popularity of the recommended items — lower = less popularity-biased),
        popularity-stratified Hit@k/NDCG@k (suffixed _head/_mid/_tail by the
        target item's popularity tertile), and the eval-target share per
        stratum (target_share_*).
    Computed when n_items is provided: Coverage@k and Gini@k (concentration of
        recommendation counts over the catalogue; 0 = even, ->1 = one item
        takes every slot).

    Args:
        tm: trained TM model with predict(X, return_class_sums=True). May be
            None when a precomputed `scores` matrix is supplied instead.
        X: (n_samples, n_features) test inputs. Columns share the item-class
            space, i.e. a 1 in column j marks item j as already present in the
            user's (target-masked) profile.
        Y: (n_samples,) ground-truth class indices.
        ks: cut-off values.
        batch_size: samples per predict() call.
        item_popularity: (n_items,) positive interaction rate per item in
            training (x_train.sum(axis=0) / n_train_users). Used for Novelty.
        n_items: total catalogue size. Used for Coverage.
        exclude_seen: when True (default), items already present in a user's
            profile are removed from that user's ranking before metrics are
            computed, as is standard for top-N recommendation. The held-out
            target is unaffected since it is masked out of X. Requires X to be
            in the same column space as the model's class scores.
        scores: optional (n_samples, n_items) precomputed score matrix, aligned
            row-for-row with X. When given, it is used in place of tm.predict —
            letting blended / CF score matrices be scored through the same
            seen-item exclusion and metric code as the model path.
    """
    if tm is None and scores is None:
        raise ValueError("evaluate requires either a model `tm` or a `scores` matrix.")

    max_k = max(ks)
    hit_counts = {k: 0 for k in ks}
    ndcg_sums = {k: 0.0 for k in ks}
    novelty_sums = {k: 0.0 for k in ks} if item_popularity is not None else None
    arp_sums = {k: 0.0 for k in ks} if item_popularity is not None else None
    rec_counts = (
        {k: np.zeros(n_items, dtype=np.int64) for k in ks}
        if n_items is not None
        else None
    )
    if item_popularity is not None:
        strata = popularity_strata(item_popularity)
        strat_hit = {k: np.zeros(len(STRATA_NAMES)) for k in ks}
        strat_ndcg = {k: np.zeros(len(STRATA_NAMES)) for k in ks}
        strat_n = np.zeros(len(STRATA_NAMES), dtype=np.int64)
    else:
        strata = None

    for start in range(0, len(X), batch_size):
        bx = X[start : start + batch_size]
        by = Y[start : start + batch_size]
        if scores is not None:
            sums = scores[start : start + batch_size]
        else:
            _, sums = tm.predict(bx, return_class_sums=True)
        if exclude_seen:
            if sums.shape != bx.shape:
                raise ValueError(
                    "exclude_seen requires X and the model's class scores to "
                    f"share a column space; got X {bx.shape} and sums {sums.shape}."
                )
            # Demote items already in the user's profile so they cannot occupy
            # ranking slots. The target is masked out of bx, so it is untouched.
            sums = np.where(bx.astype(bool), -np.inf, sums.astype(np.float64))
        ranked = np.argsort(-sums, axis=1)[:, :max_k]

        if rec_counts is not None:
            for k in ks:
                rec_counts[k] += np.bincount(ranked[:, :k].ravel(), minlength=n_items)

        for i in range(len(by)):
            pos = np.where(ranked[i] == by[i])[0]
            rank = int(pos[0]) + 1 if len(pos) > 0 else max_k + 1
            if strata is not None:
                stratum = strata[by[i]]
                strat_n[stratum] += 1
            for k in ks:
                if rank <= k:
                    hit_counts[k] += 1
                    gain = 1.0 / np.log2(rank + 1)
                    ndcg_sums[k] += gain
                    if strata is not None:
                        strat_hit[k][stratum] += 1
                        strat_ndcg[k][stratum] += gain
                if novelty_sums is not None:
                    pop = np.clip(item_popularity[ranked[i, :k]], 1e-10, 1.0)
                    novelty_sums[k] += float(-np.log2(pop).mean())
                    arp_sums[k] += float(item_popularity[ranked[i, :k]].mean())

    n = len(Y)
    result = {
        **{f"hit@{k}": hit_counts[k] / n for k in ks},
        **{f"ndcg@{k}": ndcg_sums[k] / n for k in ks},
    }
    if novelty_sums is not None:
        result.update({f"novelty@{k}": novelty_sums[k] / n for k in ks})
        result.update({f"arp@{k}": arp_sums[k] / n for k in ks})
    if strata is not None:
        for s_idx, name in enumerate(STRATA_NAMES):
            result[f"target_share_{name}"] = strat_n[s_idx] / n
            # Skip ratio metrics for strata with no eval targets rather than
            # reporting a misleading 0.
            if strat_n[s_idx] == 0:
                continue
            for k in ks:
                result[f"hit@{k}_{name}"] = strat_hit[k][s_idx] / strat_n[s_idx]
                result[f"ndcg@{k}_{name}"] = strat_ndcg[k][s_idx] / strat_n[s_idx]
    if rec_counts is not None:
        result.update(
            {f"coverage@{k}": (rec_counts[k] > 0).sum() / n_items for k in ks}
        )
        result.update({f"gini@{k}": gini(rec_counts[k]) for k in ks})
    return result


def clause_weight_stats(tm) -> dict:
    """Distribution of positive-weight clause counts across classes.

    A high std relative to mean indicates the clause bank is top-heavy: popular
    classes have accumulated many more positive-weight clauses than niche ones.

    Returns keys: clause_pos_weight_{mean,std,min,max}.
    """
    n_pos = np.array(
        [(tm.get_weights(i) > 0).sum() for i in range(tm.number_of_classes)]
    )
    return {
        "clause_pos_weight_mean": float(n_pos.mean()),
        "clause_pos_weight_std": float(n_pos.std()),
        "clause_pos_weight_min": int(n_pos.min()),
        "clause_pos_weight_max": int(n_pos.max()),
    }
