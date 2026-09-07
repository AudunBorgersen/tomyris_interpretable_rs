"""
Coalesced Weighted TM baseline for MovieLens 1M.

Formulation — masked item prediction:
  For each (user, target_movie) pair in the training data:
    X = user's binary rating profile with the target movie masked out
    Y = column index of the target movie (class label)

The single shared clause bank learns co-taste patterns that generalise across
movies; each movie gets its own weight vector over those shared clauses.

Primary goal: inspect which clause patterns the model converges to.
"""

import argparse
import json
import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

# Import for side effect: caps BLAS/OpenMP threads BEFORE numpy loads (BLAS reads
# the thread env once on import). Honours TOMYRIS_NUM_THREADS / an inherited cap;
# otherwise leaves threads uncapped (fastest when this run owns the host).
import utils.blas_cap  # noqa: F401

import numpy as np
import yaml

from datasets import get_dataset
from utils.metrics import evaluate, clause_weight_stats
from utils.wandb_logger import init_wandb
from utils.clause_analysis import clause_structure_stats

# TMCoalescedClassifier is intentionally NOT imported here.
# pycuda.autoinit fires on import and initialises CUDA against whatever devices
# are visible at that moment. We must set CUDA_VISIBLE_DEVICES first, so the
# import is deferred to main() after the env var has been written.

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent

DEFAULT_DIRECTORY = _PROJECT_ROOT / "configs" / "coalesced_baseline"


def load_config(config_path: Path) -> SimpleNamespace:
    """Load a YAML config and return a flat SimpleNamespace matching parse_args() output."""
    # Support both direct file paths and config names relative to the default directory
    if not config_path.is_file():
        config_path = f"{DEFAULT_DIRECTORY}/{config_path}.yml"

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    # Flatten the nested YAML sections into a single namespace
    flat = {}
    for section in raw.values():
        flat.update(section)
    return SimpleNamespace(**flat)


def save_checkpoint(
    tm,
    run_dir: Path,
    epoch: int,
    metrics: dict,
    metadata: dict,
) -> None:
    from tmu.clause_bank.clause_bank_cuda import ClauseBankCuda
    from utils.checkpoint import _cuda_bank_to_cpu

    ckpt_path = run_dir / f"epoch_{epoch:03d}.pkl"

    # Swap in a CPU clause bank for pickling so checkpoints load without MSVC/CUDA.
    # The GPU bank is restored immediately after so training can continue.
    if isinstance(tm.clause_bank, ClauseBankCuda):
        gpu_bank, gpu_platform = tm.clause_bank, tm.platform
        tm.clause_bank, tm.platform = _cuda_bank_to_cpu(tm), "CPU"
        try:
            # _cuda_bank_to_cpu() unconditionally syncs the device ta_state to host
            # before copying (utils.checkpoint), so an all-empty CPU bank now
            # reflects a genuinely empty model — typically a degenerate, very-low-s
            # configuration sampled during HPO — rather than the old missing-sync
            # bug. Persisting a hollow model is useless for later analysis, so skip
            # the write and let training / the HPO sweep continue instead of
            # aborting the whole run.
            total_includes = sum(
                tm.clause_bank.number_of_include_actions(j)
                for j in range(tm.number_of_clauses)
            )
            if total_includes == 0:
                log.warning(
                    f"Skipping checkpoint {ckpt_path.name}: clause bank has 0 total "
                    "literal includes (empty model — likely a degenerate "
                    "hyperparameter draw). Continuing without saving this epoch."
                )
                return
            with open(ckpt_path, "wb") as f:
                pickle.dump(tm, f)
        finally:
            tm.clause_bank, tm.platform = gpu_bank, gpu_platform
    else:
        with open(ckpt_path, "wb") as f:
            pickle.dump(tm, f)

    metadata["epochs"].append({"epoch": epoch, **metrics})
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    log.info(f"Checkpoint saved: {ckpt_path}")


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def subsample_drop_probs(item_popularity: np.ndarray, t: float) -> np.ndarray:
    """Word2vec-style frequency-subsampling drop probabilities (Mikolov et al.).

    p_drop(i) = max(0, 1 - sqrt(t / f_i)), with f_i the item's training
    popularity (fraction of training users who interacted with it). Items with
    popularity <= t are never dropped; above the threshold the drop probability
    rises smoothly towards 1. Applied to the *input* side of the masked item
    prediction samples, this flattens the frequency distribution of the input
    literals: a literal only survives Type Ib erosion if it is frequently True
    among matching samples, so damping head items lowers the popularity bar
    that keeps tail items out of the clause antecedents.
    """
    f = np.asarray(item_popularity, dtype=np.float64)
    p = np.zeros_like(f)
    pos = f > 0
    p[pos] = 1.0 - np.sqrt(t / f[pos])
    return np.clip(p, 0.0, 1.0)


def build_samples(
    x_matrix: np.ndarray,
    max_per_user: int | None = None,
    rng: np.random.Generator | None = None,
    input_drop_p: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build masked item prediction samples from a binary user-item matrix.

    For each positively-rated movie per user, produces one sample:
      X[i] = user rating vector with the target movie column zeroed out
      Y[i] = column index of the target movie

    Masking prevents the trivial solution of echoing the input feature.

    Args:
        x_matrix: (n_users, n_items) binary uint32 matrix.
        max_per_user: subsample this many positive items per user (None = all).
        rng: Generator for reproducibility.
        input_drop_p: optional (n_items,) per-item probability of zeroing an
            *input* bit, drawn independently per sample (word2vec-style
            frequency subsampling, see subsample_drop_probs). Targets are
            unaffected: every positive still yields a sample, only the profile
            the model conditions on is thinned. Training-only — evaluation
            profiles must stay intact.

    Returns:
        X: (n_samples, n_items) uint32
        Y: (n_samples,) uint32
    """
    if rng is None:
        rng = np.random.default_rng(42)

    rows_x, rows_y = [], []
    for user_vec in x_matrix:
        liked = np.where(user_vec == 1)[0]
        targets = liked
        if max_per_user is not None and len(targets) > max_per_user:
            targets = rng.choice(targets, max_per_user, replace=False)
        droppable = liked[input_drop_p[liked] > 0] if input_drop_p is not None else None
        for col in targets:
            masked = user_vec.copy()
            masked[col] = 0
            if droppable is not None and droppable.size:
                # Fresh draw per sample; col may be redrawn but is already 0.
                hits = droppable[rng.random(droppable.size) < input_drop_p[droppable]]
                masked[hits] = 0
            rows_x.append(masked)
            rows_y.append(col)

    X = np.array(rows_x, dtype=np.uint32)
    Y = np.array(rows_y, dtype=np.uint32)
    return X, Y


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    import gc
    import os
    import time

    import wandb

    if args.type_iii_feedback and args.platform.upper() == "CUDA":
        raise ValueError(
            "type_iii_feedback=True is not supported on platform=CUDA "
            "(ClauseBankCuda does not implement type_iii_feedback). "
            "Set platform: CPU in your config."
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    log.info(f"CUDA_VISIBLE_DEVICES={args.device_id}")
    # Deferred import — must come after CUDA_VISIBLE_DEVICES is set
    from tmu.models.classification.coalesced_classifier import TMCoalescedClassifier

    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    run_dir = _PROJECT_ROOT / "runs" / "coalesced_baseline" / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Run dir: {run_dir}")
    metadata: dict = {"config": vars(args), "epochs": []}

    dataset_name = getattr(args, "dataset", "movielens1m")
    log.info(f"Loading dataset: {dataset_name}...")
    val_ratio = getattr(args, "val_ratio", 0.1)
    ds = get_dataset(
        dataset_name,
        rating_threshold=getattr(args, "rating_threshold", 4),
        test_ratio=0.2,
        val_ratio=val_ratio,
        min_ratings=20,
        k_core=getattr(args, "k_core", 5),
        unit=getattr(args, "unit", "group"),
        random_state=42,
        max_items=args.max_classes,
    )
    data = ds.get()
    n_items = data["x_train"].shape[1]

    log.info(
        f"Train users: {len(data['y_train'])}  "
        f"Test users: {len(data['y_test'])}  "
        f"Items: {n_items}"
    )

    # Fold the dataset into the project name so runs land in a dataset-specific
    # project even when --dataset is overridden without touching wandb_project
    # (e.g. the HPO builds every trial on coalesced_baseline/default.yml and
    # retargets via `fixed: dataset:`). The default movielens1m dataset keeps the
    # bare base name for continuity with existing runs, and we never re-append a
    # token the base already carries (configs that already namespace by dataset).
    project_base = args.wandb_project
    if dataset_name != "movielens1m" and dataset_name not in project_base:
        project_base = f"{project_base}-{dataset_name}"
    wandb_project = f"{project_base}-{n_items}cls"
    # A caller (e.g. scripts/repeat_seeds.py) may override the display name to
    # tell seed replicas apart, and set a group so wandb aggregates them.
    wandb_run_name = (
        getattr(args, "wandb_run_name", None)
        or f"{ts}_{args.clauses}cl_T{args.T}_s{args.s}"
    )
    init_wandb(
        wandb_project,
        wandb_run_name,
        dataset_name=dataset_name,
        config=vars(args),
        group=getattr(args, "wandb_group", None),
        tags=getattr(args, "wandb_tags", None),
    )
    log.info(f"wandb project: {wandb_project}  run: {wandb_run_name}")

    rng = np.random.default_rng(args.seed)

    item_popularity = data["x_train"].sum(axis=0) / len(data["y_train"])

    # Word2vec-style input subsampling (popularity-bias mitigation): head items
    # are dropped from the *input* profiles with probability rising in their
    # popularity, flattening the literal frequency distribution the clauses
    # learn from. The corruption is redrawn every epoch, so training samples
    # are rebuilt inside the epoch loop instead of once up front.
    subsample_t = getattr(args, "input_subsample_t", 0.0) or 0.0
    subsample_mode = getattr(args, "subsample_mode", "feedback") or "feedback"
    if subsample_mode not in ("input", "feedback"):
        raise ValueError(
            f"subsample_mode must be 'input' or 'feedback', got {subsample_mode!r}"
        )
    drop_p = (
        subsample_drop_probs(item_popularity, subsample_t) if subsample_t > 0 else None
    )
    if drop_p is not None:
        eligible = drop_p > 0
        log.info(
            f"Popularity subsampling t={subsample_t} mode={subsample_mode}: "
            f"{int(eligible.sum())}/{n_items} items eligible, mean drop prob "
            f"among eligible {drop_p[eligible].mean():.2f}."
        )
    # "input" zeroes the drawn bits in the training inputs (word2vec-style
    # corruption — punishes conjunctions, samples rebuilt per epoch);
    # "feedback" leaves the inputs intact and hides the drawn bits from TA
    # feedback inside the model (masks redrawn per sample every epoch).
    input_drop_p = drop_p if subsample_mode == "input" else None
    feedback_drop_p = drop_p if subsample_mode == "feedback" else None

    max_pu = args.max_per_user if args.max_per_user else None
    if input_drop_p is None:
        log.info("Building training samples (masked item prediction)...")
        X_train, Y_train = build_samples(data["x_train"], max_pu, rng)
        log.info(
            f"  {len(X_train):,} training samples  ({n_items} features, {n_items} classes)"
        )
        mem_mb = X_train.nbytes / 1024**2
        log.info(f"  Matrix memory: {mem_mb:.0f} MB")
    else:
        X_train = Y_train = None  # rebuilt per epoch with fresh corruption
        log.info("Input-mode subsampling: training samples rebuilt each epoch.")

    log.info("Building validation + test samples (one held-out item per user)...")
    X_val, Y_val = build_samples(data["x_val"], max_per_user=None, rng=rng)
    X_test, Y_test = build_samples(data["x_test"], max_per_user=None, rng=rng)
    log.info(f"  {len(X_val):,} val samples, {len(X_test):,} test samples")

    log.info("Initialising TMCoalescedClassifier...")
    max_pos = args.max_positive_clauses if args.max_positive_clauses else None
    tm = TMCoalescedClassifier(
        number_of_clauses=args.clauses,
        T=args.T,
        s=args.s,
        weighted_clauses=True,
        focused_negative_sampling=True,
        feature_negation=args.feature_negation,
        max_included_literals=args.max_included_literals,
        output_balancing=args.output_balancing,
        type_iii_feedback=args.type_iii_feedback,
        max_positive_clauses=max_pos,
        mask_target_literal_p=args.mask_target_literal_p,
        literal_feedback_drop_p=feedback_drop_p,
        clause_drop_p=args.clause_drop_p,
        type_i_ii_ratio=args.type_i_ii_ratio,
        seed=args.seed,
        platform=args.platform,
    )

    log.info(
        f"Training ({args.clauses} clauses × {n_items} classes — "
        f"expect ~{args.clauses / 100 * 16:.0f} min/epoch on CPU)..."
    )

    # Hyperparameters are selected on the validation split; the test split is
    # evaluated every epoch only for monitoring and is never used for selection.
    # ndcg@10 (and the other rank metrics) typically peak at epoch ~3-5 then
    # decay as coverage keeps rising, so the final epoch under-represents the
    # model. Track the best validation value per metric (and the epoch it
    # occurred) so sweeps rank on the validation peak, then report the test
    # value at that same epoch.
    use_val = len(X_val) > 0
    if not use_val:
        log.warning(
            "No validation users (val_ratio=0): falling back to selecting on the "
            "TEST set (legacy behaviour). Set val_ratio > 0 for leak-free selection."
        )

    best: dict = {}
    test_by_epoch: dict[int, dict] = {}
    try:
        for epoch in range(1, args.epochs + 1):
            if input_drop_p is not None:
                # Release the previous epoch's matrix before rebuilding so peak
                # memory stays ~1x, then redraw the input corruption.
                X_train = Y_train = None
                X_train, Y_train = build_samples(
                    data["x_train"],
                    max_pu,
                    np.random.default_rng([args.seed, epoch]),
                    input_drop_p=input_drop_p,
                )
                if epoch == 1:
                    log.info(
                        f"  {len(X_train):,} training samples/epoch "
                        f"({n_items} features, {n_items} classes)"
                    )
            t0 = time.time()
            tm.fit(X_train, Y_train, shuffle=True)
            elapsed = time.time() - t0
            log.info(f"Epoch {epoch:2d} trained in {elapsed:.0f}s")

            eval_kwargs = dict(
                ks=(1, 10, 50), item_popularity=item_popularity, n_items=n_items
            )
            test_m = evaluate(tm, X_test, Y_test, **eval_kwargs)
            test_by_epoch[epoch] = test_m
            # Selection signal: validation when available, else test (legacy).
            val_m = evaluate(tm, X_val, Y_val, **eval_kwargs) if use_val else test_m

            ws = clause_weight_stats(tm)
            cs = clause_structure_stats(tm, profiles=data["x_test"])
            log.info(
                f"Epoch {epoch:2d} | "
                f"val: Hit@10={val_m['hit@10']:.4f} NDCG@10={val_m['ndcg@10']:.4f}  |  "
                f"test: Hit@10={test_m['hit@10']:.4f} NDCG@10={test_m['ndcg@10']:.4f} "
                f"NDCG@50={test_m['ndcg@50']:.4f} Cov@10={test_m['coverage@10']:.4f} "
                f"NDCG@10 h/m/t={test_m.get('ndcg@10_head', 0):.4f}/"
                f"{test_m.get('ndcg@10_mid', 0):.4f}/{test_m.get('ndcg@10_tail', 0):.4f} "
                f"Gini@10={test_m['gini@10']:.3f} ARP@10={test_m['arp@10']:.4f}  |  "
                f"ClausePos mean={ws['clause_pos_weight_mean']:.1f} std={ws['clause_pos_weight_std']:.1f}  "
                f"Size mean={cs['clause_size_mean']:.2f}  DeepCommit={cs['deep_commit_frac']:.3f}  "
                f"EffCl/cls={cs['eff_clauses_per_class']:.0f}  RankDead={cs['rank_dead_frac']:.2f}  "
                f"Core={cs['literal_core_size']}  Vis={cs['profile_visibility_mean']:.3f}  "
                f"Blind={cs['blind_user_frac']:.3f}"
            )
            wandb.log(
                {
                    **{f"val/{k}": v for k, v in val_m.items()},
                    **{f"test/{k}": v for k, v in test_m.items()},
                    **ws,
                    **cs,
                    "epoch_time_s": elapsed,
                }
            )

            for k, v in val_m.items():
                bk = f"best_{k}"
                if bk not in best or v > best[bk]:
                    best[bk] = v
                    best[f"{bk}_epoch"] = epoch

            save_checkpoint(tm, run_dir, epoch, {**val_m, **ws, **cs}, metadata)

        # Test metrics at the epoch that maximised each validation metric — the
        # honest held-out number for the val-selected operating point.
        test_at_best = {
            f"test_{k}": test_by_epoch[best[f"best_{k}_epoch"]][k]
            for k in test_by_epoch[1]
        }

        wandb.summary.update({**best, **test_at_best})
        sel_name = "validation" if use_val else "test (no val set)"
        log.info(
            f"Best on {sel_name} across {args.epochs} epochs | "
            f"NDCG@10: val={best['best_ndcg@10']:.4f} (ep {best['best_ndcg@10_epoch']}) "
            f"-> test={test_at_best['test_ndcg@10']:.4f}  "
            f"Hit@10: val={best['best_hit@10']:.4f} test={test_at_best['test_hit@10']:.4f}"
        )

        return {**best, **test_at_best}
    finally:
        # Release this run's CUDA context deterministically. Sequential HPO trials
        # call main() in one process, and the GPU clause bank's context is
        # otherwise only torn down by ClauseBankCudaDevice.__del__ at GC time —
        # non-deterministic, and its pop() targets whatever context is current
        # (not necessarily this one). On a GPU that permits a single context
        # (EXCLUSIVE_PROCESS), the next trial's make_context() then collides
        # ("after the first run"). Explicit cleanup + gc.collect() frees it now.
        cb = getattr(tm, "clause_bank", None)
        if str(getattr(args, "platform", "")).upper() == "CUDA" and hasattr(
            cb, "device"
        ):
            cb.device.cleanup()
        del tm
        gc.collect()


def parse_args(argv=None, **overrides) -> SimpleNamespace:
    """Load config from --config, then apply any CLI overrides on top.

    Pass argv=[] when calling programmatically (e.g. from HPO) to avoid
    reading sys.argv, which would contain the caller's own arguments.
    """
    p = argparse.ArgumentParser(description="Coalesced TM rec-sys baseline")
    p.add_argument(
        "--config",
        type=Path,
        default=_PROJECT_ROOT / "configs" / "coalesced_baseline" / "default.yml",
        help="Path to YAML config file.",
    )
    # Every key in the YAML can be overridden on the CLI with its flat name.
    # Types are inferred from the config value; unknown keys are passed through.
    p.add_argument("--clauses", type=int, default=None)
    p.add_argument("--T", type=int, default=None)
    p.add_argument("--s", type=float, default=None)
    p.add_argument("--max_included_literals", type=int, default=None)
    p.add_argument(
        "--feature_negation", type=lambda x: x.lower() != "false", default=None
    )
    p.add_argument("--platform", type=str, default=None)
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--rating_threshold", type=int, default=None)
    p.add_argument("--k_core", type=int, default=None)
    p.add_argument("--unit", type=str, default=None)
    p.add_argument("--val_ratio", type=float, default=None)
    p.add_argument("--max_per_user", type=int, default=None)
    p.add_argument("--input_subsample_t", type=float, default=None)
    p.add_argument(
        "--subsample_mode",
        type=str,
        default=None,
        choices=["input", "feedback"],
        help="How input_subsample_t is applied: 'input' zeroes bits in the "
        "training inputs, 'feedback' hides them from TA feedback only.",
    )
    p.add_argument("--max_classes", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Override the wandb display name (default: auto ts_clauses_T_s).",
    )
    p.add_argument(
        "--wandb_group",
        type=str,
        default=None,
        help="wandb run group — seed replicas of one config share a group.",
    )
    p.add_argument("--wandb_tags", type=str, nargs="*", default=None)
    p.add_argument(
        "--output_balancing", type=lambda x: x.lower() != "false", default=None
    )
    p.add_argument(
        "--type_iii_feedback", type=lambda x: x.lower() != "false", default=None
    )
    p.add_argument("--max_positive_clauses", type=int, default=None)
    p.add_argument("--mask_target_literal_p", type=float, default=None)
    p.add_argument("--clause_drop_p", type=float, default=None)
    p.add_argument("--type_i_ii_ratio", type=float, default=None)
    p.add_argument(
        "--device_id",
        type=int,
        default=None,
        help="GPU index to expose via CUDA_VISIBLE_DEVICES.",
    )

    cli = p.parse_args(argv)
    cfg = load_config(cli.config)

    # CLI non-None values take priority over config file
    for key, val in vars(cli).items():
        if key == "config":
            continue
        if val is not None:
            setattr(cfg, key, val)

    # Programmatic overrides (used in tests / notebooks)
    for key, val in overrides.items():
        setattr(cfg, key, val)

    log.info(f"Config: {cli.config}")
    # Print the final config after all overrides have been applied
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
