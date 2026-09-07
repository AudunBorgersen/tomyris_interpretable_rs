"""Repeat a set of coalesced_baseline configs across several seeds.

For each (config, seed) it launches experiments/coalesced_baseline.py in its own
subprocess — a clean CUDA init/teardown per run, and one crashing seed does not
abort the batch. Each run gets a distinct wandb display name
``<config-stem>_seed<seed>`` and all seeds of a config share the wandb group
``<config-stem>`` so wandb aggregates the replicas (mean/std) out of the box.

Examples:
    # 5 seeds (1..5) of both ablation arms:
    python scripts/repeat_seeds.py \
        configs/coalesced_baseline/ablation/ml_ablation_no_negation.yml \
        configs/coalesced_baseline/ablation/ml_ablation_negation.yml \
        --n 5

    # Explicit seeds, pinned to GPU 6, preview only:
    python scripts/repeat_seeds.py configs/.../ml_ablation_negation.yml \
        --seeds 42 43 44 --device_id 6 --dry-run
"""

import argparse
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_BASELINE = _PROJECT_ROOT / "experiments" / "coalesced_baseline.py"


def resolve_seeds(args: argparse.Namespace) -> list[int]:
    if args.seeds is not None:
        seeds = args.seeds
    else:
        seeds = list(range(args.base_seed, args.base_seed + args.n))
    if 0 in seeds:
        # pcg32_seed(0) zeroes the MCG state and hangs CPU training forever
        # (see the popularity-bias workstream notes). Refuse it up front.
        raise SystemExit(
            "seed 0 is forbidden (hangs TMU training); use non-zero seeds."
        )
    return seeds


def build_cmd(
    config: Path, seed: int, device_id: int | None, tags: list[str] | None
) -> list[str]:
    stem = config.stem
    cmd = [
        sys.executable,
        str(_BASELINE),
        "--config",
        str(config),
        "--seed",
        str(seed),
        "--wandb_run_name",
        f"{stem}_seed{seed}",
        "--wandb_group",
        stem,
    ]
    if device_id is not None:
        cmd += ["--device_id", str(device_id)]
    if tags:
        cmd += ["--wandb_tags", *tags]
    return cmd


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "configs", nargs="+", type=Path, help="One or more config .yml paths"
    )
    grp = p.add_mutually_exclusive_group()
    grp.add_argument("--seeds", type=int, nargs="+", help="Explicit seeds to run")
    grp.add_argument(
        "--n",
        type=int,
        default=5,
        help="Run seeds base_seed..base_seed+n-1 (default 5)",
    )
    p.add_argument(
        "--base-seed", type=int, default=1, help="First seed when using --n (default 1)"
    )
    p.add_argument(
        "--device_id", type=int, default=None, help="Override GPU for every run"
    )
    p.add_argument(
        "--wandb_tags",
        type=str,
        nargs="*",
        default=None,
        help="Tags applied to every run",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Print the commands without running"
    )
    args = p.parse_args(argv)

    configs = []
    for c in args.configs:
        c = c if c.is_absolute() else _PROJECT_ROOT / c
        if not c.is_file():
            raise SystemExit(f"config not found: {c}")
        configs.append(c)
    seeds = resolve_seeds(args)

    jobs = [(c, s) for c in configs for s in seeds]
    print(
        f"{len(configs)} config(s) x {len(seeds)} seed(s) = {len(jobs)} runs; seeds={seeds}"
    )

    results: list[tuple[str, int, int]] = []  # (stem, seed, returncode)
    for i, (config, seed) in enumerate(jobs, 1):
        cmd = build_cmd(config, seed, args.device_id, args.wandb_tags)
        print(f"\n[{i}/{len(jobs)}] {config.stem} seed={seed}")
        print("  " + " ".join(cmd))
        if args.dry_run:
            continue
        rc = subprocess.run(cmd, cwd=_PROJECT_ROOT).returncode
        results.append((config.stem, seed, rc))
        if rc != 0:
            print(f"  !! exited {rc} — continuing with remaining runs")

    if args.dry_run:
        return

    failures = [(stem, s, rc) for stem, s, rc in results if rc != 0]
    print("\n" + "=" * 60)
    print(f"Completed {len(results) - len(failures)}/{len(results)} runs.")
    for stem, s, rc in failures:
        print(f"  FAILED: {stem} seed={s} (exit {rc})")
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
