"""Quantify seed variance of the TM + clause-CF evaluation across repeated runs.

Runs experiments/clause_cf_eval.evaluate_run() on each of several seed-replica
run directories (identical config, different training seed) and compiles, per
model row and per metric, the mean and standard deviation across seeds. This is
the clause-CF analogue of the negated-literal seed ablation: it puts an error
bar on every number in the TM/CF block of the results table.

Because the val/test split is seeded independently of the training seed (see
evaluate_run), every run is scored on one fixed evaluation set — the spread
reported here is genuine model-training variance, not resampling noise.

Alpha selection (``--alpha_mode``):
  * ``pooled`` (default): the blend weight alpha is selected ONCE per CF
    approach on the seed-averaged validation curve, then every seed is
    evaluated at that single shared alpha. This decouples model-training
    variance from alpha-selection noise, so the reported std is a clean error
    bar and the table's alpha column is one reproducible value. Preferred when
    the validation curve is flat in alpha (e.g. the pos-weight blend, which by
    construction tracks the TM's own scores), where per-seed argmax jitters
    between adjacent grid points for no real gain.
  * ``per_seed``: alpha is re-tuned on each seed's own validation split; the
    reported variance then also absorbs any flip in the selected operating
    point (the older behaviour).

Discovers run dirs from the given paths: a path that itself contains
metadata.json is treated as a run; otherwise its immediate subdirectories that
contain metadata.json are used.

Outputs a console mean+/-std table, spliceable LaTeX rows printed to stdout
(and optionally written with --latex_out), and a tidy per-run CSV (--out_csv).

Example:
    python experiments/clause_cf_variance.py \
        runs/coalesced_baseline/Vibrent_final_runs \
        --out_csv results/variance/vibrent_cf_variance.csv \
        --latex_out results/tex/vibrent_cf_variance_rows.tex
"""

import argparse
import statistics as st
import sys
from pathlib import Path

_PROJ_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJ_ROOT))

from experiments.clause_cf_eval import _CF_PRETTY, _TM_ROW, evaluate_run
from utils.latex_table import _DEFAULT_METRICS, _DEFAULT_OMIT, _format_cell


def discover_runs(paths: list[Path]) -> list[Path]:
    """Expand each path to the run dirs (those containing metadata.json) under it."""
    runs: list[Path] = []
    for p in paths:
        p = p if p.is_absolute() else _PROJ_ROOT / p
        if (p / "metadata.json").is_file():
            runs.append(p)
        else:
            runs.extend(
                sorted(c for c in p.iterdir() if (c / "metadata.json").is_file())
            )
    if not runs:
        raise SystemExit(f"No run dirs (with metadata.json) found under: {paths}")
    return runs


def ordered_metric_cols(ks: list[int]) -> list[str]:
    """Metric columns in the results-table order, dropping the omitted ones."""
    return [
        f"{metric}@{k}"
        for metric in _DEFAULT_METRICS
        for k in ks
        if f"{metric}@{k}" not in _DEFAULT_OMIT
    ]


def pooled_alphas(val_sweeps, alphas, select_metric):
    """Pick one shared alpha per CF approach on the seed-averaged validation curve.

    Args:
        val_sweeps: list (one per run) of ``{(approach_key, alpha): metrics}``.
        alphas: the full alpha grid.
        select_metric: validation metric to maximise (e.g. "ndcg@10").

    Returns ``{approach_key: alpha}``. Alpha = 1.0 is excluded (it collapses the
    blend onto the pure-TM row), matching evaluate_run's per-run selection.
    """
    blend_alphas = [a for a in alphas if a < 1.0] or list(alphas)
    chosen = {}
    for key in _CF_PRETTY:
        best_a, best_v = None, float("-inf")
        for a in blend_alphas:
            vals = [vs[(key, a)][select_metric] for vs in val_sweeps if (key, a) in vs]
            if not vals:
                continue
            mean_v = st.mean(vals)
            if mean_v > best_v:
                best_v, best_a = mean_v, a
        chosen[key] = best_a
    return chosen


def aggregate(per_run: dict[str, dict[str, dict[str, float]]], cols: list[str]):
    """per_run[run_name][row_label][metric] -> {row_label: {metric: (mean, std, n)}}.

    Row labels are taken from the first run and assumed identical across runs
    (same config => same CF variants). Missing values are skipped so one broken
    run degrades gracefully rather than aborting the batch.
    """
    run_names = list(per_run)
    row_labels = list(per_run[run_names[0]])
    agg: dict[str, dict[str, tuple[float, float, int]]] = {}
    for row in row_labels:
        agg[row] = {}
        for col in cols:
            vals = [
                per_run[r][row][col]
                for r in run_names
                if row in per_run[r] and col in per_run[r][row]
            ]
            if not vals:
                continue
            mean = st.mean(vals)
            std = st.stdev(vals) if len(vals) > 1 else 0.0  # sample std (ddof=1)
            agg[row][col] = (mean, std, len(vals))
    return agg, row_labels


def print_table(agg, row_labels, cols, alpha_by_row):
    name_w = max(len(r) for r in row_labels) + 2
    cell_w = 16
    header = f"{'Model':<{name_w}}" + "".join(f"{c:>{cell_w}}" for c in cols)
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for row in row_labels:
        cells = ""
        for col in cols:
            if col in agg[row]:
                mean, std, _ = agg[row][col]
                cells += f"{mean:.4f}±{std:.4f}".rjust(cell_w)
            else:
                cells += "--".rjust(cell_w)
        print(f"{row:<{name_w}}{cells}")
    print("=" * len(header))
    print("\nSelected alpha per CF row:")
    for row, a in alpha_by_row.items():
        print(f"  {row:<40} {a}")


def render_latex(agg, row_labels, cols, alpha_by_row, precision: int) -> str:
    """Spliceable ``\\midrule``-ready rows with ``$mean \\pm std$`` cells.

    Row labels carry the selected alpha (``($\\alpha=..$)``) to match the main
    results table, and metric cells drop the leading zero for width, exactly as
    the baseline table formats them.
    """
    lines = ["\\midrule"]
    for row in row_labels:
        cells = []
        for col in cols:
            if col in agg[row]:
                mean, std, _ = agg[row][col]
                m = _format_cell(mean, precision, True)
                s = _format_cell(std, precision, True)
                cells.append(f"${m} \\pm {s}$")
            else:
                cells.append("--")
        # Row labels (both the TM row and the CF rows) carry intentional LaTeX
        # (\ac{TM}, \quad, math-mode k) and are emitted verbatim, exactly as the
        # main results table does.
        label = row
        a = alpha_by_row.get(row)
        if a is not None:
            label = f"{label} ($\\alpha={a:g}$)"
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    return "\n".join(lines)


def write_csv(path: Path, per_run, agg, row_labels, cols):
    run_names = list(per_run)
    path = path if path.is_absolute() else _PROJ_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("model,metric," + ",".join(run_names) + ",mean,std,n\n")
        for row in row_labels:
            for col in cols:
                per = [
                    f"{per_run[r][row][col]:.6f}"
                    if row in per_run[r] and col in per_run[r][row]
                    else ""
                    for r in run_names
                ]
                if col in agg[row]:
                    mean, std, n = agg[row][col]
                    f.write(
                        f"{row},{col}," + ",".join(per) + f",{mean:.6f},{std:.6f},{n}\n"
                    )
    print(f"\nWrote per-run CSV to {path}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("paths", nargs="+", type=Path, help="Run dirs or parent folder(s)")
    p.add_argument("--ks", type=int, nargs="+", default=[1, 10, 50])
    p.add_argument(
        "--alphas", type=float, nargs="+", default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    )
    p.add_argument("--select_metric", default="ndcg@10")
    p.add_argument("--uu_k", type=int, default=50)
    p.add_argument(
        "--alpha_mode",
        choices=["pooled", "per_seed"],
        default="pooled",
        help="pooled: one shared alpha per approach on the seed-averaged "
        "validation curve (default). per_seed: alpha re-tuned per seed.",
    )
    p.add_argument("--out_csv", default="results/variance/cf_variance.csv")
    p.add_argument(
        "--latex_out", default=None, help="Optional path to also write the LaTeX rows"
    )
    p.add_argument("--precision", type=int, default=4)
    args = p.parse_args(argv)

    runs = discover_runs(args.paths)
    print(f"Evaluating {len(runs)} run(s) [alpha_mode={args.alpha_mode}]:")
    for r in runs:
        print(f"  {r.name}")

    cols = ordered_metric_cols(list(args.ks))
    alphas = list(args.alphas)

    # One pass over the (expensive) checkpoints: keep each run's per-seed results
    # plus its full test & validation sweeps so alpha can be fixed afterwards.
    raw = {}
    for r in runs:
        results, alpha_by_name, test_recs, val_recs = evaluate_run(
            r,
            ks=tuple(args.ks),
            alphas=alphas,
            select_metric=args.select_metric,
            uu_k=args.uu_k,
        )
        raw[r.name] = {
            "results": results,
            "alpha_by_name": alpha_by_name,
            "test": {(k, a): m for k, a, m in test_recs},
            "val": {(k, a): m for k, a, m in val_recs},
        }

    per_run: dict[str, dict[str, dict[str, float]]] = {}
    if args.alpha_mode == "pooled":
        chosen = pooled_alphas(
            [v["val"] for v in raw.values()], alphas, args.select_metric
        )
        for name, d in raw.items():
            row = {_TM_ROW: d["results"][_TM_ROW]}  # alpha-independent
            for key, label in _CF_PRETTY.items():
                row[label] = d["test"][(key, chosen[key])]
            per_run[name] = row
        alpha_by_row = {_CF_PRETTY[k]: a for k, a in chosen.items()}
    else:  # per_seed
        for name, d in raw.items():
            per_run[name] = d["results"]
        # Report the per-row alpha only when every seed agreed on it; otherwise
        # leave it off the label (the spread already reflects the disagreement).
        alpha_by_row = {}
        for key, label in _CF_PRETTY.items():
            picks = {d["alpha_by_name"][label] for d in raw.values()}
            if len(picks) == 1:
                alpha_by_row[label] = next(iter(picks))

    agg, row_labels = aggregate(per_run, cols)
    print_table(agg, row_labels, cols, alpha_by_row)

    latex = render_latex(agg, row_labels, cols, alpha_by_row, args.precision)
    print("\n% --- mean +/- std rows; splice into the results table ---")
    print(latex)
    if args.latex_out:
        out = Path(args.latex_out)
        out = out if out.is_absolute() else _PROJ_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(latex + "\n", encoding="utf-8")
        print(f"\nWrote LaTeX rows to {out}")

    write_csv(Path(args.out_csv), per_run, agg, row_labels, cols)


if __name__ == "__main__":
    main()
