"""Render baseline recommender results as a LaTeX table.

Consumes the ``{model_name: {metric_key: value}}`` mapping produced by
``baseline_models/run_all.py`` (metric keys like ``hit@10``, ``ndcg@1``,
``novelty@50``, ``coverage@10``) and emits a booktabs ``tabular``. Requires
``\\usepackage{booktabs}`` in the document preamble.
"""

from __future__ import annotations

import re

# Pretty column headers for the metric families used by the baselines.
_METRIC_LABELS = {
    "hit": "Hit",
    "ndcg": "NDCG",
    "novelty": "Novelty",
    "coverage": "Coverage",
}
_DEFAULT_METRICS = ("hit", "ndcg", "novelty", "coverage")
# Diversity metrics aren't strictly "more is better" for a recommender (Random
# trivially maximises them), so highlight the top two rather than crowning a
# single winner.
_DIVERSITY_METRICS = ("novelty", "coverage")
# NDCG@1 is identical to Hit@1 (a single relevant item is either ranked first
# or not), so it's a redundant column in every table we print.
_DEFAULT_OMIT = ("ndcg@1",)


def _escape(text: str) -> str:
    """Escape the LaTeX special characters that show up in model names."""
    for char in ("\\", "&", "%", "$", "#", "_", "{", "}", "~", "^"):
        text = text.replace(char, "\\" + char)
    return text


def _format_cell(value: float, precision: int, drop_leading_zero: bool) -> str:
    """Format a metric value, optionally rendering 0.123 as .123."""
    cell = f"{value:.{precision}f}"
    if drop_leading_zero:
        if cell.startswith("0."):
            cell = cell[1:]
        elif cell.startswith("-0."):
            cell = "-" + cell[2:]
    return cell


def _infer_ks(results: dict[str, dict[str, float]]) -> list[int]:
    """Collect the @k cut-offs present across all metric keys, sorted."""
    ks: set[int] = set()
    for row in results.values():
        for key in row:
            m = re.search(r"@(\d+)$", key)
            if m:
                ks.add(int(m.group(1)))
    return sorted(ks)


def results_to_latex(
    results: dict[str, dict[str, float]],
    ks: list[int] | None = None,
    *,
    metrics: tuple[str, ...] = _DEFAULT_METRICS,
    diversity_metrics: tuple[str, ...] = _DIVERSITY_METRICS,
    omit: tuple[str, ...] = _DEFAULT_OMIT,
    precision: int = 4,
    bold_best: bool = True,
    group_headers: bool = True,
    drop_leading_zero: bool = True,
    tabcolsep: str | None = "3pt",
    resizebox: bool = False,
    font_size: str | None = None,
    caption: str | None = None,
    label: str | None = None,
) -> str:
    """Build a LaTeX table string from baseline results.

    Args:
        results: ``{model_name: {metric_key: value}}`` as produced by run_all.
        ks: cut-off values, ordered; inferred from the keys when omitted.
        metrics: metric families to include, in column order. Columns are
            grouped by family then by k (e.g. Hit@1, Hit@10, NDCG@1, ...),
            matching run_all's console table.
        diversity_metrics: families bolded by top-two value rather than top-one,
            since they are not strictly "higher is better" for a recommender.
        omit: metric@k columns to drop even if present in results (default:
            ``ndcg@1``, which is always identical to ``hit@1``).
        precision: decimal places for each value.
        bold_best: bold the best value(s) in each column.
        group_headers: use a two-row header (metric family spanned by
            ``\\multicolumn``/``\\cmidrule`` over bare k values). This is the main
            width saving — it shrinks each column's header from e.g. "Coverage@10"
            to "10", so column width is set by the data, not the header.
        drop_leading_zero: render 0.123 as .123 to save a character per cell.
        tabcolsep: inter-column padding, e.g. "3pt" (LaTeX default is 6pt).
            Emitted as ``\\setlength{\\tabcolsep}{...}`` before the tabular;
            pass None to leave it unchanged.
        resizebox: wrap the tabular in ``\\resizebox{\\textwidth}{!}{...}`` so it
            is scaled to the text width as a guaranteed-fit escape hatch
            (requires ``\\usepackage{graphicx}``).
        font_size: a LaTeX size command without the backslash (e.g. "small",
            "footnotesize") applied around the table body.
        caption, label: when either is given, wrap the tabular in a floating
            ``table*`` environment with ``\\caption``/``\\label``.

    Returns:
        The LaTeX source as a single string.
    """
    if not results:
        raise ValueError("results is empty")
    if ks is None:
        ks = _infer_ks(results)

    # Column order: group by metric family, then by k — only keep keys that
    # actually appear in the results.
    present = {key for row in results.values() for key in row}
    cols = [
        f"{metric}@{k}"
        for metric in metrics
        for k in ks
        if f"{metric}@{k}" in present and f"{metric}@{k}" not in omit
    ]
    if not cols:
        raise ValueError("no matching metric columns found in results")

    # Values to bold per column: top-one for accuracy metrics, top-two for the
    # diversity metrics (where a single "winner" is misleading).
    tol = 10 ** (-precision - 1)
    bold_targets: dict[str, list[float]] = {}
    for col in cols:
        n_top = 2 if col.split("@")[0] in diversity_metrics else 1
        ranked = sorted(
            {row[col] for row in results.values() if col in row}, reverse=True
        )
        bold_targets[col] = ranked[:n_top]

    def fmt(model: str, col: str) -> str:
        if col not in results[model]:
            return "--"
        val = results[model][col]
        cell = _format_cell(val, precision, drop_leading_zero)
        if bold_best and any(abs(val - t) < tol for t in bold_targets[col]):
            cell = f"\\textbf{{{cell}}}"
        return cell

    # Header rows: either a flat "Hit@10" row, or a grouped two-row header where
    # the metric family spans its k-columns and the sub-row holds bare k values.
    if group_headers:
        super_cells, sub_cells, cmidrules = [""], ["Model"], []
        start = 2  # column 1 is the model name
        for metric in metrics:
            block = [c for c in cols if c.split("@")[0] == metric]
            if not block:
                continue
            label_txt = _METRIC_LABELS.get(metric, metric.capitalize())
            super_cells.append(f"\\multicolumn{{{len(block)}}}{{c}}{{{label_txt}}}")
            end = start + len(block) - 1
            cmidrules.append(f"\\cmidrule(lr){{{start}-{end}}}")
            sub_cells.extend(c.split("@")[1] for c in block)
            start = end + 1
        header_lines = [
            " & ".join(super_cells) + " \\\\",
            " ".join(cmidrules),
            " & ".join(sub_cells) + " \\\\",
        ]
    else:
        flat = [
            f"{_METRIC_LABELS.get(c.split('@')[0], c.split('@')[0].capitalize())}@{c.split('@')[1]}"
            for c in cols
        ]
        header_lines = ["Model & " + " & ".join(flat) + " \\\\"]

    col_format = "l" + "r" * len(cols)
    lines = [
        f"\\begin{{tabular}}{{{col_format}}}",
        "\\toprule",
        *header_lines,
        "\\midrule",
    ]
    for model, row in results.items():
        cells = " & ".join(fmt(model, c) for c in cols)
        lines.append(f"{_escape(model)} & {cells} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    table = "\n".join(lines)

    if resizebox:
        table = f"\\resizebox{{\\textwidth}}{{!}}{{%\n{table}\n}}"
    if tabcolsep is not None:
        table = f"\\setlength{{\\tabcolsep}}{{{tabcolsep}}}\n{table}"

    if caption is not None or label is not None:
        wrap = ["\\begin{table*}[t]", "\\centering"]
        if font_size is not None:
            wrap.append(f"\\{font_size}")
        wrap.append(table)
        if caption is not None:
            wrap.append(f"\\caption{{{caption}}}")
        if label is not None:
            wrap.append(f"\\label{{{label}}}")
        wrap.append("\\end{table*}")
        table = "\n".join(wrap)

    return table


def latex_rows(
    results: dict[str, dict[str, float]],
    ks: list[int],
    alpha_by_name: dict[str, float] | None = None,
    *,
    metrics: tuple[str, ...] = _DEFAULT_METRICS,
    omit: tuple[str, ...] = _DEFAULT_OMIT,
    precision: int = 4,
    drop_leading_zero: bool = True,
) -> str:
    """Build spliceable ``\\midrule``-ready LaTeX rows (no header/tabular wrapper).

    For appending rows (e.g. TM/CF blend variants) to an existing table produced
    by ``results_to_latex``, in the same column order. When ``alpha_by_name`` is
    given, a matching row label is suffixed with the selected blend weight (e.g.
    ``... CF ($\\alpha=0.6$)``); rows without an entry are left unlabelled.
    """
    alpha_by_name = alpha_by_name or {}
    cols = [
        (metric, k) for metric in metrics for k in ks if f"{metric}@{k}" not in omit
    ]
    lines = ["\\midrule"]
    for name, row in results.items():
        cells = " & ".join(
            _format_cell(row[f"{metric}@{k}"], precision, drop_leading_zero)
            for metric, k in cols
        )
        label = name
        if name in alpha_by_name:
            label = f"{name} ($\\alpha={alpha_by_name[name]:g}$)"
        lines.append(f"{label} & {cells} \\\\")
    return "\n".join(lines)
