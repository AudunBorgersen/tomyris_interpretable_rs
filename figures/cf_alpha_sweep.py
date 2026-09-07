"""NDCG@10 of the clause-based CF blends as a function of the blend weight alpha.

Reads the per-alpha test sweep written by experiments/clause_cf_eval.py at
results/cf_alpha_sweep.csv (columns: approach, alpha, ndcg@10,
novelty@10, coverage@10). If that CSV is absent, it falls back to the hardcoded
numbers below (from notebooks/clause_cf.ipynb, all-positives / exclude_seen
protocol) so the figure can still be drawn — regenerate the CSV via
clause_cf_eval.py for the authoritative version.

alpha = 1 is the pure TM (every blend collapses to it), alpha = 0 the pure
clause-space recommender. The accuracy-vs-diversity trade-off is discussed in the
body; this figure only shows the accuracy dependence on alpha.

Output: results/cf_alpha_sweep.{pdf,png} (PDF is the vector version
included in the paper; PNG is for quick previews).
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt

FIG_DIR = Path(__file__).parent.parent / "results"
CSV = FIG_DIR / "cf_alpha_sweep.csv"
OUT_STEM = FIG_DIR / "cf_alpha_sweep"  # saved as both .pdf (vector) and .png

# Per-approach display style, keyed by the labels used in the eval-script CSV.
STYLE = {
    "user-user": ("User--User", "steelblue", "o"),
    "uu-knn": ("User--User (top-k)", "darkorange", "."),
    "item-item": ("Item--Item", "tomato", "v"),
    "ii-wmod": ("Item--Item (weight-mod.)", "seagreen", "^"),
    "ii-pos-weight": ("Item--Item (pos-weight)", "darkorchid", ">"),
}

# Fallback NDCG@10 per alpha (from clause_cf.ipynb, run 20260624T040526,
# all-positives + exclude_seen). Used only when the CSV is missing.
_ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
_FALLBACK = {
    "user-user": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "uu-knn": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "item-item": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "ii-wmod": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "ii-pos-weight": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}


def load_sweep():
    """Return ({approach: {"alpha": [...], "ndcg": [...]}}, used_fallback)."""
    if not CSV.exists():
        print(f"[fallback] {CSV.name} not found; using hardcoded numbers.")
        return ({k: {"alpha": _ALPHAS, "ndcg": v} for k, v in _FALLBACK.items()}, True)

    rows: dict[str, dict[str, list]] = {}
    with open(CSV, newline="", encoding="utf-8") as f:
        for r in sorted(csv.DictReader(f), key=lambda r: float(r["alpha"])):
            a = rows.setdefault(r["approach"], {"alpha": [], "ndcg": []})
            a["alpha"].append(float(r["alpha"]))
            a["ndcg"].append(float(r["ndcg@10"]))
    return rows, False


sweep, fallback = load_sweep()

plt.rcParams.update({"font.size": 9})
fig, ax = plt.subplots(figsize=(3.4, 2.7))

for key, (label, color, marker) in STYLE.items():
    if key not in sweep:
        continue
    d = sweep[key]
    ax.plot(
        d["alpha"],
        d["ndcg"],
        color=color,
        marker=marker,
        ms=4,
        lw=1.4,
        label=label,
        zorder=3,
    )

# At alpha = 1 every blend collapses to the pure TM; draw that value as a
# reference so "above the line" reads as "the blend beats the TM".
tm_ndcg = sweep["user-user"]["ndcg"][-1]
ax.axhline(tm_ndcg, color="gray", ls="--", lw=0.9, zorder=2, label="TM ($\\alpha=1$)")

ax.set_xlabel("Blend weight $\\alpha$")
ax.set_ylabel("NDCG@10")
ax.set_xlim(0, 1)
ax.set_ylim(bottom=0)
ax.legend(fontsize=6.5, loc="lower right", framealpha=0.9)
ax.grid(True, alpha=0.25)
if fallback:
    ax.set_title("fallback data — regenerate CSV", fontsize=7, color="gray")

fig.tight_layout()
for ext, kwargs in (("pdf", {}), ("png", {"dpi": 200})):
    out = OUT_STEM.with_suffix(f".{ext}")
    fig.savefig(out, bbox_inches="tight", **kwargs)
    print(f"wrote {out}")
