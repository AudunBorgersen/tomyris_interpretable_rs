"""
Dot matrix plot illustrating clause voting for a single class.

Columns  = selected clauses (top N by weight FOR, top N by weight AGAINST).
Rows     = movies that appear as 'likes' literals in those clauses.
A dot at (clause, movie) means that movie is a literal of that clause.
Blue columns = clauses voting FOR the target; red = AGAINST.

Usage:
    python figures/clause_voting.py
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from datasets import MovieLens1M
from utils.checkpoint import load_tm_cpu
from utils.clause_analysis import clause_pattern

# ── Configuration ─────────────────────────────────────────────────────────────
RUN_DIR = PROJECT_ROOT / "runs/coalesced_baseline/Movielens_final_runs/20260721T130000_movielens4"
MOVIE_COL = 1090  # column index of the target class
N_TOP = 3  # number of FOR and AGAINST clauses to show
EPOCH_INDEX = -1
IGNORE_SELF_INCLUSION = (
    True  # skip clauses including the target's own literal (inert at test time)
)
OUT_FILE = PROJECT_ROOT / "figures" / "clauses" / "clause_voting.png"
# ─────────────────────────────────────────────────────────────────────────────

# ── Load ──────────────────────────────────────────────────────────────────────
with open(RUN_DIR / "metadata.json") as f:
    cfg = SimpleNamespace(**json.load(f)["config"])

checkpoints = sorted(RUN_DIR.glob("epoch_*.pkl"))
tm = load_tm_cpu(checkpoints[EPOCH_INDEX])

ds = MovieLens1M(
    rating_threshold=cfg.rating_threshold,
    test_ratio=0.2,
    min_ratings=20,
    random_state=42,
    max_items=cfg.max_classes,
)
data = ds.get()
col_map = ds.col_to_movie_id()

# ── Select clauses ────────────────────────────────────────────────────────────
movie_title = ds.movie_id_to_title(col_map[MOVIE_COL]) or f"col {MOVIE_COL}"
weights = tm.get_weights(MOVIE_COL)
candidates = np.arange(tm.clause_bank.number_of_clauses)

if IGNORE_SELF_INCLUSION:
    self_included = np.array([bool(tm.get_ta_action(c, MOVIE_COL)) for c in candidates])
    candidates = candidates[~self_included]
    print(
        f"Ignoring {int(self_included.sum())} self-including clauses for col {MOVIE_COL}"
    )

for_idx = candidates[np.argsort(-weights[candidates])[:N_TOP]]
against_idx = candidates[np.argsort(weights[candidates])[:N_TOP]]

# ── Build long-form dataframe ─────────────────────────────────────────────────
rows = []
for c in for_idx:
    likes, _ = clause_pattern(tm, int(c), col_map, ds, max_per_polarity=999)
    w = int(weights[c])
    label = f"Clause {c}\n(w=+{w})"
    for movie in likes:
        rows.append({"clause": label, "movie": movie, "polarity": "FOR", "weight": w})

for c in against_idx:
    likes, _ = clause_pattern(tm, int(c), col_map, ds, max_per_polarity=999)
    w = int(weights[c])
    label = f"Clause {c}\n(w={w})"
    for movie in likes:
        rows.append(
            {"clause": label, "movie": movie, "polarity": "AGAINST", "weight": w}
        )

df = pd.DataFrame(rows)

# ── Axis ordering ─────────────────────────────────────────────────────────────
# X: FOR clauses (strongest first) then AGAINST (most negative first)
clause_order = [f"Clause {c}\n(w=+{int(weights[c])})" for c in for_idx] + [
    f"Clause {c}\n(w={int(weights[c])})" for c in against_idx
]

# Y: movies sorted by number of clause appearances (descending), then alpha
movie_counts = df.groupby("movie")["clause"].nunique().sort_values(ascending=False)
movie_order = movie_counts.index.tolist()

# Numeric positions
cx = {c: i for i, c in enumerate(clause_order)}
my = {m: i for i, m in enumerate(movie_order)}
df["x"] = df["clause"].map(cx)
df["y"] = df["movie"].map(my)

# ── Plot ──────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.0)

n_clauses = len(clause_order)
n_movies = len(movie_order)
fig_w = max(10, n_clauses * 1.6 + 3)
fig_h = max(6, n_movies * 0.45 + 2)

fig, ax = plt.subplots(figsize=(fig_w, fig_h))

palette = {"FOR": "#4878CF", "AGAINST": "#D65F5F"}
for polarity, grp in df.groupby("polarity"):
    ax.scatter(
        grp["x"],
        grp["y"],
        c=palette[polarity],
        s=160,
        zorder=3,
        label=polarity,
        alpha=0.85,
        linewidths=0,
    )

# Vertical separator between FOR and AGAINST groups
sep = N_TOP - 0.5
ax.axvline(sep, color="#888", linewidth=1.2, linestyle="--", zorder=2)
ax.text(
    sep - 0.1,
    -0.9,
    "FOR →",
    ha="right",
    va="bottom",
    fontsize=9,
    color="#4878CF",
    fontweight="bold",
    transform=ax.get_xaxis_transform(),
)
ax.text(
    sep + 0.1,
    -0.9,
    "← AGAINST",
    ha="left",
    va="bottom",
    fontsize=9,
    color="#D65F5F",
    fontweight="bold",
    transform=ax.get_xaxis_transform(),
)

# Colour x-tick labels to match polarity
ax.set_xticks(range(n_clauses))
ax.set_xticklabels(clause_order, rotation=45, ha="right", fontsize=8)
for i, tick in enumerate(ax.get_xticklabels()):
    tick.set_color("#4878CF" if i < N_TOP else "#D65F5F")

ax.set_yticks(range(n_movies))
ax.set_yticklabels(movie_order, fontsize=9)

ax.set_xlim(-0.5, n_clauses - 0.5)
ax.set_ylim(-0.5, n_movies - 0.5)
ax.set_xlabel("Clause", labelpad=8)
ax.set_ylabel("Movie (literal)", labelpad=8)
ax.set_title(f"Clause voting — {movie_title}", fontsize=14, fontweight="bold", pad=12)
# ax.legend(title="Clause polarity", loc="upper right", framealpha=0.9)

ax.yaxis.set_minor_locator(ticker.NullLocator())
ax.grid(axis="x", visible=False)
ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)

plt.tight_layout()
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_FILE, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved → {OUT_FILE}")
