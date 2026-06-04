"""Generate the 'Two Ceilings' hero chart for LinkedIn publication.

Run from repo root:
    python scripts/make_hero_chart.py

Output: assets/hero_two_ceilings.png (1.91:1, dark graphite editorial)
"""
from pathlib import Path

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "assets" / "hero_two_ceilings.png"

GOLD = "#8B7340"
CYAN = "#5BC0EB"
MIST = "#B0B4B8"
BG = "#1a1d22"

models_no_df = [
    "Stacking ensemble",
    "HGB Tier 1",
    "Stage 2 LightGBM",
    "CatBoost Tier 1",
    "LSTM 4ch / 60-step",
    "ARIMA auto",
    "ETS Holt-Winters",
    "Prophet per-group",
]
mae_no_df = [68.9, 69.0, 69.1, 69.1, 88.9, 89.1, 89.4, 112.0]

models_with_df = [
    "DF + HGB residual",
    "DF + RF residual",
    "DF + LightGBM residual",
    "DF puro (no model)",
]
mae_with_df = [7.43, 7.45, 7.46, 8.35]

fig, ax = plt.subplots(figsize=(12, 6.3), facecolor=BG)
ax.set_facecolor(BG)

y_top = list(range(len(models_no_df), 0, -1))
y_bot = list(range(-2, -2 - len(models_with_df), -1))

ax.barh(y_top, mae_no_df, color=GOLD, alpha=0.85, height=0.7)
ax.barh(y_bot, mae_with_df, color=CYAN, alpha=0.95, height=0.7)

for y, m, v in zip(y_top, models_no_df, mae_no_df):
    ax.text(v + 1.5, y, f"{v:.1f}", va="center", color=MIST,
            fontsize=10, family="monospace")
    ax.text(-2, y, m, va="center", ha="right", color=MIST,
            fontsize=10, family="monospace")

for y, m, v in zip(y_bot, models_with_df, mae_with_df):
    ax.text(v + 1.5, y, f"{v:.2f}", va="center", color=MIST,
            fontsize=10, family="monospace")
    ax.text(-2, y, m, va="center", ha="right", color=MIST,
            fontsize=10, family="monospace")

ax.axhline(-1, color="#404549", linewidth=0.5)
ax.text(60, -0.3, "NO DF (pure prediction) — ceiling MAE ≈ 69",
        color=GOLD, fontsize=11, family="monospace", weight="bold")
ax.text(60, -6.7, "DF as prior (residual learning) — ceiling MAE ≈ 7.4",
        color=CYAN, fontsize=11, family="monospace", weight="bold")

ax.set_xlim(-30, 130)
ax.set_ylim(-7.5, len(models_no_df) + 1)
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

ax.set_title(
    "Two Ceilings — same dataset, same models, different framing",
    color=MIST, fontsize=14, family="monospace", loc="left", pad=20,
)
ax.text(130, -8.0, "oscarponce.com",
        color="#606468", fontsize=9, family="monospace",
        ha="right", style="italic")

plt.tight_layout()
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PATH, dpi=200, facecolor=BG, bbox_inches="tight")
print(f"→ {OUT_PATH.relative_to(REPO_ROOT)}")
