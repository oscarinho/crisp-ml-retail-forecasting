# Archive — exploratory work

The official lab lives at the **repo root** (`notebooks/Inventory_Forecasting_CRISPML.ipynb`).

This folder preserves earlier exploration that informed the final lab but isn't part of the
publishable narrative. Kept here so:

1. The work isn't lost
2. The repo root stays focused on a single story (the "two ceilings" insight)
3. Curious readers can dig into the broader context if they want

## What's inside

| Folder | Contents |
|---|---|
| `notebooks/` | Three additional labs: Sales (sales_data.csv), Food Demand (Genpact/Analytics Vidhya), Store Sales (Kaggle Corporación Favorita) + a Colab-ready bundle |
| `scripts/` | Batch runners for the archived labs + the cell upserter used to retrofit Tier 1/2 models into all notebooks |
| `data/` | Datasets used by the archived labs |
| `model/` | Trained artifacts for the archived labs |
| `app/` | Sales Streamlit app |
| `docs/` | Lessons docs (Spanish + business-adult level), LinkedIn post drafts, Windows handoff guide, validation reports, test cases, model coverage notes, reference reading material |
| `examples/` | CRISP-ML notebooks from other domains (HR Attrition, Predictive Maintenance, etc.) used as templates during the build |

## Why these were archived

- **Sales lab** — same dataset family as Inventory (5 stores × 20 products, retail), produced
  redundant insights. Kept for the per-category routing finding (MAE 19.5 via 5 LightGBMs)
  but not part of the headline story.
- **Food Demand lab** — different problem structure (weekly, hierarchical, no Demand Forecast
  column) but doesn't add to the "two ceilings" narrative. Best result: LightGBM Stage 2
  MAE 69.7, CatBoost RMSLE×100 = 49.1.
- **Store Sales lab** — 3M-row Kaggle dataset, per-family LightGBM wins (RMSLE 0.394). Validated
  that per-segment routing scales, but heavy compute and doesn't change the conclusion.
- **Docs** — LESSONS_LEARNED.md, EXPLICADO_*, LINKEDIN_POSTS.md cover all 4 labs; replaced
  in root by the single-lab EXPERIMENT_DF_RESIDUAL.md.

## How to revive an archived lab

Each archived notebook still runs. Move it back to `notebooks/`, move its `data/` and `model/`
subfolders back, then `pip install -r requirements.txt` and `jupyter lab`. The scripts are
self-contained — invoke them with `python archive/scripts/<name>.py`.
