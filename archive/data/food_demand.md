# Food Demand Forecasting Dataset

- **Source:** [Analytics Vidhya · "Genpact Machine Learning Hackathon (2018-19)"](https://datahack.analyticsvidhya.com/contest/genpact-machine-learning-hackathon-1/)
- **Author / Sponsor:** Genpact
- **License:** Free for academic / portfolio use (Analytics Vidhya terms)
- **Files:** `train.csv` (18 MB) · `test.csv` (1.2 MB) · `meal_info.csv` · `fulfilment_center_info.csv` · `sample_submission.csv`
- **Layout in repo:** `data/food_demand/`

## Shape

- **Train:** 456,548 rows × 9 cols · 2018-style weekly demand
- **Grain:** `(week, center_id, meal_id)` · weekly granularity
- **Weeks:** 1 → 145 (≈ 2.8 years) for train; test starts at week 146
- **Centers:** 77 fulfilment centers
- **Meals:** 51 meals across 14 categories × 4 cuisines

## Target

`num_orders` — count of orders placed in that `(week, center, meal)` cell.
Distribution is heavy-tailed (typical retail-demand power law): median ≈ 100, p99 ≈ 1,400, max ≈ 24,000.

## Schema

| File | Key columns |
|---|---|
| `train.csv` | id, week, center_id, meal_id, checkout_price, base_price, emailer_for_promotion, homepage_featured, num_orders |
| `test.csv` | same minus num_orders |
| `meal_info.csv` | meal_id, category, cuisine |
| `fulfilment_center_info.csv` | center_id, city_code, region_code, center_type, op_area |

## Why this dataset is in the portfolio

- **Real (non-synthetic)** retail forecasting — different from the Inventory/Sales labs which use synthetic generators
- **Multi-table join** — exercises the merge step of CRISP-ML(Q) Phase 3
- **Strong within-group autocorrelation** (median lag-1 ρ ≈ 0.5) — lags genuinely matter, unlike the Inventory lab
- **Long-tail target distribution** — good test case for RMSLE optimization (Kaggle metric of choice for retail demand)

## Best-result summary

See [`notebooks/Food_Demand_Forecasting_CRISPML.ipynb`](../notebooks/Food_Demand_Forecasting_CRISPML.ipynb) for the full pipeline. Headline holdout numbers (last 15% of weeks = holdout):

| Model | MAE | RMSLE×100 | Fit time |
|---|---:|---:|---:|
| Baseline: lag-1 | 99.9 | 70.2 | — |
| Stage 2: LightGBM | 69.7 | 52.8 | 11 s |
| **Stage 2: ExtraTrees** | **68.6** | 50.9 | 90 s |
| **Stage 2: CatBoost** | 69.4 | **49.1** | 16 s |
| Per-cuisine LightGBM (routed) | 72.3 | 53.7 | 35 s |

CatBoost is the deployable RMSLE winner. ExtraTrees marginally beats it on MAE but the pickle is 2.6 GB — disqualifying for deployment. **LightGBM is the production choice** — same accuracy band, 3 MB artifact.

Per-cuisine routing (the Sales-lab winner pattern) does NOT win here — only 4 cuisines makes per-cuisine sub-models too small to learn well.

## How to (re)download

```bash
# Visit https://datahack.analyticsvidhya.com/contest/genpact-machine-learning-hackathon-1/
# Register, accept terms, download train.csv + test.csv + meal_info.csv + fulfilment_center_info.csv
# Drop them into data/food_demand/
```
