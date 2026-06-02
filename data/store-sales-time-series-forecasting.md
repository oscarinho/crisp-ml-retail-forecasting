# Store Sales — Time Series Forecasting

- **Source:** [Kaggle · Store Sales — Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)
- **Author / Sponsor:** Corporación Favorita (Ecuador's largest grocery retailer)
- **License:** Kaggle competition terms (educational / portfolio use OK)
- **Files:** `train.csv` (116 MB, **gitignored** — see below) · `test.csv` · `stores.csv` · `transactions.csv` · `oil.csv` · `holidays_events.csv` · `sample_submission.csv`
- **Layout in repo:** `data/store-sales-time-series-forecasting/`

## Shape

- **Train:** 3,000,888 rows × 6 cols · daily granularity
- **Grain:** `(date, store_nbr, family)`
- **Date range:** 2013-01-01 → 2017-08-15 (1,684 days)
- **Stores:** 54 (across 22 cities in Ecuador)
- **Families:** 33 product families (AUTOMOTIVE, BEAUTY, BREAD/BAKERY, …)
- **Series:** 1,782 total (54 × 33)

## ⚠ train.csv is NOT in this repo

The Kaggle `train.csv` is **116 MB** — exceeds GitHub's 100 MB per-file limit. It is gitignored. To run the notebook:

```bash
# Option 1 — Kaggle CLI (recommended)
pip install kaggle
# Place your kaggle.json at ~/.kaggle/kaggle.json (chmod 600)
kaggle competitions download -c store-sales-time-series-forecasting -p data/store-sales-time-series-forecasting
unzip data/store-sales-time-series-forecasting/store-sales-time-series-forecasting.zip -d data/store-sales-time-series-forecasting

# Option 2 — manual
# Visit https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data
# Download all CSVs into data/store-sales-time-series-forecasting/
```

Everything else (transactions, oil, holidays, stores, test, sample_submission) IS in the repo since they're small.

## Target

`sales` (float) — units sold for that `(date, store_nbr, family)` cell.
Distribution is **heavily zero-inflated** (~21% exact zeros, many small values). Kaggle's official metric is **RMSLE** which handles this well.

## External regressors

The competition gives unusually rich exogenous data:

| File | What it provides | Merge key |
|---|---|---|
| `stores.csv` | city, state, cluster (k-means group), type (A/B/C/D/E) | `store_nbr` |
| `transactions.csv` | daily transaction count per store | `(date, store_nbr)` |
| `oil.csv` | daily WTI oil price (Ecuador's economy is oil-dependent) | `date` (ffill weekends — oil doesn't trade) |
| `holidays_events.csv` | national/regional/local holidays with `transferred` flag | `date` (filter `transferred==False`, `locale=='National'`) |

## Why this dataset is in the portfolio

- **Real (non-synthetic)** and **large** — 3M rows tests whether our toolkit scales
- **Multi-source feature engineering** — joins, ffill, holiday handling
- **Kaggle benchmark exists** — leaderboard top sits at RMSLE ≈ 0.36-0.42, so our number is comparable
- **Per-family routing pattern** scales — 33 families × ~100k rows each is a clean stress test

## Best-result summary

See [`notebooks/Store_Sales_Forecasting_CRISPML.ipynb`](../notebooks/Store_Sales_Forecasting_CRISPML.ipynb). Headline holdout numbers (last 30 days = holdout):

| Model | MAE | RMSLE | Fit time |
|---|---:|---:|---:|
| Baseline: lag-7 (week-over-week) | 85.0 | 0.5448 | — |
| Baseline: rolling-28 | 98.5 | 0.4581 | — |
| Stage 1: LightGBM (no lags) | 67.3 | 1.0045 | 25 s |
| Stage 2: LightGBM | 53.2 | 0.5161 | 40 s |
| Stage 2: HistGradientBoosting | 54.1 | 0.5263 | 123 s |
| Stage 2: CatBoost | 63.0 | 0.3998 | 124 s |
| **Per-family LightGBM (33 models)** | **52.0** | **0.3935** | 192 s |

Per-family routing wins on both MAE and RMSLE — confirms the Sales-lab pattern scales to 3M rows × 33 families.

For Kaggle reference, the public LB top sits around RMSLE 0.36-0.42 → our **0.394 lands in the top tier** without competition-specific engineering hacks (no pseudo-labeling, no target encoding, no aggressive blending).
