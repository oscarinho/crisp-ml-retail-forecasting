# CRISP-ML(Q) Retail Demand Forecasting

Two sibling labs applying the **CRISP-ML(Q)** methodology to retail demand forecasting on two contrasting datasets. End-to-end: business understanding → leakage audits → time-aware feature engineering → model bake-off → P80 quantile newsvendor → deployment.

**Author:** [Oscar Ponce](https://oscarponce.com)
**Live demos:**
- Inventory app — *(deploy via Streamlit Cloud → see below)*
- Sales app — *(deploy via Streamlit Cloud → see below)*

---

## The two labs

| | **Inventory lab** | **Sales lab** |
|---|---|---|
| Notebook | [`notebooks/Inventory_Forecasting_CRISPML.ipynb`](notebooks/Inventory_Forecasting_CRISPML.ipynb) | [`notebooks/Sales_Forecasting_CRISPML.ipynb`](notebooks/Sales_Forecasting_CRISPML.ipynb) |
| Streamlit app | [`app/app.py`](app/app.py) | [`app/app_sales.py`](app/app_sales.py) |
| Dataset | `data/retail_store_inventory.csv` (synthetic, memoryless) | `data/sales_data.csv` (autocorrelated, censored by stockouts) |
| Rows × cols | 73,101 × 15 | 76,001 × 16 |
| Date range | 2021-12-31 → 2023-12-31 | 2022-01-01 → 2024-01-29 |
| Target | `Units Sold` | **`Demand`** (uncensored — `Units Sold ≤ Demand`) |
| Leakage trap | `Demand Forecast` column (ρ ≈ 0.997) — drop | `Units Sold` column (ρ ≈ 0.83) — drop unlagged |
| Within-group autocorr | ≈ 0 (no temporal memory) | ≈ 0.35 (modest but real) |
| Best holdout MAE | **69** (data ceiling; oracle MAE 8) | **19.5** (per-category routing) |
| Best model | Stacking ensemble | Per-category LightGBM (5 dedicated models) |

---

## Headline findings

### Inventory lab
- The dataset has zero within-group autocorrelation. Stacking, LSTM, Prophet, ARIMA, ETS all converge to MAE ~69. **MAE 69 is the noise ceiling, not a modeling failure.**
- The `Demand Forecast` column at ρ 0.997 is leakage (would not exist on real forecast day) — dropping it is the correct call.
- Stage 1 (contextual only) gets MAE 90; Stage 2 with lags gets MAE 69. Cold-start routing: ≥28 days history → Stage 2, otherwise Stage 1.

### Sales lab
- **More features hurt**: App-aligned (15 features, no lags) MAE 20.13 beats Stage 2 (29 features incl. lags) MAE 21.09. Confirmed via train-vs-holdout gap (22.7% vs 14.5%).
- **Early stopping does not rescue lag features** — best iteration 628 vs default 600; the model wanted more trees, not fewer. The overfit is structural, not a budget problem.
- **Log-transform target doesn't help here** — heteroscedasticity ratio 1.47 was misleading; tested and confirmed (MAE went up 0.2 units).
- **Per-category routing wins** (MAE 19.5, −3.2% vs global): the gains come from Clothing (−9.5%) and Electronics (−9.2%), *not* Groceries (which was the worst-performing category but barely improved at −0.7%).

---

---

## Datasets

Both datasets are public, synthetic Kaggle retail-forecasting CSVs. They share the same schema
family (5 stores × 20 products, same category & region taxonomies) but were generated under
different assumptions — see the dataset MDs in [`data/`](data/) for the full provenance.

### Inventory lab — `data/retail_store_inventory.csv`

- **Source:** [Kaggle · "Retail Store Inventory Forecasting Dataset"](https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset) by Anirudh Singh Chauhan
- **License:** CC0 (Public Domain)
- **Shape:** 73,101 rows × 15 columns · 2021-12-31 → 2023-12-31
- **Distinctive columns:** `Demand Forecast` (pre-computed signal, leakage with ρ ≈ 0.997 to
  `Units Sold` — dropped during prep), `Holiday/Promotion` (single combined flag)
- **Target:** `Units Sold` — censored by `Inventory Level` (stockouts cap observed sales)
- **EDA quirk:** within-group autocorrelation ≈ 0 — the data is memoryless by construction,
  which is why the headline MAE 69 is a *data ceiling*, not a modeling failure
- **Sample columns:** Date, Store ID, Product ID, Category, Region, Inventory Level, Units Sold,
  Units Ordered, Demand Forecast, Price, Discount, Weather Condition, Holiday/Promotion,
  Competitor Pricing, Seasonality

### Sales lab — `data/sales_data.csv`

- **Source:** [Kaggle · "Retail Store Inventory and Demand Forecasting"](https://www.kaggle.com/datasets/atomicd/retail-store-inventory-and-demand-forecasting) by WAVELET
- **License:** Apache-2.0
- **Shape:** 76,001 rows × 16 columns · 2022-01-01 → 2024-01-29
- **Mirror on Kaggle:** the same CSV is also republished by Ramin Huseyn as
  [`demand_forecasting.csv` (CC0)](https://www.kaggle.com/datasets/raminhuseyn/demand-forecasting-dataset)
  — byte-identical, just a different uploader / license. Either Kaggle page is a valid source.
- **Distinctive columns:** `Promotion` and `Epidemic` as separate binary flags · explicit `Demand`
  target (uncensored — `Units Sold ≤ Demand`, so this dataset preserves lost demand from
  stockouts), wider numeric ranges (Inventory up to 2,267 vs the inventory CSV's ~500, Price up
  to 228 vs ~100)
- **Target:** `Demand` — what customers wanted, not just what they got
- **EDA quirk:** within-group autocorrelation ≈ 0.35 — modest but exploitable, which is what
  unlocks the per-category LightGBM win (MAE 19.5)

### Compare at a glance

|  | Inventory CSV | Sales CSV |
|---|---|---|
| Promotion flag | combined `Holiday/Promotion` | separate `Promotion` |
| Epidemic flag | — | `Epidemic` ∈ {0, 1} |
| Demand signal | leaky `Demand Forecast` (drop) | clean `Demand` (target) |
| Stockout treatment | sales censored to inventory | demand recorded uncensored |
| Use it for | inventory coverage / reorder | demand under promo / epidemic / pricing |

---

## Repository structure

```
forecasting-inventory/
├── data/
│   ├── retail_store_inventory.csv   # Inventory lab dataset (~6 MB)
│   ├── retail_store_inventory.md    # Inventory dataset card (Kaggle attribution)
│   ├── sales_data.csv               # Sales lab dataset (~6 MB)
│   └── sales_data.md                # Sales dataset card (WAVELET / Apache-2.0)
├── notebooks/
│   ├── Inventory_Forecasting_CRISPML.ipynb
│   ├── Sales_Forecasting_CRISPML.ipynb
│   └── notebook_utils.py             # cached() helper for disk checkpointing
├── model/
│   ├── model.pkl                     # Inventory: Stage 2 LightGBM (full features)
│   ├── model_contextual.pkl          # Inventory: app-aligned (no lags)
│   ├── model_stage1.pkl              # Inventory: cold-start
│   ├── model_q80.pkl                 # Inventory: P80 quantile
│   ├── model_metadata.pkl
│   └── sales/
│       ├── model.pkl                 # Sales: Stage 2 LightGBM
│       ├── model_contextual.pkl      # Sales: app-aligned (fallback)
│       ├── model_stage1.pkl          # Sales: cold-start
│       ├── model_per_category.pkl    # Sales: per-category dict (5 models) ← winner
│       ├── model_q80.pkl             # Sales: P80 quantile
│       └── model_metadata.pkl
├── app/
│   ├── app.py                        # Inventory Streamlit app
│   └── app_sales.py                  # Sales Streamlit app
├── examples/                         # Reference notebooks (not run; read-only)
├── CLAUDE.md                         # Project-specific dev instructions
├── LINKEDIN_POSTS.md                 # 10-post content plan
└── requirements.txt
```

`notebooks/checkpoints/` is intentionally gitignored — local model-fit cache (210 MB RF, etc.).

---

## Local setup

```bash
# Python 3.11 recommended (matches the notebook env). On macOS, LightGBM needs libomp:
# brew install libomp                # only required for pip-only installs; conda envs bundle llvm-openmp.

pip install -r requirements.txt

# Run a notebook
jupyter lab notebooks/Sales_Forecasting_CRISPML.ipynb

# Or run an app locally
streamlit run app/app.py             # Inventory
streamlit run app/app_sales.py       # Sales
```

First notebook run trains every model (~5 min). Subsequent runs use `notebooks/checkpoints/` and complete in seconds.

---

## Deploy to Streamlit Community Cloud (free)

1. Go to <https://share.streamlit.io>, sign in with GitHub.
2. Click **New app** → **From existing repo**.
3. Repo: `oscarinho/crisp-ml-retail-forecasting`. Branch: `main`.
4. **For the Inventory app:** Main file path = `app/app.py`. App URL suffix e.g. `inventory-forecasting`.
5. **Deploy a second app** for Sales: Main file path = `app/app_sales.py`. App URL suffix e.g. `sales-forecasting`.

Both apps load their respective `model/*.pkl` and `data/*.csv` directly from the repo — no external services required.

If you hit a `libomp.dylib` error on the cloud builder, add a `packages.txt` to the repo root containing `libgomp1`.

---

## CRISP-ML(Q) methodology

Both notebooks follow the same 6-phase template:

```
Phase 1 — Business Understanding   (target, grain, aspirational metric, cost structure)
Phase 2 — Data Understanding       (EDA, autocorrelation, leakage diagnostic, stockout audit)
Phase 3 — Data Preparation         (lag/rolling features, time-split, ColumnTransformer)
Phase 4 — Modeling                 (baselines first, then ML; per-category if motivated)
Phase 5 — Evaluation               (holdout MAE/RMSE/sMAPE, per-group, residuals)
Phase 6 — Deployment               (joblib bundle + metadata + app integration)
```

Quality gates (the "Q") are inline `assert` statements: no same-day leakage, lag grouping correctness, train < holdout date.

See [`CLAUDE.md`](CLAUDE.md) for the project conventions used while building this.

---

## License & attribution

Datasets are public, synthetic Kaggle retail-forecasting datasets — see the [Datasets section](#datasets)
above for per-CSV attribution, Kaggle URLs, and licenses (CC0 / Apache-2.0). Code is provided
as-is for portfolio purposes.
