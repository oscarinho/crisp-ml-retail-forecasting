# CRISP-ML(Q) Retail Demand Forecasting

**Four labs** applying the **CRISP-ML(Q)** methodology to retail demand forecasting across contrasting datasets. End-to-end: business understanding → leakage audits → time-aware feature engineering → Tier 1/2 model bake-off → per-category routing → champion-challenger backtesting → deployment.

**Author:** [Oscar Ponce](https://oscarponce.com)
**Live demos:**
- Inventory app — *(deploy via Streamlit Cloud → see below)*
- Sales app — *(deploy via Streamlit Cloud → see below)*

---

## The four labs

| | **Inventory** | **Sales** | **Food Demand** | **Store Sales** |
|---|---|---|---|---|
| Notebook | [`Inventory_Forecasting_CRISPML.ipynb`](notebooks/Inventory_Forecasting_CRISPML.ipynb) | [`Sales_Forecasting_CRISPML.ipynb`](notebooks/Sales_Forecasting_CRISPML.ipynb) | [`Food_Demand_Forecasting_CRISPML.ipynb`](notebooks/Food_Demand_Forecasting_CRISPML.ipynb) | [`Store_Sales_Forecasting_CRISPML.ipynb`](notebooks/Store_Sales_Forecasting_CRISPML.ipynb) |
| Streamlit app | [`app/app.py`](app/app.py) | [`app/app_sales.py`](app/app_sales.py) | — (TODO) | — (TODO) |
| Dataset | `data/retail_store_inventory.csv` (synthetic) | `data/sales_data.csv` (synthetic) | `data/food_demand/` (Genpact / Analytics Vidhya) | `data/store-sales-time-series-forecasting/` (Kaggle / Corporación Favorita) |
| Rows | 73k | 76k | **457k** | **3,000k** |
| Grain | daily | daily | weekly | daily |
| Series | 100 (5 store × 20 prod) | 100 | 3,927 (77 ctr × 51 meals) | 1,782 (54 store × 33 fam) |
| Date range | 2021-12 → 2023-12 | 2022-01 → 2024-01 | 145 weeks | 2013-01 → 2017-08 |
| Target | `Units Sold` | `Demand` (uncensored) | `num_orders` | `sales` |
| Leakage trap | `Demand Forecast` (ρ≈0.997) | `Units Sold` (ρ≈0.83 unlagged) | none | none |
| Within-group autocorr | ≈ 0 | ≈ 0.35 | ≈ 0.5 | ≈ 0.6-0.8 |
| Best holdout MAE — no DF | **69** (data ceiling) | **19.5** | **68.6** (ExtraTrees) | **52.0** (per-family LGBM) |
| Best holdout MAE — DF as prior | **7.4** (residual: DF + HGB) | n/a | n/a | n/a |
| Best Kaggle-style metric | n/a | n/a | **RMSLE×100 = 49.1** (CatBoost) | **RMSLE = 0.394** (per-family LGBM) |
| Best model | Stacking ensemble (no-DF) · HGB residual (DF-prior) | Per-category LightGBM (5) | LightGBM Stage 2 (deployable winner) · CatBoost on RMSLE | **Per-family LightGBM (33)** |

---

## Headline findings

### Inventory lab
- The dataset has zero within-group autocorrelation. Stacking, LSTM, Prophet, ARIMA, ETS, CatBoost, HGB, NHITS, TFT all converge to MAE ~69. **MAE 69 is the *no-DF* noise ceiling, not a modeling failure.**
- **Two ceilings, not one.** A residual-learning experiment (§4.13 / [`EXPERIMENT_DF_RESIDUAL.md`](EXPERIMENT_DF_RESIDUAL.md)) shows that *if* `Demand Forecast` is available at inference time (which it is in real planning systems), the deployable ceiling is **MAE ≈ 7.4** via `pred = DF + HGB(features)`. The 62-unit gap between the two ceilings is **environmental, not modeling skill**.
- `Demand Forecast` at ρ 0.997 is a leakage trap when used as a direct feature, but **legitimate as a prior** in a residual-learning frame.
- Stage 1 (contextual only) gets MAE 90; Stage 2 with lags gets MAE 69. Cold-start routing: ≥28 days history → Stage 2, otherwise Stage 1.
- 18-window champion-challenger backtest (§4.14): HGB_residual wins 17/18 windows. **Framing > model choice** — Direct vs Residual changes MAE by 62 units; choice of algorithm (HGB/RF/LGBM) changes MAE by ~0.04.

### Sales lab
- **More features hurt**: App-aligned (15 features, no lags) MAE 20.13 beats Stage 2 (29 features incl. lags) MAE 21.09. Confirmed via train-vs-holdout gap (22.7% vs 14.5%).
- **Early stopping does not rescue lag features** — best iteration 628 vs default 600; the model wanted more trees, not fewer. The overfit is structural, not a budget problem.
- **Log-transform target doesn't help here** — heteroscedasticity ratio 1.47 was misleading; tested and confirmed (MAE went up 0.2 units).
- **Per-category routing wins** (MAE 19.5, −3.2% vs global): the gains come from Clothing (−9.5%) and Electronics (−9.2%), *not* Groceries (which was the worst-performing category but barely improved at −0.7%).

### Food Demand lab
- **CatBoost wins on RMSLE×100=49.1** — driven by native categorical handling (cuisine, category, center_type, city_code, region_code all have ≥10 levels)
- **ExtraTrees wins on MAE=68.6** — strong on this tabular structure with high-cardinality cats
- **Per-cuisine routing does NOT win here** (MAE 72.3 vs LightGBM 69.7) — opposite of the Sales lab. Only 4 cuisines and some have very few meals → per-cuisine models overfit
- Stage 1 (no lags) MAE 92.8 vs Stage 2 with lags MAE 69.7 → **lags add ~25% lift** (autocorr ≈ 0.5 was real)
- Naive baselines bottom out at MAE 100 — Stage 2 cuts that by 30%, CatBoost cuts RMSLE by 25%

### Store Sales lab (Corporación Favorita / Kaggle)
- 3M rows, 4.5 years, 1,782 series (54 stores × 33 families)
- External regressors merged: oil price (ffill weekends), daily transactions per store, holiday flags
- **Per-family LightGBM wins** (RMSLE 0.394, MAE 52.04) — confirms the Sales lab pattern scales to 3M rows × 33 families
- CatBoost second on RMSLE (0.400), competitive on accuracy but 3× slower
- Stage 1 (no lags) RMSLE 1.005 vs Stage 2 RMSLE 0.516 → **lags carry 50% of the signal** (autocorr ≈ 0.6-0.8 was real)
- Naive lag-7 baseline RMSLE 0.545 — surprisingly close to Stage 2 LGBM single-model (0.516); the per-family routing is what unlocks the next 30% gain

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
│   ├── retail_store_inventory.csv         # Inventory lab dataset (~6 MB)
│   ├── retail_store_inventory.md          # Inventory dataset card (Kaggle attribution)
│   ├── sales_data.csv                     # Sales lab dataset (~6 MB)
│   ├── sales_data.md                      # Sales dataset card (WAVELET / Apache-2.0)
│   ├── food_demand/                       # Genpact food demand challenge (3 CSVs)
│   └── store-sales-time-series-forecasting/  # Kaggle Corporación Favorita (6 CSVs)
├── notebooks/
│   ├── Inventory_Forecasting_CRISPML.ipynb
│   ├── Sales_Forecasting_CRISPML.ipynb
│   ├── Food_Demand_Forecasting_CRISPML.ipynb       # Lab 3 — weekly food demand
│   ├── Store_Sales_Forecasting_CRISPML.ipynb       # Lab 4 — Kaggle daily grocery
│   └── notebook_utils.py                  # cached() helper for disk checkpointing
├── model/
│   ├── model.pkl                          # Inventory: Stage 2 LightGBM (full features)
│   ├── model_contextual.pkl               # Inventory: app-aligned (no lags)
│   ├── model_stage1.pkl                   # Inventory: cold-start
│   ├── model_q80.pkl                      # Inventory: P80 quantile
│   ├── model_metadata.pkl
│   ├── sales/                             # Sales lab artifacts (5 pkls)
│   ├── food_demand/                       # Food Demand lab artifacts
│   │   ├── best_model.pkl                 # CatBoost (RMSLE winner)
│   │   └── model_metadata.pkl
│   └── store_sales/                       # Store Sales lab artifacts
├── app/
│   ├── app.py                        # Inventory Streamlit app
│   └── app_sales.py                  # Sales Streamlit app
├── examples/                         # Reference notebooks (not run; read-only)
├── CLAUDE.md                         # Project-specific dev instructions
├── LINKEDIN_POSTS.md                 # 10-post content plan
├── EXPERIMENT_DF_RESIDUAL.md         # Two-ceiling analysis (Residual + Champion-Challenger)
├── TEST_CASES.md                     # QA scenarios with exact expected predictions
├── scripts/
│   ├── add_tier12_cells.py           # Idempotent Tier-1/2 + residual cell upserter
│   ├── df_experiments.py             # Residual + Champion-Challenger backtests (Inventory)
│   ├── df_experiments_results.json   # Cached experiment outputs (Inventory)
│   ├── gen_test_cases.py             # Regenerates TEST_CASES.md predictions
│   ├── generate_new_notebooks.py     # Produces Food + Store Sales notebooks
│   ├── run_food_demand.py            # Batch experiment runner for Food Demand
│   ├── food_demand_results.json      # Cached Food Demand results
│   ├── run_store_sales.py            # Batch experiment runner for Store Sales
│   └── store_sales_results.json      # Cached Store Sales results
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
