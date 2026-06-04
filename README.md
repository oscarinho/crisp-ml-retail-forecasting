# Two Ceilings: Forecast MAE 69 vs 7.4

**A CRISP-ML(Q) study on retail demand forecasting.** The headline finding: most forecasting
notebooks chase the wrong ceiling. When the production environment already publishes a
`Demand Forecast` (every ERP / S&OP system does), using it as a **residual prior** instead
of a baseline to beat takes MAE from 69 to 7.4 on the same dataset, with the same model
families, on the same train/test split.

The framing decision is **10× more impactful** than the model decision.

**Author:** [Oscar Andres Ponce](https://oscarponce.com)
**Notebook (EN):** [`notebooks/Inventory_Forecasting_CRISPML.ipynb`](notebooks/Inventory_Forecasting_CRISPML.ipynb)
**Notebook (ES):** [`notebooks/Inventory_Forecasting_CRISPML_ES.ipynb`](notebooks/Inventory_Forecasting_CRISPML_ES.ipynb) — misma estructura, narrativa traducida
**Streamlit app:** [`app/app.py`](app/app.py) · [inventory-lab.streamlit.app](https://inventory-lab.streamlit.app)

---

## The headline finding

This dataset has **two distinct MAE ceilings**, not one:

| Regime | Best MAE | What it means |
|---|---:|---|
| **No DF available** — model only | **~69** | Real noise floor when forecasting from scratch. Validated by 10+ model families: LightGBM, RandomForest, Stacking, Prophet, ARIMA, ETS, LSTM, CatBoost, HistGradientBoosting, ExtraTrees. |
| **DF available** — residual / DF as prior | **~7.4** | What you actually get to deploy when a planning system already publishes a forecast. The model corrects ~5 units of systematic bias. |

The 62-unit gap (90% MAE reduction) is **environmental, not modeling skill**. It comes from
how the existing forecast enters the pipeline:

- ❌ As a feature → leakage (ρ=0.997 with target) → drop it
- ✅ As a prior the model corrects → `pred = DF + model(features)` → MAE drops to 7.4

Once you cross into the residual regime, the model family barely matters:

| Strategy | MAE | Bias |
|---|---:|---:|
| DF puro (no model) | 8.35 | +5.05 (systematic overshoot) |
| Stage 2 LightGBM (no DF anywhere) | 69.1 | ~−1 |
| **Residual: DF + HGB(features)** | **7.43** | **+0.10** ← 50× bias reduction |
| Residual: DF + RandomForest(features) | 7.45 | +0.13 |
| Residual: DF + LightGBM(features) | 7.46 | +0.15 |

An 18-window champion-challenger backtest (§4.14) shows **HGB residual wins 17 of 18
windows**. Direct vs Residual changes MAE by 62 units; choice of algorithm (HGB / RF / LightGBM)
changes MAE by 0.04.

→ **Full analysis:** [`EXPERIMENT_DF_RESIDUAL.md`](EXPERIMENT_DF_RESIDUAL.md)

---

## Why this matters in production

The standard ML workflow treats `Demand Forecast` as a leakage trap and drops it. That's
correct when you're benchmarking pure-prediction skill — but it's the wrong framing for
deployment.

In any real S&OP / MRP system:

- The Demand Forecast is **published 1+ weeks before** the prediction window
- It's a known input at inference time, not leaked future information
- Demand planners read it first; the model should *correct* it, not pretend it doesn't exist

This is the bridge from **portfolio ML** to **production forecasting**: the model's job is
to learn the **structured error** of the existing system (which knows seasonality, holidays,
trend) without re-deriving the easy parts from scratch.

---

## Dataset

[Kaggle · "Retail Store Inventory Forecasting Dataset"](https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset)
by Anirudh Singh Chauhan · License: CC0 · See [`data/retail_store_inventory.md`](data/retail_store_inventory.md)

- **Shape:** 73,101 rows × 15 columns · 2021-12-31 → 2023-12-31
- **Series:** 100 (5 stores × 20 products) · **grain:** daily
- **Target:** `Units Sold` — censored by `Inventory Level` (stockouts cap observed sales)
- **Distinctive column:** `Demand Forecast` (the existing system's prediction, the focus of
  this notebook's headline insight)
- **EDA quirk:** within-group autocorrelation ≈ 0 — the data is memoryless by construction,
  which is why the headline MAE 69 (no-DF) is a **data ceiling**, not a modeling failure

---

## CRISP-ML(Q) structure

```
Phase 1 — Business Understanding   target, grain, aspirational metric, cost structure
Phase 2 — Data Understanding       EDA, autocorrelation, leakage diagnostic, stockout audit
Phase 3 — Data Preparation         lag/rolling features, time-split, ColumnTransformer
Phase 4 — Modeling                 baselines → classical → ML → Tier 1 → residual + backtest
Phase 5 — Evaluation               holdout MAE / RMSE / sMAPE, per-group, residuals
Phase 6 — Deployment               joblib bundle + metadata + Streamlit app
```

Quality gates (the "Q") are inline `assert` statements: no same-day leakage, lag grouping
correctness, train < holdout date.

Phase 4 progression:

```
4.1   Naive baselines              (lag-1, lag-7, mean)
4.2   Stage 1 — Structural model   (contextual only → DF target; cold-start fallback)
4.3   Stage 2 — Sales realization  (full features incl. lags → Units Sold)
4.4   Ensemble + 4.10 Stacking
4.5   Prophet · 4.6 ARIMA · 4.7 ETS
4.8   Multivariate LSTM
4.9   Quantile P80                 (inventory decision model)
4.10  Tier 1                       CatBoost · HGB · ExtraTrees · SARIMAX · AutoTheta
4.12  Intermediate leaderboard     (no DF — all models cluster ~MAE 69)
4.13  Residual learning            (DF as prior → MAE drops to ~7.4)
4.14  Champion-Challenger backtest (18 rolling windows; framing > model choice)
4.15  Final leaderboard            (two-ceilings table)
```

---

## Repository structure

```
forecasting-inventory/
├── notebooks/
│   ├── Inventory_Forecasting_CRISPML.ipynb     # the lab (EN) — 96 cells, 6 CRISP-ML phases
│   ├── Inventory_Forecasting_CRISPML_ES.ipynb  # Spanish edition (same code, translated narrative)
│   └── notebook_utils.py                       # cached() helper for disk checkpointing
├── data/
│   ├── retail_store_inventory.csv            # Kaggle dataset (~6 MB)
│   └── retail_store_inventory.md             # Dataset card (provenance, EDA quirks)
├── model/
│   ├── model.pkl                             # Stage 2 LightGBM (full features)
│   ├── model_contextual.pkl                  # App-aligned (no lags)
│   ├── model_stage1.pkl                      # Cold-start fallback
│   ├── model_q80.pkl                         # P80 quantile for inventory decisions
│   └── model_metadata.pkl                    # Metrics, threshold, feature columns
├── app/
│   └── app.py                                # Streamlit inference UI
├── scripts/
│   ├── df_experiments.py                     # Residual + Champion-Challenger backtest runner
│   └── df_experiments_results.json           # Cached experiment outputs
├── EXPERIMENT_DF_RESIDUAL.md                 # Full analytical write-up of the headline finding
├── CLAUDE.md                                 # Project-specific dev instructions
├── requirements.txt
└── archive/                                  # Earlier exploration (3 additional labs, drafts,
                                              # lessons docs) — not part of the published story
```

`notebooks/checkpoints/` is gitignored (local model-fit cache, ~210 MB).

The `archive/` folder holds three additional labs (Sales, Food Demand, Store Sales) that
informed the build but aren't part of the published narrative — see
[`archive/README.md`](archive/README.md) for what's inside and why each was demoted.

---

## Local setup

```bash
# Python 3.11 recommended. On macOS, LightGBM needs libomp:
# brew install libomp   (only required for pip-only installs; conda envs bundle llvm-openmp)

pip install -r requirements.txt

# Run the notebook
jupyter lab notebooks/Inventory_Forecasting_CRISPML.ipynb

# Or run the Streamlit app
streamlit run app/app.py
```

First notebook run trains every model (~5 min). Subsequent runs use `notebooks/checkpoints/`
and complete in seconds.

To regenerate the residual / champion-challenger results from shell (no Jupyter required):

```bash
python scripts/df_experiments.py   # ~5 min, writes scripts/df_experiments_results.json
```

---

## Deploy to Streamlit Community Cloud (free)

1. Go to <https://share.streamlit.io>, sign in with GitHub
2. **New app → From existing repo** → branch `main`, main file `app/app.py`
3. The app loads `model/*.pkl` and `data/retail_store_inventory.csv` directly from the repo — no external services required

If you hit a `libomp.dylib` error on the cloud builder, add a `packages.txt` to the repo root containing `libgomp1`.

---

## License & attribution

Dataset is public, CC0 (Public Domain). See [`data/retail_store_inventory.md`](data/retail_store_inventory.md)
for full attribution. Code is provided as-is for portfolio purposes.
