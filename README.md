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

## The headline finding (in this dataset, under this setup)

On the Kaggle retail dataset used here, the experiment surfaces **two distinct MAE plateaus** —
an observation, not a universal law:

| Regime | MAE on this holdout | What was observed |
|---|---:|---|
| **No DF available** — model only | **~69** | Best result achieved by the multivariate ML methods (LightGBM, RandomForest, Stacking, CatBoost, HistGradientBoosting, ExtraTrees) that use the `Inventory Level` feature. Univariate time-series methods (ARIMA, ETS, LSTM, Prophet, AutoTheta) cannot use that feature and topped out near the mean baseline (~89), with Prophet at MAE 112. |
| **DF available** — residual / DF as prior | **~7.4** | Achieved by training on `Units Sold − Demand Forecast` and adding DF back at inference. The model contributes mostly bias correction (~5 unit overshoot in DF → ~0.1 in the residual prediction). |

On this dataset, the 62-unit gap (90% MAE reduction) is **largely a function of how the existing
forecast enters the pipeline**, not of model choice within either regime:

- ❌ DF as a feature → leakage (ρ=0.997 with target on this dataset) → drop it
- ✅ DF as a prior the model corrects → `pred = DF + model(features)` → MAE drops to 7.4

Within the residual regime *on this holdout*, the three algorithms tested cluster within
0.05 MAE of each other:

| Strategy | MAE | Bias |
|---|---:|---:|
| DF puro (no model) | 8.35 | +5.05 (systematic overshoot) |
| Stage 2 LightGBM (no DF anywhere) | 69.1 | ~−1 |
| **Residual: DF + HGB(features)** | **7.43** | **+0.10** |
| Residual: DF + RandomForest(features) | 7.45 | +0.13 |
| Residual: DF + LightGBM(features) | 7.46 | +0.15 |

An 18-window champion-challenger backtest (§4.14) shows **HGB residual winning 17 of 18 windows**
on this dataset. The headline contrast — **62 MAE units between regimes vs. 0.04 MAE between
algorithms within the residual regime** — is what the rest of this README and notebook unpack.

→ **Full analysis:** [`EXPERIMENT_DF_RESIDUAL.md`](EXPERIMENT_DF_RESIDUAL.md)

---

## What this experiment does and does not show

**What it shows (measured here):**

- On this synthetic Kaggle dataset, the residual-learning framing produces a much lower
  holdout MAE than training a model to predict `Units Sold` directly.
- Within the residual regime on this holdout, three tree-based algorithms converge to
  nearly identical MAE.
- DF puro on this dataset has a +5.05 unit systematic overshoot that residual learning
  reduces to +0.10.

**What it does not show (and what to be careful about):**

- This is a **synthetic dataset**. `Demand Forecast` here is a near-oracle column
  (ρ=0.997 with the target). In a real production setting, an existing planner's forecast
  typically has MAE 30–60 against actuals — closer correlation 0.7–0.85 with the target.
  The residual gain on real data is usually smaller in absolute terms (and includes
  variance reduction, not just bias correction) but the structural argument still holds.
- These ceilings are **specific to the feature set, the time split, and the noise structure
  of this dataset**. They are not universal claims about retail forecasting.
- The "framing matters more than model choice" observation is true *in this experiment, on
  this dataset, within the feature/split regime tested* — not a general law of ML.

## Why this still matters in practice

Many production S&OP / MRP systems do publish a Demand Forecast in advance of the
prediction window, in which case treating it as a known input (not as leaked future data)
is a defensible engineering choice. The design pattern this notebook illustrates —
**learn the residual of the existing forecast instead of replacing it from scratch** — is
sound regardless of whether the absolute numbers reported here transfer to your data.

The wrong move, on most real datasets, is to drop the existing forecast entirely under
the assumption that doing so is "the rigorous choice." It is rigorous only against the
narrow benchmark of pure-prediction skill; for deployment, the residual framing is usually
the more honest baseline.

---

## Dataset

[Kaggle · "Retail Store Inventory Forecasting Dataset"](https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset)
by Anirudh Singh Chauhan · License: CC0 · See [`data/retail_store_inventory.md`](data/retail_store_inventory.md)

- **Shape:** 73,101 rows × 15 columns · 2021-12-31 → 2023-12-31
- **Series:** 100 (5 stores × 20 products) · **grain:** daily
- **Target:** `Units Sold` — censored by `Inventory Level` (stockouts cap observed sales)
- **Distinctive column:** `Demand Forecast` (the existing system's prediction, the focus of
  this notebook's headline insight)
- **EDA quirk:** within-group autocorrelation ≈ 0 on this dataset — the synthetic generator
  produced memoryless series, which is why lag features add no signal and the MAE 69
  plateau (no-DF, multivariate ML methods) reflects the noise floor of the available
  features rather than a modeling failure. Real retail data typically has measurable
  autocorrelation; this finding is dataset-specific.

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
