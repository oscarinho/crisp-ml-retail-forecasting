# Two MAE plateaus, not one — what Demand Forecast does to MAE on this dataset

A 90-day holdout experiment + 18-window champion-challenger backtest. On this Kaggle retail
dataset, replacing one number ("MAE ≈ 69 is the noise floor") with a more granular framing:
**this dataset has two distinct MAE plateaus depending on whether `Demand Forecast` enters
the pipeline as a residual prior or not.**

- **Without DF** (model trained directly on `Units Sold`, multivariate ML methods using
  `Inventory Level`): MAE ≈ **69** on this holdout
- **With DF as a prior** (residual learning, model corrects DF): MAE ≈ **7.4** on this holdout

This 62-unit gap is **largely a function of how the existing forecast signal enters the
pipeline** under the synthetic noise structure of this dataset, not of choosing a better
model within either regime. Real datasets will narrow this gap (DF on real data has higher
MAE than this synthetic near-oracle); the design pattern of "use the existing forecast as
a prior, learn the residual" generally still applies.

> Date: 2026-06-01 · Dataset: `data/retail_store_inventory.csv` · 73k rows · 90-day holdout for Exp A, rolling 30-day windows for Exp B.
> Reproduce: `python scripts/df_experiments.py`

---

## What kicked this off

A collaborator (vic) published a model claiming **MAE 7.29 on the same dataset** while explicitly excluding the `Demand Forecast` column. Our notebook reports **MAE ≈ 69**. Both numbers are reproducible; one of them is leakage.

The audit (logged in conversation): vic's pipeline includes a feature `inventory_to_forecast_ratio = Inventory Level / Demand Forecast`. Even with `Demand Forecast` itself excluded, this ratio + `Inventory Level` lets a tree model **algebraically recover Demand Forecast**, which has ρ = 0.997 with `Units Sold`. That single feature collapses MAE from 68.80 → 7.32 in an ablation. The "MAE 7" is real but driven by indirect leakage, not by model quality.

That raised a fair counter-question: *"Demand Forecast should enter the model somewhere — a real planner would use it as a prior and let the model correct it."* That's a legitimate framing. Hence the two experiments below.

---

## Experiment A — Residual learning

### Setup

- **Target (residual):** `y_resid = Units Sold − Demand Forecast`
- **Features:** the same 14-column contextual set used in our deployment app (`APP_FEATURES`) — no DF in the feature matrix
- **Final prediction:** `pred = DF + model.predict(features)`, clipped at 0
- **Split:** time-based, last 90 days = holdout (matches the notebook's `SPLIT_DATE`)
- **Models tried:** HistGradientBoosting, RandomForest, LightGBM (residual targets) + same models direct on `Units Sold` for comparison

### Results

| Strategy | MAE | Bias | Δ vs DF puro |
|---|---:|---:|---:|
| DF puro (no model)                       | 8.35  | **+5.05** | — |
| Direct HGB (target=`Units Sold`, no DF)   | 69.03 | −1.37 | +60.7 ❌ |
| Direct RF (target=`Units Sold`, no DF)    | 69.33 | −0.71 | +60.9 ❌ |
| Direct LightGBM (target=`Units Sold`, no DF) | 69.10 | −1.38 | +60.7 ❌ |
| **Residual: DF + HGB(features)**          | **7.43**  | **+0.10** | **−0.92 ✓** |
| Residual: DF + RF(features)               | 7.45  | +0.13 | −0.90 ✓ |
| Residual: DF + LightGBM(features)         | 7.46  | +0.15 | −0.89 ✓ |

### What the numbers say

1. **Residual learning DOES beat DF puro** — ~11% MAE reduction. Not magic, real.
2. **The gain comes from bias correction, not variance reduction.** DF puro systematically overshoots by **+5.05 units**. The residual model drives bias to **+0.10** — a **50× reduction**. In production that translates directly to less safety-stock buffer.
3. **Model family barely matters** (HGB 7.43, RF 7.45, LGBM 7.46). The residual target has std ≈ 8.65 but is mostly noise; once the bias is captured, no model can do much more.
4. **Residual learning's MAE (7.43) is within 0.14 of vic's leaky pipeline (7.29).** Same gravitational pull, but residual learning is deployable when DF is published in advance (which it usually is in real S&OP systems). Vic's pipeline requires DF *at inference time on the prediction day*, which is the leakage definition.

---

## Experiment B — Champion-Challenger Backtesting

### Setup

- **18 rolling 30-day forecast windows**, no overlap, starting after a 6-month warm-up
- **7 contenders:**
  - `DF_puro` — no fit, just `pred = DF`
  - `HGB_direct`, `RF_direct`, `LightGBM_direct` — target = `Units Sold`, no DF in features
  - `HGB_residual`, `RF_residual`, `LightGBM_residual` — target = `Units Sold − DF`, prediction = `DF + model`
- **Protocol per window:** train on all prior data, predict the next 30 days, score MAE
- **Champion:** lowest MAE in the window

### Results — Wins per model

| Model | Wins / 18 | Win rate |
|---|---:|---:|
| **HGB_residual** | **17** 🏆 | **94.4%** |
| RF_residual | 1 | 5.6% |
| LightGBM_residual | 0 | 0% |
| DF_puro | 0 | 0% |
| HGB_direct | 0 | 0% |
| RF_direct | 0 | 0% |
| LightGBM_direct | 0 | 0% |

### Results — Overall MAE per model

| Model | Mean MAE | Std | Min | Max |
|---|---:|---:|---:|---:|
| HGB_residual         |  7.39 | 0.08 | 7.22 | 7.55 |
| RF_residual          |  7.41 | 0.08 | 7.23 | 7.56 |
| LightGBM_residual    |  7.43 | 0.09 | 7.22 | 7.58 |
| DF_puro              |  8.31 | 0.11 | 8.15 | 8.51 |
| HGB_direct           | 69.09 | 0.79 | 67.64 | 70.31 |
| RF_direct            | 69.35 | 0.89 | 67.72 | 70.70 |
| LightGBM_direct      | 69.41 | 0.87 | 67.73 | 70.74 |

### What the numbers say

1. **On this dataset, the framing decision dominates the algorithm decision.** Direct-vs-Residual moves MAE by ~62 units; the choice between HGB / RF / LightGBM within the residual regime moves MAE by ~0.04 units. The two effects are measured on different scales; the takeaway is that *how the prior signal enters the pipeline* explains far more variance in holdout MAE than the choice of gradient booster within the residual family.

2. **In this dataset, champion-challenger rotation is overhead, not value.** HGB_residual wins ~94% of windows and the runner-up sits 0.02 MAE behind. Operational cost of rotating models monthly > the ~0.02 MAE you'd recover. In datasets with real external shocks (COVID, recalls, viral products), stability would likely drop and rotation could pay off. Here it doesn't.

3. **On this dataset, direct models are dominated when DF is available.** Across every window, every direct contender lands at MAE 67–71; every residual variant lands at MAE 7.2–7.6. The plateaus are clearly separated for this dataset. Whether this separation transfers to a non-synthetic dataset depends on how close that dataset's existing forecast is to ground truth — for a synthetic ρ=0.997 column the separation is extreme, for a real ρ=0.75 forecast it would be smaller.

4. **DF puro is a respectable baseline** but always dominated. Across all 18 windows, no contender ever fell below DF puro on a single window. MAE 8.31 vs MAE 7.39 for HGB_residual — small absolute gap, big relative one *on this dataset*.

---

## The two-plateau framing (on this dataset)

| Regime | MAE plateau on this dataset | What it implies |
|---|---:|---|
| **DF unavailable** in production (e.g. green-field forecasting, no legacy planning system) | ~69 | The plateau hit by the multivariate ML methods that use `Inventory Level` (LightGBM, RandomForest, Stacking, CatBoost, HGB, ExtraTrees). Univariate methods on this dataset (ARIMA MAE 89, ETS 89, LSTM 89, Prophet 112) cannot reach this plateau because they don't use cross-sectional features. The number reflects the noise structure of this synthetic dataset and the feature set used. |
| **DF available** in production (legacy ERP / vendor forecast / vendor-managed inventory) | ~7.4 | What residual learning achieves on this holdout, with HGB / RF / LightGBM all within 0.05 MAE of each other. The improvement vs DF puro (~11%) is almost entirely from bias correction (+5.05 → +0.10). |

The "MAE 69 is the noise floor" claim in the original README was technically correct but **incomplete**. It applies only to the no-DF regime, and only to the multivariate ML methods on this feature set. For practitioners deploying inside an S&OP function with a DF column available, the relevant plateau on this dataset is **~7.4 from residual learning**, and the model that gets there is HistGradientBoosting.

> **Caveat:** these plateaus are specific to this dataset and feature set. The 1,500× gap
> between framing (~62 MAE) and algorithm choice (~0.04 MAE) is unusually extreme because
> `Demand Forecast` is a synthetic near-oracle (ρ=0.997 with target). On real data with a
> noisier existing forecast the gap will narrow — but the design principle (residual prior
> instead of dropped feature) generalizes.

---

## Implications for this repo

| Component | Status now | Decision |
|---|---|---|
| `notebooks/Inventory_Forecasting_CRISPML.ipynb` | Added §4.13 Residual Learning, §4.14 Champion-Challenger, §4.15 Two-Ceiling Leaderboard | Reproducible inside the notebook with `cached()` |
| `app/app.py` | Deploys `model/model_contextual.pkl` (no DF) | **Consider adding a `use_demand_forecast` toggle** — if user provides DF in the form, route to residual model; otherwise route to direct model |
| `model/` | Has `model_contextual.pkl` (direct, no DF) | Add `model_residual_hgb.pkl` for production residual scoring |
| `README.md` | Claims single MAE 69 ceiling | Updated to two-ceiling framing |
| `scripts/df_experiments.py` | Generates the numbers above | Re-run after any data refresh to validate ceilings still hold |

---

## On the original concern about synthetic data

> *"Los datos son sintéticos, justo eso los debería hacer mejor predecibles."*

Theoretically true if you have access to the generative process. We don't. We only see outputs.

The Kaggle synthetic generator appears to use this structure:

```
Units Sold ≈ Demand Forecast + Gaussian(0, ~5)
Demand Forecast = f(Inventory Level, Category, Region, Season, ...) + structured noise
```

The features we see (Price, Weather, Promo, Region) explain only ~33% of `Units Sold` variance directly. The remaining 67% is captured by `Demand Forecast` — but DF wasn't *computed* from those features either; it was generated upstream with its own latent variables we don't observe.

So the "noise floor 69" isn't the model failing — it's the unobservable latent state that DF embeds. Residual learning works because DF *literally is* the unobservable latent state plus 5 units of Gaussian noise.

**In a real (non-synthetic) dataset:**
- ρ(DF, Units Sold) would be 0.7–0.85, not 0.997
- The residual model would do more variance reduction (not just bias correction) — gain probably 30-50% over DF puro, not just 11%
- Champion-challenger rotation would pay off more often

---

## Reproduce

```bash
# Re-runs both experiments, caches results to scripts/df_experiments_results.json
python scripts/df_experiments.py

# Or with the project conda env
/Users/oscarponce/miniconda3/envs/ml-exp/bin/python scripts/df_experiments.py
```

Runtime: ~3 min Exp A, ~6 min Exp B on M-series Mac (CPU). Results are deterministic (`random_state=42`).
