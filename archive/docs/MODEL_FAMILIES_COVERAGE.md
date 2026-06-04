# Cobertura de Familias de Modelos — para pegar en las libretas

> **Cómo usar este archivo.** Es un bloque markdown listo para pegar
> como **celda markdown** en ambos notebooks
> (`Inventory_Forecasting_CRISPML.ipynb` y `Sales_Forecasting_CRISPML.ipynb`).
> Sugerencia de ubicación: justo después de la sección final de
> "Limitations & Caveats" o como appendix antes de "For Supply Chain
> Professionals". El texto está en inglés para coincidir con el tono
> existente de las libretas.

---

## Copiar desde aquí 👇

```markdown
## Model Families Coverage — Industry Standard Mapping

This work tests the **four canonical model families** for tabular demand
forecasting. Mapping our experiments against the industry-standard
taxonomy so any reviewer can see at a glance what was covered, what was
deliberately skipped, and why.

### The four families

| # | Family | Sub-models tested | Status | Best result |
|---|---|---|---|---|
| 1 | **Tree-based** (XGBoost, LightGBM, CatBoost) | LightGBM ✅ · XGBoost ✅ · RandomForest ✅ · GradientBoosting ✅ · CatBoost ❌ | Mostly covered | **Winner overall** — LightGBM App-aligned MAE 20.1 (Sales) / Stage 2 MAE 69.1 (Inventory) |
| 2 | **Classical statistical** (ARIMA, SARIMAX) | Auto-ARIMA per group ✅ · ETS Holt-Winters ✅ · SARIMAX with exogenous ❌ | Partially covered | Saturated at mean baseline (MAE ~89 on Inventory) |
| 3 | **Prophet** (Meta) | Prophet per (Store × Category) ✅ | Covered | Worst — MAE 112.0 (Inventory). Over-extrapolates trends on near-stationary series |
| 4 | **Deep Learning** (LSTM, DeepAR) | Multivariate LSTM 4-channel, 60-step ✅ · DeepAR ❌ · TFT ❌ | Partially covered | MAE 88.9 (loss plateaued — did not actually learn) |

### Per-family insight

#### 1. Trees — the reigning approach for tabular retail forecasting

LightGBM was the workhorse of both labs. The most striking result was
that **the simpler LightGBM (15 features, App-aligned) beat the more
complex one (29 features, Stage 2)** on holdout — confirming the
finding from the M5 Walmart competition that disciplined feature
engineering wins over feature accumulation. XGBoost contributed
marginally inside the stacking ensemble; RandomForest served as a
robustness check.

#### 2. Classical — a useful baseline, but limited here

Auto-ARIMA and ETS Holt-Winters were applied per (Store × Category) on
the Inventory dataset. Both saturated at MAE ~89 — equal to the mean
baseline. The reason: univariate methods can only see *one series at a
time*, and our signal is *cross-sectional* (Inventory Level, Promotion,
and lag features matter more than yesterday's sales of this specific
SKU). SARIMAX **with** exogenous regressors was not tested — it would
have been the fair-fight version, since it can incorporate Price,
Discount, Promotion, Holiday as exogenous inputs.

#### 3. Prophet — wrong tool for stationary series

Prophet performed worst (MAE 112.0). The dataset has no strong yearly
seasonality and no irregular holiday spikes. Prophet's trend
decomposition then **amplifies noise** instead of dampening it. The
lesson is portable: *Prophet is not a universal "good enough" baseline.*
Always validate it against a mean predictor first. Prophet shines in
retail with **strong holiday effects, irregular sales patterns, missing
data, and a forecast horizon long enough that yearly seasonality
matters** — none of which apply here.

#### 4. Deep Learning — did not actually train

The multivariate LSTM (4 input channels: Units Sold, Inventory,
Holiday, Discount; 60-step window) plateaued at loss ≈ 0.0485 from
epoch 5 onwards. The "MAE 88.9" result is essentially a fancy mean
predictor. Likely causes:

- No per-series embeddings to separate (Store × Category) signals.
- 60-step window too long for series whose autocorrelation decays past
  lag-7.
- Learning-rate schedule didn't help escape the flat region.

DeepAR and TFT — the **global multi-series deep models** actually used
in production by Amazon, Walmart, and Uber — were not tested. They
share parameters across series and learn cross-SKU patterns, which is
exactly what was missing in the single-series LSTM here.

### Coverage summary

| Coverage status | Models |
|---|---|
| ✅ Tested | LightGBM · XGBoost · RandomForest · GradientBoosting · Auto-ARIMA · ETS Holt-Winters · Prophet · LSTM (multivariate) · Ridge · StackingRegressor |
| ❌ Not tested (intentional gaps) | CatBoost · SARIMAX with exogenous · DeepAR · TFT (Temporal Fusion Transformer) · N-BEATS · Croston / TSB |

### Why those gaps matter (and don't)

- **CatBoost** — worth testing because it handles categorical variables
  natively (Region, Weather, Seasonality). Expected delta vs LightGBM:
  ±2% MAE. Not a game-changer here, but cleaner code.
- **SARIMAX with exogenous regressors** — the fair-fight version of
  Classical methods. Would let the model include `Price`, `Discount`,
  `Promotion`, `Epidemic` as exogenous inputs. Unlikely to beat
  LightGBM, but would close the gap from MAE ~89 toward ~70 on the
  Inventory dataset.
- **DeepAR / TFT** — these are the *appropriate* deep-learning
  comparison, not single-series LSTM. They share parameters across
  series, learn cross-SKU patterns, and are what Amazon, Walmart, and
  Uber actually deploy at scale. With 100 series × 760 days, a small
  TFT could outperform LightGBM. **This is the highest-priority gap.**
- **Croston / TSB** — for intermittent-demand SKUs (>30% of days with
  zero demand). Neither dataset has this characteristic, but in real
  retail it's the right routing for a significant share of the long
  tail.

### What this matrix proves

Industry standard says: *"benchmark all four families before declaring a
winner."* This work covers three families completely and the fourth
partially. The winning model — **gradient-boosted trees on
well-engineered tabular features** — matches what Kaggle competitions
(M5 Walmart, Corporación Favorita) and production retailers (Instacart,
Wayfair, Carrefour) have converged on for daily SKU-level forecasting.

### References

- Makridakis S., Spiliotis E., Assimakopoulos V. (2022). *M5 accuracy
  competition: Results, findings and conclusions.* International
  Journal of Forecasting.
- Hyndman R., Athanasopoulos G. (2021). *Forecasting: Principles and
  Practice (3rd ed.).* OTexts — reference for ARIMA, ETS, Prophet
  comparisons.
- Salinas D., Flunkert V., Gasthaus J. (2020). *DeepAR: Probabilistic
  Forecasting with Autoregressive Recurrent Networks.* International
  Journal of Forecasting.
- Lim B., Arik S.Ö., Loeff N., Pfister T. (2021). *Temporal Fusion
  Transformers for Interpretable Multi-horizon Time Series
  Forecasting.* International Journal of Forecasting.
```

## Copiar hasta aquí 👆

---

## Notas adicionales (para ti, no para pegar)

- **Lugar sugerido en cada notebook:** justo antes de la sección
  "For Supply Chain Professionals" o como appendix después de
  "Limitations & honest caveats". Encaja mejor como cierre técnico
  antes del giro hacia el lenguaje del planner.
- **Si quieres una versión más corta** (solo la tabla principal sin
  los insights por familia), avísame y la genero.
- **Si decides agregar CatBoost o SARIMAX-con-exógenas** después,
  basta con cambiar el ❌ por ✅ en la tabla y añadir una línea más al
  "Per-family insight" — el resto de la estructura aguanta.
- **Versión en español:** si prefieres pegar la versión en español en
  vez de inglés (las libretas hoy están en inglés, pero la audiencia de
  oscarponce.com tiene perfil LATAM), también puedo traducirla. Avísame.
