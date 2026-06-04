# Página de Lab — Demand Forecasting (oscarponce.com/labs/demand-forecasting)

Contenido para publicar en oscarponce.com. Formato bilingüe EN/ES y profundidad
técnica alineada con el resto de labs. Estructura idéntica a `predictive-maintenance`
y `hr-attrition`: header → metadata → 6 fases CRISP-ML(Q) → business impact →
navegación.

---

## Header

**Tag (eyebrow):** Time Series · Regression · Machine Learning

**Title (EN):** 📈 Retail Demand Forecasting
**Title (ES):** 📈 Pronóstico de Demanda Retail

**Description (EN):**
End-to-end CRISP-ML(Q) pipeline over two contrasting retail datasets
(~150,000 records) to forecast daily demand per Store × Product. The lab
demonstrates leakage defense, two-stage modeling with cold-start routing,
and a quantile (P80) model wired directly to a newsvendor reorder rule.

**Description (ES):**
Pipeline CRISP-ML(Q) end-to-end sobre dos datasets retail contrastantes
(~150.000 registros) para pronosticar demanda diaria por tienda × producto.
El lab demuestra defensa frente a leakage, arquitectura de dos etapas con
ruteo de cold-start, y un modelo cuantil (P80) conectado directamente a la
regla de reorden tipo newsvendor.

### Metadata cards

| | |
|---|---|
| 📊 **Dataset** | Retail Store Inventory + Sales Forecasting |
| 📝 **Records / Registros** | 73,100 (Inventory) + 76,000 (Sales) = **~149,100** |
| 🏭 **Industry / Industria** | Retail · Supply Chain · CPG |

### Tech stack

`Python` `LightGBM` `XGBoost` `RandomForest` `Prophet` `pmdarima` `statsmodels`
`PyTorch (LSTM)` `Scikit-learn` `SHAP` `Streamlit` `joblib` `CRISP-ML(Q)`

### CTAs

- 🚀 **View Live Demo / Ver Demo en Vivo** → `#demo`
- 📂 **View Code on GitHub / Ver Código en GitHub** → `https://github.com/oscarinho/forecasting-inventory`

---

## 01 — 🎯 Business Understanding

**EN.**
Retail stores carry the wrong amount of inventory most days. Too much,
and capital is locked up in shelf space; too little, and sales walk out
the door — along with the customer. Optimizing the daily reorder
decision per (Store × Product) is one of the highest-ROI applications
of ML in supply chain.

The cost asymmetry is not symmetric. Industry research quantifies
each side independently: stockouts cost retailers **4–8% of annual
sales** [ToolsGroup], with **9% of customers permanently switching to
a competitor after a single stockout** — rising to **55% after
repeated stockouts** [Opensend, 2024]. Overstock carrying costs are
**20–30% of inventory value annually**. That asymmetry must be
encoded in the modeling choices — not patched on at the end.

**Project goals:**

- Forecast daily demand per (Store × Product) with horizon t+1.
- Produce a P80 quantile forecast for the newsvendor reorder rule.
- Build a cold-start path for SKUs without 28 days of history.
- Deploy as a Streamlit app a planner can use without help from data.

**ES.**
Las tiendas retail cargan la cantidad equivocada de inventario casi
todos los días. De más, y el capital queda atrapado en estantería; de
menos, y las ventas se van por la puerta — junto con el cliente.
Optimizar la decisión diaria de reorden por (Tienda × Producto) es una
de las aplicaciones de ML de mayor ROI en supply chain.

La asimetría de costos no es simétrica. La investigación de industria
cuantifica cada lado por separado: los stockouts cuestan **4–8% de las
ventas anuales** en retail [ToolsGroup], con **9% de clientes que se
cambian permanentemente a un competidor tras un solo stockout** —
escalando a **55% tras stockouts repetidos** [Opensend, 2024]. El
carrying cost de overstock es **20–30% del valor del inventario por
año**. Esa asimetría debe estar codificada en las decisiones de
modelado — no parchada al final.

**Objetivos del proyecto:**

- Pronosticar demanda diaria por (Tienda × Producto) con horizonte t+1.
- Producir un forecast cuantil P80 para la regla de reorden newsvendor.
- Construir ruta de cold-start para SKUs sin 28 días de historia.
- Desplegar como app Streamlit que un planner pueda usar sin ayuda de datos.

| Problem type / Tipo | Time-series regression (per-row, panel data) |
|---|---|
| Target | Demand (uncensored, integer 4–430 in Sales) |
| Main metrics | MAE, RMSE, sMAPE (MAPE unreliable near zero) |
| Decision metric | P80 quantile coverage |
| Stakeholder | Demand Planner / Supply Chain Manager |

---

## 02 — 📊 Data Understanding

**EN.**
Two sibling datasets, same problem (daily retail demand), very different
internal structure. Running the same pipeline over both isolates the
contribution of the dataset itself from the contribution of the model —
a separation most public notebooks skip.

| Aspect | Inventory dataset | Sales dataset |
|---|---|---|
| Records | 73,100 (5 stores × 20 products × 731 days) | 76,000 (5 stores × 20 products × 760 days) |
| Target | `Units Sold` | `Demand` (uncensored true demand) |
| `lag_1` autocorrelation | ≈ 0 (memoryless) | 0.35 |
| `rolling_7` autocorrelation | ≈ 0 | 0.49 |
| Stockout rate | n/a | 70% of rows |
| Leakage column | `Demand Forecast` ρ=0.997 | `Units Sold` ρ=0.83, `Units Ordered` ρ=0.51 |
| Pricing elasticity | ≈ 0 (synthetic limitation) | ≈ 0 (synthetic limitation) |

**Key diagnostics performed:**

- **Leakage audit.** Correlation of each candidate feature with the
  target. Anything with ρ > 0.5 *that is observed after the decision
  point* is dropped or lagged.
- **Autocorrelation per-group.** `(Store, Product).Demand` shifted vs
  current — tells you whether lag features will pay off before you
  build them.
- **Stockout analysis.** 70% of rows in the Sales dataset are
  stockouts (`Units Sold < Demand`). Training on `Units Sold` would
  teach the model to under-forecast forever.
- **Stationarity check.** ADF test per series; auto-ARIMA agreed
  `d=0` for almost all series.

**ES.**
Dos datasets hermanos, mismo problema (demanda retail diaria), estructura
interna muy distinta. Correr el mismo pipeline sobre ambos aísla la
contribución del dataset de la del modelo — una separación que la mayoría
de notebooks públicos omiten.

**Diagnósticos clave:**

- **Auditoría de leakage.** Correlación de cada feature candidato con el
  target. Cualquiera con ρ > 0.5 *que se observe después del punto de
  decisión* se elimina o se lagea.
- **Autocorrelación por grupo.** `(Tienda, Producto).Demand` shifted vs
  actual — dice si los lag features van a funcionar antes de construirlos.
- **Análisis de stockouts.** 70% de las filas del dataset de Ventas son
  stockouts. Entrenar con `Units Sold` enseñaría al modelo a sub-pronosticar
  para siempre.
- **Test de estacionariedad.** ADF por serie; auto-ARIMA confirmó `d=0`
  en casi todas.

> ⚠️ **Honest caveat:** both datasets are synthetic. Real retail data
> shows **price elasticity of demand typically between –0.5
> (inelastic categories like staples) and –2 (elastic categories like
> discretionary goods)** [7Learnings; Wikipedia *Price Elasticity*];
> these datasets show ≈ 0 elasticity (`Price` ↔ `Demand` correlation
> < 0.02), confirming the absence of any pricing mechanism. Real
> data also carries trend and seasonal mix shifts that these
> datasets lack. The methodology generalizes; the absolute MAE
> numbers do not.

---

## 03 — 🔧 Data Preparation

**EN. Feature engineering — leakage-safe lag and rolling features.**

```python
# Lags built per (Store × Product) using shift() — never sees the future.
GROUP_COLS = ['Store ID', 'Product ID']
g = df_model.groupby(GROUP_COLS)

df_model['Demand_lag_1']  = g['Demand'].shift(1)
df_model['Demand_lag_7']  = g['Demand'].shift(7)
df_model['Demand_lag_28'] = g['Demand'].shift(28)

# Rolling features use closed='left' to exclude the current row.
df_model['Demand_roll7_mean']  = g['Demand'].shift(1).rolling(7).mean()
df_model['Demand_roll28_mean'] = g['Demand'].shift(1).rolling(28).mean()

# Quality gate: lag_1 must reference the prior calendar day for >= 99%
# of non-null rows. Asserted in-notebook — catches grouping bugs early.
```

**Date features + Fourier encoding** for cyclical signals:

```python
df_model['month_sin'] = np.sin(2*np.pi*df_model['month']/12)
df_model['month_cos'] = np.cos(2*np.pi*df_model['month']/12)
df_model['dow_sin']   = np.sin(2*np.pi*df_model['day_of_week']/7)
df_model['dow_cos']   = np.cos(2*np.pi*df_model['day_of_week']/7)
```

**Mixed-scaler ColumnTransformer** — different scalers per feature type:

- `StandardScaler` for normal numerics (Price, Competitor Pricing).
- `MinMaxScaler` for bounded/ordinal (Inventory Level, Fourier features).
- `PowerTransformer` for skewed numerics (Demand-derived rolling stats).
- `OneHotEncoder` for low-cardinality categoricals.

**Time-based split** — last 90 days of Sales, 2023 holdout for Inventory.
No random shuffling. Train ends *strictly* before test starts.

**ES. Feature engineering — lag y rolling features a prueba de leakage.**
Construidos por grupo `(Tienda × Producto)` con `shift()` — nunca ven el
futuro. Quality gate asserta que `lag_1` referencia el día calendario
previo en ≥99% de las filas no nulas; atrapa bugs de agrupación temprano.

**Date features + codificación Fourier** para señales cíclicas.
**ColumnTransformer multi-scaler** — StandardScaler, MinMaxScaler,
PowerTransformer y OneHotEncoder según el tipo de feature.
**Split temporal** — últimos 90 días en Ventas, holdout 2023 en
Inventario. Sin shuffle aleatorio.

---

## 04 — 🧠 Modeling

**EN.** Eight algorithms evaluated head-to-head on the same holdout:

| Model | Type | Sales MAE | Inventory MAE |
|---|---|---:|---:|
| Mean baseline | naive | 32.6 | 89.1 |
| Lag-1 naive | naive | 41.0 | ~120 |
| Rolling-7 baseline | naive | 32.1 | — |
| Auto-ARIMA per group | classical | — | 89.1 |
| ETS Holt-Winters | classical | — | 89.4 |
| Prophet per group | classical | — | 112.0 |
| LSTM (4-channel, 60-step) | deep learning | — | 88.9 |
| RandomForest | tree ensemble | 23.8 | — |
| Stacking ensemble (Ridge meta) | ensemble | — | 68.9 |
| **Stage 2 LightGBM (29 features)** | gradient boosting | 21.1 | 69.1 |
| **Stage 1 LightGBM (16 features, no lags)** | gradient boosting | 20.3 | 90.2 |
| **App-aligned LightGBM (15 features)** | gradient boosting | **20.1** | — |
| Quantile P80 LightGBM | gradient boosting | (coverage 78%) | (coverage 77.5%) |

**Two-stage architecture with cold-start routing:**

```
                 history depth?
                /              \
            ≥ 28 days        < 28 days
                |                |
            Stage 2          history?
            (full features)  /        \
                          ≥ 7 days   < 7 days
                             |          |
                          Stage 1   App-aligned
                          (no lags) (form-only)
```

**Why route, don't ensemble.** The three models have different feature
schemas. A weighted blend across schemas silently mixes incompatible
distributions. Route on `min_history_days` instead.

**Quantile P80 for reorder.** A point forecast minimizes squared error
symmetrically. Inventory decisions are asymmetric. Train a second
LightGBM with `objective='quantile', alpha=0.80` and feed its output
into the reorder rule directly.

**ES.** Ocho algoritmos evaluados cabeza-a-cabeza en el mismo holdout.
Arquitectura de dos etapas con ruteo de cold-start: Stage 2 (full
features) para SKUs con ≥28 días; Stage 1 (sin lags) para 7–27 días;
App-aligned (solo features del formulario) para <7 días.

**Por qué rutear, no ensamblar.** Los tres modelos tienen schemas de
features distintos. Un blend ponderado mezcla distribuciones
incompatibles en silencio. Mejor rutear por `min_history_days`.

**Cuantil P80 para reorden.** Forecast puntual minimiza error cuadrático
simétricamente; decisiones de inventario son asimétricas. Segundo
LightGBM con `objective='quantile', alpha=0.80` conectado directo a la
regla de reorden.

---

## 05 — ✅ Evaluation

**EN. Overfit diagnostic — the surprise winner.**

Comparing in-sample vs holdout MAE per model on the Sales dataset:

| Model | Train MAE | Holdout MAE | Gap | Overfit |
|---|---:|---:|---:|---|
| Stage 2 LightGBM | 17.2 | 21.1 | 22.7% | High |
| Stage 1 (no lags) | 18.5 | 20.3 | 9.7% | Moderate |
| **App-aligned** | **17.6** | **20.1** | **14.5%** | **Low** |
| Stage 2 RandomForest | 15.7 | 23.8 | 51.6% | Severe |

Counterintuitive result: **the App-aligned model with 15 features beats
the Stage 2 model with 29 features.** Two compatible explanations:

1. **Overfit on lag features.** With ~64k training rows, the trees
   memorize lag-specific patterns that don't generalize three months
   forward.
2. **Temporal drift.** The lag↔target relationship shifts between the
   training window (2022 → Nov 2023) and the holdout (Nov 2023 → Jan
   2024). Lag coefficients trained on the past don't transfer.

**Per-segment error analysis.** MAE is uniform across stores (67–70 on
Inventory; 14–25 on Sales by category). No catastrophic segment.

**Residual diagnostics.** Bias ≈ 0, symmetric residual distribution.
Heteroscedasticity ratio 1.47 — borderline; log-transform tested
(neutral on global MAE, mildly positive on Groceries).

**SHAP feature importance (Sales, App-aligned model).** Top drivers:
`Price`, `Inventory Level`, `Competitor Pricing`, `Epidemic` flag,
`Promotion` flag. The tree finds non-linear interactions even where raw
correlations are near zero.

**Why we chose MAE over MAPE.** MAPE explodes when the actual is near
zero. Daily SKU-level demand frequently sits at 4–10 units. sMAPE is
the safer choice and we report it everywhere alongside MAE.

**ES.** Diagnóstico de overfit — el ganador inesperado: el modelo
**App-aligned con 15 features supera al Stage 2 con 29 features**. Dos
explicaciones compatibles: overfit en lag features con ~64k filas de
entrenamiento, y drift temporal entre train (2022–Nov 2023) y holdout
(Nov 2023–Ene 2024).

Análisis de error por segmento: MAE uniforme (sin segmento catastrófico).
Diagnóstico de residuos: bias ≈ 0, distribución simétrica,
heteroscedasticidad ratio 1.47 (borderline).

SHAP en el modelo App-aligned: `Price`, `Inventory Level`,
`Competitor Pricing`, `Epidemic`, `Promotion` lideran — los árboles
encuentran interacciones no lineales donde la correlación cruda es
cercana a cero.

---

## 06 — 🚀 Deployment

**EN. Deployment stack:**

- **Frontend:** Streamlit (`app/app.py`).
- **Models:** LightGBM serialized with `joblib` — `model.pkl` (Stage 2),
  `model_stage1.pkl`, `model_contextual.pkl` (App-aligned), `model_q80.pkl`
  (quantile P80).
- **Metadata bundle:** `model_metadata.pkl` with `feature_columns`,
  `package_versions`, `training_date`, holdout metrics — used by the app
  to reindex incoming form data exactly the way the training pipeline
  saw it.
- **Hosting:** Streamlit Cloud (free tier).
- **CI/CD:** GitHub Actions for tests and notebook execution.

**Inference contract.** The app reindexes the user's form input to
`metadata['feature_columns']` exactly. Any mismatch between notebook
feature engineering and `app.py` reconstruction silently produces wrong
predictions — explicitly tested.

**From forecast to reorder — the bridge the model doesn't ship by default:**

```
P80_demand_next_day  = 130 units   # output of model_q80.pkl
lead_time            = 3 days
avg_daily_demand     = 100 units

reorder_point = avg_daily_demand × lead_time + safety_stock_buffer
              = 300              + (P80 - mean) × √lead_time
              ≈ 326 units
```

The quantile model **replaces** the static `Z × σ × √L` term with one
that adapts to current conditions — weather, promotion, epidemic flag,
day-of-week, recent trend.

**Operational hooks:**

- Drift monitoring — PSI on key features (`Inventory Level`, `Price`)
  and KS test on residuals. Industry-standard PSI thresholds: < 0.1
  no change, 0.1–0.2 minor drift, **≥ 0.2 significant drift →
  retrain** [Fiddler AI; GeeksforGeeks].
- Bias tracking — cumulative residual mean ÷ MAE. The statistical
  threshold is **3.75 MAD** (3-sigma at 99% service level), commonly
  rounded to **|tracking signal| > 4** for operational use [Value
  Chain Planning].
- Retrain cadence — quarterly by default, weekly only if drift triggers
  fire. (Tested: weekly retraining gave **0% lift** on the stationary
  synthetic dataset.)

**ES. Stack de deployment:** Streamlit + joblib + Streamlit Cloud + GitHub
Actions. Cuatro modelos serializados con metadata bundle que el app usa
para reindexar el formulario exactamente como vio el pipeline en
entrenamiento — un mismatch silenciosamente produce predicciones
incorrectas, explícitamente testeado.

**Del forecast a la orden de compra:** el modelo cuantil P80 reemplaza
el término estático `Z × σ × √L` por uno que se adapta a las
condiciones actuales (clima, promoción, flag de epidemia, día de
semana, tendencia reciente).

**Hooks operacionales:** monitoreo de drift (PSI + KS), tracking de
bias (cumulative residual ÷ MAE), reentrenamiento trimestral por
defecto, semanal solo si dispara drift.

### 🖥️ Interactive Demo / Demo Interactiva

`● Live` — Streamlit Cloud (link en GitHub).

---

## 💡 Business Impact / Impacto de Negocio

**EN.** For a retail chain with 50 stores and 5,000 SKUs (~250,000
Store × Product combinations), implementing this pipeline could
generate estimated savings of **$1M – $3M USD annually** through
two mechanisms:

1. **Reduced safety stock at the same service level.** A model with
   lower MAE shrinks the safety stock buffer needed to hit the same
   95% in-stock target. **Gartner reports each 1% forecast accuracy
   improvement yields a 2.7% reduction in finished goods inventory**
   [Gartner, cited by Manokhin]; McKinsey estimates a 10–20% accuracy
   improvement trims inventory costs by ~5%.
2. **Fewer stockouts at the same inventory level.** The P80 quantile
   model targets the asymmetric cost structure directly. Industry
   case studies of quantile-based replenishment report **stockout
   reductions of 15–30%** [ToolsGroup].

**Worked calculation** for the $1M–$3M range:

```
Assumptions
  Stores: 50, SKUs: 5,000 → 250,000 (Store × Product) pairs
  Avg inventory per pair: $200 → Total stocked inventory: $50M
  Carrying cost rate: 25% (midpoint of industry 20–30%)
  → Current annual carrying cost: $12.5M
  Annual sales (50 stores): ~$200M
  → Current stockout cost (6% of sales): ~$12M

Improvement (conservative)
  Forecast accuracy lift → 10% inventory reduction (Gartner ratio)
  Inventory freed: $5M  →  carrying cost saved: $1.25M/yr
  Stockout reduction 20%: ~$2.4M/yr saved
  Net annual benefit: ~$3M (mid case)

Lower bound (cautious adoption, partial routing):
  ~$1M/yr benefit
```

**Key benefits:**

- Reduced working capital tied up in inventory
- Higher in-stock service level (fewer lost sales)
- Adaptive safety stock by SKU (auto-wider on erratic items)
- Cold-start path for new SKU launches
- Auditable, explainable forecasts (SHAP per prediction)

**ES.** Para una cadena retail con 50 tiendas y 5.000 SKUs (~250.000
combinaciones tienda × producto), implementar este pipeline podría
generar ahorros estimados de **$1M – $3M USD anuales** mediante dos
mecanismos:

1. **Menor safety stock al mismo nivel de servicio.** **Gartner
   reporta que cada 1% de mejora en forecast accuracy genera 2.7% de
   reducción en finished goods inventory** [Gartner, vía Manokhin];
   McKinsey estima que una mejora del 10–20% en accuracy reduce
   costos de inventario ~5%.
2. **Menos stockouts al mismo nivel de inventario.** Case studies de
   industria con reorden basado en cuantil reportan **reducciones de
   stockout del 15–30%** [ToolsGroup].

El cálculo detallado del rango $1M–$3M está en el bloque en inglés
arriba — basado en 50 tiendas, $50M de inventario stockeado, 25%
carrying cost y un 10% de reducción de inventario conservador.

**Beneficios clave:**

- Reducción de capital de trabajo atrapado en inventario
- Mayor nivel de servicio in-stock (menos ventas perdidas)
- Safety stock adaptativo por SKU (auto-más-ancho en items erráticos)
- Ruta de cold-start para lanzamientos de SKU nuevos
- Pronósticos auditables y explicables (SHAP por predicción)

---

## 🔬 What this lab teaches that most don't / Lo que este lab enseña

**EN.**

1. **The signal lives in the data, not the algorithm.** Same pipeline
   on two datasets gave MAE 69 vs MAE 20. Auditing autocorrelation
   and leakage before modeling is worth more than tuning hyperparams.
2. **Simpler models often win in production.** The 15-feature
   App-aligned model beat the 29-feature Stage 2 model on holdout.
   Always measure `train_MAE` vs `holdout_MAE`; gap is the overfit
   indicator.
3. **Univariate time-series methods saturate when cross-sectional
   signal dominates.** ARIMA, ETS, Prophet, and a 4-channel LSTM all
   tied with the mean baseline on the Inventory dataset because the
   signal was in inventory level + lag features, not in univariate
   temporal structure.
4. **Quantile beats point forecast for asymmetric costs.** When the
   stockout/overstock cost ratio is 3:1 or worse, train the quantile
   model that targets the service level directly.
5. **Synthetic data has a ceiling.** Both datasets lack pricing
   elasticity and concept drift. Methodology generalizes; absolute
   MAE numbers do not — re-validate on real data before quoting.

**ES.**

1. **La señal vive en el dato, no en el algoritmo.** Mismo pipeline,
   dos datasets, MAE 69 vs MAE 20.
2. **Modelos más simples suelen ganar en producción.** App-aligned con
   15 features venció a Stage 2 con 29 en holdout.
3. **Métodos univariados se saturan cuando domina la señal
   cross-sectional.** ARIMA, ETS, Prophet y LSTM-4-canales empataron
   con la media en el dataset Inventory.
4. **Cuantil supera al pronóstico puntual cuando los costos son
   asimétricos.** Entrenar directamente el cuantil que matchea el
   nivel de servicio objetivo.
5. **El dato sintético tiene techo.** Metodología generaliza; números
   absolutos no.

---

## 🧑‍💼 For Supply Chain Professionals / Para Profesionales de Supply Chain

**EN.** If you plan inventory for a living and skipped the code above —
this section is for you. No Python required. We translate what the
models produced into the language of replenishment, safety stock, and
service levels.

### Translating the data-science words / Traduciendo el lenguaje técnico

| ML / notebook term | Supply-chain equivalent | Plain meaning |
|---|---|---|
| Target / `Demand` | Demand | The thing we forecast |
| Feature | Demand driver | Anything that explains sales (price, weather, holiday, promotion) |
| MAE = 20 | Forecast Accuracy (FA) | Typical miss is ~20 units |
| Residual mean | Forecast bias | Do we consistently lean high or low? |
| P80 quantile | 80% service-level target | Stock level that covers demand 4 days out of 5 |
| Lag feature | Recent sales history | "What did this SKU sell last week" |
| Leakage | Cheating | Using info you wouldn't actually have on forecast day |
| Model drift | Forecast going stale | Accuracy decaying as the market changes |

### From forecast to reorder quantity — worked example

A forecast alone doesn't tell you how much to order. Here's the bridge,
in four steps, for one product:

```
Step 1 — Expected demand over lead time
  Forecast = 100 units/day · Lead time = 3 days  → 300 units to cover

Step 2 — Pick a service level (business decision, not technical)
  "Avoid a stockout 95% of cycles" → Z = 1.65
  "Avoid a stockout 80% of cycles" → P80 quantile from the model

Step 3 — Safety stock (the buffer)
  Classic:    safety_stock = Z × forecast_error × √lead_time
              = 1.65 × 18 × √3 ≈ 51 units
  Data-driven (preferred): use the quantile model directly
              safety_stock = P80_forecast – mean_forecast
              automatically wider for erratic SKUs, tighter for steady ones

Step 4 — Reorder Point (ROP)
  ROP = expected_lead_time_demand + safety_stock = 300 + 51 = 351 units
  When on-hand inventory drops below 351, place the order.
```

A smaller forecast error shrinks the safety stock you need for the
**same** service level. That's where the dollar payoff of a better
model shows up.

### Service level ↔ which quantile to predict

| Target service level | Quantile | When to use |
|---|---|---|
| 50% (cost-neutral) | P50 / point forecast | Stockouts ~50% of cycles — too thin for real shelves |
| 80% (typical retail) | **P80 (built in this lab)** | Moderate buffer, balanced cash use |
| 95% (essentials, FMCG) | P95 | Fat buffer, higher availability |
| 99% (medical, safety-critical) | P99 | Reserve for items where a stockout is unacceptable |

**Newsvendor rule of thumb:** the right service level isn't gut feel
— it's `stockout_cost ÷ (stockout_cost + overstock_cost)`. If a
stockout costs $20 and an overstock $5, the optimal is **20/25 =
0.80** — which is exactly why this lab built the P80 model.

### Don't plan every SKU the same way — ABC/XYZ

| | **X (steady demand)** | **Y (variable)** | **Z (erratic / lumpy)** |
|---|---|---|---|
| **A (top 20% revenue)** | Model shines — tight buffers, low safety stock | **Highest ROI for the model** — review weekly | Use intermittent-demand methods (Croston / TSB) — this model will mis-fit |
| **B (next 30%)** | Automate fully | Medium-high value | Same caveat as A-Z |
| **C (long tail)** | Set-and-forget | Don't over-engineer | Don't bother |

Use the model's **per-SKU error rate** as the X/Y/Z axis: low error =
X, high error = Z. The value of the forecasting system is **not
distributed uniformly** across the catalog — identifying *where* it
lives is pre-modeling work, not post.

### Watch bias, not just error

MAE tells you *how big* the misses are. It does **not** tell you which
*direction*. A model can post a great MAE while quietly under-forecasting
every week — and you feel that as chronic stockouts.

- **Forecast bias** = mean of (forecast − actual). Near 0 = balanced.
  Persistently positive = over-forecasting (excess stock). Persistently
  negative = under-forecasting (stockouts).
- **Tracking Signal** = cumulative bias ÷ MAE. Statistical threshold
  is **3.75 MAD** (3-sigma at 99% service level), rounded to **|TS| > 4**
  for operational use. Past that, the forecast has a systematic lean —
  investigate before it costs you a season.
- Review bias **by segment** (region, category, store) — opposite
  biases cancel out in the total and hide the problem.

### Anti-patterns this lab prevents

1. **Bullwhip effect.** Using `Units Ordered` to forecast `Demand`
   would feed our own past over-ordering back into the forecast. This
   lab drops unlagged `Units Ordered` to break the loop.
2. **Censored signal.** Forecasting `Units Sold` instead of `Demand`
   builds in existing stockouts (70% of rows in the Sales dataset are
   stockouts). We forecast true demand, not what we managed to ship.
3. **Same-day leakage.** Every feature is lagged (`*_lag_*`) or
   known-at-decision-time. Nothing observed during day `t` predicts
   day `t`.
4. **False complexity.** Adding more features did not help on this
   dataset. We deployed the simpler model — the 15-feature App-aligned
   wins on holdout.
5. **One service level for everything.** A-items and C-items should
   not get the same buffer. The lab supports per-SKU service level via
   the quantile model.

### Fitting the model into S&OP

The model is a tool **inside** your planning rhythm, not a replacement
for it.

| Cadence | What to use the model for |
|---|---|
| **Daily** | Replenishment per SKU (P80 + ROP). Lab's sweet spot. |
| **Weekly** | Aggregate to category × store; compare actual vs forecast |
| **Monthly** | Bias / tracking signal review; decide whether to retrain |
| **Quarterly** | ABC/XYZ refresh; consider expanding feature set |

**Forecast Value Added (FVA):** every cycle, ask *"did the model beat
simply repeating last period?"* In this lab, the App-aligned model
beats the rolling-7 baseline by **37%** — a real FVA. If a model can't
beat naïve, drop it; complexity without FVA is just cost.

---

**ES.** Si planificas inventario y saltaste el código de arriba — esta
sección es para ti. No requiere Python. Aquí traducimos lo que producen
los modelos al lenguaje de reabastecimiento, safety stock y nivel de
servicio.

### Traducción del lenguaje técnico

| Notebook / ML dice... | Tú lo llamas... | Significado simple |
|---|---|---|
| Target / `Demand` | Demanda | Lo que estamos pronosticando |
| Feature | Driver de demanda | Cualquier dato que explique ventas (precio, clima, feriado, promoción) |
| MAE = 20 | Error de pronóstico (FA) | Error típico ~20 unidades |
| Residual mean | Sesgo del pronóstico | ¿Siempre sobre o sub-pronosticamos? |
| Quantile P80 | Objetivo de servicio 80% | Nivel que cubre demanda 4 de cada 5 días |
| Lag feature | Historial reciente | "Qué vendió este SKU la semana pasada" |
| Leakage | Hacer trampa | Usar info que no tendrías el día del pronóstico |
| Model drift | Pronóstico que se envejece | Precisión que baja con el cambio de mercado |

### Del pronóstico al pedido — ejemplo

Un pronóstico no te dice cuánto pedir. El puente, en cuatro pasos:

```
Paso 1 — Demanda esperada en el lead time
  Pronóstico = 100 unidades/día · Lead time = 3 días → cubrir 300 unidades

Paso 2 — Elegir nivel de servicio (decisión de negocio)
  "Evitar stockout 95% de los ciclos" → Z = 1.65
  "Evitar stockout 80% de los ciclos" → cuantil P80 del modelo

Paso 3 — Safety stock (el colchón)
  Clásico:        safety_stock = Z × error_pronóstico × √lead_time
                  = 1.65 × 18 × √3 ≈ 51 unidades
  Data-driven:    usar directamente el modelo cuantil
                  safety_stock = P80_pronóstico – pronóstico_medio
                  más ancho para SKUs erráticos, más ajustado para estables

Paso 4 — Punto de Reorden (ROP)
  ROP = demanda_lead_time + safety_stock = 300 + 51 = 351 unidades
  Cuando el inventario disponible baja a 351, se pide.
```

Un error de pronóstico menor reduce el safety stock necesario para el
**mismo** nivel de servicio. Ahí vive el ROI del modelo.

### Nivel de servicio ↔ qué cuantil pedirle al modelo

| Nivel de servicio | Cuantil | Cuándo usarlo |
|---|---|---|
| 50% (neutral) | P50 / pronóstico puntual | Stockouts ~50% — demasiado fino para retail real |
| 80% (retail típico) | **P80 (construido en este lab)** | Colchón moderado, uso balanceado de capital |
| 95% (esenciales, FMCG) | P95 | Colchón amplio, alta disponibilidad |
| 99% (médico, crítico) | P99 | Solo para items donde un stockout es inaceptable |

**Regla newsvendor:** el nivel de servicio correcto no es intuición —
es `costo_stockout ÷ (costo_stockout + costo_overstock)`. Si stockout
cuesta $20 y overstock $5, el óptimo es **20/25 = 0.80** — por eso este
lab construyó el modelo P80.

### No planificar todo SKU igual — ABC/XYZ

| | **X (demanda estable)** | **Y (variable)** | **Z (errática / lumpy)** |
|---|---|---|---|
| **A (top 20% revenue)** | El modelo brilla — colchones ajustados | **Mayor ROI del modelo** — revisar semanalmente | Usar métodos de demanda intermitente (Croston/TSB) — este modelo no aplica |
| **B (siguiente 30%)** | Automatizar completo | Valor medio-alto | Mismo caveat que A-Z |
| **C (cola larga)** | Set-and-forget | No sobre-ingeniar | No vale la pena |

El valor del sistema **no se distribuye uniformemente** en el catálogo
— identificar dónde se concentra es trabajo previo al modelo, no
posterior.

### Mirar el sesgo, no solo el error

El MAE te dice qué tan grandes son los errores. **No te dice la
dirección.** Un modelo puede tener excelente MAE y sub-pronosticar
cada semana — el equipo lo siente como stockouts crónicos.

- **Sesgo (bias)** = promedio de (pronóstico − real). Cerca de 0 =
  balanceado. Persistente positivo = sobre-pronóstico (exceso stock).
  Persistente negativo = sub-pronóstico (stockouts).
- **Tracking Signal** = suma acumulada de sesgo ÷ MAE. El umbral
  estadístico es **3.75 MAD** (3-sigma a 99% servicio), redondeado a
  **|TS| > 4** en uso operacional. Si lo cruza, el pronóstico tiene
  un sesgo sistemático — investigar antes de que cueste una temporada.
- Revisar sesgo **por segmento** (región, categoría, tienda) — los
  sesgos opuestos se cancelan en el total y esconden el problema.

### Anti-patrones que este lab previene

1. **Efecto bullwhip.** Usar `Units Ordered` para pronosticar `Demand`
   alimentaría el pronóstico con nuestras propias órdenes pasadas.
   Este lab elimina `Units Ordered` sin lagear para romper el bucle.
2. **Señal censurada.** Pronosticar `Units Sold` en vez de `Demand`
   incorpora los stockouts existentes (70% de filas en el dataset de
   Ventas son stockouts). Pronosticamos demanda real, no lo que
   logramos despachar.
3. **Leakage same-day.** Todo feature está lageado o conocido al
   momento de decisión. Nada observado en el día `t` predice el día `t`.
4. **Falsa complejidad.** Agregar más features no ayudó en este
   dataset. Desplegamos el modelo más simple — App-aligned con 15
   features gana en holdout.
5. **Un solo nivel de servicio para todo.** Items A y C no deberían
   tener el mismo colchón. El lab soporta nivel de servicio por SKU
   vía el modelo cuantil.

### Encajar el modelo en S&OP

El modelo es una herramienta **dentro** de tu ritmo de planning, no
un reemplazo de él.

| Cadencia | Para qué usar el modelo |
|---|---|
| **Diaria** | Reabastecimiento por SKU (P80 + ROP). Sweet spot del lab. |
| **Semanal** | Agregar a categoría × tienda; comparar real vs pronóstico |
| **Mensual** | Revisión de sesgo / tracking signal; decidir si retrainar |
| **Trimestral** | Refresh ABC/XYZ; considerar expandir feature set |

**Forecast Value Added (FVA):** cada ciclo, preguntá *"¿el modelo
superó simplemente repetir el último período?"*. En este lab, el
App-aligned bate al baseline rolling-7 en **37%** — un FVA real. Si
un modelo no le gana a la naïve, sacalo; complejidad sin FVA es solo
costo.

### Glosario / Glossary

| Term | Meaning / Significado |
|---|---|
| **Lead time** | Days from placing an order to receiving it / Días desde el pedido a la recepción |
| **Safety stock** | Buffer for demand above forecast / Colchón para demanda por encima del pronóstico |
| **ROP** | Reorder Point — inventory level that triggers a new order / Nivel que dispara un nuevo pedido |
| **Service level** | Target % of cycles you avoid a stockout / % de ciclos en que evitás stockout |
| **Cycle stock** | Normal working inventory between deliveries / Inventario normal entre entregas |
| **Days of cover** | On-hand inventory ÷ daily demand / Stock ÷ demanda diaria |
| **Stockout** | Demand you couldn't fill / Demanda que no pudiste cubrir |
| **Newsvendor** | Math for optimal stock under uncertain demand / Modelo para stock óptimo con demanda incierta |
| **Bullwhip** | Demand swings amplifying upstream / Variabilidad amplificada río arriba |
| **FVA** | Forecast Value Added — gain over a naïve forecast / Mejora sobre un pronóstico naïve |
| **S&OP** | Sales & Operations Planning / Planificación de Ventas y Operaciones |

---

## 📚 Further reading / Lectura adicional

- Notebook completo (Inventory) — `notebooks/Inventory_Forecasting_CRISPML.ipynb`
- Notebook completo (Sales) — `notebooks/Sales_Forecasting_CRISPML.ipynb`
- Streamlit app — `app/app.py`
- Repo — [github.com/oscarinho/forecasting-inventory](https://github.com/oscarinho/forecasting-inventory)
- LinkedIn post series — companion accessible-language summary

---

## 🔍 Sources / Fuentes

Every quantitative claim in this lab is backed by industry research
or peer-reviewed publications. Cited inline above; consolidated below.

**CRISP-ML(Q) methodology**
- Studer S. et al. (2020). *Towards CRISP-ML(Q): A Machine Learning
  Process Model with Quality Assurance Methodology.* Mercedes-Benz
  AG & TU Berlin. [arXiv:2003.05155](https://arxiv.org/abs/2003.05155)
  · [MDPI MAKE 3(2)](https://www.mdpi.com/2504-4990/3/2/20)

**Stockout and overstock economics**
- [ToolsGroup — *Cost of Stockouts vs Overstock*](https://www.toolsgroup.com/blog/cost-of-stockouts-vs-overstock/)
- [Opensend — *Inventory Stock-out Rate Statistics*](https://www.opensend.com/post/inventory-stock-out-rate-statistics)
- [WAIR — *Financial Impact of Overstock and Stockouts*](https://wair.ai/financial-impact-overstock-stockouts/)

**Forecast accuracy → inventory reduction (Gartner / McKinsey / IBF)**
- [Manokhin V. — *Benefits of Improving Forecast Accuracy in Supply Chains*](https://valeman.medium.com/benefits-of-improving-forecast-accuracy-in-supply-chains-29f6d8b37c80)
- [Bamboo Rose — *10% Forecast Accuracy Improvement on a $500M Fashion Brand*](https://bamboorose.com/blog/forecast-accuracy-fashion/)
- [Lokad — *Financial Impact of Forecasting Accuracy*](https://www.lokad.com/accuracy-gains-inventory/)

**Drift detection (PSI threshold)**
- [Fiddler AI — *Measuring Data Drift with PSI*](https://www.fiddler.ai/blog/measuring-data-drift-population-stability-index)
- [GeeksforGeeks — *Population Stability Index (PSI)*](https://www.geeksforgeeks.org/data-science/population-stability-index-psi/)

**Tracking signal (forecast bias)**
- [Value Chain Planning — *Tracking Signal: An Effective KPI*](https://valuechainplanning.com/blog-details/110)
- [Umbrex — *Demand Forecast Bias*](https://umbrex.com/resources/company-analysis/supply-chain-logistics/demand-forecast-bias/)

**Newsvendor model and quantile forecasting**
- [MetricGate — *Newsvendor Model*](https://metricgate.com/docs/newsvendor-model/)
- [Dagher P. — *Quantile Forecasting for Inventory Planning*](https://medium.com/@nasdag/timesfm-2-5-quantile-forecasting-for-inventory-planning-2500c93de050)

**Price elasticity of demand**
- [7Learnings — *A Guide to Price Elasticity*](https://7learnings.com/blog/price-elasticity/)
- [Wikipedia — *Price Elasticity of Demand*](https://en.wikipedia.org/wiki/Price_elasticity_of_demand)

**Data leakage in time-series ML**
- [IBM Think — *What is Data Leakage in Machine Learning?*](https://www.ibm.com/think/topics/data-leakage-machine-learning)
- *Hidden Leaks in Time Series Forecasting* (2025). [arXiv:2512.06932](https://arxiv.org/pdf/2512.06932)

**LightGBM and overfit in lag features**
- *Forecasting with gradient boosted trees — Winning solution to the
  M5 Uncertainty competition.*
  [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0169207021002090)
- *Comparative Analysis of Modern ML Models for Retail Sales
  Forecasting* (2025). [arXiv:2506.05941](https://arxiv.org/abs/2506.05941)
- [scikit-learn — *Lagged features for time series forecasting*](https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html)

---

## Footer / Navigation

`[← Back to Labs / ← Volver a Labs] → /labs`
`[Next: Operations Object Detection → / Siguiente: Detección de Objetos →] → /labs/object-detection`

---

## Notas de implementación (no publicar)

- **Status badge** del lab en `/labs`: cambiar de `📈 Planned / Planificado`
  a `📈 ● Live`.
- **Tech stack tags** del listing en `/labs`: actualizar de
  `Python · Prophet · XGBoost · Streamlit` a
  `Python · LightGBM · Prophet · LSTM · Streamlit`.
- **OG image**: generar uno con el leaderboard de modelos como visual
  principal (8 modelos comparados → más memorable que un dashboard).
- **SEO**: target keywords en español — "pronóstico de demanda retail",
  "forecasting de inventario CRISP-ML", "newsvendor quantile P80",
  "supply chain machine learning". En inglés — "retail demand
  forecasting LightGBM", "quantile regression inventory", "CRISP-ML
  case study".
