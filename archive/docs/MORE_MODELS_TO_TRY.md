# Modelos adicionales para probar — código ejecutable

> Bloques listos para pegar como celdas en las libretas
> (`Inventory_Forecasting_CRISPML.ipynb` o `Sales_Forecasting_CRISPML.ipynb`).
> Cada bloque asume las variables y helpers que ya existen en la libreta:
> `X_train_s2`, `y_train`, `X_test_s2`, `y_test`, `cached()`, `eval_all()`,
> `results`, `preprocessor`, `train`, `test`, `FEATURE_COLS_STAGE2`.
>
> **Recomendación de orden:** Tier 1 primero (alto ROI, bajo esfuerzo).
> Tier 4 al final (opcional, solo si te queda tiempo).

---

## Tabla resumen — qué probar y en qué orden

| Tier | Modelo | Familia | Install | Tiempo de fit | Ganancia esperada vs LightGBM |
|---|---|---|---|---|---|
| 1 | CatBoost | Trees | `pip install catboost` | ~2 min | ±2% MAE (categóricas nativas) |
| 1 | HistGradientBoosting | Trees | sklearn (ya tienes) | ~30s | ±3% MAE |
| 1 | ExtraTrees | Trees | sklearn (ya tienes) | ~1 min | -5% MAE (peor) — sirve como robustness check |
| 1 | SARIMAX con exógenas | Classical | `statsmodels` (ya tienes) | ~5 min × N grupos | Cierra brecha clásicos vs ML |
| 1 | Theta method | Classical | `pip install statsforecast` | ~10s | Baseline rápido — M-competition winner histórico |
| 2 | NeuralProphet | DL | `pip install neuralprophet` | ~10 min | Sucesor neuronal de Prophet |
| 2 | N-BEATS | DL | `pip install neuralforecast` | ~20 min | Posible mejora real en Sales |
| 2 | NHITS | DL | `pip install neuralforecast` | ~15 min | Mejor que N-BEATS típicamente |
| 2 | DLinear / NLinear | DL | `pip install neuralforecast` | ~5 min | Baseline lineal sorprendentemente fuerte |
| 2 | DeepAR (global) | DL | `pip install neuralforecast` | ~30 min | El comparador justo del LSTM |
| 2 | TFT (Temporal Fusion Transformer) | DL | `pip install neuralforecast` | ~45 min | **Posible ganador real** en Sales |
| 3 | PatchTST | DL transformer | `pip install neuralforecast` | ~30 min | SOTA reciente en time-series |
| 3 | Chronos (Amazon) | Foundation | `pip install chronos-forecasting` | Zero-shot (~10 min inferencia) | Sin entrenamiento, prueba rápida |
| 3 | TimesFM (Google) | Foundation | `pip install timesfm` | Zero-shot | Idem, alternativa |
| 4 | Croston / TSB | Intermittent | `pip install statsforecast` | ~1 min | Solo si hay SKUs con >30% ceros |
| 4 | ADIDA / IMAPA | Intermittent | `pip install statsforecast` | ~1 min | Variantes de Croston |

---

# TIER 1 — Quick wins

## 1.1 CatBoost (Trees, categóricas nativas)

**Por qué probar:** maneja categóricas (`Region`, `Weather Condition`,
`Seasonality`, `Category`) **sin one-hot encoding**. En datasets con
muchas categóricas suele empatar o superar a LightGBM con código más
limpio.

```python
# Install (solo primera vez):
# !pip install catboost --quiet

from catboost import CatBoostRegressor

# CatBoost trabaja directo con strings — no necesita ColumnTransformer
CATEGORICAL_COLS = ['Category', 'Region', 'Weather Condition', 'Seasonality']

def fit_catboost():
    X_tr = train[FEATURE_COLS_STAGE2].copy()
    X_te = test[FEATURE_COLS_STAGE2].copy()
    # Asegurar tipo string en categóricas (CatBoost lo requiere)
    for c in CATEGORICAL_COLS:
        if c in X_tr.columns:
            X_tr[c] = X_tr[c].astype(str)
            X_te[c] = X_te[c].astype(str)

    model = CatBoostRegressor(
        iterations=600,
        learning_rate=0.05,
        depth=6,
        loss_function='MAE',
        cat_features=[c for c in CATEGORICAL_COLS if c in X_tr.columns],
        random_seed=42,
        verbose=False,
    )
    model.fit(X_tr, y_train)
    return model, X_te

cat_model, X_te_cat = cached('catboost_stage2', fit_catboost)
cat_pred = np.maximum(cat_model.predict(X_te_cat), 0)
results.append(eval_all('Stage 2: CatBoost (native categoricals)', y_test, cat_pred))
print(f'CatBoost MAE: {np.mean(np.abs(y_test - cat_pred)):.2f}')
```

---

## 1.2 HistGradientBoostingRegressor (sklearn — sin instalación)

**Por qué probar:** está en `sklearn` (ya instalado), maneja NaN sin
imputación, es 2-3× más rápido que LightGBM en datasets pequeños.

```python
from sklearn.ensemble import HistGradientBoostingRegressor

def fit_histgb():
    p = Pipeline([
        ('prep', preprocessor),
        ('model', HistGradientBoostingRegressor(
            max_iter=600,
            learning_rate=0.05,
            max_leaf_nodes=63,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=42,
        ))
    ])
    return p.fit(X_train_s2, y_train)

hgb_pipe = cached('histgb_stage2', fit_histgb)
hgb_pred = np.maximum(hgb_pipe.predict(X_test_s2), 0)
results.append(eval_all('Stage 2: HistGradientBoosting (sklearn)', y_test, hgb_pred))
```

---

## 1.3 ExtraTreesRegressor (sklearn — robustness check)

**Por qué probar:** segundo *bagging-based* check además de RandomForest.
Si ExtraTrees y RandomForest dan resultados muy distintos, tu varianza
de predicción es alta y hay margen para ensembling más sofisticado.

```python
from sklearn.ensemble import ExtraTreesRegressor

def fit_extratrees():
    p = Pipeline([
        ('prep', preprocessor),
        ('model', ExtraTreesRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42,
        ))
    ])
    return p.fit(X_train_s2, y_train)

et_pipe = cached('extratrees_stage2', fit_extratrees)
et_pred = np.maximum(et_pipe.predict(X_test_s2), 0)
results.append(eval_all('Stage 2: ExtraTrees', y_test, et_pred))
```

---

## 1.4 SARIMAX con variables exógenas (Classical fair-fight)

**Por qué probar:** es el ARIMA "justo" — incluye `Price`, `Discount`,
`Promotion`, `Holiday` como exógenas. Esperable: cerrar la brecha de
los métodos clásicos contra LightGBM (de MAE ~89 a ~75 en Inventario).

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

EXOG_COLS = ['Price', 'Discount', 'Promotion', 'Holiday/Promotion', 'Inventory Level']
EXOG_COLS = [c for c in EXOG_COLS if c in train.columns]

def fit_sarimax_per_group():
    preds = []
    groups = train.groupby(['Store ID', 'Category']) if 'Category' in train.columns \
             else train.groupby(['Store ID', 'Product ID'])
    for (s, c), grp in groups:
        try:
            y_tr = grp['Units Sold'].values if 'Units Sold' in grp else grp['Demand'].values
            X_tr = grp[EXOG_COLS].fillna(0).values
            test_grp = test[(test['Store ID'] == s) & ((test.get('Category', test['Product ID']) == c))]
            if len(test_grp) == 0:
                continue
            X_te = test_grp[EXOG_COLS].fillna(0).values

            model = SARIMAX(
                y_tr, exog=X_tr,
                order=(7, 0, 0),       # AR(7) sin diferenciación
                seasonal_order=(1, 0, 1, 7),  # estacionalidad semanal
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False, maxiter=50)

            yhat = model.forecast(steps=len(test_grp), exog=X_te)
            preds.append((test_grp.index, np.maximum(yhat, 0)))
        except Exception as e:
            continue
    # Reconstruir predicción alineada al test set
    pred = np.full(len(test), np.nan)
    for idx, yhat in preds:
        pred[idx] = yhat
    # Imputar grupos fallidos con la media
    pred[np.isnan(pred)] = y_train.mean()
    return pred

sarimax_pred = cached('sarimax_exog', fit_sarimax_per_group)
results.append(eval_all('Classical: SARIMAX with exogenous', y_test, sarimax_pred))
```

> ⚠️ SARIMAX es lento (5 min × N grupos). Si tienes >50 grupos,
> considera reducir a una muestra representativa.

---

## 1.5 Theta method (M-competition winner clásico)

**Por qué probar:** ganador histórico de las competencias M3, tan
simple que cabe en 3 líneas. Excelente baseline para comparar.

```python
# !pip install statsforecast --quiet

from statsforecast import StatsForecast
from statsforecast.models import Theta, AutoTheta

def fit_theta():
    # Preparar data en formato long que statsforecast espera
    df_sf = train[['Store ID', 'Product ID', 'Date']].copy()
    df_sf['unique_id'] = df_sf['Store ID'].astype(str) + '_' + df_sf['Product ID'].astype(str)
    df_sf['ds'] = pd.to_datetime(df_sf['Date'])
    df_sf['y'] = y_train.values

    sf = StatsForecast(
        models=[AutoTheta(season_length=7)],
        freq='D',
        n_jobs=-1,
    )
    sf.fit(df_sf[['unique_id', 'ds', 'y']])

    h = test['Date'].nunique()
    fcst = sf.predict(h=h)
    return fcst

theta_fcst = cached('theta_method', fit_theta)
# Hacer merge con test para alinear predicciones
test_with_pred = test.copy()
test_with_pred['unique_id'] = test_with_pred['Store ID'].astype(str) + '_' + test_with_pred['Product ID'].astype(str)
theta_merged = test_with_pred.merge(theta_fcst, on=['unique_id'], how='left')
theta_pred = np.maximum(theta_merged['AutoTheta'].fillna(y_train.mean()).values, 0)
results.append(eval_all('Classical: AutoTheta (M-comp baseline)', y_test, theta_pred))
```

---

# TIER 2 — Deep Learning moderno

> **Stack común:** todos los modelos de Tier 2 usan `neuralforecast`
> (librería de Nixtla, instala con `pip install neuralforecast`).
> Tienen API uniforme: `nf = NeuralForecast(models=[...], freq='D')`
> + `nf.fit(df)` + `nf.predict()`.

## 2.1 Setup común para neuralforecast

```python
# !pip install neuralforecast --quiet

from neuralforecast import NeuralForecast
from neuralforecast.models import (
    NBEATS, NHITS, DLinear, NLinear, DeepAR, TFT, PatchTST,
)
from neuralforecast.losses.pytorch import MAE

# Preparar dataframe en formato Nixtla (unique_id, ds, y)
df_long = train[['Store ID', 'Product ID', 'Date']].copy()
df_long['unique_id'] = df_long['Store ID'].astype(str) + '_' + df_long['Product ID'].astype(str)
df_long['ds'] = pd.to_datetime(df_long['Date'])
df_long['y'] = y_train.values
df_long = df_long[['unique_id', 'ds', 'y']]

H = test['Date'].nunique()  # horizonte = días en test
INPUT_SIZE = 28              # ventana de contexto
```

## 2.2 N-BEATS

```python
def fit_nbeats():
    nf = NeuralForecast(
        models=[NBEATS(
            h=H, input_size=INPUT_SIZE,
            loss=MAE(),
            max_steps=500,
            batch_size=64,
            random_seed=42,
        )],
        freq='D',
    )
    nf.fit(df_long)
    return nf

nbeats_nf = cached('nbeats', fit_nbeats)
nbeats_fcst = nbeats_nf.predict()
# Merge con test...
nbeats_pred = _merge_long_predictions(test, nbeats_fcst, 'NBEATS')
results.append(eval_all('DL: N-BEATS (Nixtla)', y_test, nbeats_pred))
```

## 2.3 NHITS (sucesor mejorado de N-BEATS)

```python
def fit_nhits():
    nf = NeuralForecast(
        models=[NHITS(
            h=H, input_size=INPUT_SIZE,
            loss=MAE(),
            max_steps=500,
            batch_size=64,
            random_seed=42,
        )],
        freq='D',
    )
    nf.fit(df_long)
    return nf

nhits_nf = cached('nhits', fit_nhits)
nhits_fcst = nhits_nf.predict()
nhits_pred = _merge_long_predictions(test, nhits_fcst, 'NHITS')
results.append(eval_all('DL: NHITS (Nixtla)', y_test, nhits_pred))
```

## 2.4 DLinear / NLinear (baselines lineales sorprendentemente fuertes)

```python
def fit_dlinear():
    nf = NeuralForecast(
        models=[DLinear(h=H, input_size=INPUT_SIZE, max_steps=300, random_seed=42)],
        freq='D',
    )
    nf.fit(df_long)
    return nf

dlinear_nf = cached('dlinear', fit_dlinear)
dlinear_pred = _merge_long_predictions(test, dlinear_nf.predict(), 'DLinear')
results.append(eval_all('DL: DLinear (linear baseline)', y_test, dlinear_pred))
```

## 2.5 DeepAR (Amazon — probabilístico, global)

**Por qué probar:** es el "LSTM hecho bien" — global multi-serie con
parámetros compartidos. Es el comparador justo del LSTM que tienes.

```python
def fit_deepar():
    nf = NeuralForecast(
        models=[DeepAR(
            h=H, input_size=INPUT_SIZE,
            max_steps=500,
            batch_size=64,
            random_seed=42,
        )],
        freq='D',
    )
    nf.fit(df_long)
    return nf

deepar_nf = cached('deepar', fit_deepar)
deepar_pred = _merge_long_predictions(test, deepar_nf.predict(), 'DeepAR')
results.append(eval_all('DL: DeepAR (Amazon — global)', y_test, deepar_pred))
```

## 2.6 TFT — Temporal Fusion Transformer (Google)

**Por qué probar:** SOTA en muchos benchmarks. Interpretable
(attention sobre features). Maneja variables exógenas estáticas y
dinámicas. **Mi favorito como contender real del LightGBM.**

```python
def fit_tft():
    nf = NeuralForecast(
        models=[TFT(
            h=H, input_size=INPUT_SIZE,
            hidden_size=64,
            n_head=4,
            max_steps=500,
            batch_size=32,
            random_seed=42,
        )],
        freq='D',
    )
    nf.fit(df_long)
    return nf

tft_nf = cached('tft', fit_tft)
tft_pred = _merge_long_predictions(test, tft_nf.predict(), 'TFT')
results.append(eval_all('DL: TFT (Google — transformer + attention)', y_test, tft_pred))
```

## 2.7 Helper común — `_merge_long_predictions()`

Pegar **una sola vez** antes de los modelos Tier 2:

```python
def _merge_long_predictions(test_df, fcst_df, model_col):
    """Une predicciones en formato Nixtla (unique_id, ds, model) con el test set."""
    test_aligned = test_df.copy()
    test_aligned['unique_id'] = test_aligned['Store ID'].astype(str) + '_' + test_aligned['Product ID'].astype(str)
    test_aligned['ds'] = pd.to_datetime(test_aligned['Date'])
    merged = test_aligned.merge(fcst_df, on=['unique_id', 'ds'], how='left')
    pred = merged[model_col].fillna(y_train.mean()).values
    return np.maximum(pred, 0)
```

---

# TIER 3 — Foundation models (zero-shot)

> No requieren entrenamiento. Cargas el modelo pre-entrenado y predice
> directamente sobre tus series. Útil para benchmarking y para cuando
> no tienes datos suficientes.

## 3.1 Chronos (Amazon, basado en T5)

```python
# !pip install chronos-forecasting --quiet

from chronos import ChronosPipeline
import torch

pipeline = ChronosPipeline.from_pretrained(
    "amazon/chronos-t5-small",   # también: -base, -large
    device_map="cpu",            # cambiar a "cuda" si hay GPU
    torch_dtype=torch.float32,
)

def chronos_forecast():
    preds = []
    for uid, grp in train.assign(uid=train['Store ID'].astype(str) + '_' + train['Product ID'].astype(str)).groupby('uid'):
        context = torch.tensor(grp['Units Sold'].values, dtype=torch.float32)
        fcst = pipeline.predict(context=context, prediction_length=H, num_samples=20)
        median = fcst.median(dim=1).values.squeeze().numpy()
        preds.append((uid, median))
    return preds

chronos_preds = cached('chronos_zeroshot', chronos_forecast)
# Reconstruir alineado al test...
# (mismo merge helper)
chronos_pred_aligned = ... # implementar merge según índices
results.append(eval_all('Foundation: Chronos (Amazon, zero-shot)', y_test, chronos_pred_aligned))
```

## 3.2 TimesFM (Google)

```python
# !pip install timesfm --quiet

import timesfm

tfm = timesfm.TimesFm(
    context_len=128, horizon_len=H,
    input_patch_len=32, output_patch_len=128,
    num_layers=20, model_dims=1280,
    backend='cpu',  # o 'gpu'
)
tfm.load_from_checkpoint(repo_id="google/timesfm-1.0-200m")

# Similar a Chronos: predice por serie, mergear con test
```

---

# TIER 4 — Demanda intermitente (Croston y variantes)

> **Pre-requisito:** verifica si tu dataset tiene SKUs con >30% de
> días con `Units Sold == 0` o `Demand == 0`. Si no los tiene,
> Tier 4 no aplica.

## 4.0 Diagnóstico previo

```python
# ¿Hay SKUs intermitentes?
zero_rate = (
    train.assign(uid=train['Store ID'].astype(str) + '_' + train['Product ID'].astype(str))
         .groupby('uid')[y_train.name if hasattr(y_train, 'name') else 'Demand']
         .apply(lambda x: (x == 0).mean())
)
intermittent = zero_rate[zero_rate > 0.30]
print(f'SKUs intermitentes (>30% ceros): {len(intermittent)} de {len(zero_rate)}')
print(f'Si es 0, saltar Tier 4.')
```

## 4.1 Croston (clásico para intermittent demand)

```python
from statsforecast.models import CrostonClassic, CrostonOptimized, CrostonSBA, ADIDA, IMAPA, TSB

def fit_croston_family():
    df_sf = train[['Store ID', 'Product ID', 'Date']].copy()
    df_sf['unique_id'] = df_sf['Store ID'].astype(str) + '_' + df_sf['Product ID'].astype(str)
    df_sf['ds'] = pd.to_datetime(df_sf['Date'])
    df_sf['y'] = y_train.values

    # Filtrar solo SKUs intermitentes
    df_sf = df_sf[df_sf['unique_id'].isin(intermittent.index)]

    sf = StatsForecast(
        models=[
            CrostonClassic(),
            CrostonOptimized(),
            CrostonSBA(),
            ADIDA(),
            IMAPA(),
            TSB(alpha_d=0.2, alpha_p=0.2),
        ],
        freq='D',
        n_jobs=-1,
    )
    sf.fit(df_sf[['unique_id', 'ds', 'y']])
    return sf.predict(h=H)

croston_fcst = cached('croston_family', fit_croston_family)

# Evaluar cada variante por separado
for model_name in ['CrostonClassic', 'CrostonOptimized', 'CrostonSBA', 'ADIDA', 'IMAPA', 'TSB']:
    pred = _merge_long_predictions(test, croston_fcst, model_name)
    # Solo evaluar sobre el subset intermitente
    test_idx = test.assign(uid=test['Store ID'].astype(str) + '_' + test['Product ID'].astype(str))
    mask = test_idx['uid'].isin(intermittent.index)
    if mask.sum() > 0:
        results.append(eval_all(
            f'Intermittent ({model_name}, n={mask.sum()})',
            y_test[mask], pred[mask]
        ))
```

---

# Cómo comparar todos al final

Una vez agregados todos los modelos al `results`, regenera el
leaderboard:

```python
leaderboard = pd.DataFrame(results).sort_values('MAE').reset_index(drop=True)
print(leaderboard.to_string(index=False))

# Visualización
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, max(6, len(leaderboard) * 0.35)))
ax.barh(leaderboard['model'], leaderboard['MAE'], color='#C9A86A')
ax.set_xlabel('MAE (lower is better)')
ax.set_title('Full Model Leaderboard — All Families')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('full_leaderboard.png', dpi=120, bbox_inches='tight')
plt.show()
```

---

# Recomendación de ruta práctica

Si tienes **una tarde** disponible: corre solo Tier 1 (5 modelos,
~10 min total). Vas a obtener evidencia robusta de si LightGBM
realmente es el ganador.

Si tienes **un día completo**: agrega Tier 2 (N-BEATS, NHITS, DLinear,
DeepAR, TFT). Ahí está la mayor probabilidad de superar a LightGBM —
sobre todo TFT en el dataset de Ventas.

Si tienes **un fin de semana**: suma Tier 3 (foundation models). Es
útil más por curiosidad metodológica que por accuracy.

Solo agregues Tier 4 **si el diagnóstico de intermitencia da >0** —
si no, no aplica.

---

## Notas operativas

- **GPU:** Tier 2 entrena mucho más rápido con GPU. Si solo tienes CPU,
  reduce `max_steps` a 200 y `batch_size` a 32.
- **Reproducibilidad:** todos los modelos arriba tienen `random_seed=42`
  donde la API lo permite.
- **Caching:** todos los `fit_*()` están envueltos en `cached()` — la
  segunda corrida toma segundos.
- **Si un modelo falla:** envuelve en `try/except` y guarda `None`; el
  bloque siguiente sigue funcionando. No detengas todo el pipeline por
  un solo modelo.
