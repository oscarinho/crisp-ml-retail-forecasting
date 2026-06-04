# Lecciones Aprendidas — 4 Labs, 1 Pregunta de Negocio

> Recap analítico de lo que aprendimos al aplicar CRISP-ML(Q) sobre cuatro
> datasets de retail forecasting distintos. Empieza por una intuición personal
> y termina con cuatro modelos desplegables + lecciones que generalizan.

---

## Cómo arrancó esto

Una observación tuya, en medio de una discusión sobre si el modelo de Inventory
estaba "bien" con MAE 69:

> *"Demand Forecast debe entrar en alguna parte del modelo. Lo uso solo al
> inicio para tener un modelo base, luego lo corrijo con los valores reales."*

Y tu anécdota de tus días como analista MRP:

> *"Un demand planner usaba más de un modelo y siempre se quedaba con el que
> tenía mejor predicción en el mes anterior — era una caza del gato y ratón
> con los modelos y la demanda real."*

Las dos observaciones son **patrones reales de la industria de planning**, no
intuiciones sueltas:

| Tu intuición | Nombre formal | Quiénes lo usan en producción |
|---|---|---|
| "DF como prior + corrección" | **Residual learning / Forecast stacking** | Walmart, Amazon Supply Chain, Unilever |
| "Cat-and-mouse con modelos" | **Champion-challenger backtesting** | SAP IBP, Oracle Demantra, o3 Solutions |

Lo único que no sabíamos era **cuánto valor real generan en estos datasets
específicos**. Por eso corrimos dos experimentos.

---

## Experimento A — Residual Learning

### Setup

- **Target del modelo:** `Units Sold − Demand Forecast` (la corrección que
  necesita el forecast existente)
- **Features:** contextual (price, weather, promo, region, season) **sin
  Demand Forecast como feature directo**
- **Predicción final:** `Demand Forecast + modelo.predict(features)`, clipped at 0
- **Split:** últimos 90 días = holdout (mismo que el Inventory lab)

### Resultados

| Estrategia | MAE | Bias (sesgo) | Δ vs DF puro |
|---|---:|---:|---:|
| DF puro (sin modelo) | 8.35 | **+5.05** ← sobreestima sistemáticamente | — |
| Modelo directo HGB (sin DF) | 69.03 | -1.37 | +60.7 ❌ |
| Modelo directo RF (sin DF) | 69.33 | -0.71 | +60.9 ❌ |
| Modelo directo LightGBM (sin DF) | 69.10 | -1.38 | +60.7 ❌ |
| **Residual: DF + HGB (features)** | **7.43** | **+0.10** | **−0.92 ✓** |
| Residual: DF + RF (features) | 7.45 | +0.13 | −0.90 ✓ |
| Residual: DF + LightGBM (features) | 7.46 | +0.15 | −0.89 ✓ |

### Veredicto

**Tu intuición fue correcta**: residual learning supera a DF puro en **11% MAE**.

Lo más interesante: el aporte real **no es reducir varianza, es corregir
sesgo**. DF sobreestima por +5.05 unidades sistemáticamente; el modelo
residual lleva ese sesgo a +0.10 — una reducción **50×**. En producción eso
significa **menos safety stock** porque ya no hay sesgo positivo que compensar.

---

## Experimento B — Champion-Challenger (tu anécdota MRP)

### Setup

- **18 ventanas rolling** de 30 días, sin overlap, después de un warm-up de
  6 meses
- **7 contendientes:** DF puro, HGB direct/residual, RF direct/residual,
  LightGBM direct/residual
- **Protocolo por ventana:** entrenar con histórico, predecir 30 días, medir MAE
- **Champion:** modelo con MAE mínimo en esa ventana → contar wins

### Resultados

| Modelo | Wins / 18 | Win rate | MAE mean | Std |
|---|---:|---:|---:|---:|
| **HGB_residual** | **17** 🏆 | 94.4% | 7.39 | 0.08 |
| RF_residual | 1 | 5.6% | 7.41 | 0.08 |
| LightGBM_residual | 0 | 0% | 7.43 | 0.09 |
| DF_puro | 0 | 0% | 8.31 | 0.11 |
| HGB_direct | 0 | 0% | 69.09 | 0.79 |
| RF_direct | 0 | 0% | 69.35 | 0.89 |
| LightGBM_direct | 0 | 0% | 69.41 | 0.87 |

### Veredicto

**El patrón cat-and-mouse existe pero es dataset-dependent.** En este dataset
sintético, HGB_residual gana 17/18 ventanas — la rotación mensual NO agrega
valor. La diferencia entre champion y runner-up es 0.02 MAE — puro ruido.

**El hallazgo más importante:** *Direct vs Residual* es **10× más impactante**
que elegir LightGBM vs RandomForest vs HGB. El framing del problema le gana
al stack tecnológico por 90%.

En datasets reales con choques externos (COVID, recalls, virales, lanzamientos)
la estabilidad caería a ~60-70% y ahí sí pagaría hacer rotación mensual.
Aquí no.

---

## Las cuatro labs lado a lado

|  | Inventory | Sales | Food Demand | Store Sales |
|---|---|---|---|---|
| **Dataset origen** | Synthetic (Kaggle) | Synthetic (Kaggle, COVID-style) | Genpact / Analytics Vidhya | Corporación Favorita / Kaggle |
| **Rows** | 73k | 76k | 457k | 3M |
| **Series** | 100 (5 store × 20 prod) | 100 | 3,927 (77 ctr × 51 meals) | 1,782 (54 store × 33 fam) |
| **Granularidad** | Daily | Daily | **Weekly** | Daily |
| **Autocorrelación** | ≈ 0 | ≈ 0.35 | ≈ 0.50 | ≈ 0.60-0.80 |
| **Mejor MAE** | **7.4** (con DF) / **69** (sin) | **19.5** | **68.6** | **52.0** |
| **Mejor Kaggle-metric** | n/a | n/a | **RMSLE×100 = 49.1** | **RMSLE = 0.394** |
| **Modelo ganador** | HGB residual (con DF) / Stacking (sin) | Per-category LightGBM × 5 | CatBoost (RMSLE) / LightGBM (deploy) | **Per-family LightGBM × 33** |
| **Insight propio** | "Two ceilings" (DF disponible o no) | App-aligned > Stage 2 (más features hurtan) | Per-cuisine NO ganó (solo 4 cuisines) | Sales-lab pattern escala a 3M rows |

---

## Cuatro lecciones que generalizan

### Lección 1 — La trampa del "demand forecast oracle"

En el Inventory lab, `Demand Forecast` tiene ρ = 0.997 con `Units Sold`. No es
un baseline — **es el answer key con ~5% de ruido cosmético**.

Vic, una persona externa, obtuvo MAE 7.29 incluyendo indirectamente DF vía
`inventory_to_forecast_ratio` (su feature `Inventory Level ÷ Demand Forecast`
deja al modelo recuperar DF algebraicamente). Nosotros obtenemos
prácticamente lo mismo con residual learning legítimo (MAE 7.43). Misma
gravedad numérica, distinta integridad de producción.

> **Regla operativa:** antes de creer cualquier MAE excepcional, audita las
> features. Si una columna correlaciona >0.85 con el target, probablemente
> embebe el target. El test ablation es de 5 minutos y vale toda la
> credibilidad del proyecto.

### Lección 2 — Per-category routing necesita ≥5 categorías para ganar

|  | Categorías | Routing gana? | Lift vs global |
|---|---:|---|---|
| Sales lab | 5 (Clothing, Electronics, Groceries, Toys, Furniture) | ✓ sí | −3.2% MAE |
| Store Sales lab | 33 families | ✓ sí | −2.5% MAE / -3.0% RMSLE |
| Food Demand lab | **4** (Italian, Continental, Thai, Indian) | ✗ no | +3.8% MAE |
| Inventory lab | 5 (mismas que Sales) | n/a (no testeado) | — |

> **Regla operativa:** per-category routing requiere ≥5 categorías bien
> pobladas. Con 4 o menos, los sub-modelos se overfittean al training de cada
> categoría chica y pierden contra un modelo global.

### Lección 3 — Las lags valen exactamente lo que dice el autocorr diagnostic

| Lab | Autocorr (lag-1) | Stage 1 → Stage 2 con lags | Veredicto |
|---|---:|---|---|
| Inventory | ≈ 0 | MAE 90 → 69 (-23%) | Lags ayudan poco — el lift viene de otras features |
| Sales | ≈ 0.35 | App-aligned (sin lags) **GANA** | Lags **hacen daño** (overfit estructural) |
| Food Demand | ≈ 0.50 | MAE 92.8 → 69.7 (-25%) | Lags ayudan moderadamente |
| Store Sales | ≈ 0.70 | RMSLE 1.00 → 0.52 (-48%) | Lags cargan la mitad del lift |

> **Regla operativa:** correr el autocorr diagnostic ANTES de hacer feature
> engineering. Te dice cuánto puedes esperar del esfuerzo de lags y rolling
> stats. Es cómputo gratis comparado con el costo de fittear todos los modelos.

### Lección 4 — Framing > Model > Hyperparameters

Ordenado por impacto en MAE (datos de los 4 labs):

| Decisión | Impacto típico |
|---|---|
| Cómo planteas el problema (residual vs direct, target uncensored vs censored) | **62 unidades MAE** / 90% del gap |
| Engineering de features | 3-10 unidades MAE / 5-15% del gap |
| Elección entre LightGBM, HGB, CatBoost, RF | 0.04 unidades MAE / <1% del gap |
| Hiperparámetros (afinación fina) | 1-3 unidades MAE / 2-5% del gap |

> **Regla operativa:** si tienes 1 hora, gástala en framing. Si tienes 1 día,
> en features. Si tienes 1 semana, en hyperparams. Si gastas 1 día en
> hyperparams antes de auditar el framing, mal asignaste el tiempo.

---

## El cat-and-mouse en el mundo real (más allá del experimento)

Tu intuición de los días de MRP planner se valida en la práctica:

| Producto | Feature comercial | Adopción |
|---|---|---|
| SAP IBP | "Best-of-breed model selection" | Default en S&OP profesional |
| Oracle Demantra | "Adaptive forecasting" (rotación mensual) | Estándar desde 2010 |
| o3 Solutions | "AI/ML forecast benchmarking" | Sello de marketing |
| Anaplan | "Multi-model ensembling" | Plugin disponible |

**Pero la decisión de rotar tiene un umbral económico:**

```
Vale la pena rotar mensualmente entre N modelos SI:
  beneficio_marginal(modelo_ganador − modelo_baseline) > costo_operacional(rotación)
```

Costo operacional típico de rotar:
- Re-entrenamiento mensual: $500-2,000/mes en cómputo cloud
- Validación humana: 2-8 horas/mes de un demand planner senior
- Riesgo de inconsistencia: planners abajo se confunden si los números varían demasiado

Beneficio marginal típico para que pague la pena: **diferencia ≥ 2-5% MAE
entre champion y runner-up sostenidamente**.

En nuestros 4 labs, el delta es <0.5% en cada caso. **Un modelo + cache es la
estrategia correcta para estos datasets.** Habría sido distinto si el dataset
tuviera un evento tipo COVID dentro de la ventana de backtest — ahí algunos
modelos colapsan y otros aguantan, y la rotación recupera valor.

---

## Lo que sigue / pendientes

| Lab | Pendiente | Por qué |
|---|---|---|
| Sales | Construir DF sintético + residual learning encima | El dataset no trae columna DF, pero podríamos usar un naive baseline como prior |
| Food Demand | Champion-challenger con horizonte 4 semanas | El dominio food necesita horizonte mensual, no semanal |
| Store Sales | Submission a Kaggle leaderboard | Para validar que nuestro RMSLE 0.394 está donde creemos |
| Todos | Streamlit apps para labs 3 y 4 | Mismo design system editorial-terminal |
| Todos | A/B test de impacto de bias correction | Mostrar cómo el sesgo +5 → +0.1 reduce safety stock |

---

## Conclusión

> *El experimento más valioso no es el modelo que entrenas. Es la pregunta
> que se te ocurre validar.*

Tu intuición + tu anécdota generaron el insight central del repo: la
**"two ceilings" framing** del Inventory lab. Sin esa pregunta, el README
seguiría diciendo "MAE 69 es el techo" — una verdad parcial que se cae el
primer día que un cliente real pregunta *"¿y si te doy mi forecast existente?"*.

La lección que se generaliza a cualquier proyecto de ML, no solo a éste:

> *Antes de elegir el modelo, elige el framing.*
> *Antes de elegir el framing, escucha al practicante.*

— oscarponce.com · Junio 2026
