# Validación de claims — LinkedIn Posts + Lab Page

Auditoría de cada afirmación cuantitativa o conceptual hecha en
`LINKEDIN_POSTS.md` y `LAB_PAGE_demand_forecasting.md` contra fuentes
externas. Para cada claim: status, fuente, y ajuste recomendado (si
aplica).

Status:
- ✅ **Validado** — confirmado por fuentes externas, mantener.
- ⚠️ **Ajustar** — afirmación esencialmente correcta pero formulación
  inexacta o necesita matiz; cambio menor.
- ❌ **Corregir** — afirmación incorrecta o no soportada por fuentes;
  cambio sustantivo.

---

## 1. CRISP-ML(Q) como metodología

**Claim:** "CRISP-ML(Q) obliga a parar antes de modelar… seis fases…
fase Q de quality assurance."

**Status:** ✅ Validado.

**Fuente principal:**
- Studer et al. (2020). *Towards CRISP-ML(Q): A Machine Learning Process
  Model with Quality Assurance Methodology.* arXiv 2003.05155 / MDPI
  *Machine Learning and Knowledge Extraction* 3(2). Publicado por
  Mercedes-Benz AG y TU Berlin.

**Detalle:** Las 6 fases (Business Understanding, Data Understanding,
Data Preparation, Modeling, Evaluation, Deployment) y el componente Q
de quality assurance están explícitamente definidos en el paper. La
descripción del notebook coincide con la metodología publicada.

---

## 2. Costo de stockout vs overstock — la "asimetría 3–5×"

**Claim original (lab page):** "Stockout cost typically 3–5× more than
overstock when you include lost margin, customer churn, reputational
damage."

**Status:** ⚠️ Ajustar — la asimetría es real, pero el ratio "3–5×" no
aparece como cifra estándar de industria. Lo más defendible es citar
los dos lados por separado.

**Fuentes:**
- ToolsGroup — *Cost of Stockouts vs Overstock*: stockouts cuestan
  4–8% de ventas anuales en retail.
- Industry data (Opensend, 2024): inventory distortion (stockouts +
  overstock) cost retailers **$1.77T en 2023, ~7.2% del retail global.**
- Carrying cost de overstock: **20–30% del valor del inventario por
  año** (industria estándar, múltiples fuentes).
- Churn de cliente: **9% se cambia permanentemente a un competidor
  tras un solo stockout; 55% tras múltiples experiencias.**

**Ajuste recomendado:** Reemplazar el "3–5×" por:
> "Stockouts cost retailers 4–8% of annual sales [ToolsGroup]; overstock
> carrying costs are 20–30% of inventory value annually. 9% of customers
> permanently switch to a competitor after a single stockout, rising to
> 55% after multiple stockouts [Opensend, 2024]. The asymmetry is what
> drives the use of quantile (not mean) forecasts for inventory."

---

## 3. PSI > 0.2 como trigger de retrain

**Claim:** "Trigger retrain when drift exceeds 0.2 on any monitored feature."

**Status:** ✅ Validado — es estándar de industria.

**Fuentes:**
- *Population Stability Index (PSI)* — Medium / GeeksforGeeks /
  Fiddler AI / machinelearningplus, todos convergen en:
  - PSI < 0.1 → sin cambio significativo
  - 0.1 ≤ PSI < 0.2 → cambio leve, investigar
  - **PSI ≥ 0.2 → cambio significativo, retrain recomendado**

---

## 4. Tracking signal > 4 como trigger de retrain

**Claim:** "|tracking signal| > 4 → retrain immediately."

**Status:** ✅ Validado — con matiz.

**Fuente:**
- *Tracking Signal: An Effective KPI* — Value Chain Planning.
- *Demand Forecast Bias* — Umbrex Supply Chain.

**Detalle:** El umbral exacto es **3.75 MAD** (derivado del 3-sigma a
99% de nivel de servicio); la industria redondea a 4 por
conveniencia. Algunas fuentes usan ±4 a ±6 como ventana. La regla
"|TS| > 4 → retrain" es defendible y comúnmente citada.

**Ajuste opcional:** mencionar el 3.75 como umbral estadístico
estricto para mostrar dominio técnico.

---

## 5. Newsvendor model + cuantil P80

**Claim:** "P80 quantile model replaces static `Z × σ × √L` with one
that adapts to current conditions."

**Status:** ✅ Validado.

**Fuentes:**
- *Newsvendor Model Calculator* — MetricGate.
- Vskills tutorial on Newsvendor Model.
- *TimesFM 2.5 Quantile Forecasting for Inventory Planning* —
  Dagher, Medium.

**Detalle:** El newsvendor problem es operations research clásico
(Arrow, Harris, Marschak 1951). El enfoque moderno de "entrenar el
cuantil directamente al nivel de servicio objetivo" en vez de
"forecast + safety stock estático" está bien documentado en la
literatura de inventory planning con ML. Mi descripción es correcta.

---

## 6. Elasticidad de precio en retail — "ρ entre -0.3 y -0.7"

**Claim original:** "Real retail data shows ρ between -0.3 and -0.7."

**Status:** ❌ Corregir — la formulación es técnicamente incorrecta.

**Problema:** Estoy mezclando dos conceptos:
- **ρ (correlación)** entre `Price` y `Demand` — métrica que el notebook
  reporta (~0 en los datasets sintéticos).
- **Elasticidad-precio de la demanda** — métrica económica
  estándar (∂lnQ/∂lnP). Es lo que aparece en la literatura de retail.

Estos no son lo mismo. La correlación es lineal y unidimensional; la
elasticidad es un coeficiente económico que captura sensibilidad
porcentual.

**Fuente:**
- Wikipedia *Price Elasticity of Demand* y Conjointly *Guide to Price
  Elasticity*: típicamente entre **-0.5 (inelástica) y -2 (elástica)**
  según categoría, con la mayoría de productos retail entre **-0.7 y
  -1.5**.
- 7Learnings *Guide to Price Elasticity*: confirma estos rangos.

**Ajuste recomendado:** Reformular en lab page y posts como:
> "Real retail data shows price elasticity typically between -0.5
> (inelastic categories like staples) and -2 (elastic categories like
> discretionary goods). Even after converting to correlation, you'd
> expect at least ρ < -0.2 — the synthetic dataset shows ρ ≈ 0,
> confirming it lacks any pricing mechanism."

---

## 7. Forecast accuracy → reducción de inventario

**Claim:** "10–15% working capital reduction on stocked inventory."

**Status:** ✅ Validado — conservador.

**Fuentes (múltiples convergentes):**
- **Gartner:** cada 1% de mejora en forecast accuracy → **2.7%
  reducción en finished goods inventory**, 3.2% en costos de
  transporte, 3.9% en obsolescencia.
- **Institute of Business Forecasting and Planning:** 1% de mejora
  → 1.5% reducción en costos de inventario + 2% mejora en service
  level.
- **McKinsey:** 10–20% de mejora en forecast accuracy → ~5%
  reducción en costos de inventario.
- **Bamboo Rose** (caso fashion retail): 10% de mejora →
  **$25M USD liberados de working capital** en una marca de $500M.

**Detalle:** Mi "10–15%" como reducción de capital atrapado es
conservador comparado con las cifras top de industria (que llegan
a 20–30% en algunos casos). Defendible como rango cauto.

---

## 8. "Stockout reduction: 20–35%" con P80 quantile

**Claim:** "Estimated stockout reduction: 20–35%."

**Status:** ⚠️ Ajustar — el rango es plausible pero falta cita directa
sobre P80 específicamente.

**Fuentes parciales:**
- ToolsGroup: implementaciones de quantile forecasting reportan
  reducciones de stockout de 15–40% según categoría.
- Nestlé case study (citado en HighRadius): demand sensing redujo
  forecast error 40% → 2% mejora en perfect order fulfillment + **5%
  reducción de inventario**.

**Ajuste recomendado:** suavizar a "stockout reduction of 15–30%
reported in industry case studies of quantile-based replenishment
[ToolsGroup]" — y enlazar a un case study real en lugar de mantener
mi cifra propia.

---

## 9. Estimación "$1.5M–$4M USD savings for 50 stores / 5,000 SKUs"

**Claim:** "$1.5M – $4M USD annually."

**Status:** ⚠️ Ajustar — calcular explícitamente vs. quoting un rango
suelto.

**Cálculo defendible (basado en Gartner + benchmarks):**

```
Asunciones:
- 50 stores × 5,000 SKUs = 250,000 (Store × Product) combinations
- Inventario promedio por combinación: $200
- Inventario total stockeado: $50M USD
- Carrying cost rate: 25% (centro del rango 20–30%)
- Carrying cost anual: $12.5M USD

Si mejora de forecast accuracy → 10% reducción de inventario (Gartner
2.7% × ~4 puntos de mejora, conservador):
- Reducción de inventario: $5M USD
- Ahorro anual en carrying cost: $5M × 0.25 = $1.25M USD/año

Si la reducción llega al 20% (top de industria):
- Reducción: $10M USD
- Ahorro: $10M × 0.25 = $2.5M USD/año

Adicional por menos stockouts (4–8% de ventas; asumir $200M ventas
anuales en 50 tiendas):
- Stockout cost actual: $8M–$16M/año
- Reducción del 20%: $1.6M–$3.2M/año
```

**Ajuste recomendado:** Cambiar el rango a **"$1M – $3M USD
annually"** con el cálculo arriba como anexo, citando Gartner como
fuente de la elasticidad forecast accuracy → inventory.

---

## 10. Data leakage como problema central en forecasting

**Claim:** "Es el error más caro en forecasting… rompe modelos en
producción aunque luzcan brillantes en el notebook."

**Status:** ✅ Validado.

**Fuentes:**
- IBM Think — *What is Data Leakage in Machine Learning?*
- Machine Learning Mastery — *Data Leakage in Machine Learning.*
- arXiv 2512.06932 (2025) — *Hidden Leaks in Time Series Forecasting:
  How Data Leakage Affects LSTM Evaluation Across Configurations and
  Validation Strategies.*

**Detalle:** El paper de 2025 confirma que la leakage en time-series
sigue siendo un error frecuente incluso en publicaciones recientes,
con métricas optimistas en evaluación que colapsan en producción.
Mi framing es exacto.

---

## 11. Modelo simple gana en producción — overfit en lag features

**Claim:** "El modelo más complejo casi nunca es el mejor en
producción. App-aligned con 15 features venció a Stage 2 con 29."

**Status:** ✅ Validado como patrón general.

**Fuentes:**
- *Forecasting with gradient boosted trees: ... Winning solution to
  the M5 Uncertainty competition.* ScienceDirect. Discute
  explícitamente overfit en lag features.
- arXiv 2506.05941 — *Comparative Analysis of Modern Machine
  Learning Models for Retail Sales Forecasting.* Confirma que
  LightGBM con feature engineering parsimoniosa supera a modelos
  más complejos.
- scikit-learn — *Lagged features for time series forecasting*.

**Detalle:** En la competencia M5 de Walmart (gold standard), la
solución ganadora usó gradient boosting con cuidado en la selección
de lag features. Mi resultado replica este patrón a menor escala.

---

## 12. MAE 20.1 en retail forecasting — ¿es competitivo?

**Claim implícito:** "MAE 20.1 en el dataset de Ventas es bueno."

**Status:** ✅ Defendible con contexto.

**Fuentes (rangos de industria):**
- M5 Walmart (gold standard): MAE varía mucho por SKU; SKUs de alto
  volumen MAE ~10, SKUs de bajo volumen MAE ~3.
- *Comparative Analysis of ML Models for Retail Sales Forecasting*
  (arXiv 2506.05941): LightGBM MAE 2,325 en sales tier alto.
- Industry MAPE benchmark: 10–30%. Mi 20.1 sobre demanda media de
  ~80 unidades implica MAPE ~25%, dentro del rango aceptable para
  forecasting diario por SKU.

**Detalle:** MAE absoluto no es comparable entre datasets (escala
diferente). El benchmark relevante es la mejora vs baseline en el
**mismo** dataset — y ahí mi modelo bate al rolling-7 baseline por
**37%**, que es una mejora sustantiva en estándares de industria.

---

## 13. LightGBM vs deep learning en retail forecasting

**Claim:** "LSTM 4-channel barely beats the mean."

**Status:** ✅ Validado como hallazgo común en literatura.

**Fuente:**
- arXiv 2506.05941 (2025): LightGBM MAE 2,325 vs LSTM MAE 2,459.
  Gradient boosting supera consistentemente a LSTM en retail tabular.
- M5 competition (Makridakis et al., 2022): top 50 soluciones casi
  todas usaron gradient boosting; ningún top-10 fue deep learning puro.

---

## 14. Z = 1.65 para 95% service level

**Claim:** En la fórmula `Z × σ × √L`, Z = 1.65 para 95%.

**Status:** ✅ Validado (estadística estándar).

**Detalle:** Es el cuantil 0.95 de la distribución normal estándar.
Aparece en cualquier tabla Z. No requiere cita.

---

## 15. Lead time en supply chain — ejemplo de 3 días

**Claim:** "Supplier lead time = 3 days → cover 300 units."

**Status:** ✅ Ejemplo ilustrativo, no requiere validación externa.

---

## Resumen ejecutivo de cambios a aplicar

| # | Sección | Cambio | Severidad |
|---|---|---|---|
| 2 | Lab page §01, Business Impact | Reemplazar "3–5×" por las cifras directas (stockout 4–8% sales, overstock 20–30% inventory value, 9%/55% churn) | Media |
| 6 | Lab page §02, posts | Cambiar "ρ entre -0.3 y -0.7" por "elasticidad entre -0.5 y -2" | Alta — terminología |
| 8 | Lab page Business Impact | Suavizar "20–35%" a "15–30% reported in industry case studies" | Baja |
| 9 | Lab page Business Impact | Cambiar "$1.5M–$4M" a "$1M–$3M" con cálculo explícito | Media |
| 4 | Lab page §06 | Añadir matiz "3.75 MAD strictly, rounded to 4 in practice" | Baja |
| — | Todo | Añadir sección "Sources / Fuentes" al final del lab page | Alta — credibilidad |

---

## Fuentes consultadas (para citar en la versión final)

### Costos de inventario y stockouts

- ToolsGroup — *Cost of Stockouts vs Overstock: Impact on
  Profitability.* https://www.toolsgroup.com/blog/cost-of-stockouts-vs-overstock/
- ToolsGroup — *The Hidden Costs of Poor Inventory Management.*
  https://www.toolsgroup.com/blog/the-hidden-costs-of-poor-inventory-management-how-much-are-you-really-losing/
- Opensend — *29 Inventory Stock-out Rate Statistics for eCommerce.*
  https://www.opensend.com/post/inventory-stock-out-rate-statistics
- WAIR — *The Financial Impact of Overstock and Stockouts.*
  https://wair.ai/financial-impact-overstock-stockouts/

### Forecast accuracy → inventory reduction

- Gartner data citada por Manokhin V. — *Benefits of Improving Forecast
  Accuracy in Supply Chains.*
  https://valeman.medium.com/benefits-of-improving-forecast-accuracy-in-supply-chains-29f6d8b37c80
- Bamboo Rose — *How a 10% Forecast Accuracy Improvement Transforms a
  $500M Fashion Brand.* https://bamboorose.com/blog/forecast-accuracy-fashion/
- Lokad — *Financial Impact of the Forecasting Accuracy.*
  https://www.lokad.com/accuracy-gains-inventory/

### CRISP-ML(Q)

- Studer S. et al. (2020). *Towards CRISP-ML(Q).* arXiv:2003.05155 /
  MDPI MAKE 3(2). https://arxiv.org/abs/2003.05155
  https://www.mdpi.com/2504-4990/3/2/20

### PSI / drift detection

- *Population Stability Index (PSI).* GeeksforGeeks.
  https://www.geeksforgeeks.org/data-science/population-stability-index-psi/
- Fiddler AI — *Measuring Data Drift with the Population Stability
  Index.* https://www.fiddler.ai/blog/measuring-data-drift-population-stability-index

### Tracking signal

- Value Chain Planning — *Tracking Signal: An Effective KPI.*
  https://valuechainplanning.com/blog-details/110
- Umbrex — *Demand Forecast Bias.*
  https://umbrex.com/resources/company-analysis/supply-chain-logistics/demand-forecast-bias/

### Newsvendor / quantile forecasting

- MetricGate — *Newsvendor Model Calculator.*
  https://metricgate.com/docs/newsvendor-model/
- Dagher P. — *TimesFM 2.5 Quantile Forecasting for Inventory Planning.*
  https://medium.com/@nasdag/timesfm-2-5-quantile-forecasting-for-inventory-planning-2500c93de050

### Pricing elasticity

- 7Learnings — *A Guide to Price Elasticity.*
  https://7learnings.com/blog/price-elasticity/
- Wikipedia — *Price Elasticity of Demand.*
  https://en.wikipedia.org/wiki/Price_elasticity_of_demand

### Data leakage

- IBM Think — *What is Data Leakage in Machine Learning?*
  https://www.ibm.com/think/topics/data-leakage-machine-learning
- arXiv 2512.06932 (2025) — *Hidden Leaks in Time Series Forecasting.*
  https://arxiv.org/pdf/2512.06932

### LightGBM / M5 / overfit en lag features

- arXiv 2506.05941 (2025) — *Comparative Analysis of Modern ML Models
  for Retail Sales Forecasting.* https://arxiv.org/abs/2506.05941
- ScienceDirect — *Forecasting with gradient boosted trees: Winning
  solution to the M5 Uncertainty competition.*
  https://www.sciencedirect.com/science/article/abs/pii/S0169207021002090
