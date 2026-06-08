# Cómo funciona app.py — explicado en simple

> El objetivo de este doc: que entiendas tu propia app sin leer las 1,264 líneas. Está en orden de "de dónde salen los datos" hasta "qué ves en pantalla".

## En una frase

La app toma el dataset de retail, **predice la demanda diaria** de una combinación tienda × producto, y traduce ese número en **decisiones**: ¿estoy en riesgo de quiebre? ¿cuánto pido?

---

## El motor — 4 pasos (de datos a decisión)

### 1. Carga y prepara datos
- `load_data()` lee `data/retail_store_inventory.csv`.
- `engineer_features()` agrega columnas de calendario (mes, día de semana, trimestre, fin de semana) y `price_vs_competitor` (tu precio ÷ precio competidor). Nada de magia: solo deriva campos del `Date` y del precio.

### 2. Carga 3 modelos (.pkl) — cada uno tiene un trabajo
| Archivo | Qué hace | Dónde se usa |
|---|---|---|
| `model_contextual.pkl` | Predice demanda diaria **sin necesitar historial** (cold-start). | El simulador. Es el modelo "del formulario". |
| `model_q80.pkl` | Predice el **percentil 80** de demanda (para safety stock / reorden). | La recomendación de reorden. |
| `model_metadata.pkl` | Guarda qué features espera el modelo. | La app valida que coincidan y **falla fuerte** si no (evita predecir mal en silencio). |

> **Importante:** `model.pkl` (el Stage 2, el del MAE bajo del notebook) **NO** se usa en el formulario, porque necesita lags/historia que un formulario interactivo no tiene. Por eso el simulador **no reproduce el 7.4** — eso vive en el notebook.
> Si `model_contextual.pkl` no existe, la app entrena un RandomForest ligero al vuelo (modo demo).

### 3. Predice (2 funciones)
- `predict_demand(pipeline, row)` → mete tu escenario (un renglón con tienda, mes, precio, inventario…) al pipeline y devuelve **un número**: la demanda diaria esperada.
- `predict_p80(q80, row)` → lo mismo pero el percentil 80. Como no hay historia, rellena los lags faltantes con una proxy (el inventario) → es **direccional, no exacto** (a propósito, para no fingir precisión).

### 4. Decide — cuentas simples encima del número
- `stock_status(inventario, demanda)`: `coverage = inventario ÷ demanda`. **< 0.5** días → CRITICAL · **< 1.2** → LOW STOCK · resto → WELL STOCKED.
- `reorder_qty(demanda, inventario, safety_days=7)`: `buffer = demanda × 7 días`; pide lo que falte para cubrir esos 7 días. Si ya tienes suficiente → 0.

---

## Lo que ves arriba (siempre visible, antes de las pestañas)

1. **Header** + caja **"QUÉ HACE ESTA APP"** (explicación en español).
2. **Emergency Alert Strip** ← *esta es la alerta que sentías que faltaba.* Calcula `Days_of_Supply` por tienda × categoría usando los últimos 30 días:
   - `< 3.5 días` → **⚠ ALERTA CRÍTICA** (cuenta cuántos SKUs).
   - `3.5–7 días` → **STOCK BAJO** (amarillo).
   - resto → **STOCK OK** (verde).
3. **KPI Strip**: demanda promedio, inventario promedio, cobertura, % de días con stockout, y holiday lift.

---

## Las 3 pestañas

- **Demand Simulator** — armas un escenario (tienda, categoría, mes, precio, inventario…) → demanda predicha + status + reorden + curva precio→demanda + comparación feriado/no-feriado. *Es lo que viste en tus capturas.*
- **Inventory Dashboard** — vista agregada del dataset (KPIs y distribuciones).
- **Reorder Advisory** — lista de SKUs ordenada por días de cobertura, filtrable por riesgo, con un slider de safety buffer (3–14 días).

---

## 🔴 Lo más importante para ti ahora

Tu **archivo local** ya tiene (1) la **alerta crítica** y (2) la **caja explicativa en español**. Tus **capturas** (la app desplegada en Streamlit) **no** las muestran → la versión en vivo está **atrasada** respecto al código.
**Acción:** push a GitHub + redeploy en Streamlit, y la app en vivo mostrará las alertas y la explicación. No hay que programar nada nuevo — solo desplegar lo que ya está.

---

## Lo honesto (límites)

- El simulador corre el **modelo contextual** (cold-start). Su error es del orden de **decenas** de unidades, no 7.4. El 7.4 es del notebook (residual learning con la columna Demand Forecast).
- El dataset es **sintético**: elasticidad ≈ 0 y sin autocorrelación. Por eso la curva precio→demanda sale casi plana y los lags no aportan — el modelo refleja honestamente lo que el dato tiene.
