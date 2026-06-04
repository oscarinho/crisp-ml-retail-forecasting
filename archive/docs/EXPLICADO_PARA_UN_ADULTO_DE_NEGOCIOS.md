# 📊 Explicado para un adulto de negocios

> Esta versión asume que entiendes **P&L, working capital, costo de
> oportunidad** y que has visto los reportes de demand planning de alguna
> empresa al menos una vez. No asume Python ni machine learning.
>
> Si esto te queda chico, los notebooks están en
> [`notebooks/`](notebooks/). Si te queda grande, lee primero
> [`EXPLICADO_PARA_UN_NIÑO.md`](EXPLICADO_PARA_UN_NIÑO.md) y vuelve.

---

## El problema (no es nuevo)

Cualquier negocio que vende productos físicos enfrenta la misma pregunta
todos los días:

> *"¿Cuánto inventario de cada SKU debo tener mañana en cada tienda?"*

**Si te pasas:** capital muerto en bodega, riesgo de obsolescencia
(especialmente en perecederos, moda, electrónica), liquidación a margen
negativo.

**Si te quedas corto:** ventas perdidas (impacto directo en revenue), riesgo
de churn (un cliente que no encuentra lo que busca compra en otra parte y
a veces no vuelve), penalizaciones por OTIF en B2B.

**El número que mueve esa decisión es el forecast de demanda.** Mientras
mejor sea el forecast, menos costos asimétricos sufres.

---

## ¿Por qué no es un problema resuelto?

Compañías serias gastan **$500K-$2M USD/año en SAP IBP, Oracle Demantra
o o3 Solutions** específicamente para mejorar sus forecasts. Estos sistemas
existen porque:

1. Cada categoría se comporta distinto (carne fresca ≠ ropa de temporada ≠ electrónicos)
2. La data nunca está perfectamente limpia
3. La decisión es a 30+ días vista, pero los datos cambian a diario
4. Los modelos buenos hace 5 años hoy están obsoletos (COVID, inflación, hábitos)
5. Cada tienda, producto o región nueva rompe asunciones del modelo anterior

Forecasting **es difícil estructuralmente** — no porque la matemática sea
complicada, sino porque las condiciones de uso son hostiles.

---

## ¿Qué construimos en este repo?

**Cuatro labs.** Cada uno toma un dataset distinto y construye un sistema de
forecasting de punta a punta. Pensar en ellos como **cuatro casos de
estudio**:

| Lab | Datos | Negocio simulado |
|---|---|---|
| 1. Inventory | 73k filas, 5 tiendas × 20 productos | Retail genérico estilo Walmart |
| 2. Sales | 76k filas, dataset con flag de "epidemia" | Mismo retail bajo shock externo |
| 3. Food Demand | 457k filas, 77 cocinas × 51 platillos | Cadena tipo Rappi / Doordash |
| 4. Store Sales | 3M filas, 4.5 años de Corporación Favorita (Ecuador real) | Supermercado real Kaggle |

Cada lab corre la misma metodología (CRISP-ML(Q) — un framework de 6 fases
que la industria usa para no improvisar) y produce un modelo desplegable.
Cuatro modelos comparables sobre cuatro contextos distintos.

---

## Los 5 hallazgos importantes (en lenguaje de negocio)

### 1. Hay DOS techos de exactitud, no uno

Esto fue **el insight central**, y nació de una pregunta del autor durante el
proyecto: *"¿No debería el forecast actual entrar al modelo de alguna
manera?"*.

En el Lab 1, el dataset incluye una columna `Demand Forecast` que está
sospechosamente cerca del valor real (correlación 99.7%). Si la usas
directamente como feature, tu modelo se ve genial — pero estás haciendo
trampa: ese forecast no existe el mismo día en producción real.

**Pero si la usas como BASE + CORRECCIÓN** (lo que en planning se llama
"forecast adjustment" o "judgmental override"), no es trampa. Es práctica
estándar de demand planning maduro.

| Escenario | Error promedio (MAE) | Sesgo (bias) |
|---|---:|---:|
| Sin forecast disponible (modelo solo) | 69 unidades | -1 |
| Forecast puro (sin modelo) | 8.4 unidades | **+5.0** (sobreestima) |
| **Forecast + modelo corrige (residual learning)** | **7.4 unidades** | **+0.1** |

**Lo más importante:** la mejora no viene de reducir dispersión — viene de
**corregir el sesgo sistemático del forecast**. El sistema actual sobreestima
por 5 unidades por SKU/día. Multiplicado por miles de productos durante
todo el año, eso son **toneladas de overstock estructural**.

**Implicación de negocio:** si tu sistema actual de demand planning ya te da
un forecast — aunque sea malo — **no lo botes**. Constrúyele un modelo
encima que aprenda dónde se equivoca sistemáticamente. Mejorará el bias
antes que la dispersión, que es exactamente lo que el supply chain necesita
(menos buffer = menos working capital atrapado).

### 2. El patrón "cat-and-mouse" (champion-challenger) tiene su nicho

Si fuiste analista MRP alguna vez, recordarás que los planners usaban varios
modelos y se quedaban con el que mejor predijo el mes anterior. Es un
patrón real, hoy formalizado en sistemas de planning como
*"best-of-breed model selection"* o *"champion-challenger"*.

Lo probamos: **18 ventanas mensuales × 7 modelos compitiendo entre sí.**

**Resultado:** un solo modelo (HistGradientBoosting residual) ganó 17 de 18
ventanas (94%). La rotación mensual entre modelos **NO agregó valor** en
este dataset.

**Implicación de negocio:** champion-challenger paga el overhead operacional
SOLO si:

- Los modelos divergen consistentemente **>2-5% en MAE**
- Hay shocks externos predecibles que rompen modelos específicos
  (estacionalidad fuerte, lanzamientos, promociones)
- El equipo tiene capacidad de **evaluar y switchar mensualmente**

Si tu negocio no cumple los tres → **un solo modelo bien mantenido es más
barato y casi igual de bueno**. No te dejes vender un sistema que pelea
contra problemas que no tienes.

### 3. Más features no es mejor decisión

En el Lab 2 (Sales), probamos un modelo con **29 features** vs uno con **15**.

Ganó el de 15.

**¿Por qué?** Porque más features = más memorización del pasado = peor
generalización a datos nuevos. Es el equivalente ML del "exceso de KPIs" en
un dashboard ejecutivo: cuando todo importa, nada importa.

**Implicación de negocio:** desconfía del consultor que te dice "agreguemos
20 features más". El test correcto es **performance en datos NO VISTOS**,
no cantidad de variables.

### 4. La elección del modelo es la decisión menos importante

| Decisión | Impacto típico en error de pronóstico |
|---|---|
| Cómo planteas el problema (residual vs directo) | **62 unidades MAE** (90% del gap) |
| Engineering de features | 3-10 unidades |
| Elección entre LightGBM, Random Forest, CatBoost | 0.04 unidades |
| Hiperparámetros (afinación fina) | 1-3 unidades |

Las decisiones que importan están del lado de **cómo se formula la
pregunta**, no del lado del algoritmo. Esto va contra lo que vende la
mayoría de proveedores ("nuestro modelo usa Transformers / GenAI"). El
algoritmo es commodity. **El framing es estrategia.**

### 5. Per-category routing funciona — pero necesita masa crítica

Probamos tener **un modelo por categoría** en los 4 labs:

|  | # de categorías | ¿Routing gana? |
|---|---:|---|
| Sales lab | 5 (Clothing, Electronics, Groceries, Toys, Furniture) | ✓ sí |
| Store Sales lab | 33 families | ✓ sí |
| Food Demand lab | **4** (Italian, Continental, Thai, Indian) | ✗ no |
| Inventory lab | 5 | n/a (no probado) |

**Implicación de negocio:** el patrón "un modelo por categoría" funciona
cuando hay suficiente variedad para distinguir comportamientos distintos.
Por debajo de 5 categorías, los sub-modelos se quedan sin data para
aprender y pierden contra un modelo global más robusto.

Si tu portafolio tiene menos de 5 familias de producto bien pobladas, **no
inviertas en arquitectura per-category** — gasta el esfuerzo en mejores
features.

---

## ¿Cuánto ahorra realmente esto?

La industria reporta estos órdenes de magnitud (Gartner 2024, McKinsey 2023):

| Mejora típica con forecasting moderno | Impacto financiero |
|---|---|
| Reducción de stockouts (de 8% → 3%) | +1-3% en revenue (sales recovered) |
| Reducción de inventory carrying cost | −10-25% en working capital tied up |
| Reducción de waste / obsolescencia | −15-40% (perecederos especialmente) |
| Reducción de safety stock por mejor bias | −5-15% en inventario promedio |

**Para contexto:** una cadena con $200M USD/año en revenue y 30% inventory
turnover → upside típico de un proyecto bien ejecutado: **$1-3M USD/año
recurrente**.

Los modelos que construimos en este portfolio son **del mismo orden de
complejidad** que los que generan esos ahorros en producción real. La
diferencia entre un proyecto exitoso y uno que fracasa rara vez es el
modelo — usualmente es:

- **La calidad de los datos de input** (garbage in, garbage out — el cliché es cierto)
- **Si el equipo de planning los va a usar realmente** (más fracasos vienen de adoption que de accuracy)
- **Si el sistema de inventario está integrado para actuar sobre el forecast** (un forecast que no se ejecuta no genera ahorro)

---

## Lo que tu negocio debería hacer

Asumiendo que ya tienes algún sistema de planning (aunque sea un Excel
sofisticado):

### 1. Audita tu forecast actual
Mide su MAE y bias contra ventas reales de los últimos 90 días. **Si tienes
sesgo >2% sostenido, hay valor inmediato en construir una corrección.**

### 2. Identifica las 3 categorías que más mueven inventario
No optimices todas. Optimiza donde está el dinero. La regla 80/20 aplica
brutalmente acá.

### 3. Construye un layer de corrección sobre tu forecast existente
**No reemplaces tu forecast actual, añádele un modelo de corrección.** Eso
es exactamente lo que la Lección 1 mostró que vale (el "residual learning"
del Lab 1). Es más barato, más defendible, y compatible con tu sistema
actual.

### 4. NO te metas a champion-challenger todavía
Hasta que tengas el layer de corrección estable, rotar modelos es overhead
sin upside. Saltarse pasos en planning es lo que rompe proyectos.

### 5. Mide el costo asimétrico (stockout vs overstock)
¿Cuánto te duele un stockout vs un overstock? Si la respuesta es "no sé",
ese es el primer cálculo que debes hacer — porque ese ratio define **cuánto
buffer (safety stock) mantener**. Sin ese número, tu modelo perfecto
todavía toma decisiones imperfectas.

### 6. Reserva 10-15% de tu data para "honest holdout"
Cualquier modelo que veas — interno o de un proveedor — debe probarse en
**datos que el modelo nunca vio durante su entrenamiento**. Si el proveedor
no puede mostrarte esa métrica, asume que el modelo no es defendible.

---

## Si quieres profundizar

| Lo que quieres | A dónde ir |
|---|---|
| Entender el ánimo del proyecto en 3 minutos | [`EXPLICADO_PARA_UN_NIÑO.md`](EXPLICADO_PARA_UN_NIÑO.md) |
| Las 4 libretas completas con código y plots | [`notebooks/`](notebooks/) |
| El experimento clave sobre residual learning + cat-and-mouse | [`EXPERIMENT_DF_RESIDUAL.md`](EXPERIMENT_DF_RESIDUAL.md) |
| Las lecciones generales detrás del repo | [`LESSONS_LEARNED.md`](LESSONS_LEARNED.md) |
| Resumen ejecutivo de los 4 labs | [`README.md`](README.md) |

---

## Mini-glosario por si te encuentras a un data scientist

| Lo que dice el técnico | Lo que significa en negocio |
|---|---|
| MAE / Mean Absolute Error | Promedio de unidades que erra el pronóstico (positivo y negativo se compensan) |
| RMSE | Como MAE pero penaliza errores grandes (útil cuando un error gordo duele más) |
| Bias / Sesgo | Si sistemáticamente sobre o subestima (bias = +5 → siempre sobreestima por 5) |
| Stockout | Cuando no tienes para vender lo que el cliente quiere |
| Service level | % de veces que tienes el producto disponible (≥95% es estándar retail) |
| Lead time | Tiempo desde que ordenas hasta que llega |
| Safety stock | Buffer extra para absorber variabilidad — directamente proporcional al bias |
| Lift de modelo | Mejora % sobre el baseline actual (lo que el modelo "aporta") |
| Feature engineering | Inventar variables nuevas a partir de las que ya tienes |
| Leakage | Trampa: usar información que no vas a tener en producción |
| Holdout / Test set | Reservar 10-15% de datos para evaluar honestamente (sin esto, no hay validación) |
| Per-category routing | Un modelo distinto por familia de producto |
| Residual learning | Aprender la corrección sobre un forecast base existente |
| Champion-challenger | Mantener varios modelos en paralelo, rotar al ganador del periodo anterior |
| Newsvendor / P80 quantile | Modelo que predice un punto alto de la distribución, no el promedio — útil para reorder |

---

## Cierre

**Si llegaste hasta aquí, tienes el 90% del valor del proyecto en la cabeza.**

El 10% restante es código y matemática — y vive en los notebooks. Pero las
decisiones de negocio que necesitas tomar para que un proyecto así genere
ROI están todas en este documento.

La lección que se generaliza a cualquier inversión en ML, no solo a
forecasting:

> *El algoritmo es commodity. El framing es estrategia.*
> *Las decisiones de modelo cuestan 1 día. Las decisiones de framing cuestan 1 año.*

— oscarponce.com · Junio 2026
