# Guía de conceptos — para leer junto al notebook

Este archivo NO repite el notebook. Explica los conceptos técnicos que aparecen ahí, en lenguaje sencillo, para que cuando estés leyendo una celda y veas algo que nunca estudiaste, puedas saltar acá, entenderlo en 30 segundos, y volver.

Cada concepto tiene tres líneas:

- **Qué es** — la intuición, sin fórmulas
- **Por qué aparece acá** — para qué lo usamos en este notebook específico
- **Qué leer en el output** — qué número o gráfico te dice si funcionó

Salta directo a la fase que estés leyendo.

---

## Tabla de contenidos

- [Fase 2 — Entendimiento de los datos](#fase-2--entendimiento-de-los-datos)
- [Fase 3 — Preparación de datos](#fase-3--preparación-de-datos)
- [Fase 4 — Modelado](#fase-4--modelado)
- [Fase 5 — Evaluación](#fase-5--evaluación)
- [Fase 6 — Despliegue](#fase-6--despliegue)
- [Apéndice — Métricas en una línea](#apéndice--métricas-en-una-línea)

---

## Fase 2 — Entendimiento de los datos

### Correlación de Pearson (ρ, "rho")

- **Qué es:** un número entre −1 y +1 que mide si dos columnas se mueven juntas. +1 = perfectamente juntas, 0 = sin relación, −1 = perfectamente opuestas.
- **Por qué aparece acá:** la usamos como detector de leakage. Cuando descubrimos que `Demand Forecast` tenía ρ = 0.997 con `Units Sold`, supimos al instante que esa columna ya contiene la respuesta — no se puede usar como input "justo".
- **Qué leer en el output:** valores arriba de 0.9 son sospechosos en datos crudos. Valores cerca de 0 (como Price ρ = 0.001) dicen "esta columna no aporta señal lineal al target".

### Autocorrelación (lag-1, lag-7)

- **Qué es:** correlación de una serie *consigo misma*, desplazada en el tiempo. "Lag-1" pregunta: ¿las ventas de hoy se parecen a las de ayer? "Lag-7" pregunta: ¿se parecen a las de hace una semana?
- **Por qué aparece acá:** si la autocorrelación es alta, los features de lag (`lag_1`, `lag_7`) van a llevar mucha señal. Si es cero, los lags son inútiles. En este dataset es ≈ 0 — por eso el techo de MAE 69 no se mueve.
- **Qué leer en el output:** valores entre −0.05 y +0.05 = ruido blanco. Valores arriba de 0.3 = patrón temporal explotable.

### Test ADF (Augmented Dickey-Fuller)

- **Qué es:** un test estadístico que pregunta "¿esta serie es estable en el tiempo o está derivando?". Devuelve un p-value.
- **Por qué aparece acá:** ARIMA solo funciona bien sobre series estables ("estacionarias"). El ADF nos dice si necesitamos diferenciar la serie antes de aplicarle ARIMA.
- **Qué leer en el output:** `p < 0.05` → estable, se puede usar tal cual (en ARIMA `d=0`). `p > 0.05` → está derivando, hay que aplicar diferencia.

### Estacionariedad

- **Qué es:** una serie es "estacionaria" cuando su promedio y su variabilidad no cambian con el tiempo. Como un río calmo: siempre el mismo nivel, siempre las mismas olas pequeñas.
- **Por qué aparece acá:** ARIMA, ETS y muchos métodos clásicos asumen estacionariedad. Si la serie deriva (ventas creciendo año tras año), los modelos clásicos fallan a menos que la "destendencies" primero.
- **Qué leer en el output:** lo decide el test ADF de arriba.

---

## Fase 3 — Preparación de datos

### Split temporal (vs split aleatorio)

- **Qué es:** en lugar de mezclar las filas y agarrar un 80% al azar para entrenar, partes por fecha. Train = 2022, test = 2023.
- **Por qué aparece acá:** mezclar al azar es trampa en series temporales — el modelo vería el futuro durante el entrenamiento. El split temporal simula el mundo real: predecir mañana con lo que sabes hoy.
- **Qué leer en el output:** la celda imprime las fechas exactas del corte. Verifica que train termine antes de que test empiece.

### TimeSeriesSplit (CV temporal)

- **Qué es:** cross-validation pero respetando el tiempo. En lugar de cortar el dataset en 5 pedazos al azar, hace 5 "ventanas" donde cada fold entrena con el pasado y valida con el siguiente bloque de tiempo.
- **Por qué aparece acá:** lo usa `RandomizedSearchCV` para elegir los hiperparámetros de LightGBM. Si usáramos KFold normal, el modelo vería días del futuro durante el tuning y reportaría un MAE engañosamente bajo.
- **Qué leer en el output:** la métrica `CV MAE` que ves después del tuning. Si CV MAE y test MAE están cerca (acá 69.26 vs 69.1), generalizó bien.

### Features de lag

- **Qué es:** columnas nuevas que dicen "qué pasó hace N días". `lag_1` = ventas de ayer, `lag_7` = ventas de hace una semana.
- **Por qué aparece acá:** los modelos de árbol (LightGBM) no entienden tiempo solos. Si quieres que el modelo use el pasado, le tienes que dar el pasado como columnas explícitas.
- **Qué leer en el output:** el feature importance al final del entrenamiento. Si `lag_1` está arriba, el ayer importa. Acá no importa (autocorrelación ≈ 0).

### Rolling features (media móvil)

- **Qué es:** features que resumen una ventana del pasado. `roll_7_mean` = promedio de las últimas 7 ventas. `roll_28_mean` = promedio mensual.
- **Por qué aparece acá:** capturan tendencias suaves sin depender de un solo día. Son más robustas que un lag individual.
- **Qué leer en el output:** suelen aparecer arriba en feature importance cuando hay tendencia. Acá tampoco aportan mucho.

### ColumnTransformer

- **Qué es:** una herramienta de sklearn para aplicar diferentes transformaciones a diferentes columnas en una sola pasada. "A las numéricas StandardScaler, a las categóricas OneHotEncoder, a las skewed PowerTransformer".
- **Por qué aparece acá:** los features tienen tipos mixtos (numéricos, categóricos, sesgados). En lugar de procesar cada uno a mano, el ColumnTransformer lo hace todo en un objeto, que después se mete en el `Pipeline`.
- **Qué leer en el output:** no hay output directo — es invisible si funciona. Lo importante es que cuando salves el modelo, el ColumnTransformer va dentro y la app aplica las mismas transformaciones en inferencia.

### Scalers (Standard, MinMax, Power)

- **Qué son:**
  - `StandardScaler` — resta la media y divide por la desviación estándar. Útil para columnas que siguen una distribución normal.
  - `MinMaxScaler` — encoge el rango a [0, 1]. Útil para columnas acotadas (ej. discount %).
  - `PowerTransformer` — endereza distribuciones sesgadas (asimétricas) para que parezcan más normales.
- **Por qué aparecen acá:** algunos modelos (Ridge, LSTM, redes neuronales) son sensibles a la escala. Si una columna va de 0 a 1 y otra de 0 a 1000, el modelo le presta más atención a la segunda solo por el tamaño.
- **Qué leer en el output:** no se ve directo. Confías en que el `Pipeline` aplica el scaler correcto a la columna correcta.

### One-Hot Encoding (OHE)

- **Qué es:** convertir una columna categórica (`Category` = "Electronics", "Clothing", ...) en varias columnas binarias (`Category_Electronics` = 1 o 0). El modelo entiende solo números.
- **Por qué aparece acá:** `Store ID`, `Category`, `Region`, `Weather Condition` son categóricas. OHE las convierte en columnas que LightGBM puede usar.
- **Qué leer en el output:** el número de columnas crece. Si tenías 5 categorías, ahora hay 5 columnas nuevas (o 4 con `drop='first'`).

---

## Fase 4 — Modelado

### Baseline naive (lag-1, lag-7, media)

- **Qué es:** modelos triviales que sirven de vara de medir. "Predecir que mañana será igual que hoy" (lag-1), "que la próxima semana será igual que la pasada" (lag-7), "predecir siempre el promedio".
- **Por qué aparecen acá:** si tu modelo fancy no le gana al naive, algo está mal. Es el piso ético.
- **Qué leer en el output:** las MAE de cada baseline. Acá la media tiene MAE 89.1 — esa es la vara que cualquier modelo debe bajar.

### Stage 1 vs Stage 2 (en este notebook)

- **Qué son:**
  - **Stage 1** — entrenado con features *contextuales* solamente (fecha, clima, holiday, precio, descuento). Sin lags, sin inventario. Sirve como fallback cuando un producto nuevo no tiene historia.
  - **Stage 2** — entrenado con *todos* los features (contextuales + lags + inventario). Es el modelo principal.
- **Por qué aparecen acá:** demuestra que la señal real vive en lags + inventario. Stage 1 solo (MAE 90.2) apenas le gana a la media; Stage 2 (MAE 69.1) le gana al baseline por 22%.
- **Qué leer en el output:** la diferencia en MAE entre los dos. Si Stage 2 no le gana significativamente a Stage 1, los lags no están aportando.

### LightGBM (intuición rápida)

- **Qué es:** un modelo de "árboles boosted". Toma una predicción mediocre, calcula los errores, entrena un árbol nuevo para corregir esos errores, repite cientos de veces. Cada árbol corrige los errores del conjunto previo.
- **Por qué aparece acá:** es el caballo de batalla del retail tabular. Maneja features mixtos (numéricos + categóricos), captura no-linealidades, es rápido, no necesita escalado.
- **Qué leer en el output:** el feature importance al final. Te dice qué columnas el modelo realmente está usando.

### Gradient Boosting (la familia)

- **Qué es:** la familia de modelos a la que pertenecen LightGBM, XGBoost, CatBoost y HistGradientBoosting. Todos hacen lo mismo (árboles que corrigen errores) con detalles distintos de implementación.
- **Por qué aparecen acá:** los probamos todos (§4.10 Tier 1) para confirmar que el techo de MAE 69 no es por elegir mal el modelo — es del dataset. Todos convergen al mismo MAE.

### RandomizedSearchCV

- **Qué es:** una manera de elegir hiperparámetros (número de árboles, profundidad, learning rate) probando combinaciones al azar y midiendo cuál genera el mejor CV MAE.
- **Por qué aparece acá:** en lugar de adivinar a mano que `num_leaves=31` es bueno, dejamos que la búsqueda lo encuentre. "Randomized" en vez de "Grid" porque es mucho más rápido para conseguir resultados parecidos.
- **Qué leer en el output:** los mejores hiperparámetros encontrados (`best_params_`) y el CV MAE asociado.

### Stacking ensemble + meta-learner

- **Qué es:** entrenas varios modelos base (Ridge, RF, XGB, LGBM), después entrenas un modelo "jefe" (el meta-learner) que aprende cómo combinar las predicciones de los base learners. En este notebook el meta-learner es Ridge.
- **Por qué aparece acá:** la idea es que modelos distintos capturen señales distintas y el meta-learner las combine inteligentemente. Acá ganó solo 0.3% — porque los modelos base se equivocan en las mismas cosas (todos topan en el techo del dataset).
- **Qué leer en el output:** los pesos que el meta-learner asigna a cada base. Acá Ridge se llevó 0.69 y XGB/LGBM quedaron en 0.02 — el meta-learner detecta que XGB y LGBM no aportan información nueva.

### Prophet (intuición)

- **Qué es:** un modelo de Facebook diseñado para series con estacionalidad fuerte y holidays. Descompone una serie en tendencia + ciclo anual + ciclo semanal + holidays.
- **Por qué aparece acá:** es un baseline clásico popular. Acá falla feo (MAE 112) porque las series son básicamente ruido blanco — Prophet extrapola tendencias que no existen.
- **Qué leer en el output:** la lección es que Prophet *no* es un "default seguro". Si la serie no tiene patrón temporal real, Prophet inventa uno y empeora todo.

### ARIMA (intuición)

- **Qué es:** un modelo clásico que predice el siguiente valor usando una combinación de (a) valores pasados de la serie, (b) errores pasados de predicción. Los tres números `(p, d, q)` definen cuántos lags, cuánta diferenciación, y cuántos errores pasados usar.
- **Por qué aparece acá:** baseline clásico univariado. `auto_arima` busca los `(p, d, q)` óptimos automáticamente. Acá MAE 89.1 — empate con la media, porque las series no tienen autocorrelación.
- **Qué leer en el output:** los órdenes `(p, d, q)` que `auto_arima` elige. Si elige `(0, 0, 0)` para muchas series, la serie es ruido blanco.

### ETS (Holt-Winters)

- **Qué es:** otro clásico univariado. Modela nivel + tendencia + estacionalidad de forma "exponencialmente ponderada" (lo reciente pesa más que lo viejo).
- **Por qué aparece acá:** baseline complementario a ARIMA. Mismo veredicto: MAE 89.4 — empate con la media en este dataset.

### LSTM (intuición rápida)

- **Qué es:** una red neuronal recurrente diseñada para secuencias. Procesa un día, recuerda lo importante en su "memoria interna", procesa el siguiente día, y así. Diseñada originalmente para texto y voz.
- **Por qué aparece acá:** muchos cursos modernos dicen "para series temporales usa LSTM". Lo probamos. Resultado: MAE 88.9 — no aprende. El loss se aplana en la época 5. Sin embeddings por serie, no puede separar las 100 series del dataset.
- **Qué leer en el output:** la curva de loss durante el entrenamiento. Si se aplana muy temprano, el modelo no está aprendiendo.

### Regresión cuantil (P80)

- **Qué es:** un modelo entrenado para predecir un percentil específico, no el promedio. P80 = "valor bajo el cual queda la demanda el 80% de las veces". LightGBM lo soporta con `objective='quantile'`.
- **Por qué aparece acá:** para inventario, el promedio es inútil. Si pides exactamente el promedio, te quedas sin stock la mitad del tiempo. P80 dimensiona pedidos con margen de seguridad.
- **Qué leer en el output:** la **coverage** — el porcentaje real de veces que la demanda quedó bajo la predicción P80. Si pediste 80% y obtuviste 77.5%, el modelo infracubre ligeramente.

### Residual learning (el insight central del notebook)

- **Qué es:** en lugar de pedirle al modelo que prediga `Units Sold` desde cero, le pides que prediga `Units Sold − Demand Forecast` (el *error* del forecast existente). Después sumas: `pred = DF + model.predict(features)`.
- **Por qué aparece acá:** porque DF está disponible 1+ semanas antes de la predicción y ya contiene la mayor parte de la señal. Pedirle al modelo que re-derive estacionalidad y holidays desde cero es desperdiciar capacidad. Es mejor que aprenda solo el error que el sistema existente comete.
- **Qué leer en el output:** MAE colapsa de ~69 a ~7.4. Y el sesgo cae de +5.05 a +0.10 — el modelo está corrigiendo el sesgo sistemático del forecast original.

### Champion-Challenger backtest (rolling)

- **Qué es:** evalúas varios modelos sobre múltiples ventanas de tiempo (acá 18 ventanas de 30 días, rolling). Por cada ventana, ves cuál ganó. Si un modelo gana siempre, te quedas con él. Si los ganadores cambian, rotas mensualmente.
- **Por qué aparece acá:** simula cómo funcionaría una operación real de demand planning. Acá HGB residual gana 17 de 18 ventanas → ganador estable, no hace falta rotar.
- **Qué leer en el output:** la tabla de wins por modelo. Si un modelo concentra la mayoría, está dominando consistentemente.

---

## Fase 5 — Evaluación

### MAE (Mean Absolute Error)

- **Qué es:** "en promedio, ¿por cuántas unidades te equivocaste?". Es la métrica más intuitiva.
- **Por qué aparece acá:** es la métrica principal del notebook. MAE 69 vs MAE 7.4 = los dos techos del titular.

### RMSE (Root Mean Squared Error)

- **Qué es:** como MAE pero antes de promediar eleva los errores al cuadrado, así errores grandes pesan más. Siempre ≥ MAE.
- **Por qué aparece acá:** complementa MAE. Si RMSE es mucho mayor que MAE, hay errores feos escondidos. Si están cerca, los errores son uniformes.

### sMAPE (symmetric MAPE)

- **Qué es:** error como porcentaje, simétrico respecto a sub- y sobre-predicción, acotado 0–200%. Maneja ceros.
- **Por qué aparece acá:** permite comparar productos de tamaños distintos (no puedes comparar "errar por 69 unidades de Coca-Cola" con "69 unidades de TVs").

### AIC (Akaike Information Criterion)

- **Qué es:** una métrica que premia el buen ajuste pero penaliza la complejidad (cuántos parámetros usaste). Lower = better.
- **Por qué aparece acá:** `auto_arima` la usa internamente para elegir los `(p, d, q)`. Nunca la reportas al negocio.

### Sesgo del forecast (bias)

- **Qué es:** promedio de `(predicción − real)`. Si es positivo, el modelo sobre-predice consistentemente. Si es negativo, sub-predice.
- **Por qué aparece acá:** un modelo con MAE bajo pero sesgo positivo persistente significa exceso de stock crónico. Acá DF puro tiene sesgo +5.05 (siempre sobre-predice por 5 unidades), y el residual lo corrige a +0.10.
- **Qué leer en el output:** la columna `bias` en la tabla de residual learning.

### Coverage (cobertura empírica)

- **Qué es:** para un modelo cuantil P80, el % real de filas donde la demanda quedó bajo la predicción.
- **Por qué aparece acá:** valida si el P80 realmente cubre el 80% prometido. Acá obtuvimos 77.5% — un poco bajo, en producción tunearías α hacia arriba.

### SHAP (SHapley Additive exPlanations)

- **Qué es:** una técnica que te dice, para *cada predicción individual*, cuánto contribuyó cada feature a llevar la predicción arriba o abajo de la media. Como un "recibo" de la predicción.
- **Por qué aparece acá:** explicabilidad. Cuando un planner pregunta "¿por qué predijo 200 unidades para este SKU?", SHAP responde "porque +50 vinieron del Inventory Level alto, +30 del descuento, −20 del clima frío".
- **Qué leer en el output:** los plots de SHAP — bars que muestran el aporte de cada feature.

### PSI (Population Stability Index)

- **Qué es:** una métrica de drift. Compara dos distribuciones (la de entrenamiento vs la actual de producción) y devuelve un número. 0 = idénticas, >0.25 = drift severo.
- **Por qué aparece acá:** para monitoreo en producción. Si PSI sobre `ŷ` o sobre algún feature supera 0.2, hay que investigar — el mundo cambió respecto a cuando entrenaste.

### Hierarchical reconciliation

- **Qué es:** técnica para hacer consistente forecasts en distintos niveles de agregación. Si forecasteo cada SKU por separado y por categoría por separado, los números no van a coincidir cuando sumes los SKUs. Reconciliation los fuerza a coincidir.
- **Por qué aparece acá:** lo probamos por completitud (top-down). En este dataset *empeora* los resultados — la mezcla de categorías cambia entre train y test y las proporciones quedan desactualizadas. El método recomendado en producción es **MinT**.

---

## Fase 6 — Despliegue

### Newsvendor (la fórmula clásica)

- **Qué es:** un resultado de teoría de inventario que dice que el nivel óptimo de stock para minimizar costo total es el cuantil `q = stockout_cost / (stockout_cost + overstock_cost)` de la distribución de demanda.
- **Por qué aparece acá:** justifica matemáticamente por qué usamos P80 en este notebook. Con costo de stockout $20 y overstock $5: `q = 20/(20+5) = 0.80`.
- **Qué leer:** la fórmula da el cuantil. El modelo cuantil lo entrega.

### Lead time

- **Qué es:** los días que tardan en llegarte un pedido desde que lo colocas hasta que lo recibes.
- **Por qué aparece acá:** define cuánto stock necesitas para cubrir el "tiempo en blanco" entre colocar la orden y recibir el producto.

### Stock de seguridad (safety stock)

- **Qué es:** inventario buffer extra, por encima del expected, para cubrir días donde la demanda corre más alta de lo previsto.
- **Por qué aparece acá:** sale directo del modelo cuantil. La diferencia entre P80 y la media es tu stock de seguridad — calculada desde los datos, no desde una regla del pulgar.

### Punto de reorden (reorder point)

- **Qué es:** el nivel de stock disponible que dispara la creación de un nuevo pedido. Fórmula: `demanda esperada en lead time + stock de seguridad`.
- **Por qué aparece acá:** lo calcula el Reorder Advisory en la app Streamlit.

### Nivel de servicio (service level)

- **Qué es:** el % de ciclos en los que prometes no quedarte sin stock. 95% de servicio significa que estás dispuesto a quebrar 1 de cada 20 ciclos.
- **Por qué aparece acá:** se traduce 1:1 al cuantil que pides al modelo. 80% de servicio = pedir P80. 95% = P95.

### Cost-weighted decisioning

- **Qué es:** tomar decisiones (cuánto stockear, qué reordenar) optimizando el *costo en dólares*, no el MAE. Como los costos son asimétricos (stockout duele más que overstock), un modelo con peor MAE puede ganar en costo total si su sesgo está bien orientado.
- **Por qué aparece acá:** la simulación A/B de la Fase 6 muestra el costo en dólares de cada modelo. Es la métrica que importa al negocio.

### Online learning (refit semanal/mensual)

- **Qué es:** la práctica de re-entrenar el modelo periódicamente con datos nuevos.
- **Por qué aparece acá:** lo medimos. En este dataset estacionario el lift es 0% — reentrenar es desperdicio de cómputo. En producción real, antes de programar refits, mide drift primero.

### Drift y monitoring

- **Qué es:** "drift" significa que el mundo cambió respecto a cuando entrenaste el modelo. Puede ser drift de features (los precios subieron 20%) o drift de target (la demanda se reformuló).
- **Por qué aparece acá:** la sección de "Real-World Deployment Playbook" lista qué alertas montar en producción. PSI es el indicador típico.

---

## Apéndice — Métricas en una línea

Si solo lees una cosa de esta guía:

| Métrica | Pregunta que responde |
|---|---|
| MAE | "¿Por cuántas unidades me equivoqué en promedio?" |
| RMSE | "...pero los errores grandes, ¿qué tan grandes son?" |
| sMAPE | "¿Qué tan mal me fue, en porcentaje?" |
| CV MAE | "MAE, pero medida sin trampa (sobre datos no vistos)." |
| Coverage | "Prometí cubrir 80% — ¿lo cumplí?" |
| Bias | "¿Estoy sesgado hacia arriba o hacia abajo?" |
| AIC | "¿Cuál configuración ajusta mejor sin meter más perillas?" |
| ρ | "¿Estas dos columnas se mueven juntas?" |
| p-value (ADF) | "¿La serie es lo suficientemente estable para modelarla directo?" |
| PSI | "¿La distribución de hoy se parece a la de cuando entrené?" |
| Costo total | "¿Cuánto cuestan mis errores, en dólares?" |

Y si solo lees una idea del notebook: **en este dataset, cómo entra el `Demand Forecast` al pipeline movió MAE muchísimo más que la elección de algoritmo**. Como feature → leakage; dropeado → MAE ~69 (mesa de los métodos ML multivariados); como prior residual → MAE ~7.4. La elección entre HGB, RF o LightGBM dentro del régimen residual cambió MAE en menos de 0.05.

**Caveat:** este dataset es sintético y `Demand Forecast` tiene correlación 0.997 con el target — un near-oracle. En datos reales, el forecast existente sería más ruidoso (correlación 0.7–0.85), el gap entre los dos regímenes sería menor, y los números absolutos de MAE reportados acá no son directamente transferibles. El patrón de diseño (DF como prior, no como feature dropeado) sí generaliza.
