# Serie LinkedIn — Forecasting de Demanda (versión negocio)

Serie de **4 posts en español** dirigida a cualquier profesional —
gerente de operaciones, CEO, COO, director comercial, planner — sin
asumir conocimiento de Machine Learning. El detalle técnico vive en
**oscarponce.com**; LinkedIn es la puerta de entrada.

---

## Filosofía de esta versión

- **Cero jerga técnica.** Nada de "dataset", "lag features",
  "correlación", "overfit", "modelo". Solo conceptos que la audiencia
  ya usa en su industria.
- **Hook = cifra de negocio reconocible.** Stockouts, capital
  atrapado, ventas perdidas. Datos que un gerente intuye y validan
  con su día a día.
- **Un solo idea de negocio por post.** Si quieren profundidad, el
  link a oscarponce.com está listo.
- **Voz editorial intacta** — separadores `──────────────────`,
  subtítulos en negrita Unicode, bullets `▸`, cierre aforístico,
  flecha `👉`, hashtags al final.
- **Cada post incluye su sugerencia de imagen** justo debajo —
  con tres opciones: (A) screenshot real del proyecto,
  (B) visual diseñado en Canva/Figma, (C) prompt para IA.

**Especificaciones de imagen para LinkedIn:** 1200×627 px (landscape)
o 1080×1080 (cuadrado). Evitar mucho texto sobre la imagen.

---

# Post 1 — El costo escondido del inventario mal pronosticado

Las tiendas retail pierden entre **4% y 8%** de sus ventas anuales porque el producto no estaba el día que el cliente quería comprarlo.
El problema real no es no tener inventario. Es no saber cuánto tener.
Por eso construí un sistema que pronostica la demanda diaria por tienda y producto.

──────────────────

𝗘𝗹 𝗽𝗿𝗼𝗯𝗹𝗲𝗺𝗮
La decisión de cuánto pedir cada día se toma con intuición, planillas o reglas viejas.
Y los costos del error son asimétricos — un quiebre de stock pesa más que un sobre-stock, porque incluye venta perdida, cliente que no vuelve y daño a la marca.

𝗟𝗼 𝗾𝘂𝗲 𝗰𝗼𝗻𝘀𝘁𝗿𝘂í
▸ Pronóstico diario por tienda × producto
▸ Cálculo del colchón de inventario alineado al nivel de servicio que la empresa quiere ofrecer
▸ Una app que el responsable de planning puede usar sin pedir ayuda al área de datos

𝗘𝗹 𝗮𝗽𝗿𝗲𝗻𝗱𝗶𝘇𝗮𝗷𝗲
No se pide más porque falte producto. Se pide más porque falta certeza.
Resolver la certeza es barato. Mantener un colchón de inventario, no.

──────────────────

El inventario en exceso es capital que no rota. El inventario que falta es venta que no vuelve. El pronóstico bien hecho está en el medio — y ahí vive el retorno.

*Retomo mi serie sobre CRISP-ML aplicado al negocio — el marco que uso para que la IA no se quede en la libreta. Cuatro publicaciones, cuatro lecciones.*

👉 Caso completo en oscarponce.com

#CRISPML #SupplyChain #InventoryManagement #DemandPlanning #AplicadaANegocios

### 📸 Imagen sugerida para Post 1

**Recomendación principal:** opción (B) — la cifra "4–8%" hace el trabajo emocional.

**(A) Screenshot real — del app Streamlit**
- **Vista:** Tab **"Reorder Advisory Engine"** en `app/app.py` (la tercera pestaña).
- **Qué mostrar:** la tabla con las columnas Store, Category, Avg Demand/Day, Stock On Hand, Days of Supply, Reorder Qty, **Risk** (con celdas coloreadas rojo CRITICAL / amarillo LOW STOCK / verde OK).
- **Por qué funciona:** muestra el output operativo en lenguaje que cualquier gerente entiende — riesgo, stock, días de cobertura. Legible en 3 segundos.
- **Tip:** capturar a 1200px de ancho, recortar para mostrar 6–8 filas con un par de cada nivel de riesgo.

**(B) Visual diseñado — quote card** ⭐
- **Layout:** fondo oscuro grafito (paleta del sitio), número **"4–8%"** centrado en grande, debajo en menor tamaño: *"de las ventas anuales que pierde el retail por quiebres de stock"*.
- **Fuentes:** Orbitron para el número, IBM Plex Mono para el subtítulo.
- Se arma en Canva o Figma en 5 minutos.

**(C) Prompt para IA (Midjourney / DALL·E / Sora)**
```
Editorial photo of a half-empty supermarket shelf, soft cinematic
lighting from the side, one isolated product visible, the rest of
the shelf empty with price tags still in place. Muted color palette,
slight desaturation, shallow depth of field. Photo-realistic, top
shelf of the aisle. Aspect ratio 1.91:1.
```

---

# Post 2 — Más datos no es mejor decisión

El **80%** del éxito de un proyecto de analítica se decide antes de tocar la tecnología.
El problema real no es qué algoritmo usar. Es qué información va a tener tu equipo el día que tenga que decidir.
Lo aprendí corriendo el mismo proceso sobre dos bases de datos del mismo negocio — y obteniendo resultados muy distintos.

──────────────────

𝗘𝗹 𝗽𝗿𝗼𝗯𝗹𝗲𝗺𝗮
Muchos proyectos de IA se entrenan con información que solo existe después del momento de la decisión.
Es el equivalente operativo a darle la respuesta del examen al alumno y luego sorprenderse de que sacó 100. En el aula funciona; en producción, falla.

𝗟𝗼 𝗾𝘂𝗲 𝗼𝗯𝘀𝗲𝗿𝘃é
▸ Misma metodología, mismo equipo, dos bases de datos
▸ Una entregó resultados sólidos; la otra, ningún intento logró superar el promedio
▸ La diferencia no estuvo en la tecnología — estuvo en la calidad de la información que cada base contenía

𝗘𝗹 𝗮𝗽𝗿𝗲𝗻𝗱𝗶𝘇𝗮𝗷𝗲
Antes de invertir en IA, conviene auditar qué información tiene la empresa y cuándo la tiene. Es trabajo barato y de alto retorno.
La tecnología no arregla un proceso mal diseñado — lo amplifica.

──────────────────

Los datos no reemplazan la disciplina operativa. La revelan.

👉 Análisis completo en oscarponce.com

#CRISPML #DataStrategy #SupplyChain #AplicadaANegocios #DigitalTransformation

### 📸 Imagen sugerida para Post 2

**Recomendación principal:** opción (A) — la tabla de leaderboard es prueba visual de rigor.

**(A) Screenshot real — del notebook** ⭐
- **Notebook:** `Sales_Forecasting_CRISPML.ipynb`.
- **Celda a capturar:** el **leaderboard final** (celdas 78–79, "5.1 Holdout leaderboard") — tabla con los modelos ordenados por error.
- **Cómo prepararlo para LinkedIn:** **borrar la columna "model"** y reemplazar los nombres técnicos por etiquetas accesibles antes de capturar — por ejemplo: *"Intento completo (todo el dato)"*, *"Intento con curado mínimo"*, *"**Intento alineado al uso real (ganador)**"*. El resto de columnas (error, gap) se pueden dejar o quitar según el espacio.
- **Por qué funciona:** muestra evidencia visual del aprendizaje del post — el ganador tiene menos información y aún así mejor resultado.

**(B) Visual diseñado — comparación lado a lado**
- **Layout:** dos paneles divididos por una línea vertical fina dorada.
  - Panel izquierdo: ícono de base de datos llena, texto **"Misma metodología"**.
  - Panel derecho: ícono de base de datos también llena pero con un check verde, texto **"Distinto resultado"**.
- En la parte superior: el número **80%** grande con la frase del hook.
- Estética grafito + dorado (paleta del sitio).

**(C) Prompt para IA**
```
Editorial split-screen image: left side shows a chaotic dashboard with
many charts and metrics, dim lighting. Right side shows a single clean
chart on a minimalist desk with one notebook and a pen, warm lighting.
A thin gold vertical line separates them. Conceptual photography style,
muted graphite + warm beige palette. Aspect ratio 1.91:1.
```

---

# Post 3 — Un pronóstico no es la decisión

Un buen pronóstico de demanda no te dice cuánto pedir.
Te dice cuánto se va a vender.
Entre esas dos cosas vive el retorno — y la mayoría de proyectos se quedan en la primera.

──────────────────

𝗘𝗹 𝗽𝗿𝗼𝗯𝗹𝗲𝗺𝗮
Las empresas invierten en pronósticos más precisos y siguen sufriendo quiebres de stock.
Porque saber que se venderán 100 unidades no resuelve cuánto pedirle al proveedor. Eso depende del tiempo de entrega, del nivel de servicio que la empresa quiere ofrecer, y de cómo se gestiona el riesgo.

𝗟𝗼 𝗾𝘂𝗲 𝗰𝗼𝗻𝘀𝘁𝗿𝘂í
▸ Un puente entre el pronóstico y la orden de compra, en cuatro pasos claros
▸ Un nivel de servicio configurable — 80%, 95%, 99% — porque no todos los productos requieren la misma cobertura
▸ Un colchón de inventario que se adapta al producto, no a una fórmula promedio

𝗘𝗹 𝗮𝗽𝗿𝗲𝗻𝗱𝗶𝘇𝗮𝗷𝗲
Un pronóstico más preciso no se mide en error reducido.
Se mide en capital de trabajo liberado al mismo nivel de servicio al cliente.

──────────────────

El valor de un dato no está en su precisión. Está en la decisión que habilita.

👉 Notebook, app y playbook completo en oscarponce.com

#CRISPML #SupplyChain #InventoryManagement #DemandPlanning #AplicadaANegocios

### 📸 Imagen sugerida para Post 3

**Recomendación principal:** opción (A) — el output operativo del app cierra el arco mejor que cualquier metáfora.

**(A) Screenshot real — del app Streamlit** ⭐
- **Vista:** Tab **"Configure Scenario"** del app, con el bloque "Result" desplegado.
- **Qué mostrar:** el número de demanda predicha + el número de **Reorder Qty** debajo. Si se puede, ampliar específicamente el sub-panel donde aparece "Reorder Qty" con el slider de Safety Buffer (días) visible.
- **Por qué funciona:** muestra literalmente el puente del que habla el post — input del planner → pronóstico → orden de compra recomendada. Es la prueba visual del concepto.

**(B) Visual diseñado — diagrama de 4 pasos**
- **Layout horizontal con 4 cajas conectadas por flechas:**
  1. **Pronóstico** (ícono de gráfico) — *"Cuánto vamos a vender"*
  2. **Tiempo de entrega** (ícono de reloj) — *"Cuánto tarda el proveedor"*
  3. **Nivel de servicio** (ícono de escudo) — *"Qué cobertura quieres ofrecer"*
  4. **Orden de compra** (ícono de carrito) — *"Cuánto pedir hoy"*
- El último cuadro destacado en dorado para indicar dónde vive el ROI.
- Fuente Orbitron para los títulos.

**(C) Prompt para IA**
```
Conceptual illustration of a stone bridge connecting two cliffs across
a misty valley at sunrise. On the left cliff: a chart projection
floating in the air (the forecast). On the right cliff: a stack of
shipping boxes ready to be ordered. The bridge has 4 visible segments,
each lit by a small lamp. Editorial illustration style, muted graphite
+ gold palette, atmospheric haze. Aspect ratio 1.91:1.
```

---

# Post 4 — No todos los productos se planifican igual

El **80%** del valor del inventario está en el **20%** de los productos.
Pero la mayoría de los sistemas de planificación los trata a todos igual.
Ahí vive el mayor retorno de la analítica aplicada a supply chain.

──────────────────

𝗘𝗹 𝗽𝗿𝗼𝗯𝗹𝗲𝗺𝗮
Una sola política de inventario para todo el catálogo es la forma más rápida de fallar en los dos extremos.
Los productos de alto valor con demanda errática terminan en stockout. Los productos de bajo valor con demanda estable terminan sobre-stockeados. Y el equipo de planning gasta el mismo esfuerzo en ambos.

𝗟𝗼 𝗾𝘂𝗲 𝗼𝗯𝘀𝗲𝗿𝘃é
▸ Alta rotación + demanda estable — ahí el pronóstico se luce; ahorra capital sin sacrificar servicio
▸ Alta rotación + demanda errática — el pronóstico ayuda, pero la decisión necesita revisión semanal humana
▸ Baja rotación + demanda esporádica — el pronóstico tradicional no aplica; necesitan otro enfoque

𝗘𝗹 𝗮𝗽𝗿𝗲𝗻𝗱𝗶𝘇𝗮𝗷𝗲
El valor de un sistema de pronóstico no se distribuye uniformemente en el catálogo.
Identificar dónde se concentra ese valor — y dónde no — es trabajo previo al modelo, no posterior.

──────────────────

La pregunta no es "¿qué tan preciso es mi pronóstico?". Es "¿en qué productos vale la pena ser preciso?".

*Cuarta y última publicación de esta serie bajo CRISP-ML aplicado — gracias a quienes la siguieron. La próxima ronda toca un caso distinto.*

👉 Marco ABC/XYZ aplicado al caso completo en oscarponce.com

#CRISPML #SupplyChain #InventoryManagement #DemandPlanning #AplicadaANegocios

### 📸 Imagen sugerida para Post 4

**Recomendación principal:** opción (B) — la matriz ABC/XYZ comunica el framework completo y queda como cierre memorable de la serie.

**(A) Screenshot real — del notebook**
- **Notebook:** `Sales_Forecasting_CRISPML.ipynb`.
- **Celda a capturar:** el análisis de **error por segmento** (Phase 5, "Per-group MAE distribution" — celdas 81–82), que muestra el histograma de errores por categoría/región.
- **Cómo prepararlo:** etiquetar los grupos en lenguaje accesible antes de capturar — *"Alta rotación / estables"*, *"Alta rotación / erráticos"*, *"Baja rotación / esporádicos"*.
- **Por qué funciona:** prueba visual de que el catálogo NO es homogéneo.

**(B) Visual diseñado — matriz ABC/XYZ** ⭐
- **Layout:** matriz 3×3 con A/B/C en filas y X/Y/Z en columnas.
- **Coloreo de celdas según el rol del modelo:**
  - 🟢 Verde — "El modelo brilla" (A-X, A-Y, B-X)
  - 🟡 Amarillo — "Útil con revisión" (B-Y, C-X)
  - 🔴 Rojo — "Necesita otro enfoque" (A-Z, B-Z, C-Z)
- En la parte superior: el hook *"80% del valor en el 20% de los productos"*.
- Estética grafito + dorado.

**(C) Prompt para IA**
```
Editorial conceptual photograph: a wide supermarket aisle viewed from
above, divided into three zones by subtle gold floor markings. The
first zone is densely stocked with neat boxes (high-volume products),
the second is partially stocked (medium), the third is sparse with
just a few items (long tail). Cinematic top-down view, muted graphite
palette with gold accent lines on the floor. Aspect ratio 1.91:1.
```

---

## Notas de publicación

- **Cadencia**: 1 post por semana, martes o miércoles 09:00–10:00 LATAM.
- **Si engancha más uno**: convertir en carrusel (cada bloque = una
  slide) o en artículo largo de LinkedIn enlazando al lab.
- **CTA**: deliberadamente suave. La autoridad demostrada gana más que
  pedir el clic.

---

## Resumen visual de la serie (⭐ = recomendado)

| Post | Tema | Imagen recomendada |
|---|---|---|
| 1 | El costo escondido del inventario | ⭐ Quote card "4–8%" en grafito + dorado |
| 2 | Más datos no es mejor decisión | ⭐ Screenshot del leaderboard con etiquetas accesibles |
| 3 | Un pronóstico no es la decisión | ⭐ Screenshot del app Streamlit (Reorder Qty visible) |
| 4 | No todos los productos se planifican igual | ⭐ Matriz ABC/XYZ con celdas coloreadas |

**Patrón intencional:** alterna **diseño** (posts 1 y 4) con
**screenshot real** (posts 2 y 3). Eso mantiene variedad visual
en el feed y combina la fuerza emocional del diseño con la
credibilidad del producto real.

---

## Por qué 4 posts (y por qué el 4 cierra la serie)

- **Cada post entrega una idea de negocio distinta y accionable.**
  No se solapan: costo del problema (post 1) → por qué fallan los
  intentos comunes (post 2) → dónde está el ROI real (post 3) →
  segmentación para extraer ese ROI (post 4).
- **El Post 4 es el que más resuena con tu audiencia de supply chain
  real.** El marco ABC/XYZ es lenguaje que cualquier planner reconoce
  de inmediato — demuestra que entendés su oficio, no solo el algoritmo.
- **El arco completo:** "esto cuesta dinero" → "no es la tecnología,
  es el dato" → "el pronóstico no es la decisión" → "no todo merece el
  mismo nivel de atención". Cuatro ideas, cuatro acciones distintas.
- **Si aparece tracción**, un Post 5 podría ser un caso vertical
  (manufactura, e-commerce, distribución) o un "FVA — cómo saber si
  tu modelo realmente vale lo que cuesta".

---

## Evaluación de los notebooks y mejoras sugeridas

Resumen técnico de la revisión, para registro interno (no se publica
en LinkedIn). La narrativa accesible está arriba; aquí va la lista de
mejoras priorizadas.

### Lo que funciona bien

- **Defensa frente a leakage** — ambos notebooks aíslan columnas
  `Demand Forecast`, `Units Sold`, `Units Ordered` antes de modelar.
- **Métricas honestas** — sMAPE en lugar de MAPE cuando el target puede
  ser cero/bajo; validación con TimeSeriesSplit, no random shuffle.
- **Arquitectura Stage 1 / Stage 2** con routing por profundidad de
  historial — patrón útil para cold-start de SKUs nuevos.
- **Modelo cuantil P80** explícitamente conectado al newsvendor problem.
- **Documentación de límites** — sección "Limitations & next steps" al
  cierre de cada notebook, rara y valiosa.

### Mejoras prioritarias

1. **Rolling-origin backtest** en ambos notebooks. Hoy hay un solo
   holdout. Implementar `TimeSeriesSplit(n_splits=5)` y reportar
   `MAE media ± std`. Cambiará la confianza en la comparación
   App-aligned vs Stage 2.

2. **Multi-horizon (t+7, t+14, t+28)**. Las decisiones reales necesitan
   forecasts a lead time, no a t+1. Empezar con direct multi-horizon
   para t+7.

3. **LSTM de Inventario no aprendió** (loss plateau en epoch 5; MAE 88.9
   ≈ baseline media). Probar embeddings `(store_id, category_id)`,
   reducir `SEQ_LEN` a 14–28, subir LR inicial o agregar warmup.

4. **Per-category models en Ventas**. Spread MAE Groceries (24.5) vs
   Furniture (14.3) de 70%. Validar con CV — Furniture tiene solo ~7k
   filas y puede sobreajustar.

5. **MinT reconciliation en Inventario**. Top-down empeora los forecasts
   base (Clothing 306.8 → 541.4). Reemplazar por Minimum Trace
   (`hierarchicalforecast`).

6. **Ruta de demanda intermitente**. SKUs con `% días con demanda = 0
   > 30%` → enrutar a Croston/TSB. Saltar el modelo global.

7. **Conformal prediction** (`mapie`). Sustituye entrenar P10/P50/P90 por
   separado con un único modelo + envoltorio calibrado.

8. **Lags intermedios en Ventas**. Hoy salta de `lag_1` a `lag_7`.
   Añadir `lag_2..lag_6` y rolling-3. Bajo riesgo, posible reducción de
   MAE en patrones intra-semanales.

9. **Features cross-SKU**. Demanda media de ayer a nivel categoría y a
   nivel tienda — capturan shocks compartidos (clima, eventos) sin
   depender del histórico individual del SKU.

### Mejoras de portfolio / presentación

- Unificar artefactos en `model/inventory/` y `model/sales/`.
- README maestro que oriente al lector entre los dos labs.
- Cerrar la Streamlit del lab de Ventas — hoy está descrita en el
  notebook pero no implementada.
- Marcar explícitamente en cada notebook el límite del dataset
  sintético (sin elasticidad real, sin drift, holdout único).
