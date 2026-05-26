# Retail Store Inventory Forecasting Dataset

> A synthetic dataset for practicing inventory management and demand forecasting.

- **Autor:** Anirudh Singh Chauhan
- **Última actualización:** Hace 2 años
- **Fuente:** [Kaggle - Retail Store Inventory Forecasting Dataset](https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset)
- **Votos (upvotes):** 145
- **Usabilidad:** 10.00
- **Licencia:** CC0: Public Domain
- **Frecuencia de actualización esperada:** Never
- **Vistas:** 114K
- **Descargas:** 28K
- **Engagement:** 0.24636 descargas por vista
- **Comentarios:** 8

---

## Acerca del dataset

Este dataset proporciona datos sintéticos pero realistas para analizar y pronosticar la demanda de inventario en tiendas minoristas. Contiene más de **73,000 filas** de datos diarios a través de múltiples tiendas y productos, incluyendo atributos como ventas, niveles de inventario, precios, clima, promociones y días festivos.

Es ideal para practicar tareas de machine learning como:
- Pronóstico de demanda (demand forecasting)
- Precios dinámicos (dynamic pricing)
- Optimización de inventario

Permite a los científicos de datos explorar técnicas de pronóstico de series temporales, estudiar el impacto de factores externos como el clima y los días festivos sobre las ventas, y construir modelos avanzados para optimizar el desempeño de la cadena de suministro.

---

## Retos propuestos para Data Scientists

### Reto 1: Pronóstico de demanda con series temporales
Predecir la demanda diaria de productos en las tiendas usando datos históricos de ventas e inventario. ¿Puedes construir un modelo basado en LSTM que supere a métodos clásicos como ARIMA?

### Reto 2: Optimización de inventario
Optimizar los niveles de inventario analizando las tendencias de ventas, minimizando los desabastecimientos (stockouts) y reduciendo el exceso de stock (overstock).

### Reto 3: Precios dinámicos
Desarrollar una estrategia de precios basada en demanda, precios de la competencia y descuentos para maximizar los ingresos.

---

## Características clave de los datos

| Campo | Descripción |
|---|---|
| **Date** | Registros diarios desde la fecha de inicio hasta la fecha final. |
| **Store ID** | Identificador único de la tienda. |
| **Product ID** | Identificador único del producto. |
| **Category** | Categoría del producto (Electronics, Clothing, Groceries, Toys, Furniture, etc.). |
| **Region** | Región geográfica de la tienda. |
| **Inventory Level** | Stock disponible al inicio del día. |
| **Units Sold** | Unidades vendidas durante el día. |
| **Units Ordered** | Unidades pedidas. |
| **Demand Forecast** | Demanda pronosticada basada en tendencias pasadas. |
| **Price** | Precio del producto. |
| **Weather Condition** | Clima diario que impacta las ventas. |
| **Holiday/Promotion** | Indicadores de festividades o promociones. |

---

## Ideas para Notebooks de ejemplo

- **Análisis Exploratorio de Datos (EDA):** Analizar tendencias de ventas, visualizar datos e identificar patrones.
- **Pronóstico de series temporales:** Entrenar modelos como ARIMA, Prophet o LSTM para predecir la demanda futura.
- **Análisis de precios:** Estudiar cómo los descuentos y los precios de la competencia afectan las ventas.

---

## Información del archivo

- **Nombre del archivo:** `retail_store_inventory.csv`
- **Tamaño:** 6.19 MB
- **Versión:** 1
- **Número de columnas:** 15 (se muestran 10 en la vista compacta)
- **Rango de fechas:** 2021-12-31 a 2023-12-31

### Distribución de valores observada

- **Store ID:** 5 valores únicos (S001, S002, S003, ...)
- **Product ID:** 20 valores únicos (P0001 - P0020)
- **Categorías destacadas:** Furniture (20%), Toys (20%), Otras (60%) → incluyen Electronics, Clothing, Groceries
- **Regiones:** East (25%), South (25%), Otras (50%) → incluyen North y West
- **Inventory Level:** rango aproximado 50 – 500
- **Units Sold:** rango aproximado 0 – 499
- **Units Ordered:** rango aproximado 20 – 200
- **Demand Forecast:** rango aproximado -9.99 – 519
- **Price:** rango aproximado 10.00 – 100.00

---

## Muestra de datos

| Date | Store ID | Product ID | Category | Region | Inventory Level | Units Sold | Units Ordered | Demand Forecast | Price |
|---|---|---|---|---|---|---|---|---|---|
| 2022-01-01 | S001 | P0001 | Groceries | North | 231 | 127 | 55 | 135.47 | 33.50 |
| 2022-01-01 | S001 | P0002 | Toys | South | 204 | 150 | 66 | 144.04 | 63.01 |
| 2022-01-01 | S001 | P0003 | Toys | West | 102 | 65 | 51 | 74.02 | 27.99 |
| 2022-01-01 | S001 | P0004 | Toys | North | 469 | 61 | 164 | 62.18 | 32.72 |
| 2022-01-01 | S001 | P0005 | Electronics | East | 166 | 14 | 135 | 9.26 | 73.64 |
| 2022-01-01 | S002 | P0007 | Groceries | West | 460 | 393 | 70 | 401.48 | 91.13 |
| 2022-01-01 | S003 | P0005 | Clothing | West | 438 | 325 | 20 | 330.92 | 97.95 |

---

## Etiquetas (Tags)

- Business
- Data Analytics
- Exploratory Data Analysis
- Regression
- Python
- pandas

---

## Uso reportado por la comunidad

- **Learning:** 45
- **Research:** 16
- **LLM Fine-Tuning:** 4
- **Application:** 2

### Descripciones de la comunidad
- Well-documented: 11
- Clean data: 7
- Well-maintained: 2

---

## Notebooks relacionados

- **Retail Store Inventory Forecasting EDA & Prediction** — 147 upvotes
- **Retail Store Inventory Forecasting** — 109 upvotes
- **Demand Forecasting and Inventory Management** — 65 upvotes

---

## Datasets similares

- **Retail Store Inventory and Demand Forecasting** — Wavelet · Usability 10.0
- **Retail Sales Promotions and Demand Forecasting** — Jay Joshi · Usability 9.4
- **Demand forecasting dataset** — Ramin Huseyn · Usability 10.0
- **E-Commerce Demand Prediction dataset** — Developer · Usability 5.3