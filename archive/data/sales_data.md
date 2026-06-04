# Retail Store Inventory and Demand Forecasting

**Fuente:** [Kaggle](https://www.kaggle.com/datasets/atomicd/retail-store-inventory-and-demand-forecasting)
**Autor:** WAVELET
**Actualizado:** Hace un año
**Votos (upvotes):** 59

> ℹ️ Este dataset fue generado sintéticamente y puede no reflejar datos del mundo real.

## Descripción

Un dataset sintético para pronóstico de inventario y demanda.

Este dataset sigue el formato del *Retail Store Inventory Forecasting Dataset* y corrige entradas mal etiquetadas como IDs de tienda y de producto. Además, incluye una característica **Epidemic** para simular las condiciones del comercio minorista durante la pandemia de COVID-19, mejorando el realismo y el valor práctico de los datos. Estas mejoras buscan hacer que el dataset sea más adecuado para tareas de pronóstico de series temporales.

## Información general

| Atributo | Valor |
|----------|-------|
| **Usabilidad** | 10.00 |
| **Licencia** | Apache 2.0 |
| **Frecuencia de actualización** | Semanal |
| **Archivo** | sales_data.csv (6.35 MB) |
| **Columnas** | 16 |
| **Vistas totales** | 31,700 |
| **Descargas totales** | 8,124 |
| **Versión** | 8 |

## Etiquetas (Tags)

- Business
- Retail and Shopping
- Time Series Analysis
- Synthetic
- Linear Regression

## Estructura del Dataset

| Columna | Tipo | Descripción |
|---------|------|-------------|
| Date | Fecha | Fecha del registro |
| Store ID | Texto | Identificador único de la tienda |
| Product ID | Texto | Identificador único del producto |
| Category | Texto | Categoría del producto |
| Region | Texto | Región geográfica de la tienda |
| Inventory Level | Numérico | Unidades disponibles en stock |
| Units Sold | Numérico | Unidades vendidas en ese día |
| Units Ordered | Numérico | Unidades pedidas para reabastecimiento |
| Price | Numérico | Precio del producto |
| Discount | Numérico | Descuento aplicado (si lo hay) |
| Weather Condition | Texto | Condición climática del día del registro |
| Promotion | Binario | 1 si hubo promoción, 0 en caso contrario |
| Competitor Pricing | Numérico | Precio de un producto similar de la competencia |
| Seasonality | Texto | Estación (ej. Invierno, Primavera) |
| Epidemic | Binario | 1 si ocurrió una epidemia, 0 en caso contrario |
| Demand | Numérico | Demanda diaria estimada del producto |

## Cobertura temporal

- **Rango de fechas:** 2022-01-01 a 2024-01-29
- **Registros por periodo:** ~3,800 registros cada ~38 días
- **Tiendas únicas:** 5
- **Productos únicos:** 20

## Distribución de categorías

- Groceries: 40%
- Furniture: 18%
- Otros: 42%

## Distribución por región

- North: 40%
- South: 20%
- Otros: 40%

## Rangos numéricos principales

| Variable | Mínimo | Máximo |
|----------|--------|--------|
| Inventory Level | 0 | 2,267 |
| Units Sold | 0 | 426 |
| Units Ordered | 0 | 1,616 |
| Price | 4.74 | 228 |
| Discount | 0 | 25 |

## Casos de uso reportados por la comunidad

- Aprendizaje (Learning): 29
- Aplicación (Application): 4
- Investigación (Research): 2
- LLM Fine-Tuning: 0

## Calidad percibida

- Bien documentado: 8
- Datos limpios: 7
- Bien mantenido: 1

## Notebooks relacionados destacados

1. **Retail store inventory and demand forecasting** — 36 votos
2. **ML & DL: RFR LR XGB MLP Demand Comparison** — 20 votos
3. **Mission: Retail Category Forecasting** — 15 votos

## Datasets similares

- *Store sales dataset* — Abhishek Jaiswal (Usability 10.0)
- *Retail Sales Promotions and Demand Forecasting* — Jay Joshi (Usability 9.4)
- *Demand Forecasting Dataset* — Python Developer (Usability 4.7)
- *Synthetic Retail Dataset — 1.2M Transactions* — Amir khan (Usability 7.1)