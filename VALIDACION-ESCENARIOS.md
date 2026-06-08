# Escenarios de validación — Streamlit (inventory-lab)

App: https://inventory-lab.streamlit.app · Tab **Demand Simulator** salvo que se indique.

> Nota: el simulador usa el modelo **contextual deployable** (sin la columna `Demand Forecast`). Valida el comportamiento operativo, la lógica de reorden/P80 y las propiedades del dataset. El **MAE 7.4** es del backtest del notebook (regimen residual), no del form — eso se valida en el notebook, no aquí.

---

## Escenario A — Sanity check (baseline)
**Inputs:** Store/Category/Region cualquiera · Month 6 · Weekday Wed · Holiday OFF · Price 35 · Discount 10 · Competitor 37 · **Inventory 250**.
**Esperado:** demanda diaria razonable (decenas de unidades), Coverage ≈ inventory/predicción, status OK o LOW.
**Valida:** el modelo devuelve una predicción sana y no degenerada.

## Escenario B — Riesgo de quiebre (CRITICAL + reorden)
**Inputs:** iguales a A pero **Inventory 50** (mínimo).
**Esperado:** Coverage bajo, status **CRITICAL / LOW STOCK**, aparece "Recommended reorder: N units".
**Valida:** la lógica de reorden se dispara cuando el inventario no cubre la demanda.

## Escenario C — Sobre-stock (sin reorden)
**Inputs:** iguales a A pero **Inventory 500** (máximo).
**Esperado:** status **OK**, Coverage alto, "✓ No reorder needed".
**Valida:** no hay reorden falso cuando estás bien abastecido.

## Escenario D — P80 vs punto (regla newsvendor)
**Inputs:** iguales a B (Inventory 50). Compara la **demanda predicha** (punto) contra el **target de reorden**.
**Esperado:** el target de stock que usa para reordenar es ≥ la predicción puntual (cubre el P80, no la media).
**Valida:** el demo aplica P80 (cubrir demanda 4 de 5 días), que es la métrica de decisión del lab — no el pronóstico puntual.

## Escenario E — Elasticidad de precio ≈ 0 (caveat honesto)
**Inputs:** desde A, mueve el slider **Price** por todo su rango (o lee la **Fig.01 "Demand vs Price Sensitivity"**, banda ±20%).
**Esperado:** la curva demanda–precio es casi **plana**.
**Valida:** confirma el hallazgo documentado "pricing elasticity ≈ 0" del dataset sintético. El modelo **no inventa** una elasticidad que el dato no tiene (honestidad metodológica).

## Escenario F — Qué features sí mueven la aguja (Holiday / weekend)
**Inputs:** desde A, alterna **Holiday/Promotion** ON vs OFF (y/o Weekday Wed vs Sat), todo lo demás igual.
**Esperado:** un cambio pequeño en la predicción; el mayor driver sigue siendo **Inventory Level** y el contexto.
**Valida:** alineado con el análisis SHAP (Inventory Level domina); el dataset es de señal contextual, no temporal.

## Escenario G — Reorder Advisory: safety buffer
**Tab:** Reorder Advisory. Mueve **Safety Buffer (days)** de 3 → 14.
**Esperado:** la lista de riesgo y las cantidades de reorden crecen con un buffer mayor.
**Valida:** la regla de punto de reorden (ROP) responde al lead time / nivel de servicio.

---

## Qué deberías concluir si todo valida
- La predicción es estable y razonable (A), y la **lógica de reorden** responde correctamente a inventario alto/bajo (B, C).
- El sistema decide con **P80**, no con la media (D) → menos quiebres al mismo nivel de inventario.
- El modelo es **honesto con los límites del dato**: no finge elasticidad (E) ni señal temporal que no existe (F).
- El buffer de seguridad es ajustable al nivel de servicio del negocio (G).

Si algo NO se comporta así (ej. reorden que no aparece con inventario en 50, o elasticidad fuerte que el dataset no tiene), es una bandera para revisar el pipeline de la app.
