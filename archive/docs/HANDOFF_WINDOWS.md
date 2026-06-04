# Handoff a Windows — cómo correr las libretas

> Guía mínima para pasar este repo de la Mac (16 GB, libomp dañado) a tu
> laptop Windows 11 (64 GB + RTX 2070). Léeme una vez en la máquina nueva
> y ejecuta los pasos en orden. **Tiempo total estimado: 30-40 min** de
> setup + 1h de correr todo.

---

## Paso 0 — Antes de cerrar la Mac, push del repo

Estos commits están solo locales. Si los pierdes, los pierdes:

```bash
cd /Users/oscarponce/Documents/personal/marca-personal/labs/forecasting-inventory
git log --oneline -8        # verifica que estos commits están
git push origin main         # los manda a GitHub
```

Los commits que necesitas en GitHub para continuar en Windows:

| Hash | Mensaje |
|---|---|
| `3582235` | feat(labs): add Food Demand + Store Sales labs (4 labs total) |
| `b6afb21` | feat(notebooks): add Tier 1/2 benchmarks + residual + champion-challenger |
| `1a53f8f` | docs: enrich README + TEST_CASES with dataset provenance |
| `0dfd757` | docs: add TEST_CASES.md with QA scenarios + generator script |

Si `git push` te pide credenciales — usa un Personal Access Token de GitHub
(Settings → Developer settings → Personal access tokens).

**Archivos que NO van por git (los descargas en Windows directo):**

- `data/store-sales-time-series-forecasting/train.csv` (116 MB, gitignored)
- `CRISP-ML-vic/` (823 MB, gitignored — no lo necesitas en Windows)
- `notebooks/checkpoints/` (cache local de modelos — se regenera al correr)

---

## Paso 1 — En la laptop Windows: instalar Miniconda

1. Descarga el instalador 64-bit:
   https://docs.conda.io/projects/miniconda/en/latest/

2. Durante la instalación, **marca la casilla "Add to PATH"** (te ahorra
   dolor después).

3. Al terminar, abre **"Anaconda Prompt"** desde el menú inicio (NO
   PowerShell normal — es un wrapper distinto que ya tiene `conda`
   inicializado).

---

## Paso 2 — Clonar el repo + crear el entorno

En **Anaconda Prompt**:

```bat
cd C:\Users\TuUsuario\Documents
git clone https://github.com/oscarinho/crisp-ml-retail-forecasting.git
cd crisp-ml-retail-forecasting

REM Crear el environment
conda create -n forecasting python=3.11 -y
conda activate forecasting

REM PyTorch con CUDA 12.1 (conda baja el CUDA toolkit por ti — no hace falta instalar CUDA aparte)
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y

REM Resto del stack
pip install -r requirements.txt
```

> **Si `pip install -r requirements.txt` truena** (más probable en
> `prophet`, `pmdarima`, o `neuralforecast`):
> ```bat
> pip install pandas numpy scikit-learn lightgbm catboost statsmodels matplotlib seaborn plotly joblib jupyter streamlit
> pip install statsforecast neuralforecast
> ```
> Y deja `prophet` / `pmdarima` para después (solo se usan en celdas de
> "classical baselines" que ya tienen `try/except` defensivo).

---

## Paso 3 — Verificar que detecta la RTX 2070

```bat
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'); print('VRAM:', torch.cuda.get_device_properties(0).total_memory / 1e9, 'GB' if torch.cuda.is_available() else '')"
```

Debe imprimir algo como:
```
CUDA available: True
Device: NVIDIA GeForce RTX 2070
VRAM: 8.0 GB
```

**Si dice `CUDA available: False`** → actualiza el driver NVIDIA con GeForce
Experience (https://www.nvidia.com/en-us/geforce/geforce-experience/) y
re-corre.

---

## Paso 4 — Descargar el train.csv de Store Sales (Lab 4)

Si NO quieres correr el Lab 4: **salta este paso**.

Si sí quieres:

1. Ve a https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data
2. **Accept rules** (botón abajo) — necesitas estar logueado
3. Descarga el ZIP completo
4. Extrae `train.csv` directamente a `data/store-sales-time-series-forecasting/`

O con CLI (más rápido si ya usas Kaggle):

```bat
pip install kaggle
REM coloca tu kaggle.json en C:\Users\TuUsuario\.kaggle\kaggle.json
kaggle competitions download -c store-sales-time-series-forecasting -p data/store-sales-time-series-forecasting
cd data\store-sales-time-series-forecasting
tar -xf store-sales-time-series-forecasting.zip
```

---

## Paso 5 — Lanzar Jupyter Lab

Desde Anaconda Prompt, en el directorio del repo, con `forecasting`
activado:

```bat
jupyter lab
```

Se abre el navegador automáticamente. Si no, copia la URL que sale en
consola.

---

## Paso 6 — Orden recomendado de ejecución

Cuatro libretas en `notebooks/`. **No las corras todas a la vez — abre una
a la vez para no comer RAM innecesaria.**

### Orden sugerido

| # | Libreta | Tiempo (en tu laptop) | Crítico? |
|---|---|---:|---|
| 1 | `Inventory_Forecasting_CRISPML.ipynb` | 10-15 min | ✓ (el más importante — incluye DF residual experiment) |
| 2 | `Sales_Forecasting_CRISPML.ipynb` | 8-12 min | ✓ |
| 3 | `Food_Demand_Forecasting_CRISPML.ipynb` | 5-10 min | opcional |
| 4 | `Store_Sales_Forecasting_CRISPML.ipynb` | 20-30 min | opcional (necesita train.csv del paso 4) |

### Cómo correr cada una

1. Abre la libreta
2. `Kernel > Restart Kernel` (limpia estado)
3. `Run > Run All Cells`
4. Espera. Si truena alguna celda, anota cuál y baja a la sección "Errores
   comunes" más abajo.

### Mientras tanto, en otra Anaconda Prompt (opcional):

Puedes correr los experiment scripts directamente desde shell — más rápido
que la libreta porque no carga el preprocesamiento de matplotlib y demás:

```bat
python scripts\df_experiments.py          REM ~5 min · genera df_experiments_results.json
python scripts\run_food_demand.py         REM ~3 min · genera food_demand_results.json
python scripts\run_store_sales.py         REM ~10 min · genera store_sales_results.json
```

---

## Paso 7 — (Opcional) Acelerar Tier 2 con la GPU

Las celdas de Tier 2 (`§4.11.x` — N-BEATS, NHITS, DLinear, DeepAR, TFT)
están seteadas con `accelerator='cpu'` para protegerme del bug de Mac/MPS.
En tu Windows con CUDA, **cambia esa línea a `'gpu'`** en cada celda Tier 2:

```python
NHITS(h=H, input_size=INPUT_SIZE, loss=NFMAE(),
      max_steps=500, batch_size=64, random_seed=42,
      accelerator='gpu',  # ← era 'cpu'
)
```

**Si te quedas corto de VRAM** (RTX 2070 = 8 GB), baja `batch_size` de 64 a
32 o 16.

### CatBoost en GPU (3-5× más rápido en tu hardware)

En las celdas `§4.10.1`, añade `task_type='GPU'`:

```python
m = CatBoostRegressor(
    iterations=600, learning_rate=0.05, depth=6,
    loss_function='MAE', cat_features=CB_CAT_COLS,
    nan_mode='Min', random_seed=42, verbose=False,
    task_type='GPU',  # ← agregar
)
```

**LightGBM en GPU en Windows nativo es un infierno** de compilar. Déjalo
en CPU — corre rápido sin GPU igual.

---

## Errores comunes y fix rápido

### ❌ `Cannot convert 'S001' to float` en celda CatBoost
**Ya está fixeado en el último upsert.** Si reaparece, abre la celda y
verifica que diga:
```python
CB_CAT_COLS = [c for c in S2_CAT if c in S2_FEATURES]  # NO la versión hardcoded
```
Si dice la versión hardcoded vieja, re-corre `python scripts/add_tier12_cells.py`.

### ❌ `OMP: Error #15: libomp.dylib already initialized`
**Esto es solo de Mac.** En Windows no aparece. Si lo ves, algo raro pasó.

### ❌ `ImportError: No module named 'lightgbm'`
`pip install lightgbm` dentro del env `forecasting`. Verifica con
`conda env list` que estés en el env correcto.

### ❌ `KeyError: 'Demand Forecast'` en residual learning
La columna `Demand Forecast` solo existe en `data/retail_store_inventory.csv`
(Lab 1). En las otras labs no existe — pero el código tiene un `if HAS_DF`
check. Si te truena, revisa que estés en la libreta correcta.

### ❌ Kernel muere sin mensaje (segfault)
En Mac era el bug de libomp. En Windows con 64 GB esto NO debería pasar.
Si pasa:
- Abre Task Manager → mira el pico de RAM cuando muere
- Si >32 GB → es la libreta de Store Sales con 3M rows. Reinicia kernel y
  re-corre. La segunda vez `cached()` lee de disco y no come RAM.
- Si <16 GB → reporta el traceback completo, hay otro bug

### ❌ `CUDA out of memory` durante Tier 2
RTX 2070 tiene 8 GB, justo. Soluciones:
1. Baja `batch_size` de 64 → 32 → 16
2. Baja `input_size` de 28 → 14
3. Reduce `max_steps` de 500 → 300

### ❌ "Notebook not trusted" warning
Inofensivo. Marca la libreta como trusted: menú `File > Trust Notebook`.

### ❌ `jupyter: command not found` en Anaconda Prompt
Olvidaste `conda activate forecasting`. Re-actívalo.

---

## Qué archivos consultar al final

Después de correr todo, los resultados quedan en:

| Archivo | Contenido |
|---|---|
| `scripts/df_experiments_results.json` | Numbers del residual + champion-challenger |
| `scripts/food_demand_results.json` | Leaderboard de food demand |
| `scripts/store_sales_results.json` | Leaderboard de store sales |
| `notebooks/checkpoints/*.pkl` | Cache de modelos entrenados (re-corres en segundos) |
| `model/food_demand/best_model.pkl` | Mejor modelo deployable (food demand) |
| `model/store_sales/best_model.pkl` | Mejor modelo (per-family LightGBM, 73 MB) |

Para entender qué hicimos:
- [`README.md`](README.md) — overview de los 4 labs
- [`LESSONS_LEARNED.md`](LESSONS_LEARNED.md) — recap completo + cat-and-mouse experiment
- [`EXPERIMENT_DF_RESIDUAL.md`](EXPERIMENT_DF_RESIDUAL.md) — el experimento clave

---

## Si algo nuevo se rompe

Pásame:

1. **Qué comando o celda estabas corriendo** (paso del 1-7 de esta guía o
   número de celda en la libreta)
2. **Las últimas 15 líneas del traceback**
3. **Output de:**
   ```bat
   python -c "import sys; print(sys.version); import torch; print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
   ```

Y voy directo al fix sin pedirte más contexto.

---

## Checklist final antes de cerrar la Mac

- [ ] `git push origin main` ejecutado y subido
- [ ] Verificaste que GitHub muestra los últimos commits (`3582235` etc.)
- [ ] (Opcional) Backup de `notebooks/checkpoints/` a un USB si no quieres
      re-fitear los modelos — pero realmente no vale la pena, se regenera
      en minutos en tu laptop nueva
- [ ] Este archivo (`HANDOFF_WINDOWS.md`) en tu repo y en el push

Listo. Cierra la Mac. Continúa en Windows con esta guía.

— Junio 2026
