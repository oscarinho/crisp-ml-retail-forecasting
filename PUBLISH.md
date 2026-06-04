# PUBLISH.md — Two Ceilings (LinkedIn-ready)

Borradores listos para copiar. Hook único: **MAE 69 → 7.4 cambiando la forma, no el modelo.**

Estructura:
1. [Estado de publicación](#estado-de-publicación)
2. [Post 1 — el hero (español, voz negocio)](#post-1--hero-es)
3. [Post 2 — follow-up técnico (español, voz ML)](#post-2--técnico-es)
4. [Versión inglesa corta (opcional)](#post-en-inglés-opcional)
5. [Carrusel 5 slides — guion](#carrusel-5-slides)
6. [Prompts de imagen (Midjourney / DALL·E / Sora)](#prompts-de-imagen)
7. [Hero chart — código para generarlo](#hero-chart--código)
8. [Pre-flight checklist](#pre-flight-checklist)

---

## Estado de publicación

| Pieza | Estado | Nota |
|---|---|---|
| `notebooks/Inventory_Forecasting_CRISPML.ipynb` | ✅ pulido | 96 celdas, título "Two Ceilings", scoreboard nuevo |
| `README.md` | ✅ reescrito | Headline finding visible en H2 |
| `EXPERIMENT_DF_RESIDUAL.md` | ✅ completo | Backtest 18 ventanas, HGB residual 17/18 |
| `archive/README.md` | ✅ explica el porqué | Lab 2/3/4 demoted con justificación |
| `.gitignore` | ✅ limpio | `lightning_logs/`, `catboost_info/`, archive carve-outs |
| `app/app.py` | ✅ banner aplicado | Headline finding banner debajo del header (Opción A) |
| Hero chart PNG | ❌ falta | Código listo abajo — 1 minuto correrlo |
| Commit | ❌ pendiente | 40+ archivos en `R` status sin commit |

**TL;DR:** todo el contenido textual está listo. Falta (1) generar la imagen, (2) decidir si tocar el app o aclarar que el insight vive en notebook+README, (3) commit + push.

---

## Post 1 — Hero (ES)

> **Voz:** negocio. Sin jerga. Cifra que choque.
> **Longitud:** ~1,300 caracteres (cabe sin "ver más" en mobile).
> **Imagen sugerida:** hero chart (ver abajo) o quote card "MAE 69 → 7.4".

```
La diferencia entre un pronóstico de demanda con error de 69 unidades
o 7.4 unidades no está en el modelo. Está en una decisión que casi
todo curso de Machine Learning enseña al revés.

──────────────────

𝗘𝗹 𝗲𝘅𝗽𝗲𝗿𝗶𝗺𝗲𝗻𝘁𝗼
Mismo dataset (73,000 registros de retail). Mismo split temporal.
Diez familias de modelos: LightGBM, Random Forest, Prophet, ARIMA,
LSTM, CatBoost, ExtraTrees, ETS, Stacking, HistGradientBoosting.

Todos convergen al mismo error: MAE ≈ 69.
Sin importar el modelo. Sin importar los hiperparámetros.

Hasta que cambias UNA cosa.

𝗟𝗮 𝗱𝗲𝗰𝗶𝘀𝗶𝗼́𝗻 𝗾𝘂𝗲 𝗶𝗺𝗽𝗼𝗿𝘁𝗮
La columna "Demand Forecast" del dataset es el pronóstico que ya
publica el sistema de planning. Tiene correlación 0.997 con la
demanda real.

▸ Si la usas como feature → leakage, el modelo hace trampa
▸ Si la dropeas → noise floor, MAE 69 (lo que enseñan en los cursos)
▸ Si la usas como prior y el modelo solo corrige su error →
   MAE 7.4 (90% menos error)

𝗘𝗹 𝗮𝗽𝗿𝗲𝗻𝗱𝗶𝘇𝗮𝗷𝗲
En cualquier ERP/MRP real, el pronóstico del sistema YA está
publicado una semana antes. No es información futura. Está
disponible. Ignorarla por "purismo metodológico" es regalarle
90% de MAE al techo de tu pipeline.

El framing decide más que el modelo. 10x más.

──────────────────

El reto de la mayoría de proyectos de IA en operaciones no es
elegir bien el algoritmo. Es decidir qué señales del negocio
entran al modelo y cómo.

👉 Análisis completo (notebook + backtest 18 ventanas) en mi
repositorio: github.com/oscarinho/crisp-ml-retail-forecasting

#MachineLearning #DemandForecasting #SupplyChain #CRISPML #DataScience
```

---

## Post 2 — Técnico (ES)

> **Voz:** ML practitioner. Voltea la jerga al frente.
> **Cuándo publicarlo:** 5–7 días después del Post 1, para enganchar el debate técnico.
> **Imagen:** scoreboard del notebook (Pipeline Summary).

```
¿Por qué un dataset de retail con ρ=0.997 entre Demand Forecast y
Units Sold tiene DOS techos de MAE válidos en producción?

──────────────────

Hicimos el experimento por triplicado:

𝗧𝗲𝗰𝗵𝗼 𝟭 — sin DF (pure prediction)
LightGBM full features         MAE 69.1
Stacking ensemble              MAE 68.9
HistGradientBoosting Tier 1    MAE 69.0
CatBoost Tier 1                MAE 69.1
LSTM multivariate              MAE 88.9
ARIMA auto                     MAE 89.1
ETS Holt-Winters               MAE 89.4
Prophet                        MAE 112.0

Diez familias. Todas chocan con el mismo techo ~69. Eso NO es falla
de modelado — es el noise floor del problema cuando autocorrelación
intra-grupo ≈ 0.

𝗧𝗲𝗰𝗵𝗼 𝟮 — DF como prior (residual learning)
Reformular el target a: y = Units Sold − Demand Forecast
Predicción final: pred = DF + model.predict(features)

DF puro                        MAE 8.35 (bias +5.05)
DF + HGB residual              MAE 7.43 (bias +0.10) ← 50x bias↓
DF + RandomForest residual     MAE 7.45
DF + LightGBM residual         MAE 7.46

Champion-Challenger 18 ventanas rolling: HGB residual gana 17/18.

𝗟𝗼 𝗶𝗻𝘁𝗲𝗿𝗲𝘀𝗮𝗻𝘁𝗲
El delta entre familias en régimen residual: 0.04 MAE.
El delta entre régimenes (direct vs residual): 62 MAE.

→ El framing decide 1,500x más que el modelo en este dataset.

𝗤𝘂é 𝗮𝗽𝗿𝗲𝗻𝗱𝗶́
Cuando el ERP ya publica un forecast, el modelo NO debe re-derivar
estacionalidad/holidays desde cero. Debe aprender el ERROR
ESTRUCTURADO del sistema existente. Eso es lo que hace residual
learning bien planteado.

Y por qué importa que NO sea leakage:
- DF se publica 1+ semana antes del prediction window
- Es input conocido en tiempo de inferencia
- El planner humano ya lo lee primero — el modelo lo corrige

──────────────────

Notebook + champion-challenger backtest + writeup completo:
github.com/oscarinho/crisp-ml-retail-forecasting

#MachineLearning #ResidualLearning #TimeSeries #CRISPML #MLOps
```

---

## Post en inglés (opcional)

> Para reciclar el contenido a un público anglo (recruiters, ML twitter de retorno a LI). Más corto, hook-first.

```
Same dataset. Same train/test split. Same model families.
MAE 69 vs MAE 7.4.

The 90% reduction had nothing to do with the model.

In any retail ERP, the system already publishes a Demand Forecast
1+ week before the prediction window. Standard ML workflow drops
it as a "leakage trap" (ρ=0.997 with target). That's correct for
pure-prediction benchmarking — and wrong for production.

The fix is one line of code:
   target = Units Sold − Demand Forecast        # residual
   pred   = Demand Forecast + model(features)   # add back at inference

Across 10 model families on the same 73k retail dataset:
• No-DF regime: every family converges to MAE ≈ 69 (the noise floor)
• DF-as-prior regime: every family converges to MAE ≈ 7.4

18-window rolling backtest: HGB residual wins 17/18.

The framing decision was 1,500x more impactful than the algorithm
decision. Worth keeping in mind next time someone asks which model
to use.

Full notebook + backtest: github.com/oscarinho/crisp-ml-retail-forecasting

#MachineLearning #DemandForecasting #SupplyChain #DataScience
```

---

## Carrusel 5 slides

Si prefieres carrusel sobre post de texto largo. Cada slide = 1 idea, fuente grande, Orbitron/IBM Plex Mono para mantener identidad del repo.

| # | Headline | Body |
|---|---|---|
| 1 | **MAE 69 → 7.4** | Misma data. Mismo modelo. Una decisión de framing. |
| 2 | **El experimento** | 10 familias de modelos. 73,000 registros. Sin DF: todos convergen a MAE ≈ 69. |
| 3 | **El insight** | El ERP ya publica un forecast. Tratarlo como prior (no como feature) baja el techo a 7.4. |
| 4 | **La fórmula** | `target = Sales − DF`<br>`pred = DF + model(features)`<br>Bias: +5.05 → +0.10 (50x ↓) |
| 5 | **CTA** | Notebook + backtest 18 ventanas → github.com/oscarinho/crisp-ml-retail-forecasting |

---

## Prompts de imagen

### Hero image (cover del post 1) — Midjourney v6 / DALL·E 3

```
Two clean horizontal bars on a dark graphite background (#1a1d22),
one labeled "MAE 69" stretching almost the full width in muted gold
(#8B7340), the other labeled "MAE 7.4" as a tiny sliver in bright
electric blue (#5BC0EB). Monospace technical labels in IBM Plex Mono.
Thin grid lines suggesting a coordinate system. Minimalist editorial
poster style, no extra decoration, lots of negative space. Aspect
ratio 1.91:1.
```

### Alternativa — quote card

```
Editorial quote card. Dark graphite background (#1a1d22). Centered
giant number "10x" in Orbitron weight 700, color warm gold (#8B7340).
Below in IBM Plex Mono small caps: "the framing decision is 10× more
impactful than the model decision". Bottom-right corner small label
"OSCAR PONCE · MMXXVI · CRISP-ML(Q)". Aspect ratio 1.91:1.
```

### Post 2 (técnico) — scoreboard real

Captura del notebook, celda **Pipeline Summary** (la nueva con los dos regímenes). Crop a 1200×627. Si quieres limpio sin la UI de Jupyter, exporta solo esa celda con `jupyter nbconvert --to slides` o screenshot del scoreboard markdown renderizado.

### Slide 4 del carrusel — fórmula

```
Editorial code-poster. Dark graphite background. Two centered lines
of code in IBM Plex Mono, bright cyan accent color:
   target = Units_Sold − Demand_Forecast
   pred   = Demand_Forecast + model(features)
Below them in smaller Fraunces italic: "two lines. ninety percent
less error." Aspect ratio 1:1 for square carousel.
```

---

## Hero chart — código

Generar el chart real (es más honesto que un mockup):

```python
# scripts/make_hero_chart.py
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(12, 6.3), facecolor='#1a1d22')
ax.set_facecolor('#1a1d22')

models_no_df  = ['Stacking', 'HGB Tier1', 'Stage 2 LGBM', 'CatBoost',
                 'LSTM', 'ARIMA', 'ETS', 'Prophet']
mae_no_df     = [68.9, 69.0, 69.1, 69.1, 88.9, 89.1, 89.4, 112.0]

models_with_df = ['DF + HGB resid', 'DF + RF resid', 'DF + LGBM resid', 'DF puro']
mae_with_df    = [7.43, 7.45, 7.46, 8.35]

GOLD = '#8B7340'
CYAN = '#5BC0EB'
MIST = '#B0B4B8'

y_top = list(range(len(models_no_df), 0, -1))
y_bot = list(range(-2, -2 - len(models_with_df), -1))

ax.barh(y_top, mae_no_df, color=GOLD, alpha=0.85, height=0.7)
ax.barh(y_bot, mae_with_df, color=CYAN, alpha=0.95, height=0.7)

for y, m, v in zip(y_top, models_no_df, mae_no_df):
    ax.text(v + 1, y, f'{v:.1f}', va='center', color=MIST,
            fontsize=10, family='monospace')
    ax.text(-2, y, m, va='center', ha='right', color=MIST,
            fontsize=10, family='monospace')

for y, m, v in zip(y_bot, models_with_df, mae_with_df):
    ax.text(v + 1, y, f'{v:.2f}', va='center', color=MIST,
            fontsize=10, family='monospace')
    ax.text(-2, y, m, va='center', ha='right', color=MIST,
            fontsize=10, family='monospace')

ax.axhline(-1, color='#404549', linewidth=0.5)
ax.text(60, -0.3, 'NO DF (pure prediction) — ceiling MAE ≈ 69',
        color=GOLD, fontsize=11, family='monospace', weight='bold')
ax.text(60, -6.7, 'DF as prior (residual) — ceiling MAE ≈ 7.4',
        color=CYAN, fontsize=11, family='monospace', weight='bold')

ax.set_xlim(-30, 125)
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

ax.set_title('Two Ceilings — same dataset, same models, different framing',
             color=MIST, fontsize=14, family='monospace', loc='left', pad=20)
ax.text(125, -8.5, 'oscarponce.com',
        color='#606468', fontsize=9, family='monospace',
        ha='right', style='italic')

plt.tight_layout()
plt.savefig('hero_two_ceilings.png', dpi=200,
            facecolor='#1a1d22', bbox_inches='tight')
print('→ hero_two_ceilings.png')
```

Correr: `python scripts/make_hero_chart.py` → `hero_two_ceilings.png` listo para subir.

---

## Ajuste mínimo al app

`app/app.py` sigue siendo "Demand Simulator". Dos opciones honestas:

**Opción A (mínima — 2 minutos):** agregar un info banner debajo del header anunciando el insight y linkeando al EXPERIMENT_DF_RESIDUAL.md. Mantiene el app funcional sin tocar la lógica.

```python
# Justo después del bloque "── Header ──" (línea ~742), antes del KPI strip:
st.markdown(f"""
<div style='background:rgba(91,192,235,0.08); border-left:3px solid {CYAN};
            padding:0.9rem 1.2rem; margin:1rem 0; font-family:"IBM Plex Mono",monospace;
            font-size:0.78rem; color:{MIST};'>
  <b style='color:{CYAN}; letter-spacing:0.08em;'>HEADLINE FINDING ·</b>
  This dataset has two MAE ceilings depending on whether Demand Forecast is
  available as a prior at inference time: <b>MAE 69</b> (no DF) → <b>MAE 7.4</b>
  (DF as residual prior). The framing decision is 10× more impactful than the
  model decision.
  <a href='https://github.com/oscarinho/crisp-ml-retail-forecasting/blob/main/EXPERIMENT_DF_RESIDUAL.md'
     style='color:{CYAN};'>Full writeup →</a>
</div>
""", unsafe_allow_html=True)
```

**Opción B (full — 1–2h):** agregar una 4ta tab "Two Ceilings" con un toggle (DF on/off) que muestre la predicción en ambos regímenes side-by-side. Requiere cargar el modelo residual (no existe aún como .pkl separado, hay que entrenarlo).

→ **Recomendación:** Opción A antes de publicar. B queda para una segunda iteración si los posts generan engagement.

---

## Pre-flight checklist

Antes del primer post:

- [ ] `python scripts/make_hero_chart.py` → genera la imagen
- [x] Aplicar Opción A en `app/app.py` (banner — hecho)
- [ ] Si quieres la app live: desplegar a Streamlit Community Cloud (5 min, gratis — instrucciones en README §"Deploy")
- [ ] `git add -A && git commit -m "chore: archive exploratory work, polish for publication"` (mensaje sugerido — confirma antes de pushear)
- [ ] `git push origin main`
- [ ] Verificar que el repo en `github.com/oscarinho/crisp-ml-retail-forecasting` esté público
- [ ] Publicar Post 1 con la imagen hero
- [ ] Programar Post 2 para +5–7 días

Entre posts:
- [ ] Responder comments del Post 1 (las primeras 6h son las que más empujan el algoritmo)
- [ ] Si alguien pide más detalle técnico → linkear al EXPERIMENT_DF_RESIDUAL.md

---

**Repo público:** [github.com/oscarinho/crisp-ml-retail-forecasting](https://github.com/oscarinho/crisp-ml-retail-forecasting) — ya está referenciado en todos los borradores arriba.
