# LinkedIn Content Plan — Retail Demand Forecasting Labs

Post series built from the `forecasting_inventory` CRISP-ML(Q) project — now **two
sibling labs** on the same retail problem with two different datasets:

| Lab | Dataset | Target | Best holdout MAE | App |
|---|---|---|---:|---|
| **Inventory** | `retail_store_inventory.csv` (synthetic, memoryless) | `Units Sold` | 69 (data ceiling) | `app/app.py` |
| **Sales** | `sales_data.csv` (autocorrelated, censored by stockouts) | `Demand` (uncensored) | **19.5** (per-category routing) | `app/app_sales.py` |

Goal: showcase end-to-end ML rigor across two contrasting forecasting setups,
drive traffic to oscarponce.com, and reach two audiences —
**ML/data engineers** and **supply chain / ops professionals**.

---

## Strategy at a glance

| Item | Decision |
|---|---|
| **Cadence** | 1 post / week — 6 weeks Inventory, 4 weeks Sales, 1 closing meta = **11 weeks total** |
| **Voice** | First-person, plain-spoken, "here's what I found" — not academic |
| **Hook style** | Lead with a surprising number or a mistake, never "I built a model" |
| **CTA** | Soft — "full notebook on oscarponce.com", "what would you have done?" |
| **Visual** | Every post needs one image: chart screenshot, scoreboard table, or diagram |
| **Length** | 800–1,300 characters; first 2 lines must hook before the "...more" fold |
| **Hashtags** | 3–5 max, end of post: mix broad + niche |

**Posting order is deliberate** — Inventory series first (it has the most
counterintuitive ML stories), then Sales (more applied / deployment-flavored),
close with the dual-lab meta-post.

---

## SERIES A — Inventory Lab (Posts 1–6)

> The story: a synthetic, memoryless dataset where every ML rope you'd pull on
> snaps at the same MAE floor (~69). The lesson is *epistemic* — knowing when
> the model is constrained by the data, not by your architecture.

## Post 1 — The "perfect" forecast that was a trap

**Angle:** Data leakage. The dataset shipped a `Demand Forecast` column with ρ=0.997
to actual sales. Using it would look genius and be worthless.

**Hook:**
> A column in my dataset predicted sales with 99.7% correlation.
> I deleted it. Here's why that was the right call.

**Body beats:**
- A near-perfect feature is a red flag, not a gift.
- That column wouldn't exist on real forecast day → leakage.
- Honest baseline: oracle MAE 8 vs. a model that has to *actually work* at MAE 69.
- Lesson: the first job in forecasting is auditing what you're allowed to know.

**Visual:** correlation heatmap / leakage-audit chart (cell `30dbc994`).
**Hashtags:** #MachineLearning #DataScience #MLOps #Forecasting
**Audience:** ML engineers (primary).

---

## Post 2 — Classical vs. deep learning: the upset

**Angle:** Five model families compared on the same split. The LSTM did *not* win.

**Hook:**
> I trained an LSTM, Prophet, ARIMA, ETS and a gradient-boosted model on the
> same data. The deep learning model came 5th.

**Body beats:**
- Scoreboard (test MAE): Stacking 68.9 · LightGBM 69.1 · LSTM 88.9 · ARIMA 89.1 · Prophet 112.0.
- Tabular + engineered features → boosted trees still rule.
- Deep learning isn't a default; it's a tool for a specific shape of problem.
- The expensive model lost. Measure before you commit.

**Visual:** the Phase 5 MAE bar chart / scoreboard table.
**Hashtags:** #DeepLearning #MachineLearning #DataScience #LightGBM
**Audience:** ML engineers.

---

## Post 3 — A forecast number is useless to a planner

**Angle:** The translation gap. "MAE 69" means nothing on a shop floor. Bridge it.

**Hook:**
> "Your model is off by 69 units on average."
> A supply chain planner cannot do anything with that sentence. So I fixed it.

**Body beats:**
- Walk the 4-step bridge: forecast → lead-time demand → safety stock → reorder point.
- Service level is a business dial, not a model setting.
- Newsvendor rule: optimal service level = stockout cost ÷ (stockout + overstock cost).
- A model only earns its keep when its output becomes a *decision*.

**Visual:** the 4-step reorder-point worked-example diagram.
**Hashtags:** #SupplyChain #InventoryManagement #DemandPlanning #DataScience
**Audience:** supply chain pros (primary) — your reach-widening post.

---

## Post 4 — Why I forecast the 80th percentile, not the average

**Angle:** Point forecasts quietly cause stockouts. Quantile forecasting fixes it.

**Hook:**
> If you stock to the *average* forecast, you run out half the time.
> Here's the forecasting trick most demos skip.

**Body beats:**
- A mean forecast = a coin-flip service level.
- The P80 quantile model predicts the level demand stays under 80% of the time.
- The gap between P80 and the mean *is* your safety stock — data-driven, not a rule of thumb.
- Errors are asymmetric: a stockout ($20) hurts ~4× more than an overstock ($5).

**Visual:** actual vs. P80 prediction band chart (cell `12997c61`).
**Hashtags:** #Forecasting #SupplyChain #DataScience #InventoryManagement
**Audience:** both.

---

## Post 5 — The engineering nobody applauds (but should)

**Angle:** Checkpointing, leakage-safe lag features, time-based splits — the unglamorous
craft that separates a notebook from a pipeline.

**Hook:**
> The cell that took 8 minutes to run had a bug in the cell *after* it.
> I'd already paid that cost 5 times. So I built a fix.

**Body beats:**
- Disk checkpointing: heavy fits cache to `.pkl`, re-runs skip them.
- Leakage-safe lag features: `shift()` only looks backward, within each group.
- Time-based split, never random — you can't train on the future.
- Reproducibility is a feature. Boring engineering is what makes results trustworthy.

**Visual:** code snippet of the `cached()` helper, or the pipeline-summary diagram.
**Hashtags:** #MLOps #MachineLearning #DataEngineering #Python
**Audience:** ML engineers.

---

## Post 6 — What Series A taught me

**Angle:** Reflective wrap-up of the Inventory series. Hands off into Series B.

**Hook:**
> 6 weeks, 11 models, one retail dataset. The biggest lesson wasn't a model —
> it was learning when the dataset itself was the ceiling.

**Body beats:**
- A notebook that only ML people can read is half-built.
- Recap: leakage audit, model bake-off, quantiles for inventory, planner translation layer.
- CRISP-ML(Q) end to end: business understanding → deployment playbook.
- "Next week I run the same pipeline on a sibling dataset that *does* have signal. The findings reverse." (teases Series B)
- Full annotated notebook + supply-chain section on oscarponce.com.

**Visual:** carousel — 1 slide per earlier post's key visual.
**Hashtags:** #DataScience #MachineLearning #SupplyChain #Portfolio #CRISPML
**Audience:** both — recruiters and peers.

---

## SERIES B — Sales Lab (Posts 7–10)

> The story: same retail problem, different dataset. This one **has** temporal
> memory (lag-1 autocorr 0.35) but stockouts censor 70% of `Units Sold`. So we
> forecast `Demand` (uncensored) and the lessons flip — adding more features
> *hurts*, the cheap log transform doesn't help, and the biggest win comes from
> a routing trick the leaderboard didn't predict.

## Post 7 — I added 14 features to my model. It got worse.

**Angle:** The overfit smoking gun. Stage 2 has 29 features; App-aligned has 15
and beats it on holdout. Less truly is more — when you can prove it.

**Hook:**
> My "full" model had 29 features. My "small" model had 15.
> The small one won by 5% MAE. Here's how I caught it.

**Body beats:**
- Holdout-vs-train MAE gap: Stage 2 LightGBM **+22.7%**, App-aligned **+14.5%**. The lag/rolling features overfit.
- I tried `early_stopping` to rescue the lags. Best iteration came in at 628 (default 600) — the model wanted *more* trees, not fewer. **Not classical overfit; structural.**
- Lag-1 autocorrelation of 0.35 isn't enough to beat the noise the trees memorise.
- Deploy story: ship the smaller model. Save the bigger one for the day you have a rolling-origin backtest that proves otherwise.

**Visual:** the 4-row train-vs-holdout gap table (notebook §4.5 output).
**Hashtags:** #MachineLearning #LightGBM #Forecasting #DataScience
**Audience:** ML engineers (primary).

---

## Post 8 — I expected log-transform to fix it. It didn't.

**Angle:** Closing a hypothesis with evidence. Heteroscedasticity ratio 1.47
suggested log-target should help. We ran it. It didn't move the needle.

**Hook:**
> Textbook said: "If residual std grows with prediction, log-transform the target."
> I ran the experiment. My MAE went *up* by 0.2 units. Here's why.

**Body beats:**
- The diagnostic looked promising: residual std 21.8 (low-half) → 32.2 (high-half), ratio 1.47.
- I fit App-aligned on `log1p(Demand)`, predicted, back-transformed via `expm1`.
- Per-category delta: Furniture −0.13 (helped), Groceries +0.30 (hurt), others ≈ flat.
- The hetero ratio reflects *demand-level variability* (some products are 10×, others 100×), not the level-dependent error a log fix would address.
- **The lesson isn't "log1p is bad."** It's: *run the experiment instead of arguing about it.*
- Total cost: one cached cell (`sales_app_log1p`), ~30 seconds. Worth every second.

**Visual:** the per-category linear-vs-log MAE comparison table (§4.8).
**Hashtags:** #DataScience #MLOps #FeatureEngineering #MachineLearning
**Audience:** ML engineers (primary), data scientists.

---

## Post 9 — The 3% MAE gain came from the category I least expected

**Angle:** Per-category routing as a deployment pattern. The biggest gain came
from Clothing/Electronics, not Groceries (the worst-performing category).

**Hook:**
> Groceries was my worst category — MAE 24, vs. 14 for Furniture.
> I trained a dedicated Groceries model expecting to crush it.
> Groceries improved by less than 1%. Clothing improved by 9.5%.

**Body beats:**
- The "best" model isn't always the one fixing the worst case.
- Per-category breakdown (vs. global model): Clothing **−9.5%**, Electronics **−9.2%**, Toys −3.1%, Furniture −0.3%, Groceries **−0.7%**.
- Why Groceries didn't improve: 3,600 holdout rows, intrinsic noise from a 70%-stockout regime — model architecture can't fix dataset variance.
- Why Clothing did: smaller training set (1,440 holdout rows), more concentrated patterns, dedicated model finds them.
- Deployment pattern: `routing[Category] if Category in keys else fallback`. Five LightGBMs + one safety-net model. Live in `app/app_sales.py`.

**Visual:** the per-category delta table from §4.9 + a screenshot of the routing pill in the Streamlit app.
**Hashtags:** #SupplyChain #MachineLearning #Forecasting #DataScience
**Audience:** both.

---

## Post 10 — Two labs. Same dataset shape. Opposite conclusions.

**Angle:** Meta close of both series. Side-by-side comparison.

**Hook:**
> Same 5 stores × 20 products × ~2 years.
> Dataset A: classical ML wins. Dataset B: the simplest model wins.
> Same code. Reversed conclusions. Here's what changed.

**Body beats:**
- **Inventory dataset:** within-group lag-1 autocorr ≈ 0. MAE floor 69. Stacking, LSTM, ARIMA — all converge to the noise ceiling.
- **Sales dataset:** within-group lag-1 autocorr 0.35. **Yet lag features still hurt** because they overfit in a 64k-row training window.
- Both labs: the leakage trap was a column that looked like a great predictor (Inventory: `Demand Forecast` ρ 0.997; Sales: `Units Sold` ρ 0.83). Drop it.
- Both labs: rolling-7 baseline ≈ per-group mean baseline. The "is your ML earning its keep?" floor.
- **The lesson:** the right model is dataset-shaped. The *process* (CRISP-ML(Q), leakage audit, time-based split, baselines first, P80 for reorders) is identical across both.
- Two labs, two Streamlit apps, one set of patterns. Notebooks + apps on oscarponce.com.

**Visual:** carousel — slide 1: Inventory MAE table; slide 2: Sales MAE table; slide 3: side-by-side diagram of the two pipelines.
**Hashtags:** #DataScience #MachineLearning #CRISPML #Portfolio #SupplyChain
**Audience:** both — recruiters, peers, hiring managers.

---

## Production checklist (per post)

- [ ] Hook fits in 2 lines, lands before the "...more" fold
- [ ] One concrete number in the first 3 lines
- [ ] Exactly one visual, labeled and legible on mobile
- [ ] CTA present but soft; link to oscarponce.com in first comment, not body
- [ ] 3–5 hashtags, broad + niche mix
- [ ] Proofread for jargon — if a planner can't read Post 3/4, rewrite
- [ ] Post Tue–Thu, mid-morning; reply to every comment in first 2 hours

## Reuse / repurpose

- **Series A carousel:** Posts 1, 2, 4 → combined into a single 5-slide carousel for a second wave.
- **Series B carousel:** Posts 7, 8, 9 → combined into a 5-slide "lessons from a lab where the simple model won" carousel.
- **Post 3's reorder-point example** → standalone infographic.
- **Post 10** → a 5–7 min Medium / Substack article comparing the two labs.
- **Whole series** → one long-form essay on oscarponce.com with the two notebooks linked.
- **Conference talk angle:** "Two retail datasets, two opposite conclusions — how to read your dataset before you pick your model." 20-min slot.

## Repository pointers (link in first comment when relevant)

| Post group | Links |
|---|---|
| Series A (1–6) | `notebooks/Inventory_Forecasting_CRISPML.ipynb` · `app/app.py` |
| Series B (7–9) | `notebooks/Sales_Forecasting_CRISPML.ipynb` · `app/app_sales.py` |
| Post 10        | both notebooks side-by-side · the leaderboards · `model_metadata.pkl` from each lab |
