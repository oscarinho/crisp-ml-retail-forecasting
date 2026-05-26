# Test Cases — Inventory & Sales Apps

QA scenarios with **exact expected predictions** generated from the production `.pkl` artifacts.
Tolerance: **±2 units** (inventory) · **±1 unit** (sales). Anything outside that range means the
deployed model drifted from the notebook output — investigate.

Last verified: **2026-05-26**

---

## 🧭 What you're testing

This repo contains **two separate Streamlit apps**, each backed by its own model trained on its
own dataset. They look similar on purpose — same editorial design system, same demo-simulator
pattern — but they answer different business questions, ship different model artifacts, and
**read from different CSVs**. Below is the mental model you need before running a single test.

### App 1 — Inventory App (`app/app.py`)

- **Business question:** *"Given my current stock and context, how many units will I sell next,
  and am I covered?"*
- **Dataset:** `data/retail_store_inventory.csv` — 73,101 rows · 15 cols · 2021-12-31 → 2023-12-31.
  Synthetic. Source: [Kaggle / Anirudh Singh Chauhan](https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset) · CC0.
- **Target variable:** `Units Sold` (note: bounded by `Inventory Level` → demand is *censored*
  when stockouts happen).
- **Distinctive columns:** `Demand Forecast` (a pre-computed signal — ρ ≈ 0.997 with `Units Sold`,
  treated as leakage and dropped), `Holiday/Promotion` (single combined flag).
- **Model artifact:** `model/model_contextual.pkl` — a single sklearn `Pipeline` (preprocessing +
  regressor). One model serves every category.
- **Companion artifact:** `model/model_q80.pkl` — an 80th-percentile quantile model used to give
  a P80 reorder-up-to recommendation (newsvendor-style buffer).
- **Stock status rule:** `coverage = Inventory ÷ Predicted demand`
  - `< 0.5×` → **CRITICAL · STOCKOUT RISK** (red)
  - `0.5×` – `1.2×` → **LOW STOCK** (amber)
  - `≥ 1.2×` → **WELL STOCKED** (green)
- **Reorder rule:** 7-day buffer — `reorder = ceil(max(0, 7·demand − inventory) / demand) · demand`

### App 2 — Sales App (`app/app_sales.py`)

- **Business question:** *"What's the expected demand for this category under these
  promotion / epidemic / pricing conditions?"*
- **Dataset:** `data/sales_data.csv` — 76,001 rows · 16 cols · 2022-01-01 → 2024-01-29.
  Synthetic, sibling of the inventory dataset with mislabelled IDs fixed and an added Epidemic
  feature. Source: [Kaggle / WAVELET](https://www.kaggle.com/datasets/atomicd/retail-store-inventory-and-demand-forecasting) · Apache-2.0
  (also mirrored on Kaggle by Ramin Huseyn under CC0 — byte-identical).
- **Target variable:** `Demand` (uncensored — `Units Sold ≤ Demand` in this dataset, so it
  captures lost demand from stockouts; ρ(`Units Sold`, `Demand`) ≈ 0.83).
- **Distinctive columns:** `Promotion` and `Epidemic` are separate binary flags (vs the inventory
  dataset's combined `Holiday/Promotion`).
- **Model artifact:** `model/sales/model_per_category.pkl` — a **dict of 5 LightGBM models**,
  one per `Category` (Toys / Groceries / Electronics / Clothing / Furniture). The app routes by
  Category at inference.
- **Fallback artifact:** `model/sales/model_contextual.pkl` — single model used only if a
  category isn't in the dict. **Every test case in this doc should hit `per-category`** — if any
  shows `fallback (app-aligned)`, that's a regression.
- **Extra features unique to this app:** `Promotion`, `Epidemic`, and Fourier seasonal terms
  (`sin_year`, `cos_year`).

### Why two labs from two sibling datasets?

The two datasets are deliberately complementary — same schema family (Store × Product × Day,
5 stores × 20 products, same category & region taxonomies), but the *inventory* one is
**memoryless** (within-group autocorr ≈ 0 — Stage 1 vs Stage 2 lag features barely help,
MAE ≈ 69 is the noise floor), while the *sales* one is **autocorrelated** (within-group
ρ ≈ 0.35 — modest but real, lets per-category LightGBMs hit MAE ≈ 19.5). Running both labs
side-by-side is the portfolio's main story: *same methodology, two opposite outcomes,
explained by the data, not the model*.

---

## 📦 Inventory App — `app/app.py`

Open the **Demo Simulator** tab. Fill every field exactly as listed in the table; the prediction
should match within ±2 units. The "Stock Status" check below the prediction box is what you're
ultimately verifying — the threshold logic must match the rule stated above.

### Case 1 — Toys / Holiday / December weekend

**What this scenario represents:** Peak Christmas weekend for a toy store with a promotion
running. This is the high-demand corner of the feature space — December + Holiday flag + Weekend
should all push demand up. We want to confirm the model **does** lift demand under this combo
(it should not flatline) and that a `200`-unit inventory still resolves to WELL STOCKED.

| Field | Value |
|---|---|
| Store | `S001` |
| Category | `Toys` |
| Region | `North` |
| Month | `December` (12) |
| Weekday | `Sat` |
| Season | `Winter` |
| Weather | `Snowy` |
| Holiday/Promotion | ☑ ON |
| Price | `$25.00` |
| Discount | `15%` |
| Competitor Price | `$28.00` |
| Current Inventory | `200` |
| **→ Predicted demand** | **`95`** units (raw 94.75) |
| → Coverage | `2.1×` |
| → Stock Status | **WELL STOCKED** (green) |
| → Reorder qty | `475` units |

### Case 2 — Groceries / Summer baseline / no promo

**What this scenario represents:** The "everyday" grocery case. Mid-week, no promo, sunny day.
This is the highest baseline-demand category in the dataset — expect a prediction much larger
than Toys/Electronics for similar context. Use this to confirm the model has learned
category-level baselines.

| Field | Value |
|---|---|
| Store | `S002` |
| Category | `Groceries` |
| Region | `South` |
| Month | `June` (6) |
| Weekday | `Wed` |
| Season | `Summer` |
| Weather | `Sunny` |
| Holiday/Promotion | ☐ OFF |
| Price | `$12.00` |
| Discount | `0%` |
| Competitor Price | `$13.50` |
| Current Inventory | `400` |
| **→ Predicted demand** | **`199`** units (raw 199.29) |
| → Coverage | `2.0×` |
| → Stock Status | **WELL STOCKED** |
| → Reorder qty | `995` units |

### Case 3 — Electronics / Spring / heavy discount

**What this scenario represents:** Mid-season electronics with a healthy discount and a price
below competitor. The interesting check here is the **price-elasticity behaviour** — the
discount and the price-vs-competitor ratio should both nudge demand up. With only `60` units
of inventory we still come out WELL STOCKED because electronics is naturally low-volume.

| Field | Value |
|---|---|
| Store | `S003` |
| Category | `Electronics` |
| Region | `East` |
| Month | `April` (4) |
| Weekday | `Thu` |
| Season | `Spring` |
| Weather | `Cloudy` |
| Holiday/Promotion | ☐ OFF |
| Price | `$85.00` |
| Discount | `20%` |
| Competitor Price | `$95.00` |
| Current Inventory | `60` |
| **→ Predicted demand** | **`32`** units (raw 32.17) |
| → Coverage | `1.9×` |
| → Stock Status | **WELL STOCKED** |
| → Reorder qty | `192` units |

### Case 4 — Clothing / Autumn / promo weekend

**What this scenario represents:** Apparel on a rainy promo weekend. Clothing sits in the
mid-volume range. This case is here to verify the model handles a different category × season
combo cleanly — not a corner case, just a representative "normal" prediction for clothing.

| Field | Value |
|---|---|
| Store | `S004` |
| Category | `Clothing` |
| Region | `West` |
| Month | `November` (11) |
| Weekday | `Sun` |
| Season | `Autumn` |
| Weather | `Rainy` |
| Holiday/Promotion | ☑ ON |
| Price | `$45.00` |
| Discount | `10%` |
| Competitor Price | `$50.00` |
| Current Inventory | `50` |
| **→ Predicted demand** | **`25`** units (raw 24.51) |
| → Coverage | `2.0×` |
| → Stock Status | **WELL STOCKED** |
| → Reorder qty | `125` units |

### Case 5 — Furniture / Autumn / baseline

**What this scenario represents:** The slow-mover. Furniture is the highest-priced category and
typically the slowest turnover. Despite a low-ish unit prediction (~167), this is normal — what
you're verifying is that the per-unit reorder math still works at this volume and that status
resolves to WELL STOCKED with a 350-unit inventory.

| Field | Value |
|---|---|
| Store | `S005` |
| Category | `Furniture` |
| Region | `North` |
| Month | `September` (9) |
| Weekday | `Tue` |
| Season | `Autumn` |
| Weather | `Sunny` |
| Holiday/Promotion | ☐ OFF |
| Price | `$70.00` |
| Discount | `5%` |
| Competitor Price | `$72.00` |
| Current Inventory | `350` |
| **→ Predicted demand** | **`167`** units (raw 167.45) |
| → Coverage | `2.1×` |
| → Stock Status | **WELL STOCKED** |
| → Reorder qty | `835` units |

---

## 🛒 Sales App — `app/app_sales.py`

Open the **Demo Simulator** tab. **Every case below must show the routing badge `per-category`**
on the prediction card — if any case shows `fallback (app-aligned)`, the per-category dict isn't
loading or a category is missing from it. That's a regression worth investigating.

The two features this app exposes that the inventory app doesn't are **Promotion** and
**Epidemic** toggles. Cases 1 and 3 specifically stress those.

### Case 1 — Clothing / Promo + Epidemic / Winter

**What this scenario represents:** Worst-case combinatorial stressor — promo AND epidemic both
ON. The clothing category showed a clear interaction effect in the EDA (notebook §4.9), so a
prediction here should be **noticeably higher than baseline**. This is also a routing-test: the
Clothing model must be present and active.

| Field | Value |
|---|---|
| Store | `S001` |
| Category | `Clothing` |
| Region | `North` |
| Month | `January` (1) |
| Weekday | `Sat` |
| Season | `Winter` |
| Weather | `Snowy` |
| Promotion | ☑ ON |
| Epidemic | ☑ ON |
| Price | `$60.00` |
| Discount | `20%` |
| Competitor Price | `$70.00` |
| Current Inventory | `250` |
| **→ Predicted demand** | **`149`** units (raw 149.42) |
| → Routing | `per-category` |
| → Coverage | `1.7×` |
| → Stock Status | **WELL STOCKED** |
| → P80 stock-to | (see app) |
| → Reorder qty | `894` units |

### Case 2 — Groceries / Summer / no promo

**What this scenario represents:** Pure baseline for the highest-volume category. No promo, no
epidemic. This case anchors what "normal" looks like for Groceries — useful as a comparison
point against Case 3 (Electronics with promo) and against the same Groceries case in the
inventory app (App 1 Case 2 = 199, App 2 Case 2 = 100 — different targets, expected).

| Field | Value |
|---|---|
| Store | `S002` |
| Category | `Groceries` |
| Region | `South` |
| Month | `July` (7) |
| Weekday | `Wed` |
| Season | `Summer` |
| Weather | `Sunny` |
| Promotion | ☐ OFF |
| Epidemic | ☐ OFF |
| Price | `$15.00` |
| Discount | `0%` |
| Competitor Price | `$16.00` |
| Current Inventory | `500` |
| **→ Predicted demand** | **`100`** units (raw 99.52) |
| → Routing | `per-category` |
| → Coverage | `5.0×` |
| → Stock Status | **WELL STOCKED** |
| → Reorder qty | `200` units |

### Case 3 — Electronics / Black-Friday-style

**What this scenario represents:** A late-Q4 electronics promotion. This is the scenario where
demand **outruns inventory** — `180` on hand but `222` predicted means coverage drops below
`1.2×` and the status flips to LOW STOCK. This case is the one to use when demoing the
"reorder advisory" feature, because it actually triggers a non-trivial reorder.

| Field | Value |
|---|---|
| Store | `S003` |
| Category | `Electronics` |
| Region | `East` |
| Month | `November` (11) |
| Weekday | `Thu` |
| Season | `Autumn` |
| Weather | `Cloudy` |
| Promotion | ☑ ON |
| Epidemic | ☐ OFF |
| Price | `$120.00` |
| Discount | `25%` |
| Competitor Price | `$140.00` |
| Current Inventory | `180` |
| **→ Predicted demand** | **`222`** units (raw 221.73) |
| → Routing | `per-category` |
| → Coverage | `0.8×` |
| → Stock Status | **LOW STOCK** (amber) |
| → Reorder qty | `1,554` units |

### Case 4 — Furniture / Spring weekday / no promo

**What this scenario represents:** Slow-mover baseline. Rainy Tuesday, no promo, Spring. Use this
to confirm that **Furniture's per-category model exists and doesn't accidentally fall back** —
furniture has the smallest training partition, so it's the most likely category to be missing
or under-trained.

| Field | Value |
|---|---|
| Store | `S004` |
| Category | `Furniture` |
| Region | `West` |
| Month | `April` (4) |
| Weekday | `Tue` |
| Season | `Spring` |
| Weather | `Rainy` |
| Promotion | ☐ OFF |
| Epidemic | ☐ OFF |
| Price | `$95.00` |
| Discount | `5%` |
| Competitor Price | `$100.00` |
| Current Inventory | `200` |
| **→ Predicted demand** | **`56`** units (raw 56.24) |
| → Routing | `per-category` |
| → Coverage | `3.6×` |
| → Stock Status | **WELL STOCKED** |
| → Reorder qty | `224` units |

### Case 5 — Toys / December weekend / promo

**What this scenario represents:** The Sales-app twin of Inventory Case 1. Same category, same
month, same weekend, similar pricing — but now we have the explicit `Promotion` flag (instead of
the inventory app's combined `Holiday/Promotion`) and we expect demand even higher (`143` vs the
inventory app's `95`). The difference is because **the two apps predict different targets**:
inventory predicts `Units Sold`, sales predicts `Demand Forecast` (which is theoretical demand,
not constrained by stock).

| Field | Value |
|---|---|
| Store | `S005` |
| Category | `Toys` |
| Region | `North` |
| Month | `December` (12) |
| Weekday | `Sat` |
| Season | `Winter` |
| Weather | `Snowy` |
| Promotion | ☑ ON |
| Epidemic | ☐ OFF |
| Price | `$35.00` |
| Discount | `15%` |
| Competitor Price | `$38.00` |
| Current Inventory | `80` |
| **→ Predicted demand** | **`143`** units (raw 143.36) |
| → Routing | `per-category` |
| → Coverage | `0.6×` |
| → Stock Status | **LOW STOCK** |
| → Reorder qty | `1,001` units |

---

## 🔥 Stress tests — same prediction, different inventory

The 10 cases above mostly resolve to WELL STOCKED because they were designed to verify the
model output. To verify the **status logic** (CRITICAL · LOW · WELL thresholds), set up one of
the cases below and then **move only the Inventory slider** — the prediction stays constant,
and you watch the badge color flip.

The thresholds you're verifying:
- `coverage < 0.5×` → red CRITICAL
- `0.5× ≤ coverage < 1.2×` → amber LOW
- `coverage ≥ 1.2×` → green WELL

### Inventory app — Case 2 (predicted = 199)

| Inventory | Coverage | Expected Status |
|---|---|---|
| `80` | 0.40× | **CRITICAL — STOCKOUT RISK** (red) |
| `200` | 1.00× | **LOW STOCK** (amber) |
| `500` | 2.51× | **WELL STOCKED** (green) |

### Sales app — Case 1 (predicted = 149)

| Inventory | Coverage | Expected Status |
|---|---|---|
| `50` | 0.34× | **CRITICAL — STOCKOUT RISK** |
| `100` | 0.67× | **LOW STOCK** |
| `250` | 1.68× | **WELL STOCKED** |

### Sales app — Case 5 (predicted = 143)

| Inventory | Coverage | Expected Status |
|---|---|---|
| `60` | 0.42× | **CRITICAL** |
| `150` | 1.05× | **LOW STOCK** |
| `300` | 2.10× | **WELL STOCKED** |

---

## 📊 Chart behavior checks

These don't have a single expected number — verify **trends** instead. Each chart is checking a
specific business behavior should hold; if it doesn't, the model or the visualization wiring is
off.

### Inventory app — "Fig. 01 — Demand elasticity" (Demand vs Price sweep)
**What it shows:** The app sweeps Price ±20% around the value you entered, holds every other
feature constant, and plots predicted demand against price. This is the model's implicit
price-elasticity curve.

- The curve should slope **downward** from left to right (higher price → lower demand).
- The vertical dotted line marks the **current price** you entered.
- ⚠ If the curve goes **up** as price goes up → unexpected; capture the scenario and flag.
  That would mean the model learned a positive price coefficient, which usually means a
  collinearity issue with discount or competitor pricing.

### Sales app — "Promo × Epidemic Scenarios" (4-bar chart in simulator)
**What it shows:** The app re-predicts demand for the four combinations of the two binary flags
(Base / Promo only / Epi only / Promo+Epi) holding everything else constant. This is the
model's implicit interaction effect.

- 4 bars: Base · Promo only · Epidemic only · Promo + Epi.
- Expected pattern for **Clothing / Electronics / Toys**: Promo > Base; Epi adds further lift.
- **Groceries may surprise** — see notebook §4.9. Don't treat Groceries deviations as bugs.

### Inventory dashboard — "Fig. 04 — Store-Level Comparison"
**What it shows:** Per-store averages of demand vs inventory across the filtered window. Used
in the EDA tab to spot stores that systematically under- or over-stock.

- Two grouped bars per store: **gold = avg demand**, **indigo = avg inventory**.
- Legend should be **below** the bars, not overlapping the title.
- Should render with 5 store groups (S001…S005) when sidebar Store filter = `All`.

### Sales dashboard — "Fig. 03 — Stockouts vs Demand"
**What it shows:** Stacked bar per store of (met demand) vs (lost demand = stockouts).
Groceries should clearly dominate the stockout segment.

- Stacked bars per store: **gold = met demand**, **coral = lost (stockout)**.
- Coral segments should be **visibly larger for Groceries** stores (highest stockout regime).
- Legend should be **below**, not overlapping "Per-Store: Daily Demand Met vs Lost" title.

### Sales app — "Demand Trends" (Fig. 01)
**What it shows:** Weekly-resampled mean demand by category over the entire training window.
A simple visual smell-test for seasonality and promo / epidemic shocks.

- Weekly-resampled line by category for the full window.
- Lines should show **visible deviations during promotion / epidemic periods**.

---

## 🎨 UI smoke checks (post-redesign 2026-05-26)

After the editorial-terminal redesign (commit `3f347cb`), verify on first load that the
"newspaper" aesthetic is intact. If any of these look generic-streamlit, the CSS injection
didn't run and you're seeing the default theme.

| Element | Expected |
|---|---|
| Issue strip | Top-of-page newspaper bar with pulsing green dot ("LIVE · LAB 0X · …") |
| Hero title | Big Fraunces serif, word in italic gold (e.g. "Retail Demand *Intelligence*") |
| KPI cards | White cards with **gold L-marks in top-left & bottom-right corners** |
| Tabs | Single underline below active tab (no pill bg) — gold under active, dark text |
| Prediction box | 4px top status bar (green/amber/red), "CRISP · PRED v1" or "P80 · ROUTED v1" stamp top-right |
| Charts | Each in a white framed card with italic Fraunces caption below ("**Fig. N** — …") |
| Sidebar | Charcoal gradient, gold spine on right edge, italic "Demand Intel." / "Sales Intel." brand |
| Paper grain | Subtle noise texture on the background (visible if you zoom in) |
| Footer | "Colophon · …  oscarponce.com · MMXXVI" with hairline rule |

---

## 🐛 Known caveats

1. **Inventory all WELL STOCKED**: every test case above resolves to `WELL STOCKED`. To exercise
   LOW / CRITICAL paths, use the stress-tests section. This is because we deliberately set
   generous inventory levels in the demo cases — the goal of those cases is to verify the
   **prediction number**, not the status threshold.
2. **`use_container_width` deprecation warnings** in the Streamlit console — purely cosmetic;
   Streamlit retires the keyword after 2025-12-31. Safe to ignore until then.
3. **Sales Q80 / P80 stock-to** uses a lag-feature fallback proxy at inference (no history at
   demo time). Stock-to numbers are directional, not absolute — useful for relative comparison
   between scenarios but don't read them as ground-truth quantiles.
4. **Two apps, two targets** — when comparing predictions between App 1 and App 2 for the same
   inputs, don't expect equality. App 1 predicts `Units Sold` (constrained by stock). App 2
   predicts `Demand Forecast` (theoretical demand, unconstrained).

---

## 🔁 Regenerating

The generator script lives at [`scripts/gen_test_cases.py`](scripts/gen_test_cases.py) — it
loads the actual pkl artifacts and prints expected predictions for every case above.

```bash
# from the repo root
python scripts/gen_test_cases.py

# or with the project's conda env explicitly
/Users/oscarponce/miniconda3/envs/ml-exp/bin/python scripts/gen_test_cases.py
```

Re-run this after **any** change to the pkl files in `model/` and copy the new predictions into
the tables above. If predictions shift by more than the tolerance window (±2 inventory, ±1
sales), don't just update the doc — first confirm the retrain was intentional and that the
notebook evaluation metrics still pass.
