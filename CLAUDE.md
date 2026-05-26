# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app/app.py

# Open notebooks
jupyter lab notebooks/
```

## Project Structure

This project follows the **CRISP-ML(Q)** methodology. The intended layout:

```
forecasting_inventory/
├── data/
│   └── retail_store_inventory.csv   # Raw data — never modify
├── notebooks/
│   └── *_CRISPML.ipynb              # Main pipeline notebook
├── model/
│   ├── model.pkl                    # sklearn Pipeline (preprocessing + model)
│   └── model_metadata.pkl           # metrics, threshold, feature_columns, model_name
├── app/
│   └── app.py                       # Streamlit deployment app
├── examples/                        # Reference CRISP-ML notebooks for other projects
└── QUESTIONS.md                     # Agent clarification log (see below)
```

## Dataset

`data/retail_store_inventory.csv` — ~73k rows, retail demand forecasting:

| Column | Notes |
|--------|-------|
| Date, Store ID, Product ID | Identifiers / time dimension |
| Category, Region | Categorical groupings |
| Inventory Level, Units Sold, Units Ordered | Core inventory signals |
| Demand Forecast | Pre-computed forecast (can be a feature or comparison baseline) |
| Price, Discount, Competitor Pricing | Pricing features |
| Weather Condition, Holiday/Promotion, Seasonality | Contextual features |

## CRISP-ML(Q) Phases

Notebooks are organized with one section per phase:
```
## Phase 1 — Business Understanding
## Phase 2 — Data Understanding
## Phase 3 — Data Preparation
## Phase 4 — Modeling
## Phase 5 — Evaluation
## Phase 6 — Deployment
```

### Data Preparation rules
- Use `ColumnTransformer` with **different scalers per feature type**: `StandardScaler` for normal numerics, `MinMaxScaler` for bounded/ordinal, `PowerTransformer` for skewed.
- Guard ratio features against division by zero: `max(value, 1)`.
- For classification tasks, compare SMOTE variants (SMOTE, SMOTEENN, SMOTETomek) via CV score.

### Evaluation rules
- **Optimize the decision threshold** — don't default to 0.5. Tune to the business cost of false positives vs. false negatives.
- For regression: evaluate MAE, RMSE, and MAPE against the `Demand Forecast` baseline.

### Deployment rules
- Wrap the full pipeline (preprocessing → model) in `sklearn.Pipeline` and save with `joblib`.
- Save a metadata dict alongside:
  ```python
  metadata = {
      "model_name": "...",
      "feature_columns": [...],   # used to reindex input DataFrame in app.py
      # classification: accuracy, precision, recall, f1_score, optimal_threshold
      # regression: mae, rmse, mape
  }
  ```
- **Feature engineering must be replicated exactly in `app.py`** before calling `pipeline.predict()`. A mismatch between notebook features and `feature_columns` will silently produce wrong predictions or raise at inference time.

## Streamlit App Style (Ice Graphite Hybrid)

All apps share a unified visual identity. Copy the palette and CSS from `CRISP-ML-Starter.md`.

**Key conventions:**
- Fonts: **Orbitron** (headings, metrics, results) + **IBM Plex Mono** (body, labels, inputs)
- Hide Streamlit chrome (`#MainMenu`, `footer`, `header`)
- Input form: 3-column layout with `input-card` containers
- Secondary inputs in collapsed `st.expander`
- Output: 3-tab layout — Results / Risk Factors / Profile Summary
- Risk classification thresholds for binary classifiers: ≥0.6 HIGH, ≥0.4 MEDIUM, <0.4 LOW

## Agent Questions Protocol

When clarification is needed about data definitions, business logic, or modeling decisions, write questions to `QUESTIONS.md` — do **not** block or make silent assumptions.

```markdown
# Open Questions

## [Phase Name] — [date]

- [ ] Q1: ... (assuming X until answered)

## Resolved

- [x] Q: ... → **Answer:** ...
```

Proceed with the stated assumption; update `QUESTIONS.md` when the user replies.
