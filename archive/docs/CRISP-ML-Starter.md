# [Project Name] — CRISP-ML(Q) Project

## Project Objective

> **Problem:** <!-- What business problem are we solving? -->
> **ML Task:** <!-- Classification / Regression / Clustering -->
> **Target Variable:** <!-- Column name and what it represents -->
> **Success Criteria:** <!-- F1 ≥ X%, Recall ≥ X%, etc. -->
> **Business Value:** <!-- What decision or action does this model enable? -->

---

## Project Structure

```
project_name/
├── data/
│   └── dataset.csv               # Raw source data (never modified)
├── notebooks/
│   └── ProjectName_CRISPML.ipynb # Full CRISP-ML(Q) pipeline notebook
├── model/
│   ├── model.pkl                 # Serialized trained pipeline (preprocessing + model)
│   └── model_metadata.pkl        # Metrics + threshold + feature_columns + model_name
├── app/
│   └── app.py                    # Streamlit deployment app
├── requirements.txt
└── README.md
```

---

## CRISP-ML(Q) Phases

### 1. Business Understanding
- Define the ML task (classification / regression / clustering)
- Establish success criteria (e.g. F1 ≥ X%, ROC-AUC ≥ X%)
- Identify target variable and business impact

### 2. Data Understanding
- Load and inspect dataset (shape, dtypes, nulls, duplicates)
- Descriptive statistics and class distribution
- Correlation analysis and feature relevance
- Visualize distributions and outliers

### 3. Data Preparation
- Handle missing values and outliers
- Engineer new features from domain knowledge
- **Use different scalers for different feature types** via `ColumnTransformer`:
  - `StandardScaler` for normally distributed numerics
  - `MinMaxScaler` for bounded/ordinal features
  - `PowerTransformer` for skewed distributions
- **Compare SMOTE variants** for class imbalance (SMOTE, SMOTEENN, SMOTETomek) — pick based on CV score
- Train/test split with stratification
- **Guard ratio features against division by zero** with `max(value, 1)` before computing

### 4. Modeling
- Baseline model (LogisticRegression or DummyClassifier)
- Train candidate models (e.g. Random Forest, XGBoost, LightGBM, etc.)
- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
- Select best model based on success criteria metrics

### 5. Evaluation
- Evaluate on held-out test set: accuracy, precision, recall, F1, ROC-AUC
- Confusion matrix and classification report
- Feature importance / permutation importance
- **Optimize decision threshold** — don't default to 0.5; tune it to favor recall or precision depending on the business cost of each error type
- Cross-validation for generalization check

### 6. Deployment
- Wrap full pipeline (preprocessing → model) in `sklearn.Pipeline`
- Save with `joblib` / `pickle` → `model/model.pkl`
- Save metadata dict → `model/model_metadata.pkl`
  ```python
  metadata = {
      "model_name": "...",
      "accuracy": ..., "precision": ..., "recall": ..., "f1_score": ...,
      "optimal_threshold": ...,
      "feature_columns": [...]   # used to reindex input DataFrame in the app
  }
  ```
- Build Streamlit app that loads both artifacts and replicates feature engineering

---

## Critical Rule: Feature Engineering Sync

Any feature engineering done in the notebook **must be replicated exactly** in `app.py` before calling `pipeline.predict()`. The `feature_columns` list from `model_metadata.pkl` is used to reindex the input DataFrame — a column mismatch will raise an error at inference time.

---

## Notebook Structure

Organize the notebook with one clearly labeled section per CRISP-ML phase using markdown headers. This makes it readable as a document, not just a script.

```
## Phase 1 — Business Understanding
## Phase 2 — Data Understanding
## Phase 3 — Data Preparation
## Phase 4 — Modeling
## Phase 5 — Evaluation
## Phase 6 — Deployment
```

---

## Streamlit App Style (Ice Graphite Hybrid)

All apps follow the same visual identity. Copy the brand palette and CSS block at the top of each new `app.py`.

### Brand Palette

```python
ICE_SILVER    = "#E6E8EB"
GRAPHITE      = "#2A3038"
ESPRESSO_GOLD = "#C9A86A"   # primary accent
GRAPHITE_DEEP = "#240338"   # headings
SLATE         = "#424A53"
PEBBLE        = "#5E757D"
MIST          = "#B0B4B8"
SILVER        = "#D5D6DB"
PLATINUM      = "#EBECEF"
SUCCESS       = "#43936C"
WARNING       = "#F2AE4A"
DANGER        = "#D96B5F"
INFO          = "#4A67B0"
```

### Fonts

- **Orbitron** — headings, section titles, metric values, prediction results (uppercase)
- **IBM Plex Mono** — body, labels, inputs, supporting text

Import via Google Fonts in the CSS block:
```css
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=IBM+Plex+Mono:wght@300;400;600&display=swap');
```

### Page Config

```python
st.set_page_config(
    page_title="App Name | CRISP-ML",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="collapsed"
)
```

### Layout Conventions

- **Hide Streamlit chrome**: always hide `#MainMenu`, `footer`, `header`
- **Background**: `linear-gradient(135deg, PLATINUM 0%, ICE_SILVER 50%, SILVER 100%)`
- **Max content width**: `1400px`
- **Input form**: 3-column layout with `input-card` containers — Demographics / Job Info / Experience
- **Extra parameters**: collapsed `st.expander` for secondary inputs
- **Output**: 3-tab layout — Results / Risk Factors / Profile Summary
- **Dividers**: `st.markdown("---")` renders as a gold gradient `hr`

### CSS Component Classes

| Class | Use |
|---|---|
| `.main-header` | Page title — Orbitron 3.5rem, uppercase |
| `.sub-header` | Subtitle — IBM Plex Mono, uppercase |
| `.section-header` | Section label — Orbitron 1.8rem |
| `.metric-card` | KPI card with hover lift + gold border |
| `.prediction-box` | Result box — variants: `.success` `.warning` `.danger` |
| `.input-card` | White card wrapping form columns |
| `.input-card-header` | Card title — Orbitron, gold bottom border |
| `.info-box` | Text callout with left gold border |
| `.feature-bar` / `.feature-fill` | Custom progress bars in gold |

### Risk Classification Pattern (for binary classifiers)

```python
leave_probability = pipeline.predict_proba(df)[0][1]
optimal_threshold = metadata.get('optimal_threshold', 0.5)

if leave_probability >= 0.6:
    risk_level, color, box_class = "HIGH RISK",   DANGER,  "danger"
elif leave_probability >= 0.4:
    risk_level, color, box_class = "MEDIUM RISK", WARNING, "warning"
else:
    risk_level, color, box_class = "LOW RISK",    SUCCESS, "success"
```

---

## .gitignore Recommendations

```
# Jupyter
.ipynb_checkpoints/
catboost_info/

# macOS
.DS_Store

# Model artifacts (add to git if small; ignore if > 50MB)
# model/*.pkl
```

---

## Agent Questions Protocol

When the agent working on this project needs clarification — about the data, business logic, feature definitions, or modeling decisions — it must **write the questions to `QUESTIONS.md`** instead of making assumptions or blocking.

### QUESTIONS.md format

```markdown
# Open Questions

## [Phase Name] — [date]

- [ ] Q1: ...
- [ ] Q2: ...

## Resolved

- [x] Q: ... → **Answer:** ...
```

### Rules
- One file, append new questions under the current phase heading
- Mark resolved questions with `[x]` and inline the answer
- The agent proceeds with a clearly stated assumption until the user replies
- State the assumption explicitly next to the question: `(assuming X until answered)`

---

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app/app.py

# Open notebook
jupyter lab notebooks/
```
