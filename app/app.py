import os
import joblib
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ─── Brand Palette ────────────────────────────────────────────────────────────
ICE_SILVER    = "#E6E8EB"
GRAPHITE      = "#2A3038"
ESPRESSO_GOLD = "#C9A86A"
GRAPHITE_DEEP = "#240338"
SLATE         = "#424A53"
PEBBLE        = "#5E757D"
MIST          = "#B0B4B8"
SILVER        = "#D5D6DB"
PLATINUM      = "#EBECEF"
SUCCESS       = "#43936C"
WARNING_COL   = "#F2AE4A"
DANGER        = "#D96B5F"
INFO          = "#4A67B0"

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Demand Intelligence | CRISP-ML",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS — Editorial Terminal aesthetic ───────────────────────────────────────
INK     = "#1A1D23"
GOLD_DK = "#8B7340"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=IBM+Plex+Mono:wght@300;400;500;600&family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;1,9..144,400;1,9..144,500&display=swap');

:root {{
    --ink: {INK};
    --ink-soft: rgba(26, 29, 35, 0.55);
    --ink-faint: rgba(26, 29, 35, 0.32);
    --rule: rgba(26, 29, 35, 0.12);
    --rule-strong: rgba(26, 29, 35, 0.35);
    --gold-soft: rgba(201, 168, 106, 0.09);
    --gold-mid: rgba(201, 168, 106, 0.32);
}}

#MainMenu, footer, header {{ visibility: hidden; }}

html, body, [data-testid="stAppViewContainer"] {{
    background:
      radial-gradient(ellipse 80% 50% at top left, rgba(201, 168, 106, 0.07) 0%, transparent 55%),
      radial-gradient(ellipse 80% 50% at bottom right, rgba(36, 3, 56, 0.05) 0%, transparent 55%),
      linear-gradient(180deg, #FAFAF7 0%, #F3F2EE 100%);
    min-height: 100vh;
    font-family: 'IBM Plex Mono', monospace;
    color: var(--ink);
}}

/* Paper grain overlay */
[data-testid="stAppViewContainer"]::before {{
    content: '';
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='180' height='180'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='2' seed='3'/><feColorMatrix values='0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0.05 0'/></filter><rect width='100%' height='100%' filter='url(%23n)'/></svg>");
    mix-blend-mode: multiply;
}}

/* Sidebar — gilded charcoal */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #20242C 0%, #161A21 100%) !important;
    border-right: none;
    position: relative;
}}
[data-testid="stSidebar"]::after {{
    content: '';
    position: absolute;
    top: 0; right: 0; bottom: 0;
    width: 2px;
    background: linear-gradient(180deg, transparent 0%, {ESPRESSO_GOLD} 22%, {ESPRESSO_GOLD} 78%, transparent 100%);
}}
[data-testid="stSidebar"] * {{ color: #E6E8EB !important; }}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label {{
    color: rgba(176, 180, 184, 0.72) !important;
    font-family: 'Fraunces', Georgia, serif;
    font-style: italic;
    font-size: 0.92rem;
    letter-spacing: 0;
    text-transform: none;
    font-weight: 400;
}}

/* Issue strip — newspaper masthead */
.issue-strip {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    color: var(--ink-soft);
    letter-spacing: 0.3em;
    text-transform: uppercase;
    border-top: 1px solid var(--ink);
    border-bottom: 1px solid var(--rule);
    padding: 0.55rem 0;
    margin: 0 0 1.3rem;
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 1rem;
}}
.issue-strip .dot {{
    display: inline-block;
    width: 6px; height: 6px;
    background: {SUCCESS};
    border-radius: 50%;
    margin-right: 0.5rem;
    vertical-align: middle;
    box-shadow: 0 0 8px rgba(67, 147, 108, 0.65);
    animation: pulse 2.2s ease-in-out infinite;
}}
@keyframes pulse {{
    0%, 100% {{ opacity: 1; transform: scale(1); }}
    50%      {{ opacity: 0.4; transform: scale(0.9); }}
}}

/* Hero header — editorial display */
.main-header {{
    font-family: 'Fraunces', Georgia, serif;
    font-variation-settings: "opsz" 144;
    font-size: 3.2rem;
    font-weight: 600;
    color: var(--ink);
    letter-spacing: -0.025em;
    line-height: 1;
    margin: 0.1rem 0;
}}
.main-header em {{
    font-style: italic;
    font-weight: 400;
    color: {GOLD_DK};
}}

.sub-header {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: var(--ink-soft);
    letter-spacing: 0.24em;
    text-transform: uppercase;
    margin: 0.55rem 0 0;
    padding-bottom: 1.3rem;
    border-bottom: 2px solid var(--ink);
}}

/* Section header — Fraunces italic flanked by rules */
.section-header {{
    font-family: 'Fraunces', Georgia, serif;
    font-style: italic;
    font-size: 1.3rem;
    font-weight: 500;
    color: var(--ink);
    letter-spacing: -0.005em;
    margin: 1.8rem 0 1rem;
    display: flex;
    align-items: baseline;
    gap: 0.75rem;
    line-height: 1.2;
}}
.section-header::before {{
    content: '';
    flex: 0 0 24px;
    height: 1px;
    background: {ESPRESSO_GOLD};
    align-self: center;
}}
.section-header::after {{
    content: '';
    flex: 1 1 auto;
    height: 1px;
    background: var(--rule);
    align-self: center;
}}

/* Metric card — luxury detailing with bevel corners */
.metric-card {{
    background: #FFFFFF;
    border: 1px solid var(--rule);
    border-radius: 0;
    padding: 1.25rem 1.35rem 1rem;
    position: relative;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    box-shadow: 0 1px 0 rgba(26,29,35,0.04);
}}
.metric-card::before {{
    content: '';
    position: absolute;
    top: -1px; left: -1px;
    width: 14px; height: 14px;
    border-top: 1.5px solid {ESPRESSO_GOLD};
    border-left: 1.5px solid {ESPRESSO_GOLD};
}}
.metric-card::after {{
    content: '';
    position: absolute;
    bottom: -1px; right: -1px;
    width: 14px; height: 14px;
    border-bottom: 1.5px solid {ESPRESSO_GOLD};
    border-right: 1.5px solid {ESPRESSO_GOLD};
}}
.metric-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 14px 30px rgba(26, 29, 35, 0.08);
}}
.metric-label {{
    font-family: 'Fraunces', Georgia, serif;
    font-style: italic;
    font-size: 0.92rem;
    color: var(--ink-soft);
    letter-spacing: 0;
    text-transform: none;
    line-height: 1.2;
}}
.metric-value {{
    font-family: 'Orbitron', monospace;
    font-size: 2.1rem;
    font-weight: 700;
    color: var(--ink);
    margin: 0.4rem 0 0.15rem;
    font-feature-settings: 'tnum';
    letter-spacing: -0.02em;
    line-height: 1;
}}
.metric-delta {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    color: var(--ink-faint);
    letter-spacing: 0.16em;
    text-transform: uppercase;
}}

/* Prediction box — classified bulletin */
.prediction-box {{
    background: #FFFFFF;
    border: 1px solid var(--rule);
    border-radius: 0;
    padding: 1.7rem 1.9rem 1.4rem;
    margin: 1rem 0;
    position: relative;
    box-shadow: 0 18px 50px rgba(26, 29, 35, 0.07);
}}
.prediction-box::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 4px;
}}
.prediction-box.success::before {{ background: {SUCCESS}; }}
.prediction-box.warning::before {{ background: {WARNING_COL}; }}
.prediction-box.danger::before  {{ background: {DANGER}; }}

.pred-stamp {{
    position: absolute;
    top: 1rem; right: 1.2rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.55rem;
    letter-spacing: 0.32em;
    color: var(--ink-soft);
    background: var(--gold-soft);
    padding: 0.25rem 0.6rem;
    border: 1px solid var(--gold-mid);
    text-transform: uppercase;
}}

.prediction-label {{
    font-family: 'Fraunces', Georgia, serif;
    font-style: italic;
    font-size: 1rem;
    color: var(--ink-soft);
    letter-spacing: 0;
    text-transform: none;
    opacity: 1;
    margin-bottom: 0.2rem;
}}
.prediction-value {{
    font-family: 'Orbitron', monospace;
    font-size: 3.6rem;
    font-weight: 900;
    line-height: 1;
    margin: 0.2rem 0;
    font-feature-settings: 'tnum';
    letter-spacing: -0.03em;
}}
.prediction-advisory {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    margin-top: 0.85rem;
    font-weight: 500;
    letter-spacing: 0.03em;
    padding: 0.55rem 0.85rem;
    background: rgba(26,29,35,0.035);
    border-left: 2px solid {ESPRESSO_GOLD};
}}

/* Input card */
.input-card {{
    background: #FFFFFF;
    border: 1px solid var(--rule);
    border-radius: 0;
    padding: 1.25rem 1.5rem 1rem;
    margin-bottom: 0.9rem;
    box-shadow: 0 1px 0 rgba(26,29,35,0.04);
    position: relative;
}}
.input-card-header {{
    font-family: 'Fraunces', Georgia, serif;
    font-style: italic;
    font-size: 1.05rem;
    font-weight: 500;
    color: var(--ink);
    text-transform: none;
    letter-spacing: 0;
    border-bottom: 1px solid var(--rule);
    padding-bottom: 0.55rem;
    margin-bottom: 0.95rem;
    display: flex;
    align-items: baseline;
    gap: 0.5rem;
}}
.input-card-header::before {{
    content: '§';
    color: {ESPRESSO_GOLD};
    font-family: 'IBM Plex Mono', monospace;
    font-style: normal;
    font-weight: 600;
    font-size: 0.95rem;
}}

/* Info box — gold-tinted citation */
.info-box {{
    background: var(--gold-soft);
    border-left: 2px solid {ESPRESSO_GOLD};
    border-top: 1px solid var(--rule);
    border-right: 1px solid var(--rule);
    border-bottom: 1px solid var(--rule);
    border-radius: 0;
    padding: 0.95rem 1.1rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: var(--ink);
    margin: 0.5rem 0;
    line-height: 1.65;
}}

.stockout-row-high   {{ background: rgba(217, 107, 95, 0.10) !important; }}
.stockout-row-medium {{ background: rgba(242, 174, 74, 0.10) !important; }}

/* HR — editorial rule */
hr {{
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--ink) 25%, var(--ink) 75%, transparent);
    opacity: 0.25;
    margin: 1.5rem 0;
}}

/* Tabs — newspaper section nav */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0;
    background: transparent;
    border-radius: 0;
    padding: 0;
    border-bottom: 2px solid var(--ink);
    margin-bottom: 1.3rem;
}}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {{
    background: transparent !important;
    height: 0 !important;
}}
.stTabs [data-baseweb="tab"] {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    border-radius: 0;
    color: var(--ink-faint);
    padding: 0.78rem 1.5rem;
    transition: all 0.2s ease;
    background: transparent;
    border-bottom: 2px solid transparent;
    margin-bottom: -2px;
    font-weight: 500;
}}
.stTabs [data-baseweb="tab"]:hover {{
    color: var(--ink);
    background: var(--gold-soft);
}}
.stTabs [aria-selected="true"] {{
    background: transparent !important;
    color: var(--ink) !important;
    font-weight: 600 !important;
    border-bottom: 2px solid {ESPRESSO_GOLD} !important;
    box-shadow: none;
}}

/* Button — editorial CTA */
[data-testid="stButton"] > button {{
    background: var(--ink);
    color: #FAFAF7;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.74rem;
    font-weight: 500;
    letter-spacing: 0.24em;
    text-transform: uppercase;
    border: 1px solid var(--ink);
    border-radius: 0;
    padding: 0.78rem 2rem;
    width: 100%;
    transition: all 0.18s ease;
}}
[data-testid="stButton"] > button:hover {{
    background: {ESPRESSO_GOLD};
    color: var(--ink);
    border-color: {ESPRESSO_GOLD};
    letter-spacing: 0.26em;
}}

/* Streamlit form controls polish */
[data-baseweb="select"] > div {{
    border-radius: 0 !important;
    border-color: var(--rule) !important;
}}
.stSlider [data-baseweb="slider"] [role="slider"] {{
    background: {ESPRESSO_GOLD} !important;
    border: 2px solid var(--ink) !important;
}}

/* Plotly chart wrapper — framed paper card */
[data-testid="stPlotlyChart"] {{
    background: #FFFFFF;
    border: 1px solid var(--rule);
    padding: 0.65rem 0.85rem;
    box-shadow: 0 1px 0 rgba(26,29,35,0.04);
    margin-bottom: 0.2rem;
}}

/* Fig caption — italic editorial */
.fig-caption {{
    font-family: 'Fraunces', Georgia, serif;
    font-style: italic;
    font-size: 0.78rem;
    color: var(--ink-soft);
    margin: 0.3rem 0 1.2rem;
    padding: 0.1rem 0 0.1rem 0.6rem;
    border-left: 2px solid {ESPRESSO_GOLD};
    line-height: 1.45;
}}
.fig-caption b {{
    font-style: normal;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    font-size: 0.68rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--ink);
    padding-right: 0.4rem;
}}

/* Sidebar branding */
.sidebar-brand {{
    text-align: left;
    padding: 0.6rem 0 0.9rem;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 1rem;
}}
.sidebar-brand-mark {{
    font-family: 'Fraunces', Georgia, serif;
    font-style: italic;
    font-size: 1.55rem;
    color: {ESPRESSO_GOLD};
    line-height: 1.1;
    font-weight: 500;
}}
.sidebar-brand-mark span {{
    font-style: normal;
    color: #FAFAF7;
    font-weight: 600;
}}
.sidebar-brand-id {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.6rem;
    color: rgba(176, 180, 184, 0.55) !important;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    margin-top: 0.55rem;
}}

.sidebar-pill {{
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    padding: 0.18rem 0.55rem;
    border: 1px solid rgba(201, 168, 106, 0.4);
    background: rgba(201, 168, 106, 0.08);
    color: {ESPRESSO_GOLD} !important;
}}
.sidebar-pill .pd {{
    display: inline-block;
    width: 5px; height: 5px;
    background: {SUCCESS};
    border-radius: 50%;
    box-shadow: 0 0 6px rgba(67, 147, 108, 0.7);
    animation: pulse 2.2s ease-in-out infinite;
}}

/* Stagger fade-in for hero */
@keyframes rise {{
    from {{ opacity: 0; transform: translateY(10px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
.issue-strip  {{ animation: rise 0.55s ease-out backwards; animation-delay: 0.02s; }}
.main-header  {{ animation: rise 0.55s ease-out backwards; animation-delay: 0.10s; }}
.sub-header   {{ animation: rise 0.55s ease-out backwards; animation-delay: 0.18s; }}

/* Streamlit container neutralizing */
.block-container {{ padding-top: 1.5rem !important; }}
</style>
""", unsafe_allow_html=True)

# ─── Data Paths ───────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(ROOT, "data", "retail_store_inventory.csv")
# The app uses model_contextual.pkl — a model the notebook trains specifically
# against the DEMO_FEATURES below (no lag features required at inference).
# model.pkl is the Stage 2 batch model and expects pre-computed lag/rolling
# features; loading it here previously caused silent feature-mismatch errors.
MODEL_PATH = os.path.join(ROOT, "model", "model_contextual.pkl")
Q80_PATH   = os.path.join(ROOT, "model", "model_q80.pkl")
META_PATH  = os.path.join(ROOT, "model", "model_metadata.pkl")

DEMO_FEATURES = [
    "month", "day_of_week", "quarter", "is_weekend",
    "Inventory Level", "Price", "Discount", "price_vs_competitor",
    "Holiday/Promotion",
    "Category", "Region", "Store ID", "Weather Condition", "Seasonality",
]
NUMERIC_STD  = ["Inventory Level"]
NUMERIC_MM   = ["month", "day_of_week", "quarter", "is_weekend",
                 "Discount", "Holiday/Promotion"]
NUMERIC_POW  = ["Price", "price_vs_competitor"]
CATEGORICALS = ["Category", "Region", "Store ID", "Weather Condition", "Seasonality"]


# ─── Helpers ──────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"]             = pd.to_datetime(df["Date"])
    df["month"]            = df["Date"].dt.month
    df["day_of_week"]      = df["Date"].dt.dayofweek
    df["quarter"]          = df["Date"].dt.quarter
    df["is_weekend"]       = (df["Date"].dt.dayofweek >= 5).astype(int)
    df["price_vs_competitor"] = df["Price"] / df["Competitor Pricing"].clip(lower=0.01)
    return df


@st.cache_data(show_spinner="Loading dataset…")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return engineer_features(df)


@st.cache_resource(show_spinner="Training demo model…")
def get_model():
    """Load pkl if available; otherwise train a lightweight contextual model."""
    if os.path.exists(MODEL_PATH):
        pipeline = joblib.load(MODEL_PATH)
        meta     = joblib.load(META_PATH) if os.path.exists(META_PATH) else {}
        # Validate that the saved model's expected features match the app's
        # DEMO_FEATURES — fail loudly instead of silently mis-predicting.
        expected = meta.get("app_feature_columns") or meta.get("feature_columns")
        if expected and set(expected) != set(DEMO_FEATURES):
            missing = sorted(set(expected) - set(DEMO_FEATURES))
            extra   = sorted(set(DEMO_FEATURES) - set(expected))
            st.error(
                "Model/feature mismatch — re-run the notebook to regenerate "
                f"`model_contextual.pkl`.\n\nMissing in app: `{missing}`\nExtra in app: `{extra}`"
            )
            st.stop()
        return pipeline, meta, False  # False = not demo mode

    df = load_data()
    X  = df[DEMO_FEATURES]
    y  = df["Units Sold"]

    preprocessor = ColumnTransformer([
        ("std",    StandardScaler(),                                              NUMERIC_STD),
        ("mm",     MinMaxScaler(),                                                NUMERIC_MM),
        ("pow",    StandardScaler(),                                              NUMERIC_POW),
        ("ohe",    OneHotEncoder(drop="first", sparse_output=False,
                                 handle_unknown="ignore"),                        CATEGORICALS),
    ], remainder="drop")

    pipeline = Pipeline([
        ("prep",  preprocessor),
        ("model", RandomForestRegressor(n_estimators=80, max_depth=12,
                                        n_jobs=-1, random_state=42)),
    ])
    pipeline.fit(X, y)

    meta = {
        "model_name":      "Random Forest (Demo — no lag features)",
        "feature_columns": DEMO_FEATURES,
        "note": (
            "Trained on contextual features only. Lag features excluded for "
            "interactive simulation. Run the notebook to train the full model."
        ),
    }
    return pipeline, meta, True  # True = demo mode


def predict_demand(pipeline, row_dict: dict) -> float:
    df = pd.DataFrame([row_dict])
    missing = [c for c in DEMO_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required model features: {missing}")
    return float(pipeline.predict(df[DEMO_FEATURES])[0])


@st.cache_resource(show_spinner=False)
def get_q80_model():
    """P80 quantile model for reorder/safety-stock decisions. Optional."""
    if os.path.exists(Q80_PATH):
        try:
            return joblib.load(Q80_PATH)
        except Exception:
            return None
    return None


def predict_p80(q80_pipeline, row_dict: dict) -> float:
    """Predict the 80th-percentile demand — stock-to target for reorder advisory."""
    if q80_pipeline is None:
        return float("nan")
    df = pd.DataFrame([row_dict])
    # The q80 model is trained on Stage 2's full feature set including lag
    # features. In interactive mode we have no history, so fall back to
    # filling any missing lag/rolling columns with the row's Inventory Level
    # as a coarse proxy. This keeps the safety-stock advisory directional,
    # not deceptively precise.
    expected = getattr(q80_pipeline.named_steps.get("prep", None), "feature_names_in_", None)
    if expected is not None:
        for col in expected:
            if col not in df.columns:
                df[col] = row_dict.get("Inventory Level", 0) if "lag" in col or "roll" in col else 0
        df = df[list(expected)]
    try:
        return float(q80_pipeline.predict(df)[0])
    except Exception:
        return float("nan")


def stock_status(inventory: int, predicted: float):
    coverage = inventory / max(predicted, 1)
    if coverage < 0.5:
        return "CRITICAL — STOCKOUT RISK", DANGER, "danger", coverage
    elif coverage < 1.2:
        return "LOW STOCK", WARNING_COL, "warning", coverage
    else:
        return "WELL STOCKED", SUCCESS, "success", coverage


def reorder_qty(predicted: float, inventory: int, safety_days: int = 7) -> int:
    buffer = predicted * safety_days
    shortage = max(0, buffer - inventory)
    return int(np.ceil(shortage / max(predicted, 1)) * max(predicted, 1))


# ─── Load Resources ───────────────────────────────────────────────────────────
df              = load_data()
pipeline, meta, is_demo = get_model()
q80_pipeline    = get_q80_model()

stores      = sorted(df["Store ID"].unique())
categories  = sorted(df["Category"].unique())
regions     = sorted(df["Region"].unique())
weathers    = sorted(df["Weather Condition"].unique())
seasons     = sorted(df["Seasonality"].unique())

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div class='sidebar-brand'>
        <div class='sidebar-brand-mark'>Demand <span>Intel.</span></div>
        <div class='sidebar-brand-id'>v02 · MMXXVI · OP</div>
        <div style='margin-top:0.7rem;'><span class='sidebar-pill'><span class='pd'></span>Online</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"<div style='font-family:\"Fraunces\",serif; font-style:italic; font-size:0.95rem; color:{MIST}; margin-bottom:0.3rem;'>Dashboard filters</div>", unsafe_allow_html=True)
    sel_store    = st.selectbox("Store", ["All"] + stores)
    sel_category = st.selectbox("Category", ["All"] + categories)

    st.markdown("<hr style='border:none;height:1px;background:rgba(255,255,255,0.08);margin:1.2rem 0'>", unsafe_allow_html=True)

    model_label = "DEMO MODEL" if is_demo else "FULL MODEL"
    model_color = WARNING_COL if is_demo else SUCCESS
    st.markdown(f"""
    <div style='font-family:"Fraunces",serif; font-style:italic; font-size:0.92rem; color:{MIST};'>Model status</div>
    <div style='font-family:Orbitron,monospace; font-size:0.95rem; color:{model_color};
                font-weight:700; letter-spacing:0.06em; margin:0.4rem 0 0.3rem;'>● {model_label}</div>
    <div style='font-family:"IBM Plex Mono",monospace; font-size:0.66rem; color:rgba(176,180,184,0.7) !important; line-height:1.55;'>{meta.get("model_name","—")}</div>
    """, unsafe_allow_html=True)

    if is_demo:
        st.markdown(f"""
        <div class='info-box' style='margin-top:0.8rem; font-size:0.68rem;'>
            Run the notebook to train the full time-series model with lag features and replace this demo.
        </div>
        """, unsafe_allow_html=True)

# ─── Emergency Advisory (used by alert strip + reorder tab) ───────────────────
_recent_df = df.copy()
_recent_df["Date"] = pd.to_datetime(_recent_df["Date"])
_cutoff = _recent_df["Date"].max() - pd.Timedelta(days=30)
_recent = _recent_df[_recent_df["Date"] >= _cutoff]

_advisory_global = (
    _recent.groupby(["Store ID", "Category"])
    .agg(
        Avg_Daily_Demand=("Units Sold", "median"),
        Current_Inventory=("Inventory Level", "last"),
    )
    .reset_index()
)
_advisory_global["Days_of_Supply"] = (
    _advisory_global["Current_Inventory"]
    / _advisory_global["Avg_Daily_Demand"].clip(lower=1)
).round(1)
critical_count = int((_advisory_global["Days_of_Supply"] < 3.5).sum())
low_count = int(
    ((_advisory_global["Days_of_Supply"] >= 3.5) & (_advisory_global["Days_of_Supply"] < 7)).sum()
)
total_skus = len(_advisory_global)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class='issue-strip'>
    <span><span class='dot'></span>LIVE · LAB 01 · INVENTORY</span>
    <span>CRISP-ML(Q)</span>
    <span>OSCAR PONCE · MMXXVI</span>
</div>
<div class='main-header'>Retail Demand <em>Intelligence</em></div>
<div class='sub-header'>Inventory forecasting · 73,100 retail records · 5 stores · 20 products</div>
""", unsafe_allow_html=True)

# ─── Qué hace esta app — explicación en español plano ─────────────────────────
st.markdown(f"""
<div style='background:rgba(74,103,176,0.07); border-left:3px solid {INFO};
            padding:1rem 1.3rem; margin:1.2rem 0 0.5rem;
            font-family:"IBM Plex Mono",monospace; font-size:0.82rem;
            color:{SLATE}; line-height:1.65;'>
  <b style='color:{INFO}; letter-spacing:0.1em; font-size:0.7rem;'>QUÉ HACE ESTA APP ·</b>
  Predice <b style='color:{ESPRESSO_GOLD};'>cuántas unidades se venderán cada día</b>
  por tienda y producto, detecta qué SKUs están en
  <b style='color:{ESPRESSO_GOLD};'>riesgo de stockout</b> antes que ocurra, y recomienda
  <b style='color:{ESPRESSO_GOLD};'>cuánto pedir y cuándo</b> — el objetivo es no perder
  ventas por falta de stock ni acumular inventario muerto.
  <div style='font-size:0.7rem; color:{PEBBLE}; margin-top:0.55rem;'>
    Lab CRISP-ML(Q) sobre dataset Kaggle de retail (sintético).
    <a href='https://github.com/oscarinho/crisp-ml-retail-forecasting/blob/main/EXPERIMENT_DF_RESIDUAL.md'
       style='color:{INFO}; text-decoration:underline; text-underline-offset:2px;'
       target='_blank'>Detalle técnico (residual learning, MAE 7.4) →</a>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Emergency Alert Strip ────────────────────────────────────────────────────
if critical_count > 0:
    _low_extra = (
        f' <span style="color:{PEBBLE};">+{low_count} con stock bajo (3–7 días).</span>'
        if low_count > 0 else ''
    )
    st.markdown(f"""
    <div style='background:rgba(217,107,95,0.10); border:1px solid {DANGER};
                border-left:4px solid {DANGER};
                padding:0.7rem 1.1rem; margin:0.4rem 0 1rem;
                font-family:"IBM Plex Mono",monospace; font-size:0.78rem;
                color:{SLATE}; display:flex; justify-content:space-between;
                align-items:center; gap:1rem; flex-wrap:wrap;'>
      <span>
        <b style='color:{DANGER}; letter-spacing:0.1em; font-size:0.72rem;'>⚠ ALERTA CRÍTICA ·</b>
        <b style='font-family:Orbitron,monospace; font-size:1.05rem; color:{DANGER};'>{critical_count}</b>
        SKUs en riesgo de stockout (menos de 3 días de stock).{_low_extra}
      </span>
      <span style='font-size:0.65rem; color:{PEBBLE}; letter-spacing:0.18em; text-transform:uppercase;'>
        Ver pestaña Reorder Advisory ↓
      </span>
    </div>
    """, unsafe_allow_html=True)
elif low_count > 0:
    st.markdown(f"""
    <div style='background:rgba(242,174,74,0.08); border-left:3px solid {WARNING_COL};
                padding:0.6rem 1.1rem; margin:0.4rem 0 1rem;
                font-family:"IBM Plex Mono",monospace; font-size:0.76rem;
                color:{SLATE};'>
      <b style='color:{WARNING_COL}; letter-spacing:0.08em; font-size:0.7rem;'>STOCK BAJO ·</b>
      <b style='font-family:Orbitron,monospace; color:{WARNING_COL};'>{low_count}</b>
      SKUs por debajo del buffer de 7 días — revisar Reorder Advisory.
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div style='background:rgba(67,147,108,0.07); border-left:3px solid {SUCCESS};
                padding:0.6rem 1.1rem; margin:0.4rem 0 1rem;
                font-family:"IBM Plex Mono",monospace; font-size:0.76rem;
                color:{SLATE};'>
      <b style='color:{SUCCESS}; letter-spacing:0.08em; font-size:0.7rem;'>STOCK OK ·</b>
      Los {total_skus} SKUs activos tienen cobertura ≥ 7 días.
    </div>
    """, unsafe_allow_html=True)

# ─── KPI Strip ────────────────────────────────────────────────────────────────
view_df = df.copy()
if sel_store    != "All": view_df = view_df[view_df["Store ID"]  == sel_store]
if sel_category != "All": view_df = view_df[view_df["Category"] == sel_category]

avg_demand  = view_df["Units Sold"].mean()
avg_inv     = view_df["Inventory Level"].mean()
coverage    = avg_inv / max(avg_demand, 1)
stockout_pct = (view_df["Units Sold"] > view_df["Inventory Level"]).mean() * 100
holiday_lift = view_df.groupby("Holiday/Promotion")["Units Sold"].mean()
h_lift = ((holiday_lift.get(1, avg_demand) / max(holiday_lift.get(0, avg_demand), 1)) - 1) * 100

k1, k2, k3, k4, k5 = st.columns(5)
kpi_data = [
    (k1, "Avg Daily Demand",    f"{avg_demand:.0f}",   "units / day"),
    (k2, "Avg Inventory",       f"{avg_inv:.0f}",      "units on hand"),
    (k3, "Stock Coverage",      f"{coverage:.1f}x",    "demand days covered"),
    (k4, "Stockout Events",     f"{stockout_pct:.1f}%","days demand > stock"),
    (k5, "Holiday Demand Lift", f"+{h_lift:.1f}%",     "vs. non-holiday avg"),
]
for col, label, value, delta in kpi_data:
    col.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>{label}</div>
        <div class='metric-value'>{value}</div>
        <div class='metric-delta'>{delta}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ─── Main Tabs ────────────────────────────────────────────────────────────────
tab_sim, tab_dash, tab_reorder = st.tabs([
    "  Demand Simulator  ",
    "  Inventory Dashboard  ",
    "  Reorder Advisory  ",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DEMAND SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════════
with tab_sim:
    st.markdown("<div class='section-header'>Configure Scenario</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:0.78rem; color:{PEBBLE}; margin-bottom:1rem;'>Adjust the inputs to simulate demand for any store / product combination. The model predicts <b>Units Sold</b> based on contextual signals — no historical lag required.</div>", unsafe_allow_html=True)

    form_col, result_col = st.columns([1.1, 0.9], gap="large")

    with form_col:
        # Card 1 — Context
        st.markdown(f"""<div class='input-card'><div class='input-card-header'>Context</div>""", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        store_in    = c1.selectbox("Store",    stores,    key="sim_store")
        category_in = c2.selectbox("Category", categories, key="sim_cat")
        region_in   = c3.selectbox("Region",  regions,   key="sim_region")
        st.markdown("</div>", unsafe_allow_html=True)

        # Card 2 — Time & Environment
        st.markdown(f"""<div class='input-card'><div class='input-card-header'>Time & Environment</div>""", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        month_in   = c1.selectbox("Month", list(range(1, 13)), index=0, key="sim_month")
        weekday_in = c2.selectbox("Weekday", ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], key="sim_wd")
        season_in  = c3.selectbox("Season",  seasons,  key="sim_season")
        weather_in = c4.selectbox("Weather", weathers, key="sim_weather")
        holiday_in = st.checkbox("Holiday / Promotion active", value=False, key="sim_holiday")
        st.markdown("</div>", unsafe_allow_html=True)

        # Card 3 — Pricing & Stock
        st.markdown(f"""<div class='input-card'><div class='input-card-header'>Pricing & Inventory</div>""", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        price_in    = c1.slider("Price ($)",             5.0,  100.0, 35.0, 0.5, key="sim_price")
        discount_in = c2.slider("Discount (%)",          0,    20,    10,   5,   key="sim_disc")
        comp_in     = c3.slider("Competitor Price ($)",  5.0,  110.0, 37.0, 0.5, key="sim_comp")
        inv_in      = st.slider("Current Inventory (units)", 50, 500, 250, 10, key="sim_inv")
        st.markdown("</div>", unsafe_allow_html=True)

        predict_btn = st.button("Run Forecast", key="predict")

    with result_col:
        weekday_map = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}
        wd_num      = weekday_map[weekday_in]
        is_wknd     = 1 if wd_num >= 5 else 0

        row = {
            "month":              month_in,
            "day_of_week":        wd_num,
            "quarter":            (month_in - 1) // 3 + 1,
            "is_weekend":         is_wknd,
            "Inventory Level":    inv_in,
            "Price":              price_in,
            "Discount":           discount_in,
            "price_vs_competitor": price_in / max(comp_in, 0.01),
            "Holiday/Promotion":  int(holiday_in),
            "Category":           category_in,
            "Region":             region_in,
            "Store ID":           store_in,
            "Weather Condition":  weather_in,
            "Seasonality":        season_in,
        }

        pred        = predict_demand(pipeline, row)
        pred_int    = max(0, int(round(pred)))
        status, color, box_class, cov = stock_status(inv_in, pred_int)
        # Reorder advisory uses the P80 quantile (stock-to target) when the
        # quantile model is available — newsvendor logic: cover demand 80% of
        # cycles instead of the 50% you'd get from the point forecast.
        p80_pred = predict_p80(q80_pipeline, row) if q80_pipeline is not None else float("nan")
        stock_to_target = int(round(p80_pred)) if not (p80_pred != p80_pred) else pred_int
        reorder     = reorder_qty(max(stock_to_target, pred_int), inv_in)

        st.markdown(f"""
        <div class='prediction-box {box_class}'>
            <div class='pred-stamp'>CRISP · PRED v1</div>
            <div class='prediction-label'>Predicted daily demand</div>
            <div class='prediction-value' style='color:{color};'>{pred_int}</div>
            <div style='font-family:"IBM Plex Mono",monospace; font-size:0.68rem; margin-top:0.3rem; color:rgba(26,29,35,0.45); letter-spacing:0.12em; text-transform:uppercase;'>units · {category_in} · {store_in}</div>
            <hr style='border:none;height:1px;background:rgba(26,29,35,0.12);margin:0.9rem 0 0.7rem;'>
            <div style='font-family:"IBM Plex Mono",monospace; font-size:0.78rem; line-height:1.7;'>
                <span style='font-family:"Fraunces",serif; font-style:italic; color:rgba(26,29,35,0.55);'>Stock status</span> &nbsp;
                <span style='color:{color}; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; font-size:0.72rem;'>{status}</span><br>
                <span style='font-family:"Fraunces",serif; font-style:italic; color:rgba(26,29,35,0.55);'>Coverage</span> {cov:.1f}× &nbsp;·&nbsp;
                <span style='font-family:"Fraunces",serif; font-style:italic; color:rgba(26,29,35,0.55);'>Inventory</span> {inv_in} units
            </div>
            {"<div class='prediction-advisory'>▸ Recommended reorder: <b>" + str(reorder) + " units</b></div>" if reorder > 0 else "<div class='prediction-advisory' style='border-left-color:" + SUCCESS + "; color:" + SUCCESS + ";'>✓ No reorder needed</div>"}
        </div>
        """, unsafe_allow_html=True)

        # ─── Qué hacer — plan de acción en español ────────────────────────────
        if cov < 0.5:
            _action_color = DANGER
            _action_html = (
                f"Pide <b style='color:{DANGER};'>{reorder} unidades hoy mismo</b> — "
                f"el stock actual no cubre la demanda proyectada."
            )
        elif cov < 1.2:
            _action_color = WARNING_COL
            _action_html = (
                f"Considera pedir <b style='color:{WARNING_COL};'>{reorder} unidades esta semana</b> "
                f"para mantener un buffer de 7 días."
            )
        else:
            _action_color = SUCCESS
            _action_html = (
                f"<b style='color:{SUCCESS};'>Stock suficiente</b> — "
                f"no requiere reabastecimiento por ahora."
            )

        st.markdown(f"""
        <div style='background:#FFFFFF; border:1px solid var(--rule);
                    border-left:3px solid {_action_color};
                    padding:1rem 1.2rem; margin:0.8rem 0 1rem;
                    font-family:"IBM Plex Mono",monospace; font-size:0.82rem;
                    line-height:1.75;'>
          <div style='font-family:"Fraunces",serif; font-style:italic; font-size:1.05rem;
                      color:{INK}; margin-bottom:0.5rem;'>Qué hacer</div>
          <div style='color:{SLATE};'>
            ▸ Se proyectan <b>{pred_int} unidades</b> de venta diaria para este escenario.<br>
            ▸ El stock actual ({inv_in} unidades) alcanza para <b>{cov:.1f} días</b> a ese ritmo.<br>
            ▸ {_action_html}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # What-if: sweep price ±20%
        prices     = np.linspace(price_in * 0.8, price_in * 1.2, 9)
        preds_price = []
        for p in prices:
            r2 = {**row, "Price": p, "price_vs_competitor": p / max(comp_in, 0.01)}
            preds_price.append(predict_demand(pipeline, r2))

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=prices, y=preds_price,
            mode="lines+markers",
            line=dict(color=ESPRESSO_GOLD, width=2.5),
            marker=dict(size=6, color=ESPRESSO_GOLD),
            name="Predicted Demand",
        ))
        fig.add_vline(x=price_in, line_dash="dot", line_color=GRAPHITE, line_width=1.5)
        fig.update_layout(
            title=dict(text="Demand vs. Price Sensitivity", font_family="Orbitron",
                       font_size=12, font_color=GRAPHITE_DEEP),
            xaxis_title="Price ($)", yaxis_title="Units Sold",
            height=220, margin=dict(t=40, b=30, l=30, r=10),
            plot_bgcolor="white", paper_bgcolor="white",
            font_family="IBM Plex Mono",
            xaxis=dict(gridcolor=SILVER), yaxis=dict(gridcolor=SILVER),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("<div class='fig-caption'><b>Fig. 01</b>Demand elasticity across a ±20% price band — vertical rule marks the current price.</div>", unsafe_allow_html=True)

        # Holiday vs. baseline
        row_no_hol  = {**row, "Holiday/Promotion": 0}
        row_hol     = {**row, "Holiday/Promotion": 1}
        p_no_hol    = predict_demand(pipeline, row_no_hol)
        p_hol       = predict_demand(pipeline, row_hol)
        lift        = ((p_hol / max(p_no_hol, 1)) - 1) * 100

        fig2 = go.Figure(go.Bar(
            x=["No Holiday", "Holiday / Promo"],
            y=[p_no_hol, p_hol],
            marker_color=[MIST, ESPRESSO_GOLD],
            text=[f"{p_no_hol:.0f}", f"{p_hol:.0f}"],
            textposition="outside",
            textfont=dict(family="Orbitron", size=11),
        ))
        fig2.update_layout(
            title=dict(text=f"Holiday Lift: {lift:+.1f}%", font_family="Orbitron",
                       font_size=12, font_color=GRAPHITE_DEEP),
            height=180, margin=dict(t=40, b=20, l=30, r=10),
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(gridcolor=SILVER, showgrid=True),
            xaxis=dict(gridcolor="rgba(0,0,0,0)"),
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — INVENTORY DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
with tab_dash:
    st.markdown("<div class='section-header'>Demand Trends</div>", unsafe_allow_html=True)

    dash_df = view_df.copy()
    dash_df["Date"] = pd.to_datetime(dash_df["Date"])

    # Demand over time by category (weekly resample)
    trend = (
        dash_df.groupby(["Date", "Category"])["Units Sold"]
        .mean()
        .reset_index()
    )
    trend_weekly = (
        trend.set_index("Date")
        .groupby("Category")["Units Sold"]
        .resample("W")
        .mean()
        .reset_index()
    )

    fig3 = px.line(
        trend_weekly, x="Date", y="Units Sold", color="Category",
        color_discrete_sequence=[ESPRESSO_GOLD, INFO, SUCCESS, DANGER, PEBBLE],
    )
    fig3.update_layout(
        height=320, plot_bgcolor="white", paper_bgcolor="white",
        font_family="IBM Plex Mono",
        xaxis=dict(gridcolor=SILVER, title=None),
        yaxis=dict(gridcolor=SILVER, title="Units Sold"),
        legend=dict(orientation="h", yanchor="top", y=-0.18, x=0.5, xanchor="center", font=dict(size=10)),
        margin=dict(t=15, b=55, l=10, r=10),
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("<div class='fig-caption'><b>Fig. 02</b>Weekly-resampled mean demand by category, full two-year window.</div>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("<div class='section-header'>Demand by Season & Weather</div>", unsafe_allow_html=True)
        hm_data = (
            dash_df.groupby(["Seasonality", "Weather Condition"])["Units Sold"]
            .mean()
            .unstack(fill_value=0)
        )
        fig4 = px.imshow(
            hm_data,
            color_continuous_scale=[[0, PLATINUM], [0.5, ESPRESSO_GOLD], [1, GRAPHITE_DEEP]],
            title="Mean Units Sold",
            text_auto=".0f",
        )
        fig4.update_layout(
            height=270, plot_bgcolor="white", paper_bgcolor="white",
            font_family="IBM Plex Mono",
            title_font_family="Orbitron", title_font_size=12, title_font_color=GRAPHITE_DEEP,
            margin=dict(t=40, b=10, l=10, r=10),
        )
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown("<div class='fig-caption'><b>Fig. 03</b>Mean units sold across seasonality × weather combinations.</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown("<div class='section-header'>Inventory vs. Demand by Store</div>", unsafe_allow_html=True)
        store_agg = dash_df.groupby("Store ID").agg(
            Demand=("Units Sold", "mean"),
            Inventory=("Inventory Level", "mean"),
        ).reset_index()

        fig5 = go.Figure()
        fig5.add_trace(go.Bar(
            name="Avg Demand",   x=store_agg["Store ID"], y=store_agg["Demand"],
            marker_color=ESPRESSO_GOLD,
        ))
        fig5.add_trace(go.Bar(
            name="Avg Inventory", x=store_agg["Store ID"], y=store_agg["Inventory"],
            marker_color="rgba(74, 103, 176, 0.53)",
        ))
        fig5.update_layout(
            barmode="group", height=300, plot_bgcolor="white", paper_bgcolor="white",
            font_family="IBM Plex Mono", bargap=0.35,
            xaxis=dict(gridcolor="rgba(0,0,0,0)", title=None),
            yaxis=dict(gridcolor=SILVER),
            legend=dict(orientation="h", yanchor="top", y=-0.18, x=0.5, xanchor="center",
                        font=dict(size=10)),
            margin=dict(t=15, b=55, l=10, r=10),
        )
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown("<div class='fig-caption'><b>Fig. 04</b>Side-by-side store comparison — gold = demand, indigo = inventory.</div>", unsafe_allow_html=True)

    # Demand distribution
    st.markdown("<div class='section-header'>Demand Distribution</div>", unsafe_allow_html=True)
    fig6 = px.histogram(
        dash_df, x="Units Sold", color="Category", nbins=40,
        barmode="overlay", opacity=0.7,
        color_discrete_sequence=[ESPRESSO_GOLD, INFO, SUCCESS, DANGER, PEBBLE],
        title="Distribution of Daily Units Sold",
    )
    fig6.update_layout(
        height=260, plot_bgcolor="white", paper_bgcolor="white",
        font_family="IBM Plex Mono",
        title_font_family="Orbitron", title_font_size=13, title_font_color=GRAPHITE_DEEP,
        xaxis=dict(gridcolor=SILVER), yaxis=dict(gridcolor=SILVER),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig6, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — REORDER ADVISORY
# ═══════════════════════════════════════════════════════════════════════════════
with tab_reorder:
    st.markdown("<div class='section-header'>Reorder Advisory Engine</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-size:0.78rem; color:{PEBBLE}; margin-bottom:1rem;'>
        Predicted demand is computed for each store × category pair using the most
        recent 30-day median. Reorder quantity targets a <b>7-day safety buffer</b>.
    </div>
    """, unsafe_allow_html=True)

    safety_days = st.slider("Safety Buffer (days)", 3, 14, 7, key="safety")

    recent_df = df.copy()
    recent_df["Date"] = pd.to_datetime(recent_df["Date"])
    cutoff = recent_df["Date"].max() - pd.Timedelta(days=30)
    recent = recent_df[recent_df["Date"] >= cutoff]

    advisory = (
        recent.groupby(["Store ID", "Category"])
        .agg(
            Avg_Daily_Demand=("Units Sold", "median"),
            Current_Inventory=("Inventory Level", "last"),
        )
        .reset_index()
    )
    advisory["Days_of_Supply"] = (advisory["Current_Inventory"] / advisory["Avg_Daily_Demand"].clip(lower=1)).round(1)
    advisory["Reorder_Qty"]    = advisory.apply(
        lambda r: reorder_qty(r["Avg_Daily_Demand"], int(r["Current_Inventory"]), safety_days), axis=1
    )
    advisory["Risk"] = advisory["Days_of_Supply"].apply(
        lambda d: "CRITICAL" if d < safety_days * 0.5 else ("LOW STOCK" if d < safety_days else "OK")
    )

    risk_filter = st.selectbox("Filter by Risk Level", ["All", "CRITICAL", "LOW STOCK", "OK"])
    if risk_filter != "All":
        advisory = advisory[advisory["Risk"] == risk_filter]

    advisory_sorted = advisory.sort_values("Days_of_Supply")

    def style_risk(val):
        if val == "CRITICAL":  return f"background:{DANGER}25; color:{DANGER}; font-weight:700;"
        if val == "LOW STOCK": return f"background:{WARNING_COL}25; color:#996600; font-weight:700;"
        return f"color:{SUCCESS}; font-weight:700;"

    st.dataframe(
        advisory_sorted.rename(columns={
            "Store ID": "Store",
            "Avg_Daily_Demand": "Avg Demand/Day",
            "Current_Inventory": "Stock On Hand",
            "Days_of_Supply": "Days of Supply",
            "Reorder_Qty": "Reorder Qty",
        }),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Store":           st.column_config.TextColumn(width="small"),
            "Category":        st.column_config.TextColumn(width="medium"),
            "Avg Demand/Day":  st.column_config.NumberColumn(format="%.1f"),
            "Stock On Hand":   st.column_config.NumberColumn(format="%d"),
            "Days of Supply":  st.column_config.NumberColumn(format="%.1f"),
            "Reorder Qty":     st.column_config.NumberColumn(format="%d"),
            "Risk":            st.column_config.TextColumn(width="medium"),
        },
    )

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        risk_counts = advisory["Risk"].value_counts()
        fig7 = go.Figure(go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            hole=0.55,
            marker_colors=[DANGER, WARNING_COL, SUCCESS],
        ))
        fig7.update_layout(
            title=dict(text="Stock Health Distribution", font_family="Orbitron",
                       font_size=12, font_color=GRAPHITE_DEEP),
            height=260, paper_bgcolor="white",
            font_family="IBM Plex Mono",
            legend=dict(orientation="h"),
            margin=dict(t=50, b=10),
        )
        st.plotly_chart(fig7, use_container_width=True)

    with col_r2:
        top_reorder = advisory_sorted[advisory_sorted["Reorder_Qty"] > 0].head(8)
        fig8 = go.Figure(go.Bar(
            x=top_reorder["Reorder_Qty"],
            y=top_reorder["Store ID"] + " · " + top_reorder["Category"],
            orientation="h",
            marker_color=ESPRESSO_GOLD,
            text=top_reorder["Reorder_Qty"].astype(str) + " units",
            textposition="outside",
        ))
        fig8.update_layout(
            title=dict(text="Top Reorder Priorities", font_family="Orbitron",
                       font_size=12, font_color=GRAPHITE_DEEP),
            height=260, plot_bgcolor="white", paper_bgcolor="white",
            font_family="IBM Plex Mono",
            xaxis=dict(gridcolor=SILVER),
            yaxis=dict(gridcolor="rgba(0,0,0,0)", autorange="reversed"),
            margin=dict(t=50, b=10, l=10, r=80),
        )
        st.plotly_chart(fig8, use_container_width=True)

# ─── Colophon ─────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(f"""
<div style='display:flex; justify-content:space-between; align-items:baseline; flex-wrap:wrap; gap:1rem;
            font-family:"IBM Plex Mono",monospace; font-size:0.66rem;
            color:rgba(26,29,35,0.45); letter-spacing:0.22em; text-transform:uppercase;
            padding:0.8rem 0 1.8rem; border-top:1px solid rgba(26,29,35,0.12);'>
    <span><span style='font-family:"Fraunces",serif; font-style:italic; text-transform:none; letter-spacing:0; font-size:0.85rem; color:rgba(26,29,35,0.7);'>Colophon</span>
    &nbsp;·&nbsp; Built on CRISP-ML(Q) — six phases · time-aware splits · newsvendor P80</span>
    <span><a href='https://oscarponce.com' style='color:{ESPRESSO_GOLD} !important; text-decoration:none; border-bottom:1px solid {ESPRESSO_GOLD};'>oscarponce.com</a> &nbsp;·&nbsp; MMXXVI</span>
</div>
""", unsafe_allow_html=True)
