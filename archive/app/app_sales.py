"""Streamlit app for the Sales (Demand) lab.

Sibling to app.py (which serves the Inventory lab). Differences:
- Dataset:        data/sales_data.csv
- Target:         Demand (uncensored, vs Inventory's Units Sold)
- Models loaded:  model/sales/model_per_category.pkl (primary, dict of 5 LightGBMs)
                  model/sales/model_contextual.pkl  (fallback when Category unknown)
                  model/sales/model_q80.pkl         (P80 quantile for reorder)
- Features:       sales APP_FEATURES (Promotion + Epidemic; no Holiday/Promotion;
                  adds sin_year/cos_year Fourier).
- Deployment story: per-category routing (the winner from the notebook leaderboard).
"""
import os
import joblib
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ─── Brand Palette (same as app.py — Ice Graphite Hybrid) ─────────────────────
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

st.set_page_config(
    page_title="Sales Demand Intelligence | CRISP-ML",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS — Editorial Terminal aesthetic (matches app.py) ──────────────────────
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

[data-testid="stAppViewContainer"]::before {{
    content: '';
    position: fixed;
    inset: 0;
    pointer-events: none;
    z-index: 0;
    background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='180' height='180'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='2' seed='3'/><feColorMatrix values='0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 0.05 0'/></filter><rect width='100%' height='100%' filter='url(%23n)'/></svg>");
    mix-blend-mode: multiply;
}}

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
    font-size: 0.72rem; color: var(--ink-soft);
    letter-spacing: 0.24em; text-transform: uppercase;
    margin: 0.55rem 0 0; padding-bottom: 1.3rem;
    border-bottom: 2px solid var(--ink);
}}

.section-header {{
    font-family: 'Fraunces', Georgia, serif;
    font-style: italic;
    font-size: 1.3rem; font-weight: 500;
    color: var(--ink); letter-spacing: -0.005em;
    margin: 1.8rem 0 1rem;
    display: flex; align-items: baseline; gap: 0.75rem; line-height: 1.2;
}}
.section-header::before {{
    content: ''; flex: 0 0 24px; height: 1px;
    background: {ESPRESSO_GOLD}; align-self: center;
}}
.section-header::after {{
    content: ''; flex: 1 1 auto; height: 1px;
    background: var(--rule); align-self: center;
}}

.metric-card {{
    background: #FFFFFF;
    border: 1px solid var(--rule); border-radius: 0;
    padding: 1.25rem 1.35rem 1rem;
    position: relative;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    box-shadow: 0 1px 0 rgba(26,29,35,0.04);
}}
.metric-card::before {{
    content: ''; position: absolute;
    top: -1px; left: -1px; width: 14px; height: 14px;
    border-top: 1.5px solid {ESPRESSO_GOLD};
    border-left: 1.5px solid {ESPRESSO_GOLD};
}}
.metric-card::after {{
    content: ''; position: absolute;
    bottom: -1px; right: -1px; width: 14px; height: 14px;
    border-bottom: 1.5px solid {ESPRESSO_GOLD};
    border-right: 1.5px solid {ESPRESSO_GOLD};
}}
.metric-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 14px 30px rgba(26, 29, 35, 0.08);
}}
.metric-label  {{ font-family:'Fraunces',Georgia,serif; font-style:italic; font-size:0.92rem; color:var(--ink-soft); line-height:1.2; }}
.metric-value  {{ font-family:'Orbitron',monospace; font-size:2.1rem; font-weight:700; color:var(--ink); margin:0.4rem 0 0.15rem; font-feature-settings:'tnum'; letter-spacing:-0.02em; line-height:1; }}
.metric-delta  {{ font-family:'IBM Plex Mono',monospace; font-size:0.62rem; color:var(--ink-faint); letter-spacing:0.16em; text-transform:uppercase; }}

.prediction-box {{
    background: #FFFFFF;
    border: 1px solid var(--rule); border-radius: 0;
    padding: 1.7rem 1.9rem 1.4rem; margin: 1rem 0;
    position: relative;
    box-shadow: 0 18px 50px rgba(26, 29, 35, 0.07);
}}
.prediction-box::before {{ content: ''; position: absolute; top:0; left:0; right:0; height:4px; }}
.prediction-box.success::before {{ background: {SUCCESS}; }}
.prediction-box.warning::before {{ background: {WARNING_COL}; }}
.prediction-box.danger::before  {{ background: {DANGER}; }}

.pred-stamp {{
    position: absolute;
    top: 1rem; right: 1.2rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.55rem; letter-spacing: 0.32em;
    color: var(--ink-soft); background: var(--gold-soft);
    padding: 0.25rem 0.6rem; border: 1px solid var(--gold-mid);
    text-transform: uppercase;
}}

.prediction-label {{ font-family:'Fraunces',Georgia,serif; font-style:italic; font-size:1rem; color:var(--ink-soft); margin-bottom:0.2rem; opacity:1; }}
.prediction-value {{ font-family:'Orbitron',monospace; font-size:3.6rem; font-weight:900; line-height:1; margin:0.2rem 0; font-feature-settings:'tnum'; letter-spacing:-0.03em; }}
.prediction-advisory {{
    font-family:'IBM Plex Mono',monospace; font-size:0.78rem;
    margin-top:0.85rem; font-weight:500; letter-spacing:0.03em;
    padding:0.55rem 0.85rem;
    background:rgba(26,29,35,0.035); border-left:2px solid {ESPRESSO_GOLD};
}}

.input-card {{
    background:#FFFFFF; border:1px solid var(--rule); border-radius:0;
    padding:1.25rem 1.5rem 1rem; margin-bottom:0.9rem;
    box-shadow:0 1px 0 rgba(26,29,35,0.04); position:relative;
}}
.input-card-header {{
    font-family:'Fraunces',Georgia,serif; font-style:italic;
    font-size:1.05rem; font-weight:500; color:var(--ink);
    border-bottom:1px solid var(--rule);
    padding-bottom:0.55rem; margin-bottom:0.95rem;
    display:flex; align-items:baseline; gap:0.5rem;
}}
.input-card-header::before {{
    content:'§'; color:{ESPRESSO_GOLD};
    font-family:'IBM Plex Mono',monospace; font-style:normal;
    font-weight:600; font-size:0.95rem;
}}

.info-box {{
    background: var(--gold-soft);
    border-left: 2px solid {ESPRESSO_GOLD};
    border-top: 1px solid var(--rule);
    border-right: 1px solid var(--rule);
    border-bottom: 1px solid var(--rule);
    border-radius:0; padding:0.95rem 1.1rem;
    font-family:'IBM Plex Mono',monospace; font-size:0.78rem;
    color:var(--ink); margin:0.5rem 0; line-height:1.65;
}}

.routing-pill {{
    display: inline-block;
    background: var(--gold-soft); color: var(--ink);
    padding: 0.18rem 0.6rem; border-radius: 0;
    border: 1px solid var(--gold-mid);
    font-family:'IBM Plex Mono',monospace;
    font-size:0.6rem; letter-spacing:0.22em;
    text-transform:uppercase;
}}

hr {{
    border:none; height:1px;
    background:linear-gradient(90deg, transparent, var(--ink) 25%, var(--ink) 75%, transparent);
    opacity:0.25; margin:1.5rem 0;
}}

.stTabs [data-baseweb="tab-list"] {{
    gap:0; background:transparent; border-radius:0; padding:0;
    border-bottom:2px solid var(--ink); margin-bottom:1.3rem;
}}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {{ background:transparent !important; height:0 !important; }}
.stTabs [data-baseweb="tab"] {{
    font-family:'IBM Plex Mono',monospace; font-size:0.72rem;
    letter-spacing:0.22em; text-transform:uppercase;
    border-radius:0; color:var(--ink-faint);
    padding:0.78rem 1.5rem; transition:all 0.2s ease;
    background:transparent; border-bottom:2px solid transparent;
    margin-bottom:-2px; font-weight:500;
}}
.stTabs [data-baseweb="tab"]:hover {{ color:var(--ink); background:var(--gold-soft); }}
.stTabs [aria-selected="true"] {{
    background:transparent !important; color:var(--ink) !important;
    font-weight:600 !important;
    border-bottom:2px solid {ESPRESSO_GOLD} !important;
    box-shadow:none;
}}

[data-testid="stButton"] > button {{
    background:var(--ink); color:#FAFAF7;
    font-family:'IBM Plex Mono',monospace; font-size:0.74rem;
    font-weight:500; letter-spacing:0.24em; text-transform:uppercase;
    border:1px solid var(--ink); border-radius:0;
    padding:0.78rem 2rem; width:100%; transition:all 0.18s ease;
}}
[data-testid="stButton"] > button:hover {{
    background:{ESPRESSO_GOLD}; color:var(--ink);
    border-color:{ESPRESSO_GOLD}; letter-spacing:0.26em;
}}

[data-baseweb="select"] > div {{ border-radius:0 !important; border-color:var(--rule) !important; }}
.stSlider [data-baseweb="slider"] [role="slider"] {{ background:{ESPRESSO_GOLD} !important; border:2px solid var(--ink) !important; }}

[data-testid="stPlotlyChart"] {{
    background:#FFFFFF; border:1px solid var(--rule);
    padding:0.65rem 0.85rem; box-shadow:0 1px 0 rgba(26,29,35,0.04);
    margin-bottom:0.2rem;
}}

.fig-caption {{
    font-family:'Fraunces',Georgia,serif; font-style:italic;
    font-size:0.78rem; color:var(--ink-soft);
    margin:0.3rem 0 1.2rem; padding:0.1rem 0 0.1rem 0.6rem;
    border-left:2px solid {ESPRESSO_GOLD}; line-height:1.45;
}}
.fig-caption b {{
    font-style:normal; font-family:'IBM Plex Mono',monospace;
    font-weight:600; font-size:0.68rem;
    letter-spacing:0.2em; text-transform:uppercase;
    color:var(--ink); padding-right:0.4rem;
}}

.sidebar-brand {{
    text-align:left; padding:0.6rem 0 0.9rem;
    border-bottom:1px solid rgba(255,255,255,0.08);
    margin-bottom:1rem;
}}
.sidebar-brand-mark {{
    font-family:'Fraunces',Georgia,serif; font-style:italic;
    font-size:1.55rem; color:{ESPRESSO_GOLD};
    line-height:1.1; font-weight:500;
}}
.sidebar-brand-mark span {{ font-style:normal; color:#FAFAF7; font-weight:600; }}
.sidebar-brand-id {{
    font-family:'IBM Plex Mono',monospace; font-size:0.6rem;
    color:rgba(176,180,184,0.55) !important;
    letter-spacing:0.3em; text-transform:uppercase; margin-top:0.55rem;
}}

.sidebar-pill {{
    display:inline-flex; align-items:center; gap:0.4rem;
    font-family:'IBM Plex Mono',monospace; font-size:0.58rem;
    letter-spacing:0.22em; text-transform:uppercase;
    padding:0.18rem 0.55rem;
    border:1px solid rgba(201, 168, 106, 0.4);
    background:rgba(201, 168, 106, 0.08);
    color:{ESPRESSO_GOLD} !important;
}}
.sidebar-pill .pd {{
    display:inline-block; width:5px; height:5px;
    background:{SUCCESS}; border-radius:50%;
    box-shadow:0 0 6px rgba(67, 147, 108, 0.7);
    animation:pulse 2.2s ease-in-out infinite;
}}

@keyframes rise {{
    from {{ opacity:0; transform:translateY(10px); }}
    to   {{ opacity:1; transform:translateY(0); }}
}}
.issue-strip {{ animation:rise 0.55s ease-out backwards; animation-delay:0.02s; }}
.main-header {{ animation:rise 0.55s ease-out backwards; animation-delay:0.10s; }}
.sub-header  {{ animation:rise 0.55s ease-out backwards; animation-delay:0.18s; }}

.block-container {{ padding-top:1.5rem !important; }}
</style>
""", unsafe_allow_html=True)

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(ROOT, "data",  "sales_data.csv")
PER_CAT_PATH = os.path.join(ROOT, "model", "sales", "model_per_category.pkl")
APP_PATH     = os.path.join(ROOT, "model", "sales", "model_contextual.pkl")
Q80_PATH     = os.path.join(ROOT, "model", "sales", "model_q80.pkl")
META_PATH    = os.path.join(ROOT, "model", "sales", "model_metadata.pkl")

# Feature schema must match notebook's APP_FEATURES exactly.
DEMO_FEATURES = [
    'Category','Region','Weather Condition','Seasonality',
    'Inventory Level','Price','Discount','Competitor Pricing',
    'Promotion','Epidemic',
    'day_of_week','month','is_weekend','sin_year','cos_year',
]


# ─── Helpers ──────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Replicates §3.2 of the notebook exactly."""
    df = df.copy()
    df["Date"]        = pd.to_datetime(df["Date"])
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["month"]       = df["Date"].dt.month
    df["is_weekend"]  = (df["Date"].dt.dayofweek >= 5).astype(int)
    doy = df["Date"].dt.dayofyear
    df["sin_year"] = np.sin(2 * np.pi * doy / 365.25)
    df["cos_year"] = np.cos(2 * np.pi * doy / 365.25)
    return df


@st.cache_data(show_spinner="Loading dataset…")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return engineer_features(df)


@st.cache_resource(show_spinner="Loading models…")
def get_models():
    """Returns (per_category_dict, fallback_pipeline, q80_pipeline, meta)."""
    per_cat  = joblib.load(PER_CAT_PATH) if os.path.exists(PER_CAT_PATH) else {}
    fallback = joblib.load(APP_PATH)     if os.path.exists(APP_PATH)     else None
    q80      = joblib.load(Q80_PATH)     if os.path.exists(Q80_PATH)     else None
    meta     = joblib.load(META_PATH)    if os.path.exists(META_PATH)    else {}

    # Validate that the fallback model's expected features match DEMO_FEATURES.
    expected = meta.get("app_feature_columns")
    if expected and set(expected) != set(DEMO_FEATURES):
        missing = sorted(set(expected) - set(DEMO_FEATURES))
        extra   = sorted(set(DEMO_FEATURES) - set(expected))
        st.error(
            "Model / feature mismatch — re-run notebooks/Sales_Forecasting_CRISPML.ipynb "
            f"to regenerate artifacts.\n\nMissing in app: `{missing}`\nExtra in app: `{extra}`"
        )
        st.stop()

    if fallback is None:
        st.error(f"Required model not found at `{APP_PATH}`. Run the notebook first.")
        st.stop()

    return per_cat, fallback, q80, meta


def predict_demand(per_cat: dict, fallback, row_dict: dict):
    """Per-category routing: dispatch to the dedicated model for this Category if
    available; otherwise fall back to the App-aligned global model."""
    df = pd.DataFrame([row_dict])
    missing = [c for c in DEMO_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required model features: {missing}")
    X = df[DEMO_FEATURES]
    cat = row_dict.get("Category")
    if cat in per_cat:
        pred = float(per_cat[cat].predict(X)[0])
        route = "per-category"
    else:
        pred = float(fallback.predict(X)[0])
        route = "fallback (app-aligned)"
    return max(pred, 0.0), route


def predict_p80(q80_pipeline, row_dict: dict) -> float:
    """P80 quantile — stock-to target for newsvendor reorder advisory."""
    if q80_pipeline is None:
        return float("nan")
    df = pd.DataFrame([row_dict])
    # Q80 is trained on FEATURE_COLS_STAGE2 (includes lag/rolling). At inference
    # we have no history → fill those with the current inventory level as a
    # coarse proxy. Keeps the advisory directional rather than deceptively precise.
    expected = getattr(q80_pipeline.named_steps.get("pre", None), "feature_names_in_", None)
    if expected is not None:
        for col in expected:
            if col not in df.columns:
                df[col] = row_dict.get("Inventory Level", 0) if ("lag" in col or "roll" in col) else 0
        df = df[list(expected)]
    try:
        return float(q80_pipeline.predict(df)[0])
    except Exception:
        return float("nan")


def stock_status(inventory: int, predicted: float):
    coverage = inventory / max(predicted, 1)
    if coverage < 0.5:   return "CRITICAL — STOCKOUT RISK", DANGER, "danger", coverage
    elif coverage < 1.2: return "LOW STOCK", WARNING_COL, "warning", coverage
    else:                return "WELL STOCKED", SUCCESS, "success", coverage


def reorder_qty(predicted: float, inventory: int, safety_days: int = 7) -> int:
    buffer = predicted * safety_days
    shortage = max(0, buffer - inventory)
    return int(np.ceil(shortage / max(predicted, 1)) * max(predicted, 1))


# ─── Load Resources ───────────────────────────────────────────────────────────
df = load_data()
per_cat_models, fallback_model, q80_pipeline, meta = get_models()

stores      = sorted(df["Store ID"].unique())
categories  = sorted(df["Category"].unique())
regions     = sorted(df["Region"].unique())
weathers    = sorted(df["Weather Condition"].unique())
seasons     = sorted(df["Seasonality"].unique())

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div class='sidebar-brand'>
        <div class='sidebar-brand-mark'>Sales <span>Intel.</span></div>
        <div class='sidebar-brand-id'>v02 · MMXXVI · OP</div>
        <div style='margin-top:0.7rem;'><span class='sidebar-pill'><span class='pd'></span>Online</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"<div style='font-family:\"Fraunces\",serif; font-style:italic; font-size:0.95rem; color:{MIST}; margin-bottom:0.3rem;'>Dashboard filters</div>", unsafe_allow_html=True)
    sel_store    = st.selectbox("Store",    ["All"] + stores)
    sel_category = st.selectbox("Category", ["All"] + categories)

    st.markdown("<hr style='border:none;height:1px;background:rgba(255,255,255,0.08);margin:1.2rem 0'>", unsafe_allow_html=True)

    n_cat_models = len(per_cat_models)
    routing_label = f"PER-CATEGORY x{n_cat_models}" if n_cat_models else "FALLBACK ONLY"
    routing_color = SUCCESS if n_cat_models else WARNING_COL
    st.markdown(f"""
    <div style='font-family:"Fraunces",serif; font-style:italic; font-size:0.92rem; color:{MIST};'>Routing</div>
    <div style='font-family:Orbitron,monospace; font-size:0.95rem; color:{routing_color};
                font-weight:700; letter-spacing:0.06em; margin:0.4rem 0 0.3rem;'>● {routing_label}</div>
    <div style='font-family:"IBM Plex Mono",monospace; font-size:0.66rem; color:rgba(176,180,184,0.7) !important; line-height:1.55;'>
        Best: {meta.get("best_model", "Per-category App-aligned")}<br>
        Holdout MAE: {meta.get("metrics", {}).get("per_category", {}).get("MAE", float("nan")):.2f}
    </div>
    """, unsafe_allow_html=True)

    if not per_cat_models:
        st.markdown(f"""
        <div class='info-box' style='margin-top:0.8rem; font-size:0.68rem;'>
            Per-category models missing. Re-run notebook §4.9 to populate
            <code>model/sales/model_per_category.pkl</code>.
        </div>""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class='issue-strip'>
    <span><span class='dot'></span>LIVE · LAB 02 · SALES</span>
    <span>CRISP-ML(Q) · PER-CATEGORY ROUTING</span>
    <span>OSCAR PONCE · MMXXVI</span>
</div>
<div class='main-header'>Sales Demand <em>Intelligence</em></div>
<div class='sub-header'>Demand forecasting · 5 categories × 5 stores · routed LightGBM ensemble</div>
""", unsafe_allow_html=True)

# ─── KPI strip ────────────────────────────────────────────────────────────────
view_df = df.copy()
if sel_store    != "All": view_df = view_df[view_df["Store ID"] == sel_store]
if sel_category != "All": view_df = view_df[view_df["Category"] == sel_category]

avg_demand  = view_df["Demand"].mean()
avg_inv     = view_df["Inventory Level"].mean()
stockout_pct = (view_df["Units Sold"] < view_df["Demand"]).mean() * 100
epi_lift = view_df.groupby("Epidemic")["Demand"].mean()
epi_delta = (epi_lift.get(1, avg_demand) - epi_lift.get(0, avg_demand))
promo_lift = view_df.groupby("Promotion")["Demand"].mean()
promo_delta = ((promo_lift.get(1, avg_demand) / max(promo_lift.get(0, avg_demand), 1)) - 1) * 100

k1, k2, k3, k4, k5 = st.columns(5)
kpi_data = [
    (k1, "Avg Daily Demand",   f"{avg_demand:.0f}",  "units / day"),
    (k2, "Avg Inventory",      f"{avg_inv:.0f}",     "units on hand"),
    (k3, "Stockout Rate",      f"{stockout_pct:.0f}%", "Units Sold < Demand"),
    (k4, "Epidemic Effect",    f"{epi_delta:+.0f}",  "Δ units when active"),
    (k5, "Promotion Lift",     f"{promo_delta:+.1f}%", "vs. no-promo avg"),
]
for col, label, value, delta in kpi_data:
    col.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>{label}</div>
        <div class='metric-value'>{value}</div>
        <div class='metric-delta'>{delta}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ─── Main Tabs ────────────────────────────────────────────────────────────────
tab_sim, tab_dash, tab_reorder = st.tabs([
    "  Demand Simulator  ",
    "  Sales Dashboard  ",
    "  Reorder Advisory  ",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DEMAND SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════════
with tab_sim:
    st.markdown("<div class='section-header'>Configure Scenario</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-size:0.78rem; color:{PEBBLE}; margin-bottom:1rem;'>
        Predicts <b>Demand</b> (true demand, not censored by stock) via per-category
        routing. Each Category has its own LightGBM model — see the
        <span class='routing-pill'>routing badge</span> below the prediction.
    </div>""", unsafe_allow_html=True)

    form_col, result_col = st.columns([1.1, 0.9], gap="large")

    with form_col:
        # Card 1 — Context
        st.markdown(f"""<div class='input-card'><div class='input-card-header'>Context</div>""", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        store_in    = c1.selectbox("Store",    stores,    key="sim_store")
        category_in = c2.selectbox("Category", categories, key="sim_cat")
        region_in   = c3.selectbox("Region",   regions,   key="sim_region")
        st.markdown("</div>", unsafe_allow_html=True)

        # Card 2 — Time & Environment
        st.markdown(f"""<div class='input-card'><div class='input-card-header'>Time & Environment</div>""", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        month_in   = c1.selectbox("Month", list(range(1, 13)), index=0, key="sim_month")
        weekday_in = c2.selectbox("Weekday", ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], key="sim_wd")
        season_in  = c3.selectbox("Season",  seasons,  key="sim_season")
        weather_in = c4.selectbox("Weather", weathers, key="sim_weather")
        col_p, col_e = st.columns(2)
        promo_in   = col_p.checkbox("Promotion active", value=False, key="sim_promo")
        epidemic_in = col_e.checkbox("Epidemic active",  value=False, key="sim_epi")
        st.markdown("</div>", unsafe_allow_html=True)

        # Card 3 — Pricing & Stock
        st.markdown(f"""<div class='input-card'><div class='input-card-header'>Pricing & Inventory</div>""", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        price_in    = c1.slider("Price ($)",            5.0,  250.0, 75.0, 1.0, key="sim_price")
        discount_in = c2.slider("Discount (%)",         0,    25,    10,   5,   key="sim_disc")
        comp_in     = c3.slider("Competitor Price ($)", 5.0,  280.0, 80.0, 1.0, key="sim_comp")
        inv_in      = st.slider("Current Inventory (units)", 50, 800, 250, 10, key="sim_inv")
        st.markdown("</div>", unsafe_allow_html=True)

        predict_btn = st.button("Run Forecast", key="predict")

    with result_col:
        weekday_map = {"Mon":0,"Tue":1,"Wed":2,"Thu":3,"Fri":4,"Sat":5,"Sun":6}
        wd_num   = weekday_map[weekday_in]
        is_wknd  = 1 if wd_num >= 5 else 0
        # Pick a representative date in the chosen month for Fourier features.
        sample_date = pd.Timestamp(2024, month_in, 15)
        doy = sample_date.dayofyear
        sin_y, cos_y = np.sin(2*np.pi*doy/365.25), np.cos(2*np.pi*doy/365.25)

        row = {
            "Category":          category_in,
            "Region":            region_in,
            "Weather Condition": weather_in,
            "Seasonality":       season_in,
            "Inventory Level":   inv_in,
            "Price":             price_in,
            "Discount":          discount_in,
            "Competitor Pricing": comp_in,
            "Promotion":         int(promo_in),
            "Epidemic":          int(epidemic_in),
            "day_of_week":       wd_num,
            "month":             month_in,
            "is_weekend":        is_wknd,
            "sin_year":          sin_y,
            "cos_year":          cos_y,
        }

        pred, route = predict_demand(per_cat_models, fallback_model, row)
        pred_int = max(0, int(round(pred)))
        status, color, box_class, cov = stock_status(inv_in, pred_int)
        p80_pred = predict_p80(q80_pipeline, row)
        stock_to_target = int(round(p80_pred)) if not np.isnan(p80_pred) else pred_int
        reorder = reorder_qty(max(stock_to_target, pred_int), inv_in)

        st.markdown(f"""
        <div class='prediction-box {box_class}'>
            <div class='pred-stamp'>P80 · ROUTED v1</div>
            <div class='prediction-label'>Predicted daily demand</div>
            <div class='prediction-value' style='color:{color};'>{pred_int}</div>
            <div style='font-family:"IBM Plex Mono",monospace; font-size:0.68rem; margin-top:0.35rem; color:rgba(26,29,35,0.5); letter-spacing:0.1em; text-transform:uppercase;'>
                units · {category_in} · {store_in}
                &nbsp;&nbsp;<span class='routing-pill'>{route}</span>
            </div>
            <hr style='border:none;height:1px;background:rgba(26,29,35,0.12);margin:0.9rem 0 0.7rem;'>
            <div style='font-family:"IBM Plex Mono",monospace; font-size:0.78rem; line-height:1.75;'>
                <span style='font-family:"Fraunces",serif; font-style:italic; color:rgba(26,29,35,0.55);'>Stock status</span> &nbsp;
                <span style='color:{color}; font-weight:700; letter-spacing:0.08em; text-transform:uppercase; font-size:0.72rem;'>{status}</span><br>
                <span style='font-family:"Fraunces",serif; font-style:italic; color:rgba(26,29,35,0.55);'>Coverage</span> {cov:.1f}× &nbsp;·&nbsp;
                <span style='font-family:"Fraunces",serif; font-style:italic; color:rgba(26,29,35,0.55);'>Inventory</span> {inv_in} units<br>
                <span style='font-family:"Fraunces",serif; font-style:italic; color:rgba(26,29,35,0.55);'>P80 stock-to</span> {stock_to_target} units
                <span style='color:rgba(26,29,35,0.4); font-size:0.7rem;'>(newsvendor 80%)</span>
            </div>
            {"<div class='prediction-advisory'>▸ Recommended reorder: <b>" + str(reorder) + " units</b></div>" if reorder > 0 else "<div class='prediction-advisory' style='border-left-color:" + SUCCESS + "; color:" + SUCCESS + ";'>✓ No reorder needed</div>"}
        </div>
        """, unsafe_allow_html=True)

        # What-if: price sweep ±20%
        prices = np.linspace(price_in * 0.8, price_in * 1.2, 9)
        preds_price = []
        for p in prices:
            r2 = {**row, "Price": p}
            preds_price.append(predict_demand(per_cat_models, fallback_model, r2)[0])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=prices, y=preds_price, mode="lines+markers",
            line=dict(color=ESPRESSO_GOLD, width=2.5),
            marker=dict(size=6, color=ESPRESSO_GOLD),
            name="Predicted Demand",
        ))
        fig.add_vline(x=price_in, line_dash="dot", line_color=GRAPHITE, line_width=1.5)
        fig.update_layout(
            title=dict(text="Demand vs. Price Sensitivity", font_family="Orbitron",
                       font_size=12, font_color=GRAPHITE_DEEP),
            xaxis_title="Price ($)", yaxis_title="Demand (units)",
            height=220, margin=dict(t=40, b=30, l=30, r=10),
            plot_bgcolor="white", paper_bgcolor="white",
            font_family="IBM Plex Mono",
            xaxis=dict(gridcolor=SILVER), yaxis=dict(gridcolor=SILVER),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Epidemic / Promotion factor effect
        scenarios = []
        for promo, epi, label in [(0,0,"Base"), (1,0,"Promo only"), (0,1,"Epidemic only"), (1,1,"Promo + Epi")]:
            r3 = {**row, "Promotion": promo, "Epidemic": epi}
            scenarios.append((label, predict_demand(per_cat_models, fallback_model, r3)[0]))
        fig2 = go.Figure(go.Bar(
            x=[s[0] for s in scenarios], y=[s[1] for s in scenarios],
            marker_color=[MIST, ESPRESSO_GOLD, DANGER, GRAPHITE_DEEP],
            text=[f"{s[1]:.0f}" for s in scenarios],
            textposition="outside",
            textfont=dict(family="Orbitron", size=11),
        ))
        fig2.update_layout(
            title=dict(text="Promo × Epidemic Scenarios", font_family="Orbitron",
                       font_size=12, font_color=GRAPHITE_DEEP),
            height=200, margin=dict(t=40, b=20, l=30, r=10),
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(gridcolor=SILVER, showgrid=True),
            xaxis=dict(gridcolor="rgba(0,0,0,0)"),
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SALES DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
with tab_dash:
    st.markdown("<div class='section-header'>Demand Trends</div>", unsafe_allow_html=True)
    dash_df = view_df.copy()
    dash_df["Date"] = pd.to_datetime(dash_df["Date"])

    trend = dash_df.groupby(["Date", "Category"])["Demand"].mean().reset_index()
    trend_weekly = (
        trend.set_index("Date").groupby("Category")["Demand"].resample("W").mean().reset_index()
    )
    fig3 = px.line(
        trend_weekly, x="Date", y="Demand", color="Category",
        color_discrete_sequence=[ESPRESSO_GOLD, INFO, SUCCESS, DANGER, PEBBLE],
    )
    fig3.update_layout(
        height=320, plot_bgcolor="white", paper_bgcolor="white",
        font_family="IBM Plex Mono",
        xaxis=dict(gridcolor=SILVER, title=None),
        yaxis=dict(gridcolor=SILVER, title="Demand"),
        legend=dict(orientation="h", yanchor="top", y=-0.18, x=0.5, xanchor="center", font=dict(size=10)),
        margin=dict(t=15, b=55, l=10, r=10),
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("<div class='fig-caption'><b>Fig. 01</b>True demand (uncensored) by category, weekly-resampled. Promotion and epidemic windows visible as deviations.</div>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("<div class='section-header'>Demand by Season & Weather</div>", unsafe_allow_html=True)
        hm = dash_df.groupby(["Seasonality", "Weather Condition"])["Demand"].mean().unstack(fill_value=0)
        fig4 = px.imshow(
            hm,
            color_continuous_scale=[[0, PLATINUM], [0.5, ESPRESSO_GOLD], [1, GRAPHITE_DEEP]],
            title="Mean Demand",
            text_auto=".0f",
        )
        fig4.update_layout(
            height=270, plot_bgcolor="white", paper_bgcolor="white",
            font_family="IBM Plex Mono",
            title_font_family="Orbitron", title_font_size=12, title_font_color=GRAPHITE_DEEP,
            margin=dict(t=40, b=10, l=10, r=10),
        )
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown("<div class='fig-caption'><b>Fig. 02</b>Mean true demand across seasonality × weather — darker cells mark peak conditions.</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown("<div class='section-header'>Stockouts vs Demand (Lost Sales)</div>", unsafe_allow_html=True)
        stockout_view = dash_df.copy()
        stockout_view["Lost Sales"] = (stockout_view["Demand"] - stockout_view["Units Sold"]).clip(lower=0)
        store_loss = stockout_view.groupby("Store ID").agg(
            Demand=("Demand", "mean"),
            Lost=("Lost Sales", "mean"),
        ).reset_index()

        fig5 = go.Figure()
        fig5.add_trace(go.Bar(
            name="Met Demand", x=store_loss["Store ID"],
            y=(store_loss["Demand"] - store_loss["Lost"]),
            marker_color=ESPRESSO_GOLD,
        ))
        fig5.add_trace(go.Bar(
            name="Lost (stockout)", x=store_loss["Store ID"],
            y=store_loss["Lost"],
            marker_color=DANGER,
        ))
        fig5.update_layout(
            barmode="stack", height=300, plot_bgcolor="white", paper_bgcolor="white",
            font_family="IBM Plex Mono", bargap=0.55,
            xaxis=dict(gridcolor="rgba(0,0,0,0)", title=None),
            yaxis=dict(gridcolor=SILVER),
            legend=dict(orientation="h", yanchor="top", y=-0.18, x=0.5, xanchor="center",
                        font=dict(size=10)),
            margin=dict(t=15, b=55, l=10, r=10),
        )
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown("<div class='fig-caption'><b>Fig. 03</b>Demand stack per store — coral segments quantify daily revenue lost to stockouts.</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Per-Category MAE — Why Routing</div>", unsafe_allow_html=True)
    metrics = meta.get("metrics", {}) or {}
    global_mae   = metrics.get("app_aligned", {}).get("MAE", float("nan"))
    per_cat_mae  = metrics.get("per_category", {}).get("MAE", float("nan"))
    st.markdown(f"""
    <div class='info-box'>
        Per-category routing reduces holdout MAE from <b>{global_mae:.2f}</b> (single global model)
        to <b>{per_cat_mae:.2f}</b> — a {((1 - per_cat_mae/global_mae)*100):.1f}% improvement.
        The gain comes mostly from Clothing and Electronics, not Groceries (despite Groceries having
        the worst absolute MAE — see notebook §4.9 for the surprising breakdown).
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — REORDER ADVISORY
# ═══════════════════════════════════════════════════════════════════════════════
with tab_reorder:
    st.markdown("<div class='section-header'>Reorder Advisory Engine</div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-size:0.78rem; color:{PEBBLE}; margin-bottom:1rem;'>
        Per-store × category reorder recommendations. Demand baseline = 30-day median;
        safety buffer in days is the dial. Reorder quantity targets coverage of demand
        for the selected safety period.
    </div>""", unsafe_allow_html=True)

    safety_days = st.slider("Safety Buffer (days)", 3, 14, 7, key="safety")

    recent_df = df.copy()
    recent_df["Date"] = pd.to_datetime(recent_df["Date"])
    cutoff = recent_df["Date"].max() - pd.Timedelta(days=30)
    recent = recent_df[recent_df["Date"] >= cutoff]

    advisory = (
        recent.groupby(["Store ID", "Category"])
        .agg(
            Avg_Daily_Demand=("Demand", "median"),
            Current_Inventory=("Inventory Level", "last"),
            Avg_Stockout_Rate=("Units Sold", lambda s: (s < recent.loc[s.index, "Demand"]).mean() * 100),
        )
        .reset_index()
    )
    advisory["Days_of_Supply"] = (advisory["Current_Inventory"] / advisory["Avg_Daily_Demand"].clip(lower=1)).round(1)
    advisory["Reorder_Qty"] = advisory.apply(
        lambda r: reorder_qty(r["Avg_Daily_Demand"], int(r["Current_Inventory"]), safety_days), axis=1
    )
    advisory["Risk"] = advisory["Days_of_Supply"].apply(
        lambda d: "CRITICAL" if d < safety_days * 0.5 else ("LOW STOCK" if d < safety_days else "OK")
    )

    risk_filter = st.selectbox("Filter by Risk Level", ["All", "CRITICAL", "LOW STOCK", "OK"])
    if risk_filter != "All":
        advisory = advisory[advisory["Risk"] == risk_filter]
    advisory_sorted = advisory.sort_values("Days_of_Supply")

    st.dataframe(
        advisory_sorted.rename(columns={
            "Store ID":            "Store",
            "Avg_Daily_Demand":    "Demand/Day",
            "Current_Inventory":   "Stock On Hand",
            "Days_of_Supply":      "Days of Supply",
            "Reorder_Qty":         "Reorder Qty",
            "Avg_Stockout_Rate":   "Stockout %",
        }),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Store":           st.column_config.TextColumn(width="small"),
            "Category":        st.column_config.TextColumn(width="medium"),
            "Demand/Day":      st.column_config.NumberColumn(format="%.1f"),
            "Stock On Hand":   st.column_config.NumberColumn(format="%d"),
            "Days of Supply":  st.column_config.NumberColumn(format="%.1f"),
            "Reorder Qty":     st.column_config.NumberColumn(format="%d"),
            "Stockout %":      st.column_config.NumberColumn(format="%.0f%%"),
            "Risk":            st.column_config.TextColumn(width="medium"),
        },
    )

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        st.markdown("<div class='section-header'>Risk Distribution</div>", unsafe_allow_html=True)
        risk_counts = advisory["Risk"].value_counts()
        fig7 = go.Figure(go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            marker=dict(colors=[DANGER if r == "CRITICAL" else (WARNING_COL if r == "LOW STOCK" else SUCCESS)
                                for r in risk_counts.index]),
            textinfo="label+percent",
            textfont=dict(family="IBM Plex Mono", size=11),
            hole=0.5,
        ))
        fig7.update_layout(
            height=260, plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=10, b=10, l=10, r=10),
            showlegend=False,
        )
        st.plotly_chart(fig7, use_container_width=True)

    with col_r2:
        st.markdown("<div class='section-header'>Reorder Volume by Category</div>", unsafe_allow_html=True)
        cat_reorder = advisory.groupby("Category")["Reorder_Qty"].sum().sort_values(ascending=True)
        fig8 = go.Figure(go.Bar(
            y=cat_reorder.index, x=cat_reorder.values, orientation="h",
            marker_color=ESPRESSO_GOLD,
            text=cat_reorder.values, textposition="outside",
            textfont=dict(family="Orbitron", size=11),
        ))
        fig8.update_layout(
            height=260, plot_bgcolor="white", paper_bgcolor="white",
            font_family="IBM Plex Mono",
            xaxis=dict(gridcolor=SILVER),
            yaxis=dict(gridcolor="rgba(0,0,0,0)"),
            margin=dict(t=10, b=10, l=10, r=30),
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
    &nbsp;·&nbsp; Per-category LightGBM ensemble &nbsp;·&nbsp; Holdout MAE {meta.get("metrics", {}).get("per_category", {}).get("MAE", float("nan")):.2f}</span>
    <span><a href='https://oscarponce.com' style='color:{ESPRESSO_GOLD} !important; text-decoration:none; border-bottom:1px solid {ESPRESSO_GOLD};'>oscarponce.com</a> &nbsp;·&nbsp; MMXXVI</span>
</div>
""", unsafe_allow_html=True)
