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

# ─── CSS (matches app.py) ─────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=IBM+Plex+Mono:wght@300;400;600&display=swap');

#MainMenu, footer, header {{ visibility: hidden; }}

html, body, [data-testid="stAppViewContainer"] {{
    background: linear-gradient(135deg, {PLATINUM} 0%, {ICE_SILVER} 50%, {SILVER} 100%);
    min-height: 100vh;
    font-family: 'IBM Plex Mono', monospace;
}}

[data-testid="stSidebar"] {{
    background: {GRAPHITE} !important;
    border-right: 1px solid {ESPRESSO_GOLD}44;
}}
[data-testid="stSidebar"] * {{ color: {ICE_SILVER} !important; }}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label {{ color: {MIST} !important; font-size: 0.75rem; }}

.main-header {{
    font-family: 'Orbitron', monospace;
    font-size: 2.4rem; font-weight: 900;
    color: {GRAPHITE_DEEP};
    letter-spacing: 0.08em; text-transform: uppercase;
    margin-bottom: 0.1rem;
}}
.sub-header {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem; color: {PEBBLE};
    letter-spacing: 0.15em; text-transform: uppercase;
    margin-bottom: 1.5rem;
}}
.section-header {{
    font-family: 'Orbitron', monospace;
    font-size: 1.15rem; font-weight: 700;
    color: {GRAPHITE_DEEP};
    letter-spacing: 0.06em; text-transform: uppercase;
    margin: 1rem 0 0.5rem;
}}

.metric-card {{
    background: white; border: 1px solid {SILVER};
    border-top: 3px solid {ESPRESSO_GOLD};
    border-radius: 8px; padding: 1rem 1.2rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}}
.metric-card:hover {{ transform: translateY(-2px); box-shadow: 0 6px 16px rgba(0,0,0,0.10); }}
.metric-label  {{ font-family:'IBM Plex Mono',monospace; font-size:0.7rem; color:{PEBBLE}; text-transform:uppercase; letter-spacing:0.12em; }}
.metric-value  {{ font-family:'Orbitron',monospace;     font-size:1.8rem; font-weight:700; color:{GRAPHITE_DEEP}; margin:0.15rem 0; }}
.metric-delta  {{ font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:{PEBBLE}; }}

.prediction-box {{ border-radius:10px; padding:1.5rem 2rem; margin:1rem 0; border-left:5px solid; }}
.prediction-box.success {{ background:{SUCCESS}18; border-color:{SUCCESS}; color:{GRAPHITE}; }}
.prediction-box.warning {{ background:{WARNING_COL}20; border-color:{WARNING_COL}; color:{GRAPHITE}; }}
.prediction-box.danger  {{ background:{DANGER}18;  border-color:{DANGER};  color:{GRAPHITE}; }}
.prediction-value {{ font-family:'Orbitron',monospace; font-size:3rem; font-weight:900; line-height:1; }}
.prediction-label {{ font-family:'IBM Plex Mono',monospace; font-size:0.75rem; letter-spacing:0.15em; text-transform:uppercase; opacity:0.75; margin-bottom:0.3rem; }}
.prediction-advisory {{ font-family:'IBM Plex Mono',monospace; font-size:0.85rem; margin-top:0.6rem; font-weight:600; }}

.input-card {{ background:white; border:1px solid {SILVER}; border-radius:8px; padding:1.2rem 1.4rem; margin-bottom:0.8rem; box-shadow:0 1px 4px rgba(0,0,0,0.05); }}
.input-card-header {{ font-family:'Orbitron',monospace; font-size:0.75rem; font-weight:700; color:{ESPRESSO_GOLD}; text-transform:uppercase; letter-spacing:0.12em; border-bottom:1px solid {ESPRESSO_GOLD}44; padding-bottom:0.4rem; margin-bottom:0.8rem; }}

.info-box {{ background:{INFO}12; border-left:4px solid {INFO}; border-radius:0 6px 6px 0; padding:0.8rem 1rem; font-family:'IBM Plex Mono',monospace; font-size:0.78rem; color:{SLATE}; margin:0.5rem 0; }}
.routing-pill {{ display:inline-block; background:{ESPRESSO_GOLD}22; color:{GRAPHITE_DEEP}; padding:0.15rem 0.6rem; border-radius:12px; font-family:'IBM Plex Mono',monospace; font-size:0.7rem; letter-spacing:0.08em; }}

hr {{ border:none; height:1px; background:linear-gradient(90deg, transparent, {ESPRESSO_GOLD}, transparent); margin:1.2rem 0; }}

.stTabs [data-baseweb="tab-list"] {{
    gap:6px; background:rgba(42, 48, 56, 0.04); border-radius:10px;
    padding:6px; border-bottom:1px solid {SILVER};
}}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{
    background:transparent !important; height:0 !important;
}}
.stTabs [data-baseweb="tab"] {{
    font-family:'IBM Plex Mono',monospace; font-size:0.78rem; letter-spacing:0.12em;
    text-transform:uppercase; border-radius:7px; color:{SLATE};
    padding:0.55rem 1.4rem; transition:all 0.15s ease; background:transparent;
}}
.stTabs [data-baseweb="tab"]:hover {{
    background:rgba(201, 168, 106, 0.12); color:{GRAPHITE};
}}
.stTabs [aria-selected="true"] {{
    background:linear-gradient(135deg, {ESPRESSO_GOLD} 0%, #B89858 100%) !important;
    color:{GRAPHITE_DEEP} !important; font-weight:700 !important;
    box-shadow:0 2px 6px rgba(36, 3, 56, 0.18);
}}

[data-testid="stButton"] > button {{
    background:{ESPRESSO_GOLD}; color:{GRAPHITE};
    font-family:'Orbitron',monospace; font-size:0.8rem; font-weight:700;
    letter-spacing:0.1em; text-transform:uppercase;
    border:none; border-radius:6px; padding:0.6rem 2rem; width:100%;
    transition: background 0.15s ease;
}}
[data-testid="stButton"] > button:hover {{ background:{GRAPHITE_DEEP}; color:{ESPRESSO_GOLD}; }}
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
    <div style='text-align:center; padding: 1rem 0 0.5rem;'>
        <div style='font-family:Orbitron,monospace; font-size:1.1rem; font-weight:900;
                    color:{ESPRESSO_GOLD}; letter-spacing:0.1em;'>🛒 SALES INTEL</div>
        <div style='font-family:"IBM Plex Mono",monospace; font-size:0.65rem;
                    color:{MIST}; letter-spacing:0.12em; margin-top:0.2rem;'>
            CRISP-ML(Q) · Oscar Ponce
        </div>
    </div>
    <hr style='border:none; height:1px; background:{ESPRESSO_GOLD}44; margin:0.8rem 0;'>
    """, unsafe_allow_html=True)

    st.markdown(f"<div style='font-size:0.7rem; color:{MIST}; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:0.3rem;'>Dashboard Filters</div>", unsafe_allow_html=True)
    sel_store    = st.selectbox("Store",    ["All"] + stores)
    sel_category = st.selectbox("Category", ["All"] + categories)

    st.markdown("<hr style='border:none;height:1px;background:#ffffff22;margin:1rem 0'>", unsafe_allow_html=True)

    n_cat_models = len(per_cat_models)
    routing_label = f"PER-CATEGORY x{n_cat_models}" if n_cat_models else "FALLBACK ONLY"
    routing_color = SUCCESS if n_cat_models else WARNING_COL
    st.markdown(f"""
    <div style='font-size:0.7rem; color:{MIST}; letter-spacing:0.1em; text-transform:uppercase;'>Routing</div>
    <div style='font-family:Orbitron,monospace; font-size:0.85rem; color:{routing_color};
                font-weight:700; letter-spacing:0.08em; margin:0.3rem 0;'>● {routing_label}</div>
    <div style='font-size:0.68rem; color:{MIST}; line-height:1.5;'>
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
<div class='main-header'>Sales Demand Intelligence</div>
<div class='sub-header'>Demand Forecasting · CRISP-ML(Q) Pipeline · Oscar Ponce</div>
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
            <div class='prediction-label'>Predicted Daily Demand</div>
            <div class='prediction-value' style='color:{color};'>{pred_int}</div>
            <div style='font-family:"IBM Plex Mono",monospace; font-size:0.72rem; margin-top:0.2rem; opacity:0.7;'>
                units · {category_in} · {store_in} &nbsp;
                <span class='routing-pill'>{route}</span>
            </div>
            <hr style='border:none;height:1px;background:{color}44;margin:0.7rem 0;'>
            <div style='font-family:"IBM Plex Mono",monospace; font-size:0.8rem;'>
                <b>Stock Status:</b> <span style='color:{color}; font-weight:700;'>{status}</span><br>
                <b>Coverage:</b> {cov:.1f}x demand &nbsp;|&nbsp; <b>Inventory:</b> {inv_in} units<br>
                <b>P80 stock-to:</b> {stock_to_target} units (newsvendor 80% service level)
            </div>
            {"<div class='prediction-advisory'>⚠ Recommended Reorder: <b>" + str(reorder) + " units</b></div>" if reorder > 0 else "<div class='prediction-advisory' style='color:" + SUCCESS + ";'>✓ No reorder needed</div>"}
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
        title="Weekly Average Demand by Category",
        color_discrete_sequence=[ESPRESSO_GOLD, INFO, SUCCESS, DANGER, PEBBLE],
    )
    fig3.update_layout(
        height=320, plot_bgcolor="white", paper_bgcolor="white",
        font_family="IBM Plex Mono",
        title_font_family="Orbitron", title_font_size=13, title_font_color=GRAPHITE_DEEP,
        xaxis=dict(gridcolor=SILVER), yaxis=dict(gridcolor=SILVER),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig3, use_container_width=True)

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
            title=dict(text="Per-Store: Daily Demand Met vs Lost",
                       font_family="Orbitron", font_size=12, font_color=GRAPHITE_DEEP,
                       x=0.02, xanchor="left"),
            xaxis=dict(gridcolor="rgba(0,0,0,0)"),
            yaxis=dict(gridcolor=SILVER),
            legend=dict(orientation="h", yanchor="top", y=-0.15, x=0.5, xanchor="center",
                        font=dict(size=10)),
            margin=dict(t=40, b=55, l=10, r=10),
        )
        st.plotly_chart(fig5, use_container_width=True)

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
