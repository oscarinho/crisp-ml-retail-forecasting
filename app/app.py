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

# ─── CSS ──────────────────────────────────────────────────────────────────────
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
    font-size: 2.4rem;
    font-weight: 900;
    color: {GRAPHITE_DEEP};
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.1rem;
}}

.sub-header {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    color: {PEBBLE};
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 1.5rem;
}}

.section-header {{
    font-family: 'Orbitron', monospace;
    font-size: 1.15rem;
    font-weight: 700;
    color: {GRAPHITE_DEEP};
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin: 1rem 0 0.5rem;
}}

.metric-card {{
    background: white;
    border: 1px solid {SILVER};
    border-top: 3px solid {ESPRESSO_GOLD};
    border-radius: 8px;
    padding: 1rem 1.2rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}}
.metric-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(0,0,0,0.10);
}}
.metric-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: {PEBBLE};
    text-transform: uppercase;
    letter-spacing: 0.12em;
}}
.metric-value {{
    font-family: 'Orbitron', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: {GRAPHITE_DEEP};
    margin: 0.15rem 0;
}}
.metric-delta {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: {PEBBLE};
}}

.prediction-box {{
    border-radius: 10px;
    padding: 1.5rem 2rem;
    margin: 1rem 0;
    border-left: 5px solid;
}}
.prediction-box.success {{
    background: {SUCCESS}18;
    border-color: {SUCCESS};
    color: {GRAPHITE};
}}
.prediction-box.warning {{
    background: {WARNING_COL}20;
    border-color: {WARNING_COL};
    color: {GRAPHITE};
}}
.prediction-box.danger {{
    background: {DANGER}18;
    border-color: {DANGER};
    color: {GRAPHITE};
}}

.prediction-value {{
    font-family: 'Orbitron', monospace;
    font-size: 3rem;
    font-weight: 900;
    line-height: 1;
}}
.prediction-label {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    opacity: 0.75;
    margin-bottom: 0.3rem;
}}
.prediction-advisory {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    margin-top: 0.6rem;
    font-weight: 600;
}}

.input-card {{
    background: white;
    border: 1px solid {SILVER};
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.8rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}}
.input-card-header {{
    font-family: 'Orbitron', monospace;
    font-size: 0.75rem;
    font-weight: 700;
    color: {ESPRESSO_GOLD};
    text-transform: uppercase;
    letter-spacing: 0.12em;
    border-bottom: 1px solid {ESPRESSO_GOLD}44;
    padding-bottom: 0.4rem;
    margin-bottom: 0.8rem;
}}

.info-box {{
    background: {INFO}12;
    border-left: 4px solid {INFO};
    border-radius: 0 6px 6px 0;
    padding: 0.8rem 1rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: {SLATE};
    margin: 0.5rem 0;
}}

.stockout-row-high {{ background: {DANGER}15 !important; }}
.stockout-row-medium {{ background: {WARNING_COL}15 !important; }}

hr {{
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, {ESPRESSO_GOLD}, transparent);
    margin: 1.2rem 0;
}}

.stTabs [data-baseweb="tab-list"] {{
    gap: 6px;
    background: rgba(42, 48, 56, 0.04);
    border-radius: 10px;
    padding: 6px;
    border-bottom: 1px solid {SILVER};
}}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{
    background: transparent !important;
    height: 0 !important;
}}
.stTabs [data-baseweb="tab"] {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    border-radius: 7px;
    color: {SLATE};
    padding: 0.55rem 1.4rem;
    transition: all 0.15s ease;
    background: transparent;
}}
.stTabs [data-baseweb="tab"]:hover {{
    background: rgba(201, 168, 106, 0.12);
    color: {GRAPHITE};
}}
.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, {ESPRESSO_GOLD} 0%, #B89858 100%) !important;
    color: {GRAPHITE_DEEP} !important;
    font-weight: 700 !important;
    box-shadow: 0 2px 6px rgba(36, 3, 56, 0.18);
}}

[data-testid="stButton"] > button {{
    background: {ESPRESSO_GOLD};
    color: {GRAPHITE};
    font-family: 'Orbitron', monospace;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    border: none;
    border-radius: 6px;
    padding: 0.6rem 2rem;
    width: 100%;
    transition: background 0.15s ease;
}}
[data-testid="stButton"] > button:hover {{
    background: {GRAPHITE_DEEP};
    color: {ESPRESSO_GOLD};
}}
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
    <div style='text-align:center; padding: 1rem 0 0.5rem;'>
        <div style='font-family:Orbitron,monospace; font-size:1.1rem; font-weight:900;
                    color:{ESPRESSO_GOLD}; letter-spacing:0.1em;'>📦 DEMAND INTEL</div>
        <div style='font-family:"IBM Plex Mono",monospace; font-size:0.65rem;
                    color:{MIST}; letter-spacing:0.12em; margin-top:0.2rem;'>
            CRISP-ML(Q) · Oscar Ponce
        </div>
    </div>
    <hr style='border:none; height:1px; background:{ESPRESSO_GOLD}44; margin:0.8rem 0;'>
    """, unsafe_allow_html=True)

    st.markdown(f"<div style='font-size:0.7rem; color:{MIST}; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:0.3rem;'>Dashboard Filters</div>", unsafe_allow_html=True)
    sel_store    = st.selectbox("Store", ["All"] + stores)
    sel_category = st.selectbox("Category", ["All"] + categories)

    st.markdown("<hr style='border:none;height:1px;background:#ffffff22;margin:1rem 0'>", unsafe_allow_html=True)

    model_label = "DEMO MODEL" if is_demo else "FULL MODEL"
    model_color = WARNING_COL if is_demo else SUCCESS
    st.markdown(f"""
    <div style='font-size:0.7rem; color:{MIST}; letter-spacing:0.1em; text-transform:uppercase;'>Model Status</div>
    <div style='font-family:Orbitron,monospace; font-size:0.85rem; color:{model_color};
                font-weight:700; letter-spacing:0.08em; margin:0.3rem 0;'>● {model_label}</div>
    <div style='font-size:0.68rem; color:{MIST}; line-height:1.5;'>{meta.get("model_name","—")}</div>
    """, unsafe_allow_html=True)

    if is_demo:
        st.markdown(f"""
        <div class='info-box' style='margin-top:0.8rem; font-size:0.68rem;'>
            Run the notebook to train the full time-series model with lag features and replace this demo.
        </div>
        """, unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class='main-header'>Retail Demand Intelligence</div>
<div class='sub-header'>Inventory Forecasting · CRISP-ML(Q) Pipeline · Oscar Ponce</div>
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
            <div class='prediction-label'>Predicted Daily Demand</div>
            <div class='prediction-value' style='color:{color};'>{pred_int}</div>
            <div style='font-family:"IBM Plex Mono",monospace; font-size:0.72rem; margin-top:0.2rem; opacity:0.7;'>units · {category_in} · {store_in}</div>
            <hr style='border:none;height:1px;background:{color}44;margin:0.7rem 0;'>
            <div style='font-family:"IBM Plex Mono",monospace; font-size:0.8rem;'>
                <b>Stock Status:</b> <span style='color:{color}; font-weight:700;'>{status}</span><br>
                <b>Coverage:</b> {cov:.1f}x demand &nbsp;|&nbsp; <b>Inventory:</b> {inv_in} units
            </div>
            {"<div class='prediction-advisory'>⚠ Recommended Reorder: <b>" + str(reorder) + " units</b></div>" if reorder > 0 else "<div class='prediction-advisory' style='color:" + SUCCESS + ";'>✓ No reorder needed</div>"}
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
            title=dict(text="Store-Level Comparison", font_family="Orbitron",
                       font_size=12, font_color=GRAPHITE_DEEP, x=0.02, xanchor="left"),
            xaxis=dict(gridcolor="rgba(0,0,0,0)"),
            yaxis=dict(gridcolor=SILVER),
            legend=dict(orientation="h", yanchor="top", y=-0.15, x=0.5, xanchor="center",
                        font=dict(size=10)),
            margin=dict(t=40, b=55, l=10, r=10),
        )
        st.plotly_chart(fig5, use_container_width=True)

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

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(f"""
<div style='text-align:center; font-family:"IBM Plex Mono",monospace; font-size:0.7rem; color:{PEBBLE}; padding:0.5rem 0 1.5rem;'>
    Built with CRISP-ML(Q) methodology ·
    <a href='https://oscarponce.com' style='color:{ESPRESSO_GOLD}; text-decoration:none;'>oscarponce.com</a>
    &nbsp;·&nbsp; 73,100 retail records · 5 stores · 20 products · 2 years
</div>
""", unsafe_allow_html=True)
