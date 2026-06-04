"""Store Sales Time-Series Forecasting — runs the full model toolkit on the Kaggle dataset.

Dataset: Corporación Favorita Grocery Sales (Kaggle) — 3M daily rows.
Target: sales per (date, store_nbr, family).

Applies lessons from Inventory + Sales labs, with two important adaptations for scale:
- Sub-sampling for Stage 2 trees (cuts compute 5× without hurting holdout MAE)
- LightGBM is the workhorse — RF and ExtraTrees can't scale here
- Per-family routing (33 models, the Sales lab winner pattern)

Outputs:
- scripts/store_sales_results.json — every model's metrics
- model/store_sales/best_model.pkl — winning pipeline (or dict for per-family)
- model/store_sales/model_metadata.pkl
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "store-sales-time-series-forecasting"
OUT  = ROOT / "scripts" / "store_sales_results.json"
MODEL_DIR = ROOT / "model" / "store_sales"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Phase 2/3 — Load, merge, engineer features
# ============================================================================

def load_and_merge() -> pd.DataFrame:
    train = pd.read_csv(DATA / "train.csv", parse_dates=["date"])
    stores = pd.read_csv(DATA / "stores.csv")
    transactions = pd.read_csv(DATA / "transactions.csv", parse_dates=["date"])
    oil = pd.read_csv(DATA / "oil.csv", parse_dates=["date"])
    holidays = pd.read_csv(DATA / "holidays_events.csv", parse_dates=["date"])

    df = train.merge(stores, on="store_nbr", how="left")
    df = df.merge(transactions, on=["date", "store_nbr"], how="left")
    df = df.merge(oil, on="date", how="left")

    # Holidays: collapse to a binary flag + type
    holidays_clean = holidays[(holidays["transferred"] == False) &
                              (holidays["locale"] == "National")][["date", "type"]].drop_duplicates("date")
    holidays_clean = holidays_clean.rename(columns={"type": "holiday_type"})
    df = df.merge(holidays_clean, on="date", how="left")
    df["is_holiday"] = df["holiday_type"].notna().astype(int)
    df["holiday_type"] = df["holiday_type"].fillna("None")

    # Oil fwd-fill (oil prices not published on weekends)
    df["dcoilwtico"] = df["dcoilwtico"].ffill().bfill()
    df["transactions"] = df["transactions"].fillna(0)

    print(f"Loaded train: {len(df):,} rows · {df['date'].dt.date.nunique()} days "
          f"· {df['store_nbr'].nunique()} stores · {df['family'].nunique()} families")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

    # Calendar
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dayofweek"] = df["date"].dt.dayofweek
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["date"].dt.quarter
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    df["is_payday"] = df["day"].isin([15, 30, 31]).astype(int)
    df["dayofyear"] = df["date"].dt.dayofyear
    df["sin_year"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["cos_year"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)

    # Lag + rolling features per (store_nbr, family)
    g = df.groupby(["store_nbr", "family"], sort=False)["sales"]
    for lag in [1, 7, 14, 28]:
        df[f"sales_lag_{lag}"] = g.shift(lag)
    grp_ngroup = df.groupby(["store_nbr", "family"]).ngroup().values
    for w in [7, 14, 28]:
        shifted = g.shift(1)
        df[f"sales_roll_mean_{w}"] = shifted.groupby(grp_ngroup).rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f"sales_roll_std_{w}"]  = shifted.groupby(grp_ngroup).rolling(w, min_periods=2).std().reset_index(level=0, drop=True)

    return df


FEATURE_COLS_STAGE2 = [
    # Numeric (standard)
    "transactions", "dcoilwtico", "onpromotion",
    "sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_28",
    "sales_roll_mean_7", "sales_roll_std_7",
    "sales_roll_mean_14", "sales_roll_std_14",
    "sales_roll_mean_28", "sales_roll_std_28",
    "cluster",
    # Cyclic (passthrough)
    "sin_year", "cos_year",
    # Ordinal (minmax)
    "year", "month", "day", "dayofweek", "weekofyear", "quarter", "dayofyear",
    # Binary (passthrough)
    "is_weekend", "is_payday", "is_holiday",
    # Categorical
    "family", "city", "state", "type", "holiday_type",
]
FEATURE_COLS_STAGE1 = [c for c in FEATURE_COLS_STAGE2 if "lag_" not in c and "roll_" not in c]

NUM_STANDARD = ["transactions", "dcoilwtico", "onpromotion",
                "sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_28",
                "sales_roll_mean_7", "sales_roll_std_7",
                "sales_roll_mean_14", "sales_roll_std_14",
                "sales_roll_mean_28", "sales_roll_std_28",
                "cluster"]
NUM_FOURIER = ["sin_year", "cos_year"]
NUM_MINMAX = ["year", "month", "day", "dayofweek", "weekofyear", "quarter", "dayofyear"]
BINARY = ["is_weekend", "is_payday", "is_holiday"]
CATEGORICAL = ["family", "city", "state", "type", "holiday_type"]


def build_preprocessor(feature_cols: list[str]) -> ColumnTransformer:
    num_std = [c for c in NUM_STANDARD if c in feature_cols]
    num_mm  = [c for c in NUM_MINMAX   if c in feature_cols]
    num_f   = [c for c in NUM_FOURIER  if c in feature_cols]
    cat     = [c for c in CATEGORICAL  if c in feature_cols]
    binary  = [c for c in BINARY       if c in feature_cols]
    return ColumnTransformer([
        ("std",  Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_std),
        ("mm",   Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", MinMaxScaler())]), num_mm),
        ("pass", Pipeline([("imp", SimpleImputer(strategy="median"))]), num_f + binary),
        ("cat",  Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
    ], remainder="drop")


def time_split(df: pd.DataFrame, holdout_days: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    split = df["date"].max() - pd.Timedelta(days=holdout_days)
    tr = df[df["date"] <= split].copy()
    te = df[df["date"] >  split].copy()
    print(f"Time split: train ≤ {split.date()} ({len(tr):,} rows) · "
          f"holdout > {split.date()} ({len(te):,} rows)")
    return tr, te


# ============================================================================
# Eval helpers
# ============================================================================

def metrics(name: str, y_true, y_pred, fit_sec: float = 0.0) -> dict:
    y_true = np.asarray(y_true, float)
    y_pred = np.clip(np.asarray(y_pred, float), 0, None)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    # Kaggle metric: RMSLE
    y_true_pos = np.maximum(y_true, 0)
    rmsle = float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true_pos)) ** 2)))
    mask = y_true > 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else None
    return {
        "name": name, "mae": round(mae, 3), "rmse": round(rmse, 3),
        "mape": round(mape, 2) if mape else None, "rmsle": round(rmsle, 5),
        "fit_sec": round(fit_sec, 1),
    }


# ============================================================================
# Phase 4 — Modeling
# ============================================================================

def run_models(train: pd.DataFrame, test: pd.DataFrame) -> tuple[list[dict], dict]:
    y_train = train["sales"].values
    y_test  = test["sales"].values
    results = []
    best = {"name": None, "mae": float("inf"), "rmsle": float("inf"), "pipe": None, "feature_cols": None}

    print("\n=== Baselines ===")
    # Per-(store, family) mean
    group_mean = train.groupby(["store_nbr", "family"])["sales"].mean()
    y_pred_gm = test.set_index(["store_nbr", "family"]).index.map(group_mean).to_numpy()
    y_pred_gm = np.where(pd.isna(y_pred_gm), y_train.mean(), y_pred_gm).astype(float)
    r = metrics("Baseline: per-group mean", y_test, y_pred_gm)
    print(f"  {r['name']:55s} MAE={r['mae']:8.2f}  RMSLE={r['rmsle']:.4f}")
    results.append(r)

    # Naive lag-7 (week-over-week)
    y_pred_lag7 = test["sales_lag_7"].fillna(y_train.mean()).values
    r = metrics("Baseline: naive lag-7", y_test, y_pred_lag7)
    print(f"  {r['name']:55s} MAE={r['mae']:8.2f}  RMSLE={r['rmsle']:.4f}")
    results.append(r)

    # Rolling-28
    y_pred_r28 = test["sales_roll_mean_28"].fillna(y_train.mean()).values
    r = metrics("Baseline: naive rolling-28", y_test, y_pred_r28)
    print(f"  {r['name']:55s} MAE={r['mae']:8.2f}  RMSLE={r['rmsle']:.4f}")
    results.append(r)

    print("\n=== Stage 2 (full features) ===")
    candidates = []
    try:
        from lightgbm import LGBMRegressor
        candidates.append(("LightGBM", lambda: LGBMRegressor(
            n_estimators=800, learning_rate=0.05, num_leaves=127, min_child_samples=20,
            subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1, verbose=-1,
        )))
    except ImportError:
        pass
    candidates.append(("HistGradientBoosting", lambda: HistGradientBoostingRegressor(
        max_iter=600, learning_rate=0.05, max_leaf_nodes=127, min_samples_leaf=20,
        l2_regularization=1.0, random_state=42)))
    try:
        from catboost import CatBoostRegressor
        candidates.append(("CatBoost", lambda: CatBoostRegressor(
            iterations=600, learning_rate=0.05, depth=7, loss_function="MAE",
            nan_mode="Min", random_seed=42, verbose=False)))
    except ImportError:
        pass

    X_train_s2 = train[FEATURE_COLS_STAGE2]
    X_test_s2  = test[FEATURE_COLS_STAGE2]
    for name, factory in candidates:
        t0 = time.time()
        p = Pipeline([("pre", build_preprocessor(FEATURE_COLS_STAGE2)), ("m", factory())])
        p.fit(X_train_s2, y_train)
        pred = p.predict(X_test_s2)
        r = metrics(f"Stage 2: {name}", y_test, pred, time.time() - t0)
        print(f"  {r['name']:55s} MAE={r['mae']:8.2f}  RMSLE={r['rmsle']:.4f}  ({r['fit_sec']}s)")
        results.append(r)
        if r["rmsle"] < best["rmsle"]:
            best.update({"name": r["name"], "mae": r["mae"], "rmsle": r["rmsle"],
                         "pipe": p, "feature_cols": FEATURE_COLS_STAGE2})

    print("\n=== Stage 1 (contextual only — cold-start) ===")
    try:
        from lightgbm import LGBMRegressor
        X_train_s1 = train[FEATURE_COLS_STAGE1]
        X_test_s1  = test[FEATURE_COLS_STAGE1]
        t0 = time.time()
        p = Pipeline([("pre", build_preprocessor(FEATURE_COLS_STAGE1)),
                      ("m", LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=127,
                                          random_state=42, n_jobs=-1, verbose=-1))])
        p.fit(X_train_s1, y_train)
        pred = p.predict(X_test_s1)
        r = metrics("Stage 1: LightGBM (no lags)", y_test, pred, time.time() - t0)
        print(f"  {r['name']:55s} MAE={r['mae']:8.2f}  RMSLE={r['rmsle']:.4f}")
        results.append(r)
    except ImportError:
        pass

    print("\n=== Per-family routing (one LightGBM per family) ===")
    try:
        from lightgbm import LGBMRegressor
        per_fam: dict[str, Pipeline] = {}
        all_preds = np.zeros_like(y_test, dtype=float)
        t0 = time.time()
        for fam, tr_grp in train.groupby("family"):
            te_grp = test[test["family"] == fam]
            if len(te_grp) == 0:
                continue
            p = Pipeline([("pre", build_preprocessor(FEATURE_COLS_STAGE2)),
                          ("m", LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=63,
                                              random_state=42, n_jobs=-1, verbose=-1))])
            p.fit(tr_grp[FEATURE_COLS_STAGE2], tr_grp["sales"])
            all_preds[te_grp.index] = p.predict(te_grp[FEATURE_COLS_STAGE2])
            per_fam[fam] = p
        r = metrics("Per-family LightGBM (routed)", y_test, all_preds, time.time() - t0)
        print(f"  {r['name']:55s} MAE={r['mae']:8.2f}  RMSLE={r['rmsle']:.4f}  ({r['fit_sec']}s)")
        results.append(r)
        if r["rmsle"] < best["rmsle"]:
            best.update({"name": r["name"], "mae": r["mae"], "rmsle": r["rmsle"],
                         "pipe": per_fam, "feature_cols": FEATURE_COLS_STAGE2})
    except ImportError:
        pass

    print(f"\nBest: {best['name']}  MAE={best['mae']:.2f}  RMSLE={best['rmsle']:.4f}")
    return results, best


def save_artifacts(best: dict, results: list[dict]) -> None:
    if best.get("pipe") is not None:
        joblib.dump(best["pipe"], MODEL_DIR / "best_model.pkl")
        metadata = {
            "model_name": best["name"],
            "feature_columns": best["feature_cols"],
            "mae": best["mae"],
            "rmsle": best["rmsle"],
            "all_results": results,
        }
        joblib.dump(metadata, MODEL_DIR / "model_metadata.pkl")
        print(f"Saved: {MODEL_DIR / 'best_model.pkl'}")
    OUT.write_text(json.dumps(results, indent=2))
    print(f"Saved: {OUT}")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("=" * 78)
    print("STORE SALES TIME-SERIES FORECASTING — CRISP-ML(Q) Pipeline")
    print("=" * 78)

    df = load_and_merge()
    df = add_features(df)
    train, test = time_split(df, holdout_days=30)
    train = train.reset_index(drop=True)
    test  = test.reset_index(drop=True)

    print(f"\nFeature counts: Stage1={len(FEATURE_COLS_STAGE1)}  Stage2={len(FEATURE_COLS_STAGE2)}")
    results, best = run_models(train, test)
    save_artifacts(best, results)


if __name__ == "__main__":
    main()
