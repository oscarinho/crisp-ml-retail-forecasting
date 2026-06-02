"""Food Demand Forecasting — runs every model from our toolkit and reports best MAE.

Dataset: Genpact / Analytics Vidhya Food Demand challenge (456,548 weekly rows).
Target: num_orders per (week, center_id, meal_id).

Applies lessons learned from the Inventory + Sales labs:
- Time-based split (last 20% of weeks = holdout)
- Lag/rolling features grouped by (center_id, meal_id)
- Mixed scalers per feature type
- Tier 1 benchmarks (CatBoost, HGB, ExtraTrees, LightGBM, RandomForest)
- Per-category routing (the Sales lab winner)
- Champion-challenger backtest (commented out — ~30 min)

Outputs:
- scripts/food_demand_results.json — every model's metrics
- model/food_demand/best_model.pkl — winning pipeline
- model/food_demand/model_metadata.pkl — for app integration
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
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    PowerTransformer,
    StandardScaler,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "food_demand"
OUT  = ROOT / "scripts" / "food_demand_results.json"
MODEL_DIR = ROOT / "model" / "food_demand"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Phase 2/3 — Load, merge, engineer features
# ============================================================================

def load_and_merge() -> pd.DataFrame:
    train = pd.read_csv(DATA / "train.csv")
    centers = pd.read_csv(DATA / "fulfilment_center_info.csv")
    meals = pd.read_csv(DATA / "meal_info.csv")
    df = train.merge(meals, on="meal_id", how="left").merge(centers, on="center_id", how="left")
    print(f"Loaded train: {len(df):,} rows · {df['week'].nunique()} weeks "
          f"· {df['center_id'].nunique()} centers · {df['meal_id'].nunique()} meals")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["center_id", "meal_id", "week"]).reset_index(drop=True)

    # Price-derived features
    df["discount_abs"] = df["base_price"] - df["checkout_price"]
    df["discount_pct"] = df["discount_abs"] / df["base_price"].replace(0, np.nan)
    df["price_vs_base_ratio"] = df["checkout_price"] / df["base_price"].replace(0, np.nan)

    # Lag + rolling features per (center_id, meal_id)
    g = df.groupby(["center_id", "meal_id"], sort=False)["num_orders"]
    for lag in [1, 2, 3, 5, 10]:
        df[f"orders_lag_{lag}"] = g.shift(lag)
    for w in [3, 5, 10]:
        shifted = g.shift(1)
        df[f"orders_roll_mean_{w}"] = shifted.groupby(
            df.groupby(["center_id", "meal_id"]).ngroup().values
        ).rolling(w, min_periods=1).mean().reset_index(level=0, drop=True)
        df[f"orders_roll_std_{w}"] = shifted.groupby(
            df.groupby(["center_id", "meal_id"]).ngroup().values
        ).rolling(w, min_periods=2).std().reset_index(level=0, drop=True)

    # Calendar — week of year (treating 'week' col as ordinal week-of-dataset)
    df["week_mod52"] = df["week"] % 52
    df["sin_week"] = np.sin(2 * np.pi * df["week_mod52"] / 52)
    df["cos_week"] = np.cos(2 * np.pi * df["week_mod52"] / 52)

    return df


FEATURE_COLS_STAGE2 = [
    # Numeric (standard)
    "checkout_price", "base_price", "discount_abs", "discount_pct", "price_vs_base_ratio",
    "op_area",
    "orders_lag_1", "orders_lag_2", "orders_lag_3", "orders_lag_5", "orders_lag_10",
    "orders_roll_mean_3", "orders_roll_std_3",
    "orders_roll_mean_5", "orders_roll_std_5",
    "orders_roll_mean_10", "orders_roll_std_10",
    # Numeric (cyclic — passthrough)
    "sin_week", "cos_week",
    # Numeric (ordinal — minmax)
    "week", "week_mod52",
    # Binary (passthrough)
    "emailer_for_promotion", "homepage_featured",
    # Categorical
    "center_type", "category", "cuisine", "city_code", "region_code",
]
FEATURE_COLS_STAGE1 = [c for c in FEATURE_COLS_STAGE2
                      if "lag_" not in c and "roll_" not in c]

NUM_STANDARD = ["checkout_price", "base_price", "discount_abs", "discount_pct",
                "price_vs_base_ratio", "op_area",
                "orders_lag_1", "orders_lag_2", "orders_lag_3", "orders_lag_5", "orders_lag_10",
                "orders_roll_mean_3", "orders_roll_std_3",
                "orders_roll_mean_5", "orders_roll_std_5",
                "orders_roll_mean_10", "orders_roll_std_10"]
NUM_FOURIER = ["sin_week", "cos_week"]
NUM_MINMAX = ["week", "week_mod52"]
BINARY = ["emailer_for_promotion", "homepage_featured"]
CATEGORICAL = ["center_type", "category", "cuisine", "city_code", "region_code"]


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


def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    max_week = df["week"].max()
    split_week = int(max_week * 0.85)
    tr = df[df["week"] <= split_week].copy()
    te = df[df["week"] >  split_week].copy()
    print(f"Time split: train ≤ week {split_week} ({len(tr):,} rows) · "
          f"holdout > week {split_week} ({len(te):,} rows)")
    return tr, te


# ============================================================================
# Eval helpers
# ============================================================================

def metrics(name: str, y_true, y_pred, fit_sec: float = 0.0) -> dict:
    y_true = np.asarray(y_true, float)
    y_pred = np.clip(np.asarray(y_pred, float), 0, None)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else None
    # Kaggle uses 100 * RMSLE for this competition
    y_true_pos = np.maximum(y_true, 0)
    rmsle = float(np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true_pos)) ** 2)))
    return {
        "name": name, "mae": round(mae, 3), "rmse": round(rmse, 3),
        "mape": round(mape, 2) if mape else None, "rmsle_100": round(100 * rmsle, 3),
        "fit_sec": round(fit_sec, 1),
    }


# ============================================================================
# Phase 4 — Modeling
# ============================================================================

def run_models(train: pd.DataFrame, test: pd.DataFrame) -> list[dict]:
    y_train = train["num_orders"].values
    y_test  = test["num_orders"].values
    X_train_s2 = train[FEATURE_COLS_STAGE2]
    X_test_s2  = test[FEATURE_COLS_STAGE2]
    X_train_s1 = train[FEATURE_COLS_STAGE1]
    X_test_s1  = test[FEATURE_COLS_STAGE1]

    results = []

    print("\n=== Baselines ===")
    # Group mean
    group_mean = train.groupby(["center_id", "meal_id"])["num_orders"].mean()
    y_pred_gm = test.set_index(["center_id", "meal_id"]).index.map(group_mean).to_numpy()
    y_pred_gm = np.where(pd.isna(y_pred_gm), y_train.mean(), y_pred_gm).astype(float)
    r = metrics("Baseline: per-group mean", y_test, y_pred_gm)
    print(f"  {r['name']:55s} MAE={r['mae']:8.2f}  RMSLE×100={r['rmsle_100']:.2f}")
    results.append(r)

    # Naive lag-1
    r = metrics("Baseline: naive lag-1", y_test, X_test_s2["orders_lag_1"].fillna(y_train.mean()).values)
    print(f"  {r['name']:55s} MAE={r['mae']:8.2f}  RMSLE×100={r['rmsle_100']:.2f}")
    results.append(r)

    # Naive rolling-5
    r = metrics("Baseline: naive rolling-5", y_test, X_test_s2["orders_roll_mean_5"].fillna(y_train.mean()).values)
    print(f"  {r['name']:55s} MAE={r['mae']:8.2f}  RMSLE×100={r['rmsle_100']:.2f}")
    results.append(r)

    print("\n=== Stage 2 (full features) ===")
    best = {"name": None, "mae": float("inf"), "pipe": None, "feature_cols": None}

    candidates_s2 = []
    try:
        from lightgbm import LGBMRegressor
        candidates_s2.append(("LightGBM", lambda: LGBMRegressor(
            n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=20,
            subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=-1, verbose=-1,
        )))
    except ImportError:
        pass
    candidates_s2.extend([
        ("HistGradientBoosting", lambda: HistGradientBoostingRegressor(
            max_iter=600, learning_rate=0.05, max_leaf_nodes=63,
            min_samples_leaf=20, l2_regularization=1.0, random_state=42)),
        ("RandomForest", lambda: RandomForestRegressor(
            n_estimators=300, max_depth=18, min_samples_leaf=5, random_state=42, n_jobs=-1)),
        ("ExtraTrees", lambda: ExtraTreesRegressor(
            n_estimators=300, max_depth=None, min_samples_leaf=5, random_state=42, n_jobs=-1)),
    ])
    try:
        from catboost import CatBoostRegressor
        candidates_s2.append(("CatBoost", lambda: CatBoostRegressor(
            iterations=600, learning_rate=0.05, depth=6, loss_function="MAE",
            nan_mode="Min", random_seed=42, verbose=False)))
    except ImportError:
        pass

    for name, factory in candidates_s2:
        t0 = time.time()
        p = Pipeline([("pre", build_preprocessor(FEATURE_COLS_STAGE2)), ("m", factory())])
        p.fit(X_train_s2, y_train)
        pred = p.predict(X_test_s2)
        r = metrics(f"Stage 2: {name}", y_test, pred, time.time() - t0)
        print(f"  {r['name']:55s} MAE={r['mae']:8.2f}  RMSLE×100={r['rmsle_100']:.2f}  ({r['fit_sec']}s)")
        results.append(r)
        if r["mae"] < best["mae"]:
            best.update({"name": r["name"], "mae": r["mae"], "pipe": p, "feature_cols": FEATURE_COLS_STAGE2})

    print("\n=== Stage 1 (contextual only, no lags — cold-start) ===")
    for name, factory in candidates_s2[:1]:  # only LightGBM for stage 1
        t0 = time.time()
        p = Pipeline([("pre", build_preprocessor(FEATURE_COLS_STAGE1)), ("m", factory())])
        p.fit(X_train_s1, y_train)
        pred = p.predict(X_test_s1)
        r = metrics(f"Stage 1: {name} (no lags)", y_test, pred, time.time() - t0)
        print(f"  {r['name']:55s} MAE={r['mae']:8.2f}  RMSLE×100={r['rmsle_100']:.2f}")
        results.append(r)

    print("\n=== Per-category routing (one LightGBM per cuisine) ===")
    try:
        from lightgbm import LGBMRegressor
        per_cat: dict[str, Pipeline] = {}
        all_preds = np.zeros_like(y_test, dtype=float)
        t0 = time.time()
        for cuisine, tr_grp in train.groupby("cuisine"):
            te_grp = test[test["cuisine"] == cuisine]
            if len(te_grp) == 0:
                continue
            p = Pipeline([
                ("pre", build_preprocessor(FEATURE_COLS_STAGE2)),
                ("m", LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=63,
                                    min_child_samples=20, subsample=0.9, colsample_bytree=0.9,
                                    random_state=42, n_jobs=-1, verbose=-1)),
            ])
            p.fit(tr_grp[FEATURE_COLS_STAGE2], tr_grp["num_orders"])
            all_preds[te_grp.index] = p.predict(te_grp[FEATURE_COLS_STAGE2])
            per_cat[cuisine] = p
        r = metrics("Per-cuisine LightGBM (routed)", y_test, all_preds, time.time() - t0)
        print(f"  {r['name']:55s} MAE={r['mae']:8.2f}  RMSLE×100={r['rmsle_100']:.2f}  ({r['fit_sec']}s)")
        results.append(r)
        if r["mae"] < best["mae"]:
            best.update({"name": r["name"], "mae": r["mae"], "pipe": per_cat, "feature_cols": FEATURE_COLS_STAGE2})
    except ImportError:
        pass

    print(f"\nBest: {best['name']}  MAE={best['mae']:.2f}")
    return results, best


def save_artifacts(best: dict, results: list[dict]) -> None:
    if best.get("pipe") is not None:
        joblib.dump(best["pipe"], MODEL_DIR / "best_model.pkl")
        metadata = {
            "model_name": best["name"],
            "feature_columns": best["feature_cols"],
            "mae": best["mae"],
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
    print("FOOD DEMAND FORECASTING — CRISP-ML(Q) Pipeline")
    print("=" * 78)

    df = load_and_merge()
    df = add_features(df)
    train, test = time_split(df)
    train = train.reset_index(drop=True)
    test  = test.reset_index(drop=True)

    print(f"\nFeature counts: Stage1={len(FEATURE_COLS_STAGE1)}  Stage2={len(FEATURE_COLS_STAGE2)}")
    results, best = run_models(train, test)
    save_artifacts(best, results)


if __name__ == "__main__":
    main()
