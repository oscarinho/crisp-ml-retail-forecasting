"""Experimentos para validar el uso honesto de Demand Forecast.

A) Residual Learning — target = Units Sold − Demand Forecast.
   Comparamos: DF puro · modelo directo (sin DF) · residual model (DF + corrección).

B) Champion-Challenger Backtesting — N modelos compiten en ventana móvil de 30 días.
   Patrón clásico de MRP planning: "quién ganó el mes pasado se queda".

Resultados se guardan en scripts/df_experiments_results.json para luego pegarlos
en la libreta de Inventory.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "retail_store_inventory.csv"
OUT  = ROOT / "scripts" / "df_experiments_results.json"

# ============================================================================
# Shared setup
# ============================================================================

FEATURES = [
    "month", "day_of_week", "quarter", "is_weekend",
    "Inventory Level", "Price", "Discount", "price_vs_competitor",
    "Holiday/Promotion",
    "Category", "Region", "Store ID", "Weather Condition", "Seasonality",
]
NUM = ["month", "day_of_week", "quarter", "is_weekend",
       "Inventory Level", "Price", "Discount", "price_vs_competitor", "Holiday/Promotion"]
CAT = ["Category", "Region", "Store ID", "Weather Condition", "Seasonality"]


def load_and_prep() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Store ID", "Product ID", "Date"]).reset_index(drop=True)
    df["month"] = df["Date"].dt.month
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["quarter"] = df["Date"].dt.quarter
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["price_vs_competitor"] = df["Price"] / df["Competitor Pricing"].replace(0, np.nan)
    return df


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), NUM),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT),
    ])


def metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, float)
    y_pred = np.clip(np.asarray(y_pred, float), 0, None)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else None
    bias = float(np.mean(y_pred - y_true))
    return {"mae": round(mae, 3), "rmse": round(rmse, 3), "mape": round(mape, 2) if mape else None, "bias": round(bias, 3)}


def make_model(kind: str):
    if kind == "HGB":
        return HistGradientBoostingRegressor(max_iter=500, learning_rate=0.045,
                                             max_leaf_nodes=63, random_state=42)
    if kind == "RF":
        return RandomForestRegressor(n_estimators=300, max_depth=18, min_samples_leaf=5,
                                     random_state=42, n_jobs=-1)
    if kind == "LightGBM":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31,
                             random_state=42, n_jobs=-1, verbose=-1)
    raise ValueError(kind)


# ============================================================================
# EXPERIMENT A — Residual Learning
# ============================================================================

def experiment_a(df: pd.DataFrame) -> list[dict]:
    HOLDOUT_DAYS = 90
    SPLIT = df["Date"].max() - pd.Timedelta(days=HOLDOUT_DAYS)
    train = df[df["Date"] <= SPLIT].copy()
    test  = df[df["Date"] >  SPLIT].copy()

    y_units_train = train["Units Sold"].values
    y_units_test  = test["Units Sold"].values
    df_train = train["Demand Forecast"].values
    df_test  = test["Demand Forecast"].values
    y_resid_train = y_units_train - df_train

    print(f"Train: {len(train):,} rows  Test: {len(test):,} rows  Split @ {SPLIT.date()}")
    print(f"Residual y_train stats:  mean={y_resid_train.mean():+.2f}  "
          f"std={y_resid_train.std():.2f}  |median|={np.median(np.abs(y_resid_train)):.2f}")

    results = []

    # Baseline 0 — DF puro
    m = metrics(y_units_test, df_test)
    m["name"] = "Baseline: DF puro (sin modelo)"
    m["fit_sec"] = 0.0
    results.append(m)
    print(f"  {m['name']:55s} MAE={m['mae']:6.2f}  bias={m['bias']:+.2f}")

    # Baseline 1 — Direct model on Units Sold (sin DF feature)
    for kind in ["HGB", "RF", "LightGBM"]:
        try:
            t0 = time.time()
            p = Pipeline([("pre", make_preprocessor()), ("m", make_model(kind))])
            p.fit(train[FEATURES], y_units_train)
            pred = p.predict(test[FEATURES])
            m = metrics(y_units_test, pred)
            m["name"] = f"Direct: {kind} (target=Units Sold, sin DF)"
            m["fit_sec"] = round(time.time() - t0, 1)
            results.append(m)
            print(f"  {m['name']:55s} MAE={m['mae']:6.2f}  bias={m['bias']:+.2f}  ({m['fit_sec']}s)")
        except Exception as e:
            print(f"  [skip {kind}] {e}")

    # Residual learning
    for kind in ["HGB", "RF", "LightGBM"]:
        try:
            t0 = time.time()
            p = Pipeline([("pre", make_preprocessor()), ("m", make_model(kind))])
            p.fit(train[FEATURES], y_resid_train)
            resid_pred = p.predict(test[FEATURES])
            final_pred = df_test + resid_pred
            m = metrics(y_units_test, final_pred)
            m["name"] = f"Residual: DF + {kind}(features)"
            m["fit_sec"] = round(time.time() - t0, 1)
            results.append(m)
            print(f"  {m['name']:55s} MAE={m['mae']:6.2f}  bias={m['bias']:+.2f}  ({m['fit_sec']}s)")
        except Exception as e:
            print(f"  [skip residual-{kind}] {e}")

    return results


# ============================================================================
# EXPERIMENT B — Champion-Challenger Backtesting
# ============================================================================

def experiment_b(df: pd.DataFrame) -> tuple[list[dict], dict, dict]:
    min_train_end = df["Date"].min() + pd.Timedelta(days=180)  # 6 months warm-up
    max_date = df["Date"].max()

    # Monthly rolling windows of 30 days
    windows = []
    cur = min_train_end
    i = 0
    while True:
        win_start = cur + pd.Timedelta(days=1)
        win_end = cur + pd.Timedelta(days=30)
        if win_end > max_date:
            break
        windows.append((i, cur, win_start, win_end))
        cur = win_end
        i += 1

    print(f"Backtesting windows: {len(windows)}")

    contenders = {
        "DF_puro": None,                                    # no fit — uses Demand Forecast directly
        "HGB_direct":  lambda: ("direct",  make_model("HGB")),
        "RF_direct":   lambda: ("direct",  make_model("RF")),
        "HGB_residual": lambda: ("residual", make_model("HGB")),
        "RF_residual":  lambda: ("residual", make_model("RF")),
    }
    try:
        contenders["LightGBM_direct"]   = lambda: ("direct",   make_model("LightGBM"))
        contenders["LightGBM_residual"] = lambda: ("residual", make_model("LightGBM"))
    except Exception:
        pass

    records = []
    for (wi, train_end, win_start, win_end) in windows:
        tr = df[df["Date"] <= train_end]
        te = df[(df["Date"] >= win_start) & (df["Date"] <= win_end)]
        if len(te) == 0:
            continue
        y_units_tr = tr["Units Sold"].values
        y_units_te = te["Units Sold"].values
        df_te = te["Demand Forecast"].values

        for name, factory in contenders.items():
            t0 = time.time()
            if name == "DF_puro":
                pred = df_te
            else:
                mode, model = factory()
                p = Pipeline([("pre", make_preprocessor()), ("m", model)])
                if mode == "direct":
                    p.fit(tr[FEATURES], y_units_tr)
                    pred = p.predict(te[FEATURES])
                else:  # residual
                    y_resid_tr = y_units_tr - tr["Demand Forecast"].values
                    p.fit(tr[FEATURES], y_resid_tr)
                    pred = df_te + p.predict(te[FEATURES])
            m = metrics(y_units_te, pred)
            records.append({
                "window": wi,
                "win_start": str(win_start.date()),
                "win_end": str(win_end.date()),
                "model": name,
                "mae": m["mae"], "rmse": m["rmse"], "mape": m["mape"], "bias": m["bias"],
                "n_test": int(len(te)),
                "fit_sec": round(time.time() - t0, 1),
            })
        print(f"  W{wi+1:02d}/{len(windows)}  {win_start.date()} → {win_end.date()}  ({len(te):,} test rows)")

    rec_df = pd.DataFrame(records)
    champions = rec_df.loc[rec_df.groupby("window")["mae"].idxmin(), ["window", "win_start", "win_end", "model", "mae"]]
    win_counts = champions["model"].value_counts().to_dict()
    summary = rec_df.groupby("model")["mae"].agg(["mean", "std", "min", "max"]).round(3).reset_index().to_dict("records")

    print("\n=== Champion-challenger summary (win counts) ===")
    for k, v in sorted(win_counts.items(), key=lambda x: -x[1]):
        print(f"  {k:25s} {v:>3} wins")
    print("\n=== Overall MAE per model (mean ± std across windows) ===")
    for row in sorted(summary, key=lambda r: r["mean"]):
        print(f"  {row['model']:25s} MAE mean={row['mean']:6.2f}  std={row['std']:5.2f}  [min={row['min']:.2f}, max={row['max']:.2f}]")

    return records, win_counts, summary


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    df = load_and_prep()

    print("=" * 72)
    print("EXPERIMENT A — Residual Learning (90-day holdout)")
    print("=" * 72)
    a_results = experiment_a(df)

    print()
    print("=" * 72)
    print("EXPERIMENT B — Champion-Challenger Backtesting (rolling 30-day windows)")
    print("=" * 72)
    b_records, b_win_counts, b_summary = experiment_b(df)

    OUT.write_text(json.dumps({
        "experiment_a": a_results,
        "experiment_b_records": b_records,
        "experiment_b_win_counts": b_win_counts,
        "experiment_b_summary": b_summary,
    }, indent=2, default=str))
    print(f"\nResults saved → {OUT}")


if __name__ == "__main__":
    main()
