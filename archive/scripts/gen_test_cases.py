"""Generate exact expected predictions from both apps' models, for QA testing."""
import os, sys, joblib
import numpy as np
import pandas as pd

ROOT = "/Users/oscarponce/Documents/personal/marca-personal/labs/forecasting-inventory"

# ============================================================================
# INVENTORY APP — model_contextual.pkl
# ============================================================================
print("=" * 78)
print("INVENTORY APP — app/app.py")
print("=" * 78)

inv_model = joblib.load(os.path.join(ROOT, "model", "model_contextual.pkl"))
inv_q80   = joblib.load(os.path.join(ROOT, "model", "model_q80.pkl"))

INV_FEATURES = [
    "month", "day_of_week", "quarter", "is_weekend",
    "Inventory Level", "Price", "Discount", "price_vs_competitor",
    "Holiday/Promotion",
    "Category", "Region", "Store ID", "Weather Condition", "Seasonality",
]

def inv_predict(scenario):
    row = pd.DataFrame([scenario])
    return float(inv_model.predict(row[INV_FEATURES])[0])

inv_scenarios = [
    {
        "name": "Case 1 — Toys / Holiday / Weekend in December",
        "Store ID": "S001", "Category": "Toys", "Region": "North",
        "month": 12, "day_of_week": 5, "quarter": 4, "is_weekend": 1,
        "Weather Condition": "Snowy", "Seasonality": "Winter",
        "Holiday/Promotion": 1,
        "Price": 25.0, "Discount": 15, "Competitor Pricing": 28.0,
        "Inventory Level": 200,
    },
    {
        "name": "Case 2 — Groceries / Sunny weekday / no promo",
        "Store ID": "S002", "Category": "Groceries", "Region": "South",
        "month": 6, "day_of_week": 2, "quarter": 2, "is_weekend": 0,
        "Weather Condition": "Sunny", "Seasonality": "Summer",
        "Holiday/Promotion": 0,
        "Price": 12.0, "Discount": 0, "Competitor Pricing": 13.5,
        "Inventory Level": 400,
    },
    {
        "name": "Case 3 — Electronics / Spring / heavy discount",
        "Store ID": "S003", "Category": "Electronics", "Region": "East",
        "month": 4, "day_of_week": 4, "quarter": 2, "is_weekend": 0,
        "Weather Condition": "Cloudy", "Seasonality": "Spring",
        "Holiday/Promotion": 0,
        "Price": 85.0, "Discount": 20, "Competitor Pricing": 95.0,
        "Inventory Level": 60,
    },
    {
        "name": "Case 4 — Critical stockout risk",
        "Store ID": "S004", "Category": "Clothing", "Region": "West",
        "month": 11, "day_of_week": 6, "quarter": 4, "is_weekend": 1,
        "Weather Condition": "Rainy", "Seasonality": "Autumn",
        "Holiday/Promotion": 1,
        "Price": 45.0, "Discount": 10, "Competitor Pricing": 50.0,
        "Inventory Level": 50,
    },
    {
        "name": "Case 5 — Well-stocked baseline",
        "Store ID": "S005", "Category": "Furniture", "Region": "North",
        "month": 9, "day_of_week": 1, "quarter": 3, "is_weekend": 0,
        "Weather Condition": "Sunny", "Seasonality": "Autumn",
        "Holiday/Promotion": 0,
        "Price": 70.0, "Discount": 5, "Competitor Pricing": 72.0,
        "Inventory Level": 350,
    },
]

for sc in inv_scenarios:
    sc["price_vs_competitor"] = sc["Price"] / max(sc["Competitor Pricing"], 0.01)
    pred = inv_predict(sc)
    pred_int = max(0, int(round(pred)))
    cov = sc["Inventory Level"] / max(pred_int, 1)

    if cov < 0.5:   status = "CRITICAL — STOCKOUT RISK"
    elif cov < 1.2: status = "LOW STOCK"
    else:           status = "WELL STOCKED"

    buffer = pred_int * 7
    shortage = max(0, buffer - sc["Inventory Level"])
    reorder = int(np.ceil(shortage / max(pred_int, 1)) * max(pred_int, 1)) if pred_int else 0

    print(f"\n▸ {sc['name']}")
    print(f"   Inputs:  Store={sc['Store ID']}  Cat={sc['Category']}  Region={sc['Region']}")
    print(f"            Month={sc['month']}  WD={sc['day_of_week']}  Season={sc['Seasonality']}  Weather={sc['Weather Condition']}")
    print(f"            Price=${sc['Price']}  Discount={sc['Discount']}%  Comp=${sc['Competitor Pricing']}")
    print(f"            Holiday={'YES' if sc['Holiday/Promotion'] else 'NO'}   Inventory={sc['Inventory Level']}")
    print(f"   ──── EXPECTED ────")
    print(f"   Predicted demand:  {pred_int} units (raw {pred:.2f})")
    print(f"   Coverage:          {cov:.1f}×")
    print(f"   Status:            {status}")
    print(f"   Reorder qty:       {reorder} units (7-day buffer)")


# ============================================================================
# SALES APP — model_per_category.pkl
# ============================================================================
print("\n\n" + "=" * 78)
print("SALES APP — app/app_sales.py")
print("=" * 78)

per_cat   = joblib.load(os.path.join(ROOT, "model", "sales", "model_per_category.pkl"))
fallback  = joblib.load(os.path.join(ROOT, "model", "sales", "model_contextual.pkl"))

SALES_FEATURES = [
    'Category','Region','Weather Condition','Seasonality',
    'Inventory Level','Price','Discount','Competitor Pricing',
    'Promotion','Epidemic',
    'day_of_week','month','is_weekend','sin_year','cos_year',
]

def sales_predict(scenario):
    row = pd.DataFrame([scenario])
    X = row[SALES_FEATURES]
    cat = scenario["Category"]
    if cat in per_cat:
        return float(per_cat[cat].predict(X)[0]), "per-category"
    return float(fallback.predict(X)[0]), "fallback (app-aligned)"

def fourier(month):
    d = pd.Timestamp(2024, month, 15).dayofyear
    return np.sin(2*np.pi*d/365.25), np.cos(2*np.pi*d/365.25)

sales_scenarios = [
    {
        "name": "Case 1 — Clothing / Promotion + Epidemic / Winter",
        "Store ID": "S001", "Category": "Clothing", "Region": "North",
        "month": 1, "day_of_week": 5, "is_weekend": 1,
        "Weather Condition": "Snowy", "Seasonality": "Winter",
        "Promotion": 1, "Epidemic": 1,
        "Price": 60.0, "Discount": 20, "Competitor Pricing": 70.0,
        "Inventory Level": 250,
    },
    {
        "name": "Case 2 — Groceries / Summer baseline / no promo",
        "Store ID": "S002", "Category": "Groceries", "Region": "South",
        "month": 7, "day_of_week": 3, "is_weekend": 0,
        "Weather Condition": "Sunny", "Seasonality": "Summer",
        "Promotion": 0, "Epidemic": 0,
        "Price": 15.0, "Discount": 0, "Competitor Pricing": 16.0,
        "Inventory Level": 500,
    },
    {
        "name": "Case 3 — Electronics / Black-Friday-style (heavy promo)",
        "Store ID": "S003", "Category": "Electronics", "Region": "East",
        "month": 11, "day_of_week": 4, "is_weekend": 0,
        "Weather Condition": "Cloudy", "Seasonality": "Autumn",
        "Promotion": 1, "Epidemic": 0,
        "Price": 120.0, "Discount": 25, "Competitor Pricing": 140.0,
        "Inventory Level": 180,
    },
    {
        "name": "Case 4 — Furniture / Spring / regular weekday",
        "Store ID": "S004", "Category": "Furniture", "Region": "West",
        "month": 4, "day_of_week": 2, "is_weekend": 0,
        "Weather Condition": "Rainy", "Seasonality": "Spring",
        "Promotion": 0, "Epidemic": 0,
        "Price": 95.0, "Discount": 5, "Competitor Pricing": 100.0,
        "Inventory Level": 200,
    },
    {
        "name": "Case 5 — Toys / December weekend / promo on",
        "Store ID": "S005", "Category": "Toys", "Region": "North",
        "month": 12, "day_of_week": 6, "is_weekend": 1,
        "Weather Condition": "Snowy", "Seasonality": "Winter",
        "Promotion": 1, "Epidemic": 0,
        "Price": 35.0, "Discount": 15, "Competitor Pricing": 38.0,
        "Inventory Level": 80,
    },
]

for sc in sales_scenarios:
    sc["sin_year"], sc["cos_year"] = fourier(sc["month"])
    pred, route = sales_predict(sc)
    pred = max(pred, 0.0)
    pred_int = max(0, int(round(pred)))
    cov = sc["Inventory Level"] / max(pred_int, 1)

    if cov < 0.5:   status = "CRITICAL — STOCKOUT RISK"
    elif cov < 1.2: status = "LOW STOCK"
    else:           status = "WELL STOCKED"

    buffer = pred_int * 7
    shortage = max(0, buffer - sc["Inventory Level"])
    reorder = int(np.ceil(shortage / max(pred_int, 1)) * max(pred_int, 1)) if pred_int else 0

    print(f"\n▸ {sc['name']}")
    print(f"   Inputs:  Store={sc['Store ID']}  Cat={sc['Category']}  Region={sc['Region']}")
    print(f"            Month={sc['month']}  WD={sc['day_of_week']}  Season={sc['Seasonality']}  Weather={sc['Weather Condition']}")
    print(f"            Price=${sc['Price']}  Discount={sc['Discount']}%  Comp=${sc['Competitor Pricing']}")
    print(f"            Promo={'YES' if sc['Promotion'] else 'NO'}   Epidemic={'YES' if sc['Epidemic'] else 'NO'}   Inv={sc['Inventory Level']}")
    print(f"   ──── EXPECTED ────")
    print(f"   Predicted demand:  {pred_int} units (raw {pred:.2f})")
    print(f"   Routing:           {route}")
    print(f"   Coverage:          {cov:.1f}×")
    print(f"   Status:            {status}")
    print(f"   Reorder qty:       {reorder} units (7-day buffer)")

print("\n" + "=" * 78)
print("Done. Tolerance: ±2 units for inventory, ±1 unit for sales (model rounding).")
