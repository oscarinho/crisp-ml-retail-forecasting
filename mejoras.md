# Notebook Improvement Notes

Review target: `notebooks/Inventory_Forecasting_CRISPML.ipynb`

## Critical Issues

### 1. Lag Features Are Not Truly Temporal

The notebook creates lag features using:

```python
GROUP = ["Store ID", "Category"]
```

However, the dataset contains multiple product rows per `Store ID` + `Category` + `Date`. As a result, `lag_1` often points to another product row from the same calendar day instead of the previous day.

Observed impact:

- Around 75% of `lag_1` rows reference the same calendar day.
- Only around 24% of `lag_1` rows reference the previous day.
- This weakens the claim that the lag features are leakage-safe daily history.

Recommended fix:

- If the forecasting grain is store-product-day, create lags by `["Store ID", "Product ID"]`.
- If the forecasting grain is store-category-day, aggregate the data to `Store ID + Category + Date` first, then create lags.
- Add a validation check after lag creation to confirm that `lag_1` comes from the expected prior date.

### 2. Notebook Model and Streamlit App Are Misaligned

The notebook saves `model/model.pkl` as the Stage 2 model, which requires 33 engineered features, including lag, rolling, Fourier, and pricing features.

The Streamlit app uses a smaller interactive feature set with no lag features:

```python
DEMO_FEATURES = [
    "month", "day_of_week", "quarter", "is_weekend",
    "Inventory Level", "Price", "Discount", "price_vs_competitor",
    "Holiday/Promotion",
    "Category", "Region", "Store ID", "Weather Condition", "Seasonality",
]
```

Because `model/model.pkl` exists, the app loads the full Stage 2 model but then predicts with the demo feature set. This can fail at runtime or produce invalid inference behavior.

Recommended fix:

- Save separate artifacts:
  - `model_stage2.pkl` for batch forecasting with lag features.
  - `model_contextual.pkl` for interactive simulation without lag features.
  - `model_q80.pkl` for reorder and safety-stock decisions.
- Make the app explicitly choose the correct model for each tab.
- Use `metadata["feature_columns"]` to validate input columns before prediction.

### 3. Saved Artifacts Are Not Fully Reproducible

Loading `model/model.pkl` failed in the current environment because LightGBM requires `libomp.dylib` on macOS. There are also scikit-learn version mismatch warnings: the artifact was serialized with a different scikit-learn version than the one currently installed.

Recommended fix:

- Pin dependency versions instead of using only `>=` ranges.
- Add macOS setup notes for LightGBM:

```bash
brew install libomp
```

- Store environment metadata with each model artifact:
  - Python version
  - scikit-learn version
  - LightGBM version
  - pandas version
  - numpy version

## Methodology Improvements

### 4. Clarify the Forecasting Grain

The notebook headline says the task is daily `Units Sold` per store-product pair, but several modeling decisions operate at the store-category level.

Recommended fix:

- State the exact prediction grain in Phase 1.
- Align feature engineering, validation, evaluation, and deployment around that grain.
- If product-level forecasting is required, reconsider excluding `Product ID`.

### 5. Reframe the Success Criteria

The stated success criteria are:

- sMAPE < 40%
- MAE < 60 units
- outperform the `Demand Forecast` baseline

Final results do not meet those criteria:

- Stage 2 MAE: about 69.1
- Stage 2 sMAPE: about 58.4%
- `Demand Forecast` baseline MAE: about 8.3

Recommended fix:

- Add an explicit final conclusion that the notebook is a methodological pipeline, not a winning production model for this synthetic dataset.
- Avoid describing `Demand Forecast` as a baseline to beat unless it is clearly labeled as an oracle-like synthetic column.

### 6. Treat Stage 1 as a Cold-Start Fallback, Not as an Ensemble Contributor

The optimized ensemble gives Stage 1 almost zero weight. That means Stage 1 does not improve the main Stage 2 forecast on this dataset.

Recommended fix:

- Position Stage 1 as a fallback for new products or stores with no sales history.
- Do not present the two-stage ensemble as the main production strategy unless it shows measurable lift.

### 7. Add Multi-Horizon Forecasting

The notebook currently performs single-step forecasting. Inventory planning usually needs forecasts over lead times such as 7, 14, or 28 days.

Recommended fix:

- Add direct horizon models for `t+1`, `t+7`, `t+14`, and `t+28`.
- Evaluate forecast error by horizon.
- Connect reorder recommendations to lead-time demand instead of one-day demand.

### 8. Improve Backtesting

The notebook uses a single 2022/2023 holdout split. This is useful but limited.

Recommended fix:

- Add rolling-origin backtesting.
- Report mean and variance of MAE/sMAPE across folds.
- Compare model stability over multiple temporal cutoffs.

## Deployment Improvements

### 9. Validate Feature Availability Before Prediction

The app should fail clearly when the loaded model expects features that are not available.

Recommended fix:

```python
missing = set(metadata["feature_columns"]) - set(input_df.columns)
if missing:
    raise ValueError(f"Missing required model features: {sorted(missing)}")
```

### 10. Use the Quantile Model for Reorder Advisory

The notebook trains and saves a P80 quantile model, but the app currently appears to use the point forecast for reorder logic.

Recommended fix:

- Load `model_q80.pkl` in the app.
- Use the P80 prediction as the stock-to target.
- Show both expected demand and safety-stock demand.

### 11. Add Model Cards or Artifact Metadata

The current metadata is useful, but it can be expanded.

Recommended additions:

- Training date
- Package versions
- Model type
- Feature list
- Target
- Split date
- Evaluation metrics
- Known limitations
- Required preprocessing steps
- Whether the model supports cold-start inference

## Suggested Priority

1. Fix lag feature grouping or aggregate the dataset first.
2. Split deployment artifacts by use case.
3. Pin dependencies and document `libomp` for LightGBM.
4. Update conclusions to reflect that success criteria were not met.
5. Add multi-horizon and rolling-origin evaluation.

