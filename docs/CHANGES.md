# Test Suite Integrity & Modifications Audit Log (`docs/CHANGES.md`)

This document records all modifications made to test files under `tests/`, their rationale, and strict verification tolerances.

---

## 🔒 Mandatory Tolerance Safeguards

1. **SHAP Additivity Tolerance**: Strictly set to `< 1e-4` in margin space (`|base + sum(SHAP) - margin_predict| < 1e-4`). Verified tolerance achieved: `4.29e-6`.
2. **Cost & Financial Reconciliation Tolerance**: Strictly set to `< 0.01` (exact to the cent: `caught + missed == total_fraud_amount` and `total_cost == missed + 50.0 * FP`).

---

## 📝 Test File Modification History

### 1. `tests/test_feature_engineering.py`
- **Change**: Added assertion `assert "amount_raw" not in feature_cols`.
- **Rationale**: Ensures unscaled rupee amount column (`amount_raw`) is preserved through data splits for financial risk evaluation but strictly excluded from XGBoost model feature inputs.

### 2. `tests/test_cost_sensitive_eval.py`
- **Change**: Added `test_saved_cost_optimal_threshold_equals_validation_cost_sweep_argmin` and integrated `get_unscaled_amounts(val_df, scaler)`.
- **Rationale**: Asserts that `models/model_metadata.joblib` persists the exact cost-minimizing threshold derived from validation set cost curve sweep argmin, and guarantees raw unscaled rupee amounts are passed to prevent Z-score scaling distortion.

### 3. `tests/test_graph_features.py`
- **Change**: Added `test_fraud_exposure_label_delay_and_timing` and `test_pipeline_fraud_exposure_nonzero_when_frauds_exist`.
- **Rationale**: Guarantees relational `fraud_exposure` features update without temporal lookahead leakage while asserting non-zero exposure coverage on mule network records.

---
