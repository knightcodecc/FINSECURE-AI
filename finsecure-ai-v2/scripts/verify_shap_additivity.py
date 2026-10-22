"""
scripts/verify_shap_additivity.py
===================================
Verify SHAP TreeExplainer margin-space additivity property over 200 rows with 30+ fraud rows.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock
sys.modules["cv2"] = MagicMock()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import shap

from src.explain.shap_explainer import ShapExplainer

def main():
    print("=== SHAP ADDITIVITY VERIFICATION ===")
    print(f"XGBoost version: {xgb.__version__}")
    print(f"SHAP version:    {shap.__version__}")

    train_df = pd.read_parquet("data/features_train.parquet")
    val_df = pd.read_parquet("data/features_val.parquet")
    test_df = pd.read_parquet("data/features_test.parquet")

    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    ignore_cols = {"isFraud", "step", "mechanism", "amount_raw"}
    feature_cols = [c for c in full_df.columns if c not in ignore_cols and pd.api.types.is_numeric_dtype(full_df[c])]

    fraud_df = full_df[full_df["isFraud"] == 1]
    non_fraud_df = full_df[full_df["isFraud"] == 0]

    n_fraud = min(len(fraud_df), 40)
    sample_fraud = fraud_df.sample(n_fraud, random_state=42)
    sample_non_fraud = non_fraud_df.sample(200 - n_fraud, random_state=42)

    sample_200 = pd.concat([sample_fraud, sample_non_fraud]).sample(frac=1.0, random_state=42)
    X_200 = sample_200[feature_cols]

    print(f"Sampled 200 rows: {n_fraud} fraud rows, {200 - n_fraud} non-fraud rows")

    shap_explainer = ShapExplainer()
    additivity_valid = shap_explainer.verify_additivity(X_200, tol=1e-4)

    shap_res = shap_explainer.explainer(X_200)
    base_values = shap_res.base_values
    if np.ndim(base_values) == 0:
        base_values = np.full(len(X_200), float(base_values))
    shap_vals_matrix = shap_res.values
    if np.ndim(shap_vals_matrix) == 3:
        shap_vals_matrix = shap_vals_matrix[:, :, 1]
        base_values = base_values[:, 1]

    sum_shap = np.sum(shap_vals_matrix, axis=1)
    predicted_margins = base_values + sum_shap
    raw_margins = shap_explainer.model.predict(X_200, output_margin=True)
    max_diff = float(np.max(np.abs(predicted_margins - raw_margins)))

    print(f"Base Value (margin space): {float(np.mean(base_values)):.6f}")
    print(f"Max Absolute Additivity Difference |base + sum(SHAP) - margin_predict|: {max_diff:.8e}")
    print(f"Additivity Status (tolerance < 1e-4): {'PASS' if additivity_valid else 'FAIL'}")

if __name__ == "__main__":
    main()
