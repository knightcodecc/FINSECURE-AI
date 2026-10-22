import sys
from pathlib import Path
from unittest.mock import MagicMock
sys.modules["cv2"] = MagicMock()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import joblib
from src.explain.shap_explainer import ShapExplainer, FEATURE_HUMAN_NAMES

model = joblib.load("models/xgboost_model.joblib")
metadata = joblib.load("models/model_metadata.joblib")
test_df = pd.read_parquet("data/features_test.parquet")
scaler = joblib.load("models/scaler.joblib")

feature_cols = metadata["feature_names"]
graph_cols = {
    "orig_in_degree", "orig_out_degree", "orig_pagerank", "orig_component_size", "orig_fraud_exposure",
    "dest_in_degree", "dest_out_degree", "dest_pagerank", "dest_component_size", "dest_fraud_exposure"
}

X_test = test_df[feature_cols]
y_test = test_df["isFraud"].values
probs = model.predict_proba(X_test)[:, 1]
thresh = metadata["f2_optimal_threshold"]

explainer = ShapExplainer()

print(f"=== SEARCHING TEST SPLIT ({len(test_df)} rows, {int(np.sum(y_test==1))} fraud) FOR DEMO SCENARIOS ===")

fraud_indices = np.where(y_test == 1)[0]
flagged_frauds = []

for idx in fraud_indices:
    row_df = X_test.iloc[[idx]]
    prob = probs[idx]
    if prob >= thresh:
        flagged_frauds.append((idx, prob))

print(f"\nFlagged Fraud rows count in test split: {len(flagged_frauds)}")
if flagged_frauds:
    print(f"Sample Flagged Fraud Test Row Index: {flagged_frauds[0][0]}, proba={flagged_frauds[0][1]:.4f}")

legit_indices = np.where(y_test == 0)[0]
best_b_idx = legit_indices[0]
print(f"SCENARIO B (Legitimate Cleared): Test row index {best_b_idx}, proba={probs[best_b_idx]:.4f}")

# Find real borderline test row (where proba is closest to thresh but below thresh)
below_thresh_indices = np.where(probs < thresh)[0]
if len(below_thresh_indices) > 0:
    borderline_idx = below_thresh_indices[np.argmax(probs[below_thresh_indices])]
    print(f"SCENARIO C (Borderline Case): Test row index {borderline_idx}, proba={probs[borderline_idx]:.4f}, thresh={thresh:.4f}")
