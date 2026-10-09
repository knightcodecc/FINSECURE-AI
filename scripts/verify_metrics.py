import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score
from src.cost_sensitive_eval import threshold_sweep, expected_cost

train_df = pd.read_parquet("data/features_train.parquet")
val_df = pd.read_parquet("data/features_val.parquet")
test_df = pd.read_parquet("data/features_test.parquet")
scaler = joblib.load("models/scaler.joblib")

feature_cols = [c for c in train_df.columns if c not in ("isFraud", "step")]
graph_cols = [
    "orig_in_degree", "orig_out_degree", "orig_pagerank", "orig_component_size", "orig_fraud_exposure",
    "dest_in_degree", "dest_out_degree", "dest_pagerank", "dest_component_size", "dest_fraud_exposure"
]
tabular_cols = [c for c in feature_cols if c not in graph_cols]

print("=== 1. DATA SPLIT SIZES & FRAUD COUNTS ===")
print(f"Train split: {len(train_df)} rows, {int(train_df['isFraud'].sum())} fraud rows ({train_df['isFraud'].mean()*100:.2f}%)")
print(f"Val split:   {len(val_df)} rows, {int(val_df['isFraud'].sum())} fraud rows ({val_df['isFraud'].mean()*100:.2f}%)")
print(f"Test split:  {len(test_df)} rows, {int(test_df['isFraud'].sum())} fraud rows ({test_df['isFraud'].mean()*100:.2f}%)")

X_train = train_df[feature_cols]
y_train = train_df["isFraud"].values
X_val = val_df[feature_cols]
y_val = val_df["isFraud"].values
X_test = test_df[feature_cols]
y_test = test_df["isFraud"].values

from src.train_model import unscale_column

amt_val = unscale_column(val_df, "amount", scaler)
amt_test = unscale_column(test_df, "amount", scaler)

n_neg = np.sum(y_train == 0)
n_pos = np.sum(y_train == 1)
scale_pos_weight = n_neg / max(n_pos, 1)

# Tabular + Graph Model (v2)
model_v2 = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.8, scale_pos_weight=scale_pos_weight, random_state=42,
    eval_metric="logloss", early_stopping_rounds=30
)
model_v2.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

val_proba_v2 = model_v2.predict_proba(X_val)[:, 1]
test_proba_v2 = model_v2.predict_proba(X_test)[:, 1]

_, f2_opt_v2, cost_opt_v2 = threshold_sweep(y_val, val_proba_v2, amt_val)

# Tabular Only Model
X_train_tab = train_df[tabular_cols]
X_val_tab = val_df[tabular_cols]
X_test_tab = test_df[tabular_cols]

model_tab = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.8, scale_pos_weight=scale_pos_weight, random_state=42,
    eval_metric="logloss", early_stopping_rounds=30
)
model_tab.fit(X_train_tab, y_train, eval_set=[(X_val_tab, y_val)], verbose=False)

val_proba_tab = model_tab.predict_proba(X_val_tab)[:, 1]
test_proba_tab = model_tab.predict_proba(X_test_tab)[:, 1]

_, f2_opt_tab, cost_opt_tab = threshold_sweep(y_val, val_proba_tab, amt_val)

print("\n=== 2. THRESHOLDS TUNED ON VALIDATION ===")
print(f"Tabular Only:    F2-Opt Thresh = {f2_opt_tab:.4f}, Cost-Opt Thresh = {cost_opt_tab:.4f}")
print(f"Tabular + Graph: F2-Opt Thresh = {f2_opt_v2:.4f}, Cost-Opt Thresh = {cost_opt_v2:.4f}")

print("\n=== 3. METRICS ON TEST SPLIT (TABULAR ONLY) ===")
pr_auc_tab = average_precision_score(y_test, test_proba_tab)
roc_auc_tab = roc_auc_score(y_test, test_proba_tab)
pred_f2_tab = (test_proba_tab >= f2_opt_v2).astype(int)
rec_tab = recall_score(y_test, pred_f2_tab, zero_division=0)
prec_tab = precision_score(y_test, pred_f2_tab, zero_division=0)
f2_res_tab = expected_cost(y_test, test_proba_tab, f2_opt_v2, amt_test)
cost_res_tab = expected_cost(y_test, test_proba_tab, cost_opt_v2, amt_test)

print(f"PR-AUC:                   {pr_auc_tab:.4f}")
print(f"ROC-AUC:                  {roc_auc_tab:.4f}")
print(f"Recall (at T_F2={f2_opt_v2:.4f}):   {rec_tab:.4f}")
print(f"Precision (at T_F2={f2_opt_v2:.4f}):{prec_tab:.4f}")
print(f"F2 Score (at T_F2={f2_opt_v2:.4f}): {f2_res_tab['f2_score']:.4f}")
print(f"Total Cost at T_F2:       Rs. {f2_res_tab['total_cost']:,.2f}")
print(f"Total Cost at T_cost:     Rs. {cost_res_tab['total_cost']:,.2f}")

print("\n=== 4. METRICS ON TEST SPLIT (TABULAR + GRAPH v2) ===")
pr_auc_v2 = average_precision_score(y_test, test_proba_v2)
roc_auc_v2 = roc_auc_score(y_test, test_proba_v2)
pred_f2_v2 = (test_proba_v2 >= f2_opt_v2).astype(int)
rec_v2 = recall_score(y_test, pred_f2_v2, zero_division=0)
prec_v2 = precision_score(y_test, pred_f2_v2, zero_division=0)
f2_res_v2 = expected_cost(y_test, test_proba_v2, f2_opt_v2, amt_test)
cost_res_v2 = expected_cost(y_test, test_proba_v2, cost_opt_v2, amt_test)

print(f"PR-AUC:                   {pr_auc_v2:.4f}")
print(f"ROC-AUC:                  {roc_auc_v2:.4f}")
print(f"Recall (at T_F2={f2_opt_v2:.4f}):   {rec_v2:.4f}")
print(f"Precision (at T_F2={f2_opt_v2:.4f}):{prec_v2:.4f}")
print(f"F2 Score (at T_F2={f2_opt_v2:.4f}): {f2_res_v2['f2_score']:.4f}")
print(f"Total Cost at T_F2:       Rs. {f2_res_v2['total_cost']:,.2f}")
print(f"Total Cost at T_cost:     Rs. {cost_res_v2['total_cost']:,.2f}")
