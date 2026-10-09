"""
scripts/verify_drift_psi.py
============================
Compute and print the full per-feature Population Stability Index (PSI) table across 33 model columns for:
1. Reference Window: Train Split (steps 1..420, 169,746 rows)
2. Clean Test Window: Clean Test Split (steps 541..629, 35,954 rows)
3. Shift Window: Shift Split (steps 630..743, 45,984 rows)
4. Amount x5 Window: Clean Test Split with amount * 5 (35,954 rows)

Thresholds used: DRIFT_PSI_STABLE = 0.10, DRIFT_PSI_WARNING = 0.25
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from src.drift_monitor import DriftMonitor, EXCLUDED_PSI_COLS

TRAIN_PATH = Path("data/features_train.parquet")
VAL_PATH = Path("data/features_val.parquet")
TEST_PATH = Path("data/features_test.parquet")
SHIFT_PATH = Path("data/features_shift.parquet")
MODEL_PATH = Path("models/xgboost_model.joblib")


def main():
    print("Loading datasets and deployed XGBoost model...")
    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)
    shift_df = pd.read_parquet(SHIFT_PATH)
    model = joblib.load(MODEL_PATH)

    ignore_cols = {"isFraud", "step", "mechanism", "amount_raw"}
    feature_cols = [c for c in train_df.columns if c not in ignore_cols and pd.api.types.is_numeric_dtype(train_df[c])]
    assert len(feature_cols) == 33, f"Expected 33 feature columns, got {len(feature_cols)}"

    monitor = DriftMonitor(train_df)

    # 1. Clean Test Window (steps 541..629)
    clean_probas = model.predict_proba(test_df[feature_cols])[:, 1]
    res_clean_test = monitor.check(test_df, recent_probas=clean_probas)

    # 2. Shift Window (steps 630..743)
    shift_probas = model.predict_proba(shift_df[feature_cols])[:, 1]
    res_shift = monitor.check(shift_df, recent_probas=shift_probas)

    scaler = joblib.load(Path("models/scaler.joblib"))

    # 3. Amount x5 Window (Clean test split with amount_raw * 5.0 and recomputed dependent features)
    amt_x5_df = test_df.copy()
    raw_amt = amt_x5_df["amount_raw"] * 5.0
    amt_x5_df["amount_raw"] = raw_amt

    feature_names = list(scaler.feature_names_in_)
    def get_raw(col_name):
        idx = feature_names.index(col_name)
        return amt_x5_df[col_name] * scaler.scale_[idx] + scaler.mean_[idx]

    raw_old_orig = get_raw("oldbalanceOrg")
    raw_new_orig = get_raw("newbalanceOrig")
    raw_old_dest = get_raw("oldbalanceDest")
    raw_new_dest = get_raw("newbalanceDest")

    raw_err_orig = raw_new_orig + raw_amt - raw_old_orig
    raw_err_dest = raw_old_dest + raw_amt - raw_new_dest
    raw_log_amt = np.log1p(raw_amt)
    raw_ratio_orig = raw_amt / (raw_old_orig + 1.0)
    raw_ratio_dest = raw_amt / (raw_old_dest + 1.0)

    def set_scaled(col_name, raw_vec):
        idx = feature_names.index(col_name)
        amt_x5_df[col_name] = (raw_vec - scaler.mean_[idx]) / scaler.scale_[idx]

    set_scaled("amount", raw_amt)
    set_scaled("log_amount", raw_log_amt)
    set_scaled("errorBalanceOrig", raw_err_orig)
    set_scaled("errorBalanceDest", raw_err_dest)
    set_scaled("amount_to_oldbalanceOrg_ratio", raw_ratio_orig)
    set_scaled("amount_to_oldbalanceDest_ratio", raw_ratio_dest)

    x5_probas = model.predict_proba(amt_x5_df[feature_cols])[:, 1]
    res_amt_x5 = monitor.check(amt_x5_df, recent_probas=x5_probas)

    print("\n===================================================================================================================")
    print("FULL PER-FEATURE POPULATION STABILITY INDEX (PSI) TABLE Across 33 MODEL COLUMNS")
    print("===================================================================================================================")
    print(f"Drift Thresholds: DRIFT_PSI_STABLE = 0.10 | DRIFT_PSI_WARNING = 0.25")
    print(f"1. Reference:       Train Split (steps {train_df['step'].min()}..{train_df['step'].max()}, {len(train_df):,} rows)")
    print(f"2. Clean Test:      Clean Test Split (steps {test_df['step'].min()}..{test_df['step'].max()}, {len(test_df):,} rows)")
    print(f"3. Shift Window:    Shift Split (steps {shift_df['step'].min()}..{shift_df['step'].max()}, {len(shift_df):,} rows)")
    print(f"4. Amount x5:       Clean Test Split (amount * 5, {len(amt_x5_df):,} rows)")
    print("-" * 115)
    print(f"{'Feature Name':<33} | {'PSI Inclusion Status':<22} | {'Clean Test':<12} | {'Shift Window':<12} | {'Amount x5':<12}")
    print("-" * 115)

    for col in feature_cols:
        status_str = res_clean_test['feature_status'].get(col, "INCLUDED")
        if status_str in ("STABLE", "WARNING", "DRIFT"):
            inc_str = "INCLUDED"
        else:
            inc_str = status_str
        psi_clean = res_clean_test['feature_psi'].get(col, 0.0)
        psi_shift = res_shift['feature_psi'].get(col, 0.0)
        psi_x5 = res_amt_x5['feature_psi'].get(col, 0.0)
        print(f"{col:<33} | {inc_str:<22} | {psi_clean:<12.4f} | {psi_shift:<12.4f} | {psi_x5:<12.4f}")

    print("-" * 115)
    print(f"{'REAL MODEL SCORE PSI':<33} | {'SCORE_PSI':<22} | {res_clean_test['score_psi']:<12.4f} | {res_shift['score_psi']:<12.4f} | {res_amt_x5['score_psi']:<12.4f}")
    print(f"{'OVERALL DRIFT STATUS':<33} | {'SUMMARY':<22} | {res_clean_test['overall_status']:<12} | {res_shift['overall_status']:<12} | {res_amt_x5['overall_status']:<12}")
    print(f"{'RETRAIN RECOMMENDED?':<33} | {'FLAG':<22} | {str(res_clean_test['retrain_recommended']):<12} | {str(res_shift['retrain_recommended']):<12} | {str(res_amt_x5['retrain_recommended']):<12}")
    print("===================================================================================================================\n")

if __name__ == "__main__":
    main()
