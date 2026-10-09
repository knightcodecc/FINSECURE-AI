"""
feature_engineering.py
=======================
Feature engineering pipeline for FinSecure AI v2.

Pipeline position:
    preprocess.py -> feature_engineering.py -> train_model.py
"""

import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

CLEANED_DATA_PATH = Path("data/cleaned_transactions.parquet")
SCALER_PATH = Path("models/scaler.joblib")
TRAIN_FEATURES_PATH = Path("data/features_train.parquet")
VAL_FEATURES_PATH = Path("data/features_val.parquet")
TEST_FEATURES_PATH = Path("data/features_test.parquet")
SHIFT_FEATURES_PATH = Path("data/features_shift.parquet")


def engineer_tabular_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Port 15 hand-engineered tabular features.
    """
    df = df.copy()

    # 1. Balance discrepancy features
    df["errorBalanceOrig"] = df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
    df["errorBalanceDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]

    # 2. Log-transformed transaction amount
    df["log_amount"] = np.log1p(df["amount"])

    # 3. Account draining indicator
    df["account_drained"] = ((df["oldbalanceOrg"] > 0) & (df["newbalanceOrig"] == 0)).astype(float)

    # 4. Zero destination balance flags
    df["dest_balance_zero"] = ((df["oldbalanceDest"] == 0) & (df["newbalanceDest"] == 0)).astype(float)

    # 5. Zero origin balance flag
    df["orig_balance_zero"] = ((df["oldbalanceOrg"] == 0) & (df["newbalanceOrig"] == 0)).astype(float)

    # 6. Amount relative to account balances
    df["amount_to_oldbalanceOrg_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1.0)
    df["amount_to_oldbalanceDest_ratio"] = df["amount"] / (df["oldbalanceDest"] + 1.0)

    # 7. Temporal features
    if "step" in df.columns:
        df["hour_of_day"] = df["step"] % 24
        df["high_risk_hour"] = ((df["hour_of_day"] >= 0) & (df["hour_of_day"] <= 5)).astype(float)
        df["day_of_week"] = (df["step"] // 24) % 7

    # 8. Net balance delta features
    df["net_change_orig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["net_change_dest"] = df["newbalanceDest"] - df["oldbalanceDest"]

    return df


def split_and_scale_features(df: pd.DataFrame):
    """
    Perform 4-window chronological split using step boundaries and fit StandardScaler strictly on TRAIN.
    
    Windows:
    1. train: steps 1..420
    2. val: steps 421..540
    3. clean_test: steps 541..629 (CLEAN TEST)
    4. shift: steps 630..743 (SHIFT WINDOW)
    """
    if "step" in df.columns:
        df = df.sort_values("step").reset_index(drop=True)

    unique_steps = np.sort(df["step"].unique())
    n_steps = len(unique_steps)

    max_step = df["step"].max() if "step" in df.columns else 0
    splits_env = os.getenv("TRAIN_VAL_TEST_SPLITS")

    if splits_env:
        ratios = [float(r.strip()) for r in splits_env.split(",")]
        if len(ratios) == 4:
            c1, c2, c3 = np.cumsum(ratios)[:3]
            t_cutoff = np.quantile(unique_steps, c1)
            v_cutoff = np.quantile(unique_steps, c2)
            c_cutoff = np.quantile(unique_steps, c3)
        else:
            c1, c2 = np.cumsum(ratios)[:2]
            t_cutoff = np.quantile(unique_steps, c1)
            v_cutoff = np.quantile(unique_steps, c2)
            c_cutoff = unique_steps[-1]
    elif max_step >= 700 and n_steps >= 200:
        t_cutoff = 420
        v_cutoff = 540
        c_cutoff = 629
    else:
        t_cutoff = np.quantile(unique_steps, 0.60)
        v_cutoff = np.quantile(unique_steps, 0.75)
        c_cutoff = np.quantile(unique_steps, 0.875)

    train_df = df[df["step"] <= t_cutoff].copy()
    val_df = df[(df["step"] > t_cutoff) & (df["step"] <= v_cutoff)].copy()
    test_df = df[(df["step"] > v_cutoff) & (df["step"] <= c_cutoff)].copy()
    shift_df = df[df["step"] > c_cutoff].copy()

    ignore_cols = {"isFraud", "step", "mechanism", "amount_raw"}
    feature_cols = [c for c in df.columns if c not in ignore_cols and pd.api.types.is_numeric_dtype(df[c])]

    print("==========================================================================")
    print("      FOUR-WINDOW CHRONOLOGICAL SPLIT SUMMARY & MECHANISM BREAKDOWN      ")
    print("==========================================================================")
    for name, split in [("Train", train_df), ("Val", val_df), ("Clean Test", test_df), ("Shift Window", shift_df)]:
        n_tot = len(split)
        n_fraud = int(split["isFraud"].sum())
        fr_pct = (n_fraud / max(n_tot, 1)) * 100
        if "mechanism" in split.columns:
            mech_counts = split["mechanism"].value_counts().to_dict()
            mech_str = f"Mechs: A={mech_counts.get('MECH_A_DRAIN', 0):,}, B={mech_counts.get('MECH_B_RING', 0):,}, C={mech_counts.get('MECH_C_SHIFT', 0):,}"
        else:
            mech_str = "Mechs: N/A (real data)"
        step_min = split['step'].min() if len(split) > 0 else 0
        step_max = split['step'].max() if len(split) > 0 else 0
        print(f"{name:<12} | Steps: {step_min:>3}..{step_max:<3} | Rows: {n_tot:>7,} | Fraud: {n_fraud:>5,} ({fr_pct:>5.2f}%) | {mech_str}")
    print("==========================================================================\n")

    # Fit StandardScaler ONLY on train_df[feature_cols]
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])

    # Transform all splits using TRAIN-fitted scaler
    train_df[feature_cols] = scaler.transform(train_df[feature_cols])
    val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    shift_df[feature_cols] = scaler.transform(shift_df[feature_cols])

    return train_df, val_df, test_df, shift_df, scaler


def main():
    print(f"Loading cleaned dataset from {CLEANED_DATA_PATH}...")
    if not CLEANED_DATA_PATH.exists():
        raise FileNotFoundError(f"Cleaned data file not found at {CLEANED_DATA_PATH}. Run preprocess.py first.")

    df_clean = pd.read_parquet(CLEANED_DATA_PATH)

    print("Engineering 15 tabular features...")
    df_engineered = engineer_tabular_features(df_clean)

    print("Performing 4-window split and fitting StandardScaler on TRAIN split only...")
    train_df, val_df, test_df, shift_df, scaler = split_and_scale_features(df_engineered)

    SCALER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Fitted scaler saved to {SCALER_PATH}")

    train_df.to_parquet(TRAIN_FEATURES_PATH, index=False)
    val_df.to_parquet(VAL_FEATURES_PATH, index=False)
    test_df.to_parquet(TEST_FEATURES_PATH, index=False)
    shift_df.to_parquet(SHIFT_FEATURES_PATH, index=False)
    print(f"Saved feature splits to:\n  - {TRAIN_FEATURES_PATH}\n  - {VAL_FEATURES_PATH}\n  - {TEST_FEATURES_PATH}\n  - {SHIFT_FEATURES_PATH}")


if __name__ == "__main__":
    main()
