"""
scripts/leakage_check.py
========================
Audit synthetic dataset for feature leakage and single-feature predictive dominance.
Prints:
1. Single-feature AUC for every one of the 33 model columns on train split.
2. AUC of a shallow DecisionTree (max_depth=3) on raw numeric columns only.
3. AUC after shuffling labels on train split (must be ~0.50).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

TRAIN_FEATURES_PATH = Path("data/features_train.parquet")

def run_leakage_check():
    if not TRAIN_FEATURES_PATH.exists():
        print(f"Error: {TRAIN_FEATURES_PATH} not found. Please run feature pipeline first.")
        return

    train_df = pd.read_parquet(TRAIN_FEATURES_PATH)
    y_train = train_df["isFraud"].values
    base_rate = np.mean(y_train)

    ignore_cols = {"isFraud", "step", "mechanism"}
    feature_cols = [c for c in train_df.columns if c not in ignore_cols and pd.api.types.is_numeric_dtype(train_df[c])]
    raw_cols = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]

    print("=================================================================")
    print(f"LEAKAGE CHECK AUDIT ON TRAIN SPLIT ({len(train_df):,} rows, {np.sum(y_train):,} fraud | Base Rate: {base_rate:.4f})")
    print("=================================================================")

    # 1. Single-Feature ROC-AUC and PR-AUC
    print("\n1. SINGLE-FEATURE AUC PERFORMANCES (All 33 Columns):")
    print(f"{'Feature Name':<35} | {'ROC-AUC':<10} | {'PR-AUC':<10} | {'Leakage Warning?'}")
    print("-" * 75)

    leaking_cols = []
    for col in feature_cols:
        vals = train_df[col].values
        if np.std(vals) == 0:
            roc = 0.5000
            pr = base_rate
        else:
            roc = roc_auc_score(y_train, vals)
            if roc < 0.5:
                roc = roc_auc_score(y_train, -vals)
                pr = average_precision_score(y_train, -vals)
            else:
                pr = average_precision_score(y_train, vals)
        
        is_leak = (roc > 0.95 or pr > 0.80)
        leak_str = "LEAKING!" if is_leak else "OK"
        if is_leak:
            leaking_cols.append((col, roc, pr))
            
        print(f"{col:<35} | {roc:<10.4f} | {pr:<10.4f} | {leak_str}")

    # 2. Shallow Tree on Raw Numeric Only
    X_raw = train_df[raw_cols].values
    dt = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt.fit(X_raw, y_train)
    raw_tree_prob = dt.predict_proba(X_raw)[:, 1]
    raw_tree_roc = roc_auc_score(y_train, raw_tree_prob)
    raw_tree_pr = average_precision_score(y_train, raw_tree_prob)

    print("\n2. SHALLOW DECISION TREE (max_depth=3) ON RAW NUMERIC ONLY:")
    print(f"   - Features used: {raw_cols}")
    print(f"   - ROC-AUC: {raw_tree_roc:.4f}")
    print(f"   - PR-AUC:  {raw_tree_pr:.4f} (Base Rate: {base_rate:.4f})")

    # 3. Label Shuffling Sanity Check
    rng = np.random.RandomState(42)
    y_shuffled = rng.permutation(y_train)
    dt_shuffled = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt_shuffled.fit(X_raw, y_shuffled)
    shuffled_prob = dt_shuffled.predict_proba(X_raw)[:, 1]
    shuffled_roc = roc_auc_score(y_shuffled, shuffled_prob)
    shuffled_pr = average_precision_score(y_shuffled, shuffled_prob)

    print("\n3. LABEL SHUFFLE SANITY CHECK:")
    print(f"   - ROC-AUC with shuffled labels: {shuffled_roc:.4f} (Expected ~0.50)")
    print(f"   - PR-AUC with shuffled labels:  {shuffled_pr:.4f} (Expected ~base rate {base_rate:.4f})")

    print("\n=================================================================")
    print("LEAKAGE SUMMARY:")
    if leaking_cols:
        print(f"Detected {len(leaking_cols)} leaking columns (ROC > 0.95 or PR > 0.80):")
        for col, r, p in leaking_cols:
            print(f"  - {col}: ROC-AUC={r:.4f}, PR-AUC={p:.4f}")
    else:
        print("No individual leaking columns detected!")
    print("=================================================================\n")

if __name__ == "__main__":
    run_leakage_check()
