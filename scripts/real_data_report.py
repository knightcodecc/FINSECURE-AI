"""
scripts/real_data_report.py — Real-Data Evaluation & Ablation Report
======================================================================
Prints split sizes, fraud counts, full feature-group ablation (with CIs and validation-tuned thresholds),
no-accounting-error ablation, cost comparison against baselines (in raw rupees), and fraud_exposure coverage.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from src.cost_sensitive_eval import expected_cost


def bootstrap_pr_auc_and_cost_ci(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    amounts: np.ndarray,
    n_bootstraps: int = 200,
    seed: int = 42
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Compute 95% bootstrap confidence intervals for PR-AUC and expected cost."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    pr_scores, cost_scores = [], []
    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        yt_b, yp_b, amt_b = y_true[idx], y_proba[idx], amounts[idx]
        if len(np.unique(yt_b)) < 2:
            continue
        pr_scores.append(average_precision_score(yt_b, yp_b))
        cost_scores.append(expected_cost(yt_b, yp_b, threshold, amt_b, fp_cost=50.0)["total_cost"])
    
    pr_ci = (float(np.percentile(pr_scores, 2.5)), float(np.percentile(pr_scores, 97.5))) if pr_scores else (0.0, 0.0)
    cost_ci = (float(np.percentile(cost_scores, 2.5)), float(np.percentile(cost_scores, 97.5))) if cost_scores else (0.0, 0.0)
    return pr_ci, cost_ci


def tune_f2_threshold(y_val: np.ndarray, probas_val: np.ndarray) -> float:
    """Tune threshold optimizing F2 score on validation split."""
    best_t, best_f2 = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, 99):
        preds = (probas_val >= t).astype(int)
        tp = np.sum((preds == 1) & (y_val == 1))
        fp = np.sum((preds == 1) & (y_val == 0))
        fn = np.sum((preds == 0) & (y_val == 1))
        prec = tp / max(tp + fp, 1e-9)
        rec = tp / max(tp + fn, 1e-9)
        f2 = (5 * prec * rec) / max(4 * prec + rec, 1e-9)
        if f2 > best_f2:
            best_f2 = f2
            best_t = t
    return float(best_t)


def main():
    print("==========================================================================")
    print("         REAL DATA EVALUATION & FEATURE-GROUP ABLATION REPORT             ")
    print("==========================================================================")

    train_path = Path("data/features_train.parquet")
    val_path = Path("data/features_val.parquet")
    test_path = Path("data/features_test.parquet")
    shift_path = Path("data/features_shift.parquet")

    if not train_path.exists() or not val_path.exists() or not test_path.exists():
        print("Missing dataset feature files. Please run feature_engineering.py first.")
        return

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)
    shift_df = pd.read_parquet(shift_path) if shift_path.exists() else None

    print("\n=== 1. SPLIT SIZES & FRAUD COUNTS ===")
    print(f"Train split: {len(train_df):>7,} rows | Fraud: {train_df['isFraud'].sum():>5,} ({train_df['isFraud'].mean()*100:.2f}%)")
    print(f"Val split:   {len(val_df):>7,} rows | Fraud: {val_df['isFraud'].sum():>5,} ({val_df['isFraud'].mean()*100:.2f}%)")
    print(f"Clean Test:  {len(test_df):>7,} rows | Fraud: {test_df['isFraud'].sum():>5,} ({test_df['isFraud'].mean()*100:.2f}%)")
    if shift_df is not None and len(shift_df) > 0:
        print(f"Shift Split: {len(shift_df):>7,} rows | Fraud: {shift_df['isFraud'].sum():>5,} ({shift_df['isFraud'].mean()*100:.2f}%)")

    # Base Rate and Raw Rupee Amounts
    y_test = test_df["isFraud"].values
    base_rate = float(y_test.mean())
    amounts_test = test_df["amount_raw"].values if "amount_raw" in test_df.columns else test_df["amount"].values

    # Baselines
    cost_flag_nothing = float(np.sum(amounts_test[y_test == 1]))
    cost_flag_everything = float(50.0 * np.sum(y_test == 0))

    print(f"\n=== 2. BASELINE COSTS ON CLEAN TEST (N = {len(test_df):,}) ===")
    print(f"  - Base Rate:                 {base_rate:.4f}")
    print(f"  - No Model (Flag Nothing):   Rs. {cost_flag_nothing:,.2f}")
    print(f"  - Flag Everything Baseline:  Rs. {cost_flag_everything:,.2f}")

    # Define feature groups
    ignore_cols = {"isFraud", "step", "mechanism", "amount_raw"}
    all_feature_cols = [c for c in train_df.columns if c not in ignore_cols and pd.api.types.is_numeric_dtype(train_df[c])]

    raw_numeric = [c for c in ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"] if c in all_feature_cols]
    one_hot_type = [c for c in all_feature_cols if c.startswith("type_")]
    graph_cols = [c for c in all_feature_cols if "degree" in c or "pagerank" in c or "component" in c or "exposure" in c]
    tabular_cols = [c for c in all_feature_cols if c not in graph_cols]

    no_acct_err_tab = [c for c in tabular_cols if c not in ("errorBalanceOrig", "errorBalanceDest")]
    no_acct_err_graph = [c for c in all_feature_cols if c not in ("errorBalanceOrig", "errorBalanceDest")]

    variants = [
        ("Raw Numeric Only", raw_numeric),
        ("Raw + One-Hot Type", raw_numeric + one_hot_type),
        ("No-Accounting-Error Tabular", no_acct_err_tab),
        ("No-Accounting-Error + Graph", no_acct_err_graph),
        ("Tabular Only (23 cols)", tabular_cols),
        ("Tabular + Graph (33 cols)", all_feature_cols),
    ]

    print("\n=== 3. FEATURE-GROUP ABLATION STUDY ON CLEAN TEST SPLIT ===")

    y_val = val_df["isFraud"].values
    has_mechanism = "mechanism" in test_df.columns and "MECH_B_RING" in test_df["mechanism"].values

    eval_results = {}

    for v_name, v_cols in variants:
        if not v_cols:
            continue
        X_tr = train_df[v_cols]
        y_tr = train_df["isFraud"].values
        X_va = val_df[v_cols]
        X_te = test_df[v_cols]

        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1
        )
        model.fit(X_tr, y_tr)

        p_va = model.predict_proba(X_va)[:, 1]
        p_te = model.predict_proba(X_te)[:, 1]

        t_f2 = tune_f2_threshold(y_val, p_va)

        pr_auc = average_precision_score(y_test, p_te)
        roc_auc = roc_auc_score(y_test, p_te)
        pr_ci, cost_ci = bootstrap_pr_auc_and_cost_ci(y_test, p_te, t_f2, amounts_test)

        preds_te = (p_te >= t_f2).astype(int)
        rec = float(np.sum((preds_te == 1) & (y_test == 1)) / max(np.sum(y_test == 1), 1))

        if has_mechanism:
            ring_mask = (test_df["mechanism"] == "MECH_B_RING").values
            n_ring = int(np.sum(ring_mask))
            if n_ring > 0:
                ring_rec = float(np.sum((preds_te == 1) & ring_mask) / n_ring)
                ring_str = f"{ring_rec*100:.1f}% (n={n_ring})"
            else:
                ring_rec = None
                ring_str = "N/A"
        else:
            ring_rec = None
            ring_str = "N/A"

        cost_res = expected_cost(y_test, p_te, t_f2, amounts_test)
        tot_cost = float(cost_res["total_cost"])

        beats_fn = tot_cost < cost_flag_nothing
        beats_fe = tot_cost < cost_flag_everything

        eval_results[v_name] = {
            "pr_auc": pr_auc,
            "pr_ci": pr_ci,
            "roc_auc": roc_auc,
            "t_f2": t_f2,
            "cost": tot_cost,
            "cost_ci": cost_ci,
            "ring_rec": ring_rec,
            "beats_fn": beats_fn,
            "beats_fe": beats_fe,
        }

        print(f"[{v_name:<28}] PR-AUC: {pr_auc:.4f} (Base Rate: {base_rate:.4f}, CI: {pr_ci[0]:.4f}..{pr_ci[1]:.4f}) | ROC-AUC: {roc_auc:.4f} | T_F2: {t_f2:.4f} | Recall: {rec*100:.1f}% | Ring Recall: {ring_str} | Cost: Rs. {tot_cost:,.2f} | Beats FN: {beats_fn} | Beats FE: {beats_fe}")

    # 4. Fraud Exposure coverage
    print("\n=== 4. FRAUD EXPOSURE COVERAGE ON FRAUD ROWS ===")
    if "orig_fraud_exposure" in test_df.columns and "dest_fraud_exposure" in test_df.columns:
        fraud_test_df = test_df[test_df["isFraud"] == 1]
        n_fraud_tot = len(fraud_test_df)
        exp_mask = (fraud_test_df["orig_fraud_exposure"] > 0) | (fraud_test_df["dest_fraud_exposure"] > 0)
        n_exp = int(exp_mask.sum())
        pct_exp = (n_exp / max(n_fraud_tot, 1)) * 100
        print(f"Fraction of fraud rows in Clean Test with fraud_exposure > 0: {n_exp} / {n_fraud_tot} ({pct_exp:.2f}%)")
    else:
        n_exp, n_fraud_tot, pct_exp = 0, 0, 0.0

    # 5. Honest Computed Conclusion
    print("\n=== 5. DYNAMICALLY COMPUTED HONEST CONCLUSION ===")
    tab_res = eval_results.get("Tabular Only (23 cols)", {})
    graph_res = eval_results.get("Tabular + Graph (33 cols)", {})

    if tab_res and graph_res:
        g_pr, g_ci, g_cost, g_cost_ci = graph_res["pr_auc"], graph_res["pr_ci"], graph_res["cost"], graph_res["cost_ci"]
        t_pr, t_ci, t_cost, t_cost_ci = tab_res["pr_auc"], tab_res["pr_ci"], tab_res["cost"], tab_res["cost_ci"]

        if g_ci[0] > t_ci[1] or g_cost_ci[1] < t_cost_ci[0]:
            outcome = "lift"
        elif g_ci[1] < t_ci[0] or g_cost_ci[0] > t_cost_ci[1]:
            outcome = "worse"
        else:
            outcome = "no measurable lift"

        conclusion = f"Performance Comparison (Tabular + Graph vs Tabular Only): Graph features demonstrate {outcome.upper()} "
        conclusion += f"(PR-AUC: {g_pr:.4f} [{g_ci[0]:.4f}..{g_ci[1]:.4f}] vs {t_pr:.4f} [{t_ci[0]:.4f}..{t_ci[1]:.4f}], "
        conclusion += f"Expected Cost: Rs. {g_cost:,.2f} [{g_cost_ci[0]:,.2f}..{g_cost_ci[1]:,.2f}] vs Rs. {t_cost:,.2f} [{t_cost_ci[0]:,.2f}..{t_cost_ci[1]:,.2f}])."

        if graph_res.get("ring_rec") is not None and tab_res.get("ring_rec") is not None:
            ring_diff = (graph_res["ring_rec"] - tab_res["ring_rec"]) * 100
            conclusion += f" Ring-fraud recall difference: +{ring_diff:.1f}% ({graph_res['ring_rec']*100:.1f}% vs {tab_res['ring_rec']*100:.1f}%)."

        conclusion += f" Fraud exposure coverage: {n_exp} / {n_fraud_tot} ({pct_exp:.2f}%) of Clean Test fraud rows have orig_ or dest_fraud_exposure > 0."
        print(conclusion)

    print("==========================================================================\n")


if __name__ == "__main__":
    main()
