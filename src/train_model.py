"""
train_model.py
===============
Train XGBoost variants, perform validation-only threshold tuning (F2-optimal and cost-optimal),
evaluate metrics on held-out CLEAN TEST split with model-specific thresholds, compute bootstrap 95% CIs,
and compare against Flag-Nothing and Flag-Everything baselines.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score

from src.cost_sensitive_eval import expected_cost, threshold_sweep

TRAIN_FEATURES_PATH = Path("data/features_train.parquet")
VAL_FEATURES_PATH = Path("data/features_val.parquet")
TEST_FEATURES_PATH = Path("data/features_test.parquet")
SCALER_PATH = Path("models/scaler.joblib")
MODEL_PATH = Path("models/xgboost_model.joblib")
METADATA_PATH = Path("models/model_metadata.joblib")
ABLATION_PATH = Path("reports/ablation_graph.md")
COST_REPORT_PATH = Path("reports/cost_eval.md")
COST_CSV_PATH = Path("reports/cost_comparison.csv")
COST_SENSITIVITY_CSV_PATH = Path("reports/cost_sensitivity.csv")


def unscale_column(df: pd.DataFrame, col_name: str, scaler) -> np.ndarray:
    """Unscale a specific column using the fitted StandardScaler."""
    feature_names = scaler.feature_names_in_ if hasattr(scaler, "feature_names_in_") else None
    if feature_names is not None and col_name in feature_names:
        idx = list(feature_names).index(col_name)
    else:
        idx = 0
    mean = scaler.mean_[idx]
    scale = scaler.scale_[idx]
    return np.maximum(0.0, df[col_name].values * scale + mean)


def evaluate_variant(
    name: str,
    cols: list[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    amt_val: np.ndarray,
    amt_test: np.ndarray,
    scale_pos_weight: float,
    rng_seed: int = 42,
) -> dict:
    """Train variant model, tune OWN validation thresholds, evaluate on held-out CLEAN TEST split, and compute bootstrap 95% CIs."""
    X_train = train_df[cols]
    y_train = train_df["isFraud"].values
    X_val = val_df[cols]
    y_val = val_df["isFraud"].values
    X_test = test_df[cols]
    y_test = test_df["isFraud"].values

    clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=rng_seed,
        eval_metric="logloss",
        early_stopping_rounds=30,
    )
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    val_prob = clf.predict_proba(X_val)[:, 1]
    test_prob = clf.predict_proba(X_test)[:, 1]

    # Tune OWN validation thresholds
    _, f2_opt, cost_opt = threshold_sweep(y_val, val_prob, amt_val, fp_cost=50.0, fn_multiplier=1.0)

    # Clean Test set evaluation
    base_rate = float(np.mean(y_test))
    pr_auc = float(average_precision_score(y_test, test_prob))
    roc_auc = float(roc_auc_score(y_test, test_prob)) if len(np.unique(y_test)) > 1 else 0.5
    flag_rate = float(np.mean(test_prob >= f2_opt))

    cost_res_f2 = expected_cost(y_test, test_prob, f2_opt, amt_test, fp_cost=50.0, fn_multiplier=1.0)
    cost_res_cost = expected_cost(y_test, test_prob, cost_opt, amt_test, fp_cost=50.0, fn_multiplier=1.0)

    # 200 Bootstrap resamples for 95% CIs on CLEAN TEST split
    rng = np.random.RandomState(42)
    n = len(y_test)
    boot_pr, boot_cost, boot_ring = [], [], []
    
    is_ring_mask = (test_df["mechanism"] == "MECH_B_RING").values if "mechanism" in test_df.columns else np.zeros(n, dtype=bool)
    n_ring = int(np.sum(is_ring_mask))

    for _ in range(200):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(y_test[idx])) < 2:
            continue
        boot_pr.append(average_precision_score(y_test[idx], test_prob[idx]))
        boot_cost.append(expected_cost(y_test[idx], test_prob[idx], f2_opt, amt_test[idx], fp_cost=50.0, fn_multiplier=1.0)["total_cost"])
        
        ring_idx = idx[is_ring_mask[idx]]
        if len(ring_idx) > 0:
            boot_ring.append(np.mean(test_prob[ring_idx] >= f2_opt))

    pr_ci = (float(np.percentile(boot_pr, 2.5)), float(np.percentile(boot_pr, 97.5))) if boot_pr else (pr_auc, pr_auc)
    cost_ci = (float(np.percentile(boot_cost, 2.5)), float(np.percentile(boot_cost, 97.5))) if boot_cost else (cost_res_f2["total_cost"], cost_res_f2["total_cost"])

    if n_ring > 0:
        ring_preds = (test_prob[is_ring_mask] >= f2_opt).astype(int)
        ring_recall = float(np.mean(ring_preds))
        ring_ci = (float(np.percentile(boot_ring, 2.5)), float(np.percentile(boot_ring, 97.5))) if boot_ring else (ring_recall, ring_recall)
    else:
        ring_recall = None
        ring_ci = (None, None)

    return {
        "name": name,
        "n_cols": len(cols),
        "f2_opt_thresh": f2_opt,
        "cost_opt_thresh": cost_opt,
        "base_rate": base_rate,
        "pr_auc": pr_auc,
        "pr_ci": pr_ci,
        "roc_auc": roc_auc,
        "flag_rate": flag_rate,
        "precision_f2": cost_res_f2["precision"],
        "recall_f2": cost_res_f2["recall"],
        "n_ring": n_ring,
        "ring_recall": ring_recall,
        "ring_ci": ring_ci,
        "f2_score": cost_res_f2["f2_score"],
        "total_cost_f2": cost_res_f2["total_cost"],
        "cost_ci": cost_ci,
        "amt_caught": cost_res_f2["amount_caught"],
        "amt_missed": cost_res_f2["amount_missed"],
        "pct_caught": cost_res_f2["pct_amount_caught"],
        "total_cost_cost_opt": cost_res_cost["total_cost"],
        "clf": clf,
        "test_prob": test_prob,
    }


def train_and_evaluate():
    print("Loading feature splits...")
    train_df = pd.read_parquet(TRAIN_FEATURES_PATH)
    val_df = pd.read_parquet(VAL_FEATURES_PATH)
    test_df = pd.read_parquet(TEST_FEATURES_PATH)
    scaler = joblib.load(SCALER_PATH)

    ignore_cols = {"isFraud", "step", "mechanism", "amount_raw"}
    feature_cols = [c for c in train_df.columns if c not in ignore_cols and pd.api.types.is_numeric_dtype(train_df[c])]
    assert len(feature_cols) == 33, f"Expected 33 feature columns, got {len(feature_cols)}"

    amt_val = val_df["amount_raw"].values if "amount_raw" in val_df.columns else unscale_column(val_df, "amount", scaler)
    amt_test = test_df["amount_raw"].values if "amount_raw" in test_df.columns else unscale_column(test_df, "amount", scaler)

    y_train = train_df["isFraud"].values
    y_val = val_df["isFraud"].values
    y_test = test_df["isFraud"].values

    base_rate_test = float(np.mean(y_test))

    n_neg = np.sum(y_train == 0)
    n_pos = np.sum(y_train == 1)
    scale_pos_weight = n_neg / max(n_pos, 1)

    print("\n=== SPLIT SIZES & FRAUD COUNTS ===")
    print(f"Train split: {len(train_df):,} rows | Fraud: {np.sum(y_train==1):,} ({np.mean(y_train)*100:.2f}%)")
    print(f"Val split:   {len(val_df):,} rows | Fraud: {np.sum(y_val==1):,} ({np.mean(y_val)*100:.2f}%)")
    print(f"Clean Test:  {len(test_df):,} rows | Fraud: {np.sum(y_test==1):,} ({base_rate_test*100:.2f}%)")

    # Define feature groups for complete ablation
    raw_cols = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
    type_cols = [c for c in feature_cols if c.startswith("type_")]
    raw_type_cols = raw_cols + type_cols
    
    graph_cols = [c for c in feature_cols if c.startswith(("orig_", "dest_")) and c not in ("nameOrig", "nameDest")]
    tabular_cols = [c for c in feature_cols if c not in graph_cols]
    
    no_error_tabular_cols = [c for c in tabular_cols if c not in ("errorBalanceOrig", "errorBalanceDest")]
    no_error_graph_cols = [c for c in feature_cols if c not in ("errorBalanceOrig", "errorBalanceDest")]

    variants = [
        ("Raw Numeric Only", raw_cols),
        ("Raw + One-Hot Type", raw_type_cols),
        ("No-Accounting-Error Tabular", no_error_tabular_cols),
        ("No-Accounting-Error + Graph", no_error_graph_cols),
        ("Tabular Only (23 cols)", tabular_cols),
        ("Tabular + Graph (33 cols)", feature_cols),
    ]

    print("\n--- RUNNING FULL FEATURE GROUP ABLATION STUDY ON CLEAN TEST SPLIT ---")
    results = []
    for name, cols in variants:
        res = evaluate_variant(name, cols, train_df, val_df, test_df, amt_val, amt_test, scale_pos_weight)
        results.append(res)
        ring_str = f"{res['ring_recall']*100:.1f}% (CI: {res['ring_ci'][0]*100:.1f}..{res['ring_ci'][1]*100:.1f}%, n={res['n_ring']})" if res['ring_recall'] is not None else "N/A"
        print(f"[{name:<28}] PR-AUC: {res['pr_auc']:.4f} (Base Rate: {res['base_rate']:.4f}, CI: {res['pr_ci'][0]:.4f}..{res['pr_ci'][1]:.4f}) | ROC-AUC: {res['roc_auc']:.4f} | T_F2: {res['f2_opt_thresh']:.4f} | Overall Recall: {res['recall_f2']*100:.1f}% | Flag Rate: {res['flag_rate']*100:.1f}% | Ring-Fraud Recall: {ring_str} | Cost: Rs. {res['total_cost_f2']:,.2f}")

    # Calculate Baselines for Cost Table
    no_model_cost = float(np.sum(amt_test[y_test == 1]))  # Flag Nothing: FN = all fraud amounts
    flag_everything_cost = float(50.0 * np.sum(y_test == 0)) # Flag Everything: FP = all non-fraud * 50.0

    print("\n=== BASELINE COSTS ===")
    print(f"  - No Model (Flag Nothing) : Rs. {no_model_cost:,.2f}")
    print(f"  - Flag Everything Baseline : Rs. {flag_everything_cost:,.2f}")
    for r in results:
        beats_nothing = r["total_cost_f2"] < no_model_cost
        beats_everything = r["total_cost_f2"] < flag_everything_cost
        print(f"  - {r['name']:<28}: Cost = Rs. {r['total_cost_f2']:>12,.2f} | Beats Flag Nothing? {beats_nothing} | Beats Flag Everything? {beats_everything}")

    # Primary deployed model is Tabular + Graph
    deployed = results[-1]
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(deployed["clf"], MODEL_PATH)

    metadata = {
        "feature_names": feature_cols,
        "f2_optimal_threshold": deployed["f2_opt_thresh"],
        "cost_optimal_threshold": deployed["cost_opt_thresh"],
        "active_threshold": deployed["cost_opt_thresh"],
        "active_strategy": "cost_optimal",
        "model_expected_cost_rs": deployed["total_cost_cost_opt"],
        "baseline_flag_nothing_cost_rs": no_model_cost,
        "baseline_flag_everything_cost_rs": flag_everything_cost,
        "cost_savings_vs_flag_nothing_rs": no_model_cost - deployed["total_cost_cost_opt"],
        "cost_savings_vs_flag_everything_rs": flag_everything_cost - deployed["total_cost_cost_opt"],
    }
    joblib.dump(metadata, METADATA_PATH)

    # Build Ablation Markdown Report
    ablation_md = f"""# Full Feature Group Ablation & Impact Analysis (Clean Test Split)

Base Rate (Clean Test): **{base_rate_test:.4f}** ({base_rate_test*100:.2f}%)
Baseline Costs:
- **No Model (Flag Nothing)**: Rs. {no_model_cost:,.2f}
- **Flag Everything**: Rs. {flag_everything_cost:,.2f}

| Model Variant | Feature Count | Base Rate | Validation T_F2 | PR-AUC (95% CI) | ROC-AUC | Overall Recall | Flag Rate | Ring-Fraud Recall (n={test_df['mechanism'].eq('MECH_B_RING').sum() if 'mechanism' in test_df.columns else 0}) | Expected Cost (Rs.) | Beats Both Baselines? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **No Model (Flag Nothing)** | 0 | {base_rate_test:.4f} | N/A | N/A | N/A | 0.00% | 0.00% | N/A | Rs. {no_model_cost:,.2f} | N/A |
| **Flag Everything** | 0 | {base_rate_test:.4f} | 0.0000 | {base_rate_test:.4f} | 0.5000 | 100.00% | 100.00% | 100.00% | Rs. {flag_everything_cost:,.2f} | N/A |
"""
    for r in results:
        beats = (r["total_cost_f2"] < no_model_cost) and (r["total_cost_f2"] < flag_everything_cost)
        ring_str = f"{r['ring_recall']*100:.2f}% ({r['ring_ci'][0]*100:.1f}%..{r['ring_ci'][1]*100:.1f}%)" if r['ring_recall'] is not None else "N/A"
        ablation_md += f"| **{r['name']}** | {r['n_cols']} | {r['base_rate']:.4f} | {r['f2_opt_thresh']:.4f} | {r['pr_auc']:.4f} ({r['pr_ci'][0]:.4f}..{r['pr_ci'][1]:.4f}) | {r['roc_auc']:.4f} | {r['recall_f2']*100:.2f}% | {r['flag_rate']*100:.2f}% | {ring_str} | Rs. {r['total_cost_f2']:,.2f} | {'YES' if beats else 'NO'} |\n"

    ABLATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    ABLATION_PATH.write_text(ablation_md, encoding="utf-8")
    print(f"\nSaved report to {ABLATION_PATH}")


if __name__ == "__main__":
    train_and_evaluate()
