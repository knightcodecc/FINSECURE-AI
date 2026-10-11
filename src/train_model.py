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
        random_state=42,
        tree_method="hist",
        n_jobs=4,
        eval_metric="logloss",
        early_stopping_rounds=30,
    )
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    val_prob = clf.predict_proba(X_val)[:, 1]
    test_prob = clf.predict_proba(X_test)[:, 1]

    # Validation metrics & threshold sweep
    val_base_rate = float(np.mean(y_val))
    val_pr_auc = float(average_precision_score(y_val, val_prob))
    val_roc_auc = float(roc_auc_score(y_val, val_prob)) if len(np.unique(y_val)) > 1 else 0.5
    val_sweep_df, f2_opt, cost_opt = threshold_sweep(y_val, val_prob, amt_val, fp_cost=50.0, fn_multiplier=1.0)

    print(f"\n--- RAW VALIDATION SWEEP FOR VARIANT: {name} ---")
    print(f"Validation Metrics: Base Rate = {val_base_rate:.4f} | PR-AUC = {val_pr_auc:.4f} | ROC-AUC = {val_roc_auc:.4f}")
    min_cost_idx = val_sweep_df["total_cost"].idxmin()
    max_f2_idx = val_sweep_df["f2_score"].idxmax()
    print(f"Validation Cost Sweep Min Row (idx={min_cost_idx}): threshold={val_sweep_df.loc[min_cost_idx, 'threshold']:.4f}, total_cost=Rs. {val_sweep_df.loc[min_cost_idx, 'total_cost']:,.2f}, fn_cost=Rs. {val_sweep_df.loc[min_cost_idx, 'fn_cost']:,.2f}, fp_cost=Rs. {val_sweep_df.loc[min_cost_idx, 'fp_cost']:,.2f}")
    print(f"Validation F2 Sweep Max Row (idx={max_f2_idx}): threshold={val_sweep_df.loc[max_f2_idx, 'threshold']:.4f}, f2_score={val_sweep_df.loc[max_f2_idx, 'f2_score']:.4f}, recall={val_sweep_df.loc[max_f2_idx, 'recall']:.4f}, precision={val_sweep_df.loc[max_f2_idx, 'precision']:.4f}")
    print(f"Selected Validation Thresholds: F2-Optimal = {f2_opt:.4f} | Cost-Optimal = {cost_opt:.4f}")

    # Clean Test set evaluation
    base_rate = float(np.mean(y_test))
    pr_auc = float(average_precision_score(y_test, test_prob))
    roc_auc = float(roc_auc_score(y_test, test_prob)) if len(np.unique(y_test)) > 1 else 0.5
    flag_rate = float(np.mean(test_prob >= cost_opt))

    cost_res_f2 = expected_cost(y_test, test_prob, f2_opt, amt_test, fp_cost=50.0, fn_multiplier=1.0)
    cost_res_cost = expected_cost(y_test, test_prob, cost_opt, amt_test, fp_cost=50.0, fn_multiplier=1.0)

    # Cost table integrity assertion on test evaluation
    total_fraud_amt = float(np.sum(amt_test[y_test == 1]))
    assert abs((cost_res_cost["amount_caught"] + cost_res_cost["amount_missed"]) - total_fraud_amt) < 0.01, f"Fraud caught + missed ({cost_res_cost['amount_caught'] + cost_res_cost['amount_missed']:.2f}) != total ({total_fraud_amt:.2f})"
    calc_cost = cost_res_cost["amount_missed"] + 50.0 * cost_res_cost["fp"]
    assert abs(cost_res_cost["total_cost"] - calc_cost) < 0.01, f"Cost ({cost_res_cost['total_cost']:.2f}) != missed + 50*FP ({calc_cost:.2f})"

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
        boot_cost.append(expected_cost(y_test[idx], test_prob[idx], cost_opt, amt_test[idx], fp_cost=50.0, fn_multiplier=1.0)["total_cost"])
        
        ring_idx = idx[is_ring_mask[idx]]
        if len(ring_idx) > 0:
            boot_ring.append(np.mean(test_prob[ring_idx] >= cost_opt))

    pr_ci = (float(np.percentile(boot_pr, 2.5)), float(np.percentile(boot_pr, 97.5))) if boot_pr else (pr_auc, pr_auc)
    cost_ci = (float(np.percentile(boot_cost, 2.5)), float(np.percentile(boot_cost, 97.5))) if boot_cost else (cost_res_cost["total_cost"], cost_res_cost["total_cost"])

    if n_ring > 0:
        ring_preds = (test_prob[is_ring_mask] >= cost_opt).astype(int)
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
        "precision": cost_res_cost["precision"],
        "recall": cost_res_cost["recall"],
        "tp": cost_res_cost["tp"],
        "fp": cost_res_cost["fp"],
        "fn": cost_res_cost["fn"],
        "tn": cost_res_cost["tn"],
        "n_ring": n_ring,
        "ring_recall": ring_recall,
        "ring_ci": ring_ci,
        "f2_score": cost_res_cost["f2_score"],
        "total_cost_f2": cost_res_f2["total_cost"],
        "cost_ci": cost_ci,
        "amt_caught": cost_res_cost["amount_caught"],
        "amt_missed": cost_res_cost["amount_missed"],
        "pct_caught": cost_res_cost["pct_amount_caught"],
        "total_cost_cost_opt": cost_res_cost["total_cost"],
        "clf": clf,
        "cols": cols,
        "val_prob": val_prob,
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

    from src.cost_sensitive_eval import get_unscaled_amounts
    amt_val = get_unscaled_amounts(val_df, scaler)
    amt_test = get_unscaled_amounts(test_df, scaler)

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
        f2_edge = (res['f2_opt_thresh'] <= 1e-4) or (res['f2_opt_thresh'] >= 0.9999)
        cost_edge = (res['cost_opt_thresh'] <= 1e-4) or (res['cost_opt_thresh'] >= 0.9999)
        edge_str = f"F2 Edge: {'YES' if f2_edge else 'NO'} | Cost Edge: {'YES' if cost_edge else 'NO'}"
        ring_str = f"{res['ring_recall']*100:.1f}% (CI: {res['ring_ci'][0]*100:.1f}..{res['ring_ci'][1]*100:.1f}%, n={res['n_ring']})" if res['ring_recall'] is not None else "N/A"
        print(f"[{name:<28}] PR-AUC: {res['pr_auc']:.4f} (Base Rate: {res['base_rate']:.4f}) | ROC-AUC: {res['roc_auc']:.4f} | T_F2: {res['f2_opt_thresh']:.4f} | T_COST: {res['cost_opt_thresh']:.4f} ({edge_str}) | Recall: {res['recall']*100:.1f}% | Flag Rate: {res['flag_rate']*100:.1f}% | Ring Recall: {ring_str} | Cost (Cost-Opt): Rs. {res['total_cost_cost_opt']:,.2f}")

    # Calculate Baselines for Cost Table
    no_model_cost = float(np.sum(amt_test[y_test == 1]))  # Flag Nothing: FN = all fraud amounts
    flag_everything_cost = float(50.0 * np.sum(y_test == 0)) # Flag Everything: FP = all non-fraud * 50.0

    from src.cost_sensitive_eval import format_cost_table_row

    print("\n=== PROGRAMMATIC COST & CONFUSION MATRIX TABLE (CLEAN TEST SPLIT) ===")
    print(f"{'Variant':<35} | {'TP':<5} | {'FP':<5} | {'FN':<5} | {'TN':<6} | {'Alert Rate':<10} | {'Precision':<9} | {'Recall':<7} | {'FN Cost (Rs.)':<17} | {'FP Cost (Rs.)':<16} | {'Total Cost (Rs.)':<17} | {'Beats FN?':<10} | {'Beats FE (>5%)?':<15}")
    print("-" * 180)
    for r in results:
        formatted_row = format_cost_table_row(
            variant_name=r["name"],
            y_true=y_test,
            y_pred_proba=r["test_prob"],
            threshold=r["cost_opt_thresh"],
            amounts=amt_test,
            fp_review_cost=50.0,
            no_model_cost=no_model_cost,
            flag_everything_cost=flag_everything_cost,
        )
        print(formatted_row["formatted_line"])
    print("-" * 180)

    print("\n=== COST SENSITIVITY TABLE FOR FP REVIEW COSTS IN {50, 500, 2000, 5000} ===")
    print(f"{'FP Review Cost (Rs.)':<20} | {'Variant':<28} | {'T_COST':<8} | {'Alert Rate':<10} | {'Recall':<7} | {'Total Cost (Rs.)':<14} | {'Beats FN?':<10} | {'Beats FE (>5%)?':<15}")
    print("-" * 135)
    for fp_c in [50.0, 500.0, 2000.0, 5000.0]:
        fe_cost_fp = float(fp_c * np.sum(y_test == 0))
        for name, cols in variants:
            r = results[[res['name'] for res in results].index(name)]
            val_prob = r['val_prob']
            test_prob = r['test_prob']
            _, _, c_opt = threshold_sweep(y_val, val_prob, amt_val, fp_cost=fp_c)
            eval_c = expected_cost(y_test, test_prob, c_opt, amt_test, fp_cost=fp_c)
            ar = (eval_c['tp'] + eval_c['fp']) / len(y_test) * 100.0
            b_fn = "YES" if eval_c['total_cost'] < no_model_cost else "NO"
            b_fe = "YES" if eval_c['total_cost'] < (fe_cost_fp * 0.95) else "NO"
            print(f"Rs. {fp_c:<16,.0f} | {name:<28} | {c_opt:<8.4f} | {ar:>9.2f}% | {eval_c['recall']*100:>6.2f}% | Rs. {eval_c['total_cost']:<10,.2f} | {b_fn:<10} | {b_fe:<15}")
    print("-" * 135)

    print("\n=== RECALL AT FIXED ALERT BUDGETS (1%, 2%, 5% OF TRANSACTIONS) ===")
    print(f"{'Alert Budget %':<15} | {'Variant':<28} | {'Threshold':<9} | {'Actual Alert %':<23} | {'Overall Recall (95% CI)':<26} | {'Ring Recall (n=263, 95% CI)':<30} | {'Amount Caught (Rs.)':<18}")
    print("-" * 165)
    ring_mask = (test_df["mechanism"] == "MECH_B_RING").values if "mechanism" in test_df.columns else np.zeros(len(test_df), dtype=bool)
    n_ring = int(np.sum(ring_mask))
    rng = np.random.RandomState(42)

    for b_pct in [0.01, 0.02, 0.05]:
        for name, cols in variants:
            r = results[[res['name'] for res in results].index(name)]
            val_prob = r['val_prob']
            test_prob = r['test_prob']
            t_budget = float(np.quantile(val_prob, 1.0 - b_pct)) if len(val_prob) > 0 else 0.5
            eval_b = expected_cost(y_test, test_prob, t_budget, amt_test)
            ar = (eval_b['tp'] + eval_b['fp']) / len(y_test) * 100.0

            sat_str = " (score saturation)" if ar < (b_pct * 100.0 * 0.8) else ""
            ar_display = f"{ar:>5.2f}%{sat_str}"

            boot_rec, boot_ring = [], []
            n_samples = len(y_test)
            for _ in range(200):
                idx = rng.randint(0, n_samples, size=n_samples)
                f_idx = idx[y_test[idx] == 1]
                if len(f_idx) > 0:
                    boot_rec.append(np.mean(test_prob[f_idx] >= t_budget))
                r_idx = idx[ring_mask[idx]]
                if len(r_idx) > 0:
                    boot_ring.append(np.mean(test_prob[r_idx] >= t_budget))
            rec_ci = (float(np.percentile(boot_rec, 2.5)), float(np.percentile(boot_rec, 97.5))) if boot_rec else (eval_b['recall'], eval_b['recall'])
            ring_ci = (float(np.percentile(boot_ring, 2.5)), float(np.percentile(boot_ring, 97.5))) if boot_ring else (0.0, 0.0)

            ring_rec = float(np.mean(test_prob[ring_mask] >= t_budget)) if n_ring > 0 else 0.0
            rec_str = f"{eval_b['recall']*100:.2f}% [{rec_ci[0]*100:.2f}..{rec_ci[1]*100:.2f}%]"
            ring_str = f"{ring_rec*100:.1f}% [{ring_ci[0]*100:.1f}..{ring_ci[1]*100:.1f}%]" if n_ring > 0 else "N/A"

            print(f"{b_pct*100:<14.1f}% | {name:<28} | {t_budget:<9.4f} | {ar_display:<23} | {rec_str:<26} | {ring_str:<30} | Rs. {eval_b['amount_caught']:<15,.2f}")
    print("-" * 165)

    # Primary deployed model is Tabular + Graph
    deployed = results[-1]
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(deployed["clf"], MODEL_PATH)

    metadata = {
        "feature_names": feature_cols,
        "f2_optimal_threshold": float(deployed["f2_opt_thresh"]),
        "cost_optimal_threshold": float(deployed["cost_opt_thresh"]),
        "active_threshold": float(deployed["cost_opt_thresh"]),
        "active_strategy": "cost_optimal",
        "model_expected_cost_rs": float(deployed["total_cost_cost_opt"]),
        "baseline_flag_nothing_cost_rs": float(no_model_cost),
        "baseline_flag_everything_cost_rs": float(flag_everything_cost),
        "cost_savings_vs_flag_nothing_rs": float(no_model_cost - deployed["total_cost_cost_opt"]),
        "cost_savings_vs_flag_everything_rs": float(flag_everything_cost - deployed["total_cost_cost_opt"]),
    }
    joblib.dump(metadata, METADATA_PATH)

    # Save ablation predictions and thresholds for real_data_report.py reproducibility
    ablation_joblib_data = {
        r["name"]: {
            "val_prob": r["val_prob"],
            "test_prob": r["test_prob"],
            "f2_opt_thresh": float(r["f2_opt_thresh"]),
            "cost_opt_thresh": float(r["cost_opt_thresh"]),
            "cols": r["cols"],
        }
        for r in results
    }
    joblib.dump(ablation_joblib_data, Path("models/ablation_preds.joblib"))

    # Save .npz format
    npz_dict = {}
    for r in results:
        v_key = r["name"].replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus").replace("-", "_")
        npz_dict[f"{v_key}_val_prob"] = r["val_prob"]
        npz_dict[f"{v_key}_test_prob"] = r["test_prob"]
        npz_dict[f"{v_key}_thresholds"] = np.array([r["f2_opt_thresh"], r["cost_opt_thresh"]])
    np.savez_compressed(Path("models/ablation_preds.npz"), **npz_dict)

    # Save .parquet format (test probability dataframe)
    test_preds_df = pd.DataFrame({"y_true": y_test, "amount_test": amt_test})
    for r in results:
        test_preds_df[r["name"]] = r["test_prob"]
    test_preds_df.to_parquet(Path("models/ablation_preds.parquet"), index=False)
    print("Saved ablation prediction arrays to models/ablation_preds.{npz,joblib,parquet}")

    print("\n=== SAVED MODEL METADATA CONTENTS ===")
    for k, v in metadata.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")
    print("======================================\n")

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
        beats = (r["total_cost_cost_opt"] < no_model_cost) and (r["total_cost_cost_opt"] < flag_everything_cost)
        ring_str = f"{r['ring_recall']*100:.2f}% ({r['ring_ci'][0]*100:.1f}%..{r['ring_ci'][1]*100:.1f}%)" if r['ring_recall'] is not None else "N/A"
        ablation_md += f"| **{r['name']}** | {r['n_cols']} | {r['base_rate']:.4f} | {r['f2_opt_thresh']:.4f} | {r['pr_auc']:.4f} ({r['pr_ci'][0]:.4f}..{r['pr_ci'][1]:.4f}) | {r['roc_auc']:.4f} | {r['recall']*100:.2f}% | {r['flag_rate']*100:.2f}% | {ring_str} | Rs. {r['total_cost_cost_opt']:,.2f} | {'YES' if beats else 'NO'} |\n"

    ABLATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    ABLATION_PATH.write_text(ablation_md, encoding="utf-8")
    print(f"\nSaved report to {ABLATION_PATH}")


if __name__ == "__main__":
    train_and_evaluate()
