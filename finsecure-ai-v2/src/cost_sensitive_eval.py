"""
cost_sensitive_eval.py
========================
Cost-sensitive evaluation module for FinSecure AI v2.

Why this exists:
Standard precision/recall/F2 treats a missed ₹500 fraud the same as a missed ₹5,00,000 fraud.
This module evaluates fraud detection systems based on financial risk (money at risk in ₹),
computes expected cost curves, and performs amount-weighted threshold selection.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_unscaled_amounts(df: pd.DataFrame, scaler=None) -> np.ndarray:
    """
    Extract raw rupee amounts from DataFrame, unscaling using scaler if amount_raw is absent.
    Prevents passing scaled Z-scores to cost-sensitive evaluation functions.
    """
    if "amount_raw" in df.columns:
        return df["amount_raw"].values.astype(float)
    if "amount" in df.columns:
        if scaler is not None:
            feature_names = scaler.feature_names_in_ if hasattr(scaler, "feature_names_in_") else None
            if feature_names is not None and "amount" in feature_names:
                idx = list(feature_names).index("amount")
            else:
                idx = 0
            mean = scaler.mean_[idx]
            scale = scaler.scale_[idx]
            return np.maximum(0.0, df["amount"].values * scale + mean)
        else:
            return df["amount"].values.astype(float)
    raise KeyError("DataFrame must contain 'amount_raw' or 'amount' column")


def expected_cost(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float,
    amounts: np.ndarray,
    fp_cost: float = 50.0,
    fn_multiplier: float = 1.0,
) -> Dict[str, Any]:
    """
    Compute expected financial cost and amount metrics for a given decision threshold.
    """
    amounts = np.asarray(amounts, dtype=float)
    if len(amounts) > 0 and np.any(amounts <= 0.0):
        raise ValueError("Cost sensitive evaluation requires raw unscaled positive rupee amounts (amount_raw), but received scaled/standardized amounts!")
    y_pred = (y_pred_proba >= threshold).astype(int)

    tp = (y_true == 1) & (y_pred == 1)
    fp = (y_true == 0) & (y_pred == 1)
    fn = (y_true == 1) & (y_pred == 0)
    tn = (y_true == 0) & (y_pred == 0)

    fn_cost = np.sum(amounts[fn] * fn_multiplier)
    fp_cost_total = np.sum(fp) * fp_cost
    total_cost = float(fn_cost + fp_cost_total)

    total_fraud_amount = float(np.sum(amounts[y_true == 1]))
    amount_caught = float(np.sum(amounts[tp]))
    amount_missed = float(np.sum(amounts[fn]))
    pct_amount_caught = (amount_caught / total_fraud_amount * 100.0) if total_fraud_amount > 0 else 0.0

    tp_count = int(np.sum(tp))
    fp_count = int(np.sum(fp))
    fn_count = int(np.sum(fn))
    tn_count = int(np.sum(tn))

    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0.0
    recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0

    beta = 2.0
    f2_score = (
        (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
        if ((beta**2 * precision) + recall) > 0
        else 0.0
    )

    amount_weighted_recall = amount_caught / total_fraud_amount if total_fraud_amount > 0 else 0.0

    return {
        "threshold": float(threshold),
        "total_cost": total_cost,
        "cost_per_tx": float(total_cost / len(y_true)) if len(y_true) > 0 else 0.0,
        "fn_cost": float(fn_cost),
        "fp_cost": float(fp_cost_total),
        "amount_caught": amount_caught,
        "amount_missed": amount_missed,
        "total_fraud_amount": total_fraud_amount,
        "pct_amount_caught": pct_amount_caught,
        "amount_weighted_recall": amount_weighted_recall,
        "precision": float(precision),
        "recall": float(recall),
        "f2_score": float(f2_score),
        "tp": tp_count,
        "fp": fp_count,
        "fn": fn_count,
        "tn": tn_count,
    }


def format_cost_table_row(
    variant_name: str,
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float,
    amounts: np.ndarray,
    fp_review_cost: float = 50.0,
    no_model_cost: Optional[float] = None,
    flag_everything_cost: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Build a single cost table row with strict per-row integrity checks.
    
    Calculations:
      alert_rate = (TP + FP) / N
      precision  = TP / (TP + FP)
      recall     = TP / (TP + FN)
      fn_cost    = amount of missed fraud (total_fraud_amount - caught_amount)
      fp_cost    = FP * fp_review_cost
      total_cost = fn_cost + fp_cost
    
    Integrity Checks (raises ValueError if failed):
      1. TP + FP + FN + TN == N
      2. caught_amount + missed_amount == total_fraud_amount
      3. total_cost == fn_cost + fp_cost
    """
    eval_res = expected_cost(y_true, y_pred_proba, threshold, amounts, fp_cost=fp_review_cost)

    tp = eval_res["tp"]
    fp = eval_res["fp"]
    fn = eval_res["fn"]
    tn = eval_res["tn"]
    n = len(y_true)

    # 1. Row confusion matrix partition integrity check
    if tp + fp + fn + tn != n:
        raise ValueError(f"Integrity error in row '{variant_name}': TP ({tp}) + FP ({fp}) + FN ({fn}) + TN ({tn}) = {tp+fp+fn+tn} != N ({n})")

    total_fraud_amount = float(eval_res["total_fraud_amount"])
    caught_amount = float(eval_res["amount_caught"])
    missed_amount = float(eval_res["amount_missed"])

    # 2. Fraud amount conservation integrity check
    if abs((caught_amount + missed_amount) - total_fraud_amount) >= 0.01:
        raise ValueError(f"Integrity error in row '{variant_name}': caught ({caught_amount:.2f}) + missed ({missed_amount:.2f}) = {caught_amount+missed_amount:.2f} != total fraud ({total_fraud_amount:.2f})")

    fn_cost = missed_amount
    fp_cost = float(fp * fp_review_cost)
    total_cost = fn_cost + fp_cost

    # 3. Total financial cost reconciliation check
    if abs(total_cost - float(eval_res["total_cost"])) >= 0.01:
        raise ValueError(f"Integrity error in row '{variant_name}': total_cost ({total_cost:.2f}) != eval_res total_cost ({eval_res['total_cost']:.2f})")

    alert_rate = (tp + fp) / n if n > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    if no_model_cost is None:
        no_model_cost = total_fraud_amount
    if flag_everything_cost is None:
        flag_everything_cost = float((n - int(np.sum(y_true))) * fp_review_cost)

    beats_fn = total_cost < no_model_cost
    beats_fe = total_cost < (flag_everything_cost * 0.95)

    is_near_fe = alert_rate > 0.20
    variant_label = variant_name + (" (Near Flag-Everything)" if is_near_fe else "")

    formatted_line = (
        f"{variant_label:<35} | {tp:<5} | {fp:<5} | {fn:<5} | {tn:<6} | "
        f"{alert_rate*100:>9.2f}% | {precision*100:>8.2f}% | {recall*100:>6.2f}% | "
        f"Rs. {fn_cost:<13,.2f} | Rs. {fp_cost:<12,.2f} | Rs. {total_cost:<13,.2f} | "
        f"{'YES' if beats_fn else 'NO':<10} | {'YES' if beats_fe else 'NO':<15}"
    )

    return {
        "variant_name": variant_name,
        "variant_label": variant_label,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n": n,
        "alert_rate": alert_rate,
        "precision": precision,
        "recall": recall,
        "fn_cost": fn_cost,
        "fp_cost": fp_cost,
        "total_cost": total_cost,
        "caught_amount": caught_amount,
        "missed_amount": missed_amount,
        "total_fraud_amount": total_fraud_amount,
        "beats_fn": beats_fn,
        "beats_fe": beats_fe,
        "threshold": threshold,
        "formatted_line": formatted_line,
    }



def threshold_sweep(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    amounts: np.ndarray,
    fp_cost: float = 50.0,
    fn_multiplier: float = 1.0,
    n_steps: int = 200,
) -> Tuple[pd.DataFrame, float, float]:
    """
    Sweep decision thresholds on validation set to find F2-optimal and cost-optimal thresholds.
    Uses a fine wide grid combining log-spaced thresholds down to 1e-4 and score quantiles.
    """
    grid_log = np.logspace(-4, -0.0001, 150)
    grid_lin = np.linspace(0.0001, 0.9999, 150)
    grid_q = np.quantile(y_pred_proba, np.linspace(0.001, 0.999, 100)) if len(y_pred_proba) > 0 else np.array([])
    thresholds = np.unique(np.sort(np.clip(np.concatenate([grid_log, grid_lin, grid_q]), 1e-4, 0.9999)))

    results = []

    for thresh in thresholds:
        res = expected_cost(
            y_true=y_true,
            y_pred_proba=y_pred_proba,
            threshold=thresh,
            amounts=amounts,
            fp_cost=fp_cost,
            fn_multiplier=fn_multiplier,
        )
        results.append(res)

    sweep_df = pd.DataFrame(results)

    best_f2_idx = sweep_df["f2_score"].idxmax()
    best_cost_idx = sweep_df["total_cost"].idxmin()

    f2_optimal_threshold = float(sweep_df.loc[best_f2_idx, "threshold"])
    cost_optimal_threshold = float(sweep_df.loc[best_cost_idx, "threshold"])

    return sweep_df, f2_optimal_threshold, cost_optimal_threshold


def plot_cost_vs_threshold(
    sweep_df: pd.DataFrame,
    f2_thresh: float,
    cost_thresh: float,
    output_path: Path = Path("reports/cost_vs_threshold.png")
):
    """Plot expected financial cost vs. decision threshold."""
    plt.figure(figsize=(10, 6))
    plt.plot(sweep_df["threshold"], sweep_df["total_cost"], label="Total Expected Cost (Rs.)", color="#1a73e8", lw=2)
    plt.plot(sweep_df["threshold"], sweep_df["fn_cost"], label="False Negative Cost (Missed Fraud)", color="#ea4335", linestyle="--")
    plt.plot(sweep_df["threshold"], sweep_df["fp_cost"], label="False Positive Cost (Analyst Review)", color="#fbbc04", linestyle="--")

    plt.axvline(f2_thresh, color="#34a853", linestyle=":", label=f"F2-Optimal ({f2_thresh:.4f})")
    plt.axvline(cost_thresh, color="#ea4335", linestyle=":", label=f"Cost-Optimal ({cost_thresh:.4f})")

    plt.xlabel("Decision Threshold")
    plt.ylabel("Cost (Rs.)")
    plt.title("Financial Risk vs. Decision Threshold (Validation Split)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved cost vs threshold plot to {output_path}")


def plot_model_cost_comparison(
    comparison_df: pd.DataFrame,
    output_path: Path = Path("reports/model_cost_comparison.png")
):
    """Bar chart comparing baseline, Model A (tabular), and Model B (tabular+graph) costs."""
    plt.figure(figsize=(9, 5))
    bars = plt.bar(comparison_df["Strategy"], comparison_df["Total Cost (Rs.)"], color=["#ea4335", "#fbbc04", "#34a853"])

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 50, f"Rs. {height:,.2f}", ha="center", va="bottom", fontweight="bold")

    plt.ylabel("Total Expected Cost (Rs.)")
    plt.title("Expected Financial Cost Comparison on Test Split")
    plt.grid(axis="y", alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved model cost comparison plot to {output_path}")


def run_cost_evaluation():
    """Run full cost-sensitive evaluation pipeline and save report assets."""
    from src.train_model import unscale_column

    train_df = pd.read_parquet("data/features_train.parquet")
    val_df = pd.read_parquet("data/features_val.parquet")
    test_df = pd.read_parquet("data/features_test.parquet")
    scaler = joblib.load("models/scaler.joblib")
    model = joblib.load("models/xgboost_model.joblib")
    metadata = joblib.load("models/model_metadata.joblib")

    feature_cols = metadata.get("feature_names", [c for c in val_df.columns if c not in ("isFraud", "step")])

    X_val = val_df[feature_cols]
    y_val = val_df["isFraud"].values
    amt_val = unscale_column(val_df, "amount", scaler)

    X_test = test_df[feature_cols]
    y_test = test_df["isFraud"].values
    amt_test = unscale_column(test_df, "amount", scaler)

    val_proba = model.predict_proba(X_val)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]

    # Threshold sweep on VALIDATION set
    sweep_df, f2_thresh, cost_thresh = threshold_sweep(y_val, val_proba, amt_val)
    plot_cost_vs_threshold(sweep_df, f2_thresh, cost_thresh)

    # Evaluate on TEST set with fixed thresholds
    test_f2_eval = expected_cost(y_test, test_proba, f2_thresh, amt_test)
    test_cost_eval = expected_cost(y_test, test_proba, cost_thresh, amt_test)

    # Baselines: (a) No model (flag nothing, thresh=1.0), (b) Tabular-only model
    no_model_eval = expected_cost(y_test, test_proba, threshold=1.0, amounts=amt_test)
    
    # Savings vs no model
    savings_f2 = no_model_eval["total_cost"] - test_f2_eval["total_cost"]
    savings_cost = no_model_eval["total_cost"] - test_cost_eval["total_cost"]

    # Generate comparison DataFrame
    comp_df = pd.DataFrame([
        {"Strategy": "No Model (Flag Nothing)", "Total Cost (Rs.)": no_model_eval["total_cost"], "Amount Caught": no_model_eval["amount_caught"], "Amount Missed": no_model_eval["amount_missed"], "Pct Caught": "0.0%"},
        {"Strategy": f"Deployed Model (F2 Thresh = {f2_thresh:.4f})", "Total Cost (Rs.)": test_f2_eval["total_cost"], "Amount Caught": test_f2_eval["amount_caught"], "Amount Missed": test_f2_eval["amount_missed"], "Pct Caught": f"{test_f2_eval['pct_amount_caught']:.1f}%"},
        {"Strategy": f"Deployed Model (Cost Thresh = {cost_thresh:.4f})", "Total Cost (Rs.)": test_cost_eval["total_cost"], "Amount Caught": test_cost_eval["amount_caught"], "Amount Missed": test_cost_eval["amount_missed"], "Pct Caught": f"{test_cost_eval['pct_amount_caught']:.1f}%"},
    ])

    comp_csv_path = Path("reports/cost_comparison.csv")
    comp_csv_path.parent.mkdir(parents=True, exist_ok=True)
    comp_df.to_csv(comp_csv_path, index=False)
    print(f"Saved cost comparison CSV to {comp_csv_path}")

    plot_model_cost_comparison(comp_df.iloc[[0, 1, 2]])

    report_md = f"""# Cost-Sensitive Evaluation Report (FinSecure AI v2)

## Cost Model Parameters
- **False Negative (FN) Cost**: Transaction Amount ($\text{{amount}} \\times 1.0$)
- **False Positive (FP) Cost**: Fixed Analyst Review Cost (Rs. 50.0 per alert)
- **True Positive (TP) / True Negative (TN) Cost**: Rs. 0.0

## Threshold Selection (Validation Split Only)
- **F2-Optimal Threshold**: `{f2_thresh:.4f}`
- **Cost-Optimal Threshold**: `{cost_thresh:.4f}`

## Financial Risk Metrics on Held-Out Test Split
- **No Model Baseline (Flag Nothing)**:
  - Total Cost: **Rs. {no_model_eval['total_cost']:,.2f}**
  - Fraud Amount Missed: Rs. {no_model_eval['amount_missed']:,.2f}
- **Deployed Model under F2-Threshold ({f2_thresh:.4f})**:
  - Total Cost: **Rs. {test_f2_eval['total_cost']:,.2f}**
  - Financial Savings vs No Model: **Rs. {savings_f2:,.2f}**
  - Amount-Weighted Recall: **{test_f2_eval['amount_weighted_recall']:.1%}**
  - Precision: {test_f2_eval['precision']:.4f} | Recall: {test_f2_eval['recall']:.4f} | F2: {test_f2_eval['f2_score']:.4f}
- **Deployed Model under Cost-Threshold ({cost_thresh:.4f})**:
  - Total Cost: **Rs. {test_cost_eval['total_cost']:,.2f}**
  - Financial Savings vs No Model: **Rs. {savings_cost:,.2f}**
  - Amount-Weighted Recall: **{test_cost_eval['amount_weighted_recall']:.1%}**

## Artifact Links
- CSV Data: [`reports/cost_comparison.csv`](file:///c:/Users/Pratham%20Bhat/Downloads/finsecure-ai-v2-scaffold/finsecure-ai-v2/reports/cost_comparison.csv)
- Cost Curve Plot: [`reports/cost_vs_threshold.png`](file:///c:/Users/Pratham%20Bhat/Downloads/finsecure-ai-v2-scaffold/finsecure-ai-v2/reports/cost_vs_threshold.png)
- Model Comparison Plot: [`reports/model_cost_comparison.png`](file:///c:/Users/Pratham%20Bhat/Downloads/finsecure-ai-v2-scaffold/finsecure-ai-v2/reports/model_cost_comparison.png)
"""
    report_md_path = Path("reports/cost_eval.md")
    report_md_path.write_text(report_md, encoding="utf-8")
    print(f"Saved cost evaluation report to {report_md_path}")


if __name__ == "__main__":
    run_cost_evaluation()
