"""
Diagnostic Script: High Alert Rate Diagnosis across 3 Graph Modes and Training Distribution Comparison
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from src.batch_scoring import compute_evaluation_metrics, ALLOWED_TYPES, get_warmed_graph_state

def run_diagnosis():
    print("================================================================================")
    print("DIAGNOSIS 1: HIGH ALERT RATE ON UPLOADED FILES ACROSS THREE GRAPH MODES")
    print("================================================================================")

    model = joblib.load("models/xgboost_model.joblib")
    scaler = joblib.load("models/scaler.joblib")
    meta = joblib.load("models/model_metadata.joblib")
    threshold = float(meta.get("active_threshold", meta.get("cost_optimal_threshold", 0.0484)))
    feature_names = meta.get("feature_names", list(scaler.feature_names_in_))

    sample_path = Path("frontend/samples/sample_training_rows.csv")
    if not sample_path.exists():
        sample_path = Path("frontend/samples/sample_transactions.csv")
    df_raw = pd.read_csv(sample_path)
    print(f"Sample loaded: {len(df_raw)} rows, {df_raw['isFraud'].sum()} frauds. Step range: {df_raw['step'].min()}..{df_raw['step'].max()}")

    # 1. Mode A: Graph computed from this file only (current)
    df_mode_a = df_raw.copy()
    df_graph_a = attach_graph_features_expanding(df_mode_a, step_batch_size=72)
    # No labels for graph
    df_graph_a["orig_fraud_exposure"] = 0.0
    df_graph_a["dest_fraud_exposure"] = 0.0
    df_eng_a = engineer_tabular_features(df_graph_a)
    for t in ALLOWED_TYPES:
        c = f"type_{t}"
        if c not in df_eng_a.columns:
            df_eng_a[c] = (df_eng_a["type"] == t).astype(float)
    X_a = scaler.transform(df_eng_a[feature_names])
    probs_a = model.predict_proba(X_a)[:, 1]
    y_pred_a = (probs_a >= threshold).astype(int)
    m_a = compute_evaluation_metrics(df_raw["isFraud"].values, y_pred_a, probs_a, df_raw["amount"].values, threshold)

    # 2. Mode B: Neutral cold-start graph features
    df_mode_b = df_raw.copy()
    for prefix in ["orig", "dest"]:
        df_mode_b[f"{prefix}_in_degree"] = 0.0
        df_mode_b[f"{prefix}_out_degree"] = 0.0
        df_mode_b[f"{prefix}_pagerank"] = 0.0
        df_mode_b[f"{prefix}_component_size"] = 1.0
        df_mode_b[f"{prefix}_fraud_exposure"] = 0.0
    df_eng_b = engineer_tabular_features(df_mode_b)
    for t in ALLOWED_TYPES:
        c = f"type_{t}"
        if c not in df_eng_b.columns:
            df_eng_b[c] = (df_eng_b["type"] == t).astype(float)
    X_b = scaler.transform(df_eng_b[feature_names])
    probs_b = model.predict_proba(X_b)[:, 1]
    y_pred_b = (probs_b >= threshold).astype(int)
    m_b = compute_evaluation_metrics(df_raw["isFraud"].values, y_pred_b, probs_b, df_raw["amount"].values, threshold)

    # 3. Mode C: Graph context warmed from training history (step <= 420)
    print("\nWarming graph context from training history (data/paysim_synth.csv)...")
    df_train_raw = pd.read_csv("data/paysim_synth.csv")
    train_history = df_train_raw[df_train_raw["step"] <= 420].copy()
    graph_state = InferenceGraphState()
    graph_state.warm_from_history(train_history)
    print(f"Warmed graph nodes: {graph_state.G.number_of_nodes():,}, edges: {graph_state.G.number_of_edges():,}")

    df_mode_c = df_raw.copy()
    orig_feats = [graph_state.get_node_features(n) for n in df_mode_c["nameOrig"]]
    dest_feats = [graph_state.get_node_features(n) for n in df_mode_c["nameDest"]]
    for f in ["in_degree", "out_degree", "pagerank", "component_size", "fraud_exposure"]:
        df_mode_c[f"orig_{f}"] = [of[f] for of in orig_feats]
        df_mode_c[f"dest_{f}"] = [df_[f] for df_ in dest_feats]

    df_eng_c = engineer_tabular_features(df_mode_c)
    for t in ALLOWED_TYPES:
        c = f"type_{t}"
        if c not in df_eng_c.columns:
            df_eng_c[c] = (df_eng_c["type"] == t).astype(float)
    X_c = scaler.transform(df_eng_c[feature_names])
    probs_c = model.predict_proba(X_c)[:, 1]
    y_pred_c = (probs_c >= threshold).astype(int)
    m_c = compute_evaluation_metrics(df_raw["isFraud"].values, y_pred_c, probs_c, df_raw["amount"].values, threshold)

    print("\n--------------------------------------------------------------------------------")
    print("RESULTS: THREE GRAPH MODES COMPARISON ON sample_transactions.csv (N=2,000, Fraud=50)")
    print("--------------------------------------------------------------------------------")
    modes = [
        ("File-only expanding graph (current)", probs_a, y_pred_a, m_a),
        ("Neutral cold-start graph (unseen defaults)", probs_b, y_pred_b, m_b),
        ("Training-warmed graph context", probs_c, y_pred_c, m_c),
    ]

    print(f"{'Mode':<42} | {'Flagged':<7} | {'Alert Rate':<10} | {'Precision':<9} | {'Recall':<7} | {'F1':<7} | {'F2':<7}")
    print("-" * 100)
    for name, pr, yp, m in modes:
        fl_cnt = int(yp.sum())
        fl_rate = (fl_cnt / len(yp)) * 100
        prec = m["precision"] * 100
        rec = m["recall"] * 100
        f1 = m["f1"]
        f2 = m["f2"]
        print(f"{name:<42} | {fl_cnt:<7} | {fl_rate:>9.2f}% | {prec:>8.2f}% | {rec:>6.2f}% | {f1:>6.4f} | {f2:>6.4f}")

    # 4. Compare min/median/max of 10 graph features: Upload vs Training Set
    print("\n--------------------------------------------------------------------------------")
    print("DISTRIBUTION CHECK: 10 GRAPH FEATURES (Upload File-Only vs Full Training Set)")
    print("--------------------------------------------------------------------------------")
    graph_cols = [
        "orig_in_degree", "orig_out_degree", "orig_pagerank", "orig_component_size", "orig_fraud_exposure",
        "dest_in_degree", "dest_out_degree", "dest_pagerank", "dest_component_size", "dest_fraud_exposure"
    ]

    # Training set graph features from data/transactions_with_graph_features.parquet
    train_gf_df = pd.read_parquet("data/transactions_with_graph_features.parquet")
    train_gf = train_gf_df[train_gf_df["step"] <= 420]

    print(f"{'Feature Name':<24} | {'Upload Min':<10} | {'Upload Med':<10} | {'Upload Max':<10} | {'Train Min':<10} | {'Train Med':<10} | {'Train Max':<10}")
    print("-" * 92)
    for col in graph_cols:
        u_min = df_graph_a[col].min()
        u_med = df_graph_a[col].median()
        u_max = df_graph_a[col].max()
        t_min = train_gf[col].min()
        t_med = train_gf[col].median()
        t_max = train_gf[col].max()
        print(f"{col:<24} | {u_min:<10.6f} | {u_med:<10.6f} | {u_max:<10.6f} | {t_min:<10.6f} | {t_med:<10.6f} | {t_max:<10.6f}")

if __name__ == "__main__":
    run_diagnosis()
