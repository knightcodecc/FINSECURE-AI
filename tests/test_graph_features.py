"""
Tests for src/graph_features.py

Key things to verify (leakage discipline is the whole point of this module):
- attach_graph_features_expanding() never uses same-step-or-later
  transactions when computing features for a given step
- fraud_exposure for a node with no fraud history is 0.0
- component_size defaults to 1 for an isolated node
- output row count matches input row count (no rows dropped/duplicated)
"""


import pandas as pd
import networkx as nx
from src.graph_features import (
    build_graph,
    compute_node_features,
    attach_graph_features_expanding,
)


def test_no_future_leakage():
    """Verify that expanding window does not use transactions from future steps."""
    # Step 1: Clean transaction between A and B
    # Step 50: Fraudulent transaction between B and C
    df = pd.DataFrame([
        {"step": 1, "nameOrig": "A", "nameDest": "B", "amount": 100.0, "isFraud": 0},
        {"step": 50, "nameOrig": "B", "nameDest": "C", "amount": 500.0, "isFraud": 1},
    ])
    
    result = attach_graph_features_expanding(df)
    
    # Row for step 1 should have history strictly before step 1 (empty graph)
    # Thus orig_fraud_exposure and dest_fraud_exposure for step 1 must be 0.0
    step1_row = result[result["step"] == 1].iloc[0]
    assert step1_row["orig_fraud_exposure"] == 0.0
    assert step1_row["dest_fraud_exposure"] == 0.0
    assert step1_row["orig_in_degree"] == 0.0


def test_fraud_exposure_zero_for_clean_node():
    """Verify that a node with only clean transactions has fraud_exposure == 0.0."""
    df_edges = pd.DataFrame([
        {"nameOrig": "User1", "nameDest": "User2", "amount": 50.0, "isFraud": 0},
        {"nameOrig": "User2", "nameDest": "User3", "amount": 150.0, "isFraud": 0},
    ])
    G = build_graph(df_edges)
    feats = compute_node_features(G)
    
    assert (feats["fraud_exposure"] == 0.0).all()


def test_row_count_preserved():
    """Verify that attach_graph_features_expanding preserves exact row count."""
    df = pd.DataFrame([
        {"step": 1, "nameOrig": "A", "nameDest": "B", "amount": 100.0, "isFraud": 0},
        {"step": 2, "nameOrig": "C", "nameDest": "D", "amount": 200.0, "isFraud": 0},
        {"step": 25, "nameOrig": "B", "nameDest": "C", "amount": 300.0, "isFraud": 1},
    ])
    result = attach_graph_features_expanding(df)
    assert len(result) == len(df)


def test_inference_graph_state_cold_start_and_no_future_lookahead():
    """Verify neutral defaults for cold-start accounts and zero future lookahead (Amendment 3)."""
    dummy_df = pd.DataFrame([{
        "step": 1,
        "nameOrig": "NEW_USER_A",
        "nameDest": "NEW_USER_B",
        "amount": 100.0,
        "isFraud": 0
    }])
    
    df_out = attach_graph_features_expanding(dummy_df)
    
    assert df_out.loc[0, "orig_in_degree"] == 0
    assert df_out.loc[0, "orig_out_degree"] == 0
    assert df_out.loc[0, "orig_pagerank"] == 0.0
    assert df_out.loc[0, "orig_component_size"] == 1
    assert df_out.loc[0, "orig_fraud_exposure"] == 0.0

    two_step_df = pd.DataFrame([
        {"step": 1, "nameOrig": "USER_1", "nameDest": "USER_2", "amount": 50.0, "isFraud": 0},
        {"step": 2, "nameOrig": "USER_1", "nameDest": "USER_3", "amount": 500.0, "isFraud": 1},
    ])
    
    res = attach_graph_features_expanding(two_step_df)
    step1_exposure = res[res["step"] == 1]["orig_fraud_exposure"].values[0]
    assert step1_exposure == 0.0, "Leakage detected! Step 1 transaction saw future Step 2 fraud label!"


def test_fraud_exposure_label_delay_and_timing():
    """Verify that a fraud edge at step t populates exposure for transactions at step >= t + delay, but 0 before."""
    df = pd.DataFrame([
        {"step": 5, "nameOrig": "ACC_V", "nameDest": "ACC_M", "amount": 1000.0, "isFraud": 0},
        {"step": 10, "nameOrig": "ACC_M", "nameDest": "HUB_1", "amount": 5000.0, "isFraud": 1},
        {"step": 12, "nameOrig": "ACC_X", "nameDest": "HUB_1", "amount": 2000.0, "isFraud": 0},
    ])
    
    res = attach_graph_features_expanding(df, step_batch_size=1)
    
    # Step 5 transaction before fraud edge must have exposure 0.0
    row_step5 = res[res["step"] == 5].iloc[0]
    assert row_step5["dest_fraud_exposure"] == 0.0

    # Step 12 transaction touching HUB_1 (confirmed fraud at step 10) must have dest_fraud_exposure > 0.0
    row_step12 = res[res["step"] == 12].iloc[0]
    assert row_step12["dest_fraud_exposure"] > 0.0, f"Expected dest_fraud_exposure > 0 at step 12, got {row_step12['dest_fraud_exposure']}"


def test_pipeline_fraud_exposure_nonzero_when_frauds_exist():
    """Pipeline-level assertion: fails if fraud_exposure is all zero when confirmed frauds exist."""
    df = pd.DataFrame([
        {"step": 1, "nameOrig": "USER_A", "nameDest": "HUB_X", "amount": 100.0, "isFraud": 1},
        {"step": 10, "nameOrig": "USER_B", "nameDest": "HUB_X", "amount": 200.0, "isFraud": 0},
        {"step": 20, "nameOrig": "USER_C", "nameDest": "HUB_X", "amount": 300.0, "isFraud": 0},
    ])
    res = attach_graph_features_expanding(df, step_batch_size=1)
    
    total_exposure_sum = res["orig_fraud_exposure"].sum() + res["dest_fraud_exposure"].sum()
    assert total_exposure_sum > 0.0, "Pipeline failure: fraud_exposure is all zero despite confirmed fraud edges!"


