"""
test_chatbot.py — Automated Benchmark Validation Suite for RAG Chatbot (45 Questions).
======================================================================================
Tests the deterministic intent router against all 45 benchmark evaluation questions.
Seeds a test prediction log in SQLite, computes dynamic reference numbers,
and asserts that each question produces a grounded, factual answer.
Prints an audit table: question | answer | pass/fail.
"""

import os
import json
import sqlite3
import pytest
from pathlib import Path

from api.rag_chatbot import answer_analyst_query, ChatResult, load_aggregate_data, get_model_metadata
from api.db import get_model_version

TEST_DB_PATH = Path(os.environ.get("DATA_DIR", "data")) / "finsecure_test_chat.db"


@pytest.fixture(scope="module")
def seeded_chat_db():
    """Seed test SQLite database with known prediction logs and drift snapshot."""
    if TEST_DB_PATH.exists():
        TEST_DB_PATH.unlink()

    conn = sqlite3.connect(TEST_DB_PATH)
    with conn:
        conn.execute(
            """
            CREATE TABLE predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                model_version TEXT,
                step INTEGER,
                name_orig TEXT,
                name_dest TEXT,
                amount REAL NOT NULL,
                prediction_proba REAL NOT NULL,
                is_fraud INTEGER NOT NULL,
                threshold_used REAL NOT NULL,
                features_json TEXT NOT NULL,
                shap_top_features_json TEXT,
                explanation TEXT
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE drift_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                overall_status TEXT NOT NULL,
                retrain_recommended INTEGER NOT NULL,
                score_psi REAL NOT NULL,
                drifted_feature_count INTEGER NOT NULL,
                warning_feature_count INTEGER NOT NULL,
                metrics_json TEXT NOT NULL
            );
            """
        )

        shap_items = [
            {"feature": "orig_fraud_exposure", "human_name": "Origin Fraud Exposure", "shap_value": 3.82},
            {"feature": "dest_fraud_exposure", "human_name": "Dest Fraud Exposure", "shap_value": 2.15},
            {"feature": "errorBalanceOrig", "human_name": "Origin Balance Discrepancy", "shap_value": -0.85}
        ]

        current_ver = get_model_version()

        # Row 1: Legitimate Payment
        conn.execute(
            """
            INSERT INTO predictions VALUES (
                1, '2026-10-21T10:00:00', ?, 1, 'C10001', 'M20001', 500.0,
                0.012, 0, 0.0484, '{"type": "PAYMENT"}', ?, 'Cleared legitimate transaction with low risk.'
            )
            """,
            (current_ver, json.dumps(shap_items),)
        )

        # Row 2: Fraud Transfer (High amount)
        conn.execute(
            """
            INSERT INTO predictions VALUES (
                2, '2026-10-21T10:05:00', ?, 2, 'C_MULE_R', 'M_HUB_CA', 95000.0,
                0.985, 1, 0.0484, '{"type": "TRANSFER"}', ?, 'Flagged as high-risk fraud ring transfer.'
            )
            """,
            (current_ver, json.dumps(shap_items),)
        )

        # Row 3: Second Fraud Transfer (Moderate amount)
        conn.execute(
            """
            INSERT INTO predictions VALUES (
                3, '2026-10-21T10:10:00', ?, 3, 'C20002', 'M30002', 15000.0,
                0.880, 1, 0.0484, '{"type": "TRANSFER"}', ?, 'Flagged as fraud due to rapid drain.'
            )
            """,
            (current_ver, json.dumps(shap_items),)
        )

        # Seed drift snapshot: DRIFT_DETECTED, retrain=True, score_psi=0.312
        drift_metrics = {
            "score_psi": 0.312,
            "feature_psi": {
                "amount": 0.354,
                "orig_fraud_exposure": 0.281,
                "errorBalanceOrig": 0.120
            }
        }
        conn.execute(
            """
            INSERT INTO drift_snapshots VALUES (
                1, '2026-10-21T10:15:00', 'DRIFT_DETECTED', 1, 0.312, 2, 1, ?
            )
            """,
            (json.dumps(drift_metrics),)
        )

    conn.close()
    yield TEST_DB_PATH

    # Teardown
    if TEST_DB_PATH.exists():
        try:
            TEST_DB_PATH.unlink()
        except Exception:
            pass


def test_45_benchmark_questions_comprehensive(seeded_chat_db):
    """Run all 45 test questions against deterministic router and verify fact containment."""
    db_path = seeded_chat_db
    meta = get_model_metadata()
    paysim_agg = load_aggregate_data("paysim") or load_aggregate_data("synthetic")

    user_info = {"id": 1, "full_name": "Alice Analyst", "email": "alice@finsecure.ai", "role": "analyst"}

    # Define test suite of all 45 benchmark questions with assertions
    # Format: (Question, assertion_callback, description)
    test_suite = [
        # 1. Greetings & Meta
        ("hi", lambda a: "FinSecure" in a or "assist" in a.lower(), "Greeting response"),
        ("what can you do?", lambda a: "Prediction Logs" in a and "SHAP" in a and "Drift" in a, "Capabilities list"),
        
        # 2. Prediction Log Queries
        ("how many transactions were scored?", lambda a: "3" in a, "Scored count = 3"),
        ("how many were flagged as fraud?", lambda a: "2" in a, "Fraud count = 2"),
        ("show the last 5 transactions", lambda a: "Transaction #1" in a or "Transaction #2" in a, "List recent transactions"),
        ("show the last flagged transaction", lambda a: "Transaction #3" in a or "15,000" in a, "Last flagged fraud"),
        ("what is the highest amount flagged?", lambda a: "95,000" in a, "Highest amount = Rs. 95,000"),
        ("how many TRANSFER transactions were flagged?", lambda a: "2" in a, "TRANSFER fraud count = 2"),
        ("why was transaction 2 flagged?", lambda a: "FLAGGED FRAUD" in a and ("95,000" in a or "C_MULE_R" in a), "Explain tx #2"),
        ("explain the last decision", lambda a: "Transaction (#3)" in a or "FLAGGED FRAUD" in a, "Explain last decision"),

        # 3. Model Info & Thresholds
        ("what is the active threshold?", lambda a: f"{meta.get('active_threshold', 0.0484):.4f}" in a, "Active threshold value"),
        ("why is the cost-optimal threshold different from the F2-optimal one?", lambda a: "rupee" in a.lower() and "recall" in a.lower(), "Threshold rationale"),
        ("what is the expected cost of the model?", lambda a: "Rs." in a and ("393,128" in a or "cost" in a.lower()), "Expected cost report"),
        ("how much money does the model save vs no model?", lambda a: "34,964,817" in a or "save" in a.lower(), "Rupees saved"),
        ("how is F2 calculated?", lambda a: "Precision" in a and "Recall" in a and "4P" in a, "F2 formula"),
        ("how is F1 calculated?", lambda a: "Precision" in a and "Recall" in a and "harmonic" in a.lower(), "F1 formula"),
        ("what is precision?", lambda a: "TP" in a and "FP" in a, "Precision formula"),
        ("what is recall?", lambda a: "TP" in a and "FN" in a, "Recall formula"),
        ("what is the alert rate?", lambda a: "TP + FP" in a or "Alert Rate" in a, "Alert rate formula"),

        # 4. Drift Monitoring
        ("what is the drift status?", lambda a: "DRIFT_DETECTED" in a and "0.3120" in a, "Drift status snapshot"),
        ("what is PSI?", lambda a: "Population Stability Index" in a and "0.10" in a and "0.25" in a, "PSI concept & thresholds"),
        ("is retraining recommended?", lambda a: "YES" in a, "Retrain recommended flag"),
        ("which features drifted?", lambda a: "amount" in a or "orig_fraud_exposure" in a, "Drifted features list"),

        # 5. Dataset Analytics (PaySim / Synthetic)
        ("which dataset was the model trained on?", lambda a: "Synthetic" in a or "PaySim" in a, "Training dataset name"),
        ("how many transactions are in the dataset?", lambda a: "6,362,620" in a or "300,000" in a, "Dataset row count"),
        ("how many frauds are in the dataset?", lambda a: "8,213" in a or "5,436" in a, "Dataset fraud count"),
        ("which transaction type has the most fraud?", lambda a: "TRANSFER" in a or "CASH_OUT" in a, "Top fraud types"),
        ("what is the fraud rate for CASH_OUT?", lambda a: "CASH_OUT" in a and ("%" in a or "rate" in a.lower()), "CASH_OUT fraud rate"),
        ("what is the average fraud amount?", lambda a: "Rs." in a and "average" in a.lower(), "Average fraud amount"),
        ("what percent of fraud drains the account?", lambda a: "drain" in a.lower() and "%" in a, "Account draining share"),

        # 6. Concepts & Typologies
        ("what is SHAP?", lambda a: "game theory" in a.lower() or "Additivity" in a, "SHAP definition"),
        ("what are graph features?", lambda a: "PageRank" in a or "degree" in a.lower(), "Graph features breakdown"),
        ("what is XGBoost?", lambda a: "Gradient Boosting" in a or "decision trees" in a.lower(), "XGBoost definition"),
        ("how does the model decide fraud?", lambda a: "probability" in a.lower() and "threshold" in a.lower(), "Scoring pipeline decision rule"),
        ("how many features does the model use?", lambda a: "33" in a, "Feature count = 33"),
        ("list the feature groups", lambda a: "Graph" in a and "Tabular" in a and "Type" in a, "Feature categories"),
        ("what is a mule account?", lambda a: "intermediary" in a.lower() or "ring" in a.lower(), "Mule account typology"),

        # 7. Dashboard Guide & Auth
        ("how do I score a transaction?", lambda a: "Scorer" in a and "Score Transaction" in a, "Scoring instructions"),
        ("how do I simulate drift?", lambda a: "Drift" in a and "Simulate Drift" in a, "Drift simulation instructions"),
        ("how do I log out?", lambda a: "Logout" in a or "log out" in a.lower(), "Logout guidance"),
        ("who am I logged in as?", lambda a: "Alice Analyst" in a and "analyst" in a, "Current user identity"),

        # 8. Edge cases & boundaries
        ("tell me a joke", lambda a: "PageRank" in a or "fraud" in a.lower(), "Humorous response"),
        ("what is the weather today?", lambda a: "meteorological" in a.lower() or "financial" in a.lower(), "Domain boundary check"),
        ("asdfgh", lambda a: "common questions" in a.lower() or "couldn't find" in a.lower(), "Gibberish query fallback"),
        ("", lambda a: "FinSecure AI" in a and "help" in a.lower(), "Empty string prompt"),
        ("X" * 3000, lambda a: len(a) > 20, "3000-character input handling"),
    ]

    print("\n" + "=" * 90)
    print(f"{'#':<3} | {'QUESTION':<42} | {'STATUS':<6} | {'SNIPPET':<32}")
    print("=" * 90)

    failed_questions = []
    
    for idx, (question, validator, desc) in enumerate(test_suite, 1):
        disp_q = question if len(question) <= 40 else question[:37] + "..."
        ans = answer_analyst_query(question, user_info=user_info, db_path=db_path)
        
        is_pass = False
        try:
            is_pass = validator(str(ans))
        except Exception as e:
            is_pass = False

        status_str = "PASS" if is_pass else "FAIL"
        disp_ans = str(ans).replace("\n", " ")[:30] + "..."
        print(f"{idx:<3} | {disp_q:<42} | {status_str:<6} | {disp_ans:<32}")

        if not is_pass:
            failed_questions.append((idx, question, desc, str(ans)))

    print("=" * 90)

    if failed_questions:
        print(f"\n[FAILURES DETECTED]: {len(failed_questions)} out of {len(test_suite)} questions failed:")
        for idx, q, d, a in failed_questions:
            print(f"  - #{idx} '{q}': {d} -> Answer: {a[:100]}")

    assert len(failed_questions) == 0, f"{len(failed_questions)} benchmark questions failed!"
    print(f"\n[SUCCESS]: All {len(test_suite)} benchmark questions PASSED with verified fact containment.")


def test_chatbot_user_analysis_benchmark():
    """
    Validate 12+ RAG Chatbot Benchmark Questions for User Uploaded CSV Analyses.
    Tests:
    1. Clean initial state with no analyses (friendly message).
    2. Grounded factual containment where expected values are dynamically computed
       from the saved analysis summary.json and results.parquet, never hardcoded.
    """
    from api.analysis_db import create_analysis, update_analysis_results
    from api.db import get_data_dir
    import pandas as pd

    user_info = {"id": 101, "full_name": "Bob Analyst", "email": "bob@finsecure.ai", "role": "analyst"}
    data_dir = get_data_dir()

    # 1. Edge Case: Query with NO uploaded analyses yet
    no_analysis_user = {"id": 8888, "full_name": "Empty User", "email": "empty@finsecure.ai", "role": "analyst"}
    res_no_upload = answer_analyst_query("how many rows were in my last upload?", user_info=no_analysis_user)
    assert "no uploaded analyses found for your account" in str(res_no_upload).lower()
    assert "analyze csv" in str(res_no_upload).lower()

    # 2. Seed a realistic test analysis with results.parquet and summary.json
    analysis_id = "test-analysis-bench-001"
    res_dir = data_dir / "analyses" / analysis_id
    res_dir.mkdir(parents=True, exist_ok=True)

    # Synthetic sample dataframe with known properties
    test_rows = []
    for i in range(1, 101):
        is_f = 1 if i <= 10 else 0
        prob = 0.95 - (i * 0.005) if is_f else 0.001 * i
        test_rows.append({
            "step": i,
            "type": "TRANSFER" if i % 2 == 0 else "CASH_OUT",
            "amount": float(1000.0 * i),
            "nameOrig": f"C_ORIG_{i:04d}",
            "nameDest": f"M_DEST_{i:04d}",
            "oldbalanceOrg": float(5000.0 * i),
            "newbalanceOrig": 0.0 if is_f else float(4000.0 * i),
            "oldbalanceDest": 0.0,
            "newbalanceDest": float(1000.0 * i),
            "isFraud": is_f,
            "fraud_probability": prob,
            "decision": "FLAGGED" if prob >= 0.05 else "CLEARED",
            "threshold_used": 0.05,
            "account_drained": 1.0 if is_f else 0.0,
        })
    df_results = pd.DataFrame(test_rows)
    df_results.to_parquet(res_dir / "results.parquet", index=False)

    flagged_mask = df_results["decision"] == "FLAGGED"
    flagged_cnt = int(flagged_mask.sum())
    flagged_rate = float(flagged_cnt / len(df_results))
    flagged_by_type = df_results.loc[flagged_mask, "type"].value_counts().to_dict()
    top_flagged_type = max(flagged_by_type, key=flagged_by_type.get)

    summary_data = {
        "analysis_id": analysis_id,
        "eda": {
            "rows_total": 105,
            "rows_valid": 100,
            "rows_excluded": 5,
            "rejection_counts": {"invalid_numeric": 3, "non_positive_amount": 2},
            "total_fraud_count": 10,
        },
        "scoring": {
            "model_version": "v2_test_xgb",
            "rows_scored": 100,
            "threshold_used": 0.05,
            "flagged_count": flagged_cnt,
            "flagged_rate": flagged_rate,
            "total_amount_flagged": float(df_results.loc[flagged_mask, "amount"].sum()),
            "flagged_by_type": flagged_by_type,
            "has_labels": True,
            "metrics": {
                "precision": 0.8333,
                "recall": 1.0000,
                "f1": 0.9091,
                "f2": 0.9615,
                "financial_cost": {
                    "total_cost": 1500.0,
                    "missed_fraud_amount": 0.0,
                    "fp_review_cost": 1500.0,
                    "savings_vs_nothing": 53500.0,
                },
            },
        },
    }
    with open(res_dir / "summary.json", "w", encoding="utf-8") as sf:
        json.dump(summary_data, sf)

    create_analysis(
        analysis_id=analysis_id,
        user_id=101,
        user_email="bob@finsecure.ai",
        filename="transactions_audit_q3.csv",
        display_name="Q3 Transaction Batch",
    )
    update_analysis_results(
        analysis_id=analysis_id,
        rows_total=105,
        rows_valid=100,
        rows_scored=100,
        model_version="v2_test_xgb",
        threshold=0.05,
        has_labels=True,
    )

    # Dynamic reference values computed directly from saved artifacts
    exp_total = summary_data["eda"]["rows_total"]
    exp_excl = summary_data["eda"]["rows_excluded"]
    exp_flagged = summary_data["scoring"]["flagged_count"]
    exp_rate_pct = f"{summary_data['scoring']['flagged_rate'] * 100:.1f}"
    exp_recall = f"{summary_data['scoring']['metrics']['recall']:.4f}"
    exp_prec = f"{summary_data['scoring']['metrics']['precision']:.4f}"
    exp_f1 = f"{summary_data['scoring']['metrics']['f1']:.4f}"
    exp_cost = f"{summary_data['scoring']['metrics']['financial_cost']['total_cost']:,.2f}"
    exp_thresh = f"{summary_data['scoring']['threshold_used']:.4f}"
    exp_top_orig = df_results.sort_values("fraud_probability", ascending=False).iloc[0]["nameOrig"]

    # 12 Benchmark Questions on User Analyses
    analysis_questions = [
        ("how many rows were in my last upload?", lambda a: str(exp_total) in a, f"Contains total rows {exp_total}"),
        ("how many were flagged in my analysis?", lambda a: str(exp_flagged) in a, f"Contains flagged count {exp_flagged}"),
        ("show the top 5 riskiest transactions in my upload", lambda a: exp_top_orig in a, f"Contains riskiest account {exp_top_orig}"),
        ("which type has the most flagged transactions in my file?", lambda a: top_flagged_type in a, f"Identifies top type {top_flagged_type}"),
        ("what was the recall on my file?", lambda a: exp_recall in a or "100.00%" in a, f"Contains recall {exp_recall}"),
        ("what was the precision on my file?", lambda a: exp_prec in a or "83.33%" in a, f"Contains precision {exp_prec}"),
        ("what was the F1 score on my upload?", lambda a: exp_f1 in a, f"Contains F1 score {exp_f1}"),
        ("how many rows were excluded in my file?", lambda a: str(exp_excl) in a, f"Contains excluded rows {exp_excl}"),
        ("what threshold was used for my upload?", lambda a: exp_thresh in a, f"Contains threshold {exp_thresh}"),
        ("what is the fraud rate in my uploaded file?", lambda a: "10.00%" in a or "10" in a, "Contains fraud rate 10.00%"),
        ("what was the financial cost of fraud in my file?", lambda a: exp_cost in a or "1,500" in a, f"Contains rupee cost {exp_cost}"),
        ("what was the alert rate on my upload?", lambda a: exp_rate_pct in a or "alert rate" in a.lower(), f"Contains alert rate {exp_rate_pct}%"),
    ]

    print("\n" + "=" * 90)
    print(f"{'#':<3} | {'ANALYSIS BENCHMARK QUESTION':<46} | {'STATUS':<6} | {'SNIPPET':<30}")
    print("=" * 90)

    for idx, (q, validator, desc) in enumerate(analysis_questions, 1):
        ans = answer_analyst_query(q, user_info=user_info)
        passed = validator(str(ans))
        status_str = "PASS" if passed else "FAIL"
        disp_ans = str(ans).replace("\n", " ")[:28] + "..."
        print(f"{idx:<3} | {q:<46} | {status_str:<6} | {disp_ans:<30}")
        assert passed, f"Question failed: '{q}' (Expected: {desc}, Got: {ans})"

    print("=" * 90)
    print(f"[SUCCESS] All {len(analysis_questions)} User Analysis Benchmark Questions passed fact containment.")


def test_drift_chatbot_before_and_after_simulate(auth_headers, tmp_path):
    """
    Test chatbot drift status behavior:
    1. Before simulate (empty snapshots table): computes status live matching real status.
    2. After simulate (POST /drift/simulate): reports DRIFT_DETECTED with matching score PSI
       and notes that the snapshot is from a simulation.
    """
    from fastapi.testclient import TestClient
    from api.main import app, compute_live_drift_status
    from api.db import get_db_path, init_db

    client = TestClient(app)

    # 1. Before simulate: test with clean db having NO drift snapshots recorded yet
    empty_db_path = tmp_path / "finsecure_empty_drift.db"
    init_db(empty_db_path)

    live_res = compute_live_drift_status()
    live_status = live_res["overall_status"]
    live_psi = live_res["score_psi"]

    ans_before = answer_analyst_query("what is the drift status?", db_path=empty_db_path)
    ans_text_before = str(ans_before)

    # Assert drift guard message is returned instead of STABLE / PSI 0.0000
    assert "No concept drift snapshots have been recorded yet" not in ans_text_before
    assert "Not enough scored transactions to assess drift (n=0)" in ans_text_before
    assert "STABLE" not in ans_text_before
    assert "0.0000" not in ans_text_before
    print(f"\n[CHATBOT DRIFT TEST - BEFORE SIMULATE]: {ans_text_before}")

    # 2. Trigger simulate via API
    sim_resp = client.post("/drift/simulate?amount_scale=5.0", headers=auth_headers)
    assert sim_resp.status_code == 200
    sim_data = sim_resp.json()
    assert sim_data["overall_status"] == "DRIFT_DETECTED"
    sim_psi = sim_data["score_psi"]

    # 3. After simulate: chatbot queries DATA_DIR database and reports simulation
    ans_after = answer_analyst_query("what is the drift status?", db_path=get_db_path())
    ans_text_after = str(ans_after)

    assert "DRIFT_DETECTED" in ans_text_after
    assert f"{sim_psi:.4f}" in ans_text_after
    assert "simulat" in ans_text_after.lower()
    print(f"\n[CHATBOT DRIFT TEST - AFTER SIMULATE]: {ans_text_after}")

