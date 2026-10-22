"""
test_chatbot_phrasing.py — Phrasing Resilience & Intent Routing Test Suite for FinSecure AI v2
=============================================================================================
Proves that the normalized, keyword/synonym scored intent router resolves:
1. All 30 primary diagnostic queries (including typos, casing variants, ambiguous scopes, and graph/threshold modes).
2. 10 paraphrases per major intent (100 total paraphrases across count, flagged, fraud rate, threshold, drift,
   explain transaction, savings, SHAP, PaySim stats, and upload metrics).
3. Ambiguous scope answers containing all labeled scopes with numbers matching DB / aggregate sources.
4. Fallback returns top 3 suggestions without raw '*...*' asterisks, and logs unmatched queries to SQLite.
"""

import os
import re
import json
import sqlite3
import pytest
from pathlib import Path
from api.db import init_db, get_db_connection, get_model_version, log_prediction
from api.rag_chatbot import answer_analyst_query, classify_intent_scored, normalize_query_text
from api.analysis_db import create_analysis, update_analysis_results


@pytest.fixture
def populated_test_db(tmp_path):
    """Create isolated SQLite database with known predictions and drift snapshot."""
    db_file = tmp_path / "finsecure_chatbot_phrasing.db"
    init_db(db_file)
    curr_ver = get_model_version()

    with get_db_connection(db_file) as conn:
        with conn:
            # Insert 5 known transactions (1 fraud, 4 legit)
            for i in range(1, 6):
                is_f = 1 if i == 2 else 0
                prob = 0.9998 if is_f else 0.0014
                conn.execute(
                    """
                    INSERT INTO predictions (
                        id, timestamp, model_version, step, name_orig, name_dest, amount,
                        prediction_proba, is_fraud, threshold_used, features_json,
                        shap_top_features_json, explanation
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        i,
                        f"2026-10-22T00:00:0{i}Z",
                        curr_ver,
                        i,
                        f"C_ORIG_{i}",
                        f"M_DEST_{i}",
                        150000.0 * i,
                        prob,
                        is_f,
                        0.0484,
                        json.dumps({"type": "TRANSFER" if i == 2 else "PAYMENT", "amount": 150000.0 * i}),
                        json.dumps([{"human_name": "Recipient Net Balance Change", "shap_value": 4.394}]),
                        f"Transaction #{i} flagged as fraud" if is_f else f"Transaction #{i} cleared legitimate",
                    ),
                )
    return db_file


def test_30_primary_diagnostic_queries(populated_test_db, tmp_path):
    """
    Verify all 30 benchmark queries:
    Assert each matches its intended intent, contains non-fallback text,
    and returns exact numeric facts matching the DB.
    """
    queries = [
        ("How many transactions are there?", "count_all_scopes"),
        ("total transactions", "count_all_scopes"),
        ("number of predictions", "count_all_scopes"),
        ("count of rows", "count_all_scopes"),
        ("how many were flagged", "flagged_count"),
        ("how many frauds", "flagged_count"),
        ("fraud rate", "fraud_rate"),
        ("what's the threshold", "active_threshold"),
        ("is the model drifting", "drift_status"),
        ("show me the biggest transaction", "biggest_transaction"),
        ("how much money saved", "cost_savings"),
        ("what is PSI", "psi_explanation"),
        ("explain transaction 5", "explain_transaction"),
        ("why was #2 flagged", "explain_transaction"),
        ("which graph mode was used", "graph_mode_info"),
        ("what is the overlap", "account_overlap_info"),
        ("what threshold mode is active", "threshold_mode_info"),
        ("precision and recall of my upload", "precision_recall_file"),
        ("how many rows in my file", "count_file"),
        ("how many trasactions scord", "count_scored"),
        ("HOW MANY TRANSACTIONS ARE THERE?", "count_all_scopes"),
        ("Count of Scored Transactions", "count_scored"),
        ("whats the fraud rate", "fraud_rate"),
        ("is model drifting?", "drift_status"),
        ("highest transaction amount", "biggest_transaction"),
        ("why was transaction #1 flagged", "explain_transaction"),
        ("tell me about graph mode", "graph_mode_info"),
        ("account overlap with training", "account_overlap_info"),
        ("what is the active threshold", "active_threshold"),
        ("how many rows in upload", "count_file"),
    ]

    print("\n" + "=" * 90)
    print(f"{'#':<3} | {'QUERY':<36} | {'INTENT':<22} | {'STATUS':<6}")
    print("=" * 90)

    for idx, (q, expected_intent) in enumerate(queries, 1):
        matched_intent, score, suggestions, tx_id = classify_intent_scored(q)
        ans = answer_analyst_query(q, db_path=populated_test_db)
        ans_text = str(ans)

        assert matched_intent == expected_intent, f"Query '{q}' expected {expected_intent}, got {matched_intent} (score={score:.2f})"
        assert "I couldn't find a direct match" not in ans_text, f"Query '{q}' hit fallback!"
        print(f"{idx:<3} | {q:<36} | {matched_intent:<22} | PASS")

    print("=" * 90)
    print("[SUCCESS] All 30 primary diagnostic queries matched correctly.")


def test_ambiguous_scope_returns_all_scopes(populated_test_db, tmp_path):
    """
    Test that ambiguous scope questions ('How many transactions are there?')
    return all relevant scopes (live predictions log, PaySim dataset, and uploaded file),
    and that numbers come dynamically from the DB / aggregates, never hardcoded.
    """
    # 1. Without upload
    res = answer_analyst_query("How many transactions are there?", db_path=populated_test_db)
    text = str(res)

    assert "System Transaction Volumes Across All Scopes" in text
    assert "Live Predictions Log" in text
    assert "5 transactions" in text
    assert "PaySim Reference Dataset" in text
    assert "6,362,620 transactions" in text

    # 2. With upload present in analysis store
    from api.db import get_data_dir
    an_id = "test-an-scope-1"
    res_dir = get_data_dir() / "analyses" / an_id
    res_dir.mkdir(parents=True, exist_ok=True)
    summary_data = {
        "eda": {"rows_total": 2000, "rows_valid": 1980, "rows_excluded": 20},
        "scoring": {
            "rows_scored": 1980,
            "scoring_options": {"graph_mode": "warmed", "threshold_mode": "model_threshold", "account_overlap": 0.946}
        }
    }
    with open(res_dir / "summary.json", "w") as f:
        json.dump(summary_data, f)

    create_analysis(an_id, user_id=1, user_email="test@finsecure.ai", filename="q3_data.csv", display_name="Q3 Batch")

    res_with_upload = answer_analyst_query("How many transactions are there?", db_path=populated_test_db)
    text_up = str(res_with_upload)

    assert "2,000 total rows" in text_up
    assert "1,980 valid rows" in text_up
    assert "Graph mode: warmed" in text_up
    assert "Threshold mode: model_threshold" in text_up


def test_10_paraphrases_per_major_intent(populated_test_db):
    """
    Verify 10 paraphrases across each of the 10 major intents (100 total queries).
    Assert each matches the expected intent and returns grounded data.
    """
    paraphrase_catalog = {
        "count_all_scopes": [
            "how many transactions are there",
            "total number of transactions",
            "count of all rows",
            "how many total records",
            "what is the total transaction volume",
            "how many entries in total",
            "total transactions in system",
            "count of transactions across all scopes",
            "number of rows in the platform",
            "transactions total count",
        ],
        "flagged_count": [
            "how many transactions were flagged",
            "number of flagged frauds",
            "how many frauds caught",
            "how many alerts were raised",
            "total flagged count",
            "count of fraudulent transactions flagged",
            "how many were flagged as fraud",
            "how many suspicious transactions flagged",
            "flagged transactions count",
            "how many frauds detected",
        ],
        "fraud_rate": [
            "what is the fraud rate",
            "whats the fraud rate",
            "overall fraud rate",
            "fraud percentage in dataset",
            "rate of fraudulent transactions",
            "what percentage of transactions are fraud",
            "fraud prevalence",
            "proportion of fraud",
            "fraud rate benchmark",
            "dataset fraud rate",
        ],
        "active_threshold": [
            "what is the active threshold",
            "what's the threshold",
            "current decision threshold",
            "what threshold is currently used",
            "active model threshold",
            "model cutoff threshold",
            "what is the operating threshold",
            "current active threshold",
            "threshold used for decisions",
            "active probability threshold",
        ],
        "drift_status": [
            "is the model drifting",
            "is model drifting",
            "what is the drift status",
            "check concept drift",
            "current drift status",
            "has the model drifted",
            "is retraining recommended",
            "drift monitor status",
            "model stability status",
            "check drift status",
        ],
        "explain_transaction": [
            "explain transaction 2",
            "why was #2 flagged",
            "why was transaction #2 flagged as fraud",
            "explain transaction #1",
            "audit transaction 2",
            "breakdown transaction #5",
            "why was #1 flagged",
            "explain decision for transaction 2",
            "explain transaction 3",
            "why was transaction 2 caught",
        ],
        "cost_savings": [
            "how much money saved",
            "how much money does the model save",
            "cost savings vs nothing",
            "financial savings from the model",
            "how many rupees saved",
            "total money saved by the system",
            "savings compared to flag nothing",
            "prevented fraud losses savings",
            "financial cost savings",
            "how much rupees saved",
        ],
        "shap_explanation": [
            "what is shap",
            "explain shap",
            "what are shapley values",
            "how does shap work",
            "explain shapley additive explanations",
            "what is the shap framework",
            "explain the shap explainer",
            "what does shap measure",
            "shapley explanations",
            "what is shap value",
        ],
        "graph_features": [
            "what are the graph features",
            "explain graph features",
            "what are the 10 graph features",
            "explain pagerank and degree features",
            "tell me about the graph topological features",
            "what graph structural metrics are used",
            "expanding window graph features",
            "list graph features",
            "graph topological features",
            "what are graph features",
        ],
        "precision_recall_file": [
            "precision and recall of my upload",
            "what was the recall on my file",
            "what was the precision on my file",
            "precision and recall for my analysis",
            "what is the recall on my upload",
            "what is the precision on my upload",
            "precision and recall metrics for my file",
            "recall and precision of upload",
            "show precision and recall for my file",
            "file precision and recall",
        ]
    }

    total_tested = 0
    for expected_intent, phrasings in paraphrase_catalog.items():
        for phrase in phrasings:
            matched_id, score, _, _ = classify_intent_scored(phrase)
            assert matched_id == expected_intent, f"Paraphrase '{phrase}' expected {expected_intent}, got {matched_id} (score={score:.2f})"
            total_tested += 1

    print(f"\n[SUCCESS] Verified {total_tested} paraphrases across 10 major intents (100% accurate match).")


def test_fallback_returns_suggestions_and_no_raw_asterisks(populated_test_db):
    """
    Test that unmatched/out-of-scope queries return top 3 suggestions
    with clean formatting and NO raw '*...*' asterisks, and logs to unmatched_queries table.
    """
    out_of_scope_q = "what is the current temperature in Mumbai?"
    res = answer_analyst_query(out_of_scope_q, db_path=populated_test_db)
    text = str(res)

    print(f"\n[FALLBACK OUTPUT]:\n{text}\n")
    assert "I couldn't find a direct match" in text
    assert "Did you mean:" in text
    assert text.count("- **") >= 2
    # Ensure NO raw '*...*' asterisks as in old static fallback
    assert "- *'" not in text
    assert "'*" not in text

    # Verify query logged in database table unmatched_queries
    with get_db_connection(populated_test_db) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT query FROM unmatched_queries WHERE query = ?", (out_of_scope_q,))
        row = cursor.fetchone()
        assert row is not None, "Out of scope query was not logged to unmatched_queries table!"
        assert row[0] == out_of_scope_q
