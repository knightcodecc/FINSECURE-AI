"""
test_api.py — API, Integration, System, and Security Test Suite for FinSecure AI v2.
=====================================================================================
Covers:
- Unit: API validation returning 422 on malformed or missing transaction inputs
- Integration: FastAPI endpoint -> XGBoost Model -> SHAP Explainer -> SQLite DB
- System: Full end-to-end flow: score a transaction, see it in Live Predictions, get explanation
- Security: Unauthorized access returning 401, SQL safety guardrails
"""

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.db import DB_PATH, get_db_connection
from api.chat_agent import run_readonly_sql


def test_predict_validation_missing_fields_returns_422(auth_headers):
    """Unit: Verify that sending an incomplete payload to /predict returns HTTP 422."""
    client = TestClient(app, headers=auth_headers)
    # Missing required amount, nameOrig, nameDest, etc.
    resp = client.post("/predict", json={"step": 1, "type": "TRANSFER"})
    assert resp.status_code == 422, f"Expected 422 for missing fields, got {resp.status_code}"


def test_predict_validation_invalid_data_types_returns_422(auth_headers):
    """Unit: Verify that sending non-numeric amount returns HTTP 422."""
    client = TestClient(app, headers=auth_headers)
    bad_payload = {
        "step": "not-an-int",
        "type": "TRANSFER",
        "amount": "invalid-amount-string",
        "nameOrig": "C_TEST",
        "oldbalanceOrg": 1000.0,
        "newbalanceOrig": 0.0,
        "nameDest": "M_TEST",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 1000.0
    }
    resp = client.post("/predict", json=bad_payload)
    assert resp.status_code == 422, f"Expected 422 for invalid types, got {resp.status_code}"


def test_integration_model_to_sqlite_logging(auth_headers):
    """Integration: Verify that /predict scores via XGBoost and commits to SQLite with SHAP JSON."""
    client = TestClient(app, headers=auth_headers)
    tx_payload = {
        "step": 605,
        "type": "CASH_OUT",
        "amount": 75000.0,
        "nameOrig": "C_INTEG_ORIG_1",
        "oldbalanceOrg": 75000.0,
        "newbalanceOrig": 0.0,
        "nameDest": "M_INTEG_DEST_1",
        "oldbalanceDest": 5000.0,
        "newbalanceDest": 80000.0
    }
    resp = client.post("/predict", json=tx_payload)
    assert resp.status_code == 200, f"Predict failed: {resp.text}"
    data = resp.json()
    assert "id" in data
    assert "prediction_proba" in data
    assert "shap_top_features" in data
    assert "explanation" in data
    tx_id = data["id"]

    # Verify SQLite row
    with get_db_connection(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name_orig, amount, prediction_proba, shap_top_features_json FROM predictions WHERE id = ?", (tx_id,))
        row = cursor.fetchone()
        assert row is not None
        assert row[1] == "C_INTEG_ORIG_1"
        assert abs(row[2] - 75000.0) < 1e-4
        assert row[4] is not None


def test_system_full_flow_score_and_retrieve_in_recent_predictions(auth_headers):
    """System: Full flow: score a transaction, verify presence in /predictions/recent, and query chat."""
    client = TestClient(app, headers=auth_headers)
    unique_orig = "C_SYSTEM_FLOW_999"
    tx_payload = {
        "step": 610,
        "type": "TRANSFER",
        "amount": 350000.0,
        "nameOrig": unique_orig,
        "oldbalanceOrg": 350000.0,
        "newbalanceOrig": 0.0,
        "nameDest": "M_SYSTEM_HUB_888",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 350000.0
    }

    # 1. Score
    score_resp = client.post("/predict", json=tx_payload)
    assert score_resp.status_code == 200
    score_data = score_resp.json()
    created_id = score_data["id"]

    # 2. See in Live Predictions recent list
    recent_resp = client.get("/predictions/recent?limit=10")
    assert recent_resp.status_code == 200
    recent_items = recent_resp.json()
    matching = [item for item in recent_items if item["id"] == created_id]
    assert len(matching) == 1, f"Transaction #{created_id} not found in recent predictions list"
    assert matching[0]["nameOrig"] == unique_orig

    # 3. Ask Chatbot about the last decision
    chat_resp = client.post("/chat", json={"query": "why was the last transaction flagged"})
    assert chat_resp.status_code == 200
    chat_data = chat_resp.json()
    assert "answer" in chat_data
    assert len(chat_data["answer"]) > 0


def test_security_unauthenticated_request_returns_401():
    """Security: Verify protected endpoints reject requests lacking valid Bearer token."""
    client = TestClient(app)
    # /predict without auth
    resp = client.post("/predict", json={"step": 1, "type": "TRANSFER", "amount": 100.0})
    assert resp.status_code == 401, f"Expected 401 Unauthorized, got {resp.status_code}"

    # /model/info without auth
    info_resp = client.get("/model/info")
    assert info_resp.status_code == 401


def test_security_sql_injection_defense():
    """Security: Verify read-only SQL tool rejects SQL injection and mutation payloads."""
    # Attempt DROP
    drop_attempt = run_readonly_sql("SELECT 1; DROP TABLE predictions;--")
    assert "error" in drop_attempt

    # Attempt UPDATE
    update_attempt = run_readonly_sql("UPDATE predictions SET is_fraud = 0")
    assert "error" in update_attempt
    assert "read-only" in update_attempt["error"].lower()
