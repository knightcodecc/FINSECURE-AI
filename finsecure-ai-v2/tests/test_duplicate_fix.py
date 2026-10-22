"""
test_duplicate_fix.py — Verification Suite for Duplicate Transaction Prevention.
================================================================================
Assimilates requirements for Section 1:
- Exactly one log_prediction call per request
- 1 /predict call -> exactly +1 row in the predictions table
- 5 /predict calls -> exactly +5 rows in the predictions table
- Returned response strictly includes server-side prediction primary key ID
"""

import sqlite3
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from api.main import app
from api.db import DB_PATH, get_db_connection


def get_prediction_row_count() -> int:
    """Helper to query current prediction row count directly from SQLite."""
    if not DB_PATH.exists():
        return 0
    with get_db_connection(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM predictions")
        return cursor.fetchone()[0]


def test_single_predict_call_adds_exactly_one_row(auth_headers):
    """Verify that calling POST /predict increases predictions table row count by exactly +1 and returns server ID."""
    client = TestClient(app, headers=auth_headers)
    initial_count = get_prediction_row_count()

    tx_payload = {
        "step": 600,
        "type": "PAYMENT",
        "amount": 1250.75,
        "nameOrig": "C_TEST_DUP_1",
        "oldbalanceOrg": 5000.0,
        "newbalanceOrig": 3749.25,
        "nameDest": "M_TEST_MERCH_1",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0
    }

    resp = client.post("/predict", json=tx_payload)
    assert resp.status_code == 200, f"Predict failed: {resp.text}"
    data = resp.json()

    # 1. Returned payload MUST contain the server primary key ID
    assert "id" in data
    assert isinstance(data["id"], int)
    assert data["id"] > 0
    server_id = data["id"]

    # 2. SQLite row count must increase by exactly +1
    new_count = get_prediction_row_count()
    assert new_count == initial_count + 1, f"Expected {initial_count + 1} rows, found {new_count}"

    # 3. Verify record in SQLite matches the returned ID
    with get_db_connection(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name_orig, amount FROM predictions WHERE id = ?", (server_id,))
        row = cursor.fetchone()
        assert row is not None, f"Prediction with ID {server_id} not found in SQLite"
        assert row[1] == "C_TEST_DUP_1"
        assert abs(row[2] - 1250.75) < 1e-4


def test_multiple_predict_calls_strictly_sequential(auth_headers):
    """Verify that submitting 5 distinct transactions results in exactly +5 rows with unique sequential IDs."""
    client = TestClient(app, headers=auth_headers)
    start_count = get_prediction_row_count()
    created_ids = []

    for i in range(5):
        tx_payload = {
            "step": 601 + i,
            "type": "TRANSFER",
            "amount": 1000.0 * (i + 1),
            "nameOrig": f"C_SEQ_ORIG_{i}",
            "oldbalanceOrg": 10000.0,
            "newbalanceOrig": 10000.0 - 1000.0 * (i + 1),
            "nameDest": f"M_SEQ_DEST_{i}",
            "oldbalanceDest": 500.0,
            "newbalanceDest": 500.0 + 1000.0 * (i + 1)
        }
        resp = client.post("/predict", json=tx_payload)
        assert resp.status_code == 200
        p_id = resp.json()["id"]
        assert p_id not in created_ids, f"Duplicate ID generated: {p_id}"
        created_ids.append(p_id)

    final_count = get_prediction_row_count()
    assert final_count == start_count + 5, f"Expected exactly +5 rows ({start_count + 5}), got {final_count}"
    assert len(set(created_ids)) == 5
