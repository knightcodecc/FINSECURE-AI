"""
test_serving.py
================
Unit tests for the serving layer:
- Serving features equal training features for the exact same input (zero train/serve skew)
- Single shared Scanner instance across /ws/scan and /ws/stats
- SQLite connection manager operates in WAL mode and supports batch logging
"""

import json
import sqlite3
import numpy as np
import pandas as pd
import pytest

from api.predictor import get_predictor, Predictor
from api.realtime_scanner import get_scanner, RealtimeScanner
from api.db import get_db_connection, log_prediction, log_predictions_batch, DB_PATH
from src.feature_engineering import engineer_tabular_features


from pathlib import Path

def test_serving_features_equal_training_features():
    """Verify that serving feature generation is identical to training feature engineering (Amendment 4)."""
    if not Path("models/xgboost_model.joblib").exists():
        pytest.skip("models/xgboost_model.joblib missing")
    predictor = get_predictor()

    raw_tx = {
        "step": 1,
        "type": "TRANSFER",
        "amount": 10000.0,
        "nameOrig": "C_TEST_SERVE_ORIG",
        "oldbalanceOrg": 50000.0,
        "newbalanceOrig": 40000.0,
        "nameDest": "M_TEST_SERVE_DEST",
        "oldbalanceDest": 1000.0,
        "newbalanceDest": 11000.0,
    }

    # Serving feature extraction dataframe
    df_serving = predictor.prepare_features(raw_tx)

    # Re-compute via training code path manually
    tx_dict = {
        "step": raw_tx["step"],
        "amount": raw_tx["amount"],
        "oldbalanceOrg": raw_tx["oldbalanceOrg"],
        "newbalanceOrig": raw_tx["newbalanceOrig"],
        "oldbalanceDest": raw_tx["oldbalanceDest"],
        "newbalanceDest": raw_tx["newbalanceDest"],
        "type_CASH_IN": 0.0,
        "type_CASH_OUT": 0.0,
        "type_DEBIT": 0.0,
        "type_PAYMENT": 0.0,
        "type_TRANSFER": 1.0,
        "orig_in_degree": 0.0,
        "orig_out_degree": 0.0,
        "orig_pagerank": 0.0,
        "orig_component_size": 1.0,
        "orig_fraud_exposure": 0.0,
        "dest_in_degree": 0.0,
        "dest_out_degree": 0.0,
        "dest_pagerank": 0.0,
        "dest_component_size": 1.0,
        "dest_fraud_exposure": 0.0,
    }
    df_training_raw = engineer_tabular_features(pd.DataFrame([tx_dict]))[predictor.feature_names]
    df_training_scaled = df_training_raw.copy()
    df_training_scaled[predictor.feature_names] = predictor.scaler.transform(df_training_raw)

    # Check value equality across all 33 features
    diff = np.max(np.abs(df_serving.values - df_training_scaled.values))
    assert diff < 1e-6, f"Train/serve skew detected! Maximum feature difference: {diff}"


def test_shared_scanner_singleton():
    """Verify single shared Scanner instance across endpoints (Amendment 4)."""
    scanner1 = get_scanner()
    scanner2 = get_scanner()

    assert scanner1 is scanner2, "Scanner singleton failed! Multiple instances exist."


def test_sqlite_wal_and_batch_logging(tmp_path):
    """Verify SQLite connection manager operates in WAL mode and supports batch inserts."""
    test_db = tmp_path / "test_finsecure.db"

    with get_db_connection(test_db) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode;")
        mode = cursor.fetchone()[0]
        assert mode.lower() == "wal", f"Expected WAL mode, got {mode}"

    batch_records = [
        {
            "timestamp": "2026-10-09T10:00:00",
            "step": 1,
            "nameOrig": f"C_{i}",
            "nameDest": f"M_{i}",
            "amount": 1000.0 * i,
            "prediction_proba": 0.1 * i,
            "is_fraud": i > 5,
            "threshold_used": 0.25,
            "features": {"amount": 1000.0 * i},
            "shap_top_features": [],
            "explanation": "Test batch item",
        }
        for i in range(1, 11)
    ]

    log_predictions_batch(batch_records, db_path=test_db)

    with get_db_connection(test_db) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM predictions;")
        count = cursor.fetchone()[0]
        assert count == 10, f"Expected 10 logged predictions, got {count}"


def test_get_metrics_cost_endpoint():
    """Verify GET /metrics/cost endpoint returns 200 and expected threshold keys."""
    from fastapi.testclient import TestClient
    from api.main import app

    if not Path("models/model_metadata.joblib").exists():
        pytest.skip("models/model_metadata.joblib missing")
    client = TestClient(app)
    response = client.get("/metrics/cost")
    assert response.status_code == 200
    data = response.json()
    assert "f2_optimal_threshold" in data
    assert "cost_optimal_threshold" in data
    assert "active_threshold" in data


def test_predict_threshold_used_matches_metrics_cost_active_threshold():
    """Verify that /predict's threshold_used strictly equals /metrics/cost's active_threshold (Amendment 1/2)."""
    from fastapi.testclient import TestClient
    from api.main import app

    if not Path("models/model_metadata.joblib").exists() or not Path("models/xgboost_model.joblib").exists():
        pytest.skip("Model or metadata joblib missing")

    client = TestClient(app)
    cost_res = client.get("/metrics/cost").json()
    active_thresh = cost_res["active_threshold"]

    raw_tx = {
        "step": 500,
        "type": "TRANSFER",
        "amount": 250000.0,
        "nameOrig": "C_TEST_SYNC_ORIG",
        "oldbalanceOrg": 300000.0,
        "newbalanceOrig": 50000.0,
        "nameDest": "M_TEST_SYNC_DEST",
        "oldbalanceDest": 10000.0,
        "newbalanceDest": 260000.0
    }
    pred_res = client.post("/predict", json=raw_tx).json()
    threshold_used = pred_res["threshold_used"]

    assert abs(threshold_used - active_thresh) < 1e-6, f"Predict threshold {threshold_used} != active threshold {active_thresh}"


def test_drift_endpoints_fastapi():
    """Verify GET /drift/status and POST /drift/simulate via FastAPI TestClient."""
    from fastapi.testclient import TestClient
    from api.main import app

    if not (Path("data/features_train.parquet").exists() and Path("data/features_val.parquet").exists()):
        pytest.skip("Training/Val feature data missing")

    client = TestClient(app)

    # 1. Check GET /drift/status
    res_status = client.get("/drift/status")
    assert res_status.status_code == 200
    status_data = res_status.json()
    assert "overall_status" in status_data
    assert "feature_psi" in status_data

    # 2. Check POST /drift/simulate (amount_scale=5.0)
    res_sim = client.post("/drift/simulate?amount_scale=5.0")
    assert res_sim.status_code == 200
    sim_data = res_sim.json()
    assert sim_data["retrain_recommended"] is True
    assert sim_data["overall_status"] in ("DRIFT_DETECTED", "WARNING")


def test_send_alert_webhook_mocked(monkeypatch):
    """Verify alert webhook dispatcher sends JSON payload when URL configured."""
    from unittest.mock import MagicMock
    from api.alerts import send_alert_webhook

    monkeypatch.setenv("ALERT_WEBHOOK_URL", "https://example.com/webhook")

    mock_resp = MagicMock()
    mock_resp.status_code = 200

    monkeypatch.setattr("httpx.post", lambda url, json, timeout: mock_resp)
    success = send_alert_webhook("HIGH_FRAUD_ALERT", {"tx_id": "123", "amount": 50000})
    assert success is True


def test_sqlite_shap_top_features_json_column(tmp_path):
    """Verify SQLite predictions table logs and parses shap_top_features_json column."""
    test_db = tmp_path / "test_shap_db.db"

    record = {
        "timestamp": "2026-10-09T12:00:00",
        "step": 1,
        "nameOrig": "C1111",
        "nameDest": "M2222",
        "amount": 5000.0,
        "prediction_proba": 0.85,
        "is_fraud": True,
        "threshold_used": 0.2971,
        "features": {"amount": 5000.0},
        "shap_top_features": [
            {"feature": "amount", "human_name": "Transaction Amount", "raw_value": 5000.0, "shap_value": 1.5, "direction": "pushes_toward_fraud"}
        ],
        "explanation": "High transaction amount",
    }

    log_prediction(record, db_path=test_db)

    with get_db_connection(test_db) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT shap_top_features_json FROM predictions WHERE name_orig = 'C1111';")
        row = cursor.fetchone()
        assert row is not None
        shap_json_str = row[0]
        parsed_shap = json.loads(shap_json_str)
        assert len(parsed_shap) == 1
        assert parsed_shap[0]["feature"] == "amount"
        assert parsed_shap[0]["shap_value"] == 1.5


def test_predictor_strictly_positive_amount_assertion():
    """Verify that predictor asserts amount > 0 and fails loudly on scaled or negative values."""
    predictor = get_predictor()
    bad_tx = {
        "step": 1,
        "type": "CASH_OUT",
        "amount": -0.1259,  # Scaled z-score value
        "nameOrig": "C_TEST",
        "nameDest": "M_TEST",
    }
    with pytest.raises(AssertionError, match="strictly positive"):
        predictor.prepare_features(bad_tx)


def test_serving_parity_small_slice():
    """Verify live serving parity against batch inference on a small slice of clean-test transactions."""
    raw_path = Path("data/paysim_synth.csv") if Path("data/paysim_synth.csv").exists() else Path("data/transactions_with_graph_features.parquet")
    if not raw_path.exists() or not Path("data/features_test.parquet").exists():
        pytest.skip("Dataset files missing")
        
    predictor = get_predictor(warm=True)
    raw_df = pd.read_csv(raw_path) if raw_path.suffix == ".csv" else pd.read_parquet(raw_path)
    raw_sorted = raw_df.sort_values("step").reset_index(drop=True)
    clean_test_raw = raw_sorted[(raw_sorted["step"] > 540) & (raw_sorted["step"] <= 629)].reset_index(drop=True).iloc[:20]
    batch_feat = pd.read_parquet("data/features_test.parquet").iloc[:20]
    
    live_probs = []
    for i in range(len(clean_test_raw)):
        r = clean_test_raw.iloc[i]
        tx = {
            "step": int(r["step"]),
            "type": str(r["type"]),
            "amount": float(r["amount"]),
            "nameOrig": str(r["nameOrig"]),
            "oldbalanceOrg": float(r["oldbalanceOrg"]),
            "newbalanceOrig": float(r["newbalanceOrig"]),
            "nameDest": str(r["nameDest"]),
            "oldbalanceDest": float(r["oldbalanceDest"]),
            "newbalanceDest": float(r["newbalanceDest"]),
        }
        res = predictor.predict_one(tx)
        live_probs.append(res["prediction_proba"])
        
    batch_probs = predictor.model.predict_proba(batch_feat[predictor.feature_names])[:, 1]
    diffs = np.abs(np.array(live_probs) - batch_probs)
    assert np.mean(diffs) < 0.10, f"Mean diff {np.mean(diffs)} exceeded threshold"
    assert not np.isnan(diffs).any()


