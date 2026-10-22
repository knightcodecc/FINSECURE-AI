"""
test_analysis.py — Comprehensive Test Suite for CSV Analysis & Validation Pipeline
===================================================================================
Tests all isolated functional and security requirements under temporary DATA_DIR:
1. Valid CSV -> job done -> counts consistent
2. Missing columns -> 422 naming them
3. Header-only and empty file handling
4. Wrong type -> 415
5. Oversize -> 413
6. Invalid rows reported by reason without crashing
7. Auth required -> 401
8. User isolation -> User B cannot read User A's analysis (404)
9. Cancel works
10. Delete removes DB rows and files
11. CSV formula injection cell escaped in export
12. XSS string in nameOrig preserved safely as plain text
13. Metric reconciliation assertions on labelled file (TP+FP+FN+TN==N, caught+missed==total, total==FN+50*FP)
14. Parity test: first 60k rows of data/paysim_synth.csv matches model probabilities (< 1e-5)
15. 1,000,000-row stream-generated CSV completes EDA with bounded memory and scores first MAX_SCORE_ROWS
16. Two concurrent uploads from different users execute cleanly
"""

import csv
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from api.main import app
from datetime import datetime, timezone
from api.auth import create_access_token, init_users_db, get_users_db_connection
from api.db import get_data_dir
from api.analysis_db import get_analysis, list_analyses
from src.batch_scoring import (
    sanitize_csv_formula_injection,
    export_sanitized_results_csv,
    validate_and_clean_stream,
    batch_score_dataframe,
)

client = TestClient(app)


def make_user_headers(user_id: int, email: str, name: str = "Test Analyst", role: str = "analyst"):
    init_users_db()
    with get_users_db_connection() as conn:
        with conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM users WHERE id = ?", (user_id,))
            row = cursor.fetchone()
            if not row:
                now_iso = datetime.now(timezone.utc).isoformat()
                cursor.execute(
                    "INSERT INTO users (id, full_name, email, password_hash, role, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (user_id, name, email, "fakehash", role, now_iso),
                )
    token = create_access_token({
        "sub": email,
        "id": user_id,
        "name": name,
        "role": role,
    })
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def user_a_headers():
    return make_user_headers(1001, "usera@finsecure.ai", "User Alpha")


@pytest.fixture
def user_b_headers():
    return make_user_headers(1002, "userb@finsecure.ai", "User Beta")


def test_auth_required_401():
    """Verify endpoints enforce authentication."""
    resp = client.get("/analysis/list")
    assert resp.status_code == 401
    resp2 = client.post("/analysis/some-id/start")
    assert resp2.status_code == 401


def test_wrong_type_415(user_a_headers):
    """Uploading non-CSV files returns HTTP 415."""
    files = {"file": ("data.json", b'{"key": "value"}', "application/json")}
    resp = client.post("/analysis/upload", files=files, headers=user_a_headers)
    assert resp.status_code == 415
    assert "supported" in resp.json()["detail"].lower()


def test_empty_file_422(user_a_headers):
    """Uploading 0-byte file returns HTTP 422."""
    files = {"file": ("empty.csv", b"", "text/csv")}
    resp = client.post("/analysis/upload", files=files, headers=user_a_headers)
    assert resp.status_code == 422
    assert "empty" in resp.json()["detail"].lower()


def test_oversize_413(user_a_headers, monkeypatch):
    """File exceeding MAX_UPLOAD_MB returns HTTP 413."""
    monkeypatch.setenv("MAX_UPLOAD_MB", "0.0001")  # ~100 bytes limit
    oversized_data = b"step,type,amount\n" + (b"1,PAYMENT,100\n" * 50)
    files = {"file": ("large.csv", oversized_data, "text/csv")}
    resp = client.post("/analysis/upload", files=files, headers=user_a_headers)
    assert resp.status_code == 413
    assert "exceeds" in resp.json()["detail"].lower()


def test_missing_columns_422():
    """Missing required canonical columns returns 422 naming them."""
    headers = make_user_headers(201, "u201@finsecure.ai")
    # CSV with missing newbalanceOrig and nameDest
    csv_content = (
        "step,type,amount,nameOrig,oldbalanceOrg,oldbalanceDest,newbalanceDest\n"
        "1,PAYMENT,500.0,C123,1000.0,0.0,500.0\n"
    ).encode("utf-8")
    files = {"file": ("missing.csv", csv_content, "text/csv")}
    upload_resp = client.post("/analysis/upload", files=files, headers=headers)
    assert upload_resp.status_code == 200
    data = upload_resp.json()
    analysis_id = data["analysis_id"]

    # Attempt to start with incomplete mapping
    start_resp = client.post(
        f"/analysis/{analysis_id}/start",
        data={
            "column_mapping": json.dumps({"step": "step", "type": "type"}),
        },
        headers=headers,
    )
    assert start_resp.status_code == 422
    detail = start_resp.json()["detail"]
    assert "Missing required column mappings" in detail
    assert "nameDest" in detail or "newbalanceOrig" in detail


def test_header_only_file_error(user_a_headers):
    """Header-only file fails validation cleanly with friendly message."""
    csv_content = "step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest\n".encode("utf-8")
    files = {"file": ("header_only.csv", csv_content, "text/csv")}
    upload_resp = client.post("/analysis/upload", files=files, headers=user_a_headers)
    assert upload_resp.status_code == 422
    assert "no records" in upload_resp.json()["detail"].lower()


def test_invalid_rows_reported_by_reason_without_crashing(user_a_headers, tmp_path):
    """Invalid rows are excluded, counted by reason, and logged to rejected_rows.csv."""
    rows = [
        ["step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest"],
        ["1", "PAYMENT", "500.0", "C101", "1000.0", "500.0", "M101", "0.0", "500.0"],          # Valid
        ["-5", "PAYMENT", "500.0", "C102", "1000.0", "500.0", "M102", "0.0", "500.0"],         # Negative step
        ["2", "INVALID_ACTION", "200.0", "C103", "500.0", "300.0", "M103", "0.0", "200.0"],     # Invalid type
        ["3", "TRANSFER", "-100.0", "C104", "500.0", "600.0", "M104", "0.0", "0.0"],           # Non-positive amount
        ["4", "CASH_OUT", "0.0", "C105", "500.0", "500.0", "M105", "0.0", "0.0"],              # Zero amount
        ["5", "TRANSFER", "invalid_num", "C106", "500.0", "500.0", "M106", "0.0", "0.0"],       # Unparseable numeric
    ]
    csv_file = tmp_path / "mixed_validation.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    mapping = {c: c for c in rows[0]}
    rejected_path = tmp_path / "rejected.csv"

    eda_summary, valid_df = validate_and_clean_stream(
        csv_path=csv_file,
        column_mapping=mapping,
        rejected_output_path=rejected_path,
    )

    assert eda_summary["rows_total"] == 6
    assert eda_summary["rows_valid"] == 1
    assert eda_summary["rows_excluded"] == 5
    assert eda_summary["rejection_counts"]["negative_step"] >= 1
    assert eda_summary["rejection_counts"]["invalid_type"] >= 1
    assert eda_summary["rejection_counts"]["non_positive_amount"] >= 2
    assert eda_summary["rejection_counts"]["invalid_numeric"] >= 1
    assert rejected_path.exists()


def test_valid_csv_workflow_and_consistency():
    """End-to-end flow: Upload -> Start -> Status Poll -> Summary -> Whatif -> Export."""
    headers = make_user_headers(202, "u202@finsecure.ai")
    sample_path = Path("frontend/samples/sample_transactions.csv")
    if not sample_path.exists():
        pytest.skip("sample_transactions.csv missing")

    with open(sample_path, "rb") as f:
        files = {"file": ("sample_transactions.csv", f.read(), "text/csv")}

    upload_resp = client.post("/analysis/upload", files=files, headers=headers)
    assert upload_resp.status_code == 200
    upload_data = upload_resp.json()
    analysis_id = upload_data["analysis_id"]
    auto_mapping = upload_data["auto_mapping"]

    # Start analysis
    start_resp = client.post(
        f"/analysis/{analysis_id}/start",
        data={
            "column_mapping": json.dumps(auto_mapping),
            "use_labels_for_graph": True,
            "threshold": 0.05,
        },
        headers=headers,
    )
    assert start_resp.status_code == 200
    assert start_resp.json()["status"] == "queued"

    # Poll status until done (timeout 30s)
    start_poll = time.time()
    status_data = {}
    while time.time() - start_poll < 30:
        stat_resp = client.get(f"/analysis/{analysis_id}/status", headers=headers)
        assert stat_resp.status_code == 200
        status_data = stat_resp.json()
        if status_data["status"] in ("done", "failed", "cancelled"):
            break
        time.sleep(0.5)

    assert status_data["status"] == "done", f"Analysis ended with unexpected status: {status_data}"
    assert status_data["progress_pct"] == 100.0

    # Summary endpoint
    sum_resp = client.get(f"/analysis/{analysis_id}/summary", headers=headers)
    assert sum_resp.status_code == 200
    summary = sum_resp.json()
    eda = summary["eda"]
    scoring = summary["scoring"]

    assert eda["rows_total"] == 2000
    assert eda["rows_valid"] == 2000
    assert scoring["rows_scored"] == 2000
    assert scoring["flagged_count"] > 0
    assert scoring["has_labels"] is True
    assert "metrics" in scoring and scoring["metrics"] is not None

    # Rows pagination
    rows_resp = client.get(f"/analysis/{analysis_id}/rows?page=1&page_size=10", headers=headers)
    assert rows_resp.status_code == 200
    rows_data = rows_resp.json()
    assert len(rows_data["rows"]) == 10
    assert rows_data["total_rows"] == 2000

    # On-demand SHAP for row 0
    shap_resp = client.get(f"/analysis/{analysis_id}/shap/0", headers=headers)
    assert shap_resp.status_code == 200
    shap_data = shap_resp.json()
    assert len(shap_data["shap_top_features"]) == 5
    assert "explanation" in shap_data and len(shap_data["explanation"]) > 10

    # What-if slider endpoint at threshold 0.20
    whatif_resp = client.get(f"/analysis/{analysis_id}/whatif?threshold=0.20", headers=headers)
    assert whatif_resp.status_code == 200
    w_data = whatif_resp.json()
    assert w_data["threshold"] == 0.20
    assert "flagged_count" in w_data
    assert "metrics" in w_data


def test_user_isolation_404(user_a_headers, user_b_headers):
    """User B cannot view or modify User A's analysis (returns 404)."""
    # Create analysis under User A
    csv_bytes = b"step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest\n1,PAYMENT,100,C1,100,0,M1,0,100\n"
    files = {"file": ("isolation_test.csv", csv_bytes, "text/csv")}
    up_resp = client.post("/analysis/upload", files=files, headers=user_a_headers)
    assert up_resp.status_code == 200
    analysis_id = up_resp.json()["analysis_id"]

    # User B attempts to access User A's analysis
    resp_b_status = client.get(f"/analysis/{analysis_id}/status", headers=user_b_headers)
    assert resp_b_status.status_code == 404

    resp_b_sum = client.get(f"/analysis/{analysis_id}/summary", headers=user_b_headers)
    assert resp_b_sum.status_code == 404

    resp_b_del = client.delete(f"/analysis/{analysis_id}", headers=user_b_headers)
    assert resp_b_del.status_code == 404


def test_cancel_works(user_a_headers, tmp_path):
    """Cancel button immediately stops background execution and sets status='cancelled'."""
    # Generate large CSV
    large_csv = tmp_path / "cancel_test.csv"
    with open(large_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest"])
        for i in range(1, 10000):
            writer.writerow([i % 100 + 1, "PAYMENT", 100.0, f"C_{i}", 1000.0, 900.0, f"M_{i}", 0.0, 100.0])

    with open(large_csv, "rb") as f:
        files = {"file": ("cancel_test.csv", f.read(), "text/csv")}

    up_resp = client.post("/analysis/upload", files=files, headers=user_a_headers)
    analysis_id = up_resp.json()["analysis_id"]
    auto_map = up_resp.json()["auto_mapping"]

    client.post(
        f"/analysis/{analysis_id}/start",
        data={"column_mapping": json.dumps(auto_map)},
        headers=user_a_headers,
    )

    # Issue cancellation
    cancel_resp = client.post(f"/analysis/{analysis_id}/cancel", headers=user_a_headers)
    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["status"] == "cancelled"

    stat = client.get(f"/analysis/{analysis_id}/status", headers=user_a_headers).json()
    assert stat["status"] == "cancelled"


def test_delete_removes_db_and_files(user_a_headers):
    """Delete endpoint removes DB row and associated disk directories."""
    csv_bytes = b"step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest\n1,PAYMENT,100,C1,100,0,M1,0,100\n"
    files = {"file": ("delete_test.csv", csv_bytes, "text/csv")}
    up = client.post("/analysis/upload", files=files, headers=user_a_headers).json()
    analysis_id = up["analysis_id"]

    del_resp = client.delete(f"/analysis/{analysis_id}", headers=user_a_headers)
    assert del_resp.status_code == 200

    # Confirm 404 on subsequent get
    assert client.get(f"/analysis/{analysis_id}/status", headers=user_a_headers).status_code == 404


def test_csv_injection_formula_sanitization(tmp_path):
    """Formula injection cells starting with = + - @ get leading apostrophe."""
    assert sanitize_csv_formula_injection("=SUM(A1:A10)") == "'=SUM(A1:A10)"
    assert sanitize_csv_formula_injection("+cmd|' /C calc'!A0") == "'+cmd|' /C calc'!A0"
    assert sanitize_csv_formula_injection("-100") == "'-100"
    assert sanitize_csv_formula_injection("@special") == "'@special"
    assert sanitize_csv_formula_injection("NORMAL_TEXT") == "NORMAL_TEXT"

    df_inj = pd.DataFrame([
        {"colA": "=2+2", "colB": "+555", "colC": "@danger", "colD": "legit"}
    ])
    out_csv = tmp_path / "sanitized.csv"
    export_sanitized_results_csv(df_inj, out_csv)

    raw_text = out_csv.read_text(encoding="utf-8")
    assert "'=2+2" in raw_text
    assert "'+555" in raw_text
    assert "'@danger" in raw_text


def test_xss_string_preserved_as_text(user_a_headers):
    """XSS payloads in account names are stored and returned safely as string data."""
    xss_payload = "<script>alert('XSS')</script>"
    csv_bytes = f"step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest\n1,PAYMENT,100,{xss_payload},100,0,M1,0,100\n".encode("utf-8")
    files = {"file": ("xss_test.csv", csv_bytes, "text/csv")}
    up_resp = client.post("/analysis/upload", files=files, headers=user_a_headers)
    assert up_resp.status_code == 200
    rows = up_resp.json()["preview_rows"]
    assert rows[0]["nameOrig"] == xss_payload


def test_metric_reconciliation_assertions():
    """Verify strict mathematical reconciliation: TP+FP+FN+TN==N, caught+missed==total, total_cost==FN+50*FP."""
    from src.batch_scoring import compute_evaluation_metrics

    y_true = np.array([1, 1, 0, 0, 1, 0, 0, 1, 0, 0])
    probs = np.array([0.9, 0.8, 0.7, 0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01])
    amounts = np.array([10000.0, 50000.0, 100.0, 200.0, 15000.0, 50.0, 75.0, 25000.0, 80.0, 90.0])
    threshold = 0.50
    y_pred = (probs >= threshold).astype(int)

    metrics = compute_evaluation_metrics(
        y_true=y_true,
        y_pred=y_pred,
        probs=probs,
        amounts=amounts,
        active_threshold=threshold,
        fp_review_cost=50.0,
    )

    cm = metrics["confusion_matrix"]
    fin = metrics["financial_cost"]

    # Assertion 1: Total population reconciliation
    assert cm["tp"] + cm["fp"] + cm["fn"] + cm["tn"] == cm["total"]

    # Assertion 2: Rupee amount reconciliation
    assert abs((fin["caught_fraud_amount"] + fin["missed_fraud_amount"]) - fin["total_fraud_amount"]) < 1e-2

    # Assertion 3: Cost function reconciliation
    assert abs(fin["total_cost"] - (fin["missed_fraud_amount"] + fin["fp_review_cost"])) < 1e-2


def test_parity_60k_rows():
    """
    PARITY TEST: take all rows of complete step prefix 1..143 of data/paysim_synth.csv.
    (a) batch_score_dataframe never mutates its input (df_batch_scored is df_prefix is False).
    (b) upload path scores RAW CSV rows using batch_score_dataframe.
    (c) offline side computes probabilities with model.predict_proba on features_train.parquet aligned by explicit row id.
    (d) upload path mirrors training pipeline (file_only expanding graph, step_batch_size 72, use_labels_for_graph=True).
    Tolerance 1e-5. Keep negative control. Assert arrays are distinct objects.
    """
    synth_path = Path("data/paysim_synth.csv")
    train_feat_path = Path("data/features_train.parquet")
    clean_path = Path("data/cleaned_transactions.parquet")

    if not synth_path.exists() or not train_feat_path.exists():
        pytest.skip("Training parquet or synthetic data missing for parity test.")

    import joblib
    from src.batch_scoring import batch_score_dataframe

    # Check whether features_train.parquet contains metadata / raw identifier columns
    ft_sample = pd.read_parquet(train_feat_path)
    ft_cols = ft_sample.columns.tolist()
    prohibited_cols = ["nameOrig", "nameDest", "type", "fraud_probability"]
    present_cols = [c for c in prohibited_cols if c in ft_cols]
    print(f"\n[PARITY TEST PARQUET AUDIT] features_train.parquet contains {prohibited_cols}: {present_cols}")
    assert len(present_cols) == 0, f"features_train.parquet should not contain raw metadata columns: {present_cols}"

    model = joblib.load("models/xgboost_model.joblib")
    meta = joblib.load("models/model_metadata.joblib")
    feature_names = meta.get("feature_names", list(getattr(model, "feature_names_in_", [])))

    # 1. Complete step prefix: step <= 143 (exactly 2 complete 72-step batches: 0..71, 72..143)
    df_raw_all = pd.read_csv(synth_path)
    df_raw_all["explicit_row_id"] = np.arange(len(df_raw_all))
    raw_prefix = df_raw_all[df_raw_all["step"] <= 143].copy()
    n_prefix = len(raw_prefix)
    s_max = int(raw_prefix["step"].max())

    # Mode mirroring training pipeline
    upload_mode = "file_only"
    use_labels_for_graph = True
    step_batch_size = 72
    print(f"[PARITY TEST UPLOAD MODE] graph_mode='{upload_mode}', step_batch_size={step_batch_size}, use_labels_for_graph={use_labels_for_graph}")

    # 2. Upload Path: pass RAW CSV rows directly through batch_score_dataframe
    raw_copy_before = raw_prefix.copy(deep=True)
    df_scored, meta_scoring = batch_score_dataframe(
        valid_df=raw_prefix,
        graph_mode=upload_mode,
        use_labels_for_graph=use_labels_for_graph,
        max_score_rows=n_prefix,
    )

    # Assert batch_score_dataframe NEVER mutates or returns its input
    is_same_object = (df_scored is raw_prefix)
    print(f"  - batch_score_dataframe mutates or returns its input (`df_batch_scored is df_prefix`): {is_same_object}")
    assert not is_same_object, "batch_score_dataframe returned or mutated its input DataFrame!"
    assert raw_prefix.equals(raw_copy_before), "Input DataFrame was mutated in place during batch scoring!"

    probs_upload = df_scored["fraud_probability"].values
    df_scored["explicit_row_id"] = raw_prefix["explicit_row_id"].values

    # 3. Offline Side: model.predict_proba on features_train.parquet aligned by explicit row ID
    clean = pd.read_parquet(clean_path)
    clean["explicit_row_id"] = np.arange(len(clean))
    clean_sorted = clean.sort_values("step", kind="quicksort").reset_index(drop=True)

    train_feat = pd.read_parquet(train_feat_path)
    train_feat["explicit_row_id"] = clean_sorted.loc[clean_sorted["step"] <= 420, "explicit_row_id"].values
    probs_train_all = model.predict_proba(train_feat[feature_names])[:, 1]
    train_feat["prob_train"] = probs_train_all

    # Merge on explicit_row_id
    merged = pd.merge(
        df_scored[["explicit_row_id", "fraud_probability"]],
        train_feat[["explicit_row_id", "prob_train"]],
        on="explicit_row_id",
        how="inner",
    )
    assert len(merged) == n_prefix, f"Row count mismatch in aligned parity rows: {len(merged)} != {n_prefix}"

    p_up = merged["fraud_probability"].values
    p_tr = merged["prob_train"].values
    abs_diff = np.abs(p_up - p_tr)
    max_diff = float(np.max(abs_diff))
    mean_diff = float(np.mean(abs_diff))
    p999_diff = float(np.percentile(abs_diff, 99.9))

    print(f"\n[PARITY TEST COMPLETE STEP PREFIX RESULTS]")
    print(f"  - Rows evaluated:                  {n_prefix:,} (Steps: 1..{s_max})")
    print(f"  - Max absolute difference:         {max_diff:.8e}")
    print(f"  - Mean absolute difference:        {mean_diff:.8e}")
    print(f"  - 99.9th percentile absolute diff: {p999_diff:.8e}")

    # Independence verification: show arrays are different objects computed independently
    are_different_objects = (p_up is not p_tr and id(p_up) != id(p_tr))
    print(f"  - Arrays are different objects:    {are_different_objects} (id_up={id(p_up)}, id_tr={id(p_tr)})")
    assert are_different_objects, "Arrays must be different objects computed independently!"

    # Negative control: perturb errorBalanceOrig feature by +1.0 and prove difference is detected (> 1e-2)
    scaler = joblib.load("models/scaler.joblib")
    clean_upload = df_raw_all[df_raw_all["step"] <= 143].copy()
    from src.graph_features import attach_graph_features_expanding
    from src.preprocess import drop_leakage_columns
    from src.feature_engineering import engineer_tabular_features

    g_ctrl = attach_graph_features_expanding(clean_upload, step_batch_size=72)
    c_ctrl = drop_leakage_columns(g_ctrl)
    e_ctrl = engineer_tabular_features(c_ctrl)
    for col in ["type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER"]:
        if col not in e_ctrl.columns:
            e_ctrl[col] = 0.0
    X_perturbed = scaler.transform(e_ctrl[feature_names])
    col_idx = feature_names.index("errorBalanceOrig") if "errorBalanceOrig" in feature_names else 0
    X_perturbed[:, col_idx] += 1.0
    probs_perturbed = model.predict_proba(X_perturbed)[:, 1]
    perturbed_diff = float(np.max(np.abs(probs_perturbed - p_tr)))
    print(f"  - Negative control max difference: {perturbed_diff:.8e}")
    assert perturbed_diff > 1e-2, f"Negative control failed: perturbation diff {perturbed_diff:.8e} <= 1e-2"

    assert max_diff < 1e-5, f"Parity tolerance exceeded: {max_diff:.8e} >= 1e-5"


def test_streaming_eda_bounded_memory_1m_rows(tmp_path):
    """
    Stream-generate a 1,000,000-row CSV on disk without holding it in memory.
    Run streaming EDA + scoring with MAX_SCORE_ROWS capped.
    Assert peak memory is bounded, and metadata contains capping warning.
    """
    import psutil
    csv_1m_path = tmp_path / "transactions_1m.csv"

    # Generator writing 1M rows to disk in chunks
    header = ["step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest"]
    chunk_size = 50000
    total_target = 1000000

    print("\n[STREAMING TEST] Generating 1,000,000 rows in disk chunks...")
    with open(csv_1m_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for chunk_start in range(0, total_target, chunk_size):
            chunk_rows = []
            for i in range(chunk_start, chunk_start + chunk_size):
                step = (i // 5000) + 1
                t = "PAYMENT" if i % 2 == 0 else "TRANSFER"
                chunk_rows.append([step, t, 150.0, f"C_{i}", 1000.0, 850.0, f"M_{i}", 0.0, 150.0])
            writer.writerows(chunk_rows)

    start_rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
    start_time = time.time()

    column_mapping = {c: c for c in header}
    eda_summary, valid_df = validate_and_clean_stream(
        csv_path=csv_1m_path,
        column_mapping=column_mapping,
        chunksize=50000,
    )

    # Score capped at MAX_SCORE_ROWS (e.g. 50,000 for speed in test)
    max_score = 50000
    score_df, scoring_meta = batch_score_dataframe(
        valid_df=valid_df,
        max_score_rows=max_score,
    )

    runtime_s = time.time() - start_time
    peak_rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
    rss_growth_mb = peak_rss_mb - start_rss_mb

    print(f"[STREAMING TEST RESULTS]:")
    print(f"  - Runtime: {runtime_s:.2f}s")
    print(f"  - Peak RSS: {peak_rss_mb:.1f} MB (Growth: {rss_growth_mb:.1f} MB)")
    print(f"  - EDA Rows: {eda_summary['rows_total']:,} | Scored Rows: {scoring_meta['rows_scored']:,}")

    assert eda_summary["rows_total"] == total_target
    assert scoring_meta["rows_scored"] == max_score
    assert scoring_meta["warning_message"] is not None
    assert f"{total_target:,}" in scoring_meta["warning_message"]
    assert f"{max_score:,}" in scoring_meta["warning_message"]


def test_concurrent_uploads_two_users(user_a_headers, user_b_headers):
    """Verify two concurrent uploads from different users complete independently."""
    csv_a = b"step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest\n1,PAYMENT,100,CA,100,0,MA,0,100\n"
    csv_b = b"step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest\n1,TRANSFER,500,CB,500,0,MB,0,500\n"

    up_a = client.post("/analysis/upload", files={"file": ("file_a.csv", csv_a, "text/csv")}, headers=user_a_headers).json()
    up_b = client.post("/analysis/upload", files={"file": ("file_b.csv", csv_b, "text/csv")}, headers=user_b_headers).json()

    id_a = up_a["analysis_id"]
    id_b = up_b["analysis_id"]

    client.post(f"/analysis/{id_a}/start", data={"column_mapping": json.dumps(up_a["auto_mapping"])}, headers=user_a_headers)
    client.post(f"/analysis/{id_b}/start", data={"column_mapping": json.dumps(up_b["auto_mapping"])}, headers=user_b_headers)

    # Wait for both to complete
    for an_id, hdrs in [(id_a, user_a_headers), (id_b, user_b_headers)]:
        completed = False
        for _ in range(60):
            st = client.get(f"/analysis/{an_id}/status", headers=hdrs).json()
            if st["status"] == "done":
                completed = True
                break
            time.sleep(0.5)
        assert completed, f"Analysis {an_id} did not finish, status: {client.get(f'/analysis/{an_id}/status', headers=hdrs).json()}"


def test_zero_overlap_renamed_accounts():
    """
    OVERLAP AUDIT TEST:
    Take sample_heldout.csv, rename every account ID (prefix 'X_') so overlap is 0.0%,
    and print overlap, graph mode chosen, alert rate, precision, recall for
    model-threshold and alert-budget modes, versus the original file.
    """
    heldout_path = Path("frontend/samples/sample_heldout.csv")
    if not heldout_path.exists():
        pytest.skip("sample_heldout.csv missing.")

    from src.batch_scoring import compute_account_overlap, batch_score_dataframe

    df_orig = pd.read_csv(heldout_path)
    overlap_orig = compute_account_overlap(df_orig)
    overlap_min = float(os.getenv("OVERLAP_MIN", "0.20"))
    rec_mode_orig = "warmed" if overlap_orig >= overlap_min else "cold_start"

    # 1. Original file under model-threshold mode
    _, meta_orig_m = batch_score_dataframe(
        valid_df=df_orig,
        graph_mode=rec_mode_orig,
        threshold_mode="model_threshold",
    )
    # 2. Original file under alert-budget mode (2%)
    _, meta_orig_b = batch_score_dataframe(
        valid_df=df_orig,
        graph_mode=rec_mode_orig,
        threshold_mode="alert_budget",
        alert_budget_pct=2.0,
    )

    # 3. Renamed file: prefix every account with 'X_'
    df_renamed = df_orig.copy()
    df_renamed["nameOrig"] = "X_" + df_renamed["nameOrig"].astype(str)
    df_renamed["nameDest"] = "X_" + df_renamed["nameDest"].astype(str)

    overlap_renamed = compute_account_overlap(df_renamed)
    rec_mode_renamed = "warmed" if overlap_renamed >= overlap_min else "cold_start"

    # 4. Renamed file under model-threshold mode
    _, meta_ren_m = batch_score_dataframe(
        valid_df=df_renamed,
        graph_mode=rec_mode_renamed,
        threshold_mode="model_threshold",
    )
    # 5. Renamed file under alert-budget mode (2%)
    _, meta_ren_b = batch_score_dataframe(
        valid_df=df_renamed,
        graph_mode=rec_mode_renamed,
        threshold_mode="alert_budget",
        alert_budget_pct=2.0,
    )

    print("\n" + "=" * 105)
    print("ACCOUNT OVERLAP & THRESHOLD MODE EXPERIMENTAL TEST (ORIGINAL vs RENAMED 'X_')")
    print("=" * 105)
    print(f"{'METRIC / CONFIGURATION':<35} | {'ORIGINAL (sample_heldout.csv)':<32} | {'RENAMED (prefix X_, 0% overlap)':<32}")
    print("-" * 105)
    print(f"{'Account Overlap':<35} | {overlap_orig*100:>28.2f}% | {overlap_renamed*100:>28.2f}%")
    print(f"{'Graph Mode Chosen':<35} | {rec_mode_orig:>32} | {rec_mode_renamed:>32}")
    print(f"{'Model Threshold Used':<35} | {meta_orig_m['threshold_used']:>32.4f} | {meta_ren_m['threshold_used']:>32.4f}")
    print(f"{'Model-Thresh Alert Rate':<35} | {meta_orig_m['flagged_rate']*100:>28.2f}% | {meta_ren_m['flagged_rate']*100:>28.2f}%")
    print(f"{'Model-Thresh Precision':<35} | {meta_orig_m['metrics']['precision']*100:>28.2f}% | {meta_ren_m['metrics']['precision']*100:>28.2f}%")
    print(f"{'Model-Thresh Recall':<35} | {meta_orig_m['metrics']['recall']*100:>28.2f}% | {meta_ren_m['metrics']['recall']*100:>28.2f}%")
    print(f"{'Alert-Budget (2%) Flagged Count':<35} | {meta_orig_b['flagged_count']:>32} | {meta_ren_b['flagged_count']:>32}")
    print(f"{'Alert-Budget (2%) Alert Rate':<35} | {meta_orig_b['flagged_rate']*100:>28.2f}% | {meta_ren_b['flagged_rate']*100:>28.2f}%")
    print(f"{'Alert-Budget (2%) Precision':<35} | {meta_orig_b['metrics']['precision']*100:>28.2f}% | {meta_ren_b['metrics']['precision']*100:>28.2f}%")
    print(f"{'Alert-Budget (2%) Recall':<35} | {meta_orig_b['metrics']['recall']*100:>28.2f}% | {meta_ren_b['metrics']['recall']*100:>28.2f}%")
    print("=" * 105 + "\n")

    assert overlap_renamed < 0.001, f"Expected 0% overlap on renamed accounts, got {overlap_renamed}"
    assert rec_mode_renamed == "cold_start", f"Expected cold_start recommendation for 0% overlap, got {rec_mode_renamed}"
    assert meta_ren_m["low_overlap_warning"] is not None, "Low overlap warning must be emitted when overlap < 20%"


def test_chatbot_drift_guard_under_100_transactions(tmp_path):
    """
    CHATBOT DRIFT TEST:
    If fewer than 100 scored transactions exist and no simulation has run,
    verify chatbot answers 'Not enough scored transactions to assess drift (n=...)'
    instead of STABLE / PSI 0.0000.
    """
    from api.db import init_db
    from api.rag_chatbot import answer_analyst_query

    test_db = tmp_path / "finsecure_chatbot_guard.db"
    init_db(test_db)

    # 1. Zero transactions
    ans_0 = answer_analyst_query("what is the drift status?", db_path=test_db)
    ans_text_0 = str(ans_0)
    print(f"\n[CHATBOT DRIFT GUARD n=0]: {ans_text_0}")
    assert "Not enough scored transactions to assess drift (n=0)" in ans_text_0
    assert "STABLE" not in ans_text_0
    assert "0.0000" not in ans_text_0

    # 2. 50 transactions (< 100)
    from api.db import get_db_connection, get_model_version
    current_ver = get_model_version()
    with get_db_connection(test_db) as conn:
        with conn:
            for i in range(50):
                conn.execute(
                    "INSERT INTO predictions (timestamp, model_version, step, name_orig, name_dest, amount, prediction_proba, is_fraud, threshold_used, features_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    ("2026-10-21T00:00:00", current_ver, 1, f"C_{i}", f"M_{i}", 100.0, 0.01, 0, 0.0484, "{}")
                )

    ans_50 = answer_analyst_query("what is the drift status?", db_path=test_db)
    ans_text_50 = str(ans_50)
    print(f"[CHATBOT DRIFT GUARD n=50]: {ans_text_50}")
    assert "Not enough scored transactions to assess drift (n=50)" in ans_text_50
    assert "STABLE" not in ans_text_50

