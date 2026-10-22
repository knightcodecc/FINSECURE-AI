"""
analysis.py — FastAPI Router & Background Execution Engine for CSV Analysis
============================================================================
Handles asynchronous CSV upload, streaming validation, background batch inference,
interactive what-if analytics, on-demand SHAP explanations, and sanitized exports.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gc
import io
import json
import math
import os
from pathlib import Path
import shutil
import threading
import time
from typing import Dict, List, Any, Optional
import uuid

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query, status
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
import joblib
import numpy as np
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
import pandas as pd

from api.auth import get_current_user
from api.db import get_data_dir
from api.analysis_db import (
    create_analysis,
    update_analysis_status,
    update_analysis_results,
    get_analysis,
    list_analyses,
    rename_analysis,
    delete_analysis,
    count_active_jobs,
)
from src.batch_scoring import (
    detect_delimiter_and_encoding,
    auto_map_columns,
    validate_and_clean_stream,
    batch_score_dataframe,
    compute_evaluation_metrics,
    sanitize_csv_formula_injection,
    export_sanitized_results_csv,
    compute_account_overlap,
    REQUIRED_CANONICAL_COLUMNS,
    OPTIONAL_CANONICAL_COLUMNS,
    DEFAULT_MAX_SCORE_ROWS,
)

router = APIRouter(prefix="/analysis", tags=["CSV Analysis"])

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "500"))
UPLOAD_TTL_DAYS = int(os.getenv("UPLOAD_TTL_DAYS", "7"))
MAX_CONCURRENT_JOBS_PER_USER = int(os.getenv("MAX_CONCURRENT_JOBS_PER_USER", "1"))
MAX_CONCURRENT_JOBS_TOTAL = int(os.getenv("MAX_CONCURRENT_JOBS_TOTAL", "2"))

# Worker execution pool & cancellation tokens
_ANALYSIS_EXECUTOR = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_JOBS_TOTAL)
_CANCEL_TOKENS: Dict[str, threading.Event] = {}
_ACTIVE_JOB_TIMES: Dict[str, float] = {}


def get_uploads_dir(user_id: int) -> Path:
    """Return isolated uploads directory for user under DATA_DIR."""
    p = get_data_dir() / "uploads" / str(user_id)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_analysis_results_dir(analysis_id: str) -> Path:
    """Return storage directory for an analysis under DATA_DIR."""
    p = get_data_dir() / "analyses" / analysis_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def cleanup_expired_uploads():
    """Remove uploaded raw CSV files older than UPLOAD_TTL_DAYS."""
    uploads_root = get_data_dir() / "uploads"
    if not uploads_root.exists():
        return
    now = time.time()
    ttl_seconds = UPLOAD_TTL_DAYS * 86400
    for user_folder in uploads_root.iterdir():
        if user_folder.is_dir():
            for f in user_folder.glob("*.csv"):
                try:
                    if now - f.stat().st_mtime > ttl_seconds:
                        f.unlink(missing_ok=True)
                except Exception:
                    pass


# Run upload TTL cleanup on startup
try:
    cleanup_expired_uploads()
except Exception:
    pass


@router.post("/upload")
async def upload_transaction_csv(
    file: UploadFile = File(...),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Stream upload raw transaction CSV chunk by chunk directly to disk.
    Enforces extension (.csv only), UTF-8 BOM detection, delimiter detection,
    and returns detected columns plus preview rows for column mapping.
    """
    # 1. Validate file extension (415 if wrong type)
    filename = file.filename or "upload.csv"
    if not filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported media type. Only .csv files are supported.",
        )

    user_id = current_user["id"]
    analysis_id = str(uuid.uuid4())
    upload_dir = get_uploads_dir(user_id)
    target_csv_path = (upload_dir / f"{analysis_id}.csv").resolve()

    # Prevent path traversal
    if not str(target_csv_path).startswith(str(upload_dir.resolve())):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid file destination path.")

    max_mb = float(os.getenv("MAX_UPLOAD_MB", "500"))
    max_bytes = int(max_mb * 1024 * 1024)
    total_bytes_written = 0
    first_chunk_sample = b""

    # 2. Stream chunk by chunk to disk (bounded memory)
    with open(target_csv_path, "wb") as dest:
        while True:
            chunk = await file.read(1024 * 1024)  # 1MB chunks
            if not chunk:
                break
            if not first_chunk_sample:
                first_chunk_sample = chunk[:16384]
            total_bytes_written += len(chunk)
            if total_bytes_written > max_bytes:
                dest.close()
                target_csv_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Uploaded file exceeds maximum limit of {MAX_UPLOAD_MB} MB.",
                )
            dest.write(chunk)

    if total_bytes_written == 0:
        target_csv_path.unlink(missing_ok=True)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Uploaded file is empty (0 bytes).")

    # 3. Detect delimiter and encoding
    delimiter, encoding = detect_delimiter_and_encoding(first_chunk_sample)

    # 4. Read header and preview of first 10 rows
    try:
        preview_df = pd.read_csv(
            target_csv_path,
            nrows=10,
            delimiter=delimiter,
            encoding=encoding,
            dtype=str,
        )
    except Exception as e:
        target_csv_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to parse CSV file: {str(e)}",
        )

    raw_columns = list(preview_df.columns)
    if not raw_columns or len(preview_df) == 0:
        target_csv_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="CSV contains no records or invalid header structure.",
        )

    auto_mapping, missing_required = auto_map_columns(raw_columns)

    # Compute account overlap and step span at upload time
    account_overlap = 0.0
    account_overlap_pct = 0.0
    step_span = 0
    span_under_72 = False
    overlap_min = float(os.getenv("OVERLAP_MIN", "0.20"))
    recommended_graph_mode = "cold_start"

    orig_col = auto_mapping.get("nameOrig")
    dest_col = auto_mapping.get("nameDest")
    step_col = auto_mapping.get("step")
    usecols = [c for c in [orig_col, dest_col, step_col] if c is not None and c in raw_columns]

    if orig_col and dest_col and orig_col in raw_columns and dest_col in raw_columns:
        try:
            df_cols = pd.read_csv(
                target_csv_path,
                usecols=usecols,
                delimiter=delimiter,
                encoding=encoding,
            )
            renamed = df_cols.rename(columns={orig_col: "nameOrig", dest_col: "nameDest"})
            account_overlap = compute_account_overlap(renamed)
            account_overlap_pct = round(account_overlap * 100.0, 2)
            if step_col and step_col in df_cols.columns:
                steps = pd.to_numeric(df_cols[step_col], errors="coerce").dropna()
                if not steps.empty:
                    s_min = int(steps.min())
                    s_max = int(steps.max())
                    step_span = max(0, s_max - s_min + 1)
                    span_under_72 = bool(step_span < 72)
            recommended_graph_mode = "warmed" if account_overlap >= overlap_min else "cold_start"
        except Exception:
            pass

    # Create initial queued record in SQLite
    create_analysis(
        analysis_id=analysis_id,
        user_id=user_id,
        user_email=current_user["email"],
        filename=filename,
        params_json=json.dumps({
            "delimiter": delimiter,
            "encoding": encoding,
            "raw_columns": raw_columns,
            "auto_mapping": auto_mapping,
            "file_size_mb": round(total_bytes_written / (1024 * 1024), 2),
            "account_overlap": account_overlap,
            "account_overlap_pct": account_overlap_pct,
            "recommended_graph_mode": recommended_graph_mode,
            "step_span": step_span,
            "span_under_72": span_under_72,
        }),
    )

    # Format preview rows safely (limit to 10 rows)
    preview_rows = preview_df.fillna("").head(10).to_dict(orient="records")

    return {
        "analysis_id": analysis_id,
        "filename": filename,
        "file_size_mb": round(total_bytes_written / (1024 * 1024), 2),
        "delimiter": delimiter,
        "encoding": encoding,
        "raw_columns": raw_columns,
        "headers": raw_columns,
        "required_columns": REQUIRED_CANONICAL_COLUMNS,
        "optional_columns": OPTIONAL_CANONICAL_COLUMNS,
        "auto_mapping": auto_mapping,
        "missing_required": missing_required,
        "preview_rows": preview_rows,
        "row_preview": preview_rows,
        "account_overlap": account_overlap,
        "account_overlap_pct": account_overlap_pct,
        "recommended_graph_mode": recommended_graph_mode,
        "step_span": step_span,
        "span_under_72": span_under_72,
        "overlap_min_pct": round(overlap_min * 100.0, 1),
    }


def _run_analysis_worker(
    analysis_id: str,
    user_id: int,
    csv_path: Path,
    column_mapping: Dict[str, str],
    use_labels_for_graph: bool,
    graph_mode: str,
    active_threshold: Optional[float],
    threshold_mode: str,
    alert_budget_pct: Optional[float],
    max_score_rows: int,
    cancel_event: threading.Event,
):
    """Background worker executing streaming validation, batch scoring, and artifact serialization."""
    start_time = time.time()
    results_dir = get_analysis_results_dir(analysis_id)
    rejected_path = results_dir / "rejected_rows.csv"
    summary_path = results_dir / "summary.json"
    results_parquet_path = results_dir / "results.parquet"

    def cancel_check() -> bool:
        return cancel_event.is_set()

    def progress_callback(pct: float, stage: str):
        if cancel_event.is_set():
            raise InterruptedError("Analysis cancelled by user.")
        elapsed = time.time() - start_time
        eta = 0.0
        if pct > 5.0:
            total_est = (elapsed / pct) * 100.0
            eta = max(0.0, total_est - elapsed)
        update_analysis_status(
            analysis_id=analysis_id,
            status="running",
            stage=stage,
            progress_pct=round(pct, 1),
            eta_seconds=round(eta, 1),
        )

    try:
        update_analysis_status(analysis_id, status="running", stage="reading", progress_pct=5.0)

        # 1. Delimiter & encoding
        sample_bytes = b""
        with open(csv_path, "rb") as f:
            sample_bytes = f.read(16384)
        delimiter, encoding = detect_delimiter_and_encoding(sample_bytes)

        # 2. Streaming Validation & EDA
        progress_callback(10.0, "validating")
        eda_summary, valid_df = validate_and_clean_stream(
            csv_path=csv_path,
            column_mapping=column_mapping,
            rejected_output_path=rejected_path,
            delimiter=delimiter,
            encoding=encoding,
            cancel_check=cancel_check,
            progress_callback=progress_callback,
        )

        if cancel_check():
            update_analysis_status(analysis_id, status="cancelled", stage="finishing")
            return

        # 3. Batch Scoring
        score_df, scoring_metadata = batch_score_dataframe(
            valid_df=valid_df,
            active_threshold=active_threshold,
            threshold_mode=threshold_mode,
            alert_budget_pct=alert_budget_pct,
            use_labels_for_graph=use_labels_for_graph,
            graph_mode=graph_mode,
            max_score_rows=max_score_rows,
            cancel_check=cancel_check,
            progress_callback=progress_callback,
        )

        if cancel_check():
            update_analysis_status(analysis_id, status="cancelled", stage="finishing")
            return

        # 4. Serialize artifacts to DATA_DIR/analyses/<id>/
        score_df.to_parquet(results_parquet_path, index=False)

        # Consolidate overall summary
        full_summary = {
            "analysis_id": analysis_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "eda": eda_summary,
            "scoring": scoring_metadata,
            "execution_time_seconds": round(time.time() - start_time, 2),
            "rejected_file_exists": rejected_path.exists() and eda_summary["rows_excluded"] > 0,
        }

        with open(summary_path, "w", encoding="utf-8") as sf:
            json.dump(full_summary, sf, indent=2)

        # 5. Update SQLite record
        update_analysis_results(
            analysis_id=analysis_id,
            rows_total=eda_summary["rows_total"],
            rows_valid=eda_summary["rows_valid"],
            rows_scored=scoring_metadata["rows_scored"],
            model_version=scoring_metadata["model_version"],
            threshold=scoring_metadata["threshold_used"],
            has_labels=scoring_metadata["has_labels"],
            params_json=json.dumps({
                "column_mapping": column_mapping,
                "use_labels_for_graph": use_labels_for_graph,
                "graph_mode": graph_mode,
                "threshold_mode": threshold_mode,
                "alert_budget_pct": alert_budget_pct,
                "max_score_rows": max_score_rows,
                "threshold": scoring_metadata["threshold_used"],
            }),
        )

    except InterruptedError:
        update_analysis_status(analysis_id, status="cancelled", stage="finishing")
    except Exception as e:
        update_analysis_status(analysis_id, status="failed", error=str(e))
    finally:
        _CANCEL_TOKENS.pop(analysis_id, None)
        _ACTIVE_JOB_TIMES.pop(analysis_id, None)
        gc.collect()


@router.post("/{analysis_id}/start")
async def start_analysis(
    analysis_id: str,
    column_mapping: str = Form(...),  # JSON string
    use_labels_for_graph: bool = Form(False),
    graph_mode: str = Form("warmed"),
    threshold: Optional[float] = Form(None),
    threshold_mode: str = Form("model_threshold"),
    alert_budget_pct: Optional[float] = Form(2.0),
    max_score_rows: int = Form(DEFAULT_MAX_SCORE_ROWS),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Launch background analysis job with concurrency limit enforcement and column validation.
    """
    user_id = current_user["id"]
    analysis = get_analysis(analysis_id)
    if not analysis or (analysis["user_id"] != user_id and current_user.get("role") != "admin"):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis record not found.")

    # Concurrency limit enforcement
    user_active = count_active_jobs(user_id=user_id, exclude_analysis_id=analysis_id)
    if user_active >= MAX_CONCURRENT_JOBS_PER_USER:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Concurrent analysis limit exceeded for user (max {MAX_CONCURRENT_JOBS_PER_USER}). Please wait for your running analysis to complete.",
        )

    total_active = count_active_jobs(exclude_analysis_id=analysis_id)
    if total_active >= MAX_CONCURRENT_JOBS_TOTAL:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"System-wide analysis capacity reached (max {MAX_CONCURRENT_JOBS_TOTAL} concurrent jobs). Please retry shortly.",
        )

    try:
        mapping_dict = json.loads(column_mapping)
    except Exception:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid JSON column mapping provided.")

    # Validate all 9 required columns are mapped
    missing_required = [c for c in REQUIRED_CANONICAL_COLUMNS if not mapping_dict.get(c)]
    if missing_required:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Missing required column mappings for: {', '.join(missing_required)}",
        )

    csv_path = get_uploads_dir(user_id) / f"{analysis_id}.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Uploaded CSV file not found on disk.")

    cancel_event = threading.Event()
    _CANCEL_TOKENS[analysis_id] = cancel_event
    _ACTIVE_JOB_TIMES[analysis_id] = time.time()

    update_analysis_status(analysis_id, status="queued", stage="reading", progress_pct=0.0)

    _ANALYSIS_EXECUTOR.submit(
        _run_analysis_worker,
        analysis_id=analysis_id,
        user_id=user_id,
        csv_path=csv_path,
        column_mapping=mapping_dict,
        use_labels_for_graph=use_labels_for_graph,
        graph_mode=graph_mode,
        active_threshold=threshold,
        threshold_mode=threshold_mode,
        alert_budget_pct=alert_budget_pct,
        max_score_rows=max_score_rows,
        cancel_event=cancel_event,
    )

    return {
        "status": "queued",
        "analysis_id": analysis_id,
        "message": "Analysis successfully queued for processing.",
    }


@router.get("/{analysis_id}/status")
async def get_analysis_status(
    analysis_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Retrieve progress, stage, ETA, and status for polling."""
    analysis = get_analysis(analysis_id)
    if not analysis or (analysis["user_id"] != current_user["id"] and current_user.get("role") != "admin"):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis record not found.")

    return {
        "id": analysis["id"],
        "status": analysis["status"],
        "stage": analysis["stage"],
        "progress_pct": analysis["progress_pct"],
        "eta_seconds": analysis["eta_seconds"],
        "error": analysis["error"],
        "display_name": analysis["display_name"],
        "rows_total": analysis["rows_total"],
        "rows_scored": analysis["rows_scored"],
        "created_at": analysis["created_at"],
    }


@router.post("/{analysis_id}/cancel")
async def cancel_analysis(
    analysis_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Cancel a running analysis and free memory."""
    analysis = get_analysis(analysis_id)
    if not analysis or (analysis["user_id"] != current_user["id"] and current_user.get("role") != "admin"):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis record not found.")

    token = _CANCEL_TOKENS.get(analysis_id)
    if token:
        token.set()

    update_analysis_status(analysis_id, status="cancelled", stage="finishing")
    gc.collect()

    return {"status": "cancelled", "message": "Analysis cancelled successfully."}


@router.get("/list")
async def list_user_analyses(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """List analyses: regular users see their own; admins see all."""
    is_admin = current_user.get("role") == "admin"
    analyses = list_analyses(user_id=current_user["id"], is_admin=is_admin)
    return {"analyses": analyses}


@router.patch("/{analysis_id}/rename")
async def rename_user_analysis(
    analysis_id: str,
    new_name: str = Form(...),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Rename an analysis record."""
    is_admin = current_user.get("role") == "admin"
    success = rename_analysis(
        analysis_id=analysis_id,
        new_display_name=new_name.strip(),
        user_id=current_user["id"],
        is_admin=is_admin,
    )
    if not success:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis record not found or access denied.")
    return {"status": "success", "message": "Analysis renamed successfully."}


@router.delete("/{analysis_id}")
async def delete_user_analysis(
    analysis_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Delete an analysis and remove all associated files and DB rows."""
    analysis = get_analysis(analysis_id)
    if not analysis or (analysis["user_id"] != current_user["id"] and current_user.get("role") != "admin"):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis record not found.")

    is_admin = current_user.get("role") == "admin"
    delete_analysis(analysis_id=analysis_id, user_id=current_user["id"], is_admin=is_admin)

    # Delete results folder
    res_dir = get_analysis_results_dir(analysis_id)
    if res_dir.exists():
        shutil.rmtree(res_dir, ignore_errors=True)

    # Delete upload CSV
    upload_file = get_uploads_dir(analysis["user_id"]) / f"{analysis_id}.csv"
    upload_file.unlink(missing_ok=True)

    return {"status": "deleted", "message": "Analysis deleted successfully."}


@router.get("/{analysis_id}/summary")
async def get_analysis_summary(
    analysis_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Return complete analysis summary JSON (KPIs, charts data, metrics)."""
    analysis = get_analysis(analysis_id)
    if not analysis or (analysis["user_id"] != current_user["id"] and current_user.get("role") != "admin"):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis record not found.")

    summary_file = get_analysis_results_dir(analysis_id) / "summary.json"
    if not summary_file.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis summary not yet available.")

    with open(summary_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["analysis_info"] = analysis
    return data


@router.get("/{analysis_id}/rows")
async def get_analysis_rows(
    analysis_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=200),
    sort_by: str = Query("fraud_probability"),
    order: str = Query("desc"),
    search: Optional[str] = Query(None),
    filter_type: Optional[str] = Query(None),
    decision: Optional[str] = Query(None),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Paginated, sortable, and searchable flagged transactions table."""
    analysis = get_analysis(analysis_id)
    if not analysis or (analysis["user_id"] != current_user["id"] and current_user.get("role") != "admin"):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis record not found.")

    parquet_file = get_analysis_results_dir(analysis_id) / "results.parquet"
    if not parquet_file.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis results not yet available.")

    df = pd.read_parquet(parquet_file)

    # Attach integer row_id for reference
    if "row_id" not in df.columns:
        df["row_id"] = np.arange(len(df)) + 1

    # Apply filters
    if decision:
        df = df[df["decision"].str.upper() == decision.upper()]
    if filter_type:
        df = df[df["type"].str.upper() == filter_type.upper()]
    if search:
        s = search.strip().lower()
        orig_match = df["nameOrig"].str.lower().str.contains(s, na=False)
        dest_match = df["nameDest"].str.lower().str.contains(s, na=False)
        df = df[orig_match | dest_match]

    total_matching = len(df)

    # Sort
    ascending = order.lower() == "asc"
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=ascending)

    start_idx = (page - 1) * page_size
    page_df = df.iloc[start_idx : start_idx + page_size]

    records = page_df.to_dict(orient="records")

    return {
        "page": page,
        "page_size": page_size,
        "total_rows": total_matching,
        "total_pages": math.ceil(total_matching / page_size) if total_matching > 0 else 1,
        "rows": records,
    }


@router.get("/{analysis_id}/whatif")
async def whatif_threshold_simulation(
    analysis_id: str,
    threshold: Optional[float] = Query(None, ge=0.0001, le=0.9999),
    threshold_mode: str = Query("model_threshold"),
    alert_budget_pct: Optional[float] = Query(2.0, ge=0.01, le=100.0),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Simulate what-if threshold adjustment dynamically from cached predictions.
    Supports both model threshold and alert-budget (top p%) modes.
    Recomputes flagged count, rate, and if labeled: confusion matrix, precision, recall, F1, F2, and rupee costs.
    """
    analysis = get_analysis(analysis_id)
    if not analysis or (analysis["user_id"] != current_user["id"] and current_user.get("role") != "admin"):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis record not found.")

    parquet_file = get_analysis_results_dir(analysis_id) / "results.parquet"
    if not parquet_file.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Results parquet not found.")

    df = pd.read_parquet(parquet_file)
    probs = df["fraud_probability"].values
    amounts = df["amount"].values
    total_scored = len(df)

    if threshold_mode == "alert_budget":
        pct = float(alert_budget_pct if alert_budget_pct is not None else 2.0)
        pct = max(0.01, min(100.0, pct))
        budget_k = max(1, int(round(total_scored * (pct / 100.0))))
        kth_val = float(np.sort(probs)[-budget_k])
        effective_threshold = kth_val
        flagged_mask = probs >= effective_threshold
        if np.sum(flagged_mask) > budget_k:
            top_k_indices = np.argsort(probs)[-budget_k:]
            flagged_mask = np.zeros(len(probs), dtype=bool)
            flagged_mask[top_k_indices] = True
        active_threshold_used = effective_threshold
    else:
        threshold_mode = "model_threshold"
        active_threshold_used = float(threshold if threshold is not None else 0.0484)
        flagged_mask = probs >= active_threshold_used

    flagged_count = int(np.sum(flagged_mask))
    flagged_rate = float(flagged_count / max(total_scored, 1))
    total_amount_flagged = float(np.sum(amounts[flagged_mask]))
    amount_cleared = float(np.sum(amounts[~flagged_mask]))

    has_labels = "isFraud" in df.columns
    metrics = None
    if has_labels:
        y_true = df["isFraud"].astype(int).values
        y_pred = flagged_mask.astype(int)
        metrics = compute_evaluation_metrics(
            y_true=y_true,
            y_pred=y_pred,
            probs=probs,
            amounts=amounts,
            active_threshold=active_threshold_used,
        )

    return {
        "threshold": active_threshold_used,
        "threshold_mode": threshold_mode,
        "alert_budget_pct": alert_budget_pct,
        "flagged_count": flagged_count,
        "flagged_rate": flagged_rate,
        "total_amount_flagged": total_amount_flagged,
        "amount_cleared": amount_cleared,
        "has_labels": has_labels,
        "metrics": metrics,
    }


@router.get("/{analysis_id}/shap/{row_idx}")
async def get_row_shap_explanation(
    analysis_id: str,
    row_idx: int,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Compute on-demand SHAP explanation (top-5 features & explanation) for a specific row only.
    """
    analysis = get_analysis(analysis_id)
    if not analysis or (analysis["user_id"] != current_user["id"] and current_user.get("role") != "admin"):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis record not found.")

    parquet_file = get_analysis_results_dir(analysis_id) / "results.parquet"
    if not parquet_file.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Results parquet not found.")

    df = pd.read_parquet(parquet_file)
    if row_idx < 0 or row_idx >= len(df):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Row index {row_idx} out of bounds.")

    row = df.iloc[row_idx].to_dict()

    # Load predictor to transform and explain single row
    from api.predictor import get_predictor
    from api.xai_engine import shap_explain, generate_explanation

    predictor = get_predictor(warm=False)
    df_scaled = predictor.prepare_features(row)
    proba = float(row.get("fraud_probability", predictor.model.predict_proba(df_scaled)[0, 1]))
    threshold = float(row.get("threshold_used", predictor.active_threshold))

    top_features, base_val, _ = shap_explain(df_scaled, top_k=5, raw_tx_dict=row)
    explanation = generate_explanation(top_features, proba, threshold)

    return {
        "row_idx": row_idx,
        "fraud_probability": proba,
        "decision": row.get("decision", "FLAGGED" if proba >= threshold else "CLEARED"),
        "threshold_used": threshold,
        "shap_top_features": top_features,
        "explanation": explanation,
        "base_value": base_val,
        "raw_transaction": {
            "step": row.get("step"),
            "type": row.get("type"),
            "amount": row.get("amount"),
            "nameOrig": row.get("nameOrig"),
            "nameDest": row.get("nameDest"),
            "oldbalanceOrg": row.get("oldbalanceOrg"),
            "newbalanceOrig": row.get("newbalanceOrig"),
            "oldbalanceDest": row.get("oldbalanceDest"),
            "newbalanceDest": row.get("newbalanceDest"),
        },
    }


@router.get("/{analysis_id}/export/csv")
async def export_analysis_csv(
    analysis_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Export results CSV with formula-injection sanitization.
    """
    analysis = get_analysis(analysis_id)
    if not analysis or (analysis["user_id"] != current_user["id"] and current_user.get("role") != "admin"):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis record not found.")

    parquet_file = get_analysis_results_dir(analysis_id) / "results.parquet"
    if not parquet_file.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Results parquet not found.")

    df = pd.read_parquet(parquet_file)
    csv_export_path = get_analysis_results_dir(analysis_id) / "results_sanitized.csv"
    export_sanitized_results_csv(df, csv_export_path)

    return FileResponse(
        path=csv_export_path,
        media_type="text/csv",
        filename=f"finsecure_analysis_{analysis_id[:8]}_results.csv",
    )


@router.get("/{analysis_id}/export/xlsx")
async def export_analysis_xlsx(
    analysis_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Generate professional multi-tab Excel workbook:
    1. Overview (KPIs, parameters, date, model version)
    2. Metrics (if labeled: confusion matrix, precision, recall, F1, F2, rupee costs, alert budgets)
    3. Flagged Transactions (top 1000 flagged rows)
    """
    analysis = get_analysis(analysis_id)
    if not analysis or (analysis["user_id"] != current_user["id"] and current_user.get("role") != "admin"):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis record not found.")

    res_dir = get_analysis_results_dir(analysis_id)
    summary_file = res_dir / "summary.json"
    parquet_file = res_dir / "results.parquet"

    if not summary_file.exists() or not parquet_file.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis results not yet available.")

    with open(summary_file, "r", encoding="utf-8") as f:
        summary_data = json.load(f)

    df = pd.read_parquet(parquet_file)
    eda = summary_data.get("eda", {})
    scoring = summary_data.get("scoring", {})
    metrics = scoring.get("metrics")

    wb = openpyxl.Workbook()
    # Sheet 1: Overview
    ws_over = wb.active
    ws_over.title = "Overview KPIs"

    # Styling
    header_font = Font(name="Calibri", size=14, bold=True, color="1E3A8A")
    sec_font = Font(name="Calibri", size=11, bold=True, color="1F2937")
    hdr_fill = PatternFill(start_color="F3F4F6", end_color="F3F4F6", fill_type="solid")
    border_thin = Border(
        left=Side(style="thin", color="E5E7EB"),
        right=Side(style="thin", color="E5E7EB"),
        top=Side(style="thin", color="E5E7EB"),
        bottom=Side(style="thin", color="E5E7EB"),
    )

    ws_over["A1"] = "FINSECURE AI v2 — CSV ANALYSIS REPORT"
    ws_over["A1"].font = header_font
    ws_over["A2"] = f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} | Filename: {analysis['filename']}"

    overview_rows = [
        ["Analysis ID", analysis["id"]],
        ["Display Name", analysis["display_name"]],
        ["User Email", analysis["user_email"]],
        ["Model Version", scoring.get("model_version", "v2_xgboost")],
        ["Dataset Name", scoring.get("dataset_name", "PaySim")],
        ["Threshold Used", scoring.get("threshold_used", 0.017033)],
        ["Total Rows in File", eda.get("rows_total", 0)],
        ["Valid Rows", eda.get("rows_valid", 0)],
        ["Excluded Rows", eda.get("rows_excluded", 0)],
        ["Rows Scored", scoring.get("rows_scored", 0)],
        ["Flagged Count", scoring.get("flagged_count", 0)],
        ["Flagged Rate", f"{scoring.get('flagged_rate', 0.0) * 100:.2f}%"],
        ["Total Amount Flagged (INR)", round(scoring.get("total_amount_flagged", 0.0), 2)],
        ["Amount Cleared (INR)", round(scoring.get("amount_cleared", 0.0), 2)],
        ["Step Range", f"{eda.get('step_min', 0)} to {eda.get('step_max', 0)}"],
    ]

    ws_over.append([])
    for row in overview_rows:
        ws_over.append(row)

    # Sheet 2: Evaluation Metrics (if ground truth exists)
    if metrics:
        ws_met = wb.create_sheet(title="Model Metrics")
        ws_met["A1"] = "EVALUATION METRICS & FINANCIAL RECONCILIATION"
        ws_met["A1"].font = header_font

        cm = metrics.get("confusion_matrix", {})
        fin = metrics.get("financial_cost", {})

        metric_rows = [
            ["Metric", "Value", "Formula / Description"],
            ["True Positives (TP)", cm.get("tp", 0), "Frauds correctly caught"],
            ["False Positives (FP)", cm.get("fp", 0), "Legitimate transactions flagged"],
            ["False Negatives (FN)", cm.get("fn", 0), "Frauds missed by model"],
            ["True Negatives (TN)", cm.get("tn", 0), "Legitimate transactions cleared"],
            ["Total Evaluated (N)", cm.get("total", 0), "TP + FP + FN + TN == N"],
            ["Precision", f"{metrics.get('precision', 0.0):.4f}", "TP / (TP + FP)"],
            ["Recall", f"{metrics.get('recall', 0.0):.4f}", "TP / (TP + FN)"],
            ["F1 Score", f"{metrics.get('f1', 0.0):.4f}", "2 * P * R / (P + R)"],
            ["F2 Score (Recall-weighted)", f"{metrics.get('f2', 0.0):.4f}", "5 * P * R / (4 * P + R)"],
            ["PR-AUC", f"{metrics.get('pr_auc', 0.0):.4f}", "Precision-Recall Area Under Curve"],
            ["ROC-AUC", f"{metrics.get('roc_auc', 0.0):.4f}", "Receiver Operating Characteristic AUC"],
            [],
            ["Financial Rupee Metric", "Amount (INR)", "Reconciliation"],
            ["Caught Fraud Amount", round(fin.get("caught_fraud_amount", 0.0), 2), "Sum of fraud amounts caught"],
            ["Missed Fraud Amount", round(fin.get("missed_fraud_amount", 0.0), 2), "Sum of fraud amounts missed"],
            ["Total Fraud in Upload", round(fin.get("total_fraud_amount", 0.0), 2), "Caught + Missed == Total Fraud"],
            ["FP Review Cost (@ Rs 50)", round(fin.get("fp_review_cost", 0.0), 2), "FP * 50.0"],
            ["Total Financial Cost", round(fin.get("total_cost", 0.0), 2), "Missed Fraud + FP Review Cost"],
            ["Cost (Flag Nothing)", round(fin.get("flag_nothing_cost", 0.0), 2), "Total Fraud Amount"],
            ["Cost (Flag Everything)", round(fin.get("flag_everything_cost", 0.0), 2), "Non-Fraud Rows * Rs 50"],
            ["Financial Savings vs Nothing", round(fin.get("savings_vs_nothing", 0.0), 2), "Flag Nothing Cost - Total Cost"],
        ]

        ws_met.append([])
        for r in metric_rows:
            ws_met.append(r)

    # Sheet 3: Flagged Transactions
    ws_flag = wb.create_sheet(title="Flagged Transactions")
    flagged_df = df[df["decision"] == "FLAGGED"].sort_values("fraud_probability", ascending=False).head(1000)

    cols_to_export = [
        "step", "type", "amount", "nameOrig", "nameDest",
        "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
        "fraud_probability", "decision"
    ]
    if "isFraud" in flagged_df.columns:
        cols_to_export.append("isFraud")

    ws_flag.append(cols_to_export)
    for row in flagged_df[cols_to_export].itertuples(index=False):
        sanitized_cells = [sanitize_csv_formula_injection(c) for c in row]
        ws_flag.append(sanitized_cells)

    # Adjust column widths
    for sheet in wb.worksheets:
        for col in sheet.columns:
            max_len = max(len(str(cell.value or "")) for cell in col)
            col_letter = openpyxl.utils.get_column_letter(col[0].column)
            sheet.column_dimensions[col_letter].width = min(max(max_len + 3, 12), 45)

    xlsx_path = res_dir / "summary.xlsx"
    wb.save(xlsx_path)

    return FileResponse(
        path=xlsx_path,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=f"finsecure_analysis_{analysis_id[:8]}_summary.xlsx",
    )


@router.get("/{analysis_id}/export/rejected")
async def export_rejected_csv(
    analysis_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Download rejected rows CSV if invalid rows were excluded during validation."""
    analysis = get_analysis(analysis_id)
    if not analysis or (analysis["user_id"] != current_user["id"] and current_user.get("role") != "admin"):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis record not found.")

    rejected_path = get_analysis_results_dir(analysis_id) / "rejected_rows.csv"
    if not rejected_path.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No rejected rows file exists for this analysis.")

    return FileResponse(
        path=rejected_path,
        media_type="text/csv",
        filename=f"finsecure_analysis_{analysis_id[:8]}_rejected_rows.csv",
    )
