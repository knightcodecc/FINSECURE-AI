"""
main.py — FastAPI Application Entrypoint for FinSecure AI v2.
==============================================================
Provides REST API endpoints and WebSocket live transaction streams for:
- Real-time fraud detection (/predict, /predict/batch)
- Grounded SHAP XAI explainability
- Cost-sensitive metrics (/metrics/cost)
- Concept drift monitoring & simulation (/drift/status, /drift/simulate)
- Grounded RAG chatbot (/chat) with structured sources & conversation history
- PaySim dataset precomputed analytics (/analytics/*)
- User authentication & role management (/auth/*)
- WebSocket live streaming (/ws/scan, /ws/stats) with token authentication
"""

import os
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from api.chat_agent import ensure_project_env
    ensure_project_env()
except Exception:
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env", override=True)
    except Exception:
        pass

from unittest.mock import MagicMock
sys.modules["cv2"] = MagicMock()

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api.schemas import (
    TransactionInput,
    PredictionResponse,
    BatchTransactionInput,
    BatchPredictionResponse,
    DriftStatusResponse,
    ModelInfoResponse,
    ChatRequest,
    ChatResponse,
    RegisterRequest,
    LoginRequest,
    TokenResponse,
)
from api.predictor import get_predictor
from api.realtime_scanner import get_scanner
from api.rag_chatbot import answer_analyst_query
from api.chat_agent import ask_agent, get_last_tool_calls
from api.analytics import router as analytics_router
from api.analysis import router as analysis_router
from api.analysis_db import init_analyses_db
from api.auth import (
    init_users_db,
    get_users_db_connection,
    hash_password,
    verify_password,
    validate_email_format,
    validate_password_strength,
    check_rate_limit,
    record_failed_login,
    clear_failed_logins,
    create_access_token,
    decode_access_token,
    get_current_user,
    get_current_admin,
    export_users_to_excel,
    log_login_event,
    get_excel_export_path,
    get_users_db_path,
    EXCEL_EXPORT_PATH,
    USERS_DB_PATH,
)
from src.drift_monitor import DriftMonitor, simulate_drift, DB_PATH
from src.cost_sensitive_eval import expected_cost, threshold_sweep

import joblib
import pandas as pd


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize SQLite databases, predictors, scanners, and warm up SHAP explainer at application startup."""
    init_users_db()
    init_analyses_db()
    predictor = get_predictor(warm=True)
    scanner = get_scanner()
    # Explicitly warm up SHAP explainer with a dummy row so JIT/TreeExplainer initialization is complete
    try:
        from src.explain.shap_explainer import ShapExplainer
        _exp = ShapExplainer()
        dummy_df = pd.DataFrame([{col: 0.0 for col in predictor.feature_names}])
        _ = _exp.explain_transaction(dummy_df)
    except Exception as e:
        print(f"[WARMUP]: SHAP warmup note: {e}")
    print("[SERVER STARTUP]: Initialized User DB, XGBoost Model, Scaler, SHAP Explainer, and Graph State.")
    yield
    print("[SERVER SHUTDOWN]: Closing server connections.")


app = FastAPI(
    title="FinSecure AI v2 — Real-Time Financial Fraud Detection & XAI",
    version="2.0.0",
    description="VTU Major Project: Graph-enhanced XGBoost fraud detection with SHAP-grounded LLM explainability, cost-sensitive evaluation, concept drift monitoring, and user authentication.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# PUBLIC ENDPOINTS (No Authentication Required)
# =============================================================================

@app.get("/health")
def health_check():
    """System health check endpoint."""
    return {
        "status": "healthy",
        "service": "FinSecure AI v2 API",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/debug/keys")
def debug_keys():
    """Debug endpoint to inspect environment keys visibility."""
    g_raw = os.getenv("GEMINI_API_KEY", "")
    o_raw = os.getenv("OPENAI_API_KEY", "")
    return {
        "gemini_present": bool(g_raw),
        "gemini_len": len(g_raw),
        "gemini_prefix": g_raw[:4] if g_raw else "",
        "openai_present": bool(o_raw),
        "openai_len": len(o_raw),
        "openai_prefix": o_raw[:4] if o_raw else "",
        "cwd": os.getcwd(),
        "project_root": str(PROJECT_ROOT),
        "env_path": str((PROJECT_ROOT / ".env").resolve()),
        "env_exists": (PROJECT_ROOT / ".env").exists(),
    }


@app.post("/auth/register", response_model=TokenResponse)
def register_user(req: RegisterRequest, request: Request):
    """Register a new analyst account. Password minimum 8 characters, email validated."""
    ip = request.client.host if request.client else "unknown"

    validate_email_format(req.email)
    validate_password_strength(req.password)

    email_clean = req.email.strip().lower()
    full_name_clean = req.full_name.strip()

    with get_users_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE email = ?", (email_clean,))
        if cursor.fetchone():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="An account with this email address already exists."
            )

        pw_hash = hash_password(req.password)
        now_iso = datetime.now(timezone.utc).isoformat()

        with conn:
            cursor.execute(
                """
                INSERT INTO users (full_name, email, password_hash, role, created_at, last_login)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (full_name_clean, email_clean, pw_hash, "analyst", now_iso, now_iso),
            )
            new_id = cursor.lastrowid

    # Record login event & auto-export Excel
    log_login_event(new_id, email_clean, True, ip)
    try:
        export_users_to_excel()
    except Exception as e:
        print(f"[AUTH EXCEL EXPORT WARNING]: {e}")

    token_data = {"sub": email_clean, "id": new_id, "name": full_name_clean, "role": "analyst"}
    token = create_access_token(token_data)

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        user={"id": new_id, "full_name": full_name_clean, "email": email_clean, "role": "analyst"}
    )


@app.post("/auth/login", response_model=TokenResponse)
def login_user(req: LoginRequest, request: Request):
    """Authenticate user with rate limiting (5 failed attempts per 15 mins)."""
    ip = request.client.host if request.client else "unknown"
    email_clean = req.email.strip().lower()

    # Rate limiting safeguard
    check_rate_limit(email_clean, ip)

    with get_users_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, full_name, email, password_hash, role FROM users WHERE email = ?",
            (email_clean,)
        )
        user_row = cursor.fetchone()

    if not user_row or not verify_password(req.password, user_row[3]):
        record_failed_login(email_clean, ip)
        log_login_event(user_row[0] if user_row else None, email_clean, False, ip)
        try:
            export_users_to_excel()
        except Exception:
            pass
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password."
        )

    # Success: clear failed rate limit counters and update last_login
    clear_failed_logins(email_clean, ip)
    user_id, full_name, email, _, role = user_row
    now_iso = datetime.now(timezone.utc).isoformat()

    with get_users_db_connection() as conn:
        with conn:
            conn.execute("UPDATE users SET last_login = ? WHERE id = ?", (now_iso, user_id))

    log_login_event(user_id, email, True, ip)
    try:
        export_users_to_excel()
    except Exception as e:
        print(f"[AUTH EXCEL EXPORT WARNING]: {e}")

    token_data = {"sub": email, "id": user_id, "name": full_name, "role": role}
    token = create_access_token(token_data)

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        user={"id": user_id, "full_name": full_name, "email": email, "role": role}
    )


# =============================================================================
# PROTECTED USER & ADMIN ENDPOINTS
# =============================================================================

@app.get("/auth/me")
def get_my_profile(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Return currently authenticated user information."""
    return current_user


@app.get("/auth/users")
def list_users(current_admin: Dict[str, Any] = Depends(get_current_admin)):
    """Admin-only: list all registered users (passwords excluded)."""
    with get_users_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, full_name, email, role, created_at, last_login FROM users ORDER BY id ASC")
        rows = cursor.fetchall()

    return [
        {
            "id": r[0],
            "full_name": r[1],
            "email": r[2],
            "role": r[3],
            "created_at": r[4],
            "last_login": r[5],
        }
        for r in rows
    ]


@app.get("/auth/users/export")
def download_users_excel(current_admin: Dict[str, Any] = Depends(get_current_admin)):
    """Admin-only: download the auto-exported users_export.xlsx spreadsheet."""
    export_path = get_excel_export_path()
    export_users_to_excel(export_path=export_path)
    if not export_path.exists():
        raise HTTPException(status_code=404, detail="Excel export file not found.")

    return FileResponse(
        path=str(export_path),
        filename="users_export.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


# =============================================================================
# PROTECTED CORE ML & OPERATIONAL ENDPOINTS
# =============================================================================

@app.get("/model/info", response_model=ModelInfoResponse)
def get_model_info(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Return active model parameters, threshold calibration, and training dataset name."""
    predictor = get_predictor()
    return ModelInfoResponse(
        model_type="XGBoost Classifier + SHAP TreeExplainer",
        feature_count=len(predictor.feature_names),
        dataset_name=getattr(predictor, "dataset_name", "Synthetic PaySim (300K)"),
        active_threshold=predictor.active_threshold,
        active_strategy=predictor.active_strategy,
        f2_optimal_threshold=predictor.f2_optimal_threshold,
        cost_optimal_threshold=predictor.cost_optimal_threshold,
        feature_names=predictor.feature_names,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_transaction(input_data: TransactionInput, current_user: Dict[str, Any] = Depends(get_current_user)):
    """
    Score a single financial transaction.
    Returns prediction result including the server-side database primary key ID.
    Broadcasts the scored transaction to connected WebSocket scanner clients.
    """
    try:
        predictor = get_predictor()
        scanner = get_scanner()

        tx_dict = input_data.model_dump()
        result = predictor.predict_one(tx_dict)

        # Broadcast live scored transaction with server-generated ID to WebSocket clients
        broadcast_payload = {**tx_dict, **result}
        await scanner.process_and_broadcast(broadcast_payload)

        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(batch_input: BatchTransactionInput, current_user: Dict[str, Any] = Depends(get_current_user)):
    """Score a batch of transactions and broadcast each with server database ID."""
    try:
        predictor = get_predictor()
        scanner = get_scanner()
        results = []

        for tx in batch_input.transactions:
            tx_dict = tx.model_dump()
            res = predictor.predict_one(tx_dict)
            results.append(PredictionResponse(**res))

            broadcast_payload = {**tx_dict, **res}
            await scanner.process_and_broadcast(broadcast_payload)

        return BatchPredictionResponse(
            predictions=results,
            total_processed=len(results)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def compute_live_drift_status() -> Dict[str, Any]:
    """Evaluate live Population Stability Index (PSI) against reference baseline."""
    train_df = pd.read_parquet("data/features_train.parquet")
    test_path = Path("data/features_test.parquet")
    val_path = Path("data/features_val.parquet")
    eval_df = pd.read_parquet(test_path) if test_path.exists() else pd.read_parquet(val_path)

    monitor = DriftMonitor(train_df)
    return monitor.check(eval_df)


@app.get("/drift/status", response_model=DriftStatusResponse)
def get_drift_status(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Evaluate current Population Stability Index (PSI) against reference baseline."""
    try:
        status_res = compute_live_drift_status()
        return DriftStatusResponse(**status_res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/drift/simulate", response_model=DriftStatusResponse)
def trigger_drift_simulation(amount_scale: float = 5.0, current_user: Dict[str, Any] = Depends(get_current_user)):
    """Simulate covariate distribution shift (5x transaction amounts) and recalculate PSI."""
    try:
        train_df = pd.read_parquet("data/features_train.parquet")
        test_path = Path("data/features_test.parquet")
        val_path = Path("data/features_val.parquet")
        eval_df = pd.read_parquet(test_path) if test_path.exists() else pd.read_parquet(val_path)

        shifted_df = simulate_drift(eval_df, amount_scale=amount_scale)

        monitor = DriftMonitor(train_df)
        drift_res = monitor.check(shifted_df)

        from api.db import log_drift_snapshot
        log_drift_snapshot(drift_res, is_simulation=True)

        return DriftStatusResponse(**drift_res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/metrics/cost")
def get_cost_metrics(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Return financial expected costs and savings vs baselines."""
    try:
        metadata = joblib.load("models/model_metadata.joblib")
        return {
            "f2_optimal_threshold": metadata.get("f2_optimal_threshold"),
            "cost_optimal_threshold": metadata.get("cost_optimal_threshold"),
            "active_strategy": metadata.get("active_strategy"),
            "active_threshold": metadata.get("active_threshold"),
            "model_expected_cost_rs": metadata.get("model_expected_cost_rs"),
            "baseline_flag_nothing_cost_rs": metadata.get("baseline_flag_nothing_cost_rs"),
            "baseline_flag_everything_cost_rs": metadata.get("baseline_flag_everything_cost_rs"),
            "cost_savings_vs_flag_nothing_rs": metadata.get("cost_savings_vs_flag_nothing_rs"),
            "cost_savings_vs_flag_everything_rs": metadata.get("cost_savings_vs_flag_everything_rs"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
def chat_rag_query(request: ChatRequest, current_user: Dict[str, Any] = Depends(get_current_user)):
    """Tool-calling LLM audit agent answering analyst queries over prediction logs and metrics."""
    history_turns = (request.history or [])[-10:]
    reply = ask_agent(request.query, history=history_turns)
    sources = getattr(reply, "sources", None) or get_last_tool_calls() or []

    return ChatResponse(
        query=request.query,
        answer=str(reply),
        sources=sources,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/predictions/recent")
def get_recent_predictions_endpoint(limit: int = 200, current_user: Dict[str, Any] = Depends(get_current_user)):
    """Retrieve recent predictions ordered by primary key ID descending."""
    try:
        from api.db import get_db_path, get_db_connection
        db_path = get_db_path()
        if not db_path.exists():
            return []
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, timestamp, step, name_orig, name_dest, amount,
                       prediction_proba, is_fraud, threshold_used, explanation
                FROM predictions
                ORDER BY id DESC LIMIT ?
                """,
                (limit,)
            )
            rows = cursor.fetchall()
            return [
                {
                    "id": r[0],
                    "timestamp": r[1],
                    "step": r[2],
                    "nameOrig": r[3],
                    "nameDest": r[4],
                    "amount": r[5],
                    "prediction_proba": r[6],
                    "is_fraud": bool(r[7]),
                    "threshold_used": r[8],
                    "explanation": r[9],
                }
                for r in rows
            ]
    except Exception as e:
        print(f"[ERROR /predictions/recent]: {e}")
        return []


# =============================================================================
# WEBSOCKET ENDPOINTS (Authenticated via ?token=)
# =============================================================================

@app.websocket("/ws/scan")
async def websocket_scan_endpoint(websocket: WebSocket):
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=1008, reason="Authentication token required")
        return
    try:
        decode_access_token(token)
    except Exception:
        await websocket.close(code=1008, reason="Invalid authentication token")
        return

    scanner = get_scanner()
    await scanner.connect_scan(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        scanner.disconnect_scan(websocket)


@app.websocket("/ws/stats")
async def websocket_stats_endpoint(websocket: WebSocket):
    token = websocket.query_params.get("token")
    if not token:
        await websocket.close(code=1008, reason="Authentication token required")
        return
    try:
        decode_access_token(token)
    except Exception:
        await websocket.close(code=1008, reason="Invalid authentication token")
        return

    scanner = get_scanner()
    await scanner.connect_stats(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        scanner.disconnect_stats(websocket)


# Mount PaySim analytics router
app.include_router(analytics_router)

# Mount CSV analysis router
app.include_router(analysis_router)

# Mount static frontend files
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
