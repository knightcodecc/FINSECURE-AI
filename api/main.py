"""
main.py — FastAPI Application Entrypoint for FinSecure AI v2.
==============================================================
Provides REST API endpoints and WebSocket live transaction streams for:
- Real-time fraud detection (/predict, /predict/batch)
- Grounded SHAP XAI explainability
- Cost-sensitive metrics (/metrics/cost)
- Concept drift monitoring & simulation (/drift/status, /drift/simulate)
- RAG chatbot over prediction logs (/chat)
- WebSocket streaming (/ws/scan, /ws/stats)
"""

from contextlib import asynccontextmanager
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Dict, List, Any

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.schemas import (
    TransactionInput,
    PredictionResponse,
    BatchTransactionInput,
    BatchPredictionResponse,
    DriftStatusResponse,
    ModelInfoResponse,
    ChatRequest,
    ChatResponse,
)
from api.predictor import get_predictor
from api.realtime_scanner import get_scanner
from api.rag_chatbot import answer_analyst_query
from src.drift_monitor import DriftMonitor, simulate_drift, DB_PATH
from src.cost_sensitive_eval import expected_cost, threshold_sweep

import joblib
import pandas as pd


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model predictors and SQLite database at startup."""
    predictor = get_predictor()
    scanner = get_scanner()
    print("[SERVER STARTUP]: Loaded XGBoost model, Scaler, SHAP Explainer, and Graph State.")
    yield
    print("[SERVER SHUTDOWN]: Closing server connections.")


app = FastAPI(
    title="FinSecure AI v2 — Real-Time Financial Fraud Detection & XAI",
    version="2.0.0",
    description="VTU Major Project: Graph-enhanced XGBoost fraud detection with SHAP-grounded LLM explainability, cost-sensitive evaluation, and concept drift monitoring.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "FinSecure AI v2 API",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/model/info", response_model=ModelInfoResponse)
def get_model_info():
    predictor = get_predictor()
    return ModelInfoResponse(
        model_type="XGBoost Classifier + SHAP TreeExplainer",
        feature_count=len(predictor.feature_names),
        active_threshold=predictor.active_threshold,
        active_strategy=predictor.active_strategy,
        f2_optimal_threshold=predictor.f2_optimal_threshold,
        cost_optimal_threshold=predictor.cost_optimal_threshold,
        feature_names=predictor.feature_names,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_transaction(input_data: TransactionInput):
    try:
        predictor = get_predictor()
        scanner = get_scanner()
        
        tx_dict = input_data.model_dump()
        result = predictor.predict_one(tx_dict)

        # Broadcast live scored transaction to WebSocket clients
        broadcast_payload = {**tx_dict, **result}
        await scanner.process_and_broadcast(broadcast_payload)

        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(batch_input: BatchTransactionInput):
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


@app.get("/drift/status", response_model=DriftStatusResponse)
def get_drift_status():
    try:
        train_df = pd.read_parquet("data/features_train.parquet")
        val_df = pd.read_parquet("data/features_val.parquet")
        feature_cols = [c for c in train_df.columns if c not in ("isFraud", "step", "mechanism")]

        monitor = DriftMonitor(train_df)
        status_res = monitor.check(val_df[feature_cols])

        return DriftStatusResponse(**status_res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/drift/simulate", response_model=DriftStatusResponse)
def trigger_drift_simulation(amount_scale: float = 5.0):
    try:
        train_df = pd.read_parquet("data/features_train.parquet")
        test_path = Path("data/features_test.parquet")
        val_path = Path("data/features_val.parquet")
        eval_df = pd.read_parquet(test_path) if test_path.exists() else pd.read_parquet(val_path)

        shifted_df = simulate_drift(eval_df, amount_scale=amount_scale)
        
        monitor = DriftMonitor(train_df)
        drift_res = monitor.check(shifted_df)

        return DriftStatusResponse(**drift_res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/cost")
def get_cost_metrics():
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
def chat_rag_query(request: ChatRequest):
    answer = answer_analyst_query(request.query)
    return ChatResponse(
        query=request.query,
        answer=answer,
        timestamp=datetime.now().isoformat(),
    )


# WebSocket Endpoints
@app.websocket("/ws/scan")
async def websocket_scan_endpoint(websocket: WebSocket):
    scanner = get_scanner()
    await scanner.connect_scan(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        scanner.disconnect_scan(websocket)


@app.websocket("/ws/stats")
async def websocket_stats_endpoint(websocket: WebSocket):
    scanner = get_scanner()
    await scanner.connect_stats(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        scanner.disconnect_stats(websocket)
