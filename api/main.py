"""
FastAPI Application Module for FinSecure AI Fraud Detection System.

This module provides the complete FastAPI application with REST endpoints,
WebSocket handlers, middleware, and startup/shutdown events.

Author: FinSecure AI Team
"""

import asyncio
import time
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

from fastapi import FastAPI, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Load environment variables
import os
from dotenv import load_dotenv

load_dotenv()

# Import modules
from api.schemas import (
    TransactionInput,
    PredictionResponse,
    BatchTransactionInput,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    StatsSummaryResponse,
    RiskLevel,
    ErrorResponse
)
from api.predictor import FraudPredictor, create_predictor
from api.xai_engine import XAIEngine, create_xai_engine
from api.realtime_scanner import RealtimeScanner, create_scanner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global state
app_state = {
    'predictor': None,
    'xai_engine': None,
    'scanner': None,
    'start_time': None,
    'stats': {
        'total_predictions': 0,
        'fraud_detected': 0,
        'total_fraud_amount': 0.0,
        'risk_level_counts': {
            'LOW': 0,
            'MEDIUM': 0,
            'HIGH': 0,
            'CRITICAL': 0
        }
    }
}

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


def get_predictor() -> FraudPredictor:
    """Dependency to get the predictor."""
    if app_state['predictor'] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return app_state['predictor']


def get_xai_engine() -> XAIEngine:
    """Dependency to get the XAI engine."""
    if app_state['xai_engine'] is None:
        raise HTTPException(status_code=503, detail="XAI engine not initialized")
    return app_state['xai_engine']


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown."""
    # Startup
    logger.info("Starting FinSecure AI API")
    app_state['start_time'] = time.time()

    # Load predictor
    try:
        predictor = create_predictor(MODELS_DIR)
        app_state['predictor'] = predictor
        logger.info("FraudPredictor loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load predictor: {e}")
        raise

    # Load XAI engine
    try:
        timeout = int(os.getenv("LLM_TIMEOUT_SECONDS", "3"))
        xai_engine = create_xai_engine(timeout_seconds=timeout)
        app_state['xai_engine'] = xai_engine
        logger.info("XAIEngine initialized")
    except Exception as e:
        logger.error(f"Failed to initialize XAI engine: {e}")

    # Create scanner
    try:
        scanner = create_scanner(app_state['predictor'], app_state['xai_engine'])
        app_state['scanner'] = scanner
        logger.info("RealtimeScanner created")
    except Exception as e:
        logger.error(f"Failed to create scanner: {e}")

    logger.info("FinSecure AI API ready")

    yield

    # Shutdown
    logger.info("Shutting down FinSecure AI API")


# Create FastAPI app
app = FastAPI(
    title="FinSecure AI",
    description="End-to-end AI model for financial fraud detection with real-time scanning",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time and request ID to response headers."""
    start_time = time.time()
    request_id = str(uuid4())

    # Add request ID header
    request.state.request_id = request_id

    response = await call_next(request)

    # Add headers
    process_time = (time.time() - start_time) * 1000
    response.headers["X-Process-Time"] = f"{process_time:.2f}ms"
    response.headers["X-Request-ID"] = request_id

    return response


# Exception handler for 422 errors
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "request_id": getattr(request.state, "request_id", None)
        }
    )


# ===== REST Endpoints =====

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "name": "FinSecure AI API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    uptime = time.time() - app_state['start_time'] if app_state['start_time'] else 0

    return HealthResponse(
        status="healthy" if app_state['predictor'] else "degraded",
        model_loaded=app_state['predictor'] is not None,
        uptime_seconds=round(uptime, 2),
        version="1.0.0"
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info(predictor: FraudPredictor = Depends(get_predictor)):
    """Get model metadata."""
    metadata = predictor.metadata

    return ModelInfoResponse(
        threshold=metadata['threshold'],
        feature_count=len(metadata['feature_columns']),
        training_date=metadata.get('training_date', 'unknown'),
        model_version=metadata.get('model_version', '1.0.0')
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(transaction: TransactionInput, request: Request,
                  predictor: FraudPredictor = Depends(get_predictor)):
    """
    Predict fraud for a single transaction.

    Args:
        transaction: TransactionInput with transaction details.

    Returns:
        PredictionResponse with fraud prediction and explanation.
    """
    # Run prediction
    prediction = predictor.predict_single(transaction)

    # Generate explanation if fraud
    explanation = None
    if prediction['is_fraud']:
        xai_engine = get_xai_engine()
        context = {
            'transaction': transaction,
            'fraud_probability': prediction['fraud_probability'],
            'risk_level': prediction['risk_level'],
            'top_risk_factors': prediction['top_risk_factors']
        }
        try:
            explanation = await xai_engine.generate_explanation(context)
        except Exception as e:
            logger.warning(f"Explanation generation failed: {e}")

    # Update statistics
    app_state['stats']['total_predictions'] += 1
    if prediction['is_fraud']:
        app_state['stats']['fraud_detected'] += 1
        app_state['stats']['total_fraud_amount'] += transaction.amount
    app_state['stats']['risk_level_counts'][prediction['risk_level'].value] += 1

    # Build response
    return PredictionResponse(
        is_fraud=prediction['is_fraud'],
        fraud_probability=prediction['fraud_probability'],
        risk_level=prediction['risk_level'],
        explanation=explanation,
        top_risk_factors=prediction['top_risk_factors'],
        processing_time_ms=prediction['processing_time_ms'],
        transaction_id=uuid4(),
        timestamp=datetime.now()
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse,
           tags=["Prediction"])
async def predict_batch(batch: BatchTransactionInput,
                        predictor: FraudPredictor = Depends(get_predictor)):
    """
    Predict fraud for multiple transactions (up to 1000).

    Args:
        batch: BatchTransactionInput with list of transactions.

    Returns:
        BatchPredictionResponse with predictions.
    """
    start_time = time.time()

    # Process batch
    predictions = predictor.predict_batch(batch.transactions)

    # Build responses
    results = []
    for i, pred in enumerate(predictions):
        transaction = batch.transactions[i]

        results.append(PredictionResponse(
            is_fraud=pred['is_fraud'],
            fraud_probability=pred['fraud_probability'],
            risk_level=pred['risk_level'],
            explanation=None,
            top_risk_factors=pred['top_risk_factors'],
            processing_time_ms=pred['processing_time_ms'],
            transaction_id=uuid4(),
            timestamp=datetime.now()
        ))

        # Update stats
        app_state['stats']['total_predictions'] += 1
        if pred['is_fraud']:
            app_state['stats']['fraud_detected'] += 1
            app_state['stats']['total_fraud_amount'] += transaction.amount

    total_time = (time.time() - start_time) * 1000

    return BatchPredictionResponse(
        predictions=results,
        total_transactions=len(results),
        processing_time_ms=round(total_time, 2)
    )


@app.get("/stats/summary", response_model=StatsSummaryResponse, tags=["Statistics"])
async def stats_summary():
    """Get aggregate fraud statistics since server start."""
    stats = app_state['stats']

    total = stats['total_predictions']
    fraud_count = stats['fraud_detected']

    if total > 0:
        fraud_rate = fraud_count / total
        avg_probability = (stats['fraud_detected'] * 0.7) / total  # Rough estimate
    else:
        fraud_rate = 0.0
        avg_probability = 0.0

    return StatsSummaryResponse(
        total_predictions=total,
        fraud_detected=fraud_count,
        fraud_rate=round(fraud_rate, 4),
        total_fraud_amount=round(stats['total_fraud_amount'], 2),
        average_fraud_probability=round(avg_probability, 4),
        risk_level_distribution=stats['risk_level_counts']
    )


# ===== WebSocket Endpoints =====

@app.websocket("/ws/scan")
async def websocket_scan(websocket: WebSocket):
    """
    WebSocket endpoint for real-time transaction scanning.

    Clients connect to receive live transaction scan results.
    """
    await websocket.accept()
    logger.info("WebSocket connection accepted for /ws/scan")

    if app_state['predictor'] is None or app_state['xai_engine'] is None:
        await websocket.send_json({"error": "Model not loaded"})
        await websocket.close()
        return

    scanner = RealtimeScanner(app_state['predictor'], app_state['xai_engine'])

    try:
        await scanner.run_scan(websocket)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        scanner.stop_scan()


@app.websocket("/ws/stats")
async def websocket_stats(websocket: WebSocket):
    """
    WebSocket endpoint for live statistics broadcast.

    Clients connect to receive periodic statistics updates.
    """
    await websocket.accept()
    logger.info("WebSocket connection accepted for /ws/stats")

    try:
        while True:
            if app_state['scanner']:
                stats = app_state['scanner'].get_statistics()
                await websocket.send_json(stats.model_dump())
            await asyncio.sleep(5)
    except WebSocketDisconnect:
        logger.info("Stats WebSocket disconnected")
    except Exception as e:
        logger.error(f"Stats WebSocket error: {e}")


# Add tags for API documentation
def tags_metadata():
    return [
        {"name": "Root", "description": "Root endpoints"},
        {"name": "Health", "description": "Health check endpoints"},
        {"name": "Model", "description": "Model information endpoints"},
        {"name": "Prediction", "description": "Fraud prediction endpoints"},
        {"name": "Statistics", "description": "Statistics endpoints"},
    ]


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True
    )