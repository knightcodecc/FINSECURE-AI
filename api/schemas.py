"""
schemas.py — Pydantic request/response models for FinSecure AI v2.
===================================================================
Single source of truth for FastAPI endpoints.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class TransactionInput(BaseModel):
    step: int = Field(default=1, description="Step hour index")
    type: str = Field(..., description="Transaction type: CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER")
    amount: float = Field(..., gt=0.0, description="Transaction amount")
    nameOrig: str = Field(..., description="Origin account ID")
    oldbalanceOrg: float = Field(..., ge=0.0, description="Origin initial balance")
    newbalanceOrig: float = Field(..., ge=0.0, description="Origin final balance")
    nameDest: str = Field(..., description="Destination account ID")
    oldbalanceDest: float = Field(..., ge=0.0, description="Destination initial balance")
    newbalanceDest: float = Field(..., ge=0.0, description="Destination final balance")


class ShapFeatureItem(BaseModel):
    feature: str
    human_name: str
    raw_value: float
    shap_value: float
    direction: str


class PredictionResponse(BaseModel):
    prediction_proba: float
    is_fraud: bool
    threshold_used: float
    active_strategy: str
    f2_optimal_threshold: float
    cost_optimal_threshold: float
    shap_top_features: List[ShapFeatureItem]
    explanation: str
    timestamp: str


class BatchTransactionInput(BaseModel):
    transactions: List[TransactionInput]


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_processed: int


class DriftStatusResponse(BaseModel):
    timestamp: str
    overall_status: str
    retrain_recommended: bool
    score_psi: float
    drifted_feature_count: int
    warning_feature_count: int
    drifted_features: List[str]
    feature_psi: Dict[str, float]
    feature_status: Dict[str, str]


class ModelInfoResponse(BaseModel):
    model_type: str = "XGBoost Classifier + SHAP TreeExplainer"
    feature_count: int = 33
    active_threshold: float
    active_strategy: str
    f2_optimal_threshold: float
    cost_optimal_threshold: float
    feature_names: List[str]


class ChatRequest(BaseModel):
    query: str = Field(..., description="Analyst question regarding fraud logs or drift status")


class ChatResponse(BaseModel):
    query: str
    answer: str
    timestamp: str
