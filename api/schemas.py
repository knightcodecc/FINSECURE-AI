"""
API Schemas Module for FinSecure AI Fraud Detection System.

This module defines Pydantic v2 request/response models for the FastAPI
application, including validation and serialization.

Author: FinSecure AI Team
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional
from uuid import uuid4, UUID
from pydantic import BaseModel, Field, field_validator


class TransactionType(str, Enum):
    """Valid transaction types."""
    TRANSFER = "TRANSFER"
    CASH_OUT = "CASH_OUT"
    PAYMENT = "PAYMENT"
    CASH_IN = "CASH_IN"
    DEBIT = "DEBIT"


class TransactionInput(BaseModel):
    """
    Request model for single transaction fraud prediction.

    Fields:
        step: Time step (simulated hour).
        type: Transaction type (TRANSFER, CASH_OUT, PAYMENT, CASH_IN, DEBIT).
        amount: Transaction amount (> 0).
        oldbalanceOrg: Sender balance before transaction (>= 0).
        newbalanceOrig: Sender balance after transaction (>= 0).
        oldbalanceDest: Receiver balance before transaction (>= 0).
        newbalanceDest: Receiver balance after transaction (>= 0).
    """
    step: int = Field(..., ge=0, description="Time step (simulated hour)")
    type: TransactionType = Field(..., description="Transaction type")
    amount: float = Field(..., gt=0, description="Transaction amount (must be > 0)")
    oldbalanceOrg: float = Field(..., ge=0, description="Sender balance before transaction")
    newbalanceOrig: float = Field(..., ge=0, description="Sender balance after transaction")
    oldbalanceDest: float = Field(..., ge=0, description="Receiver balance before transaction")
    newbalanceDest: float = Field(..., ge=0, description="Receiver balance after transaction")

    @field_validator('type', mode='before')
    @classmethod
    def validate_type(cls, v: str) -> TransactionType:
        """Validate transaction type is one of the valid values."""
        if isinstance(v, TransactionType):
            return v
        if isinstance(v, str):
            v_upper = v.upper()
            for enum_val in TransactionType:
                if enum_val.value == v_upper:
                    return enum_val
        raise ValueError(f"Invalid transaction type: {v}")


class RiskLevel(str, Enum):
    """Risk level classification."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RiskFactor(BaseModel):
    """Individual risk factor with importance score."""
    feature: str = Field(..., description="Feature name")
    value: float = Field(..., description="Feature value")
    importance: float = Field(..., description="Feature importance score")


class PredictionResponse(BaseModel):
    """
    Response model for fraud prediction.

    Fields:
        is_fraud: Whether the transaction is flagged as fraud.
        fraud_probability: Fraud probability score (0-1).
        risk_level: Risk level classification (LOW/MEDIUM/HIGH/CRITICAL).
        explanation: LLM-generated explanation (optional).
        top_risk_factors: List of top 5 risk factors.
        processing_time_ms: Time taken to process the prediction.
        transaction_id: Auto-generated UUID for the transaction.
        timestamp: Timestamp of the prediction.
    """
    is_fraud: bool = Field(..., description="Whether transaction is flagged as fraud")
    fraud_probability: float = Field(..., ge=0, le=1, description="Fraud probability score (0-1)")
    risk_level: RiskLevel = Field(..., description="Risk level classification")
    explanation: Optional[str] = Field(None, description="LLM-generated explanation")
    top_risk_factors: List[RiskFactor] = Field(default_factory=list, description="Top 5 risk factors")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    transaction_id: UUID = Field(default_factory=uuid4, description="Transaction UUID")
    timestamp: datetime = Field(default_factory=datetime.now, description="Prediction timestamp")


class BatchTransactionInput(BaseModel):
    """
    Request model for batch transaction prediction.

    A list of TransactionInput objects, max 1000 items.
    """
    transactions: List[TransactionInput] = Field(
        ...,
        max_length=1000,
        description="List of transactions (max 1000)"
    )


class BatchPredictionResponse(BaseModel):
    """Response model for batch prediction."""
    predictions: List[PredictionResponse] = Field(..., description="List of predictions")
    total_transactions: int = Field(..., description="Total number of transactions processed")
    processing_time_ms: float = Field(..., description="Total processing time")


class RealtimeScanStatus(BaseModel):
    """
    WebSocket message model for real-time scan status.

    Fields:
        transaction_id: Transaction UUID.
        is_fraud: Whether flagged as fraud.
        fraud_probability: Fraud probability score.
        risk_level: Risk level.
        amount: Transaction amount.
        type: Transaction type.
        timestamp: Transaction timestamp.
        explanation: Optional explanation for fraud cases.
    """
    transaction_id: UUID = Field(default_factory=uuid4, description="Transaction UUID")
    is_fraud: bool = Field(..., description="Whether flagged as fraud")
    fraud_probability: float = Field(..., ge=0, le=1, description="Fraud probability score")
    risk_level: RiskLevel = Field(..., description="Risk level")
    amount: float = Field(..., description="Transaction amount")
    type: TransactionType = Field(..., description="Transaction type")
    timestamp: datetime = Field(default_factory=datetime.now, description="Transaction timestamp")
    explanation: Optional[str] = Field(None, description="Fraud explanation (if applicable)")


class ScannerStats(BaseModel):
    """Statistics model for real-time scanner."""
    total_scanned: int = Field(..., description="Total transactions scanned")
    total_flagged: int = Field(..., description="Total transactions flagged as fraud")
    total_fraud_amount: float = Field(..., description="Total fraud amount detected")
    scan_rate_per_second: float = Field(..., description="Current scan rate per second")
    fraud_rate: float = Field(..., description="Fraud detection rate")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    uptime_seconds: float = Field(..., description="Server uptime in seconds")
    version: str = Field(..., description="API version")


class ModelInfoResponse(BaseModel):
    """Model metadata response."""
    threshold: float = Field(..., description="Classification threshold")
    feature_count: int = Field(..., description="Number of features")
    training_date: str = Field(..., description="Model training date")
    model_version: str = Field(..., description="Model version")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    request_id: Optional[str] = Field(None, description="Request ID for debugging")


class StatsSummaryResponse(BaseModel):
    """Aggregate fraud statistics response."""
    total_predictions: int = Field(..., description="Total predictions since server start")
    fraud_detected: int = Field(..., description="Total fraud cases detected")
    fraud_rate: float = Field(..., description="Overall fraud detection rate")
    total_fraud_amount: float = Field(..., description="Total amount involved in fraud")
    average_fraud_probability: float = Field(..., description="Average fraud probability")
    risk_level_distribution: dict = Field(..., description="Distribution by risk level")