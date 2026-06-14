"""
Real-Time Scanner Module for FinSecure AI Fraud Detection System.

This module provides WebSocket-based real-time transaction scanning with
synthetic data generation and fraud detection.

Author: FinSecure AI Team
"""

import asyncio
import logging
import random
import time
from datetime import datetime
from typing import AsyncGenerator, Dict, Any, Optional
from dataclasses import dataclass, field
from uuid import uuid4

import numpy as np

# Import schemas
from api.schemas import (
    TransactionInput,
    TransactionType,
    RealtimeScanStatus,
    ScannerStats,
    RiskLevel
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Constants
DEFAULT_FRAUD_RATE = 0.02  # 2% of transactions are fraudulent
MIN_INTERVAL = 0.5  # seconds
MAX_INTERVAL = 2.0  # seconds

# Transaction type probabilities
TYPE_WEIGHTS = {
    TransactionType.TRANSFER: 0.15,
    TransactionType.CASH_OUT: 0.20,
    TransactionType.PAYMENT: 0.35,
    TransactionType.CASH_IN: 0.20,
    TransactionType.DEBIT: 0.10
}


@dataclass
class ScannerStatistics:
    """Running statistics for the scanner."""
    total_scanned: int = 0
    total_flagged: int = 0
    total_fraud_amount: float = 0.0
    fraud_rate: float = 0.0
    scan_rate_per_second: float = 0.0
    scan_start_time: float = field(default_factory=time.time)
    last_update_time: float = field(default_factory=time.time)

    def update(self, is_fraud: bool, amount: float) -> None:
        """Update statistics with a new transaction."""
        self.total_scanned += 1

        if is_fraud:
            self.total_flagged += 1
            self.total_fraud_amount += amount

        # Update rates
        elapsed = time.time() - self.scan_start_time
        if elapsed > 0:
            self.scan_rate_per_second = self.total_scanned / elapsed

        if self.total_scanned > 0:
            self.fraud_rate = self.total_flagged / self.total_scanned

        self.last_update_time = time.time()

    def to_model(self) -> ScannerStats:
        """Convert to Pydantic model."""
        return ScannerStats(
            total_scanned=self.total_scanned,
            total_flagged=self.total_flagged,
            total_fraud_amount=round(self.total_fraud_amount, 2),
            scan_rate_per_second=round(self.scan_rate_per_second, 2),
            fraud_rate=round(self.fraud_rate, 4)
        )


class TransactionStreamSimulator:
    """
    Generates synthetic transactions mimicking PaySim1 schema.

    Produces realistic transaction data at random intervals with
    intentional fraud injection based on known fraud patterns.
    """

    def __init__(self, fraud_rate: float = DEFAULT_FRAUD_RATE):
        """
        Initialize transaction stream simulator.

        Args:
            fraud_rate: Probability of generating fraudulent transactions.
        """
        self.fraud_rate = fraud_rate
        self.step_counter = 0

    def _generate_legitimate_transaction(self) -> TransactionInput:
        """Generate a legitimate transaction."""
        # Choose transaction type based on weights
        transaction_type = random.choices(
            list(TYPE_WEIGHTS.keys()),
            weights=list(TYPE_WEIGHTS.values())
        )[0]

        # Generate realistic amounts based on type
        amount_ranges = {
            TransactionType.TRANSFER: (100, 100000),
            TransactionType.CASH_OUT: (50, 50000),
            TransactionType.PAYMENT: (10, 5000),
            TransactionType.CASH_IN: (100, 50000),
            TransactionType.DEBIT: (20, 10000)
        }

        amount = random.uniform(*amount_ranges[transaction_type])

        # Generate balances
        oldbalanceOrg = random.uniform(0, amount * 5) if random.random() > 0.3 else amount + random.uniform(0, 10000)
        newbalanceOrig = oldbalanceOrg - amount if transaction_type != TransactionType.CASH_IN else oldbalanceOrg + amount
        newbalanceOrig = max(0, newbalanceOrig)

        oldbalanceDest = random.uniform(0, 50000)
        newbalanceDest = oldbalanceDest + amount

        return TransactionInput(
            step=self.step_counter,
            type=transaction_type,
            amount=round(amount, 2),
            oldbalanceOrg=round(oldbalanceOrg, 2),
            newbalanceOrig=round(newbalanceOrig, 2),
            oldbalanceDest=round(oldbalanceDest, 2),
            newbalanceDest=round(newbalanceDest, 2)
        )

    def _generate_fraudulent_transaction(self) -> TransactionInput:
        """Generate a fraudulent transaction using known fraud patterns."""
        # Fraud patterns: account draining + TRANSFER type + new destination
        self.step_counter += 1

        amount = random.uniform(10000, 100000)

        # Account with some balance that gets drained
        oldbalanceOrg = amount + random.uniform(0, 5000)  # Just enough or slightly more
        newbalanceOrig = 0  # Completely drained

        # New destination account (zero or low balance)
        oldbalanceDest = random.uniform(0, 1000)
        newbalanceDest = oldbalanceDest + amount

        return TransactionInput(
            step=self.step_counter,
            type=TransactionType.TRANSFER,
            amount=round(amount, 2),
            oldbalanceOrg=round(oldbalanceOrg, 2),
            newbalanceOrig=round(newbalanceOrig, 2),
            oldbalanceDest=round(oldbalanceDest, 2),
            newbalanceDest=round(newbalanceDest, 2)
        )

    async def stream(self) -> AsyncGenerator[TransactionInput, None]:
        """
        Async generator that yields transactions at random intervals.

        Yields:
            TransactionInput objects at random intervals (0.5-2 seconds).
        """
        while True:
            # Decide if this should be fraudulent
            is_fraud = random.random() < self.fraud_rate

            if is_fraud:
                transaction = self._generate_fraudulent_transaction()
            else:
                self.step_counter += 1
                transaction = self._generate_legitimate_transaction()

            yield transaction

            # Random delay between transactions
            delay = random.uniform(MIN_INTERVAL, MAX_INTERVAL)
            await asyncio.sleep(delay)


class RealtimeScanner:
    """
    Real-time WebSocket transaction scanner.

    Connects to the stream simulator, runs fraud detection on each
    transaction, and broadcasts results via WebSocket.
    """

    def __init__(self, predictor: Any, xai_engine: Any):
        """
        Initialize real-time scanner.

        Args:
            predictor: FraudPredictor instance.
            xai_engine: XAIEngine instance.
        """
        self.predictor = predictor
        self.xai_engine = xai_engine
        self.simulator = TransactionStreamSimulator()
        self.statistics = ScannerStatistics()
        self._running = False

    async def scan_transaction(self, transaction: TransactionInput,
                               generate_explanation: bool = False) -> RealtimeScanStatus:
        """
        Scan a single transaction for fraud.

        Args:
            transaction: Transaction to scan.
            generate_explanation: Whether to generate LLM explanation.

        Returns:
            RealtimeScanStatus with scan results.
        """
        # Run prediction
        prediction = self.predictor.predict_single(transaction)

        # Generate explanation if fraud and requested
        explanation = None
        if prediction['is_fraud'] and generate_explanation:
            context = {
                'transaction': transaction,
                'fraud_probability': prediction['fraud_probability'],
                'risk_level': prediction['risk_level'],
                'top_risk_factors': prediction['top_risk_factors']
            }
            explanation = await self.xai_engine.generate_explanation(context)

        # Update statistics
        self.statistics.update(
            is_fraud=prediction['is_fraud'],
            amount=transaction.amount
        )

        return RealtimeScanStatus(
            transaction_id=uuid4(),
            is_fraud=prediction['is_fraud'],
            fraud_probability=prediction['fraud_probability'],
            risk_level=prediction['risk_level'],
            amount=transaction.amount,
            type=transaction.type,
            timestamp=datetime.now(),
            explanation=explanation
        )

    async def run_scan(self, websocket: Any, duration: Optional[float] = None) -> None:
        """
        Run continuous scan on a WebSocket connection.

        Args:
            websocket: WebSocket connection to send results to.
            duration: Optional duration in seconds (None for infinite).
        """
        self._running = True
        start_time = time.time()

        logger.info("Starting real-time scan")

        async for transaction in self.simulator.stream():
            if not self._running:
                break

            # Scan transaction
            result = await self.scan_transaction(
                transaction,
                generate_explanation=True   # always attempt; XAIEngine handles the threshold internally
            )

            # Send result via WebSocket
            await websocket.send_json(result.model_dump())

            # Check duration limit
            if duration and (time.time() - start_time) >= duration:
                break

        logger.info("Real-time scan stopped")

    def stop_scan(self) -> None:
        """Stop the running scan."""
        self._running = False

    def get_statistics(self) -> ScannerStats:
        """Get current scanner statistics."""
        return self.statistics.to_model()


# WebSocket endpoint handlers
async def handle_scan_websocket(websocket: Any, predictor: Any,
                                xai_engine: Any) -> None:
    """
    Handle WebSocket connection for real-time scanning.

    Args:
        websocket: WebSocket connection.
        predictor: FraudPredictor instance.
        xai_engine: XAIEngine instance.
    """
    scanner = RealtimeScanner(predictor, xai_engine)

    try:
        await scanner.run_scan(websocket)
    except Exception as e:
        logger.error(f"WebSocket scan error: {e}")
    finally:
        scanner.stop_scan()


async def stats_broadcast_websocket(websocket: Any, scanner: RealtimeScanner) -> None:
    """
    Handle WebSocket connection for statistics broadcast.

    Args:
        websocket: WebSocket connection.
        scanner: RealtimeScanner instance to get stats from.
    """
    try:
        while True:
            stats = scanner.get_statistics()
            await websocket.send_json(stats.model_dump())
            await asyncio.sleep(5)  # Broadcast every 5 seconds
    except Exception as e:
        logger.error(f"WebSocket stats error: {e}")


def create_scanner(predictor: Any, xai_engine: Any) -> RealtimeScanner:
    """
    Factory function to create a scanner.

    Args:
        predictor: FraudPredictor instance.
        xai_engine: XAIEngine instance.

    Returns:
        Initialized RealtimeScanner instance.
    """
    return RealtimeScanner(predictor, xai_engine)