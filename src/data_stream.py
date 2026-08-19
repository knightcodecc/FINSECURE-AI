"""
CSV-based transaction streamer for FinSecure AI.

This module provides the TransactionFileStreamer for replaying PaySim1-formatted
transaction CSVs through the real-time scanning pipeline.
"""

import asyncio
import logging
import random
from pathlib import Path
from typing import AsyncGenerator, Optional, Tuple

import pandas as pd

from api.schemas import TransactionInput, TransactionType
from api.realtime_scanner import MIN_INTERVAL, MAX_INTERVAL

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    'step',
    'type',
    'amount',
    'oldbalanceOrg',
    'newbalanceOrig',
    'oldbalanceDest',
    'newbalanceDest'
]

VALID_TRANSACTION_TYPES = {t.value for t in TransactionType}


class TransactionFileStreamer:
    """Stream transactions from a CSV file in PaySim1 schema."""

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Transaction file not found: {self.file_path}")

        self.df = pd.read_csv(self.file_path)
        self._validate_columns(self.df.columns)
        self.has_ground_truth = 'isFraud' in self.df.columns
        self.row_count = len(self.df)

    def _validate_columns(self, columns) -> None:
        found_columns = list(columns)
        missing = [col for col in REQUIRED_COLUMNS if col not in found_columns]
        if missing:
            raise ValueError(
                "CSV missing required columns: {}. Found columns: {}".format(
                    ", ".join(missing),
                    ", ".join(found_columns)
                )
            )

    async def stream(self, speed: float = 1.0) -> AsyncGenerator[Tuple[Optional[TransactionInput], Optional[int]], None]:
        """Yield transactions and optional ground truth label at a paced interval."""
        if speed <= 0:
            raise ValueError("speed must be greater than 0")

        if self.row_count > 2000:
            logger.warning(
                "Transaction file has %d rows; streaming may take a long time. "
                "Consider truncating the file or increasing speed.",
                self.row_count
            )

        for idx, row in self.df.iterrows():
            row_number = idx + 2
            transaction_type = str(row['type']).strip().upper() if pd.notna(row['type']) else ''
            if transaction_type not in VALID_TRANSACTION_TYPES:
                logger.warning(
                    "Skipping row %d: invalid transaction type '%s'",
                    row_number,
                    row['type']
                )
                continue

            try:
                transaction = TransactionInput(
                    step=int(row['step']),
                    type=transaction_type,
                    amount=float(row['amount']),
                    oldbalanceOrg=float(row['oldbalanceOrg']),
                    newbalanceOrig=float(row['newbalanceOrig']),
                    oldbalanceDest=float(row['oldbalanceDest']),
                    newbalanceDest=float(row['newbalanceDest'])
                )
            except Exception as exc:
                logger.warning(
                    "Skipping row %d due to invalid data: %s",
                    row_number,
                    exc
                )
                continue

            label = None
            if self.has_ground_truth:
                label_value = row.get('isFraud')
                try:
                    label = int(label_value) if pd.notna(label_value) else None
                except Exception:
                    logger.warning(
                        "Skipping row %d due to invalid isFraud label: %s",
                        row_number,
                        label_value
                    )
                    continue

            yield transaction, label

            delay = random.uniform(MIN_INTERVAL, MAX_INTERVAL) / speed
            await asyncio.sleep(delay)

        yield None, None
