"""
Unit tests for CSV-based transaction streaming.
"""

import asyncio
import csv
import sys
from pathlib import Path
import tempfile

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_stream import TransactionFileStreamer
from api.schemas import TransactionType


def write_csv(path, rows, fieldnames):
    with open(path, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.asyncio
async def test_transaction_file_streamer_valid_csv(tmp_path):
    """Valid CSV should stream all rows and preserve ground truth."""
    file_path = tmp_path / 'valid.csv'
    fieldnames = [
        'step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
        'oldbalanceDest', 'newbalanceDest', 'isFraud'
    ]
    rows = [
        {
            'step': 1,
            'type': 'TRANSFER',
            'amount': 1000.0,
            'oldbalanceOrg': 5000.0,
            'newbalanceOrig': 4000.0,
            'oldbalanceDest': 1000.0,
            'newbalanceDest': 2000.0,
            'isFraud': 0
        },
        {
            'step': 2,
            'type': 'PAYMENT',
            'amount': 200.0,
            'oldbalanceOrg': 300.0,
            'newbalanceOrig': 100.0,
            'oldbalanceDest': 100.0,
            'newbalanceDest': 300.0,
            'isFraud': 1
        }
    ]
    write_csv(file_path, rows, fieldnames)

    streamer = TransactionFileStreamer(str(file_path))
    assert streamer.row_count == 2
    assert streamer.has_ground_truth is True

    received = []
    async for item, label in streamer.stream(speed=10.0):
        if item is None and label is None:
            received.append((None, None))
            break
        received.append((item, label))

    assert len(received) == 3
    assert received[0][0].type == TransactionType.TRANSFER
    assert received[0][1] == 0
    assert received[1][0].type == TransactionType.PAYMENT
    assert received[1][1] == 1
    assert received[-1] == (None, None)


def test_transaction_file_streamer_missing_required_columns(tmp_path):
    """Missing required columns should raise a ValueError with details."""
    file_path = tmp_path / 'missing.csv'
    fieldnames = ['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceDest']
    rows = [
        {
            'step': 1,
            'type': 'TRANSFER',
            'amount': 1000.0,
            'oldbalanceOrg': 5000.0,
            'newbalanceDest': 2000.0
        }
    ]
    write_csv(file_path, rows, fieldnames)

    with pytest.raises(ValueError) as exc_info:
        TransactionFileStreamer(str(file_path))

    assert 'missing required columns' in str(exc_info.value).lower()
    assert 'newbalanceOrig' in str(exc_info.value)
    assert 'oldbalanceDest' in str(exc_info.value)


@pytest.mark.asyncio
async def test_transaction_file_streamer_invalid_type_rows_are_skipped(tmp_path, caplog):
    """Rows with invalid transaction types should be skipped and logged."""
    file_path = tmp_path / 'invalid_type.csv'
    fieldnames = [
        'step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
        'oldbalanceDest', 'newbalanceDest'
    ]
    rows = [
        {
            'step': 1,
            'type': 'UNKNOWN',
            'amount': 100.0,
            'oldbalanceOrg': 1000.0,
            'newbalanceOrig': 900.0,
            'oldbalanceDest': 50.0,
            'newbalanceDest': 150.0
        },
        {
            'step': 2,
            'type': 'CASH_IN',
            'amount': 500.0,
            'oldbalanceOrg': 100.0,
            'newbalanceOrig': 600.0,
            'oldbalanceDest': 200.0,
            'newbalanceDest': 700.0
        }
    ]
    write_csv(file_path, rows, fieldnames)

    streamer = TransactionFileStreamer(str(file_path))
    caplog.clear()
    received = []

    async for item, label in streamer.stream(speed=10.0):
        if item is None and label is None:
            received.append((None, None))
            break
        received.append((item, label))

    assert len(received) == 2
    assert received[0][0].type == TransactionType.CASH_IN
    assert received[0][1] is None
    assert received[1] == (None, None)
    assert any('Skipping row' in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_transaction_file_streamer_yields_sentinel_at_end(tmp_path):
    """Streamer yields a final None sentinel after the last row."""
    file_path = tmp_path / 'sentinel.csv'
    fieldnames = [
        'step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
        'oldbalanceDest', 'newbalanceDest'
    ]
    rows = [
        {
            'step': 1,
            'type': 'DEBIT',
            'amount': 150.0,
            'oldbalanceOrg': 200.0,
            'newbalanceOrig': 50.0,
            'oldbalanceDest': 20.0,
            'newbalanceDest': 170.0
        }
    ]
    write_csv(file_path, rows, fieldnames)

    streamer = TransactionFileStreamer(str(file_path))
    messages = []

    async for item, label in streamer.stream(speed=10.0):
        messages.append((item, label))

    assert messages[-1] == (None, None)
    assert messages[0][0].type == TransactionType.DEBIT
    assert messages[0][1] is None
