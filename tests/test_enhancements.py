import pytest
import asyncio
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
import sys
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.predictor import FraudPredictor
from api.schemas import TransactionInput, TransactionType, RiskFactor, RiskLevel
from api.xai_engine import XAIEngine


class MockModel:
    def __init__(self, prob_class_1=0.85):
        self.feature_importances_ = np.array([0.6, 0.4])
        self.prob_class_1 = prob_class_1

    def predict_proba(self, X):
        # Return probability array of shape (N, 2)
        n_samples = X.shape[0]
        probs = np.zeros((n_samples, 2))
        probs[:, 0] = 1.0 - self.prob_class_1
        probs[:, 1] = self.prob_class_1
        return probs


@pytest.fixture
def dummy_predictor():
    # Initialize FraudPredictor without calling _load_model
    predictor = FraudPredictor.__new__(FraudPredictor)
    predictor.feature_columns = ['balance_error_orig', 'account_drained']
    predictor.metadata = {'threshold': 0.5}
    
    # Set up mock model
    predictor.model = MockModel()
    
    # Set up scaler (only numeric features in self.feature_columns are scaled)
    scaler = StandardScaler()
    # Fit scaler on dummy data for balance_error_orig
    df_fit = pd.DataFrame({
        'balance_error_orig': [10.0, 20.0, 30.0]
    })
    scaler.fit(df_fit)
    predictor.scaler = scaler
    
    return predictor


def test_transactions_to_dataframe(dummy_predictor):
    tx1 = TransactionInput(
        step=1, type=TransactionType.TRANSFER, amount=100.0,
        oldbalanceOrg=100.0, newbalanceOrig=0.0,
        oldbalanceDest=0.0, newbalanceDest=100.0
    )
    tx2 = TransactionInput(
        step=2, type=TransactionType.CASH_OUT, amount=50.0,
        oldbalanceOrg=50.0, newbalanceOrig=0.0,
        oldbalanceDest=0.0, newbalanceDest=50.0
    )
    
    df = dummy_predictor._transactions_to_dataframe([tx1, tx2])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert df.iloc[0]['amount'] == 100.0
    assert df.iloc[1]['amount'] == 50.0
    assert df.iloc[0]['type_TRANSFER'] == 1
    assert df.iloc[1]['type_CASH_OUT'] == 1


def test_predict_batch_vectorized(dummy_predictor):
    tx1 = TransactionInput(
        step=1, type=TransactionType.TRANSFER, amount=100.0,
        oldbalanceOrg=100.0, newbalanceOrig=0.0,
        oldbalanceDest=0.0, newbalanceDest=100.0
    )
    tx2 = TransactionInput(
        step=2, type=TransactionType.CASH_OUT, amount=50.0,
        oldbalanceOrg=50.0, newbalanceOrig=0.0,
        oldbalanceDest=0.0, newbalanceDest=50.0
    )
    
    # Call predict_batch
    results = dummy_predictor.predict_batch([tx1, tx2])
    
    assert len(results) == 2
    assert results[0]['is_fraud'] is True
    assert results[0]['fraud_probability'] == 0.85
    assert results[0]['risk_level'] == RiskLevel.CRITICAL
    assert 'top_risk_factors' in results[0]
    assert 'processing_time_ms' in results[0]
    
    # Test empty batch
    assert dummy_predictor.predict_batch([]) == []


def test_transaction_specific_risk_factors(dummy_predictor):
    # Setup two distinct test DataFrames representing two transactions
    # Tx A: high balance error orig, account_drained = 0
    df_a = pd.DataFrame({
        'balance_error_orig': [50.0],  # Mean is 20.0, Scale is ~8.16, so std_val will be high
        'account_drained': [0]
    })
    
    # Tx B: low balance error orig, account_drained = 1
    df_b = pd.DataFrame({
        'balance_error_orig': [20.0],  # std_val will be 0.0
        'account_drained': [1]
    })
    
    factors_a = dummy_predictor._get_risk_factors(df_a, 0.85)
    factors_b = dummy_predictor._get_risk_factors(df_b, 0.85)
    
    # Assert features are sorted differently due to transaction-specific values
    assert factors_a[0].feature == 'balance_error_orig'
    assert factors_b[0].feature == 'account_drained'


def test_generate_explanation_sync_no_loop():
    engine = XAIEngine()
    context = {
        'transaction': TransactionInput(
            step=1, type=TransactionType.TRANSFER, amount=100.0,
            oldbalanceOrg=100.0, newbalanceOrig=0.0,
            oldbalanceDest=0.0, newbalanceDest=100.0
        ),
        'fraud_probability': 0.85,
        'risk_level': RiskLevel.CRITICAL,
        'top_risk_factors': [
            RiskFactor(feature='account_drained', value=1.0, importance=1.8)
        ]
    }
    
    # Call synchronous wrapper from a non-async thread
    explanation = engine.generate_explanation_sync(context)
    assert isinstance(explanation, str)
    assert len(explanation) > 0
    # Rule-based fallback explanation check
    assert "drained" in explanation.lower() or "transaction flagged" in explanation.lower()


@pytest.mark.asyncio
async def test_generate_explanation_sync_with_running_loop():
    # This runs within a running event loop (due to pytest-asyncio)
    engine = XAIEngine()
    context = {
        'transaction': TransactionInput(
            step=1, type=TransactionType.TRANSFER, amount=100.0,
            oldbalanceOrg=100.0, newbalanceOrig=0.0,
            oldbalanceDest=0.0, newbalanceDest=100.0
        ),
        'fraud_probability': 0.85,
        'risk_level': RiskLevel.CRITICAL,
        'top_risk_factors': [
            RiskFactor(feature='account_drained', value=1.0, importance=1.8)
        ]
    }
    
    # Run the sync wrapper in a background thread to simulate how FastAPI calls sync endpoints.
    # This prevents blocking the main event loop thread where pytest-asyncio runs.
    loop = asyncio.get_running_loop()
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(engine.generate_explanation_sync, context)
        explanation = await loop.run_in_executor(None, future.result)
        
    assert isinstance(explanation, str)
    assert len(explanation) > 0
    assert "drained" in explanation.lower() or "transaction flagged" in explanation.lower()
