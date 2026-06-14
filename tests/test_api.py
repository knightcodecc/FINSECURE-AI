"""
Unit Tests for API Endpoints.

Tests FastAPI endpoints using TestClient.

Author: FinSecure AI Team
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock the model loading since we don't have trained models in tests
import api.main as main
from api.schemas import TransactionInput


# Create a mock predictor for testing
class MockPredictor:
    """Mock predictor for testing without trained model."""

    def __init__(self):
        self.metadata = {'threshold': 0.8, 'feature_columns': []}

    def predict_single(self, transaction):
        """Mock prediction based on transaction patterns."""
        # Mock logic: flag as fraud if account drained + TRANSFER type
        is_fraud = (transaction.newbalanceOrig == 0 and
                   transaction.oldbalanceOrg > 0 and
                   transaction.type.value == 'TRANSFER')

        fraud_probability = 0.95 if is_fraud else 0.1
        risk_level = 'CRITICAL' if fraud_probability >= 0.8 else 'LOW'

        from api.schemas import RiskFactor, RiskLevel

        return {
            'is_fraud': is_fraud,
            'fraud_probability': fraud_probability,
            'risk_level': RiskLevel(risk_level),
            'top_risk_factors': [
                RiskFactor(feature='test', value=1.0, importance=0.5)
            ],
            'processing_time_ms': 10.0
        }

    def predict_batch(self, transactions):
        """Mock batch prediction."""
        return [self.predict_single(t) for t in transactions]


class MockXAIEngine:
    """Mock XAI engine for testing."""

    async def generate_explanation(self, context):
        return "Mock explanation for testing."


# Override the predictor and XAI engine in the app state
def setup_mock_app():
    """Setup app with mock components."""
    main.app_state['predictor'] = MockPredictor()
    main.app_state['xai_engine'] = MockXAIEngine()
    main.app_state['start_time'] = 1000.0
    main.app_state['stats'] = {
        'total_predictions': 0,
        'fraud_detected': 0,
        'total_fraud_amount': 0.0,
        'risk_level_counts': {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0, 'CRITICAL': 0}
    }


# Create test client
client = TestClient(main.app)


# Test fixtures
FAKE_TRANSACTION = {
    "step": 1,
    "type": "TRANSFER",
    "amount": 1000.0,
    "oldbalanceOrg": 5000.0,
    "newbalanceOrig": 4000.0,
    "oldbalanceDest": 1000.0,
    "newbalanceDest": 2000.0
}

FRAUD_TRANSACTION = {
    "step": 1,
    "type": "TRANSFER",
    "amount": 10000.0,
    "oldbalanceOrg": 10000.0,
    "newbalanceOrig": 0.0,
    "oldbalanceDest": 0.0,
    "newbalanceDest": 10000.0
}

INVALID_TRANSACTION = {
    "step": 1,
    "type": "INVALID_TYPE",
    "amount": 1000.0,
    "oldbalanceOrg": 5000.0,
    "newbalanceOrig": 4000.0,
    "oldbalanceDest": 1000.0,
    "newbalanceDest": 2000.0
}

MISSING_FIELDS_TRANSACTION = {
    "step": 1,
    "type": "TRANSFER"
    # Missing required fields
}


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_200(self):
        """Test health check returns 200."""
        setup_mock_app()
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_structure(self):
        """Test health response has correct fields."""
        setup_mock_app()
        response = client.get("/health")
        data = response.json()
        assert 'status' in data
        assert 'model_loaded' in data
        assert 'uptime_seconds' in data
        assert 'version' in data


class TestPredictEndpoint:
    """Tests for /predict endpoint."""

    def test_predict_legitimate_transaction(self):
        """Test prediction for legitimate transaction."""
        setup_mock_app()
        response = client.post("/predict", json=FAKE_TRANSACTION)
        assert response.status_code == 200

        data = response.json()
        assert 'is_fraud' in data
        assert 'fraud_probability' in data
        assert 'risk_level' in data

    def test_predict_fraudulent_transaction(self):
        """Test prediction for fraudulent transaction."""
        setup_mock_app()
        response = client.post("/predict", json=FRAUD_TRANSACTION)
        assert response.status_code == 200

        data = response.json()
        assert data['is_fraud'] == True

    def test_predict_missing_fields(self):
        """Test prediction with missing fields returns 422."""
        setup_mock_app()
        response = client.post("/predict", json=MISSING_FIELDS_TRANSACTION)
        assert response.status_code == 422

    def test_predict_invalid_type(self):
        """Test prediction with invalid transaction type returns 422."""
        setup_mock_app()
        response = client.post("/predict", json=INVALID_TRANSACTION)
        assert response.status_code == 422

    def test_predict_negative_amount(self):
        """Test prediction with negative amount returns 422."""
        setup_mock_app()
        transaction = FAKE_TRANSACTION.copy()
        transaction['amount'] = -100.0
        response = client.post("/predict", json=transaction)
        assert response.status_code == 422

    def test_predict_response_fields(self):
        """Test response contains all required fields."""
        setup_mock_app()
        response = client.post("/predict", json=FAKE_TRANSACTION)
        data = response.json()

        assert 'transaction_id' in data
        assert 'timestamp' in data
        assert 'processing_time_ms' in data
        assert 'top_risk_factors' in data
        assert 'explanation' in data


class TestBatchPredictEndpoint:
    """Tests for /predict/batch endpoint."""

    def test_batch_predict_multiple(self):
        """Test batch prediction with multiple transactions."""
        setup_mock_app()
        batch = {
            "transactions": [
                FAKE_TRANSACTION,
                FRAUD_TRANSACTION,
                FAKE_TRANSACTION
            ]
        }
        response = client.post("/predict/batch", json=batch)
        assert response.status_code == 200

        data = response.json()
        assert data['total_transactions'] == 3
        assert len(data['predictions']) == 3

    def test_batch_predict_single(self):
        """Test batch prediction with single transaction."""
        setup_mock_app()
        batch = {"transactions": [FAKE_TRANSACTION]}
        response = client.post("/predict/batch", json=batch)
        assert response.status_code == 200

    def test_batch_predict_empty(self):
        """Test batch prediction with empty list."""
        setup_mock_app()
        batch = {"transactions": []}
        response = client.post("/predict/batch", json=batch)
        # Should handle empty list
        assert response.status_code in [200, 422]

    def test_batch_response_structure(self):
        """Test batch response has correct structure."""
        setup_mock_app()
        batch = {"transactions": [FAKE_TRANSACTION]}
        response = client.post("/predict/batch", json=batch)
        data = response.json()

        assert 'predictions' in data
        assert 'total_transactions' in data
        assert 'processing_time_ms' in data


class TestStatsEndpoint:
    """Tests for /stats/summary endpoint."""

    def test_stats_returns_200(self):
        """Test stats endpoint returns 200."""
        setup_mock_app()
        response = client.get("/stats/summary")
        assert response.status_code == 200

    def test_stats_response_fields(self):
        """Test stats response contains expected fields."""
        setup_mock_app()
        response = client.get("/stats/summary")
        data = response.json()

        assert 'total_predictions' in data
        assert 'fraud_detected' in data
        assert 'fraud_rate' in data
        assert 'total_fraud_amount' in data


class TestModelInfoEndpoint:
    """Tests for /model/info endpoint."""

    def test_model_info_returns_200(self):
        """Test model info returns 200."""
        setup_mock_app()
        response = client.get("/model/info")
        assert response.status_code == 200

    def test_model_info_response(self):
        """Test model info response fields."""
        setup_mock_app()
        response = client.get("/model/info")
        data = response.json()

        assert 'threshold' in data
        assert 'feature_count' in data
        assert 'training_date' in data
        assert 'model_version' in data


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_returns_200(self):
        """Test root endpoint returns 200."""
        response = client.get("/")
        assert response.status_code == 200

    def test_root_response(self):
        """Test root response contains basic info."""
        response = client.get("/")
        data = response.json()
        assert 'name' in data
        assert 'version' in data


class TestMiddleware:
    """Tests for middleware functionality."""

    def test_request_id_header(self):
        """Test that X-Request-ID header is added."""
        setup_mock_app()
        response = client.post("/predict", json=FAKE_TRANSACTION)
        assert 'x-request-id' in response.headers

    def test_process_time_header(self):
        """Test that X-Process-Time header is added."""
        setup_mock_app()
        response = client.post("/predict", json=FAKE_TRANSACTION)
        assert 'x-process-time' in response.headers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])