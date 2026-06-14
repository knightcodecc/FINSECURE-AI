# FinSecure AI - End-to-End Financial Fraud Detection System

A production-ready fraud detection system that ingests financial transaction data, detects fraud using XGBoost, explains predictions using an LLM, and serves everything through a real-time REST API with a live web dashboard.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FinSecure AI                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐ │
│  │   PaySim1    │───▶│  Preprocessing   │───▶│ Feature Engine   │ │
│  │   Dataset    │    │   (src/preprocess)│    │(src/feature_eng) │ │
│  │   (6.3M)     │    └──────────────────┘    └──────────────────┘ │
│  └──────────────┘              │                      │            │
│                                ▼                      ▼            │
│                    ┌──────────────────┐    ┌──────────────────┐    │
│                    │  Train Model     │    │    Scaler        │    │
│                    │(src/train_model) │───▶│  (StandardScaler)│    │
│                    └──────────────────┘    └──────────────────┘    │
│                                │                                    │
│                                ▼                                    │
│                    ┌──────────────────┐                            │
│                    │ XGBoost Model     │                            │
│                    │ (models/)         │                            │
│                    └──────────────────┘                            │
│                                │                                    │
│                                ▼                                    │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                         API Layer                              ││
│  │  ┌─────────────┐  ┌──────────────┐  ┌───────────────────────┐  ││
│  │  │   FastAPI  │  │  Predictor   │  │   XAI Engine (LLM)   │  ││
│  │  │  (main.py) │  │ (predictor)  │  │    (xai_engine)     │  ││
│  │  └─────────────┘  └──────────────┘  └───────────────────────┘  ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                │                                    │
│                                ▼                                    │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                    Frontend Dashboards                         ││
│  │  ┌────────────────────────┐  ┌───────────────────────────────┐ ││
│  │  │  Analyst Dashboard     │  │   Real-Time Scanner          │ ││
│  │  │  (index.html)          │  │   (realtime.html)            │ ││
│  │  └────────────────────────┘  └───────────────────────────────┘ ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Features

- **XGBoost Fraud Detection**: Trained on PaySim1 dataset with 15 engineered behavioral features
- **Real-Time Scanning**: WebSocket-based live transaction streaming and fraud detection
- **Explainable AI**: LLM-generated explanations for fraud predictions with rule-based fallback
- **REST API**: Full-featured API with single/batch predictions, health checks, and statistics
- **Interactive Dashboards**: Dark-themed web dashboards for analysts and real-time monitoring

## Prerequisites

- Python 3.10+
- PaySim1 dataset (download from [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1))

## Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd finsecure
pip install -r requirements.txt
cp .env.example .env
```

### 2. Download Dataset

Download `paysim.csv` from Kaggle and place it in the `data/` directory.

### 3. Preprocess Data

```bash
python src/preprocess.py
```

This will:
- Load the PaySim1 dataset (6.3M rows)
- Drop leakage columns (nameOrig, nameDest, isFlaggedFraud)
- Remove duplicates and handle missing values
- One-hot encode transaction types
- Save cleaned data to `data/cleaned_transactions.parquet`

### 4. Train Model

```bash
python src/train_model.py
```

This will:
- Apply 15 feature engineering transformations
- Split data chronologically (70/15/15)
- Train XGBoost with early stopping
- Tune threshold for optimal F2-score
- Save model to `models/`

### 5. Run API Server

```bash
cd api
uvicorn main:app --reload
```

Or from project root:

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Open Dashboard

Open `frontend/index.html` in your browser to access the analyst dashboard.

For real-time scanning, open `frontend/realtime.html`.

## API Usage Examples

### Single Transaction Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "step": 1,
    "type": "TRANSFER",
    "amount": 10000.0,
    "oldbalanceOrg": 10000.0,
    "newbalanceOrig": 0.0,
    "oldbalanceDest": 0.0,
    "newbalanceDest": 10000.0
  }'
```

### Batch Prediction

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {
        "step": 1,
        "type": "TRANSFER",
        "amount": 1000.0,
        "oldbalanceOrg": 5000.0,
        "newbalanceOrig": 4000.0,
        "oldbalanceDest": 1000.0,
        "newbalanceDest": 2000.0
      },
      {
        "step": 2,
        "type": "PAYMENT",
        "amount": 100.0,
        "oldbalanceOrg": 1000.0,
        "newbalanceOrig": 900.0,
        "oldbalanceDest": 500.0,
        "newbalanceDest": 600.0
      }
    ]
  }'
```

### Health Check

```bash
curl http://localhost:8000/health
```

### Model Info

```bash
curl http://localhost:8000/model/info
```

## WebSocket Usage Example

### Python WebSocket Client

```python
import asyncio
import websockets
import json

async def scan():
    uri = "ws://localhost:8000/ws/scan"
    async with websockets.connect(uri) as ws:
        for i in range(10):
            message = await ws.recv()
            data = json.loads(message)
            print(f"Transaction: {data['type']} - ₹{data['amount']} - "
                  f"Fraud: {data['is_fraud']} ({data['fraud_probability']:.1%})")

asyncio.run(scan())
```

### Stats WebSocket

```python
import asyncio
import websockets
import json

async def stats():
    uri = "ws://localhost:8000/ws/stats"
    async with websockets.connect(uri) as ws:
        for i in range(5):
            message = await ws.recv()
            data = json.loads(message)
            print(f"Scanned: {data['total_scanned']}, "
                  f"Fraud: {data['total_flagged']}, "
                  f"Rate: {data['fraud_rate']:.2%}")

asyncio.run(stats())
```

## Engineered Features

| Feature | Formula/Logic |
|---------|---------------|
| balance_error_orig | (oldbalanceOrg - amount) - newbalanceOrig |
| balance_error_dest | (oldbalanceDest + amount) - newbalanceDest |
| log_amount | log1p(amount) |
| log_oldbalanceOrg | log1p(oldbalanceOrg) |
| log_newbalanceOrig | log1p(newbalanceOrig) |
| log_oldbalanceDest | log1p(oldbalanceDest) |
| log_newbalanceDest | log1p(newbalanceDest) |
| orig_balance_zero | 1 if oldbalanceOrg == 0 else 0 |
| dest_balance_zero | 1 if oldbalanceDest == 0 else 0 |
| account_drained | 1 if newbalanceOrig == 0 AND oldbalanceOrg > 0 else 0 |
| amount_gt_orig_balance | 1 if amount > oldbalanceOrg else 0 |
| dest_balance_unchanged | 1 if newbalanceDest == oldbalanceDest else 0 |
| orig_to_dest_ratio | amount / (oldbalanceDest + 1), clipped at 100 |
| hour_of_day | step % 24 |
| is_high_risk_hour | 1 if hour between 0-5 else 0 |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/predict` | Single transaction fraud prediction |
| POST | `/predict/batch` | Batch prediction (up to 1000) |
| GET | `/health` | Health check |
| GET | `/model/info` | Model metadata |
| GET | `/stats/summary` | Aggregate statistics |
| GET | `/docs` | Swagger documentation |
| WS | `/ws/scan` | Real-time transaction stream |
| WS | `/ws/stats` | Live statistics broadcast |

## Performance Metrics

Based on PaySim1 test set evaluation:

| Metric | Score |
|--------|-------|
| ROC-AUC | 99.98% |
| Recall | 99.20% |
| Precision | 80.34% |
| F2-Score | 95.67% |
| Latency | <42ms |

## Project Structure

```
finsecure/
├── data/
│   ├── paysim.csv              # Raw PaySim1 dataset
│   └── cleaned_transactions.parquet  # Cleaned data
├── src/
│   ├── preprocess.py           # Data preprocessing
│   ├── feature_engineering.py  # Feature engineering
│   ├── train_model.py          # Model training
│   └── evaluate_model.py       # Model evaluation
├── api/
│   ├── main.py                 # FastAPI application
│   ├── schemas.py              # Pydantic models
│   ├── predictor.py            # ML inference
│   ├── xai_engine.py           # LLM explanations
│   └── realtime_scanner.py     # Real-time scanner
├── frontend/
│   ├── index.html              # Analyst dashboard
│   ├── realtime.html            # Real-time scanner
│   └── styles.css              # Shared styles
├── models/
│   ├── xgboost_model.joblib    # Trained model
│   ├── scaler.joblib           # Feature scaler
│   └── model_metadata.joblib   # Model metadata
├── tests/
│   ├── test_features.py        # Feature tests
│   ├── test_api.py             # API tests
│   └── test_integration.py     # WebSocket tests
├── requirements.txt
├── .env.example
└── README.md
```

## Future Enhancements

- [ ] Add model versioning and A/B testing
- [ ] Implement real-time data ingestion from Kafka
- [ ] Add user authentication and role-based access
- [ ] Deploy to Kubernetes with auto-scaling
- [ ] Add model drift detection
- [ ] Implement feedback loop for model improvement

## License

MIT License

---

Built with ❤️ by FinSecure AI Team"# FINSECURE-AI" 
