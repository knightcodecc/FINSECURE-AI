# FinSecure AI v2 — End-to-End Financial Fraud Detection System

A complete rebuild of FinSecure AI addressing 4 core gaps identified in modern fraud-detection literature — scoped per VTU final-year major project guidelines.

---

## 🚀 Objectives & Architecture

| # | Objective | Literature Gap Addressed | Implementation |
|---|-----------|---------------------------|----------------|
| 1 | Relational Fraud Signals | Tabular-only models miss ring/mule networks | `src/graph_features.py` (10 leakage-safe expanding graph metrics) |
| 2 | SHAP-Grounded Explainability | LLMs without SHAP hallucinate explanations | `src/explain/shap_explainer.py`, `api/xai_engine.py` (TreeExplainer + JSON Faithfulness) |
| 3 | Cost-Sensitive Evaluation | Standard metrics treat all false negatives equally | `src/cost_sensitive_eval.py` (FN cost = transaction amount, FP cost = ₹50 review) |
| 4 | Concept Drift Monitoring | Static models silently decay over time | `src/drift_monitor.py` (PSI/KS drift checks + automated retraining banner) |

---

## 🛠️ Operational Setup & Quickstart Instructions

### 1. Installation & Environment Setup

Ensure Python 3.10+ is installed. Clone the repository and install all required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Model Training & Threshold Calibration

To generate dataset features, train the XGBoost model, perform threshold tuning on **Validation Split Only**, and output test evaluations/ablation reports:

```bash
python src/train_model.py
```
*(Artifacts generated in `models/`: `xgboost_model.joblib`, `scaler.joblib`, `model_metadata.joblib`)*
*(Reports generated in `reports/`: `cost_eval.md`, `cost_comparison.csv`, `ablation_graph.md`, `shap_summary.png`)*

### 3. Launching the Backend REST & WebSocket API Server

Run the FastAPI application from the **project root**:

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```
- Interactive Swagger API Documentation: `http://localhost:8000/docs`
- Health check endpoint: `http://localhost:8000/health`
- WebSocket Real-time Feed: `ws://localhost:8000/ws/feed`

### 4. Opening the Analyst Dashboard

Open `frontend/index.html` directly in any standard web browser or serve via HTTP server:
```bash
# Alternative static server:
python -m http.server 3000 --directory frontend
```
*(Provides real-time transaction scoring, dual-threshold toggle, SHAP feature contribution bar chart, concept drift panel, and RAG Chatbot)*

### 5. Running End-to-End System Demonstration

Run the automated validation script that demonstrates transaction scoring, LLM faithful explanations, RAG log querying, and concept drift triggering:

```bash
python scripts/demo.py
```

### 6. Executing the Test Suite

Run the full pytest suite covering feature column counts, SHAP additivity in margin space, faithfulness validation, cost tuning, drift simulation, and train/serve skew:

```bash
pytest -v
```

---

## 📁 Repository Structure

```
finsecure-ai-v2/
├── data/                       # PaySim dataset & SQLite database (finsecure.db)
├── src/                        # Core ML pipeline
│   ├── graph_features.py       # Expanding-window graph extraction (10 features)
│   ├── preprocess.py           # Preprocessing & train/val/test chronological split
│   ├── feature_engineering.py  # Ported 13 tabular + 10 graph features (33 total)
│   ├── train_model.py          # Model training, validation tuning, test reporting
│   ├── cost_sensitive_eval.py  # Expected cost evaluation & threshold tuning
│   └── drift_monitor.py        # PSI & KS drift monitoring engine
│   └── explain/
│       └── shap_explainer.py   # SHAP TreeExplainer & margin additivity
├── api/                        # FastAPI Serving Layer
│   ├── main.py                 # REST endpoints & WebSocket handler
│   ├── predictor.py            # Real-time predictor & shared prepare_features
│   ├── xai_engine.py           # Structured JSON LLM explanation generator & validator
│   ├── db.py                   # SQLite WAL connection manager & schema migrations
│   ├── alerts.py               # Webhook alert dispatcher
│   └── rag_chatbot.py          # RAG query interface over predictions table
├── frontend/                   # Web Dashboards
│   ├── index.html              # Main Analyst Dashboard & Real-time Scanner
│   ├── style.css               # Styling rules & dynamic SHAP charts
│   └── app.js                  # WebSocket client & REST handler
├── scripts/
│   └── demo.py                 # End-to-end operational demo script
├── tests/                      # Pytest verification suite
│   ├── test_feature_engineering.py
│   ├── test_cost_sensitive_eval.py
│   ├── test_xai.py
│   ├── test_drift_monitor.py
│   └── test_serving.py
├── reports/                    # Generated charts, ablation reports & markdown summaries
├── models/                     # Saved joblib model artifacts
└── docs/                       # Architecture specs & future enhancements doc
```

---

## 🔮 Future Enhancements

The project scope is strictly maintained around the core 4 objectives. Advanced capabilities like dynamic per-segment thresholds and federated multi-institution learning are documented in [docs/FUTURE_ENHANCEMENTS.md](docs/FUTURE_ENHANCEMENTS.md).
