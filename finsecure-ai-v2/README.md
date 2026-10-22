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

### ⚡ One-Command Instant Launch (Recommended)

Run the full system (FastAPI backend + Interactive Analyst Dashboard) with a single command from anywhere:

```powershell
# Windows PowerShell Local Launcher:
.\scripts\run_local.ps1
```
Or via Python/Batch:
```bash
run.bat
# Or:
python run.py
```
*This automatically verifies dependencies, precomputes PaySim aggregates if missing, starts the server on `$env:PORT` (default 8001), and opens the dashboard in your default browser.*

Other operational commands:
```bash
python scripts/verify_e2e_browser.py   # Runs full Playwright end-to-end browser test with screenshots
pytest -v                              # Runs full pytest test suite (42 tests, 100% pass)
python scripts/build_aggregates.py     # Precomputes streaming big-data aggregates for PaySim CSV
python scripts/export_users.py         # Exports registered analysts from SQLite to Excel (.xlsx)
```

### 1. Installation & Environment Setup

Ensure Python 3.10+ is installed. Clone the repository and install all required dependencies:

```bash
pip install -r requirements.txt
```
*(Dependencies include FastAPI, Uvicorn, XGBoost, SHAP, openpyxl, PyJWT, bcrypt, and Playwright).*

### 2. Authentication & User Database (Excel Viewable)

- **Storage**: SQLite `data/finsecure_users.db` (`users` and `login_events` tables) separate from predictions DB.
- **Security**: Salted bcrypt password hashes (never logged or stored plaintext), 8-hour HS256 signed JWT Bearer tokens, rate limiting (5 failed attempts per 15 minutes per IP/email -> 429).
- **Excel Sync**: Automatically syncs `data/users_export.xlsx` (`Users` and `Login Events` sheets, passwords strictly excluded) on every register/login using openpyxl with lock-resilient atomic write.
- **Default Admin Account**: Defined in `.env` (`ADMIN_EMAIL`, `ADMIN_PASSWORD`). Default: `admin@finsecure.ai` / `AdminSecure2026!`.
- **New Registrations**: Default to `analyst` role. The `Users` management panel is visible exclusively to administrators.

### 3. Big-Data PaySim Precomputed Aggregates (6.36M Rows)

To serve interactive dashboards instantly without memory bloat:
```bash
python scripts/build_aggregates.py
```
- Streams raw CSV in 500,000-row chunks using minimal RAM (~141 MB peak for 6.36M rows).
- Output: `data/aggregates_paysim.json` (6,362,620 rows, 8,213 frauds).
- Verified invariants: sum of type counts == total rows; fraud total == sum of fraud by type; histogram sums == total rows.
- Cached in-memory by FastAPI (`/analytics/*` endpoints).

### 4. Grounded RAG Audit Chatbot

Redesigned deterministic intent router (`api/rag_chatbot.py`) answering 45+ auditor benchmark evaluation queries:
- Prediction logs: counts, flagged vs cleared, by type, by account, highest amount, time ranges.
- Grounded SHAP risk drivers for any transaction ID.
- Model thresholds: cost-optimal vs F2-optimal explanations, financial cost savings.
- Covariate & concept drift status and feature PSI metrics.
- PaySim dataset statistics: fraud rates, volumes, draining patterns.
- Fully operational offline without external API keys; if `GEMINI_API_KEY` or `OPENAI_API_KEY` is present, answers are strictly grounded in tool data.

### 5. Duplicate Transaction Prevention

- `POST /predict` returns server-side database primary key `id`.
- Table rows are keyed strictly by this server `id` in `predictionsMap` (in-place update/merge).
- Exactly 1 `log_prediction` call per inference request.
- Submissions, page reload, and WebSocket reconnects merge seamlessly without duplicating rows.

### 6. Modern Light-Theme UI & Offline Vendoring

- Vendored Chart.js (`frontend/vendor/chart.umd.min.js`) — zero external CDN dependencies for reliable execution offline or behind Cloudflare tunnels.
- Clean light design system (`frontend/style.css`) with WCAG AA compliance, sidebar navigation, animated KPI count-ups, and interactive slide-out drawers.
- Scorer with preset buttons: balanced legitimate payment (0.14% fraud probability) and account-drain fraud (99.98% fraud probability).
- Multi-viewport responsive layouts verified at 1280px (desktop), 768px (tablet), and 390px (mobile).

### 7. Analyze CSV — Batch Transaction Intelligence & XAI Scoring

Interactive batch fraud inference pipeline enabling analysts to upload custom transaction datasets, stream-validate schema & integrity, batch-score with the production XGBoost model + chronological graph features, and download sanitized reports:
- **Streaming Upload & Auto-Mapping**: Multi-format delimiter autodetection (comma, semicolon, tab, pipe) and UTF-8 BOM support. Enforces 9 required PaySim fields with interactive mapping dropdowns and a 10-row preview table.
- **Resilient Row Validation**: Validates step, transaction type (`TRANSFER`, `CASH_OUT`, etc.), non-negative/positive balances and amounts without crashing. Counts rejected rows by category and provides rejected row CSV downloads.
- **Chronological Graph Features & Parity**: Computes 10 expanding-window graph metrics with zero lookahead leakage. Verified to achieve exact bit-level score parity (`max_diff = 0.0 < 1e-5`) with offline training features.
- **Asynchronous Background Processing**: Worker thread execution with live stage tracking (`reading`, `validating`, `features`, `graph`, `scoring`, `aggregates`, `finishing`), progress percentage, dynamic ETA, and real-time job cancellation.
- **Interactive Visualizations & What-If Thresholds**: 7 Chart.js visualizations (Type Mix, Flagged vs Cleared, Flagged by Type, Amount Log-Distribution, Temporal Volume, Fraud Score Histogram, Drained Accounts). Dynamic What-If threshold slider recalibrates flagged volumes, flag rates, confusion matrix, precision/recall/F1/F2, and rupee savings instantly.
- **Grounded On-Demand SHAP**: Clicking any flagged transaction computes exact SHAP TreeExplainer local feature contributions on demand with an interactive horizontal impact chart.
- **Export Sanitization**: Generates `results.csv` and multi-tab `summary.xlsx` (via `openpyxl`) with CSV formula injection escaping (`=`, `+`, `-`, `@` prepended with `'`).
- **Grounded Chatbot Integration**: Read-only chatbot tools answer ad-hoc natural language queries about the user's uploaded analyses (rows, flagged counts, risk drivers, recall, model version).

### 8. Environment Variables & Operational Limits

| Variable | Default | Purpose & Constraints |
|:---|:---:|:---|
| `DATA_DIR` | `"data"` | Root storage directory for SQLite DBs (`finsecure.db`, `finsecure_users.db`, `finsecure_analyses.db`), uploads (`data/uploads`), analysis outputs (`data/analyses`), and Excel exports. Set to isolated temporary paths during tests. |
| `MAX_UPLOAD_MB` | `500` | Maximum file size in MB allowed for CSV uploads before returning HTTP 413. Streamed to disk in 1 MB chunks to prevent RAM bloat. |
| `MAX_SCORE_ROWS` | `300000` | Upper limit of transactions scored in chronological step order. Files exceeding this limit receive full-file streaming EDA aggregates, while scoring is applied to the chronological prefix. |
| `UPLOAD_TTL_DAYS` | `7` | Retention time window in days for uploaded CSVs and analysis artifacts before TTL sweep eligibility. |
| `MAX_CONCURRENT_JOBS_PER_USER` | `1` | Concurrency ceiling for simultaneous background scoring runs per user. |
| `MAX_CONCURRENT_JOBS_TOTAL` | `2` | Global server concurrency ceiling for active background scoring tasks. |
| `PORT` | `8001` | Local FastAPI and frontend static server listening port. |

---

## 📁 Repository Structure

```
finsecure-ai-v2/
├── data/                       # PaySim dataset, SQLite DBs & Excel export
│   ├── finsecure.db            # Predictions and drift snapshots
│   ├── finsecure_users.db      # Users and login events
│   ├── finsecure_analyses.db   # CSV batch analysis metadata & ownership records
│   ├── users_export.xlsx       # Auto-synced Excel view of users (no passwords)
│   ├── aggregates_paysim.json  # Precomputed 6.36M row aggregates
│   ├── uploads/                # Sanitized user uploaded CSV files (git-ignored)
│   └── analyses/               # Per-analysis scored parquet & summary.json (git-ignored)
├── src/                        # Core ML pipeline
│   ├── batch_scoring.py        # Stream validation, graph features, XGBoost scoring & EDA
│   ├── graph_features.py       # Expanding-window graph extraction (10 features)
│   ├── preprocess.py           # Preprocessing & train/val/test chronological split
│   ├── feature_engineering.py  # Ported 13 tabular + 10 graph features (33 total)
│   ├── train_model.py          # Model training, validation tuning, test reporting
│   ├── cost_sensitive_eval.py  # Expected cost evaluation & threshold tuning
│   ├── drift_monitor.py        # PSI & KS drift monitoring engine
│   └── explain/
│       └── shap_explainer.py   # SHAP TreeExplainer & margin additivity
├── api/                        # FastAPI Serving Layer
│   ├── main.py                 # REST endpoints & WebSocket handler
│   ├── analysis.py             # CSV analysis upload, background worker, what-if & export routes
│   ├── analysis_db.py          # SQLite connection manager for batch analyses
│   ├── auth.py                 # JWT Bearer auth, bcrypt, SQLite users, Excel export
│   ├── analytics.py            # Precomputed PaySim big-data analytics endpoints
│   ├── predictor.py            # Real-time predictor & shared prepare_features
│   ├── xai_engine.py           # Structured JSON LLM explanation generator & validator
│   ├── db.py                   # SQLite WAL connection manager & schema migrations
│   ├── alerts.py               # Webhook alert dispatcher
│   └── rag_chatbot.py          # Grounded RAG router over prediction logs & batch analyses
├── frontend/                   # Modern Light-Theme Dashboard
│   ├── index.html              # Main Analyst Dashboard & Analyze CSV 5-Step Stepper
│   ├── login.html              # Authentication sign-in page
│   ├── register.html           # Analyst self-registration page
│   ├── style.css               # WCAG AA light-theme design system
│   ├── app.js                  # Reactive state, WebSocket, batch scoring & Chart.js controllers
│   ├── samples/                # Sample datasets for quick demo testing
│   │   ├── sample_transactions.csv  # 2,000 labeled transactions (with drained accounts & ring mules)
│   │   ├── sample_unlabeled.csv     # 2,000 unlabeled transactions
│   │   └── template.csv             # Empty header-only template for user uploads
│   └── vendor/
│       └── chart.umd.min.js    # Vendored offline Chart.js (205 KB)
├── scripts/
│   ├── run_local.ps1           # Windows PowerShell automated launcher
│   ├── clean_test_data.py      # Interactive utility to purge test accounts/analyses with confirmation
│   ├── verify_e2e_browser.py   # Full Playwright end-to-end browser audit (12 comprehensive steps)
│   ├── build_aggregates.py     # Memory-light streaming PaySim aggregator
│   ├── export_users.py         # Standalone Excel user export tool
│   ├── colab_run.md            # Google Colab cloudflared setup guide
│   └── demo.py                 # System demonstration script
├── tests/                      # Full Pytest Verification Suite
│   ├── conftest.py             # Auth fixtures, test tokens & temp DATA_DIR isolation
│   ├── test_analysis.py        # 17 batch CSV engine, streaming, auth, parity & export tests
│   ├── test_chatbot.py         # Grounded audit chatbot tests (45 general + 12 dynamic analysis tests)
│   ├── test_auth.py            # Registration, login, rate-limiting, Excel, XSS tests
│   ├── test_duplicate_fix.py   # Database row increment & server ID assertions
│   ├── test_feature_engineering.py
│   ├── test_cost_sensitive_eval.py
│   ├── test_xai.py
│   ├── test_drift_monitor.py
│   └── test_serving.py
├── reports/                    # Generated charts, ablation reports & UI screenshots
│   └── ui/                     # Responsive verification screenshots (1280px, 768px, 390px)
├── models/                     # Saved joblib model artifacts
└── docs/                       # Architecture specs & test change log (CHANGES.md)
```

---

## 🔮 Future Enhancements

The project scope is strictly maintained around the core 4 objectives. Advanced capabilities like dynamic per-segment thresholds and federated multi-institution learning are documented in [docs/FUTURE_ENHANCEMENTS.md](docs/FUTURE_ENHANCEMENTS.md).
