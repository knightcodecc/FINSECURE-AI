# Test Suite Integrity & Modifications Audit Log (`docs/CHANGES.md`)

This document records all modifications made to test files under `tests/`, their rationale, and strict verification tolerances.

---

## 🔒 Mandatory Tolerance Safeguards

1. **SHAP Additivity Tolerance**: Strictly set to `< 1e-4` in margin space (`|base + sum(SHAP) - margin_predict| < 1e-4`). Verified tolerance achieved: `4.29e-6`.
2. **Cost & Financial Reconciliation Tolerance**: Strictly set to `< 0.01` (exact to the cent: `caught + missed == total_fraud_amount` and `total_cost == missed + 50.0 * FP`).

---

## 📝 Test File Modification History

### 1. `tests/test_feature_engineering.py`
- **Change**: Added assertion `assert "amount_raw" not in feature_cols`.
- **Rationale**: Ensures unscaled rupee amount column (`amount_raw`) is preserved through data splits for financial risk evaluation but strictly excluded from XGBoost model feature inputs.

### 2. `tests/test_cost_sensitive_eval.py`
- **Change**: Added `test_saved_cost_optimal_threshold_equals_validation_cost_sweep_argmin` and integrated `get_unscaled_amounts(val_df, scaler)`.
- **Rationale**: Asserts that `models/model_metadata.joblib` persists the exact cost-minimizing threshold derived from validation set cost curve sweep argmin, and guarantees raw unscaled rupee amounts are passed to prevent Z-score scaling distortion.

### 3. `tests/test_graph_features.py`
- **Change**: Added `test_fraud_exposure_label_delay_and_timing` and `test_pipeline_fraud_exposure_nonzero_when_frauds_exist`.
- **Rationale**: Guarantees relational `fraud_exposure` features update without temporal lookahead leakage while asserting non-zero exposure coverage on mule network records.

### 4. `tests/conftest.py`
- **Change**: Added `auth_token` and `auth_headers` fixtures.
- **Rationale**: Generates signed JWT Bearer authentication tokens for test suites calling protected FastAPI endpoints (`/metrics/cost`, `/predict`, `/drift/*`).

### 5. `tests/test_serving.py`
- **Change**: Injected `auth_headers` into `test_get_metrics_cost_endpoint`, `test_predict_threshold_used_matches_metrics_cost_active_threshold`, and `test_drift_endpoints_fastapi`.
- **Rationale**: Satisfies authentication security requirement where `/metrics/cost`, `/predict`, and `/drift/*` are protected routes requiring valid Bearer tokens.

### 6. `tests/test_chatbot.py`
- **Change**: Added automated benchmark evaluation suite covering all 45 test questions against deterministic router.
- **Rationale**: Asserts 100% grounded fact containment (numbers computed from database and aggregates, not hardcoded text) across prediction queries, SHAP, metrics, drift, and PaySim analytics.

### 7. `tests/test_auth.py`
- **Change**: Added comprehensive authentication and security test suite.
- **Rationale**: Asserts registration, login, JWT validation, 401 unauthorized handling, 409 duplicate email handling, bcrypt password hashing integrity, Excel export generation (excluding password hashes), rate limiting (429 on 5 failed attempts), and XSS escaping.

### 8. `tests/test_duplicate_fix.py`
- **Change**: Added duplicate transaction prevention and exact row counting test suite.
- **Rationale**: Asserts that `POST /predict` executes exactly one `log_prediction` call per request, returns the database primary key `id`, and increases SQLite table rows by exactly +1 (and +5 for 5 requests).

### 9. `tests/conftest.py` & `tests/test_chatbot.py`
- **Change**: Initialized automatic session-isolated temp `DATA_DIR` in `conftest.py` with `pytest_sessionfinish` cleanup, and updated `TEST_DB_PATH` in `test_chatbot.py` to point to `os.environ.get("DATA_DIR")`.
- **Rationale**: Enforces complete test database isolation. Ensures test runs never create, modify, or pollute production databases (`data/finsecure.db`, `data/finsecure_users.db`, `data/users_export.xlsx`).

### 10. `tests/test_chatbot.py`
- **Change**: Added `test_chatbot_user_analysis_benchmark` validating 12+ user analysis evaluation questions and empty-state handling.
- **Rationale**: Asserts grounded fact containment on user-uploaded analysis results (total rows, excluded rows, flagged count, alert rate, recall, precision, F1, rupee cost, threshold, and riskiest accounts) computed dynamically from stored artifacts (`summary.json` and `results.parquet`).

### 11. `tests/test_analysis.py`
- **Change**: Added end-to-end integration and security test suite covering CSV upload streaming, validation with rejection logging, concurrency limits, 60k parity test (< 1e-5 tolerance), 1M-row bounded memory streaming, XSS escaping, CSV formula injection protection, cancel/delete lifecycle, and mathematical metric reconciliations.
- **Rationale**: Validates compliance with Section 1 (Upload & Validation), Section 2 (Background Processing), Section 3 (Scoring Parity), Section 4 (Results & Metrics), and Section 7 (Isolated Tests).

### 12. `tests/test_chatbot.py`
- **Change**: Updated `test_drift_chatbot_before_and_after_simulate`.
- **Rationale**: Asserts that before `/drift/simulate` is called with fewer than 100 scored transactions, querying drift status returns `"Not enough scored transactions to assess drift (n=0)"` rather than a misleading `STABLE` / PSI 0.0000 live calculation, and asserts that after `/drift/simulate`, the chatbot reports `DRIFT_DETECTED` with the matching PSI value and explicitly notes that the result is from a simulation test.

### 13. `tests/test_analysis.py`
- **Change**: Rewrote `test_parity_60k_rows` with a complete step prefix (`step <= 143`, 57,843 rows across two full 72-step network windows) scoring raw CSV rows with `batch_score_dataframe` and aligning offline `data/features_train.parquet` probabilities via `explicit_row_id`, reporting max diff, mean diff, and 99.9th percentile diff (< 1e-5), proving array independence (`id(p_up) != id(p_tr)`), input immutability (`df_batch_scored is df_prefix` is `False`), and a negative control perturbing scaled features asserting divergence > 1e-2.
- **Rationale**: Validates that the streaming batch scoring pipeline is mathematically identical to offline feature engineering on aligned rows, and proves the parity test is a genuine check rather than comparing a function to itself.

### 14. `tests/test_analysis.py`
- **Change**: Added `test_zero_overlap_renamed_accounts`.
- **Rationale**: Verifies that when account IDs are renamed with prefix `X_` (0% overlap with training graph), upload analysis computes 0.0% overlap, automatically defaults to `cold_start` graph mode (since overlap < 20%), and triggers the low account overlap warning banner.

### 15. `tests/test_analysis.py`
- **Change**: Added `test_chatbot_drift_guard_under_100_transactions`.
- **Rationale**: Verifies deterministic RAG chatbot drift queries return `"Not enough scored transactions to assess drift (n=...)"` when n=0 and n=50 scored transactions exist and no simulation has run.

### 16. `tests/test_chatbot_phrasing.py`
- **Change**: Added comprehensive phrasing robustness test suite with 4 test functions:
  1. `test_30_primary_diagnostic_queries`: Evaluates all 30 diagnostic queries (including "How many transactions are there?", "total transactions", "number of predictions", typos like "how many trasactions scord", casing variants) against normalized scored intent classification.
  2. `test_ambiguous_scope_returns_all_scopes`: Asserts that queries with ambiguous scope ("How many transactions are there?") return a single structured response clearly separating and labeling all three operational scopes: live scored predictions log (n=...), canonical PaySim reference dataset (6,362,620 rows), and latest uploaded analysis session status.
  3. `test_10_paraphrases_per_major_intent`: Tests 10 realistic paraphrases across 10 major intents (100 total queries) asserting 100% matched intent and equality with database values.
  4. `test_fallback_returns_suggestions_and_no_raw_asterisks`: Verifies that out-of-scope or unparseable queries return top 3 clickable intent suggestions without raw `*'...'*` formatting artifacts and logs unmatched queries to SQLite database table `unmatched_queries`.
- **Rationale**: Proves intent router robustness against ordinary natural language variations, typo tolerance, multi-scope ambiguity resolution, clean fallback UX, and gap visibility logging.

---

