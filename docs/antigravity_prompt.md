# Antigravity build prompt — FinSecure AI v2 (4-feature scope)

Paste the block below into Antigravity (or Claude Code, Cursor, etc. — it's
tool-agnostic) as your first message once you've opened this repo folder.

---

## PROMPT

You are working in an existing repo scaffold called `finsecure-ai-v2`. Every
file under `src/` and `api/` already has a docstring explaining its
objective, why it exists, and a TODO list — read every docstring before
writing any code. Do not restructure the folder layout; fill in the stubs.

**Project**: FinSecure AI v2, a real-time financial fraud detection system
trained on the PaySim1 dataset (Kaggle: `ealaxi/paysim1`, ~6.3M rows, columns:
step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest,
oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud). This is a VTU
final-year major project.

This is a rebuild of an earlier version (v1) that used plain XGBoost on 15
hand-engineered tabular features with a fixed 0.8 decision threshold and
GPT-3.5 paraphrase-only explanations. v2 closes exactly 4 gaps raised
against v1 — the guide capped scope at 4-5 features, so **do not add a 5th
major feature** (adaptive per-segment thresholding and federated learning
are explicitly out of scope — they belong only in the report's "Future
Enhancements" section, not in code):

1. Relational/graph fraud signal (`src/graph_features.py`)
2. SHAP-grounded explainability (`api/xai_engine.py`)
3. Cost-sensitive evaluation (`src/cost_sensitive_eval.py`)
4. Concept drift monitoring (`src/drift_monitor.py`)

Treat every design decision as answerable with "which of these 4 objectives
does this serve" — if a change doesn't serve one of them, it's scope creep,
flag it to the user instead of building it.

**Build order** (follow `README.md`'s build order exactly — don't skip
ahead to `api/` before `src/` is working and tested):

1. **`src/graph_features.py`** is already fully implemented (not a stub) —
   read it first, it defines the leakage-safety pattern (expanding time
   window, no future information) that every other module must respect.
   Verify it runs end-to-end against a real `data/paysim.csv` once the user
   provides one; it currently defaults to a 200K-row sample via
   `SAMPLE_ROWS`.

2. **`src/preprocess.py`**: implement `load_raw()`, `merge_graph_features()`,
   `drop_leakage_columns()`. Critical ordering: graph features must be
   joined onto the raw dataframe (which still has `nameOrig`/`nameDest`)
   BEFORE those ID columns are dropped. Run `graph_features.py` as a
   separate prior step (its own script invocation), then load its parquet
   output here and merge on `(step, nameOrig, nameDest)`.

3. **`src/feature_engineering.py`**: port the 15 v1 tabular features (ask
   the user for the v1 formulas if not present in this repo) unchanged. Fit
   `StandardScaler` on the TRAIN split only — never on validation/test —
   and persist it to `models/scaler.joblib`.

4. **`src/cost_sensitive_eval.py`**: implement `expected_cost()` and
   `threshold_sweep()`. Cost model: false negative cost = transaction amount
   (configurable multiplier for regulatory/reputational weight), false
   positive cost = a fixed analyst-review cost. Produce a plot of expected
   cost vs. threshold and identify the cost-minimizing threshold. This will
   be compared against the F2-optimal threshold from v1 — do not assume
   they're the same value; report both, pick one as "active" via a static
   config flag, not adaptive logic.

5. **`src/train_model.py`**: chronological 70/15/15 split (unchanged
   philosophy from v1). Train XGBoost with early stopping. Run
   `cost_sensitive_eval.threshold_sweep()` on the validation set and save
   BOTH the F2-optimal and cost-optimal thresholds to
   `models/model_metadata.joblib`, tagged by strategy name. Also run an
   ablation: train once on tabular-only features, once on tabular+graph
   features, and log ROC-AUC/precision/recall/F2 for both — write this
   comparison to `docs/ablation_results.md` as a markdown table. This
   ablation is the evidence that graph features are worth the added
   complexity; report the real numbers even if the margin is small.

6. **`src/drift_monitor.py`**: implement `compute_psi()` (standard
   Population Stability Index formula, binned) and the `DriftMonitor`
   class. Reference distribution = training set feature distributions.
   Current distribution = a rolling window pulled from the prediction log
   (reuse `data/finsecure.db` from v1 if present, otherwise create a
   `predictions` table with columns: timestamp, features_json,
   prediction_proba, actual_label_if_known). When PSI exceeds
   `DRIFT_PSI_THRESHOLD` (from `.env`) for any feature or the overall
   prediction distribution, log a `RETRAIN_RECOMMENDED` event — do NOT
   auto-retrain. Surface this via the API, not silently.
   For the demo: also add a small helper/notebook that artificially shifts
   a sample batch's feature distribution (e.g. scale `amount` up) so drift
   can be triggered and shown live in under 2 minutes without needing
   months of real data.

7. **`api/*`**: wire `predictor.py` to load the model + scaler and accept
   the expanded feature set, with a documented cold-start fallback for
   transactions where a counterparty account has no graph history yet
   (use the same neutral defaults `graph_features.py` uses: 0 for
   degree/pagerank/fraud_exposure, 1 for component_size).
   `xai_engine.py`: implement SHAP TreeExplainer on the trained XGBoost
   model, extract top-5 features by |SHAP value| for the flagged
   transaction, and build a prompt that asks the LLM to narrate those
   SPECIFIC features in plain English — do not let the LLM free-associate
   a reason. Keep v1's rule-based fallback for LLM failures.
   Add `GET /drift/status` returning the `DriftMonitor`'s latest check
   result, and extend `GET /model/info` to report both threshold
   candidates. Wire a scheduled job (APScheduler, interval from `.env`) to
   run the drift check periodically. Do NOT build a threshold_manager or
   any per-segment threshold logic — out of scope.

8. **`frontend/`**: extend v1's `index.html` analyst dashboard with a drift
   status panel (poll `/drift/status`) and update the explanation panel to
   show the SHAP top-5 features alongside the LLM narrative, not the
   narrative alone. Show both threshold candidates from `/model/info` as
   plain text, no toggle UI needed.

9. **`tests/`**: fill in every stub, especially the leakage tests in
   `test_graph_features.py` — these prove the no-future-leakage claim
   made throughout the codebase and are the single most important tests
   in this repo for defending the project's methodology in viva.

10. Once the API is running, load-test with Locust at realistic concurrent
    request volume and report p50/p95/p99 latency (not just single-request
    latency) in `docs/load_test_results.md`.

**Constraints while building:**
- Don't introduce new top-level folders or rename existing files.
- Don't build adaptive/per-segment thresholding or federated learning —
  if you find yourself tempted to add either, stop and note it as a
  "Future Enhancements" line in the report instead.
- Every new function needs a docstring explaining WHY, not just what
  (matching the style already in the stubs) — this repo doubles as
  documentation for an academic project review.
- Flag, rather than silently work around, anything that looks like it
  would reintroduce label leakage (e.g., using post-transaction information
  in a feature, or computing graph stats on the full dataset instead of an
  expanding window).
- After each numbered step above, stop and summarize what you built and
  what you verified before moving to the next step — don't build the whole
  repo in one uninterrupted pass.

Start with step 1: confirm you've read `src/graph_features.py` in full and
summarize its leakage-safety design back to me before touching anything
else.
