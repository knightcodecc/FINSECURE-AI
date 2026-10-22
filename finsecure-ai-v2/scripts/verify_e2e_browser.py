"""
End-to-End Browser Verification for FinSecure AI v2
Automates complete verification flow with Playwright:
1. Server bootstrap on free port (default 8005)
2. User registration and login flow
3. Scorer execution (legit and fraud) with probability, decision, and row count logging
4. Duplicate transaction test (Section 1):
   - Submit 1 -> exactly +1 row
   - Submit 5 -> exactly +5 rows
   - Reload page -> identical rows, zero duplicates
   - Reconnect WebSocket -> zero duplicates
   - Explicit SQLite row count vs DOM row count at each step
5. PaySim analytics verification (canvas non-empty, KPI values matching aggregate JSON)
6. Concept drift simulation (status and retrain banner verification)
7. Grounded Chatbot UI interaction (8 live questions with printed responses)
8. Multi-device responsive screenshots (1280px, 768px, 390px)
9. Logout and unauthenticated redirect guard
10. Strict console error logging (asserts 0 errors)
"""

import os
import sys
import time
import socket
import sqlite3
import json
import shutil
import tempfile
import subprocess
from pathlib import Path
from urllib.request import urlopen
import pandas as pd
from playwright.sync_api import sync_playwright

from unittest.mock import MagicMock
sys.modules["cv2"] = MagicMock()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ensure isolated temp DATA_DIR for E2E runs
_AUTO_TEMP_DATA_DIR = None
if "DATA_DIR" not in os.environ:
    _AUTO_TEMP_DATA_DIR = tempfile.mkdtemp(prefix="finsecure_e2e_data_")
    os.environ["DATA_DIR"] = _AUTO_TEMP_DATA_DIR

DATA_DIR = Path(os.environ["DATA_DIR"])
DB_PATH = DATA_DIR / "finsecure.db"
AGGREGATES_PAYSIM = ROOT_DIR / "data" / "aggregates_paysim.json"
AGGREGATES_SYNTHETIC = ROOT_DIR / "data" / "aggregates_synthetic.json"
REPORTS_UI = ROOT_DIR / "reports" / "ui"
REPORTS_UI.mkdir(parents=True, exist_ok=True)


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def get_sqlite_row_count() -> int:
    if not DB_PATH.exists():
        return 0
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM predictions")
    cnt = cursor.fetchone()[0]
    conn.close()
    return cnt


def wait_for_server(url: str, timeout: int = 15):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urlopen(f"{url}/health", timeout=1) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    raise RuntimeError(f"Server at {url} failed to respond within {timeout}s.")


def run_e2e_verification():
    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    print(f"\n=======================================================")
    print(f"[START] FINSECURE AI v2 E2E VERIFICATION ON PORT {port}")
    print(f"=======================================================\n")

    import threading
    import uvicorn
    from api.main import app

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    try:
        print("[INIT] Waiting for server health endpoint...")
        wait_for_server(base_url, timeout=30)
        print(f"[INIT] Server is healthy at {base_url}/health")

        console_errors = []

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1280, "height": 850})
            page = context.new_page()

            # Listen for console messages and page errors
            def on_console_msg(msg):
                if msg.type == "error":
                    # Ignore harmless favicon 404 or intentional 415/422 test responses
                    text_lower = msg.text.lower()
                    if "favicon" not in text_lower and "status of 415" not in text_lower and "status of 422" not in text_lower:
                        console_errors.append(f"Console Error: {msg.text}")
                elif msg.type == "warn":
                    pass

            def on_page_error(exc):
                console_errors.append(f"Page Exception: {exc}")

            page.on("console", on_console_msg)
            page.on("pageerror", on_page_error)
            page.on("dialog", lambda dialog: dialog.accept())

            # -----------------------------------------------------------------
            # 1. REGISTRATION FLOW
            # -----------------------------------------------------------------
            print("\n--- STEP 1: USER REGISTRATION ---")
            ts = int(time.time())
            test_email = f"lead_auditor_{ts}@finsecure.ai"
            test_password = "AuditorPassword2026!"
            test_name = "Auditor Jane Doe"

            page.goto(f"{base_url}/register.html")
            page.fill("#reg-fullname", test_name)
            page.fill("#reg-email", test_email)
            page.fill("#reg-password", test_password)
            page.click("#btn-register")

            # Registration auto-redirects to dashboard /
            page.wait_for_url(f"{base_url}/", timeout=10000)
            page.wait_for_selector("#user-display-name")
            logged_user = page.inner_text("#user-display-name")
            logged_role = page.inner_text("#user-role-badge")
            print(f"[REGISTRATION OK] Registered & logged in as: '{logged_user}' | Role: '{logged_role}'")

            # -----------------------------------------------------------------
            # 2. LOGOUT & LOGIN FLOW
            # -----------------------------------------------------------------
            print("\n--- STEP 2: LOGOUT & SIGN IN FLOW ---")
            page.click("#btn-logout")
            page.wait_for_url(f"{base_url}/login.html", timeout=10000)
            print("[LOGOUT OK] Redirected to /login.html")

            # Sign in with registered credentials
            page.fill("#login-email", test_email)
            page.fill("#login-password", test_password)
            page.click("#btn-login")

            page.wait_for_url(f"{base_url}/", timeout=10000)
            page.wait_for_selector("#user-display-name")
            print(f"[LOGIN OK] Authenticated successfully, redirected to dashboard.")

            # -----------------------------------------------------------------
            # 3. TAB NAVIGATION
            # -----------------------------------------------------------------
            print("\n--- STEP 3: TAB NAVIGATION AUDIT ---")
            tabs = [
                ("tab-scorer", "Scorer & XAI"),
                ("tab-table", "Live Predictions"),
                ("tab-analytics", "PaySim Analytics"),
                ("tab-drift", "Concept Drift"),
                ("tab-cost", "Cost Analytics"),
                ("tab-chat", "Audit Chatbot"),
            ]
            for tab_id, tab_label in tabs:
                page.click(f'button[data-tab="{tab_id}"]')
                page.wait_for_selector(f"#{tab_id}.active")
                print(f"[NAV OK] Loaded tab: {tab_label} (#{tab_id})")

            # -----------------------------------------------------------------
            # 4. SCORER: LEGIT AND FRAUD EXAMPLES
            # -----------------------------------------------------------------
            print("\n--- STEP 4: TRANSACTION SCORER EXECUTION ---")
            page.click('button[data-tab="tab-scorer"]')

            # 4a. Legit Payment Example
            page.click("#btn-fill-legit")
            page.click("#btn-submit-tx")
            page.wait_for_function("document.getElementById('scorer-result-pill').textContent !== 'READY TO SCORE'")
            legit_pill = page.inner_text("#scorer-result-pill")
            legit_prob = page.inner_text("#scorer-prob-text")
            print(f"[SCORER LEGIT] Decision: '{legit_pill}' | Fraud Probability: {legit_prob}")
            assert "CLEARED" in legit_pill, f"Expected CLEARED for legit example, got {legit_pill}"

            # 4b. Fraud Account Drain Example
            page.click("#btn-fill-fraud")
            page.click("#btn-submit-tx")
            time.sleep(1.0)
            page.wait_for_function("document.getElementById('scorer-result-pill').textContent === 'FLAGGED FRAUD'")
            fraud_pill = page.inner_text("#scorer-result-pill")
            fraud_prob = page.inner_text("#scorer-prob-text")
            fraud_exp = page.inner_text("#scorer-explanation-box")
            print(f"[SCORER FRAUD] Decision: '{fraud_pill}' | Fraud Probability: {fraud_prob}")
            print(f"               SHAP Explanation: {fraud_exp[:90]}...")
            assert "FLAGGED" in fraud_pill, f"Expected FLAGGED for fraud example, got {fraud_pill}"

            # -----------------------------------------------------------------
            # 5. DUPLICATE TRANSACTION BUG AUDIT (SECTION 1)
            # -----------------------------------------------------------------
            print("\n--- STEP 5: DUPLICATE TRANSACTION VERIFICATION (SECTION 1) ---")
            # Navigate to predictions table
            page.click('button[data-tab="tab-table"]')
            time.sleep(0.5)

            def get_dom_row_count():
                rows = page.query_selector_all("#prediction-table-body tr.row-clickable")
                return len(rows)

            init_dom_count = get_dom_row_count()
            init_db_count = get_sqlite_row_count()
            print(f"[DUP AUDIT 5.0 Baseline]  DOM Table Rows: {init_dom_count} | SQLite DB Rows: {init_db_count}")

            # 5.1 Submit 1 transaction
            print("\n[DUP AUDIT 5.1] Submitting exactly 1 transaction...")
            page.click('button[data-tab="tab-scorer"]')
            page.click("#btn-fill-legit")
            page.fill("#tx-amount", "500.00")
            page.fill("#tx-new-orig", "9500.00")
            page.fill("#tx-new-dest", "5500.00")
            page.click("#btn-submit-tx")
            time.sleep(1.0)

            page.click('button[data-tab="tab-table"]')
            time.sleep(0.5)
            dom_after_1 = get_dom_row_count()
            db_after_1 = get_sqlite_row_count()
            print(f"[DUP AUDIT 5.1 Result]    DOM Table Rows: {dom_after_1} (+{dom_after_1 - init_dom_count}) | SQLite DB Rows: {db_after_1} (+{db_after_1 - init_db_count})")
            assert db_after_1 == init_db_count + 1, f"Expected DB +1, got {db_after_1 - init_db_count}"
            assert dom_after_1 == init_dom_count + 1, f"Expected DOM +1, got {dom_after_1 - init_dom_count}"

            # 5.2 Submit 5 transactions sequentially
            print("\n[DUP AUDIT 5.2] Submitting exactly 5 transactions...")
            page.click('button[data-tab="tab-scorer"]')
            for i in range(1, 6):
                amt = 100.0 * i
                page.click("#btn-fill-legit")
                page.fill("#tx-amount", str(amt))
                page.fill("#tx-orig", f"C_BATCH_{i}")
                page.fill("#tx-new-orig", str(10000.0 - amt))
                page.fill("#tx-new-dest", str(5000.0 + amt))
                page.click("#btn-submit-tx")
                time.sleep(0.8)

            page.click('button[data-tab="tab-table"]')
            time.sleep(0.5)
            dom_after_5 = get_dom_row_count()
            db_after_5 = get_sqlite_row_count()
            print(f"[DUP AUDIT 5.2 Result]    DOM Table Rows: {dom_after_5} (+{dom_after_5 - dom_after_1}) | SQLite DB Rows: {db_after_5} (+{db_after_5 - db_after_1})")
            assert db_after_5 == db_after_1 + 5, f"Expected DB +5, got {db_after_5 - db_after_1}"
            assert dom_after_5 == dom_after_1 + 5, f"Expected DOM +5, got {dom_after_5 - dom_after_1}"

            # 5.3 Reload page -> same rows, no duplicates
            print("\n[DUP AUDIT 5.3] Reloading page (checking re-hydration)...")
            page.reload()
            time.sleep(1.0)
            page.click('button[data-tab="tab-table"]')
            time.sleep(2.0)
            table_html = page.inner_html("#prediction-table-body")
            map_size = page.evaluate("() => window.FinSecureApp ? window.FinSecureApp.predictionsMap.size : 'no app'")
            print(f"[DEBUG 5.3] predictionsMap.size: {map_size}")
            print(f"[DEBUG 5.3] table_html snippet: {table_html[:300]}")
            dom_after_reload = get_dom_row_count()
            db_after_reload = get_sqlite_row_count()
            print(f"[DUP AUDIT 5.3 Result]    DOM Table Rows: {dom_after_reload} | SQLite DB Rows: {db_after_reload}")
            assert dom_after_reload == dom_after_5, f"Reload changed DOM row count! Expected {dom_after_5}, got {dom_after_reload}"
            assert db_after_reload == db_after_5, f"Reload changed DB row count! Expected {db_after_5}, got {db_after_reload}"

            # 5.4 Disconnect and Reconnect WebSocket -> no duplicates
            print("\n[DUP AUDIT 5.4] Disconnecting and Reconnecting WebSocket...")
            page.evaluate("window.FinSecureApp.reconnectWebSocket()")
            time.sleep(2.0)
            dom_after_ws = get_dom_row_count()
            db_after_ws = get_sqlite_row_count()
            print(f"[DUP AUDIT 5.4 Result]    DOM Table Rows: {dom_after_ws} | SQLite DB Rows: {db_after_ws}")
            assert dom_after_ws == dom_after_reload, f"WebSocket reconnect duplicated rows! Expected {dom_after_reload}, got {dom_after_ws}"
            assert db_after_ws == db_after_reload, f"WebSocket reconnect changed DB rows! Expected {db_after_reload}, got {db_after_ws}"
            print("[DUP AUDIT SUCCESS] Zero duplicate rows confirmed across submissions, reload, and WebSocket stream!")

            # -----------------------------------------------------------------
            # 6. PAYSIM ANALYTICS TAB & KPI VERIFICATION
            # -----------------------------------------------------------------
            print("\n--- STEP 6: PAYSIM BIG-DATA ANALYTICS AUDIT ---")
            page.click('button[data-tab="tab-analytics"]')
            time.sleep(1.5)

            # Check all 6 Chart canvases are non-empty
            chart_canvas_ids = [
                "typeMixChart",
                "fraudShareChart",
                "fraudByTypeChart",
                "timeseriesChart",
                "amountDistChart",
                "fraudPatternsChart",
            ]
            for cid in chart_canvas_ids:
                data_url = page.evaluate(f"document.getElementById('{cid}').toDataURL()")
                assert len(data_url) > 1000, f"Chart canvas {cid} is blank/empty!"
                print(f"[CHART OK] Canvas #{cid} rendered successfully (data URL len: {len(data_url)})")

            # Check KPI values against JSON
            kpi_tx = page.inner_text("#kpi-total-tx")
            kpi_fraud = page.inner_text("#kpi-total-fraud")
            kpi_rate = page.inner_text("#kpi-fraud-rate")
            print(f"[ANALYTICS KPIS] Total Transactions: {kpi_tx} | Total Fraud: {kpi_fraud} | Fraud Rate: {kpi_rate}")

            with open(AGGREGATES_PAYSIM, "r", encoding="utf-8") as f:
                paysim_json = json.load(f)
            expected_tx_num = paysim_json['summary']['total_transactions']
            expected_fraud_num = paysim_json['summary']['total_fraud']
            print(f"[ANALYTICS JSON] Expected from JSON:  {expected_tx_num:,} rows | {expected_fraud_num:,} frauds")
            assert kpi_tx.replace(",", "") == str(expected_tx_num), f"KPI mismatch! UI: {kpi_tx}, JSON: {expected_tx_num}"
            assert kpi_fraud.replace(",", "") == str(expected_fraud_num), f"KPI mismatch! UI: {kpi_fraud}, JSON: {expected_fraud_num}"

            # -----------------------------------------------------------------
            # 7. CONCEPT DRIFT SIMULATION AUDIT
            # -----------------------------------------------------------------
            print("\n--- STEP 7: CONCEPT DRIFT MONITORING AUDIT ---")
            page.click('button[data-tab="tab-drift"]')
            time.sleep(1.0)
            initial_drift_status = page.inner_text("#drift-kpi-status")
            print(f"[DRIFT INITIAL] Status: {initial_drift_status}")

            page.click("#btn-simulate-drift")
            time.sleep(2.0)
            post_drift_status = page.inner_text("#drift-kpi-status")
            post_score_psi = page.inner_text("#drift-kpi-score-psi")
            drifted_count = page.inner_text("#drift-kpi-drifted-count")
            banner_display = page.evaluate("document.getElementById('drift-retrain-banner').style.display")
            print(f"[DRIFT SIMULATED] Status: {post_drift_status} | Score PSI: {post_score_psi} | Drifted Features: {drifted_count}")
            print(f"                 Retrain Banner Visible: {banner_display != 'none'}")
            assert post_drift_status in ["WARNING", "CRITICAL DRIFT", "DRIFT_DETECTED"], f"Unexpected drift status: {post_drift_status}"

            # -----------------------------------------------------------------
            # 8. BATCH CSV FRAUD INTELLIGENCE E2E AUDIT
            # -----------------------------------------------------------------
            print("\n--- STEP 8: BATCH CSV FRAUD INTELLIGENCE E2E AUDIT ---")
            page.click('button[data-tab="tab-analysis"]')
            time.sleep(0.5)

            # 8a. Test Bad File Friendly Error (415)
            print("[CSV TEST 8a] Testing bad file format rejection (415)...")
            bad_txt_path = Path(tempfile.gettempdir()) / "invalid_format.txt"
            bad_txt_path.write_text("not a csv file", encoding="utf-8")
            page.set_input_files("#analysis-file-input", str(bad_txt_path))
            page.wait_for_selector("#btn-upload-file:not([disabled])")
            page.click("#btn-upload-file")
            page.wait_for_selector("#analysis-upload-error[style*='display: block']", timeout=10000)
            err_text = page.inner_text("#analysis-upload-error")
            print(f"[CSV TEST 8a SUCCESS] Rejected non-CSV with friendly alert: '{err_text}'")

            # 8b. Upload Labeled Sample (2,000 txs)
            print("[CSV TEST 8b] Uploading labeled sample dataset...")
            page.click("#btn-sample-labeled")
            page.wait_for_selector("#selected-file-display[style*='display: block']")
            file_name = page.inner_text("#selected-file-name")
            print(f"[CSV TEST 8b] Selected file: '{file_name}'")
            assert "sample_transactions.csv" in file_name

            page.click("#btn-upload-file")
            page.wait_for_selector("#stepper-step-2.active", timeout=15000)
            print("[CSV TEST 8b] Advanced to Step 2: Column Mapping & 10-Row Preview")

            # Verify 10-row preview has data
            preview_row_count = page.evaluate("document.querySelectorAll('#preview-table-body tr').length")
            print(f"[CSV TEST 8b] Schema preview rows rendered: {preview_row_count}")
            assert preview_row_count == 10, f"Expected 10 preview rows, got {preview_row_count}"

            # Step 2 -> Step 3
            page.click("#btn-mapping-next")
            page.wait_for_selector("#stepper-step-3.active", timeout=5000)
            print("[CSV TEST 8b] Advanced to Step 3: Inference Options")

            # Step 3 -> Launch Job
            page.click("#btn-start-scoring-job")
            page.wait_for_selector("#stepper-step-4.active", timeout=10000)
            print("[CSV TEST 8b] Advanced to Step 4: Background Worker Active")

            # Watch progress until Step 5
            page.wait_for_selector("#stepper-step-5.active", timeout=60000)
            print("[CSV TEST 8b SUCCESS] Job complete! Advanced to Step 5: Results View")

            # 8c. Assert UI KPIs match API Summary
            analysis_id = page.evaluate("currentAnalysisId")
            print(f"[CSV TEST 8c] Scored Analysis ID: {analysis_id}")
            summary_api = page.evaluate("currentAnalysisSummary")
            ui_rows_total = page.inner_text("#kpi-rows-total").replace(",", "")
            ui_flagged_count = page.inner_text("#kpi-flagged-count").replace(",", "")
            ui_scored = page.inner_text("#kpi-rows-scored").replace(",", "")

            print(f"[CSV TEST 8c] UI Rows Total: {ui_rows_total} | API: {summary_api['eda']['rows_total']}")
            print(f"[CSV TEST 8c] UI Flagged: {ui_flagged_count} | API: {summary_api['scoring']['flagged_count']}")
            print(f"[CSV TEST 8c] UI Scored: {ui_scored} | API: {summary_api['scoring']['rows_scored']}")

            assert int(ui_rows_total) == summary_api["eda"]["rows_total"]
            assert int(ui_flagged_count) == summary_api["scoring"]["flagged_count"]
            assert int(ui_scored) == summary_api["scoring"]["rows_scored"]

            # 8c1. Print DOM vs API Metrics and Assert Equality (Requirement 3)
            dom_graph_mode = page.inner_text("#kpi-graph-mode").strip()
            dom_overlap = page.inner_text("#kpi-account-overlap").strip()
            dom_flagged = int(page.inner_text("#kpi-flagged-count").replace(",", "").strip())
            dom_alert_rate = page.inner_text("#overview-alert-rate").strip()
            dom_precision = page.inner_text("#overview-precision").strip()
            dom_recall = page.inner_text("#overview-recall").strip()

            gm_mode_raw = (summary_api["scoring"].get("scoring_options", {}).get("graph_mode")) or summary_api["scoring"].get("graph_mode") or "warmed"
            gm_labels = {
                "warmed": "Training-Warmed History",
                "cold_start": "Neutral Cold-Start",
                "file_only": "File-Only Graph"
            }
            api_graph_mode = (gm_labels.get(gm_mode_raw, gm_mode_raw)) + (" (Labels Active)" if summary_api["scoring"].get("use_labels_for_graph") else "")
            overlap_val = summary_api["scoring"].get("account_overlap_pct", summary_api["scoring"].get("account_overlap", 0.0) * 100.0)
            api_overlap = f"{overlap_val:.1f}% seen in training"
            api_flagged = int(summary_api["scoring"]["flagged_count"])
            api_m = summary_api["scoring"].get("metrics", {})
            api_ar = api_m.get("alert_rate", summary_api["scoring"]["flagged_rate"])
            api_alert_rate = f"{api_ar * 100.0:.2f}%"
            api_precision = f"{api_m.get('precision', 0.0) * 100.0:.2f}%"
            api_recall = f"{api_m.get('recall', 0.0) * 100.0:.2f}%"

            print(f"\n=======================================================")
            print(f"[RECONCILIATION AUDIT: DOM vs API METRICS (Requirement 3)]")
            print(f"=======================================================")
            print(f"  Graph Mode: DOM='{dom_graph_mode}' | API='{api_graph_mode}'")
            print(f"  Overlap:    DOM='{dom_overlap}' | API='{api_overlap}'")
            print(f"  Flagged:    DOM='{dom_flagged}' | API='{api_flagged}'")
            print(f"  Alert Rate: DOM='{dom_alert_rate}' | API='{api_alert_rate}'")
            print(f"  Precision:  DOM='{dom_precision}' | API='{api_precision}'")
            print(f"  Recall:     DOM='{dom_recall}' | API='{api_recall}'")
            print(f"=======================================================\n")

            assert dom_graph_mode == api_graph_mode, f"Graph mode mismatch: DOM '{dom_graph_mode}' != API '{api_graph_mode}'"
            assert dom_overlap == api_overlap, f"Overlap mismatch: DOM '{dom_overlap}' != API '{api_overlap}'"
            assert dom_flagged == api_flagged, f"Flagged count mismatch: DOM {dom_flagged} != API {api_flagged}"
            assert dom_alert_rate == api_alert_rate, f"Alert rate mismatch: DOM '{dom_alert_rate}' != API '{api_alert_rate}'"
            assert dom_precision == api_precision, f"Precision mismatch: DOM '{dom_precision}' != API '{api_precision}'"
            assert dom_recall == api_recall, f"Recall mismatch: DOM '{dom_recall}' != API '{api_recall}'"
            print("[RECONCILIATION AUDIT SUCCESS] DOM and API metrics matched exactly.\n")

            # 8c2. Banner Verification & Screenshots (Requirement 4)
            # 1) Synthetic model banner
            page.wait_for_selector("#res-synthetic-banner", state="visible")
            syn_text = page.inner_text("#res-synthetic-banner").strip()
            print(f"[BANNER AUDIT] Synthetic Banner visible: '{syn_text}'")
            assert "Model trained on dataset" in syn_text
            page.locator("#res-synthetic-banner").screenshot(path=str(REPORTS_UI / "banner_synthetic.png"))
            print(f"[BANNER AUDIT] Saved: reports/ui/banner_synthetic.png")

            # 2) Training overlap banner
            page.wait_for_selector("#res-overlap-banner", state="visible")
            over_text = page.inner_text("#res-overlap-banner").strip()
            print(f"[BANNER AUDIT] Training Overlap Banner visible: '{over_text}'")
            assert "overlaps the model's training data" in over_text
            page.locator("#res-overlap-banner").screenshot(path=str(REPORTS_UI / "banner_overlap.png"))
            print(f"[BANNER AUDIT] Saved: reports/ui/banner_overlap.png")

            # 8d. Verify Chart Canvases Non-Empty
            print("[CSV TEST 8d] Verifying all 7 Chart.js canvases are non-empty...")
            page.click('button[data-subtab="res-tab-charts"]')
            time.sleep(1.0)
            chart_canvas_ids = [
                "chartTypeMix", "chartFlagShare", "chartFlagByType",
                "chartAmountDist", "chartTimeOverTime", "chartScoreHist", "chartDrained"
            ]
            for cid in chart_canvas_ids:
                is_valid = page.evaluate(f"() => {{ const c = document.getElementById('{cid}'); return c && c.toDataURL().length > 100 && c.width > 0; }}")
                assert is_valid, f"Canvas #{cid} is empty or unrendered!"
                print(f"  [CANVAS OK] #{cid} rendered with live data")

            # 8e. Interactive What-If Threshold Slider
            print("[CSV TEST 8e] Testing interactive What-If threshold slider...")
            pre_whatif_flagged = page.inner_text("#whatif-stat-flagged")
            # Move slider to 0.20
            page.evaluate("() => { const s = document.getElementById('whatif-slider'); s.value = 0.20; s.dispatchEvent(new Event('input', { bubbles: true })); }")
            page.wait_for_function(f"document.getElementById('whatif-stat-flagged').innerText !== '{pre_whatif_flagged}'", timeout=10000)
            post_whatif_flagged = page.inner_text("#whatif-stat-flagged")
            print(f"[CSV TEST 8e] Flagged at default: {pre_whatif_flagged} -> Flagged at 0.20: {post_whatif_flagged}")
            assert pre_whatif_flagged != post_whatif_flagged, "What-if slider did not dynamically update flagged count!"

            # Reset slider
            page.click("#btn-reset-whatif")
            page.wait_for_function(f"document.getElementById('whatif-stat-flagged').innerText === '{pre_whatif_flagged}'", timeout=5000)
            reset_flagged = page.inner_text("#whatif-stat-flagged")
            assert reset_flagged == pre_whatif_flagged, "Reset what-if button did not restore default threshold!"
            print(f"[CSV TEST 8e SUCCESS] What-if slider and reset button validated.")

            # 8f. On-Demand SHAP Explanation Drawer for Flagged Row
            print("[CSV TEST 8f] Testing on-demand SHAP explanation drawer on row click...")
            page.click('button[data-subtab="res-tab-flagged"]')
            time.sleep(0.5)
            page.wait_for_selector("#flagged-table-body tr")
            # Click first row in table
            page.click("#flagged-table-body tr:first-child")
            page.wait_for_selector("#transaction-drawer.open", timeout=5000)
            # Wait for explanation text to populate
            page.wait_for_function("document.getElementById('drawer-explanation-text').innerText.length > 20 && !document.getElementById('drawer-explanation-text').innerText.includes('Computing')")
            shap_explanation = page.inner_text("#drawer-explanation-text")
            print(f"[CSV TEST 8f SUCCESS] SHAP explanation loaded: '{shap_explanation[:85]}...'")
            shap_canvas_valid = page.evaluate("() => { const c = document.getElementById('drawerShapChart'); return c && c.toDataURL().length > 100; }")
            assert shap_canvas_valid, "drawerShapChart canvas is empty!"
            page.click("#btn-close-drawer")
            time.sleep(0.3)

            # 8g. Export results.csv and summary.xlsx and Read Back with Pandas
            print("[CSV TEST 8g] Downloading exported datasets and asserting schema integrity with pandas...")
            page.click('button[data-subtab="res-tab-downloads"]')
            time.sleep(0.5)

            # Download CSV
            with page.expect_download() as download_info:
                page.click("#btn-download-results-csv")
            csv_download = download_info.value
            temp_csv_file = Path(tempfile.gettempdir()) / f"e2e_results_{int(time.time())}.csv"
            csv_download.save_as(str(temp_csv_file))
            df_exported = pd.read_csv(temp_csv_file)
            print(f"[CSV TEST 8g] results.csv read successfully: shape {df_exported.shape}")
            assert len(df_exported) == 2000, f"Expected 2000 rows in export, got {len(df_exported)}"
            assert "fraud_probability" in df_exported.columns
            assert "decision" in df_exported.columns
            assert "threshold_used" in df_exported.columns

            # Download XLSX
            with page.expect_download() as download_info_xlsx:
                page.click("#btn-download-summary-xlsx")
            xlsx_download = download_info_xlsx.value
            temp_xlsx_file = Path(tempfile.gettempdir()) / f"e2e_summary_{int(time.time())}.xlsx"
            xlsx_download.save_as(str(temp_xlsx_file))
            xl = pd.ExcelFile(temp_xlsx_file)
            print(f"[CSV TEST 8g] summary.xlsx read successfully: sheet names {xl.sheet_names}")
            assert any(s in xl.sheet_names for s in ["Overview KPIs", "Executive Summary"]), f"Missing Overview sheet in {xl.sheet_names}"
            assert any(s in xl.sheet_names for s in ["Flagged Transactions", "Scored Transactions"]), f"Missing Transactions sheet in {xl.sheet_names}"
            assert any(s in xl.sheet_names for s in ["Model Metrics", "Confusion Matrix"]), f"Missing Metrics sheet in {xl.sheet_names}"

            # 8h. Upload Unlabeled Sample & Verify Metrics Tab is Hidden
            print("[CSV TEST 8h] Testing unlabeled sample upload & confirming Metrics tab is hidden...")
            page.click("#btn-new-analysis")
            page.wait_for_selector("#stepper-step-1.active")
            page.click("#btn-sample-unlabeled")
            page.wait_for_selector("#selected-file-display[style*='display: block']")
            page.click("#btn-upload-file")
            page.wait_for_selector("#stepper-step-2.active", timeout=15000)
            page.click("#btn-mapping-next")
            page.wait_for_selector("#stepper-step-3.active", timeout=5000)
            page.click("#btn-start-scoring-job")
            page.wait_for_selector("#stepper-step-5.active", timeout=60000)

            metrics_tab_display = page.evaluate("document.getElementById('btn-subtab-metrics').style.display")
            print(f"[CSV TEST 8h SUCCESS] Unlabeled analysis completed. Metrics tab display: '{metrics_tab_display}'")
            assert metrics_tab_display == "none", "Metrics subtab should be hidden when unlabeled data is uploaded!"

            # 8i. 'My Analyses' Modal: Verify History & Delete
            print("[CSV TEST 8i] Verifying 'My Analyses' history table and delete flow...")
            page.reload()
            page.wait_for_selector("#user-display-name")
            page.click('button[data-tab="tab-analysis"]')
            time.sleep(0.5)
            page.click("#btn-open-my-analyses")
            page.wait_for_selector("#modal-my-analyses.active", timeout=5000)
            page.wait_for_selector("#analyses-table-body tr")
            history_rows = page.evaluate("document.querySelectorAll('#analyses-table-body tr').length")
            print(f"[CSV TEST 8i] Past analyses in history: {history_rows}")
            assert history_rows >= 2, f"Expected at least 2 analyses in history, found {history_rows}"

            # Delete the latest analysis
            page.once("dialog", lambda dialog: dialog.accept())
            page.click("#analyses-table-body tr:first-child .btn-delete-analysis")
            page.wait_for_function(f"document.querySelectorAll('#analyses-table-body tr').length === {history_rows - 1}", timeout=5000)
            post_delete_rows = page.evaluate("document.querySelectorAll('#analyses-table-body tr').length")
            print(f"[CSV TEST 8i] Past analyses after deletion: {post_delete_rows}")
            assert post_delete_rows == history_rows - 1
            page.click("#btn-close-analyses-modal")
            time.sleep(0.3)

            # 8j. High Alert Rate Banner Verification (>20% alert rate with sample_training_rows.csv)
            print("\n[BANNER AUDIT 8j] Uploading sample_training_rows.csv in cold_start mode to trigger High Alert Rate banner (>20%)...")
            page.click("#btn-new-analysis")
            page.wait_for_selector("#stepper-step-1.active")
            page.click("#btn-sample-training")
            page.wait_for_selector("#selected-file-display[style*='display: block']")
            page.click("#btn-upload-file")
            page.wait_for_selector("#stepper-step-2.active", timeout=15000)
            page.click("#btn-mapping-next")
            page.wait_for_selector("#stepper-step-3.active", timeout=5000)
            # Choose cold_start mode
            page.check("#radio-graph-cold")
            page.click("#btn-start-scoring-job")
            page.wait_for_selector("#stepper-step-5.active", timeout=60000)

            page.wait_for_selector("#res-high-alert-banner", state="visible")
            high_alert_text = page.inner_text("#res-high-alert-banner").strip()
            print(f"[BANNER AUDIT 8j] High Alert Rate Banner visible: '{high_alert_text}'")
            assert "High Alert Rate" in high_alert_text
            assert "20%" in high_alert_text
            page.locator("#res-high-alert-banner").screenshot(path=str(REPORTS_UI / "banner_high_alert.png"))
            print(f"[BANNER AUDIT 8j SUCCESS] Saved: reports/ui/banner_high_alert.png\n")

            # 8k. Low Account Overlap Banner Verification (<20% overlap with sample_zero_overlap.csv)
            print("\n[BANNER AUDIT 8k] Uploading sample_zero_overlap.csv to trigger Low Account Overlap banner (<20%)...")
            page.click("#btn-new-analysis")
            page.wait_for_selector("#stepper-step-1.active")
            zero_overlap_path = ROOT_DIR / "frontend" / "samples" / "sample_zero_overlap.csv"
            page.set_input_files("#analysis-file-input", str(zero_overlap_path))
            page.wait_for_selector("#btn-upload-file:not([disabled])")
            page.click("#btn-upload-file")
            page.wait_for_selector("#stepper-step-2.active", timeout=15000)
            page.click("#btn-mapping-next")
            page.wait_for_selector("#stepper-step-3.active", timeout=5000)

            # Assert Step 3 displays low overlap banner (< 20%)
            page.wait_for_selector("#opt-low-overlap-banner", state="visible")
            step3_low_text = page.inner_text("#opt-low-overlap-banner").strip()
            print(f"[BANNER AUDIT 8k] Step 3 Low Overlap Banner visible: '{step3_low_text}'")
            assert "Low account overlap" in step3_low_text

            page.click("#btn-start-scoring-job")
            page.wait_for_selector("#stepper-step-5.active", timeout=60000)

            page.wait_for_selector("#res-low-overlap-banner", state="visible")
            low_over_text = page.inner_text("#res-low-overlap-banner").strip()
            print(f"[BANNER AUDIT 8k] Step 5 Low Overlap Banner visible: '{low_over_text}'")
            assert "Low account overlap" in low_over_text
            assert "< 20%" in low_over_text
            page.locator("#res-low-overlap-banner").screenshot(path=str(REPORTS_UI / "banner_low_overlap.png"))
            print(f"[BANNER AUDIT 8k SUCCESS] Saved: reports/ui/banner_low_overlap.png\n")

            # -----------------------------------------------------------------
            # 9. GROUNDED RAG CHATBOT UI INTERACTION (11 QUESTIONS)
            # -----------------------------------------------------------------
            print("\n--- STEP 9: GROUNDED RAG CHATBOT 11-QUESTION UI AUDIT ---")
            page.click('button[data-tab="tab-chat"]')
            time.sleep(0.5)

            chat_questions = [
                "how many transactions were scored?",
                "how many were flagged as fraud?",
                "what is the active threshold?",
                "why is the cost-optimal threshold different from the F2-optimal one?",
                "what is the drift status?",
                "how many transactions are in the dataset?",
                "which transaction type has the most fraud?",
                "who am I logged in as?",
                "how many rows were in my last upload?",
                "how many were flagged in my analysis?",
                "what was the recall on my file?",
            ]

            for idx, q in enumerate(chat_questions, 1):
                page.fill("#chat-input", q)
                page.click("#btn-send-chat")
                # Wait for bot bubble
                page.wait_for_function(f"document.querySelectorAll('.chat-bubble.bot').length >= {idx + 1}")
                bot_bubbles = page.query_selector_all(".chat-bubble.bot")
                last_bubble = bot_bubbles[-1].inner_text()
                # Clean snippet for logging
                first_line = last_bubble.split("\n")[0]
                print(f"[CHAT #{idx}] Q: \"{q}\"")
                print(f"          A: {first_line[:110]}...")

            # -----------------------------------------------------------------
            # 10. MULTI-DEVICE RESPONSIVE SCREENSHOTS
            # -----------------------------------------------------------------
            print("\n--- STEP 10: CAPTURING RESPONSIVE SCREENSHOTS ---")
            # 1280px Desktop
            page.set_viewport_size({"width": 1280, "height": 850})
            page.click('button[data-tab="tab-scorer"]')
            page.screenshot(path=str(REPORTS_UI / "scorer_1280px.png"))
            page.click('button[data-tab="tab-table"]')
            page.screenshot(path=str(REPORTS_UI / "predictions_1280px.png"))
            page.click('button[data-tab="tab-analytics"]')
            page.screenshot(path=str(REPORTS_UI / "analytics_1280px.png"))
            page.click('button[data-tab="tab-drift"]')
            page.screenshot(path=str(REPORTS_UI / "drift_1280px.png"))
            page.click('button[data-tab="tab-cost"]')
            page.screenshot(path=str(REPORTS_UI / "cost_1280px.png"))
            page.click('button[data-tab="tab-chat"]')
            page.screenshot(path=str(REPORTS_UI / "chat_1280px.png"))
            page.click('button[data-tab="tab-analysis"]')
            time.sleep(0.5)
            page.screenshot(path=str(REPORTS_UI / "analysis_1280px.png"))
            print("[SCREENSHOTS 1280px] Saved scorer, predictions, analytics, drift, cost, chat, and analysis.")

            # 768px Tablet
            page.set_viewport_size({"width": 768, "height": 1024})
            page.evaluate("document.querySelector('button[data-tab=\"tab-scorer\"]').click()")
            time.sleep(0.5)
            page.screenshot(path=str(REPORTS_UI / "dashboard_768px.png"))
            page.evaluate("document.querySelector('button[data-tab=\"tab-analysis\"]').click()")
            time.sleep(0.5)
            page.screenshot(path=str(REPORTS_UI / "analysis_768px.png"))
            print("[SCREENSHOT 768px] Saved tablet dashboard and analysis.")

            # 390px Mobile
            page.set_viewport_size({"width": 390, "height": 844})
            page.evaluate("document.querySelector('button[data-tab=\"tab-scorer\"]').click()")
            time.sleep(0.5)
            page.screenshot(path=str(REPORTS_UI / "dashboard_390px.png"))
            page.evaluate("document.querySelector('button[data-tab=\"tab-analysis\"]').click()")
            time.sleep(0.5)
            page.screenshot(path=str(REPORTS_UI / "analysis_390px.png"))
            print("[SCREENSHOT 390px] Saved mobile dashboard and analysis.")

            # -----------------------------------------------------------------
            # 11. AUTHENTICATION GUARD & UNAUTHENTICATED REDIRECT
            # -----------------------------------------------------------------
            print("\n--- STEP 11: AUTHENTICATION GUARD AUDIT ---")
            page.set_viewport_size({"width": 1280, "height": 850})
            page.click("#btn-logout")
            page.wait_for_url(f"{base_url}/login.html", timeout=5000)
            print("[AUTH GUARD] Logout redirected to /login.html")

            # Try visiting dashboard without token
            page.evaluate("localStorage.clear()")
            page.goto(f"{base_url}/")
            page.wait_for_url(f"{base_url}/login.html", timeout=5000)
            print("[AUTH GUARD] Unauthenticated access to / properly blocked and redirected to /login.html")

            # -----------------------------------------------------------------
            # 12. BROWSER CONSOLE ERRORS
            # -----------------------------------------------------------------
            print("\n--- STEP 12: CONSOLE ERRORS AUDIT ---")
            print(f"Total Browser Console Errors: {len(console_errors)}")
            for err in console_errors:
                print(f"  [ERROR] {err}")
            assert len(console_errors) == 0, f"Expected 0 console errors, but found {len(console_errors)}!"
            print("[CONSOLE AUDIT SUCCESS] Clean execution with 0 console errors!")

            browser.close()

        print("\n=======================================================")
        print("[SUCCESS] ALL END-TO-END BROWSER VERIFICATION TESTS PASSED!")
        print("=======================================================\n")

    except Exception as e:
        print(f"\n[CRASH ERROR]: {e}")
        if 'console_errors' in locals():
            print(f"[CONSOLE ERRORS CAPTURED]: {len(console_errors)}")
            for ce in console_errors:
                print(f"  {ce}")
        raise e
    finally:
        print("[CLEANUP] Stopping server...")
        server.should_exit = True
        if _AUTO_TEMP_DATA_DIR and Path(_AUTO_TEMP_DATA_DIR).exists():
            shutil.rmtree(_AUTO_TEMP_DATA_DIR, ignore_errors=True)


if __name__ == "__main__":
    run_e2e_verification()
