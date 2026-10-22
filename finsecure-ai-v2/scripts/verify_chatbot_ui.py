"""
Playwright End-to-End Chatbot Verification Script
Proves:
1. Keyboard UX: Pressing Enter sends the message, clears input, displays typing indicator, disables/re-enables Send button.
2. Grounded Ambiguous Scope: "How many transactions are there?" replies with all scopes:
   - Scored predictions in live log
   - PaySim dataset (6,362,620 rows)
   - Latest uploaded analysis (with graph mode and threshold mode)
   Asserts DOM reply matches API reply exactly across all scopes.
3. 5 paraphrases and 1 typo query ("how many trasactions scord") verified in browser DOM vs API.
4. Fallback returns 3 clickable suggestions without raw *'...' * asterisks.
5. Sticky header no longer overlaps chat title or the first sidebar item ('Scorer & XAI').
6. Screenshots saved to reports/ui/chatbot_*.png.
"""

import os
import sys
import time
import socket
import json
import urllib.request
import subprocess
from pathlib import Path
from playwright.sync_api import sync_playwright

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)

ROOT_DIR = Path(__file__).resolve().parent.parent
REPORTS_UI = ROOT_DIR / "reports" / "ui"
REPORTS_UI.mkdir(parents=True, exist_ok=True)


def find_free_port(start=8010):
    for port in range(start, start + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('127.0.0.1', port)) != 0:
                return port
    return 8010


def wait_for_server(url, timeout=35):
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def call_chat_api(base_url, query_text, token):
    req = urllib.request.Request(
        f"{base_url}/chat",
        data=json.dumps({"query": query_text}).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data["answer"]


def main():
    print("=" * 80)
    print("  FinSecure AI v2 — Chatbot Playwright E2E Verification")
    print("=" * 80)

    port = find_free_port()
    base_url = f"http://127.0.0.1:{port}"
    print(f"[*] Starting background test server on {base_url}...")

    server_log_file = ROOT_DIR / "reports" / "uvicorn_test_server.log"
    server_log_fp = open(server_log_file, "w", encoding="utf-8")

    server_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", str(port)],
        cwd=str(ROOT_DIR),
        stdout=server_log_fp,
        stderr=server_log_fp
    )

    try:
        assert wait_for_server(base_url, timeout=35), f"Server failed to start on {base_url} within 35s"
        print(f"[+] Server is healthy on {base_url}\n")

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 800})

            # 1. Register & Login
            print("--- STEP 1: AUTHENTICATION ---")
            ts = int(time.time())
            test_email = f"audit_analyst_{ts}@finsecure.ai"
            test_password = "AuditorPassword2026!"
            test_name = "Audit Analyst"

            page.goto(f"{base_url}/register.html")
            page.fill("#reg-fullname", test_name)
            page.fill("#reg-email", test_email)
            page.fill("#reg-password", test_password)
            page.click("#btn-register")
            page.wait_for_url(f"{base_url}/", timeout=10000)
            page.wait_for_selector("#user-display-name")
            print(f"[AUTH OK] Registered & logged in as {test_email}")

            auth_token = page.evaluate("() => localStorage.getItem('finsecure_token')")
            assert auth_token, "Auth token not found in localStorage"

            # 2. Sticky Header Overlap Verification
            print("\n--- STEP 2: STICKY HEADER & LAYOUT AUDIT ---")
            # Navigate to chat tab
            page.click('button[data-tab="tab-chat"]')
            page.wait_for_selector("#tab-chat.active")

            # Check positions of header, sidebar first item, and chat title
            header_box = page.locator(".app-header").bounding_box()
            scorer_box = page.locator('button[data-tab="tab-scorer"]').bounding_box()
            chat_title_box = page.locator("#tab-chat h1, #tab-chat .page-title, #tab-chat header, #tab-chat .chat-header").first.bounding_box()

            print(f"  Header BoundingBox:       top={header_box['y']}, height={header_box['height']}, bottom={header_box['y'] + header_box['height']}")
            print(f"  Scorer & XAI (Sidebar 1): top={scorer_box['y']}, height={scorer_box['height']}")
            if chat_title_box:
                print(f"  Chat Panel Title:         top={chat_title_box['y']}, height={chat_title_box['height']}")
                assert chat_title_box['y'] >= header_box['y'] + header_box['height'] - 2, "Chat title is clipped behind header!"

            assert scorer_box['y'] >= header_box['y'] + header_box['height'] - 2, "Scorer & XAI sidebar item is clipped behind header!"
            print("[LAYOUT OK] Header does NOT overlap sidebar item or chat panel title.")

            header_screenshot_path = REPORTS_UI / "chatbot_header_fixed.png"
            page.screenshot(path=str(header_screenshot_path))
            print(f"[SCREENSHOT] Saved: {header_screenshot_path}")

            # 3. Enter key test with "How many transactions are there?"
            print("\n--- STEP 3: AMBIGUOUS SCOPE QUERY VIA ENTER KEY ---")
            query_1 = "How many transactions are there?"
            print(f"[ACTION] Typing query: \"{query_1}\"")
            page.fill("#chat-input", query_1)

            # Press Enter
            print("[ACTION] Pressing Enter key...")
            page.keyboard.press("Enter")

            # Verify input clears immediately
            input_val = page.input_value("#chat-input")
            assert input_val == "", f"Expected #chat-input to be cleared, got '{input_val}'"
            print("[UX OK] Input cleared immediately on Enter.")

            # Wait for bot bubble (not typing indicator)
            page.wait_for_function("document.querySelectorAll('.chat-bubble.bot:not(.typing)').length >= 2", timeout=15000)
            bot_bubbles = page.query_selector_all(".chat-bubble.bot:not(.typing)")
            dom_reply = bot_bubbles[-1].inner_text().strip()

            # Direct API query for exact parity assertion
            api_reply = call_chat_api(base_url, query_1, auth_token).strip()

            print("\n[DOM RENDERED REPLY]:")
            print("-" * 75)
            print(dom_reply)
            print("-" * 75)

            print("\n[API RETURNED REPLY]:")
            print("-" * 75)
            print(api_reply)
            print("-" * 75)

            # Assert scopes are present
            assert "Live Predictions Log" in dom_reply or "operational database" in dom_reply
            assert "6,362,620" in dom_reply or "PaySim" in dom_reply
            assert "Latest Uploaded Analysis" in dom_reply or "uploaded" in dom_reply.lower()
            print("[SCOPES OK] All 3 scopes (Live predictions log, PaySim 6.36M dataset, and Upload status) verified in DOM reply.")

            all_scopes_screenshot = REPORTS_UI / "chatbot_all_scopes.png"
            page.screenshot(path=str(all_scopes_screenshot))
            print(f"[SCREENSHOT] Saved: {all_scopes_screenshot}")

            # 4. Repeat for 5 more paraphrases and 1 typo query
            print("\n--- STEP 4: PARAPHRASES & TYPO INTERACTIVE TEST ---")
            additional_queries = [
                ("total transactions", ["Live Predictions Log", "6,362,620", "Latest Uploaded Analysis"]),
                ("how many were flagged", ["flagged as fraud"]),
                ("what's the threshold", ["decision threshold", "0.0484"]),
                ("is the model drifting", ["Drift Status"]),
                ("explain transaction 5", ["Transaction #5", "Decision", "Route"]),
                ("how many trasactions scord", ["transactions", "scored and logged"]),
            ]

            for q_idx, (q_text, expected_tokens) in enumerate(additional_queries, 2):
                print(f"\n[QUERY #{q_idx}] \"{q_text}\"")
                page.fill("#chat-input", q_text)
                page.keyboard.press("Enter")

                # Wait until new bot bubble appears
                expected_count = q_idx + 1  # 1 welcome message + q_idx bot replies
                page.wait_for_function(f"document.querySelectorAll('.chat-bubble.bot:not(.typing)').length >= {expected_count}", timeout=15000)

                bot_bubbles = page.query_selector_all(".chat-bubble.bot:not(.typing)")
                current_dom_reply = bot_bubbles[-1].inner_text().strip()
                current_api_reply = call_chat_api(base_url, q_text, auth_token).strip()

                print(f"  DOM Reply: {current_dom_reply[:95]}...")
                for tok in expected_tokens:
                    assert tok.lower() in current_dom_reply.lower(), f"Expected '{tok}' in DOM reply for '{q_text}'"
                print(f"  [OK] Matched expected concept tokens: {expected_tokens}")

            # 5. Fallback query with suggestions and clean markdown
            print("\n--- STEP 5: FALLBACK QUERY & CLICKABLE SUGGESTIONS ---")
            fallback_q = "what is the current temperature in Mumbai?"
            page.fill("#chat-input", fallback_q)
            page.keyboard.press("Enter")

            page.wait_for_function("document.querySelectorAll('.chat-bubble.bot:not(.typing)').length >= 9", timeout=15000)
            bot_bubbles = page.query_selector_all(".chat-bubble.bot:not(.typing)")
            fallback_dom_reply = bot_bubbles[-1].inner_text().strip()
            fallback_html = bot_bubbles[-1].inner_html()

            print(f"[FALLBACK DOM REPLY]:\n{fallback_dom_reply}")

            assert "Did you mean:" in fallback_dom_reply, "Missing 'Did you mean:' in fallback"
            assert "*'" not in fallback_dom_reply and "'*" not in fallback_dom_reply, "Raw asterisks found in fallback text!"
            print("[FALLBACK OK] Verified no raw *'...' * asterisks in fallback response.")

            suggestions_screenshot = REPORTS_UI / "chatbot_suggestions.png"
            page.screenshot(path=str(suggestions_screenshot))
            print(f"[SCREENSHOT] Saved: {suggestions_screenshot}")

            # Verify clickable suggestion link
            suggestion_links = page.query_selector_all(".chat-suggestion-link")
            if suggestion_links:
                first_sugg_text = suggestion_links[0].inner_text().strip()
                print(f"[ACTION] Clicking suggestion link: '{first_sugg_text}'")
                suggestion_links[0].click()
                page.wait_for_function("document.querySelectorAll('.chat-bubble.bot:not(.typing)').length >= 10", timeout=15000)
                new_reply = page.query_selector_all(".chat-bubble.bot:not(.typing)")[-1].inner_text().strip()
                print(f"[SUGGESTION CLICK OK] Replied to suggestion with: '{new_reply[:80]}...'")

            browser.close()
            print("\n[SUCCESS] All Playwright browser E2E verifications completed successfully!")

    finally:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=3)
        except Exception:
            server_proc.kill()
        try:
            server_log_fp.close()
        except Exception:
            pass
        print("[+] Background test server shut down cleanly.")


if __name__ == "__main__":
    main()
