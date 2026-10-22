"""
FinSecure AI v2 — One-Command Runner
====================================
Unified launcher for FinSecure AI v2.

Usage:
    python run.py          # Starts Backend API & serves Dashboard at http://localhost:8000 (opens browser)
    python run.py demo     # Runs the End-to-End Audit & Demonstration script (scripts/demo.py)
    python run.py train    # Trains XGBoost model & tunes cost-optimal threshold (src/train_model.py)
    python run.py test     # Executes pytest test suite (30 unit & integration tests)
    python run.py help     # Displays usage and available commands
"""

import os
import sys
import time
import threading
import subprocess
import webbrowser
from pathlib import Path


def resolve_project_root() -> Path:
    """Find the directory containing api/ and src/."""
    current = Path(__file__).resolve().parent
    if (current / "finsecure-ai-v2" / "api").exists():
        return current / "finsecure-ai-v2"
    if (current / "api").exists() and (current / "src").exists():
        return current
    if (current.parent / "api").exists():
        return current.parent
    raise FileNotFoundError("Could not locate FinSecure AI v2 project directory.")


def open_browser_later(url: str, delay: float = 1.5):
    """Open web browser after a short delay to give the server time to start."""
    def _open():
        time.sleep(delay)
        try:
            webbrowser.open(url)
        except Exception:
            pass
    threading.Thread(target=_open, daemon=True).start()


def main():
    project_dir = resolve_project_root()
    os.chdir(str(project_dir))
    sys.path.insert(0, str(project_dir))

    cmd = (sys.argv[1].lower() if len(sys.argv) > 1 else "server")

    banner = r"""
===========================================================================
  FinSecure AI v2 — Real-Time Financial Fraud Detection & XAI System
===========================================================================
"""
    print(banner)

    if cmd in ("demo", "run-demo", "audit"):
        print("[RUNNING]: End-to-End System Demonstration (scripts/demo.py)...")
        print("-" * 75)
        sys.exit(subprocess.call([sys.executable, "scripts/demo.py"]))

    elif cmd in ("train", "train-model"):
        print("[RUNNING]: Model Training & Threshold Tuning Pipeline (src/train_model.py)...")
        print("-" * 75)
        sys.exit(subprocess.call([sys.executable, "src/train_model.py"]))

    elif cmd in ("test", "tests", "pytest"):
        print("[RUNNING]: Full PyTest Validation Suite (30 tests)...")
        print("-" * 75)
        sys.exit(subprocess.call([sys.executable, "-m", "pytest", "-v"]))

    elif cmd in ("help", "-h", "--help"):
        print("Available commands:")
        print("  python run.py          Start the full system (API + UI Dashboard on http://localhost:8000)")
        print("  python run.py demo     Run the end-to-end demonstration & assertion checks")
        print("  python run.py train    Run XGBoost model training and threshold tuning")
        print("  python run.py test     Run full pytest test suite")
        sys.exit(0)

    elif cmd in ("server", "app", "start"):
        port = 8000
        host = "127.0.0.1"
        url = f"http://{host}:{port}"
        docs_url = f"{url}/docs"

        print(f"[*] Starting unified FinSecure AI v2 Server...")
        print(f"[*] Dashboard UI:   {url}")
        print(f"[*] API Docs:       {docs_url}")
        print(f"[*] WebSocket Feed: ws://{host}:{port}/ws/scan")
        print("-" * 75)
        print(f"[+] Opening {url} in your default browser in 1.5 seconds...")
        print(f"[!] Press CTRL+C to stop the server.\n")

        open_browser_later(url, delay=1.5)

        try:
            subprocess.run([
                sys.executable, "-m", "uvicorn", "api.main:app",
                "--host", host,
                "--port", str(port),
                "--reload"
            ])
        except KeyboardInterrupt:
            print("\n[!] FinSecure AI v2 server stopped by user.")
    else:
        print(f"Unknown command: '{cmd}'. Run 'python run.py help' for available commands.")
        sys.exit(1)


if __name__ == "__main__":
    main()
