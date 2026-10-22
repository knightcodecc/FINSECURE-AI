"""
FinSecure AI v2 — One-Command Root Runner
=========================================
Delegates to finsecure-ai-v2/run.py
"""

import sys
from pathlib import Path

sub_run = Path(__file__).resolve().parent / "finsecure-ai-v2" / "run.py"
if sub_run.exists():
    with open(sub_run, "rb") as f:
        code = compile(f.read(), str(sub_run), "exec")
        exec(code, {"__name__": "__main__", "__file__": str(sub_run)})
else:
    print(f"Error: Could not find {sub_run}")
    sys.exit(1)
