import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import httpx
from api.auth import create_access_token

def test_chips():
    token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiaGF0cHJhdGhhbTQzQGdtYWlsLmNvbSIsImlkIjo0MSwibmFtZSI6IlByYXRoYW0iLCJyb2xlIjoiYW5hbHlzdCIsImV4cCI6MTc5MjY3OTM3NH0.EnK44o2enRQU7n-ElOuSAtQofcXx0CwZtlTvTtcCJno"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    test_queries = [
        ("Chip 1 (How many scored?)", "How many transactions were scored?"),
        ("Chip 2 (Fraud count?)", "How many were flagged as fraud?"),
        ("Chip 3 (Explain last decision)", "Explain the last decision"),
        ("Chip 4 (Active threshold?)", "What is the active threshold?"),
        ("Chip 5 (Why thresholds differ?)", "Why is the cost-optimal threshold different from the F2-optimal one?"),
        ("Chip 6 (Drift status?)", "What is the drift status?"),
        ("Chip 7 (PaySim frauds?)", "How many frauds are in the dataset?"),
        ("Chip 8 (What is SHAP?)", "What is SHAP?"),
        ("Plus: Explain the last decision", "Explain the last decision"),
    ]

    print("=" * 80)
    print("FINSECURE AI v2 -- LIVE HTTP CHAT ENDPOINT BENCHMARK (/chat)")
    print("Testing 8 Quick-Reply Chips + 'Explain the last decision'")
    print("Requirement: Every one must respond in under 5.0 seconds")
    print("=" * 80)

    results = []
    with httpx.Client(base_url="http://127.0.0.1:8000", timeout=10.0) as client:
        for label, query in test_queries:
            t0 = time.perf_counter()
            resp = client.post("/chat", json={"query": query, "history": []}, headers=headers)
            elapsed = time.perf_counter() - t0

            assert resp.status_code == 200, f"Failed: {resp.status_code} {resp.text}"
            data = resp.json()
            answer = data.get("answer", "")
            sources = data.get("sources", [])

            under_5s = elapsed < 5.0
            results.append({
                "label": label,
                "query": query,
                "latency_sec": elapsed,
                "under_5s": under_5s,
                "sources": sources,
                "answer_preview": answer[:120].replace("\n", " ") + ("..." if len(answer) > 120 else "")
            })

            status_str = "PASS (<5s)" if under_5s else "FAIL (>5s)"
            print(f"\n[{status_str}] {label}")
            print(f"  Query: \"{query}\"")
            print(f"  Latency: {elapsed*1000:.2f} ms ({elapsed:.4f}s)")
            print(f"  Sources: {sources}")
            print(f"  Answer: {answer[:140]}...")

    print("\n" + "=" * 80)
    print("LATENCY SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Item':<35} | {'Latency (ms)':<15} | {'Under 5s?':<10} | {'Status'}")
    print("-" * 75)
    for r in results:
        ms = r["latency_sec"] * 1000
        status = "PASSED" if r["under_5s"] else "FAILED"
        print(f"{r['label']:<35} | {ms:>10.2f} ms   | {str(r['under_5s']):<10} | {status}")
    print("=" * 80)

if __name__ == "__main__":
    test_chips()
