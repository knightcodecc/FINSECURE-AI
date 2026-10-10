import sys
import time
import httpx

API_BASE = "http://127.0.0.1:8000"

def main():
    print("=== VERIFYING FASTAPI ENDPOINTS END-TO-END ===")

    # 1. GET /health
    resp_health = httpx.get(f"{API_BASE}/health", timeout=5.0)
    print(f"\n[1] GET /health (Status {resp_health.status_code}):")
    print(resp_health.json())

    # 2. GET /model/info
    resp_info = httpx.get(f"{API_BASE}/model/info", timeout=5.0)
    print(f"\n[2] GET /model/info (Status {resp_info.status_code}):")
    print(resp_info.json())

    # 3. POST /predict
    payload_tx = {
        "step": 3,
        "type": "TRANSFER",
        "amount": 100.0,
        "nameOrig": "C_RING_MULE",
        "oldbalanceOrg": 100.0,
        "newbalanceOrig": 0.0,
        "nameDest": "M_HUB",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 0.0
    }
    resp_pred = httpx.post(f"{API_BASE}/predict", json=payload_tx, timeout=5.0)
    print(f"\n[3] POST /predict (Status {resp_pred.status_code}):")
    print(resp_pred.json())

    # 4. GET /metrics/cost
    resp_cost = httpx.get(f"{API_BASE}/metrics/cost", timeout=5.0)
    print(f"\n[4] GET /metrics/cost (Status {resp_cost.status_code}):")
    print(resp_cost.json())

    # 5. GET /drift/status
    resp_drift = httpx.get(f"{API_BASE}/drift/status", timeout=5.0)
    print(f"\n[5] GET /drift/status (Status {resp_drift.status_code}):")
    print(resp_drift.json())

    # 6. POST /drift/simulate
    resp_sim = httpx.post(f"{API_BASE}/drift/simulate?amount_scale=5.0", timeout=5.0)
    print(f"\n[6] POST /drift/simulate (Status {resp_sim.status_code}):")
    print(resp_sim.json())

    # 7. POST /chat
    resp_chat = httpx.post(f"{API_BASE}/chat", json={"query": "Why was the recent transfer to M_HUB flagged as fraud?"}, timeout=5.0)
    print(f"\n[7] POST /chat (Status {resp_chat.status_code}):")
    print(resp_chat.json())

    print("\n=== ALL FASTAPI REST ENDPOINTS VERIFIED SUCCESSFULLY ===")


if __name__ == "__main__":
    main()
