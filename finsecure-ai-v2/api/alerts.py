"""
alerts.py — Alert Webhook Dispatcher
=====================================
Dispatches HTTP webhooks when critical events (high-risk fraud or concept drift) occur.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx


def send_alert_webhook(alert_type: str, payload: Dict[str, Any]) -> bool:
    """
    Post alert payload to configured ALERT_WEBHOOK_URL.
    
    alert_type: 'HIGH_FRAUD_ALERT' or 'RETRAIN_RECOMMENDED'
    """
    webhook_url = os.getenv("ALERT_WEBHOOK_URL")
    if not webhook_url:
        print(f"[ALERT LOCAL LOG]: ({alert_type}) {payload}")
        return True

    try:
        body = {
            "alert_type": alert_type,
            "payload": payload
        }
        resp = httpx.post(webhook_url, json=body, timeout=3.0)
        return resp.status_code == 200
    except Exception as e:
        print(f"[ALERT WEBHOOK ERROR]: Failed to dispatch {alert_type}: {e}")
        return False
