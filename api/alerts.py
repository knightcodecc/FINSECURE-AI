"""
Alert Webhook & Notification Dispatcher for FinSecure AI.

Triggers structured alert notifications (logged webhook payload / HTTP POST stub / email stub)
whenever a transaction crosses the fraud threshold (is_fraud == True).

Author: FinSecure AI Team
"""

import logging
import asyncio
import os
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Configurable webhook endpoint
WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL", "https://api.bank-security-example.com/alerts/fraud")

async def dispatch_fraud_alert(prediction: Dict[str, Any], transaction_payload: Dict[str, Any]) -> bool:
    """
    Dispatch a fraud alert webhook when a transaction is flagged as fraud.

    Args:
        prediction: Prediction result dictionary (is_fraud, probability, risk_level, top_risk_factors).
        transaction_payload: Transaction details (amount, step, type, oldbalanceOrg, etc.).

    Returns:
        bool: True if alert dispatched successfully.
    """
    if not prediction.get('is_fraud'):
        return False

    alert_payload = {
        'alert_id': f"ALT-{datetime.now().strftime('%Y%m%d%H%M%S')}",
        'event_type': 'FRAUD_DETECTED',
        'severity': prediction.get('risk_level', 'HIGH'),
        'timestamp': datetime.now().isoformat(),
        'transaction_id': str(prediction.get('transaction_id', '')),
        'fraud_probability': prediction.get('fraud_probability'),
        'amount': transaction_payload.get('amount'),
        'transaction_type': str(transaction_payload.get('type')),
        'step': transaction_payload.get('step'),
        'top_risk_factors': [
            {'feature': f.feature if hasattr(f, 'feature') else f.get('feature'),
             'importance': f.importance if hasattr(f, 'importance') else f.get('importance')}
            for f in prediction.get('top_risk_factors', [])[:3]
        ]
    }

    # Log structured alert to system logs (Extension point for Webhook / SMTP email)
    logger.warning(
        f"🚨 [ALERT WEBHOOK DISPATCH] Fraud Alert Sent to Bank/Admin Security Service:\n"
        f"   Alert ID: {alert_payload['alert_id']} | TxID: {alert_payload['transaction_id']}\n"
        f"   Probability: {alert_payload['fraud_probability']*100:.1f}% | Risk: {alert_payload['severity']}\n"
        f"   Type: {alert_payload['transaction_type']} | Amount: ${alert_payload['amount']:,.2f}\n"
        f"   Target Webhook Endpoint: {WEBHOOK_URL}"
    )

    # In production, this can perform `aiohttp` or `httpx` POST to WEBHOOK_URL
    return True
