"""
realtime_scanner.py — WebSocket Live Transaction Stream & Stats Broadcast.
=============================================================================
Provides a single shared Scanner singleton for /ws/scan and /ws/stats endpoints,
broadcasting live transaction scores, fraud rate statistics, and drift alerts.
"""

import asyncio
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Dict, List, Any, Set

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import WebSocket


class RealtimeScanner:
    """
    Singleton Scanner managing WebSocket connections for live transaction streaming
    and periodic stats broadcast.
    """
    def __init__(self):
        self.scan_clients: Set[WebSocket] = set()
        self.stats_clients: Set[WebSocket] = set()
        
        self.total_processed: int = 0
        self.total_fraud: int = 0
        self.total_amount: float = 0.0
        self.recent_alerts: List[Dict[str, Any]] = []

    async def connect_scan(self, websocket: WebSocket):
        await websocket.accept()
        self.scan_clients.add(websocket)

    def disconnect_scan(self, websocket: WebSocket):
        self.scan_clients.discard(websocket)

    async def connect_stats(self, websocket: WebSocket):
        await websocket.accept()
        self.stats_clients.add(websocket)

    def disconnect_stats(self, websocket: WebSocket):
        self.stats_clients.discard(websocket)

    async def process_and_broadcast(self, pred_result: Dict[str, Any]):
        """Process a scored transaction and broadcast to connected clients."""
        self.total_processed += 1
        amount = float(pred_result.get("amount", 0.0))
        self.total_amount += amount

        if pred_result.get("is_fraud"):
            self.total_fraud += 1
            if len(self.recent_alerts) >= 50:
                self.recent_alerts.pop(0)
            self.recent_alerts.append(pred_result)

        # Broadcast live scored transaction payload to /ws/scan subscribers
        scan_msg = json.dumps(pred_result)
        disconnected = set()
        for client in self.scan_clients:
            try:
                await client.send_text(scan_msg)
            except Exception:
                disconnected.add(client)
        for client in disconnected:
            self.scan_clients.discard(client)

        # Broadcast updated statistics to /ws/stats subscribers
        fraud_rate = (self.total_fraud / self.total_processed * 100.0) if self.total_processed > 0 else 0.0
        stats_payload = {
            "timestamp": datetime.now().isoformat(),
            "total_processed": self.total_processed,
            "total_fraud": self.total_fraud,
            "total_amount": self.total_amount,
            "fraud_rate_pct": fraud_rate,
            "active_threshold": pred_result.get("threshold_used", 0.25),
        }
        stats_msg = json.dumps(stats_payload)
        
        disconnected_stats = set()
        for client in self.stats_clients:
            try:
                await client.send_text(stats_msg)
            except Exception:
                disconnected_stats.add(client)
        for client in disconnected_stats:
            self.stats_clients.discard(client)


# Singleton RealtimeScanner instance
_SCANNER: RealtimeScanner | None = None


def get_scanner() -> RealtimeScanner:
    global _SCANNER
    if _SCANNER is None:
        _SCANNER = RealtimeScanner()
    return _SCANNER
