"""
rag_chatbot.py — RAG Chatbot over SQLite Prediction Log & Drift Status.
========================================================================
Queries the prediction database (data/finsecure.db) and drift snapshots
to answer analyst questions with grounded context. Strictly filters predictions
by the current deployed model_version to ensure serving data hygiene.
"""

from datetime import datetime
import json
from pathlib import Path
import sqlite3
import sys
from typing import Dict, List, Any

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.db import get_db_connection, get_model_version


def answer_analyst_query(query: str, db_path: Path = Path("data/finsecure.db")) -> str:
    """
    RAG Chatbot logic querying prediction logs and drift snapshots for the CURRENT model version.
    """
    q_lower = query.lower()
    current_version = get_model_version()

    # Query 1: Drift status questions
    if "drift" in q_lower or "retrain" in q_lower or "psi" in q_lower:
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT timestamp, overall_status, retrain_recommended, score_psi, metrics_json FROM drift_snapshots ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            
            if row:
                ts, status, retrain, psi, metrics_raw = row
                retrain_str = "YES (Retraining Action Recommended)" if retrain else "NO (Model Operating Within Normal Thresholds)"
                return f"Drift Status Update [{ts}]: Overall Status = '{status}', Prediction Score PSI = {psi:.4f}. Retrain Recommended = {retrain_str}."
            else:
                return "No concept drift snapshots logged yet. Run a drift check or POST /drift/simulate to view drift metrics."

    # Query 2: Fraud summary & SHAP questions (Filter strictly by current model_version)
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*), SUM(is_fraud), SUM(amount) FROM predictions WHERE model_version = ?", (current_version,))
        row = cursor.fetchone()
        total_tx, total_fraud, total_amt = row if row and row[0] is not None else (0, 0, 0.0)

        cursor.execute(
            "SELECT name_orig, name_dest, amount, prediction_proba, explanation FROM predictions WHERE model_version = ? AND is_fraud = 1 ORDER BY id DESC LIMIT 3",
            (current_version,)
        )
        fraud_rows = cursor.fetchall()

    if "shap" in q_lower or "reason" in q_lower or "why" in q_lower or "factor" in q_lower:
        if fraud_rows:
            reasons = []
            for orig, dest, amt, proba, exp in fraud_rows:
                reasons.append(f"- Transaction {orig} -> {dest} (Rs. {amt:,.2f}, proba {proba:.1%}): {exp}")
            return f"Recent flagged fraud cases under current model version ({current_version}) and their primary SHAP reasons:\n" + "\n".join(reasons)
        else:
            return f"No fraud transactions logged in predictions table for current model version ({current_version}) yet."

    # Default summary response
    return f"FinSecure AI v2 System Summary [Model Version: {current_version}]: Total Processed Transactions = {total_tx}, Flagged Frauds = {total_fraud}, Total Volume = Rs. {total_amt:,.2f}. Ask me about recent SHAP explanations, fraud counts, or concept drift status!"
