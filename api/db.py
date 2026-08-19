"""
Database module for FinSecure AI Fraud Detection System.

Provides persistent SQLite storage (with WAL mode) for:
1. Predictions (payload, features, probability, risk level, verdict)
2. XAI explanations
3. Analyst feedback (CONFIRMED_FRAUD, FALSE_POSITIVE)
4. Persistent velocity/novelty feature lookups (is_dest_new, txn_count_24h, recency_hours)
5. Aggregate stats persistence across server restarts
6. RAG Context Retrieval for Process Q&A Chatbot

Author: FinSecure AI Team
"""

import sqlite3
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "finsecure.db"

def get_db_connection() -> sqlite3.Connection:
    """Connect to SQLite database with WAL mode enabled for high concurrency."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn

def init_db() -> None:
    """Initialize database tables and indexes."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Predictions table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT UNIQUE NOT NULL,
            step INTEGER NOT NULL,
            type TEXT NOT NULL,
            amount REAL NOT NULL,
            oldbalanceOrg REAL NOT NULL,
            newbalanceOrig REAL NOT NULL,
            oldbalanceDest REAL NOT NULL,
            newbalanceDest REAL NOT NULL,
            nameOrig TEXT DEFAULT '',
            nameDest TEXT DEFAULT '',
            fraud_probability REAL NOT NULL,
            is_fraud INTEGER NOT NULL,
            risk_level TEXT NOT NULL,
            top_risk_factors TEXT,
            timestamp TEXT NOT NULL
        );
        """)
        
        # Indexes for fast velocity lookups and analytics
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_nameOrig ON predictions(nameOrig);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_nameDest ON predictions(nameDest);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_step ON predictions(step);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_timestamp ON predictions(timestamp);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_is_fraud ON predictions(is_fraud);")
        
        # Explanations table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS explanations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT NOT NULL,
            explanation_text TEXT NOT NULL,
            risk_factors TEXT,
            timestamp TEXT NOT NULL,
            FOREIGN KEY (transaction_id) REFERENCES predictions(transaction_id)
        );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_exp_txid ON explanations(transaction_id);")

        # Analyst Feedback table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS analyst_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT NOT NULL,
            feedback_label TEXT NOT NULL,
            notes TEXT,
            timestamp TEXT NOT NULL
        );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_txid ON analyst_feedback(transaction_id);")
        
        conn.commit()
        logger.info(f"SQLite Database initialized at {DB_PATH}")

def log_prediction(
    transaction_id: str,
    step: int,
    type_str: str,
    amount: float,
    oldbalanceOrg: float,
    newbalanceOrig: float,
    oldbalanceDest: float,
    newbalanceDest: float,
    fraud_probability: float,
    is_fraud: bool,
    risk_level: str,
    top_risk_factors: List[Dict[str, Any]],
    nameOrig: str = "",
    nameDest: str = ""
) -> None:
    """Log a single prediction to the predictions table."""
    try:
        factors_json = json.dumps([
            {'feature': f.feature if hasattr(f, 'feature') else f.get('feature'),
             'value': f.value if hasattr(f, 'value') else f.get('value'),
             'importance': f.importance if hasattr(f, 'importance') else f.get('importance')}
            for f in top_risk_factors
        ])
        with get_db_connection() as conn:
            conn.execute("""
            INSERT OR REPLACE INTO predictions (
                transaction_id, step, type, amount, oldbalanceOrg, newbalanceOrig,
                oldbalanceDest, newbalanceDest, nameOrig, nameDest,
                fraud_probability, is_fraud, risk_level, top_risk_factors, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                transaction_id, step, type_str, amount, oldbalanceOrg, newbalanceOrig,
                oldbalanceDest, newbalanceDest, nameOrig, nameDest,
                fraud_probability, int(is_fraud), risk_level, factors_json, datetime.now().isoformat()
            ))
            conn.commit()
    except Exception as e:
        logger.error(f"Failed to log prediction to DB: {e}")

def log_explanation(transaction_id: str, explanation_text: str, risk_factors: List[Dict[str, Any]]) -> None:
    """Log an XAI explanation to the explanations table."""
    try:
        factors_json = json.dumps(risk_factors) if risk_factors else None
        with get_db_connection() as conn:
            conn.execute("""
            INSERT INTO explanations (transaction_id, explanation_text, risk_factors, timestamp)
            VALUES (?, ?, ?, ?);
            """, (transaction_id, explanation_text, factors_json, datetime.now().isoformat()))
            conn.commit()
    except Exception as e:
        logger.error(f"Failed to log explanation to DB: {e}")

def log_feedback(transaction_id: str, feedback_label: str, notes: Optional[str] = None) -> None:
    """Log analyst feedback (CONFIRMED_FRAUD / FALSE_POSITIVE)."""
    try:
        with get_db_connection() as conn:
            conn.execute("""
            INSERT INTO analyst_feedback (transaction_id, feedback_label, notes, timestamp)
            VALUES (?, ?, ?, ?);
            """, (transaction_id, feedback_label, notes, datetime.now().isoformat()))
            conn.commit()
    except Exception as e:
        logger.error(f"Failed to log analyst feedback: {e}")

def query_realtime_velocity(nameOrig: str, nameDest: str, current_step: int) -> Tuple[int, float, int]:
    """
    Query historical predictions in DB to compute real-time velocity features:
    - is_dest_new: 1 if (nameOrig, nameDest) pair not seen before, 0 if seen
    - recency_hours: step delta since last transaction by nameOrig (999.0 if first)
    - txn_count_24h: count of transactions by nameOrig in last 24 steps
    """
    if not nameOrig:
        return 1, 999.0, 1

    try:
        with get_db_connection() as conn:
            # 1. is_dest_new
            cur = conn.cursor()
            cur.execute("""
            SELECT COUNT(*) FROM predictions WHERE nameOrig = ? AND nameDest = ?;
            """, (nameOrig, nameDest))
            dest_count = cur.fetchone()[0]
            is_dest_new = 1 if dest_count == 0 else 0

            # 2. recency_hours & 3. txn_count_24h
            cur.execute("""
            SELECT step FROM predictions WHERE nameOrig = ? ORDER BY step DESC LIMIT 1;
            """, (nameOrig,))
            last_row = cur.fetchone()
            if last_row:
                last_step = last_row[0]
                recency_hours = float(current_step - last_step)
            else:
                recency_hours = 999.0

            cur.execute("""
            SELECT COUNT(*) FROM predictions WHERE nameOrig = ? AND step >= ?;
            """, (nameOrig, max(0, current_step - 24)))
            txn_count_24h = cur.fetchone()[0] + 1  # count including current

            return is_dest_new, recency_hours, txn_count_24h
    except Exception as e:
        logger.warning(f"Error querying velocity features from DB: {e}")
        return 1, 999.0, 1

def get_persisted_stats() -> Dict[str, Any]:
    """Compute aggregate statistics directly from persisted prediction records."""
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(*) FROM predictions;")
            total_predictions = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*), SUM(amount) FROM predictions WHERE is_fraud = 1;")
            fraud_row = cur.fetchone()
            fraud_detected = fraud_row[0] or 0
            total_fraud_amount = fraud_row[1] or 0.0

            cur.execute("SELECT AVG(fraud_probability) FROM predictions;")
            avg_prob = cur.fetchone()[0] or 0.0

            cur.execute("""
            SELECT risk_level, COUNT(*) FROM predictions GROUP BY risk_level;
            """)
            risk_counts = {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0, 'CRITICAL': 0}
            for row in cur.fetchall():
                if row[0] in risk_counts:
                    risk_counts[row[0]] = row[1]

            fraud_rate = (fraud_detected / total_predictions * 100) if total_predictions > 0 else 0.0

            return {
                'total_predictions': total_predictions,
                'fraud_detected': fraud_detected,
                'fraud_rate': round(fraud_rate, 2),
                'total_fraud_amount': round(total_fraud_amount, 2),
                'average_fraud_probability': round(avg_prob, 4),
                'risk_level_distribution': risk_counts
            }
    except Exception as e:
        logger.error(f"Error fetching persisted stats: {e}")
        return {
            'total_predictions': 0, 'fraud_detected': 0, 'fraud_rate': 0.0,
            'total_fraud_amount': 0.0, 'average_fraud_probability': 0.0,
            'risk_level_distribution': {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0, 'CRITICAL': 0}
        }

def get_rag_context_for_chatbot(query: str) -> Dict[str, Any]:
    """
    RAG Context Builder: Retrieve relevant logged data for Chatbot Q&A based on user query.
    Extracts transaction IDs, stats summaries, recent alerts, and analyst feedback.
    """
    context = {
        'stats_summary': get_persisted_stats(),
        'recent_fraud_alerts': [],
        'queried_transaction': None,
        'recent_feedback': []
    }
    
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            
            # 1. Fetch recent fraud transactions (last 10 flagged)
            cur.execute("""
            SELECT transaction_id, step, type, amount, fraud_probability, risk_level, timestamp
            FROM predictions WHERE is_fraud = 1 ORDER BY id DESC LIMIT 10;
            """)
            context['recent_fraud_alerts'] = [dict(r) for r in cur.fetchall()]

            # 2. Check if a specific transaction ID or UUID is mentioned in query
            words = query.replace(",", " ").replace(":", " ").split()
            for word in words:
                word_clean = word.strip().strip("'\"")
                if len(word_clean) >= 6:
                    cur.execute("""
                    SELECT p.*, e.explanation_text
                    FROM predictions p
                    LEFT JOIN explanations e ON p.transaction_id = e.transaction_id
                    WHERE p.transaction_id LIKE ? LIMIT 1;
                    """, (f"%{word_clean}%",))
                    row = cur.fetchone()
                    if row:
                        context['queried_transaction'] = dict(row)
                        break

            # 3. Fetch recent analyst feedback
            cur.execute("""
            SELECT transaction_id, feedback_label, notes, timestamp
            FROM analyst_feedback ORDER BY id DESC LIMIT 5;
            """)
            context['recent_feedback'] = [dict(r) for r in cur.fetchall()]

    except Exception as e:
        logger.error(f"Error fetching RAG context: {e}")
        
    return context
