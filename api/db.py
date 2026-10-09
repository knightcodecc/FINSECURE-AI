"""
db.py — Shared SQLite Connection Manager with WAL mode & Serving Data Hygiene
=============================================================================
Provides connection pooling, model version tracking, and schema migrations for high-throughput
prediction logging and drift snapshot storage.
"""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
import time
from typing import Dict, List, Any, Generator

DB_PATH = Path("data/finsecure.db")
MODEL_PATH = Path("models/xgboost_model.joblib")


def get_model_version(model_path: Path = MODEL_PATH) -> str:
    """Compute hash version of the deployed model file."""
    if not model_path.exists():
        return "v0_no_model"
    md5 = hashlib.md5(model_path.read_bytes()).hexdigest()[:12]
    return f"v_{md5}"


def init_db(db_path: Path = DB_PATH):
    """Ensure SQLite database tables exist and enforce model_version hygiene on startup."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    
    current_version = get_model_version()

    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                model_version TEXT,
                step INTEGER,
                name_orig TEXT,
                name_dest TEXT,
                amount REAL NOT NULL,
                prediction_proba REAL NOT NULL,
                is_fraud INTEGER NOT NULL,
                threshold_used REAL NOT NULL,
                features_json TEXT NOT NULL,
                shap_top_features_json TEXT,
                explanation TEXT
            );
            """
        )
        
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(predictions);")
        existing_cols = {col[1] for col in cursor.fetchall()}
        
        if "model_version" not in existing_cols:
            conn.execute("ALTER TABLE predictions ADD COLUMN model_version TEXT;")
        if "step" not in existing_cols:
            conn.execute("ALTER TABLE predictions ADD COLUMN step INTEGER;")
        if "name_orig" not in existing_cols:
            conn.execute("ALTER TABLE predictions ADD COLUMN name_orig TEXT;")
        if "name_dest" not in existing_cols:
            conn.execute("ALTER TABLE predictions ADD COLUMN name_dest TEXT;")

        # Serving Data Hygiene: check if stored log contains predictions from a different model_version
        cursor.execute("SELECT model_version FROM predictions WHERE model_version IS NOT NULL ORDER BY id DESC LIMIT 1;")
        row = cursor.fetchone()
        if row and row[0] != current_version:
            archive_table_name = f"predictions_archive_{int(time.time())}"
            print(f"[SERVING DATA HYGIENE]: Model version mismatch detected (old='{row[0]}', new='{current_version}'). Archiving predictions to {archive_table_name}...")
            conn.execute(f"ALTER TABLE predictions RENAME TO {archive_table_name};")
            conn.execute(
                """
                CREATE TABLE predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    step INTEGER,
                    name_orig TEXT,
                    name_dest TEXT,
                    amount REAL NOT NULL,
                    prediction_proba REAL NOT NULL,
                    is_fraud INTEGER NOT NULL,
                    threshold_used REAL NOT NULL,
                    features_json TEXT NOT NULL,
                    shap_top_features_json TEXT,
                    explanation TEXT
                );
                """
            )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS drift_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                overall_status TEXT NOT NULL,
                retrain_recommended INTEGER NOT NULL,
                score_psi REAL NOT NULL,
                drifted_feature_count INTEGER NOT NULL,
                metrics_json TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analyst_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                transaction_id INTEGER NOT NULL,
                confirmed_is_fraud INTEGER NOT NULL
            );
            """
        )
    conn.close()


@contextmanager
def get_db_connection(db_path: Path = DB_PATH) -> Generator[sqlite3.Connection, None, None]:
    """Context manager providing a SQLite connection with WAL mode enabled."""
    init_db(db_path)
    conn = sqlite3.connect(db_path, timeout=10.0)
    try:
        yield conn
    finally:
        conn.close()


def log_prediction(pred_data: Dict[str, Any], db_path: Path = DB_PATH):
    """Log a single prediction record with model_version into SQLite."""
    current_version = get_model_version()
    with get_db_connection(db_path) as conn:
        with conn:
            conn.execute(
                """
                INSERT INTO predictions (
                    timestamp, model_version, step, name_orig, name_dest, amount, prediction_proba,
                    is_fraud, threshold_used, features_json, shap_top_features_json, explanation
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pred_data.get("timestamp"),
                    current_version,
                    pred_data.get("step", 0),
                    pred_data.get("nameOrig", ""),
                    pred_data.get("nameDest", ""),
                    pred_data.get("amount", 0.0),
                    pred_data.get("prediction_proba", 0.0),
                    1 if pred_data.get("is_fraud") else 0,
                    pred_data.get("threshold_used", 0.5),
                    json.dumps(pred_data.get("features", {})),
                    json.dumps(pred_data.get("shap_top_features", [])),
                    pred_data.get("explanation", ""),
                ),
            )


def log_predictions_batch(batch_records: List[Dict[str, Any]], db_path: Path = DB_PATH):
    """Bulk log prediction records with model_version in a single SQLite transaction."""
    if not batch_records:
        return
    current_version = get_model_version()
    with get_db_connection(db_path) as conn:
        with conn:
            conn.executemany(
                """
                INSERT INTO predictions (
                    timestamp, model_version, step, name_orig, name_dest, amount, prediction_proba,
                    is_fraud, threshold_used, features_json, shap_top_features_json, explanation
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        r.get("timestamp"),
                        current_version,
                        r.get("step", 0),
                        r.get("nameOrig", ""),
                        r.get("nameDest", ""),
                        r.get("amount", 0.0),
                        r.get("prediction_proba", 0.0),
                        1 if r.get("is_fraud") else 0,
                        r.get("threshold_used", 0.5),
                        json.dumps(r.get("features", {})),
                        json.dumps(r.get("shap_top_features", [])),
                        r.get("explanation", ""),
                    )
                    for r in batch_records
                ],
            )
