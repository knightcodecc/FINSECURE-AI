"""
db.py — Shared SQLite Connection Manager with WAL mode & Serving Data Hygiene
=============================================================================
Provides connection pooling, model version tracking, and schema migrations for high-throughput
prediction logging and drift snapshot storage.
"""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
from typing import Dict, List, Any, Generator, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_data_dir() -> Path:
    """Return configured DATA_DIR (default: project_root/data) ensuring directory exists."""
    raw = os.getenv("DATA_DIR", "data")
    p = Path(raw)
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_db_path() -> Path:
    """Dynamically return the predictions database path under DATA_DIR."""
    return get_data_dir() / "finsecure.db"


DB_PATH = get_db_path()
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_model.joblib"


def get_model_version(model_path: Path = MODEL_PATH) -> str:
    """Compute hash version of the deployed model file."""
    if not model_path.exists():
        return "v0_no_model"
    md5 = hashlib.md5(model_path.read_bytes()).hexdigest()[:12]
    return f"v_{md5}"


def init_db(db_path: Optional[Path] = None):
    """Ensure SQLite database tables exist and enforce model_version hygiene on startup."""
    if db_path is None:
        db_path = get_db_path()
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
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS unmatched_queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                query TEXT NOT NULL,
                user_email TEXT
            );
            """
        )
    conn.close()


@contextmanager
def get_db_connection(db_path: Optional[Path] = None) -> Generator[sqlite3.Connection, None, None]:
    """Context manager providing a SQLite connection with WAL mode enabled."""
    if db_path is None:
        db_path = get_db_path()
    init_db(db_path)
    conn = sqlite3.connect(db_path, timeout=10.0)
    try:
        yield conn
    finally:
        conn.close()


def log_prediction(pred_data: Dict[str, Any], db_path: Optional[Path] = None) -> int:
    """Log a single prediction record with model_version into SQLite and return its inserted row ID."""
    if db_path is None:
        db_path = get_db_path()
    current_version = get_model_version()
    with get_db_connection(db_path) as conn:
        with conn:
            cursor = conn.cursor()
            cursor.execute(
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
            return cursor.lastrowid


def log_predictions_batch(batch_records: List[Dict[str, Any]], db_path: Optional[Path] = None):
    """Bulk log prediction records with model_version in a single SQLite transaction."""
    if not batch_records:
        return
    if db_path is None:
        db_path = get_db_path()
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


def log_drift_snapshot(
    drift_data: Dict[str, Any],
    is_simulation: bool = False,
    db_path: Optional[Path] = None,
) -> int:
    """Store concept drift snapshot in SQLite under DATA_DIR."""
    if db_path is None:
        db_path = get_db_path()
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).isoformat()
    overall_status = drift_data.get("overall_status", "STABLE")
    retrain_recommended = 1 if drift_data.get("retrain_recommended") else 0
    score_psi = float(drift_data.get("score_psi", 0.0))
    drifted_count = int(drift_data.get("drifted_feature_count", 0))
    metrics_payload = dict(drift_data)
    metrics_payload["is_simulation"] = is_simulation
    metrics_json = json.dumps(metrics_payload)

    with get_db_connection(db_path) as conn:
        with conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO drift_snapshots (
                    timestamp, overall_status, retrain_recommended, score_psi,
                    drifted_feature_count, metrics_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (ts, overall_status, retrain_recommended, score_psi, drifted_count, metrics_json),
            )
            return cursor.lastrowid


def log_unmatched_query(
    query: str,
    user_email: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> int:
    """Log unmatched chatbot query to database for gap analysis (no other user data logged)."""
    if db_path is None:
        db_path = get_db_path()
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).isoformat()
    with get_db_connection(db_path) as conn:
        with conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO unmatched_queries (timestamp, query, user_email)
                VALUES (?, ?, ?)
                """,
                (ts, query[:500], user_email or "anonymous"),
            )
            return cursor.lastrowid

