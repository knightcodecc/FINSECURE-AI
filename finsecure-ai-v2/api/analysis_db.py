"""
analysis_db.py — Isolated SQLite Storage Manager for User CSV Analyses
========================================================================
Maintains isolated persistence for background analysis metadata:
- Analyses table with progress percentage, stage tracking, and ETA calculation.
- Role-based multi-tenant security: analysts view only their own analyses; admins view all.
- WAL mode with dynamic DATA_DIR resolution.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import time
from typing import Dict, List, Any, Optional, Generator

from api.db import get_data_dir


def get_analyses_db_path() -> Path:
    """Return path to finsecure_analyses.db inside configured DATA_DIR."""
    return get_data_dir() / "finsecure_analyses.db"


@contextmanager
def get_analyses_db_connection(db_path: Optional[Path] = None) -> Generator[sqlite3.Connection, None, None]:
    """Provide thread-safe connection to finsecure_analyses.db with WAL mode."""
    target_path = db_path or get_analyses_db_path()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(
        str(target_path),
        timeout=15.0,
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    try:
        yield conn
    finally:
        conn.close()


def init_analyses_db(db_path: Optional[Path] = None):
    """Initialize schema for finsecure_analyses.db."""
    with get_analyses_db_connection(db_path) as conn:
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS analyses (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    user_email TEXT,
                    filename TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    rows_total INTEGER DEFAULT 0,
                    rows_valid INTEGER DEFAULT 0,
                    rows_scored INTEGER DEFAULT 0,
                    status TEXT NOT NULL, -- queued, running, done, failed, cancelled
                    stage TEXT,          -- reading, validating, features, graph, scoring, aggregates, finishing
                    progress_pct REAL DEFAULT 0.0,
                    eta_seconds REAL DEFAULT 0.0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT,
                    model_version TEXT,
                    threshold REAL,
                    has_labels INTEGER DEFAULT 0,
                    params_json TEXT,
                    error TEXT
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_analyses_user ON analyses(user_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_analyses_status ON analyses(status);")


def create_analysis(
    analysis_id: str,
    user_id: int,
    user_email: str,
    filename: str,
    display_name: Optional[str] = None,
    params_json: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> str:
    """Create a new analysis record in queued status."""
    init_analyses_db(db_path)
    now = datetime.now(timezone.utc).isoformat()
    disp = display_name or filename

    with get_analyses_db_connection(db_path) as conn:
        with conn:
            conn.execute(
                """
                INSERT INTO analyses (
                    id, user_id, user_email, filename, display_name, status, stage,
                    progress_pct, eta_seconds, created_at, updated_at, params_json
                ) VALUES (?, ?, ?, ?, ?, 'uploaded', 'reading', 0.0, 0.0, ?, ?, ?)
                """,
                (analysis_id, user_id, user_email, filename, disp, now, now, params_json),
            )
    return analysis_id


def update_analysis_status(
    analysis_id: str,
    status: str,
    stage: Optional[str] = None,
    progress_pct: Optional[float] = None,
    eta_seconds: Optional[float] = None,
    error: Optional[str] = None,
    db_path: Optional[Path] = None,
):
    """Update progress, stage, ETA, or terminal status of an analysis."""
    now = datetime.now(timezone.utc).isoformat()
    updates = ["status = ?", "updated_at = ?"]
    params = [status, now]

    if stage is not None:
        updates.append("stage = ?")
        params.append(stage)
    if progress_pct is not None:
        updates.append("progress_pct = ?")
        params.append(progress_pct)
    if eta_seconds is not None:
        updates.append("eta_seconds = ?")
        params.append(eta_seconds)
    if error is not None:
        updates.append("error = ?")
        params.append(error)

    params.append(analysis_id)
    if status == "running":
        sql = f"UPDATE analyses SET {', '.join(updates)} WHERE id = ? AND status NOT IN ('cancelled', 'failed', 'done')"
    else:
        sql = f"UPDATE analyses SET {', '.join(updates)} WHERE id = ?"

    with get_analyses_db_connection(db_path) as conn:
        with conn:
            conn.execute(sql, params)


def update_analysis_results(
    analysis_id: str,
    rows_total: int,
    rows_valid: int,
    rows_scored: int,
    model_version: str,
    threshold: float,
    has_labels: bool,
    params_json: Optional[str] = None,
    db_path: Optional[Path] = None,
):
    """Store final dataset dimensions and model configuration upon completion."""
    now = datetime.now(timezone.utc).isoformat()
    with get_analyses_db_connection(db_path) as conn:
        with conn:
            conn.execute(
                """
                UPDATE analyses SET
                    rows_total = ?,
                    rows_valid = ?,
                    rows_scored = ?,
                    model_version = ?,
                    threshold = ?,
                    has_labels = ?,
                    status = 'done',
                    stage = 'finishing',
                    progress_pct = 100.0,
                    eta_seconds = 0.0,
                    updated_at = ?,
                    params_json = COALESCE(?, params_json)
                WHERE id = ?
                """,
                (
                    rows_total,
                    rows_valid,
                    rows_scored,
                    model_version,
                    threshold,
                    1 if has_labels else 0,
                    now,
                    params_json,
                    analysis_id,
                ),
            )


def get_analysis(analysis_id: str, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Retrieve an analysis record by unique ID."""
    init_analyses_db(db_path)
    with get_analyses_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,))
        row = cursor.fetchone()
        if not row:
            return None
        return dict(row)


def list_analyses(
    user_id: Optional[int] = None,
    is_admin: bool = False,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    List analyses:
    - Regular users see ONLY their own records.
    - Admins see all records across the organization.
    """
    init_analyses_db(db_path)
    with get_analyses_db_connection(db_path) as conn:
        cursor = conn.cursor()
        if is_admin:
            cursor.execute("SELECT * FROM analyses ORDER BY created_at DESC")
        else:
            cursor.execute("SELECT * FROM analyses WHERE user_id = ? ORDER BY created_at DESC", (user_id,))
        rows = cursor.fetchall()
        return [dict(r) for r in rows]


def rename_analysis(
    analysis_id: str,
    new_display_name: str,
    user_id: Optional[int] = None,
    is_admin: bool = False,
    db_path: Optional[Path] = None,
) -> bool:
    """Rename an analysis display name enforcing ownership."""
    init_analyses_db(db_path)
    now = datetime.now(timezone.utc).isoformat()
    with get_analyses_db_connection(db_path) as conn:
        with conn:
            if is_admin:
                cursor = conn.execute(
                    "UPDATE analyses SET display_name = ?, updated_at = ? WHERE id = ?",
                    (new_display_name, now, analysis_id),
                )
            else:
                cursor = conn.execute(
                    "UPDATE analyses SET display_name = ?, updated_at = ? WHERE id = ? AND user_id = ?",
                    (new_display_name, now, analysis_id, user_id),
                )
            return cursor.rowcount > 0


def delete_analysis(
    analysis_id: str,
    user_id: Optional[int] = None,
    is_admin: bool = False,
    db_path: Optional[Path] = None,
) -> bool:
    """Delete an analysis record from SQLite enforcing ownership."""
    init_analyses_db(db_path)
    with get_analyses_db_connection(db_path) as conn:
        with conn:
            if is_admin:
                cursor = conn.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
            else:
                cursor = conn.execute("DELETE FROM analyses WHERE id = ? AND user_id = ?", (analysis_id, user_id))
            return cursor.rowcount > 0


def count_active_jobs(
    user_id: Optional[int] = None,
    exclude_analysis_id: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> int:
    """Count currently queued or running jobs (optionally filtered by user_id and excluding a specific analysis)."""
    init_analyses_db(db_path)
    with get_analyses_db_connection(db_path) as conn:
        cursor = conn.cursor()
        query = "SELECT COUNT(*) FROM analyses WHERE status IN ('queued', 'running')"
        params = []
        if user_id is not None:
            query += " AND user_id = ?"
            params.append(user_id)
        if exclude_analysis_id is not None:
            query += " AND id != ?"
            params.append(exclude_analysis_id)
        cursor.execute(query, params)
        return int(cursor.fetchone()[0])
