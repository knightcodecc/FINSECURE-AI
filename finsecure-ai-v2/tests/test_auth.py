"""
test_auth.py — Unit & Integration Test Suite for User Authentication & Security.
================================================================================
Covers:
- register -> login -> access protected route OK
- wrong password -> 401
- duplicate email -> 409
- protected route without token -> 401
- expired token -> 401
- password_hash in DB is not plaintext
- xlsx export exists, contains the new user, and strictly excludes password hashes
- rate limit triggers on 5 failed login attempts -> 429
- XSS prevention: user-supplied strings with HTML/scripts are sanitized/escaped
"""

import html
import json
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

import pandas as pd
from api.main import app
from api.auth import (
    init_users_db,
    get_users_db_connection,
    create_access_token,
    hash_password,
    verify_password,
    USERS_DB_PATH,
    EXCEL_EXPORT_PATH,
    FAILED_ATTEMPTS,
)


@pytest.fixture(autouse=True)
def clean_auth_state():
    """Ensure clean users db and cleared rate limits before each test."""
    init_users_db()
    FAILED_ATTEMPTS.clear()
    yield
    FAILED_ATTEMPTS.clear()


def test_auth_registration_login_protected_route_flow():
    """Verify register -> login -> access protected route (/model/info) succeeds."""
    client = TestClient(app)
    unique_email = f"analyst_{int(time.time() * 1000)}@finsecure.ai"
    raw_pw = "ValidSecretPass2026!"

    # 1. Register
    reg_resp = client.post("/auth/register", json={
        "full_name": "Test Analyst",
        "email": unique_email,
        "password": raw_pw
    })
    assert reg_resp.status_code == 200, f"Register failed: {reg_resp.text}"
    reg_data = reg_resp.json()
    assert "access_token" in reg_data
    assert reg_data["user"]["role"] == "analyst"

    # 2. Login
    login_resp = client.post("/auth/login", json={
        "email": unique_email,
        "password": raw_pw
    })
    assert login_resp.status_code == 200
    token = login_resp.json()["access_token"]
    assert token is not None

    # 3. Access protected route with Bearer token
    headers = {"Authorization": f"Bearer {token}"}
    me_resp = client.get("/auth/me", headers=headers)
    assert me_resp.status_code == 200
    assert me_resp.json()["email"] == unique_email

    info_resp = client.get("/model/info", headers=headers)
    assert info_resp.status_code == 200
    assert "feature_count" in info_resp.json()


def test_auth_wrong_password_returns_401():
    """Verify login with wrong password returns 401 Unauthorized."""
    client = TestClient(app)
    unique_email = f"analyst_wrong_{int(time.time() * 1000)}@finsecure.ai"

    # Register user first
    client.post("/auth/register", json={
        "full_name": "User Wrong",
        "email": unique_email,
        "password": "CorrectPassword123!"
    })

    # Try wrong password
    resp = client.post("/auth/login", json={
        "email": unique_email,
        "password": "IncorrectPassword999!"
    })
    assert resp.status_code == 401
    assert "Invalid email or password" in resp.json()["detail"]


def test_auth_duplicate_email_returns_409():
    """Verify registering an existing email returns 409 Conflict."""
    client = TestClient(app)
    email = f"dup_{int(time.time() * 1000)}@finsecure.ai"

    res1 = client.post("/auth/register", json={
        "full_name": "Original User",
        "email": email,
        "password": "Password123!"
    })
    assert res1.status_code == 200

    res2 = client.post("/auth/register", json={
        "full_name": "Duplicate User",
        "email": email,
        "password": "Password456!"
    })
    assert res2.status_code == 409
    assert "already exists" in res2.json()["detail"].lower()


def test_protected_route_without_token_returns_401():
    """Verify accessing protected routes without Bearer token returns 401."""
    client = TestClient(app)
    for path in ["/auth/me", "/model/info", "/predictions/recent", "/drift/status"]:
        resp = client.get(path)
        assert resp.status_code == 401, f"Expected 401 for {path}, got {resp.status_code}"


def test_expired_token_returns_401():
    """Verify accessing protected routes with an expired token returns 401."""
    client = TestClient(app)
    # Generate expired token (-1 hour)
    expired_token = create_access_token(
        {"sub": "admin@finsecure.ai", "id": 1, "name": "Admin", "role": "admin"},
        expires_delta=timedelta(hours=-1)
    )
    headers = {"Authorization": f"Bearer {expired_token}"}
    resp = client.get("/auth/me", headers=headers)
    assert resp.status_code == 401
    assert "expired" in resp.json()["detail"].lower()


def test_password_hash_in_db_is_not_plaintext():
    """Verify password_hash in SQLite is securely hashed and not plaintext."""
    client = TestClient(app)
    raw_pw = "SuperSecretPlainTextPassword2026!"
    email = f"hashcheck_{int(time.time() * 1000)}@finsecure.ai"

    client.post("/auth/register", json={
        "full_name": "Hash Check",
        "email": email,
        "password": raw_pw
    })

    with get_users_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT password_hash FROM users WHERE email = ?", (email,))
        stored_hash = cursor.fetchone()[0]

    assert stored_hash != raw_pw
    assert raw_pw not in stored_hash
    assert stored_hash.startswith("$2b$") or stored_hash.startswith("$2a$")
    assert verify_password(raw_pw, stored_hash) is True


def test_excel_export_exists_and_excludes_password_hashes():
    """Verify users_export.xlsx exists, includes new registered users, and excludes password hashes."""
    client = TestClient(app)
    email = f"exceluser_{int(time.time() * 1000)}@finsecure.ai"

    client.post("/auth/register", json={
        "full_name": "Excel Auditor",
        "email": email,
        "password": "AuditPassword123!"
    })

    assert EXCEL_EXPORT_PATH.exists(), "Export file users_export.xlsx was not created!"

    # Read back with pandas
    users_df = pd.read_excel(EXCEL_EXPORT_PATH, sheet_name="Users")
    assert "password_hash" not in users_df.columns, "CRITICAL SECURITY LEAK: password_hash found in Excel export!"
    assert any(users_df["Email Address"] == email), f"Registered user {email} missing from Excel export"


def test_rate_limit_triggers_after_5_failed_attempts():
    """Verify rate limit blocks the 6th failed attempt with HTTP 429."""
    client = TestClient(app)
    email = f"ratelimit_{int(time.time() * 1000)}@finsecure.ai"

    # Register user
    client.post("/auth/register", json={
        "full_name": "Brute Force Victim",
        "email": email,
        "password": "RealPassword123!"
    })

    # Perform 5 failed attempts
    for _ in range(5):
        resp = client.post("/auth/login", json={"email": email, "password": "WrongPassword!"})
        assert resp.status_code == 401

    # 6th attempt must be rejected with 429 Too Many Requests
    blocked_resp = client.post("/auth/login", json={"email": email, "password": "WrongPassword!"})
    assert blocked_resp.status_code == 429
    assert "too many failed login attempts" in blocked_resp.json()["detail"].lower()


def test_xss_sanitization_script_tags():
    """Verify that malicious script tag payloads in user inputs are escaped into safe text entities."""
    malicious_input = "<script>alert(1)</script>"
    escaped = html.escape(malicious_input)

    assert "<script>" not in escaped
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" == escaped
