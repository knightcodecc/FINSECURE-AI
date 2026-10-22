"""
conftest.py — Pytest Configuration & Test Fixtures for FinSecure AI v2.
=======================================================================
Provides test fixtures including JWT authentication token for protected API endpoints.
"""

import os
import shutil
import tempfile
from pathlib import Path
import pytest

# Ensure isolated temp DATA_DIR for all test runs
if "DATA_DIR" not in os.environ:
    _TEST_TEMP_DIR = tempfile.mkdtemp(prefix="finsecure_test_data_")
    os.environ["DATA_DIR"] = _TEST_TEMP_DIR

from api.auth import init_users_db, create_access_token


def pytest_sessionfinish(session, exitstatus):
    """Clean up temp DATA_DIR after test session if auto-generated."""
    temp_dir = os.environ.get("DATA_DIR")
    if temp_dir and "finsecure_test_data_" in temp_dir and Path(temp_dir).exists():
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="session", autouse=True)
def setup_users_database():
    """Ensure users database and default admin exist for tests."""
    init_users_db()


@pytest.fixture
def auth_token():
    """Generate a valid test JWT authentication token for admin."""
    init_users_db()
    return create_access_token({
        "sub": "admin@finsecure.ai",
        "id": 1,
        "name": "System Administrator",
        "role": "admin"
    })


@pytest.fixture
def auth_headers(auth_token):
    """Return Authorization header dictionary with valid Bearer token."""
    return {"Authorization": f"Bearer {auth_token}"}
