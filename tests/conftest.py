"""Shared test fixtures."""

from __future__ import annotations

import os
import tempfile

import pytest
import pytest_asyncio

from app import db


@pytest.fixture(autouse=True, scope="session")
def _init_db_for_sync_tests():
    """Ensure DB is initialised for sync TestClient tests."""
    import asyncio
    tmp = tempfile.mkdtemp()
    db_path = os.path.join(tmp, "test.db")
    asyncio.get_event_loop_policy().new_event_loop().run_until_complete(db.init_db(db_path))
    yield
