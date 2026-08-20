"""Smoke tests for the Phase 4 FastAPI skeleton (no GPU required)."""

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from jarvis.inference.api import app  # noqa: E402


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def test_health_works_without_model(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is False


def test_toggle_off_locks_chat(client):
    assert client.post("/twin/toggle", json={"enabled": False}).status_code == 200
    r = client.post("/chat", json={"message": "γεια"})
    assert r.status_code == 423                     # locked — the safety switch works
    client.post("/twin/toggle", json={"enabled": True})


def test_chat_returns_503_without_runtime(client):
    """On GPU-less machines the skeleton must fail gracefully, not crash."""
    r = client.post("/chat", json={"message": "γεια"})
    assert r.status_code in (200, 503)
