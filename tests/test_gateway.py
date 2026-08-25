import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from fastapi.testclient import TestClient
import gateway
from gateway import app, resolve_engine_from_string, detect_engine, get_worker_url


def test_gateway_engine_resolution_disabled():
    """When Gemma is disabled (default), engine strings resolve to dots-ocr."""
    with patch("gateway.ENABLE_GEMMA", False):
        assert resolve_engine_from_string("dots-ocr") == "dots-ocr"
        assert resolve_engine_from_string("gemma-4") == "dots-ocr"
        assert resolve_engine_from_string("metadata") == "dots-ocr"
        assert resolve_engine_from_string(None) == "dots-ocr"


def test_gateway_engine_resolution_enabled():
    """When Gemma is enabled, gemma/metadata strings resolve to gemma-4."""
    with patch("gateway.ENABLE_GEMMA", True):
        assert resolve_engine_from_string("dots-ocr") == "dots-ocr"
        assert resolve_engine_from_string("gemma-4") == "gemma-4"
        assert resolve_engine_from_string("metadata") == "gemma-4"
        assert resolve_engine_from_string("archival") == "gemma-4"
        assert resolve_engine_from_string(None) == "dots-ocr"


def test_gateway_get_engines_disabled():
    """When Gemma is disabled, /v1/engines lists only dots-ocr."""
    client = TestClient(app)
    with patch("gateway.ENABLE_GEMMA", False):
        resp = client.get("/v1/engines")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["engines"] == ["dots-ocr"]
        assert "dots-ocr" in data["workers"]
        assert "gemma-4" not in data["workers"]
        assert data["gemma_enabled"] is False


def test_gateway_get_engines_enabled():
    """When Gemma is enabled, /v1/engines lists dots-ocr and gemma-4."""
    client = TestClient(app)
    with patch("gateway.ENABLE_GEMMA", True):
        resp = client.get("/v1/engines")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "dots-ocr" in data["engines"]
        assert "gemma-4" in data["engines"]
        assert data["gemma_enabled"] is True


def test_gateway_metadata_disabled_rejection():
    """When Gemma is disabled, /v1/metadata returns 503 Service Unavailable."""
    client = TestClient(app)
    with patch("gateway.ENABLE_GEMMA", False):
        resp = client.post("/v1/metadata", json={"some": "payload"})
        assert resp.status_code == 503
        assert "disabled" in resp.json()["detail"].lower()


def test_gateway_health_check_disabled_gemma():
    """When Gemma is disabled, querying /health for gemma returns 503 disabled."""
    client = TestClient(app)
    with patch("gateway.ENABLE_GEMMA", False):
        resp = client.get("/health?engine=gemma-4")
        assert resp.status_code == 503
        assert resp.json()["status"] == "disabled"
