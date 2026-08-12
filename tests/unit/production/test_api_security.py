"""Unit tests for FastAPI security, auth, and guardrails."""
import asyncio

import pytest
from fastapi import HTTPException, Request, status

from quant.production.api import (
    _app_state,
    validate_request_bounds,
    verify_api_key,
)
from quant.production.config import ProductionConfig, SecurityConfig, reset_config_cache


def test_unauthenticated_request_rejected():
    """Test that requests missing required API key header raise HTTP 401."""
    reset_config_cache()
    config = ProductionConfig()
    config.security = SecurityConfig(api_key="test-secret-key-123", api_key_header="X-API-Key")
    _app_state["config"] = config

    scope = {"type": "http", "headers": []}
    req = Request(scope)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(verify_api_key(req))

    assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
    assert exc_info.value.detail == "Invalid or missing API Key"
    reset_config_cache()


def test_authenticated_request_accepted():
    """Test that request with valid X-API-Key header passes verification."""
    reset_config_cache()
    config = ProductionConfig()
    config.security = SecurityConfig(api_key="test-secret-key-123", api_key_header="X-API-Key")
    _app_state["config"] = config

    headers = [(b"x-api-key", b"test-secret-key-123")]
    scope = {"type": "http", "headers": headers}
    req = Request(scope)

    asyncio.run(verify_api_key(req))
    reset_config_cache()


def test_oversized_symbol_list_rejected():
    """Test that symbol count > 50 raises HTTP 422."""
    symbols = [f"SYM{i}" for i in range(55)]
    with pytest.raises(HTTPException) as exc_info:
        validate_request_bounds(symbols, "2023-01-01", "2023-06-01")

    assert exc_info.value.status_code == 422
    assert "Symbol count exceeds maximum limit" in exc_info.value.detail


def test_excessive_date_range_rejected():
    """Test that date range > 3650 days raises HTTP 422."""
    with pytest.raises(HTTPException) as exc_info:
        validate_request_bounds(["AAPL"], "2000-01-01", "2025-01-01")

    assert exc_info.value.status_code == 422
    assert "Date range exceeds maximum limit" in exc_info.value.detail


def test_invalid_date_order_rejected():
    """Test that start_date >= end_date raises HTTP 422."""
    with pytest.raises(HTTPException) as exc_info:
        validate_request_bounds(["AAPL"], "2023-06-01", "2023-01-01")

    assert exc_info.value.status_code == 422
    assert "start_date must be earlier than end_date" in exc_info.value.detail
