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


def test_production_fail_closed_without_api_key():
    """Test that creating app in production without API key raises RuntimeError at startup."""
    from quant.production.api import create_app
    config = ProductionConfig(environment="production")
    config.security.api_key = ""
    with pytest.raises(RuntimeError) as exc_info:
        create_app(config)
    assert "API key authentication must be configured in production" in str(exc_info.value)


def test_dev_environment_auth_optional():
    """Test that creating app in development environment without API key succeeds."""
    from quant.production.api import create_app
    config = ProductionConfig(environment="development")
    config.security.api_key = ""
    app = create_app(config)
    assert app.title == "Quant Platform API"


def test_generic_error_messages_returned_to_clients():
    """Test that unexpected server exceptions return generic error detail rather than raw traceback/exception string."""
    from quant.production.api import MLExperimentRequest, create_app

    config = ProductionConfig(environment="development")
    config.security.api_key = ""
    app = create_app(config)
    compare_route = next(r for r in app.routes if getattr(r, "path", None) == "/api/v1/ml/compare")
    req = MLExperimentRequest(symbols=["INVALID_SYM"], start_date="2023-01-01", end_date="2023-06-01")
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(compare_route.endpoint(req))

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Internal server error, see logs"


def test_production_allowed_hosts_wildcard_rejected():
    """Test that production environment rejects wildcard allowed_hosts."""
    from quant.production.api import create_app
    config = ProductionConfig(environment="production")
    config.security.api_key = "secret"
    config.security.allowed_hosts = ["*"]
    with pytest.raises(ValueError) as exc_info:
        create_app(config)
    assert "allowed_hosts must be explicitly configured in production" in str(exc_info.value)


def test_production_explicit_allowed_hosts_accepted():
    """Test that production environment accepts explicit allowed_hosts."""
    from quant.production.api import create_app
    config = ProductionConfig(environment="production")
    config.security.api_key = "secret"
    config.security.allowed_hosts = ["api.quant.com", "localhost"]
    app = create_app(config)
    assert app.title == "Quant Platform API"


def test_production_cors_wildcard_rejected():
    """Test that production environment rejects wildcard cors_origins when allow_credentials=True."""
    from quant.production.api import create_app
    config = ProductionConfig(environment="production")
    config.security.api_key = "secret"
    config.security.allowed_hosts = ["api.quant.com"]
    config.api.cors_origins = ["*"]
    with pytest.raises(ValueError) as exc_info:
        create_app(config)
    assert "cors_origins must be explicitly configured" in str(exc_info.value)


def test_production_cors_empty_rejected():
    """Test that production environment rejects empty cors_origins."""
    from quant.production.api import create_app
    config = ProductionConfig(environment="production")
    config.security.api_key = "secret"
    config.security.allowed_hosts = ["api.quant.com"]
    config.api.cors_origins = []
    with pytest.raises(ValueError) as exc_info:
        create_app(config)
    assert "cors_origins must be explicitly configured" in str(exc_info.value)


def test_production_cors_explicit_accepted():
    """Test that production environment accepts explicit cors_origins."""
    from quant.production.api import create_app
    config = ProductionConfig(environment="production")
    config.security.api_key = "secret"
    config.security.allowed_hosts = ["api.quant.com"]
    config.api.cors_origins = ["https://dashboard.quant.com"]
    app = create_app(config)
    assert app.title == "Quant Platform API"






