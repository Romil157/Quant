"""Unit tests for FastAPI security, auth, middleware, and guardrails."""
import asyncio

import pytest
from fastapi import HTTPException, Request, status
from fastapi.testclient import TestClient

from quant.production.api import (
    _app_state,
    create_app,
    validate_request_bounds,
    verify_api_key,
)
from quant.production.config import ProductionConfig, SecurityConfig, reset_config_cache


def test_unauthenticated_request_rejected():
    """Test that requests missing required API key header raise HTTP 401."""
    reset_config_cache()
    config = ProductionConfig()
    config.security = SecurityConfig(api_key="test-secret-key-123", api_key_header="X-API-Key", require_auth=True)
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
    config.security = SecurityConfig(api_key="test-secret-key-123", api_key_header="X-API-Key", require_auth=True)
    _app_state["config"] = config

    headers = [(b"x-api-key", b"test-secret-key-123")]
    scope = {"type": "http", "headers": headers}
    req = Request(scope)

    asyncio.run(verify_api_key(req))
    reset_config_cache()


def test_fail_closed_when_require_auth_true_and_key_unset():
    """Test that creating app with require_auth=True and no API key raises RuntimeError at startup."""
    config = ProductionConfig(environment="development")
    config.security.require_auth = True
    config.security.api_key = ""
    with pytest.raises(RuntimeError) as exc_info:
        create_app(config)
    assert "API key authentication must be configured when require_auth=True" in str(exc_info.value)


def test_dev_environment_explicitly_disabled_auth():
    """Test that creating app with require_auth=False in development succeeds without API key."""
    config = ProductionConfig(environment="development")
    config.security.require_auth = False
    config.security.api_key = ""
    app = create_app(config)
    assert app.title == "Quant Platform API"


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
    config = ProductionConfig(environment="production")
    config.security.api_key = ""
    with pytest.raises(RuntimeError) as exc_info:
        create_app(config)
    assert "API key authentication must be configured" in str(exc_info.value)


def test_security_headers_present_on_responses():
    """Test that security headers and X-Request-ID are attached to HTTP responses."""
    config = ProductionConfig(environment="development")
    config.security.require_auth = False
    app = create_app(config)
    client = TestClient(app)

    response = client.get("/health")
    assert response.status_code == 200
    assert response.headers.get("X-Content-Type-Options") == "nosniff"
    assert response.headers.get("X-Frame-Options") == "DENY"
    assert response.headers.get("X-XSS-Protection") == "1; mode=block"
    assert "X-Request-ID" in response.headers
    assert "X-RateLimit-Limit" in response.headers


def test_rate_limiting_middleware():
    """Test that excessive requests trigger HTTP 429 Too Many Requests."""
    from quant.production.api import reset_rate_limits
    reset_rate_limits()

    config = ProductionConfig(environment="development")
    config.security.require_auth = False
    config.api.rate_limit = 5  # Set small limit for testing
    app = create_app(config)
    client = TestClient(app)

    # 5 requests should succeed
    for _ in range(5):
        resp = client.get("/live")
        assert resp.status_code == 200

    # 6th request should be rate limited
    resp6 = client.get("/live")
    assert resp6.status_code == 429
    assert resp6.json().get("detail") == "Rate limit exceeded. Try again later."
    assert "X-RateLimit-Reset" in resp6.headers
    reset_rate_limits()


def test_generic_error_messages_returned_to_clients(monkeypatch):
    """Test that unexpected server exceptions return generic error detail rather than raw traceback/exception string."""
    from quant.production.api import reset_rate_limits
    reset_rate_limits()

    config = ProductionConfig(environment="development")
    config.security.require_auth = False
    config.security.api_key = ""
    app = create_app(config)
    client = TestClient(app)

    # Force download_data to raise an internal exception with sensitive string
    def boom(*_args, **_kwargs):
        raise RuntimeError("simulated secret internal server failure 0xDEADBEEF")

    import quant.production.api as api_module
    monkeypatch.setattr(api_module, "download_data", boom)

    response = client.post(
        "/api/v1/ml/compare",
        json={"symbols": ["AAPL"], "start_date": "2023-01-01", "end_date": "2023-06-01"},
    )


    assert response.status_code == 500
    data = response.json()
    assert data["detail"] == "Internal server error, see logs"
    assert "request_id" in data
    # The raw exception text must NOT leak to client
    assert "0xDEADBEEF" not in str(data)


def test_production_allowed_hosts_wildcard_rejected():
    """Test that production environment rejects wildcard allowed_hosts."""
    config = ProductionConfig(environment="production")
    config.security.api_key = "secret"
    config.security.allowed_hosts = ["*"]
    with pytest.raises(ValueError) as exc_info:
        create_app(config)
    assert "allowed_hosts must be explicitly configured in production" in str(exc_info.value)


def test_production_explicit_allowed_hosts_accepted():
    """Test that production environment accepts explicit allowed_hosts."""
    config = ProductionConfig(environment="production")
    config.security.api_key = "secret"
    config.security.allowed_hosts = ["api.quant.com", "localhost", "testserver"]
    app = create_app(config)
    assert app.title == "Quant Platform API"


def test_production_cors_wildcard_rejected():
    """Test that production environment rejects wildcard cors_origins when allow_credentials=True."""
    config = ProductionConfig(environment="production")
    config.security.api_key = "secret"
    config.security.allowed_hosts = ["api.quant.com"]
    config.api.cors_origins = ["*"]
    with pytest.raises(ValueError) as exc_info:
        create_app(config)
    assert "cors_origins must be explicitly configured" in str(exc_info.value)


def test_production_cors_empty_rejected():
    """Test that production environment rejects empty cors_origins."""
    config = ProductionConfig(environment="production")
    config.security.api_key = "secret"
    config.security.allowed_hosts = ["api.quant.com"]
    config.api.cors_origins = []
    with pytest.raises(ValueError) as exc_info:
        create_app(config)
    assert "cors_origins must be explicitly configured" in str(exc_info.value)


def test_production_cors_explicit_accepted():
    """Test that production environment accepts explicit cors_origins."""
    config = ProductionConfig(environment="production")
    config.security.api_key = "secret"
    config.security.allowed_hosts = ["api.quant.com"]
    config.api.cors_origins = ["https://dashboard.quant.com"]
    app = create_app(config)
    assert app.title == "Quant Platform API"







