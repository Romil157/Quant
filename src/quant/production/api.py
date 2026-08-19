"""FastAPI server for production API."""
from __future__ import annotations

import hmac
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    HTTPException,
    Request,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from quant.backtest.engine import BacktestConfig, BacktestEngine
from quant.data import download_data, validate_data
from quant.ml import compare_models, run_ml_experiment
from quant.portfolio.construction import PortfolioConstraints
from quant.production.config import ProductionConfig, get_config
from quant.production.monitoring import (
    configure_logging,
    get_health_check,
    get_logger,
    get_metrics_collector,
    log_context,
)
from quant.strategies import STRATEGY_REGISTRY, create_strategy

# Input bounds constants
MAX_SYMBOLS = 50
MAX_DATE_RANGE_DAYS = 3650  # 10 years


def validate_request_bounds(symbols: list[str], start_date_str: str, end_date_str: str) -> None:
    """Validate request size and date range guardrails."""
    if len(symbols) > MAX_SYMBOLS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Symbol count exceeds maximum limit of {MAX_SYMBOLS}",
        )
    try:
        start = datetime.fromisoformat(start_date_str)
        end = datetime.fromisoformat(end_date_str)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid ISO date format for start_date or end_date",
        ) from e
    if start >= end:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="start_date must be earlier than end_date",
        )
    if (end - start).days > MAX_DATE_RANGE_DAYS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Date range exceeds maximum limit of {MAX_DATE_RANGE_DAYS} days",
        )


# Request/Response models
class BacktestRequest(BaseModel):
    strategy: str
    symbols: list[str]
    start_date: str
    end_date: str
    initial_capital: float = 100000
    parameters: dict[str, Any] = Field(default_factory=dict)
    provider: str = "mock"


class BacktestResponse(BaseModel):
    status: str
    results: dict[str, Any] | None = None
    error: str | None = None
    execution_time: float


class MLExperimentRequest(BaseModel):
    model_name: str = "rf"
    task: str = "regression"
    symbols: list[str]
    start_date: str
    end_date: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    tune: bool = False


class MLExperimentResponse(BaseModel):
    status: str
    results: dict[str, Any] | None = None
    error: str | None = None
    execution_time: float


class DataDownloadRequest(BaseModel):
    symbols: list[str]
    start_date: str
    end_date: str
    provider: str = "mock"


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    uptime_seconds: float
    checks: dict[str, Any]


# Global state
_app_state: dict[str, Any] = {}


async def verify_api_key(request: Request) -> None:
    """Verify API key authentication dependency for protected endpoints."""
    config = _app_state.get("config") or get_config()
    expected_key = config.security.api_key
    if not expected_key:
        if config.environment == "production":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Server configuration error: authentication required in production",
            )
        return  # Auth disabled if no QUANT_API_KEY is configured in dev/test

    header_name = config.security.api_key_header
    provided_key = request.headers.get(header_name)
    if not provided_key or not hmac.compare_digest(provided_key, expected_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    config = get_config()

    if config.environment == "production" and not config.security.api_key:
        raise RuntimeError("API key authentication must be configured in production environment (QUANT_API_KEY environment variable missing)")

    # Configure logging
    configure_logging(
        level=config.monitoring.log_level,
        format=config.monitoring.log_format,
        log_file=Path(config.monitoring.log_file) if config.monitoring.log_file else None,
    )

    logger = get_logger("api")
    logger.info("api_starting", environment=config.environment)

    if not config.security.api_key:
        logger.warning("api_key_auth_disabled", environment=config.environment, message="API Key authentication is disabled because QUANT_API_KEY is missing")

    # Initialize metrics
    metrics = get_metrics_collector()

    # Initialize health checks
    health = get_health_check()
    health.register("database", lambda: True)  # Placeholder
    health.register("redis", lambda: True)      # Placeholder
    health.register("storage", lambda: True)    # Placeholder

    _app_state["logger"] = logger
    _app_state["metrics"] = metrics
    _app_state["health"] = health
    _app_state["config"] = config

    logger.info("api_started", port=config.api.port)

    yield

    logger.info("api_shutting_down")


def create_app(config: ProductionConfig | None = None) -> FastAPI:
    """Create FastAPI application."""
    if config is None:
        config = get_config()

    if config.environment == "production":
        if not config.security.api_key:
            raise RuntimeError("API key authentication must be configured in production environment (QUANT_API_KEY environment variable missing)")
        if not config.security.allowed_hosts or "*" in config.security.allowed_hosts:
            raise ValueError("SecurityConfig.allowed_hosts must be explicitly configured in production (cannot contain wildcard '*')")
        if not config.api.cors_origins or "*" in config.api.cors_origins:
            raise ValueError("APIConfig.cors_origins must be explicitly configured (non-empty and no wildcard '*') when allow_credentials=True in production environment.")


    app = FastAPI(
        title="Quant Platform API",
        description="Quantitative research and backtesting platform",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs" if config.api.enable_docs else None,
        redoc_url="/redoc" if config.api.enable_docs else None,
    )

    # Trusted Host Middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=config.security.allowed_hosts,
    )

    # CORS

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        logger = _app_state.get("logger") or get_logger("api")
        metrics = _app_state.get("metrics") or get_metrics_collector()

        start_time = time.time()
        request_id = request.headers.get("X-Request-ID", "")

        with log_context(request_id=request_id, method=request.method, path=request.url.path):
            try:
                response = await call_next(request)
                duration = time.time() - start_time

                metrics.record_request(
                    method=request.method,
                    endpoint=request.url.path,
                    status=response.status_code,
                    duration=duration,
                )

                logger.info(
                    "request_completed",
                    method=request.method,
                    path=request.url.path,
                    status=response.status_code,
                    duration=duration,
                )

                return response
            except Exception as e:
                duration = time.time() - start_time
                metrics.record_error("api", type(e).__name__)
                logger.exception("request_failed", error=str(e))
                raise

    # Health check endpoints (Unauthenticated)
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        health = _app_state.get("health") or get_health_check()
        result = health.run_checks()
        return HealthResponse(**result)

    @app.get("/health/{check_name}")
    async def health_check_detail(check_name: str):
        """Detailed health check."""
        health = _app_state.get("health") or get_health_check()
        result = health.run_check(check_name)
        return JSONResponse(content=result)

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics_endpoint():
        """Prometheus metrics endpoint."""
        metrics = _app_state.get("metrics") or get_metrics_collector()
        return metrics.get_metrics_text()

    @app.get("/ready")
    async def readiness():
        """Readiness probe."""
        return {"status": "ready"}

    @app.get("/live")
    async def liveness():
        """Liveness probe."""
        return {"status": "alive"}

    # Protected Backtest endpoints
    @app.post(
        "/api/v1/backtest",
        response_model=BacktestResponse,
        dependencies=[Depends(verify_api_key)],
    )
    async def run_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
        """Run a backtest."""
        logger = _app_state.get("logger") or get_logger("api")
        metrics = _app_state.get("metrics") or get_metrics_collector()

        start_time = time.time()

        try:
            validate_request_bounds(request.symbols, request.start_date, request.end_date)
            logger.info("backtest_started", strategy=request.strategy, symbols=request.symbols)

            start_date = datetime.fromisoformat(request.start_date)
            end_date = datetime.fromisoformat(request.end_date)

            # Download data
            data = download_data(
                symbols=request.symbols,
                start_date=request.start_date,
                end_date=request.end_date,
                provider=request.provider,
            )

            # Validate
            for symbol in request.symbols:
                validation = validate_data(symbol, start_date, end_date)
                if not validation["valid"]:
                    logger.warning("data_validation_failed", symbol=symbol, issues=validation["issues"])

            bt_config = BacktestConfig(
                initial_capital=request.initial_capital,
                start_date=request.start_date,
                end_date=request.end_date,
            )
            bt_config.portfolio_constraints = PortfolioConstraints(
                max_position=1.0,
                max_gross_exposure=1.0,
                max_net_exposure=1.0,
                long_only=True,
            )

            if request.strategy not in STRATEGY_REGISTRY:
                valid = ", ".join(sorted(STRATEGY_REGISTRY))
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unknown strategy {request.strategy!r}. Valid: {valid}",
                )

            # Filter params to those accepted by the strategy's __init__ to
            # avoid brittle TypeErrors from mismatched JSON payloads.
            import inspect

            sig = inspect.signature(STRATEGY_REGISTRY[request.strategy].__init__)
            accepted = {
                n for n, p in sig.parameters.items()
                if n != "self" and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
            }
            filtered_params = {k: v for k, v in request.parameters.items() if k in accepted}

            try:
                strategy = create_strategy(request.strategy, **filtered_params)
            except TypeError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Strategy {request.strategy!r} rejected parameters {filtered_params}: {e}",
                ) from e

            engine = BacktestEngine(bt_config)
            engine.set_strategy(strategy)
            results = engine.run(data)

            results_payload = {
                "final_equity": float(results.get("final_equity", request.initial_capital)),
                "total_return": float(results.get("total_return", 0.0)),
                "max_drawdown_hit": bool(results.get("max_drawdown_hit", False)),
                "num_orders": len(results.get("orders", [])),
                "num_fills": len(results.get("fills", [])),
            }

            duration = time.time() - start_time
            metrics.record_backtest(request.strategy, "success", duration)
            logger.info("backtest_completed", strategy=request.strategy, duration=duration)

            return BacktestResponse(
                status="success",
                results=results_payload,
                execution_time=duration,
            )

        except HTTPException:
            raise
        except Exception as e:
            duration = time.time() - start_time
            metrics.record_backtest(request.strategy, "error", duration)
            metrics.record_error("backtest", type(e).__name__)
            logger.error("backtest_failed", strategy=request.strategy, error=str(e))

            return BacktestResponse(
                status="error",
                error="Internal server error, see logs",
                execution_time=duration,
            )

    @app.post(
        "/api/v1/ml/experiment",
        response_model=MLExperimentResponse,
        dependencies=[Depends(verify_api_key)],
    )
    async def run_ml_experiment_endpoint(request: MLExperimentRequest):
        """Run ML experiment."""
        logger = _app_state.get("logger") or get_logger("api")
        metrics = _app_state.get("metrics") or get_metrics_collector()

        start_time = time.time()

        try:
            validate_request_bounds(request.symbols, request.start_date, request.end_date)
            logger.info("ml_experiment_started", model=request.model_name, symbols=request.symbols)

            data = download_data(
                symbols=request.symbols,
                start_date=request.start_date,
                end_date=request.end_date,
            )

            result = run_ml_experiment(
                data=data,
                model_name=request.model_name,
                task=request.task,
                tune=request.tune,
            )

            duration = time.time() - start_time
            metrics.record_ml_training(request.model_name, "success", duration)
            logger.info("ml_experiment_completed", model=request.model_name, duration=duration)

            return MLExperimentResponse(
                status="success",
                results={
                    "test_metrics": result.test_metrics,
                    "cv_summary": {
                        "mean_test_score": result.cv_summary.mean_test_score,
                        "std_test_score": result.cv_summary.std_test_score,
                    } if result.cv_summary else None,
                    "feature_importance": result.feature_importance.to_dict(),
                },
                execution_time=duration,
            )

        except HTTPException:
            raise
        except Exception as e:
            duration = time.time() - start_time
            metrics.record_ml_training(request.model_name, "error", duration)
            metrics.record_error("ml", type(e).__name__)
            logger.error("ml_experiment_failed", model=request.model_name, error=str(e))

            return MLExperimentResponse(
                status="error",
                error="Internal server error, see logs",
                execution_time=duration,
            )

    @app.post(
        "/api/v1/ml/compare",
        dependencies=[Depends(verify_api_key)],
    )
    async def compare_models_endpoint(request: MLExperimentRequest):
        """Compare multiple models."""
        logger = _app_state.get("logger") or get_logger("api")

        try:
            validate_request_bounds(request.symbols, request.start_date, request.end_date)
            data = download_data(
                symbols=request.symbols,
                start_date=request.start_date,
                end_date=request.end_date,
            )

            model_names = ["linear", "ridge", "rf", "gbr"]
            results = compare_models(
                data=data,
                model_names=model_names,
                task=request.task,
            )

            return {"results": results.to_dict()}

        except HTTPException:
            raise
        except Exception as e:
            logger.error("model_comparison_failed", error=str(e))
            raise HTTPException(status_code=500, detail="Internal server error, see logs") from e

    # Data endpoints
    @app.post(
        "/api/v1/data/download",
        dependencies=[Depends(verify_api_key)],
    )
    async def download_data_endpoint(request: DataDownloadRequest):
        """Download market data."""
        logger = _app_state.get("logger") or get_logger("api")
        metrics = _app_state.get("metrics") or get_metrics_collector()

        try:
            validate_request_bounds(request.symbols, request.start_date, request.end_date)
            data = download_data(
                symbols=request.symbols,
                start_date=request.start_date,
                end_date=request.end_date,
                provider=request.provider,
            )

            for symbol in request.symbols:
                metrics.record_data_download(request.provider, symbol, "success")

            return {
                "status": "success",
                "symbols": list(data.keys()),
                "date_range": f"{request.start_date} to {request.end_date}",
            }

        except HTTPException:
            raise
        except Exception as e:
            for symbol in request.symbols:
                metrics.record_data_download(request.provider, symbol, "error")
            logger.error("data_download_failed", error=str(e))
            raise HTTPException(status_code=500, detail="Internal server error, see logs") from e

    @app.get(
        "/api/v1/data/validate/{symbol}",
        dependencies=[Depends(verify_api_key)],
    )
    async def validate_data_endpoint(symbol: str, start_date: str, end_date: str):
        """Validate data quality."""
        logger = _app_state.get("logger") or get_logger("api")

        try:
            validate_request_bounds([symbol], start_date, end_date)
            start = datetime.fromisoformat(start_date)
            end = datetime.fromisoformat(end_date)

            result = validate_data(symbol, start, end)
            return result

        except HTTPException:
            raise
        except Exception as e:
            logger.error("data_validation_failed", symbol=symbol, error=str(e))
            raise HTTPException(status_code=500, detail="Internal server error, see logs") from e

    # Config endpoint
    @app.get(
        "/api/v1/config",
        dependencies=[Depends(verify_api_key)],
    )
    async def get_config_endpoint():
        """Get current configuration (sanitized)."""
        config = _app_state.get("config") or get_config()
        return config.to_dict()

    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    workers: int = 1,
    config: ProductionConfig | None = None,
):
    """Run the API server."""
    import uvicorn

    if config is None:
        config = get_config()

    uvicorn.run(
        "quant.production.api:create_app",
        host=host,
        port=port,
        workers=workers,
        factory=True,
        log_level=config.monitoring.log_level.lower(),
    )


if __name__ == "__main__":
    run_server()
