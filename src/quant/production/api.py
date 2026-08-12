"""FastAPI server for production API."""
from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import asyncio

from fastapi import FastAPI, HTTPException, Request, Response, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

from quant.production.config import get_config, ProductionConfig, APIConfig
from quant.production.monitoring import (
    get_logger, get_metrics_collector, get_health_check,
    configure_logging, MetricsCollector, HealthCheck
)
from quant.backtest.engine import BacktestEngine, BacktestConfig
from quant.data import download_data, validate_data
from quant.ml import run_ml_experiment, compare_models


# Request/Response models
class BacktestRequest(BaseModel):
    strategy: str
    symbols: List[str]
    start_date: str
    end_date: str
    initial_capital: float = 100000
    parameters: Dict[str, Any] = Field(default_factory=dict)


class BacktestResponse(BaseModel):
    status: str
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: float


class MLExperimentRequest(BaseModel):
    model_name: str = "rf"
    task: str = "regression"
    symbols: List[str]
    start_date: str
    end_date: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    tune: bool = False


class MLExperimentResponse(BaseModel):
    status: str
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: float


class DataDownloadRequest(BaseModel):
    symbols: List[str]
    start_date: str
    end_date: str
    provider: str = "mock"


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    uptime_seconds: float
    checks: Dict[str, Any]


# Global state
_app_state: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    config = get_config()
    
    # Configure logging
    configure_logging(
        level=config.monitoring.log_level,
        format=config.monitoring.log_format,
        log_file=Path(config.monitoring.log_file) if config.monitoring.log_file else None,
    )
    
    logger = get_logger("api")
    logger.info("api_starting", environment=config.environment)
    
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


def create_app(config: Optional[ProductionConfig] = None) -> FastAPI:
    """Create FastAPI application."""
    if config is None:
        config = get_config()
    
    app = FastAPI(
        title="Quant Platform API",
        description="Quantitative research and backtesting platform",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs" if config.api.enable_docs else None,
        redoc_url="/redoc" if config.api.enable_docs else None,
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
    
    # Health check endpoints
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
    
    # Backtest endpoints
    @app.post("/api/v1/backtest", response_model=BacktestResponse)
    async def run_backtest(request: BacktestRequest, background_tasks: BackgroundTasks):
        """Run a backtest."""
        logger = _app_state.get("logger") or get_logger("api")
        metrics = _app_state.get("metrics") or get_metrics_collector()
        
        start_time = time.time()
        
        try:
            logger.info("backtest_started", strategy=request.strategy, symbols=request.symbols)
            
            # Parse dates
            start_date = datetime.fromisoformat(request.start_date)
            end_date = datetime.fromisoformat(request.end_date)
            
            # Download data
            data = download_data(
                symbols=request.symbols,
                start_date=request.start_date,
                end_date=request.end_date,
            )
            
            # Validate
            for symbol in request.symbols:
                validation = validate_data(symbol, start_date, end_date)
                if not validation["valid"]:
                    logger.warning("data_validation_failed", symbol=symbol, issues=validation["issues"])
            
            # Run backtest (simplified - would need strategy factory)
            bt_config = BacktestConfig(
                initial_capital=request.initial_capital,
                start_date=start_date,
                end_date=end_date,
            )
            
            # This is a placeholder - real implementation would use strategy factory
            engine = BacktestEngine(bt_config)
            # results = engine.run(data, strategy)
            results = {"placeholder": "backtest results"}
            
            duration = time.time() - start_time
            
            metrics.record_backtest(request.strategy, "success", duration)
            
            logger.info("backtest_completed", strategy=request.strategy, duration=duration)
            
            return BacktestResponse(
                status="success",
                results=results,
                execution_time=duration,
            )
            
        except Exception as e:
            duration = time.time() - start_time
            metrics.record_backtest(request.strategy, "error", duration)
            metrics.record_error("backtest", type(e).__name__)
            
            logger.error("backtest_failed", strategy=request.strategy, error=str(e))
            
            return BacktestResponse(
                status="error",
                error=str(e),
                execution_time=duration,
            )
    
    @app.post("/api/v1/ml/experiment", response_model=MLExperimentResponse)
    async def run_ml_experiment_endpoint(request: MLExperimentRequest):
        """Run ML experiment."""
        logger = _app_state.get("logger") or get_logger("api")
        metrics = _app_state.get("metrics") or get_metrics_collector()
        
        start_time = time.time()
        
        try:
            logger.info("ml_experiment_started", model=request.model_name, symbols=request.symbols)
            
            # Download data
            data = download_data(
                symbols=request.symbols,
                start_date=request.start_date,
                end_date=request.end_date,
            )
            
            # Run experiment
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
            
        except Exception as e:
            duration = time.time() - start_time
            metrics.record_ml_training(request.model_name, "error", duration)
            metrics.record_error("ml", type(e).__name__)
            
            logger.error("ml_experiment_failed", model=request.model_name, error=str(e))
            
            return MLExperimentResponse(
                status="error",
                error=str(e),
                execution_time=duration,
            )
    
    @app.post("/api/v1/ml/compare")
    async def compare_models_endpoint(request: MLExperimentRequest):
        """Compare multiple models."""
        logger = _app_state.get("logger") or get_logger("api")
        
        try:
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
            
        except Exception as e:
            logger.error("model_comparison_failed", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    # Data endpoints
    @app.post("/api/v1/data/download")
    async def download_data_endpoint(request: DataDownloadRequest):
        """Download market data."""
        logger = _app_state.get("logger") or get_logger("api")
        metrics = _app_state.get("metrics") or get_metrics_collector()
        
        try:
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
            
        except Exception as e:
            for symbol in request.symbols:
                metrics.record_data_download(request.provider, symbol, "error")
            logger.error("data_download_failed", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/api/v1/data/validate/{symbol}")
    async def validate_data_endpoint(symbol: str, start_date: str, end_date: str):
        """Validate data quality."""
        logger = _app_state.get("logger") or get_logger("api")
        
        try:
            start = datetime.fromisoformat(start_date)
            end = datetime.fromisoformat(end_date)
            
            result = validate_data(symbol, start, end)
            return result
            
        except Exception as e:
            logger.error("data_validation_failed", symbol=symbol, error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    # Config endpoint
    @app.get("/api/v1/config")
    async def get_config_endpoint():
        """Get current configuration (sanitized)."""
        config = _app_state.get("config") or get_config()
        return config.to_dict()
    
    return app


# Import time for middleware
import time


def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    workers: int = 1,
    config: Optional[ProductionConfig] = None,
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