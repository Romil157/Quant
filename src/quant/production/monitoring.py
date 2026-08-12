"""Structured logging and metrics for production."""
from __future__ import annotations

import json
import logging
import logging.config
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from functools import wraps

import structlog
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
from prometheus_client.core import REGISTRY


# Prometheus metrics
REQUEST_COUNT = Counter(
    'quant_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'quant_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint']
)

ACTIVE_CONNECTIONS = Gauge(
    'quant_active_connections',
    'Active connections'
)

BACKTEST_RUNS = Counter(
    'quant_backtest_runs_total',
    'Total backtest runs',
    ['strategy', 'status']
)

BACKTEST_DURATION = Histogram(
    'quant_backtest_duration_seconds',
    'Backtest execution time',
    ['strategy']
)

ML_TRAINING_RUNS = Counter(
    'quant_ml_training_runs_total',
    'Total ML training runs',
    ['model', 'status']
)

ML_TRAINING_DURATION = Histogram(
    'quant_ml_training_duration_seconds',
    'ML training duration',
    ['model']
)

DATA_DOWNLOADS = Counter(
    'quant_data_downloads_total',
    'Total data downloads',
    ['provider', 'symbol', 'status']
)

ERROR_COUNT = Counter(
    'quant_errors_total',
    'Total errors',
    ['component', 'error_type']
)


@dataclass
class LogContext:
    """Context for structured logging."""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    user_id: Optional[str] = None
    component: str = "quant"
    extra: Dict[str, Any] = field(default_factory=dict)


class StructuredLogger:
    """Wrapper for structured logging with context."""
    
    def __init__(self, name: str = "quant"):
        self.logger = structlog.get_logger(name)
        self._context: Optional[LogContext] = None
    
    @contextmanager
    def context(self, **kwargs):
        """Temporary context for logging."""
        old_context = self._context
        self._context = LogContext(**kwargs)
        try:
            yield self
        finally:
            self._context = old_context
    
    def bind(self, **kwargs) -> 'StructuredLogger':
        """Bind additional context."""
        new_logger = StructuredLogger(self.logger._context.get('name', 'quant'))
        if self._context:
            new_logger._context = LogContext(
                request_id=self._context.request_id,
                user_id=self._context.user_id,
                component=self._context.component,
                extra={**self._context.extra, **kwargs}
            )
        else:
            new_logger._context = LogContext(extra=kwargs)
        return new_logger
    
    def _log(self, level: str, event: str, **kwargs):
        """Internal log method."""
        if self._context:
            kwargs.update({
                'request_id': self._context.request_id,
                'user_id': self._context.user_id,
                'component': self._context.component,
                **self._context.extra,
            })
        getattr(self.logger, level)(event, **kwargs)
    
    def debug(self, event: str, **kwargs):
        self._log('debug', event, **kwargs)
    
    def info(self, event: str, **kwargs):
        self._log('info', event, **kwargs)
    
    def warning(self, event: str, **kwargs):
        self._log('warning', event, **kwargs)
    
    def error(self, event: str, **kwargs):
        self._log('error', event, **kwargs)
    
    def critical(self, event: str, **kwargs):
        self._log('critical', event, **kwargs)
    
    def exception(self, event: str, **kwargs):
        self._log('exception', event, **kwargs)


def configure_logging(
    level: str = "INFO",
    format: str = "json",
    log_file: Optional[Path] = None,
) -> None:
    """Configure structured logging."""
    
    # Configure stdlib logging
    handlers = ["console"]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append("file")
    
    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": structlog.stdlib.ProcessorFormatter,
                "processor": structlog.processors.JSONRenderer(),
            },
            "console": {
                "()": structlog.stdlib.ProcessorFormatter,
                "processor": structlog.dev.ConsoleRenderer(colors=True),
            },
        },
        "handlers": {
            "console": {
                "level": level,
                "class": "logging.StreamHandler",
                "formatter": format,
                "stream": sys.stdout,
            },
        },
        "loggers": {
            "quant": {
                "handlers": handlers,
                "level": level,
                "propagate": False,
            },
            "uvicorn": {
                "handlers": handlers,
                "level": level,
                "propagate": False,
            },
        },
        "root": {
            "handlers": handlers,
            "level": level,
        },
    }
    
    if log_file:
        logging_config["handlers"]["file"] = {
            "level": level,
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": format,
            "filename": str(log_file),
            "maxBytes": 10_000_000,
            "backupCount": 5,
        }
    
    logging.config.dictConfig(logging_config)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "quant") -> StructuredLogger:
    """Get structured logger instance."""
    return StructuredLogger(name)


@contextmanager
def log_context(**kwargs):
    """Context manager for adding context to logs."""
    logger = get_logger()
    with logger.context(**kwargs):
        yield logger


def log_execution_time(logger: StructuredLogger, operation: str):
    """Decorator to log execution time."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start
                logger.info(f"{operation}_completed", duration=duration)
                return result
            except Exception as e:
                duration = time.time() - start
                logger.error(f"{operation}_failed", duration=duration, error=str(e))
                raise
        return wrapper
    return decorator


async def log_execution_time_async(logger: StructuredLogger, operation: str):
    """Async decorator to log execution time."""
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start
                logger.info(f"{operation}_completed", duration=duration)
                return result
            except Exception as e:
                duration = time.time() - start
                logger.error(f"{operation}_failed", duration=duration, error=str(e))
                raise
        return wrapper
    return decorator


class MetricsCollector:
    """Collect and expose metrics."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or REGISTRY
    
    def record_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record HTTP request metrics."""
        REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
        REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)
    
    def record_backtest(self, strategy: str, status: str, duration: float):
        """Record backtest metrics."""
        BACKTEST_RUNS.labels(strategy=strategy, status=status).inc()
        BACKTEST_DURATION.labels(strategy=strategy).observe(duration)
    
    def record_ml_training(self, model: str, status: str, duration: float):
        """Record ML training metrics."""
        ML_TRAINING_RUNS.labels(model=model, status=status).inc()
        ML_TRAINING_DURATION.labels(model=model).observe(duration)
    
    def record_data_download(self, provider: str, symbol: str, status: str):
        """Record data download metrics."""
        DATA_DOWNLOADS.labels(provider=provider, symbol=symbol, status=status).inc()
    
    def record_error(self, component: str, error_type: str):
        """Record error metrics."""
        ERROR_COUNT.labels(component=component, error_type=error_type).inc()
    
    def set_active_connections(self, count: int):
        """Set active connections gauge."""
        ACTIVE_CONNECTIONS.set(count)
    
    def get_metrics(self) -> bytes:
        """Get Prometheus metrics."""
        return generate_latest(self.registry)
    
    def get_metrics_text(self) -> str:
        """Get Prometheus metrics as text."""
        return self.get_metrics().decode('utf-8')


# Global metrics collector
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


class HealthCheck:
    """Health check utilities."""
    
    def __init__(self):
        self.checks: Dict[str, Callable] = {}
        self.start_time = time.time()
    
    def register(self, name: str, check: Callable[[], bool]):
        """Register a health check."""
        self.checks[name] = check
    
    def run_checks(self) -> Dict[str, Any]:
        """Run all health checks."""
        results = {}
        all_healthy = True
        
        for name, check in self.checks.items():
            try:
                healthy = check()
                results[name] = {"healthy": healthy, "error": None}
                if not healthy:
                    all_healthy = False
            except Exception as e:
                results[name] = {"healthy": False, "error": str(e)}
                all_healthy = False
        
        return {
            "status": "healthy" if all_healthy else "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_seconds": time.time() - self.start_time,
            "checks": results,
        }
    
    def run_check(self, name: str) -> Dict[str, Any]:
        """Run a specific health check."""
        if name not in self.checks:
            return {"healthy": False, "error": "Check not found"}
        
        try:
            healthy = self.checks[name]()
            return {"healthy": healthy, "error": None}
        except Exception as e:
            return {"healthy": False, "error": str(e)}


# Global health check
_health_check: Optional[HealthCheck] = None


def get_health_check() -> HealthCheck:
    """Get global health check."""
    global _health_check
    if _health_check is None:
        _health_check = HealthCheck()
    return _health_check