"""Production configuration management."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path


@dataclass
class DatabaseConfig:
    """Database configuration."""
    host: str = "localhost"
    port: int = 5432
    database: str = "quant"
    user: str = "quant"
    password: str = ""
    pool_size: int = 10
    max_overflow: int = 20

    @property
    def url(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    def __repr__(self) -> str:
        pwd_str = "'***'" if self.password else "''"
        return (
            f"DatabaseConfig(host={self.host!r}, port={self.port!r}, database={self.database!r}, "
            f"user={self.user!r}, password={pwd_str}, pool_size={self.pool_size!r}, "
            f"max_overflow={self.max_overflow!r})"
        )

    def __str__(self) -> str:
        return repr(self)


@dataclass
class RedisConfig:
    """Redis configuration."""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: str | None = None
    max_connections: int = 50

    @property
    def url(self) -> str:
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"

    def __repr__(self) -> str:
        pwd_str = "'***'" if self.password else "None"
        return (
            f"RedisConfig(host={self.host!r}, port={self.port!r}, db={self.db!r}, "
            f"password={pwd_str}, max_connections={self.max_connections!r})"
        )

    def __str__(self) -> str:
        return repr(self)


@dataclass
class APIConfig:
    """API server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    timeout: int = 30
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    cors_origins: list[str] = field(default_factory=lambda: ["http://localhost:3000", "http://localhost:8000"])
    rate_limit: int = 100  # requests per minute
    enable_docs: bool = True



@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration."""
    enable_metrics: bool = True
    metrics_port: int = 9090
    enable_tracing: bool = False
    tracing_sample_rate: float = 0.1
    log_level: str = "INFO"
    log_format: str = "json"
    log_file: str | None = None
    sentry_dsn: str | None = None
    prometheus_multiproc_dir: str | None = None

    def __repr__(self) -> str:
        dsn_str = "'***'" if self.sentry_dsn else "None"
        return (
            f"MonitoringConfig(enable_metrics={self.enable_metrics!r}, metrics_port={self.metrics_port!r}, "
            f"enable_tracing={self.enable_tracing!r}, tracing_sample_rate={self.tracing_sample_rate!r}, "
            f"log_level={self.log_level!r}, log_format={self.log_format!r}, log_file={self.log_file!r}, "
            f"sentry_dsn={dsn_str}, prometheus_multiproc_dir={self.prometheus_multiproc_dir!r})"
        )

    def __str__(self) -> str:
        return repr(self)


@dataclass
class SchedulerConfig:
    """Job scheduler configuration."""
    enabled: bool = True
    timezone: str = "UTC"
    max_concurrent_jobs: int = 10
    job_defaults: dict = field(default_factory=lambda: {
        "coalesce": True,
        "max_instances": 1,
        "misfire_grace_time": 300,
    })
    persistent: bool = True
    jobstore_url: str | None = None


@dataclass
class StorageConfig:
    """Storage configuration."""
    data_root: Path = Path("data")
    raw_root: Path = Path("data/raw")
    processed_root: Path = Path("data/processed")
    cache_root: Path = Path("data/cache")
    reports_root: Path = Path("reports")
    models_root: Path = Path("models")

    def __post_init__(self):
        for path in [self.raw_root, self.processed_root, self.cache_root,
                     self.reports_root, self.models_root]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class SecurityConfig:
    """Security configuration."""
    api_key: str = ""
    secret_key: str = ""
    api_key_header: str = "X-API-Key"
    jwt_algorithm: str = "HS256"
    jwt_expiration_minutes: int = 60
    allowed_hosts: list[str] = field(default_factory=lambda: ["*"])
    ssl_certfile: str | None = None
    ssl_keyfile: str | None = None

    def __repr__(self) -> str:
        key_str = "'***'" if self.api_key else "''"
        sec_str = "'***'" if self.secret_key else "''"
        return (
            f"SecurityConfig(api_key={key_str}, secret_key={sec_str}, api_key_header={self.api_key_header!r}, "
            f"jwt_algorithm={self.jwt_algorithm!r}, jwt_expiration_minutes={self.jwt_expiration_minutes!r}, "
            f"allowed_hosts={self.allowed_hosts!r}, ssl_certfile={self.ssl_certfile!r}, "
            f"ssl_keyfile={self.ssl_keyfile!r})"
        )

    def __str__(self) -> str:
        return repr(self)


@dataclass
class ProductionConfig:
    """Complete production configuration."""
    environment: str = "development"
    debug: bool = False


    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    api: APIConfig = field(default_factory=APIConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)

    # Feature flags
    enable_backtesting: bool = True
    enable_ml: bool = True
    enable_research: bool = True

    def to_dict(self) -> dict:
        """Convert to dictionary (excluding secrets)."""
        import dataclasses

        def convert(obj):
            if dataclasses.is_dataclass(obj):
                result = {}
                for f in dataclasses.fields(obj):
                    value = getattr(obj, f.name)
                    if f.name in ('password', 'secret_key', 'sentry_dsn', 'api_key'):
                        result[f.name] = "***"
                    elif isinstance(value, Path):
                        result[f.name] = str(value)
                    else:
                        result[f.name] = convert(value)
                return result
            elif isinstance(obj, list):
                return [convert(v) for v in obj]
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            else:
                return obj

        res = convert(self)
        return dict(res) if isinstance(res, dict) else {}


    def save(self, path: Path) -> None:
        """Save configuration to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


def load_config_from_env(config: ProductionConfig) -> ProductionConfig:
    """Override config with environment variables."""
    # Database
    if db_host := os.getenv("DB_HOST"):
        config.database.host = db_host
    if db_port := os.getenv("DB_PORT"):
        config.database.port = int(db_port)
    if db_name := os.getenv("DB_NAME"):
        config.database.database = db_name
    if db_user := os.getenv("DB_USER"):
        config.database.user = db_user
    if db_pass := os.getenv("DB_PASSWORD"):
        config.database.password = db_pass

    # Redis
    if redis_host := os.getenv("REDIS_HOST"):
        config.redis.host = redis_host
    if redis_port := os.getenv("REDIS_PORT"):
        config.redis.port = int(redis_port)

    # API
    if api_port := os.getenv("API_PORT"):
        config.api.port = int(api_port)
    if api_host := os.getenv("API_HOST"):
        config.api.host = api_host
    if cors_origins := os.getenv("CORS_ORIGINS"):
        config.api.cors_origins = [s.strip() for s in cors_origins.split(",") if s.strip()]

    # Security
    if api_key := (os.getenv("QUANT_API_KEY") or os.getenv("API_KEY")):
        config.security.api_key = api_key
    if secret_key := os.getenv("SECRET_KEY"):
        config.security.secret_key = secret_key
    if sentry_dsn := os.getenv("SENTRY_DSN"):
        config.monitoring.sentry_dsn = sentry_dsn

    # Environment
    if env := os.getenv("ENVIRONMENT"):
        config.environment = env
    if debug := os.getenv("DEBUG"):
        config.debug = debug.lower() == "true"

    return config



def load_config_from_file(path: Path, config: ProductionConfig) -> ProductionConfig:
    """Load configuration from JSON file."""
    if not path.exists():
        return config

    with open(path) as f:
        data = json.load(f)

    # Apply loaded config (simplified - would need recursive merge)
    if "database" in data:
        for k, v in data["database"].items():
            if hasattr(config.database, k) and v != "***":
                setattr(config.database, k, v)

    if "redis" in data:
        for k, v in data["redis"].items():
            if hasattr(config.redis, k) and v != "***":
                setattr(config.redis, k, v)

    if "api" in data:
        for k, v in data["api"].items():
            if hasattr(config.api, k):
                setattr(config.api, k, v)

    if "monitoring" in data:
        for k, v in data["monitoring"].items():
            if hasattr(config.monitoring, k) and v != "***":
                setattr(config.monitoring, k, v)

    if "scheduler" in data:
        for k, v in data["scheduler"].items():
            if hasattr(config.scheduler, k):
                setattr(config.scheduler, k, v)

    if "security" in data:
        for k, v in data["security"].items():
            if hasattr(config.security, k) and v != "***":
                setattr(config.security, k, v)

    if "environment" in data:
        config.environment = data["environment"]
    if "debug" in data:
        config.debug = data["debug"]

    return config


@lru_cache
def get_config() -> ProductionConfig:
    """Get singleton production configuration."""
    config = ProductionConfig()
    config = load_config_from_env(config)

    # Try to load from config file
    config_path = Path("config/production.json")
    if config_path.exists():
        config = load_config_from_file(config_path, config)

    return config


def reset_config_cache() -> None:
    """Reset configuration cache (for testing)."""
    get_config.cache_clear()
