"""Unit tests for configuration loading."""
from pathlib import Path

from quant.config.loader import AppConfig, load_config


def test_load_development_config() -> None:
    cfg_path = Path("configs/development.yaml")
    assert cfg_path.exists()
    cfg = load_config(cfg_path)
    assert isinstance(cfg, AppConfig)
    assert cfg.project.name == "quant"
    assert cfg.logging.level == "DEBUG"


def test_production_config_repr_masks_secrets() -> None:
    from quant.production.config import (
        DatabaseConfig,
        MonitoringConfig,
        ProductionConfig,
        RedisConfig,
        SecurityConfig,
    )

    db_cfg = DatabaseConfig(password="supersecretpass")
    redis_cfg = RedisConfig(password="redissecretpass")
    sec_cfg = SecurityConfig(secret_key="myjwtsecretkey")
    mon_cfg = MonitoringConfig(sentry_dsn="https://secret@sentry.io/123")

    prod_cfg = ProductionConfig(
        database=db_cfg,
        redis=redis_cfg,
        security=sec_cfg,
        monitoring=mon_cfg,
    )

    repr_str = repr(prod_cfg)
    assert "supersecretpass" not in repr_str
    assert "redissecretpass" not in repr_str
    assert "myjwtsecretkey" not in repr_str
    assert "https://secret@sentry.io/123" not in repr_str

    assert "password='***'" in repr(db_cfg)
    assert "password='***'" in repr(redis_cfg)
    assert "secret_key='***'" in repr(sec_cfg)
    assert "sentry_dsn='***'" in repr(mon_cfg)

