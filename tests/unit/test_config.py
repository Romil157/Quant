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
