"""Configuration loading utilities."""
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    name: str
    version: str


class DataConfig(BaseModel):
    raw_root: Path
    processed_root: Path
    cache_root: Path
    metadata_root: Path


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "%(asctime)s %(levelname)s %(name)s: %(message)s"


class ProvidersConfig(BaseModel):
    mock: dict[str, Any] = Field(default_factory=dict)


class AppConfig(BaseModel):
    project: ProjectConfig
    data: DataConfig
    logging: LoggingConfig
    providers: ProvidersConfig


def load_config(path: Path) -> AppConfig:
    """Load YAML configuration file into AppConfig."""
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return AppConfig(**raw)
