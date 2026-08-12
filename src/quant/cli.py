"""Command line interface for the quant platform."""
from pathlib import Path
from typing import Annotated

import typer

from quant.config.loader import load_config

app = typer.Typer(help="Quant research platform CLI")


@app.command()
def hello() -> None:
    """Simple health check."""
    typer.echo("Quant platform ready.")


DEFAULT_CONFIG = Path("configs/development.yaml")


@app.command()
def show_config(config_path: Annotated[Path, typer.Option(help="Path to YAML config")] = DEFAULT_CONFIG) -> None:
    """Load and display configuration."""
    cfg = load_config(config_path)
    typer.echo(cfg.model_dump_json(indent=2))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
