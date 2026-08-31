"""Command line interface for the AegisQuant platform."""
from pathlib import Path
from typing import Annotated

import typer

from quant.config.loader import load_config

app = typer.Typer(help="AegisQuant research platform CLI")


@app.command()
def hello() -> None:
    """Simple health check."""
    typer.echo("AegisQuant platform ready.")


DEFAULT_CONFIG = Path("configs/development.yaml")


@app.command()
def show_config(config_path: Annotated[Path, typer.Option(help="Path to YAML config")] = DEFAULT_CONFIG) -> None:
    """Load and display configuration."""
    cfg = load_config(config_path)
    typer.echo(cfg.model_dump_json(indent=2))


@app.callback(invoke_without_command=True)
def main_callback(ctx: typer.Context) -> None:
    """AegisQuant research platform CLI."""
    if ctx.invoked_subcommand is None:
        typer.echo("=========================================")
        typer.echo("AegisQuant Research Platform v0.2.0")
        typer.echo("=========================================")
        typer.echo("Platform status: Ready.")
        typer.echo("Run 'python -m quant --help' to list available commands.")


def main() -> None:
    app()


if __name__ == "__main__":
    main()

