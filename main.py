"""Unified Typer CLI for the Context-Aware Recommender System."""
from __future__ import annotations

import typer

app = typer.Typer(add_completion=False)


@app.callback()
def cli() -> None:
    """Context-Aware Recommender System CLI."""


@app.command()
def preprocess() -> None:
    """Download (if needed) and preprocess the MovieLens dataset."""
    from src.data.movielens import run_preprocess

    run_preprocess()


@app.command()
def featurize() -> None:
    """Run the full feature engineering pipeline (temporal, session, device)."""
    from src.data.features.pipeline import run_featurize

    run_featurize()


@app.command()
def train(
    config: str = typer.Option(..., "--config", "-c", help="Path to experiment YAML config"),
    smoke_test: bool = typer.Option(False, "--smoke-test", help="Run 1 epoch on 1000 rows"),
) -> None:
    """Train a model variant defined by a YAML config file."""
    from src.training.train import train as _train

    _train(config_path=config, smoke_test=smoke_test)


@app.command()
def evaluate(
    checkpoint: str = typer.Option(..., "--checkpoint", "-k", help="Path to .ckpt file"),
) -> None:
    """Evaluate a saved checkpoint on the test set."""
    typer.echo(f"[evaluate] Loading checkpoint: {checkpoint}")
    typer.echo("[evaluate] Not yet implemented — coming in Phase 5.")


@app.command()
def serve() -> None:
    """Start the FastAPI inference server."""
    typer.echo("[serve] Not yet implemented — coming in Phase 9.")


if __name__ == "__main__":
    app()
