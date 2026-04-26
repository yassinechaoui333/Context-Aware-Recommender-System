from __future__ import annotations

import typer

from src.data.features.pipeline import run_featurize
from src.data.movielens import run_preprocess

app = typer.Typer()


@app.callback()
def cli() -> None:
    pass


@app.command()
def preprocess() -> None:
    run_preprocess()


@app.command()
def featurize() -> None:
    run_featurize()


if __name__ == "__main__":
    app()
