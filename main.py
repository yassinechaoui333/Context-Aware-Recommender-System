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
    from src.evaluation.evaluator import Evaluator
    from src.data.datamodule import RecSysDataModule
    from omegaconf import OmegaConf
    import json

    cfg = OmegaConf.load("params.yaml")
    dm = RecSysDataModule(params=OmegaConf.to_container(cfg, resolve=True))
    dm.setup("test")

    from src.models.context_ncf_attn import ContextNCFAttn
    import pickle

    with open("data/processed/encoders.pkl", "rb") as fh:
        enc = pickle.load(fh)
    n_users = len(enc["user"].classes_)
    n_items = len(enc["item"].classes_)

    model = ContextNCFAttn.load_from_checkpoint(
        checkpoint, n_users=n_users, n_items=n_items
    )

    evaluator = Evaluator(
        model=model,
        test_dataloader=dm.test_dataloader(),
        k_values=list(cfg.eval.k_values),
        n_items=n_items,
    )
    results = evaluator.run()
    typer.echo(evaluator.summary_table().to_string())

    import pathlib
    out = pathlib.Path("outputs/final_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    typer.echo(f"[evaluate] Results saved to {out}")


@app.command()
def export(
    checkpoint: str = typer.Option(
        "outputs/checkpoints/context_ncf_attn/best.ckpt",
        "--checkpoint",
        "-k",
        help="Path to ContextNCFAttn .ckpt file",
    ),
    out_dir: str = typer.Option("outputs", "--out-dir", "-o", help="Output directory"),
) -> None:
    """Export the best model to TorchScript and ONNX."""
    from src.api.model_export import export as _export

    _export(checkpoint_path=checkpoint, out_dir=out_dir)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="Bind host"),
    port: int = typer.Option(8000, "--port", "-p", help="Bind port"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload (dev mode)"),
) -> None:
    """Start the FastAPI inference server."""
    import uvicorn

    uvicorn.run(
        "src.api.app:app",
        host=host,
        port=port,
        reload=reload,
    )


@app.command()
def interpret(
    checkpoint: str = typer.Option(
        "outputs/checkpoints/context_ncf_attn/best.ckpt",
        "--checkpoint",
        "-k",
        help="Path to ContextNCFAttn .ckpt file",
    ),
    skip_shap: bool = typer.Option(False, "--skip-shap", help="Skip slow SHAP analysis"),
) -> None:
    """Run interpretability analysis (SHAP + gate heatmap)."""
    from src.interpretability.attention_viz import run_attention_viz

    typer.echo("[interpret] Generating gate heatmap …")
    run_attention_viz(checkpoint_path=checkpoint)

    if not skip_shap:
        from src.interpretability.shap_analysis import run_shap_analysis

        typer.echo("[interpret] Running SHAP analysis (this may take a few minutes) …")
        run_shap_analysis(checkpoint_path=checkpoint)
    else:
        typer.echo("[interpret] SHAP analysis skipped (--skip-shap).")


if __name__ == "__main__":
    app()
