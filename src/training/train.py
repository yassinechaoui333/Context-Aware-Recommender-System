"""Training entry point for all NCF model variants.

Usage
-----
Called via ``main.py train --config configs/ncf.yaml`` or programmatically
as ``from src.training.train import train; train("configs/ncf.yaml")``.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import mlflow
import pytorch_lightning as pl
from dotenv import load_dotenv
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# Load env vars (.env file may define MLFLOW_TRACKING_URI etc.)
load_dotenv()

# Model registry — maps config type strings to classes
from src.models.ncf import NCF
from src.models.context_ncf_late import ContextNCFLate
from src.models.context_ncf_early import ContextNCFEarly
from src.models.context_ncf_attn import ContextNCFAttn

MODEL_REGISTRY: dict = {
    "NCF": NCF,
    "ContextNCFLate": ContextNCFLate,
    "ContextNCFEarly": ContextNCFEarly,
    "ContextNCFAttn": ContextNCFAttn,
}


def train(config_path: str, smoke_test: bool = False) -> None:
    """Train a single model variant defined by *config_path*.

    Parameters
    ----------
    config_path:
        Path to a YAML experiment config (e.g. ``configs/ncf.yaml``).
    smoke_test:
        If ``True``, run 1 epoch on the first 1 000 rows of each split
        (used by CI to verify the pipeline end-to-end quickly).
    """
    # ── 1. Merge global params with experiment config ──────────────────────
    base_cfg = OmegaConf.load("params.yaml")
    exp_cfg = OmegaConf.load(config_path)
    cfg = OmegaConf.merge(base_cfg, exp_cfg)

    experiment_name: str = cfg.get("experiment_name", "unnamed_experiment")
    model_type: str = cfg.model.type
    context_dim: Optional[int] = cfg.model.get("context_dim", None)

    # ── 2. MLflow tracking ─────────────────────────────────────────────────
    tracking_uri = os.environ.get(
        "MLFLOW_TRACKING_URI", "outputs/logs/mlruns"
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("context-recsys")

    with mlflow.start_run(run_name=experiment_name):
        flat_params = OmegaConf.to_container(cfg, resolve=True)
        mlflow.log_params(_flatten(flat_params))

        # ── 3. DataModule ──────────────────────────────────────────────────
        ablation_cfg = cfg.get("ablation", None)

        if ablation_cfg is not None:
            from src.ablation.ablation_dataset import AblationDataModule

            zeroed_cols = list(ablation_cfg.get("zeroed_cols", []))
            datamodule = AblationDataModule(
                params=OmegaConf.to_container(cfg, resolve=True),
                zeroed_cols=zeroed_cols,
            )
        else:
            from src.data.datamodule import RecSysDataModule

            datamodule = RecSysDataModule(
                params=OmegaConf.to_container(cfg, resolve=True)
            )

        # Smoke test: limit rows
        if smoke_test:
            datamodule = _wrap_smoke(datamodule)

        # ── 4. Instantiate model ───────────────────────────────────────────
        if model_type not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model type '{model_type}'. "
                f"Valid options: {list(MODEL_REGISTRY)}"
            )
        model_cls = MODEL_REGISTRY[model_type]
        model_kwargs: dict = {
            "n_users": datamodule.n_users,
            "n_items": datamodule.n_items,
            "embedding_dim": int(cfg.model.embedding_dim),
            "mlp_layers": list(cfg.model.mlp_layers),
            "dropout": float(cfg.model.dropout),
            "lr": float(cfg.model.lr),
            "weight_decay": float(cfg.model.weight_decay),
            "max_epochs": int(cfg.model.max_epochs),
        }
        if context_dim is not None:
            model_kwargs["context_dim"] = int(context_dim)
        model = model_cls(**model_kwargs)

        # ── 5. Callbacks ───────────────────────────────────────────────────
        ckpt_dir = Path(f"outputs/checkpoints/{experiment_name}")
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        callbacks = [
            EarlyStopping(
                monitor="val_ndcg_10",
                patience=int(cfg.model.patience),
                mode="max",
            ),
            ModelCheckpoint(
                dirpath=str(ckpt_dir),
                filename="best",
                monitor="val_ndcg_10",
                mode="max",
                save_top_k=1,
            ),
        ]

        # ── 6. Trainer ─────────────────────────────────────────────────────
        max_epochs = 1 if smoke_test else int(cfg.model.max_epochs)
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=callbacks,
            enable_progress_bar=True,
            log_every_n_steps=10,
        )

        # ── 7. Fit + test ──────────────────────────────────────────────────
        trainer.fit(model, datamodule)
        trainer.test(model, datamodule)

        # ── 8. Log checkpoint artifact ─────────────────────────────────────
        best_ckpt = ckpt_dir / "best.ckpt"
        if best_ckpt.exists():
            mlflow.log_artifact(str(best_ckpt))


# ── Helpers ────────────────────────────────────────────────────────────────────


def _flatten(d: dict, prefix: str = "") -> dict:
    """Recursively flatten a nested dict for MLflow log_params."""
    out: dict = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def _wrap_smoke(datamodule):
    """Monkey-patch datasets to first 1000 rows for smoke tests."""
    original_setup = datamodule.setup

    def smoke_setup(stage=None):
        original_setup(stage)
        for attr in ("train_dataset", "val_dataset", "test_dataset"):
            ds = getattr(datamodule, attr, None)
            if ds is not None:
                ds.df = ds.df.head(1000).reset_index(drop=True)

    datamodule.setup = smoke_setup
    return datamodule
