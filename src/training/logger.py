from __future__ import annotations

import os
from typing import Any, Dict

import mlflow


class ExperimentLogger:
    def __init__(self, cfg):
        tracking_uri = os.environ.get(
            "MLFLOW_TRACKING_URI", "outputs/logs/mlruns"
        )
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("context-recsys")

        self._experiment_name = cfg.get("experiment_name", "unnamed")
        self._run = mlflow.start_run(run_name=self._experiment_name)

    @property
    def run_id(self) -> str:
        return self._run.info.run_id

    @property
    def experiment_name(self) -> str:
        return self._experiment_name

    def log_params(self, params: dict) -> None:
        mlflow.log_params(_flatten(params))

    def log_metrics(self, metrics: Dict[str, float], step: int | None = None) -> None:
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: str) -> None:
        mlflow.log_artifact(path)

    def end(self) -> None:
        mlflow.end_run()

    def __enter__(self) -> ExperimentLogger:
        return self

    def __exit__(self, *args: Any) -> None:
        self.end()


def _flatten(d: dict, prefix: str = "") -> dict:
    out: dict = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        elif isinstance(v, (list, tuple)):
            out[key] = str(v)
        else:
            out[key] = v
    return out
