from __future__ import annotations

from glob import glob
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
from omegaconf import OmegaConf
from tabulate import tabulate

from src.data.datamodule import RecSysDataModule
from src.evaluation.evaluator import Evaluator

EXPERIMENT_MODEL_MAP = {
    "ncf_baseline": "NCF",
    "context_ncf_late": "ContextNCFLate",
    "context_ncf_early": "ContextNCFEarly",
    "context_ncf_attn": "ContextNCFAttn",
    "ablation_no_temporal": "ContextNCFAttn",
    "ablation_no_session": "ContextNCFAttn",
    "ablation_no_device": "ContextNCFAttn",
    "ablation_context_only": "ContextNCFAttn",
}

MODEL_REGISTRY = {
    "NCF": __import__("src.models.ncf", fromlist=["NCF"]).NCF,
    "ContextNCFLate": __import__(
        "src.models.context_ncf_late", fromlist=["ContextNCFLate"]
    ).ContextNCFLate,
    "ContextNCFEarly": __import__(
        "src.models.context_ncf_early", fromlist=["ContextNCFEarly"]
    ).ContextNCFEarly,
    "ContextNCFAttn": __import__(
        "src.models.context_ncf_attn", fromlist=["ContextNCFAttn"]
    ).ContextNCFAttn,
}


def run_ablation() -> pd.DataFrame:
    params = OmegaConf.load("params.yaml")
    params_dict = OmegaConf.to_container(params, resolve=True)

    dm = RecSysDataModule(params=params_dict)
    dm.setup("test")

    item_pop = Evaluator.compute_item_popularity("data/processed/train.parquet")
    k_values = list(params_dict["eval"]["k_values"])
    n_items = dm.n_items

    checkpoints = sorted(glob("outputs/checkpoints/*/best.ckpt"))
    if not checkpoints:
        raise FileNotFoundError(
            "No checkpoints found under outputs/checkpoints/*/best.ckpt"
        )

    rows: list[dict] = []

    for ckpt_path in checkpoints:
        exp_name = Path(ckpt_path).parent.name
        model_type = EXPERIMENT_MODEL_MAP.get(exp_name)

        if model_type is None:
            print(f"Skipping unknown experiment '{exp_name}' — no model map entry")
            continue

        model_class = MODEL_REGISTRY[model_type]

        try:
            model: pl.LightningModule = model_class.load_from_checkpoint(
                ckpt_path, map_location="cpu"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to load checkpoint '{ckpt_path}': {exc}")
            continue

        model.eval()
        evaluator = Evaluator(
            model,
            dm.test_dataloader(),
            k_values=k_values,
            n_items=n_items,
            item_pop=item_pop,
        )
        results = evaluator.run()

        row: dict[str, object] = {"model_name": exp_name}
        for k in k_values:
            row[f"NDCG@{k}"] = round(results["NDCG"][k], 4)
            row[f"HR@{k}"] = round(results["HR"][k], 4)
        row["MRR"] = round(results["MRR"], 4)
        row["Coverage"] = round(results["Coverage"], 4)
        row["Novelty"] = round(results["Novelty"], 4)
        rows.append(row)

    df = pd.DataFrame(rows)

    baseline_ndcg = None
    for _, rdf in df.iterrows():
        if rdf["model_name"] == "ncf_baseline":
            baseline_ndcg = rdf["NDCG@10"]
            break

    deltas: list[float] = []
    for _, rdf in df.iterrows():
        if baseline_ndcg is not None and baseline_ndcg != 0:
            delta = ((rdf["NDCG@10"] - baseline_ndcg) / baseline_ndcg) * 100
        else:
            delta = 0.0
        deltas.append(round(delta, 2))

    df["Δ_NDCG@10"] = deltas
    df = df.sort_values("NDCG@10", ascending=False).reset_index(drop=True)

    out_path = Path("outputs/ablation_table.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(tabulate(df, headers="keys", tablefmt="grid", floatfmt=".4f"))
    print(f"\nSaved to {out_path}")
    return df
