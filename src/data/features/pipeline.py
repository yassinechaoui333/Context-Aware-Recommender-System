from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.data.features.context import build_context_vector, save_context_stats
from src.data.features.device import encode_device_proxy
from src.data.features.negative_sampling import sample_negatives_eval, sample_negatives_train
from src.data.features.session import encode_session, synthesize_sessions
from src.data.features.temporal import encode_temporal

SESSION_SCALER_PATH = "data/features/session_scaler.pkl"
CONTEXT_STATS_PATH = "data/features/context_stats.json"


def _load_params(params_path: str = "params.yaml"):
    return OmegaConf.load(params_path)


def _n_items_from_encoder(path: str = "data/processed/encoders.pkl") -> int:
    encoder_path = Path(path)
    if not encoder_path.exists():
        raise FileNotFoundError(f"Encoders not found at '{encoder_path}'")

    with encoder_path.open("rb") as file_obj:
        encoders = pickle.load(file_obj)
    return int(len(encoders["item"].classes_))


def run_feature_pipeline(
    split: str,
    fit_scalers: bool = False,
    params_path: str = "params.yaml",
) -> None:
    params = _load_params(params_path)
    split_path = Path(f"data/processed/{split}.parquet")
    if not split_path.exists():
        raise FileNotFoundError(f"Missing processed split at '{split_path}'")

    df = pd.read_parquet(split_path)
    df = synthesize_sessions(df, gap_seconds=int(params.data.session_gap_seconds))
    df = encode_temporal(df)
    df = encode_session(df, SESSION_SCALER_PATH, fit=fit_scalers)
    df = encode_device_proxy(df)

    context_array = build_context_vector(df)
    if fit_scalers:
        save_context_stats(context_array, CONTEXT_STATS_PATH)

    if split == "train":
        df = sample_negatives_train(
            df,
            n_neg=int(params.data.negative_samples_train),
            seed=int(params.data.seed),
        )
    else:
        n_items = _n_items_from_encoder()
        all_items = np.arange(n_items, dtype=np.int64)
        df = sample_negatives_eval(
            df,
            all_items,
            n_neg=int(params.data.negative_samples_eval),
            seed=int(params.data.seed),
        )

    out_dir = Path("data/features")
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / f"{split}.parquet", index=False)


def run_featurize(params_path: str = "params.yaml") -> None:
    run_feature_pipeline("train", fit_scalers=True, params_path=params_path)
    run_feature_pipeline("val", fit_scalers=False, params_path=params_path)
    run_feature_pipeline("test", fit_scalers=False, params_path=params_path)
