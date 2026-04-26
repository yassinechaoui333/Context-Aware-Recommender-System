from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

CONTEXT_COLS = [
    "sin_hour",
    "cos_hour",
    "sin_dow",
    "cos_dow",
    "session_pos_norm",
    "session_len_norm",
    "device_0",
    "device_1",
    "device_2",
]


def build_context_vector(df: pd.DataFrame) -> np.ndarray:
    missing = [col for col in CONTEXT_COLS if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required context columns: {missing}")
    return df[CONTEXT_COLS].to_numpy(dtype=np.float32)


def save_context_stats(arr: np.ndarray, path: str) -> None:
    if arr.ndim != 2 or arr.shape[1] != len(CONTEXT_COLS):
        raise ValueError(
            f"Expected context array shape (N, {len(CONTEXT_COLS)}), got {arr.shape}"
        )

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "columns": CONTEXT_COLS,
        "mean": arr.mean(axis=0, dtype=np.float64).tolist(),
        "std": arr.std(axis=0, dtype=np.float64).tolist(),
        "dim": len(CONTEXT_COLS),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
