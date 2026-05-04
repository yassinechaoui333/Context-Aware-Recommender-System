"""Feature builder — converts raw API context inputs into a model-ready tensor.

Applies the exact same transformations as the offline feature pipeline so that
inference is consistent with training.

Reference encoding order (SCHEMA_CONTRACT):
    [sin_hour, cos_hour, sin_dow, cos_dow,
     session_pos_norm, session_len_norm,
     device_0, device_1, device_2]
"""
from __future__ import annotations

import math
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.api.schemas import ContextInput

CONTEXT_DIM = 9
SESSION_SCALER_PATH = Path("data/features/session_scaler.pkl")


@lru_cache(maxsize=1)
def _load_session_scaler(path: str = str(SESSION_SCALER_PATH)):
    """Load and cache the MinMaxScaler fitted on training session lengths."""
    scaler_path = Path(path)
    if not scaler_path.exists():
        # Graceful fallback: return None — session_len_norm will use raw value
        return None
    with open(scaler_path, "rb") as fh:
        return pickle.load(fh)


def build_context_tensor(
    ctx: ContextInput,
    scaler_path: Optional[str] = None,
) -> torch.Tensor:
    """Convert a :class:`ContextInput` to a float32 tensor of shape ``(9,)``.

    Parameters
    ----------
    ctx:
        Raw context from the API request.
    scaler_path:
        Override path to the session-length scaler pickle.
        Defaults to ``data/features/session_scaler.pkl``.

    Returns
    -------
    torch.Tensor
        Shape ``(9,)``, dtype ``float32``.
    """
    # ── Cyclic temporal encoding ─────────────────────────────────────────────
    sin_hour = math.sin(2.0 * math.pi * ctx.hour / 24.0)
    cos_hour = math.cos(2.0 * math.pi * ctx.hour / 24.0)
    sin_dow = math.sin(2.0 * math.pi * ctx.day_of_week / 7.0)
    cos_dow = math.cos(2.0 * math.pi * ctx.day_of_week / 7.0)

    # ── Session features ─────────────────────────────────────────────────────
    session_pos_norm = ctx.session_pos / (ctx.session_len - 1 + 1e-8)

    # Scale session_len using the offline-fitted MinMaxScaler
    scaler = _load_session_scaler(scaler_path or str(SESSION_SCALER_PATH))
    if scaler is not None:
        session_len_norm = float(
            scaler.transform(np.array([[ctx.session_len]], dtype=np.float32))[0, 0]
        )
        # Clip to [0, 1] — scaler may extrapolate for unseen values
        session_len_norm = max(0.0, min(1.0, session_len_norm))
    else:
        # Fallback: rough normalisation using typical max session length (20)
        session_len_norm = min(ctx.session_len / 20.0, 1.0)

    # ── Device one-hot ───────────────────────────────────────────────────────
    device_ohe = [float(ctx.device == i) for i in range(3)]

    # ── Assemble in SCHEMA_CONTRACT order ────────────────────────────────────
    values = [
        sin_hour,
        cos_hour,
        sin_dow,
        cos_dow,
        session_pos_norm,
        session_len_norm,
        *device_ohe,
    ]
    assert len(values) == CONTEXT_DIM, f"Expected {CONTEXT_DIM} dims, got {len(values)}"

    return torch.tensor(values, dtype=torch.float32)
