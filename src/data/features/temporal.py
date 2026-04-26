from __future__ import annotations

import numpy as np
import pandas as pd


def encode_temporal(df: pd.DataFrame) -> pd.DataFrame:
    encoded = df.copy()
    dt = pd.to_datetime(encoded["timestamp"], unit="s")

    hour = dt.dt.hour.astype(np.float32)
    day_of_week = dt.dt.dayofweek.astype(np.float32)
    month = dt.dt.month.astype(np.float32)

    encoded["hour"] = hour
    encoded["day_of_week"] = day_of_week
    encoded["month"] = month

    encoded["sin_hour"] = np.sin(2.0 * np.pi * hour / 24.0)
    encoded["cos_hour"] = np.cos(2.0 * np.pi * hour / 24.0)
    encoded["sin_dow"] = np.sin(2.0 * np.pi * day_of_week / 7.0)
    encoded["cos_dow"] = np.cos(2.0 * np.pi * day_of_week / 7.0)
    encoded["sin_month"] = np.sin(2.0 * np.pi * month / 12.0)
    encoded["cos_month"] = np.cos(2.0 * np.pi * month / 12.0)

    encoded = encoded.drop(columns=["hour", "day_of_week", "month"])
    return encoded
