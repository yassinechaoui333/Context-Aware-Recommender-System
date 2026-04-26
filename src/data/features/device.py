from __future__ import annotations

import pandas as pd


def encode_device_proxy(df: pd.DataFrame) -> pd.DataFrame:
    encoded = df.copy()
    encoded["device_idx"] = (encoded["user_id"] % 3).astype("int8")

    encoded["device_0"] = (encoded["device_idx"] == 0).astype("int8")
    encoded["device_1"] = (encoded["device_idx"] == 1).astype("int8")
    encoded["device_2"] = (encoded["device_idx"] == 2).astype("int8")

    encoded = encoded.drop(columns=["device_idx"])
    return encoded
