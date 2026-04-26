from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def synthesize_sessions(df: pd.DataFrame, gap_seconds: int = 3600) -> pd.DataFrame:
    sessions = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True).copy()

    time_delta = sessions.groupby("user_id")["timestamp"].diff().fillna(gap_seconds + 1)
    is_new_session = (time_delta > gap_seconds).astype("int32")

    user_session_id = is_new_session.groupby(sessions["user_id"]).cumsum().astype("int64")
    sessions["session_id"] = pd.factorize(
        sessions["user_id"].astype("int64") * 10_000_000 + user_session_id
    )[0].astype("int64")

    sessions["session_pos"] = sessions.groupby("session_id").cumcount().astype("int32")
    sessions["session_len"] = (
        sessions.groupby("session_id")["session_pos"].transform("count").astype("int32")
    )
    return sessions


def encode_session(df: pd.DataFrame, scaler_path: str, fit: bool = False) -> pd.DataFrame:
    encoded = df.copy()

    session_len_float = encoded["session_len"].astype("float32")
    encoded["session_pos_norm"] = encoded["session_pos"] / (session_len_float - 1.0 + 1e-8)

    scaler_file = Path(scaler_path)
    scaler_file.parent.mkdir(parents=True, exist_ok=True)

    session_len_arr = encoded[["session_len"]].astype("float32").to_numpy()
    if fit:
        scaler = MinMaxScaler()
        scaler.fit(session_len_arr)
        with scaler_file.open("wb") as file_obj:
            pickle.dump(scaler, file_obj)
    else:
        if not scaler_file.exists():
            raise FileNotFoundError(f"Session scaler not found at '{scaler_file}'")
        with scaler_file.open("rb") as file_obj:
            scaler = pickle.load(file_obj)

    encoded["session_len_norm"] = scaler.transform(session_len_arr).astype("float32")
    return encoded
