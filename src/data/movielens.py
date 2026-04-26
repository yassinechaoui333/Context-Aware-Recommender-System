from __future__ import annotations

import pickle
from pathlib import Path
from typing import Tuple

import pandas as pd
from omegaconf import OmegaConf
from sklearn.preprocessing import LabelEncoder

from src.data.download import download_movielens

RATING_COLUMNS = ["user_id", "item_id", "rating", "timestamp"]
PROCESSED_DIR = Path("data/processed")
ENCODER_PATH = PROCESSED_DIR / "encoders.pkl"
MOVIES_PATH = PROCESSED_DIR / "movies.parquet"


def _load_params(params_path: str = "params.yaml"):
    return OmegaConf.load(params_path)


def _validate_version(version: str) -> None:
    if version not in {"1m", "100k"}:
        raise ValueError("version must be either '1m' or '100k'")


def load_movielens(path: str | Path, version: str = "1m") -> pd.DataFrame:
    _validate_version(version)

    root = Path(path)
    if version == "1m":
        ratings_path = root / "ml-1m" / "ratings.dat"
        df = pd.read_csv(
            ratings_path,
            sep="::",
            engine="python",
            header=None,
            names=RATING_COLUMNS,
            encoding="latin-1",
        )
    else:
        ratings_path = root / "ml-100k" / "u.data"
        df = pd.read_csv(
            ratings_path,
            sep="\t",
            header=None,
            names=RATING_COLUMNS,
            encoding="latin-1",
        )

    df = df.astype(
        {
            "user_id": "int32",
            "item_id": "int32",
            "rating": "float32",
            "timestamp": "int64",
        }
    )

    params = _load_params()
    min_user_interactions = int(params.data.min_user_interactions)
    user_counts = df["user_id"].value_counts()
    valid_users = user_counts[user_counts >= min_user_interactions].index
    df = df[df["user_id"].isin(valid_users)].copy()

    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    df["user_id"] = user_encoder.fit_transform(df["user_id"]).astype("int32")
    df["item_id"] = item_encoder.fit_transform(df["item_id"]).astype("int32")

    if not df.empty and (int(df["user_id"].min()) != 0 or int(df["item_id"].min()) != 0):
        raise ValueError("LabelEncoder output must be 0-indexed for user_id and item_id")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with ENCODER_PATH.open("wb") as file_obj:
        pickle.dump({"user": user_encoder, "item": item_encoder}, file_obj)

    return df.sort_values("timestamp", ascending=True).reset_index(drop=True)


def load_movie_metadata(path: str | Path, version: str = "1m") -> pd.DataFrame:
    _validate_version(version)
    if not ENCODER_PATH.exists():
        raise FileNotFoundError(f"Missing encoders at '{ENCODER_PATH}'. Run load_movielens first.")

    root = Path(path)
    if version == "1m":
        movies_path = root / "ml-1m" / "movies.dat"
        movies = pd.read_csv(
            movies_path,
            sep="::",
            engine="python",
            header=None,
            names=["item_id_raw", "title", "genres"],
            encoding="latin-1",
        )
    else:
        movies_path = root / "ml-100k" / "u.item"
        movies = pd.read_csv(
            movies_path,
            sep="|",
            engine="python",
            header=None,
            usecols=[0, 1],
            names=["item_id_raw", "title"],
            encoding="latin-1",
        )
        movies["genres"] = ""

    with ENCODER_PATH.open("rb") as file_obj:
        encoders = pickle.load(file_obj)

    item_encoder: LabelEncoder = encoders["item"]
    movies["item_id_raw"] = movies["item_id_raw"].astype("int32")
    valid_item_ids = set(item_encoder.classes_.tolist())
    movies = movies[movies["item_id_raw"].isin(valid_item_ids)].copy()
    movies["item_id"] = item_encoder.transform(movies["item_id_raw"]).astype("int32")

    movies = movies[["item_id", "title", "genres"]].sort_values("item_id").reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    movies.to_parquet(MOVIES_PATH, index=False)
    return movies


def time_split(
    df: pd.DataFrame,
    val_ratio: float,
    test_ratio: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if val_ratio < 0 or test_ratio < 0:
        raise ValueError("val_ratio and test_ratio must be non-negative")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be < 1.0")

    df_sorted = df.sort_values("timestamp", ascending=True).reset_index(drop=True)
    n_rows = len(df_sorted)

    val_size = int(n_rows * val_ratio)
    test_size = int(n_rows * test_ratio)
    train_size = n_rows - val_size - test_size
    if train_size <= 0:
        raise ValueError("Split produced empty train set; adjust val_ratio/test_ratio")

    train_df = df_sorted.iloc[:train_size].copy()
    val_df = df_sorted.iloc[train_size : train_size + val_size].copy()
    test_df = df_sorted.iloc[train_size + val_size :].copy()

    train_users = set(train_df["user_id"].unique())
    val_df = val_df[val_df["user_id"].isin(train_users)].reset_index(drop=True)
    test_df = test_df[test_df["user_id"].isin(train_users)].reset_index(drop=True)

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    val_df.to_parquet(PROCESSED_DIR / "val.parquet", index=False)
    test_df.to_parquet(PROCESSED_DIR / "test.parquet", index=False)

    return train_df, val_df, test_df


def run_preprocess(
    params_path: str = "params.yaml",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    params = _load_params(params_path)
    version = str(params.data.movielens_version)
    raw_root = Path("data/raw/movielens")

    download_movielens(version=version, dest=str(raw_root))
    interactions = load_movielens(raw_root, version=version)
    load_movie_metadata(raw_root, version=version)

    return time_split(
        interactions,
        val_ratio=float(params.data.val_split_ratio),
        test_ratio=float(params.data.test_split_ratio),
    )
