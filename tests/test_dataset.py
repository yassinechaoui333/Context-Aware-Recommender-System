from __future__ import annotations

import pickle

import pandas as pd
import torch
from omegaconf import OmegaConf

from src.data.datamodule import RecSysDataModule
from src.data.dataset import RecSysDataset


def test_dataset_lengths_match_parquet_rows() -> None:
    train_df = pd.read_parquet("data/features/train.parquet")
    val_df = pd.read_parquet("data/features/val.parquet")

    train_ds = RecSysDataset("data/features/train.parquet", mode="train")
    eval_ds = RecSysDataset("data/features/val.parquet", mode="eval")

    assert len(train_ds) == len(train_df)
    assert len(eval_ds) == len(val_df)


def test_train_item_types_and_context_shape() -> None:
    train_ds = RecSysDataset("data/features/train.parquet", mode="train")
    sample = train_ds[0]

    assert sample["user"].dtype == torch.long
    assert sample["item_pos"].dtype == torch.long
    assert sample["item_neg"].dtype == torch.long
    assert sample["context"].dtype == torch.float32
    assert sample["context"].shape == (9,)


def test_eval_items_shape_and_label_idx() -> None:
    eval_ds = RecSysDataset("data/features/val.parquet", mode="eval")
    sample = eval_ds[0]

    assert sample["user"].dtype == torch.long
    assert sample["items"].dtype == torch.long
    assert sample["items"].shape == (100,)
    assert sample["context"].dtype == torch.float32
    assert sample["context"].shape == (9,)
    assert sample["label_idx"].dtype == torch.long
    assert int(sample["label_idx"].item()) == 0


def test_datamodule_train_loader_iterates_two_batches() -> None:
    params = OmegaConf.load("params.yaml")
    dm = RecSysDataModule(OmegaConf.to_container(params, resolve=True))
    dm.setup("fit")

    loader = dm.train_dataloader()
    iterator = iter(loader)
    batch_1 = next(iterator)
    batch_2 = next(iterator)

    assert batch_1["user"].dtype == torch.long
    assert batch_1["item_pos"].dtype == torch.long
    assert batch_1["item_neg"].dtype == torch.long
    assert batch_1["context"].dtype == torch.float32

    assert batch_2["user"].dtype == torch.long
    assert batch_2["item_pos"].dtype == torch.long
    assert batch_2["item_neg"].dtype == torch.long
    assert batch_2["context"].dtype == torch.float32


def test_datamodule_n_users_n_items_match_encoders() -> None:
    params = OmegaConf.load("params.yaml")
    dm = RecSysDataModule(OmegaConf.to_container(params, resolve=True))

    with open("data/processed/encoders.pkl", "rb") as file_obj:
        encoders = pickle.load(file_obj)

    assert isinstance(dm.n_users, int)
    assert isinstance(dm.n_items, int)
    assert dm.n_users > 0
    assert dm.n_items > 0
    assert dm.n_users == len(encoders["user"].classes_)
    assert dm.n_items == len(encoders["item"].classes_)
