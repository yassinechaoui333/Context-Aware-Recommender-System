from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.data.dataset import RecSysDataset


class RecSysDataModule(LightningDataModule):
    def __init__(self, params: dict[str, Any]):
        super().__init__()
        self.params = params
        self.batch_size = int(self.params["model"]["batch_size"])

        encoder_path = Path("data/processed/encoders.pkl")
        if not encoder_path.exists():
            raise FileNotFoundError(f"Missing encoders file at '{encoder_path}'")

        with encoder_path.open("rb") as file_obj:
            encoders = pickle.load(file_obj)

        self.n_users = int(len(encoders["user"].classes_))
        self.n_items = int(len(encoders["item"].classes_))
        self.context_dim = 9

        self.train_dataset: RecSysDataset | None = None
        self.val_dataset: RecSysDataset | None = None
        self.test_dataset: RecSysDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        train_path = Path("data/features/train.parquet")
        val_path = Path("data/features/val.parquet")
        test_path = Path("data/features/test.parquet")

        if stage in (None, "fit"):
            self.train_dataset = RecSysDataset(str(train_path), mode="train")
            self.val_dataset = RecSysDataset(str(val_path), mode="eval")

        if stage in (None, "test", "validate"):
            if self.val_dataset is None:
                self.val_dataset = RecSysDataset(str(val_path), mode="eval")
            self.test_dataset = RecSysDataset(str(test_path), mode="eval")

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Call setup('fit') before requesting train_dataloader")

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("Call setup('fit') before requesting val_dataloader")

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("Call setup('test') before requesting test_dataloader")

        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
