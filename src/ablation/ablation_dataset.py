from __future__ import annotations

from pathlib import Path
from typing import Any, List

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from src.data.datamodule import RecSysDataModule
from src.data.dataset import RecSysDataset
from src.data.features.context import CONTEXT_COLS


class AblationDataset(RecSysDataset):
    def __init__(
        self,
        parquet_path: str,
        mode: str,
        zeroed_cols: List[str],
    ):
        super().__init__(parquet_path, mode)
        missing = [c for c in zeroed_cols if c not in CONTEXT_COLS]
        if missing:
            raise KeyError(f"Unknown columns to zero: {missing}")
        self.zeroed_indices = [CONTEXT_COLS.index(c) for c in zeroed_cols]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = super().__getitem__(idx)
        for i in self.zeroed_indices:
            sample["context"][i] = 0.0
        return sample


class AblationDataModule(LightningDataModule):
    def __init__(self, params: dict[str, Any], zeroed_cols: List[str]):
        super().__init__()
        self._base = RecSysDataModule(params)
        self.zeroed_cols = list(zeroed_cols)
        self.train_dataset: AblationDataset | None = None
        self.val_dataset: AblationDataset | None = None
        self.test_dataset: AblationDataset | None = None

    @property
    def n_users(self) -> int:
        return self._base.n_users

    @property
    def n_items(self) -> int:
        return self._base.n_items

    @property
    def context_dim(self) -> int:
        return self._base.context_dim

    def setup(self, stage: str | None = None) -> None:
        train_path = Path("data/features/train.parquet")
        val_path = Path("data/features/val.parquet")
        test_path = Path("data/features/test.parquet")

        if stage in (None, "fit"):
            self.train_dataset = AblationDataset(
                str(train_path), "train", self.zeroed_cols
            )
            self.val_dataset = AblationDataset(
                str(val_path), "eval", self.zeroed_cols
            )

        if stage in (None, "test", "validate"):
            if self.val_dataset is None:
                self.val_dataset = AblationDataset(
                    str(val_path), "eval", self.zeroed_cols
                )
            self.test_dataset = AblationDataset(
                str(test_path), "eval", self.zeroed_cols
            )

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Call setup first")
        return DataLoader(
            self.train_dataset,
            batch_size=self._base.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("Call setup first")
        return DataLoader(
            self.val_dataset,
            batch_size=self._base.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("Call setup first")
        return DataLoader(
            self.test_dataset,
            batch_size=self._base.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
