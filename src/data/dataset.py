from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.features.context import CONTEXT_COLS


class RecSysDataset(Dataset):
    def __init__(self, parquet_path: str, mode: Literal["train", "eval"]):
        if mode not in {"train", "eval"}:
            raise ValueError("mode must be one of {'train', 'eval'}")

        self.df = pd.read_parquet(parquet_path)
        self.mode = mode

        missing = [col for col in CONTEXT_COLS if col not in self.df.columns]
        if missing:
            raise KeyError(f"Missing context columns in '{parquet_path}': {missing}")

        if self.mode == "train" and "item_neg" not in self.df.columns:
            raise KeyError("Train dataset requires an 'item_neg' column")
        if self.mode == "eval" and "candidates" not in self.df.columns:
            raise KeyError("Eval dataset requires a 'candidates' column")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        ctx_np = row[CONTEXT_COLS].to_numpy(dtype=np.float32)
        ctx = torch.tensor(ctx_np, dtype=torch.float32)

        if self.mode == "train":
            return {
                "user": torch.tensor(int(row["user_id"]), dtype=torch.long),
                "item_pos": torch.tensor(int(row["item_id"]), dtype=torch.long),
                "item_neg": torch.tensor(int(row["item_neg"]), dtype=torch.long),
                "context": ctx,
            }

        candidates = np.asarray(row["candidates"], dtype=np.int64)
        if candidates.ndim != 1:
            raise ValueError(
                f"Expected 1D candidates array at idx={idx}, got shape={candidates.shape}"
            )

        return {
            "user": torch.tensor(int(row["user_id"]), dtype=torch.long),
            "items": torch.tensor(candidates, dtype=torch.long),
            "context": ctx,
            "label_idx": torch.tensor(0, dtype=torch.long),
        }
