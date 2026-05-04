from __future__ import annotations

from typing import Dict

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.evaluation.metrics import coverage, hit_rate_at_k, mrr, ndcg_at_k, novelty


class Evaluator:
    def __init__(
        self,
        model,
        test_dataloader: DataLoader,
        k_values: list[int] = (5, 10, 20),
        n_items: int = 0,
        item_pop: Dict[int, float] | None = None,
    ):
        self.model = model
        self.test_dataloader = test_dataloader
        self.k_values = list(k_values)
        self.n_items = n_items
        self.item_pop = item_pop

    def run(self) -> dict:
        self.model.eval()
        device = next(self.model.parameters()).device

        all_ndcg: dict[int, list[float]] = {k: [] for k in self.k_values}
        all_hr: dict[int, list[float]] = {k: [] for k in self.k_values}
        all_mrr: list[float] = []
        all_recs: list[list[int]] = []

        with torch.no_grad():
            for batch in self.test_dataloader:
                users = batch["user"].to(device)
                items = batch["items"].to(device)
                ctx = batch["context"].to(device)
                B, C = items.shape

                users_exp = users.unsqueeze(1).expand(B, C).reshape(-1)
                items_flat = items.reshape(-1)

                if ctx is not None:
                    ctx_exp = ctx.unsqueeze(1).expand(B, C, -1).reshape(B * C, -1)
                else:
                    ctx_exp = None

                scores = self.model.predict_score(users_exp, items_flat, ctx_exp)
                scores = scores.squeeze(-1).reshape(B, C)
                ranked_idx = scores.argsort(dim=1, descending=True)

                for i in range(B):
                    ranked_item_ids = items[i][ranked_idx[i]].cpu().tolist()
                    true_item = items[i][0].item()

                    for k in self.k_values:
                        all_ndcg[k].append(ndcg_at_k(ranked_item_ids, true_item, k))
                        all_hr[k].append(hit_rate_at_k(ranked_item_ids, true_item, k))

                    all_mrr.append(mrr(ranked_item_ids, true_item))
                    all_recs.append(ranked_item_ids[: max(self.k_values)])

        ndcg_result = {
            k: float(torch.tensor(all_ndcg[k]).mean().item()) for k in self.k_values
        }
        hr_result = {
            k: float(torch.tensor(all_hr[k]).mean().item()) for k in self.k_values
        }
        mrr_result = float(torch.tensor(all_mrr).mean().item())

        cov = coverage(all_recs, self.n_items) if self.n_items > 0 else 0.0

        nov = 0.0
        if self.item_pop is not None:
            nov = novelty(all_recs, self.item_pop)

        return {
            "NDCG": ndcg_result,
            "HR": hr_result,
            "MRR": mrr_result,
            "Coverage": cov,
            "Novelty": nov,
        }

    def summary_table(self) -> pd.DataFrame:
        results = self.run()
        flat: dict[str, float] = {}
        for k in self.k_values:
            flat[f"NDCG@{k}"] = round(results["NDCG"][k], 4)
            flat[f"HR@{k}"] = round(results["HR"][k], 4)
        flat["MRR"] = round(results["MRR"], 4)
        flat["Coverage"] = round(results["Coverage"], 4)
        flat["Novelty"] = round(results["Novelty"], 4)
        return pd.DataFrame([flat])

    @staticmethod
    def compute_item_popularity(parquet_path: str) -> Dict[int, float]:
        df = pd.read_parquet(parquet_path)
        counts = df["item_id"].value_counts()
        total = counts.sum()
        return {int(item): float(cnt / total) for item, cnt in counts.items()}
