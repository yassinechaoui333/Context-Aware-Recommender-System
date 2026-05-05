from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def _build_user_item_sets(df: pd.DataFrame) -> Dict[int, set[int]]:
    return (
        df.groupby("user_id")["item_id"]
        .apply(lambda items: set(int(item) for item in items.tolist()))
        .to_dict()
    )


def sample_negatives_train(df: pd.DataFrame, n_neg: int = 4, seed: int = 42) -> pd.DataFrame:
    sampled = df.copy().reset_index(drop=True)
    if sampled.empty:
        sampled["item_neg"] = []
        return sampled

    if n_neg <= 0:
        raise ValueError("n_neg must be >= 1")

    user_items = _build_user_item_sets(sampled)

    item_counts = sampled["item_id"].value_counts().sort_index()
    all_items = item_counts.index.to_numpy(dtype=np.int64)
    pop_weights = item_counts.to_numpy(dtype=np.float64)
    pop_weights /= pop_weights.sum()

    rng = np.random.default_rng(seed)

    user_candidate_items: dict[int, np.ndarray] = {}
    user_candidate_weights: dict[int, np.ndarray] = {}
    for user_id, seen_items in user_items.items():
        valid_mask = np.array([item not in seen_items for item in all_items], dtype=bool)
        candidate_items = all_items[valid_mask]
        if candidate_items.size == 0:
            raise ValueError(f"No train negatives available for user_id={user_id}")

        candidate_weights = pop_weights[valid_mask]
        candidate_weights = candidate_weights / candidate_weights.sum()

        user_candidate_items[int(user_id)] = candidate_items
        user_candidate_weights[int(user_id)] = candidate_weights

    negatives_col: list[list[int]] = []
    for _, row in sampled.iterrows():
        user_id = int(row["user_id"])
        candidate_items = user_candidate_items[user_id]
        candidate_weights = user_candidate_weights[user_id]

        sampled_negs = rng.choice(
            candidate_items,
            size=n_neg,
            replace=True,
            p=candidate_weights,
        )
        negatives_col.append(sampled_negs.astype(int).tolist())

    sampled["item_neg"] = negatives_col
    return sampled


def sample_negatives_eval(
    df: pd.DataFrame,
    all_items: np.ndarray,
    n_neg: int = 99,
    seed: int = 42,
) -> pd.DataFrame:
    sampled = df.copy().reset_index(drop=True)
    if sampled.empty:
        sampled["candidates"] = []
        return sampled

    if n_neg <= 0:
        raise ValueError("n_neg must be >= 1")

    all_items_arr = np.asarray(all_items, dtype=np.int64)
    if all_items_arr.size == 0:
        raise ValueError("all_items must not be empty")

    user_items = _build_user_item_sets(sampled)
    rng = np.random.default_rng(seed)

    user_candidate_pools: dict[int, np.ndarray] = {}
    for user_id, seen_items in user_items.items():
        seen_arr = np.fromiter(seen_items, dtype=np.int64)
        pool = np.setdiff1d(all_items_arr, seen_arr, assume_unique=False)
        if pool.size == 0:
            raise ValueError(f"No eval negatives available for user_id={user_id}")
        user_candidate_pools[int(user_id)] = pool

    candidates_col: list[list[int]] = []
    for _, row in sampled.iterrows():
        user_id = int(row["user_id"])
        pos_item = int(row["item_id"])
        candidate_pool = user_candidate_pools[user_id]

        replace = len(candidate_pool) < n_neg
        negatives = rng.choice(
            candidate_pool,
            size=n_neg,
            replace=replace,
        )
        candidates = [pos_item, *negatives.astype(int).tolist()]
        candidates_col.append(candidates)

    sampled["candidates"] = candidates_col
    return sampled
