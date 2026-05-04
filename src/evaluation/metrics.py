from __future__ import annotations

import math
from typing import Dict, List


def ndcg_at_k(ranked_list: List[int], true_item: int, k: int) -> float:
    if true_item not in ranked_list[:k]:
        return 0.0
    rank = ranked_list[:k].index(true_item)
    return 1.0 / math.log2(rank + 2)


def hit_rate_at_k(ranked_list: List[int], true_item: int, k: int) -> float:
    return 1.0 if true_item in ranked_list[:k] else 0.0


def mrr(ranked_list: List[int], true_item: int) -> float:
    try:
        rank = ranked_list.index(true_item)
    except ValueError:
        return 0.0
    return 1.0 / (rank + 1)


def coverage(all_recs: List[List[int]], n_items: int) -> float:
    unique = {item for recs in all_recs for item in recs}
    return len(unique) / n_items


def novelty(all_recs: List[List[int]], item_pop: Dict[int, float]) -> float:
    total = 0.0
    count = 0
    for recs in all_recs:
        for item in recs:
            pop = item_pop.get(item, 1e-10)
            total += -math.log2(pop + 1e-10)
            count += 1
    if count == 0:
        return 0.0
    return total / count
