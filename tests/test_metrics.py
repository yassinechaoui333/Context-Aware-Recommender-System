from __future__ import annotations

from src.evaluation.metrics import coverage, hit_rate_at_k, mrr, ndcg_at_k, novelty


def test_ndcg_top_rank():
    assert ndcg_at_k([3, 1, 2], true_item=3, k=10) == 1.0


def test_ndcg_mid_rank():
    score = ndcg_at_k([1, 2, 3], true_item=3, k=10)
    assert 0.0 < score < 1.0


def test_ndcg_not_present():
    assert ndcg_at_k([1, 2, 3], true_item=9, k=10) == 0.0


def test_hit_rate_found():
    assert hit_rate_at_k([5, 3, 1], true_item=5, k=1) == 1.0


def test_hit_rate_not_found():
    assert hit_rate_at_k([1, 2, 3], true_item=5, k=3) == 0.0


def test_mrr_first():
    assert mrr([5, 3, 1], true_item=5) == 1.0


def test_mrr_third():
    assert mrr([5, 3, 1], true_item=1) == 1.0 / 3.0


def test_mrr_not_found():
    assert mrr([1, 2, 3], true_item=9) == 0.0


def test_coverage():
    assert coverage([[1, 2], [2, 3]], n_items=10) == 0.3


def test_coverage_empty():
    assert coverage([], n_items=100) == 0.0


def test_novelty():
    pop = {1: 0.5, 2: 0.3, 3: 0.2}
    recs = [[1, 2], [2, 3]]
    result = novelty(recs, pop)
    assert result > 0.0


def test_novelty_empty():
    assert novelty([], {}) == 0.0
