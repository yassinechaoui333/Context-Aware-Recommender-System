"""Tests for NCF baseline model — Step 4.2."""
from __future__ import annotations

import torch
import pytest

from src.models.ncf import NCF


# ── fixtures ──────────────────────────────────────────────────────────────────

N_USERS = 100
N_ITEMS = 200
EMB_DIM = 16
MLP_LAYERS = [32, 16]
BATCH = 8


@pytest.fixture()
def model() -> NCF:
    return NCF(
        n_users=N_USERS,
        n_items=N_ITEMS,
        embedding_dim=EMB_DIM,
        mlp_layers=MLP_LAYERS,
        dropout=0.0,
        lr=1e-3,
        weight_decay=1e-4,
        max_epochs=10,
    )


def _rand_batch(B: int = BATCH):
    users = torch.randint(1, N_USERS + 1, (B,))
    items = torch.randint(1, N_ITEMS + 1, (B,))
    return users, items


# ── forward ───────────────────────────────────────────────────────────────────


def test_forward_shape(model: NCF) -> None:
    users, items = _rand_batch()
    out = model(users, items)
    assert out.shape == (BATCH, 1), f"Expected shape ({BATCH}, 1), got {out.shape}"


def test_forward_values_in_01(model: NCF) -> None:
    users, items = _rand_batch()
    out = model(users, items)
    assert (out >= 0.0).all() and (out <= 1.0).all(), "Output must be in [0, 1]"


# ── BPR loss ──────────────────────────────────────────────────────────────────


def test_bpr_loss_is_scalar_and_positive(model: NCF) -> None:
    users, items_pos = _rand_batch()
    _, items_neg = _rand_batch()
    pos_score = model(users, items_pos).squeeze(-1)
    neg_score = model(users, items_neg).squeeze(-1)
    loss = NCF._bpr_loss(pos_score, neg_score)
    assert loss.shape == torch.Size([]), "BPR loss must be a scalar"
    assert loss.item() > 0.0, "BPR loss must be positive"


def test_bpr_loss_backward(model: NCF) -> None:
    users, items_pos = _rand_batch()
    _, items_neg = _rand_batch()
    pos_score = model(users, items_pos).squeeze(-1)
    neg_score = model(users, items_neg).squeeze(-1)
    loss = NCF._bpr_loss(pos_score, neg_score)
    loss.backward()  # must not raise


# ── predict_score ─────────────────────────────────────────────────────────────


def test_predict_score_in_01(model: NCF) -> None:
    users, items = _rand_batch()
    out = model.predict_score(users, items)
    assert (out >= 0.0).all() and (out <= 1.0).all()


def test_predict_score_accepts_none_context(model: NCF) -> None:
    users, items = _rand_batch()
    out = model.predict_score(users, items, context=None)
    assert out.shape == (BATCH, 1)


# ── gradient flow ─────────────────────────────────────────────────────────────


def test_gradients_flow_to_embeddings(model: NCF) -> None:
    users, items_pos = _rand_batch()
    _, items_neg = _rand_batch()
    model.train()
    pos_score = model(users, items_pos).squeeze(-1)
    neg_score = model(users, items_neg).squeeze(-1)
    loss = NCF._bpr_loss(pos_score, neg_score)
    loss.backward()

    user_emb_grad = model.user_emb.weight.grad
    item_emb_grad = model.item_emb.weight.grad
    assert user_emb_grad is not None, "User embedding must have gradients"
    assert item_emb_grad is not None, "Item embedding must have gradients"
    # At least one gradient value should be nonzero
    assert user_emb_grad.abs().sum() > 0
    assert item_emb_grad.abs().sum() > 0
