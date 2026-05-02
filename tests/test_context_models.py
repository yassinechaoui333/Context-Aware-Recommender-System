"""Tests for all three context model variants — Step 4.5."""
from __future__ import annotations

import torch
import pytest

from src.models.context_ncf_late import ContextNCFLate
from src.models.context_ncf_early import ContextNCFEarly
from src.models.context_ncf_attn import ContextNCFAttn

# ── constants ─────────────────────────────────────────────────────────────────

N_USERS = 100
N_ITEMS = 200
EMB_DIM = 16
MLP_LAYERS = [32, 16]
CTX_DIM = 9
BATCH = 8


# ── fixtures ──────────────────────────────────────────────────────────────────


def _make_model(cls):
    return cls(
        n_users=N_USERS,
        n_items=N_ITEMS,
        embedding_dim=EMB_DIM,
        mlp_layers=MLP_LAYERS,
        dropout=0.0,
        lr=1e-3,
        weight_decay=1e-4,
        max_epochs=10,
        context_dim=CTX_DIM,
    )


@pytest.fixture()
def late_model():
    return _make_model(ContextNCFLate)


@pytest.fixture()
def early_model():
    return _make_model(ContextNCFEarly)


@pytest.fixture()
def attn_model():
    return _make_model(ContextNCFAttn)


def _rand_inputs(B: int = BATCH):
    users = torch.randint(1, N_USERS + 1, (B,))
    items = torch.randint(1, N_ITEMS + 1, (B,))
    context = torch.rand(B, CTX_DIM)
    return users, items, context


# ── identical output shapes ───────────────────────────────────────────────────


@pytest.mark.parametrize("model_fixture", ["late_model", "early_model", "attn_model"])
def test_output_shape(model_fixture, request) -> None:
    model = request.getfixturevalue(model_fixture)
    users, items, context = _rand_inputs()
    out = model(users, items, context)
    assert out.shape == (BATCH, 1), f"{model.__class__.__name__}: expected ({BATCH},1), got {out.shape}"


@pytest.mark.parametrize("model_fixture", ["late_model", "early_model", "attn_model"])
def test_output_in_01(model_fixture, request) -> None:
    model = request.getfixturevalue(model_fixture)
    users, items, context = _rand_inputs()
    out = model(users, items, context)
    assert (out >= 0.0).all() and (out <= 1.0).all()


# ── distinct architectures produce different outputs ──────────────────────────


def test_models_produce_different_outputs(late_model, early_model, attn_model) -> None:
    torch.manual_seed(99)
    users, items, context = _rand_inputs()

    # Force all to train mode so BN works correctly with batch
    late_out = late_model(users, items, context).detach()
    early_out = early_model(users, items, context).detach()
    attn_out = attn_model(users, items, context).detach()

    assert not torch.allclose(late_out, early_out, atol=1e-4), \
        "Late and Early models should produce different outputs"
    assert not torch.allclose(late_out, attn_out, atol=1e-4), \
        "Late and Attn models should produce different outputs"
    assert not torch.allclose(early_out, attn_out, atol=1e-4), \
        "Early and Attn models should produce different outputs"


# ── ContextNCFAttn gate ───────────────────────────────────────────────────────


def test_attn_gate_shape_and_range(attn_model: ContextNCFAttn) -> None:
    users, items, context = _rand_inputs()
    attn_model(users, items, context)  # trigger forward to populate last_gate

    gate = attn_model.gate.last_gate
    assert gate is not None, "last_gate should be populated after forward()"
    assert gate.shape == (BATCH, EMB_DIM), \
        f"Expected gate shape ({BATCH}, {EMB_DIM}), got {gate.shape}"
    assert (gate >= 0.0).all() and (gate <= 1.0).all(), \
        "Gate values must be in [0, 1]"


def test_attn_gate_is_detached(attn_model: ContextNCFAttn) -> None:
    users, items, context = _rand_inputs()
    attn_model(users, items, context)
    assert not attn_model.gate.last_gate.requires_grad, \
        "last_gate should be detached from the computation graph"


# ── predict_score interface ───────────────────────────────────────────────────


@pytest.mark.parametrize("model_fixture", ["late_model", "early_model", "attn_model"])
def test_predict_score_accepts_context(model_fixture, request) -> None:
    model = request.getfixturevalue(model_fixture)
    users, items, context = _rand_inputs()
    out = model.predict_score(users, items, context=context)
    assert out.shape == (BATCH, 1)
    assert (out >= 0.0).all() and (out <= 1.0).all()


# ── context required ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("model_fixture", ["late_model", "early_model", "attn_model"])
def test_context_required(model_fixture, request) -> None:
    model = request.getfixturevalue(model_fixture)
    users, items, _ = _rand_inputs()
    with pytest.raises((ValueError, TypeError)):
        model(users, items, context=None)
