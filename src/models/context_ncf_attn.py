"""Context-NCF Attention Gate variant (primary model).

A learned context gate modulates the user and item embeddings
before they enter the GMF and MLP branches.  Gate activations are
stored for interpretability analysis.
"""
from __future__ import annotations

from typing import List, Optional

import torch
from torch import Tensor

from src.models.ncf import NCF
from src.models.modules.gate import ContextGate


class ContextNCFAttn(NCF):
    """NCF with an attention-style context gate (primary model).

    A :class:`~src.models.modules.gate.ContextGate` produces a
    per-dimension weight vector ``g ∈ [0, 1]^{embedding_dim}`` from
    the context input.  Both user and item embeddings are scaled
    element-wise by ``g`` before being passed to the GMF and MLP
    branches.

    The gate's activations after each forward pass are accessible via
    ``model.gate.last_gate`` (shape ``(B, embedding_dim)``).

    Parameters
    ----------
    context_dim:
        Dimensionality of the context vector (9 for this project).
    All other parameters are identical to :class:`NCF`.
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        mlp_layers: List[int] = (128, 64, 32),
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        max_epochs: int = 50,
        context_dim: int = 9,
    ) -> None:
        super().__init__(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=embedding_dim,
            mlp_layers=mlp_layers,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            max_epochs=max_epochs,
            context_dim=context_dim,
        )
        self.context_dim = context_dim
        # The gate uses the original embedding_dim (not augmented)
        self.gate = ContextGate(context_dim, embedding_dim)

    def _forward_tower(
        self,
        user_emb: Tensor,
        item_emb: Tensor,
        context: Optional[Tensor] = None,
    ) -> Tensor:
        if context is None:
            raise ValueError("ContextNCFAttn requires a context tensor.")

        # 1. Compute gate weights from context
        g = self.gate(context)  # (B, embedding_dim), values in [0, 1]

        # 2. Apply gate to both embeddings
        user_emb_gated = g * user_emb
        item_emb_gated = g * item_emb

        # 3. Feed gated embeddings to the standard NCF branches
        gmf_out = self.gmf(user_emb_gated, item_emb_gated)
        mlp_in = torch.cat([user_emb_gated, item_emb_gated], dim=-1)
        mlp_out = self.mlp(mlp_in)
        combined = torch.cat([gmf_out, mlp_out], dim=-1)
        return self.head(combined)

    def predict_score(
        self,
        user_ids: Tensor,
        item_ids: Tensor,
        context: Optional[Tensor] = None,
    ) -> Tensor:
        self.eval()
        with torch.no_grad():
            return self(user_ids, item_ids, context)
