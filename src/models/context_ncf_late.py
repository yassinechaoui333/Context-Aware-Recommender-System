"""Context-NCF Late Fusion variant.

Context is concatenated to the combined GMF+MLP representation
*after* both branches have processed the embeddings, then passed
through a wider final linear head.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from src.models.ncf import NCF


class ContextNCFLate(NCF):
    """NCF with late context fusion.

    The context vector is appended to the ``[gmf_out || mlp_out]``
    concatenation before the final linear → sigmoid head.

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
        # Build the base NCF without the head (we override it below)
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

        # Replace the head to accommodate the extra context features
        final_input_dim = embedding_dim + self.mlp.output_dim + context_dim
        self.head = nn.Sequential(
            nn.Linear(final_input_dim, 1),
            nn.Sigmoid(),
        )

    def _forward_tower(
        self,
        user_emb: Tensor,
        item_emb: Tensor,
        context: Optional[Tensor] = None,
    ) -> Tensor:
        if context is None:
            raise ValueError("ContextNCFLate requires a context tensor.")
        gmf_out = self.gmf(user_emb, item_emb)
        mlp_in = torch.cat([user_emb, item_emb], dim=-1)
        mlp_out = self.mlp(mlp_in)
        combined = torch.cat([gmf_out, mlp_out, context], dim=-1)
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
