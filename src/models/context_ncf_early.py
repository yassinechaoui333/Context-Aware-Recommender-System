"""Context-NCF Early Fusion variant.

Context is concatenated to the user and item embeddings *before*
they enter either branch, so both GMF and MLP operate on
context-augmented representations from the start.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from src.models.modules.gmf import GMFBranch
from src.models.modules.mlp import MLPBranch
from src.models.ncf import NCF


class ContextNCFEarly(NCF):
    """NCF with early context fusion.

    After embedding lookup, the context vector is concatenated with
    both the user and item embeddings.  The GMF and MLP branches then
    operate on these ``(embedding_dim + context_dim)``-dimensional
    vectors.

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
        # Build the base NCF so embeddings and hyper-params are initialised.
        # We intentionally pass context_dim=None so the parent head is
        # constructed; we overwrite the branches and head below.
        super().__init__(
            n_users=n_users,
            n_items=n_items,
            embedding_dim=embedding_dim,
            mlp_layers=mlp_layers,
            dropout=dropout,
            lr=lr,
            weight_decay=weight_decay,
            max_epochs=max_epochs,
            context_dim=None,
        )
        self.context_dim = context_dim
        augmented_dim = embedding_dim + context_dim

        # Rebuild branches to accept augmented embedding dimension
        self.gmf = GMFBranch(augmented_dim)
        self.mlp = MLPBranch(2 * augmented_dim, list(mlp_layers), dropout)

        # Rebuild head for new dimensions
        final_input_dim = augmented_dim + self.mlp.output_dim
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
            raise ValueError("ContextNCFEarly requires a context tensor.")

        # Augment both embeddings with the context vector
        user_in = torch.cat([user_emb, context], dim=-1)
        item_in = torch.cat([item_emb, context], dim=-1)

        gmf_out = self.gmf(user_in, item_in)
        mlp_in = torch.cat([user_in, item_in], dim=-1)
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
