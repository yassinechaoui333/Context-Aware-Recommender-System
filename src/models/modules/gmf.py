"""Generalised Matrix Factorisation (GMF) branch."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class GMFBranch(nn.Module):
    """Elementwise product of user and item embeddings.

    Parameters
    ----------
    embedding_dim:
        Dimensionality of both the user and item embedding vectors.
    """

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, user_emb: Tensor, item_emb: Tensor) -> Tensor:
        """Compute the Hadamard product.

        Parameters
        ----------
        user_emb:
            Shape ``(B, embedding_dim)``.
        item_emb:
            Shape ``(B, embedding_dim)``.

        Returns
        -------
        Tensor
            Shape ``(B, embedding_dim)``.
        """
        return user_emb * item_emb
