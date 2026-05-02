"""Context gating module for attention-based context fusion."""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ContextGate(nn.Module):
    """Two-layer gate network that maps a context vector to per-dimension weights.

    Architecture::

        Linear(context_dim, embedding_dim) → ReLU
        → Linear(embedding_dim, embedding_dim) → Sigmoid

    The output is in ``[0, 1]`` and is used to scale user and item embeddings
    element-wise before they are fed into the GMF and MLP branches.  The gate
    activations from the most recent forward pass are stored in
    ``self.last_gate`` (detached from the computation graph) for
    interpretability analysis.

    Parameters
    ----------
    context_dim:
        Dimensionality of the raw context input vector (9 in this project).
    embedding_dim:
        Dimensionality of the user/item embeddings.
    """

    def __init__(self, context_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(context_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.Sigmoid(),
        )
        # Populated after every forward() for interpretability
        self.last_gate: Tensor | None = None

    def forward(self, context_vec: Tensor) -> Tensor:
        """Compute gate activations.

        Parameters
        ----------
        context_vec:
            Shape ``(B, context_dim)``.

        Returns
        -------
        Tensor
            Gate weights in ``[0, 1]``, shape ``(B, embedding_dim)``.
        """
        gate = self.gate_net(context_vec)
        self.last_gate = gate.detach()
        return gate
