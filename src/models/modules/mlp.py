"""Multi-Layer Perceptron (MLP) branch."""
from __future__ import annotations

from typing import List

import torch.nn as nn
from torch import Tensor


class MLPBranch(nn.Module):
    """Stack of Linear → BatchNorm1d → ReLU → Dropout blocks.

    The input to this branch is the concatenation of user and item embeddings:
    ``[user_emb || item_emb]``, so ``input_dim`` should be
    ``2 * embedding_dim`` for the standard NCF setup.

    Parameters
    ----------
    input_dim:
        Dimensionality of the input (typically ``2 * embedding_dim``).
    layer_sizes:
        Output size of each hidden layer.
    dropout:
        Dropout probability applied after each ReLU.
    """

    def __init__(
        self,
        input_dim: int,
        layer_sizes: List[int],
        dropout: float,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        in_dim = input_dim
        for out_dim in layer_sizes:
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            ])
            in_dim = out_dim

        self.net = nn.Sequential(*layers)
        self.output_dim = in_dim  # last layer size

    def forward(self, x: Tensor) -> Tensor:
        """Run the MLP.

        Parameters
        ----------
        x:
            Concatenated user and item embeddings, shape ``(B, input_dim)``.

        Returns
        -------
        Tensor
            Last hidden representation, shape ``(B, layer_sizes[-1])``.
        """
        return self.net(x)
