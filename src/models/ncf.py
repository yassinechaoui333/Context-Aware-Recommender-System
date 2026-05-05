"""Neural Collaborative Filtering (NCF) — baseline model.

Reference: He et al., "Neural Collaborative Filtering", WWW 2017.
"""
from __future__ import annotations

import math
from typing import List, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor

from src.models.modules.gmf import GMFBranch
from src.models.modules.mlp import MLPBranch


class NCF(pl.LightningModule):
    """NeuMF-style NCF combining GMF and MLP branches.

    The model ignores the optional ``context`` argument so that all context
    variants share the same ``predict_score`` interface.

    Parameters
    ----------
    n_users:
        Number of unique users (encoder classes count).
    n_items:
        Number of unique items (encoder classes count).
    embedding_dim:
        Size of the user and item embedding vectors.
    mlp_layers:
        Hidden layer sizes for the MLP branch.
    dropout:
        Dropout probability.
    lr:
        Learning rate for AdamW.
    weight_decay:
        Weight decay (L2 regularisation) for AdamW.
    max_epochs:
        Total training epochs — used by CosineAnnealingLR.
    context_dim:
        Accepted for interface compatibility; unused in base NCF.
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
        context_dim: Optional[int] = None,  # noqa: ARG002  — kept for compat
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.embedding_dim = embedding_dim

        # Embedding tables — padding_idx=0 so the 0 vector is never updated
        self.user_emb = nn.Embedding(n_users + 1, embedding_dim, padding_idx=0)
        self.item_emb = nn.Embedding(n_items + 1, embedding_dim, padding_idx=0)

        # GMF branch
        self.gmf = GMFBranch(embedding_dim)

        # MLP branch — input = concat(user, item)
        self.mlp = MLPBranch(2 * embedding_dim, list(mlp_layers), dropout)

        # Final prediction head
        final_input_dim = embedding_dim + self.mlp.output_dim
        self.head = nn.Sequential(
            nn.Linear(final_input_dim, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        nn.init.xavier_normal_(self.user_emb.weight)
        nn.init.xavier_normal_(self.item_emb.weight)
        # Zero out the padding embedding
        with torch.no_grad():
            self.user_emb.weight[0].fill_(0.0)
            self.item_emb.weight[0].fill_(0.0)

    # ------------------------------------------------------------------
    # Forward helpers (overridden by subclasses)
    # ------------------------------------------------------------------

    def _embed(self, user_ids: Tensor, item_ids: Tensor) -> tuple[Tensor, Tensor]:
        """Return base user and item embeddings."""
        return self.user_emb(user_ids), self.item_emb(item_ids)

    def _forward_tower(
        self,
        user_emb: Tensor,
        item_emb: Tensor,
        context: Optional[Tensor] = None,
    ) -> Tensor:
        """Run GMF + MLP and produce a sigmoid score.

        Parameters
        ----------
        user_emb, item_emb:
            Shape ``(B, embedding_dim)``.
        context:
            Ignored in base NCF; accepted for interface compatibility.

        Returns
        -------
        Tensor
            Predicted scores, shape ``(B, 1)``.
        """
        gmf_out = self.gmf(user_emb, item_emb)
        mlp_in = torch.cat([user_emb, item_emb], dim=-1)
        mlp_out = self.mlp(mlp_in)
        combined = torch.cat([gmf_out, mlp_out], dim=-1)
        return self.head(combined)

    def forward(
        self,
        user_ids: Tensor,
        item_ids: Tensor,
        context: Optional[Tensor] = None,
    ) -> Tensor:
        """Predict a single score per (user, item) pair.

        Parameters
        ----------
        user_ids: ``(B,)``
        item_ids: ``(B,)``
        context: ``(B, context_dim)`` — optional in base NCF

        Returns
        -------
        Tensor
            Shape ``(B, 1)``, values in ``[0, 1]``.
        """
        user_emb, item_emb = self._embed(user_ids, item_ids)
        return self._forward_tower(user_emb, item_emb, context)

    # ------------------------------------------------------------------
    # BPR loss
    # ------------------------------------------------------------------

    @staticmethod
    def _bpr_loss(pos_score: Tensor, neg_score: Tensor) -> Tensor:
        """Bayesian Personalised Ranking loss.

        ``L = -mean(log(σ(pos - neg) + ε))``
        """
        return -torch.mean(torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8))

    # ------------------------------------------------------------------
    # Lightning steps
    # ------------------------------------------------------------------

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:  # noqa: ARG002
        user = batch["user"]
        pos = batch["item_pos"]
        neg = batch["item_neg"]
        ctx = batch.get("context")

        pos_score = self(user, pos, ctx).squeeze(-1)  # (B,)

        if neg.ndim == 1:
            neg_score = self(user, neg, ctx).squeeze(-1)
            bpr_losses = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-8)
            loss = bpr_losses.mean()
        else:
            B, n_neg = neg.shape
            neg_flat = neg.reshape(-1)
            user_exp = user.unsqueeze(1).expand(B, n_neg).reshape(-1)

            if ctx is not None:
                ctx_exp = ctx.unsqueeze(1).expand(B, n_neg, -1).reshape(B * n_neg, -1)
            else:
                ctx_exp = None

            neg_score = self(
                user_exp, neg_flat, ctx_exp
            ).reshape(B, n_neg).squeeze(-1)
            bpr_losses = -torch.log(torch.sigmoid(
                pos_score.unsqueeze(1) - neg_score
            ) + 1e-8)
            loss = bpr_losses.mean()

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(
        self, batch: dict[str, Tensor], batch_idx: int  # noqa: ARG002
    ) -> None:
        ndcg = self._eval_batch(batch)
        self.log("val_ndcg_10", ndcg, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(
        self, batch: dict[str, Tensor], batch_idx: int  # noqa: ARG002
    ) -> None:
        ndcg = self._eval_batch(batch)
        self.log("test_ndcg_10", ndcg, on_step=False, on_epoch=True)

    def _eval_batch(self, batch: dict[str, Tensor]) -> Tensor:
        """Compute mean NDCG@10 for an eval batch.

        Each row has 100 candidate items where index 0 is the positive.
        """
        users = batch["user"]         # (B,)
        items = batch["items"]        # (B, 100)
        context = batch.get("context")  # (B, 9) or None
        B, C = items.shape

        # Repeat user ids across all candidates
        users_exp = users.unsqueeze(1).expand(B, C).reshape(-1)  # (B*C,)
        items_flat = items.reshape(-1)                            # (B*C,)

        if context is not None:
            ctx_exp = context.unsqueeze(1).expand(B, C, -1).reshape(B * C, -1)
        else:
            ctx_exp = None

        scores = self(users_exp, items_flat, ctx_exp).squeeze(-1)  # (B*C,)
        scores = scores.reshape(B, C)                              # (B, 100)

        # Rank descending; positive is at index 0 in candidate list
        ranked = scores.argsort(dim=1, descending=True)  # (B, 100)
        k = 10
        ndcg_sum = torch.tensor(0.0, device=scores.device)
        for i in range(B):
            rank_of_pos = (ranked[i] == 0).nonzero(as_tuple=True)[0].item()
            if rank_of_pos < k:
                ndcg_sum += 1.0 / math.log2(rank_of_pos + 2)
        return ndcg_sum / B

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------

    def configure_optimizers(self):  # type: ignore[override]
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    # ------------------------------------------------------------------
    # Public inference interface
    # ------------------------------------------------------------------

    def predict_score(
        self,
        user_ids: Tensor,
        item_ids: Tensor,
        context: Optional[Tensor] = None,
    ) -> Tensor:
        """Score user–item pairs (inference-only convenience method).

        Parameters
        ----------
        user_ids: ``(B,)``
        item_ids: ``(B,)``
        context:  ``(B, context_dim)`` — optional in base NCF

        Returns
        -------
        Tensor
            Scores in ``[0, 1]``, shape ``(B, 1)``.
        """
        self.eval()
        with torch.no_grad():
            return self(user_ids, item_ids, context)
