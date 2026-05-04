"""SHAP analysis for the ContextNCFAttn model.

Wraps the model so that only the 9-dim context vector varies (user and item
embeddings are fixed to a "median" representative), then runs
``shap.KernelExplainer`` over 500 held-out test samples.

Outputs
-------
outputs/figures/shap_values.npy   — raw SHAP values, shape (500, 9)
outputs/figures/shap_summary.png  — beeswarm summary plot
outputs/figures/shap_importance.png — mean |SHAP| bar chart
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")  # headless backend

CONTEXT_COLS = [
    "sin_hour",
    "cos_hour",
    "sin_dow",
    "cos_dow",
    "session_pos_norm",
    "session_len_norm",
    "device_0",
    "device_1",
    "device_2",
]

FIGURES_DIR = Path("outputs/figures")
FEATURES_DIR = Path("data/features")
PROCESSED_DIR = Path("data/processed")


# ---------------------------------------------------------------------------
# Model wrapper — context-only callable for SHAP
# ---------------------------------------------------------------------------


class _ContextOnlyWrapper:
    """Wraps ContextNCFAttn so only the context vector varies.

    A fixed *median user* and the most popular item are used for all calls.
    This isolates the contribution of context features to the model's output.
    """

    def __init__(self, model, user_id: int, item_id: int, device: torch.device) -> None:
        self.model = model
        self.user_id = user_id
        self.item_id = item_id
        self.device = device
        self.model.eval()

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Score N context vectors and return predicted scores (N,).

        Parameters
        ----------
        X:
            Shape ``(N, 9)``, float32.
        """
        N = X.shape[0]
        users = torch.tensor([self.user_id] * N, dtype=torch.long, device=self.device)
        items = torch.tensor([self.item_id] * N, dtype=torch.long, device=self.device)
        ctx = torch.tensor(X, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            scores = self.model(users, items, ctx).squeeze(-1).cpu().numpy()
        return scores.astype(np.float32)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_shap_analysis(
    checkpoint_path: str = "outputs/checkpoints/context_ncf_attn/best.ckpt",
    n_background: int = 100,
    n_test_samples: int = 500,
    seed: int = 42,
    encoders_path: str = "data/processed/encoders.pkl",
    test_features_path: str = "data/features/test.parquet",
) -> None:
    """Run SHAP analysis and save all artefacts.

    Parameters
    ----------
    checkpoint_path:
        Path to the ``ContextNCFAttn`` ``.ckpt`` checkpoint.
    n_background:
        Number of background samples for KernelExplainer.
    n_test_samples:
        Number of test samples to explain.
    seed:
        Random seed for reproducibility.
    encoders_path:
        Path to the label-encoder pickle (needed to resolve n_users/n_items).
    test_features_path:
        Path to the featurised test parquet (provides context vectors).
    """
    import shap

    rng = np.random.default_rng(seed)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load model ────────────────────────────────────────────────────────
    from src.models.context_ncf_attn import ContextNCFAttn

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at '{ckpt}'.  "
            "Run `python main.py train --config configs/context_attn.yaml` first."
        )

    with open(encoders_path, "rb") as fh:
        encoders = pickle.load(fh)
    n_users = len(encoders["user"].classes_)
    n_items = len(encoders["item"].classes_)

    model = ContextNCFAttn.load_from_checkpoint(
        str(ckpt),
        n_users=n_users,
        n_items=n_items,
    ).to(device)

    # ── 2. Load context arrays ───────────────────────────────────────────────
    df_test = pd.read_parquet(test_features_path)

    missing_cols = [c for c in CONTEXT_COLS if c not in df_test.columns]
    if missing_cols:
        raise KeyError(f"Missing context columns in test data: {missing_cols}")

    test_ctx = df_test[CONTEXT_COLS].to_numpy(dtype=np.float32)

    # Clamp samples to available rows
    n_test_samples = min(n_test_samples, len(test_ctx))

    # Pick a representative "median user" and most popular item
    user_id = int(df_test["user_id"].median())
    item_counts = df_test["item_id"].value_counts()
    item_id = int(item_counts.index[0])

    # ── 3. Background and explanation sets ───────────────────────────────────
    bg_indices = rng.choice(len(test_ctx), size=min(n_background, len(test_ctx)), replace=False)
    background = test_ctx[bg_indices]

    test_indices = rng.choice(len(test_ctx), size=n_test_samples, replace=False)
    test_samples = test_ctx[test_indices]

    # ── 4. SHAP KernelExplainer ──────────────────────────────────────────────
    wrapper = _ContextOnlyWrapper(model, user_id, item_id, device)
    explainer = shap.KernelExplainer(wrapper, background)

    print(f"[shap_analysis] Computing SHAP values for {n_test_samples} samples …")
    shap_values = explainer.shap_values(test_samples, nsamples=200)  # shape (N, 9)
    shap_values = np.array(shap_values, dtype=np.float32)

    # ── 5. Save raw values ───────────────────────────────────────────────────
    npy_path = FIGURES_DIR / "shap_values.npy"
    np.save(str(npy_path), shap_values)
    print(f"[shap_analysis] Saved SHAP values → {npy_path}  shape={shap_values.shape}")

    # ── 6. Summary plot (beeswarm) ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        test_samples,
        feature_names=CONTEXT_COLS,
        show=False,
        plot_type="dot",
    )
    summary_path = FIGURES_DIR / "shap_summary.png"
    plt.tight_layout()
    plt.savefig(str(summary_path), dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"[shap_analysis] Saved summary plot → {summary_path}")

    # ── 7. Importance bar chart ──────────────────────────────────────────────
    mean_abs_shap = np.abs(shap_values).mean(axis=0)  # (9,)
    order = np.argsort(mean_abs_shap)[::-1]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(
        [CONTEXT_COLS[i] for i in order],
        mean_abs_shap[order],
        color="#4c72b0",
        edgecolor="white",
    )
    ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=9)
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title("Context Feature Importance (ContextNCFAttn)", fontsize=13)
    ax.invert_yaxis()
    plt.tight_layout()
    importance_path = FIGURES_DIR / "shap_importance.png"
    plt.savefig(str(importance_path), dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"[shap_analysis] Saved importance plot → {importance_path}")

    # ── 8. Brief console summary ─────────────────────────────────────────────
    print("\n[shap_analysis] Top context features by mean |SHAP|:")
    for rank, idx in enumerate(order, 1):
        print(f"  {rank:>2}. {CONTEXT_COLS[idx]:<22}  {mean_abs_shap[idx]:.5f}")

    return shap_values


if __name__ == "__main__":
    run_shap_analysis()
