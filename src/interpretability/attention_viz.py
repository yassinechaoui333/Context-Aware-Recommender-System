"""Attention gate visualisation for ContextNCFAttn.

Defines 6 canonical context scenarios (morning/evening × session start/mid/end),
runs inference for a batch of users, and plots a heatmap of the mean gate
activations per scenario across embedding dimensions.

Output
------
outputs/figures/gate_heatmap.png
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

matplotlib.use("Agg")

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


def _build_context_row(
    hour: float,
    day_of_week: float,
    session_pos_norm: float,
    session_len_norm: float,
    device: int,
) -> np.ndarray:
    """Build a single 9-dim context row using the same cyclic encoding as the pipeline."""
    sin_hour = np.sin(2.0 * np.pi * hour / 24.0)
    cos_hour = np.cos(2.0 * np.pi * hour / 24.0)
    sin_dow = np.sin(2.0 * np.pi * day_of_week / 7.0)
    cos_dow = np.cos(2.0 * np.pi * day_of_week / 7.0)

    device_ohe = [float(device == i) for i in range(3)]  # device_0, device_1, device_2

    return np.array(
        [sin_hour, cos_hour, sin_dow, cos_dow, session_pos_norm, session_len_norm] + device_ohe,
        dtype=np.float32,
    )


# ── 6 canonical scenarios ────────────────────────────────────────────────────

SCENARIOS: list[dict] = [
    {
        "label": "Morning · Session Start",
        "hour": 8.0,
        "day_of_week": 1.0,
        "session_pos_norm": 0.0,
        "session_len_norm": 0.2,
        "device": 0,  # mobile
    },
    {
        "label": "Morning · Session Mid",
        "hour": 8.0,
        "day_of_week": 1.0,
        "session_pos_norm": 0.5,
        "session_len_norm": 0.5,
        "device": 0,
    },
    {
        "label": "Morning · Session End",
        "hour": 8.0,
        "day_of_week": 1.0,
        "session_pos_norm": 1.0,
        "session_len_norm": 0.8,
        "device": 0,
    },
    {
        "label": "Evening · Session Start",
        "hour": 22.0,
        "day_of_week": 5.0,  # Saturday
        "session_pos_norm": 0.0,
        "session_len_norm": 0.2,
        "device": 2,  # desktop
    },
    {
        "label": "Evening · Session Mid",
        "hour": 22.0,
        "day_of_week": 5.0,
        "session_pos_norm": 0.5,
        "session_len_norm": 0.5,
        "device": 2,
    },
    {
        "label": "Evening · Session End",
        "hour": 22.0,
        "day_of_week": 5.0,
        "session_pos_norm": 1.0,
        "session_len_norm": 0.8,
        "device": 2,
    },
]


def run_attention_viz(
    checkpoint_path: str = "outputs/checkpoints/context_ncf_attn/best.ckpt",
    n_users: Optional[int] = None,
    encoders_path: str = "data/processed/encoders.pkl",
    seed: int = 42,
) -> np.ndarray:
    """Generate and save the gate heatmap.

    Parameters
    ----------
    checkpoint_path:
        Path to the ``ContextNCFAttn`` ``.ckpt`` checkpoint.
    n_users:
        Number of users to average gate activations over (default: min(10, n_users)).
    encoders_path:
        Path to the label-encoder pickle.
    seed:
        Random seed for selecting users.

    Returns
    -------
    gate_matrix:
        Shape ``(n_scenarios, embedding_dim)``.
    """
    rng = np.random.default_rng(seed)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Load encoders & model ─────────────────────────────────────────────
    from src.models.context_ncf_attn import ContextNCFAttn

    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at '{ckpt}'. "
            "Run training first: `python main.py train --config configs/context_attn.yaml`"
        )

    with open(encoders_path, "rb") as fh:
        encoders = pickle.load(fh)
    total_users = len(encoders["user"].classes_)
    total_items = len(encoders["item"].classes_)

    device_torch = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ContextNCFAttn.load_from_checkpoint(
        str(ckpt),
        n_users=total_users,
        n_items=total_items,
    ).to(device_torch)
    model.eval()

    # ── 2. Select sample users ───────────────────────────────────────────────
    if n_users is None:
        n_users = min(10, total_users)
    user_sample = rng.choice(total_users, size=n_users, replace=False).tolist()
    # Use the most popular item (index 0 after encoding, offset by +1 for padding)
    item_id = 1

    # ── 3. Compute mean gate per scenario ───────────────────────────────────
    gate_matrix = np.zeros((len(SCENARIOS), model.embedding_dim), dtype=np.float32)

    with torch.no_grad():
        for s_idx, scenario in enumerate(SCENARIOS):
            ctx_row = _build_context_row(
                hour=scenario["hour"],
                day_of_week=scenario["day_of_week"],
                session_pos_norm=scenario["session_pos_norm"],
                session_len_norm=scenario["session_len_norm"],
                device=scenario["device"],
            )
            ctx_batch = (
                torch.tensor(ctx_row, dtype=torch.float32, device=device_torch)
                .unsqueeze(0)
                .expand(n_users, -1)
            )  # (n_users, 9)

            users_t = torch.tensor(user_sample, dtype=torch.long, device=device_torch)
            items_t = torch.tensor([item_id] * n_users, dtype=torch.long, device=device_torch)

            # Forward triggers gate activation storage
            _ = model(users_t, items_t, ctx_batch)

            gate_act = model.gate.last_gate  # (n_users, embedding_dim)
            gate_matrix[s_idx] = gate_act.mean(dim=0).cpu().numpy()

    # ── 4. Plot heatmap ──────────────────────────────────────────────────────
    scenario_labels = [s["label"] for s in SCENARIOS]

    # Only display first 32 dims for readability if embedding_dim is large
    display_dims = min(32, gate_matrix.shape[1])
    data = gate_matrix[:, :display_dims]

    fig, ax = plt.subplots(figsize=(max(10, display_dims * 0.35), 5))
    im = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=0.0, vmax=1.0)

    ax.set_yticks(range(len(scenario_labels)))
    ax.set_yticklabels(scenario_labels, fontsize=10)
    ax.set_xlabel(f"Embedding dimension (first {display_dims} of {gate_matrix.shape[1]})", fontsize=10)
    ax.set_title("ContextNCFAttn — Gate Activations per Scenario", fontsize=13, pad=12)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Gate value (0=blocked · 1=full pass)", fontsize=9)

    plt.tight_layout()
    heatmap_path = FIGURES_DIR / "gate_heatmap.png"
    plt.savefig(str(heatmap_path), dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"[attention_viz] Saved gate heatmap → {heatmap_path}")

    # ── 5. Console summary ───────────────────────────────────────────────────
    print("\n[attention_viz] Mean gate activation per scenario (first 8 dims):")
    header = "Scenario".ljust(32) + "  " + "  ".join(f"d{i:02d}" for i in range(min(8, display_dims)))
    print(header)
    for label, row in zip(scenario_labels, gate_matrix):
        vals = "  ".join(f"{v:.3f}" for v in row[:8])
        print(f"  {label:<30}  {vals}")

    return gate_matrix


if __name__ == "__main__":
    run_attention_viz()
