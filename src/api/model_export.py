"""Export ContextNCFAttn to TorchScript and ONNX formats.

Usage
-----
    python -m src.api.model_export
    # or via main.py (after adding the export command)

Outputs
-------
outputs/model.pt    — TorchScript (CPU)
outputs/model.onnx  — ONNX opset 17
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import torch

OUTPUTS_DIR = Path("outputs")
CKPT_PATH = OUTPUTS_DIR / "checkpoints" / "context_ncf_attn" / "best.ckpt"
ENCODERS_PATH = Path("data/processed/encoders.pkl")

BATCH_SIZE = 4   # dummy batch for tracing
CONTEXT_DIM = 9
EMBEDDING_DIM = 64  # default; loaded from checkpoint hparams


def export(
    checkpoint_path: str = str(CKPT_PATH),
    encoders_path: str = str(ENCODERS_PATH),
    out_dir: str = str(OUTPUTS_DIR),
) -> None:
    """Export the best ContextNCFAttn checkpoint to TorchScript and ONNX.

    Parameters
    ----------
    checkpoint_path:
        Path to the ``.ckpt`` file produced by training.
    encoders_path:
        Path to ``encoders.pkl`` (provides n_users / n_items).
    out_dir:
        Directory to write ``model.pt`` and ``model.onnx``.
    """
    from src.models.context_ncf_attn import ContextNCFAttn

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at '{ckpt}'. "
            "Train first: `python main.py train --config configs/context_attn.yaml`"
        )

    with open(encoders_path, "rb") as fh:
        encoders = pickle.load(fh)
    n_users = len(encoders["user"].classes_)
    n_items = len(encoders["item"].classes_)

    print(f"[export] Loading checkpoint: {ckpt}")
    model = ContextNCFAttn.load_from_checkpoint(
        str(ckpt),
        n_users=n_users,
        n_items=n_items,
    ).cpu()
    model.eval()

    # ── Dummy inputs ──────────────────────────────────────────────────────────
    user_ids = torch.zeros(BATCH_SIZE, dtype=torch.long)
    item_ids = torch.ones(BATCH_SIZE, dtype=torch.long)
    context = torch.zeros(BATCH_SIZE, CONTEXT_DIM)

    # ── TorchScript ──────────────────────────────────────────────────────────
    ts_path = out_path / "model.pt"
    print(f"[export] TorchScript → {ts_path}")
    scripted = torch.jit.trace(model, (user_ids, item_ids, context))
    scripted.save(str(ts_path))

    # ── ONNX ─────────────────────────────────────────────────────────────────
    onnx_path = out_path / "model.onnx"
    print(f"[export] ONNX       → {onnx_path}")
    torch.onnx.export(
        model,
        (user_ids, item_ids, context),
        str(onnx_path),
        opset_version=17,
        input_names=["user_ids", "item_ids", "context"],
        output_names=["score"],
        dynamic_axes={
            "user_ids": {0: "batch"},
            "item_ids": {0: "batch"},
            "context": {0: "batch"},
            "score": {0: "batch"},
        },
    )

    # ── Verify ONNX vs PyTorch ────────────────────────────────────────────────
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_inputs = {
        "user_ids": user_ids.numpy(),
        "item_ids": item_ids.numpy(),
        "context": context.numpy(),
    }
    ort_out = sess.run(["score"], ort_inputs)[0]  # (B, 1)
    with torch.no_grad():
        pt_out = model(user_ids, item_ids, context).numpy()

    max_diff = float(np.abs(ort_out - pt_out).max())
    assert max_diff < 1e-4, f"ONNX/PyTorch mismatch: max diff = {max_diff:.2e}"
    print(f"[export] Export verified ✓  (max abs diff = {max_diff:.2e})")


if __name__ == "__main__":
    export()
