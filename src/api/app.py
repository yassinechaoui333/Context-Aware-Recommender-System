"""FastAPI application for the Context-Aware Recommender System.

Endpoints
---------
GET  /health               — liveness check
POST /recommend            — top-k recommendations for a user + context
GET  /items/{item_id}      — item metadata

Startup loads:
  • TorchScript model (outputs/model.pt)
  • Movies lookup DataFrame (data/processed/movies.parquet)
  • Encoders pickle (data/processed/encoders.pkl)
  • Redis client (optional; skipped gracefully if Redis unavailable)
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException

from src.api.feature_builder import build_context_tensor
from src.api.schemas import (
    HealthResponse,
    ItemResponse,
    RecommendRequest,
    RecommendResponse,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

MODEL_PT_PATH = Path("outputs/model.pt")
MOVIES_PATH = Path("data/processed/movies.parquet")
ENCODERS_PATH = Path("data/processed/encoders.pkl")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
CACHE_TTL = 300  # seconds

# ---------------------------------------------------------------------------
# Application state (populated at startup)
# ---------------------------------------------------------------------------

_state: dict = {}


def _try_connect_redis():
    """Return a Redis client or None if Redis is unavailable."""
    try:
        import redis  # type: ignore

        client = redis.from_url(REDIS_URL, socket_connect_timeout=1)
        client.ping()
        return client
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Lifespan — load resources once at startup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Model ────────────────────────────────────────────────────────────────
    if MODEL_PT_PATH.exists():
        _state["model"] = torch.jit.load(str(MODEL_PT_PATH), map_location="cpu")
        _state["model"].eval()
    else:
        # Fallback: load directly from checkpoint (requires src.models to be importable)
        _load_model_from_checkpoint()

    # ── Movies lookup ────────────────────────────────────────────────────────
    if MOVIES_PATH.exists():
        movies_df = pd.read_parquet(MOVIES_PATH)
        # Build a fast dict: item_id (encoded) → {title, genres}
        _state["movies"] = {
            int(row["item_id"]): {"title": str(row["title"]), "genres": str(row.get("genres", ""))}
            for _, row in movies_df.iterrows()
        }
    else:
        _state["movies"] = {}

    # ── Encoders ─────────────────────────────────────────────────────────────
    if ENCODERS_PATH.exists():
        with open(ENCODERS_PATH, "rb") as fh:
            encoders = pickle.load(fh)
        _state["n_users"] = len(encoders["user"].classes_)
        _state["n_items"] = len(encoders["item"].classes_)
        _state["all_items"] = np.arange(_state["n_items"], dtype=np.int64)
    else:
        _state["n_users"] = 10_000
        _state["n_items"] = 5_000
        _state["all_items"] = np.arange(_state["n_items"], dtype=np.int64)

    # ── Redis ────────────────────────────────────────────────────────────────
    _state["redis"] = _try_connect_redis()
    if _state["redis"] is None:
        print("[api] Redis unavailable — caching disabled.")

    yield

    # ── Teardown ─────────────────────────────────────────────────────────────
    _state.clear()


def _load_model_from_checkpoint() -> None:
    """Fallback: load ContextNCFAttn directly from the .ckpt file."""
    ckpt_path = Path("outputs/checkpoints/context_ncf_attn/best.ckpt")
    if not ckpt_path.exists():
        print(
            "[api] WARNING: No model found at outputs/model.pt or "
            f"'{ckpt_path}'. /recommend will return 503."
        )
        _state["model"] = None
        return

    from src.models.context_ncf_attn import ContextNCFAttn

    if ENCODERS_PATH.exists():
        with open(ENCODERS_PATH, "rb") as fh:
            enc = pickle.load(fh)
        n_users = len(enc["user"].classes_)
        n_items = len(enc["item"].classes_)
    else:
        n_users, n_items = 10_000, 5_000

    model = ContextNCFAttn.load_from_checkpoint(
        str(ckpt_path), n_users=n_users, n_items=n_items
    ).cpu()
    model.eval()
    _state["model"] = model


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Context-Aware Recommender API",
    description="Neural Collaborative Filtering with context-aware attention gate.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Helper — cache key
# ---------------------------------------------------------------------------


def _cache_key(user_id: int, ctx_hash: str, k: int) -> str:
    return f"rec:u{user_id}:ctx:{ctx_hash}:k{k}"


def _hash_context(req: RecommendRequest) -> str:
    ctx = req.context
    raw = f"{ctx.hour}:{ctx.day_of_week}:{ctx.session_pos}:{ctx.session_len}:{ctx.device}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------


def _infer(user_id: int, context_tensor: torch.Tensor, k: int) -> tuple[list[int], list[float]]:
    """Score all items and return top-k (item_ids, scores)."""
    model = _state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    n_items = _state["n_items"]

    user_t = torch.tensor([user_id] * n_items, dtype=torch.long)
    items_t = torch.tensor(list(range(n_items)), dtype=torch.long)
    ctx_t = context_tensor.unsqueeze(0).expand(n_items, -1)  # (n_items, 9)

    with torch.no_grad():
        scores = model(user_t, items_t, ctx_t).squeeze(-1).numpy()  # (n_items,)

    top_k_idx = np.argsort(scores)[::-1][:k]
    top_item_ids = top_k_idx.tolist()
    top_scores = scores[top_k_idx].tolist()
    return top_item_ids, top_scores


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model="ContextNCFAttn")


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest) -> RecommendResponse:
    t0 = time.perf_counter()
    n_users = _state.get("n_users", 10_000)

    if req.user_id < 0 or req.user_id >= n_users:
        raise HTTPException(
            status_code=422, detail=f"user_id must be in [0, {n_users - 1}]."
        )

    ctx_hash = _hash_context(req)
    cache_key = _cache_key(req.user_id, ctx_hash, req.k)

    # ── Cache hit ─────────────────────────────────────────────────────────────
    redis = _state.get("redis")
    if redis is not None:
        cached = redis.get(cache_key)
        if cached:
            data = json.loads(cached)
            latency_ms = (time.perf_counter() - t0) * 1000
            return RecommendResponse(latency_ms=latency_ms, **data)

    # ── Inference ─────────────────────────────────────────────────────────────
    context_tensor = build_context_tensor(req.context)
    top_item_ids, top_scores = _infer(req.user_id, context_tensor, req.k)

    # ── Enrich with titles ────────────────────────────────────────────────────
    movies = _state.get("movies", {})
    titles = [movies.get(iid, {}).get("title", f"item_{iid}") for iid in top_item_ids]

    payload = {
        "user_id": req.user_id,
        "items": top_item_ids,
        "titles": titles,
        "scores": [round(s, 6) for s in top_scores],
    }

    # ── Cache write ───────────────────────────────────────────────────────────
    if redis is not None:
        redis.setex(cache_key, CACHE_TTL, json.dumps(payload))

    latency_ms = (time.perf_counter() - t0) * 1000
    return RecommendResponse(latency_ms=latency_ms, **payload)


@app.get("/items/{item_id}", response_model=ItemResponse)
def get_item(item_id: int) -> ItemResponse:
    movies = _state.get("movies", {})
    if item_id not in movies:
        raise HTTPException(status_code=404, detail=f"item_id {item_id} not found.")
    meta = movies[item_id]
    return ItemResponse(id=item_id, title=meta["title"], genres=meta.get("genres", ""))
