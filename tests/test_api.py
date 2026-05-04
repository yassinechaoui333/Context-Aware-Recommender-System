"""Tests for the FastAPI recommendation API (Phase 9 · Step 9.5).

Uses ``httpx.AsyncClient`` with the ``ASGITransport`` to avoid spinning up a
live server.  The tests patch expensive startup operations (model loading,
Redis) so they run entirely in-process with no external dependencies.
"""
from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from httpx import ASGITransport, AsyncClient


# ---------------------------------------------------------------------------
# Fixtures — patch heavy startup so tests are fast / offline
# ---------------------------------------------------------------------------


def _make_dummy_model(n_items: int = 200) -> MagicMock:
    """Return a mock that mimics the TorchScript/Lightning model interface."""
    mock = MagicMock()

    def _forward(user_ids, item_ids, context):
        B = user_ids.shape[0]
        return torch.rand(B, 1)

    mock.__call__ = _forward
    mock.side_effect = _forward
    return mock


def _make_state(n_users: int = 100, n_items: int = 200) -> dict[str, Any]:
    return {
        "model": _make_dummy_model(n_items),
        "movies": {i: {"title": f"Movie {i}", "genres": "Drama"} for i in range(n_items)},
        "n_users": n_users,
        "n_items": n_items,
        "all_items": np.arange(n_items, dtype=np.int64),
        "redis": None,
    }


@pytest.fixture()
async def client():
    """Async HTTP client with patched application state."""
    from src.api import app as app_module

    state = _make_state()

    with patch.object(app_module, "_state", state):
        async with AsyncClient(
            transport=ASGITransport(app=app_module.app), base_url="http://test"
        ) as c:
            yield c


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.anyio
async def test_health(client: AsyncClient) -> None:
    resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "model" in body


@pytest.mark.anyio
async def test_recommend_valid(client: AsyncClient) -> None:
    payload = {
        "user_id": 1,
        "context": {
            "hour": 10,
            "day_of_week": 1,
            "session_pos": 0,
            "session_len": 5,
            "device": 0,
        },
        "k": 5,
    }
    resp = await client.post("/recommend", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert len(body["items"]) == 5
    assert len(body["titles"]) == 5
    assert len(body["scores"]) == 5
    assert body["user_id"] == 1
    assert "latency_ms" in body


@pytest.mark.anyio
async def test_recommend_default_k(client: AsyncClient) -> None:
    """k defaults to 10 when not supplied."""
    payload = {
        "user_id": 0,
        "context": {
            "hour": 20,
            "day_of_week": 5,
            "session_pos": 2,
            "session_len": 8,
            "device": 2,
        },
    }
    resp = await client.post("/recommend", json=payload)
    assert resp.status_code == 200
    assert len(resp.json()["items"]) == 10


@pytest.mark.anyio
async def test_recommend_invalid_hour(client: AsyncClient) -> None:
    """hour=99 must fail with 422 Unprocessable Entity."""
    payload = {
        "user_id": 1,
        "context": {
            "hour": 99,          # invalid
            "day_of_week": 1,
            "session_pos": 0,
            "session_len": 5,
            "device": 0,
        },
        "k": 10,
    }
    resp = await client.post("/recommend", json=payload)
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_recommend_invalid_device(client: AsyncClient) -> None:
    """device=5 must fail with 422."""
    payload = {
        "user_id": 1,
        "context": {
            "hour": 10,
            "day_of_week": 1,
            "session_pos": 0,
            "session_len": 5,
            "device": 5,         # invalid (max=2)
        },
    }
    resp = await client.post("/recommend", json=payload)
    assert resp.status_code == 422


@pytest.mark.anyio
async def test_get_item_found(client: AsyncClient) -> None:
    resp = await client.get("/items/1")
    assert resp.status_code == 200
    body = resp.json()
    assert "title" in body
    assert body["id"] == 1


@pytest.mark.anyio
async def test_get_item_not_found(client: AsyncClient) -> None:
    resp = await client.get("/items/999999")
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_recommend_latency(client: AsyncClient) -> None:
    """End-to-end latency (mocked model) must be < 200 ms."""
    payload = {
        "user_id": 1,
        "context": {
            "hour": 10,
            "day_of_week": 1,
            "session_pos": 0,
            "session_len": 5,
            "device": 0,
        },
        "k": 10,
    }
    t0 = time.perf_counter()
    resp = await client.post("/recommend", json=payload)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert resp.status_code == 200
    assert elapsed_ms < 200, f"Latency {elapsed_ms:.1f} ms exceeds 200 ms"
