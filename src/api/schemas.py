"""Pydantic request/response schemas for the recommendation API."""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class ContextInput(BaseModel):
    """Raw context inputs supplied by the client."""

    hour: int = Field(..., ge=0, le=23, description="Hour of day (0–23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Mon … 6=Sun)")
    session_pos: int = Field(..., ge=0, description="0-indexed position within the session")
    session_len: int = Field(..., ge=1, description="Total number of items in the session")
    device: int = Field(..., ge=0, le=2, description="Device proxy: 0=mobile · 1=tablet · 2=desktop")


class RecommendRequest(BaseModel):
    """Payload for ``POST /recommend``."""

    user_id: int = Field(..., ge=0, description="Encoded user ID (0-indexed)")
    context: ContextInput
    k: int = Field(default=10, ge=1, le=50, description="Number of items to recommend")


class RecommendResponse(BaseModel):
    """Response payload for ``POST /recommend``."""

    user_id: int
    items: List[int]
    titles: List[str]
    scores: List[float]
    latency_ms: float


class HealthResponse(BaseModel):
    """Response payload for ``GET /health``."""

    status: str
    model: str


class ItemResponse(BaseModel):
    """Response payload for ``GET /items/{item_id}``."""

    id: int
    title: str
    genres: str
