"""Pydantic fixture models used by heuristics tests."""

from pydantic import BaseModel, Field


class UserPayload(BaseModel):
    """Capture minimal user payload for heuristic config checks."""

    id: int
    name: str = Field(..., max_length=64)
