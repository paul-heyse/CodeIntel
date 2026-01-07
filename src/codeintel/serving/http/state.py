"""Typed FastAPI state for the serving application."""

from __future__ import annotations

from codeintel.serving.context import ServingContext

ServingState = ServingContext

__all__ = ["ServingState"]
