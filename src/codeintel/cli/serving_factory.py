"""CLI-facing serving app factories for multi-worker Uvicorn."""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.serving.http.app import create_serving_app
from codeintel.serving.settings import get_serving_settings

if TYPE_CHECKING:
    from fastapi import FastAPI


def create_serving_app_from_env() -> FastAPI:
    """Create the serving FastAPI app using environment-derived settings.

    Returns
    -------
    FastAPI
        Configured serving application.
    """
    settings = get_serving_settings()
    return create_serving_app(settings)


__all__ = ["create_serving_app_from_env"]
