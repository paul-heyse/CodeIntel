"""Ibis view builders and discovery helpers.

View builder functions are defined in `codeintel.storage.views.ibis_views`.
Deterministic discovery for materialization lives in `codeintel.storage.views.discovery`.

This package intentionally keeps its public surface minimal to avoid import
cycles during storage bootstrap.
"""

from __future__ import annotations

__all__: list[str] = []
