"""Rustworkx-backed GraphEngine implementation."""

from __future__ import annotations

from codeintel.build.graphs.engine.nx_engine import NxGraphEngine


class RxGraphEngine(NxGraphEngine):
    """Rustworkx-backed graph engine with NetworkX compatibility outputs."""


__all__ = ["RxGraphEngine"]
