"""Deprecated shim - use GraphRuntimeDouble."""

from __future__ import annotations

from tests._helpers.fakes.graph_runtime import GraphCallRecord as GraphCall
from tests._helpers.fakes.graph_runtime import GraphRuntimeDouble as StubGraphEngine

__all__ = ["GraphCall", "StubGraphEngine"]
