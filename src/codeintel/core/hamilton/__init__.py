"""Shared Hamilton integration primitives (core-owned).

This package contains small, dependency-light utilities shared across build,
storage, and serving that relate to Hamilton tagging and execution records.

It exists to avoid layering violations (e.g., storage importing build) while
keeping the Hamilton-first architecture explicit.
"""

from __future__ import annotations

__all__: list[str] = []
