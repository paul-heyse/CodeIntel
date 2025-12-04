"""Hotspot computation module.

This package provides pure computation functions for analyzing file churn
and complexity hotspots.
"""

from __future__ import annotations

from codeintel.analytics.compute.hotspots.metrics import (
    FileChurn,
    build_hotspots,
)

__all__ = [
    "FileChurn",
    "build_hotspots",
]
