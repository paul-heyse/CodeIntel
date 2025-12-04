"""Entrypoint detection computation module.

This package provides pure computation functions for detecting HTTP/CLI/job
entrypoints from source modules.
"""

from __future__ import annotations

from codeintel.analytics.compute.entrypoints.detection import (
    DetectorSettings,
    EntryPointCandidate,
    detect_entrypoints,
)

__all__ = [
    "DetectorSettings",
    "EntryPointCandidate",
    "detect_entrypoints",
]
