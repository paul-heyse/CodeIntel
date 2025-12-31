"""Target metadata helpers for the build system."""

from __future__ import annotations

from typing import Literal

from codeintel.build.hamilton.dag_catalog import TargetDescriptor

TargetModule = Literal["ingestion", "graphs", "analytics", "export", "views"]
"""Classification of which target module produces an output."""


__all__ = [
    "TargetDescriptor",
    "TargetModule",
]
