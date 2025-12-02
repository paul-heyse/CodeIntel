"""Re-export facade for pipeline run registry types and utilities.

This module provides convenient access to run tracking types for use in the
pipeline orchestration layer (CLI, HTTP handlers, etc.). The actual
implementation lives in codeintel.storage.run_tracking to respect layering
boundaries.

For engine implementations (analytics, graphs, ingestion), use the gateway-based
API via `gateway.runs.*` instead of importing from this module.
"""

from __future__ import annotations

# Re-export all types and constants from the storage layer
from codeintel.storage.run_tracking import (
    ModuleKind,
    PipelineRunRecord,
    PipelineRunTracking,
    PipelineStatus,
    PipelineStepRecord,
    StepStatus,
)

__all__ = [
    "ModuleKind",
    "PipelineRunRecord",
    "PipelineRunTracking",
    "PipelineStatus",
    "PipelineStepRecord",
    "StepStatus",
]
