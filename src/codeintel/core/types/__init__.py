"""Core type definitions and relationships.

This package provides type definitions, aliases, and documentation for
status types and other shared type constructs.
"""

from __future__ import annotations

from codeintel.core.types.domain import (
    GraphBackendConfig,
    GraphFeatureFlags,
    PytestCallEntry,
    PytestTestEntry,
    ScipDocument,
    ScipOccurrence,
    ScipRange,
    SnapshotRef,
    normalize_pytest_entry,
    normalize_scip_document,
    validate_pytest_entry,
    validate_scip_document,
)
from codeintel.core.types.status import (
    ExecutionStatus,
    PipelineStatus,
    PluginStatus,
    StepStatus,
)

__all__ = [
    "ExecutionStatus",
    "GraphBackendConfig",
    "GraphFeatureFlags",
    "PipelineStatus",
    "PluginStatus",
    "PytestCallEntry",
    "PytestTestEntry",
    "ScipDocument",
    "ScipOccurrence",
    "ScipRange",
    "SnapshotRef",
    "StepStatus",
    "normalize_pytest_entry",
    "normalize_scip_document",
    "validate_pytest_entry",
    "validate_scip_document",
]
