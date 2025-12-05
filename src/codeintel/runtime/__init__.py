"""Unified runtime context and orchestration for CodeIntel pipelines.

This package provides the canonical RunContext type and orchestration utilities
that unify run identity across ingestion, graphs, and analytics engines.
"""

from __future__ import annotations

from codeintel.runtime.context import RunContext, RunKind, TriggerKind
from codeintel.runtime.ids import (
    RUN_PREFIX_ANALYTICS,
    RUN_PREFIX_GRAPHS,
    RUN_PREFIX_INGEST,
    RUN_PREFIX_PIPELINE,
    RUN_PREFIX_PLAN,
    new_run_id,
)
from codeintel.runtime.orchestrator import new_run_context

__all__ = [
    "RUN_PREFIX_ANALYTICS",
    "RUN_PREFIX_GRAPHS",
    "RUN_PREFIX_INGEST",
    "RUN_PREFIX_PIPELINE",
    "RUN_PREFIX_PLAN",
    "RunContext",
    "RunKind",
    "TriggerKind",
    "new_run_context",
    "new_run_id",
]
