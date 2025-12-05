"""Status type definitions and relationships.

This module documents the hierarchy of status types used across the codebase.

Status Hierarchy
----------------
Most specific to least specific:

1. PluginStatus (core.plugins.result)
   Values: "succeeded", "failed", "skipped"
   Use: Individual plugin execution outcomes

2. StepStatus (storage.tracking.run_tracking)
   Values: "pending", "running", "succeeded", "failed", "skipped"
   Use: Pipeline step tracking (superset of PluginStatus + lifecycle)

3. ExecutionStatus (core.plugins.report)
   Values: "succeeded", "failed", "partial"
   Use: Aggregate execution report status

4. PipelineStatus (storage.tracking.run_tracking)
   Values: "running", "succeeded", "failed", "partial"
   Use: Overall pipeline run status

Type Relationships
------------------
- PluginStatus values are a subset of StepStatus values
- ExecutionStatus has "partial" instead of "skipped" (for aggregate outcomes)
- PipelineStatus is like ExecutionStatus but includes "running" state

Semantic Equivalence
--------------------
Some types across domains are semantically identical:

- AnalyticsStatus (analytics.runtime.manifest) == PluginStatus
  Both use: "succeeded", "failed", "skipped"

- GraphPluginStatus (implicitly from result) == PluginStatus
  Both represent individual plugin execution outcomes
"""

from __future__ import annotations

from codeintel.core.plugins.report import ExecutionStatus
from codeintel.core.plugins.result import PluginStatus
from codeintel.storage.tracking.run_tracking import PipelineStatus, StepStatus

__all__ = [
    "ExecutionStatus",
    "PipelineStatus",
    "PluginStatus",
    "StepStatus",
]
