"""Unified data pipeline for analytics datasets.

This package provides a typed dataset system with dependency DAG,
contract validation, and lineage tracking.

Key Components
--------------
DatasetSpec
    Specification for a typed dataset including schema and contracts.
DatasetComputation
    Protocol for dataset computation implementations.
PipelineScheduler
    Schedules dataset computations respecting dependencies.
DatasetLineage
    Tracks provenance of dataset rows.

Example
-------
>>> from codeintel.analytics.pipeline import DatasetSpec, PipelineScheduler
>>> spec = DatasetSpec(
...     name="analytics.function_metrics",
...     inputs=("core.goids",),
...     outputs=("analytics.function_metrics",),
... )
>>> scheduler = PipelineScheduler()
>>> plan = scheduler.plan(["analytics.function_metrics"])
"""

from __future__ import annotations

from codeintel.analytics.pipeline.contracts import (
    DatasetContract,
    DatasetContractValidator,
)
from codeintel.analytics.pipeline.lineage import (
    DatasetLineage,
    LineageStore,
)
from codeintel.analytics.pipeline.protocol import (
    DatasetComputation,
    DatasetSpec,
    PipelineContext,
)
from codeintel.analytics.pipeline.scheduler import (
    ExecutionPlan,
    PipelineReport,
    PipelineScheduler,
)

__all__ = [
    "DatasetComputation",
    "DatasetContract",
    "DatasetContractValidator",
    "DatasetLineage",
    "DatasetSpec",
    "ExecutionPlan",
    "LineageStore",
    "PipelineContext",
    "PipelineReport",
    "PipelineScheduler",
]
