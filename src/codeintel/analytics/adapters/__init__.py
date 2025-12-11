"""Persistence adapters for analytics data access.

This package provides adapter classes that handle all database I/O for
analytics modules. Adapters encapsulate:
- Loading data from DuckDB tables
- Persisting computed results
- Managing transaction boundaries

By separating I/O into adapters, the computation layer remains pure
and easily testable.

Modules
-------
functions
    Adapters for function metrics and types tables.
graphs
    Adapters for graph metrics tables.
profiles
    Adapters for profile aggregation tables.
subsystems
    Adapters for subsystem classification tables.
semantic_roles
    Adapters for semantic role classification tables.
entrypoints
    Adapters for entrypoint detection tables.
data_models
    Adapters for data model usage tables.
base
    Base adapter classes and protocols.
"""

from __future__ import annotations

from codeintel.analytics.adapters.base import (
    AnalyticsAdapter,
    DeleteScope,
)
from codeintel.analytics.adapters.data_models import (
    DataModelUsageAdapter,
)
from codeintel.analytics.adapters.entrypoints import (
    EntrypointsAdapter,
    EntrypointTestsAdapter,
)
from codeintel.analytics.adapters.functions import (
    FunctionMetricsAdapter,
    FunctionTypesAdapter,
)
from codeintel.analytics.adapters.profiles import (
    FileProfileAdapter,
    FunctionProfileAdapter,
    ModuleProfileAdapter,
)
from codeintel.analytics.adapters.schema_adapter import (
    SchemaAwareBatchAdapter,
    SchemaValidationMixin,
)
from codeintel.analytics.adapters.semantic_roles import (
    SemanticRolesFunctionsAdapter,
    SemanticRolesModulesAdapter,
)
from codeintel.analytics.adapters.subsystems import (
    SubsystemModulesAdapter,
    SubsystemsAdapter,
)

__all__ = [
    "AnalyticsAdapter",
    "DataModelUsageAdapter",
    "DeleteScope",
    "EntrypointTestsAdapter",
    "EntrypointsAdapter",
    "FileProfileAdapter",
    "FunctionMetricsAdapter",
    "FunctionProfileAdapter",
    "FunctionTypesAdapter",
    "ModuleProfileAdapter",
    "SchemaAwareBatchAdapter",
    "SchemaValidationMixin",
    "SemanticRolesFunctionsAdapter",
    "SemanticRolesModulesAdapter",
    "SubsystemModulesAdapter",
    "SubsystemsAdapter",
]
