"""Plugin architecture for ingestion pipelines.

This package provides the plugin protocol, registry, decorators,
and contracts for building extensible ingestion pipelines.
"""

from __future__ import annotations

# Import builtin plugins to trigger registration
from codeintel.ingestion.plugins import builtin as _builtin  # noqa: F401
from codeintel.ingestion.plugins.contracts import (
    ColumnConstraint,
    ContractValidationResult,
    ContractViolation,
    ForeignKeyConstraint,
    IngestContractSpec,
    IngestContractValidator,
    foreign_key_contract,
    not_null_contract,
    row_count_contract,
)
from codeintel.ingestion.plugins.decorators import (
    ClassBasedIngestPlugin,
    FunctionalIngestPlugin,
    ingest_plugin,
    register_class_plugin,
)
from codeintel.ingestion.plugins.protocol import (
    DEFAULT_INGEST_PLUGINS,
    IngestIsolationKind,
    IngestPluginContext,
    IngestPluginMetadata,
    IngestPluginPlan,
    IngestPluginProtocol,
    IngestPluginResult,
    IngestPluginSkip,
    IngestResourceHints,
    IngestRuntimeScratch,
    IngestSeverity,
    IngestStage,
)
from codeintel.ingestion.plugins.registry import (
    IngestPluginRegistry,
    get_ingest_registry,
    list_ingest_plugins,
    plan_ingest_plugins,
    register_ingest_plugin,
)

__all__ = [
    "DEFAULT_INGEST_PLUGINS",
    "ClassBasedIngestPlugin",
    "ColumnConstraint",
    "ContractValidationResult",
    "ContractViolation",
    "ForeignKeyConstraint",
    "FunctionalIngestPlugin",
    "IngestContractSpec",
    "IngestContractValidator",
    "IngestIsolationKind",
    "IngestPluginContext",
    "IngestPluginMetadata",
    "IngestPluginPlan",
    "IngestPluginProtocol",
    "IngestPluginRegistry",
    "IngestPluginResult",
    "IngestPluginSkip",
    "IngestResourceHints",
    "IngestRuntimeScratch",
    "IngestSeverity",
    "IngestStage",
    "foreign_key_contract",
    "get_ingest_registry",
    "ingest_plugin",
    "list_ingest_plugins",
    "not_null_contract",
    "plan_ingest_plugins",
    "register_class_plugin",
    "register_ingest_plugin",
    "row_count_contract",
]
