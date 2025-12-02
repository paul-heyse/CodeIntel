"""Plugin architecture for ingestion pipelines.

This package provides the plugin protocol, registry, decorators,
harness, config factory, and contracts for building extensible ingestion pipelines.
"""

from __future__ import annotations

# Import builtin plugins to trigger registration
from codeintel.ingestion.plugins import builtin as _builtin
from codeintel.ingestion.plugins.config_factory import (
    DEFAULT_CONTEXT_MAPPINGS,
    ConfigFactory,
    ConfigMapping,
    get_config_fields,
    infer_config_mapping,
)
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
from codeintel.ingestion.plugins.harness import (
    HarnessConfig,
    HarnessContext,
    IngestExecutionHarness,
    with_harness,
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
    PlanOptions,
    get_ingest_registry,
    list_ingest_plugins,
    plan_ingest_plugins,
    register_ingest_plugin,
)

__all__ = [
    "DEFAULT_CONTEXT_MAPPINGS",
    "DEFAULT_INGEST_PLUGINS",
    "ClassBasedIngestPlugin",
    "ColumnConstraint",
    "ConfigFactory",
    "ConfigMapping",
    "ContractValidationResult",
    "ContractViolation",
    "ForeignKeyConstraint",
    "FunctionalIngestPlugin",
    "HarnessConfig",
    "HarnessContext",
    "IngestContractSpec",
    "IngestContractValidator",
    "IngestExecutionHarness",
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
    "PlanOptions",
    "_builtin",
    "foreign_key_contract",
    "get_config_fields",
    "get_ingest_registry",
    "infer_config_mapping",
    "ingest_plugin",
    "list_ingest_plugins",
    "not_null_contract",
    "plan_ingest_plugins",
    "register_class_plugin",
    "register_ingest_plugin",
    "row_count_contract",
    "with_harness",
]
