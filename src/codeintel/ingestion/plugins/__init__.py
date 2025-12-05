"""Plugin architecture for ingestion pipelines.

This package provides the plugin protocol, registry, config factory, and
contracts for building extensible ingestion pipelines.

Class-Based Plugins
-------------------
The following class-based plugins are available:

- `RepoScanPlugin`: Scan repository modules and build change-tracker state.
- `AstExtractPlugin`: Parse Python AST and persist rows + metrics.
- `CstExtractPlugin`: Parse CST via LibCST and write rows.
- `ScipIngestPlugin`: Run scip-python and persist symbols.
- `TypingIngestPlugin`: Populate typedness and static diagnostics.
- `CoverageIngestPlugin`: Load coverage.py data.
- `TestsIngestPlugin`: Ingest pytest JSON report.
- `DocstringsIngestPlugin`: Extract docstrings and persist structured rows.
- `ConfigIngestPlugin`: Flatten config files into config_values.

These class-based plugins inherit from the base classes in `codeintel.ingestion.core`
and provide a composable architecture with traits, middleware, and resource providers.

NOTE: Class-based plugin imports are lazy to avoid circular dependencies with core.base.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

# Import protocol types first (these have no circular dependency issues)
from codeintel.ingestion.plugins.protocol import (
    DEFAULT_INGEST_PLUGINS,
    IngestIsolationKind,
    IngestPluginMetadata,
    IngestPluginPlan,
    IngestPluginProtocol,
    IngestPluginResult,
    IngestPluginSkip,
    IngestSeverity,
    IngestStage,
)

# Import contracts from validation/ (canonical location)
from codeintel.ingestion.validation import (
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

if TYPE_CHECKING:
    from codeintel.ingestion.plugins.ast_extract import (
        AstExtractPlugin as AstExtractPlugin,
    )
    from codeintel.ingestion.plugins.config_factory import (
        DEFAULT_CONTEXT_MAPPINGS as DEFAULT_CONTEXT_MAPPINGS,
    )
    from codeintel.ingestion.plugins.config_factory import (
        ConfigFactory as ConfigFactory,
    )
    from codeintel.ingestion.plugins.config_factory import (
        ConfigMapping as ConfigMapping,
    )
    from codeintel.ingestion.plugins.config_factory import (
        get_config_fields as get_config_fields,
    )
    from codeintel.ingestion.plugins.config_factory import (
        infer_config_mapping as infer_config_mapping,
    )
    from codeintel.ingestion.plugins.config_plugin import (
        ConfigIngestPlugin as ConfigIngestPlugin,
    )
    from codeintel.ingestion.plugins.coverage_plugin import (
        CoverageIngestPlugin as CoverageIngestPlugin,
    )
    from codeintel.ingestion.plugins.cst_extract import (
        CstExtractPlugin as CstExtractPlugin,
    )
    from codeintel.ingestion.plugins.docstrings_plugin import (
        DocstringsIngestPlugin as DocstringsIngestPlugin,
    )
    from codeintel.ingestion.plugins.registry import (
        IngestPluginRegistry as IngestPluginRegistry,
    )
    from codeintel.ingestion.plugins.registry import (
        PlanOptions as PlanOptions,
    )
    from codeintel.ingestion.plugins.registry import (
        get_ingest_registry as get_ingest_registry,
    )
    from codeintel.ingestion.plugins.registry import (
        list_ingest_plugins as list_ingest_plugins,
    )
    from codeintel.ingestion.plugins.registry import (
        plan_ingest_plugins as plan_ingest_plugins,
    )
    from codeintel.ingestion.plugins.registry import (
        register_class_based_plugins as register_class_based_plugins,
    )
    from codeintel.ingestion.plugins.registry import (
        register_ingest_plugin as register_ingest_plugin,
    )
    from codeintel.ingestion.plugins.repo_scan import (
        RepoScanPlugin as RepoScanPlugin,
    )
    from codeintel.ingestion.plugins.scip_plugin import (
        ScipIngestPlugin as ScipIngestPlugin,
    )
    from codeintel.ingestion.plugins.tests_plugin import (
        TestsIngestPlugin as TestsIngestPlugin,
    )
    from codeintel.ingestion.plugins.typing_plugin import (
        TypingIngestPlugin as TypingIngestPlugin,
    )


# Lazy imports to avoid circular dependencies with core.base
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "AstExtractPlugin": ("codeintel.ingestion.plugins.ast_extract", "AstExtractPlugin"),
    "ConfigFactory": ("codeintel.ingestion.plugins.config_factory", "ConfigFactory"),
    "ConfigMapping": ("codeintel.ingestion.plugins.config_factory", "ConfigMapping"),
    "DEFAULT_CONTEXT_MAPPINGS": (
        "codeintel.ingestion.plugins.config_factory",
        "DEFAULT_CONTEXT_MAPPINGS",
    ),
    "get_config_fields": ("codeintel.ingestion.plugins.config_factory", "get_config_fields"),
    "infer_config_mapping": ("codeintel.ingestion.plugins.config_factory", "infer_config_mapping"),
    "ConfigIngestPlugin": ("codeintel.ingestion.plugins.config_plugin", "ConfigIngestPlugin"),
    "CoverageIngestPlugin": ("codeintel.ingestion.plugins.coverage_plugin", "CoverageIngestPlugin"),
    "CstExtractPlugin": ("codeintel.ingestion.plugins.cst_extract", "CstExtractPlugin"),
    "DocstringsIngestPlugin": (
        "codeintel.ingestion.plugins.docstrings_plugin",
        "DocstringsIngestPlugin",
    ),
    "IngestPluginRegistry": ("codeintel.ingestion.plugins.registry", "IngestPluginRegistry"),
    "PlanOptions": ("codeintel.ingestion.plugins.registry", "PlanOptions"),
    "get_ingest_registry": ("codeintel.ingestion.plugins.registry", "get_ingest_registry"),
    "list_ingest_plugins": ("codeintel.ingestion.plugins.registry", "list_ingest_plugins"),
    "plan_ingest_plugins": ("codeintel.ingestion.plugins.registry", "plan_ingest_plugins"),
    "register_class_based_plugins": (
        "codeintel.ingestion.plugins.registry",
        "register_class_based_plugins",
    ),
    "register_ingest_plugin": ("codeintel.ingestion.plugins.registry", "register_ingest_plugin"),
    "RepoScanPlugin": ("codeintel.ingestion.plugins.repo_scan", "RepoScanPlugin"),
    "ScipIngestPlugin": ("codeintel.ingestion.plugins.scip_plugin", "ScipIngestPlugin"),
    "TestsIngestPlugin": ("codeintel.ingestion.plugins.tests_plugin", "TestsIngestPlugin"),
    "TypingIngestPlugin": ("codeintel.ingestion.plugins.typing_plugin", "TypingIngestPlugin"),
}

_LOADED: dict[str, object] = {}


def __getattr__(name: str) -> object:
    """Lazily import class-based plugins and registry functions.

    Parameters
    ----------
    name
        The name of the attribute to retrieve.

    Returns
    -------
    object
        The requested attribute.

    Raises
    ------
    AttributeError
        If the attribute does not exist.
    """
    if name in _LOADED:
        return _LOADED[name]
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        module = importlib.import_module(module_path)
        value = getattr(module, attr_name)
        _LOADED[name] = value
        return value
    message = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(message)


# Note: ruff's F822 doesn't understand __getattr__ for lazy loading.
# The lazy-loaded names below are all valid and provided by __getattr__.
__all__ = [
    "DEFAULT_CONTEXT_MAPPINGS",
    "DEFAULT_INGEST_PLUGINS",
    "AstExtractPlugin",
    "ColumnConstraint",
    "ConfigFactory",
    "ConfigIngestPlugin",
    "ConfigMapping",
    "ContractValidationResult",
    "ContractViolation",
    "CoverageIngestPlugin",
    "CstExtractPlugin",
    "DocstringsIngestPlugin",
    "ForeignKeyConstraint",
    "IngestContractSpec",
    "IngestContractValidator",
    "IngestIsolationKind",
    "IngestPluginMetadata",
    "IngestPluginPlan",
    "IngestPluginProtocol",
    "IngestPluginRegistry",
    "IngestPluginResult",
    "IngestPluginSkip",
    "IngestSeverity",
    "IngestStage",
    "PlanOptions",
    "RepoScanPlugin",
    "ScipIngestPlugin",
    "TestsIngestPlugin",
    "TypingIngestPlugin",
    "foreign_key_contract",
    "get_config_fields",
    "get_ingest_registry",
    "infer_config_mapping",
    "list_ingest_plugins",
    "not_null_contract",
    "plan_ingest_plugins",
    "register_class_based_plugins",
    "register_ingest_plugin",
    "row_count_contract",
]
