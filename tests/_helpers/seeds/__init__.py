"""Composable seed packs for hexagonal test architecture."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Final

_MODULE_EXPORTS: Final[dict[str, tuple[str, ...]]] = {
    "tests._helpers.fixtures.coverage": ("COVERAGE_PACK", "CoveragePack"),
    "tests._helpers.seeds.ast_metrics": ("AST_METRICS_PACK", "AstMetricsPack"),
    "tests._helpers.seeds.cli": (
        "CLI_CORE_PACK",
        "GRAPH_HANDLER_PACK",
        "OPERATION_REGISTRY_PACK",
        "STORAGE_PROFILE_PACK",
        "SUBSYSTEM_HANDLER_PACK",
        "CliCorePack",
        "GraphHandlerPack",
        "OperationRegistryPack",
        "StorageProfilePack",
        "SubsystemHandlerPack",
    ),
    "tests._helpers.seeds.config": ("CONFIG_PACK", "ConfigPack"),
    "tests._helpers.seeds.core": ("CORE_PACK", "CorePack"),
    "tests._helpers.seeds.coverage_lines": ("COVERAGE_LINES_PACK", "CoverageLinesPack"),
    "tests._helpers.seeds.data_models": ("DATA_MODELS_PACK", "DataModelsPack"),
    "tests._helpers.seeds.docs": (
        "DOCS_EXPORT_PACK",
        "MCP_BACKEND_PACK",
        "PROFILE_DATA_PACK",
        "DocsExportPack",
        "McpBackendPack",
        "ProfileDataPack",
    ),
    "tests._helpers.seeds.docstrings": ("DOCSTRING_PACK", "DocstringPack"),
    "tests._helpers.seeds.entrypoints": ("ENTRYPOINTS_PACK", "EntrypointsPack"),
    "tests._helpers.seeds.function_types": ("FUNCTION_TYPES_PACK", "FunctionTypesPack"),
    "tests._helpers.seeds.graph": ("GRAPH_PACK", "GraphPack"),
    "tests._helpers.seeds.metrics": ("METRICS_PACK", "MetricsPack"),
    "tests._helpers.seeds.subsystems": ("SUBSYSTEM_PACK", "SubsystemPack"),
    "tests._helpers.seeds.subsystems_analytics": (
        "SUBSYSTEM_ANALYTICS_PACK",
        "SubsystemAnalyticsPack",
    ),
    "tests._helpers.seeds.symbols": ("SYMBOL_PACK", "SymbolPack"),
}

_EXPORT_TO_MODULE: Final[dict[str, str]] = {
    name: module for module, names in _MODULE_EXPORTS.items() for name in names
}

__all__ = [
    "AST_METRICS_PACK",
    "CLI_CORE_PACK",
    "CONFIG_PACK",
    "CORE_PACK",
    "COVERAGE_LINES_PACK",
    "COVERAGE_PACK",
    "DATA_MODELS_PACK",
    "DOCSTRING_PACK",
    "DOCS_EXPORT_PACK",
    "ENTRYPOINTS_PACK",
    "FUNCTION_TYPES_PACK",
    "GRAPH_HANDLER_PACK",
    "GRAPH_PACK",
    "MCP_BACKEND_PACK",
    "METRICS_PACK",
    "OPERATION_REGISTRY_PACK",
    "PROFILE_DATA_PACK",
    "STORAGE_PROFILE_PACK",
    "SUBSYSTEM_ANALYTICS_PACK",
    "SUBSYSTEM_HANDLER_PACK",
    "SUBSYSTEM_PACK",
    "SYMBOL_PACK",
    "AstMetricsPack",
    "CliCorePack",
    "ConfigPack",
    "CorePack",
    "CoverageLinesPack",
    "CoveragePack",
    "DataModelsPack",
    "DocsExportPack",
    "DocstringPack",
    "EntrypointsPack",
    "FunctionTypesPack",
    "GraphHandlerPack",
    "GraphPack",
    "McpBackendPack",
    "MetricsPack",
    "OperationRegistryPack",
    "ProfileDataPack",
    "StorageProfilePack",
    "SubsystemAnalyticsPack",
    "SubsystemHandlerPack",
    "SubsystemPack",
    "SymbolPack",
]

if TYPE_CHECKING:
    from tests._helpers.fixtures.coverage import COVERAGE_PACK, CoveragePack
    from tests._helpers.seeds.ast_metrics import AST_METRICS_PACK, AstMetricsPack
    from tests._helpers.seeds.cli import (
        CLI_CORE_PACK,
        GRAPH_HANDLER_PACK,
        OPERATION_REGISTRY_PACK,
        STORAGE_PROFILE_PACK,
        SUBSYSTEM_HANDLER_PACK,
        CliCorePack,
        GraphHandlerPack,
        OperationRegistryPack,
        StorageProfilePack,
        SubsystemHandlerPack,
    )
    from tests._helpers.seeds.config import CONFIG_PACK, ConfigPack
    from tests._helpers.seeds.core import CORE_PACK, CorePack
    from tests._helpers.seeds.coverage_lines import COVERAGE_LINES_PACK, CoverageLinesPack
    from tests._helpers.seeds.data_models import DATA_MODELS_PACK, DataModelsPack
    from tests._helpers.seeds.docs import (
        DOCS_EXPORT_PACK,
        MCP_BACKEND_PACK,
        PROFILE_DATA_PACK,
        DocsExportPack,
        McpBackendPack,
        ProfileDataPack,
    )
    from tests._helpers.seeds.docstrings import DOCSTRING_PACK, DocstringPack
    from tests._helpers.seeds.entrypoints import ENTRYPOINTS_PACK, EntrypointsPack
    from tests._helpers.seeds.function_types import FUNCTION_TYPES_PACK, FunctionTypesPack
    from tests._helpers.seeds.graph import GRAPH_PACK, GraphPack
    from tests._helpers.seeds.metrics import METRICS_PACK, MetricsPack
    from tests._helpers.seeds.subsystems import SUBSYSTEM_PACK, SubsystemPack
    from tests._helpers.seeds.subsystems_analytics import (
        SUBSYSTEM_ANALYTICS_PACK,
        SubsystemAnalyticsPack,
    )
    from tests._helpers.seeds.symbols import SYMBOL_PACK, SymbolPack

    _TYPE_CHECKING_EXPORTS = (
        AST_METRICS_PACK,
        CLI_CORE_PACK,
        CONFIG_PACK,
        CORE_PACK,
        COVERAGE_LINES_PACK,
        COVERAGE_PACK,
        DATA_MODELS_PACK,
        DOCSTRING_PACK,
        DOCS_EXPORT_PACK,
        ENTRYPOINTS_PACK,
        FUNCTION_TYPES_PACK,
        GRAPH_HANDLER_PACK,
        GRAPH_PACK,
        MCP_BACKEND_PACK,
        METRICS_PACK,
        OPERATION_REGISTRY_PACK,
        PROFILE_DATA_PACK,
        STORAGE_PROFILE_PACK,
        SUBSYSTEM_ANALYTICS_PACK,
        SUBSYSTEM_HANDLER_PACK,
        SUBSYSTEM_PACK,
        SYMBOL_PACK,
        AstMetricsPack,
        CliCorePack,
        ConfigPack,
        CorePack,
        CoverageLinesPack,
        CoveragePack,
        DataModelsPack,
        DocsExportPack,
        DocstringPack,
        EntrypointsPack,
        FunctionTypesPack,
        GraphHandlerPack,
        GraphPack,
        McpBackendPack,
        MetricsPack,
        OperationRegistryPack,
        ProfileDataPack,
        StorageProfilePack,
        SubsystemAnalyticsPack,
        SubsystemHandlerPack,
        SubsystemPack,
        SymbolPack,
    )


def __getattr__(name: str) -> object:
    """Lazily resolve seed pack exports.

    Parameters
    ----------
    name
        Attribute name requested from this package.

    Returns
    -------
    object
        Resolved attribute value.

    Raises
    ------
    AttributeError
        If ``name`` is not a known export.
    """
    module_name = _EXPORT_TO_MODULE.get(name)
    if module_name is None:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)

    module = importlib.import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
