"""Composable seed packs for hexagonal test architecture.

This package provides reusable seed packs that populate test data in a
composable way. Each pack seeds specific tables and can depend on other packs.

Seed packs follow the Testing Charter principles:
- Use real DuckDB tables (same technology as production)
- Data is realistic in structure and content
- Packs can be composed for complex test scenarios

Available Packs
---------------
CORE_PACK
    Minimal core data: repo_map, modules, goids.
GRAPH_PACK
    Graph data: call graph nodes/edges, import graph, cfg/dfg.
COVERAGE_PACK
    Coverage data: test catalog, coverage edges, coverage functions.
METRICS_PACK
    Metrics data: function metrics, risk factors, graph metrics.
DOCSTRING_PACK
    Documentation data: parsed docstrings.
SUBSYSTEM_PACK
    Architecture data: subsystems and subsystem_modules.
SUBSYSTEM_ANALYTICS_PACK
    Extended subsystem data with risk factors for analytics tests.
SYMBOL_PACK
    Symbol data: symbol use edges.
CONFIG_PACK
    Configuration data: config file references.
DATA_MODELS_PACK
    Data models and config data flow for model heuristics tests.
FUNCTION_TYPES_PACK
    Function type annotations for typing analytics tests.
"""

from __future__ import annotations

from tests._helpers.seeds.architecture import (
    open_seeded_architecture_gateway,
    seed_architecture,
)
from tests._helpers.seeds.ast_metrics import AST_METRICS_PACK, AstMetricsPack
from tests._helpers.seeds.config import CONFIG_PACK, ConfigPack
from tests._helpers.seeds.core import CORE_PACK, CorePack
from tests._helpers.seeds.coverage import COVERAGE_PACK, CoveragePack
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
from tests._helpers.seeds.function_types import FUNCTION_TYPES_PACK, FunctionTypesPack
from tests._helpers.seeds.graph import GRAPH_PACK, GraphPack
from tests._helpers.seeds.metrics import METRICS_PACK, MetricsPack
from tests._helpers.seeds.entrypoints import ENTRYPOINTS_PACK, EntrypointsPack
from tests._helpers.seeds.subsystems import SUBSYSTEM_PACK, SubsystemPack
from tests._helpers.seeds.subsystems_analytics import (
    SUBSYSTEM_ANALYTICS_PACK,
    SubsystemAnalyticsPack,
)
from tests._helpers.seeds.symbols import SYMBOL_PACK, SymbolPack

__all__ = [
    "AST_METRICS_PACK",
    "CONFIG_PACK",
    "CORE_PACK",
    "COVERAGE_PACK",
    "COVERAGE_LINES_PACK",
    "DATA_MODELS_PACK",
    "DOCSTRING_PACK",
    "DOCS_EXPORT_PACK",
    "FUNCTION_TYPES_PACK",
    "GRAPH_PACK",
    "MCP_BACKEND_PACK",
    "METRICS_PACK",
    "ENTRYPOINTS_PACK",
    "PROFILE_DATA_PACK",
    "SUBSYSTEM_ANALYTICS_PACK",
    "SUBSYSTEM_PACK",
    "SYMBOL_PACK",
    "ConfigPack",
    "CorePack",
    "CoveragePack",
    "AstMetricsPack",
    "CoverageLinesPack",
    "DataModelsPack",
    "DocsExportPack",
    "DocstringPack",
    "FunctionTypesPack",
    "GraphPack",
    "EntrypointsPack",
    "McpBackendPack",
    "MetricsPack",
    "ProfileDataPack",
    "SubsystemAnalyticsPack",
    "SubsystemPack",
    "SymbolPack",
    "open_seeded_architecture_gateway",
    "seed_architecture",
]
