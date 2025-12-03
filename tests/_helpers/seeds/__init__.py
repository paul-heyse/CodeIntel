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
SYMBOL_PACK
    Symbol data: symbol use edges.
CONFIG_PACK
    Configuration data: config file references.
"""

from __future__ import annotations

from tests._helpers.seeds.config import CONFIG_PACK, ConfigPack
from tests._helpers.seeds.core import CORE_PACK, CorePack
from tests._helpers.seeds.coverage import COVERAGE_PACK, CoveragePack
from tests._helpers.seeds.docstrings import DOCSTRING_PACK, DocstringPack
from tests._helpers.seeds.graph import GRAPH_PACK, GraphPack
from tests._helpers.seeds.metrics import METRICS_PACK, MetricsPack
from tests._helpers.seeds.subsystems import SUBSYSTEM_PACK, SubsystemPack
from tests._helpers.seeds.symbols import SYMBOL_PACK, SymbolPack

__all__ = [
    "CONFIG_PACK",
    "CORE_PACK",
    "COVERAGE_PACK",
    "DOCSTRING_PACK",
    "GRAPH_PACK",
    "METRICS_PACK",
    "SUBSYSTEM_PACK",
    "SYMBOL_PACK",
    "ConfigPack",
    "CorePack",
    "CoveragePack",
    "DocstringPack",
    "GraphPack",
    "MetricsPack",
    "SubsystemPack",
    "SymbolPack",
]
