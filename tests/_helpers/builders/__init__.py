"""Typed row dataclasses for seeded DuckDB test data.

This package provides Row dataclasses that implement the InsertableRow protocol.
Use the generic insert_rows() function from row_protocol.py to insert these rows.

The package is organized by database schema domain:
- core: Core entity tables (modules, goids, docstrings, etc.)
- graph: Graph structure tables (call graph, import graph, CFG/DFG)
- analytics: Analytics and metrics tables (function metrics, coverage, risk, etc.)
"""

from __future__ import annotations

from tests._helpers.builders.analytics import (
    ConfigValueRow,
    CoverageFunctionRow,
    CoverageLineRow,
    FunctionMetricsRow,
    FunctionTypesRow,
    FunctionValidationRow,
    GraphMetricsModulesExtRow,
    HotspotRow,
    RiskFactorRow,
    StaticDiagnosticsRow,
    SubsystemModuleRow,
    SubsystemRow,
    SymbolGraphMetricsModulesRow,
    TestCatalogRow,
    TestCoverageEdgeRow,
    TypednessRow,
)
from tests._helpers.builders.core import (
    AstMetricsRow,
    DocstringRow,
    GoidCrosswalkRow,
    GoidRow,
    ModuleRow,
    RepoMapRow,
)
from tests._helpers.builders.function_context import FunctionContextBuilder
from tests._helpers.builders.graph import (
    CallGraphEdgeRow,
    CallGraphNodeRow,
    CFGBlockRow,
    CFGEdgeRow,
    DFGEdgeRow,
    ImportGraphEdgeRow,
    SymbolEdgeOptions,
    SymbolUseEdgeInput,
    SymbolUseEdgeRow,
    insert_symbol_use_edges,
    make_symbol_use_edge_row,
)
from tests._helpers.builders.metadata import DatasetDataflowEdgeRow, DatasetDataflowNodeRow
from tests._helpers.builders.row_protocol import InsertableRow, insert_rows

__all__ = [
    "AstMetricsRow",
    "CFGBlockRow",
    "CFGEdgeRow",
    "CallGraphEdgeRow",
    "CallGraphNodeRow",
    "ConfigValueRow",
    "CoverageFunctionRow",
    "CoverageLineRow",
    "DFGEdgeRow",
    "DatasetDataflowEdgeRow",
    "DatasetDataflowNodeRow",
    "DocstringRow",
    "FunctionContextBuilder",
    "FunctionMetricsRow",
    "FunctionTypesRow",
    "FunctionValidationRow",
    "GoidCrosswalkRow",
    "GoidRow",
    "GraphMetricsModulesExtRow",
    "HotspotRow",
    "ImportGraphEdgeRow",
    "InsertableRow",
    "ModuleRow",
    "RepoMapRow",
    "RiskFactorRow",
    "StaticDiagnosticsRow",
    "SubsystemModuleRow",
    "SubsystemRow",
    "SymbolEdgeOptions",
    "SymbolGraphMetricsModulesRow",
    "SymbolUseEdgeInput",
    "SymbolUseEdgeRow",
    "TestCatalogRow",
    "TestCoverageEdgeRow",
    "TypednessRow",
    "insert_rows",
    "insert_symbol_use_edges",
    "make_symbol_use_edge_row",
]
