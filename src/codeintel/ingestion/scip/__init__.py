"""SCIP parsing helpers for ingestion and build targets."""

from __future__ import annotations

from codeintel.ingestion.scip.protobuf_parser import ScipParsedIndex, parse_index
from codeintel.ingestion.scip.rows import (
    SCIP_DIAGNOSTICS_TABLE_KEY,
    SCIP_EXTERNAL_SYMBOLS_TABLE_KEY,
    SCIP_MODULE_STATE_TABLE_KEY,
    SCIP_OCCURRENCES_TABLE_KEY,
    SCIP_RELATIONSHIPS_TABLE_KEY,
    SCIP_SYMBOL_INFO_TABLE_KEY,
    SCIP_SYMBOLS_TABLE_KEY,
    ScipRowContext,
    build_diagnostic_rows,
    build_external_symbol_rows,
    build_module_state_rows,
    build_occurrence_rows,
    build_symbol_information_rows,
    build_symbol_relationship_rows,
    build_symbol_rows,
)

__all__ = [
    "SCIP_DIAGNOSTICS_TABLE_KEY",
    "SCIP_EXTERNAL_SYMBOLS_TABLE_KEY",
    "SCIP_MODULE_STATE_TABLE_KEY",
    "SCIP_OCCURRENCES_TABLE_KEY",
    "SCIP_RELATIONSHIPS_TABLE_KEY",
    "SCIP_SYMBOLS_TABLE_KEY",
    "SCIP_SYMBOL_INFO_TABLE_KEY",
    "ScipParsedIndex",
    "ScipRowContext",
    "build_diagnostic_rows",
    "build_external_symbol_rows",
    "build_module_state_rows",
    "build_occurrence_rows",
    "build_symbol_information_rows",
    "build_symbol_relationship_rows",
    "build_symbol_rows",
    "parse_index",
]
