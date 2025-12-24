"""SCIP parsing helpers for ingestion and build targets."""

from __future__ import annotations

from codeintel.ingestion.scip.protobuf_parser import ScipParsedIndex, parse_index
from codeintel.ingestion.scip.rows import (
    build_diagnostic_rows,
    build_external_symbol_rows,
    build_occurrence_rows,
    build_symbol_information_rows,
    build_symbol_relationship_rows,
    build_symbol_rows,
)

__all__ = [
    "ScipParsedIndex",
    "build_diagnostic_rows",
    "build_external_symbol_rows",
    "build_occurrence_rows",
    "build_symbol_information_rows",
    "build_symbol_relationship_rows",
    "build_symbol_rows",
    "parse_index",
]
