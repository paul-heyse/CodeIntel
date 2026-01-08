"""Typed models for parsed SCIP protobuf data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ScipSymbolInfo:
    """Parsed symbol metadata."""

    symbol: str
    documentation: str | None
    kind: int | None
    display_name: str | None
    signature: str | None
    enclosing_symbol: str | None


@dataclass(frozen=True)
class ScipSymbolRelationship:
    """Relationship between symbols (implements, reference, type definition)."""

    symbol: str
    related_symbol: str
    relationship_kind: str


@dataclass(frozen=True)
class ScipDiagnostic:
    """Diagnostic emitted for a source range."""

    rel_path: str
    start_line: int
    start_col: int
    end_line: int
    end_col: int
    position_encoding: int | None
    text_document_encoding: str | None
    severity: str
    code: str | None
    message: str
    source: str | None


@dataclass(frozen=True)
class ScipExternalSymbol:
    """External symbol reference with package metadata."""

    symbol: str
    package_manager: str | None
    package_name: str | None
    package_version: str | None


@dataclass(frozen=True)
class ScipIndexMetadata:
    """Metadata captured from the SCIP index header."""

    project_root: Path | None
    text_document_encoding: str | None
    tool_name: str | None
    tool_version: str | None
    tool_arguments: tuple[str, ...] | None


__all__ = [
    "ScipDiagnostic",
    "ScipExternalSymbol",
    "ScipIndexMetadata",
    "ScipSymbolInfo",
    "ScipSymbolRelationship",
]
