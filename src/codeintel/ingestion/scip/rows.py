"""Helpers for building SCIP row payloads."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from codeintel.ingestion.ports.tools import ScipDocument
from codeintel.ingestion.scip.models import (
    ScipDiagnostic,
    ScipExternalSymbol,
    ScipSymbolInfo,
    ScipSymbolRelationship,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from codeintel.core.schemas.row_models import RowSerializer


def build_symbol_rows(
    documents: Sequence[ScipDocument],
    repo: str,
    commit: str,
    created_at: datetime,
    *,
    serializer: RowSerializer | None = None,
) -> list[tuple[object, ...]]:
    """Build symbol rows from SCIP documents.

    Deduplicates by (rel_path, symbol) to avoid primary key violations.

    Parameters
    ----------
    documents
        SCIP documents containing symbols.
    repo
        Repository identifier.
    commit
        Commit identifier.
    created_at
        Row creation timestamp.
    serializer
        Optional row serializer for deterministic column ordering.

    Returns
    -------
    list[tuple[object, ...]]
        Symbol rows for storage (deduplicated).
    """
    seen: dict[tuple[str, str], dict[str, object]] = {}
    for doc in documents:
        for sym in doc.symbols:
            key = (doc.relative_path, sym.symbol)
            if key not in seen:
                seen[key] = {
                    "repo": repo,
                    "commit": commit,
                    "rel_path": doc.relative_path,
                    "symbol": sym.symbol,
                    "documentation": sym.documentation,
                    "created_at": created_at,
                }

    rows: list[tuple[object, ...]] = []
    for payload in seen.values():
        if serializer is not None:
            rows.append(serializer(payload))
        else:
            rows.append(
                (
                    payload["repo"],
                    payload["commit"],
                    payload["rel_path"],
                    payload["symbol"],
                    payload["documentation"],
                    payload["created_at"],
                )
            )
    return rows


def build_occurrence_rows(
    documents: Sequence[ScipDocument],
    repo: str,
    commit: str,
    created_at: datetime,
    *,
    serializer: RowSerializer | None = None,
) -> list[tuple[object, ...]]:
    """Build occurrence rows from SCIP documents.

    Deduplicates by (rel_path, symbol, start_line, start_col) to avoid
    primary key violations.

    Parameters
    ----------
    documents
        SCIP documents containing occurrences.
    repo
        Repository identifier.
    commit
        Commit identifier.
    created_at
        Row creation timestamp.
    serializer
        Optional row serializer for deterministic column ordering.

    Returns
    -------
    list[tuple[object, ...]]
        Occurrence rows for storage (deduplicated).
    """
    seen: dict[tuple[str, str, int, int], dict[str, object]] = {}
    for doc in documents:
        for occ in doc.occurrences:
            key = (doc.relative_path, occ.symbol, occ.range_start_line, occ.range_start_col)
            if key not in seen:
                seen[key] = {
                    "repo": repo,
                    "commit": commit,
                    "rel_path": doc.relative_path,
                    "symbol": occ.symbol,
                    "start_line": occ.range_start_line,
                    "start_col": occ.range_start_col,
                    "end_line": occ.range_end_line,
                    "end_col": occ.range_end_col,
                    "roles": occ.symbol_roles,
                    "created_at": created_at,
                }

    rows: list[tuple[object, ...]] = []
    for payload in seen.values():
        if serializer is not None:
            rows.append(serializer(payload))
        else:
            rows.append(
                (
                    payload["repo"],
                    payload["commit"],
                    payload["rel_path"],
                    payload["symbol"],
                    payload["start_line"],
                    payload["start_col"],
                    payload["end_line"],
                    payload["end_col"],
                    payload["roles"],
                    payload["created_at"],
                )
            )
    return rows


def build_symbol_information_rows(
    symbol_infos: Sequence[ScipSymbolInfo],
    repo: str,
    commit: str,
    created_at: datetime,
    *,
    serializer: RowSerializer | None = None,
) -> list[tuple[object, ...]]:
    """Build rows for core.scip_symbol_information.

    Returns
    -------
    list[tuple[object, ...]]
        Serialized row tuples for symbol information.
    """
    rows: list[tuple[object, ...]] = []
    for info in symbol_infos:
        payload = {
            "repo": repo,
            "commit": commit,
            "symbol": info.symbol,
            "documentation": info.documentation,
            "kind": info.kind,
            "display_name": info.display_name,
            "signature": info.signature,
            "enclosing_symbol": info.enclosing_symbol,
            "created_at": created_at,
        }
        if serializer is not None:
            rows.append(serializer(payload))
        else:
            rows.append(
                (
                    payload["repo"],
                    payload["commit"],
                    payload["symbol"],
                    payload["documentation"],
                    payload["kind"],
                    payload["display_name"],
                    payload["signature"],
                    payload["enclosing_symbol"],
                    payload["created_at"],
                )
            )
    return rows


def build_symbol_relationship_rows(
    relationships: Sequence[ScipSymbolRelationship],
    repo: str,
    commit: str,
    created_at: datetime,
    *,
    serializer: RowSerializer | None = None,
) -> list[tuple[object, ...]]:
    """Build rows for core.scip_symbol_relationships.

    Returns
    -------
    list[tuple[object, ...]]
        Serialized row tuples for symbol relationships.
    """
    rows: list[tuple[object, ...]] = []
    for rel in relationships:
        payload = {
            "repo": repo,
            "commit": commit,
            "symbol": rel.symbol,
            "related_symbol": rel.related_symbol,
            "relationship_kind": rel.relationship_kind,
            "created_at": created_at,
        }
        if serializer is not None:
            rows.append(serializer(payload))
        else:
            rows.append(
                (
                    payload["repo"],
                    payload["commit"],
                    payload["symbol"],
                    payload["related_symbol"],
                    payload["relationship_kind"],
                    payload["created_at"],
                )
            )
    return rows


def build_diagnostic_rows(
    diagnostics: Sequence[ScipDiagnostic],
    repo: str,
    commit: str,
    created_at: datetime,
    *,
    serializer: RowSerializer | None = None,
) -> list[tuple[object, ...]]:
    """Build rows for core.scip_diagnostics.

    Returns
    -------
    list[tuple[object, ...]]
        Serialized row tuples for diagnostics.
    """
    rows: list[tuple[object, ...]] = []
    for diag in diagnostics:
        payload = {
            "repo": repo,
            "commit": commit,
            "rel_path": diag.rel_path,
            "start_line": diag.start_line,
            "start_col": diag.start_col,
            "end_line": diag.end_line,
            "end_col": diag.end_col,
            "severity": diag.severity,
            "code": diag.code,
            "message": diag.message,
            "source": diag.source,
            "created_at": created_at,
        }
        if serializer is not None:
            rows.append(serializer(payload))
        else:
            rows.append(
                (
                    payload["repo"],
                    payload["commit"],
                    payload["rel_path"],
                    payload["start_line"],
                    payload["start_col"],
                    payload["end_line"],
                    payload["end_col"],
                    payload["severity"],
                    payload["code"],
                    payload["message"],
                    payload["source"],
                    payload["created_at"],
                )
            )
    return rows


def build_external_symbol_rows(
    external_symbols: Sequence[ScipExternalSymbol],
    repo: str,
    commit: str,
    created_at: datetime,
    *,
    serializer: RowSerializer | None = None,
) -> list[tuple[object, ...]]:
    """Build rows for core.scip_external_symbols.

    Returns
    -------
    list[tuple[object, ...]]
        Serialized row tuples for external symbols.
    """
    rows: list[tuple[object, ...]] = []
    for sym in external_symbols:
        payload = {
            "repo": repo,
            "commit": commit,
            "symbol": sym.symbol,
            "package_manager": sym.package_manager,
            "package_name": sym.package_name,
            "package_version": sym.package_version,
            "created_at": created_at,
        }
        if serializer is not None:
            rows.append(serializer(payload))
        else:
            rows.append(
                (
                    payload["repo"],
                    payload["commit"],
                    payload["symbol"],
                    payload["package_manager"],
                    payload["package_name"],
                    payload["package_version"],
                    payload["created_at"],
                )
            )
    return rows


__all__ = [
    "build_diagnostic_rows",
    "build_external_symbol_rows",
    "build_occurrence_rows",
    "build_symbol_information_rows",
    "build_symbol_relationship_rows",
    "build_symbol_rows",
]
