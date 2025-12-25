"""Helpers for building SCIP row payloads."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from codeintel.ingestion.ports.tools import ScipDocument
from codeintel.ingestion.scip.manifest import ScipShardManifest
from codeintel.ingestion.scip.models import (
    ScipDiagnostic,
    ScipExternalSymbol,
    ScipSymbolInfo,
    ScipSymbolRelationship,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from codeintel.core.schemas.row_models import RowSerializer

_ROLE_DEFINITION = 1
_ROLE_REFERENCE = 2


@dataclass(frozen=True)
class ScipRowContext:
    """Shared context for SCIP row building."""

    repo: str
    commit: str
    created_at: datetime
    include_references: bool = True
    include_implementations: bool = True


def build_symbol_rows(
    documents: Sequence[ScipDocument],
    context: ScipRowContext,
    *,
    serializer: RowSerializer | None = None,
) -> list[tuple[object, ...]]:
    """Build symbol rows from SCIP documents.

    Deduplicates by (rel_path, symbol) to avoid primary key violations.

    Parameters
    ----------
    documents
        SCIP documents containing symbols.
    context
        Shared row context (repo, commit, created_at).
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
                    "repo": context.repo,
                    "commit": context.commit,
                    "rel_path": doc.relative_path,
                    "symbol": sym.symbol,
                    "documentation": sym.documentation,
                    "created_at": context.created_at,
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
    context: ScipRowContext,
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
    context
        Shared row context (repo, commit, created_at, include_references).
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
            if not context.include_references and _is_reference_only(occ.symbol_roles):
                continue
            key = (doc.relative_path, occ.symbol, occ.range_start_line, occ.range_start_col)
            if key not in seen:
                seen[key] = {
                    "repo": context.repo,
                    "commit": context.commit,
                    "rel_path": doc.relative_path,
                    "symbol": occ.symbol,
                    "start_line": occ.range_start_line,
                    "start_col": occ.range_start_col,
                    "end_line": occ.range_end_line,
                    "end_col": occ.range_end_col,
                    "roles": occ.symbol_roles,
                    "created_at": context.created_at,
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
    context: ScipRowContext,
    *,
    serializer: RowSerializer | None = None,
) -> list[tuple[object, ...]]:
    """Build rows for core.scip_symbol_information.

    Parameters
    ----------
    symbol_infos
        Parsed symbol info records.
    context
        Shared row context (repo, commit, created_at).
    serializer
        Optional row serializer for deterministic column ordering.

    Returns
    -------
    list[tuple[object, ...]]
        Serialized row tuples for symbol information.
    """
    rows: list[tuple[object, ...]] = []
    seen: dict[str, ScipSymbolInfo] = {}
    for info in symbol_infos:
        existing = seen.get(info.symbol)
        if existing is None:
            seen[info.symbol] = info
            continue
        seen[info.symbol] = _prefer_symbol_info(existing, info)
    for info in seen.values():
        payload = {
            "repo": context.repo,
            "commit": context.commit,
            "symbol": info.symbol,
            "documentation": info.documentation,
            "kind": info.kind,
            "display_name": info.display_name,
            "signature": info.signature,
            "enclosing_symbol": info.enclosing_symbol,
            "created_at": context.created_at,
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


def _symbol_info_score(info: ScipSymbolInfo) -> int:
    score = 0
    if info.documentation:
        score += 1
    if info.display_name:
        score += 1
    if info.signature:
        score += 1
    if info.enclosing_symbol:
        score += 1
    if info.kind is not None:
        score += 1
    return score


def _prefer_symbol_info(current: ScipSymbolInfo, candidate: ScipSymbolInfo) -> ScipSymbolInfo:
    if _symbol_info_score(candidate) > _symbol_info_score(current):
        return candidate
    return current


def build_symbol_relationship_rows(
    relationships: Sequence[ScipSymbolRelationship],
    context: ScipRowContext,
    *,
    serializer: RowSerializer | None = None,
) -> list[tuple[object, ...]]:
    """Build rows for core.scip_symbol_relationships.

    Parameters
    ----------
    relationships
        Parsed symbol relationships.
    context
        Shared row context (repo, commit, created_at, include_* flags).
    serializer
        Optional row serializer for deterministic column ordering.

    Returns
    -------
    list[tuple[object, ...]]
        Serialized row tuples for symbol relationships.
    """
    rows: list[tuple[object, ...]] = []
    for rel in relationships:
        if not context.include_references and rel.relationship_kind == "reference":
            continue
        if not context.include_implementations and rel.relationship_kind == "implementation":
            continue
        payload = {
            "repo": context.repo,
            "commit": context.commit,
            "symbol": rel.symbol,
            "related_symbol": rel.related_symbol,
            "relationship_kind": rel.relationship_kind,
            "created_at": context.created_at,
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


def _is_reference_only(symbol_roles: int) -> bool:
    if (symbol_roles & _ROLE_REFERENCE) == 0:
        return False
    return (symbol_roles & _ROLE_DEFINITION) == 0


def build_diagnostic_rows(
    diagnostics: Sequence[ScipDiagnostic],
    context: ScipRowContext,
    *,
    serializer: RowSerializer | None = None,
) -> list[tuple[object, ...]]:
    """Build rows for core.scip_diagnostics.

    Parameters
    ----------
    diagnostics
        Parsed diagnostics entries.
    context
        Shared row context (repo, commit, created_at).
    serializer
        Optional row serializer for deterministic column ordering.

    Returns
    -------
    list[tuple[object, ...]]
        Serialized row tuples for diagnostics.
    """
    rows: list[tuple[object, ...]] = []
    for diag in diagnostics:
        payload = {
            "repo": context.repo,
            "commit": context.commit,
            "rel_path": diag.rel_path,
            "start_line": diag.start_line,
            "start_col": diag.start_col,
            "end_line": diag.end_line,
            "end_col": diag.end_col,
            "severity": diag.severity,
            "code": diag.code,
            "message": diag.message,
            "source": diag.source,
            "created_at": context.created_at,
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
    context: ScipRowContext,
    *,
    serializer: RowSerializer | None = None,
) -> list[tuple[object, ...]]:
    """Build rows for core.scip_external_symbols.

    Parameters
    ----------
    external_symbols
        Parsed external symbols.
    context
        Shared row context (repo, commit, created_at).
    serializer
        Optional row serializer for deterministic column ordering.

    Returns
    -------
    list[tuple[object, ...]]
        Serialized row tuples for external symbols.
    """
    rows: list[tuple[object, ...]] = []
    for sym in external_symbols:
        payload = {
            "repo": context.repo,
            "commit": context.commit,
            "symbol": sym.symbol,
            "package_manager": sym.package_manager,
            "package_name": sym.package_name,
            "package_version": sym.package_version,
            "created_at": context.created_at,
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


def build_module_state_rows(
    manifest: ScipShardManifest,
    repo: str,
    commit: str,
    *,
    serializer: RowSerializer | None = None,
) -> list[tuple[object, ...]]:
    """Build rows for core.scip_module_state.

    Returns
    -------
    list[tuple[object, ...]]
        Serialized row tuples for module state records.
    """
    rows: list[tuple[object, ...]] = []
    for rel_path, record in sorted(manifest.records.items()):
        payload = {
            "repo": repo,
            "commit": commit,
            "rel_path": rel_path,
            "content_hash": record.content_hash,
            "options_hash": record.options_hash,
            "tool_version": record.tool_version,
            "shard_path": record.shard_path,
            "updated_at": record.updated_at,
        }
        if serializer is not None:
            rows.append(serializer(payload))
        else:
            rows.append(
                (
                    payload["repo"],
                    payload["commit"],
                    payload["rel_path"],
                    payload["content_hash"],
                    payload["options_hash"],
                    payload["tool_version"],
                    payload["shard_path"],
                    payload["updated_at"],
                )
            )
    return rows


__all__ = [
    "ScipRowContext",
    "build_diagnostic_rows",
    "build_external_symbol_rows",
    "build_module_state_rows",
    "build_occurrence_rows",
    "build_symbol_information_rows",
    "build_symbol_relationship_rows",
    "build_symbol_rows",
]
