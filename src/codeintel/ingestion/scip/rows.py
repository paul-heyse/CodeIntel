"""Helpers for parsing SCIP JSON and building row payloads."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from codeintel.ingestion.ports.tools import ScipDocument, ScipOccurrence, ScipSymbol

if TYPE_CHECKING:
    from collections.abc import Sequence

    from codeintel.core.schemas.row_models import RowSerializer

log = logging.getLogger(__name__)

_SCIP_RANGE_END_CHAR_IDX = 3


def find_scip_json(index_json_path: Path | None, output_scip: Path) -> Path | None:
    """Find the SCIP JSON file from multiple candidate locations.

    Returns
    -------
    Path | None
        Path to existing JSON file or None if not found.
    """
    candidates: list[Path] = []
    if index_json_path is not None:
        candidates.append(index_json_path)
    candidates.extend(
        [
            output_scip.with_suffix(".scip.json"),
            output_scip.parent / "index.scip.json",
            output_scip.parent / "index.json",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def parse_scip_symbols(raw_symbols: list[Any]) -> list[ScipSymbol]:
    """Parse symbols from raw JSON data.

    Returns
    -------
    list[ScipSymbol]
        Parsed symbol objects.
    """
    symbols: list[ScipSymbol] = []
    for sym in raw_symbols:
        if not isinstance(sym, dict):
            continue
        symbol_str = sym.get("symbol", "")
        documentation = sym.get("documentation", [])
        doc_str = "\n".join(documentation) if isinstance(documentation, list) else ""
        symbols.append(ScipSymbol(symbol=symbol_str, documentation=doc_str))
    return symbols


def parse_scip_occurrences(raw_occurrences: list[Any]) -> list[ScipOccurrence]:
    """Parse occurrences from raw JSON data.

    Returns
    -------
    list[ScipOccurrence]
        Parsed occurrence objects.
    """
    occurrences: list[ScipOccurrence] = []
    for occ in raw_occurrences:
        if not isinstance(occ, dict):
            continue
        rng = occ.get("range", [])
        if not isinstance(rng, list) or not rng:
            continue

        roles = occ.get("symbolRoles") or occ.get("symbol_roles", 0)
        role_int = roles if isinstance(roles, int) else 0
        occurrences.append(
            ScipOccurrence(
                symbol=occ.get("symbol", ""),
                range_start_line=int(rng[0]),
                range_start_col=int(rng[1]) if len(rng) > 1 else 0,
                range_end_line=int(rng[0]),
                range_end_col=int(rng[_SCIP_RANGE_END_CHAR_IDX])
                if len(rng) > _SCIP_RANGE_END_CHAR_IDX
                else 0,
                symbol_roles=role_int,
            )
        )
    return occurrences


def parse_scip_document(doc: dict[str, Any]) -> ScipDocument | None:
    """Parse a single SCIP document from JSON.

    Returns
    -------
    ScipDocument | None
        Parsed document or None if invalid.
    """
    rel_path = doc.get("relativePath") or doc.get("relative_path", "")
    if not rel_path:
        return None
    raw_symbols = doc.get("symbols", [])
    symbols = parse_scip_symbols(raw_symbols) if isinstance(raw_symbols, list) else []
    raw_occurrences = doc.get("occurrences", [])
    occurrences = (
        parse_scip_occurrences(raw_occurrences) if isinstance(raw_occurrences, list) else []
    )
    return ScipDocument(
        relative_path=str(rel_path),
        symbols=tuple(symbols),
        occurrences=tuple(occurrences),
    )


def parse_scip_json_file(
    index_json_path: Path | None,
    output_scip: Path,
) -> list[ScipDocument]:
    """Parse SCIP documents from JSON file.

    Try multiple JSON file locations and parse documents with symbols/occurrences.

    Parameters
    ----------
    index_json_path
        Optional JSON file path to check first.
    output_scip
        SCIP binary file path (used to find alternate JSON locations).

    Returns
    -------
    list[ScipDocument]
        Parsed SCIP documents.
    """
    json_path = find_scip_json(index_json_path, output_scip)
    if json_path is None:
        log.debug("No SCIP JSON file found")
        return []

    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Failed to parse SCIP JSON file %s: %s", json_path, exc)
        return []

    docs: list[Any] = []
    if isinstance(payload, dict):
        docs_field = payload.get("documents", [])
        docs = docs_field if isinstance(docs_field, list) else []
    elif isinstance(payload, list):
        docs = payload

    documents: list[ScipDocument] = []
    for doc in docs:
        if isinstance(doc, dict):
            parsed = parse_scip_document(doc)
            if parsed is not None:
                documents.append(parsed)

    log.debug("Parsed %d SCIP documents from %s", len(documents), json_path)
    return documents


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
                    "range_start_line": occ.range_start_line,
                    "range_start_col": occ.range_start_col,
                    "range_end_line": occ.range_end_line,
                    "range_end_col": occ.range_end_col,
                    "symbol_roles": occ.symbol_roles,
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
                    payload["range_start_line"],
                    payload["range_start_col"],
                    payload["range_end_line"],
                    payload["range_end_col"],
                    payload["symbol_roles"],
                    payload["created_at"],
                )
            )
    return rows


__all__ = [
    "build_occurrence_rows",
    "build_symbol_rows",
    "find_scip_json",
    "parse_scip_document",
    "parse_scip_json_file",
    "parse_scip_occurrences",
    "parse_scip_symbols",
]
