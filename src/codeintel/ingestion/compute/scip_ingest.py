"""SCIP indexing step with port injection.

This module provides a pure domain logic implementation for running
SCIP indexing and ingesting symbol data, using ports for all I/O operations.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from codeintel.ingestion.compute.base import StepResult
from codeintel.ingestion.ports.tools import ScipDocument, ScipOccurrence, ScipSymbol, ToolStatus

if TYPE_CHECKING:
    from codeintel.ingestion.ports.discovery import ModuleRecord
    from codeintel.ingestion.ports.storage import IngestStoragePort
    from codeintel.ingestion.ports.tools import IngestToolPort

log = logging.getLogger(__name__)

# Threshold for incremental vs full indexing
INCREMENTAL_INDEX_THRESHOLD = 100


@dataclass(frozen=True)
class ScipIngestResult:
    """Outcome of SCIP ingestion.

    Attributes
    ----------
    status
        Result status: "success", "unavailable", or "failed".
    index_scip
        Path to SCIP index file if created.
    index_json
        Path to JSON export if created.
    reason
        Reason for failure or unavailability.
    """

    status: Literal["success", "unavailable", "failed"]
    index_scip: Path | None
    index_json: Path | None
    reason: str | None = None


@dataclass
class ScipIngestConfig:
    """Configuration for SCIP indexing.

    Attributes
    ----------
    repo
        Repository identifier.
    commit
        Commit identifier.
    repo_root
        Repository root path.
    output_scip
        Path for SCIP index output.
    output_json
        Path for JSON export output.
    target_dir
        Optional target directory to index.
    """

    repo: str
    commit: str
    repo_root: Path
    output_scip: Path
    output_json: Path
    target_dir: Path | None = None


class ScipIngestStep:
    """SCIP indexing step with port injection.

    This step runs SCIP indexing and persists symbol data,
    using ports for all I/O operations.

    Parameters
    ----------
    storage
        Storage port for persisting data.
    tools
        Tool port for running SCIP.
    """

    def __init__(
        self,
        storage: IngestStoragePort,
        tools: IngestToolPort,
    ) -> None:
        """Initialize the step.

        Parameters
        ----------
        storage
            Storage port for persisting data.
        tools
            Tool port for running SCIP.
        """
        self._storage = storage
        self._tools = tools

    async def execute_async(
        self,
        modules: Sequence[ModuleRecord],
        config: ScipIngestConfig,
    ) -> StepResult:
        """Execute SCIP indexing.

        Parameters
        ----------
        modules
            Modules to index (can be subset for incremental).
        config
            SCIP indexing configuration.

        Returns
        -------
        StepResult
            Execution result with row counts.
        """
        created_at = datetime.now(UTC)

        # Determine if doing incremental (specific files) or full index
        rel_paths = None
        if len(modules) < INCREMENTAL_INDEX_THRESHOLD:
            rel_paths = [m.rel_path for m in modules if m.rel_path.endswith(".py")]

        # Run SCIP indexing
        result = await self._tools.run_scip(
            config.repo_root,
            output_scip=config.output_scip,
            output_json=config.output_json,
            target_dir=config.target_dir,
            rel_paths=rel_paths,
        )

        if result.status != ToolStatus.OK:
            log.warning("SCIP indexing failed: %s", result.error)
            return StepResult.fail(f"SCIP indexing failed: {result.error}")

        # Get documents from result or parse from JSON file
        documents = result.documents
        if not documents:
            # Try to parse documents from JSON file if it exists
            documents = _parse_scip_json_file(config.output_json, config.output_scip)
            if documents:
                log.info("SCIP documents loaded from JSON file: %d", len(documents))

        # Build rows
        symbol_rows = _build_symbol_rows(documents, config.repo, config.commit, created_at)
        occurrence_rows = _build_occurrence_rows(documents, config.repo, config.commit, created_at)

        # Delete existing data for this repo/commit before inserting
        self._storage.execute_query(
            "DELETE FROM core.scip_symbols WHERE repo = ? AND commit = ?",
            [config.repo, config.commit],
        )
        self._storage.execute_query(
            "DELETE FROM core.scip_occurrences WHERE repo = ? AND commit = ?",
            [config.repo, config.commit],
        )

        # Persist rows
        table_counts: dict[str, int] = {}
        total_rows = 0

        if symbol_rows:
            scope = f"{config.repo}@{config.commit}"
            write_result = self._storage.write_batch("core.scip_symbols", symbol_rows, scope=scope)
            table_counts["core.scip_symbols"] = write_result.rows_written
            total_rows += write_result.rows_written

        if occurrence_rows:
            write_result = self._storage.write_batch("core.scip_occurrences", occurrence_rows)
            table_counts["core.scip_occurrences"] = write_result.rows_written
            total_rows += write_result.rows_written

        log.info(
            "SCIP ingest: repo=%s commit=%s symbols=%d occurrences=%d",
            config.repo,
            config.commit,
            len(symbol_rows),
            len(occurrence_rows),
        )

        return StepResult(rows_written=total_rows, table_counts=table_counts)


# SCIP range array indices
_SCIP_RANGE_END_CHAR_IDX = 3


def _find_scip_json(output_json: Path, output_scip: Path) -> Path | None:
    """Find the SCIP JSON file from multiple candidate locations.

    Returns
    -------
    Path | None
        Path to existing JSON file or None if not found.
    """
    candidates = [
        output_json,
        output_scip.with_suffix(".scip.json"),
        output_scip.parent / "index.scip.json",
        output_scip.parent / "index.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _parse_scip_symbols(raw_symbols: list[Any]) -> list[ScipSymbol]:
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


def _parse_scip_occurrences(raw_occurrences: list[Any]) -> list[ScipOccurrence]:
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
        # Try both camelCase and snake_case field names
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


def _parse_scip_document(doc: Mapping[str, Any]) -> ScipDocument | None:
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
    symbols = _parse_scip_symbols(raw_symbols) if isinstance(raw_symbols, list) else []
    raw_occurrences = doc.get("occurrences", [])
    occurrences = (
        _parse_scip_occurrences(raw_occurrences) if isinstance(raw_occurrences, list) else []
    )
    return ScipDocument(
        relative_path=str(rel_path),
        symbols=tuple(symbols),
        occurrences=tuple(occurrences),
    )


def _parse_scip_json_file(output_json: Path, output_scip: Path) -> list[ScipDocument]:
    """Parse SCIP documents from JSON file.

    Try multiple JSON file locations and parse documents with symbols/occurrences.

    Parameters
    ----------
    output_json
        Primary JSON file path.
    output_scip
        SCIP binary file path (used to find alternate JSON locations).

    Returns
    -------
    list[ScipDocument]
        Parsed SCIP documents.
    """
    json_path = _find_scip_json(output_json, output_scip)
    if json_path is None:
        log.debug("No SCIP JSON file found")
        return []

    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Failed to parse SCIP JSON file %s: %s", json_path, exc)
        return []

    # Extract documents from payload
    docs: list[Any] = []
    if isinstance(payload, dict):
        docs_field = payload.get("documents", [])
        docs = docs_field if isinstance(docs_field, list) else []
    elif isinstance(payload, list):
        docs = payload

    # Convert to ScipDocument objects
    documents: list[ScipDocument] = []
    for doc in docs:
        if isinstance(doc, dict):
            parsed = _parse_scip_document(doc)
            if parsed is not None:
                documents.append(parsed)

    log.debug("Parsed %d SCIP documents from %s", len(documents), json_path)
    return documents


def _build_symbol_rows(
    documents: list[ScipDocument],
    repo: str,
    commit: str,
    created_at: datetime,
) -> list[list[object]]:
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

    Returns
    -------
    list[list[object]]
        Symbol rows for storage (deduplicated).
    """
    # Use dict to deduplicate by (rel_path, symbol) - keep first occurrence
    seen: dict[tuple[str, str], list[object]] = {}
    for doc in documents:
        for sym in doc.symbols:
            key = (doc.relative_path, sym.symbol)
            if key not in seen:
                seen[key] = [
                    repo,
                    commit,
                    doc.relative_path,
                    sym.symbol,
                    sym.documentation,
                    created_at,
                ]
    return list(seen.values())


def _build_occurrence_rows(
    documents: list[ScipDocument],
    repo: str,
    commit: str,
    created_at: datetime,
) -> list[list[object]]:
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

    Returns
    -------
    list[list[object]]
        Occurrence rows for storage (deduplicated).
    """
    # Use dict to deduplicate by primary key fields - keep first occurrence
    seen: dict[tuple[str, str, int, int], list[object]] = {}
    for doc in documents:
        for occ in doc.occurrences:
            key = (doc.relative_path, occ.symbol, occ.range_start_line, occ.range_start_col)
            if key not in seen:
                seen[key] = [
                    repo,
                    commit,
                    doc.relative_path,
                    occ.symbol,
                    occ.range_start_line,
                    occ.range_start_col,
                    occ.range_end_line,
                    occ.range_end_col,
                    occ.symbol_roles,
                    created_at,
                ]
    return list(seen.values())


__all__ = ["ScipIngestConfig", "ScipIngestResult", "ScipIngestStep"]
