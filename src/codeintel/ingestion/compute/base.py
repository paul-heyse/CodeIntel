"""Base types for ingestion compute layer.

This module defines common types used by all ingestion compute modules,
analogous to base types in graphs/compute/.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.build.hamilton.execution_result import ExecutionResult

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from codeintel.ingestion.ports.discovery import ModuleDiscoveryPort, ModuleRecord
    from codeintel.ingestion.ports.storage import IngestStoragePort
    from codeintel.ingestion.ports.tools import IngestToolPort


class BaseExtractStep:
    """Base class for module extraction steps with port injection.

    Provides shared initialization and helper methods for steps that:

    - Accept storage and discovery ports via constructor
    - Iterate over Python modules and read source
    - Write rows to tables with scope tracking

    Parameters
    ----------
    storage
        Storage port for persisting data.
    discovery
        Discovery port for reading module source.
    """

    _storage: IngestStoragePort
    _discovery: ModuleDiscoveryPort

    def __init__(
        self,
        storage: IngestStoragePort,
        discovery: ModuleDiscoveryPort,
    ) -> None:
        """Initialize the step with storage and discovery ports.

        Parameters
        ----------
        storage
            Storage port for persisting data.
        discovery
            Discovery port for reading module source.
        """
        self._storage = storage
        self._discovery = discovery

    def _iter_python_sources(
        self, modules: Sequence[ModuleRecord]
    ) -> Iterator[tuple[ModuleRecord, str]]:
        """Yield (module, source) pairs for Python files with readable source.

        Parameters
        ----------
        modules
            Sequence of module records to iterate.

        Yields
        ------
        tuple[ModuleRecord, str]
            Module record and its source code for each readable Python file.
        """
        for module in modules:
            if not module.rel_path.endswith(".py"):
                continue
            source = self._discovery.read_module_source(module)
            if source is not None:
                yield module, source

    def _write_and_count(
        self,
        table_key: str,
        rows: Sequence[Sequence[object]],
        *,
        repo: str,
        commit: str,
    ) -> dict[str, int]:
        """Write rows and return table counts dictionary.

        Parameters
        ----------
        table_key
            Target table identifier.
        rows
            Rows to write.
        repo
            Repository identifier.
        commit
            Commit identifier.

        Returns
        -------
        dict[str, int]
            Mapping of table key to rows written count.
        """
        if not rows:
            return {}
        scope = f"{repo}@{commit}"
        result = self._storage.write_batch(table_key, list(rows), scope=scope)
        return {table_key: result.rows_affected}

    def _finalize_result(
        self,
        table_rows: Mapping[str, Sequence[Sequence[object]]],
        *,
        repo: str,
        commit: str,
        warnings: Sequence[str] | None = None,
    ) -> ExecutionResult:
        """Write rows to multiple tables and build ExecutionResult.

        Parameters
        ----------
        table_rows
            Mapping of table keys to row sequences.
        repo
            Repository identifier.
        commit
            Commit identifier.
        warnings
            Optional list of warning messages.

        Returns
        -------
        ExecutionResult
            Result with total rows written and table counts.
        """
        table_counts: dict[str, int] = {}
        scope = f"{repo}@{commit}"

        for table_key, rows in table_rows.items():
            if rows:
                result = self._storage.write_batch(table_key, list(rows), scope=scope)
                table_counts[table_key] = result.rows_affected

        result_warnings = tuple(warnings) if warnings else ()
        return ExecutionResult.ok(table_counts=table_counts, warnings=result_warnings)


class BaseToolIngestStep:
    """Base class for ingestion steps requiring tool execution.

    Provides shared initialization for steps that need storage and tool ports
    but not discovery (like CoverageIngestStep, TestsIngestStep).

    Parameters
    ----------
    storage
        Storage port for persisting data.
    tools
        Tool port for running external tools.
    """

    _storage: IngestStoragePort
    _tools: IngestToolPort

    def __init__(
        self,
        storage: IngestStoragePort,
        tools: IngestToolPort,
    ) -> None:
        """Initialize the step with storage and tool ports.

        Parameters
        ----------
        storage
            Storage port for persisting data.
        tools
            Tool port for running external tools.
        """
        self._storage = storage
        self._tools = tools


__all__ = ["BaseExtractStep", "BaseToolIngestStep", "ExecutionResult"]
