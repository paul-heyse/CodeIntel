"""Typed fakes for ingestion and analytics tests."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
from anyio import to_thread
from coverage import Coverage

from codeintel.config import TestCoverageStepConfig
from codeintel.config.models import ToolsConfig
from codeintel.ingestion.infrastructure_utilities.tool_runner import (
    ToolName,
    ToolResult,
    ToolRunner,
)
from codeintel.ingestion.ports.storage import BatchResult, QueryResult
from codeintel.ingestion.tool_service import CoverageFileReport, ToolService
from codeintel.ingestion.tools.results import ScipIndexResult


def _mkdir_parents(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_text(path: Path, payload: str) -> None:
    path.write_text(payload, encoding="utf8")


class FakeCoverageData:
    """Lightweight coverage data implementing measured_files/contexts_by_lineno."""

    def __init__(self, contexts_by_file: dict[str, dict[int, set[str]]]) -> None:
        self._contexts_by_file = contexts_by_file

    def measured_files(self) -> list[str]:
        """
        Return measured file paths.

        Returns
        -------
        list[str]
            File paths observed in coverage data.
        """
        return list(self._contexts_by_file.keys())

    def contexts_by_lineno(self, filename: str) -> dict[int, set[str]]:
        """
        Return contexts keyed by line number for a file.

        Parameters
        ----------
        filename
            File path to resolve contexts for.

        Returns
        -------
        dict[int, set[str]]
            Mapping of line numbers to context identifiers.
        """
        return self._contexts_by_file.get(filename, {})


class FakeCoverage:
    """Coverage shim providing deterministic statements/contexts."""

    def __init__(
        self,
        statements: dict[str, list[int]],
        contexts: dict[str, dict[int, set[str]]],
    ) -> None:
        self._statements = statements
        self._contexts = contexts

    def analysis2(self, filename: str) -> tuple[str, list[int], list[int], list[int], list[int]]:
        stmts = self._statements.get(filename, [])
        return filename, stmts, [], [], stmts

    def get_data(self) -> FakeCoverageData:
        """
        Return deterministic coverage data wrapper.

        Returns
        -------
        FakeCoverageData
            Coverage data exposing measured files and contexts.
        """
        return FakeCoverageData(self._contexts)


class CoverageLoader(Protocol):
    """Protocol for injecting coverage loaders."""

    def __call__(self, cfg: TestCoverageStepConfig | object) -> Coverage:
        """Return a Coverage-compatible object."""
        raise NotImplementedError


class FakeToolRunner(ToolRunner):
    """ToolRunner stub returning canned payloads."""

    def __init__(
        self,
        cache_dir: Path,
        *,
        payloads: dict[str, Any] | None = None,
        on_run: Callable[[ToolName, list[str]], None] | None = None,
    ) -> None:
        super().__init__(cache_dir=cache_dir)
        self.payloads = payloads or {}
        self.calls: list[tuple[ToolName, list[str]]] = []
        self.on_run = on_run

    async def run_async(
        self,
        tool: ToolName | str,
        args: Sequence[str],
        *,
        cwd: Path | None = None,
        output_path: Path | None = None,
        timeout_s: float | None = None,
    ) -> ToolResult:
        """
        Execute a tool invocation with canned outputs.

        Returns
        -------
        ToolResult
            Structured result capturing stdout/stderr and codes.
        """
        _ = cwd
        _ = timeout_s
        tool_enum = tool if isinstance(tool, ToolName) else ToolName(str(tool))
        args_list = list(args)
        self.calls.append((tool_enum, args_list))
        if self.on_run is not None:
            self.on_run(tool_enum, args_list)
        payload_stdout = self.payloads.get(tool_enum.value, "")
        if output_path is not None and tool_enum in {ToolName.COVERAGE, ToolName.PYREFLY}:
            json_payload = self.payloads.get(
                f"{tool_enum.value}_json",
                self.payloads.get("json", {}),
            )
            await to_thread.run_sync(_mkdir_parents, output_path.parent)
            await to_thread.run_sync(_write_text, output_path, json.dumps(json_payload))
        return ToolResult(
            tool=tool_enum,
            args=tuple(args_list),
            returncode=0,
            stdout=str(payload_stdout),
            stderr="",
            output_path=output_path,
            duration_s=0.0,
        )


@dataclass(frozen=True)
class FakeScipResult:
    """SCIP result stand-in mirroring dataclass fields."""

    status: str = "success"
    index_scip: Path | None = None
    index_json: Path | None = None
    reason: str | None = None


def write_dummy_scip_files(base_dir: Path, *, index_content: str = "[]") -> tuple[Path, Path]:
    """
    Create minimal SCIP artifacts for tests.

    Returns
    -------
    tuple[Path, Path]
        Paths to index.scip and index.scip.json.
    """
    scip_dir = base_dir / "scip"
    scip_dir.mkdir(parents=True, exist_ok=True)
    index_scip = scip_dir / "index.scip"
    index_json = scip_dir / "index.scip.json"
    index_scip.write_text("scip-binary", encoding="utf8")
    index_json.write_text(index_content, encoding="utf8")
    return index_scip, index_json


def utcnow() -> datetime:
    """
    Return timezone-aware now for deterministic tests.

    Returns
    -------
    datetime
        Current timezone-aware datetime.
    """
    return datetime.now().astimezone()


# =============================================================================
# FakeIngestStorage - Protocol-compliant in-memory storage
# =============================================================================


@dataclass
class FakeIngestStorage:
    """Protocol-compliant in-memory storage implementing IngestStoragePort.

    This fake implements the full IngestStoragePort protocol with in-memory
    data structures, enabling tests to verify storage behavior without a
    real database while maintaining protocol compliance.

    Attributes
    ----------
    data : dict[str, list[Sequence[object]]]
        In-memory data store keyed by table_key.
    schemas : set[str]
        Set of table keys for which schema has been ensured.
    operations : list[tuple[str, str, object]]
        Log of operations for verification (operation_type, table_key, details).
    """

    data: dict[str, list[Sequence[object]]] = field(default_factory=dict)
    schemas: set[str] = field(default_factory=set)
    operations: list[tuple[str, str, object]] = field(default_factory=list)

    def ensure_schema(self, table_key: str) -> None:
        """Ensure the schema exists for a table.

        Parameters
        ----------
        table_key
            Registry table key (e.g., "core.ast_nodes").
        """
        self.schemas.add(table_key)
        if table_key not in self.data:
            self.data[table_key] = []
        self.operations.append(("ensure_schema", table_key, None))

    def write_batch(
        self,
        table_key: str,
        rows: Sequence[Sequence[object]],
        *,
        scope: str | None = None,
    ) -> BatchResult:
        """Write a batch of rows to a table.

        Parameters
        ----------
        table_key
            Registry table key (e.g., "core.ast_nodes").
        rows
            Row data matching the table's column order.
        scope
            Optional scope identifier for logging.

        Returns
        -------
        BatchResult
            Metadata about the write operation.
        """
        if table_key not in self.data:
            self.data[table_key] = []
        self.data[table_key].extend(rows)
        self.operations.append(("write_batch", table_key, {"rows": len(rows), "scope": scope}))
        return BatchResult(table_key=table_key, rows_written=len(rows), duration_s=0.0)

    def delete_by_params(
        self,
        table_key: str,
        params: Sequence[object],
    ) -> int:
        """Delete rows matching the given parameters.

        Parameters
        ----------
        table_key
            Registry table key.
        params
            Parameters for the delete statement.

        Returns
        -------
        int
            Number of rows deleted (always 0 in this fake).
        """
        self.operations.append(("delete_by_params", table_key, {"params": params}))
        return 0

    def delete_by_paths(
        self,
        table_key: str,
        paths: Sequence[str],
        *,
        path_column: str = "rel_path",
    ) -> int:
        """Delete rows where path_column matches any of the provided paths.

        Parameters
        ----------
        table_key
            Registry table key.
        paths
            List of path values to delete.
        path_column
            Name of the column containing paths.

        Returns
        -------
        int
            Number of rows deleted (always 0 in this fake).
        """
        self.operations.append(
            ("delete_by_paths", table_key, {"paths": paths, "path_column": path_column})
        )
        return 0

    def execute_query(
        self,
        sql: str,
        params: Sequence[object] | None = None,
    ) -> QueryResult:
        """Execute a query and return results.

        Parameters
        ----------
        sql
            SQL query string.
        params
            Optional query parameters.

        Returns
        -------
        QueryResult
            Empty query results (queries not supported in fake).
        """
        self.operations.append(("execute_query", sql, {"params": params}))
        return QueryResult(rows=[], columns=(), row_count=0)

    def fetch_dataframe(
        self,
        sql: str,
        params: Sequence[object] | None = None,
    ) -> object:
        """Execute a query and return results as a DataFrame.

        Parameters
        ----------
        sql
            SQL query string.
        params
            Optional query parameters.

        Returns
        -------
        object
            Empty DataFrame-like object.
        """
        self.operations.append(("fetch_dataframe", sql, {"params": params}))
        return pd.DataFrame()


# =============================================================================
# FakeToolService - Protocol-compliant ToolService with deterministic results
# =============================================================================


@dataclass
class FakeToolServiceConfig:
    """Configuration for FakeToolService behavior.

    Attributes
    ----------
    pyright_errors : dict[str, int]
        Mapping of file paths to error counts for pyright.
    pyrefly_errors : dict[str, int]
        Mapping of file paths to error counts for pyrefly.
    ruff_errors : dict[str, int]
        Mapping of file paths to error counts for ruff.
    coverage_reports : list[CoverageFileReport] | None
        Coverage reports to return, or None for empty.
    scip_result : ScipIndexResult | None
        SCIP result to return, or None for empty.
    pytest_success : bool
        Whether pytest should succeed.
    raise_on_pyright : Exception | None
        Exception to raise on pyright calls.
    raise_on_pyrefly : Exception | None
        Exception to raise on pyrefly calls.
    raise_on_ruff : Exception | None
        Exception to raise on ruff calls.
    raise_on_coverage : Exception | None
        Exception to raise on coverage calls.
    raise_on_scip : Exception | None
        Exception to raise on scip calls.
    raise_on_pytest : Exception | None
        Exception to raise on pytest calls.
    """

    pyright_errors: dict[str, int] = field(default_factory=dict)
    pyrefly_errors: dict[str, int] = field(default_factory=dict)
    ruff_errors: dict[str, int] = field(default_factory=dict)
    coverage_reports: list[CoverageFileReport] | None = None
    scip_result: ScipIndexResult | None = None
    pytest_success: bool = True
    raise_on_pyright: Exception | None = None
    raise_on_pyrefly: Exception | None = None
    raise_on_ruff: Exception | None = None
    raise_on_coverage: Exception | None = None
    raise_on_scip: Exception | None = None
    raise_on_pytest: Exception | None = None


class FakeToolService(ToolService):
    """ToolService subclass with deterministic, configurable results.

    This fake extends the real ToolService with configurable responses,
    enabling tests to verify tool integration behavior without running real
    external tools. It inherits from ToolService for full type compatibility.

    Parameters
    ----------
    config : FakeToolServiceConfig | None
        Configuration for fake behavior. Defaults to empty/success responses.
    cache_dir : Path | None
        Cache directory for the underlying FakeToolRunner.

    Attributes
    ----------
    fake_config : FakeToolServiceConfig
        Current configuration.
    calls : list[tuple[str, dict[str, object]]]
        Log of method calls for verification.
    """

    def __init__(
        self, config: FakeToolServiceConfig | None = None, cache_dir: Path | None = None
    ) -> None:
        """Initialize with optional configuration."""
        effective_cache = cache_dir or Path(tempfile.gettempdir()) / "fake_tool_cache"
        fake_runner = FakeToolRunner(cache_dir=effective_cache)
        super().__init__(fake_runner)
        self.fake_config = config or FakeToolServiceConfig()
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def run_pyright(self, repo_root: Path) -> dict[str, int]:
        """Run pyright and return configured error counts.

        Parameters
        ----------
        repo_root
            Repository root (logged but not used).

        Returns
        -------
        dict[str, int]
            Configured pyright errors.
        """
        self.calls.append(("run_pyright", {"repo_root": repo_root}))
        if self.fake_config.raise_on_pyright is not None:
            raise self.fake_config.raise_on_pyright
        return dict(self.fake_config.pyright_errors)

    async def run_pyrefly(self, repo_root: Path) -> dict[str, int]:
        """Run pyrefly and return configured error counts.

        Parameters
        ----------
        repo_root
            Repository root (logged but not used).

        Returns
        -------
        dict[str, int]
            Configured pyrefly errors.
        """
        self.calls.append(("run_pyrefly", {"repo_root": repo_root}))
        if self.fake_config.raise_on_pyrefly is not None:
            raise self.fake_config.raise_on_pyrefly
        return dict(self.fake_config.pyrefly_errors)

    async def run_ruff(self, repo_root: Path) -> dict[str, int]:
        """Run ruff and return configured error counts.

        Parameters
        ----------
        repo_root
            Repository root (logged but not used).

        Returns
        -------
        dict[str, int]
            Configured ruff errors.
        """
        self.calls.append(("run_ruff", {"repo_root": repo_root}))
        if self.fake_config.raise_on_ruff is not None:
            raise self.fake_config.raise_on_ruff
        return dict(self.fake_config.ruff_errors)

    async def run_coverage_json(
        self,
        repo_root: Path,
        *,
        coverage_file: Path | None = None,
        output_path: Path | None = None,
    ) -> list[CoverageFileReport]:
        """Run coverage and return configured reports.

        Parameters
        ----------
        repo_root
            Repository root (logged but not used).
        coverage_file
            Coverage file path (logged but not used).
        output_path
            Output path (logged but not used).

        Returns
        -------
        list[CoverageFileReport]
            Configured coverage reports.
        """
        self.calls.append(
            (
                "run_coverage_json",
                {"repo_root": repo_root, "coverage_file": coverage_file, "output_path": output_path},
            )
        )
        if self.fake_config.raise_on_coverage is not None:
            raise self.fake_config.raise_on_coverage
        return list(self.fake_config.coverage_reports or [])

    async def run_scip_full(
        self,
        repo_root: Path,
        *,
        output_scip: Path,
        output_json: Path,
        target_dir: Path | None = None,
    ) -> ScipIndexResult:
        """Run full SCIP indexing and return configured result.

        Parameters
        ----------
        repo_root
            Repository root (logged but not used).
        output_scip
            Output SCIP file path (logged but not used).
        output_json
            Output JSON file path (logged but not used).
        target_dir
            Target directory (logged but not used).

        Returns
        -------
        ScipIndexResult
            Configured SCIP result.
        """
        self.calls.append(
            (
                "run_scip_full",
                {
                    "repo_root": repo_root,
                    "output_scip": output_scip,
                    "output_json": output_json,
                    "target_dir": target_dir,
                },
            )
        )
        if self.fake_config.raise_on_scip is not None:
            raise self.fake_config.raise_on_scip
        return self.fake_config.scip_result or ScipIndexResult.empty()

    async def run_scip_shard(
        self,
        repo_root: Path,
        *,
        rel_paths: list[str],
        output_scip: Path,
        output_json: Path,
        target_dir: Path | None = None,
    ) -> ScipIndexResult:
        """Run SCIP indexing for a shard and return configured result.

        Parameters
        ----------
        repo_root
            Repository root (logged but not used).
        rel_paths
            Relative paths to index (logged but not used).
        output_scip
            Output SCIP file path (logged but not used).
        output_json
            Output JSON file path (logged but not used).
        target_dir
            Target directory (logged but not used).

        Returns
        -------
        ScipIndexResult
            Configured SCIP result.
        """
        self.calls.append(
            (
                "run_scip_shard",
                {
                    "repo_root": repo_root,
                    "rel_paths": rel_paths,
                    "output_scip": output_scip,
                    "output_json": output_json,
                    "target_dir": target_dir,
                },
            )
        )
        if self.fake_config.raise_on_scip is not None:
            raise self.fake_config.raise_on_scip
        return self.fake_config.scip_result or ScipIndexResult.empty()

    async def run_pytest_report(
        self,
        repo_root: Path,
        *,
        json_report_path: Path,
    ) -> bool:
        """Run pytest and return configured success.

        Parameters
        ----------
        repo_root
            Repository root (logged but not used).
        json_report_path
            JSON report path (logged but not used).

        Returns
        -------
        bool
            Configured pytest success status.
        """
        self.calls.append(
            ("run_pytest_report", {"repo_root": repo_root, "json_report_path": json_report_path})
        )
        if self.fake_config.raise_on_pytest is not None:
            raise self.fake_config.raise_on_pytest
        return self.fake_config.pytest_success


# =============================================================================
# Typed Fake Config Primitives for Config Factory Tests
# =============================================================================


@dataclass(frozen=True)
class FakeSnapshotRef:
    """Fake SnapshotRef for config factory and plugin tests.

    Mirrors the real SnapshotRef interface with sensible test defaults.

    Attributes
    ----------
    repo : str
        Repository slug.
    commit : str
        Commit identifier.
    repo_root : Path
        Path to repository root.
    branch : str | None
        Optional branch name.
    """

    repo: str = "test/repo"
    commit: str = "testcommit"
    repo_root: Path = field(default_factory=lambda: Path("/repo"))
    branch: str | None = None


@dataclass(frozen=True)
class FakeBuildPaths:
    """Fake BuildPaths for config factory and plugin tests.

    Mirrors the real BuildPaths interface with sensible test defaults.

    Attributes
    ----------
    build_dir : Path
        Root build directory.
    db_path : Path
        Path to DuckDB database.
    document_output_dir : Path
        Directory for output documents.
    scip_dir : Path
        Directory for SCIP artifacts.
    coverage_json : Path
        Path for coverage JSON.
    pytest_report : Path
        Path for pytest JSON report.
    tool_cache : Path
        Cache directory for tools.
    log_db_path : Path
        Path to logging database.
    """

    build_dir: Path = field(default_factory=lambda: Path("/build"))
    db_path: Path = field(default_factory=lambda: Path("/build/codeintel.duckdb"))
    document_output_dir: Path = field(default_factory=lambda: Path("/build/docs"))
    scip_dir: Path = field(default_factory=lambda: Path("/build/scip"))
    coverage_json: Path = field(default_factory=lambda: Path("/build/coverage.json"))
    pytest_report: Path = field(default_factory=lambda: Path("/build/pytest.json"))
    tool_cache: Path = field(default_factory=lambda: Path("/cache"))
    log_db_path: Path = field(default_factory=lambda: Path("/build/log.duckdb"))


@dataclass
class FakePluginContext:
    """Fake IngestExecutionContext for config factory tests.

    Mirrors the real IngestExecutionContext interface with typed fields
    for proper static analysis.

    Attributes
    ----------
    snapshot : FakeSnapshotRef
        Snapshot reference.
    paths : FakeBuildPaths
        Build paths.
    tools : ToolsConfig | None
        Optional tools configuration.
    tracker : object | None
        Optional change tracker (using object to avoid circular imports).
    tool_service : ToolService | None
        Optional tool service.
    code_profile : object | None
        Optional code scan profile.
    config_profile : object | None
        Optional config scan profile.
    """

    snapshot: FakeSnapshotRef = field(default_factory=FakeSnapshotRef)
    paths: FakeBuildPaths = field(default_factory=FakeBuildPaths)
    tools: ToolsConfig | None = None
    tracker: object | None = None
    tool_service: ToolService | None = None
    code_profile: object | None = None
    config_profile: object | None = None
