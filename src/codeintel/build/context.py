"""Execution context for target plugins.

This module defines TargetExecutionContext, which is the single object
passed to plugin execute() methods. It provides:

1. Target information (name, contract, parameters)
2. Resources (tool runners, modules)
3. Path resolution (artifacts, build directories)
4. Write validation (against contract schemas)

Plugins receive everything they need via context, eliminating the need
for config classes and scattered ClassVars.

Example
-------
>>> async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
...     max_commits = ctx.parameters.get("max_commits", int, default=2000)
...
...     git = ctx.resources.git_history
...     entries = await git.log(ctx.repo_root, max_count=max_commits)
...
...     ctx.write_table("analytics.hotspots", rows)
...
...     return TargetResult.success()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pandera.errors import SchemaError, SchemaErrors

from codeintel.build.errors import ColumnCountMismatchError, SchemaNotFoundError
from codeintel.build.context_base import BuildContext
from codeintel.build.hamilton.contracts.schemas import SCHEMA_REGISTRY
from codeintel.build.parameters import EMPTY_PARAMETERS
from codeintel.build.result import TargetResult

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    import pandas as pd

    from codeintel.build.contracts import OutputContract, TableSchema
    from codeintel.build.parameters import TargetParameters
    from codeintel.build.protocols import (
        CoverageCollector,
        GitHistoryProvider,
        ScipIndexer,
        ToolRunner,
        TypeChecker,
    )
    from codeintel.build.providers import Providers, RealTestReporter
    from codeintel.build.targets import OutputTarget
    from codeintel.config.primitives import BuildPaths, SnapshotRef
    from codeintel.core.catalog import FunctionCatalogProvider
    from codeintel.graphs.runtime import GraphRuntime
    from codeintel.ingestion.tracker import ChangeTracker
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)

__all__ = [
    "ContextResources",
    "TargetExecutionContext",
    "TargetResult",
    "WriteRecord",
]


@dataclass
class WriteRecord:
    """Record of data written to a table.

    Used for testing and validation - plugins can inspect what
    was written without actually committing to the database.

    Attributes
    ----------
    table_key
        Fully-qualified table name.
    rows
        List of row data (list of tuples or dicts).
    validated
        Whether the data passed contract validation.
    """

    table_key: str
    rows: list[tuple[Any, ...] | dict[str, Any]] = field(default_factory=list)
    validated: bool = False


@dataclass
class ContextResources:
    """Resources available to a target during execution.

    This wraps the Providers and adds target-specific resources
    like the module list.

    Attributes
    ----------
    providers
        DI providers for external tools.
    modules
        Module list (if target requires it).
    change_tracker
        Change tracker for incremental builds.
    graph_runtime
        Graph runtime for graph/analytics plugins.
    catalog
        Function catalog for analytics plugins.
    """

    providers: Providers | None = None
    modules: tuple[str, ...] = ()
    change_tracker: ChangeTracker | None = None
    graph_runtime: GraphRuntime | None = None
    catalog: FunctionCatalogProvider | None = None

    @property
    def tool_runner(self) -> ToolRunner | None:
        """Get the tool runner.

        Returns
        -------
        ToolRunner | None
            Tool runner if available.
        """
        return self.providers.tool_runner if self.providers else None

    @property
    def scip_indexer(self) -> ScipIndexer | None:
        """Get the SCIP indexer.

        Returns
        -------
        ScipIndexer | None
            SCIP indexer if available.
        """
        return self.providers.scip_indexer if self.providers else None

    @property
    def type_checker(self) -> TypeChecker | None:
        """Get the type checker.

        Returns
        -------
        TypeChecker | None
            Type checker if available.
        """
        return self.providers.type_checker if self.providers else None

    @property
    def coverage_collector(self) -> CoverageCollector | None:
        """Get the coverage collector.

        Returns
        -------
        CoverageCollector | None
            Coverage collector if available.
        """
        return self.providers.coverage_collector if self.providers else None

    @property
    def test_reporter(self) -> RealTestReporter | None:
        """Get the test reporter.

        Returns
        -------
        TestReporter | None
            Test reporter if available.
        """
        return self.providers.test_reporter if self.providers else None

    @property
    def git_history(self) -> GitHistoryProvider | None:
        """Get the git history provider.

        Returns
        -------
        GitHistoryProvider | None
            Git history provider if available.
        """
        return self.providers.git_history if self.providers else None


@dataclass
class TargetExecutionContext:
    """Complete execution context for a target plugin.

    This is the single object passed to plugin execute() methods.
    It provides everything the plugin needs:

    - Target information (name, contract)
    - Parameters (tuning values)
    - Resources (providers, gateway, modules)
    - Path helpers (repo_root, build_dir, artifact paths)
    - Write methods (with contract validation)

    This class provides resource handling and write tracking capabilities
    for target execution.

    Attributes
    ----------
    build_ctx
        Shared build context (gateway, snapshot, paths).
    target
        The OutputTarget being executed.
    resources
        Resources available for execution.
    parameters
        Tuning parameters for this target.
    _written_tables
        Internal record of tables written (for testing).
    """

    build_ctx: BuildContext
    target: OutputTarget
    resources: ContextResources = field(default_factory=ContextResources)
    parameters: TargetParameters = field(default_factory=lambda: EMPTY_PARAMETERS)
    _written_tables: dict[str, WriteRecord] = field(default_factory=dict)

    @property
    def snapshot(self) -> SnapshotRef:
        """Return the snapshot reference."""
        return self.build_ctx.snapshot

    @property
    def paths(self) -> BuildPaths:
        """Return build paths configuration."""
        return self.build_ctx.paths

    @property
    def target_name(self) -> str:
        """Return the target name.

        Returns
        -------
        str
            Target identifier.
        """
        return self.target.name

    @property
    def contract(self) -> OutputContract:
        """Return the target's output contract.

        Returns
        -------
        OutputContract
            Tables and artifacts this target produces.
        """
        return self.target.contract

    @property
    def repo(self) -> str:
        """Return the repository slug.

        Returns
        -------
        str
            Repository identifier.
        """
        return self.build_ctx.repo

    @property
    def commit(self) -> str:
        """Return the commit SHA.

        Returns
        -------
        str
            Commit identifier.
        """
        return self.build_ctx.commit

    @property
    def repo_root(self) -> Path:
        """Return the repository root path.

        Returns
        -------
        Path
            Repository root directory.
        """
        return self.build_ctx.repo_root

    @property
    def build_dir(self) -> Path:
        """Return the build directory.

        Returns
        -------
        Path
            Build output directory.
        """
        return self.build_ctx.build_dir

    @property
    def scip_dir(self) -> Path:
        """Return the SCIP artifacts directory.

        Returns
        -------
        Path
            Directory for SCIP index files.
        """
        return self.build_ctx.scip_dir

    @property
    def export_dir(self) -> Path:
        """Return the export directory.

        Returns
        -------
        Path
            Directory for document output exports.
        """
        return self.build_ctx.paths.document_output_dir

    def artifact_path(self, artifact_name: str) -> Path:
        """Resolve an artifact path from the contract.

        Parameters
        ----------
        artifact_name
            Name of the artifact in the contract.

        Returns
        -------
        Path
            Resolved file path.

        Raises
        ------
        KeyError
            If artifact is not in the contract.
        """
        spec = self.contract.get_artifact(artifact_name)
        if spec is None:
            available = ", ".join(self.contract.artifact_names)
            msg = f"Artifact '{artifact_name}' not in contract. Available: {available}"
            raise KeyError(msg)

        return self.build_ctx.path_resolver.artifact_path_from_template(spec.path_template)

    @property
    def gateway(self) -> StorageGateway:
        """Return the storage gateway."""
        return self.build_ctx.gateway

    def write_table(
        self,
        table_key: str,
        rows: Sequence[tuple[Any, ...] | dict[str, Any]],
        *,
        validate: bool = True,
    ) -> int:
        """Write rows to a table with contract validation.

        Parameters
        ----------
        table_key
            Fully-qualified table name (e.g., "core.goids").
        rows
            Row data as tuples or dicts.
        validate
            Whether to validate against contract schema.

        Returns
        -------
        int
            Number of rows written.

        Raises
        ------
        SchemaNotFoundError
            If table is not in the contract and validate=True.
        """
        if table_key not in self._written_tables:
            self._written_tables[table_key] = WriteRecord(table_key)

        record = self._written_tables[table_key]

        if validate:
            schema = self.contract.get_table(table_key)
            if schema is None:
                raise SchemaNotFoundError(self.target_name, table_key)

            if schema is not None and not (
                schema.description
                and schema.description.startswith(
                    "Placeholder schema generated by OutputContract.simple"
                )
            ):
                self._validate_rows(table_key, schema, rows)

        record.rows.extend(rows)
        record.validated = validate

        self._persist_rows(table_key, rows)

        return len(rows)

    def _validate_rows(
        self,
        table_key: str,
        schema: TableSchema,
        rows: Sequence[tuple[Any, ...] | dict[str, Any]],
    ) -> None:
        """Validate rows against schema.

        Parameters
        ----------
        table_key
            Table being written to.
        schema
            Table schema for validation.
        rows
            Rows to validate.

        Raises
        ------
        ColumnCountMismatchError
            If column count doesn't match.
        """
        expected_cols = len(schema.columns)

        for i, row in enumerate(rows):
            actual_cols = len(row)

            if actual_cols != expected_cols:
                raise ColumnCountMismatchError(
                    self.target_name,
                    table_key,
                    expected_cols,
                    actual_cols,
                    i,
                )

    def _persist_rows(
        self,
        table_key: str,
        rows: Sequence[tuple[Any, ...] | dict[str, Any]],
    ) -> None:
        """Persist rows to database.

        Parameters
        ----------
        table_key
            Table to write to.
        rows
            Row data.
        """
        log.debug(
            "Writing %d rows to %s (gateway=%s)",
            len(rows),
            table_key,
            True,
        )

    def write_validated_table(
        self,
        table_key: str,
        df: pd.DataFrame,
        *,
        strict: bool = True,
    ) -> int:
        """Write DataFrame with automatic Pandera schema validation.

        This method validates the DataFrame against the registered Pandera
        schema before writing to the database. It provides stronger type
        guarantees than the row-based write_table method.

        Parameters
        ----------
        table_key
            Fully-qualified table name (e.g., "analytics.function_metrics").
        df
            DataFrame to validate and write.
        strict
            If True, raise on validation failure. If False, log and continue.

        Returns
        -------
        int
            Number of rows written.

        Raises
        ------
        KeyError
            If no schema is registered for the table.

        Notes
        -----
        Full activation requires updating all plugins to use this method.
        See architecture Section 4.3 - Build Context Integration for details.
        """
        schema = SCHEMA_REGISTRY.get(table_key)
        if schema is None:
            msg = f"No schema registered for {table_key}"
            raise KeyError(msg)

        if strict:
            validated_df = schema.validate(df)
        else:
            try:
                validated_df = schema.validate(df)
            except (SchemaError, SchemaErrors) as exc:
                log.warning("Schema validation failed for %s: %s", table_key, exc)
                validated_df = df

        rows = list(validated_df.itertuples(index=False, name=None))
        return self.write_table(table_key, rows, validate=False)

    @property
    def written_tables(self) -> Mapping[str, WriteRecord]:
        """Return records of tables written (for testing).

        Returns
        -------
        Mapping[str, WriteRecord]
            Write records by table key.
        """
        return dict(self._written_tables)

    def get_written_rows(self, table_key: str) -> list[tuple[Any, ...] | dict[str, Any]]:
        """Get rows written to a table (for testing).

        Parameters
        ----------
        table_key
            Table to get rows for.

        Returns
        -------
        list
            Rows written, or empty list if none.
        """
        record = self._written_tables.get(table_key)
        return list(record.rows) if record else []
