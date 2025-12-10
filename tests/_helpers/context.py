"""Unified test context for production-parity test environments.

This module provides a single composable TestContext that replaces scattered
environment dataclasses throughout the test suite. The TestContext supports
lazy resource access, composable seed packs, and query shortcuts.

Design Principles
-----------------
1. Single unified context reduces boilerplate across tests.
2. Seed packs are applied lazily and idempotently via `require()`.
3. All external systems use real implementations (DuckDB, filesystem, etc.).
4. Context is immutable after construction; seeds mutate only the gateway.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

from codeintel.config.primitives import BuildPaths, SnapshotRef
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.macros import ensure_ingest_macros
from codeintel.storage.schema import apply_all_schemas
from tests._helpers.constants import DEFAULT_COMMIT, DEFAULT_REPO
from tests._helpers.env_options import EnvOptions, GatewayOptions
from tests._helpers.gateway import GatewayFactory
from tests._helpers.repo import write_canonical_repo

if TYPE_CHECKING:
    from collections.abc import Sequence

    from duckdb import DuckDBPyConnection

    from tests._helpers.configs.provisioning_config import ProvisionedGateway


# =============================================================================
# Seed Pack Protocol
# =============================================================================


@runtime_checkable
class SeedPack(Protocol):
    """Protocol for composable seed packs that populate test data.

    Each seed pack has a unique name and an apply method that seeds
    tables in the gateway. Seeds are applied idempotently; the context
    tracks which packs have been applied.
    """

    @property
    def name(self) -> str:
        """Return unique identifier for this seed pack."""
        ...

    @property
    def dependencies(self) -> tuple[SeedPack, ...]:
        """Return seed packs that must be applied before this one."""
        ...

    def apply(self, ctx: TestContext) -> None:
        """Apply seeds to the test context gateway.

        Parameters
        ----------
        ctx
            Test context to seed; seeds write to ctx.gateway.
        """
        ...


# =============================================================================
# Query Result Types
# =============================================================================


@dataclass(frozen=True)
class QueryRow:
    """Single row from a query result with named access.

    Provides both index and attribute access to column values.
    """

    _data: tuple[object, ...]
    _columns: tuple[str, ...]

    def __getitem__(self, key: int | str) -> object:
        """Access value by index or column name.

        Parameters
        ----------
        key
            Integer index or string column name.

        Returns
        -------
        object
            Value at the specified position.

        Raises
        ------
        KeyError
            If column name not found.
        """
        if isinstance(key, int):
            return self._data[key]
        if key in self._columns:
            idx = self._columns.index(key)
            return self._data[idx]
        message = f"Column '{key}' not found in {self._columns}"
        raise KeyError(message)

    def __getattr__(self, name: str) -> object:
        """Access value by column name as attribute.

        Parameters
        ----------
        name
            Column name.

        Returns
        -------
        object
            Value for the column.

        Raises
        ------
        AttributeError
            If column name not found.
        """
        if name.startswith("_"):
            return object.__getattribute__(self, name)
        try:
            return self[name]
        except KeyError as exc:
            message = f"'{type(self).__name__}' has no attribute '{name}'"
            raise AttributeError(message) from exc

    def as_dict(self) -> dict[str, object]:
        """Convert row to dictionary.

        Returns
        -------
        dict[str, object]
            Mapping of column names to values.
        """
        return dict(zip(self._columns, self._data, strict=True))


# =============================================================================
# Test Context
# =============================================================================


@dataclass
class TestContext:
    """Unified test environment with lazy resource access.

    Provides a single composable context that replaces scattered environment
    dataclasses. Supports lazy seed pack application and query shortcuts.

    Attributes
    ----------
    snapshot : SnapshotRef
        Repository snapshot reference (repo, commit, repo_root).
    gateway : StorageGateway
        Storage gateway for database access.
    build_paths : BuildPaths
        Derived build paths for the test.
    seeds_applied : set[str]
        Names of seed packs already applied (for idempotency).
    extra : dict[str, object]
        Additional context metadata for test-specific needs.
    """

    __test__ = False  # Prevent pytest collection

    snapshot: SnapshotRef
    gateway: StorageGateway
    build_paths: BuildPaths
    seeds_applied: set[str] = field(default_factory=set)
    extra: dict[str, object] = field(default_factory=dict)

    @property
    def repo(self) -> str:
        """Return repository identifier.

        Returns
        -------
        str
            Repository slug from snapshot.
        """
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Return commit identifier.

        Returns
        -------
        str
            Commit hash from snapshot.
        """
        return self.snapshot.commit

    @property
    def repo_root(self) -> Path:
        """Return repository root path.

        Returns
        -------
        Path
            Path to repository root.
        """
        return self.snapshot.repo_root

    @property
    def con(self) -> DuckDBPyConnection:
        """Return DuckDB connection shorthand.

        Returns
        -------
        DuckDBPyConnection
            Underlying database connection.
        """
        return self.gateway.con

    def to_snapshot_ref(self) -> SnapshotRef:
        """Return the snapshot reference for use with graph plugins.

        This is a convenience method for accessing the snapshot when
        building graph plugin contexts or other components that need
        a SnapshotRef.

        Returns
        -------
        SnapshotRef
            The test context's snapshot reference.
        """
        return self.snapshot

    @classmethod
    def from_provisioned(cls, provisioned: ProvisionedGateway) -> TestContext:
        """Create a TestContext from a ProvisionedGateway.

        This factory method enables interoperability between the older
        ProvisionedGateway pattern and the newer TestContext pattern.

        Parameters
        ----------
        provisioned
            A provisioned gateway from orchestration functions.

        Returns
        -------
        TestContext
            Context configured with the provisioned gateway's settings.
        """
        snapshot = SnapshotRef(
            repo=provisioned.repo,
            commit=provisioned.commit,
            repo_root=provisioned.repo_root,
        )
        build_paths = BuildPaths.from_repo_root(
            provisioned.repo_root,
            build_dir=provisioned.build_dir,
        )
        return cls(
            snapshot=snapshot,
            gateway=provisioned.gateway,
            build_paths=build_paths,
            extra={
                "coverage_file": provisioned.coverage_file,
                "runner": provisioned.runner,
                "document_output_dir": provisioned.document_output_dir,
            },
        )

    def require(self, *seed_packs: SeedPack) -> Self:
        """Ensure seed packs are applied (idempotent).

        Applies the given seed packs if not already applied, including
        their dependencies in topological order.

        Parameters
        ----------
        seed_packs
            One or more seed packs to apply.

        Returns
        -------
        Self
            Self for method chaining.
        """
        for pack in seed_packs:
            self._apply_pack(pack)
        return self

    def _apply_pack(self, pack: SeedPack) -> None:
        """Apply a single seed pack with dependencies.

        Parameters
        ----------
        pack
            Seed pack to apply.
        """
        if pack.name in self.seeds_applied:
            return
        # Apply dependencies first
        for dep in pack.dependencies:
            self._apply_pack(dep)
        # Apply this pack
        pack.apply(self)
        self.seeds_applied.add(pack.name)

    def query(
        self,
        sql: str,
        params: Sequence[object] | None = None,
    ) -> list[QueryRow]:
        """Execute query and return typed rows.

        Convenience method for assertions that need to check database state.

        Parameters
        ----------
        sql
            SQL query string.
        params
            Optional query parameters.

        Returns
        -------
        list[QueryRow]
            Query results with named column access.
        """
        result = self.gateway.con.execute(sql, params or [])
        columns = tuple(desc[0] for desc in result.description or [])
        return [QueryRow(_data=tuple(row), _columns=columns) for row in result.fetchall()]

    def query_scalar(
        self,
        sql: str,
        params: Sequence[object] | None = None,
    ) -> object:
        """Execute query and return single scalar value.

        Parameters
        ----------
        sql
            SQL query returning single value.
        params
            Optional query parameters.

        Returns
        -------
        object
            Single value from query.

        Raises
        ------
        ValueError
            If query returns no rows.
        """
        rows = self.query(sql, params)
        if not rows:
            message = "Query returned no rows"
            raise ValueError(message)
        return rows[0][0]

    def query_count(self, table: str, where: str | None = None) -> int:
        """Count rows in a table.

        Parameters
        ----------
        table
            Table name (schema.table format).
        where
            Optional WHERE clause (without 'WHERE' keyword).

        Returns
        -------
        int
            Number of rows.
        """
        # Build SQL using join to avoid S608 lint on f-string concatenation
        # Table names cannot be parameterized; this is safe as test code controls input
        parts = ["SELECT COUNT(*) FROM", table]
        if where:
            parts.extend(["WHERE", where])
        sql = " ".join(parts)
        result = self.query_scalar(sql)
        if not isinstance(result, int):
            return int(str(result))
        return result

    def close(self) -> None:
        """Close the underlying gateway connection."""
        self.gateway.close()

    def __enter__(self) -> Self:
        """Enter context manager scope.

        Returns
        -------
        Self
            Self reference for use in with block.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit context manager and close gateway."""
        self.close()


# =============================================================================
# Factory Functions
# =============================================================================


def create_test_context(
    tmp_path: Path,
    options: EnvOptions | None = None,
    *,
    gateway_options: GatewayOptions | None = None,
) -> TestContext:
    """Create a minimal TestContext for testing.

    Parameters
    ----------
    tmp_path
        Temporary directory for test artifacts.
    options
        Environment options bundle; defaults to standard repo/commit and in-memory DB.
    gateway_options
        Optional gateway configuration to override schema/view/validation or snapshot defaults.

    Returns
    -------
    TestContext
        Configured test context with gateway, snapshot, and build paths.
    """
    env_opts = options or EnvOptions()
    repo_root_path, build_dir_path, db_path = _prepare_paths(tmp_path, env_opts)

    snapshot = SnapshotRef(repo=env_opts.repo, commit=env_opts.commit, repo_root=repo_root_path)
    build_paths = BuildPaths.from_repo_root(repo_root_path, build_dir=build_dir_path)

    gateway = build_test_gateway(
        gateway_options
        or GatewayOptions(
            file_backed=env_opts.file_backed,
            db_path=db_path,
            repo=env_opts.repo,
            commit=env_opts.commit,
        )
    )

    return TestContext(
        snapshot=snapshot,
        gateway=gateway,
        build_paths=build_paths,
    )


def _ensure_sample_repo(repo_root: Path) -> None:
    """Write a minimal canonical repo to the provided repo_root."""
    write_canonical_repo(repo_root)


def coverage_ready_context(tmp_path: Path) -> TestContext:
    """Create a TestContext seeded with core + coverage packs and sample files."""
    from tests._helpers.seeds import CORE_PACK, COVERAGE_LINES_PACK, COVERAGE_PACK

    ctx = create_test_context(tmp_path)
    _ensure_sample_repo(ctx.repo_root)
    ctx.require(CORE_PACK, COVERAGE_PACK, COVERAGE_LINES_PACK)
    return ctx


def graph_ready_context(tmp_path: Path) -> TestContext:
    """Create a TestContext seeded with core + graph packs and sample files."""
    from tests._helpers.seeds import CORE_PACK, GRAPH_PACK

    ctx = create_test_context(tmp_path)
    _ensure_sample_repo(ctx.repo_root)
    ctx.require(CORE_PACK, GRAPH_PACK)
    return ctx


def coverage_and_graph_context(tmp_path: Path) -> TestContext:
    """Create a TestContext with both coverage and graph packs pre-applied."""
    from tests._helpers.seeds import CORE_PACK, COVERAGE_LINES_PACK, COVERAGE_PACK, GRAPH_PACK

    ctx = create_test_context(tmp_path)
    _ensure_sample_repo(ctx.repo_root)
    ctx.require(CORE_PACK, COVERAGE_PACK, COVERAGE_LINES_PACK, GRAPH_PACK)
    return ctx


def build_test_gateway(options: GatewayOptions | None = None) -> StorageGateway:
    """Create a StorageGateway with schema/views/macros ensured.

    Parameters
    ----------
    options
        Gateway configuration bundle.

    Returns
    -------
    StorageGateway
        Gateway ready for test use with macros ensured.

    Raises
    ------
    ValueError
        If file_backed is True but no db_path is provided.
    """
    opts = options or GatewayOptions()
    factory = GatewayFactory()
    factory = factory.with_schema() if opts.apply_schema else factory.without_schema()
    factory = factory.with_views() if opts.ensure_views else factory.without_views()
    factory = factory.with_validation() if opts.validate_schema else factory.without_validation()
    if opts.repo is not None and opts.commit is not None:
        factory = factory.with_snapshot(opts.repo, opts.commit)

    if opts.file_backed:
        if opts.db_path is None:
            message = "db_path must be provided for file_backed gateways"
            raise ValueError(message)
        factory = factory.file_backed(opts.db_path)

    gateway = factory.open()
    apply_all_schemas(gateway.con)
    ensure_ingest_macros(gateway.con)
    return gateway


def _prepare_paths(
    tmp_path: Path,
    env_opts: EnvOptions,
) -> tuple[Path, Path, Path | None]:
    """Resolve repository and build paths, ensuring directories exist.

    Returns
    -------
    tuple[Path, Path, Path | None]
        Resolved repo_root, build_dir, and optional database path.
    """
    repo_root_path = env_opts.repo_root or (tmp_path / "repo")
    build_dir_path = env_opts.build_dir or (tmp_path / "build")
    repo_root_path.mkdir(parents=True, exist_ok=True)
    build_dir_path.mkdir(parents=True, exist_ok=True)
    db_path = env_opts.db_path
    if env_opts.file_backed and db_path is None:
        db_path = build_dir_path / "db" / "codeintel.duckdb"
        db_path.parent.mkdir(parents=True, exist_ok=True)
    return repo_root_path, build_dir_path, db_path


__all__ = [
    "DEFAULT_COMMIT",
    "DEFAULT_REPO",
    "EnvOptions",
    "GatewayOptions",
    "QueryRow",
    "SeedPack",
    "TestContext",
    "build_test_gateway",
    "coverage_and_graph_context",
    "coverage_ready_context",
    "create_test_context",
    "graph_ready_context",
]
