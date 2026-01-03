"""Shared helpers for isolated gateway/DuckDB test setup (relation-first).

This module builds gateways without ingest macro registration. Tests should
exercise DuckDB relation and SQLGlot paths and avoid legacy storage adapters.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.build.config import BuildConfig
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.meta.contract_catalog import persist_contract_catalog_to_connection
from codeintel.build.providers import create_default_providers
from codeintel.build.schemas.contract_service import get_contract_service
from codeintel.config.models import ToolsConfig
from codeintel.config.primitives import BuildPaths, SnapshotRef
from codeintel.core.config.settings import (
    BuildSettings,
    ExportAuditSettings,
    HamiltonExecutionSettings,
)
from codeintel.core.schemas.provider import MappingSchemaProvider
from codeintel.runtime.compose import compose_runtime
from codeintel.storage.gateway import (
    DuckDBConnection,
    StorageConfig,
    open_gateway,
    open_inference_gateway,
)
from codeintel.storage.gateway import open_memory_gateway as _open_memory_gateway
from codeintel.storage.gateway.factory import MemoryGatewayOptions
from tests._helpers.assertions import ModulesAssertions
from tests._helpers.modules_expectations import modules_expected_from_repo_tree

if TYPE_CHECKING:
    from collections.abc import Iterator

    from codeintel.storage.gateway import StorageGateway
    from tests._helpers.env_options import GatewayOptions


class GatewayFactory:
    """Unified gateway creation with composable options.

    Provide a fluent builder interface for creating test gateways with
    consistent configuration. This consolidates the various gateway creation
    functions into a single, composable interface.

    Example
    -------
    >>> gateway = GatewayFactory().file_backed(db_path).with_schema().open()
    >>> gateway = GatewayFactory.from_options(opts).open()
    """

    def __init__(self) -> None:
        """Initialize factory with defaults."""
        self._apply_schema: bool = True
        self._ensure_views: bool = True
        self._validate_schema: bool = True
        self._strict_schema: bool = True
        self._file_backed: bool = False
        self._db_path: Path | None = None
        self._repo: str | None = None
        self._commit: str | None = None

    @classmethod
    def from_options(cls, options: GatewayOptions) -> GatewayFactory:
        """Create a factory configured from a GatewayOptions dataclass.

        Parameters
        ----------
        options
            Gateway configuration options.

        Returns
        -------
        GatewayFactory
            Factory configured with the provided options.
        """
        factory = cls()
        factory._apply_schema = options.apply_schema
        factory._ensure_views = options.ensure_views
        factory._validate_schema = options.validate_schema
        factory._strict_schema = options.strict_schema
        factory._file_backed = options.file_backed
        factory._db_path = options.db_path
        factory._repo = options.repo
        factory._commit = options.commit
        return factory

    def with_schema(self) -> GatewayFactory:
        """Enable schema application (default).

        Returns
        -------
        GatewayFactory
            Self for chaining.
        """
        self._apply_schema = True
        return self

    def without_schema(self) -> GatewayFactory:
        """Disable schema application.

        Returns
        -------
        GatewayFactory
            Self for chaining.
        """
        self._apply_schema = False
        return self

    def with_views(self) -> GatewayFactory:
        """Enable view creation (default).

        Returns
        -------
        GatewayFactory
            Self for chaining.
        """
        self._ensure_views = True
        return self

    def without_views(self) -> GatewayFactory:
        """Disable view creation.

        Returns
        -------
        GatewayFactory
            Self for chaining.
        """
        self._ensure_views = False
        return self

    def with_validation(self) -> GatewayFactory:
        """Enable schema validation (default).

        Returns
        -------
        GatewayFactory
            Self for chaining.
        """
        self._validate_schema = True
        return self

    def without_validation(self) -> GatewayFactory:
        """Disable schema validation.

        Returns
        -------
        GatewayFactory
            Self for chaining.
        """
        self._validate_schema = False
        return self

    def strict(self) -> GatewayFactory:
        """Enable strict schema mode (default).

        Returns
        -------
        GatewayFactory
            Self for chaining.
        """
        self._strict_schema = True
        return self

    def relaxed(self) -> GatewayFactory:
        """Disable strict schema mode.

        Returns
        -------
        GatewayFactory
            Self for chaining.
        """
        self._strict_schema = False
        return self

    def file_backed(self, db_path: Path) -> GatewayFactory:
        """Use a file-backed database instead of in-memory.

        Parameters
        ----------
        db_path
            Path to the database file.

        Returns
        -------
        GatewayFactory
            Self for chaining.
        """
        self._file_backed = True
        self._db_path = db_path
        return self

    def in_memory(self) -> GatewayFactory:
        """Use an in-memory database (default).

        Returns
        -------
        GatewayFactory
            Self for chaining.
        """
        self._file_backed = False
        self._db_path = None
        return self

    def with_snapshot(self, repo: str, commit: str) -> GatewayFactory:
        """Set the repository snapshot.

        Parameters
        ----------
        repo
            Repository identifier.
        commit
            Commit hash.

        Returns
        -------
        GatewayFactory
            Self for chaining.
        """
        self._repo = repo
        self._commit = commit
        return self

    def open(self) -> StorageGateway:
        """Create and return the configured gateway.

        Returns
        -------
        StorageGateway
            Configured gateway ready for use.

        Raises
        ------
        ValueError
            If db_path is not set for file-backed gateway.
        """
        if self._file_backed:
            if self._db_path is None:
                msg = "db_path must be set for file-backed gateway"
                raise ValueError(msg)
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            cfg = StorageConfig(
                db_path=self._db_path,
                read_only=False,
                apply_schema=self._apply_schema,
                ensure_views=self._ensure_views,
                validate_schema=self._validate_schema,
            )
            gateway = open_gateway(cfg, seed_contract_catalog=seed_contract_catalog)
        else:
            effective_ensure_views = self._ensure_views or self._strict_schema
            effective_validate_schema = self._validate_schema or self._strict_schema
            gateway = _open_memory_gateway(
                options=MemoryGatewayOptions(
                    apply_schema=self._apply_schema,
                    ensure_views=effective_ensure_views,
                    validate_schema=effective_validate_schema,
                    repo=self._repo,
                    commit=self._commit,
                ),
                seed_contract_catalog=seed_contract_catalog,
            )

        return gateway

    @classmethod
    def open_on_disk(
        cls,
        db_path: Path,
        *,
        options: GatewayOptions | None = None,
    ) -> StorageGateway:
        """Open a file-backed gateway with optional overrides.

        Parameters
        ----------
        db_path
            Path to the DuckDB file on disk.
        options
            Optional GatewayOptions to apply before opening.

        Returns
        -------
        StorageGateway
            File-backed gateway opened with schema and views.
        """
        factory = cls.from_options(options) if options else cls()
        return factory.file_backed(db_path).open()


@contextmanager
def analytics_gateway(options: GatewayOptions | None = None) -> Iterator[StorageGateway]:
    """Context-managed gateway creation for analytics tests.

    Parameters
    ----------
    options
        Optional GatewayOptions to configure the gateway.

    Yields
    ------
    StorageGateway
        Gateway with schema/views applied.
    """
    factory = GatewayFactory.from_options(options) if options else GatewayFactory()
    gateway = factory.open()
    try:
        yield gateway
    finally:
        gateway.close()


def _ensure_contract_service_configured() -> None:
    try:
        get_contract_service()
    except RuntimeError:
        providers = create_default_providers(ToolsConfig.default())
        snapshot = SnapshotRef.from_args(
            repo="demo/repo",
            commit="deadbeef",
            repo_root=Path.cwd(),
        )
        gateway = open_inference_gateway(schema_provider=MappingSchemaProvider({}))
        try:
            env = BuildEnv(
                gateway=gateway,
                snapshot=snapshot,
                paths=BuildPaths.from_repo_root(snapshot.repo_root),
                providers=providers,
                config=BuildConfig.empty(),
                settings=BuildSettings(
                    engine_version="test",
                    export_audit=ExportAuditSettings(),
                ),
                execution_settings=HamiltonExecutionSettings(),
            )
            config = env.variants.as_hamilton_config()
            config["variant_fingerprint"] = env.variants.variant_fingerprint
            compose_runtime(env=env, config=config)
        finally:
            gateway.close()


def seed_contract_catalog(con: DuckDBConnection) -> None:
    """Seed the canonical dataset contract catalog into a DuckDB connection."""
    _ensure_contract_service_configured()
    persist_contract_catalog_to_connection(
        con,
        inputs={"source": "tests"},
    )


def seed_tables(gateway: StorageGateway, ddl: list[str]) -> None:
    """Apply defensive DDL statements (DROP/CREATE) to avoid cross-test conflicts."""
    for stmt in ddl:
        gateway.con.execute(stmt)


def seed_repo_identity(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
    modules: dict[str, str] | None = None,
    repo_root: Path | None = None,
) -> None:
    """
    Insert a repo identity row for serving-layer verification.

    Parameters
    ----------
    gateway
        Target gateway with an applied schema.
    repo
        Repository slug to record.
    commit
        Commit hash to record.
    modules
        Optional module->path mappings to persist alongside identity.
    repo_root
        Optional repo root to derive module mappings when not provided.
    """
    modules_payload = modules
    if modules_payload is None and repo_root is not None:
        path_map = modules_expected_from_repo_tree(repo_root)
        modules_payload = {module: path for path, module in path_map.items()}
    if modules_payload is None:
        modules_payload = {}
    gateway.con.execute(
        "DELETE FROM core.repo_map WHERE repo = ? AND commit = ?",
        [repo, commit],
    )
    gateway.con.execute(
        """
        INSERT INTO core.repo_map (repo, commit, modules, overlays, generated_at)
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
        [repo, commit, modules_payload, {}],
    )
    if modules_payload:
        gateway.con.execute(
            "DELETE FROM core.modules WHERE repo = ? AND commit = ?",
            [repo, commit],
        )
        gateway.con.executemany(
            """
            INSERT INTO core.modules (module, path, repo, commit, language, tags, owners)
            VALUES (?, ?, ?, ?, 'python', ?, ?)
            """,
            [(module, path, repo, commit, [], []) for module, path in modules_payload.items()],
        )
        snapshot = SnapshotRef(repo=repo, commit=commit, repo_root=repo_root or Path.cwd())
        ModulesAssertions(gateway, snapshot).inventory_consistent()


__all__ = [
    "DuckDBConnection",
    "GatewayFactory",
    "seed_contract_catalog",
    "seed_repo_identity",
    "seed_tables",
]
