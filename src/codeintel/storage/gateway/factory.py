"""Factory functions for creating StorageGateway instances."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import duckdb

from codeintel.core.errors.storage import StorageConnectionError
from codeintel.core.schemas import MappingSchemaProvider, SchemaService
from codeintel.core.schemas.service import get_schema_service, set_schema_service
from codeintel.storage.backend import DuckDBSession
from codeintel.storage.contracts.catalog_state import contract_catalog_table_schemas
from codeintel.storage.contracts.provider import load_contract_catalog_from_connection
from codeintel.storage.datasets.registry import load_dataset_registry
from codeintel.storage.gateway.accessors import DuckDBGateway
from codeintel.storage.gateway.config import StorageConfig
from codeintel.storage.gateway.inference import InferenceGateway
from codeintel.storage.metadata import bootstrap_metadata_datasets
from codeintel.storage.schema import assert_schema_alignment
from codeintel.storage.validation import validate_contract_or_raise

if TYPE_CHECKING:
    from codeintel.core.schemas.provider import SchemaProvider
    from codeintel.storage.gateway.protocol import SnapshotGatewayResolver, StorageGateway

__all__ = [
    "build_snapshot_gateway_resolver",
    "open_gateway",
    "open_inference_gateway",
    "open_memory_gateway",
]


def _maybe_set_schema_service_from_catalog() -> None:
    try:
        get_schema_service()
    except RuntimeError:
        schemas = contract_catalog_table_schemas()
        if schemas:
            service = SchemaService(table_provider=MappingSchemaProvider(schemas))
            set_schema_service(service)


def open_gateway(config: StorageConfig) -> StorageGateway:
    """Create a StorageGateway bound to a DuckDB database.

    Parameters
    ----------
    config
        Storage configuration describing connection options.

    Returns
    -------
    StorageGateway
        Gateway exposing typed accessors and dataset registry.

    Raises
    ------
    StorageConnectionError
        If the database connection cannot be established.
    """
    try:
        session = DuckDBSession(config)
        con = session.open_reader() if config.read_only else session.open()
        if not config.read_only:
            include_views = config.ensure_views and config.apply_schema
            bootstrap_metadata_datasets(
                con,
                include_views=include_views,
                validate_schema_registry=config.validate_schema,
            )
        load_contract_catalog_from_connection(con)
        _maybe_set_schema_service_from_catalog()
        datasets = load_dataset_registry(con)
        gateway = DuckDBGateway(config=config, datasets=datasets, con=con)
        if config.ensure_views and not config.read_only:
            gateway.policy.ensure_all_views(overwrite=True, strict=config.validate_schema)
        if config.validate_schema:
            assert_schema_alignment(
                con,
                include_views=config.ensure_views and not config.read_only,
                strict=True,
            )
            validate_contract_or_raise(
                con,
                include_views=config.ensure_views and not config.read_only,
            )
    except duckdb.Error as exc:
        raise StorageConnectionError(str(exc), cause=exc) from exc
    return gateway


def open_inference_gateway(*, schema_provider: SchemaProvider) -> InferenceGateway:
    """Create a minimal in-memory gateway for schema inference.

    This bypasses metadata bootstrap and contract catalog loading, providing a
    lightweight DuckDB connection with a policy backend seeded by the supplied
    schema provider.

    Parameters
    ----------
    schema_provider
        Schema provider used for DDL and column-order enforcement.

    Returns
    -------
    MinimalStorageGateway
        Minimal gateway backed by an in-memory DuckDB connection.
    """
    con = duckdb.connect(":memory:")
    return InferenceGateway(con=con, schema_provider=schema_provider)


def build_snapshot_gateway_resolver(
    *,
    db_dir: Path,
    repo: str | None = None,
    primary_gateway: StorageGateway | None = None,
) -> SnapshotGatewayResolver:
    """
    Build a resolver that opens per-commit snapshot databases as StorageGateways.

    Parameters
    ----------
    db_dir:
        Directory containing per-commit DuckDB snapshots, named
        ``codeintel-<commit>.duckdb``.
    repo:
        Optional repository slug to record in the StorageConfig for observability.
    primary_gateway:
        Optional gateway to reuse when the requested commit resolves to the same
        database path, avoiding duplicate connections with conflicting settings.

    Returns
    -------
    SnapshotGatewayResolver
        Callable that returns a read-only StorageGateway for the given commit.
    """

    def _resolve(commit: str) -> StorageGateway:
        db_path = db_dir / f"codeintel-{commit}.duckdb"
        if (
            primary_gateway is not None
            and db_path.resolve() == primary_gateway.config.db_path.resolve()
        ):
            return primary_gateway
        if not db_path.is_file():
            message = f"Missing snapshot database for commit {commit}: {db_path}"
            raise FileNotFoundError(message)
        cfg = StorageConfig(
            db_path=db_path,
            read_only=True,
            apply_schema=False,
            ensure_views=False,
            validate_schema=False,
            repo=repo,
            commit=commit,
        )
        return open_gateway(cfg)

    return _resolve


def open_memory_gateway(
    *,
    apply_schema: bool = True,
    ensure_views: bool = False,
    validate_schema: bool = True,
    repo: str | None = None,
    commit: str | None = None,
) -> StorageGateway:
    """
    Create an in-memory StorageGateway for tests.

    Parameters
    ----------
    apply_schema
        When True, apply all table schemas to the in-memory database.
    ensure_views
        When True, create docs views after schema application.
    validate_schema
        When True, validate schema alignment after setup.
    repo
        Optional repository slug to record in the StorageConfig for observability.
    commit
        Optional commit hash to record in the StorageConfig for observability.

    Returns
    -------
    StorageGateway
        Gateway backed by an in-memory DuckDB connection.
    """
    repo_value = repo if repo is not None else "demo/repo"
    commit_value = commit if commit is not None else "deadbeef"
    cfg = StorageConfig(
        db_path=Path(":memory:"),
        read_only=False,
        apply_schema=apply_schema,
        ensure_views=ensure_views,
        validate_schema=validate_schema,
        repo=repo_value,
        commit=commit_value,
    )
    return open_gateway(cfg)
