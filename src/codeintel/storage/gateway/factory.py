"""Factory functions for creating StorageGateway instances."""

from __future__ import annotations

from pathlib import Path

from codeintel.storage.datasets.registry import load_dataset_registry
from codeintel.storage.gateway.accessors import DuckDBGateway
from codeintel.storage.gateway.config import StorageConfig
from codeintel.storage.gateway.connection import (
    _apply_schema_and_views,
    _ensure_macros_and_schema,
    connect,
)
from codeintel.storage.gateway.protocol import SnapshotGatewayResolver, StorageGateway
from codeintel.storage.metadata import bootstrap_metadata_datasets
from codeintel.storage.validation import validate_contract_or_raise

__all__ = [
    "build_snapshot_gateway_resolver",
    "open_gateway",
    "open_memory_gateway",
]


def open_gateway(config: StorageConfig) -> StorageGateway:
    """
    Create a StorageGateway bound to a DuckDB database.

    Parameters
    ----------
    config
        Storage configuration describing connection options.

    Returns
    -------
    StorageGateway
        Gateway exposing typed accessors and dataset registry.
    """
    con = connect(config)
    if not config.read_only:
        _apply_schema_and_views(con, config)
        _ensure_macros_and_schema(con, config)
        bootstrap_metadata_datasets(con)
    datasets = load_dataset_registry(con)
    validate_contract_or_raise(con)
    return DuckDBGateway(config=config, datasets=datasets, con=con)


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
    cfg = StorageConfig(
        db_path=Path(":memory:"),
        read_only=False,
        apply_schema=apply_schema,
        ensure_views=ensure_views,
        validate_schema=validate_schema,
        repo=repo,
        commit=commit,
    )
    return open_gateway(cfg)
