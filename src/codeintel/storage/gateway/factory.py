"""Factory functions for creating StorageGateway instances."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb

from codeintel.core.errors.storage import StorageConnectionError
from codeintel.core.schemas import MappingSchemaProvider, SchemaService
from codeintel.core.schemas.provider import FallbackSchemaProvider
from codeintel.core.schemas.service import get_schema_service, set_schema_service
from codeintel.storage.backend import DuckDBSession
from codeintel.storage.constants import META_CATALOG_NAME
from codeintel.storage.contracts.catalog_state import (
    contract_catalog_table_schemas,
    get_contract_catalog,
)
from codeintel.storage.contracts.provider import load_contract_catalog_from_connection
from codeintel.storage.contracts.schema_provider import clear_schema_provider_cache
from codeintel.storage.datasets.registry import load_dataset_registry
from codeintel.storage.gateway.accessors import DuckDBGateway
from codeintel.storage.gateway.config import StorageConfig
from codeintel.storage.gateway.inference import InferenceGateway
from codeintel.storage.metadata import (
    SchemaValidationRun,
    bootstrap_metadata_datasets,
    record_schema_validation_run,
)
from codeintel.storage.metadata.ddl import apply_metadata_ddl
from codeintel.storage.metadata.meta_catalog import attach_meta_database, meta_table_ref
from codeintel.storage.schema import apply_all_schemas, assert_schema_alignment
from codeintel.storage.schema.arrow_schema import RegistryArrowSchemaProvider
from codeintel.storage.tracking.schema_catalog import SchemaCatalogProvider
from codeintel.storage.validation import (
    ContractValidationMode,
    clear_contract_validation_cache,
    collect_contract_issues,
    collect_contract_issues_lenient,
    validate_contract_or_raise,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from codeintel.core.schemas.provider import SchemaProvider
    from codeintel.storage.gateway.protocol import SnapshotGatewayResolver, StorageGateway

    CatalogSeeder = Callable[[duckdb.DuckDBPyConnection], None]

__all__ = [
    "MemoryGatewayOptions",
    "build_snapshot_gateway_resolver",
    "open_gateway",
    "open_inference_gateway",
    "open_memory_gateway",
]

LOG = logging.getLogger(__name__)
ISSUE_PREVIEW_LIMIT = 10


@dataclass(frozen=True, slots=True)
class MemoryGatewayOptions:
    """Options for configuring an in-memory gateway."""

    apply_schema: bool = True
    ensure_views: bool = False
    validate_schema: bool = True
    suppress_registry_health_log: bool = False
    repo: str | None = None
    commit: str | None = None


def _registry_has_schemas(con: duckdb.DuckDBPyConnection) -> bool:
    registry_ref = meta_table_ref("metadata.table_schema_registry")
    filter_clause = (
        "AND ("
        "registry.inference_status IN ('inferred', 'override') "
        "OR registry.derivation_kind IN ('inferred_relation', 'view_inferred')"
        ")"
    )
    try:
        row = con.execute(
            f"""
            SELECT 1
            FROM {registry_ref} AS registry
            WHERE 1 = 1
            {filter_clause}
            LIMIT 1
            """
        ).fetchone()
    except (duckdb.Error, RuntimeError, TypeError, ValueError):
        return False
    return row is not None


def _is_catalog_provider(provider: SchemaProvider) -> bool:
    if isinstance(provider, SchemaCatalogProvider):
        return True
    if isinstance(provider, FallbackSchemaProvider):
        return isinstance(provider.primary, SchemaCatalogProvider)
    return False


def _maybe_set_schema_service_from_catalog(con: duckdb.DuckDBPyConnection) -> None:
    try:
        service = get_schema_service()
    except RuntimeError:
        service = None
    schemas = contract_catalog_table_schemas()
    fallback_provider = MappingSchemaProvider(schemas) if schemas else None
    if _registry_has_schemas(con):
        provider: SchemaProvider
        catalog_provider = SchemaCatalogProvider(con)
        if fallback_provider is not None:
            provider = FallbackSchemaProvider(
                primary=catalog_provider,
                fallback=fallback_provider,
            )
        else:
            provider = catalog_provider
        if service is not None and _is_catalog_provider(service.table_provider):
            return
        service = SchemaService(
            table_provider=provider,
            arrow_provider=RegistryArrowSchemaProvider(con),
        )
        set_schema_service(service)
        clear_schema_provider_cache()
        return
    if service is not None:
        return
    if fallback_provider is not None:
        service = SchemaService(table_provider=fallback_provider)
        set_schema_service(service)
        clear_schema_provider_cache()


def _schema_service_mismatches() -> list[str]:
    try:
        schema_service = get_schema_service()
    except RuntimeError:
        return []
    catalog = get_contract_catalog()
    if catalog is None:
        return []
    mismatches: list[str] = []
    for table_key, contract in catalog.items():
        if contract.is_view or table_key.startswith("tmp_") or contract.schema is None:
            continue
        schema = schema_service.get_table_schema(table_key)
        if schema is None or schema != contract.schema:
            mismatches.append(table_key)
    return mismatches


def _schema_service_available() -> bool:
    try:
        get_schema_service()
    except RuntimeError:
        return False
    return True


def _ensure_contract_catalog(con: duckdb.DuckDBPyConnection) -> None:
    load_contract_catalog_from_connection(con)
    clear_contract_validation_cache()
    catalog = get_contract_catalog()
    if catalog is None:
        msg = (
            "Contract catalog missing. Run `codeintel meta sync` to populate "
            "metadata.canonical_catalogs before opening a storage gateway."
        )
        raise RuntimeError(msg)

    mismatches = _schema_service_mismatches()
    if mismatches:
        if _schema_service_available():
            LOG.warning(
                "Contract catalog mismatch detected for %d tables; "
                "runtime schema service will be preferred",
                len(mismatches),
            )
        else:
            LOG.warning(
                "Contract catalog mismatch detected for %d tables; using stored catalog",
                len(mismatches),
            )
    clear_schema_provider_cache()
    clear_contract_validation_cache()


def _write_contract_validation_summary(
    *,
    issues: list[str],
    config: StorageConfig,
    include_views: bool,
) -> None:
    summary_path = config.validation_summary_path
    if summary_path is None or not issues:
        return
    payload = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "validation_mode": config.validation_mode.value,
        "db_path": str(config.db_path),
        "repo": config.repo,
        "commit": config.commit,
        "include_views": include_views,
        "issue_count": len(issues),
        "issues": issues,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _log_contract_issues(issues: list[str]) -> None:
    if not issues:
        return
    preview = "; ".join(issues[:ISSUE_PREVIEW_LIMIT])
    remaining = len(issues) - ISSUE_PREVIEW_LIMIT
    suffix = f" (+{remaining} more)" if remaining > 0 else ""
    LOG.warning("Contract validation warnings: %s%s", preview, suffix)


def _apply_contract_validation(
    *,
    con: duckdb.DuckDBPyConnection,
    config: StorageConfig,
    include_views: bool,
) -> None:
    if not config.validate_schema:
        return
    if config.validation_mode == ContractValidationMode.OFF:
        return

    drift_issues = assert_schema_alignment(
        con,
        include_views=include_views,
        strict=False,
    )
    if config.validation_mode == ContractValidationMode.LENIENT:
        contract_issues = collect_contract_issues_lenient(con, include_views=include_views)
    else:
        contract_issues = collect_contract_issues(
            con, include_views=include_views, missing_ok=False
        )
    issues = [*drift_issues, *contract_issues]

    if config.attach_meta and not config.read_only:
        validation_run = SchemaValidationRun(
            repo=config.repo,
            commit=config.commit,
            validation_mode=config.validation_mode.value,
            include_views=include_views,
            issues=issues,
        )
        record_schema_validation_run(con, validation_run)

    if config.validation_mode == ContractValidationMode.STRICT:
        if issues:
            assert_schema_alignment(con, include_views=include_views, strict=True)
            validate_contract_or_raise(con, include_views=include_views)
        return

    _log_contract_issues(issues)
    _write_contract_validation_summary(
        issues=issues,
        config=config,
        include_views=include_views,
    )


def _log_registry_health(gateway: StorageGateway) -> None:
    try:
        snapshot = gateway.schemas.registry_health_snapshot()
    except (RuntimeError, TypeError, ValueError, duckdb.Error):
        return
    if snapshot.get("registry_stale"):
        LOG.warning(
            "Schema registry appears stale (latest_manifest=%s)",
            snapshot.get("latest_manifest"),
        )
    drift = snapshot.get("contract_drift")
    if not isinstance(drift, dict):
        return
    missing_contracts = int(drift.get("missing_contracts", 0) or 0)
    missing_metadata = int(drift.get("missing_contract_metadata", 0) or 0)
    hash_mismatches = int(drift.get("hash_mismatches", 0) or 0)
    digest_mismatches = int(drift.get("digest_mismatches", 0) or 0)
    if not any((missing_contracts, missing_metadata, hash_mismatches, digest_mismatches)):
        return
    samples = drift.get("mismatch_samples") or drift.get("missing_contract_samples") or []
    preview = ", ".join(samples[:ISSUE_PREVIEW_LIMIT]) if samples else ""
    LOG.warning(
        "Arrow contract drift detected (missing=%d missing_metadata=%d hash=%d digest=%d)%s",
        missing_contracts,
        missing_metadata,
        hash_mismatches,
        digest_mismatches,
        f": {preview}" if preview else "",
    )


def open_gateway(
    config: StorageConfig,
    *,
    seed_contract_catalog: CatalogSeeder | None = None,
) -> StorageGateway:
    """Create a StorageGateway bound to a DuckDB database.

    Parameters
    ----------
    config
        Storage configuration describing connection options.
    seed_contract_catalog
        Callback to seed the contract catalog before opening the gateway.
        Required for in-memory gateways because metadata catalogs start empty.

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
        session_config = config
        include_views_for_bootstrap = True
        if config.ensure_views:
            LOG.warning("DuckDB view materialization disabled; ignoring ensure_views=True.")
        if not config.read_only and config.apply_schema:
            session_config = replace(config, apply_schema=False)
        session = DuckDBSession(session_config)
        con = session.open_reader() if config.read_only else session.open()
        attach_meta_database(con, config=config)
        if not config.read_only:
            apply_metadata_ddl(con, catalog=META_CATALOG_NAME)
            if seed_contract_catalog is not None:
                seed_contract_catalog(con)
        _ensure_contract_catalog(con)
        if not config.read_only and config.apply_schema:
            apply_all_schemas(con)
        if not config.read_only:
            bootstrap_metadata_datasets(
                con,
                include_views=include_views_for_bootstrap,
            )
        _maybe_set_schema_service_from_catalog(con)
        datasets = load_dataset_registry(con)
        gateway = DuckDBGateway(config=config, datasets=datasets, con=con)
        include_views = False
        _apply_contract_validation(
            con=con,
            config=config,
            include_views=include_views,
        )
        if not config.suppress_registry_health_log:
            _log_registry_health(gateway)
    except duckdb.Error as exc:
        raise StorageConnectionError(str(exc), cause=exc) from exc
    return gateway


def open_inference_gateway(*, schema_provider: SchemaProvider) -> InferenceGateway:
    """Create a minimal in-memory gateway for schema inference.

    This bypasses metadata bootstrap and contract catalog loading, providing a
    lightweight DuckDB connection with a policy backend seeded by the supplied
    schema provider. Metadata DDL is still applied to ensure meta tables exist.

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
    config = StorageConfig(
        db_path=Path(":memory:"),
        read_only=False,
        apply_schema=False,
        ensure_views=False,
        validate_schema=False,
    )
    attach_meta_database(con, config=config)
    apply_metadata_ddl(con, catalog=META_CATALOG_NAME)
    return InferenceGateway(con=con, schema_provider=schema_provider, config=config)


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
    options: MemoryGatewayOptions | None = None,
    seed_contract_catalog: CatalogSeeder | None = None,
) -> StorageGateway:
    """
    Create an in-memory StorageGateway for tests.

    Parameters
    ----------
    options
        Configuration options for schema application, view creation, validation,
        and repo/commit metadata.
    seed_contract_catalog
        Callback to seed the contract catalog before opening the gateway.

    Returns
    -------
    StorageGateway
        Gateway backed by an in-memory DuckDB connection.

    Raises
    ------
    RuntimeError
        If no seed_contract_catalog callback is provided.
    """
    if seed_contract_catalog is None:
        msg = (
            "open_memory_gateway requires seed_contract_catalog to populate "
            "metadata.canonical_catalogs. Pass an explicit seed callback or use "
            "open_inference_gateway for schema-only workflows."
        )
        raise RuntimeError(msg)
    resolved = options or MemoryGatewayOptions()
    repo_value = resolved.repo or "demo/repo"
    commit_value = resolved.commit or "deadbeef"
    cfg = StorageConfig(
        db_path=Path(":memory:"),
        read_only=False,
        apply_schema=resolved.apply_schema,
        ensure_views=resolved.ensure_views,
        validate_schema=resolved.validate_schema,
        suppress_registry_health_log=resolved.suppress_registry_health_log,
        repo=repo_value,
        commit=commit_value,
    )
    return open_gateway(cfg, seed_contract_catalog=seed_contract_catalog)
