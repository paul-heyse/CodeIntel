"""Handlers for metadata catalog commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.build.meta.contract_catalog import persist_contract_catalog
from codeintel.build.schemas import get_schema_provider
from codeintel.build.schemas.compile import (
    SchemaManifestContext,
    SchemaManifestRequest,
    compile_schema_manifest,
)
from codeintel.cli.core import CliResult
from codeintel.cli.errors.results import (
    fail_execution_failed,
    fail_invalid_value,
    fail_not_found,
)
from codeintel.cli.handlers.runtime_helpers import compose_cli_runtime_bundle
from codeintel.core.execution.ids import new_run_id
from codeintel.storage.tracking.schema_catalog import SchemaCatalogRequest

if TYPE_CHECKING:
    from codeintel.cli.context import CommandContext


def meta_sync_handler(ctx: CommandContext) -> CliResult[dict[str, object]]:
    """Regenerate and persist canonical meta catalogs.

    Parameters
    ----------
    ctx
        Command context with runtime and storage access.

    Returns
    -------
    CliResult[dict[str, object]]
        Summary payload for the sync operation.
    """
    if not ctx.has_runtime or not ctx.has_storage:
        return fail_execution_failed("meta", "meta.sync requires runtime and storage access")

    snapshot = ctx.runtime.snapshot
    if not snapshot.repo or not snapshot.commit:
        return fail_execution_failed("meta", "repo and commit must be set for meta.sync")

    runtime_bundle = compose_cli_runtime_bundle(runtime=ctx.runtime, gateway=ctx.gateway)
    schema_index = runtime_bundle.schema_index
    if schema_index is None:
        return fail_execution_failed("meta", "Runtime schema_index is required")
    schema_provider = get_schema_provider()
    request = SchemaManifestRequest(
        all_targets=True,
        stable=True,
        version="v2",
        include_views=True,
        include_artifacts=True,
        include_provenance=True,
    )

    try:
        manifest = compile_schema_manifest(
            provider=schema_provider,
            context=SchemaManifestContext(
                catalog=runtime_bundle.catalog,
                schema_index=schema_index,
                tag_query=runtime_bundle.tag_query,
            ),
            request=request,
            con=ctx.gateway.con,
        )
        run_id = new_run_id("meta")
        catalog_request = SchemaCatalogRequest(
            run_id=run_id,
            repo=snapshot.repo,
            commit=snapshot.commit,
            catalog_inputs={"source": "meta.sync"},
        )
        schema_result = ctx.gateway.schemas.persist_schema_manifest(
            manifest,
            request=catalog_request,
        )
        override_result = ctx.gateway.schemas.refresh_override_registry_from_manifest(
            manifest,
            request=catalog_request,
            catalog_hash=schema_result.catalog_hash,
        )
        contract_result = persist_contract_catalog(
            ctx.gateway,
            inputs={"source": "meta.sync"},
        )
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        return fail_execution_failed("meta", str(exc), status=500)

    payload: dict[str, object] = {
        "run_id": run_id,
        "repo": snapshot.repo,
        "commit": snapshot.commit,
        "schema_manifest_hash": schema_result.catalog_hash,
        "schema_manifest_tables": schema_result.tables,
        "schema_manifest_views": schema_result.views,
        "schema_versions_rows": schema_result.schema_versions_rows,
        "table_schema_registry_rows": schema_result.table_schema_registry_rows,
        "schema_manifest_runs_rows": schema_result.schema_manifest_runs_rows,
        "override_refresh_status": override_result.status,
        "override_refresh_reason": override_result.reason,
        "override_refresh_version_id": override_result.version_id,
        "override_refresh_tables": override_result.tables,
        "override_refresh_schema_versions_rows": override_result.schema_versions_rows,
        "override_refresh_versions_rows": override_result.override_versions_rows,
        "override_refresh_registry_rows": override_result.override_registry_rows,
        "contract_catalog_hash": contract_result.catalog_hash,
        "contract_count": contract_result.contract_count,
    }
    return CliResult.ok(payload)


def meta_override_pin_handler(ctx: CommandContext) -> CliResult[dict[str, object]]:
    """Pin the override registry to a specific schema version.

    Returns
    -------
    CliResult[dict[str, object]]
        Result payload for the updated override registry record.
    """
    if not ctx.has_storage:
        return fail_execution_failed("meta", "meta.override-pin requires storage access")
    table_key = ctx.params.get_str("table_key")
    if not table_key:
        return fail_invalid_value(
            "table_key",
            table_key,
            "table_key is required",
            suggestion="Provide --table-key",
        )
    schema_digest = ctx.params.get_str("schema_digest")
    version_id = ctx.params.get_str("version_id")
    if schema_digest is None and version_id is None:
        return fail_invalid_value(
            "schema_digest",
            None,
            "schema_digest or version_id must be provided",
            suggestion="Provide --schema-digest or --version-id",
        )

    try:
        record = ctx.gateway.schemas.set_override_registry_version(
            table_key=table_key,
            schema_digest=schema_digest,
            version_id=version_id,
        )
    except KeyError:
        return fail_not_found(
            "override_version",
            table_key,
            suggestion="Verify table_key and schema_digest/version_id",
        )
    except (RuntimeError, ValueError) as exc:
        return fail_execution_failed("meta", str(exc), status=500)

    payload: dict[str, object] = {
        "table_key": record.table_key,
        "schema_digest": record.schema_digest,
        "schema_hash": record.schema_hash,
        "version_id": record.version_id,
        "updated_at": record.updated_at.isoformat() if record.updated_at is not None else None,
    }
    return CliResult.ok(payload)


def meta_registry_health_handler(ctx: CommandContext) -> CliResult[dict[str, object]]:
    """Return schema registry health diagnostics for the connected gateway.

    Returns
    -------
    CliResult[dict[str, object]]
        Health snapshot payload for the schema registry.
    """
    if not ctx.has_storage:
        return fail_execution_failed("meta", "meta.health requires storage access")
    try:
        payload = ctx.gateway.schemas.registry_health_snapshot()
    except (RuntimeError, TypeError, ValueError) as exc:
        return fail_execution_failed("meta", str(exc), status=500)
    return CliResult.ok(payload)
