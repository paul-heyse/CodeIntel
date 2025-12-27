"""Handlers for metadata catalog commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.build.meta.contract_catalog import persist_contract_catalog
from codeintel.build.schemas import get_schema_provider
from codeintel.build.schemas.compile import SchemaManifestRequest, compile_schema_manifest
from codeintel.cli.core import CliResult
from codeintel.cli.errors.results import fail_execution_failed
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
            request=request,
            con=ctx.gateway.con,
        )
        run_id = new_run_id("meta")
        schema_result = ctx.gateway.schemas.persist_schema_manifest(
            manifest,
            request=SchemaCatalogRequest(
                run_id=run_id,
                repo=snapshot.repo,
                commit=snapshot.commit,
                catalog_inputs={"source": "meta.sync"},
            ),
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
        "contract_catalog_hash": contract_result.catalog_hash,
        "contract_count": contract_result.contract_count,
    }
    return CliResult.ok(payload)
