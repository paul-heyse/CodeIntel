"""Handlers for metadata catalog commands."""

from __future__ import annotations

from typing import TYPE_CHECKING

from codeintel.cli.core import CliResult
from codeintel.cli.errors.results import (
    fail_execution_failed,
    fail_invalid_value,
    fail_not_found,
)
from codeintel.cli.handlers.metadata_bundle import BundleIngestRequest, ingest_metadata_bundle

if TYPE_CHECKING:
    from codeintel.cli.context import CommandContext


def meta_sync_handler(ctx: CommandContext) -> CliResult[dict[str, object]]:
    """Ingest build metadata bundles into the meta catalog.

    Parameters
    ----------
    ctx
        Command context with runtime access.

    Returns
    -------
    CliResult[dict[str, object]]
        Summary payload for the ingest operation.
    """
    if not ctx.has_runtime:
        return fail_execution_failed("meta", "meta.sync requires runtime access")

    snapshot = ctx.runtime.snapshot
    if not snapshot.repo or not snapshot.commit:
        return fail_execution_failed("meta", "repo and commit must be set for meta.sync")
    bundle_root = ctx.params.get_path("bundle_root") or ctx.runtime.paths.build_dir / "metadata"
    if not bundle_root.exists():
        return fail_not_found(
            "bundle_root",
            str(bundle_root),
            suggestion="Run `codeintel build run --all` to generate build metadata",
        )

    try:
        outcome = ingest_metadata_bundle(
            BundleIngestRequest(
                bundle_root=bundle_root,
                db_path=ctx.runtime.db_path,
                repo=snapshot.repo,
                commit=snapshot.commit,
                catalog_inputs={"source": "meta.sync"},
            )
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        return fail_execution_failed("meta", str(exc), status=500)

    payload = outcome.report.to_payload()
    payload.update(
        {
            "run_id": outcome.run_id,
            "repo": snapshot.repo,
            "commit": snapshot.commit,
            "bundle_root": str(bundle_root),
            "bundle_manifest_repo": outcome.bundle_manifest.repo,
            "bundle_manifest_commit": outcome.bundle_manifest.commit,
            "bundle_manifest_run_id": outcome.bundle_manifest.run_id,
            "renderer_cache_backfill_rows": outcome.renderer_cache_rows,
        }
    )
    if outcome.override_refresh is None:
        payload.update(
            {
                "override_refresh_status": "skipped",
                "override_refresh_reason": "schema_manifest_missing",
                "override_refresh_version_id": None,
                "override_refresh_tables": 0,
                "override_refresh_schema_versions_rows": 0,
                "override_refresh_versions_rows": 0,
                "override_refresh_registry_rows": 0,
            }
        )
    else:
        payload.update(
            {
                "override_refresh_status": outcome.override_refresh.status,
                "override_refresh_reason": outcome.override_refresh.reason,
                "override_refresh_version_id": outcome.override_refresh.version_id,
                "override_refresh_tables": outcome.override_refresh.tables,
                "override_refresh_schema_versions_rows": outcome.override_refresh.schema_versions_rows,
                "override_refresh_versions_rows": outcome.override_refresh.override_versions_rows,
                "override_refresh_registry_rows": outcome.override_refresh.override_registry_rows,
            }
        )
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


def meta_drift_report_handler(ctx: CommandContext) -> CliResult[dict[str, object]]:
    """Return latest schema drift summaries from schema observations.

    Parameters
    ----------
    ctx
        CLI command context with gateway access.

    Returns
    -------
    CliResult[dict[str, object]]
        Drift summary payload for recent observations.
    """
    if not ctx.has_storage:
        return fail_execution_failed("meta", "meta.drift requires storage access")
    limit = ctx.params.get_int("limit", 50)
    try:
        payload = ctx.gateway.schemas.drift_summary_report(limit=limit)
    except (RuntimeError, TypeError, ValueError) as exc:
        return fail_execution_failed("meta", str(exc), status=500)
    return CliResult.ok(payload)
