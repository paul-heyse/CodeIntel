"""Serving-layer introspection payload assembly.

This module consolidates snapshot + environment + feature/limit reporting for
both HTTP and FastMCP surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from codeintel.serving.meta.tooling import runtime_versions, tooling_mismatch_warnings
from codeintel.serving.operations.ops import ServingOperations
from codeintel.serving.settings import ServingSettings

if TYPE_CHECKING:
    from codeintel.serving.db.manager import ServingDBManager


def build_resource_templates_payload(
    ops: ServingOperations,
    *,
    templates: object,
    generated_at: datetime,
) -> dict[str, object]:
    """Build the canonical resource-templates payload for discovery resources.

    Returns
    -------
    dict[str, object]
        Serialized payload for resource template discovery.
    """
    pointer = ops.db.current_pointer()
    return {
        "uri": "codeintel://meta/resources",
        "generated_at": generated_at.isoformat(),
        "snapshot": {
            "repo": pointer.repo,
            "commit": pointer.commit,
            "run_id": pointer.run_id,
            "published_at": pointer.published_at.isoformat(),
        },
        "templates": templates,
    }


def build_kernel_meta_payload(db: ServingDBManager) -> dict[str, object]:
    """Build the canonical serving meta payload used by HTTP `/meta` and MCP resources.

    This is a refactor of the legacy `SemanticQueryKernel.meta()` implementation into
    a reusable service function to keep adapters thin.

    Returns
    -------
    dict[str, object]
        Serialized kernel metadata payload.
    """
    pointer = db.current_pointer()
    context = db.snapshot_context(pointer)
    registry = context.registry
    spec = context.buildspec
    env_meta = context.environment or {}

    tables = sum(1 for d in spec.datasets if not d.table_key.startswith("docs.v_"))
    views = sum(1 for d in spec.datasets if d.table_key.startswith("docs.v_"))

    return {
        "repo": pointer.repo,
        "commit": pointer.commit,
        "run_id": pointer.run_id,
        "published_at": pointer.published_at.isoformat(),
        "semantic_layer_version": pointer.semantic_layer_version,
        "buildspec_hash": spec.buildspec_hash,
        "buildspec_version": spec.spec_version,
        "duckdb": {"db_path": str(pointer.db_path), "read_only": True},
        "environment": env_meta,
        "semantic_views": [
            {"id": v.id, "table_key": v.table_key, "entity": v.entity, "grain": v.grain}
            for v in registry.views
            if not v.deprecated
        ],
        "datasets": [
            {"table_key": dataset.table_key, "schema_hash": dataset.schema_hash}
            for dataset in spec.datasets
        ],
        "targets": [
            {
                "name": t.name,
                "domain": t.domain,
                "impl_kind": t.impl_kind,
                "deps": list(t.deps),
                "outputs": list(t.outputs),
                "artifacts": [
                    {"name": artifact.name, "kind": artifact.kind} for artifact in t.artifacts
                ],
            }
            for t in spec.targets
        ],
        "schema_inventory": {"tables": tables, "views": views},
    }


def build_environment_meta_payload(
    ops: ServingOperations,
    *,
    settings: ServingSettings,
) -> dict[str, object]:
    """Build environment/tooling payload for `codeintel://meta/environment`.

    Returns
    -------
    dict[str, object]
        Environment payload including runtime versions and export limits.
    """
    meta = ops.meta()
    env_obj = meta.get("environment")
    environment = env_obj if isinstance(env_obj, dict) else {}
    pointer = ops.db.current_pointer()
    runtime = runtime_versions()
    return {
        "snapshot": {
            "repo": pointer.repo,
            "commit": pointer.commit,
            "run_id": pointer.run_id,
            "published_at": pointer.published_at.isoformat(),
        },
        "environment": environment,
        "runtime_versions": runtime,
        "warnings": list(tooling_mismatch_warnings(environment, runtime=runtime)),
        "mcp_export_limits": {
            "max_full_read_bytes": settings.mcp_export_max_full_read_bytes,
            "max_chunk_bytes": settings.mcp_export_max_chunk_bytes,
            "max_chunk_lines": settings.mcp_export_max_chunk_lines,
            "ttl_seconds": settings.mcp_export_ttl_seconds,
        },
    }


@dataclass(frozen=True, slots=True)
class ServingMetaExtras:
    """Extra data needed to build the MCP `serving_meta` payload."""

    features: dict[str, bool]
    inventories: dict[str, int]
    resource_templates: object


def build_serving_meta_payload(
    ops: ServingOperations,
    *,
    settings: ServingSettings,
    started_at: datetime,
    extras: ServingMetaExtras,
) -> dict[str, object]:
    """Build the canonical payload consumed by the FastMCP `serving_meta` tool.

    Returns
    -------
    dict[str, object]
        Serving meta payload combining snapshot info, features, and limits.
    """
    pointer = ops.db.current_pointer()
    meta = ops.meta()

    env_obj = meta.get("environment")
    environment = env_obj if isinstance(env_obj, dict) else {}
    runtime = runtime_versions()
    warnings = tooling_mismatch_warnings(environment, runtime=runtime)

    return {
        "server_version": runtime.get("codeintel", "not-installed"),
        "started_at": started_at.isoformat(),
        "snapshot": {
            "repo": pointer.repo,
            "commit": pointer.commit,
            "run_id": pointer.run_id,
            "published_at": pointer.published_at.isoformat(),
        },
        "semantic_layer": {
            "version": str(meta.get("semantic_layer_version", "unknown")),
            "hash": str(meta.get("semantic_layer_hash", "unknown")),
            "view_count": extras.inventories.get("views", 0),
            "schema_manifest_hash": str(meta.get("schema_manifest_hash"))
            if meta.get("schema_manifest_hash") is not None
            else None,
        },
        "buildspec": {
            "version": str(meta.get("buildspec_version", "unknown")),
            "hash": str(meta.get("buildspec_hash", "unknown")),
            "compiled_at": pointer.published_at.isoformat(),
        },
        "read_only": True,
        "features": extras.features,
        "limits": {
            "export_max_rows": settings.export_max_rows,
            "export_ttl_seconds": settings.mcp_export_ttl_seconds,
        },
        "resource_templates": extras.resource_templates,
        "inventories": extras.inventories,
        "warnings": list(warnings),
    }


__all__ = [
    "ServingMetaExtras",
    "build_environment_meta_payload",
    "build_kernel_meta_payload",
    "build_resource_templates_payload",
    "build_serving_meta_payload",
]
