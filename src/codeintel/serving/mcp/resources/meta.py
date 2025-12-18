"""FastMCP resources: semantic/meta discovery and snapshot metadata."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from codeintel.serving.errors import MetaArtifactNotFoundError, MetaSqlUnsafeError
from codeintel.serving.mcp.models import (
    DEFAULT_RESOURCE_TEMPLATES,
    ResourceTemplatesResponse,
)
from codeintel.serving.mcp.protocols import ServingSnapshotPointerProtocol
from codeintel.serving.meta.service import (
    build_environment_meta_payload,
    build_resource_templates_payload,
)
from codeintel.serving.semantic.models import (
    SemanticCatalogResponse,
    SemanticViewDescriptionResponse,
)
from codeintel.storage.queries.safe import UnsafeSqlError, assert_single_select_statement

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.serving.mcp._compat import FastMCP
    from codeintel.serving.operations.ops import ServingOperations
    from codeintel.serving.settings import ServingSettings

LOG = logging.getLogger(__name__)


def _read_json_file(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        msg = f"Expected JSON object in {path.name}"
        raise TypeError(msg)
    return {str(k): v for k, v in raw.items()}


def _artifact_dir_for_pointer(pointer: ServingSnapshotPointerProtocol) -> Path:
    return pointer.schema_manifest_path.parent


def _read_views_sql(pointer: ServingSnapshotPointerProtocol) -> dict[str, object]:
    artifact_name = "views_sql.json"
    path = _artifact_dir_for_pointer(pointer) / artifact_name
    if not path.is_file():
        raise MetaArtifactNotFoundError(artifact_name)
    views_sql = _read_json_file(path)
    for view_id, sql in views_sql.items():
        if not isinstance(sql, str):
            msg = f"views_sql.json entry for {view_id!r} is not a string"
            raise TypeError(msg)
        try:
            assert_single_select_statement(sql)
        except UnsafeSqlError as exc:
            raise MetaSqlUnsafeError(str(view_id)) from exc
    return views_sql


def _read_views_sql_diff(pointer: ServingSnapshotPointerProtocol) -> dict[str, object]:
    artifact_name = "views_sql_diff.json"
    path = _artifact_dir_for_pointer(pointer) / artifact_name
    if not path.is_file():
        raise MetaArtifactNotFoundError(artifact_name)
    return _read_json_file(path)


def _build_resource_templates_response(ops: ServingOperations) -> dict[str, object]:
    payload = build_resource_templates_payload(
        ops,
        templates=DEFAULT_RESOURCE_TEMPLATES,
        generated_at=datetime.now(UTC),
    )
    return ResourceTemplatesResponse.model_validate(payload).model_dump(mode="json")


def register_meta_resources(
    mcp: FastMCP,
    ops: ServingOperations,
    *,
    settings: ServingSettings,
) -> None:
    """Register meta and semantic discovery resources."""

    @mcp.resource("codeintel://semantic/views")
    def semantic_views() -> dict[str, object]:
        return SemanticCatalogResponse.model_validate(ops.catalog()).model_dump(mode="json")

    @mcp.resource("codeintel://semantic/views/{view_id}")
    def view_description(view_id: str) -> dict[str, object]:
        return SemanticViewDescriptionResponse.model_validate(ops.describe(view_id)).model_dump(
            mode="json"
        )

    @mcp.resource("codeintel://meta/serving")
    def serving_meta_resource() -> dict[str, object]:
        return ops.meta()

    @mcp.resource("codeintel://meta/resources")
    def resource_templates() -> dict[str, object]:
        return _build_resource_templates_response(ops)

    @mcp.resource("codeintel://meta/environment", mime_type="application/json", tags={"meta"})
    def environment() -> dict[str, object]:
        return build_environment_meta_payload(ops, settings=settings)

    @mcp.resource("codeintel://meta/views_sql", mime_type="application/json", tags={"meta"})
    def views_sql() -> dict[str, object]:
        return _read_views_sql(ops.db.current_pointer())

    @mcp.resource("codeintel://meta/views_sql_diff", mime_type="application/json", tags={"meta"})
    def views_sql_diff() -> dict[str, object]:
        return _read_views_sql_diff(ops.db.current_pointer())


__all__ = ["register_meta_resources"]
