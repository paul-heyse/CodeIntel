"""Tests for MCP response models.

Verify that all response models are correctly defined, validate input,
and serialize/deserialize to JSON properly.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from codeintel.serving.mcp.models import (
    DEFAULT_RESOURCE_TEMPLATES,
    BuildSpecInfo,
    ExportHandleResponse,
    ExportMetaResponse,
    ExportQuerySpec,
    ExportSchemaSummary,
    ExportSnapshot,
    ExportURIs,
    QueryLimits,
    QueryPreview,
    ResourceTemplate,
    ResourceTemplatesResponse,
    SemanticLayerInfo,
    SemanticQueryToolResponse,
    ServingMetaResponse,
    SnapshotRef,
)
from codeintel.serving.semantic.models import SemanticQueryResponse
from codeintel.serving.snapshot.models import ServingSnapshotIdentity
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_false,
    expect_is_not_none,
    expect_true,
)

if TYPE_CHECKING:
    from codeintel.serving.export.formats import ExportFormat
    from codeintel.serving.mcp.models.exports import ExportStatus

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_snapshot_ref() -> SnapshotRef:
    """Provide a sample SnapshotRef for testing.

    Returns
    -------
    SnapshotRef
        Sample snapshot reference with test data.
    """
    return SnapshotRef(
        repo="org/repo",
        commit="abc123def456",
        run_id="run-001",
        published_at=datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC),
    )


@pytest.fixture
def sample_export_snapshot(sample_snapshot_ref: SnapshotRef) -> ExportSnapshot:
    """Provide a sample ExportSnapshot for testing.

    Returns
    -------
    ExportSnapshot
        Sample export snapshot with test data.
    """
    return ExportSnapshot(
        snapshot=sample_snapshot_ref,
        semantic_layer_hash="sl_abc123",
        buildspec_hash="bs_def456",
    )


@pytest.fixture
def sample_semantic_layer_info() -> SemanticLayerInfo:
    """Provide a sample SemanticLayerInfo for testing.

    Returns
    -------
    SemanticLayerInfo
        Sample semantic layer info with test data.
    """
    return SemanticLayerInfo(
        version="1.0.0",
        hash="hash123",
        view_count=10,
        schema_manifest_hash="manifest_hash",
    )


@pytest.fixture
def sample_buildspec_info() -> BuildSpecInfo:
    """Provide a sample BuildSpecInfo for testing.

    Returns
    -------
    BuildSpecInfo
        Sample buildspec info with test data.
    """
    return BuildSpecInfo(
        version="2.0.0",
        hash="buildspec_hash",
        compiled_at=datetime(2024, 1, 10, 8, 0, 0, tzinfo=UTC),
    )


@pytest.fixture
def sample_query_limits() -> QueryLimits:
    """Provide a sample QueryLimits for testing.

    Returns
    -------
    QueryLimits
        Sample query limits with test data.
    """
    return QueryLimits(
        default_limit=200,
        max_limit=5000,
        export_max_rows=100000,
        export_ttl_seconds=3600,
    )


# =============================================================================
# SnapshotRef Tests
# =============================================================================


def test_snapshot_ref_creation(sample_snapshot_ref: SnapshotRef) -> None:
    """Verify SnapshotRef model creation."""
    expect_equal(sample_snapshot_ref.repo, "org/repo")
    expect_equal(sample_snapshot_ref.commit, "abc123def456")
    expect_equal(sample_snapshot_ref.run_id, "run-001")
    expect_is_not_none(sample_snapshot_ref.published_at)


def test_snapshot_ref_json_round_trip(sample_snapshot_ref: SnapshotRef) -> None:
    """Verify SnapshotRef serializes and deserializes correctly."""
    json_str = sample_snapshot_ref.model_dump_json()
    restored = SnapshotRef.model_validate_json(json_str)
    expect_equal(restored.repo, sample_snapshot_ref.repo)
    expect_equal(restored.commit, sample_snapshot_ref.commit)


def test_snapshot_ref_forbids_extra_fields() -> None:
    """Verify SnapshotRef rejects extra fields."""
    with pytest.raises(ValidationError):
        SnapshotRef(
            repo="org/repo",
            commit="abc123",
            run_id="run-001",
            published_at=datetime.now(UTC),
            extra_field="not_allowed",  # type: ignore[call-arg]
        )


# =============================================================================
# ResourceTemplate Tests
# =============================================================================


def test_resource_template_creation() -> None:
    """Verify ResourceTemplate model creation."""
    template = ResourceTemplate(
        uri="codeintel://semantic/views/{view_id}",
        description="Semantic view descriptor.",
        mime_type="application/json",
        tags=("semantic", "describe"),
    )
    expect_equal(template.uri, "codeintel://semantic/views/{view_id}")
    expect_equal(template.description, "Semantic view descriptor.")
    expect_equal(template.mime_type, "application/json")
    expect_equal(template.tags, ("semantic", "describe"))


def test_resource_template_optional_fields() -> None:
    """Verify ResourceTemplate handles optional fields."""
    template = ResourceTemplate(
        uri="codeintel://exports/{export_id}",
        description="Export payload.",
    )
    expect_equal(template.mime_type, None)
    expect_equal(template.tags, ())


# =============================================================================
# SemanticLayerInfo Tests
# =============================================================================


def test_semantic_layer_info_creation(sample_semantic_layer_info: SemanticLayerInfo) -> None:
    """Verify SemanticLayerInfo model creation."""
    expect_equal(sample_semantic_layer_info.version, "1.0.0")
    expect_equal(sample_semantic_layer_info.hash, "hash123")
    expect_equal(sample_semantic_layer_info.view_count, 10)


def test_semantic_layer_info_view_count_validation() -> None:
    """Verify SemanticLayerInfo rejects negative view_count."""
    with pytest.raises(ValidationError):
        SemanticLayerInfo(
            version="1.0.0",
            hash="hash123",
            view_count=-1,
        )


# =============================================================================
# BuildSpecInfo Tests
# =============================================================================


def test_buildspec_info_creation(sample_buildspec_info: BuildSpecInfo) -> None:
    """Verify BuildSpecInfo model creation."""
    expect_equal(sample_buildspec_info.version, "2.0.0")
    expect_equal(sample_buildspec_info.hash, "buildspec_hash")
    expect_is_not_none(sample_buildspec_info.compiled_at)


# =============================================================================
# QueryLimits Tests
# =============================================================================


def test_query_limits_creation(sample_query_limits: QueryLimits) -> None:
    """Verify QueryLimits model creation."""
    expect_equal(sample_query_limits.default_limit, 200)
    expect_equal(sample_query_limits.max_limit, 5000)
    expect_equal(sample_query_limits.export_max_rows, 100000)
    expect_equal(sample_query_limits.export_ttl_seconds, 3600)


def test_query_limits_defaults() -> None:
    """Verify QueryLimits uses sensible defaults."""
    limits = QueryLimits()
    expect_equal(limits.default_limit, 200)
    expect_equal(limits.max_limit, 5000)
    expect_equal(limits.export_max_rows, 100000)
    expect_equal(limits.export_ttl_seconds, None)


def test_query_limits_validation() -> None:
    """Verify QueryLimits rejects invalid values."""
    with pytest.raises(ValidationError):
        QueryLimits(default_limit=0)


# =============================================================================
# ExportSnapshot Tests
# =============================================================================


def test_export_snapshot_creation(sample_export_snapshot: ExportSnapshot) -> None:
    """Verify ExportSnapshot model creation."""
    expect_is_not_none(sample_export_snapshot.snapshot)
    expect_equal(sample_export_snapshot.semantic_layer_hash, "sl_abc123")
    expect_equal(sample_export_snapshot.buildspec_hash, "bs_def456")


# =============================================================================
# ExportURIs Tests
# =============================================================================


def test_export_uris_creation() -> None:
    """Verify ExportURIs model creation."""
    uris = ExportURIs(
        payload_uri="codeintel://exports/exp123",
        meta_uri="codeintel://exports/exp123/meta",
        preview_uri="codeintel://exports/exp123/preview",
        sql_uri="codeintel://exports/exp123/sql",
    )
    expect_equal(uris.payload_uri, "codeintel://exports/exp123")
    expect_equal(uris.meta_uri, "codeintel://exports/exp123/meta")
    expect_equal(uris.preview_uri, "codeintel://exports/exp123/preview")
    expect_equal(uris.sql_uri, "codeintel://exports/exp123/sql")


def test_export_uris_optional_fields() -> None:
    """Verify ExportURIs handles optional fields."""
    uris = ExportURIs(
        payload_uri="codeintel://exports/exp123",
        meta_uri="codeintel://exports/exp123/meta",
    )
    expect_equal(uris.preview_uri, None)
    expect_equal(uris.sql_uri, None)


# =============================================================================
# ExportQuerySpec Tests
# =============================================================================


def test_export_query_spec_creation() -> None:
    """Verify ExportQuerySpec model creation."""
    spec = ExportQuerySpec(
        view_id="function_metrics",
        select=("repo", "commit", "loc"),
        order_by=("-loc",),
        filters=({"column": "loc", "op": "gt", "value": 100},),
        limit=1000,
        offset=0,
        query_hash="q_abc123",
    )
    expect_equal(spec.view_id, "function_metrics")
    expect_equal(spec.select, ("repo", "commit", "loc"))
    expect_equal(len(spec.filters), 1)


def test_export_query_spec_defaults() -> None:
    """Verify ExportQuerySpec uses sensible defaults."""
    spec = ExportQuerySpec()
    expect_equal(spec.view_id, None)
    expect_equal(spec.select, None)
    expect_equal(spec.order_by, ())
    expect_equal(spec.filters, ())


# =============================================================================
# ExportSchemaSummary Tests
# =============================================================================


def test_export_schema_summary_creation() -> None:
    """Verify ExportSchemaSummary model creation."""
    schema = ExportSchemaSummary(
        columns=("repo", "commit", "loc"),
        types={"repo": "VARCHAR", "commit": "VARCHAR", "loc": "INTEGER"},
        schema_hash="schema_abc123",
    )
    expect_equal(schema.columns, ("repo", "commit", "loc"))
    expect_equal(schema.types["loc"], "INTEGER")


# =============================================================================
# ExportHandleResponse Tests
# =============================================================================


def test_export_handle_response_creation(sample_export_snapshot: ExportSnapshot) -> None:
    """Verify ExportHandleResponse model creation."""
    handle = ExportHandleResponse(
        export_id="exp123456789",
        format="jsonl",
        mime_type="application/x-ndjson",
        filename="function_metrics.jsonl",
        uri="codeintel://exports/exp123456789",
        meta_uri="codeintel://exports/exp123456789/meta",
        created_at=datetime.now(UTC),
        snapshot=sample_export_snapshot,
    )
    expect_equal(handle.export_id, "exp123456789")
    expect_equal(handle.format, "jsonl")
    expect_equal(handle.mime_type, "application/x-ndjson")


def test_export_handle_response_with_optional_fields(
    sample_export_snapshot: ExportSnapshot,
) -> None:
    """Verify ExportHandleResponse handles optional fields."""
    handle = ExportHandleResponse(
        export_id="exp123456789",
        format="parquet",
        mime_type="application/vnd.apache.parquet",
        filename="data.parquet",
        uri="codeintel://exports/exp123456789",
        meta_uri="codeintel://exports/exp123456789/meta",
        preview_uri="codeintel://exports/exp123456789/preview",
        sql_uri="codeintel://exports/exp123456789/sql",
        created_at=datetime.now(UTC),
        expires_at=datetime(2024, 12, 31, tzinfo=UTC),
        row_count=1000,
        byte_size=50000,
        snapshot=sample_export_snapshot,
        note="Large result spilled to export.",
    )
    expect_equal(handle.row_count, 1000)
    expect_equal(handle.byte_size, 50000)
    expect_equal(handle.note, "Large result spilled to export.")


# =============================================================================
# ExportMetaResponse Tests
# =============================================================================


def test_export_meta_response_creation(sample_export_snapshot: ExportSnapshot) -> None:
    """Verify ExportMetaResponse model creation."""
    uris = ExportURIs(
        payload_uri="codeintel://exports/exp123",
        meta_uri="codeintel://exports/exp123/meta",
    )
    meta = ExportMetaResponse(
        export_id="exp123456789",
        status="ready",
        created_at=datetime.now(UTC),
        format="jsonl",
        mime_type="application/x-ndjson",
        filename="data.jsonl",
        snapshot=sample_export_snapshot,
        uris=uris,
    )
    expect_equal(meta.export_id, "exp123456789")
    expect_equal(meta.status, "ready")
    expect_equal(meta.format, "jsonl")


@pytest.mark.parametrize("status", ["ready", "expired", "missing", "error"])
def test_export_meta_response_status_values(
    status: ExportStatus, sample_export_snapshot: ExportSnapshot
) -> None:
    """Verify ExportMetaResponse accepts all valid status values."""
    uris = ExportURIs(
        payload_uri="codeintel://exports/exp123",
        meta_uri="codeintel://exports/exp123/meta",
    )
    meta = ExportMetaResponse(
        export_id="exp123456789",
        status=status,
        created_at=datetime.now(UTC),
        format="json",
        mime_type="application/json",
        filename="data.json",
        snapshot=sample_export_snapshot,
        uris=uris,
    )
    expect_equal(meta.status, status)


# =============================================================================
# QueryPreview Tests
# =============================================================================


def test_query_preview_creation() -> None:
    """Verify QueryPreview model creation."""
    preview = QueryPreview(
        columns=("repo", "commit", "loc"),
        rows=(
            {"repo": "org/repo", "commit": "abc123", "loc": 100},
            {"repo": "org/repo", "commit": "def456", "loc": 200},
        ),
        truncated=True,
    )
    expect_equal(preview.columns, ("repo", "commit", "loc"))
    expect_equal(len(preview.rows), 2)
    expect_true(preview.truncated)


def test_query_preview_defaults() -> None:
    """Verify QueryPreview uses sensible defaults."""
    preview = QueryPreview()
    expect_equal(preview.columns, ())
    expect_equal(preview.rows, ())
    expect_true(preview.truncated)


# =============================================================================
# SemanticQueryToolResponse Tests
# =============================================================================


def test_semantic_query_tool_response_creation() -> None:
    """Verify SemanticQueryToolResponse model creation."""
    result = SemanticQueryResponse(
        view_id="function_metrics",
        columns=["repo", "commit", "loc"],
        rows=[{"repo": "org/repo", "commit": "abc123", "loc": 100}],
        truncated=False,
        snapshot=ServingSnapshotIdentity(repo="org/repo", commit="abc123", run_id="run-001"),
    )
    response = SemanticQueryToolResponse(result=result)
    expect_equal(response.result.view_id, "function_metrics")
    expect_false(response.result.truncated)
    expect_equal(response.export, None)


def test_semantic_query_tool_response_with_export(sample_export_snapshot: ExportSnapshot) -> None:
    """Verify SemanticQueryToolResponse handles export spillover."""
    result = SemanticQueryResponse(
        view_id="function_metrics",
        columns=["repo", "commit", "loc"],
        rows=[{"repo": "org/repo", "commit": "abc123", "loc": 100}],
        truncated=True,
        snapshot=ServingSnapshotIdentity(repo="org/repo", commit="abc123", run_id="run-001"),
    )
    export_handle = ExportHandleResponse(
        export_id="exp123456789",
        format="jsonl",
        mime_type="application/x-ndjson",
        filename="function_metrics.jsonl",
        uri="codeintel://exports/exp123456789",
        meta_uri="codeintel://exports/exp123456789/meta",
        created_at=datetime.now(UTC),
        snapshot=sample_export_snapshot,
    )
    preview = QueryPreview(
        columns=("repo", "commit", "loc"),
        rows=({"repo": "org/repo", "commit": "abc123", "loc": 100},),
        truncated=True,
    )
    response = SemanticQueryToolResponse(
        result=result,
        preview=preview,
        export=export_handle,
        export_uri="codeintel://exports/exp123456789",
        export_meta_uri="codeintel://exports/exp123456789/meta",
        note="Result truncated; use export_uri for full dataset.",
    )
    expect_true(response.result.truncated)
    expect_is_not_none(response.export)
    expect_is_not_none(response.preview)
    expect_equal(response.note, "Result truncated; use export_uri for full dataset.")


# =============================================================================
# ServingMetaResponse Tests
# =============================================================================


def test_serving_meta_response_creation(
    sample_snapshot_ref: SnapshotRef,
    sample_semantic_layer_info: SemanticLayerInfo,
    sample_buildspec_info: BuildSpecInfo,
    sample_query_limits: QueryLimits,
) -> None:
    """Verify ServingMetaResponse model creation."""
    response = ServingMetaResponse(
        server_version="1.0.0",
        started_at=datetime.now(UTC),
        snapshot=sample_snapshot_ref,
        semantic_layer=sample_semantic_layer_info,
        buildspec=sample_buildspec_info,
        limits=sample_query_limits,
    )
    expect_equal(response.service, "codeintel")
    expect_equal(response.protocol, "mcp")
    expect_equal(response.server_version, "1.0.0")
    expect_true(response.read_only)


def test_serving_meta_response_with_all_fields(
    sample_snapshot_ref: SnapshotRef,
    sample_semantic_layer_info: SemanticLayerInfo,
    sample_buildspec_info: BuildSpecInfo,
    sample_query_limits: QueryLimits,
) -> None:
    """Verify ServingMetaResponse with all optional fields."""
    templates = (
        ResourceTemplate(
            uri="codeintel://meta/serving",
            description="Serving metadata.",
            mime_type="application/json",
            tags=("meta",),
        ),
    )
    response = ServingMetaResponse(
        server_version="1.0.0",
        started_at=datetime.now(UTC),
        snapshot=sample_snapshot_ref,
        semantic_layer=sample_semantic_layer_info,
        buildspec=sample_buildspec_info,
        read_only=True,
        features={"supports_explain": True, "supports_export": True},
        limits=sample_query_limits,
        resource_templates=templates,
        inventories={"views": 10, "tables": 5},
    )
    expect_equal(len(response.resource_templates), 1)
    expect_true(response.features["supports_explain"])
    expect_equal(response.inventories["views"], 10)


# =============================================================================
# ResourceTemplatesResponse Tests
# =============================================================================


def test_resource_templates_response_creation(sample_snapshot_ref: SnapshotRef) -> None:
    """Verify ResourceTemplatesResponse model creation."""
    response = ResourceTemplatesResponse(
        generated_at=datetime.now(UTC),
        snapshot=sample_snapshot_ref,
    )
    expect_equal(response.uri, "codeintel://meta/resources")
    expect_is_not_none(response.generated_at)


def test_resource_templates_response_with_templates(
    sample_snapshot_ref: SnapshotRef,
    sample_semantic_layer_info: SemanticLayerInfo,
    sample_buildspec_info: BuildSpecInfo,
) -> None:
    """Verify ResourceTemplatesResponse with all fields."""
    response = ResourceTemplatesResponse(
        generated_at=datetime.now(UTC),
        snapshot=sample_snapshot_ref,
        semantic_layer=sample_semantic_layer_info,
        buildspec=sample_buildspec_info,
        templates=DEFAULT_RESOURCE_TEMPLATES,
        notes=("Exports expire after 1 hour.",),
    )
    expect_equal(len(response.templates), len(DEFAULT_RESOURCE_TEMPLATES))
    expect_equal(response.notes, ("Exports expire after 1 hour.",))


# =============================================================================
# DEFAULT_RESOURCE_TEMPLATES Tests
# =============================================================================


def test_default_resource_templates_count() -> None:
    """Verify DEFAULT_RESOURCE_TEMPLATES has expected count."""
    expect_equal(len(DEFAULT_RESOURCE_TEMPLATES), 11)


def test_default_resource_templates_structure() -> None:
    """Verify DEFAULT_RESOURCE_TEMPLATES entries are valid."""
    for template in DEFAULT_RESOURCE_TEMPLATES:
        expect_true(template.uri.startswith("codeintel://"))
        expect_true(len(template.description) > 0, message="Template should have description")
        expect_is_not_none(template.tags)


def test_default_resource_templates_uris() -> None:
    """Verify expected URIs are present in DEFAULT_RESOURCE_TEMPLATES."""
    uris = {t.uri for t in DEFAULT_RESOURCE_TEMPLATES}
    expected_uris = {
        "codeintel://meta/serving",
        "codeintel://meta/resources",
        "codeintel://meta/environment",
        "codeintel://semantic/views",
        "codeintel://semantic/views/{view_id}",
        "codeintel://exports/{export_id}",
        "codeintel://exports/{export_id}/meta",
        "codeintel://exports/{export_id}/preview",
        "codeintel://exports/{export_id}/sql",
        "codeintel://exports/{export_id}/lines{?offset,limit}",
        "codeintel://exports/{export_id}/bytes{?offset,limit}",
    }
    expect_equal(uris, expected_uris)


# =============================================================================
# Literal Type Tests
# =============================================================================


@pytest.mark.parametrize("fmt", ["jsonl", "json", "parquet", "arrow"])
def test_export_format_values(fmt: ExportFormat, sample_export_snapshot: ExportSnapshot) -> None:
    """Verify ExportFormat accepts all valid values."""
    handle = ExportHandleResponse(
        export_id="exp123456789",
        format=fmt,
        mime_type="application/octet-stream",
        filename=f"data.{fmt}",
        uri="codeintel://exports/exp123456789",
        meta_uri="codeintel://exports/exp123456789/meta",
        created_at=datetime.now(UTC),
        snapshot=sample_export_snapshot,
    )
    expect_equal(handle.format, fmt)


def test_export_format_rejects_invalid() -> None:
    """Verify ExportFormat rejects invalid values."""
    with pytest.raises(ValidationError):
        # Create a dict and validate it
        ExportHandleResponse.model_validate(
            {
                "export_id": "exp123456789",
                "format": "csv",  # Invalid format
                "mime_type": "text/csv",
                "filename": "data.csv",
                "uri": "codeintel://exports/exp123456789",
                "meta_uri": "codeintel://exports/exp123456789/meta",
                "created_at": datetime.now(UTC).isoformat(),
                "snapshot": {
                    "snapshot": {
                        "repo": "org/repo",
                        "commit": "abc123",
                        "run_id": "run-001",
                        "published_at": datetime.now(UTC).isoformat(),
                    },
                    "semantic_layer_hash": "sl_hash",
                    "buildspec_hash": "bs_hash",
                },
            }
        )


# =============================================================================
# JSON Serialization Round-Trip Tests
# =============================================================================


def test_serving_meta_response_json_round_trip(
    sample_snapshot_ref: SnapshotRef,
    sample_semantic_layer_info: SemanticLayerInfo,
    sample_buildspec_info: BuildSpecInfo,
    sample_query_limits: QueryLimits,
) -> None:
    """Verify ServingMetaResponse JSON serialization round-trip."""
    original = ServingMetaResponse(
        server_version="1.0.0",
        started_at=datetime.now(UTC),
        snapshot=sample_snapshot_ref,
        semantic_layer=sample_semantic_layer_info,
        buildspec=sample_buildspec_info,
        limits=sample_query_limits,
        resource_templates=DEFAULT_RESOURCE_TEMPLATES,
    )
    json_str = original.model_dump_json()
    restored = ServingMetaResponse.model_validate_json(json_str)
    expect_equal(restored.server_version, original.server_version)
    expect_equal(restored.service, original.service)
    expect_equal(len(restored.resource_templates), len(original.resource_templates))


def test_export_meta_response_json_round_trip(sample_export_snapshot: ExportSnapshot) -> None:
    """Verify ExportMetaResponse JSON serialization round-trip."""
    uris = ExportURIs(
        payload_uri="codeintel://exports/exp123",
        meta_uri="codeintel://exports/exp123/meta",
    )
    original = ExportMetaResponse(
        export_id="exp123456789",
        status="ready",
        created_at=datetime.now(UTC),
        format="jsonl",
        mime_type="application/x-ndjson",
        filename="data.jsonl",
        row_count=1000,
        snapshot=sample_export_snapshot,
        uris=uris,
    )
    json_str = original.model_dump_json()
    restored = ExportMetaResponse.model_validate_json(json_str)
    expect_equal(restored.export_id, original.export_id)
    expect_equal(restored.status, original.status)
    expect_equal(restored.row_count, original.row_count)


# =============================================================================
# Model Immutability Tests
# =============================================================================


def test_snapshot_ref_is_frozen(sample_snapshot_ref: SnapshotRef) -> None:
    """Verify SnapshotRef is immutable (frozen)."""
    with pytest.raises(ValidationError):
        sample_snapshot_ref.repo = "new/repo"  # type: ignore[misc]


def test_resource_template_is_frozen() -> None:
    """Verify ResourceTemplate is immutable (frozen)."""
    template = ResourceTemplate(
        uri="codeintel://meta/serving",
        description="Test",
    )
    with pytest.raises(ValidationError):
        template.uri = "codeintel://meta/new"  # type: ignore[misc]


def test_query_limits_is_frozen(sample_query_limits: QueryLimits) -> None:
    """Verify QueryLimits is immutable (frozen)."""
    with pytest.raises(ValidationError):
        sample_query_limits.max_limit = 10000  # type: ignore[misc]
