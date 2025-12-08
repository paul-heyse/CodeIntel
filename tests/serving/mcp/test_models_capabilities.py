"""Tests for DatasetSchemaResponse capability normalization."""

from types import SimpleNamespace
from typing import cast

import pytest

from codeintel.serving.mcp import models
from tests._helpers.assertions import expect_equal


def _make_schema(
    *,
    dataset_name: str,
    table_key: str,
    capabilities: object,
) -> SimpleNamespace:
    return SimpleNamespace(
        dataset_name=dataset_name,
        table_key=table_key,
        duckdb_schema=[],
        json_schema=None,
        sample_rows=[],
        capabilities=capabilities,
        owner=None,
        freshness_sla=None,
        retention_policy=None,
        schema_version=None,
        stable_id=None,
        validation_profile=None,
        meta=None,
    )


def test_from_domain_decodes_bytes_keys_and_values() -> None:
    """Bytes keys/values are decoded and normalized through from_domain/to_domain."""
    schema = cast(
        "models.dm.DatasetSchema",
        _make_schema(
            dataset_name="ds",
            table_key="tbl",
            capabilities={b"export": b"1", "docs_view": True},
        ),
    )

    response = models.DatasetSchemaResponse.from_domain(schema)

    expect_equal(response.capabilities, {"export": "1", "docs_view": True})
    domain = response.to_domain()
    expect_equal(domain.capabilities, {"export": True, "docs_view": True})


def test_from_domain_rejects_unsupported_value_types() -> None:
    """Unsupported capability value types raise a clear error."""
    schema = cast(
        "models.dm.DatasetSchema",
        _make_schema(
            dataset_name="ds",
            table_key="tbl",
            capabilities={"export": object()},
        ),
    )

    with pytest.raises(TypeError):
        models.DatasetSchemaResponse.from_domain(schema)
