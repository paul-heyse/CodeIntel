"""Tests for SchemaIndex override handling and inference error reporting."""

from __future__ import annotations

from datetime import UTC, datetime
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

import pytest

from codeintel.build.contracts import OutputContract, TableOutputDescriptor
from codeintel.build.schemas.inference_service import (
    SchemaInferenceService,
    get_schema_inference_service,
)
from codeintel.build.schemas.schema_index import SchemaDerivation, SchemaIndex, build_schema_index
from codeintel.build.target_metadata import TargetSystem, get_target_metadata_service
from codeintel.build.targets import TargetDescriptor
from codeintel.core.schemas.primitives import Column, TableSchema
from codeintel.core.schemas.provider import MappingSchemaProvider
from tests._helpers.catalog import build_catalog, make_target_descriptor

if TYPE_CHECKING:
    from codeintel.core.schemas.provider import SchemaProvider


def _build_target_system(targets: tuple[TargetDescriptor, ...]) -> TargetSystem:
    catalog = build_catalog(targets=targets)
    by_name: dict[str, TargetDescriptor] = {}
    by_table_key: dict[str, TargetDescriptor] = {}
    by_artifact_name: dict[str, TargetDescriptor] = {}

    for target in targets:
        by_name[target.name] = target
        for table_key in target.contract.table_keys:
            by_table_key[table_key] = target
        for artifact_name in target.contract.artifact_names:
            by_artifact_name[artifact_name] = target

    runtime = get_target_metadata_service().system.runtime
    return TargetSystem(
        runtime=runtime,
        catalog=catalog,
        by_name=MappingProxyType(by_name),
        by_table_key=MappingProxyType(by_table_key),
        by_artifact_name=MappingProxyType(by_artifact_name),
    )


def test_schema_index_accepts_explicit_override_for_non_inferable_outputs() -> None:
    """Explicit overrides resolve schemas for non-inferable outputs."""
    table_key = "analytics.override_ok"
    override_schema = TableSchema(
        schema="analytics",
        name="override_ok",
        columns=[Column("id", "INTEGER", nullable=False)],
    )
    target = make_target_descriptor(
        name="override_ok_target",
        module="analytics",
        contract=OutputContract(tables=(TableOutputDescriptor(table_key=table_key),)),
        description="Test target with explicit override.",
    )
    declared_provider = MappingSchemaProvider({table_key: override_schema})
    system = _build_target_system((target,))
    schema_index = build_schema_index(
        system=system,
        declared_provider=declared_provider,
        inference_service=get_schema_inference_service(),
    )

    resolved = schema_index.get_table_schema(table_key, allow_inference=False)
    if resolved != override_schema:
        pytest.fail("SchemaIndex did not return the explicit override schema")


def test_schema_index_records_inference_errors() -> None:
    """Inference errors should be recorded deterministically."""

    class _FailingInferenceService:
        @staticmethod
        def infer_table_schema(
            table_key: str,
            *,
            declared_provider: SchemaProvider,
        ) -> TableSchema:
            if declared_provider is None:
                msg = "declared_provider is required"
                raise ValueError(msg)
            msg = f"boom:{table_key}"
            raise ValueError(msg)

    table_key = "analytics.inference_failure"
    schema_index = SchemaIndex(
        derivations={
            table_key: SchemaDerivation(
                table_key=table_key,
                kind="inferred_ibis",
                source="test",
                override_schema=None,
            )
        },
        inferable_table_keys=frozenset({table_key}),
        declared_provider=MappingSchemaProvider({}),
        inference_service=cast("SchemaInferenceService", _FailingInferenceService()),
    )

    schema = schema_index.get_table_schema(table_key)
    if schema is not None:
        pytest.fail("Expected inference failure to return None without overrides")
    error = schema_index.get_inference_error(table_key)
    if error != "ValueError: boom:analytics.inference_failure":
        msg = f"Unexpected inference error message: {error}"
        pytest.fail(msg)


def test_schema_index_inference_error_rows() -> None:
    """Inference error rows should include run metadata."""

    class _FailingInferenceService:
        @staticmethod
        def infer_table_schema(
            table_key: str,
            *,
            declared_provider: SchemaProvider,
        ) -> TableSchema:
            _ = declared_provider
            msg = f"boom:{table_key}"
            raise ValueError(msg)

    table_key = "analytics.inference_failure"
    schema_index = SchemaIndex(
        derivations={
            table_key: SchemaDerivation(
                table_key=table_key,
                kind="inferred_ibis",
                source="test",
                override_schema=None,
            )
        },
        inferable_table_keys=frozenset({table_key}),
        declared_provider=MappingSchemaProvider({}),
        inference_service=cast("SchemaInferenceService", _FailingInferenceService()),
    )

    _ = schema_index.get_table_schema(table_key)
    occurred_at = datetime(2024, 1, 1, tzinfo=UTC)
    rows = list(
        schema_index.iter_inference_error_rows(
            repo="org/repo",
            commit="deadbeef",
            run_id="run-123",
            occurred_at=occurred_at,
        )
    )

    assert rows == [
        {
            "table_key": table_key,
            "repo": "org/repo",
            "commit": "deadbeef",
            "error": "ValueError: boom:analytics.inference_failure",
            "occurred_at": occurred_at,
            "run_id": "run-123",
        }
    ]
