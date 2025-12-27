"""Tests for schema index derivations based on catalog outputs."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, cast

import pytest

from codeintel.build.schemas.schema_index import build_schema_index
from codeintel.build.target_metadata import TargetSystem, get_target_metadata_service
from codeintel.core.schemas.primitives import Column, TableSchema
from codeintel.core.schemas.provider import MappingSchemaProvider
from tests._helpers.catalog import build_catalog, make_target_descriptor

if TYPE_CHECKING:
    from codeintel.build.hamilton.dag_catalog import TargetDescriptor
    from codeintel.build.schemas.inference_service import SchemaInferenceService


class _FakeInferenceService:
    @staticmethod
    def inferable_table_keys(*, catalog: object) -> frozenset[str]:
        _ = catalog
        return frozenset({"analytics.inferable"})

    @staticmethod
    def infer_table_schema(
        table_key: str,
        *,
        declared_provider: MappingSchemaProvider,
    ) -> TableSchema:
        _ = declared_provider
        return TableSchema(
            schema="analytics",
            name=table_key.split(".", maxsplit=1)[1],
            columns=[Column("id", "INTEGER", nullable=False)],
        )

    @staticmethod
    def infer_table_schemas(
        table_keys: list[str],
        *,
        declared_provider: MappingSchemaProvider,
    ) -> dict[str, TableSchema]:
        return {
            table_key: _FakeInferenceService.infer_table_schema(
                table_key,
                declared_provider=declared_provider,
            )
            for table_key in table_keys
        }


def _build_target_system() -> TargetSystem:
    target = make_target_descriptor(
        name="alpha",
        module="analytics",
        description="Test target",
    )
    catalog = build_catalog(
        targets=(target,),
        table_keys_by_target={
            "alpha": ("analytics.explicit", "analytics.inferable"),
        },
    )
    by_name = {target.name: target}
    by_table_key = {
        table_key: catalog.targets[output.producer_target]
        for table_key, output in catalog.table_outputs.items()
    }
    by_artifact_name: dict[str, TargetDescriptor] = {}
    runtime = get_target_metadata_service().system.runtime
    return TargetSystem(
        runtime=runtime,
        catalog=catalog,
        by_name=MappingProxyType(by_name),
        by_table_key=MappingProxyType(by_table_key),
        by_artifact_name=MappingProxyType(by_artifact_name),
    )


def test_schema_index_requires_explicit_schema_for_non_inferable() -> None:
    """Non-inferable outputs must have explicit registry schemas."""
    system = _build_target_system()
    declared_provider = MappingSchemaProvider({})
    with pytest.raises(ValueError, match="Missing explicit schema overrides"):
        build_schema_index(
            system=system,
            declared_provider=declared_provider,
            inference_service=cast("SchemaInferenceService", _FakeInferenceService()),
        )


def test_schema_index_uses_catalog_outputs() -> None:
    """Schema index derivations should come from catalog outputs."""
    system = _build_target_system()
    declared_provider = MappingSchemaProvider(
        {
            "analytics.explicit": TableSchema(
                schema="analytics",
                name="explicit",
                columns=[Column("id", "INTEGER", nullable=False)],
            )
        }
    )
    schema_index = build_schema_index(
        system=system,
        declared_provider=declared_provider,
        inference_service=cast("SchemaInferenceService", _FakeInferenceService()),
    )

    explicit_schema = schema_index.get_table_schema("analytics.explicit", allow_inference=False)
    if explicit_schema is None:
        pytest.fail("Expected explicit schema for analytics.explicit")

    inferred_schema = schema_index.get_table_schema("analytics.inferable", allow_inference=True)
    if inferred_schema is None:
        pytest.fail("Expected inferred schema for analytics.inferable")
