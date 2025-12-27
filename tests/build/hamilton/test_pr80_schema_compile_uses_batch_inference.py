"""Tests for PR-80: batch schema inference for schema compilation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

from codeintel.build.schemas import declared_schema_provider
from codeintel.build.schemas.compile import SchemaManifestRequest, compile_schema_manifest
from codeintel.build.target_metadata import get_target_metadata_service
from codeintel.core.schemas.primitives import Column, TableSchema
from codeintel.storage.helpers.table_key import split_table_key

if TYPE_CHECKING:
    from collections.abc import Iterable

    from codeintel.core.schemas.provider import SchemaProvider


def _schema_for_key(table_key: str) -> TableSchema:
    schema, name = split_table_key(table_key)
    return TableSchema(
        schema=schema,
        name=name,
        columns=[Column("id", "VARCHAR", nullable=False)],
    )


@dataclass
class _SpyBatchInferer:
    calls: int = 0
    seen: list[tuple[str, ...]] = field(default_factory=list)

    def __call__(
        self,
        table_keys: Iterable[str],
        *,
        declared_provider: SchemaProvider,
    ) -> dict[str, TableSchema]:
        if declared_provider is None:
            msg = "declared_provider is required"
            raise ValueError(msg)
        self.calls += 1
        keys = tuple(sorted(table_keys))
        self.seen.append(keys)
        return {key: _schema_for_key(key) for key in keys}


def test_pr80_schema_compile_uses_batch_inference() -> None:
    """Ensure compile_schema_manifest pre-infers schemas in a single batch call."""
    provider = declared_schema_provider()
    spy = _SpyBatchInferer()

    request = SchemaManifestRequest(
        targets=("risk_factors", "hotspots"),
        infer_native=True,
        batch_infer_native=True,
        stable=True,
    )

    manifest = compile_schema_manifest(provider=provider, request=request, batch_inferer=spy)
    if not manifest.tables:
        pytest.fail("Expected schema manifest to include tables")
    if spy.calls != 1:
        pytest.fail(f"Expected batch inferer to be called once, got {spy.calls}")

    catalog = get_target_metadata_service().system.catalog
    risk_table_keys = set(catalog.get("risk_factors").table_keys)
    hotspot_table_keys = set(catalog.get("hotspots").table_keys)
    expected_keys = sorted(risk_table_keys | hotspot_table_keys)
    if spy.seen != [tuple(expected_keys)]:
        pytest.fail(f"Expected batch inferer keys {expected_keys}, got {spy.seen}")
