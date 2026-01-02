"""Tests for PR-80: batch schema inference for schema compilation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

from codeintel.build.schemas import declared_schema_provider
from codeintel.build.schemas.compile import (
    SchemaManifestContext,
    SchemaManifestRequest,
    compile_schema_manifest,
)
from codeintel.core.schemas.primitives import Column, TableSchema
from codeintel.runtime.runtime_bundle import RuntimeBundle
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


def test_pr80_schema_compile_uses_batch_inference(hamilton_runtime: RuntimeBundle) -> None:
    """Ensure compile_schema_manifest pre-infers schemas in a single batch call."""
    provider = declared_schema_provider(runtime=hamilton_runtime)
    spy = _SpyBatchInferer()

    request = SchemaManifestRequest(
        targets=("function_types",),
        infer_native=True,
        batch_infer_native=True,
        stable=True,
    )

    schema_index = hamilton_runtime.schema_index
    if schema_index is None:
        pytest.fail("Runtime bundle missing schema_index")
    manifest = compile_schema_manifest(
        provider=provider,
        context=SchemaManifestContext(
            catalog=hamilton_runtime.catalog,
            schema_index=schema_index,
            tag_query=hamilton_runtime.tag_query,
        ),
        request=request,
        batch_inferer=spy,
    )
    if not manifest.tables:
        pytest.fail("Expected schema manifest to include tables")
    if spy.calls != 1:
        pytest.fail(f"Expected batch inferer to be called once, got {spy.calls}")

    function_table_keys = set(hamilton_runtime.catalog.get("function_types").table_keys)
    expected_keys = sorted(function_table_keys)
    if spy.seen != [tuple(expected_keys)]:
        pytest.fail(f"Expected batch inferer keys {expected_keys}, got {spy.seen}")
