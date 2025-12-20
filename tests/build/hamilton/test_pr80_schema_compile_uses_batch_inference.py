"""Tests for PR-80: batch schema inference for schema compilation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

from codeintel.build.contracts import placeholder_table_schema
from codeintel.build.schemas.compile import SchemaManifestRequest, compile_schema_manifest
from codeintel.build.schemas.provider_declared import declared_schema_provider
from codeintel.build.target_metadata import get_target_metadata_service

if TYPE_CHECKING:
    from collections.abc import Iterable

    from codeintel.core.schemas.primitives import TableSchema
    from codeintel.core.schemas.provider import SchemaProvider


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
        return {key: placeholder_table_schema(key) for key in keys}


def test_pr80_schema_compile_uses_batch_inference() -> None:
    """Ensure compile_schema_manifest pre-infers schemas in a single batch call."""
    provider = declared_schema_provider()
    spy = _SpyBatchInferer()

    request = SchemaManifestRequest(
        targets=("risk_factors", "hotspots"),
        only_native=True,
        infer_native=True,
        batch_infer_native=True,
        stable=True,
    )

    manifest = compile_schema_manifest(provider=provider, request=request, batch_inferer=spy)
    if not manifest.tables:
        pytest.fail("Expected schema manifest to include tables")
    if spy.calls != 1:
        pytest.fail(f"Expected batch inferer to be called once, got {spy.calls}")

    graph = get_target_metadata_service().system.graph
    risk_table_keys = set(graph.get("risk_factors").contract.table_keys)
    hotspot_table_keys = set(graph.get("hotspots").contract.table_keys)
    expected_keys = sorted(risk_table_keys | hotspot_table_keys)
    if spy.seen != [tuple(expected_keys)]:
        pytest.fail(f"Expected batch inferer keys {expected_keys}, got {spy.seen}")
