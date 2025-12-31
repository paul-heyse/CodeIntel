"""Tests for storage export service boundary usage in build exports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import duckdb

from codeintel.build.exports import common as export_common
from codeintel.storage.duckdb_types import DuckDBRelation
from codeintel.storage.exports.service import ExportService
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.gateway.minimal import MinimalStorageGateway
from codeintel.storage.protocols import ExportRelation
from tests._helpers.assertions.expectation_assertions import expect_equal, expect_is_not_none
from tests._helpers.schemas import ensure_production_schemas

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

    from codeintel.storage.duckdb_policy_backend import DuckDBPolicyBackend


@dataclass
class _RecordingExportService:
    gateway: MinimalStorageGateway
    last_relation: DuckDBRelation | None = None

    def build_export_relation(self, *, relation: DuckDBRelation) -> ExportRelation:
        self.last_relation = relation
        return ExportService(self.gateway).build_export_relation(relation=relation)


class _RecordingGateway:
    def __init__(self, con: DuckDBPyConnection) -> None:
        self._base = MinimalStorageGateway(con)
        self._exports = _RecordingExportService(self._base)

    @property
    def con(self) -> DuckDBPyConnection:
        return self._base.con

    @property
    def policy(self) -> DuckDBPolicyBackend:
        return self._base.policy

    @property
    def exports(self) -> _RecordingExportService:
        return self._exports


def _seed_export_table(con: DuckDBPyConnection) -> None:
    ensure_production_schemas(con)
    con.execute("INSERT INTO analytics.function_metrics (function_goid_h128) VALUES (1)")


def test_build_export_relation_uses_storage_export_service() -> None:
    """Verify build export relation creation stays in storage boundary."""
    con = duckdb.connect(":memory:")
    _seed_export_table(con)
    gateway = _RecordingGateway(con)

    result = export_common.build_export_relation(
        cast("StorageGateway", gateway),
        "analytics.function_metrics",
        10,
        0,
    )

    last_relation = expect_is_not_none(gateway.exports.last_relation)
    expect_is_not_none(last_relation)
    row = expect_is_not_none(result.fetchone())
    expect_equal(row[0], 1)
