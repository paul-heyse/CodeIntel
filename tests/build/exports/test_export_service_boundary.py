"""Tests for storage export service boundary usage in build exports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import duckdb

from codeintel.build.exports import common as export_common
from codeintel.storage.exports.service import ExportService
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.gateway.minimal import MinimalStorageGateway
from codeintel.storage.protocols import ExportRelation
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_in,
    expect_is_not_none,
)

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

    from codeintel.storage.duckdb_policy_backend import DuckDBPolicyBackend
    from codeintel.storage.ibis_adapter import IbisGateway


@dataclass
class _RecordingExportService:
    gateway: MinimalStorageGateway
    last_sql: str | None = None

    def build_export_relation(self, *, sql: str) -> ExportRelation:
        self.last_sql = sql
        return ExportService(self.gateway).build_export_relation(sql=sql)


class _RecordingGateway:
    def __init__(self, con: DuckDBPyConnection) -> None:
        self._base = MinimalStorageGateway(con)
        self._exports = _RecordingExportService(self._base)

    @property
    def con(self) -> DuckDBPyConnection:
        return self._base.con

    @property
    def ibis(self) -> IbisGateway:
        return self._base.ibis

    @property
    def policy(self) -> DuckDBPolicyBackend:
        return self._base.policy

    @property
    def exports(self) -> _RecordingExportService:
        return self._exports


def _seed_export_table(con: DuckDBPyConnection) -> None:
    con.execute("CREATE SCHEMA analytics")
    con.execute("CREATE TABLE analytics.function_metrics (id INTEGER)")
    con.execute("INSERT INTO analytics.function_metrics VALUES (1)")


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

    last_sql = expect_is_not_none(gateway.exports.last_sql)
    expect_in("analytics.function_metrics", last_sql)
    expect_equal(result.fetchone(), (1,))
