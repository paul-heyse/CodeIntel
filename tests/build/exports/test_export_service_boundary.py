"""Tests for storage export service boundary usage in build exports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest

from codeintel.build.exports import common as export_common
from codeintel.storage.protocols import ExportRelation
from tests._helpers.assertions.expectation_assertions import expect_equal


@dataclass
class _StubExportService:
    relation: ExportRelation
    last_sql: str | None = None

    def build_export_relation(self, *, sql: str) -> ExportRelation:
        self.last_sql = sql
        return self.relation


@dataclass
class _StubGateway:
    exports: _StubExportService


def test_build_export_relation_uses_storage_export_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify build export relation creation stays in storage boundary."""
    relation = cast("ExportRelation", object())
    export_service = _StubExportService(relation=relation)
    gateway = _StubGateway(exports=export_service)

    monkeypatch.setattr(export_common, "build_export_expr", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(export_common, "compile_export_sql", lambda *_args, **_kwargs: "SELECT 1")

    result = export_common.build_export_relation(gateway, "analytics.function_metrics", 10, 0)

    expect_equal(result, relation)
    expect_equal(export_service.last_sql, "SELECT 1")
