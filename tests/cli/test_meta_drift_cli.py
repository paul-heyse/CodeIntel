"""Tests for meta drift CLI output."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pyarrow as pa

from codeintel.core.columnar.ipc import schema_to_ipc_payload
from codeintel.core.time import utc_now
from codeintel.storage.tracking.schema_catalog_models import SchemaObservationRecord
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_is_instance,
    expect_true,
)
from tests._helpers.cli import assert_success
from tests._helpers.cli_project import CLIProjectContext

if TYPE_CHECKING:
    from collections.abc import Callable

    from tests._helpers.cli import CliResult


def _record_drift_observation(ctx: CLIProjectContext) -> None:
    table_key = "analytics.demo_drift"
    observation = SchemaObservationRecord(
        table_key=table_key,
        schema_digest="digest",
        schema_hash="hash",
        arrow_schema_ipc_b64=schema_to_ipc_payload(pa.schema([("id", pa.int64())])),
        drift_summary={"missing_columns": ["owner"], "extra_columns": ["extra"]},
        observed_at=utc_now(),
    )
    if ctx.gateway is None:
        msg = "Expected gateway to be initialized for CLI project context"
        raise ValueError(msg)
    ctx.gateway.schemas.record_schema_observations_batch([observation])


def test_meta_drift_cli_reports_latest(
    cli_project_ctx: CLIProjectContext,
    cli_project_runner: Callable[[list[str]], CliResult],
) -> None:
    """Meta drift CLI should surface latest drift summaries."""
    _record_drift_observation(cli_project_ctx)

    result = cli_project_runner(["meta", "drift", "--output-format", "json"])
    assert_success(result)

    payload = json.loads(result.stdout)
    data = payload.get("data")
    expect_is_instance(data, dict)
    latest = data.get("latest")
    expect_is_instance(latest, list)
    expect_true(len(latest) >= 1, message="Expected drift summaries in latest list")
    first = latest[0]
    expect_is_instance(first, dict)
    expect_equal(first.get("table_key"), expected="analytics.demo_drift")
