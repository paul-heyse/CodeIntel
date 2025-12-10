"""Tests for ingestion macros and tooling service wiring."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

from codeintel.ingestion.engine.infrastructure import ToolRunner
from codeintel.ingestion.infrastructure.macros import INGEST_MACRO_TABLES, macro_exists
from tests._helpers.assertions import (
    expect_equal,
    expect_in,
    expect_is_instance,
    expect_true,
)
from tests._helpers.constants import DEFAULT_COMMIT, DEFAULT_REPO
from tests._helpers.fakes.tools import FakeToolRunner, FakeToolService, FakeToolServiceConfig
from tests._helpers.macros import (
    assert_all_ingest_macros,
    assert_ingest_macros_registered,
    assert_macro_perf,
    measure_ingest_perf,
)
from tests._helpers.rows import function_metrics_row

EXPECTED_TABLE_KEY_PARTS = 2
PERF_TABLE_KEYS = tuple(
    table_key
    for table_key in sorted(INGEST_MACRO_TABLES)
    if table_key in {"analytics.function_metrics", "analytics.function_effects"}
)


def _function_effects_row(
    repo: str = DEFAULT_REPO,
    commit: str = DEFAULT_COMMIT,
    goid: int = 1,
) -> tuple[object, ...]:
    """Row payload matching analytics.function_effects schema order."""
    created_at = datetime.now(tz=UTC).isoformat()
    return (
        repo,
        commit,
        goid,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        1.0,
        "{}",
        created_at,
    )


def test_ingest_macro_tables_cover_expected_entries() -> None:
    """INGEST_MACRO_TABLES should expose required table keys."""
    expect_is_instance(INGEST_MACRO_TABLES, frozenset)
    required = {
        "core.ast_nodes",
        "core.cst_nodes",
        "core.docstrings",
        "core.modules",
        "analytics.coverage_lines",
        "analytics.typedness",
        "graph.call_graph_edges",
        "graph.call_graph_nodes",
    }
    for table_key in required:
        expect_in(table_key, INGEST_MACRO_TABLES)


@pytest.mark.parametrize("table_key", sorted(INGEST_MACRO_TABLES))
def test_ingest_macro_table_keys_are_well_formed(table_key: str) -> None:
    """Each table key should be well-formed schema.table."""
    parts = table_key.split(".")
    expect_equal(
        len(parts),
        EXPECTED_TABLE_KEY_PARTS,
        label=f"Table key '{table_key}' should have format 'schema.table'",
    )
    schema, table = parts
    expect_true(bool(schema), message=f"Table key '{table_key}' has empty schema")
    expect_true(bool(table), message=f"Table key '{table_key}' has empty table name")


def test_ingest_macros_registered_and_listed(fresh_gateway) -> None:
    """Ensure ingest macros are present for all registered tables."""
    assert_all_ingest_macros(fresh_gateway.con)
    assert_ingest_macros_registered(fresh_gateway.con)


def test_macro_exists_handles_malformed_table_key(fresh_gateway) -> None:
    """macro_exists should raise on table keys without a schema prefix."""
    with pytest.raises(ValueError, match="not enough values to unpack"):
        macro_exists(fresh_gateway.con, "no_dot_in_name")


@pytest.mark.parametrize("table_key", PERF_TABLE_KEYS)
def test_ingest_macro_perf_with_prepared_statements(
    fresh_gateway, table_key: str
) -> None:
    """Macro ingest should remain within acceptable bounds versus prepared statements."""
    if table_key == "analytics.function_metrics":
        row = function_metrics_row(
            goid=101,
            rel_path="pkg/mod.py",
            qualname="pkg.mod:func",
            snapshot=(DEFAULT_REPO, DEFAULT_COMMIT),
        ).to_tuple()
    elif table_key == "analytics.function_effects":
        row = _function_effects_row()
    else:  # pragma: no cover
        pytest.skip(f"Perf measurement unsupported for {table_key}")

    result = measure_ingest_perf(fresh_gateway, table_key, [row])

    assert_macro_perf(result)


def test_fake_tool_service_uses_shared_runner(tmp_path: Path) -> None:
    """FakeToolService should wire the shared FakeToolRunner without subprocesses."""
    config = FakeToolServiceConfig(pyright_errors={"main.py": 2})
    service = FakeToolService(config=config, cache_dir=tmp_path / "cache")

    result = asyncio.run(service.run_pyright(tmp_path))

    expect_equal(result, {"main.py": 2})
    expect_is_instance(service.runner, FakeToolRunner)
    expect_is_instance(service.runner, ToolRunner)
