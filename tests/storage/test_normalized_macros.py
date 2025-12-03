"""Ensure normalized macros execute for all registered datasets."""

from __future__ import annotations

import io

import pytest

from codeintel.pipeline.export.export_jsonl import NORMALIZED_MACROS
from codeintel.storage.gateway import DuckDBError, StorageGateway
from codeintel.storage.normalized_macros import render_macro
from codeintel.storage.sql_helpers import safe_macro_call

pytestmark = pytest.mark.smoke


def test_normalized_macros_execute(fresh_gateway: StorageGateway) -> None:
    """
    Every normalized macro should execute with a zero-row limit.

    This guards against missing macro definitions or signature drift.
    """
    con = fresh_gateway.con
    failures: list[str] = []
    for table_key, macro in sorted(NORMALIZED_MACROS.items()):
        try:
            sql, params = safe_macro_call(
                macro, [table_key, 0], allowed=set(NORMALIZED_MACROS.values())
            )
            con.execute(sql, params)
        except (DuckDBError, RuntimeError, ValueError) as exc:
            failures.append(f"{table_key} via {macro}: {exc}")
    if failures:
        message = "Normalized macro failures: " + "; ".join(failures)
        pytest.fail(message)


def test_render_macro_valid_table_key() -> None:
    """Verify render_macro returns RenderedMacro for valid table key."""
    result = render_macro("core.ast_nodes")

    assert result.macro_name.startswith("metadata.normalized_")
    assert "CREATE OR REPLACE MACRO" in result.ddl


def test_render_macro_unknown_table_key_raises() -> None:
    """Verify render_macro raises KeyError for unknown table key."""
    with pytest.raises(KeyError, match="Unknown table key"):
        render_macro("unknown.table_that_does_not_exist")


def test_render_macro_with_custom_limit() -> None:
    """Verify render_macro accepts custom default_limit."""
    result = render_macro("core.ast_nodes", default_limit=1000)

    assert ":= 1000" in result.ddl


def test_render_macro_includes_date_cast() -> None:
    """Verify render_macro correctly casts DATE columns through internal _cast_expression."""
    # Find a table with DATE columns (core.file_state likely has date columns)
    # We test the output DDL to verify DATE handling
    result = render_macro("analytics.function_history")

    # The DDL should contain proper casting for timestamp columns
    assert "CAST" in result.ddl


def test_render_macro_includes_goid_cast() -> None:
    """Verify render_macro casts goid_h128 columns to BIGINT."""
    result = render_macro("analytics.function_profile")

    # Should include BIGINT cast for goid columns
    assert "AS BIGINT" in result.ddl
    assert "goid_h128" in result.ddl.lower()


def test_render_macro_outputs_ddl_to_buffer() -> None:
    """Verify render_macro produces DDL strings for provided tables."""
    buffer = io.StringIO()

    for table in ("core.ast_nodes", "core.goids"):
        rendered = render_macro(table)
        buffer.write(rendered.ddl)

    output = buffer.getvalue()
    assert "metadata.normalized_ast_nodes" in output
    assert "metadata.normalized_goids" in output
