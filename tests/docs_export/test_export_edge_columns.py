"""Ensure graph edge exports include repo/commit columns."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb

from codeintel.config.schemas.tables import TABLE_SCHEMAS
from codeintel.docs_export.export_jsonl import export_jsonl_for_table
from tests._helpers.fixtures import provision_graph_ready_repo


def _setup_edge_table(con: duckdb.DuckDBPyConnection, table: str) -> None:
    con.execute("CREATE SCHEMA IF NOT EXISTS graph;")
    schema = TABLE_SCHEMAS[table]
    cols = ", ".join(f"{col.name} {col.type}" for col in schema.columns)
    con.execute(f"DROP TABLE IF EXISTS {table};")
    con.execute(f"CREATE TABLE {table} ({cols});")


def test_call_graph_edges_export_includes_repo_commit(tmp_path: Path) -> None:
    """
    Exported call graph edges should carry repo/commit columns.

    Raises
    ------
    AssertionError
        If repo/commit columns are missing or keys drift.
    """
    ctx = provision_graph_ready_repo(tmp_path, repo="r1", commit="c1")
    gateway = ctx.gateway
    con = gateway.con
    table = "graph.call_graph_edges"
    _setup_edge_table(con, table)
    con.execute(
        """
        INSERT INTO graph.call_graph_edges VALUES
        ('r1','c1',1,2,'a.py',1,0,'python','direct','local_name',1.0,'{}')
        """
    )

    out = tmp_path / "call_graph_edges.jsonl"
    export_jsonl_for_table(gateway, table, out)

    content = out.read_text(encoding="utf-8").splitlines()
    if not content:
        message = "Expected exported rows"
        raise AssertionError(message)
    first = json.loads(content[0])
    expected_keys = TABLE_SCHEMAS[table].column_names()
    if set(first) != set(expected_keys):
        unexpected = set(first)
        message = f"Unexpected keys {unexpected}"
        raise AssertionError(message)
    if first["repo"] != "r1" or first["commit"] != "c1":
        message = "Repo/commit missing in export"
        raise AssertionError(message)


def test_import_graph_edges_export_includes_repo_commit(tmp_path: Path) -> None:
    """
    Exported import graph edges should carry repo/commit columns.

    Raises
    ------
    AssertionError
        If repo/commit columns are missing or keys drift.
    """
    ctx = provision_graph_ready_repo(tmp_path, repo="r1", commit="c1")
    gateway = ctx.gateway
    con = gateway.con
    table = "graph.import_graph_edges"
    _setup_edge_table(con, table)
    con.execute(
        """
        INSERT INTO graph.import_graph_edges VALUES
        ('r1','c1','pkg.a','pkg.b',1,2,0,NULL)
        """
    )

    out = tmp_path / "import_graph_edges.jsonl"
    export_jsonl_for_table(gateway, table, out)

    content = out.read_text(encoding="utf-8").splitlines()
    if not content:
        message = "Expected exported rows"
        raise AssertionError(message)
    first = json.loads(content[0])
    expected_keys = TABLE_SCHEMAS[table].column_names()
    if set(first) != set(expected_keys):
        unexpected = set(first)
        message = f"Unexpected keys {unexpected}"
        raise AssertionError(message)
    if first["repo"] != "r1" or first["commit"] != "c1":
        message = "Repo/commit missing in export"
        raise AssertionError(message)
