"""Smoke test for exporting datasets to Document Output."""

from __future__ import annotations

from pathlib import Path

import duckdb

from codeintel.docs_export.export_jsonl import export_all_jsonl
from codeintel.docs_export.export_parquet import export_all_parquet


def _seed_minimal_db(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("CREATE SCHEMA IF NOT EXISTS core;")
    con.execute("CREATE SCHEMA IF NOT EXISTS graph;")
    con.execute("CREATE SCHEMA IF NOT EXISTS analytics;")
    con.execute(
        """
        CREATE TABLE core.repo_map (
            repo TEXT,
            commit TEXT,
            modules JSON,
            overlays JSON,
            generated_at TIMESTAMP
        );
        """
    )

    con.execute(
        """
        CREATE TABLE core.goids (
            goid_h128 DECIMAL(38,0),
            urn TEXT,
            repo TEXT,
            commit TEXT
        );
        """
    )
    con.execute(
        """
        CREATE TABLE core.goid_crosswalk (
            repo TEXT,
            commit TEXT,
            goid TEXT
        );
        """
    )
    con.execute(
        """
        CREATE TABLE core.modules (
            repo TEXT,
            commit TEXT,
            path TEXT,
            module TEXT,
            language TEXT,
            tags JSON,
            owners JSON
        );
        """
    )
    con.execute(
        """
        CREATE TABLE graph.call_graph_nodes (
            goid_h128 DECIMAL(38,0),
            repo TEXT,
            commit TEXT,
            rel_path TEXT,
            qualname TEXT
        );
        """
    )
    con.execute(
        """
        CREATE TABLE graph.call_graph_edges (
            repo TEXT,
            commit TEXT,
            caller_goid_h128 DECIMAL(38,0),
            callee_goid_h128 DECIMAL(38,0)
        );
        """
    )
    con.execute(
        """
        CREATE TABLE graph.cfg_blocks (
            function_goid_h128 DECIMAL(38,0),
            block_idx INTEGER
        );
        """
    )
    con.execute(
        """
        CREATE TABLE graph.cfg_edges (
            src_block_id TEXT,
            dst_block_id TEXT
        );
        """
    )
    con.execute(
        """
        CREATE TABLE graph.dfg_edges (
            src_block_id TEXT,
            dst_block_id TEXT
        );
        """
    )
    con.execute(
        """
        CREATE TABLE graph.import_graph_edges (
            repo TEXT,
            commit TEXT,
            src_module TEXT,
            dst_module TEXT
        );
        """
    )
    con.execute(
        """
        CREATE TABLE graph.symbol_use_edges (
            symbol TEXT,
            def_path TEXT,
            use_path TEXT,
            same_file BOOLEAN,
            same_module BOOLEAN
        );
        """
    )
    con.execute(
        """
        CREATE TABLE core.ast_nodes(path TEXT);
        """
    )
    con.execute(
        """
        CREATE TABLE core.ast_metrics(rel_path TEXT, function_count INTEGER);
        """
    )
    con.execute(
        """
        CREATE TABLE core.cst_nodes(path TEXT);
        """
    )
    con.execute(
        """
        CREATE TABLE core.docstrings(rel_path TEXT, qualname TEXT, doc JSON);
        """
    )
    con.execute(
        """
        CREATE TABLE analytics.config_values(key TEXT, value TEXT);
        """
    )
    con.execute(
        """
        CREATE TABLE analytics.static_diagnostics(rel_path TEXT, total_errors INTEGER);
        """
    )
    con.execute(
        """
        CREATE TABLE analytics.hotspots(rel_path TEXT, score DOUBLE);
        """
    )
    con.execute(
        """
        CREATE TABLE analytics.typedness(path TEXT, annotation_ratio JSON);
        """
    )
    con.execute(
        """
        CREATE TABLE analytics.function_metrics(function_goid_h128 DECIMAL(38,0), repo TEXT, commit TEXT, rel_path TEXT, qualname TEXT);
        """
    )
    con.execute(
        """
        CREATE TABLE analytics.function_types(function_goid_h128 DECIMAL(38,0), typedness_bucket TEXT);
        """
    )
    con.execute(
        """
        CREATE TABLE analytics.coverage_lines(rel_path TEXT);
        """
    )
    con.execute(
        """
        CREATE TABLE analytics.coverage_functions(function_goid_h128 DECIMAL(38,0), coverage_ratio DOUBLE);
        """
    )
    con.execute(
        """
        CREATE TABLE analytics.test_catalog(test_id TEXT);
        """
    )
    con.execute(
        """
        CREATE TABLE analytics.test_coverage_edges(test_id TEXT, function_goid_h128 DECIMAL(38,0));
        """
    )
    con.execute(
        """
        CREATE TABLE analytics.goid_risk_factors(function_goid_h128 DECIMAL(38,0), repo TEXT, commit TEXT, rel_path TEXT, qualname TEXT);
        """
    )

    con.execute(
        """
        INSERT INTO core.goids VALUES (1, 'urn:foo', 'r', 'c');
        INSERT INTO core.goid_crosswalk VALUES ('r', 'c', 'urn:foo');
        INSERT INTO core.modules VALUES ('r','c','foo.py','pkg.foo','python','[]','[]');
        INSERT INTO core.repo_map VALUES ('r','c','{\"pkg.foo\": \"foo.py\"}','{}', CURRENT_TIMESTAMP);
        INSERT INTO graph.call_graph_nodes VALUES (1,'r','c','foo.py','foo');
        INSERT INTO graph.call_graph_edges VALUES ('r','c',1,1);
        INSERT INTO graph.cfg_blocks VALUES (1,0);
        INSERT INTO graph.import_graph_edges VALUES ('r','c','pkg.foo','pkg.bar');
        INSERT INTO core.docstrings VALUES ('foo.py','foo','{\"summary\": \"demo\"}');
        INSERT INTO analytics.function_metrics VALUES (1,'r','c','foo.py','foo');
        INSERT INTO analytics.function_types VALUES (1,'typed');
        INSERT INTO analytics.coverage_functions VALUES (1,1.0);
        INSERT INTO analytics.test_catalog VALUES ('t1');
        INSERT INTO analytics.test_coverage_edges VALUES ('t1',1);
        INSERT INTO analytics.goid_risk_factors VALUES (1,'r','c','foo.py','foo');
        """
    )


def test_export_all_writes_expected_files(tmp_path: Path) -> None:
    """
    Seed a minimal DB and verify Parquet/JSONL exports are produced.

    This ensures the export mappings are usable end-to-end.

    Raises
    ------
    AssertionError
        If any expected export is missing after running both exporters.
    """
    db_path = tmp_path / "test.duckdb"
    con = duckdb.connect(str(db_path))
    _seed_minimal_db(con)

    output_dir = tmp_path / "Document Output"
    export_all_parquet(con, output_dir)
    export_all_jsonl(con, output_dir)

    expected_basenames = {
        "goids.parquet",
        "goid_crosswalk.parquet",
        "call_graph_nodes.parquet",
        "call_graph_edges.parquet",
        "cfg_blocks.parquet",
        "import_graph_edges.parquet",
        "docstrings.parquet",
        "function_metrics.parquet",
        "function_types.parquet",
        "coverage_functions.parquet",
        "test_catalog.parquet",
        "test_coverage_edges.parquet",
        "goid_risk_factors.parquet",
        "goids.jsonl",
        "goid_crosswalk.jsonl",
        "call_graph_nodes.jsonl",
        "call_graph_edges.jsonl",
        "cfg_blocks.jsonl",
        "import_graph_edges.jsonl",
        "docstrings.jsonl",
        "function_metrics.jsonl",
        "function_types.jsonl",
        "coverage_functions.jsonl",
        "test_catalog.jsonl",
        "test_coverage_edges.jsonl",
        "goid_risk_factors.jsonl",
        "repo_map.json",
        "index.json",
    }

    written = {p.name for p in output_dir.iterdir() if p.is_file()}

    missing = expected_basenames - written
    if missing:
        message = f"Expected exports missing: {sorted(missing)}"
        raise AssertionError(message)
