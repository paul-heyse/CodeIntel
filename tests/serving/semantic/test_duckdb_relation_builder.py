"""Tests for DuckDB relation builder AST handling."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pyarrow as pa
import pytest
from sqlglot import exp, parse_one

from codeintel.serving.semantic.datasets import DatasetManifestEntry, DatasetManifestIndex
from codeintel.serving.semantic.duckdb_relation_builder import (
    DuckDBRelationQueryBuilderError,
    RelationBuildContext,
    RelationScanOptions,
    apply_query_ast,
    build_relation_plan,
)
from codeintel.serving.semantic.inventory import SchemaInventory
from codeintel.serving.semantic.models import FilterSpec
from codeintel.serving.semantic.specs import SemanticQuerySpec
from codeintel.serving.semantic.sqlglot_query_builder import build_sqlglot_query
from codeintel.storage.datasets.arrow_store import ArrowDatasetWriteOptions, write_dataset
from codeintel.storage.datasets.manifests import dataset_manifest_path, read_dataset_manifest
from tests._helpers.assertions.expectation_assertions import expect_equal

pytestmark = pytest.mark.no_runtime_env


def _demo_connection() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE demo (id INTEGER, label VARCHAR)")
    con.execute("INSERT INTO demo VALUES (1, 'one'), (2, 'two'), (3, 'three')")
    return con


def _write_metadata_dataset(
    tmp_path: Path,
) -> tuple[str, DatasetManifestIndex, SchemaInventory]:
    table_key = "analytics.demo"
    snapshot_id = "snap-1"
    table = pa.table({"id": [1, 2], "label": ["a", "b"]})
    metadata = {
        "codeintel.table_key": table_key,
        "codeintel.domain": "analytics",
        "codeintel.target": "demo_target",
        "codeintel.schema_hash": "demo_hash",
        "codeintel.schema_digest": "demo_digest",
        "codeintel.columns_json": {"id": "INTEGER", "label": "VARCHAR"},
        "codeintel.nullability_json": {"id": False, "label": True},
        "codeintel.primary_keys_json": ["id"],
        "codeintel.partition_columns_json": [],
        "codeintel.build_id": "demo-build",
        "codeintel.repo": "demo-repo",
        "codeintel.commit": "demo-commit",
        "codeintel.snapshot_id": snapshot_id,
        "codeintel.generated_at": "2025-01-01T00:00:00Z",
        "codeintel.hamilton.node": "demo_node",
        "codeintel.hamilton.graph_version": "demo-version",
        "codeintel.inputs_json": [],
    }
    write_dataset(
        dataset_root=tmp_path,
        table_key=table_key,
        snapshot_id=snapshot_id,
        data=table,
        options=ArrowDatasetWriteOptions(
            schema_metadata=metadata,
            persist_manifest=True,
        ),
    )
    manifest_path = dataset_manifest_path(
        dataset_root=tmp_path,
        table_key=table_key,
        snapshot_id=snapshot_id,
    )
    manifest = read_dataset_manifest(manifest_path)
    entry = DatasetManifestEntry(manifest=manifest, manifest_path=manifest_path)
    manifests = DatasetManifestIndex(by_table_key={table_key: entry})
    inventory = SchemaInventory.from_dataset_manifests(manifests)
    return table_key, manifests, inventory


def test_duckdb_ast_builder_filters_orders_and_paginates() -> None:
    """DuckDB AST builder applies filters, ordering, and pagination."""
    con = _demo_connection()
    try:
        relation = con.table("demo")
        spec = SemanticQuerySpec(
            view_id="demo.view",
            table_key="main.demo",
            allowed_columns=frozenset({"id", "label"}),
            columns=["id", "label"],
            filters=[FilterSpec(column="id", op="gte", value=2)],
            order_by=["-id"],
            limit=10,
            offset=0,
            column_types={"id": "INTEGER", "label": "VARCHAR"},
        )
        ast = build_sqlglot_query(spec=spec)
        rows = apply_query_ast(
            relation,
            ast=ast,
            allowed_columns=spec.allowed_columns,
            column_types=spec.column_types,
        ).fetchall()
        expect_equal(rows, expected=[(3, "three"), (2, "two")])
    finally:
        con.close()


def test_duckdb_ast_builder_supports_alias_and_lower() -> None:
    """DuckDB AST builder supports projection aliases and lower()."""
    con = _demo_connection()
    try:
        relation = con.table("demo")
        ast = parse_one(
            "SELECT lower(label) AS label_lower FROM main.demo",
            dialect="duckdb",
        )
        rows = apply_query_ast(
            relation,
            ast=ast,
            allowed_columns=frozenset({"label"}),
            column_types={"label": "VARCHAR"},
        ).fetchall()
        expect_equal(rows, expected=[("one",), ("two",), ("three",)])
    finally:
        con.close()


def test_duckdb_ast_builder_rejects_unknown_column() -> None:
    """Unknown columns in AST selections are rejected."""
    con = _demo_connection()
    try:
        relation = con.table("demo")
        ast = exp.select(exp.Column(this=exp.to_identifier("oops")))
        with pytest.raises(DuckDBRelationQueryBuilderError, match="Unknown select column"):
            apply_query_ast(
                relation,
                ast=ast,
                allowed_columns=frozenset({"id"}),
                column_types={"id": "INTEGER"},
            )
    finally:
        con.close()


def test_duckdb_relation_builder_uses_parquet_metadata(
    tmp_path: Path,
) -> None:
    """DuckDB relation plans should rely on schema metadata-derived column types."""
    table_key, manifests, inventory = _write_metadata_dataset(tmp_path)
    schema = inventory.require(table_key)
    column_types = {col.name: col.type for col in schema.columns}

    con = duckdb.connect(":memory:")
    try:
        spec = SemanticQuerySpec(
            view_id="demo.view",
            table_key=table_key,
            allowed_columns=frozenset({"id", "label"}),
            columns=["id", "label"],
            filters=[FilterSpec(column="id", op="gte", value=2)],
            order_by=["id"],
            limit=10,
            offset=0,
            column_types=column_types,
        )
        ast = build_sqlglot_query(spec=spec)
        relation = build_relation_plan(
            con=con,
            spec=spec,
            ast=ast,
            context=RelationBuildContext(
                dataset_manifests=manifests,
                scan_options=RelationScanOptions(batch_size=128),
                column_types=column_types,
                contract_schema=None,
            ),
        )
        expect_equal(relation.fetchall(), expected=[(2, "b")])
    finally:
        con.close()


def test_duckdb_ast_builder_rejects_invalid_operator() -> None:
    """Invalid operator usage should raise a builder error."""
    con = _demo_connection()
    try:
        relation = con.table("demo")
        table = exp.Table(this=exp.to_identifier("demo"), db=exp.to_identifier("main"))
        predicate = exp.Anonymous(
            this="contains",
            expressions=[
                exp.Column(this=exp.to_identifier("id")),
                exp.Literal.string("oops"),
            ],
        )
        ast = exp.select(exp.Column(this=exp.to_identifier("id"))).from_(table).where(predicate)
        with pytest.raises(
            DuckDBRelationQueryBuilderError,
            match="Operator contains is not supported for column type INTEGER",
        ):
            apply_query_ast(
                relation,
                ast=ast,
                allowed_columns=frozenset({"id"}),
                column_types={"id": "INTEGER"},
            )
    finally:
        con.close()
