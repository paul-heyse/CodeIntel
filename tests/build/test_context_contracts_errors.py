"""Tests for execution context, contracts, and build error handling."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.build.context import ContextResources, TargetExecutionContext
from codeintel.build.contracts import ArtifactSpec, OutputContract
from codeintel.build.errors import (
    BuildErrorCollection,
    ColumnCountMismatchError,
    SchemaNotFoundError,
)
from codeintel.build.parameters import TargetParameters
from codeintel.build.targets import OutputTarget
from codeintel.config.datasets.primitives import Column, TableSchema
from tests._helpers import make_build_paths, make_snapshot


def _context_with_contract(contract: OutputContract, tmp_path: Path) -> TargetExecutionContext:
    target = OutputTarget(
        name="demo",
        module="analytics",
        plugin="demo_plugin",
        contract=contract,
        tables=contract.table_keys,
    )
    return TargetExecutionContext(
        target=target,
        snapshot=make_snapshot(tmp_path),
        paths=make_build_paths(tmp_path),
        resources=ContextResources(),
        parameters=TargetParameters({"limit": 5}),
    )


def test_artifact_path_resolution(tmp_path: Path) -> None:
    """Artifact placeholders resolve to BuildPaths locations."""
    contract = OutputContract(artifacts=(ArtifactSpec("scip_index", "{scip_dir}/index.scip"),))
    ctx = _context_with_contract(contract, tmp_path)

    resolved = ctx.artifact_path("scip_index")
    assert resolved == make_build_paths(tmp_path).scip_dir / "index.scip"


def test_write_table_validation_and_recording(tmp_path: Path) -> None:
    """Valid writes are recorded and validated against schemas."""
    schema = TableSchema(
        schema="core",
        name="items",
        columns=[Column("id", "INTEGER", nullable=False), Column("name", "VARCHAR")],
    )
    contract = OutputContract(tables=(schema,))
    ctx = _context_with_contract(contract, tmp_path)
    rows = [(1, "a"), (2, "b")]

    written = ctx.write_table("core.items", rows)
    assert written == 2
    record = ctx.written_tables["core.items"]
    assert record.validated is True
    assert record.rows == rows


def test_write_table_legacy_tables_skip_schema(tmp_path: Path) -> None:
    """Legacy tables in target.tables bypass schema lookup."""
    contract = OutputContract()
    target = OutputTarget(
        name="legacy",
        module="analytics",
        plugin="legacy_plugin",
        contract=contract,
        tables=("core.legacy",),
    )
    ctx = TargetExecutionContext(
        target=target,
        snapshot=make_snapshot(tmp_path),
        paths=make_build_paths(tmp_path),
        resources=ContextResources(),
        parameters=TargetParameters.empty(),
    )

    rows = [(1, "x")]
    written = ctx.write_table("core.legacy", rows)
    assert written == 1
    assert ctx.written_tables["core.legacy"].validated is True


def test_write_table_missing_schema_raises(tmp_path: Path) -> None:
    """Missing schema raises SchemaNotFoundError."""
    schema = TableSchema(
        schema="core",
        name="items",
        columns=[Column("id", "INTEGER")],
    )
    contract = OutputContract(tables=(schema,))
    ctx = _context_with_contract(contract, tmp_path)

    with pytest.raises(SchemaNotFoundError):
        ctx.write_table("core.missing", [(1,)])


def test_write_table_column_mismatch_raises(tmp_path: Path) -> None:
    """Column count mismatches raise detailed errors."""
    schema = TableSchema(
        schema="core",
        name="items",
        columns=[Column("id", "INTEGER"), Column("name", "VARCHAR")],
    )
    contract = OutputContract(tables=(schema,))
    ctx = _context_with_contract(contract, tmp_path)

    with pytest.raises(ColumnCountMismatchError) as exc_info:
        ctx.write_table("core.items", [(1,), (2, "ok")])
    assert "core.items" in str(exc_info.value)
    assert exc_info.value.actual == 1
    assert exc_info.value.expected == 2


def test_gateway_property_requires_resource(tmp_path: Path) -> None:
    """Gateway property enforces presence of configured gateway."""
    contract = OutputContract()
    ctx = TargetExecutionContext(
        target=OutputTarget(
            name="noop",
            module="analytics",
            plugin="noop",
            contract=contract,
        ),
        snapshot=make_snapshot(tmp_path),
        paths=make_build_paths(tmp_path),
        resources=ContextResources(),
        parameters=TargetParameters.empty(),
    )

    with pytest.raises(RuntimeError):
        _ = ctx.gateway


def test_contract_validation_reports_duplicates() -> None:
    """Contracts report duplicate table keys and artifacts."""
    schema = TableSchema(schema="core", name="items", columns=[Column("id", "INTEGER")])
    contract = OutputContract(
        tables=(
            schema,
            TableSchema(schema="core", name="items", columns=[Column("id", "INTEGER")]),
        ),
        artifacts=(
            ArtifactSpec("a1", "{build_dir}/a1"),
            ArtifactSpec("a1", "{build_dir}/a1"),
        ),
    )

    errors = contract.validate()
    assert "Duplicate table key: core.items" in errors
    assert "Duplicate artifact name: a1" in errors


def test_build_error_collection_formats_and_filters() -> None:
    """Error collection aggregates, filters, formats, and raises."""
    collection = BuildErrorCollection()
    collection.add(SchemaNotFoundError("target", "core.table"))
    collection.add_warning("non-fatal")

    assert collection.has_errors is True
    assert collection.has_warnings is True
    assert len(collection.by_type(SchemaNotFoundError)) == 1
    summary = collection.format_summary()
    assert "SCHEMANOTFOUNDERROR" in summary
    assert "Hint: Add schema for 'core.table'" in summary

    with pytest.raises(SchemaNotFoundError):
        collection.raise_if_errors()
