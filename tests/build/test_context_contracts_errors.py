"""Tests for execution context, contracts, and build error handling."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from codeintel.build.context import ContextResources, TargetExecutionContext
from codeintel.build.context_base import BuildContext
from codeintel.build.contracts import ArtifactSpec, OutputContract
from codeintel.build.errors import (
    BuildErrorCollection,
    ColumnCountMismatchError,
    SchemaNotFoundError,
)
from codeintel.build.parameters import TargetParameters
from codeintel.build.targets import OutputTarget
from codeintel.config.datasets.primitives import Column, TableSchema
from tests._helpers import build_test_gateway, make_build_paths, make_snapshot
from tests._helpers.assertions import expect_equal, expect_in, expect_true

if TYPE_CHECKING:
    from pathlib import Path


def _context_with_contract(contract: OutputContract, tmp_path: Path) -> TargetExecutionContext:
    target = OutputTarget(
        name="demo",
        module="analytics",
        plugin="demo_plugin",
        contract=contract,
    )
    snapshot = make_snapshot(tmp_path)
    build_ctx = BuildContext(
        gateway=build_test_gateway(),
        snapshot=snapshot,
        paths=make_build_paths(tmp_path),
    )
    return TargetExecutionContext(
        build_ctx=build_ctx,
        target=target,
        resources=ContextResources(),
        parameters=TargetParameters({"limit": 5}),
    )


def test_artifact_path_resolution(tmp_path: Path) -> None:
    """Artifact placeholders resolve to BuildPaths locations."""
    contract = OutputContract(artifacts=(ArtifactSpec("scip_index", "{scip_dir}/index.scip"),))
    ctx = _context_with_contract(contract, tmp_path)

    resolved = ctx.artifact_path("scip_index")
    expect_equal(resolved, make_build_paths(tmp_path).scip_dir / "index.scip")


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
    expect_equal(written, 2)
    record = ctx.written_tables["core.items"]
    expect_true(record.validated is True)
    expect_equal(record.rows, rows)


def test_write_table_requires_contract_schema(tmp_path: Path) -> None:
    """Writing to table not in contract raises SchemaNotFoundError."""
    target = OutputTarget.from_tables(
        name="demo",
        module="analytics",
        plugin="demo_plugin",
        tables=("core.declared",),
    )
    snapshot = make_snapshot(tmp_path)
    ctx = TargetExecutionContext(
        build_ctx=BuildContext(
            gateway=build_test_gateway(),
            snapshot=snapshot,
            paths=make_build_paths(tmp_path),
        ),
        target=target,
        resources=ContextResources(),
        parameters=TargetParameters.empty(),
    )

    with pytest.raises(SchemaNotFoundError):
        ctx.write_table("core.undeclared", [(1, "x")])


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
    expect_true("core.items" in str(exc_info.value))
    expect_equal(exc_info.value.actual, 1)
    expect_equal(exc_info.value.expected, 2)


def test_gateway_property_returns_build_ctx_gateway(tmp_path: Path) -> None:
    """Gateway property delegates to build context gateway."""
    contract = OutputContract()
    gateway = build_test_gateway()
    snapshot = make_snapshot(tmp_path)
    ctx = TargetExecutionContext(
        build_ctx=BuildContext(
            gateway=gateway,
            snapshot=snapshot,
            paths=make_build_paths(tmp_path),
        ),
        target=OutputTarget(
            name="noop",
            module="analytics",
            plugin="noop",
            contract=contract,
        ),
        resources=ContextResources(),
        parameters=TargetParameters.empty(),
    )

    expect_equal(ctx.gateway, gateway)


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
    expect_true("Duplicate table key: core.items" in errors)
    expect_true("Duplicate artifact name: a1" in errors)


def test_build_error_collection_formats_and_filters() -> None:
    """Error collection aggregates, filters, formats, and raises."""
    collection = BuildErrorCollection()
    collection.add(SchemaNotFoundError("target", "core.table"))
    collection.add_warning("non-fatal")

    expect_true(collection.has_errors is True)
    expect_true(collection.has_warnings is True)
    expect_equal(len(collection.by_type(SchemaNotFoundError)), 1)
    summary = collection.format_summary()
    expect_in("SCHEMANOTFOUNDERROR", summary)
    expect_in("Hint: Add schema for 'core.table'", summary)

    with pytest.raises(SchemaNotFoundError):
        collection.raise_if_errors()
