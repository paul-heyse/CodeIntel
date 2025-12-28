"""Unit tests for storage repository helpers."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import pytest

from codeintel.build.schemas import configure_schema_service, get_schema_provider
from codeintel.runtime.runtime_bundle import RuntimeBundle
from codeintel.storage.repositories import (
    DataModelsRepository,
    DatasetReadRepository,
    FunctionRepository,
    GraphRepository,
    ModuleRepository,
    SubsystemRepository,
    TestRepository,
)
from codeintel.storage.warehouse import Warehouse
from tests._helpers.fixtures.rows import (
    DataModelFieldSeed,
    DataModelRelationshipSeed,
    DataModelSeed,
    data_model_field_row,
    data_model_relationship_row,
    data_model_row,
)
from tests._helpers.seeds.subsystems_analytics import SUBSYSTEM_ANALYTICS_PACK

if TYPE_CHECKING:
    from collections.abc import Sequence

    from codeintel.storage.gateway import StorageGateway
    from tests._helpers.context import TestContext


def _expect_true(condition: object, message: str) -> None:
    if bool(condition):
        return
    raise AssertionError(message)


def _expect_equal(actual: object, expected: object, message: str) -> None:
    if actual == expected:
        return
    raise AssertionError(message)


def _expect_in(member: object, container: Sequence[object], message: str) -> None:
    if member in container:
        return
    raise AssertionError(message)


@pytest.fixture(autouse=True)
def _configure_schema_provider(hamilton_runtime: RuntimeBundle) -> None:
    configure_schema_service(runtime=hamilton_runtime)


def _as_mapping(row: tuple[object, ...], table_key: str) -> dict[str, object]:
    table_schema = get_schema_provider().get_table_schema(table_key)
    if table_schema is None:
        message = f"Unknown table key: {table_key}"
        raise AssertionError(message)
    columns = tuple(table_schema.column_names())
    return dict(zip(columns, row, strict=True))


def _repos_for_gateway(
    gateway: StorageGateway,
    repo: str,
    commit: str,
) -> tuple[
    FunctionRepository,
    ModuleRepository,
    TestRepository,
    GraphRepository,
    SubsystemRepository,
    DatasetReadRepository,
]:
    return (
        FunctionRepository(gateway, repo, commit),
        ModuleRepository(gateway, repo, commit),
        TestRepository(gateway, repo, commit),
        GraphRepository(gateway, repo, commit),
        SubsystemRepository(gateway, repo, commit),
        DatasetReadRepository(gateway, repo, commit),
    )


def _repos(
    provisioned_ctx: TestContext,
) -> tuple[
    FunctionRepository,
    ModuleRepository,
    TestRepository,
    GraphRepository,
    SubsystemRepository,
    DatasetReadRepository,
]:
    return _repos_for_gateway(
        provisioned_ctx.gateway,
        provisioned_ctx.repo,
        provisioned_ctx.commit,
    )


def _require_function_goid(functions: FunctionRepository) -> int:
    goid = functions.resolve_function_goid(urn="urn:foo")
    if goid is None:
        message = "resolve_function_goid should return goid"
        raise AssertionError(message)
    return goid


def _assert_function_summary(functions: FunctionRepository, goid: int) -> None:
    summary = functions.get_function_summary_by_goid(goid)
    if summary is None:
        message = "function summary exists"
        raise AssertionError(message)
    _expect_equal(summary["qualname"], "pkg.foo:func", "qualname mismatch")

    per_file = functions.list_function_summaries_for_file("foo.py")
    _expect_true(bool(per_file), "file summaries should be present")
    _expect_in("pkg.foo:func", [row["qualname"] for row in per_file], "missing qualname")

    high_risk = functions.list_high_risk_functions(min_risk=0.0, limit=5, tested_only=False)
    _expect_true(bool(high_risk), "high risk list should not be empty")
    _expect_in(goid, [row["function_goid_h128"] for row in high_risk], "goid missing")
    _expect_in(goid, functions.list_function_goids(), "goid missing from list_function_goids")


def _assert_tests_for_function(tests_repo: TestRepository, goid: int) -> None:
    tests_for_fn = tests_repo.get_tests_for_function(goid, limit=5)
    _expect_equal(len(tests_for_fn), 1, "tests_for_function length mismatch")
    _expect_equal(tests_for_fn[0]["test_id"], "t1", "unexpected test id")


def _assert_graph_neighbors(graphs: GraphRepository, goid: int) -> None:
    outgoing = graphs.get_outgoing_callgraph_neighbors(goid, limit=5)
    _expect_equal(len(outgoing), 1, "outgoing neighbor count mismatch")
    _expect_equal(outgoing[0]["callee_goid_h128"], goid, "callee id mismatch")

    incoming = graphs.get_incoming_callgraph_neighbors(goid, limit=5)
    _expect_equal(len(incoming), 1, "incoming neighbor count mismatch")
    _expect_equal(incoming[0]["caller_goid_h128"], goid, "caller id mismatch")


def _assert_dataset_reads(datasets: DatasetReadRepository, goid: int) -> None:
    dataset_rows = datasets.read_dataset_rows("analytics.function_metrics", limit=10, offset=0)
    if not dataset_rows:
        message = "dataset rows should be readable"
        raise AssertionError(message)
    _expect_equal(dataset_rows[0]["function_goid_h128"], goid, "dataset goid mismatch")
    dataset_df = datasets.read_dataset_dataframe("analytics.function_metrics", limit=10, offset=0)
    _expect_true(not dataset_df.is_empty(), "dataset dataframe should not be empty")
    first_row = dataset_df.row(0, named=True)
    _expect_equal(int(first_row["function_goid_h128"]), goid, "dataframe goid mismatch")


@pytest.fixture
def subsystem_repo_ctx(test_ctx: TestContext) -> TestContext:
    """
    Provide a TestContext seeded with subsystem analytics data.

    Returns
    -------
    TestContext
        Context with subsystem analytics seeds applied.
    """
    return test_ctx.require(SUBSYSTEM_ANALYTICS_PACK)


def test_function_repository_reads(docs_export_gateway: TestContext) -> None:
    """Function repository should resolve GOIDs and surface summaries."""
    functions, _, tests_repo, graphs, _, datasets = _repos(docs_export_gateway)

    goid = _require_function_goid(functions)
    _assert_function_summary(functions, goid)
    _assert_tests_for_function(tests_repo, goid)
    _assert_graph_neighbors(graphs, goid)
    _assert_dataset_reads(datasets, goid)


def test_module_repository_reads(docs_export_gateway: TestContext) -> None:
    """
    Module repository should surface file metadata and IDE hints.

    Raises
    ------
    AssertionError
        If expected summary rows are missing.
    """
    _, modules, _, _, _, _ = _repos(docs_export_gateway)

    summary = modules.get_file_summary("foo.py")
    if summary is None:
        message = "file summary exists"
        raise AssertionError(message)
    _expect_equal(summary["rel_path"], "foo.py", "summary path mismatch")

    module_ids = modules.list_modules()
    _expect_in("pkg.foo", module_ids, "module missing from list_modules")

    hints = modules.get_file_hints("foo.py")
    _expect_true(bool(hints), "IDE hints should exist for module path")
    _expect_equal(hints[0]["module"], "pkg.foo", "hint module mismatch")


def test_subsystem_repository_reads(subsystem_repo_ctx: TestContext) -> None:
    """Subsystem repository should return seeded subsystem summaries and memberships."""
    subsystems = SubsystemRepository(
        subsystem_repo_ctx.gateway,
        subsystem_repo_ctx.repo,
        subsystem_repo_ctx.commit,
    )

    summaries = subsystems.list_subsystems(limit=5)
    _expect_true(len(summaries) > 0, "subsystem summary count should be non-zero")

    modules = subsystems.list_subsystem_modules(str(summaries[0]["subsystem_id"]))
    _expect_true(bool(modules), "subsystem module count mismatch")

    memberships = subsystems.list_subsystems_for_module(str(modules[0]["module"]))
    _expect_true(bool(memberships), "module membership count mismatch")


def test_data_model_accessors(docs_export_gateway: TestContext) -> None:
    """Data model accessors should surface normalized rows directly."""
    ctx = docs_export_gateway
    gateway = ctx.gateway
    warehouse = Warehouse(gateway)
    repo = ctx.repo
    commit = ctx.commit
    now = datetime.now().astimezone()

    model_row = data_model_row(
        DataModelSeed(
            model_id="ModelA",
            model_name="ModelA",
            module="pkg.foo",
            rel_path="foo.py",
            model_kind="dataclass",
            repo=repo,
            commit=commit,
            doc_short="short",
            doc_long="long",
            created_at=now,
        )
    )
    field_row = data_model_field_row(
        DataModelFieldSeed(
            model_id="ModelA",
            field_name="field_a",
            field_type="int",
            required=True,
            has_default=False,
            repo=repo,
            commit=commit,
            source="source",
            rel_path="foo.py",
            lineno=1,
            created_at=now,
        )
    )
    relationship_row = data_model_relationship_row(
        DataModelRelationshipSeed(
            source_model_id="ModelA",
            target_model_id="ModelB",
            target_module=None,
            target_model_name=None,
            field_name="field_a",
            relationship_kind="association",
            multiplicity=None,
            repo=repo,
            commit=commit,
            rel_path="foo.py",
            lineno=1,
            created_at=now,
        )
    )

    warehouse.materialize_mappings(
        "analytics.data_models", [_as_mapping(model_row, "analytics.data_models")]
    )
    warehouse.materialize_mappings(
        "analytics.data_model_fields", [_as_mapping(field_row, "analytics.data_model_fields")]
    )
    warehouse.materialize_mappings(
        "analytics.data_model_relationships",
        [_as_mapping(relationship_row, "analytics.data_model_relationships")],
    )

    normalized = DataModelsRepository(gateway, repo, commit).list_models_normalized()
    _expect_equal(len(normalized), 1, "normalized model count mismatch")
    model = normalized[0]
    _expect_equal(model.model_id, "ModelA", "model id mismatch")
    _expect_equal(len(model.fields), 1, "model fields mismatch")
    _expect_equal(len(model.relationships), 1, "model relationships mismatch")
