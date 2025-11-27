"""Unit tests for storage repository helpers."""

from __future__ import annotations

from datetime import datetime

from codeintel.storage.repositories import (
    DatasetReadRepository,
    FunctionRepository,
    GraphRepository,
    ModuleRepository,
    SubsystemRepository,
    TestRepository,
)
from codeintel.storage.repositories.data_models import DataModelRepository


def _expect_true(condition: object, message: str) -> None:
    if bool(condition):
        return
    raise AssertionError(message)


def _expect_equal(actual: object, expected: object, message: str) -> None:
    if actual == expected:
        return
    raise AssertionError(message)


def _expect_in(member: object, container: list[object], message: str) -> None:
    if member in container:
        return
    raise AssertionError(message)


def _repos(
    provisioned_ctx: object,
) -> tuple[
    FunctionRepository,
    ModuleRepository,
    TestRepository,
    GraphRepository,
    SubsystemRepository,
    DatasetReadRepository,
]:
    gateway = provisioned_ctx.gateway
    repo = provisioned_ctx.repo
    commit = provisioned_ctx.commit
    return (
        FunctionRepository(gateway, repo, commit),
        ModuleRepository(gateway, repo, commit),
        TestRepository(gateway, repo, commit),
        GraphRepository(gateway, repo, commit),
        SubsystemRepository(gateway, repo, commit),
        DatasetReadRepository(gateway, repo, commit),
    )


def test_function_repository_reads(docs_export_gateway: object) -> None:
    """Function repository should resolve GOIDs and surface summaries."""
    functions, _, tests_repo, graphs, _, datasets = _repos(docs_export_gateway)

    goid = functions.resolve_function_goid(urn="urn:foo")
    _expect_true(goid is not None, "resolve_function_goid should return goid")

    summary = functions.get_function_summary_by_goid(goid)
    _expect_true(summary is not None, "function summary exists")
    _expect_equal(summary["qualname"], "pkg.foo:func", "qualname mismatch")

    per_file = functions.list_function_summaries_for_file("foo.py")
    _expect_true(bool(per_file), "file summaries should be present")
    _expect_in("pkg.foo:func", [row["qualname"] for row in per_file], "missing qualname")

    high_risk = functions.list_high_risk_functions(min_risk=0.0, limit=5, tested_only=False)
    _expect_true(bool(high_risk), "high risk list should not be empty")
    _expect_in(goid, [row["function_goid_h128"] for row in high_risk], "goid missing")

    tests_for_fn = tests_repo.get_tests_for_function(goid, limit=5)
    _expect_equal(len(tests_for_fn), 1, "tests_for_function length mismatch")
    _expect_equal(tests_for_fn[0]["test_id"], "t1", "unexpected test id")

    outgoing = graphs.get_outgoing_callgraph_neighbors(goid, limit=5)
    _expect_equal(len(outgoing), 1, "outgoing neighbor count mismatch")
    _expect_equal(outgoing[0]["callee_goid_h128"], goid, "callee id mismatch")

    incoming = graphs.get_incoming_callgraph_neighbors(goid, limit=5)
    _expect_equal(len(incoming), 1, "incoming neighbor count mismatch")
    _expect_equal(incoming[0]["caller_goid_h128"], goid, "caller id mismatch")

    dataset_rows = datasets.read_dataset_rows("analytics.function_metrics", limit=10, offset=0)
    _expect_true(bool(dataset_rows), "dataset rows should be readable")
    _expect_equal(dataset_rows[0]["function_goid_h128"], goid, "dataset goid mismatch")


def test_module_repository_reads(docs_export_gateway: object) -> None:
    """Module repository should surface file metadata and IDE hints."""
    _, modules, _, _, _, _ = _repos(docs_export_gateway)

    summary = modules.get_file_summary("foo.py")
    _expect_true(summary is not None, "file summary exists")
    _expect_equal(summary["rel_path"], "foo.py", "summary path mismatch")

    hints = modules.get_file_hints("foo.py")
    _expect_true(bool(hints), "IDE hints should exist for module path")
    _expect_equal(hints[0]["module"], "pkg.foo", "hint module mismatch")


def _seed_subsystem_data(provisioned_ctx: object) -> None:
    now = datetime.now().astimezone()
    gateway = provisioned_ctx.gateway
    repo = provisioned_ctx.repo
    commit = provisioned_ctx.commit
    gateway.analytics.insert_subsystems(
        [
            (
                repo,
                commit,
                "subsystem-1",
                "Subsystem One",
                "demo subsystem",
                1,
                '["pkg.foo"]',
                '["foo.entry"]',
                0,
                0,
                0,
                0,
                1,
                0.1,
                0.1,
                0,
                "low",
                now,
            )
        ]
    )
    gateway.analytics.insert_subsystem_modules(
        [(repo, commit, "subsystem-1", "pkg.foo", "owner")]
    )


def test_subsystem_repository_reads(docs_export_gateway: object) -> None:
    """Subsystem repository should return seeded subsystem summaries and memberships."""
    _, _, _, _, subsystems, _ = _repos(docs_export_gateway)
    _seed_subsystem_data(docs_export_gateway)

    summaries = subsystems.list_subsystems(limit=5)
    _expect_equal(len(summaries), 1, "subsystem summary count mismatch")
    _expect_equal(summaries[0]["subsystem_id"], "subsystem-1", "subsystem id mismatch")

    modules = subsystems.list_subsystem_modules("subsystem-1")
    _expect_equal(len(modules), 1, "subsystem module count mismatch")
    _expect_equal(modules[0]["module"], "pkg.foo", "subsystem module mismatch")

    memberships = subsystems.list_subsystems_for_module("pkg.foo")
    _expect_equal(len(memberships), 1, "module membership count mismatch")
    _expect_equal(memberships[0]["subsystem_id"], "subsystem-1", "membership id mismatch")


def test_data_model_repository_reads(docs_export_gateway: object) -> None:
    """Data model repository should surface normalized rows."""
    ctx = docs_export_gateway
    gateway = ctx.gateway
    repo = ctx.repo
    commit = ctx.commit
    repo_dm = DataModelRepository(gateway)
    now = datetime.now().astimezone()

    gateway.con.execute(
        """
        INSERT INTO analytics.data_models (
            repo, commit, model_id, goid_h128, model_name, module, rel_path, model_kind,
            base_classes_json, fields_json, relationships_json, doc_short, doc_long, created_at
        ) VALUES (?, ?, ?, NULL, ?, ?, ?, ?, '[]', '[]', '[]', 'short', 'long', ?)
        """,
        [repo, commit, "ModelA", "ModelA", "pkg.foo", "foo.py", "dataclass", now],
    )
    gateway.con.execute(
        """
        INSERT INTO analytics.data_model_fields (
            repo, commit, model_id, field_name, field_type, required, has_default, default_expr,
            constraints_json, source, rel_path, lineno, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, '{}', ?, ?, ?, ?)
        """,
        [repo, commit, "ModelA", "field_a", "int", True, False, "source", "foo.py", 1, now],
    )
    gateway.con.execute(
        """
        INSERT INTO analytics.data_model_relationships (
            repo, commit, source_model_id, target_model_id, target_module, target_model_name,
            field_name, relationship_kind, multiplicity, via, evidence_json, rel_path, lineno,
            created_at
        ) VALUES (?, ?, ?, ?, NULL, NULL, ?, ?, NULL, NULL, '{}', ?, ?, ?)
        """,
        [repo, commit, "ModelA", "ModelB", "field_a", "association", "foo.py", 1, now],
    )

    normalized = repo_dm.models_normalized(repo, commit)
    _expect_equal(len(normalized), 1, "normalized model count mismatch")
    model = normalized[0]
    _expect_equal(model.model_id, "ModelA", "model id mismatch")
    _expect_equal(len(model.fields), 1, "model fields mismatch")
    _expect_equal(len(model.relationships), 1, "model relationships mismatch")
