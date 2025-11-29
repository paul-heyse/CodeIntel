"""Integration-style fixtures for test profile helpers."""
# ruff: noqa: E402

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest

legacy_module = ModuleType("codeintel.analytics.tests.profiles")
legacy_module.load_functions_covered = lambda *_args, **_kwargs: {}  # type: ignore[attr-defined]
legacy_module.load_subsystems_covered = lambda *_args, **_kwargs: {}  # type: ignore[attr-defined]
legacy_module.load_test_graph_metrics_public = (  # type: ignore[attr-defined]
    lambda *_args, **_kwargs: {}
)
legacy_module.load_test_records_public = lambda *_args, **_kwargs: []  # type: ignore[attr-defined]
legacy_module.build_test_ast_index = lambda *_args, **_kwargs: {}  # type: ignore[attr-defined]
legacy_module.DEFAULT_IO_SPEC = {}  # type: ignore[attr-defined]
legacy_module.CONCURRENCY_LIBS = set()  # type: ignore[attr-defined]
legacy_module.load_test_profile_context = (  # type: ignore[attr-defined]
    lambda *_args, **_kwargs: {}
)
legacy_module.BehavioralContext = object  # type: ignore[attr-defined]
legacy_module.build_behavior_row = lambda *_args, **_kwargs: ()  # type: ignore[attr-defined]
legacy_module.infer_behavior_tags = lambda *_args, **_kwargs: []  # type: ignore[attr-defined]
legacy_module.compute_flakiness_score = lambda *_args, **_kwargs: 0.0  # type: ignore[attr-defined]
legacy_module.compute_importance_score = (  # type: ignore[attr-defined]
    lambda *_args, **_kwargs: 0.0
)
legacy_module.build_test_profile_row = lambda *_args, **_kwargs: ()  # type: ignore[attr-defined]
legacy_module.build_test_profile = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
legacy_module.build_behavioral_coverage = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
sys.modules.setdefault("codeintel.analytics.tests.profiles", legacy_module)

from codeintel.analytics.tests_profiles import coverage_inputs, importance, rows
from codeintel.analytics.tests_profiles.types import (
    ImportanceInputs,
    IoFlags,
    TestAstInfo,
    TestProfileContext,
    TestRecord,
)
from codeintel.config import BehavioralCoverageStepConfig, TestProfileStepConfig
from codeintel.config.primitives import SnapshotRef
from codeintel.storage.gateway import DuckDBConnection as DuckDBConnectionType
from codeintel.storage.rows import TestProfileRowModel

_SLOW_THRESHOLD_MS = 1500.0


@contextmanager
def _override(obj: object, name: str, value: object) -> Iterator[None]:
    original = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, original)


def _snapshot_cfg() -> tuple[TestProfileStepConfig, BehavioralCoverageStepConfig]:
    snapshot = SnapshotRef(repo="r", commit="c", repo_root=Path.cwd())
    return TestProfileStepConfig(snapshot=snapshot), BehavioralCoverageStepConfig(snapshot=snapshot)


@dataclass(frozen=True)
class _FixtureData:
    functions_by_test: dict[str, dict[str, object]]
    subsystems_by_test: dict[str, dict[str, object]]
    graph_metrics_by_test: dict[str, dict[str, object]]
    ast_info: dict[str, TestAstInfo]
    tests: list[TestRecord]


def _make_fixture_data() -> _FixtureData:
    functions_by_test = {
        "t1": {
            "functions": [{"function_goid_h128": 1, "covered_lines": 8, "executable_lines": 10}],
            "count": 1,
            "primary": [1],
        },
        "t2": {
            "functions": [{"function_goid_h128": 2, "covered_lines": 4, "executable_lines": 20}],
            "count": 1,
            "primary": [2],
        },
        "t3": {
            "functions": [{"function_goid_h128": 3, "covered_lines": 6, "executable_lines": 10}],
            "count": 1,
            "primary": [3],
        },
        "t4": {
            "functions": [{"function_goid_h128": 4, "covered_lines": 3, "executable_lines": 5}],
            "count": 1,
            "primary": [4],
        },
    }
    subsystems_by_test = {
        "t1": {
            "subsystems": [{"subsystem_id": "subA", "risk": 0.6}],
            "count": 1,
            "primary_subsystem_id": "subA",
            "max_risk_score": 0.6,
        },
        "t2": {
            "subsystems": [{"subsystem_id": "subB", "risk": 0.2}],
            "count": 1,
            "primary_subsystem_id": "subB",
            "max_risk_score": 0.2,
        },
        "t3": {
            "subsystems": [{"subsystem_id": "subC", "risk": 0.4}],
            "count": 1,
            "primary_subsystem_id": "subC",
            "max_risk_score": 0.4,
        },
        "t4": {
            "subsystems": [{"subsystem_id": "subD", "risk": 0.9}],
            "count": 1,
            "primary_subsystem_id": "subD",
            "max_risk_score": 0.9,
        },
    }
    graph_metrics_by_test = {
        "t1": {
            "degree": 2,
            "weighted_degree": 3.5,
            "proj_degree": 1,
            "proj_weight": 1.5,
            "proj_clustering": 0.1,
            "proj_betweenness": 0.05,
        },
        "t2": {
            "degree": 4,
            "weighted_degree": 1.0,
            "proj_degree": 1,
            "proj_weight": 1.0,
            "proj_clustering": 0.2,
            "proj_betweenness": 0.01,
        },
        "t3": {
            "degree": 3,
            "weighted_degree": 2.0,
            "proj_degree": 2,
            "proj_weight": 2.5,
            "proj_clustering": 0.15,
            "proj_betweenness": 0.02,
        },
        "t4": {
            "degree": 5,
            "weighted_degree": 4.5,
            "proj_degree": 3,
            "proj_weight": 4.0,
            "proj_clustering": 0.3,
            "proj_betweenness": 0.08,
        },
    }
    ast_info = {
        "t1": TestAstInfo(
            assert_count=2, raise_count=0, uses_fixtures=True, io_flags=IoFlags(uses_network=True)
        ),
        "t2": TestAstInfo(assert_count=1, raise_count=1, uses_fixtures=False, io_flags=IoFlags()),
        "t3": TestAstInfo(
            assert_count=3,
            raise_count=0,
            uses_fixtures=True,
            io_flags=IoFlags(uses_db=True, uses_filesystem=True),
        ),
        "t4": TestAstInfo(
            assert_count=4,
            raise_count=2,
            uses_fixtures=True,
            io_flags=IoFlags(uses_filesystem=True, uses_subprocess=True),
        ),
    }
    tests = [
        TestRecord(
            test_id="t1",
            test_goid_h128=101,
            urn="urn:t1",
            rel_path="tests/t1.py",
            module="tests.t1",
            qualname="TestT1.test_one",
            language="python",
            kind="function",
            status="failed",
            duration_ms=2000.0,
            markers=["slow", "network"],
            flaky=True,
            start_line=1,
            end_line=10,
        ),
        TestRecord(
            test_id="t2",
            test_goid_h128=202,
            urn="urn:t2",
            rel_path="tests/t2.py",
            module="tests.t2",
            qualname="TestT2.test_two",
            language="python",
            kind="function",
            status="passed",
            duration_ms=200.0,
            markers=["fast"],
            flaky=False,
            start_line=5,
            end_line=15,
        ),
        TestRecord(
            test_id="t3",
            test_goid_h128=303,
            urn="urn:t3",
            rel_path="tests/t3.py",
            module="tests.t3",
            qualname="TestT3.test_three",
            language="python",
            kind="function",
            status="passed",
            duration_ms=1200.0,
            markers=["db", "integration"],
            flaky=False,
            start_line=10,
            end_line=25,
        ),
        TestRecord(
            test_id="t4",
            test_goid_h128=404,
            urn="urn:t4",
            rel_path="tests/t4.py",
            module="tests.t4",
            qualname="TestT4.test_four",
            language="python",
            kind="function",
            status="failed",
            duration_ms=3000.0,
            markers=["flaky", "integration"],
            flaky=True,
            start_line=15,
            end_line=40,
        ),
    ]
    return _FixtureData(
        functions_by_test=functions_by_test,
        subsystems_by_test=subsystems_by_test,
        graph_metrics_by_test=graph_metrics_by_test,
        ast_info=ast_info,
        tests=tests,
    )


def _flakiness_score(status: str | None, duration_ms: float | None, io_flags: IoFlags) -> float:
    base = 0.1
    if status == "failed":
        base += 0.3
    if duration_ms is not None and duration_ms > _SLOW_THRESHOLD_MS:
        base += 0.2
    if io_flags.uses_network:
        base += 0.15
    if io_flags.uses_db or io_flags.uses_filesystem:
        base += 0.1
    if io_flags.uses_subprocess:
        base += 0.1
    return min(base, 1.0)


def _importance_score(inputs: ImportanceInputs) -> float:
    weight_share = inputs.functions_covered_count / max(inputs.max_function_count, 1)
    degree_share = (inputs.weighted_degree or 0.0) / max(inputs.max_weighted_degree, 1.0)
    risk_share = (inputs.subsystem_risk or 0.0) / max(inputs.max_subsystem_risk, 1.0)
    return max(0.0, min((weight_share + degree_share + risk_share) / 3.0, 1.0))


def _build_test_profile_row(test: TestRecord, ctx_param: TestProfileContext) -> tuple[object, ...]:
    coverage = cast("dict[str, object]", ctx_param.functions_covered[test.test_id])
    subs = cast("dict[str, object]", ctx_param.subsystems_covered[test.test_id])
    metrics = cast("dict[str, object]", ctx_param.tg_metrics[test.test_id])
    ast = ctx_param.ast_info[test.test_id]
    flakiness = importance.compute_flakiness_score(
        status=test.status,
        markers=test.markers,
        duration_ms=test.duration_ms,
        io_flags=ast.io_flags,
        slow_test_threshold_ms=_SLOW_THRESHOLD_MS,
    )
    importance_score = importance.compute_importance_score(
        ImportanceInputs(
            functions_covered_count=cast("int", coverage["count"]),
            weighted_degree=cast("float | None", metrics["weighted_degree"]),
            max_function_count=ctx_param.max_function_count,
            max_weighted_degree=ctx_param.max_weighted_degree,
            subsystem_risk=cast("float | None", subs["max_risk_score"]),
            max_subsystem_risk=ctx_param.max_subsystem_risk,
        )
    )
    return (
        ctx_param.cfg.repo,
        ctx_param.cfg.commit,
        test.test_id,
        test.test_goid_h128,
        test.urn,
        test.rel_path,
        test.module,
        test.qualname,
        test.language,
        test.kind,
        test.status,
        test.duration_ms,
        test.markers,
        test.flaky,
        ctx_param.now,
        cast("list[dict[str, object]]", coverage["functions"]),
        cast("int", coverage["count"]),
        cast("list[int]", coverage["primary"]),
        cast("list[dict[str, object]]", subs["subsystems"]),
        cast("int", subs["count"]),
        cast("str | None", subs["primary_subsystem_id"]),
        ast.assert_count,
        ast.raise_count,
        False,
        ast.uses_fixtures,
        ast.io_flags.io_bound,
        ast.io_flags.uses_network,
        ast.io_flags.uses_db,
        ast.io_flags.uses_filesystem,
        ast.io_flags.uses_subprocess,
        flakiness,
        importance_score,
        None,
        metrics["degree"],
        metrics["weighted_degree"],
        metrics["proj_degree"],
        metrics["proj_weight"],
        metrics["proj_clustering"],
        metrics["proj_betweenness"],
        ctx_param.now,
    )


def _assert_primary_and_counts(by_id: dict[str, TestProfileRowModel]) -> None:
    expected_counts = {"t1": 1, "t2": 1, "t3": 1, "t4": 1}
    expected_functions = {"t1": [1], "t2": [2], "t3": [3], "t4": [4]}
    expected_subsystems = {"t1": "subA", "t2": "subB", "t3": "subC", "t4": "subD"}
    for test_id, model in by_id.items():
        if model["functions_covered_count"] != expected_counts[test_id]:
            msg = f"Function coverage count mismatch for {test_id}."
            pytest.fail(msg)
        if model["primary_function_goids"] != expected_functions[test_id]:
            msg = f"Primary function goids mismatch for {test_id}."
            pytest.fail(msg)
        if model["primary_subsystem_id"] != expected_subsystems[test_id]:
            msg = f"Primary subsystem mismatch for {test_id}."
            pytest.fail(msg)


def _assert_scores_and_flags(by_id: dict[str, TestProfileRowModel], now: datetime) -> None:
    expected_importance_order = ["t4", "t1", "t3", "t2"]
    importance_sorted = sorted(
        by_id.items(),
        key=lambda kv: cast("float", kv[1]["importance_score"]),
        reverse=True,
    )
    if [k for k, _ in importance_sorted] != expected_importance_order:
        msg = "Importance ordering did not match expected risk hierarchy."
        pytest.fail(msg)
    expected_flakiness_order = ["t4", "t1", "t3", "t2"]
    flakiness_sorted = sorted(
        by_id.items(),
        key=lambda kv: cast("float", kv[1]["flakiness_score"]),
        reverse=True,
    )
    if [k for k, _ in flakiness_sorted] != expected_flakiness_order:
        msg = "Flakiness ordering did not match expected risk hierarchy."
        pytest.fail(msg)

    expected_network = {"t1": True, "t2": False, "t3": False, "t4": False}
    expected_db = {"t1": False, "t2": False, "t3": True, "t4": False}
    expected_fs = {"t1": False, "t2": False, "t3": True, "t4": True}
    expected_subprocess = {"t1": False, "t2": False, "t3": False, "t4": True}
    for test_id, model in by_id.items():
        if model["created_at"] != now:
            msg = f"Created_at timestamp mismatch for {test_id}."
            pytest.fail(msg)
        if model["uses_network"] is not expected_network[test_id]:
            msg = f"Network flag mismatch for {test_id}."
            pytest.fail(msg)
        if model["uses_db"] is not expected_db[test_id]:
            msg = f"DB flag mismatch for {test_id}."
            pytest.fail(msg)
        if model["uses_filesystem"] is not expected_fs[test_id]:
            msg = f"Filesystem flag mismatch for {test_id}."
            pytest.fail(msg)
        if model["uses_subprocess"] is not expected_subprocess[test_id]:
            msg = f"Subprocess flag mismatch for {test_id}."
            pytest.fail(msg)


def _assert_models(models: list[TestProfileRowModel], now: datetime) -> None:
    by_id = {model["test_id"]: model for model in models}
    if set(by_id) != {"t1", "t2", "t3", "t4"}:
        msg = "Expected both tests in assembled profile rows."
        pytest.fail(msg)
    _assert_primary_and_counts(by_id)
    _assert_scores_and_flags(by_id, now)


def test_profile_row_assembly_end_to_end() -> None:
    """Stitch coverage, metrics, and importance into normalized profile rows."""
    now = datetime(2024, 2, 1, tzinfo=UTC)
    test_cfg, beh_cfg = _snapshot_cfg()
    fixture = _make_fixture_data()

    with ExitStack() as stack:
        stack.enter_context(
            _override(
                coverage_inputs.legacy,
                "load_functions_covered",
                lambda _con, repo, commit: fixture.functions_by_test
                if (repo, commit) == (test_cfg.repo, test_cfg.commit)
                else {},
            )
        )
        stack.enter_context(
            _override(
                coverage_inputs.legacy,
                "load_subsystems_covered",
                lambda _con, repo, commit: fixture.subsystems_by_test
                if (repo, commit) == (beh_cfg.repo, beh_cfg.commit)
                else {},
            )
        )
        stack.enter_context(
            _override(
                coverage_inputs.legacy,
                "load_test_graph_metrics_public",
                lambda _con, repo, commit: fixture.graph_metrics_by_test
                if (repo, commit) == (test_cfg.repo, test_cfg.commit)
                else {},
            )
        )
        stack.enter_context(
            _override(
                importance.legacy,
                "compute_flakiness_score",
                lambda **kwargs: _flakiness_score(
                    status=cast("str | None", kwargs.get("status")),
                    duration_ms=cast("float | None", kwargs.get("duration_ms")),
                    io_flags=cast("IoFlags", kwargs.get("io_flags")),
                ),
            )
        )
        stack.enter_context(
            _override(importance.legacy, "compute_importance_score", _importance_score)
        )
        stack.enter_context(
            _override(rows.legacy, "build_test_profile_row", _build_test_profile_row)
        )

        fake_con = cast("DuckDBConnectionType", object())
        functions_covered = coverage_inputs.aggregate_test_coverage_by_function(fake_con, test_cfg)
        subsystems_covered = coverage_inputs.aggregate_test_coverage_by_subsystem(fake_con, beh_cfg)
        tg_metrics = coverage_inputs.load_test_graph_metrics(fake_con, test_cfg)

        ctx = TestProfileContext(
            cfg=test_cfg,
            now=now,
            max_function_count=2,
            max_weighted_degree=5.0,
            max_subsystem_risk=1.0,
            functions_covered=functions_covered,
            subsystems_covered=subsystems_covered,
            tg_metrics=tg_metrics,
            ast_info=fixture.ast_info,
        )

        models = rows.build_test_profile_rows(fixture.tests, ctx)
        _assert_models(models, now)
