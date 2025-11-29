"""Unit fixtures for analytics.tests_profiles helpers (legacy-free)."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from codeintel.analytics.tests_profiles import behavioral_tags, coverage_inputs, importance, rows
from codeintel.analytics.tests_profiles.types import (
    BehavioralLLMRequest,
    BehavioralLLMResult,
    ImportanceInputs,
    IoFlags,
    TestAstInfo,
    TestProfileContext,
    TestRecord,
)
from codeintel.config import BehavioralCoverageStepConfig, TestProfileStepConfig
from codeintel.config.primitives import SnapshotRef
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.rows import (
    BEHAVIORAL_COVERAGE_COLUMNS,
    TEST_PROFILE_COLUMNS,
    TestProfileRowModel,
    behavioral_coverage_row_to_tuple,
    serialize_test_profile_row,
)


@contextmanager
def _override(obj: object, name: str, value: object) -> Iterator[None]:
    original = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, original)


class _FakeCon:
    def __init__(self) -> None:
        self.executed: list[tuple[str, list[object] | None]] = []
        self.executemany_calls: list[tuple[str, list[list[object]]]] = []

    def execute(self, sql: str, params: list[object] | None = None) -> _FakeCon:
        self.executed.append((sql, params))
        return self

    def executemany(self, sql: str, params_list: list[list[object]]) -> None:
        self.executemany_calls.append((sql, params_list))


def _snapshot_cfg() -> tuple[TestProfileStepConfig, BehavioralCoverageStepConfig]:
    snapshot = SnapshotRef(repo="r", commit="c", repo_root=Path.cwd())
    return TestProfileStepConfig(snapshot=snapshot), BehavioralCoverageStepConfig(snapshot=snapshot)


def test_importance_guardrails_and_monotonicity() -> None:
    """Importance/flakiness scoring should remain bounded and monotonic."""
    io_none = IoFlags()
    io_network = IoFlags(uses_network=True)
    slow_threshold = 1000.0
    fast_score = importance.compute_flakiness_score(
        status="passed",
        markers=["fast"],
        duration_ms=100.0,
        io_flags=io_none,
        slow_test_threshold_ms=slow_threshold,
    )
    slow_score = importance.compute_flakiness_score(
        status="failed",
        markers=["slow"],
        duration_ms=5000.0,
        io_flags=io_network,
        slow_test_threshold_ms=slow_threshold,
    )
    if not 0.0 <= fast_score <= 1.0 or not 0.0 <= slow_score <= 1.0:
        msg = "Flakiness scores escaped [0, 1] bounds."
        pytest.fail(msg)
    if slow_score <= fast_score:
        msg = "Flakiness scoring is not monotonic with worse signals."
        pytest.fail(msg)

    baseline = importance.compute_importance_score(
        ImportanceInputs(
            functions_covered_count=1,
            weighted_degree=0.5,
            max_function_count=5,
            max_weighted_degree=5.0,
            subsystem_risk=0.1,
            max_subsystem_risk=1.0,
        )
    )
    improved = importance.compute_importance_score(
        ImportanceInputs(
            functions_covered_count=3,
            weighted_degree=4.0,
            max_function_count=5,
            max_weighted_degree=5.0,
            subsystem_risk=0.5,
            max_subsystem_risk=1.0,
        )
    )
    if baseline is None or improved is None:
        msg = "Importance scoring returned None unexpectedly."
        pytest.fail(msg)
    if improved <= baseline:
        msg = "Importance score did not increase with stronger signals."
        pytest.fail(msg)


def test_build_test_profile_rows_round_trip() -> None:
    """Tuple-to-model mapping should align with schema constants for new helpers."""
    created_at = datetime(2024, 1, 1, tzinfo=UTC)
    test_cfg, _ = _snapshot_cfg()
    test_record = TestRecord(
        test_id="test-id",
        test_goid_h128=1,
        urn="urn",
        rel_path="rel.py",
        module="mod",
        qualname="qual",
        language="python",
        kind="function",
        status="passed",
        duration_ms=12.5,
        markers=["fast"],
        flaky=False,
        start_line=1,
        end_line=2,
    )
    functions_covered = {
        "test-id": coverage_inputs.FunctionCoverageEntry(
            functions=[{"function_goid_h128": 1}],
            count=1,
            primary=[1],
        )
    }
    subsystems_covered = {
        "test-id": coverage_inputs.SubsystemCoverageEntry(
            subsystems=[{"subsystem_id": "sub"}],
            count=1,
            primary_subsystem_id="sub",
            max_risk_score=0.2,
        )
    }
    tg_metrics = {
        "test-id": coverage_inputs.TestGraphMetrics(
            degree=1,
            weighted_degree=2.0,
            proj_degree=1,
            proj_weight=2.0,
            proj_clustering=0.1,
            proj_betweenness=0.2,
        )
    }
    ast_info = {"test-id": TestAstInfo(assert_count=1, raise_count=0)}
    ctx = rows.build_test_profile_context(
        cfg=test_cfg,
        functions_covered=functions_covered,
        subsystems_covered=subsystems_covered,
        tg_metrics=tg_metrics,
        ast_info=ast_info,
    )
    # Freeze time for determinism
    frozen_ctx = TestProfileContext(
        cfg=ctx.cfg,
        now=created_at,
        max_function_count=ctx.max_function_count,
        max_weighted_degree=ctx.max_weighted_degree,
        max_subsystem_risk=ctx.max_subsystem_risk,
        functions_covered=ctx.functions_covered,
        subsystems_covered=ctx.subsystems_covered,
        tg_metrics=ctx.tg_metrics,
        ast_info=ctx.ast_info,
    )
    models = rows.build_test_profile_rows([test_record], frozen_ctx)
    if len(models) != 1:
        msg = "Expected exactly one model."
        pytest.fail(msg)
    serialized = serialize_test_profile_row(models[0])
    if len(serialized) != len(TEST_PROFILE_COLUMNS):
        msg = "Serialized tuple length mismatch for test_profile."
        pytest.fail(msg)
    if models[0]["created_at"] != created_at:
        msg = "Created_at did not preserve frozen context value."
        pytest.fail(msg)


def test_build_behavioral_coverage_rows_normalization() -> None:
    """Behavioral coverage rows should align with schema constants."""
    created_at = datetime(2024, 1, 1, tzinfo=UTC)
    tuple_row = (
        "repo",
        "commit",
        "test-id",
        1,
        "rel.py",
        "qual",
        ["network_interaction"],
        "heuristic",
        "v1",
        "gpt",
        "run-id",
        created_at,
    )
    models = rows.build_behavioral_coverage_rows([tuple_row])
    if len(models) != 1:
        msg = "Expected exactly one behavioral coverage model."
        pytest.fail(msg)
    serialized = behavioral_coverage_row_to_tuple(models[0])
    if len(serialized) != len(BEHAVIORAL_COVERAGE_COLUMNS):
        msg = "Serialized tuple length mismatch for behavioral_coverage."
        pytest.fail(msg)


def test_build_behavior_rows_mixed_sources() -> None:
    """Behavior rows should preserve mixed heuristic/LLM metadata without legacy hooks."""
    fake_con = _FakeCon()
    gateway = cast("StorageGateway", SimpleNamespace(con=fake_con))
    _, beh_cfg = _snapshot_cfg()

    sample_tests = [
        TestRecord(
            test_id="t1",
            test_goid_h128=None,
            urn=None,
            rel_path="a.py",
            module=None,
            qualname="A::test",
            language="python",
            kind="function",
            status="passed",
            duration_ms=10.0,
            markers=["network"],
            flaky=False,
            start_line=1,
            end_line=5,
        ),
        TestRecord(
            test_id="t2",
            test_goid_h128=None,
            urn=None,
            rel_path="b.py",
            module=None,
            qualname="B::test",
            language="python",
            kind="function",
            status="failed",
            duration_ms=20.0,
            markers=["db", "io"],
            flaky=False,
            start_line=1,
            end_line=5,
        ),
    ]

    profile_ctx = {
        "t1": {"markers": ["network"], "functions_covered": [], "subsystems_covered": []},
        "t2": {"markers": ["db"], "functions_covered": [], "subsystems_covered": []},
    }
    ast_info = {"t1": TestAstInfo(), "t2": TestAstInfo(io_flags=IoFlags(uses_db=True))}
    ctx_seen: dict[str, object] = {}

    def _fake_build_behavior_row(
        test: TestRecord, ctx: behavioral_tags.BehavioralContext
    ) -> tuple[object, ...]:
        ctx_seen.setdefault("llm_runner", getattr(ctx, "llm_runner", None))
        tag_source = "llm" if test.test_id == "t2" else "heuristic"
        llm_model = "gpt" if test.test_id == "t2" else None
        llm_run_id = "run-123" if test.test_id == "t2" else None
        tags = ["db"] if test.test_id == "t2" else ["network"]
        return (
            beh_cfg.repo,
            beh_cfg.commit,
            test.test_id,
            None,
            test.rel_path,
            test.qualname or test.test_id,
            tags,
            tag_source,
            beh_cfg.heuristic_version,
            llm_model,
            llm_run_id,
            getattr(ctx, "now", datetime.now(tz=UTC)),
        )

    def _fake_llm_runner(_request: BehavioralLLMRequest) -> BehavioralLLMResult:
        return BehavioralLLMResult(tags=["db"])

    hooks = behavioral_tags.BehaviorRowHooks(
        load_tests=lambda _con, _cfg: sample_tests,
        build_ast=lambda _root, _tests, _io_spec, _concurrency: ast_info,
        load_profile_ctx=lambda _con, _cfg: profile_ctx,
        row_builder=_fake_build_behavior_row,
    )
    with _override(behavioral_tags, "ensure_schema", lambda _con, _table: None):
        tuples = behavioral_tags.build_behavior_rows(
            gateway,
            beh_cfg,
            llm_runner=_fake_llm_runner,
            hooks=hooks,
        )
    models = rows.build_behavioral_coverage_rows(tuples)
    if {model["test_id"] for model in models} != {"t1", "t2"}:
        msg = "Behavioral coverage rows missing expected tests."
        pytest.fail(msg)
    tag_sources = {model["test_id"]: model["tag_source"] for model in models}
    if tag_sources["t1"] != "heuristic" or tag_sources["t2"] != "llm":
        msg = "Tag sources were not preserved per test."
        pytest.fail(msg)
    if ctx_seen.get("llm_runner") is not _fake_llm_runner:
        msg = "LLM runner was not threaded through behavioral context."
        pytest.fail(msg)


def test_write_test_profile_rows_with_stubs() -> None:
    """Writer should honor registry columns and insert via prepared statement."""
    fake_con = _FakeCon()
    gateway = cast("StorageGateway", SimpleNamespace(con=fake_con))
    test_cfg, _ = _snapshot_cfg()
    sample_row: TestProfileRowModel = dict.fromkeys(TEST_PROFILE_COLUMNS)  # type: ignore[arg-type]
    sample_row["repo"] = "r"
    sample_row["commit"] = "c"
    sample_row["test_id"] = "t"
    sample_row["rel_path"] = "p"
    sample_row["markers"] = []
    sample_row["functions_covered"] = []
    sample_row["primary_function_goids"] = []
    sample_row["subsystems_covered"] = []
    sample_row["created_at"] = datetime.now(tz=UTC)

    with ExitStack() as stack:
        stack.enter_context(_override(rows, "ensure_schema", lambda _con, _table_key: None))
        stack.enter_context(
            _override(
                rows,
                "load_registry_columns",
                lambda _con: {"analytics.test_profile": list(TEST_PROFILE_COLUMNS)},
            )
        )
        stack.enter_context(
            _override(
                rows,
                "prepared_statements_dynamic",
                lambda _con, _table_key: SimpleNamespace(
                    insert_sql="INSERT INTO analytics.test_profile"
                ),
            )
        )

        inserted = rows.write_test_profile_rows(gateway, test_cfg, [sample_row])
        if inserted != 1:
            msg = "Writer did not report one inserted row."
            pytest.fail(msg)
        if not fake_con.executed[0][0].startswith("DELETE FROM analytics.test_profile"):
            msg = "Delete was not issued for test_profile."
            pytest.fail(msg)
        if not fake_con.executemany_calls[0][0].startswith("INSERT INTO analytics.test_profile"):
            msg = "Insert was not issued for test_profile."
            pytest.fail(msg)
