"""Unit tests for analytics.tests_profiles module.

This module consolidates unit tests for test profiles helpers, wrappers,
registry guards, and snapshot validation.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path as PathLib
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from codeintel.analytics.testing.behavioral import importance
from codeintel.analytics.testing.behavioral import tags as behavioral_tags
from codeintel.analytics.testing.behavioral.importance import (
    compute_flakiness_score,
    compute_importance_score,
)
from codeintel.analytics.testing.behavioral.tags import infer_behavior_tags
from codeintel.analytics.testing.coverage import inputs as coverage_inputs
from codeintel.analytics.testing.coverage.inputs import (
    aggregate_test_coverage_by_function,
    aggregate_test_coverage_by_subsystem,
    load_test_graph_metrics,
)
from codeintel.analytics.testing.profiles import rows
from codeintel.analytics.testing.profiles.types import (
    BehavioralCoverageOptions,
    BehavioralLLMResult,
    ImportanceInputs,
    IoFlags,
    TestAstInfo,
    TestProfileContext,
    TestProfileOptions,
    TestRecord,
)
from codeintel.config.datasets import (
    BEHAVIORAL_COVERAGE_COLUMNS,
    TEST_PROFILE_COLUMNS,
    behavioral_coverage_row_to_tuple,
    serialize_test_profile_row,
)
from codeintel.config.primitives import SnapshotRef
from tests._helpers.factories import (
    blank_behavioral_coverage_row,
    blank_test_profile_row,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from pathlib import Path

    import duckdb

    from codeintel.analytics.testing.profiles.types import (
        BehavioralLLMRequest,
    )
    from codeintel.storage.gateway import StorageGateway


@contextmanager
def _override(obj: object, name: str, value: object) -> Iterator[None]:
    """Context manager to temporarily override an attribute."""
    original = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, original)


class _FakeCon:
    """Fake database connection for testing without real DB."""

    def __init__(self) -> None:
        self.executed: list[tuple[str, list[object] | None]] = []
        self.executemany_calls: list[tuple[str, list[list[object]]]] = []

    def execute(self, sql: str, params: list[object] | None = None) -> _FakeCon:
        """Record execute call and return self for chaining.

        Returns
        -------
        _FakeCon
            Self for method chaining.
        """
        self.executed.append((sql, params))
        return self

    def executemany(self, sql: str, params_list: list[list[object]]) -> None:
        """Record executemany call."""
        self.executemany_calls.append((sql, params_list))


class _FakeIbis:
    """Fake IbisGateway for testing without real DB."""

    def __init__(self) -> None:
        self.delete_calls: list[tuple[str, object | None]] = []
        self.table_calls: list[str] = []
        self.write_calls: list[tuple[str, list[tuple[object, ...]], list[str] | None]] = []

    def table(self, table_key: str) -> SimpleNamespace:
        """Return a fake table object with repo/commit columns.

        Returns
        -------
        SimpleNamespace
            Fake table exposing repo and commit columns.
        """
        self.table_calls.append(table_key)
        return SimpleNamespace(repo=_FakeIbisColumn("repo"), commit=_FakeIbisColumn("commit"))

    def delete(self, table_key: str, *, where: object | None = None) -> int:
        """Record delete call and return a fake deleted row count.

        Returns
        -------
        int
            Sentinel row count indicating deletion attempt.
        """
        self.delete_calls.append((table_key, where))
        return -1

    def write(
        self,
        table_key: str,
        data: list[tuple[object, ...]],
        *,
        columns: list[str] | None = None,
    ) -> SimpleNamespace:
        """Record write call and return fake result.

        Returns
        -------
        SimpleNamespace
            Fake WriteResult.
        """
        self.write_calls.append((table_key, data, columns))
        return SimpleNamespace(rows_affected=len(data), method="insert_values")


class _FakeIbisColumn:
    """Fake Ibis column expression used by _FakeIbis.table()."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __eq__(self, _other: object) -> bool:
        return True

    def __hash__(self) -> int:
        return hash(self.name)


def _default_options() -> tuple[TestProfileOptions, BehavioralCoverageOptions]:
    """Create default test and behavioral coverage options.

    Returns
    -------
    tuple[TestProfileOptions, BehavioralCoverageOptions]
        Tuple of test profile and behavioral coverage options.
    """
    return TestProfileOptions(), BehavioralCoverageOptions()


def _make_snapshot(repo_root: Path | None = None) -> SnapshotRef:
    """Create a standard test snapshot reference.

    Parameters
    ----------
    repo_root
        Optional repo root path; defaults to Path.cwd() if not provided.

    Returns
    -------
    SnapshotRef
        Configured snapshot reference with test defaults.
    """
    return SnapshotRef(
        repo="demo/repo",
        commit="deadbeef",
        repo_root=repo_root if repo_root is not None else PathLib.cwd(),
    )


def _snapshot_cfg() -> tuple[SnapshotRef, TestProfileOptions, BehavioralCoverageOptions]:
    """Create snapshot and options for tests without file I/O.

    Returns
    -------
    tuple[SnapshotRef, TestProfileOptions, BehavioralCoverageOptions]
        Snapshot reference, test profile options, and behavioral coverage options.
    """
    return _make_snapshot(), TestProfileOptions(), BehavioralCoverageOptions()


def _configs(tmp_path: Path) -> SnapshotRef:
    """Create snapshot reference for integration-style tests.

    Parameters
    ----------
    tmp_path
        Temporary directory path for repo root.

    Returns
    -------
    SnapshotRef
        Snapshot reference with tmp_path as repo root.
    """
    return _make_snapshot(repo_root=tmp_path)


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


def test_importance_and_flakiness_scoring() -> None:
    """Validate flakiness and importance scoring produce bounded values.

    Raises
    ------
    AssertionError
        If scores fall outside expected ranges.
    """
    io_flags = IoFlags(
        uses_network=True, uses_db=False, uses_filesystem=False, uses_subprocess=False
    )
    flakiness = compute_flakiness_score(
        status="xfail",
        markers=["slow", "network"],
        duration_ms=3000.0,
        io_flags=io_flags,
        slow_test_threshold_ms=2000.0,
    )
    min_expected_flakiness = 0.4
    if not (min_expected_flakiness <= flakiness <= 1.0):
        message = "Flakiness score out of expected range."
        raise AssertionError(message)

    inputs = ImportanceInputs(
        functions_covered_count=2,
        weighted_degree=1.0,
        max_function_count=4,
        max_weighted_degree=2.0,
        subsystem_risk=0.5,
        max_subsystem_risk=1.0,
    )
    imp_score = compute_importance_score(inputs)
    if imp_score is None or not (0.0 <= imp_score <= 1.0):
        message = "Importance score out of expected range."
        raise AssertionError(message)


def test_build_test_profile_rows_round_trip() -> None:
    """Tuple-to-model mapping should align with schema constants for new helpers."""
    created_at = datetime(2024, 1, 1, tzinfo=UTC)
    snapshot, test_options, _ = _snapshot_cfg()
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
    inputs = rows.TestProfileInputs(
        functions_covered=functions_covered,
        subsystems_covered=subsystems_covered,
        tg_metrics=tg_metrics,
        ast_info=ast_info,
    )
    ctx = rows.build_test_profile_context(
        snapshot=snapshot,
        inputs=inputs,
        options=test_options,
    )

    frozen_ctx = TestProfileContext(
        snapshot=ctx.snapshot,
        options=ctx.options,
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


def test_infer_behavior_tags_basic() -> None:
    """Ensure behavior tag inference captures core markers.

    Raises
    ------
    AssertionError
        If expected tags are missing.
    """
    ast_info = TestAstInfo(
        assert_count=0,
        raise_count=0,
        uses_pytest_raises=True,
        uses_concurrency_lib=False,
        has_boundary_asserts=False,
        uses_fixtures=False,
        io_flags=IoFlags(),
    )
    tags = infer_behavior_tags(
        name="test_network_error_path",
        markers=["network", "slow"],
        io_flags=IoFlags(
            uses_network=True, uses_db=False, uses_filesystem=False, uses_subprocess=False
        ),
        ast_info=ast_info,
    )
    if "network_interaction" not in tags or "error_paths" not in tags:
        message = f"Unexpected tags: {tags}"
        raise AssertionError(message)


def test_build_behavior_rows_mixed_sources() -> None:
    """Behavior rows should preserve mixed heuristic/LLM metadata without legacy hooks."""
    fake_con = _FakeCon()
    gateway = cast("StorageGateway", SimpleNamespace(con=fake_con))
    snapshot, _, beh_options = _snapshot_cfg()

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

    profile_ctx: Mapping[str, dict[str, object]] = {
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
            snapshot.repo,
            snapshot.commit,
            test.test_id,
            None,
            test.rel_path,
            test.qualname or test.test_id,
            tags,
            tag_source,
            beh_options.heuristic_version,
            llm_model,
            llm_run_id,
            getattr(ctx, "now", datetime.now(tz=UTC)),
        )

    def _fake_llm_runner(_request: BehavioralLLMRequest) -> BehavioralLLMResult:
        return BehavioralLLMResult(tags=["db"])

    hooks = behavioral_tags.BehaviorRowHooks(
        load_tests=lambda _con, _cfg: sample_tests,
        build_ast=lambda _root, _tests, _patterns: ast_info,
        load_profile_ctx=lambda _con, _cfg: profile_ctx,
        row_builder=_fake_build_behavior_row,
    )
    with _override(
        behavioral_tags.DuckDBPolicyBackend,
        "ensure_table",
        lambda _self, _table_key: None,
    ):
        tuples = behavioral_tags.build_behavior_rows(
            gateway,
            snapshot,
            options=beh_options,
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


def test_coverage_wrappers_empty(
    tmp_path: Path, coverage_profiles_conn: duckdb.DuckDBPyConnection
) -> None:
    """Ensure coverage aggregation wrappers handle empty tables.

    Raises
    ------
    AssertionError
        If any aggregation returns a non-empty result.
    """
    snapshot = _configs(tmp_path)
    if (
        aggregate_test_coverage_by_function(coverage_profiles_conn, snapshot, loader=lambda *_: {})
        != {}
    ):
        message = "Expected empty function coverage aggregation."
        raise AssertionError(message)
    if (
        aggregate_test_coverage_by_subsystem(coverage_profiles_conn, snapshot, loader=lambda *_: {})
        != {}
    ):
        message = "Expected empty subsystem coverage aggregation."
        raise AssertionError(message)
    if load_test_graph_metrics(coverage_profiles_conn, snapshot, loader=lambda *_: {}) != {}:
        message = "Expected empty test graph metrics aggregation."
        raise AssertionError(message)


def test_test_profile_model_snapshot() -> None:
    """Deterministic snapshot of test_profile row model to catch drift."""
    snapshot, test_options, _ = _snapshot_cfg()
    created_at = datetime(2024, 1, 1, tzinfo=UTC)
    test_record = TestRecord(
        test_id="t1",
        test_goid_h128=101,
        urn="urn:t1",
        rel_path="rel.py",
        module="mod.a",
        qualname="A::test",
        language="python",
        kind="function",
        status="passed",
        duration_ms=10.5,
        markers=["fast"],
        flaky=False,
        start_line=1,
        end_line=5,
    )
    functions = {
        "t1": coverage_inputs.FunctionCoverageEntry(
            functions=[{"function_goid_h128": 1}],
            count=1,
            primary=[1],
        )
    }
    subsystems = {
        "t1": coverage_inputs.SubsystemCoverageEntry(
            subsystems=[{"subsystem_id": "s1"}],
            count=1,
            primary_subsystem_id="s1",
            max_risk_score=0.4,
        )
    }
    tg_metrics = {
        "t1": coverage_inputs.TestGraphMetrics(
            degree=2,
            weighted_degree=3.0,
            proj_degree=1,
            proj_weight=1.5,
            proj_clustering=0.2,
            proj_betweenness=0.1,
        )
    }
    ast_info = {"t1": TestAstInfo(io_flags=IoFlags(uses_network=True))}
    inputs = rows.TestProfileInputs(
        functions_covered=functions,
        subsystems_covered=subsystems,
        tg_metrics=tg_metrics,
        ast_info=ast_info,
    )
    ctx = rows.build_test_profile_context(
        snapshot=snapshot,
        inputs=inputs,
        options=test_options,
    )
    frozen_ctx = TestProfileContext(
        snapshot=ctx.snapshot,
        options=ctx.options,
        now=created_at,
        max_function_count=ctx.max_function_count,
        max_weighted_degree=ctx.max_weighted_degree,
        max_subsystem_risk=ctx.max_subsystem_risk,
        functions_covered=ctx.functions_covered,
        subsystems_covered=ctx.subsystems_covered,
        tg_metrics=ctx.tg_metrics,
        ast_info=ctx.ast_info,
    )
    model = rows.build_test_profile_rows([test_record], frozen_ctx)[0]

    expected = {
        "repo": "demo/repo",
        "commit": "deadbeef",
        "test_id": "t1",
        "test_goid_h128": 101,
        "urn": "urn:t1",
        "rel_path": "rel.py",
        "module": "mod.a",
        "qualname": "A::test",
        "language": "python",
        "kind": "function",
        "status": "passed",
        "duration_ms": 10.5,
        "markers": ["fast"],
        "flaky": False,
        "last_run_at": created_at,
        "functions_covered": [{"function_goid_h128": 1}],
        "functions_covered_count": 1,
        "primary_function_goids": [1],
        "subsystems_covered": [{"subsystem_id": "s1"}],
        "subsystems_covered_count": 1,
        "primary_subsystem_id": "s1",
        "assert_count": 0,
        "raise_count": 0,
        "uses_parametrize": False,
        "uses_fixtures": False,
        "io_bound": True,
        "uses_network": True,
        "uses_db": False,
        "uses_filesystem": False,
        "uses_subprocess": False,
        "flakiness_score": pytest.approx(0.15),
        "importance_score": pytest.approx(1.0),
        "notes": None,
        "tg_degree": 2,
        "tg_weighted_degree": 3.0,
        "tg_proj_degree": 1,
        "tg_proj_weight": 1.5,
        "tg_proj_clustering": 0.2,
        "tg_proj_betweenness": 0.1,
        "created_at": created_at,
    }
    if model != expected:
        pytest.fail(f"Snapshot mismatch for test_profile model: {model}")
    serialized = serialize_test_profile_row(model)
    if len(serialized) != len(TEST_PROFILE_COLUMNS):
        pytest.fail("Serialized tuple length mismatch for test_profile.")


def test_behavioral_coverage_serialization() -> None:
    """Ensure behavioral coverage row serialization produces correct tuple length."""
    row = blank_behavioral_coverage_row()
    row["repo"] = "r"
    row["commit"] = "c"
    row["test_id"] = "t1"
    row["rel_path"] = "p"
    row["qualname"] = "q"
    row["behavior_tags"] = []
    row["tag_source"] = "heuristic"
    row["created_at"] = datetime.now(tz=UTC)

    serialized = behavioral_coverage_row_to_tuple(row)
    if len(serialized) != len(BEHAVIORAL_COVERAGE_COLUMNS):
        pytest.fail("Serialized tuple length mismatch for behavioral_coverage.")


def test_test_profile_serialization() -> None:
    """Ensure test profile row serialization produces correct tuple length."""
    sample_row = blank_test_profile_row()
    sample_row["repo"] = "r"
    sample_row["commit"] = "c"
    sample_row["test_id"] = "t"
    sample_row["rel_path"] = "p"
    sample_row["markers"] = []
    sample_row["functions_covered"] = []
    sample_row["primary_function_goids"] = []
    sample_row["subsystems_covered"] = []
    sample_row["created_at"] = datetime.now(tz=UTC)

    serialized = serialize_test_profile_row(sample_row)
    if len(serialized) != len(TEST_PROFILE_COLUMNS):
        pytest.fail("Serialized tuple length mismatch for test_profile.")
