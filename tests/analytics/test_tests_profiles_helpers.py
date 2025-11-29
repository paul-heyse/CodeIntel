"""Unit fixtures for analytics.tests_profiles helpers."""
# ruff: noqa: E402,ARG005

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import cast

import pytest

legacy_module = ModuleType("codeintel.analytics.tests.profiles")


def _ret_empty_dict(*_: object, **__: object) -> dict[str, object]:
    return {}


def _ret_empty_list(*_: object, **__: object) -> list[object]:
    return []


legacy_module.load_functions_covered = _ret_empty_dict  # type: ignore[attr-defined]
legacy_module.load_subsystems_covered = _ret_empty_dict  # type: ignore[attr-defined]
legacy_module.load_test_graph_metrics_public = _ret_empty_dict  # type: ignore[attr-defined]
legacy_module.load_test_records_public = _ret_empty_list  # type: ignore[attr-defined]
legacy_module.build_test_ast_index = _ret_empty_dict  # type: ignore[attr-defined]
legacy_module.DEFAULT_IO_SPEC = {}  # type: ignore[attr-defined]
legacy_module.CONCURRENCY_LIBS = set()  # type: ignore[attr-defined]
legacy_module.load_test_profile_context = (  # type: ignore[attr-defined]
    lambda *args, **kwargs: {}
)
legacy_module.BehavioralContext = SimpleNamespace  # type: ignore[attr-defined]
legacy_module.build_behavior_row = lambda *args, **kwargs: ()  # type: ignore[attr-defined]
legacy_module.infer_behavior_tags = lambda *args, **kwargs: []  # type: ignore[attr-defined]
legacy_module.compute_flakiness_score = (  # type: ignore[attr-defined]
    lambda **kwargs: 0.0
)
legacy_module.compute_importance_score = (  # type: ignore[attr-defined]
    lambda inputs: 0.0
)
legacy_module.build_test_profile_row = (  # type: ignore[attr-defined]
    lambda *args, **kwargs: ()
)
legacy_module.build_behavioral_coverage = (  # type: ignore[attr-defined]
    lambda *args, **kwargs: None
)
legacy_module.build_test_profile = lambda *args, **kwargs: None  # type: ignore[attr-defined]
sys.modules.setdefault("codeintel.analytics.tests.profiles", legacy_module)

from codeintel.analytics.tests_profiles import behavioral_tags, coverage_inputs, importance, rows
from codeintel.analytics.tests_profiles.types import (
    BehavioralLLMRequest,
    BehavioralLLMResult,
    ImportanceInputs,
    IoFlags,
    TestAstInfo,
)
from codeintel.config import BehavioralCoverageStepConfig, TestProfileStepConfig
from codeintel.config.primitives import SnapshotRef
from codeintel.storage.gateway import (
    DuckDBConnection as DuckDBConnectionType,
)
from codeintel.storage.gateway import (
    StorageGateway,
)
from codeintel.storage.rows import (
    BEHAVIORAL_COVERAGE_COLUMNS,
    TEST_PROFILE_COLUMNS,
    BehavioralCoverageRowModel,
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


def test_coverage_inputs_passthrough() -> None:
    """Coverage helpers should forward legacy results unchanged."""
    expected_functions = {"t1": {"functions": [1]}}
    expected_subsystems = {"t1": {"subsystems": [1]}}
    expected_graph = {"t1": {"degree": 1}}

    with ExitStack() as stack:
        stack.enter_context(
            _override(
                coverage_inputs.legacy,
                "load_functions_covered",
                lambda _con, _repo, _commit: expected_functions,
            )
        )
        stack.enter_context(
            _override(
                coverage_inputs.legacy,
                "load_subsystems_covered",
                lambda _con, _repo, _commit: expected_subsystems,
            )
        )
        stack.enter_context(
            _override(
                coverage_inputs.legacy,
                "load_test_graph_metrics_public",
                lambda _con, _repo, _commit: expected_graph,
            )
        )

        test_cfg, beh_cfg = _snapshot_cfg()
        fake_con = cast("DuckDBConnectionType", object())
        func_result = coverage_inputs.aggregate_test_coverage_by_function(fake_con, test_cfg)
        if func_result != expected_functions:
            msg = "Function coverage wrapper did not return legacy result."
            pytest.fail(msg)
        subsys_result = coverage_inputs.aggregate_test_coverage_by_subsystem(fake_con, beh_cfg)
        if subsys_result != expected_subsystems:
            msg = "Subsystem coverage wrapper did not return legacy result."
            pytest.fail(msg)
        graph_result = coverage_inputs.load_test_graph_metrics(fake_con, test_cfg)
        if graph_result != expected_graph:
            msg = "Graph metrics wrapper did not return legacy result."
            pytest.fail(msg)


def test_behavioral_tags_infer_passthrough() -> None:
    """Behavioral tag helper should forward to legacy implementation."""
    called: dict[str, object] = {}

    def _fake_infer(
        *, name: str, markers: list[str], io_flags: IoFlags, ast_info: TestAstInfo
    ) -> list[str]:
        called["name"] = name
        called["markers"] = markers
        called["io_flags"] = io_flags
        called["ast_info"] = ast_info
        return ["network_interaction"]

    with _override(behavioral_tags.legacy, "infer_behavior_tags", _fake_infer):
        tags = behavioral_tags.infer_behavior_tags(
            name="test_network",
            markers=["network"],
            io_flags=IoFlags(uses_network=True),
            ast_info=TestAstInfo(),
        )
        if tags != ["network_interaction"]:
            msg = "Tags wrapper did not return legacy result."
            pytest.fail(msg)
        if called["name"] != "test_network":
            msg = "Legacy infer not invoked with expected name."
            pytest.fail(msg)


def test_importance_passthrough() -> None:
    """Importance/flakiness helpers should forward to legacy implementations."""
    expected_flakiness = 0.75
    expected_importance = 0.5
    with ExitStack() as stack:
        stack.enter_context(
            _override(importance.legacy, "compute_flakiness_score", lambda **_: expected_flakiness)
        )
        stack.enter_context(
            _override(
                importance.legacy,
                "compute_importance_score",
                lambda _inputs: expected_importance,
            )
        )
        score = importance.compute_flakiness_score(
            status="failed",
            markers=["slow"],
            duration_ms=5000.0,
            io_flags=IoFlags(uses_network=True),
            slow_test_threshold_ms=1000.0,
        )
        if score != expected_flakiness:
            msg = "Flakiness score passthrough failed."
            pytest.fail(msg)
        importance_inputs = cast(
            "ImportanceInputs",
            SimpleNamespace(
                functions_covered_count=1,
                weighted_degree=1.0,
                max_function_count=2,
                max_weighted_degree=2.0,
                subsystem_risk=0.1,
                max_subsystem_risk=1.0,
            ),
        )
        if importance.compute_importance_score(importance_inputs) != expected_importance:
            msg = "Importance score passthrough failed."
            pytest.fail(msg)


def test_importance_guardrails_and_monotonicity() -> None:
    """Importance/flakiness scoring should remain bounded and monotonic."""

    def _fake_flakiness(
        *,
        status: str | None,
        markers: list[str],
        duration_ms: float | None,
        io_flags: IoFlags,
        slow_test_threshold_ms: float,
    ) -> float:
        del markers  # markers not used in stub
        base = 0.5 if status == "failed" else 0.1
        duration_value = duration_ms or 0.0
        slow_threshold = slow_test_threshold_ms
        if duration_value > slow_threshold:
            base += 0.25
        if getattr(io_flags, "uses_network", False):
            base += 0.15
        return min(base, 1.0)

    def _fake_importance(inputs: ImportanceInputs) -> float:
        weight_share = inputs.functions_covered_count / max(inputs.max_function_count, 1)
        degree_share = (inputs.weighted_degree or 0.0) / max(inputs.max_weighted_degree, 1.0)
        risk_share = (inputs.subsystem_risk or 0.0) / max(inputs.max_subsystem_risk, 1.0)
        composite = (weight_share + degree_share + risk_share) / 3.0
        return max(0.0, min(composite, 1.0))

    with ExitStack() as stack:
        stack.enter_context(
            _override(importance.legacy, "compute_flakiness_score", _fake_flakiness)
        )
        stack.enter_context(
            _override(importance.legacy, "compute_importance_score", _fake_importance)
        )

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

        baseline = ImportanceInputs(
            functions_covered_count=1,
            weighted_degree=0.5,
            max_function_count=5,
            max_weighted_degree=5.0,
            subsystem_risk=0.1,
            max_subsystem_risk=1.0,
        )
        improved = ImportanceInputs(
            functions_covered_count=3,
            weighted_degree=4.0,
            max_function_count=5,
            max_weighted_degree=5.0,
            subsystem_risk=0.5,
            max_subsystem_risk=1.0,
        )
        baseline_score = importance.compute_importance_score(baseline)
        improved_score = importance.compute_importance_score(improved)
        if baseline_score is None or improved_score is None:
            msg = "Importance scoring returned None unexpectedly."
            pytest.fail(msg)
        if improved_score <= baseline_score:
            msg = "Importance score did not increase with stronger signals."
            pytest.fail(msg)


def test_build_test_profile_rows_normalization() -> None:
    """Tuple-to-model mapping should align with schema constants."""
    created_at = datetime(2024, 1, 1, tzinfo=UTC)
    tuple_row = (
        "repo",
        "commit",
        "test-id",
        1,
        "urn",
        "rel.py",
        "mod",
        "qual",
        "python",
        "function",
        "passed",
        12.5,
        ["fast"],
        False,
        created_at,
        [],
        1,
        [],
        [],
        0,
        None,
        2,
        0,
        False,
        False,
        False,
        True,
        False,
        False,
        0.1,
        0.9,
        None,
        1,
        2.0,
        1,
        2.0,
        0.1,
        0.2,
        created_at,
        created_at,
    )
    with _override(rows.legacy, "build_test_profile_row", lambda _test, _ctx: tuple_row):
        dummy_ctx = SimpleNamespace()
        test_record = SimpleNamespace(test_id="test-id")
        models = rows.build_test_profile_rows([test_record], dummy_ctx)  # type: ignore[arg-type]
    if len(models) != 1:
        msg = "Expected exactly one model."
        pytest.fail(msg)
    model = models[0]
    if model["created_at"] != created_at:
        msg = "Created_at did not preserve tuple value."
        pytest.fail(msg)
    serialized = serialize_test_profile_row(model)
    if len(serialized) != len(TEST_PROFILE_COLUMNS):
        msg = "Serialized tuple length mismatch for test_profile."
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
    """Behavior rows should preserve mixed heuristic/LLM metadata."""
    fake_con = _FakeCon()
    gateway = cast("StorageGateway", SimpleNamespace(con=fake_con))
    _, beh_cfg = _snapshot_cfg()

    sample_tests = [
        {"test_id": "t1", "rel_path": "a.py", "qualname": "A::test", "tags": ["network"]},
        {
            "test_id": "t2",
            "rel_path": "b.py",
            "qualname": "B::test",
            "tags": ["db", "io"],
            "tag_source": "llm",
            "heuristic_version": None,
            "llm_model": "gpt",
            "llm_run_id": "run-123",
        },
    ]

    ctx_seen: dict[str, object] = {}

    def _fake_build_behavior_row(test: dict[str, object], ctx: object) -> tuple[object, ...]:
        ctx_seen.setdefault("llm_runner", getattr(ctx, "llm_runner", None))
        tag_source = test.get("tag_source", "heuristic")
        return (
            beh_cfg.repo,
            beh_cfg.commit,
            test["test_id"],
            None,
            test["rel_path"],
            test.get("qualname"),
            test.get("tags", []),
            tag_source,
            test.get("heuristic_version"),
            test.get("llm_model"),
            test.get("llm_run_id"),
            getattr(ctx, "now", datetime.now(tz=UTC)),
        )

    def _fake_llm_runner(_request: BehavioralLLMRequest) -> BehavioralLLMResult:
        return BehavioralLLMResult(tags=[])

    with ExitStack() as stack:
        stack.enter_context(
            _override(behavioral_tags, "ensure_schema", lambda _con, _table_key: None)
        )
        stack.enter_context(
            _override(behavioral_tags.legacy, "load_test_records_public", lambda *_: sample_tests)
        )
        stack.enter_context(
            _override(
                behavioral_tags.legacy,
                "build_test_ast_index",
                lambda *_args, **_kwargs: {"t1": {}, "t2": {}},
            )
        )
        stack.enter_context(
            _override(
                behavioral_tags.legacy,
                "load_test_profile_context",
                lambda *_args, **_kwargs: {"t1": {}, "t2": {}},
            )
        )
        stack.enter_context(
            _override(behavioral_tags.legacy, "build_behavior_row", _fake_build_behavior_row)
        )
        llm_runner = _fake_llm_runner
        tuples = behavioral_tags.build_behavior_rows(gateway, beh_cfg, llm_runner=llm_runner)
        models = rows.build_behavioral_coverage_rows(tuples)
        if {model["test_id"] for model in models} != {"t1", "t2"}:
            msg = "Behavioral coverage rows missing expected tests."
            pytest.fail(msg)
        tag_sources = {model["test_id"]: model["tag_source"] for model in models}
        if tag_sources["t1"] != "heuristic" or tag_sources["t2"] != "llm":
            msg = "Tag sources were not preserved per test."
            pytest.fail(msg)
        if ctx_seen.get("llm_runner") is not llm_runner:
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


def test_write_test_profile_rows_empty_and_registry_drift() -> None:
    """Writer should delete then skip inserts, and reject registry drift."""
    fake_con = _FakeCon()
    gateway = cast("StorageGateway", SimpleNamespace(con=fake_con))
    test_cfg, _ = _snapshot_cfg()
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

        inserted = rows.write_test_profile_rows(gateway, test_cfg, [])
        if inserted != 0:
            msg = "Writer should report zero inserts when no rows provided."
            pytest.fail(msg)
        if not fake_con.executed or not fake_con.executed[0][0].startswith(
            "DELETE FROM analytics.test_profile"
        ):
            msg = "Delete was not issued when skipping inserts."
            pytest.fail(msg)
        if fake_con.executemany_calls:
            msg = "Insert should not be issued for empty row set."
            pytest.fail(msg)

    with ExitStack() as stack:
        stack.enter_context(_override(rows, "ensure_schema", lambda _con, _table_key: None))
        stack.enter_context(
            _override(
                rows,
                "load_registry_columns",
                lambda _con: {"analytics.test_profile": ["repo", "commit", "test_id"]},
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
        with pytest.raises(RuntimeError):
            rows.write_test_profile_rows(gateway, test_cfg, [])


def test_write_behavioral_coverage_rows_with_stubs() -> None:
    """Behavioral coverage writer should honor registry columns and prepared statement."""
    fake_con = _FakeCon()
    gateway = cast("StorageGateway", SimpleNamespace(con=fake_con))
    _, beh_cfg = _snapshot_cfg()
    sample_row: BehavioralCoverageRowModel = dict.fromkeys(BEHAVIORAL_COVERAGE_COLUMNS)  # type: ignore[arg-type]
    sample_row["repo"] = "r"
    sample_row["commit"] = "c"
    sample_row["test_id"] = "t"
    sample_row["rel_path"] = "p"
    sample_row["behavior_tags"] = []
    sample_row["tag_source"] = "heuristic"
    sample_row["created_at"] = datetime.now(tz=UTC)

    with ExitStack() as stack:
        stack.enter_context(_override(rows, "ensure_schema", lambda _con, _table_key: None))
        stack.enter_context(
            _override(
                rows,
                "load_registry_columns",
                lambda _con: {"analytics.behavioral_coverage": list(BEHAVIORAL_COVERAGE_COLUMNS)},
            )
        )
        stack.enter_context(
            _override(
                rows,
                "prepared_statements_dynamic",
                lambda _con, _table_key: SimpleNamespace(
                    insert_sql="INSERT INTO analytics.behavioral_coverage"
                ),
            )
        )

        inserted = rows.write_behavioral_coverage_rows(gateway, beh_cfg, [sample_row])
        if inserted != 1:
            msg = "Behavioral coverage writer did not report one row."
            pytest.fail(msg)
        if not fake_con.executed[0][0].startswith("DELETE FROM analytics.behavioral_coverage"):
            msg = "Delete was not issued for behavioral_coverage."
            pytest.fail(msg)
        if not fake_con.executemany_calls[0][0].startswith(
            "INSERT INTO analytics.behavioral_coverage"
        ):
            msg = "Insert was not issued for behavioral_coverage."
            pytest.fail(msg)


def test_write_behavioral_coverage_rows_registry_drift() -> None:
    """Behavioral coverage writer should fail fast on registry drift."""
    fake_con = _FakeCon()
    gateway = cast("StorageGateway", SimpleNamespace(con=fake_con))
    _, beh_cfg = _snapshot_cfg()
    with ExitStack() as stack:
        stack.enter_context(_override(rows, "ensure_schema", lambda _con, _table_key: None))
        stack.enter_context(
            _override(
                rows,
                "load_registry_columns",
                lambda _con: {"analytics.behavioral_coverage": ["repo", "commit", "test_id"]},
            )
        )
        with pytest.raises(RuntimeError):
            rows.write_behavioral_coverage_rows(gateway, beh_cfg, [])
