"""Registry and snapshot guards for analytics tests_profiles helpers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from codeintel.analytics.tests_profiles import coverage_inputs, rows
from codeintel.analytics.tests_profiles.types import (
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
    BehavioralCoverageRowModel,
    behavioral_coverage_row_to_tuple,
    serialize_test_profile_row,
)


class _FakeCon:
    def __init__(self) -> None:
        self.executed: list[tuple[str, list[object] | None]] = []
        self.executemany_calls: list[tuple[str, list[list[object]]]] = []

    def execute(self, sql: str, params: list[object] | None = None) -> _FakeCon:
        self.executed.append((sql, params))
        return self

    def executemany(self, sql: str, params_list: list[list[object]]) -> None:
        self.executemany_calls.append((sql, params_list))


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


def test_test_profile_model_snapshot() -> None:
    """Deterministic snapshot of test_profile row model to catch drift."""
    test_cfg, _ = _snapshot_cfg()
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
    ctx = rows.build_test_profile_context(
        cfg=test_cfg,
        functions_covered=functions,
        subsystems_covered=subsystems,
        tg_metrics=tg_metrics,
        ast_info=ast_info,
    )
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
    model = rows.build_test_profile_rows([test_record], frozen_ctx)[0]

    expected = {
        "repo": "r",
        "commit": "c",
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


def test_behavioral_writer_registry_guard() -> None:
    """Ensure behavioral writer honors registry columns and schema guardrails."""
    fake_con = _FakeCon()
    gateway = cast("StorageGateway", SimpleNamespace(con=fake_con))
    _, beh_cfg = _snapshot_cfg()
    row: BehavioralCoverageRowModel = dict.fromkeys(BEHAVIORAL_COVERAGE_COLUMNS)  # type: ignore[arg-type]
    row["repo"] = "r"
    row["commit"] = "c"
    row["test_id"] = "t1"
    row["rel_path"] = "p"
    row["qualname"] = "q"
    row["behavior_tags"] = []
    row["tag_source"] = "heuristic"
    row["created_at"] = datetime.now(tz=UTC)

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

        inserted = rows.write_behavioral_coverage_rows(gateway, beh_cfg, [row])
        if inserted != 1:
            pytest.fail("Writer did not report one inserted row.")
        if not fake_con.executed[0][0].startswith("DELETE FROM analytics.behavioral_coverage"):
            pytest.fail("Delete was not issued for behavioral_coverage.")
        if not fake_con.executemany_calls[0][0].startswith(
            "INSERT INTO analytics.behavioral_coverage"
        ):
            pytest.fail("Insert was not issued for behavioral_coverage.")
        serialized = behavioral_coverage_row_to_tuple(row)
        if len(serialized) != len(BEHAVIORAL_COVERAGE_COLUMNS):
            pytest.fail("Serialized tuple length mismatch for behavioral_coverage.")
