"""Contract and drift guards for function profile outputs."""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import pytest

from codeintel.analytics.profiles import files, functions, modules
from codeintel.analytics.testing.profiles import rows as test_rows
from codeintel.config import (
    BehavioralCoverageStepConfig,
    ProfilesAnalyticsStepConfig,
    TestProfileStepConfig,
)
from codeintel.config.datasets import (
    BEHAVIORAL_COVERAGE_COLUMNS,
    FILE_PROFILE_COLUMNS,
    FUNCTION_PROFILE_COLUMNS,
    MODULE_PROFILE_COLUMNS,
    TEST_PROFILE_COLUMNS,
    BehavioralCoverageRowModel,
    FileProfileRowModel,
    FunctionProfileRowModel,
    ModuleProfileRowModel,
    ProfileRowModel,
    behavioral_coverage_row_to_tuple,
    file_profile_row_to_tuple,
    function_profile_row_to_tuple,
    module_profile_row_to_tuple,
    serialize_test_profile_row,
)
from tests._helpers.assertions import expect_equal, expect_true
from tests._helpers.factories import make_snapshot
from tests._helpers.factories.row_factories import (
    blank_behavioral_coverage_row,
    blank_file_profile_row,
    blank_function_profile_row,
    blank_module_profile_row,
    blank_test_profile_row,
    sample_behavioral_coverage_rows,
    sample_file_profile_rows,
    sample_function_profile_rows,
    sample_module_profile_rows,
    sample_test_profile_rows,
)
from tests._helpers.gateway import GatewayFactory

WARNING_LEVEL = logging.WARNING


def _assert_no_warnings(caplog: pytest.LogCaptureFixture) -> None:
    warning_records = [record for record in caplog.records if record.levelno >= WARNING_LEVEL]
    if warning_records:
        pytest.fail(f"Unexpected warnings emitted: {warning_records}")


def _cfg() -> ProfilesAnalyticsStepConfig:
    snapshot = make_snapshot()
    return ProfilesAnalyticsStepConfig(snapshot=snapshot)


def _test_cfg() -> TestProfileStepConfig:
    snapshot = make_snapshot()
    return TestProfileStepConfig(snapshot=snapshot)


def _behavior_cfg() -> BehavioralCoverageStepConfig:
    snapshot = make_snapshot()
    return BehavioralCoverageStepConfig(snapshot=snapshot)


def _function_rows(repo: str, commit: str) -> list[FunctionProfileRowModel]:
    rows: list[FunctionProfileRowModel] = []
    for base in sample_function_profile_rows(repo, commit):
        module_name = base.get("module") or base.get("rel_path", "").replace("/", ".").removesuffix(
            ".py"
        )
        row = blank_function_profile_row()
        row.update(base)
        row.setdefault("module", module_name)
        if row.get("created_at") is None:
            row["created_at"] = datetime.now(tz=UTC)
        rows.append(row)
    return rows


def _file_rows(repo: str, commit: str) -> list[FileProfileRowModel]:
    rows: list[FileProfileRowModel] = []
    for base in sample_file_profile_rows(repo, commit):
        row = blank_file_profile_row()
        row.update(base)
        row.setdefault("language", "python")
        if row.get("created_at") is None:
            row["created_at"] = datetime.now(tz=UTC)
        rows.append(row)
    return rows


def _module_rows(repo: str, commit: str) -> list[ModuleProfileRowModel]:
    rows: list[ModuleProfileRowModel] = []
    for base in sample_module_profile_rows(repo, commit):
        row = blank_module_profile_row()
        row.update(base)
        row.setdefault("language", "python")
        if row.get("created_at") is None:
            row["created_at"] = datetime.now(tz=UTC)
        rows.append(row)
    return rows


def _test_rows(repo: str, commit: str) -> list[ProfileRowModel]:
    rows: list[ProfileRowModel] = []
    for base in sample_test_profile_rows(repo, commit):
        module_name = base.get("module") or base.get("rel_path", "").replace("/", ".").removesuffix(
            ".py"
        )
        row = blank_test_profile_row()
        row.update(base)
        row.setdefault("module", module_name)
        row.setdefault("qualname", base.get("qualname", base["test_id"].split("::")[-1]))
        if row.get("created_at") is None:
            row["created_at"] = datetime.now(tz=UTC)
        rows.append(row)
    return rows


def _behavior_rows(repo: str, commit: str) -> list[BehavioralCoverageRowModel]:
    rows: list[BehavioralCoverageRowModel] = []
    for base in sample_behavioral_coverage_rows(repo, commit):
        row = blank_behavioral_coverage_row()
        row.update(base)
        if row.get("created_at") is None:
            row["created_at"] = datetime.now(tz=UTC)
        rows.append(row)
    return rows


def test_function_profile_tuple_alignment() -> None:
    """Serializer should align with FUNCTION_PROFILE_COLUMNS."""
    row = _function_rows("r", "c")[0]

    serialized = function_profile_row_to_tuple(row)
    if len(serialized) != len(FUNCTION_PROFILE_COLUMNS):
        msg = "Function profile tuple length mismatch with column constants."
        pytest.fail(msg)


def test_file_profile_tuple_alignment() -> None:
    """Serializer should align with FILE_PROFILE_COLUMNS."""
    row = _file_rows("r", "c")[0]

    serialized = file_profile_row_to_tuple(row)
    if len(serialized) != len(FILE_PROFILE_COLUMNS):
        msg = "File profile tuple length mismatch with column constants."
        pytest.fail(msg)


def test_module_profile_tuple_alignment() -> None:
    """Serializer should align with MODULE_PROFILE_COLUMNS."""
    row = _module_rows("r", "c")[1]

    serialized = module_profile_row_to_tuple(row)
    if len(serialized) != len(MODULE_PROFILE_COLUMNS):
        msg = "Module profile tuple length mismatch with column constants."
        pytest.fail(msg)


def test_test_profile_tuple_alignment() -> None:
    """Serializer should align with TEST_PROFILE_COLUMNS."""
    row = _test_rows("r", "c")[0]

    serialized = serialize_test_profile_row(row)
    if len(serialized) != len(TEST_PROFILE_COLUMNS):
        msg = "Test profile tuple length mismatch with column constants."
        pytest.fail(msg)


def test_behavioral_coverage_tuple_alignment() -> None:
    """Serializer should align with BEHAVIORAL_COVERAGE_COLUMNS."""
    row = _behavior_rows("r", "c")[0]

    serialized = behavioral_coverage_row_to_tuple(row)
    if len(serialized) != len(BEHAVIORAL_COVERAGE_COLUMNS):
        msg = "Behavioral coverage tuple length mismatch with column constants."
        pytest.fail(msg)


def test_function_profile_writer_registry_and_prepared_statements() -> None:
    """Writer should delete then insert with registry alignment."""
    gateway = GatewayFactory().with_macros().open()
    rows = _function_rows("r", "c")
    try:
        inserted_first = functions.write_function_profile_rows(gateway, rows)
        stored_first = gateway.con.execute(
            """
            select function_goid_h128, tags, owners
            from analytics.function_profile
            order by function_goid_h128
            """
        ).fetchall()
        expect_equal(inserted_first, len(rows))
        expect_equal(len(stored_first), len(rows))
        expect_true(any(owner is None for _, _, owner in stored_first))
        expect_true(any("unicode" in (tags or "") for _, tags, _ in stored_first))

        inserted_second = functions.write_function_profile_rows(gateway, rows[:1])
        expect_equal(inserted_second, 1)
        stored_second = gateway.con.execute(
            """
            select function_goid_h128, tags, owners
            from analytics.function_profile
            order by function_goid_h128
            """
        ).fetchall()
        expect_equal(len(stored_second), 1)
        expect_equal(stored_second[0][0], rows[0]["function_goid_h128"])
    finally:
        gateway.close()


def test_file_profile_writer_registry_and_prepared_statements(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Writer should delete then insert with registry alignment."""
    gateway = GatewayFactory().with_macros().open()
    rows = _file_rows("r", "c")
    try:
        caplog.set_level("WARNING")
        inserted_first = files.write_file_profile_rows(gateway, rows)
        stored_first = gateway.con.execute(
            """
            select rel_path, module, tags
            from analytics.file_profile
            order by rel_path
            """
        ).fetchall()
        expect_equal(inserted_first, len(rows))
        expect_equal(len(stored_first), len(rows))
        expect_true(any("delta" in module for _, module, _ in stored_first))
        expect_true(any(tag_json and "unicode" in tag_json for _, _, tag_json in stored_first))

        inserted_second = files.write_file_profile_rows(gateway, rows[:1])
        expect_equal(inserted_second, 1)
        stored_second = gateway.con.execute(
            """
            select rel_path, module, tags
            from analytics.file_profile
            order by rel_path
            """
        ).fetchall()
        expect_equal(len(stored_second), 1)
        expect_true("alpha" in stored_second[0][0])
        _assert_no_warnings(caplog)
    finally:
        gateway.close()


def test_module_profile_writer_registry_and_prepared_statements(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Writer should delete then insert with registry alignment."""
    gateway = GatewayFactory().with_macros().open()
    rows = _module_rows("r", "c")
    try:
        caplog.set_level("WARNING")
        inserted_first = modules.write_module_profile_rows(gateway, rows)
        stored_first = gateway.con.execute(
            """
            select module, path, tags
            from analytics.module_profile
            order by module
            """
        ).fetchall()
        expect_equal(inserted_first, len(rows))
        expect_equal(len(stored_first), len(rows))
        expect_true(any("unicode" in path for _, path, _ in stored_first))

        inserted_second = modules.write_module_profile_rows(gateway, rows[:1])
        expect_equal(inserted_second, 1)
        stored_second = gateway.con.execute(
            """
            select module, path, tags
            from analytics.module_profile
            order by module
            """
        ).fetchall()
        expect_equal(len(stored_second), 1)
        expect_true("alpha" in stored_second[0][0])
        _assert_no_warnings(caplog)
    finally:
        gateway.close()


def test_test_profile_writer_registry_and_prepared_statements(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Writer should delete then insert with registry alignment."""
    gateway = GatewayFactory().with_macros().open()
    cfg = _test_cfg()
    rows = _test_rows(cfg.repo, cfg.commit)
    try:
        caplog.set_level("WARNING")
        inserted_first = test_rows.write_test_profile_rows(gateway, cfg, rows)
        stored_first = gateway.con.execute(
            """
            select test_id, rel_path
            from analytics.test_profile
            order by test_id
            """
        ).fetchall()
        expect_equal(inserted_first, len(rows))
        expect_equal(len(stored_first), len(rows))
        expect_true(any("test_δelta_flow" in test_id for test_id, _ in stored_first))

        inserted_second = test_rows.write_test_profile_rows(gateway, cfg, rows[:1])
        expect_equal(inserted_second, 1)
        stored_second = gateway.con.execute(
            """
            select test_id, rel_path
            from analytics.test_profile
            order by test_id
            """
        ).fetchall()
        expect_equal(len(stored_second), 1)
        expect_true("TestAlpha" in stored_second[0][0])
        _assert_no_warnings(caplog)
    finally:
        gateway.close()


def test_behavioral_coverage_writer_registry_and_prepared_statements(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Writer should delete then insert with registry alignment."""
    gateway = GatewayFactory().with_macros().open()
    cfg = _behavior_cfg()
    rows = _behavior_rows(cfg.repo, cfg.commit)
    try:
        caplog.set_level("WARNING")
        inserted_first = test_rows.write_behavioral_coverage_rows(gateway, cfg, rows)
        stored_first = gateway.con.execute(
            """
            select test_id, qualname, behavior_tags
            from analytics.behavioral_coverage
            order by test_id
            """
        ).fetchall()
        expect_equal(inserted_first, len(rows))
        expect_equal(len(stored_first), len(rows))
        expect_true(any("test_δelta_flow" in test_id for test_id, _, _ in stored_first))

        inserted_second = test_rows.write_behavioral_coverage_rows(gateway, cfg, rows[:1])
        expect_equal(inserted_second, 1)
        stored_second = gateway.con.execute(
            """
            select test_id, qualname, behavior_tags
            from analytics.behavioral_coverage
            order by test_id
            """
        ).fetchall()
        expect_equal(len(stored_second), 1)
        expect_true("TestAlpha" in stored_second[0][0])
        _assert_no_warnings(caplog)
    finally:
        gateway.close()
