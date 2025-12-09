"""Contract and drift guards for function profile outputs."""

from __future__ import annotations

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
    behavioral_coverage_row_to_tuple,
    file_profile_row_to_tuple,
    function_profile_row_to_tuple,
    module_profile_row_to_tuple,
    serialize_test_profile_row,
)
from tests._helpers.assertions import expect_equal, expect_true
from tests._helpers.factories import (
    make_snapshot,
    sample_behavioral_coverage_rows,
    sample_file_profile_rows,
    sample_function_profile_rows,
    sample_module_profile_rows,
    sample_test_profile_rows,
)
from tests._helpers.gateway import GatewayFactory


def _cfg() -> ProfilesAnalyticsStepConfig:
    snapshot = make_snapshot()
    return ProfilesAnalyticsStepConfig(snapshot=snapshot)


def _test_cfg() -> TestProfileStepConfig:
    snapshot = make_snapshot()
    return TestProfileStepConfig(snapshot=snapshot)


def _behavior_cfg() -> BehavioralCoverageStepConfig:
    snapshot = make_snapshot()
    return BehavioralCoverageStepConfig(snapshot=snapshot)


def test_function_profile_tuple_alignment() -> None:
    """Serializer should align with FUNCTION_PROFILE_COLUMNS."""
    row = sample_function_profile_rows("r", "c")[0]

    serialized = function_profile_row_to_tuple(row)
    if len(serialized) != len(FUNCTION_PROFILE_COLUMNS):
        msg = "Function profile tuple length mismatch with column constants."
        pytest.fail(msg)


def test_file_profile_tuple_alignment() -> None:
    """Serializer should align with FILE_PROFILE_COLUMNS."""
    row = sample_file_profile_rows("r", "c")[0]

    serialized = file_profile_row_to_tuple(row)
    if len(serialized) != len(FILE_PROFILE_COLUMNS):
        msg = "File profile tuple length mismatch with column constants."
        pytest.fail(msg)


def test_module_profile_tuple_alignment() -> None:
    """Serializer should align with MODULE_PROFILE_COLUMNS."""
    row = sample_module_profile_rows("r", "c")[1]

    serialized = module_profile_row_to_tuple(row)
    if len(serialized) != len(MODULE_PROFILE_COLUMNS):
        msg = "Module profile tuple length mismatch with column constants."
        pytest.fail(msg)


def test_test_profile_tuple_alignment() -> None:
    """Serializer should align with TEST_PROFILE_COLUMNS."""
    row = sample_test_profile_rows("r", "c")[0]

    serialized = serialize_test_profile_row(row)
    if len(serialized) != len(TEST_PROFILE_COLUMNS):
        msg = "Test profile tuple length mismatch with column constants."
        pytest.fail(msg)


def test_behavioral_coverage_tuple_alignment() -> None:
    """Serializer should align with BEHAVIORAL_COVERAGE_COLUMNS."""
    row = sample_behavioral_coverage_rows("r", "c")[0]

    serialized = behavioral_coverage_row_to_tuple(row)
    if len(serialized) != len(BEHAVIORAL_COVERAGE_COLUMNS):
        msg = "Behavioral coverage tuple length mismatch with column constants."
        pytest.fail(msg)


def test_function_profile_writer_registry_and_prepared_statements() -> None:
    """Writer should delete then insert with registry alignment."""
    gateway = GatewayFactory().with_macros().open()
    rows = sample_function_profile_rows("r", "c")
    try:
        inserted_first = functions.write_function_profile_rows(gateway, rows)
        inserted_second = functions.write_function_profile_rows(gateway, rows[:1])
        expect_equal(inserted_first, len(rows))
        expect_equal(inserted_second, 1)
        stored = gateway.con.execute(
            """
            select function_goid_h128, tags, owners
            from analytics.function_profile
            order by function_goid_h128
            """
        ).fetchall()
        expect_equal(len(stored), 1)
        expect_equal(stored[0][0], rows[0]["function_goid_h128"])
        expect_true("io" in stored[0][1])
    finally:
        gateway.close()


def test_file_profile_writer_registry_and_prepared_statements() -> None:
    """Writer should delete then insert with registry alignment."""
    gateway = GatewayFactory().with_macros().open()
    rows = sample_file_profile_rows("r", "c")
    try:
        inserted_first = files.write_file_profile_rows(gateway, rows)
        inserted_second = files.write_file_profile_rows(gateway, rows[:1])
        expect_equal(inserted_first, len(rows))
        expect_equal(inserted_second, 1)
        stored = gateway.con.execute(
            """
            select rel_path, module, tags
            from analytics.file_profile
            order by rel_path
            """
        ).fetchall()
        expect_equal(len(stored), 1)
        expect_true("alpha" in stored[0][0])
    finally:
        gateway.close()


def test_module_profile_writer_registry_and_prepared_statements() -> None:
    """Writer should delete then insert with registry alignment."""
    gateway = GatewayFactory().with_macros().open()
    rows = sample_module_profile_rows("r", "c")
    try:
        inserted_first = modules.write_module_profile_rows(gateway, rows)
        inserted_second = modules.write_module_profile_rows(gateway, rows[:1])
        expect_equal(inserted_first, len(rows))
        expect_equal(inserted_second, 1)
        stored = gateway.con.execute(
            """
            select module, path, tags
            from analytics.module_profile
            order by module
            """
        ).fetchall()
        expect_equal(len(stored), 1)
        expect_true("alpha" in stored[0][0])
    finally:
        gateway.close()


def test_test_profile_writer_registry_and_prepared_statements() -> None:
    """Writer should delete then insert with registry alignment."""
    gateway = GatewayFactory().with_macros().open()
    cfg = _test_cfg()
    rows = sample_test_profile_rows(cfg.repo, cfg.commit)
    try:
        inserted_first = test_rows.write_test_profile_rows(gateway, cfg, rows)
        inserted_second = test_rows.write_test_profile_rows(gateway, cfg, rows[:1])
        expect_equal(inserted_first, len(rows))
        expect_equal(inserted_second, 1)
        stored = gateway.con.execute(
            """
            select test_id, rel_path
            from analytics.test_profile
            order by test_id
            """
        ).fetchall()
        expect_equal(len(stored), 1)
        expect_true("TestAlpha" in stored[0][0])
    finally:
        gateway.close()


def test_behavioral_coverage_writer_registry_and_prepared_statements() -> None:
    """Writer should delete then insert with registry alignment."""
    gateway = GatewayFactory().with_macros().open()
    cfg = _behavior_cfg()
    rows = sample_behavioral_coverage_rows(cfg.repo, cfg.commit)
    try:
        inserted_first = test_rows.write_behavioral_coverage_rows(gateway, cfg, rows)
        inserted_second = test_rows.write_behavioral_coverage_rows(gateway, cfg, rows[:1])
        expect_equal(inserted_first, len(rows))
        expect_equal(inserted_second, 1)
        stored = gateway.con.execute(
            """
            select test_id, qualname, behavior_tags
            from analytics.behavioral_coverage
            order by test_id
            """
        ).fetchall()
        expect_equal(len(stored), 1)
        expect_true("TestAlpha" in stored[0][0])
    finally:
        gateway.close()
