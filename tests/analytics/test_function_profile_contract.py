"""Contract and drift guards for function profile outputs."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from codeintel.analytics.profiles import files, functions, modules
from codeintel.analytics.tests_profiles import rows as test_rows
from codeintel.config import (
    BehavioralCoverageStepConfig,
    ProfilesAnalyticsStepConfig,
    TestProfileStepConfig,
)
from codeintel.config.primitives import SnapshotRef
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.rows import (
    BEHAVIORAL_COVERAGE_COLUMNS,
    FILE_PROFILE_COLUMNS,
    FUNCTION_PROFILE_COLUMNS,
    MODULE_PROFILE_COLUMNS,
    TEST_PROFILE_COLUMNS,
    BehavioralCoverageRowModel,
    FileProfileRowModel,
    FunctionProfileRowModel,
    ModuleProfileRowModel,
    TestProfileRowModel,
    behavioral_coverage_row_to_tuple,
    file_profile_row_to_tuple,
    function_profile_row_to_tuple,
    module_profile_row_to_tuple,
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


def _cfg() -> ProfilesAnalyticsStepConfig:
    snapshot = SnapshotRef(repo="r", commit="c", repo_root=Path.cwd())
    return ProfilesAnalyticsStepConfig(snapshot=snapshot)


def _test_cfg() -> TestProfileStepConfig:
    snapshot = SnapshotRef(repo="r", commit="c", repo_root=Path.cwd())
    return TestProfileStepConfig(snapshot=snapshot)


def _behavior_cfg() -> BehavioralCoverageStepConfig:
    snapshot = SnapshotRef(repo="r", commit="c", repo_root=Path.cwd())
    return BehavioralCoverageStepConfig(snapshot=snapshot)


def test_function_profile_tuple_alignment() -> None:
    """Serializer should align with FUNCTION_PROFILE_COLUMNS."""
    row: FunctionProfileRowModel = dict.fromkeys(FUNCTION_PROFILE_COLUMNS)  # type: ignore[arg-type]
    row["repo"] = "r"
    row["commit"] = "c"
    row["function_goid_h128"] = 1
    row["urn"] = "urn:fn"
    row["rel_path"] = "path.py"
    row["language"] = "python"
    row["kind"] = "function"
    row["qualname"] = "mod.fn"
    row["created_at"] = datetime.now(tz=UTC)

    serialized = function_profile_row_to_tuple(row)
    if len(serialized) != len(FUNCTION_PROFILE_COLUMNS):
        msg = "Function profile tuple length mismatch with column constants."
        pytest.fail(msg)


def test_file_profile_tuple_alignment() -> None:
    """Serializer should align with FILE_PROFILE_COLUMNS."""
    row: FileProfileRowModel = dict.fromkeys(FILE_PROFILE_COLUMNS)  # type: ignore[arg-type]
    row["repo"] = "r"
    row["commit"] = "c"
    row["rel_path"] = "a.py"
    row["module"] = "m"
    row["tags"] = "[]"
    row["owners"] = "[]"
    row["created_at"] = datetime.now(tz=UTC)

    serialized = file_profile_row_to_tuple(row)
    if len(serialized) != len(FILE_PROFILE_COLUMNS):
        msg = "File profile tuple length mismatch with column constants."
        pytest.fail(msg)


def test_module_profile_tuple_alignment() -> None:
    """Serializer should align with MODULE_PROFILE_COLUMNS."""
    row: ModuleProfileRowModel = dict.fromkeys(MODULE_PROFILE_COLUMNS)  # type: ignore[arg-type]
    row["repo"] = "r"
    row["commit"] = "c"
    row["module"] = "pkg.mod"
    row["path"] = "pkg/mod.py"
    row["tags"] = "[]"
    row["owners"] = "[]"
    row["created_at"] = datetime.now(tz=UTC)

    serialized = module_profile_row_to_tuple(row)
    if len(serialized) != len(MODULE_PROFILE_COLUMNS):
        msg = "Module profile tuple length mismatch with column constants."
        pytest.fail(msg)


def test_test_profile_tuple_alignment() -> None:
    """Serializer should align with TEST_PROFILE_COLUMNS."""
    row: TestProfileRowModel = dict.fromkeys(TEST_PROFILE_COLUMNS)  # type: ignore[arg-type]
    row["repo"] = "r"
    row["commit"] = "c"
    row["test_id"] = "t::case"
    row["rel_path"] = "tests/test_a.py"
    row["markers"] = []
    row["functions_covered"] = []
    row["primary_function_goids"] = []
    row["subsystems_covered"] = []
    row["created_at"] = datetime.now(tz=UTC)

    serialized = serialize_test_profile_row(row)
    if len(serialized) != len(TEST_PROFILE_COLUMNS):
        msg = "Test profile tuple length mismatch with column constants."
        pytest.fail(msg)


def test_behavioral_coverage_tuple_alignment() -> None:
    """Serializer should align with BEHAVIORAL_COVERAGE_COLUMNS."""
    row: BehavioralCoverageRowModel = dict.fromkeys(BEHAVIORAL_COVERAGE_COLUMNS)  # type: ignore[arg-type]
    row["repo"] = "r"
    row["commit"] = "c"
    row["test_id"] = "t::case"
    row["rel_path"] = "tests/test_a.py"
    row["qualname"] = "TestA::test_a"
    row["behavior_tags"] = []
    row["tag_source"] = "heuristic"
    row["created_at"] = datetime.now(tz=UTC)

    serialized = behavioral_coverage_row_to_tuple(row)
    if len(serialized) != len(BEHAVIORAL_COVERAGE_COLUMNS):
        msg = "Behavioral coverage tuple length mismatch with column constants."
        pytest.fail(msg)


def test_function_profile_writer_registry_and_prepared_statements() -> None:
    """Writer should delete then insert with registry alignment."""
    fake_con = _FakeCon()
    gateway = cast("StorageGateway", SimpleNamespace(con=fake_con))
    cfg = _cfg()
    sample_row: FunctionProfileRowModel = dict.fromkeys(FUNCTION_PROFILE_COLUMNS)  # type: ignore[arg-type]
    sample_row["repo"] = cfg.repo
    sample_row["commit"] = cfg.commit
    sample_row["function_goid_h128"] = 1
    sample_row["rel_path"] = "p.py"
    sample_row["language"] = "python"
    sample_row["kind"] = "function"
    sample_row["qualname"] = "Q"
    sample_row["created_at"] = datetime.now(tz=UTC)

    with ExitStack() as stack:
        stack.enter_context(
            _override(
                functions,
                "prepared_statements_dynamic",
                lambda _con, _table_key: SimpleNamespace(
                    insert_sql="INSERT INTO analytics.function_profile"
                ),
            )
        )
        stack.enter_context(
            _override(
                functions,
                "load_registry_columns",
                lambda _con: {"analytics.function_profile": list(FUNCTION_PROFILE_COLUMNS)},
            )
        )
        stack.enter_context(
            _override(
                functions,
                "ensure_schema",
                lambda _con, _table_key: None,
            )
        )
        inserted = functions.write_function_profile_rows(gateway, [sample_row])

    if inserted != 1:
        msg = "Function profile writer did not report one inserted row."
        pytest.fail(msg)
    if not fake_con.executed or not fake_con.executed[0][0].startswith(
        "DELETE FROM analytics.function_profile"
    ):
        msg = "Delete was not issued for function_profile."
        pytest.fail(msg)
    if not fake_con.executemany_calls or not fake_con.executemany_calls[0][0].startswith(
        "INSERT INTO analytics.function_profile"
    ):
        msg = "Insert was not issued for function_profile."
        pytest.fail(msg)


def test_file_profile_writer_registry_and_prepared_statements() -> None:
    """Writer should delete then insert with registry alignment."""
    fake_con = _FakeCon()
    gateway = cast("StorageGateway", SimpleNamespace(con=fake_con))
    sample_row: FileProfileRowModel = dict.fromkeys(FILE_PROFILE_COLUMNS)  # type: ignore[arg-type]
    sample_row["repo"] = "r"
    sample_row["commit"] = "c"
    sample_row["rel_path"] = "a.py"
    sample_row["module"] = "m"
    sample_row["tags"] = "[]"
    sample_row["owners"] = "[]"
    sample_row["created_at"] = datetime.now(tz=UTC)

    with ExitStack() as stack:
        stack.enter_context(
            _override(
                files,
                "prepared_statements_dynamic",
                lambda _con, _table: SimpleNamespace(
                    insert_sql="INSERT INTO analytics.file_profile"
                ),
            )
        )
        stack.enter_context(
            _override(
                files,
                "load_registry_columns",
                lambda _con: {"analytics.file_profile": list(FILE_PROFILE_COLUMNS)},
            )
        )
        stack.enter_context(_override(files, "ensure_schema", lambda _con, _table: None))
        inserted = files.write_file_profile_rows(gateway, [sample_row])

    if inserted != 1:
        msg = "File profile writer did not report one inserted row."
        pytest.fail(msg)
    if not fake_con.executed or not fake_con.executed[0][0].startswith(
        "DELETE FROM analytics.file_profile"
    ):
        msg = "Delete was not issued for file_profile."
        pytest.fail(msg)
    if not fake_con.executemany_calls or not fake_con.executemany_calls[0][0].startswith(
        "INSERT INTO analytics.file_profile"
    ):
        msg = "Insert was not issued for file_profile."
        pytest.fail(msg)


def test_module_profile_writer_registry_and_prepared_statements() -> None:
    """Writer should delete then insert with registry alignment."""
    fake_con = _FakeCon()
    gateway = cast("StorageGateway", SimpleNamespace(con=fake_con))
    sample_row: ModuleProfileRowModel = dict.fromkeys(MODULE_PROFILE_COLUMNS)  # type: ignore[arg-type]
    sample_row["repo"] = "r"
    sample_row["commit"] = "c"
    sample_row["module"] = "pkg.mod"
    sample_row["path"] = "pkg/mod.py"
    sample_row["tags"] = "[]"
    sample_row["owners"] = "[]"
    sample_row["created_at"] = datetime.now(tz=UTC)

    with ExitStack() as stack:
        stack.enter_context(
            _override(
                modules,
                "prepared_statements_dynamic",
                lambda _con, _table: SimpleNamespace(
                    insert_sql="INSERT INTO analytics.module_profile"
                ),
            )
        )
        stack.enter_context(
            _override(
                modules,
                "load_registry_columns",
                lambda _con: {"analytics.module_profile": list(MODULE_PROFILE_COLUMNS)},
            )
        )
        stack.enter_context(_override(modules, "ensure_schema", lambda _con, _table: None))
        inserted = modules.write_module_profile_rows(gateway, [sample_row])

    if inserted != 1:
        msg = "Module profile writer did not report one inserted row."
        pytest.fail(msg)
    if not fake_con.executed or not fake_con.executed[0][0].startswith(
        "DELETE FROM analytics.module_profile"
    ):
        msg = "Delete was not issued for module_profile."
        pytest.fail(msg)
    if not fake_con.executemany_calls or not fake_con.executemany_calls[0][0].startswith(
        "INSERT INTO analytics.module_profile"
    ):
        msg = "Insert was not issued for module_profile."
        pytest.fail(msg)


def test_test_profile_writer_registry_and_prepared_statements() -> None:
    """Writer should delete then insert with registry alignment."""
    fake_con = _FakeCon()
    gateway = cast("StorageGateway", SimpleNamespace(con=fake_con))
    cfg = _test_cfg()
    sample_row: TestProfileRowModel = dict.fromkeys(TEST_PROFILE_COLUMNS)  # type: ignore[arg-type]
    sample_row["repo"] = cfg.repo
    sample_row["commit"] = cfg.commit
    sample_row["test_id"] = "t::case"
    sample_row["rel_path"] = "tests/test_a.py"
    sample_row["markers"] = []
    sample_row["functions_covered"] = []
    sample_row["primary_function_goids"] = []
    sample_row["subsystems_covered"] = []
    sample_row["created_at"] = datetime.now(tz=UTC)

    with ExitStack() as stack:
        stack.enter_context(
            _override(
                test_rows,
                "prepared_statements_dynamic",
                lambda _con, _table: SimpleNamespace(
                    insert_sql="INSERT INTO analytics.test_profile"
                ),
            )
        )
        stack.enter_context(
            _override(
                test_rows,
                "load_registry_columns",
                lambda _con: {"analytics.test_profile": list(TEST_PROFILE_COLUMNS)},
            )
        )
        stack.enter_context(_override(test_rows, "ensure_schema", lambda _con, _table: None))
        inserted = test_rows.write_test_profile_rows(gateway, cfg, [sample_row])

    if inserted != 1:
        msg = "Test profile writer did not report one inserted row."
        pytest.fail(msg)
    if not fake_con.executed or not fake_con.executed[0][0].startswith(
        "DELETE FROM analytics.test_profile"
    ):
        msg = "Delete was not issued for test_profile."
        pytest.fail(msg)
    if not fake_con.executemany_calls or not fake_con.executemany_calls[0][0].startswith(
        "INSERT INTO analytics.test_profile"
    ):
        msg = "Insert was not issued for test_profile."
        pytest.fail(msg)


def test_behavioral_coverage_writer_registry_and_prepared_statements() -> None:
    """Writer should delete then insert with registry alignment."""
    fake_con = _FakeCon()
    gateway = cast("StorageGateway", SimpleNamespace(con=fake_con))
    cfg = _behavior_cfg()
    sample_row: BehavioralCoverageRowModel = dict.fromkeys(BEHAVIORAL_COVERAGE_COLUMNS)  # type: ignore[arg-type]
    sample_row["repo"] = cfg.repo
    sample_row["commit"] = cfg.commit
    sample_row["test_id"] = "t::case"
    sample_row["rel_path"] = "tests/test_a.py"
    sample_row["qualname"] = "TestA::test_a"
    sample_row["behavior_tags"] = []
    sample_row["tag_source"] = "heuristic"
    sample_row["created_at"] = datetime.now(tz=UTC)

    with ExitStack() as stack:
        stack.enter_context(
            _override(
                test_rows,
                "prepared_statements_dynamic",
                lambda _con, _table: SimpleNamespace(
                    insert_sql="INSERT INTO analytics.behavioral_coverage"
                ),
            )
        )
        stack.enter_context(
            _override(
                test_rows,
                "load_registry_columns",
                lambda _con: {"analytics.behavioral_coverage": list(BEHAVIORAL_COVERAGE_COLUMNS)},
            )
        )
        stack.enter_context(_override(test_rows, "ensure_schema", lambda _con, _table: None))
        inserted = test_rows.write_behavioral_coverage_rows(gateway, cfg, [sample_row])

    if inserted != 1:
        msg = "Behavioral coverage writer did not report one inserted row."
        pytest.fail(msg)
    if not fake_con.executed or not fake_con.executed[0][0].startswith(
        "DELETE FROM analytics.behavioral_coverage"
    ):
        msg = "Delete was not issued for behavioral_coverage."
        pytest.fail(msg)
    if not fake_con.executemany_calls or not fake_con.executemany_calls[0][0].startswith(
        "INSERT INTO analytics.behavioral_coverage"
    ):
        msg = "Insert was not issued for behavioral_coverage."
        pytest.fail(msg)
