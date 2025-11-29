"""Row assembly and writers for test and behavioral coverage profiles."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime

from codeintel.analytics.profiles.utils import (
    optional_bool,
    optional_float,
    optional_int,
    optional_str,
)
from codeintel.analytics.tests import profiles as legacy
from codeintel.analytics.tests_profiles.types import TestProfileContext, TestRecord
from codeintel.config import BehavioralCoverageStepConfig, TestProfileStepConfig
from codeintel.config.schemas.registry_adapter import load_registry_columns
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.rows import (
    BEHAVIORAL_COVERAGE_COLUMNS,
    TEST_PROFILE_COLUMNS,
    BehavioralCoverageRowModel,
    TestProfileRowModel,
    behavioral_coverage_row_to_tuple,
    serialize_test_profile_row,
)
from codeintel.storage.sql_helpers import ensure_schema, prepared_statements_dynamic


def build_test_profile_rows(
    tests: Iterable[TestRecord],
    ctx: TestProfileContext,
) -> list[TestProfileRowModel]:
    """
    Build test_profile row models using legacy row construction.

    Returns
    -------
    list[TestProfileRowModel]
        Row models ready for insertion.
    """
    rows: list[TestProfileRowModel] = []

    for test in tests:
        row_tuple = legacy.build_test_profile_row(test, ctx)
        rows.append(_tuple_to_test_profile_model(row_tuple))
    return rows


def _tuple_to_test_profile_model(row: tuple[object, ...]) -> TestProfileRowModel:
    """
    Convert the legacy tuple order into a TestProfileRowModel.

    This keeps the writers strongly typed and aligned with table schema.

    Returns
    -------
    TestProfileRowModel
        Row model with normalized types for analytics.test_profile.
    """
    created_at = row[39] if isinstance(row[39], datetime) else datetime.now(tz=UTC)
    return TestProfileRowModel(
        repo=str(row[0]),
        commit=str(row[1]),
        test_id=str(row[2]),
        test_goid_h128=optional_int(row[3]),
        urn=optional_str(row[4]),
        rel_path=str(row[5]),
        module=optional_str(row[6]),
        qualname=optional_str(row[7]),
        language=optional_str(row[8]),
        kind=optional_str(row[9]),
        status=optional_str(row[10]),
        duration_ms=optional_float(row[11]),
        markers=row[12],
        flaky=optional_bool(row[13]),
        last_run_at=row[14] if isinstance(row[14], datetime) else None,
        functions_covered=row[15],
        functions_covered_count=optional_int(row[16]),
        primary_function_goids=row[17],
        subsystems_covered=row[18],
        subsystems_covered_count=optional_int(row[19]),
        primary_subsystem_id=optional_str(row[20]),
        assert_count=optional_int(row[21]),
        raise_count=optional_int(row[22]),
        uses_parametrize=optional_bool(row[23]),
        uses_fixtures=optional_bool(row[24]),
        io_bound=optional_bool(row[25]),
        uses_network=optional_bool(row[26]),
        uses_db=optional_bool(row[27]),
        uses_filesystem=optional_bool(row[28]),
        uses_subprocess=optional_bool(row[29]),
        flakiness_score=optional_float(row[30]),
        importance_score=optional_float(row[31]),
        notes=optional_str(row[32]),
        tg_degree=optional_int(row[33]),
        tg_weighted_degree=optional_float(row[34]),
        tg_proj_degree=optional_int(row[35]),
        tg_proj_weight=optional_float(row[36]),
        tg_proj_clustering=optional_float(row[37]),
        tg_proj_betweenness=optional_float(row[38]),
        created_at=created_at,
    )


def write_test_profile_rows(
    gateway: StorageGateway,
    cfg: TestProfileStepConfig,
    rows: Iterable[TestProfileRowModel],
) -> int:
    """
    Insert rows into analytics.test_profile.

    Returns
    -------
    int
        Number of rows inserted.

    Raises
    ------
    RuntimeError
        If registry column order drifts from serializer constants.
    """
    rows = list(rows)
    con = gateway.con
    ensure_schema(con, "analytics.test_profile")
    registry_cols = load_registry_columns(con).get("analytics.test_profile")
    if registry_cols is None or tuple(registry_cols) != TEST_PROFILE_COLUMNS:
        message = "Registry columns for analytics.test_profile differ from serializer constants."
        raise RuntimeError(message)
    stmt = prepared_statements_dynamic(con, "analytics.test_profile")
    con.execute(
        "DELETE FROM analytics.test_profile WHERE repo = ? AND commit = ?",
        [cfg.repo, cfg.commit],
    )
    if not rows:
        return 0
    tuples = [serialize_test_profile_row(row) for row in rows]
    con.executemany(stmt.insert_sql, tuples)
    return len(rows)


def build_behavioral_coverage_rows(
    rows: Iterable[tuple[object, ...]],
) -> list[BehavioralCoverageRowModel]:
    """
    Build BehavioralCoverageRowModel entries from tuples returned by legacy builder.

    Returns
    -------
    list[BehavioralCoverageRowModel]
        Row models for behavioral coverage.
    """
    models: list[BehavioralCoverageRowModel] = []
    for row in rows:
        (
            repo,
            commit,
            test_id,
            test_goid_h128,
            rel_path,
            qualname,
            behavior_tags,
            tag_source,
            heuristic_version,
            llm_model,
            llm_run_id,
            created_at,
        ) = row
        created_at_value = created_at if isinstance(created_at, datetime) else datetime.now(tz=UTC)
        models.append(
            BehavioralCoverageRowModel(
                repo=str(repo),
                commit=str(commit),
                test_id=str(test_id),
                test_goid_h128=optional_int(test_goid_h128),
                rel_path=str(rel_path),
                qualname=str(qualname) if qualname is not None else None,
                behavior_tags=behavior_tags,
                tag_source=str(tag_source),
                heuristic_version=str(heuristic_version) if heuristic_version is not None else None,
                llm_model=str(llm_model) if llm_model is not None else None,
                llm_run_id=str(llm_run_id) if llm_run_id is not None else None,
                created_at=created_at_value,
            )
        )
    return models


def write_behavioral_coverage_rows(
    gateway: StorageGateway,
    cfg: BehavioralCoverageStepConfig,
    rows: Iterable[BehavioralCoverageRowModel],
) -> int:
    """
    Insert rows into analytics.behavioral_coverage.

    Returns
    -------
    int
        Number of rows inserted.

    Raises
    ------
    RuntimeError
        If registry column order drifts from serializer constants.
    """
    rows = list(rows)
    con = gateway.con
    ensure_schema(con, "analytics.behavioral_coverage")
    registry_cols = load_registry_columns(con).get("analytics.behavioral_coverage")
    if registry_cols is None or tuple(registry_cols) != BEHAVIORAL_COVERAGE_COLUMNS:
        message = (
            "Registry columns for analytics.behavioral_coverage differ from serializer constants."
        )
        raise RuntimeError(message)
    stmt = prepared_statements_dynamic(con, "analytics.behavioral_coverage")
    con.execute(
        "DELETE FROM analytics.behavioral_coverage WHERE repo = ? AND commit = ?",
        [cfg.repo, cfg.commit],
    )
    if not rows:
        return 0
    tuples = [behavioral_coverage_row_to_tuple(row) for row in rows]
    con.executemany(stmt.insert_sql, tuples)
    return len(rows)
