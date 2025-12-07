"""Tests for analytics.entrypoints.core coverage and summarization."""

from __future__ import annotations

from datetime import UTC, datetime

from tests._helpers.assertions import assert_mapping_value
from tests._helpers.builders import (
    CoverageFunctionRow,
    TestCatalogRow,
    TestCoverageEdgeRow,
    insert_rows,
)
from tests._helpers.fakes.function_catalogs import MockFunctionCatalog
from tests.analytics.integration.sample_repo import SampleRepo

from codeintel.analytics.entrypoints.core import build_entrypoints
from codeintel.config.steps_analytics import EntryPointsStepConfig
from codeintel.graphs.catalog import FunctionCatalogService
from codeintel.storage.sql.builder import ensure_schema


def test_entrypoints_materialize_with_test_summary(sample_repo: SampleRepo) -> None:
    """Entry points and entrypoint_tests rows capture coverage and test meta."""
    catalog = FunctionCatalogService.from_db(
        sample_repo.gateway,
        repo=sample_repo.snapshot.repo,
        commit=sample_repo.snapshot.commit,
    )
    con = sample_repo.gateway.con
    ensure_schema(con, "analytics.subsystem_modules")
    ensure_schema(con, "analytics.subsystems")
    con.execute(
        "DELETE FROM core.modules WHERE repo = ? AND commit = ?",
        [sample_repo.snapshot.repo, sample_repo.snapshot.commit],
    )

    now = datetime.now(tz=UTC)
    func_ast = sample_repo.ast_map[sample_repo.goid_route]
    insert_rows(
        sample_repo.gateway,
        [
            CoverageFunctionRow(
                function_goid_h128=sample_repo.goid_route,
                urn="urn:pkg.api.list_items",
                repo=sample_repo.snapshot.repo,
                commit=sample_repo.snapshot.commit,
                rel_path=func_ast.rel_path,
                language="python",
                kind="function",
                qualname="list_items",
                start_line=func_ast.start_line,
                end_line=func_ast.end_line,
                executable_lines=10,
                covered_lines=6,
                coverage_ratio=0.6,
                tested=True,
                untested_reason=None,
                created_at=now,
            )
        ],
    )
    insert_rows(
        sample_repo.gateway,
        [
            TestCatalogRow(
                test_id="tests/test_api.py::test_failed",
                repo=sample_repo.snapshot.repo,
                commit=sample_repo.snapshot.commit,
                rel_path="tests/test_api.py",
                qualname="test_failed",
                status="failed",
                duration_ms=1500,
                flaky=False,
                created_at=now,
            ),
            TestCatalogRow(
                test_id="tests/test_api.py::test_slow_flaky",
                repo=sample_repo.snapshot.repo,
                commit=sample_repo.snapshot.commit,
                rel_path="tests/test_api.py",
                qualname="test_slow_flaky",
                status="passed",
                duration_ms=1500,
                flaky=True,
                created_at=now,
            ),
        ],
    )
    insert_rows(
        sample_repo.gateway,
        [
            TestCoverageEdgeRow(
                test_id="tests/test_api.py::test_failed",
                function_goid_h128=sample_repo.goid_route,
                urn="urn:pkg.api.list_items",
                repo=sample_repo.snapshot.repo,
                commit=sample_repo.snapshot.commit,
                rel_path=func_ast.rel_path,
                qualname="list_items",
                covered_lines=5,
                executable_lines=10,
                coverage_ratio=0.5,
                last_status="failed",
                created_at=now,
            ),
            TestCoverageEdgeRow(
                test_id="tests/test_api.py::test_slow_flaky",
                function_goid_h128=sample_repo.goid_route,
                urn="urn:pkg.api.list_items",
                repo=sample_repo.snapshot.repo,
                commit=sample_repo.snapshot.commit,
                rel_path=func_ast.rel_path,
                qualname="list_items",
                covered_lines=4,
                executable_lines=10,
                coverage_ratio=0.4,
                last_status="passed",
                created_at=now,
            ),
        ],
    )

    build_entrypoints(
        sample_repo.gateway,
        EntryPointsStepConfig(snapshot=sample_repo.snapshot),
        catalog_provider=catalog,
        module_map=sample_repo.module_map,
        features_map=sample_repo.features,
    )

    entry_row = sample_repo.gateway.con.execute(
        """
        SELECT tests_touching, failing_tests, slow_tests, flaky_tests,
               entrypoint_coverage_ratio, last_test_status, extra
        FROM analytics.entrypoints
        WHERE repo = ? AND commit = ?
        """,
        [sample_repo.snapshot.repo, sample_repo.snapshot.commit],
    ).fetchone()
    assert entry_row is not None
    assert entry_row[0] == 2
    assert entry_row[1] == 1
    assert entry_row[2] == 1
    assert entry_row[3] == 1
    assert entry_row[4] == 0.6
    assert entry_row[5] == "some_failing"

    extra_payload = assert_mapping_value(entry_row[6], "ast_features", dict)
    assert extra_payload["uses_network"] is True
    assert extra_payload["http_server_libs"] == ["fastapi"]

    test_rows = sample_repo.gateway.con.execute(
        """
        SELECT test_id, coverage_ratio, status, duration_ms
        FROM analytics.entrypoint_tests
        WHERE repo = ? AND commit = ?
        ORDER BY test_id
        """,
        [sample_repo.snapshot.repo, sample_repo.snapshot.commit],
    ).fetchall()
    assert len(test_rows) == 2
    assert test_rows[0][2] == "failed"
    assert test_rows[1][3] == 1500


def test_entrypoints_no_modules_skip_detection(sample_repo: SampleRepo) -> None:
    """Early exit occurs when no module map is available."""
    catalog = MockFunctionCatalog()
    con = sample_repo.gateway.con
    con.execute(
        "DELETE FROM core.modules WHERE repo = ? AND commit = ?",
        [sample_repo.snapshot.repo, sample_repo.snapshot.commit],
    )

    build_entrypoints(
        sample_repo.gateway,
        EntryPointsStepConfig(snapshot=sample_repo.snapshot),
        catalog_provider=catalog,
        module_map={},
        features_map={},
    )

    count = sample_repo.gateway.con.execute(
        """
        SELECT COUNT(*) FROM analytics.entrypoints WHERE repo = ? AND commit = ?
        """,
        [sample_repo.snapshot.repo, sample_repo.snapshot.commit],
    ).fetchone()
    assert count is not None
    assert count[0] == 0
