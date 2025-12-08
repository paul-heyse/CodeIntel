"""Tests for analytics.entrypoints.core coverage and summarization."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from codeintel.analytics.ast_features.model import FunctionAstFeatures, IoFlags
from codeintel.analytics.entrypoints.core import build_entrypoints
from codeintel.analytics.parsing.ast_cache import FunctionAst
from codeintel.config.primitives import SnapshotRef
from codeintel.config.steps_analytics import EntryPointsStepConfig
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.sql.builder import ensure_schema
from tests._helpers.assertions import (
    assert_mapping_value,
    expect_equal,
    expect_is_not_none,
    expect_length,
    expect_true,
)
from tests._helpers.builders import (
    CoverageFunctionRow,
    TestCatalogRow,
    TestCoverageEdgeRow,
    insert_rows,
)
from tests._helpers.fakes.function_catalogs import MockFunctionCatalog, MockFunctionMeta
from tests._helpers.gateway import GatewayFactory
from tests._helpers.graphs import build_ast_map, build_module_map, insert_goids, insert_modules


def _seed_entrypoint_repo(
    tmp_path: Path,
) -> tuple[
    SnapshotRef,
    MockFunctionCatalog,
    dict[str, str],
    FunctionAstFeatures,
    Path,
    FunctionAst,
    StorageGateway,
]:
    repo_root = tmp_path / "repo"
    module_path = repo_root / "pkg" / "api.py"
    module_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.write_text(
        "\n".join(
            [
                "from fastapi import FastAPI",
                "",
                "app = FastAPI()",
                "",
                "@app.get('/items')",
                "def list_items(limit: int | None = None) -> int:",
                "    if limit is None:",
                "        return 0",
                "    return limit",
            ]
        ),
        encoding="utf-8",
    )
    snapshot = SnapshotRef(repo="demo", commit="entry", repo_root=repo_root)
    gateway_factory = GatewayFactory().with_snapshot(snapshot.repo, snapshot.commit)
    gateway = gateway_factory.open()

    paths = {"pkg.api": module_path}
    goids = {"list_items": 7001}
    qualname_full = "pkg.api.list_items"
    ast_map = build_ast_map(
        paths,
        goids,
        snapshot.repo_root,
        target_names={"pkg.api": "list_items"},
    )
    insert_modules(gateway, snapshot, paths)
    insert_goids(gateway, snapshot, ast_map, now=datetime.now(tz=UTC))
    module_map = build_module_map(ast_map, {goids["list_items"]: "pkg.api"})
    func_ast = ast_map[goids["list_items"]]

    catalog = MockFunctionCatalog(
        functions=[
            MockFunctionMeta(
                goid=goids["list_items"],
                urn="urn:pkg.api.list_items",
                rel_path=func_ast.rel_path,
                qualname=qualname_full,
                start_line=func_ast.start_line,
                end_line=func_ast.end_line,
            )
        ],
        module_by_path={func_ast.rel_path: "pkg.api"},
    )
    ensure_schema(gateway.con, "analytics.coverage_functions")
    ensure_schema(gateway.con, "analytics.test_catalog")
    ensure_schema(gateway.con, "analytics.test_coverage_edges")
    features = FunctionAstFeatures(
        goid=goids["list_items"],
        rel_path=func_ast.rel_path,
        qualname=qualname_full,
        is_async=False,
        decorators=("@app.get('/items')",),
        imports={"fastapi": "fastapi"},
        libraries_used=frozenset({"fastapi"}),
        io_flags=IoFlags(uses_network=True),
        uses_concurrency_lib=False,
        uses_threading=False,
        uses_asyncio_lib=False,
        http_client_libs=frozenset(),
        http_server_libs=frozenset({"fastapi"}),
        db_libs=frozenset(),
        message_libs=frozenset(),
        config_read_count=0,
        feature_flag_count=0,
    )
    return snapshot, catalog, module_map, features, module_path, func_ast, gateway


def test_entrypoints_materialize_with_test_summary(tmp_path: Path) -> None:
    """Entry points and entrypoint_tests rows capture coverage and test meta."""
    snapshot, catalog, module_map, features, module_path, func_ast, gateway = _seed_entrypoint_repo(
        tmp_path
    )
    con = gateway.con
    ensure_schema(con, "analytics.subsystem_modules")
    ensure_schema(con, "analytics.subsystems")

    now = datetime.now(tz=UTC)
    insert_rows(
        gateway,
        [
            CoverageFunctionRow(
                function_goid_h128=features.goid,
                urn="urn:pkg.api.list_items",
                repo=snapshot.repo,
                commit=snapshot.commit,
                rel_path=module_path.relative_to(snapshot.repo_root).as_posix(),
                language="python",
                kind="function",
                qualname=features.qualname,
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
        gateway,
        [
            TestCatalogRow(
                test_id="tests/test_api.py::test_failed",
                repo=snapshot.repo,
                commit=snapshot.commit,
                rel_path="tests/test_api.py",
                qualname="test_failed",
                status="failed",
                duration_ms=800,
                flaky=False,
                created_at=now,
            ),
            TestCatalogRow(
                test_id="tests/test_api.py::test_slow_flaky",
                repo=snapshot.repo,
                commit=snapshot.commit,
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
        gateway,
        [
            TestCoverageEdgeRow(
                test_id="tests/test_api.py::test_failed",
                function_goid_h128=features.goid,
                urn="urn:pkg.api.list_items",
                repo=snapshot.repo,
                commit=snapshot.commit,
                rel_path=module_path.relative_to(snapshot.repo_root).as_posix(),
                qualname=features.qualname,
                covered_lines=5,
                executable_lines=10,
                coverage_ratio=0.5,
                last_status="failed",
                created_at=now,
            ),
            TestCoverageEdgeRow(
                test_id="tests/test_api.py::test_slow_flaky",
                function_goid_h128=features.goid,
                urn="urn:pkg.api.list_items",
                repo=snapshot.repo,
                commit=snapshot.commit,
                rel_path=module_path.relative_to(snapshot.repo_root).as_posix(),
                qualname=features.qualname,
                covered_lines=4,
                executable_lines=10,
                coverage_ratio=0.4,
                last_status="passed",
                created_at=now,
            ),
        ],
    )

    try:
        build_entrypoints(
            gateway,
            EntryPointsStepConfig(snapshot=snapshot),
            catalog_provider=catalog,
            module_map=module_map,
            features_map={features.goid: features},
        )

        entry_row = gateway.con.execute(
            """
            SELECT tests_touching, failing_tests, slow_tests, flaky_tests,
                   entrypoint_coverage_ratio, last_test_status, extra
            FROM analytics.entrypoints
            WHERE repo = ? AND commit = ?
            """,
            [snapshot.repo, snapshot.commit],
        ).fetchone()
        row = expect_is_not_none(entry_row)
        expect_equal(row[0], 2)
        expect_equal(row[1], 1)
        expect_equal(row[2], 1)
        expect_equal(row[3], 1)
        expect_equal(row[4], 0.6)
        expect_equal(row[5], "some_failing")

        extra = row[6]
        if isinstance(extra, str):
            extra = json.loads(extra)
        extra_payload = assert_mapping_value(extra, "ast_features", dict)
        expect_true(extra_payload["uses_network"])
        expect_equal(extra_payload["http_server_libs"], ["fastapi"])

        test_rows = gateway.con.execute(
            """
            SELECT test_id, coverage_ratio, status, duration_ms
            FROM analytics.entrypoint_tests
            WHERE repo = ? AND commit = ?
            ORDER BY test_id
            """,
            [snapshot.repo, snapshot.commit],
        ).fetchall()
        expect_length(test_rows, 2)
        expect_equal(test_rows[0][2], "failed")
        expect_equal(test_rows[1][3], 1500)
    finally:
        gateway.close()


def test_entrypoints_no_modules_skip_detection(tmp_path: Path) -> None:
    """Early exit occurs when no module map or module context is available."""
    snapshot, _, _, _, _, _, gateway = _seed_entrypoint_repo(tmp_path)
    gateway.con.execute(
        "DELETE FROM core.modules WHERE repo = ? AND commit = ?",
        [snapshot.repo, snapshot.commit],
    )

    try:
        build_entrypoints(
            gateway,
            EntryPointsStepConfig(snapshot=snapshot),
            catalog_provider=MockFunctionCatalog(),
            module_map={},
            features_map={},
        )

        count = gateway.con.execute(
            """
            SELECT COUNT(*) FROM analytics.entrypoints WHERE repo = ? AND commit = ?
            """,
            [snapshot.repo, snapshot.commit],
        ).fetchone()
        row = expect_is_not_none(count)
        expect_equal(row[0], 0)
    finally:
        gateway.close()
