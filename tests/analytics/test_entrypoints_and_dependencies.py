"""Integration tests for entrypoint and dependency analytics."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from codeintel.analytics.dependencies import (
    build_external_dependencies,
    build_external_dependency_calls,
)
from codeintel.analytics.entrypoints import build_entrypoints
from codeintel.config import ConfigBuilder
from codeintel.graphs.function_catalog_service import FunctionCatalogService
from codeintel.graphs.goid_builder import build_goids
from codeintel.ingestion.py_ast_extract import ingest_python_ast
from codeintel.ingestion.repo_scan import ingest_repo
from codeintel.storage.gateway import DuckDBConnection
from tests._helpers.builders import (
    CoverageFunctionRow,
    ModuleRow,
    TestCatalogRow,
    TestCoverageEdgeRow,
    insert_coverage_functions,
    insert_modules,
    insert_test_catalog,
    insert_test_coverage_edges,
)
from tests._helpers.fixtures import ProvisionedGateway, provision_gateway_with_repo

HTTP_CREATED = 201
COVERAGE_TOLERANCE = 1e-6
GoidRow = tuple[int, str, str, str | None, str, str, int | None, int | None]


def _write_sample_repo(repo_root: Path) -> None:
    pkg_dir = repo_root / "pkg"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    (pkg_dir / "__init__.py").write_text("", encoding="utf8")
    (pkg_dir / "app.py").write_text(
        "\n".join(
            [
                "from fastapi import FastAPI",
                "import boto3",
                "import click",
                "import redis",
                "",
                "app = FastAPI()",
                "",
                "@app.get('/hello', status_code=201)",
                "def hello(name: str) -> str:",
                "    client = boto3.client('s3')",
                "    client.list_buckets()",
                "    return name",
                "",
                "@click.command()",
                "@click.option('--loud', is_flag=True)",
                "def cli(loud: bool) -> None:",
                "    redis.Redis().get('key')",
            ]
        ),
        encoding="utf8",
    )
    patterns_source = Path(__file__).resolve().parents[2] / "config" / "dependency_patterns.yml"
    patterns_dir = repo_root / "config"
    patterns_dir.mkdir(parents=True, exist_ok=True)
    patterns_dir.joinpath("dependency_patterns.yml").write_text(
        patterns_source.read_text(encoding="utf8"), encoding="utf8"
    )


def _get_goid_row(con: DuckDBConnection, qualname: str) -> GoidRow:
    row = con.execute(
        """
        SELECT goid_h128, urn, rel_path, language, kind, qualname, start_line, end_line
        FROM core.goids
        WHERE qualname = ?
        """,
        [qualname],
    ).fetchone()
    if row is None:
        message = f"GOID row missing for {qualname}"
        raise AssertionError(message)
    return (
        int(row[0]),
        str(row[1]),
        str(row[2]),
        str(row[3]) if row[3] is not None else None,
        str(row[4]),
        str(row[5]),
        int(row[6]) if row[6] is not None else None,
        int(row[7]) if row[7] is not None else None,
    )


def _ensure(condition: object, message: str) -> None:
    if not condition:
        pytest.fail(message)


def _seed_coverage_and_tests(ctx: ProvisionedGateway, hello_row: GoidRow, now: datetime) -> None:
    (
        hello_goid,
        hello_urn,
        hello_rel_path,
        language,
        kind,
        qualname,
        start_line,
        end_line,
    ) = hello_row
    normalized_language = language or "python"
    normalized_start = start_line or 1
    normalized_end = end_line or normalized_start
    test_id = "tests.test_app::test_hello"
    insert_coverage_functions(
        ctx.gateway,
        [
            CoverageFunctionRow(
                function_goid_h128=hello_goid,
                urn=hello_urn,
                repo=ctx.repo,
                commit=ctx.commit,
                rel_path=hello_rel_path,
                language=normalized_language,
                kind=kind,
                qualname=qualname,
                start_line=normalized_start,
                end_line=normalized_end,
                executable_lines=2,
                covered_lines=2,
                coverage_ratio=1.0,
                tested=True,
                untested_reason=None,
                created_at=now,
            )
        ],
    )
    insert_test_catalog(
        ctx.gateway,
        [
            TestCatalogRow(
                test_id=test_id,
                repo=ctx.repo,
                commit=ctx.commit,
                rel_path="tests/test_app.py",
                qualname="tests.test_app.test_hello",
                status="passed",
                created_at=now,
            )
        ],
    )
    insert_test_coverage_edges(
        ctx.gateway,
        [
            TestCoverageEdgeRow(
                test_id=test_id,
                function_goid_h128=hello_goid,
                urn=hello_urn,
                repo=ctx.repo,
                commit=ctx.commit,
                rel_path=hello_rel_path,
                qualname=qualname,
                covered_lines=2,
                executable_lines=2,
                coverage_ratio=1.0,
                last_status="passed",
                created_at=now,
            )
        ],
    )


def _validate_entrypoint_rows(con: DuckDBConnection, repo: str, commit: str) -> None:
    entry_row = con.execute(
        """
        SELECT http_method, route_path, status_codes, tests_touching,
               entrypoint_coverage_ratio, last_test_status
        FROM analytics.entrypoints
        WHERE handler_qualname = 'pkg.app.hello'
          AND repo = ?
          AND commit = ?
        """,
        [repo, commit],
    ).fetchone()
    if entry_row is None:
        pytest.fail("entrypoint row missing for FastAPI handler")
    http_method, route_path, status_codes, tests_touching, coverage_ratio, last_status = entry_row
    status_list = _coerce_status_codes(status_codes)
    _ensure(http_method == "GET", "http_method not captured")
    _ensure(route_path == "/hello", "route_path not captured")
    _ensure(HTTP_CREATED in status_list, "status_codes missing HTTP_CREATED")
    _ensure(tests_touching == 1, "tests_touching should count seeded test")
    _ensure(
        abs(float(coverage_ratio) - 1.0) < COVERAGE_TOLERANCE,
        "entrypoint coverage ratio did not propagate",
    )
    _ensure(last_status == "all_passing", "last_test_status not aggregated")

    test_edge_count = con.execute(
        """
        SELECT COUNT(*)
        FROM analytics.entrypoint_tests
        WHERE entrypoint_id = (
            SELECT entrypoint_id FROM analytics.entrypoints
            WHERE handler_qualname = 'pkg.app.hello'
              AND repo = ?
              AND commit = ?
        )
        """,
        [repo, commit],
    ).fetchone()
    if test_edge_count is None:
        pytest.fail("entrypoint_tests missing")
    _ensure(int(test_edge_count[0]) == 1, "entrypoint_tests count mismatch")

    cli_entry = con.execute(
        """
        SELECT kind, command_name
        FROM analytics.entrypoints
        WHERE handler_qualname = 'pkg.app.cli'
          AND repo = ?
          AND commit = ?
        """,
        [repo, commit],
    ).fetchone()
    if cli_entry is None:
        pytest.fail("CLI entrypoint row missing")
    kind, command_name = cli_entry
    _ensure(kind == "cli", "CLI kind not recorded")
    _ensure(command_name == "cli", "CLI command_name not recorded")


def _validate_dependency_rows(con: DuckDBConnection, repo: str, commit: str) -> None:
    libraries = {
        row[0]
        for row in con.execute(
            """
            SELECT library, callsite_count, modes
            FROM analytics.external_dependency_calls
            WHERE repo = ? AND commit = ?
            """,
            [repo, commit],
        ).fetchall()
    }
    _ensure({"boto3", "redis"} <= libraries, "dependency calls not captured")

    dep_by_lib = {
        row[0]: row
        for row in con.execute(
            """
            SELECT library, function_count, callsite_count, risk_level
            FROM analytics.external_dependencies
            WHERE repo = ? AND commit = ?
            """,
            [repo, commit],
        ).fetchall()
    }
    _ensure(dep_by_lib["boto3"][1] == 1, "boto3 function_count mismatch")
    _ensure(dep_by_lib["boto3"][3] == "high", "boto3 risk level unexpected")
    _ensure(dep_by_lib["redis"][1] == 1, "redis function_count mismatch")
    _ensure(dep_by_lib["redis"][3] in {"medium", "high"}, "redis risk level unexpected")


def _coerce_status_codes(raw: object) -> list[int]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [int(code) for code in raw]
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [int(code) for code in parsed]
        except json.JSONDecodeError:
            return []
    return []


def test_entrypoints_and_dependencies_round_trip(tmp_path: Path) -> None:
    """Validate entrypoint and dependency tables on a small repo snapshot."""
    repo_root = tmp_path / "repo"
    _write_sample_repo(repo_root)

    with provision_gateway_with_repo(repo_root) as ctx:
        builder = ConfigBuilder.from_snapshot(
            repo=ctx.repo,
            commit=ctx.commit,
            repo_root=ctx.repo_root,
            build_dir=ctx.build_dir,
        )
        tracker = ingest_repo(
            ctx.gateway,
            cfg=builder.repo_scan(),
        )
        insert_modules(
            ctx.gateway,
            [
                ModuleRow(
                    module="pkg.app",
                    path="pkg/app.py",
                    repo=ctx.repo,
                    commit=ctx.commit,
                )
            ],
        )
        ingest_python_ast(tracker)
        build_goids(ctx.gateway, builder.goid_builder())

        con = ctx.gateway.con
        hello_row = _get_goid_row(con, "pkg.app.hello")

        now = datetime.now(tz=UTC)
        _seed_coverage_and_tests(ctx, hello_row, now)

        catalog = FunctionCatalogService.from_db(ctx.gateway, repo=ctx.repo, commit=ctx.commit)
        entry_cfg = builder.entrypoints()
        build_entrypoints(ctx.gateway, entry_cfg, catalog_provider=catalog)

        dep_cfg = builder.external_dependencies()
        build_external_dependency_calls(ctx.gateway, dep_cfg, catalog_provider=catalog)
        build_external_dependencies(ctx.gateway, dep_cfg)

        _validate_entrypoint_rows(con, ctx.repo, ctx.commit)
        _validate_dependency_rows(con, ctx.repo, ctx.commit)
