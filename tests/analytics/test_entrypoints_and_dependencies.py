"""Integration tests for entrypoint and dependency analytics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from codeintel.analytics.ast_features.model import FunctionAstFeatures
from codeintel.analytics.dependencies import build_external_dependencies
from codeintel.analytics.dependencies.core import (
    ExternalDependencyInputs,
    build_external_dependency_calls,
)
from codeintel.analytics.entrypoints import build_entrypoints
from codeintel.analytics.function_ast_cache import FunctionAst
from codeintel.analytics.resources.asts import AstProvider
from codeintel.analytics.resources.features import FeaturesProvider
from codeintel.analytics.resources.module_map import ModuleMapProvider
from codeintel.config import ConfigBuilder
from codeintel.config.primitives import SnapshotRef
from codeintel.graphs.catalog import FunctionCatalogService
from codeintel.graphs.plugins.builders.goid import build_goids
from codeintel.ingestion import (
    AstExtractStep,
    DuckDBStorageAdapter,
    FilesystemDiscoveryAdapter,
    HashChangeDetectionAdapter,
    RepoScanStep,
)
from codeintel.ingestion.infrastructure_utilities.source_scanner import default_code_profile
from codeintel.storage.gateway import DuckDBConnection, StorageGateway
from tests._helpers.builders import CoverageFunctionRow, insert_coverage_functions
from tests._helpers.configs import CoverageSeedConfig
from tests._helpers.orchestration import provision_gateway_with_repo

HTTP_CREATED = 201
COVERAGE_TOLERANCE = 1e-6
GoidRow = tuple[int, str, str, str | None, str, str, int | None, int | None]


@dataclass
class EntrypointTestProviders:
    """Providers needed for entrypoint and dependency testing.

    Consolidate provider instances to reduce local variable count
    in test functions.
    """

    catalog: FunctionCatalogService
    module_map: dict[str, str]
    ast_by_goid: dict[int, FunctionAst]
    missing_goids: set[int]
    features_map: dict[int, FunctionAstFeatures]


def _build_test_providers(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    repo: str,
    commit: str,
) -> EntrypointTestProviders:
    """Build all providers needed for entrypoint testing.

    Parameters
    ----------
    gateway
        Storage gateway instance.
    snapshot
        Snapshot reference.
    repo
        Repository identifier.
    commit
        Commit hash.

    Returns
    -------
    EntrypointTestProviders
        Container with all required providers.
    """
    catalog = FunctionCatalogService.from_db(gateway, repo=repo, commit=commit)
    module_map_provider = ModuleMapProvider(gateway, snapshot)
    ast_provider = AstProvider(gateway, snapshot)
    ast_data = ast_provider.get()
    features_provider = FeaturesProvider(gateway, snapshot)

    return EntrypointTestProviders(
        catalog=catalog,
        module_map=module_map_provider.get(),
        ast_by_goid=ast_data.function_ast_map,
        missing_goids=ast_data.missing_function_goids,
        features_map=features_provider.get(),
    )


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

        # Set up ingestion adapters and run repo scan
        storage = DuckDBStorageAdapter(ctx.gateway)
        discovery = FilesystemDiscoveryAdapter(ctx.repo_root)
        change_detection = HashChangeDetectionAdapter(storage)
        scan_step = RepoScanStep(
            storage=storage,
            discovery=discovery,
            change_detection=change_detection,
        )
        profile = default_code_profile(ctx.repo_root)
        _, modules, _ = scan_step.execute(
            repo=ctx.repo,
            commit=ctx.commit,
            repo_root=ctx.repo_root,
            profile=profile,
        )

        # Extract ASTs and build GOIDs
        ast_step = AstExtractStep(storage=storage, discovery=discovery)
        ast_step.execute(list(modules), repo=ctx.repo, commit=ctx.commit)
        build_goids(ctx.gateway, builder.goid_builder())

        hello_row = _get_goid_row(ctx.gateway.con, "pkg.app.hello")
        seed_cfg = CoverageSeedConfig(
            module_import="pkg.app",
            function_name="hello",
            test_id="tests.test_app::test_hello",
            repo=ctx.repo,
            commit=ctx.commit,
            function_goid=hello_row[0],
            test_goid=hello_row[0] + 1000,
        )
        now = datetime.now(tz=UTC)
        insert_coverage_functions(
            ctx.gateway,
            [
                CoverageFunctionRow(
                    function_goid_h128=seed_cfg.function_goid,
                    urn=f"goid:{ctx.repo}#python:function:pkg.app.hello",
                    repo=ctx.repo,
                    commit=ctx.commit,
                    rel_path="pkg/app.py",
                    language="python",
                    kind="function",
                    qualname="pkg.app.hello",
                    start_line=9,
                    end_line=15,
                    executable_lines=2,
                    covered_lines=2,
                    coverage_ratio=1.0,
                    tested=True,
                    untested_reason=None,
                    created_at=now,
                )
            ],
        )
        ctx.gateway.con.execute("DELETE FROM analytics.test_coverage_edges")
        ctx.gateway.con.execute("DELETE FROM analytics.test_catalog")
        ctx.gateway.con.execute(
            """
            INSERT INTO analytics.test_catalog (test_id, rel_path, qualname, repo, commit, status, created_at)
            VALUES (?, ?, ?, ?, ?, 'passed', ?)
            """,
            [
                seed_cfg.test_id,
                "tests/test_app.py",
                "tests.test_app.test_hello",
                seed_cfg.repo,
                seed_cfg.commit,
                now,
            ],
        )
        ctx.gateway.con.execute(
            """
            INSERT INTO analytics.test_coverage_edges (
                test_id, function_goid_h128, urn, repo, commit, rel_path, qualname,
                covered_lines, executable_lines, coverage_ratio, last_status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 2, 2, 1.0, 'passed', ?)
            """,
            [
                seed_cfg.test_id,
                seed_cfg.function_goid,
                f"goid:{seed_cfg.repo}#python:function:{seed_cfg.module_import}.{seed_cfg.function_name}",
                seed_cfg.repo,
                seed_cfg.commit,
                "pkg/app.py",
                "pkg.app.hello",
                now,
            ],
        )

        # Build providers and run analytics
        providers = _build_test_providers(ctx.gateway, builder.snapshot, ctx.repo, ctx.commit)

        build_entrypoints(
            ctx.gateway,
            builder.entrypoints(),
            catalog_provider=providers.catalog,
            module_map=providers.module_map,
            features_map=providers.features_map,
        )

        dep_cfg = builder.external_dependencies()
        build_external_dependency_calls(
            ctx.gateway,
            dep_cfg,
            inputs=ExternalDependencyInputs(
                catalog_provider=providers.catalog,
                module_map=providers.module_map,
                ast_by_goid=providers.ast_by_goid,
                features_map=providers.features_map,
                missing_goids=providers.missing_goids,
            ),
        )
        build_external_dependencies(ctx.gateway, dep_cfg)

        _validate_entrypoint_rows(ctx.gateway.con, ctx.repo, ctx.commit)
        _validate_dependency_rows(ctx.gateway.con, ctx.repo, ctx.commit)
