"""Tests for analytics.dependencies.core coverage and aggregation."""

from __future__ import annotations

import ast
from pathlib import Path

from tests._helpers.assertions import assert_mapping_list
from tests._helpers.builders import ConfigValueRow, insert_rows
from tests._helpers.fakes.function_catalogs import MockFunctionCatalog, MockFunctionMeta
from tests._helpers.gateway import GatewayFactory

from codeintel.analytics.ast_features.model import FunctionAstFeatures, IoFlags
from codeintel.analytics.dependencies.core import (
    ExternalDependencyInputs,
    build_external_dependencies,
    build_external_dependency_calls,
)
from codeintel.analytics.parsing.ast_cache import FunctionAst
from codeintel.config.primitives import SnapshotRef
from codeintel.config.steps_graphs import ExternalDependenciesStepConfig


def _build_function_ast(module_path: Path, qualname: str, goid: int, repo_root: Path) -> FunctionAst:
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == qualname
    )
    start_line = getattr(node, "lineno", 0)
    end_line = getattr(node, "end_lineno", start_line)
    return FunctionAst(
        goid=goid,
        rel_path=module_path.relative_to(repo_root).as_posix(),
        qualname=qualname,
        start_line=start_line,
        end_line=end_line,
        node=node,
        lines=list(source.splitlines()),
    )


def test_dependency_calls_and_aggregation(tmp_path: Path) -> None:
    """Dependency calls are collected, then aggregated with config keys and modes."""
    repo_root = tmp_path / "repo"
    module_path = repo_root / "pkg" / "client.py"
    module_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.write_text(
        "\n".join(
            [
                "import requests as rq",
                "import httpx",
                "",
                "def fetch_data(url: str) -> int:",
                "    response = rq.get(url)",
                "    httpx.post(url)",
                "    return response.status_code if hasattr(response, 'status_code') else 0",
            ]
        ),
        encoding="utf-8",
    )
    patterns_path = repo_root / "config" / "dependency_patterns.yml"
    patterns_path.parent.mkdir(parents=True, exist_ok=True)
    patterns_path.write_text(
        "libs:\n"
        "  requests:\n"
        "    severity: medium\n"
        "    patterns:\n"
        '      - mode: ["read"]\n'
        '        method: "get"\n'
        "  httpx:\n"
        "    patterns:\n"
        '      - mode: ["write"]\n'
        '        method_prefix: "httpx.post"\n'
        "        criticality: 2.0\n",
        encoding="utf-8",
    )

    snapshot = SnapshotRef(repo="demo", commit="abc123", repo_root=repo_root)
    gateway = GatewayFactory().with_snapshot(snapshot.repo, snapshot.commit).open()

    goid = 5001
    func_ast = _build_function_ast(module_path, "fetch_data", goid, repo_root)
    catalog = MockFunctionCatalog(
        functions=[
            MockFunctionMeta(
                goid=goid,
                urn="urn:pkg.client.fetch_data",
                rel_path=func_ast.rel_path,
                qualname="fetch_data",
                start_line=func_ast.start_line,
                end_line=func_ast.end_line,
            )
        ],
        module_by_path={func_ast.rel_path: "pkg.client"},
    )
    cfg = ExternalDependenciesStepConfig(snapshot=snapshot)
    inputs = ExternalDependencyInputs(
        catalog_provider=catalog,
        module_map={func_ast.rel_path: "pkg.client"},
        ast_by_goid={goid: func_ast},
        features_map={},
    )

    try:
        build_external_dependency_calls(gateway, cfg, inputs=inputs)

        call_rows = gateway.con.execute(
            """
            SELECT library, modes, evidence_json, callsite_count, rel_path
            FROM analytics.external_dependency_calls
            WHERE repo = ? AND commit = ?
            ORDER BY library
            """,
            [snapshot.repo, snapshot.commit],
        ).fetchall()
        assert len(call_rows) == 2
        requests_row = next(row for row in call_rows if row[0] == "requests")
        httpx_row = next(row for row in call_rows if row[0] == "httpx")

        assert requests_row[1] == ["read"]
        evidence = assert_mapping_list({"samples": requests_row[2]}, "samples")
        assert evidence, "expected requests evidence samples"
        assert requests_row[3] == 1
        assert requests_row[4] == func_ast.rel_path

        assert httpx_row[1] == ["write"]

        insert_rows(
            gateway,
            [
                ConfigValueRow(
                    repo=snapshot.repo,
                    commit=snapshot.commit,
                    config_path="config/settings.yml",
                    format="yaml",
                    key="API_TOKEN",
                    reference_paths=[func_ast.rel_path],
                    reference_modules=["pkg.client"],
                    reference_count=1,
                )
            ],
        )

        build_external_dependencies(gateway, cfg)

        dep_rows = gateway.con.execute(
            """
            SELECT library, usage_modes, config_keys, risk_level, callsite_count, function_count
            FROM analytics.external_dependencies
            WHERE repo = ? AND commit = ?
            ORDER BY library
            """,
            [snapshot.repo, snapshot.commit],
        ).fetchall()
        assert len(dep_rows) == 2
        requests_dep = next(row for row in dep_rows if row[0] == "requests")
        httpx_dep = next(row for row in dep_rows if row[0] == "httpx")

        assert requests_dep[1] == ["read"]
        assert requests_dep[2] == ["API_TOKEN"]
        assert requests_dep[3] == "medium"
        assert requests_dep[4] == 1
        assert requests_dep[5] == 1

        assert httpx_dep[1] == ["write"]
        assert httpx_dep[2] == ["API_TOKEN"]
        assert httpx_dep[3] == "high"
    finally:
        gateway.close()


def test_dependency_calls_respect_feature_gates(tmp_path: Path) -> None:
    """Calls are skipped when features indicate no IO usage."""
    repo_root = tmp_path / "repo"
    module_path = repo_root / "pkg" / "client.py"
    module_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.write_text(
        "\n".join(
            [
                "import requests as rq",
                "",
                "def fetch_data(url: str) -> int:",
                "    return rq.get(url).status_code",
            ]
        ),
        encoding="utf-8",
    )
    snapshot = SnapshotRef(repo="demo", commit="def456", repo_root=repo_root)
    cfg = ExternalDependenciesStepConfig(snapshot=snapshot)
    patterns_path = repo_root / "config" / "dependency_patterns.yml"
    patterns_path.parent.mkdir(parents=True, exist_ok=True)
    patterns_path.write_text(
        "libs:\n"
        "  requests:\n"
        "    patterns:\n"
        '      - mode: ["read"]\n'
        '        method: "get"\n',
        encoding="utf-8",
    )
    gateway = GatewayFactory().with_snapshot(snapshot.repo, snapshot.commit).open()

    goid = 6001
    func_ast = _build_function_ast(module_path, "fetch_data", goid, repo_root)
    catalog = MockFunctionCatalog(
        functions=[
            MockFunctionMeta(
                goid=goid,
                urn="urn:pkg.client.fetch_data",
                rel_path=func_ast.rel_path,
                qualname="fetch_data",
                start_line=func_ast.start_line,
                end_line=func_ast.end_line,
            )
        ],
        module_by_path={func_ast.rel_path: "pkg.client"},
    )
    features = FunctionAstFeatures(
        goid=goid,
        rel_path=func_ast.rel_path,
        qualname="fetch_data",
        is_async=False,
        decorators=(),
        imports={"requests": "requests"},
        libraries_used=frozenset({"requests"}),
        io_flags=IoFlags(),
        uses_concurrency_lib=False,
        uses_threading=False,
        uses_asyncio_lib=False,
        http_client_libs=frozenset(),
        http_server_libs=frozenset(),
        db_libs=frozenset(),
        message_libs=frozenset(),
        config_read_count=0,
        feature_flag_count=0,
    )
    inputs = ExternalDependencyInputs(
        catalog_provider=catalog,
        module_map={func_ast.rel_path: "pkg.client"},
        ast_by_goid={goid: func_ast},
        features_map={goid: features},
    )

    try:
        build_external_dependency_calls(gateway, cfg, inputs=inputs)

        rows = gateway.con.execute(
            """
            SELECT COUNT(*) FROM analytics.external_dependency_calls
            WHERE repo = ? AND commit = ?
            """,
            [snapshot.repo, snapshot.commit],
        ).fetchone()
        assert rows is not None
        assert rows[0] == 0
    finally:
        gateway.close()
