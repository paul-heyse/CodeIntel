"""Tests for analytics.dependencies.core coverage and aggregation."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from codeintel.analytics.ast_features.model import FunctionAstFeatures, IoFlags
from codeintel.analytics.dependencies.core import (
    ExternalDependencyInputs,
    build_external_dependencies,
    build_external_dependency_calls,
)
from codeintel.analytics.parsing.ast_cache import FunctionAst
from codeintel.config.steps_graphs import ExternalDependenciesStepConfig
from tests._helpers import CORE_PACK, create_test_context
from tests._helpers.assertions import (
    assert_mapping_list,
    expect_equal,
    expect_is_not_none,
    expect_true,
)
from tests._helpers.builders import ConfigValueRow, insert_rows
from tests._helpers.fakes.function_catalogs import MockFunctionCatalog
from tests._helpers.rows import function_meta


def _as_list(value: object) -> list[object]:
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
            if isinstance(loaded, list):
                return loaded
        except json.JSONDecodeError:
            return [value]
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _build_function_ast(
    module_path: Path, qualname: str, goid: int, repo_root: Path
) -> FunctionAst:
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == qualname)
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
    ctx = create_test_context(tmp_path)
    ctx.require(CORE_PACK)
    repo_root = ctx.repo_root
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
    (repo_root / "config").mkdir(parents=True, exist_ok=True)
    (repo_root / "config" / "dependency_patterns.yml").write_text(
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

    snapshot = ctx.to_snapshot_ref()
    gateway = ctx.gateway

    goid = 5001
    func_ast = _build_function_ast(module_path, "fetch_data", goid, repo_root)
    cfg = ExternalDependenciesStepConfig(snapshot=snapshot)
    inputs = ExternalDependencyInputs(
        catalog_provider=MockFunctionCatalog(
            functions=[
                function_meta(
                    goid=goid,
                    rel_path=func_ast.rel_path,
                    qualname="fetch_data",
                    snapshot=(snapshot.repo, snapshot.commit),
                    line_span=(func_ast.start_line, func_ast.end_line),
                )
            ],
            module_by_path={func_ast.rel_path: "pkg.client"},
        ),
        module_map={func_ast.rel_path: "pkg.client"},
        ast_by_goid={goid: func_ast},
        features_map={},
    )

    try:
        build_external_dependency_calls(gateway, cfg, inputs=inputs)

        rows_by_library = {
            row[0]: row
            for row in gateway.con.execute(
                """
                SELECT library, modes, evidence_json, callsite_count, rel_path
                FROM analytics.external_dependency_calls
                WHERE repo = ? AND commit = ?
                ORDER BY library
                """,
                [snapshot.repo, snapshot.commit],
            ).fetchall()
        }
        expect_equal(len(rows_by_library), 2)
        requests_row = rows_by_library["requests"]
        httpx_row = rows_by_library["httpx"]

        expect_equal(_as_list(requests_row[1]), ["read"])
        evidence_data = requests_row[2]
        if isinstance(evidence_data, str):
            evidence_data = json.loads(evidence_data)
        expect_true(assert_mapping_list({"samples": evidence_data}, "samples"))
        expect_equal(requests_row[3], 1)
        expect_equal(requests_row[4], func_ast.rel_path)

        expect_equal(_as_list(httpx_row[1]), ["write"])

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

        deps_by_library = {
            row[0]: row
            for row in gateway.con.execute(
                """
                SELECT library, usage_modes, config_keys, risk_level, callsite_count, function_count
                FROM analytics.external_dependencies
                WHERE repo = ? AND commit = ?
                ORDER BY library
                """,
                [snapshot.repo, snapshot.commit],
            ).fetchall()
        }
        expect_equal(len(deps_by_library), 2)
        requests_dep = deps_by_library["requests"]
        httpx_dep = deps_by_library["httpx"]

        expect_equal(_as_list(requests_dep[1]), ["read"])
        expect_equal(_as_list(requests_dep[2]), ["API_TOKEN"])
        expect_equal(requests_dep[3], "medium")
        expect_equal(requests_dep[4], 1)
        expect_equal(requests_dep[5], 1)

        expect_equal(_as_list(httpx_dep[1]), ["write"])
        expect_equal(_as_list(httpx_dep[2]), ["API_TOKEN"])
        expect_equal(httpx_dep[3], "high")
    finally:
        ctx.close()


def test_dependency_calls_respect_feature_gates(tmp_path: Path) -> None:
    """Calls are skipped when features indicate no IO usage."""
    ctx = create_test_context(tmp_path)
    ctx.require(CORE_PACK)
    repo_root = ctx.repo_root
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
    snapshot = ctx.to_snapshot_ref()
    cfg = ExternalDependenciesStepConfig(snapshot=snapshot)
    patterns_path = repo_root / "config" / "dependency_patterns.yml"
    patterns_path.parent.mkdir(parents=True, exist_ok=True)
    patterns_path.write_text(
        'libs:\n  requests:\n    patterns:\n      - mode: ["read"]\n        method: "get"\n',
        encoding="utf-8",
    )
    gateway = ctx.gateway

    goid = 6001
    func_ast = _build_function_ast(module_path, "fetch_data", goid, repo_root)
    catalog = MockFunctionCatalog(
        functions=[
            function_meta(
                goid=goid,
                rel_path=func_ast.rel_path,
                qualname="fetch_data",
                snapshot=(snapshot.repo, snapshot.commit),
                line_span=(func_ast.start_line, func_ast.end_line),
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
        expect_is_not_none(rows)
        if rows is not None:
            expect_equal(rows[0], 0)
    finally:
        ctx.close()
