"""Shared seeded sample repository and runtime helpers for analytics integration tests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import networkx as nx

from codeintel.analytics.ast_features.model import FunctionAstFeatures, IoFlags
from codeintel.analytics.parsing.ast_cache import FunctionAst
from codeintel.analytics.runtime.graph import GraphRuntime, GraphRuntimeOptions
from codeintel.config.primitives import SnapshotRef
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.sql.builder import ensure_schema
from tests._helpers.contracts import count_rows
from tests._helpers.gateway import GatewayFactory
from tests._helpers.graphs import (
    build_ast_map,
    build_graph_engine_double,
    build_module_map,
    insert_goids,
    insert_modules,
)

TABLE_QUERIES: dict[str, str] = {
    "analytics.function_contracts": """
        SELECT COUNT(*) FROM analytics.function_contracts WHERE repo = ? AND commit = ?
    """,
    "analytics.function_effects": """
        SELECT COUNT(*) FROM analytics.function_effects WHERE repo = ? AND commit = ?
    """,
    "analytics.function_history": """
        SELECT COUNT(*) FROM analytics.function_history WHERE repo = ? AND commit = ?
    """,
    "analytics.data_models": """
        SELECT COUNT(*) FROM analytics.data_models WHERE repo = ? AND commit = ?
    """,
    "analytics.external_dependency_calls": """
        SELECT COUNT(*) FROM analytics.external_dependency_calls WHERE repo = ? AND commit = ?
    """,
    "analytics.entrypoints": """
        SELECT COUNT(*) FROM analytics.entrypoints WHERE repo = ? AND commit = ?
    """,
}


@dataclass
class SampleRepo:
    """Bundle of shared test fixtures for analytics pipelines."""

    snapshot: SnapshotRef
    gateway: StorageGateway
    rel_path: str
    goid_route: int
    goid_method: int
    class_goid: int
    ast_map: dict[int, FunctionAst]
    features: dict[int, FunctionAstFeatures]
    module_map: dict[str, str]


def write_sample_repo(tmp_path: Path) -> SampleRepo:
    """
    Seed a small repository layout for analytics integration tests.

    Parameters
    ----------
    tmp_path
        Temporary directory provided by pytest.

    Returns
    -------
    SampleRepo
        Seeded repository metadata and handles.
    """
    repo_root = tmp_path / "repo"
    api_path = repo_root / "pkg" / "api.py"
    api_path.parent.mkdir(parents=True, exist_ok=True)
    source = "\n".join(
        [
            "from fastapi import FastAPI",
            "import requests",
            "",
            "app = FastAPI()",
            "",
            "@app.get('/items')",
            "def list_items(limit: int | None = None):",
            "    if limit is None:",
            "        raise ValueError('limit required')",
            "    if not isinstance(limit, int):",
            "        return 0",
            "    data = requests.get('http://example.com')",
            "    return data.status_code",
            "",
            "class User:",
            '    """User model"""',
            "    id: int",
            "    name: str",
            "",
            "    def to_dict(self):",
            "        return {'id': self.id, 'name': self.name}",
        ]
    )
    api_path.write_text(source, encoding="utf-8")

    config_path = repo_root / "config" / "dependency_patterns.yml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "libs:\n"
        "  requests:\n"
        "    patterns:\n"
        '      - mode: ["http"]\n'
        '        match: "requests.get"\n',
        encoding="utf-8",
    )

    snapshot = SnapshotRef(repo="demo", commit="abc123", repo_root=repo_root)
    gateway = GatewayFactory().with_snapshot(repo="demo", commit="abc123").open()
    now = datetime.now(tz=UTC)

    ensure_schema(gateway.con, "analytics.coverage_functions")
    ensure_schema(gateway.con, "analytics.test_coverage_edges")
    ensure_schema(gateway.con, "analytics.test_catalog")

    paths = {"pkg.api": api_path}
    insert_modules(gateway, snapshot, paths)

    goids = {
        "list_items": 1001,
        "User.to_dict": 1002,
        "User": 2001,
    }
    ast_map = build_ast_map(
        paths,
        goids,
        snapshot.repo_root,
        target_names={"pkg.api": ("list_items", "User.to_dict", "User")},
    )
    insert_goids(gateway, snapshot, ast_map, now=now)

    user_ast = ast_map[goids["User"]]
    gateway.con.execute(
        """
        INSERT INTO core.docstrings (
            repo, commit, rel_path, module, qualname, kind, lineno, end_lineno,
            raw_docstring, style, short_desc, long_desc, params, returns, raises,
            examples, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            snapshot.repo,
            snapshot.commit,
            user_ast.rel_path,
            "pkg.api",
            "User",
            "class",
            user_ast.start_line,
            user_ast.end_line,
            "User model",
            "numpy",
            "User model",
            None,
            json.dumps([]),
            None,
            None,
            None,
            now,
        ),
    )

    features = {
        goids["list_items"]: FunctionAstFeatures(
            goid=goids["list_items"],
            rel_path=ast_map[goids["list_items"]].rel_path,
            qualname="list_items",
            is_async=False,
            decorators=("app.get('/items')",),
            imports={"requests": "requests"},
            libraries_used=frozenset({"requests", "fastapi"}),
            io_flags=IoFlags(uses_network=True),
            uses_concurrency_lib=False,
            uses_threading=False,
            uses_asyncio_lib=False,
            http_client_libs=frozenset({"requests"}),
            http_server_libs=frozenset({"fastapi"}),
            db_libs=frozenset(),
            message_libs=frozenset(),
            config_read_count=0,
            feature_flag_count=0,
        )
    }

    return SampleRepo(
        snapshot=snapshot,
        gateway=gateway,
        rel_path=ast_map[goids["list_items"]].rel_path,
        goid_route=goids["list_items"],
        goid_method=goids["User.to_dict"],
        class_goid=goids["User"],
        ast_map=ast_map,
        features=features,
        module_map=build_module_map(
            ast_map,
            {
                goids["list_items"]: "pkg.api",
                goids["User.to_dict"]: "pkg.api",
                goids["User"]: "pkg.api",
            },
        ),
    )


def count_table_rows(sample: SampleRepo, table: str) -> int:
    """
    Count rows for the seeded repo scoped by repo/commit.

    Parameters
    ----------
    sample
        Seeded repository wrapper.
    table
        Table name to query.

    Returns
    -------
    int
        Row count for the requested table.

    Raises
    ------
    ValueError
        If the requested table is not supported.
    """
    query = TABLE_QUERIES.get(table)
    if query is None:
        message = f"Unsupported table for count: {table}"
        raise ValueError(message)
    return count_rows(sample.gateway.con, query, [sample.snapshot.repo, sample.snapshot.commit])


def build_runtime(sample: SampleRepo) -> GraphRuntime:
    """
    Construct runtime with seeded call graph.

    Parameters
    ----------
    sample
        Seeded repository wrapper.

    Returns
    -------
    GraphRuntime
        Runtime configured with stubbed graphs.
    """
    call_graph = nx.DiGraph()
    call_graph.add_edge(sample.goid_route, sample.goid_method)

    engine = build_graph_engine_double(
        sample.gateway,
        sample.snapshot,
        call_graph=call_graph,
    )
    runtime = GraphRuntime(GraphRuntimeOptions(snapshot=sample.snapshot), engine)
    runtime.ensure_call_graph()
    return runtime
