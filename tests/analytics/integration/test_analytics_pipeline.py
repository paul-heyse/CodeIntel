"""Integration-style tests for analytics compute modules."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import networkx as nx

from codeintel.analytics.ast_features.model import FunctionAstFeatures, IoFlags
from codeintel.analytics.data_models.core import compute_data_models
from codeintel.analytics.dependencies.core import (
    ExternalDependencyInputs,
    build_external_dependency_calls,
)
from codeintel.analytics.entrypoints.core import build_entrypoints
from codeintel.analytics.functions.function_contracts import compute_function_contracts
from codeintel.analytics.functions.function_effects import (
    FunctionEffectsInputs,
    compute_function_effects,
)
from codeintel.analytics.functions.function_history import compute_function_history
from codeintel.analytics.functions.metrics import compute_function_metrics_and_types
from codeintel.analytics.parsing.ast_cache import FunctionAst
from codeintel.analytics.runtime.graph import GraphRuntime, GraphRuntimeOptions
from codeintel.config.primitives import SnapshotRef
from codeintel.config.steps_analytics import (
    DataModelsStepConfig,
    EntryPointsStepConfig,
    FunctionAnalyticsStepConfig,
    FunctionContractsStepConfig,
    FunctionEffectsStepConfig,
    FunctionHistoryStepConfig,
)
from codeintel.config.steps_graphs import ExternalDependenciesStepConfig
from codeintel.graphs.catalog import FunctionCatalogService
from codeintel.storage.sql.builder import ensure_schema
from tests._helpers.gateway import GatewayFactory


@dataclass
class SampleRepo:
    """Bundle of shared test fixtures for analytics pipelines."""

    snapshot: SnapshotRef
    gateway: object
    rel_path: str
    goid_route: int
    goid_method: int
    class_goid: int
    ast_map: dict[int, FunctionAst]
    features: dict[int, FunctionAstFeatures]
    module_map: dict[str, str]


def _write_sample_repo(tmp_path: Path) -> SampleRepo:
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
            "    \"\"\"User model\"\"\"",
            "    id: int",
            "    name: str",
            "",
            "    def to_dict(self):",
            "        return {'id': self.id, 'name': self.name}",
        ]
    )
    rel_path = "pkg/api.py"
    api_path.write_text(source, encoding="utf-8")

    config_path = repo_root / "config" / "dependency_patterns.yml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "libs:\n"
        "  requests:\n"
        "    patterns:\n"
        "      - mode: [\"http\"]\n"
        "        match: \"requests.get\"\n",
        encoding="utf-8",
    )

    snapshot = SnapshotRef(repo="demo", commit="abc123", repo_root=repo_root)
    gateway = GatewayFactory().with_snapshot(repo="demo", commit="abc123").open()
    now = datetime.now(tz=UTC)

    ensure_schema(gateway.con, "analytics.coverage_functions")
    ensure_schema(gateway.con, "analytics.test_coverage_edges")
    ensure_schema(gateway.con, "analytics.test_catalog")

    gateway.con.execute(
        """
        INSERT INTO core.modules(module, path, repo, commit, language, tags, owners)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        ("pkg.api", rel_path, snapshot.repo, snapshot.commit, "python", json.dumps([]), json.dumps([])),
    )

    goids = {
        "route": 1001,
        "method": 1002,
        "class": 2001,
    }
    for goid, urn, kind, qualname, start, end in [
        (goids["route"], "urn:route", "function", "list_items", 7, 13),
        (goids["method"], "urn:method", "function", "User.to_dict", 20, 21),
        (goids["class"], "urn:class", "class", "User", 15, 21),
    ]:
        gateway.con.execute(
            """
            INSERT INTO core.goids (
                goid_h128, urn, repo, commit, rel_path, language, kind, qualname,
                start_line, end_line, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                goid,
                urn,
                snapshot.repo,
                snapshot.commit,
                rel_path,
                "python",
                kind,
                qualname,
                start,
                end,
                now,
            ),
        )

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
            rel_path,
            "pkg.api",
            "User",
            "class",
            15,
            21,
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

    tree = ast.parse(source)
    list_func = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "list_items"
    )
    class_method = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "to_dict"
    )
    ast_map = {
        goids["route"]: FunctionAst(
            goid=goids["route"],
            rel_path=rel_path,
            qualname="list_items",
            start_line=7,
            end_line=13,
            node=list_func,
            lines=source.splitlines(),
        ),
        goids["method"]: FunctionAst(
            goid=goids["method"],
            rel_path=rel_path,
            qualname="User.to_dict",
            start_line=20,
            end_line=21,
            node=class_method,
            lines=source.splitlines(),
        ),
    }

    features = {
        goids["route"]: FunctionAstFeatures(
            goid=goids["route"],
            rel_path=rel_path,
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
        rel_path=rel_path,
        goid_route=goids["route"],
        goid_method=goids["method"],
        class_goid=goids["class"],
        ast_map=ast_map,
        features=features,
        module_map={rel_path: "pkg.api"},
    )


def _row_count(sample: SampleRepo, table: str) -> int:
    queries = {
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
    if table not in queries:
        message = f"Unsupported table for count: {table}"
        raise ValueError(message)
    return int(
        sample.gateway.con.execute(
            queries[table],
            [sample.snapshot.repo, sample.snapshot.commit],
        ).fetchone()[0]
    )


def _build_runtime(sample: SampleRepo) -> GraphRuntime:
    call_graph = nx.DiGraph()
    call_graph.add_edge(sample.goid_route, sample.goid_method)

    class _FakeEngine:
        def load_call_graph(self) -> nx.DiGraph:
            return call_graph

    runtime = GraphRuntime(GraphRuntimeOptions(snapshot=sample.snapshot), _FakeEngine())
    runtime.ensure_call_graph()
    return runtime


def test_full_analytics_pipeline(tmp_path: Path) -> None:
    """Execute compute flows end-to-end on a small in-memory snapshot."""
    sample = _write_sample_repo(tmp_path)
    catalog = FunctionCatalogService.from_db(
        sample.gateway,
        repo=sample.snapshot.repo,
        commit=sample.snapshot.commit,
    )

    summary = compute_function_metrics_and_types(
        sample.gateway,
        FunctionAnalyticsStepConfig(snapshot=sample.snapshot),
    )
    assert summary["metrics_rows"] >= 2

    compute_function_contracts(
        sample.gateway,
        FunctionContractsStepConfig(snapshot=sample.snapshot, max_conditions_per_func=5),
        function_ast_map=sample.ast_map,
        catalog=catalog,
    )
    assert _row_count(sample, "analytics.function_contracts") >= 2

    runtime = _build_runtime(sample)
    compute_function_effects(
        sample.gateway,
        FunctionEffectsStepConfig(snapshot=sample.snapshot),
        inputs=FunctionEffectsInputs(
            catalog_provider=catalog,
            runtime=runtime,
            ast_map=sample.ast_map,
            missing_goids=set(),
        ),
    )
    assert _row_count(sample, "analytics.function_effects") >= 2

    compute_function_history(
        sample.gateway,
        FunctionHistoryStepConfig(
            snapshot=sample.snapshot,
            max_history_days=7,
            min_lines_threshold=1,
            default_branch="HEAD",
        ),
    )
    assert _row_count(sample, "analytics.function_history") >= 2

    compute_data_models(
        sample.gateway,
        DataModelsStepConfig(snapshot=sample.snapshot),
    )
    assert _row_count(sample, "analytics.data_models") >= 1

    build_external_dependency_calls(
        sample.gateway,
        ExternalDependenciesStepConfig(snapshot=sample.snapshot),
        inputs=ExternalDependencyInputs(
            catalog_provider=catalog,
            module_map=sample.module_map,
            ast_by_goid=sample.ast_map,
            features_map=sample.features,
        ),
    )
    assert _row_count(sample, "analytics.external_dependency_calls") >= 1

    build_entrypoints(
        sample.gateway,
        EntryPointsStepConfig(snapshot=sample.snapshot),
        catalog_provider=catalog,
        module_map=sample.module_map,
        features_map=sample.features,
    )
    assert _row_count(sample, "analytics.entrypoints") >= 1

    sample.gateway.close()
