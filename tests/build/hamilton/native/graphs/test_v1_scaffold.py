"""Scaffolding tests for v1 graph Hamilton nodes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.graphs.call_graph import (
    call_graph_edges_compute,
    call_graph_nodes_compute,
)
from codeintel.build.hamilton.native.graphs.cfg_dfg import (
    cfg_blocks_compute,
    cfg_dfg_analysis,
    cfg_edges_compute,
    dfg_edges_compute,
)
from codeintel.build.hamilton.native.graphs.goids import (
    GOID_CROSSWALK_COLUMNS,
    GOIDS_COLUMNS,
    goid_crosswalk__base,
    goids__base,
    goids_analysis,
    goids_inputs,
)
from codeintel.build.hamilton.native.graphs.import_graph import (
    import_graph_analysis,
    import_graph_edges_compute,
    import_modules_compute,
)
from codeintel.build.hamilton.native.graphs.symbol_use import symbol_use_edges_compute
from codeintel.build.tabular.conversion import tabular_to_lazyframe
from codeintel.build.tabular.types import InferableTabularInput

try:
    import polars as pl
except ModuleNotFoundError:
    pytest.skip("polars is required for graph scaffold tests", allow_module_level=True)

pytestmark = pytest.mark.no_runtime_env


@dataclass(frozen=True)
class _FakeSnapshot:
    repo: str
    commit: str
    repo_root: Path


@dataclass(frozen=True)
class _FakeEnv:
    snapshot: _FakeSnapshot
    repo: str
    commit: str


def _fake_env(repo_root: Path) -> BuildEnv:
    snapshot = _FakeSnapshot(repo="repo", commit="commit", repo_root=repo_root)
    env = _FakeEnv(snapshot=snapshot, repo=snapshot.repo, commit=snapshot.commit)
    return cast("BuildEnv", env)


def _sample_goids_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "goid_h128": [1, 2],
            "urn": ["urn:goid:1", "urn:goid:2"],
            "repo": ["repo", "repo"],
            "commit": ["commit", "commit"],
            "rel_path": ["src/app.py", "src/app.py"],
            "language": ["python", "python"],
            "kind": ["function", "function"],
            "qualname": ["app.main", "app.helper"],
            "start_line": [7, 3],
            "end_line": [8, 5],
            "created_at": [
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 1, tzinfo=UTC),
            ],
        }
    )


def _sample_modules_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "repo": ["repo"],
            "commit": ["commit"],
            "module": ["app"],
            "path": ["src/app.py"],
            "language": ["python"],
        }
    )


def _sample_ast_nodes_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "path": ["src/app.py", "src/app.py", "src/app.py"],
            "node_type": ["Module", "FunctionDef", "FunctionDef"],
            "name": ["app", "helper", "main"],
            "qualname": ["app", "app.helper", "app.main"],
            "lineno": [None, 3, 7],
            "end_lineno": [None, 5, 8],
        }
    )


def _sample_scip_occurrences_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": ["local app.main", "local app.main"],
            "rel_path": ["src/app.py", "src/app.py"],
            "start_line": [7, 7],
            "roles": [1, 2],
        }
    )


def _collect_frame(frame: InferableTabularInput) -> pl.DataFrame:
    return tabular_to_lazyframe(frame).collect()


def _write_sample_module(repo_root: Path) -> None:
    module_path = repo_root / "src" / "app.py"
    module_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.write_text(
        "import json\n\n"
        "def helper() -> None:\n"
        "    value = json.dumps({})\n"
        "    print(value)\n\n"
        "def main() -> None:\n"
        "    helper()\n",
        encoding="utf-8",
    )


def test_call_graph_nodes_compute_columns(tmp_path: Path) -> None:
    """Ensure call graph nodes compute columns match the scaffold."""
    _write_sample_module(tmp_path)
    env = _fake_env(tmp_path)
    frame = call_graph_nodes_compute(
        env,
        _sample_goids_frame(),
        _sample_modules_frame(),
    )
    result = _collect_frame(frame)
    assert result.columns == [
        "goid_h128",
        "language",
        "kind",
        "arity",
        "is_public",
        "rel_path",
    ]


def test_call_graph_edges_compute_columns(tmp_path: Path) -> None:
    """Ensure call graph edges compute columns match the scaffold."""
    _write_sample_module(tmp_path)
    env = _fake_env(tmp_path)
    goids = _sample_goids_frame()
    modules = _sample_modules_frame()
    frame = call_graph_edges_compute(env, goids, modules)
    result = _collect_frame(frame)
    assert result.columns == [
        "repo",
        "commit",
        "caller_goid_h128",
        "callee_goid_h128",
        "callsite_path",
        "callsite_line",
        "callsite_col",
        "language",
        "kind",
        "resolved_via",
        "confidence",
        "evidence_json",
    ]


def test_import_graph_compute_columns(tmp_path: Path) -> None:
    """Ensure import graph compute columns match the scaffold."""
    _write_sample_module(tmp_path)
    env = _fake_env(tmp_path)
    analysis = import_graph_analysis(env, _sample_modules_frame())
    modules = import_modules_compute(env, analysis)
    result_modules = _collect_frame(modules)
    assert result_modules.columns == [
        "repo",
        "commit",
        "module",
        "scc_id",
        "component_size",
        "layer",
        "cycle_group",
    ]

    edges = import_graph_edges_compute(env, analysis)
    result_edges = _collect_frame(edges)
    assert result_edges.columns == [
        "repo",
        "commit",
        "src_module",
        "dst_module",
        "src_fan_out",
        "dst_fan_in",
        "cycle_group",
        "module_layer",
    ]


def test_cfg_dfg_compute_columns(tmp_path: Path) -> None:
    """Ensure CFG/DFG compute columns match the scaffold."""
    _write_sample_module(tmp_path)
    env = _fake_env(tmp_path)
    analysis = cfg_dfg_analysis(env, _sample_goids_frame(), _sample_ast_nodes_frame())
    blocks = cfg_blocks_compute(analysis)
    result_blocks = _collect_frame(blocks)
    assert result_blocks.columns == [
        "function_goid_h128",
        "block_idx",
        "block_id",
        "label",
        "file_path",
        "start_line",
        "end_line",
        "kind",
        "stmts_json",
        "in_degree",
        "out_degree",
    ]

    cfg_edges = cfg_edges_compute(analysis)
    result_cfg_edges = _collect_frame(cfg_edges)
    assert result_cfg_edges.columns == [
        "function_goid_h128",
        "src_block_id",
        "dst_block_id",
        "edge_kind",
    ]

    dfg_edges = dfg_edges_compute(analysis)
    result_dfg_edges = _collect_frame(dfg_edges)
    assert result_dfg_edges.columns == [
        "function_goid_h128",
        "src_block_id",
        "dst_block_id",
        "src_var",
        "dst_var",
        "edge_kind",
        "via_phi",
        "use_kind",
    ]


def test_goids_compute_columns(tmp_path: Path) -> None:
    """Ensure GOID compute columns match the schema."""
    env = _fake_env(tmp_path)
    inputs = goids_inputs(_sample_modules_frame(), _sample_ast_nodes_frame())
    analysis = goids_analysis(env, inputs)
    goids = _collect_frame(goids__base(analysis))
    crosswalk = _collect_frame(goid_crosswalk__base(analysis))
    assert goids.columns == list(GOIDS_COLUMNS)
    assert crosswalk.columns == list(GOID_CROSSWALK_COLUMNS)


def test_symbol_use_edges_compute_columns(tmp_path: Path) -> None:
    """Ensure symbol use edges compute columns match the schema."""
    _write_sample_module(tmp_path)
    frame = symbol_use_edges_compute(
        _sample_scip_occurrences_frame(),
        _sample_modules_frame(),
        _sample_goids_frame(),
    )
    result = _collect_frame(frame)
    assert result.columns == [
        "symbol",
        "def_path",
        "use_path",
        "same_file",
        "same_module",
        "def_goid_h128",
        "use_goid_h128",
    ]
