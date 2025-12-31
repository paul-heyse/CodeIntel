"""Scaffolding tests for v1 graph Hamilton nodes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.graphs.call_graph import (
    call_graph_edges_compute,
    call_graph_nodes_compute,
)
from codeintel.build.hamilton.native.graphs.cfg_dfg import (
    cfg_blocks_compute,
    cfg_edges_compute,
    dfg_edges_compute,
)
from codeintel.build.hamilton.native.graphs.import_graph import (
    import_graph_analysis,
    import_graph_edges_compute,
    import_modules_compute,
)

if TYPE_CHECKING:
    import polars as pl

pl = pytest.importorskip("polars")
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
            "start_line": [6, 3],
            "end_line": [7, 4],
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


def _write_sample_module(repo_root: Path) -> None:
    module_path = repo_root / "src" / "app.py"
    module_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.write_text(
        "import json\n\n"
        "def helper() -> None:\n"
        "    json.dumps({})\n\n"
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
    result = frame.collect()
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
    result = frame.collect()
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
    result_modules = modules.collect()
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
    result_edges = edges.collect()
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


def test_cfg_dfg_compute_columns() -> None:
    """Ensure CFG/DFG compute columns match the scaffold."""
    blocks = cfg_blocks_compute(_sample_goids_frame())
    result_blocks = blocks.collect()
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

    cfg_edges = cfg_edges_compute(blocks)
    result_cfg_edges = cfg_edges.collect()
    assert result_cfg_edges.columns == [
        "function_goid_h128",
        "src_block_id",
        "dst_block_id",
        "edge_kind",
    ]

    dfg_edges = dfg_edges_compute(blocks)
    result_dfg_edges = dfg_edges.collect()
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
