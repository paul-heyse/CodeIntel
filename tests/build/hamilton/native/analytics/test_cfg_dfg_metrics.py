"""Scaffolding tests for CFG/DFG analytics Hamilton nodes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from codeintel.build.hamilton.env import BuildEnv
from tests._helpers.schemas import ensure_schema_service

try:
    from codeintel.build.hamilton.native.analytics.cfg_dfg_metrics import (
        cfg_block_metrics__base,
        cfg_dfg_metrics_analysis,
        cfg_dfg_metrics_inputs,
        cfg_function_metrics__base,
        cfg_function_metrics_ext__base,
        dfg_block_metrics__base,
        dfg_function_metrics__base,
        dfg_function_metrics_ext__base,
    )
    from codeintel.build.hamilton.native.graphs.cfg_dfg import (
        cfg_blocks_compute,
        cfg_dfg_analysis,
        cfg_edges_compute,
        dfg_edges_compute,
    )
except RuntimeError as exc:
    if "SchemaService has not been configured" in str(exc):
        pytest.skip(
            "SchemaService is required for CFG/DFG metrics nodes.",
            allow_module_level=True,
        )
    raise

try:
    import polars as pl
except ModuleNotFoundError:
    pytest.skip("polars is required for CFG/DFG metrics tests", allow_module_level=True)

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


def test_cfg_dfg_metrics_columns(tmp_path: Path) -> None:
    """Ensure CFG/DFG metrics nodes expose expected columns.

    Raises
    ------
    RuntimeError
        If required schema services are unavailable for the test runtime.
    """
    try:
        ensure_schema_service()
    except RuntimeError as exc:
        if "ContractService has not been configured" in str(exc):
            pytest.skip("ContractService is required for CFG/DFG metrics nodes.")
        raise
    _write_sample_module(tmp_path)
    env = _fake_env(tmp_path)
    graph_analysis = cfg_dfg_analysis(
        env,
        _sample_goids_frame(),
        _sample_ast_nodes_frame(),
    )
    cfg_blocks = cfg_blocks_compute(graph_analysis)
    cfg_edges = cfg_edges_compute(graph_analysis)
    dfg_edges = dfg_edges_compute(graph_analysis)

    metrics_inputs = cfg_dfg_metrics_inputs(
        cfg_blocks,
        cfg_edges,
        dfg_edges,
        _sample_goids_frame(),
        _sample_modules_frame(),
    )
    metrics_analysis = cfg_dfg_metrics_analysis(env, metrics_inputs)

    def _columns(frame: object) -> list[str]:
        if hasattr(frame, "column_names"):
            return list(frame.column_names)
        if hasattr(frame, "columns"):
            return list(frame.columns)
        if hasattr(frame, "collect"):
            return list(frame.collect().columns)
        msg = f"Unsupported metrics frame type: {type(frame).__name__}"
        raise TypeError(msg)

    cfg_fn = cfg_function_metrics__base(metrics_analysis)
    assert _columns(cfg_fn) == [
        "function_goid_h128",
        "repo",
        "commit",
        "rel_path",
        "module",
        "qualname",
        "cfg_block_count",
        "cfg_edge_count",
        "cfg_has_cycles",
        "cfg_scc_count",
        "cfg_longest_path_len",
        "cfg_avg_shortest_path_len",
        "cfg_branching_factor_mean",
        "cfg_branching_factor_max",
        "cfg_linear_block_fraction",
        "cfg_dom_tree_height",
        "cfg_dominance_frontier_size_mean",
        "cfg_dominance_frontier_size_max",
        "cfg_loop_count",
        "cfg_loop_nesting_depth_max",
        "cfg_bc_betweenness_max",
        "cfg_bc_betweenness_mean",
        "cfg_bc_closeness_mean",
        "cfg_bc_eigenvector_max",
        "created_at",
        "metrics_version",
    ]

    cfg_blocks_metrics = cfg_block_metrics__base(metrics_analysis)
    assert _columns(cfg_blocks_metrics) == [
        "function_goid_h128",
        "repo",
        "commit",
        "block_idx",
        "is_entry",
        "is_exit",
        "is_branch",
        "is_join",
        "dom_depth",
        "dominates_exit",
        "bc_betweenness",
        "bc_closeness",
        "bc_eigenvector",
        "in_loop_scc",
        "loop_header",
        "loop_nesting_depth",
        "created_at",
        "metrics_version",
    ]

    cfg_ext = cfg_function_metrics_ext__base(metrics_analysis)
    assert _columns(cfg_ext) == [
        "function_goid_h128",
        "repo",
        "commit",
        "unreachable_block_count",
        "loop_header_count",
        "true_edge_count",
        "false_edge_count",
        "back_edge_count",
        "exception_edge_count",
        "fallthrough_edge_count",
        "loop_edge_count",
        "entry_exit_simple_paths",
        "created_at",
        "metrics_version",
    ]

    dfg_fn = dfg_function_metrics__base(metrics_analysis)
    assert _columns(dfg_fn) == [
        "function_goid_h128",
        "repo",
        "commit",
        "rel_path",
        "module",
        "qualname",
        "dfg_block_count",
        "dfg_edge_count",
        "dfg_phi_edge_count",
        "dfg_symbol_count",
        "dfg_component_count",
        "dfg_scc_count",
        "dfg_has_cycles",
        "dfg_longest_chain_len",
        "dfg_avg_shortest_path_len",
        "dfg_avg_in_degree",
        "dfg_avg_out_degree",
        "dfg_max_in_degree",
        "dfg_max_out_degree",
        "dfg_branchy_block_fraction",
        "dfg_bc_betweenness_max",
        "dfg_bc_betweenness_mean",
        "dfg_bc_eigenvector_max",
        "created_at",
        "metrics_version",
    ]

    dfg_blocks_metrics = dfg_block_metrics__base(metrics_analysis)
    assert _columns(dfg_blocks_metrics) == [
        "function_goid_h128",
        "repo",
        "commit",
        "block_idx",
        "dfg_in_degree",
        "dfg_out_degree",
        "dfg_phi_in_degree",
        "dfg_phi_out_degree",
        "dfg_bc_betweenness",
        "dfg_bc_closeness",
        "dfg_bc_eigenvector",
        "dfg_in_chain",
        "dfg_in_scc",
        "created_at",
        "metrics_version",
    ]

    dfg_ext = dfg_function_metrics_ext__base(metrics_analysis)
    assert _columns(dfg_ext) == [
        "function_goid_h128",
        "repo",
        "commit",
        "data_flow_edge_count",
        "intra_block_edge_count",
        "use_kind_phi_count",
        "use_kind_data_flow_count",
        "use_kind_intra_block_count",
        "use_kind_other_count",
        "phi_edge_ratio",
        "entry_exit_simple_paths",
        "created_at",
        "metrics_version",
    ]
