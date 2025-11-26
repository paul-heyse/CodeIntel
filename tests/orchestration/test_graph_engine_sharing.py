"""Smoke coverage for shared graph engine wiring in orchestration."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from codeintel.config import (
    BuildPaths,
    ConfigBuilder,
    ExecutionConfig,
    SnapshotRef,
)
from codeintel.config.models import ToolsConfig
from codeintel.config.primitives import GraphBackendConfig
from codeintel.ingestion.source_scanner import ScanProfile
from codeintel.orchestration import steps as orchestration_steps
from codeintel.orchestration.steps import SemanticRolesStep
from codeintel.storage.gateway import StorageGateway
from tests._helpers.gateway import open_ingestion_gateway


def _scan_profile(repo_root: Path) -> ScanProfile:
    return ScanProfile(
        repo_root=repo_root,
        source_roots=(repo_root,),
        include_globs=("*",),
    )


def test_graph_engine_reused_with_backend_flags(tmp_path: Path) -> None:
    """Ensure orchestration uses a shared, backend-aware graph engine."""
    gateway = open_ingestion_gateway(apply_schema=True, ensure_views=True, validate_schema=True)
    snapshot = SnapshotRef(repo_root=tmp_path, repo="demo/repo", commit="deadbeef")
    graph_backend = GraphBackendConfig(use_gpu=True, backend="auto", strict=False)
    execution = ExecutionConfig(
        build_dir=tmp_path / "build",
        tools=ToolsConfig.default(),
        code_profile=_scan_profile(tmp_path),
        config_profile=_scan_profile(tmp_path),
        graph_backend=graph_backend,
    )
    paths = BuildPaths.from_layout(
        repo_root=tmp_path,
        build_dir=execution.build_dir,
        db_path=gateway.config.db_path,
    )
    ctx = orchestration_steps.PipelineContext(
        snapshot=snapshot,
        execution=execution,
        paths=paths,
        gateway=gateway,
    )

    try:
        engine_first = orchestration_steps.ensure_graph_engine(ctx)
        engine_second = orchestration_steps.ensure_graph_engine(ctx)
        if engine_first is not engine_second:
            pytest.fail("Graph engine was not reused across calls")
        if engine_first.use_gpu != graph_backend.use_gpu:
            pytest.fail("Graph engine did not respect backend GPU preference")

        graph = engine_first.call_graph()
        if graph is None:
            pytest.fail("Graph engine did not return a call graph instance")
    finally:
        gateway.close()


def _seed_semantic_roles_prereqs(gateway: StorageGateway, repo: str, commit: str) -> None:
    con = gateway.con
    now = datetime.now(tz=UTC)
    goid = 1
    rel_path = "src/app.py"
    module = "app"
    qualname = "app.main"
    con.execute(
        "INSERT INTO core.modules (module, path, repo, commit, language, tags, owners) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [module, rel_path, repo, commit, "python", "[]", "[]"],
    )
    con.execute(
        """
        INSERT INTO analytics.function_metrics (
            function_goid_h128, repo, commit, rel_path, language, kind, qualname,
            start_line, end_line, loc, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [goid, repo, commit, rel_path, "python", "function", qualname, 1, 5, 5, now],
    )
    con.execute(
        """
        INSERT INTO analytics.function_effects (
            repo, commit, function_goid_h128, is_pure, uses_io, touches_db, uses_time,
            uses_randomness, modifies_globals, modifies_closure, spawns_threads_or_tasks,
            has_transitive_effects, purity_confidence, effects_json, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            repo,
            commit,
            goid,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            False,
            1.0,
            "{}",
            now,
        ],
    )
    con.execute(
        """
        INSERT INTO analytics.function_contracts (
            repo, commit, function_goid_h128, preconditions_json, postconditions_json,
            raises_json, param_nullability_json, return_nullability, contract_confidence, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [repo, commit, goid, "[]", "[]", "[]", "{}", "unknown", 0.5, now],
    )
    con.execute(
        """
        INSERT INTO analytics.graph_metrics_functions (
            repo, commit, function_goid_h128, call_fan_in, call_fan_out, call_in_degree,
            call_out_degree, call_pagerank, call_betweenness, call_closeness, call_cycle_member,
            call_cycle_id, call_layer, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [repo, commit, goid, 0, 1, 0, 1, 0.1, 0.0, 0.0, False, None, 0, now],
    )


def test_engine_reuse_across_semantic_roles(tmp_path: Path) -> None:
    """Semantic roles should reuse the shared engine without rebuilding."""
    gateway = open_ingestion_gateway(apply_schema=True, ensure_views=True, validate_schema=True)
    snapshot = SnapshotRef(repo_root=tmp_path, repo="demo/repo", commit="deadbeef")
    graph_backend = GraphBackendConfig(use_gpu=False, backend="cpu", strict=False)
    execution = ExecutionConfig(
        build_dir=tmp_path / "build",
        tools=ToolsConfig.default(),
        code_profile=_scan_profile(tmp_path),
        config_profile=_scan_profile(tmp_path),
        graph_backend=graph_backend,
    )
    paths = BuildPaths.from_layout(
        repo_root=tmp_path,
        build_dir=execution.build_dir,
        db_path=gateway.config.db_path,
    )
    ctx = orchestration_steps.PipelineContext(
        snapshot=snapshot,
        execution=execution,
        paths=paths,
        gateway=gateway,
    )

    try:
        _seed_semantic_roles_prereqs(gateway, snapshot.repo_slug, snapshot.commit)
        engine = orchestration_steps.ensure_graph_engine(ctx)
        step = SemanticRolesStep()
        step.run(ctx)
        if ctx.graph_engine is not engine:
            pytest.fail("Engine was rebuilt during semantic roles step")
        rows = gateway.con.execute(
            """
            SELECT COUNT(*) FROM analytics.semantic_roles_functions
            WHERE repo = ? AND commit = ?
            """,
            [snapshot.repo_slug, snapshot.commit],
        ).fetchone()
        count = int(rows[0]) if rows is not None else 0
        if count == 0:
            pytest.fail("Semantic roles step did not persist any classifications")
    finally:
        gateway.close()


def test_graph_runtime_reuses_engine(tmp_path: Path) -> None:
    """Shared runtime should expose the same engine instance."""
    gateway = open_ingestion_gateway(apply_schema=True, ensure_views=True, validate_schema=True)
    snapshot = SnapshotRef(repo_root=tmp_path, repo="demo/repo", commit="deadbeef")
    graph_backend = GraphBackendConfig(use_gpu=False, backend="cpu", strict=False)
    execution = ExecutionConfig(
        build_dir=tmp_path / "build",
        tools=ToolsConfig.default(),
        code_profile=_scan_profile(tmp_path),
        config_profile=_scan_profile(tmp_path),
        graph_backend=graph_backend,
    )
    paths = BuildPaths.from_layout(
        repo_root=tmp_path,
        build_dir=execution.build_dir,
        db_path=gateway.config.db_path,
    )
    ctx = orchestration_steps.PipelineContext(
        snapshot=snapshot,
        execution=execution,
        paths=paths,
        gateway=gateway,
    )

    try:
        engine = orchestration_steps.ensure_graph_engine(ctx)
        runtime = orchestration_steps.ensure_graph_runtime(ctx, acx=None)
        if runtime.engine is not engine:
            pytest.fail("Runtime did not reuse the shared engine instance")
        rebuilt = runtime.build_engine(gateway, snapshot.repo_slug, snapshot.commit)
        if rebuilt is not engine:
            pytest.fail("Runtime.build_engine did not return the shared engine")
    finally:
        gateway.close()


def test_runtime_reuse_with_graph_context(tmp_path: Path) -> None:
    """Runtime reuse should hold even when a graph context is provided."""
    gateway = open_ingestion_gateway(apply_schema=True, ensure_views=True, validate_schema=True)
    snapshot = SnapshotRef(repo_root=tmp_path, repo="demo/repo", commit="deadbeef")
    graph_backend = GraphBackendConfig(use_gpu=False, backend="cpu", strict=False)
    execution = ExecutionConfig(
        build_dir=tmp_path / "build",
        tools=ToolsConfig.default(),
        code_profile=_scan_profile(tmp_path),
        config_profile=_scan_profile(tmp_path),
        graph_backend=graph_backend,
    )
    paths = BuildPaths.from_layout(
        repo_root=tmp_path,
        build_dir=execution.build_dir,
        db_path=gateway.config.db_path,
    )
    ctx = orchestration_steps.PipelineContext(
        snapshot=snapshot,
        execution=execution,
        paths=paths,
        gateway=gateway,
    )
    builder = ConfigBuilder.from_snapshot(
        repo=snapshot.repo_slug,
        commit=snapshot.commit,
        repo_root=snapshot.repo_root,
        build_dir=paths.build_dir,
    )

    try:
        graph_ctx = orchestration_steps.build_graph_context(
            builder.graph_metrics(),
            now=datetime.now(tz=UTC),
            use_gpu=False,
        )
        runtime_one = orchestration_steps.ensure_graph_runtime(ctx, graph_ctx=graph_ctx)
        runtime_two = orchestration_steps.ensure_graph_runtime(ctx, graph_ctx=graph_ctx)
        if runtime_one.engine is not runtime_two.engine:
            pytest.fail("Graph runtime did not reuse the shared engine with graph context")
    finally:
        gateway.close()
