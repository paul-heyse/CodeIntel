"""Tests for native pipeline runner functionality."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import pytest
from typer.testing import CliRunner

from codeintel.cli import app
from codeintel.config import SnapshotRef
from codeintel.config.primitives import BuildPaths, GraphBackendConfig
from codeintel.ingestion.infrastructure.scanning import default_code_profile
from codeintel.pipeline.execution.context import PipelineContext
from codeintel.pipeline.execution.step_runner import (
    ExportArgs,
    ExportHooks,
    HistoryTimeseriesParams,
    close_gateways,
    gateway_cache_stats,
    run_export_docs,
    run_full_pipeline,
    run_pipeline_with_retries,
)
from codeintel.pipeline.export.export_jsonl import ExportCallOptions
from codeintel.pipeline.export.runner import ExportOptions
from codeintel.pipeline.steps.base import StepPhase
from codeintel.pipeline.steps.registry import build_registry
from codeintel.storage.gateway import (
    DuckDBConnection,
    StorageConfig,
    StorageGateway,
    open_gateway,
)
from codeintel.storage.views import create_all_views
from tests._helpers import seed_docs_export_minimal
from tests._helpers.gateway import open_fresh_duckdb

runner = CliRunner()


class _StubStep:
    """Stub pipeline step for testing."""

    name: str
    description: str
    phase: StepPhase
    deps: Sequence[str]
    run_count: int
    last_ctx: PipelineContext | None

    def __init__(self, name: str, deps: Sequence[str] = ()) -> None:
        self.name = name
        self.description = f"Stub step: {name}"
        self.phase = StepPhase.INGESTION
        self.deps = deps
        self.run_count = 0
        self.last_ctx = None

    def run(self, ctx: PipelineContext) -> None:
        """Execute stub step."""
        self.run_count += 1
        self.last_ctx = ctx


def test_runner_module_imports() -> None:
    """Runner module can be imported without side effects."""
    if not callable(run_full_pipeline):
        pytest.fail("run_full_pipeline is not callable")
    if not callable(run_pipeline_with_retries):
        pytest.fail("run_pipeline_with_retries is not callable")


def test_run_pipeline_with_retries_executes_steps_in_order(tmp_path: Path) -> None:
    """Pipeline steps execute in topological order with retry support."""
    # Create stub steps with dependencies
    step_a = _StubStep("step_a")
    step_b = _StubStep("step_b", deps=["step_a"])
    step_c = _StubStep("step_c", deps=["step_b"])

    registry = build_registry(
        {
            "step_a": step_a,
            "step_b": step_b,
            "step_c": step_c,
        }
    )

    # Create minimal context
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / ".git").mkdir(parents=True, exist_ok=True)
    build_dir = repo_root / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    db_path = build_dir / "db" / "codeintel.duckdb"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    gateway = open_fresh_duckdb(db_path)
    try:
        snapshot = SnapshotRef(repo_root=repo_root, repo="test/repo", commit="abc123")
        paths = BuildPaths.from_layout(
            repo_root=repo_root,
            build_dir=build_dir,
            db_path=db_path,
        )

        ctx = PipelineContext(
            snapshot=snapshot,
            paths=paths,
            gateway=gateway,
            code_profile_cfg=default_code_profile(repo_root),
            config_profile_cfg=default_code_profile(repo_root),
            graph_backend_cfg=GraphBackendConfig(),
        )

        # Execute pipeline
        run_pipeline_with_retries(ctx, registry)

        # Verify all steps executed
        assert step_a.run_count == 1, "step_a should execute once"
        assert step_b.run_count == 1, "step_b should execute once"
        assert step_c.run_count == 1, "step_c should execute once"

        # Verify context was passed
        assert step_a.last_ctx is ctx
        assert step_b.last_ctx is ctx
        assert step_c.last_ctx is ctx
    finally:
        gateway.close()


def test_run_pipeline_with_selected_steps(tmp_path: Path) -> None:
    """Pipeline respects selected_steps with dependency expansion."""
    step_a = _StubStep("step_a")
    step_b = _StubStep("step_b", deps=["step_a"])
    step_c = _StubStep("step_c", deps=["step_b"])
    step_d = _StubStep("step_d")  # Unrelated step

    registry = build_registry(
        {
            "step_a": step_a,
            "step_b": step_b,
            "step_c": step_c,
            "step_d": step_d,
        }
    )

    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / ".git").mkdir(parents=True, exist_ok=True)
    build_dir = repo_root / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    db_path = build_dir / "db" / "codeintel.duckdb"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    gateway = open_fresh_duckdb(db_path)
    try:
        snapshot = SnapshotRef(repo_root=repo_root, repo="test/repo", commit="abc123")
        paths = BuildPaths.from_layout(
            repo_root=repo_root,
            build_dir=build_dir,
            db_path=db_path,
        )

        ctx = PipelineContext(
            snapshot=snapshot,
            paths=paths,
            gateway=gateway,
            code_profile_cfg=default_code_profile(repo_root),
            config_profile_cfg=default_code_profile(repo_root),
            graph_backend_cfg=GraphBackendConfig(),
        )

        # Execute only step_c (should include step_a, step_b as dependencies)
        run_pipeline_with_retries(ctx, registry, selected_steps=["step_c"])

        # Verify step_c and its dependencies executed
        assert step_a.run_count == 1, "step_a should execute (dependency)"
        assert step_b.run_count == 1, "step_b should execute (dependency)"
        assert step_c.run_count == 1, "step_c should execute (selected)"

        # Verify unrelated step did not execute
        assert step_d.run_count == 0, "step_d should not execute (not selected)"
    finally:
        gateway.close()


def test_export_args_dataclass(tmp_path: Path) -> None:
    """ExportArgs dataclass provides expected methods."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    args = ExportArgs(
        repo_root=repo_root,
        repo="test/repo",
        commit="abc123",
        db_path=tmp_path / "db.duckdb",
        build_dir=tmp_path / "build",
    )

    snapshot = args.snapshot_config()
    assert snapshot.repo == "test/repo"
    assert snapshot.commit == "abc123"

    tools = args.resolved_tools()
    assert tools is not None

    profiles = args.resolved_profiles()
    assert profiles.code is not None
    assert profiles.config is not None

    backend = args.resolved_graph_backend()
    assert backend is not None


def test_gateway_cache_operations(tmp_path: Path) -> None:
    """Gateway caching and cleanup works correctly."""
    # Clear any existing cache
    close_gateways()
    initial_stats = gateway_cache_stats()
    assert initial_stats["size"] == 0

    # Open a gateway through cache
    db_path = tmp_path / "db.duckdb"
    gateway = open_fresh_duckdb(db_path)
    gateway.close()

    # Clear cache
    close_gateways()
    final_stats = gateway_cache_stats()
    assert final_stats["size"] == 0


def test_run_export_docs_invokes_validator_before_export(tmp_path: Path) -> None:
    """Export docs validates registry before exporting datasets."""
    events: list[str] = []
    opened: list[StorageGateway] = []

    def validator(_gateway: StorageGateway) -> None:
        events.append("validator")

    def export_runner(
        *, gateway: StorageGateway, output_dir: Path, options: ExportOptions | None = None
    ) -> list[Path]:
        if options is None:
            pytest.fail("Expected options")
        options.validator(gateway)
        events.append(f"export:{output_dir}")
        return []

    def gateway_factory(_db_path: Path) -> StorageGateway:
        cfg = StorageConfig(
            db_path=_db_path,
            apply_schema=True,
            ensure_views=True,
            validate_schema=True,
        )
        gateway = open_gateway(cfg)
        opened.append(gateway)
        return gateway

    def create_views_fn(con: DuckDBConnection) -> None:
        events.append("views")
        create_all_views(con)

    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    hooks = ExportHooks(
        validator=validator,
        export_runner=export_runner,
        gateway_factory=gateway_factory,
        create_views=create_views_fn,
    )
    export_options = ExportOptions(
        export=ExportCallOptions(validate_exports=True, schemas=["public"])
    )
    try:
        run_export_docs(
            db_path=tmp_path / "db.duckdb",
            document_output_dir=output_dir,
            options=export_options,
            hooks=hooks,
        )
    finally:
        for gateway in opened:
            gateway.close()

    expected = ["views", "validator", f"export:{output_dir}"]
    assert events == expected, f"Unexpected event order: {events}"


def test_run_full_pipeline_preflight_only(tmp_path: Path) -> None:
    """Run full pipeline with no targets to ensure preflight completes."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / ".git").mkdir(parents=True, exist_ok=True)
    build_dir = repo_root / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    db_path = build_dir / "db" / "codeintel.duckdb"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    previous_env = {key: value for key, value in os.environ.items() if key.startswith("CODEINTEL_")}
    os.environ["CODEINTEL_SKIP_SCIP"] = "true"
    try:
        run_full_pipeline(
            args=ExportArgs(
                repo_root=repo_root,
                repo="demo/repo",
                commit="deadbeef",
                db_path=db_path,
                build_dir=build_dir,
            ),
            targets=[],
        )
    finally:
        for key in list(os.environ.keys()):
            if key.startswith("CODEINTEL_") and key not in previous_env:
                os.environ.pop(key, None)
        for key, value in previous_env.items():
            os.environ[key] = value


def test_cli_docs_export_with_validation(tmp_path: Path) -> None:
    """Export via the real CLI with validation enabled against a minimal seeded DB."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / ".git").mkdir(parents=True, exist_ok=True)
    build_dir = repo_root / "build"
    db_path = build_dir / "db" / "codeintel.duckdb"
    gateway = open_fresh_duckdb(db_path)
    seed_docs_export_minimal(gateway, repo="demo/repo", commit="deadbeef")
    gateway.close()

    document_output_dir = repo_root / "Document Output"
    argv = [
        "docs",
        "export",
        "--repo-root",
        str(repo_root),
        "--repo",
        "demo/repo",
        "--commit",
        "deadbeef",
        "--db-path",
        str(db_path),
        "--build-dir",
        str(build_dir),
        "--document-output-dir",
        str(document_output_dir),
        "--validate",
        "--schema",
        "function_profile",
    ]

    result = runner.invoke(app, argv)
    if result.exit_code != 0:
        pytest.fail(f"CLI docs export failed with exit code {result.exit_code}")

    manifest = document_output_dir / "index.json"
    if not manifest.exists():
        pytest.fail("Expected Document Output manifest from CLI export")


def test_history_timeseries_params_dataclass(tmp_path: Path) -> None:
    """HistoryTimeseriesParams dataclass has expected fields."""
    params = HistoryTimeseriesParams(
        repo_root=tmp_path / "repo",
        repo="test/repo",
        commits=("abc123", "def456"),
        history_db_dir=tmp_path / "history",
        db_path=tmp_path / "db.duckdb",
    )

    assert params.repo == "test/repo"
    assert params.commits == ("abc123", "def456")
    assert params.runner is None
