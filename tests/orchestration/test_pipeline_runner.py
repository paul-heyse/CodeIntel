"""Tests for native pipeline runner functionality."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
from typer.testing import CliRunner

from codeintel.cli import app
from codeintel.config.models import ToolsConfig
from codeintel.core.plugins.types.protocol import PluginMetadata
from codeintel.pipeline import (
    FULL_PIPELINE,
    CliPipelineArgs,
    PipelinePlanOptions,
    run_pipeline,
)
from codeintel.pipeline.execution.context import PipelineContext
from codeintel.pipeline.execution.step_runner import (
    ExportHooks,
    HistoryTimeseriesParams,
    run_export_docs,
)
from codeintel.pipeline.export.export_jsonl import ExportCallOptions
from codeintel.pipeline.export.runner import ExportOptions
from codeintel.pipeline.steps.base import StepPhase, step_to_plugin_metadata
from codeintel.pipeline.steps.registry import build_registry
from codeintel.storage.gateway import (
    DuckDBConnection,
    StorageConfig,
    StorageGateway,
    open_gateway,
)
from codeintel.storage.gateway_cache import (
    close_gateways,
    gateway_cache_stats,
    get_gateway,
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

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata for registry compatibility."""
        return step_to_plugin_metadata(self.name, self.description, self.phase, self.deps)

    def run(self, ctx: PipelineContext) -> None:
        """Execute stub step."""
        self.run_count += 1
        self.last_ctx = ctx


def test_runner_module_imports() -> None:
    """Runner module can be imported without side effects."""
    if not callable(run_pipeline):
        pytest.fail("run_pipeline is not callable")


def test_step_registry_topological_order() -> None:
    """Step registry returns steps in topological order."""
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

    ordered = registry.topological_order(("step_a", "step_b", "step_c"))
    # step_a should come first, then step_b, then step_c
    assert ordered.index("step_a") < ordered.index("step_b")
    assert ordered.index("step_b") < ordered.index("step_c")


def test_step_registry_expand_with_deps() -> None:
    """Step registry expands step selection to include dependencies."""
    step_a = _StubStep("step_a")
    step_b = _StubStep("step_b", deps=["step_a"])
    step_c = _StubStep("step_c", deps=["step_b"])
    step_d = _StubStep("step_d")

    registry = build_registry(
        {
            "step_a": step_a,
            "step_b": step_b,
            "step_c": step_c,
            "step_d": step_d,
        }
    )

    # Selecting step_c should include step_a and step_b
    expanded = registry.expand_with_deps(("step_c",))
    assert "step_a" in expanded
    assert "step_b" in expanded
    assert "step_c" in expanded
    assert "step_d" not in expanded


def test_gateway_cache_operations(tmp_path: Path) -> None:
    """Gateway caching and cleanup works correctly."""
    # Clear any existing cache
    close_gateways()
    initial_stats = gateway_cache_stats()
    assert initial_stats["size"] == 0

    # Open a gateway through the cache
    db_path = tmp_path / "db.duckdb"
    config = StorageConfig.for_ingest(db_path)
    _ = get_gateway(config)

    # Stats should show one open
    stats = gateway_cache_stats()
    assert stats["opens"] == 1

    # Close gateways
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


# -----------------------------------------------------------------------------
# New Spec-Based Execution Tests
# -----------------------------------------------------------------------------


def test_cli_pipeline_args_to_plan_options(tmp_path: Path) -> None:
    """CliPipelineArgs converts correctly to PipelinePlanOptions."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    db_path = tmp_path / "db.duckdb"
    build_dir = tmp_path / "build"

    cli_args = CliPipelineArgs(
        repo_root=repo_root,
        repo="test/repo",
        commit="abc123",
        db_path=db_path,
        build_dir=build_dir,
    )

    # Verify snapshot_ref method
    snapshot = cli_args.snapshot_ref()
    assert snapshot.repo == "test/repo"
    assert snapshot.commit == "abc123"
    assert snapshot.repo_root == repo_root

    # Verify build_paths method
    paths = cli_args.build_paths()
    assert paths.db_path == db_path
    assert paths.build_dir == build_dir


def test_cli_pipeline_args_full_conversion(tmp_path: Path) -> None:
    """CliPipelineArgs converts to PipelinePlanOptions with gateway."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    db_path = tmp_path / "db.duckdb"
    build_dir = tmp_path / "build"

    cli_args = CliPipelineArgs(
        repo_root=repo_root,
        repo="test/repo",
        commit="abc123",
        db_path=db_path,
        build_dir=build_dir,
    )

    tools = ToolsConfig.default()
    gateway = open_fresh_duckdb(db_path)

    try:
        options = cli_args.to_plan_options(gateway, tools)
        assert isinstance(options, PipelinePlanOptions)
        assert options.snapshot.repo == "test/repo"
        assert options.snapshot.commit == "abc123"
        assert options.gateway is gateway
        assert options.tools is tools
        assert options.trigger == "cli"
    finally:
        gateway.close()


def test_gateway_cache_reuse(tmp_path: Path) -> None:
    """Gateway cache reuses existing gateways."""
    close_gateways()

    db_path = tmp_path / "db.duckdb"
    config = StorageConfig.for_ingest(db_path)

    # First access creates gateway
    gateway1 = get_gateway(config)
    stats1 = gateway_cache_stats()
    assert stats1["opens"] == 1
    assert stats1["hits"] == 0

    # Second access returns same gateway
    gateway2 = get_gateway(config)
    stats2 = gateway_cache_stats()
    assert stats2["opens"] == 1
    assert stats2["hits"] == 1

    # Should be same object
    assert gateway1 is gateway2

    close_gateways()


FULL_PIPELINE_STAGE_COUNT = 3


def test_full_pipeline_spec_has_all_stages() -> None:
    """FULL_PIPELINE spec includes all expected stages."""
    assert FULL_PIPELINE.id == "full"
    assert len(FULL_PIPELINE.stages) == FULL_PIPELINE_STAGE_COUNT

    modules = {stage.module for stage in FULL_PIPELINE.stages}
    assert "ingestion" in modules
    assert "graphs" in modules
    assert "analytics" in modules
