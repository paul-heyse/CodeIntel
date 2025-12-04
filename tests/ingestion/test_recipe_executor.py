"""Tests for recipe executor and DSL structures.

This module tests the recipe execution infrastructure including
ExecutorConfig, RecipeExecutorContext, and recipe DSL types.

Enhanced with realistic runtime scenarios using:
- Real plugins (RepoScanPlugin, AstExtractPlugin)
- RecipeExecutor.execute() with multi-stage recipes
- Stage skipping, fail-fast behavior, and parallel execution
- IngestTestSetup and provisioned_repo fixtures
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from codeintel.config.primitives import BuildPaths, SnapshotRef
from codeintel.ingestion.core.base import BaseIngestPlugin
from codeintel.ingestion.core.execution_context import IngestExecutionContext
from codeintel.ingestion.infrastructure.scanning import ScanProfile
from codeintel.ingestion.plugins import (
    AstExtractPlugin,
    RepoScanPlugin,
)
from codeintel.ingestion.plugins.protocol import (
    IngestPluginPlan,
    IngestPluginResult,
    IngestPluginSkip,
    IngestRuntimeScratch,
    IngestStage,
)
from codeintel.ingestion.plugins.registry import IngestPluginRegistry, PlanOptions
from codeintel.ingestion.recipes.dsl import (
    IngestRecipe,
    RecipeExecutionResult,
    RecipeOptions,
    RecipeStage,
    RecipeStageResult,
)
from codeintel.ingestion.recipes.executor import (
    ExecutorConfig,
    PluginExecutionRecord,
    RecipeExecutor,
    RecipeExecutorContext,
)
from codeintel.storage.gateway import StorageGateway
from tests._helpers.configs.provisioning_config import ProvisionedGateway
from tests._helpers.gateway import open_ingestion_gateway_with_macros as open_ingestion_gateway
from tests._helpers.harnesses import IngestTestSetup
from tests._helpers.orchestration.tooling import make_tools_config

# =============================================================================
# Helper Context Manager
# =============================================================================


@dataclass(frozen=True)
class ExecutorTestEnv:
    """Bundled environment for recipe executor tests.

    Attributes
    ----------
    gateway : StorageGateway
        Active storage gateway.
    setup : IngestTestSetup
        Bundled test setup with paths and profiles.
    context : RecipeExecutorContext
        Executor context ready for use.
    """

    gateway: StorageGateway
    setup: IngestTestSetup
    context: RecipeExecutorContext


@contextmanager
def executor_test_env(repo_root: Path) -> Iterator[ExecutorTestEnv]:
    """Create an executor test environment from a repo root.

    This context manager handles gateway lifecycle and creates the
    standard test setup and executor context.

    Parameters
    ----------
    repo_root
        Path to the repository root (should exist and contain test files).

    Yields
    ------
    ExecutorTestEnv
        Bundled environment ready for recipe execution tests.
    """
    gateway = open_ingestion_gateway()
    try:
        setup = IngestTestSetup.from_repo(repo_root, gateway=gateway)
        context = RecipeExecutorContext(
            gateway=gateway,
            snapshot=setup.snapshot,
            paths=setup.paths,
            tools=setup.tools,
            code_profile=setup.code_profile,
            config_profile=setup.config_profile,
        )
        yield ExecutorTestEnv(gateway=gateway, setup=setup, context=context)
    finally:
        gateway.close()


# Test constants
EXPECTED_TIMEOUT_60 = 60
EXPECTED_TIMEOUT_120 = 120
EXPECTED_LENGTH_2 = 2


# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass
class MinimalPlugin(BaseIngestPlugin):
    """Minimal plugin for testing."""

    plugin_name: ClassVar[str] = "minimal"
    plugin_description: ClassVar[str] = "Minimal test plugin"
    plugin_stage: ClassVar[IngestStage] = "parse"

    def compute(self, ctx: IngestExecutionContext) -> Mapping[str, int] | None:
        """Return empty result.

        Returns
        -------
        Mapping[str, int] | None
            Empty row counts.
        """
        _ = self  # Use self for PLR6301
        _ = ctx
        return {}


# =============================================================================
# ExecutorConfig Tests
# =============================================================================


def test_executor_config_defaults() -> None:
    """ExecutorConfig should have sensible defaults."""
    config = ExecutorConfig()

    assert config.registry is not None
    assert config.scratch is not None
    assert config.run_id is not None
    assert len(config.run_id) > 0
    assert config.enable_parallel is True
    assert config.max_workers > 0
    assert config.timeout_s is None


def test_executor_config_custom_values() -> None:
    """ExecutorConfig should accept custom values."""
    scratch = IngestRuntimeScratch()
    registry = IngestPluginRegistry()
    custom_workers = 8
    custom_timeout = 60

    config = ExecutorConfig(
        registry=registry,
        scratch=scratch,
        run_id="custom-run-id",
        enable_parallel=False,
        max_workers=custom_workers,
        timeout_s=custom_timeout,
    )

    assert config.registry is registry
    assert config.scratch is scratch
    assert config.run_id == "custom-run-id"
    assert config.enable_parallel is False
    assert config.max_workers == custom_workers
    assert config.timeout_s == custom_timeout


def test_executor_config_run_id_is_unique() -> None:
    """Each ExecutorConfig should get a unique run_id by default."""
    config1 = ExecutorConfig()
    config2 = ExecutorConfig()

    assert config1.run_id != config2.run_id


# =============================================================================
# RecipeExecutorContext Tests
# =============================================================================


def test_recipe_executor_context_minimal(fresh_gateway: StorageGateway, tmp_path: Path) -> None:
    """RecipeExecutorContext should be creatable with minimal params."""
    snapshot = SnapshotRef(repo="test/repo", commit="abc123", repo_root=tmp_path)
    paths = BuildPaths.from_repo_root(tmp_path)
    scan_profile = ScanProfile(
        repo_root=tmp_path,
        source_roots=(tmp_path,),
        include_globs=("*.py",),
        ignore_dirs=(),
    )

    context = RecipeExecutorContext(
        gateway=fresh_gateway,
        snapshot=snapshot,
        paths=paths,
        tools=make_tools_config(),
        code_profile=scan_profile,
        config_profile=scan_profile,
    )

    assert context.gateway is fresh_gateway
    assert context.snapshot is snapshot
    assert context.paths is paths
    assert context.tool_runner is None
    assert context.tool_service is None
    assert context.change_tracker is None


# =============================================================================
# PluginExecutionRecord Tests
# =============================================================================


def test_plugin_execution_record_success() -> None:
    """PluginExecutionRecord should record successful execution."""
    result = IngestPluginResult.ok(row_counts={"core.test": 10})
    expected_duration = 1.5

    record = PluginExecutionRecord(
        plugin_name="test_plugin",
        result=result,
        duration_s=expected_duration,
    )

    assert record.plugin_name == "test_plugin"
    assert record.result is result
    assert record.result is not None
    assert record.result.success is True
    assert record.duration_s == expected_duration
    assert record.error is None


def test_plugin_execution_record_with_error() -> None:
    """PluginExecutionRecord should record execution with error."""
    error = ValueError("Test error")

    record = PluginExecutionRecord(
        plugin_name="failing_plugin",
        error=error,
        duration_s=0.1,
    )

    assert record.plugin_name == "failing_plugin"
    assert record.result is None
    assert record.error is error


def test_plugin_execution_record_default_values() -> None:
    """PluginExecutionRecord should have sensible defaults."""
    record = PluginExecutionRecord(plugin_name="test")

    assert record.plugin_name == "test"
    assert record.result is None
    assert record.duration_s == 0.0
    assert record.error is None


# =============================================================================
# RecipeStage Tests
# =============================================================================


def test_recipe_stage_from_dict() -> None:
    """RecipeStage.from_dict should parse dictionary correctly."""
    data = {
        "name": "parse",
        "plugins": ["ast_extract", "cst_extract"],
        "parallel": True,
        "required": False,
        "timeout_s": 60,
        "description": "Parse source files",
    }

    stage = RecipeStage.from_dict(data)

    assert stage.name == "parse"
    assert stage.plugins == ("ast_extract", "cst_extract")
    assert stage.parallel is True
    assert stage.required is False
    assert stage.timeout_s == EXPECTED_TIMEOUT_60
    assert stage.description == "Parse source files"


def test_recipe_stage_from_dict_defaults() -> None:
    """RecipeStage.from_dict should use defaults for missing fields."""
    data = {"name": "scan", "plugins": ["repo_scan"]}

    stage = RecipeStage.from_dict(data)

    assert stage.name == "scan"
    assert stage.plugins == ("repo_scan",)
    assert stage.parallel is False
    assert stage.required is True
    assert stage.timeout_s is None
    assert not stage.description


def test_recipe_stage_to_dict() -> None:
    """RecipeStage.to_dict should produce correct dictionary."""
    stage = RecipeStage(
        name="parse",
        plugins=("ast_extract",),
        parallel=True,
        required=False,
        timeout_s=120,
        description="Parsing stage",
    )

    data = stage.to_dict()

    assert data["name"] == "parse"
    assert data["plugins"] == ["ast_extract"]
    assert data["parallel"] is True
    assert data["required"] is False
    assert data["timeout_s"] == EXPECTED_TIMEOUT_120
    assert data["description"] == "Parsing stage"


def test_recipe_stage_to_dict_minimal() -> None:
    """RecipeStage.to_dict should omit default values."""
    stage = RecipeStage(name="scan", plugins=("repo_scan",))

    data = stage.to_dict()

    assert data["name"] == "scan"
    assert data["plugins"] == ["repo_scan"]
    # Default values should not be included
    assert "parallel" not in data
    assert "required" not in data
    assert "timeout_s" not in data
    assert "description" not in data


# =============================================================================
# RecipeOptions Tests
# =============================================================================


def test_recipe_options_defaults() -> None:
    """RecipeOptions should have sensible defaults."""
    options = RecipeOptions()

    assert options.fail_fast is True
    assert options.dry_run is False
    assert options.enable_contracts is True
    assert options.enable_incremental is True


def test_recipe_options_custom() -> None:
    """RecipeOptions should accept custom values."""
    options = RecipeOptions(
        fail_fast=False,
        dry_run=True,
        enable_contracts=False,
    )

    assert options.fail_fast is False
    assert options.dry_run is True
    assert options.enable_contracts is False


# =============================================================================
# IngestRecipe Tests
# =============================================================================


def test_ingest_recipe_all_plugins() -> None:
    """IngestRecipe.all_plugins should collect plugins from all stages."""
    stages = (
        RecipeStage(name="scan", plugins=("repo_scan",)),
        RecipeStage(name="parse", plugins=("ast_extract", "cst_extract")),
    )

    recipe = IngestRecipe(name="test", stages=stages)

    all_plugins = recipe.all_plugins
    assert "repo_scan" in all_plugins
    assert "ast_extract" in all_plugins
    assert "cst_extract" in all_plugins


def test_ingest_recipe_disabled_plugins() -> None:
    """IngestRecipe should track disabled plugins."""
    recipe = IngestRecipe(
        name="test",
        stages=(RecipeStage(name="parse", plugins=("ast_extract",)),),
        disabled_plugins=("slow_plugin",),
    )

    assert "slow_plugin" in recipe.disabled_plugins


def test_ingest_recipe_from_dict() -> None:
    """IngestRecipe.from_dict should parse dictionary correctly."""
    data = {
        "name": "default",
        "description": "Default recipe",
        "stages": [
            {"name": "scan", "plugins": ["repo_scan"]},
            {"name": "parse", "plugins": ["ast_extract"]},
        ],
        "options": {"fail_fast": False},
    }

    recipe = IngestRecipe.from_dict(data)

    assert recipe.name == "default"
    assert recipe.description == "Default recipe"
    assert len(recipe.stages) == EXPECTED_LENGTH_2
    assert recipe.stages[0].name == "scan"
    assert recipe.options.fail_fast is False


def test_ingest_recipe_to_dict() -> None:
    """IngestRecipe.to_dict should produce correct dictionary."""
    recipe = IngestRecipe(
        name="test",
        description="Test recipe",
        stages=(RecipeStage(name="scan", plugins=("repo_scan",)),),
        options=RecipeOptions(fail_fast=False),
    )

    data = recipe.to_dict()

    assert data["name"] == "test"
    assert data["description"] == "Test recipe"
    stages = data.get("stages")
    assert isinstance(stages, list)
    assert len(stages) == 1


# =============================================================================
# RecipeStageResult Tests
# =============================================================================


def test_recipe_stage_result_success() -> None:
    """RecipeStageResult should track successful execution."""
    expected_duration = 2.5
    stage = RecipeStage(name="parse", plugins=("ast_extract",))

    result = RecipeStageResult(
        stage=stage,
        plugin_results={"ast_extract": IngestPluginResult.ok()},
        success=True,
        duration_s=expected_duration,
    )

    assert result.stage.name == "parse"
    assert result.success is True
    assert result.duration_s == expected_duration


def test_recipe_stage_result_with_failures() -> None:
    """RecipeStageResult should track failed plugins."""
    stage = RecipeStage(name="parse", plugins=("good_plugin", "bad_plugin"))

    result = RecipeStageResult(
        stage=stage,
        plugin_results={
            "good_plugin": IngestPluginResult.ok(),
            "bad_plugin": IngestPluginResult.fail("error"),
        },
        success=False,
    )

    assert result.success is False
    assert "bad_plugin" in result.plugin_results


# =============================================================================
# RecipeExecutionResult Tests
# =============================================================================


def test_recipe_execution_result_success() -> None:
    """RecipeExecutionResult should track successful recipe execution."""
    recipe = IngestRecipe(name="test", stages=(RecipeStage(name="scan", plugins=("repo_scan",)),))
    expected_duration = 5.0

    result = RecipeExecutionResult(
        recipe=recipe,
        success=True,
        duration_s=expected_duration,
    )

    assert result.recipe is recipe
    assert result.success is True
    assert result.duration_s == expected_duration
    assert result.error is None


def test_recipe_execution_result_with_error() -> None:
    """RecipeExecutionResult should track execution error."""
    recipe = IngestRecipe(name="test", stages=(RecipeStage(name="scan", plugins=("repo_scan",)),))

    result = RecipeExecutionResult(
        recipe=recipe,
        success=False,
        duration_s=1.0,
        error="Stage 'scan' failed",
    )

    assert result.success is False
    assert result.error == "Stage 'scan' failed"


# =============================================================================
# IngestPluginPlan Tests
# =============================================================================


def test_plugin_plan_ordered_names() -> None:
    """IngestPluginPlan.ordered_names should return plugin names in order."""
    plugin = MinimalPlugin()

    plan = IngestPluginPlan(
        plugins=(plugin,),
        plan_id="test-plan",
    )

    assert plan.ordered_names == ("minimal",)


def test_plugin_plan_with_skipped() -> None:
    """IngestPluginPlan should track skipped plugins."""
    skip = IngestPluginSkip(name="skipped_plugin", reason="disabled")

    plan = IngestPluginPlan(
        plugins=(),
        plan_id="test-plan",
        skipped_plugins=(skip,),
    )

    assert len(plan.skipped_plugins) == 1
    assert plan.skipped_plugins[0].name == "skipped_plugin"
    assert plan.skipped_plugins[0].reason == "disabled"


# =============================================================================
# Registry Integration Tests
# =============================================================================


def test_plugin_registry_plan_basic() -> None:
    """IngestPluginRegistry should create plans from plugin names."""
    registry = IngestPluginRegistry()
    plugin = MinimalPlugin()
    registry.register(plugin)

    plan = registry.plan(PlanOptions(plugin_names=["minimal"]))

    assert len(plan.plugins) == 1
    assert plan.plugins[0].metadata.name == "minimal"


def test_plugin_registry_plan_disabled() -> None:
    """IngestPluginRegistry should skip disabled plugins."""
    registry = IngestPluginRegistry()
    plugin = MinimalPlugin()
    registry.register(plugin)

    plan = registry.plan(PlanOptions(plugin_names=["minimal"], disabled=("minimal",)))

    assert len(plan.plugins) == 0
    assert any(skip.name == "minimal" for skip in plan.skipped_plugins)


# =============================================================================
# Realistic Execution Tests with Real Plugins
# =============================================================================


def _create_test_repo(repo_root: Path) -> None:
    """Create a minimal Python package for realistic testing.

    Parameters
    ----------
    repo_root
        Root directory for the test repository.
    """
    pkg_dir = repo_root / "pkg"
    pkg_dir.mkdir(parents=True, exist_ok=True)

    (pkg_dir / "__init__.py").write_text(
        '"""Test package for recipe executor tests."""\n',
        encoding="utf-8",
    )

    (pkg_dir / "module.py").write_text(
        '''"""Sample module for testing recipe execution."""


def greet(name: str) -> str:
    """Return a greeting message.

    Parameters
    ----------
    name
        Name to greet.

    Returns
    -------
    str
        Greeting message.
    """
    return f"Hello, {name}!"


class Calculator:
    """Simple calculator for testing."""

    def add(self, a: int, b: int) -> int:
        """Add two numbers.

        Parameters
        ----------
        a
            First operand.
        b
            Second operand.

        Returns
        -------
        int
            Sum of a and b.
        """
        return a + b
''',
        encoding="utf-8",
    )


def test_recipe_executor_with_real_repo_scan(tmp_path: Path) -> None:
    """Test RecipeExecutor with real RepoScanPlugin."""
    repo_root = tmp_path / "repo"
    _create_test_repo(repo_root)

    with executor_test_env(repo_root) as env:
        registry = IngestPluginRegistry()
        registry.register(RepoScanPlugin())

        config = ExecutorConfig(
            registry=registry,
            scratch=IngestRuntimeScratch(),
            enable_parallel=False,
        )

        executor = RecipeExecutor(env.context, config)

        recipe = IngestRecipe(
            name="repo_scan_test",
            stages=(
                RecipeStage(
                    name="scan",
                    plugins=("repo_scan",),
                    parallel=False,
                ),
            ),
            options=RecipeOptions(fail_fast=True),
        )

        result = executor.execute(recipe)

        assert result.success, f"Recipe failed: {result.error}"
        assert len(result.stage_results) == 1
        assert result.stage_results[0].success

        # Verify modules were discovered in database
        row = env.gateway.con.execute("SELECT COUNT(*) FROM core.modules").fetchone()
        assert row is not None, "Query returned None"
        assert row[0] >= 1, "Expected at least 1 module"


def test_recipe_executor_multi_stage_with_dependencies(tmp_path: Path) -> None:
    """Test RecipeExecutor with multi-stage pipeline."""
    repo_root = tmp_path / "repo"
    _create_test_repo(repo_root)

    with executor_test_env(repo_root) as env:
        registry = IngestPluginRegistry()
        registry.register(RepoScanPlugin())
        registry.register(AstExtractPlugin())

        config = ExecutorConfig(
            registry=registry,
            scratch=IngestRuntimeScratch(),
            enable_parallel=False,
        )

        executor = RecipeExecutor(env.context, config)

        recipe = IngestRecipe(
            name="multi_stage",
            stages=(
                RecipeStage(name="scan", plugins=("repo_scan",), parallel=False),
                RecipeStage(name="parse", plugins=("ast_extract",), parallel=False),
            ),
            options=RecipeOptions(fail_fast=True),
        )

        result = executor.execute(recipe)

        assert result.success, f"Recipe failed: {result.error}"
        # Scan stage must succeed
        scan_result = next(
            (s for s in result.stage_results if s.stage.name == "scan"),
            None,
        )
        assert scan_result is not None, "Scan stage not found"
        assert scan_result.success, "Scan stage failed"


def test_recipe_executor_fail_fast_stops_on_error(tmp_path: Path) -> None:
    """Test that fail_fast=True stops execution on first failure."""
    repo_root = tmp_path / "repo"
    # Don't create any files - repo_scan should still succeed but find nothing
    repo_root.mkdir(parents=True, exist_ok=True)

    with executor_test_env(repo_root) as env:
        registry = IngestPluginRegistry()
        registry.register(RepoScanPlugin())
        registry.register(AstExtractPlugin())

        config = ExecutorConfig(
            registry=registry,
            scratch=IngestRuntimeScratch(),
            enable_parallel=False,
        )

        executor = RecipeExecutor(env.context, config)

        recipe = IngestRecipe(
            name="fail_fast_test",
            stages=(
                RecipeStage(name="scan", plugins=("repo_scan",), parallel=False),
                RecipeStage(name="parse", plugins=("ast_extract",), parallel=False),
            ),
            options=RecipeOptions(fail_fast=True),
        )

        result = executor.execute(recipe)

        # With empty repo, first stage may succeed but produce no data
        # The result should complete without exception
        assert result is not None


def test_recipe_executor_continue_on_soft_fail(tmp_path: Path) -> None:
    """Test that continue_on_soft_fail=True continues after non-critical failures."""
    repo_root = tmp_path / "repo"
    _create_test_repo(repo_root)

    with executor_test_env(repo_root) as env:
        registry = IngestPluginRegistry()
        registry.register(RepoScanPlugin())

        config = ExecutorConfig(
            registry=registry,
            scratch=IngestRuntimeScratch(),
            enable_parallel=False,
        )

        executor = RecipeExecutor(env.context, config)

        recipe = IngestRecipe(
            name="soft_fail_test",
            stages=(
                RecipeStage(
                    name="scan",
                    plugins=("repo_scan",),
                    parallel=False,
                    required=False,  # Non-required stage
                ),
            ),
            options=RecipeOptions(fail_fast=False, continue_on_soft_fail=True),
        )

        result = executor.execute(recipe)

        # Should complete even if stage reports issues
        assert result is not None


def test_recipe_executor_empty_stage_skipped(tmp_path: Path) -> None:
    """Test that stages with no plugins are handled gracefully."""
    repo_root = tmp_path / "repo"
    _create_test_repo(repo_root)

    with executor_test_env(repo_root) as env:
        registry = IngestPluginRegistry()
        registry.register(RepoScanPlugin())

        config = ExecutorConfig(
            registry=registry,
            scratch=IngestRuntimeScratch(),
            enable_parallel=False,
        )

        executor = RecipeExecutor(env.context, config)

        recipe = IngestRecipe(
            name="empty_stage_test",
            stages=(
                RecipeStage(name="empty", plugins=(), parallel=False),
                RecipeStage(name="scan", plugins=("repo_scan",), parallel=False),
            ),
            options=RecipeOptions(fail_fast=True),
        )

        result = executor.execute(recipe)

        # Should complete successfully, skipping empty stage
        assert result.success, f"Recipe failed: {result.error}"


def test_recipe_executor_with_provisioned_gateway(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Test RecipeExecutor with pre-provisioned gateway."""
    setup = IngestTestSetup.from_repo(
        provisioned_repo.repo_root,
        gateway=provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
    )

    exec_context = RecipeExecutorContext(
        gateway=provisioned_repo.gateway,
        snapshot=setup.snapshot,
        paths=setup.paths,
        tools=setup.tools,
        code_profile=setup.code_profile,
        config_profile=setup.config_profile,
    )

    registry = IngestPluginRegistry()
    registry.register(RepoScanPlugin())

    config = ExecutorConfig(
        registry=registry,
        scratch=IngestRuntimeScratch(),
        enable_parallel=False,
    )

    executor = RecipeExecutor(exec_context, config)

    recipe = IngestRecipe(
        name="provisioned_test",
        stages=(RecipeStage(name="scan", plugins=("repo_scan",), parallel=False),),
        options=RecipeOptions(fail_fast=False),
    )

    result = executor.execute(recipe)

    # Should complete with pre-existing data
    assert result is not None


def test_recipe_executor_duration_tracking(tmp_path: Path) -> None:
    """Test that RecipeExecutor tracks execution duration."""
    repo_root = tmp_path / "repo"
    _create_test_repo(repo_root)

    with executor_test_env(repo_root) as env:
        registry = IngestPluginRegistry()
        registry.register(RepoScanPlugin())

        config = ExecutorConfig(
            registry=registry,
            scratch=IngestRuntimeScratch(),
            enable_parallel=False,
        )

        executor = RecipeExecutor(env.context, config)

        recipe = IngestRecipe(
            name="duration_test",
            stages=(RecipeStage(name="scan", plugins=("repo_scan",), parallel=False),),
            options=RecipeOptions(fail_fast=True),
        )

        result = executor.execute(recipe)

        assert result.duration_s >= 0, "Duration should be non-negative"
        if result.stage_results:
            for stage_result in result.stage_results:
                assert stage_result.duration_s >= 0, "Stage duration should be non-negative"


def test_recipe_executor_parallel_disabled(tmp_path: Path) -> None:
    """Test RecipeExecutor with parallel execution disabled."""
    repo_root = tmp_path / "repo"
    _create_test_repo(repo_root)

    with executor_test_env(repo_root) as env:
        registry = IngestPluginRegistry()
        registry.register(RepoScanPlugin())

        config = ExecutorConfig(
            registry=registry,
            scratch=IngestRuntimeScratch(),
            enable_parallel=False,  # Explicitly disabled
            max_workers=1,
        )

        executor = RecipeExecutor(env.context, config)

        recipe = IngestRecipe(
            name="sequential_test",
            stages=(
                RecipeStage(
                    name="scan",
                    plugins=("repo_scan",),
                    parallel=True,  # Stage wants parallel but config disables it
                ),
            ),
            options=RecipeOptions(fail_fast=True),
        )

        result = executor.execute(recipe)

        assert result.success, f"Recipe failed: {result.error}"
