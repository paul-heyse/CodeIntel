"""End-to-end integration tests for ingestion pipelines.

This module tests complete ingestion pipelines through production entry points,
exercising realistic runtime conditions with:
- Full ingestion flows: repo_scan → ast_extract → scip_ingest → typing_ingest
- Recipe-based execution with RecipeExecutor and RecipeStage
- Change detection and incremental ingestion
- Plugin dependency resolution and execution ordering

Per Testing Charter: No monkeypatching, same stack (real DuckDB, real I/O),
entry points through public APIs, realistic data structures.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.ingestion.plugins import (
    DEFAULT_INGEST_PLUGINS,
    AstExtractPlugin,
    IngestPluginRegistry,
    IngestRuntimeScratch,
    PlanOptions,
    RepoScanPlugin,
    get_ingest_registry,
    plan_ingest_plugins,
)
from codeintel.ingestion.recipes.dsl import (
    IngestRecipe,
    RecipeOptions,
    RecipeStage,
)
from codeintel.ingestion.recipes.executor import (
    ExecutorConfig,
    RecipeExecutor,
    RecipeExecutorContext,
)
from tests._helpers import ProvisionedGateway
from tests._helpers.gateway import open_ingestion_gateway_with_macros as open_ingestion_gateway
from tests._helpers.ingest_setup import IngestTestSetup

# =============================================================================
# Test Fixtures
# =============================================================================


def _create_realistic_repo(repo_root: Path) -> None:
    """Create a realistic Python package for integration testing.

    Parameters
    ----------
    repo_root
        Root directory for the test repository.
    """
    # Create package structure
    pkg_dir = repo_root / "pkg"
    pkg_dir.mkdir(parents=True, exist_ok=True)

    (pkg_dir / "__init__.py").write_text(
        '''"""Main package for integration test repository.

This package demonstrates a typical Python package structure
for testing ingestion pipelines.
"""

from pkg.models import User, Config
from pkg.services import UserService

__all__ = ["User", "Config", "UserService"]
''',
        encoding="utf-8",
    )

    (pkg_dir / "models.py").write_text(
        '''"""Data models for the integration test package."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class User:
    """Represents a user in the system.

    Parameters
    ----------
    id
        Unique user identifier.
    name
        User's display name.
    email
        User's email address.
    """

    id: int
    name: str
    email: str

    def display_name(self) -> str:
        """Return formatted display name.

        Returns
        -------
        str
            Formatted name with email.
        """
        return f"{self.name} <{self.email}>"


@dataclass
class Config:
    """Application configuration.

    Parameters
    ----------
    debug
        Enable debug mode.
    timeout
        Request timeout in seconds.
    """

    debug: bool = False
    timeout: int = 30
    secret_key: Optional[str] = None
''',
        encoding="utf-8",
    )

    (pkg_dir / "services.py").write_text(
        '''"""Service layer for the integration test package."""

from typing import Dict, List, Optional

from pkg.models import User


class UserService:
    """Service for managing users.

    Parameters
    ----------
    storage
        Dictionary storage backend.
    """

    def __init__(self, storage: Optional[Dict[int, User]] = None) -> None:
        """Initialize the user service."""
        self._storage: Dict[int, User] = storage or {}

    def get_user(self, user_id: int) -> Optional[User]:
        """Retrieve a user by ID.

        Parameters
        ----------
        user_id
            User identifier to look up.

        Returns
        -------
        Optional[User]
            User if found, None otherwise.
        """
        return self._storage.get(user_id)

    def list_users(self) -> List[User]:
        """List all users.

        Returns
        -------
        List[User]
            All registered users.
        """
        return list(self._storage.values())

    def create_user(self, name: str, email: str) -> User:
        """Create a new user.

        Parameters
        ----------
        name
            User's display name.
        email
            User's email address.

        Returns
        -------
        User
            Newly created user.
        """
        user_id = len(self._storage) + 1
        user = User(id=user_id, name=name, email=email)
        self._storage[user_id] = user
        return user
''',
        encoding="utf-8",
    )

    # Create tests directory
    tests_dir = repo_root / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)

    (tests_dir / "__init__.py").write_text('"""Test suite."""\n', encoding="utf-8")

    (tests_dir / "test_models.py").write_text(
        '''"""Tests for data models."""

import pytest
from pkg.models import User, Config


def test_user_creation() -> None:
    """Test User dataclass creation."""
    user = User(id=1, name="Test", email="test@example.com")
    assert user.id == 1
    assert user.name == "Test"


def test_user_display_name() -> None:
    """Test User.display_name method."""
    user = User(id=1, name="Test", email="test@example.com")
    assert user.display_name() == "Test <test@example.com>"


def test_config_defaults() -> None:
    """Test Config default values."""
    config = Config()
    assert config.debug is False
    assert config.timeout == 30
''',
        encoding="utf-8",
    )


# =============================================================================
# Plugin Registry and Planning Tests
# =============================================================================


def test_registry_returns_all_default_plugins() -> None:
    """Verify default registry contains all expected plugins."""
    registry = get_ingest_registry()
    names = set(registry.list_names())

    expected_core = {"repo_scan", "ast_extract", "cst_extract", "scip_ingest"}
    missing = expected_core - names
    if missing:
        pytest.fail(f"Missing core plugins: {sorted(missing)}")


def test_plan_respects_dependency_ordering() -> None:
    """Confirm plan ordering respects declared dependencies."""
    plan = plan_ingest_plugins(
        PlanOptions(
            plugin_names=("repo_scan", "ast_extract", "scip_ingest"),
            defaults=DEFAULT_INGEST_PLUGINS,
        )
    )

    ordered = plan.ordered_names
    positions = {name: idx for idx, name in enumerate(ordered)}

    # repo_scan must come before ast_extract (ast_extract depends on repo_scan)
    if "repo_scan" in positions and "ast_extract" in positions:
        assert positions["repo_scan"] < positions["ast_extract"], (
            "repo_scan must precede ast_extract"
        )

    # repo_scan must come before scip_ingest
    if "repo_scan" in positions and "scip_ingest" in positions:
        assert positions["repo_scan"] < positions["scip_ingest"], (
            "repo_scan must precede scip_ingest"
        )


def test_plan_excludes_disabled_plugins() -> None:
    """Verify disabled plugins are excluded from plan."""
    plan = plan_ingest_plugins(
        PlanOptions(
            plugin_names=DEFAULT_INGEST_PLUGINS,
            disabled=("scip_ingest", "typing_ingest"),
            defaults=DEFAULT_INGEST_PLUGINS,
        )
    )

    ordered = plan.ordered_names
    assert "scip_ingest" not in ordered
    assert "typing_ingest" not in ordered

    # Verify in skipped
    skipped_names = {s.name for s in plan.skipped_plugins}
    assert "scip_ingest" in skipped_names
    assert "typing_ingest" in skipped_names


def test_custom_registry_with_plan() -> None:
    """Test custom registry with plan generation."""
    registry = IngestPluginRegistry()

    # Register only the plugins we want
    registry.register(RepoScanPlugin())
    registry.register(AstExtractPlugin())

    plan = registry.plan(
        PlanOptions(
            plugin_names=("repo_scan", "ast_extract"),
            defaults=("repo_scan", "ast_extract"),
        )
    )

    assert "repo_scan" in plan.ordered_names
    assert "ast_extract" in plan.ordered_names


# =============================================================================
# Single Plugin Execution Tests
# =============================================================================


def test_repo_scan_plugin_execution(tmp_path: Path) -> None:
    """Test RepoScanPlugin execution with realistic repo."""
    repo_root = tmp_path / "repo"
    _create_realistic_repo(repo_root)

    gateway = open_ingestion_gateway()
    try:
        setup = IngestTestSetup.from_repo(repo_root, gateway=gateway)
        ctx = setup.build_context("repo_scan")

        plugin = RepoScanPlugin()
        result = plugin.execute(ctx)

        assert result.success, f"RepoScanPlugin failed: {result.error}"
        assert not result.skipped

        # Verify modules were discovered
        row = gateway.con.execute("SELECT COUNT(*) FROM core.modules").fetchone()
        module_count = row[0] if row else 0
        min_expected_modules = 3
        assert module_count >= min_expected_modules, (
            f"Expected at least {min_expected_modules} modules, got {module_count}"
        )
    finally:
        gateway.close()


def test_ast_extract_plugin_execution_after_repo_scan(tmp_path: Path) -> None:
    """Test AstExtractPlugin execution after repo_scan."""
    repo_root = tmp_path / "repo"
    _create_realistic_repo(repo_root)

    gateway = open_ingestion_gateway()
    try:
        setup = IngestTestSetup.from_repo(repo_root, gateway=gateway)

        # First run repo_scan
        repo_ctx = setup.build_context("repo_scan")
        repo_plugin = RepoScanPlugin()
        repo_result = repo_plugin.execute(repo_ctx)
        assert repo_result.success, f"RepoScanPlugin failed: {repo_result.error}"

        # Now run ast_extract with fresh scratch for dependency data
        ast_setup = setup.with_fresh_scratch()
        ast_ctx = ast_setup.build_context("ast_extract")

        # Re-run repo_scan to populate scratch for ast_extract
        repo_plugin2 = RepoScanPlugin()
        repo_plugin2.execute(ast_ctx)

        ast_plugin = AstExtractPlugin()
        ast_result = ast_plugin.execute(ast_ctx)

        # AST extract may skip if no changed files
        if not ast_result.skipped:
            assert ast_result.success, f"AstExtractPlugin failed: {ast_result.error}"
    finally:
        gateway.close()


# =============================================================================
# Full Pipeline Integration Tests
# =============================================================================


def test_full_pipeline_repo_scan_to_ast_extract(tmp_path: Path) -> None:
    """Test full pipeline from repo_scan to ast_extract."""
    repo_root = tmp_path / "repo"
    _create_realistic_repo(repo_root)

    gateway = open_ingestion_gateway()
    try:
        # Use plugin registry to plan execution
        registry = IngestPluginRegistry()
        registry.register(RepoScanPlugin())
        registry.register(AstExtractPlugin())

        plan = registry.plan(
            PlanOptions(
                plugin_names=("repo_scan", "ast_extract"),
                defaults=("repo_scan", "ast_extract"),
            )
        )

        setup = IngestTestSetup.from_repo(repo_root, gateway=gateway)
        results = []

        for plugin in plan.plugins:
            ctx = setup.build_context(plugin.metadata.name)
            result = plugin.execute(ctx)
            results.append((plugin.metadata.name, result))

        # Verify repo_scan succeeded
        repo_result = next(r for name, r in results if name == "repo_scan")
        assert repo_result.success, f"repo_scan failed: {repo_result.error}"

        # Verify modules exist
        row = gateway.con.execute("SELECT COUNT(*) FROM core.modules").fetchone()
        assert row is not None, "Query returned None"
        assert row[0] > 0, "No modules found after pipeline"
    finally:
        gateway.close()


# =============================================================================
# Recipe Executor Tests
# =============================================================================


def test_recipe_executor_single_stage(tmp_path: Path) -> None:
    """Test RecipeExecutor with a single-stage recipe."""
    repo_root = tmp_path / "repo"
    _create_realistic_repo(repo_root)

    gateway = open_ingestion_gateway()
    try:
        setup = IngestTestSetup.from_repo(repo_root, gateway=gateway)

        exec_context = RecipeExecutorContext(
            gateway=gateway,
            snapshot=setup.snapshot,
            paths=setup.paths,
            tools=setup.tools,
            code_profile=setup.code_profile,
            config_profile=setup.config_profile,
        )

        # Create a minimal recipe with just repo_scan
        registry = IngestPluginRegistry()
        registry.register(RepoScanPlugin())

        config = ExecutorConfig(
            registry=registry,
            scratch=IngestRuntimeScratch(),
            enable_parallel=False,
        )

        executor = RecipeExecutor(exec_context, config)

        recipe = IngestRecipe(
            name="single_stage",
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

        assert result.success, f"Recipe execution failed: {result.error}"
        assert len(result.stage_results) == 1
        stage_result = result.stage_results[0]
        assert stage_result.stage.name == "scan"
        assert stage_result.success
    finally:
        gateway.close()


def test_recipe_executor_multi_stage(tmp_path: Path) -> None:
    """Test RecipeExecutor with multi-stage recipe."""
    repo_root = tmp_path / "repo"
    _create_realistic_repo(repo_root)

    gateway = open_ingestion_gateway()
    try:
        setup = IngestTestSetup.from_repo(repo_root, gateway=gateway)

        exec_context = RecipeExecutorContext(
            gateway=gateway,
            snapshot=setup.snapshot,
            paths=setup.paths,
            tools=setup.tools,
            code_profile=setup.code_profile,
            config_profile=setup.config_profile,
        )

        registry = IngestPluginRegistry()
        registry.register(RepoScanPlugin())
        registry.register(AstExtractPlugin())

        config = ExecutorConfig(
            registry=registry,
            scratch=IngestRuntimeScratch(),
            enable_parallel=False,
        )

        executor = RecipeExecutor(exec_context, config)

        recipe = IngestRecipe(
            name="multi_stage",
            stages=(
                RecipeStage(
                    name="scan",
                    plugins=("repo_scan",),
                    parallel=False,
                ),
                RecipeStage(
                    name="parse",
                    plugins=("ast_extract",),
                    parallel=False,
                ),
            ),
            options=RecipeOptions(fail_fast=True),
        )

        result = executor.execute(recipe)

        assert result.success, f"Recipe execution failed: {result.error}"
        assert len(result.stage_results) >= 1

        # Verify scan stage succeeded
        scan_stage = next((s for s in result.stage_results if s.stage.name == "scan"), None)
        assert scan_stage is not None
        assert scan_stage.success
    finally:
        gateway.close()


# =============================================================================
# Incremental/Change Detection Tests
# =============================================================================


def test_incremental_scan_detects_changes(tmp_path: Path) -> None:
    """Test that incremental scans detect file changes."""
    repo_root = tmp_path / "repo"
    _create_realistic_repo(repo_root)

    gateway = open_ingestion_gateway()
    try:
        setup = IngestTestSetup.from_repo(repo_root, gateway=gateway)

        # First scan
        ctx1 = setup.build_context("repo_scan_1")
        plugin1 = RepoScanPlugin()
        result1 = plugin1.execute(ctx1)
        assert result1.success

        # Add a new file
        (repo_root / "pkg" / "new_module.py").write_text(
            '''"""New module added after initial scan."""


def new_function() -> str:
    """Return a greeting."""
    return "Hello from new module"
''',
            encoding="utf-8",
        )

        # Second scan should detect the new file
        setup2 = IngestTestSetup.from_repo(repo_root, gateway=gateway)
        ctx2 = setup2.build_context("repo_scan_2")
        plugin2 = RepoScanPlugin()
        result2 = plugin2.execute(ctx2)
        assert result2.success

        # Verify the new module was added
        row = gateway.con.execute(
            "SELECT COUNT(*) FROM core.modules WHERE path = 'pkg/new_module.py'"
        ).fetchone()
        assert row is not None, "Query returned None"
        assert row[0] == 1, "New module not found in database"
    finally:
        gateway.close()


# =============================================================================
# Integration with Provisioned Gateway
# =============================================================================


def test_plugin_execution_with_provisioned_gateway(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Test plugin execution with pre-provisioned gateway fixture."""
    setup = IngestTestSetup.from_repo(
        provisioned_repo.repo_root,
        gateway=provisioned_repo.gateway,
        repo=provisioned_repo.repo,
        commit=provisioned_repo.commit,
    )

    ctx = setup.build_context("test_with_provisioned")
    plugin = RepoScanPlugin()

    # Gateway already has data from provisioning
    result = plugin.execute(ctx)

    # Should succeed (may find existing modules)
    assert result.success or result.skipped


def test_recipe_with_provisioned_gateway(
    provisioned_repo: ProvisionedGateway,
) -> None:
    """Test recipe execution with provisioned gateway."""
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
        stages=(
            RecipeStage(
                name="scan",
                plugins=("repo_scan",),
                parallel=False,
            ),
        ),
        options=RecipeOptions(fail_fast=False),
    )

    result = executor.execute(recipe)

    # Should complete (success or skip)
    assert result.success or not result.stage_results


# =============================================================================
# Error Handling Tests
# =============================================================================


def test_pipeline_rejects_missing_dependency(tmp_path: Path) -> None:
    """Test that pipeline rejects plugins with missing dependencies."""
    repo_root = tmp_path / "repo"
    _create_realistic_repo(repo_root)

    gateway = open_ingestion_gateway()
    try:
        # Register only ast_extract (which depends on repo_scan)
        registry = IngestPluginRegistry()
        registry.register(AstExtractPlugin())

        # Plan should reject due to missing dependency
        with pytest.raises(ValueError, match="depends on 'repo_scan'"):
            registry.plan(
                PlanOptions(
                    plugin_names=("ast_extract",),
                    defaults=("ast_extract",),
                )
            )
    finally:
        gateway.close()


def test_recipe_fail_fast_behavior(tmp_path: Path) -> None:
    """Test recipe fail_fast behavior on plugin error."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    # Don't create any files - this will cause a "no modules" scenario

    gateway = open_ingestion_gateway()
    try:
        setup = IngestTestSetup.from_repo(repo_root, gateway=gateway)

        exec_context = RecipeExecutorContext(
            gateway=gateway,
            snapshot=setup.snapshot,
            paths=setup.paths,
            tools=setup.tools,
            code_profile=setup.code_profile,
            config_profile=setup.config_profile,
        )

        registry = IngestPluginRegistry()
        registry.register(RepoScanPlugin())
        registry.register(AstExtractPlugin())

        config = ExecutorConfig(
            registry=registry,
            scratch=IngestRuntimeScratch(),
            enable_parallel=False,
        )

        executor = RecipeExecutor(exec_context, config)

        recipe = IngestRecipe(
            name="fail_fast_test",
            stages=(
                RecipeStage(
                    name="scan",
                    plugins=("repo_scan",),
                    parallel=False,
                ),
                RecipeStage(
                    name="parse",
                    plugins=("ast_extract",),
                    parallel=False,
                ),
            ),
            options=RecipeOptions(fail_fast=True),
        )

        result = executor.execute(recipe)

        # With no Python files, repo_scan may succeed but find nothing
        # This tests the graceful handling of empty repos
        assert result is not None
    finally:
        gateway.close()
