"""Test recipe models from codeintel.core.recipes.model.

This module tests:
- BaseRecipeStage, BaseRecipeOptions, BaseRecipe
- RecipeStage with optional field
- RecipeOptions with all options
- Recipe construction and properties
- RecipeScope literal type
- RecipePluginRecord and RecipeExecutionReport
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from codeintel.core.recipes.model import (
    BaseRecipe,
    BaseRecipeOptions,
    BaseRecipeStage,
    Recipe,
    RecipeExecutionReport,
    RecipeOptions,
    RecipePluginRecord,
    RecipeScope,
    RecipeStage,
)

DEFAULT_MAX_PARALLEL = BaseRecipeOptions().max_parallel
CUSTOM_MAX_PARALLEL = 8
HIGH_MAX_PARALLEL = 16
DEFAULT_TIMEOUT_MS = 5000
DEFAULT_MAX_DURATION_MS = 60000
FAILURE_ATTEMPTS = 3

# =============================================================================
# BaseRecipeStage Tests
# =============================================================================


def test_base_recipe_stage_construction() -> None:
    """Verify BaseRecipeStage can be constructed with required fields."""
    stage = BaseRecipeStage(
        name="build",
        plugins=("plugin1", "plugin2"),
    )

    assert stage.name == "build"
    assert stage.plugins == ("plugin1", "plugin2")
    assert stage.parallel is False  # Default
    assert stage.fail_fast is True  # Default


def test_base_recipe_stage_all_fields() -> None:
    """Verify BaseRecipeStage accepts all fields."""
    stage = BaseRecipeStage(
        name="analysis",
        plugins=("p1", "p2"),
        parallel=True,
        fail_fast=False,
    )

    assert stage.parallel is True
    assert stage.fail_fast is False


def test_base_recipe_stage_is_frozen() -> None:
    """Verify BaseRecipeStage is immutable."""
    stage = BaseRecipeStage(name="test", plugins=())

    with pytest.raises(AttributeError):
        stage.name = "modified"  # type: ignore[misc]


# =============================================================================
# BaseRecipeOptions Tests
# =============================================================================


def test_base_recipe_options_defaults() -> None:
    """Verify BaseRecipeOptions has sensible defaults."""
    options = BaseRecipeOptions()

    assert options.dry_run is False
    assert options.max_parallel == DEFAULT_MAX_PARALLEL
    assert options.fail_fast is True


def test_base_recipe_options_custom() -> None:
    """Verify BaseRecipeOptions accepts custom values."""
    custom_max_parallel = CUSTOM_MAX_PARALLEL
    options = BaseRecipeOptions(
        dry_run=True,
        max_parallel=custom_max_parallel,
        fail_fast=False,
    )

    assert options.dry_run is True
    assert options.max_parallel == custom_max_parallel
    assert options.fail_fast is False


# =============================================================================
# BaseRecipe Tests
# =============================================================================


def test_base_recipe_construction() -> None:
    """Verify BaseRecipe can be constructed."""
    base_recipe = BaseRecipe(name="base_recipe")

    assert base_recipe.name == "base_recipe"
    assert not base_recipe.description  # Default empty
    assert base_recipe.version == "1.0"  # Default
    assert base_recipe.tags == ()  # Default


def test_base_recipe_all_fields() -> None:
    """Verify BaseRecipe accepts all fields."""
    base_recipe = BaseRecipe(
        name="my_recipe",
        description="A test recipe",
        version="2.0",
        tags=("tag1", "tag2"),
    )

    assert base_recipe.description == "A test recipe"
    assert base_recipe.version == "2.0"
    assert base_recipe.tags == ("tag1", "tag2")


# =============================================================================
# RecipeStage Tests
# =============================================================================


def test_recipe_stage_extends_base() -> None:
    """Verify RecipeStage extends BaseRecipeStage with optional field."""
    stage = RecipeStage(
        name="optional_stage",
        plugins=("p1",),
        optional=True,
    )

    assert stage.name == "optional_stage"
    assert stage.optional is True


def test_recipe_stage_optional_defaults_false() -> None:
    """Verify RecipeStage optional defaults to False."""
    stage = RecipeStage(name="test", plugins=())

    assert stage.optional is False


# =============================================================================
# RecipeOptions Tests
# =============================================================================


def test_recipe_options_defaults() -> None:
    """Verify RecipeOptions has sensible defaults."""
    options = RecipeOptions()

    assert options.dry_run is False
    assert options.skip_on_unchanged is False
    assert options.max_parallel == DEFAULT_MAX_PARALLEL
    assert options.timeout_ms is None
    assert options.fail_fast is True
    assert options.max_duration_ms is None


def test_recipe_options_all_fields() -> None:
    """Verify RecipeOptions accepts all fields."""
    custom_max_parallel = HIGH_MAX_PARALLEL
    timeout_ms = DEFAULT_TIMEOUT_MS
    max_duration_ms = DEFAULT_MAX_DURATION_MS
    options = RecipeOptions(
        dry_run=True,
        skip_on_unchanged=True,
        max_parallel=custom_max_parallel,
        timeout_ms=timeout_ms,
        fail_fast=False,
        max_duration_ms=max_duration_ms,
    )

    assert options.dry_run is True
    assert options.skip_on_unchanged is True
    assert options.max_parallel == custom_max_parallel
    assert options.timeout_ms == timeout_ms
    assert options.fail_fast is False
    assert options.max_duration_ms == max_duration_ms


# =============================================================================
# RecipeScope Tests
# =============================================================================


def test_recipe_scope_defaults() -> None:
    """Verify RecipeScope has empty defaults."""
    scope = RecipeScope()

    assert scope.paths == ()
    assert scope.modules == ()
    assert scope.time_window is None
    assert scope.labels == {}


def test_recipe_scope_with_paths() -> None:
    """Verify RecipeScope accepts paths."""
    scope = RecipeScope(paths=("src/", "tests/"))

    assert scope.paths == ("src/", "tests/")


def test_recipe_scope_with_time_window() -> None:
    """Verify RecipeScope accepts time window."""
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 12, 31, tzinfo=UTC)

    scope = RecipeScope(time_window=(start, end))

    assert scope.time_window is not None
    assert scope.time_window[0] == start
    assert scope.time_window[1] == end


def test_recipe_scope_with_labels() -> None:
    """Verify RecipeScope accepts labels."""
    scope = RecipeScope(labels={"env": "prod", "team": "core"})

    assert scope.labels["env"] == "prod"
    assert scope.labels["team"] == "core"


# =============================================================================
# Recipe Tests
# =============================================================================


def test_recipe_minimal() -> None:
    """Verify Recipe can be created with minimal fields."""
    recipe_obj = Recipe(name="minimal")

    assert recipe_obj.name == "minimal"
    assert recipe_obj.stages == ()
    assert recipe_obj.plugins == ()


def test_recipe_with_plugins() -> None:
    """Verify Recipe can be created with plugins."""
    recipe_obj = Recipe(
        name="with_plugins",
        plugins=("plugin1", "plugin2", "plugin3"),
    )

    assert recipe_obj.plugins == ("plugin1", "plugin2", "plugin3")


def test_recipe_with_stages() -> None:
    """Verify Recipe can be created with stages."""
    stages = (
        RecipeStage(name="build", plugins=("builder",)),
        RecipeStage(name="analyze", plugins=("analyzer",)),
    )

    recipe_obj = Recipe(name="staged", stages=stages)

    assert recipe_obj.is_staged is True
    assert len(recipe_obj.stages) == len(stages)


def test_recipe_all_plugins_from_stages() -> None:
    """Verify all_plugins collects plugins from stages."""
    stages = (
        RecipeStage(name="s1", plugins=("p1", "p2")),
        RecipeStage(name="s2", plugins=("p3",)),
    )

    recipe_obj = Recipe(name="test", stages=stages)

    assert recipe_obj.all_plugins == ("p1", "p2", "p3")


def test_recipe_all_plugins_from_both() -> None:
    """Verify all_plugins collects from stages and plugins."""
    stages = (RecipeStage(name="s1", plugins=("p1",)),)

    recipe_obj = Recipe(
        name="test",
        stages=stages,
        plugins=("p2", "p3"),
    )

    assert recipe_obj.all_plugins == ("p1", "p2", "p3")


def test_recipe_all_plugins_deduplicates() -> None:
    """Verify all_plugins removes duplicates."""
    stages = (RecipeStage(name="s1", plugins=("p1", "p2")),)

    recipe_obj = Recipe(
        name="test",
        stages=stages,
        plugins=("p2", "p3"),  # p2 is duplicate
    )

    assert recipe_obj.all_plugins == ("p1", "p2", "p3")


def test_recipe_is_staged_true() -> None:
    """Verify is_staged returns True when stages exist."""
    recipe_obj = Recipe(
        name="staged",
        stages=(RecipeStage(name="s", plugins=()),),
    )

    assert recipe_obj.is_staged is True


def test_recipe_is_staged_false() -> None:
    """Verify is_staged returns False when no stages."""
    recipe_obj = Recipe(name="flat", plugins=("p1",))

    assert recipe_obj.is_staged is False


def test_recipe_with_plugins_method() -> None:
    """Verify with_plugins returns new recipe with additional plugins."""
    original = Recipe(name="test", plugins=("p1",))

    extended = original.with_plugins("p2", "p3")

    assert original.plugins == ("p1",)  # Original unchanged
    assert extended.plugins == ("p1", "p2", "p3")


def test_recipe_with_config_method() -> None:
    """Verify with_config returns new recipe with config override."""
    original = Recipe(name="test")

    with_config = original.with_config("plugin1", {"key": "value"})

    assert original.default_configs == {}  # Original unchanged
    assert with_config.default_configs["plugin1"] == {"key": "value"}


def test_recipe_with_config_merges() -> None:
    """Verify with_config merges with existing config."""
    original = Recipe(
        name="test",
        default_configs={"plugin1": {"a": 1}},
    )

    updated = original.with_config("plugin1", {"b": 2})

    assert updated.default_configs["plugin1"] == {"a": 1, "b": 2}


def test_recipe_with_options_method() -> None:
    """Verify with_options returns new recipe with updated options."""
    original = Recipe(name="test")

    updated = original.with_options(dry_run=True, max_parallel=HIGH_MAX_PARALLEL)

    assert original.options.dry_run is False  # Original unchanged
    assert updated.options.dry_run is True
    assert updated.options.max_parallel == HIGH_MAX_PARALLEL


def test_recipe_with_options_preserves_unset() -> None:
    """Verify with_options preserves options not explicitly set."""
    original = Recipe(
        name="test",
        options=RecipeOptions(timeout_ms=DEFAULT_TIMEOUT_MS),
    )

    updated = original.with_options(dry_run=True)

    assert updated.options.timeout_ms == DEFAULT_TIMEOUT_MS
    assert updated.options.dry_run is True


# =============================================================================
# RecipePluginRecord Tests
# =============================================================================


def test_recipe_plugin_record_minimal() -> None:
    """Verify RecipePluginRecord can be created with required fields."""
    now = datetime.now(tz=UTC)
    record = RecipePluginRecord(
        plugin_name="test.plugin",
        status="succeeded",
        started_at=now,
        ended_at=now,
        duration_ms=100.0,
    )

    assert record.plugin_name == "test.plugin"
    assert record.status == "succeeded"
    assert record.attempts == 1  # Default
    assert record.error is None  # Default


def test_recipe_plugin_record_failed() -> None:
    """Verify RecipePluginRecord can record failure."""
    now = datetime.now(tz=UTC)
    record = RecipePluginRecord(
        plugin_name="failing.plugin",
        status="failed",
        started_at=now,
        ended_at=now,
        duration_ms=50.0,
        attempts=FAILURE_ATTEMPTS,
        error="Something went wrong",
    )

    assert record.status == "failed"
    assert record.attempts == FAILURE_ATTEMPTS
    assert record.error == "Something went wrong"


def test_recipe_plugin_record_with_row_counts() -> None:
    """Verify RecipePluginRecord can include row counts."""
    now = datetime.now(tz=UTC)
    record = RecipePluginRecord(
        plugin_name="producer",
        status="succeeded",
        started_at=now,
        ended_at=now,
        duration_ms=100.0,
        row_counts={"table1": 100, "table2": 50},
    )

    assert record.row_counts == {"table1": 100, "table2": 50}


# =============================================================================
# RecipeExecutionReport Tests
# =============================================================================


def test_recipe_execution_report_minimal() -> None:
    """Verify RecipeExecutionReport can be created."""
    now = datetime.now(tz=UTC)
    report = RecipeExecutionReport(
        recipe_name="test_recipe",
        run_id="run-123",
        repo="test/repo",
        commit="abc123",
        scope=RecipeScope(),
        started_at=now,
        ended_at=now,
        duration_ms=1000.0,
        status="succeeded",
        plugin_records=(),
    )

    assert report.recipe_name == "test_recipe"
    assert report.status == "succeeded"


def test_recipe_execution_report_succeeded_count() -> None:
    """Verify succeeded_count returns correct count."""
    now = datetime.now(tz=UTC)
    records = (
        RecipePluginRecord("p1", "succeeded", now, now, 100.0),
        RecipePluginRecord("p2", "failed", now, now, 50.0),
        RecipePluginRecord("p3", "succeeded", now, now, 75.0),
        RecipePluginRecord("p4", "skipped", now, now, 0.0),
    )

    report = RecipeExecutionReport(
        recipe_name="test",
        run_id="run-1",
        repo="repo",
        commit="abc",
        scope=RecipeScope(),
        started_at=now,
        ended_at=now,
        duration_ms=225.0,
        status="partial",
        plugin_records=records,
    )

    expected_succeeded = sum(record.status == "succeeded" for record in records)
    assert report.succeeded_count == expected_succeeded


def test_recipe_execution_report_failed_count() -> None:
    """Verify failed_count returns correct count."""
    now = datetime.now(tz=UTC)
    records = (
        RecipePluginRecord("p1", "succeeded", now, now, 100.0),
        RecipePluginRecord("p2", "failed", now, now, 50.0),
        RecipePluginRecord("p3", "failed", now, now, 50.0),
    )

    report = RecipeExecutionReport(
        recipe_name="test",
        run_id="run-1",
        repo="repo",
        commit="abc",
        scope=RecipeScope(),
        started_at=now,
        ended_at=now,
        duration_ms=200.0,
        status="failed",
        plugin_records=records,
    )

    expected_failed = sum(record.status == "failed" for record in records)
    assert report.failed_count == expected_failed


def test_recipe_execution_report_total_row_counts() -> None:
    """Verify total_row_counts aggregates across plugins."""
    now = datetime.now(tz=UTC)
    records = (
        RecipePluginRecord("p1", "succeeded", now, now, 100.0, row_counts={"t1": 10, "t2": 20}),
        RecipePluginRecord("p2", "succeeded", now, now, 50.0, row_counts={"t1": 5, "t3": 15}),
    )

    report = RecipeExecutionReport(
        recipe_name="test",
        run_id="run-1",
        repo="repo",
        commit="abc",
        scope=RecipeScope(),
        started_at=now,
        ended_at=now,
        duration_ms=150.0,
        status="succeeded",
        plugin_records=records,
    )

    totals = report.total_row_counts

    expected_totals = {
        "t1": sum(record.row_counts["t1"] for record in records if record.row_counts),
        "t2": records[0].row_counts["t2"],
        "t3": records[1].row_counts["t3"],
    }

    assert totals["t1"] == expected_totals["t1"]
    assert totals["t2"] == expected_totals["t2"]
    assert totals["t3"] == expected_totals["t3"]


def test_recipe_execution_report_with_skipped() -> None:
    """Verify RecipeExecutionReport tracks skipped plugins."""
    now = datetime.now(tz=UTC)
    report = RecipeExecutionReport(
        recipe_name="test",
        run_id="run-1",
        repo="repo",
        commit="abc",
        scope=RecipeScope(),
        started_at=now,
        ended_at=now,
        duration_ms=100.0,
        status="partial",
        plugin_records=(),
        skipped_plugins=("skipped1", "skipped2"),
    )

    assert report.skipped_plugins == ("skipped1", "skipped2")


def test_recipe_execution_report_with_error() -> None:
    """Verify RecipeExecutionReport can include overall error."""
    now = datetime.now(tz=UTC)
    report = RecipeExecutionReport(
        recipe_name="test",
        run_id="run-1",
        repo="repo",
        commit="abc",
        scope=RecipeScope(),
        started_at=now,
        ended_at=now,
        duration_ms=100.0,
        status="failed",
        plugin_records=(),
        error="Recipe execution aborted",
    )

    assert report.error == "Recipe execution aborted"
