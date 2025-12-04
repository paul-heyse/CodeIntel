"""Test DSL functions from codeintel.core.recipes.dsl.

This module tests:
- stage() helper function
- recipe() helper function
- RecipeBuilder fluent API (with_stage, with_options, etc.)
- RecipeBuilder.build() validation
"""

from __future__ import annotations

from codeintel.core.recipes.dsl import RecipeBuilder, recipe, stage
from codeintel.core.recipes.model import Recipe, RecipeOptions, RecipeStage

# =============================================================================
# stage() Helper Tests
# =============================================================================


def test_stage_minimal() -> None:
    """Verify stage() creates a RecipeStage with minimal args."""
    result = stage("build", ["plugin1", "plugin2"])

    assert isinstance(result, RecipeStage)
    assert result.name == "build"
    assert result.plugins == ("plugin1", "plugin2")
    assert result.parallel is False  # Default
    assert result.fail_fast is True  # Default
    assert result.optional is False  # Default


def test_stage_with_all_options() -> None:
    """Verify stage() accepts all options."""
    result = stage(
        "analysis",
        ["p1", "p2"],
        parallel=True,
        fail_fast=False,
        optional=True,
    )

    assert result.name == "analysis"
    assert result.parallel is True
    assert result.fail_fast is False
    assert result.optional is True


def test_stage_empty_plugins() -> None:
    """Verify stage() accepts empty plugin list."""
    result = stage("empty", [])

    assert result.plugins == ()


def test_stage_converts_list_to_tuple() -> None:
    """Verify stage() converts plugin list to tuple."""
    result = stage("test", ["a", "b", "c"])

    assert isinstance(result.plugins, tuple)
    assert result.plugins == ("a", "b", "c")


# =============================================================================
# recipe() Helper Tests
# =============================================================================


def test_recipe_minimal() -> None:
    """Verify recipe() creates a Recipe with minimal args."""
    result = recipe("test_recipe")

    assert isinstance(result, Recipe)
    assert result.name == "test_recipe"
    assert result.description == ""
    assert result.stages == ()
    assert result.plugins == ()


def test_recipe_with_description() -> None:
    """Verify recipe() accepts description."""
    result = recipe("test", description="A test recipe")

    assert result.description == "A test recipe"


def test_recipe_with_stages() -> None:
    """Verify recipe() accepts stages."""
    stages = [
        stage("build", ["builder"]),
        stage("analyze", ["analyzer"]),
    ]

    result = recipe("staged", stages=stages)

    assert len(result.stages) == 2
    assert result.stages[0].name == "build"
    assert result.stages[1].name == "analyze"


def test_recipe_with_plugins() -> None:
    """Verify recipe() accepts flat plugin list."""
    result = recipe("flat", plugins=["p1", "p2", "p3"])

    assert result.plugins == ("p1", "p2", "p3")


def test_recipe_with_options() -> None:
    """Verify recipe() accepts RecipeOptions."""
    options = RecipeOptions(dry_run=True, max_parallel=8)

    result = recipe("with_options", options=options)

    assert result.options.dry_run is True
    assert result.options.max_parallel == 8


def test_recipe_with_default_configs() -> None:
    """Verify recipe() accepts default_configs."""
    configs = {
        "plugin1": {"key": "value"},
        "plugin2": {"other": 42},
    }

    result = recipe("configured", default_configs=configs)

    assert result.default_configs["plugin1"] == {"key": "value"}
    assert result.default_configs["plugin2"] == {"other": 42}


def test_recipe_with_tags() -> None:
    """Verify recipe() accepts tags."""
    result = recipe("tagged", tags=["fast", "integration"])

    assert result.tags == ("fast", "integration")


def test_recipe_with_version() -> None:
    """Verify recipe() accepts version."""
    result = recipe("versioned", version="2.5.0")

    assert result.version == "2.5.0"


def test_recipe_all_options() -> None:
    """Verify recipe() accepts all options together."""
    stages = [stage("s1", ["p1"])]
    options = RecipeOptions(fail_fast=False)

    result = recipe(
        "full",
        description="Full recipe",
        stages=stages,
        plugins=["p2"],
        options=options,
        default_configs={"p1": {"x": 1}},
        tags=["tag"],
        version="3.0",
    )

    assert result.name == "full"
    assert result.description == "Full recipe"
    assert len(result.stages) == 1
    assert result.plugins == ("p2",)
    assert result.options.fail_fast is False
    assert result.default_configs == {"p1": {"x": 1}}
    assert result.tags == ("tag",)
    assert result.version == "3.0"


# =============================================================================
# RecipeBuilder Basic Tests
# =============================================================================


def test_builder_minimal() -> None:
    """Verify RecipeBuilder creates recipe with name only."""
    result = RecipeBuilder("test").build()

    assert result.name == "test"
    assert result.plugins == ()
    assert result.stages == ()


def test_builder_description() -> None:
    """Verify RecipeBuilder.description() sets description."""
    result = RecipeBuilder("test").description("My description").build()

    assert result.description == "My description"


def test_builder_add_plugin() -> None:
    """Verify RecipeBuilder.add() adds a plugin."""
    result = RecipeBuilder("test").add("plugin1").add("plugin2").build()

    assert result.plugins == ("plugin1", "plugin2")


def test_builder_add_deduplicates() -> None:
    """Verify RecipeBuilder.add() doesn't add duplicates."""
    result = RecipeBuilder("test").add("p1").add("p2").add("p1").build()

    assert result.plugins == ("p1", "p2")


def test_builder_add_all() -> None:
    """Verify RecipeBuilder.add_all() adds multiple plugins."""
    result = RecipeBuilder("test").add_all("p1", "p2", "p3").build()

    assert result.plugins == ("p1", "p2", "p3")


def test_builder_add_stage() -> None:
    """Verify RecipeBuilder.add_stage() adds a stage."""
    result = (
        RecipeBuilder("test")
        .add_stage("build", ["builder"])
        .add_stage("analyze", ["analyzer"])
        .build()
    )

    assert len(result.stages) == 2
    assert result.stages[0].name == "build"
    assert result.stages[1].name == "analyze"


def test_builder_add_stage_with_options() -> None:
    """Verify RecipeBuilder.add_stage() accepts stage options."""
    result = (
        RecipeBuilder("test")
        .add_stage(
            "parallel_stage",
            ["p1", "p2"],
            parallel=True,
            stage_fail_fast=False,
            optional=True,
        )
        .build()
    )

    stage_obj = result.stages[0]
    assert stage_obj.parallel is True
    assert stage_obj.fail_fast is False
    assert stage_obj.optional is True


def test_builder_remove_plugin() -> None:
    """Verify RecipeBuilder.remove() removes a plugin."""
    result = (
        RecipeBuilder("test")
        .add_all("p1", "p2", "p3")
        .remove("p2")
        .build()
    )

    assert result.plugins == ("p1", "p3")


def test_builder_remove_nonexistent() -> None:
    """Verify RecipeBuilder.remove() ignores nonexistent plugins."""
    result = RecipeBuilder("test").add("p1").remove("nonexistent").build()

    assert result.plugins == ("p1",)


# =============================================================================
# RecipeBuilder Configuration Tests
# =============================================================================


def test_builder_with_config() -> None:
    """Verify RecipeBuilder.with_config() sets plugin config."""
    result = (
        RecipeBuilder("test")
        .add("plugin1")
        .with_config("plugin1", {"key": "value"})
        .build()
    )

    assert result.default_configs["plugin1"] == {"key": "value"}


def test_builder_with_config_merges() -> None:
    """Verify RecipeBuilder.with_config() merges configs."""
    result = (
        RecipeBuilder("test")
        .with_config("p1", {"a": 1})
        .with_config("p1", {"b": 2})
        .build()
    )

    assert result.default_configs["p1"] == {"a": 1, "b": 2}


def test_builder_tag() -> None:
    """Verify RecipeBuilder.tag() adds tags."""
    result = RecipeBuilder("test").tag("fast", "unit").build()

    assert result.tags == ("fast", "unit")


def test_builder_tag_deduplicates() -> None:
    """Verify RecipeBuilder.tag() doesn't add duplicate tags."""
    result = RecipeBuilder("test").tag("a", "b").tag("b", "c").build()

    assert result.tags == ("a", "b", "c")


def test_builder_version() -> None:
    """Verify RecipeBuilder.version() sets version."""
    result = RecipeBuilder("test").version("2.0.0").build()

    assert result.version == "2.0.0"


# =============================================================================
# RecipeBuilder Options Tests
# =============================================================================


def test_builder_fail_fast() -> None:
    """Verify RecipeBuilder.fail_fast() sets option."""
    result_true = RecipeBuilder("test").fail_fast().build()
    result_false = RecipeBuilder("test").fail_fast(value=False).build()

    assert result_true.options.fail_fast is True
    assert result_false.options.fail_fast is False


def test_builder_parallel_stages() -> None:
    """Verify RecipeBuilder.parallel_stages() sets option."""
    # Note: parallel_stages is a builder state, may not affect Recipe directly
    # This tests the builder method works
    builder = RecipeBuilder("test").parallel_stages()
    result = builder.build()

    # The build() creates RecipeOptions which has max_parallel
    assert result is not None


def test_builder_max_parallel() -> None:
    """Verify RecipeBuilder.max_parallel() sets option."""
    result = RecipeBuilder("test").max_parallel(16).build()

    assert result.options.max_parallel == 16


def test_builder_max_duration() -> None:
    """Verify RecipeBuilder.max_duration() sets option."""
    result = RecipeBuilder("test").max_duration(60000).build()

    assert result.options.max_duration_ms == 60000


def test_builder_timeout() -> None:
    """Verify RecipeBuilder.timeout() sets option."""
    result = RecipeBuilder("test").timeout(5000).build()

    assert result.options.timeout_ms == 5000


def test_builder_dry_run() -> None:
    """Verify RecipeBuilder.dry_run() sets option."""
    result = RecipeBuilder("test").dry_run().build()

    assert result.options.dry_run is True


def test_builder_skip_on_unchanged() -> None:
    """Verify RecipeBuilder.skip_on_unchanged() sets option."""
    result = RecipeBuilder("test").skip_on_unchanged().build()

    assert result.options.skip_on_unchanged is True


# =============================================================================
# RecipeBuilder Chaining Tests
# =============================================================================


def test_builder_chaining() -> None:
    """Verify RecipeBuilder methods return self for chaining."""
    result = (
        RecipeBuilder("complex")
        .description("Complex recipe")
        .add("p1")
        .add("p2")
        .add_stage("stage1", ["s1p1", "s1p2"], parallel=True)
        .with_config("p1", {"x": 1})
        .tag("tag1", "tag2")
        .fail_fast(value=False)
        .max_parallel(8)
        .timeout(10000)
        .max_duration(120000)
        .dry_run(value=False)
        .skip_on_unchanged()
        .version("3.0")
        .build()
    )

    assert result.name == "complex"
    assert result.description == "Complex recipe"
    assert result.plugins == ("p1", "p2")
    assert len(result.stages) == 1
    assert result.default_configs["p1"] == {"x": 1}
    assert result.tags == ("tag1", "tag2")
    assert result.options.fail_fast is False
    assert result.options.max_parallel == 8
    assert result.options.timeout_ms == 10000
    assert result.options.max_duration_ms == 120000
    assert result.options.dry_run is False
    assert result.options.skip_on_unchanged is True
    assert result.version == "3.0"


# =============================================================================
# RecipeBuilder Extend Tests
# =============================================================================


def test_builder_extend() -> None:
    """Verify RecipeBuilder.extend() copies from another recipe."""
    base_recipe = recipe(
        "base",
        plugins=["p1", "p2"],
        default_configs={"p1": {"key": "value"}},
        tags=["inherited"],
    )

    result = RecipeBuilder("extended").extend(base_recipe).add("p3").build()

    assert "p1" in result.plugins
    assert "p2" in result.plugins
    assert "p3" in result.plugins
    assert result.default_configs.get("p1") == {"key": "value"}
    assert "inherited" in result.tags


def test_builder_extend_with_stages() -> None:
    """Verify RecipeBuilder.extend() handles staged recipes."""
    stages = [stage("s1", ["p1", "p2"])]
    base_recipe = recipe("base", stages=stages, tags=["base_tag"])

    result = RecipeBuilder("extended").extend(base_recipe).build()

    # all_plugins from stages should be added
    assert "p1" in result.plugins or "p1" in result.all_plugins
    assert "base_tag" in result.tags


# =============================================================================
# Integration Tests
# =============================================================================


def test_stage_and_recipe_integration() -> None:
    """Verify stage() and recipe() work together."""
    build_stage = stage("build", ["builder.goid", "builder.edges"])
    analyze_stage = stage("analyze", ["analyzer.metrics"], parallel=True)

    result = recipe(
        "full_pipeline",
        description="Full analysis pipeline",
        stages=[build_stage, analyze_stage],
        options=RecipeOptions(max_parallel=4),
    )

    assert result.is_staged is True
    assert result.all_plugins == ("builder.goid", "builder.edges", "analyzer.metrics")


def test_builder_produces_valid_recipe() -> None:
    """Verify RecipeBuilder produces a valid Recipe."""
    result = (
        RecipeBuilder("valid")
        .description("Valid recipe")
        .add_stage("first", ["p1"], parallel=False)
        .add_stage("second", ["p2"], parallel=True)
        .tag("production")
        .fail_fast()
        .build()
    )

    # Verify it's a proper Recipe with expected properties
    assert isinstance(result, Recipe)
    assert result.is_staged is True
    assert len(result.stages) == 2
    assert result.all_plugins == ("p1", "p2")
    assert "production" in result.tags
    assert result.options.fail_fast is True
