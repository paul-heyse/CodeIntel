"""Tests for the analytics recipe system."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from codeintel.analytics.recipes.dsl import RecipeBuilder, recipe
from codeintel.analytics.recipes.model import Recipe, RecipeOptions, RecipeScope
from codeintel.analytics.recipes.registry import RecipeRegistry
from codeintel.core.recipes import RecipeBuilder as CoreRecipeBuilder

FAST_RECIPES_COUNT = 2
MAX_DURATION_MS = 60_000


@pytest.fixture
def recipe_registry() -> RecipeRegistry:
    """Return a fresh recipe registry for each test.

    Returns
    -------
    RecipeRegistry
        Fresh registry instance.
    """
    return RecipeRegistry()


# -----------------------------------------------------------------------------
# Recipe
# -----------------------------------------------------------------------------


def test_create_minimal_recipe() -> None:
    """Create a minimal recipe with sensible defaults."""
    rec = Recipe(
        name="test",
        description="A test recipe",
        plugins=("plugin1", "plugin2"),
    )

    assert rec.name == "test"
    assert rec.description == "A test recipe"
    assert rec.plugins == ("plugin1", "plugin2")
    assert rec.tags == ()
    assert rec.options.fail_fast is True


def test_recipe_with_plugins_is_immutable() -> None:
    """with_plugins should return a new instance without mutating the original."""
    rec = Recipe(
        name="base",
        description="Base recipe",
        plugins=("plugin1",),
    )

    extended = rec.with_plugins("plugin2", "plugin3")

    assert extended.plugins == ("plugin1", "plugin2", "plugin3")
    assert rec.plugins == ("plugin1",)


def test_recipe_with_config_returns_new_instance() -> None:
    """with_config should add configuration without mutating the source recipe."""
    rec = Recipe(
        name="test",
        description="Test",
        plugins=("plugin1",),
    )

    configured = rec.with_config("plugin1", {"key": "value"})

    assert configured.default_configs.get("plugin1") == {"key": "value"}
    assert rec.default_configs == {}


@dataclass(frozen=True)
class FailFastCase:
    """Case for toggling fail_fast on a recipe."""

    initial: bool
    updated: bool


@pytest.mark.parametrize(
    "case",
    [
        FailFastCase(initial=True, updated=False),
        FailFastCase(initial=False, updated=True),
    ],
)
def test_recipe_with_options_returns_new_instance(case: FailFastCase) -> None:
    """with_options should update fail_fast in a copied recipe."""
    rec = Recipe(
        name="test",
        description="Test",
        plugins=(),
        options=RecipeOptions(fail_fast=case.initial),
    )

    updated = rec.with_options(fail_fast=case.updated)

    assert updated.options.fail_fast is case.updated
    assert rec.options.fail_fast is case.initial


# -----------------------------------------------------------------------------
# RecipeScope
# -----------------------------------------------------------------------------


def test_create_empty_scope() -> None:
    """RecipeScope defaults to an empty scope."""
    scope = RecipeScope()

    assert scope.paths == ()
    assert scope.modules == ()
    assert scope.time_window is None
    assert scope.labels == {}


def test_create_scope_with_paths_and_modules() -> None:
    """RecipeScope stores provided paths and modules."""
    scope = RecipeScope(
        paths=("src/", "tests/"),
        modules=("module1", "module2"),
    )

    assert scope.paths == ("src/", "tests/")
    assert scope.modules == ("module1", "module2")


# -----------------------------------------------------------------------------
# RecipeRegistry
# -----------------------------------------------------------------------------


def test_register_and_get_recipe(recipe_registry: RecipeRegistry) -> None:
    """Registering a recipe should make it retrievable by name."""
    rec = Recipe(
        name="test",
        description="Test",
        plugins=(),
    )

    recipe_registry.register(rec)

    assert recipe_registry.get("test") is rec


def test_register_duplicate_raises(recipe_registry: RecipeRegistry) -> None:
    """Registering a duplicate recipe should raise a ValueError."""
    rec1 = Recipe(name="test", description="", plugins=())
    rec2 = Recipe(name="test", description="", plugins=())

    recipe_registry.register(rec1)

    with pytest.raises(ValueError, match="Duplicate recipe"):
        recipe_registry.register(rec2)


def test_get_unknown_raises(recipe_registry: RecipeRegistry) -> None:
    """Getting an unknown recipe should raise KeyError."""
    with pytest.raises(KeyError, match="Unknown recipe"):
        recipe_registry.get("nonexistent")


def test_get_optional_unknown_returns_none(recipe_registry: RecipeRegistry) -> None:
    """get_optional should return None for unknown recipes."""
    assert recipe_registry.get_optional("unknown") is None


def test_list_by_tag_filters_recipes(recipe_registry: RecipeRegistry) -> None:
    """list_by_tag should only return recipes with the requested tag."""
    rec1 = Recipe(name="r1", description="", plugins=(), tags=("fast",))
    rec2 = Recipe(name="r2", description="", plugins=(), tags=("fast", "test"))
    rec3 = Recipe(name="r3", description="", plugins=(), tags=("slow",))

    recipe_registry.register(rec1)
    recipe_registry.register(rec2)
    recipe_registry.register(rec3)

    fast_recipes = recipe_registry.list_by_tag("fast")
    assert len(fast_recipes) == FAST_RECIPES_COUNT
    assert rec1 in fast_recipes
    assert rec2 in fast_recipes


def test_compose_recipes_deduplicates_plugins_and_merges_tags(
    recipe_registry: RecipeRegistry,
) -> None:
    """Compose should merge plugin lists and tags across recipes."""
    rec1 = Recipe(
        name="r1",
        description="",
        plugins=("p1", "p2"),
        tags=("tag1",),
    )
    rec2 = Recipe(
        name="r2",
        description="",
        plugins=("p2", "p3"),
        tags=("tag2",),
    )

    recipe_registry.register(rec1)
    recipe_registry.register(rec2)

    composed = recipe_registry.compose(
        name="composed",
        description="Composed recipe",
        recipes=["r1", "r2"],
    )

    assert composed.plugins == ("p1", "p2", "p3")
    assert set(composed.tags) == {"tag1", "tag2"}


# -----------------------------------------------------------------------------
# RecipeBuilder
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class RecipeBuilderCase:
    """Case for verifying builder output."""

    builder: CoreRecipeBuilder
    name: str
    plugins: tuple[str, ...]
    tags: tuple[str, ...]
    duration: int | None
    fail_fast: bool


@pytest.mark.parametrize(
    "case",
    [
        RecipeBuilderCase(
            builder=RecipeBuilder("test")
            .description("A test recipe")
            .add("plugin1")
            .add("plugin2"),
            name="test",
            plugins=("plugin1", "plugin2"),
            tags=(),
            duration=None,
            fail_fast=True,
        ),
        RecipeBuilderCase(
            builder=RecipeBuilder("test").description("Test").add_all("p1", "p2", "p3"),
            name="test",
            plugins=("p1", "p2", "p3"),
            tags=(),
            duration=None,
            fail_fast=True,
        ),
        RecipeBuilderCase(
            builder=RecipeBuilder("test")
            .description("Test")
            .add_all("p1", "p2", "p3")
            .remove("p2"),
            name="test",
            plugins=("p1", "p3"),
            tags=(),
            duration=None,
            fail_fast=True,
        ),
        RecipeBuilderCase(
            builder=(
                RecipeBuilder("test")
                .description("Test")
                .add("plugin1")
                .with_config("plugin1", {"key": "value"})
                .tag("fast", "audit")
                .fail_fast(value=False)
                .max_duration(ms=MAX_DURATION_MS)
            ),
            name="test",
            plugins=("plugin1",),
            tags=("fast", "audit"),
            duration=MAX_DURATION_MS,
            fail_fast=False,
        ),
    ],
)
def test_recipe_builder_variations(case: RecipeBuilderCase) -> None:
    """RecipeBuilder should construct recipes matching the configured options."""
    rec = case.builder.build()

    assert rec.name == case.name
    assert rec.plugins == case.plugins
    assert rec.tags == case.tags
    if case.duration is None:
        assert rec.options.max_duration_ms is None
    else:
        assert rec.options.max_duration_ms == case.duration
    assert rec.options.fail_fast is case.fail_fast


# -----------------------------------------------------------------------------
# recipe() helper
# -----------------------------------------------------------------------------


def test_recipe_function_creates_builder() -> None:
    """recipe() convenience function should return a RecipeBuilder."""
    builder = recipe("my_recipe")

    assert isinstance(builder, RecipeBuilder)

    rec = builder.description("My recipe").add("plugin1").build()

    assert rec.name == "my_recipe"
    assert rec.plugins == ("plugin1",)
