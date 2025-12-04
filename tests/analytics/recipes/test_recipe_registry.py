"""Tests for recipe registry.

This module tests:
- RecipeRegistry for managing analytics recipes
- Recipe registration, lookup, and discovery
- Tag-based filtering
- Recipe composition
"""

from __future__ import annotations

import pytest

from codeintel.analytics.recipes.model import Recipe
from codeintel.analytics.recipes.registry import (
    RecipeRegistry,
    get_recipe_registry,
    register_recipe,
    reset_recipe_registry,
)

# Test constants
EXPECTED_TWO_RECIPES = 2
EXPECTED_THREE_PLUGINS = 3
EXPECTED_FOUR_UNIQUE_PLUGINS = 4


@pytest.fixture
def empty_registry() -> RecipeRegistry:
    """Create an empty recipe registry.

    Returns
    -------
    RecipeRegistry
        A fresh empty registry.
    """
    return RecipeRegistry()


@pytest.fixture
def sample_recipe() -> Recipe:
    """Create a sample analytics recipe.

    Returns
    -------
    Recipe
        A sample recipe for testing.
    """
    return Recipe(
        name="test_recipe",
        description="A test recipe",
        plugins=("plugin.one", "plugin.two"),
        default_configs={"plugin.one": {"enabled": True}},
        tags=("testing", "sample"),
    )


@pytest.fixture
def another_recipe() -> Recipe:
    """Create another sample analytics recipe.

    Returns
    -------
    Recipe
        Another sample recipe for testing.
    """
    return Recipe(
        name="another_recipe",
        description="Another test recipe",
        plugins=("plugin.three",),
        default_configs={},
        tags=("testing", "other"),
    )


def test_registry_empty(empty_registry: RecipeRegistry) -> None:
    """Empty registry has no recipes."""
    assert empty_registry.list_all() == ()
    assert empty_registry.list_names() == ()
    assert empty_registry.list_tags() == ()


def test_registry_register(empty_registry: RecipeRegistry, sample_recipe: Recipe) -> None:
    """Register a recipe successfully."""
    empty_registry.register(sample_recipe)

    assert sample_recipe.name in empty_registry.list_names()
    assert len(empty_registry.list_all()) == 1


def test_registry_register_duplicate_raises(
    empty_registry: RecipeRegistry, sample_recipe: Recipe
) -> None:
    """Registering duplicate recipe raises ValueError."""
    empty_registry.register(sample_recipe)

    with pytest.raises(ValueError, match="Duplicate recipe"):
        empty_registry.register(sample_recipe)


def test_registry_get(empty_registry: RecipeRegistry, sample_recipe: Recipe) -> None:
    """Get a recipe by name."""
    empty_registry.register(sample_recipe)

    result = empty_registry.get(sample_recipe.name)

    assert result is sample_recipe


def test_registry_get_not_found(empty_registry: RecipeRegistry) -> None:
    """Get unknown recipe raises KeyError."""
    with pytest.raises(KeyError, match="Unknown recipe"):
        empty_registry.get("nonexistent")


def test_registry_get_optional(
    empty_registry: RecipeRegistry, sample_recipe: Recipe
) -> None:
    """Get optional returns recipe when found."""
    empty_registry.register(sample_recipe)

    result = empty_registry.get_optional(sample_recipe.name)

    assert result is sample_recipe


def test_registry_get_optional_not_found(empty_registry: RecipeRegistry) -> None:
    """Get optional returns None when not found."""
    result = empty_registry.get_optional("nonexistent")

    assert result is None


def test_registry_unregister(
    empty_registry: RecipeRegistry, sample_recipe: Recipe
) -> None:
    """Unregister removes recipe from registry."""
    empty_registry.register(sample_recipe)
    assert sample_recipe.name in empty_registry.list_names()

    empty_registry.unregister(sample_recipe.name)

    assert sample_recipe.name not in empty_registry.list_names()


def test_registry_unregister_nonexistent(empty_registry: RecipeRegistry) -> None:
    """Unregister nonexistent recipe does nothing."""
    # Should not raise
    empty_registry.unregister("nonexistent")


def test_registry_unregister_removes_from_tags(
    empty_registry: RecipeRegistry, sample_recipe: Recipe
) -> None:
    """Unregister removes recipe from tag index."""
    empty_registry.register(sample_recipe)
    assert len(empty_registry.list_by_tag("testing")) == 1

    empty_registry.unregister(sample_recipe.name)

    assert len(empty_registry.list_by_tag("testing")) == 0


def test_registry_list_all(
    empty_registry: RecipeRegistry,
    sample_recipe: Recipe,
    another_recipe: Recipe,
) -> None:
    """List all returns all registered recipes."""
    empty_registry.register(sample_recipe)
    empty_registry.register(another_recipe)

    result = empty_registry.list_all()

    assert len(result) == EXPECTED_TWO_RECIPES
    assert sample_recipe in result
    assert another_recipe in result


def test_registry_list_names(
    empty_registry: RecipeRegistry,
    sample_recipe: Recipe,
    another_recipe: Recipe,
) -> None:
    """List names returns all recipe names."""
    empty_registry.register(sample_recipe)
    empty_registry.register(another_recipe)

    result = empty_registry.list_names()

    assert sample_recipe.name in result
    assert another_recipe.name in result


def test_registry_list_tags(
    empty_registry: RecipeRegistry,
    sample_recipe: Recipe,
    another_recipe: Recipe,
) -> None:
    """List tags returns all unique tags."""
    empty_registry.register(sample_recipe)
    empty_registry.register(another_recipe)

    result = empty_registry.list_tags()

    assert "testing" in result
    assert "sample" in result
    assert "other" in result


def test_registry_list_by_tag(
    empty_registry: RecipeRegistry,
    sample_recipe: Recipe,
    another_recipe: Recipe,
) -> None:
    """List by tag returns recipes with specific tag."""
    empty_registry.register(sample_recipe)
    empty_registry.register(another_recipe)

    # Both have "testing" tag
    testing_recipes = empty_registry.list_by_tag("testing")
    assert len(testing_recipes) == EXPECTED_TWO_RECIPES

    # Only sample_recipe has "sample" tag
    sample_recipes = empty_registry.list_by_tag("sample")
    assert len(sample_recipes) == 1
    assert sample_recipe in sample_recipes


def test_registry_list_by_tag_unknown(empty_registry: RecipeRegistry) -> None:
    """List by unknown tag returns empty tuple."""
    result = empty_registry.list_by_tag("unknown")

    assert result == ()


def test_registry_compose(
    empty_registry: RecipeRegistry,
    sample_recipe: Recipe,
    another_recipe: Recipe,
) -> None:
    """Compose creates new recipe from existing recipes."""
    empty_registry.register(sample_recipe)
    empty_registry.register(another_recipe)

    composed = empty_registry.compose(
        name="composed_recipe",
        description="A composed recipe",
        recipes=[sample_recipe.name, another_recipe.name],
    )

    assert composed.name == "composed_recipe"
    # Plugins deduplicated and in order
    assert "plugin.one" in composed.plugins
    assert "plugin.two" in composed.plugins
    assert "plugin.three" in composed.plugins
    # Configs merged
    assert "plugin.one" in composed.default_configs


def test_registry_compose_with_instances(
    empty_registry: RecipeRegistry,
    sample_recipe: Recipe,
    another_recipe: Recipe,
) -> None:
    """Compose accepts recipe instances directly."""
    # Not registering - using instances directly
    composed = empty_registry.compose(
        name="direct_composed",
        description="Composed from instances",
        recipes=[sample_recipe, another_recipe],
    )

    assert composed.name == "direct_composed"
    assert len(composed.plugins) == EXPECTED_THREE_PLUGINS  # 2 + 1 plugins


def test_registry_compose_deduplicates_plugins(
    empty_registry: RecipeRegistry,
) -> None:
    """Compose deduplicates plugins while preserving order."""
    recipe1 = Recipe(
        name="recipe1",
        description="Recipe 1",
        plugins=("plugin.a", "plugin.b", "plugin.c"),
        tags=(),
    )
    recipe2 = Recipe(
        name="recipe2",
        description="Recipe 2",
        plugins=("plugin.b", "plugin.c", "plugin.d"),  # b and c overlap
        tags=(),
    )

    composed = empty_registry.compose(
        name="dedup_test",
        description="Dedup test",
        recipes=[recipe1, recipe2],
    )

    # Should have 4 unique plugins in order
    assert len(composed.plugins) == EXPECTED_FOUR_UNIQUE_PLUGINS
    assert composed.plugins == ("plugin.a", "plugin.b", "plugin.c", "plugin.d")


def test_registry_compose_merges_tags(
    empty_registry: RecipeRegistry,
    sample_recipe: Recipe,
    another_recipe: Recipe,
) -> None:
    """Compose merges tags from all recipes."""
    composed = empty_registry.compose(
        name="tag_merge_test",
        description="Tag merge test",
        recipes=[sample_recipe, another_recipe],
    )

    # Tags from both recipes
    assert "testing" in composed.tags
    assert "sample" in composed.tags
    assert "other" in composed.tags


def test_global_registry_singleton() -> None:
    """Global registry is a singleton."""
    reset_recipe_registry()

    registry1 = get_recipe_registry()
    registry2 = get_recipe_registry()

    assert registry1 is registry2


def test_global_registry_reset() -> None:
    """Reset creates a new registry instance."""
    registry1 = get_recipe_registry()
    reset_recipe_registry()
    registry2 = get_recipe_registry()

    # After reset, should be a different instance
    assert registry1 is not registry2


def test_global_registry_has_builtins() -> None:
    """Global registry includes builtin recipes."""
    reset_recipe_registry()
    registry = get_recipe_registry()

    names = registry.list_names()

    # Should have builtin recipes
    assert "quick_audit" in names
    assert "full_analysis" in names


def test_register_recipe_function() -> None:
    """Register recipe function adds to global registry."""
    reset_recipe_registry()

    custom_recipe = Recipe(
        name="custom_global_recipe",
        description="Custom recipe for global registration",
        plugins=("plugin.custom",),
        tags=("custom",),
    )

    register_recipe(custom_recipe)

    registry = get_recipe_registry()
    assert registry.get_optional("custom_global_recipe") is custom_recipe
