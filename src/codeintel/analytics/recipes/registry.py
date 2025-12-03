"""Recipe registry for managing analytics recipes.

This module provides a centralized registry for analytics recipes,
enabling recipe lookup, discovery, and management.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from codeintel.analytics.recipes.builtins import (
    COVERAGE_FOCUS,
    FULL_ANALYSIS,
    GRAPH_METRICS,
    QUICK_AUDIT,
    RISK_ANALYSIS,
    TEST_ANALYSIS,
)
from codeintel.analytics.recipes.model import AnalyticsRecipe
from codeintel.core.singleton import SingletonHolder

log = logging.getLogger(__name__)


class RecipeRegistry:
    """Central registry for analytics recipes.

    The registry provides:
    - Recipe registration and lookup
    - Tag-based discovery
    - Recipe composition helpers
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._recipes: dict[str, AnalyticsRecipe] = {}
        self._by_tag: dict[str, set[str]] = {}

    def register(self, recipe: AnalyticsRecipe) -> None:
        """Register a recipe.

        Parameters
        ----------
        recipe
            Recipe to register.

        Raises
        ------
        ValueError
            If a recipe with the same name is already registered.
        """
        if recipe.name in self._recipes:
            message = f"Duplicate recipe name: {recipe.name}"
            raise ValueError(message)

        self._recipes[recipe.name] = recipe

        for tag in recipe.tags:
            self._by_tag.setdefault(tag, set()).add(recipe.name)

        log.debug("Registered recipe %s (plugins=%d)", recipe.name, len(recipe.plugins))

    def unregister(self, name: str) -> None:
        """Remove a recipe from the registry.

        Parameters
        ----------
        name
            Recipe name to remove.
        """
        recipe = self._recipes.pop(name, None)
        if recipe is None:
            return

        for tag in recipe.tags:
            if tag in self._by_tag:
                self._by_tag[tag].discard(name)

    def get(self, name: str) -> AnalyticsRecipe:
        """Return a recipe by name.

        Parameters
        ----------
        name
            Recipe name to look up.

        Returns
        -------
        AnalyticsRecipe
            The registered recipe.

        Raises
        ------
        KeyError
            If no recipe is registered with the given name.
        """
        if name not in self._recipes:
            message = f"Unknown recipe: {name}"
            raise KeyError(message)
        return self._recipes[name]

    def get_optional(self, name: str) -> AnalyticsRecipe | None:
        """Return a recipe by name if it exists.

        Parameters
        ----------
        name
            Recipe name to look up.

        Returns
        -------
        AnalyticsRecipe | None
            The recipe or None if not found.
        """
        return self._recipes.get(name)

    def list_all(self) -> tuple[AnalyticsRecipe, ...]:
        """Return all registered recipes.

        Returns
        -------
        tuple[AnalyticsRecipe, ...]
            All registered recipes.
        """
        return tuple(self._recipes.values())

    def list_by_tag(self, tag: str) -> tuple[AnalyticsRecipe, ...]:
        """Return recipes with a specific tag.

        Parameters
        ----------
        tag
            Tag to filter by.

        Returns
        -------
        tuple[AnalyticsRecipe, ...]
            Recipes with the tag.
        """
        names = self._by_tag.get(tag, set())
        return tuple(self._recipes[name] for name in names if name in self._recipes)

    def list_names(self) -> tuple[str, ...]:
        """Return all registered recipe names.

        Returns
        -------
        tuple[str, ...]
            Recipe names.
        """
        return tuple(self._recipes.keys())

    def list_tags(self) -> tuple[str, ...]:
        """Return all known tags.

        Returns
        -------
        tuple[str, ...]
            All tags used by registered recipes.
        """
        return tuple(self._by_tag.keys())

    def compose(
        self,
        name: str,
        description: str,
        recipes: Sequence[str | AnalyticsRecipe],
    ) -> AnalyticsRecipe:
        """Create a new recipe by composing existing recipes.

        Plugins are deduplicated while preserving order.

        Parameters
        ----------
        name
            Name for the composed recipe.
        description
            Description for the composed recipe.
        recipes
            Recipe names or instances to compose.

        Returns
        -------
        AnalyticsRecipe
            New composed recipe.
        """
        seen: set[str] = set()
        plugins: list[str] = []
        configs: dict[str, dict[str, object]] = {}
        tags: set[str] = set()

        for recipe_ref in recipes:
            recipe = self.get(recipe_ref) if isinstance(recipe_ref, str) else recipe_ref
            for plugin in recipe.plugins:
                if plugin not in seen:
                    seen.add(plugin)
                    plugins.append(plugin)
            for plugin_name, config in recipe.default_configs.items():
                if plugin_name not in configs:
                    configs[plugin_name] = {}
                configs[plugin_name].update(config)
            tags.update(recipe.tags)

        return AnalyticsRecipe(
            name=name,
            description=description,
            plugins=tuple(plugins),
            default_configs=configs,
            tags=tuple(sorted(tags)),
        )


# Singleton holder for recipe registry
class _RecipeRegistryHolder(SingletonHolder["RecipeRegistry"]):
    """Thread-safe singleton holder for RecipeRegistry."""


def _create_recipe_registry() -> RecipeRegistry:
    """Create and initialize a new recipe registry with builtin recipes.

    Returns
    -------
    RecipeRegistry
        A new registry with all builtin recipes registered.
    """
    registry = RecipeRegistry()
    _register_builtin_recipes(registry)
    return registry


def get_recipe_registry() -> RecipeRegistry:
    """Return the global recipe registry.

    Returns
    -------
    RecipeRegistry
        The singleton registry instance.
    """
    return _RecipeRegistryHolder.get(_create_recipe_registry)


def reset_recipe_registry() -> None:
    """Reset the global recipe registry.

    Primarily useful for testing to ensure clean state between tests.
    """
    _RecipeRegistryHolder.reset()


def register_recipe(recipe: AnalyticsRecipe) -> None:
    """Register a recipe with the global registry.

    Parameters
    ----------
    recipe
        Recipe to register.
    """
    get_recipe_registry().register(recipe)


def _register_builtin_recipes(registry: RecipeRegistry) -> None:
    """Register built-in recipes with the registry."""
    for recipe in (
        QUICK_AUDIT,
        FULL_ANALYSIS,
        COVERAGE_FOCUS,
        TEST_ANALYSIS,
        GRAPH_METRICS,
        RISK_ANALYSIS,
    ):
        registry.register(recipe)


__all__ = [
    "RecipeRegistry",
    "get_recipe_registry",
    "register_recipe",
]
