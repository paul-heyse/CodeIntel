"""Fluent DSL for building recipes.

This module provides both a fluent builder pattern and simple helper
functions for constructing recipes with a clean, chainable API.
"""

from __future__ import annotations

from collections.abc import Mapping

from codeintel.core.recipes.model import Recipe, RecipeOptions, RecipeStage


def stage(
    name: str,
    plugins: list[str],
    *,
    parallel: bool = False,
    fail_fast: bool = True,
    optional: bool = False,
) -> RecipeStage:
    """Create a recipe stage.

    Parameters
    ----------
    name
        Stage identifier.
    plugins
        Plugin names to include in this stage.
    parallel
        Whether plugins can run in parallel within the stage.
    fail_fast
        Whether to abort stage on first failure.
    optional
        Whether the stage can be skipped.

    Returns
    -------
    RecipeStage
        Stage definition.
    """
    return RecipeStage(
        name=name,
        plugins=tuple(plugins),
        parallel=parallel,
        fail_fast=fail_fast,
        optional=optional,
    )


def recipe(
    name: str,
    *,
    description: str = "",
    stages: list[RecipeStage] | None = None,
    plugins: list[str] | None = None,
    options: RecipeOptions | None = None,
    default_configs: Mapping[str, Mapping[str, object]] | None = None,
    tags: list[str] | None = None,
    version: str = "1.0",
) -> Recipe:
    """Create a recipe.

    Parameters
    ----------
    name
        Recipe identifier.
    description
        Human-readable description.
    stages
        Ordered stages for stage-based execution.
    plugins
        Flat plugin list for simple recipes.
    options
        Global execution options.
    default_configs
        Configuration overrides by plugin name.
    tags
        Free-form tags.
    version
        Recipe version string.

    Returns
    -------
    Recipe
        Recipe definition.
    """
    return Recipe(
        name=name,
        description=description,
        stages=tuple(stages) if stages else (),
        plugins=tuple(plugins) if plugins else (),
        options=options or RecipeOptions(),
        default_configs=default_configs or {},
        tags=tuple(tags) if tags else (),
        version=version,
    )


class RecipeBuilder:
    """Fluent builder for constructing recipes.

    Provides a clean API for building recipes incrementally with chaining.

    Example
    -------
    >>> my_recipe = (
    ...     RecipeBuilder("custom_analysis")
    ...     .description("Custom analysis workflow")
    ...     .add("functions.metrics")
    ...     .add("hotspots.build")
    ...     .with_config("hotspots.build", {"max_commits": 500})
    ...     .tag("custom")
    ...     .fail_fast()
    ...     .build()
    ... )
    """

    def __init__(self, name: str) -> None:
        """Initialize a recipe builder.

        Parameters
        ----------
        name
            Name for the recipe being built.
        """
        self._name = name
        self._description = ""
        self._stages: list[RecipeStage] = []
        self._plugins: list[str] = []
        self._configs: dict[str, dict[str, object]] = {}
        self._tags: list[str] = []
        self._fail_fast = True
        self._parallel_stages = False
        self._max_parallel = 4
        self._max_duration_ms: int | None = None
        self._timeout_ms: int | None = None
        self._dry_run = False
        self._skip_on_unchanged = False
        self._version = "1.0"

    def description(self, desc: str) -> RecipeBuilder:
        """Set the recipe description.

        Parameters
        ----------
        desc
            Human-readable description.

        Returns
        -------
        RecipeBuilder
            Self for chaining.
        """
        self._description = desc
        return self

    def add(self, plugin_name: str) -> RecipeBuilder:
        """Add a plugin to the recipe.

        Parameters
        ----------
        plugin_name
            Name of the plugin to add.

        Returns
        -------
        RecipeBuilder
            Self for chaining.
        """
        if plugin_name not in self._plugins:
            self._plugins.append(plugin_name)
        return self

    def add_all(self, *plugin_names: str) -> RecipeBuilder:
        """Add multiple plugins to the recipe.

        Parameters
        ----------
        plugin_names
            Names of plugins to add.

        Returns
        -------
        RecipeBuilder
            Self for chaining.
        """
        for name in plugin_names:
            self.add(name)
        return self

    def add_stage(
        self,
        name: str,
        plugins: list[str],
        *,
        parallel: bool = False,
        stage_fail_fast: bool = True,
        optional: bool = False,
    ) -> RecipeBuilder:
        """Add a stage to the recipe.

        Parameters
        ----------
        name
            Stage identifier.
        plugins
            Plugin names for this stage.
        parallel
            Whether plugins can run in parallel.
        stage_fail_fast
            Whether to abort stage on first failure.
        optional
            Whether the stage can be skipped.

        Returns
        -------
        RecipeBuilder
            Self for chaining.
        """
        self._stages.append(
            RecipeStage(
                name=name,
                plugins=tuple(plugins),
                parallel=parallel,
                fail_fast=stage_fail_fast,
                optional=optional,
            )
        )
        return self

    def remove(self, plugin_name: str) -> RecipeBuilder:
        """Remove a plugin from the recipe.

        Parameters
        ----------
        plugin_name
            Name of the plugin to remove.

        Returns
        -------
        RecipeBuilder
            Self for chaining.
        """
        if plugin_name in self._plugins:
            self._plugins.remove(plugin_name)
        return self

    def with_config(
        self,
        plugin_name: str,
        config: Mapping[str, object],
    ) -> RecipeBuilder:
        """Set configuration for a specific plugin.

        Parameters
        ----------
        plugin_name
            Plugin to configure.
        config
            Configuration mapping.

        Returns
        -------
        RecipeBuilder
            Self for chaining.
        """
        if plugin_name not in self._configs:
            self._configs[plugin_name] = {}
        self._configs[plugin_name].update(config)
        return self

    def tag(self, *tags: str) -> RecipeBuilder:
        """Add tags to the recipe.

        Parameters
        ----------
        tags
            Tags to add.

        Returns
        -------
        RecipeBuilder
            Self for chaining.
        """
        for t in tags:
            if t not in self._tags:
                self._tags.append(t)
        return self

    def fail_fast(self, *, value: bool = True) -> RecipeBuilder:
        """Set the fail_fast behavior.

        Parameters
        ----------
        value
            Whether to stop on first failure.

        Returns
        -------
        RecipeBuilder
            Self for chaining.
        """
        self._fail_fast = value
        return self

    def parallel_stages(self, *, value: bool = True) -> RecipeBuilder:
        """Set whether stages can be parallelized.

        Parameters
        ----------
        value
            Whether to allow parallel execution.

        Returns
        -------
        RecipeBuilder
            Self for chaining.
        """
        self._parallel_stages = value
        return self

    def max_parallel(self, count: int) -> RecipeBuilder:
        """Set maximum concurrent executions.

        Parameters
        ----------
        count
            Maximum parallel plugin executions.

        Returns
        -------
        RecipeBuilder
            Self for chaining.
        """
        self._max_parallel = count
        return self

    def max_duration(self, ms: int | None) -> RecipeBuilder:
        """Set maximum execution duration.

        Parameters
        ----------
        ms
            Maximum duration in milliseconds.

        Returns
        -------
        RecipeBuilder
            Self for chaining.
        """
        self._max_duration_ms = ms
        return self

    def timeout(self, ms: int | None) -> RecipeBuilder:
        """Set per-plugin timeout.

        Parameters
        ----------
        ms
            Timeout in milliseconds.

        Returns
        -------
        RecipeBuilder
            Self for chaining.
        """
        self._timeout_ms = ms
        return self

    def dry_run(self, *, value: bool = True) -> RecipeBuilder:
        """Set dry run mode.

        Parameters
        ----------
        value
            Whether to simulate execution.

        Returns
        -------
        RecipeBuilder
            Self for chaining.
        """
        self._dry_run = value
        return self

    def skip_on_unchanged(self, *, value: bool = True) -> RecipeBuilder:
        """Set skip on unchanged behavior.

        Parameters
        ----------
        value
            Whether to skip unchanged plugins.

        Returns
        -------
        RecipeBuilder
            Self for chaining.
        """
        self._skip_on_unchanged = value
        return self

    def version(self, v: str) -> RecipeBuilder:
        """Set the recipe version.

        Parameters
        ----------
        v
            Version string.

        Returns
        -------
        RecipeBuilder
            Self for chaining.
        """
        self._version = v
        return self

    def extend(self, other: Recipe) -> RecipeBuilder:
        """Extend this recipe with plugins from another recipe.

        Parameters
        ----------
        other
            Recipe to extend from.

        Returns
        -------
        RecipeBuilder
            Self for chaining.
        """
        for plugin in other.all_plugins:
            self.add(plugin)
        for plugin_name, config in other.default_configs.items():
            self.with_config(plugin_name, config)
        for t in other.tags:
            self.tag(t)
        return self

    def build(self) -> Recipe:
        """Build the recipe.

        Returns
        -------
        Recipe
            The constructed recipe.
        """
        options = RecipeOptions(
            dry_run=self._dry_run,
            skip_on_unchanged=self._skip_on_unchanged,
            max_parallel=self._max_parallel,
            timeout_ms=self._timeout_ms,
            fail_fast=self._fail_fast,
            max_duration_ms=self._max_duration_ms,
        )

        return Recipe(
            name=self._name,
            description=self._description,
            stages=tuple(self._stages),
            plugins=tuple(self._plugins),
            options=options,
            default_configs=dict(self._configs),
            tags=tuple(self._tags),
            version=self._version,
        )


__all__ = [
    "RecipeBuilder",
    "recipe",
    "stage",
]
