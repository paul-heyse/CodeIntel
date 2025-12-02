"""Graph recipe DSL for declarative pipeline composition.

This module provides a declarative domain-specific language for composing
graph construction and analysis pipelines, mirroring the ingestion recipe
system but specialized for graph workflows.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GraphStage:
    """Stage within a graph recipe.

    Attributes
    ----------
    name
        Stage identifier.
    plugins
        Plugin names to execute in this stage.
    parallel
        Whether plugins can run in parallel.
    fail_fast
        Whether to abort on first failure.
    optional
        Whether the stage can be skipped.
    """

    name: str
    plugins: tuple[str, ...]
    parallel: bool = False
    fail_fast: bool = True
    optional: bool = False


@dataclass(frozen=True)
class GraphRecipeOptions:
    """Global options for recipe execution.

    Attributes
    ----------
    dry_run
        Whether to simulate execution.
    skip_on_unchanged
        Whether to skip unchanged plugins.
    max_parallel
        Maximum parallel executions.
    timeout_ms
        Default timeout in milliseconds.
    """

    dry_run: bool = False
    skip_on_unchanged: bool = False
    max_parallel: int = 4
    timeout_ms: int | None = None


@dataclass(frozen=True)
class GraphRecipe:
    """Declarative graph recipe definition.

    Attributes
    ----------
    name
        Recipe identifier.
    description
        Human-readable description.
    stages
        Ordered stages to execute.
    options
        Global recipe options.
    version
        Recipe version string.
    """

    name: str
    description: str
    stages: tuple[GraphStage, ...]
    options: GraphRecipeOptions = field(default_factory=GraphRecipeOptions)
    version: str = "1.0"

    @property
    def all_plugins(self) -> tuple[str, ...]:
        """Return all plugin names across all stages.

        Returns
        -------
        tuple[str, ...]
            Unique plugin names in stage order.
        """
        plugins: list[str] = []
        for stage in self.stages:
            for plugin in stage.plugins:
                if plugin not in plugins:
                    plugins.append(plugin)
        return tuple(plugins)


def graph_stage(
    name: str,
    plugins: list[str],
    *,
    parallel: bool = False,
    fail_fast: bool = True,
    optional: bool = False,
) -> GraphStage:
    """Create a graph stage.

    Parameters
    ----------
    name
        Stage identifier.
    plugins
        Plugin names.
    parallel
        Whether plugins can run in parallel.
    fail_fast
        Whether to abort on first failure.
    optional
        Whether the stage can be skipped.

    Returns
    -------
    GraphStage
        Stage definition.
    """
    return GraphStage(
        name=name,
        plugins=tuple(plugins),
        parallel=parallel,
        fail_fast=fail_fast,
        optional=optional,
    )


def graph_recipe(
    name: str,
    *,
    description: str = "",
    stages: list[GraphStage],
    options: GraphRecipeOptions | None = None,
    version: str = "1.0",
) -> GraphRecipe:
    """Create a graph recipe.

    Parameters
    ----------
    name
        Recipe identifier.
    description
        Human-readable description.
    stages
        Ordered stages to execute.
    options
        Global recipe options.
    version
        Recipe version string.

    Returns
    -------
    GraphRecipe
        Recipe definition.
    """
    return GraphRecipe(
        name=name,
        description=description,
        stages=tuple(stages),
        options=options or GraphRecipeOptions(),
        version=version,
    )


__all__ = [
    "GraphRecipe",
    "GraphRecipeOptions",
    "GraphStage",
    "graph_recipe",
    "graph_stage",
]
