"""Unified recipe model for all CodeIntel pipelines.

This module provides a unified recipe abstraction that can represent any
pipeline type: ingestion, graphs, analytics, or full pipeline runs.

The UnifiedRecipe extends the core Recipe model with:

- Kind field for pipeline type classification
- Module field for stage dispatch
- Compatibility with both PipelineSpec and domain-specific recipes
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from codeintel.core.recipes.model import (
    BaseRecipeOptions,
    BaseRecipeStage,
    Recipe,
    RecipeOptions,
    RecipeScope,
    RecipeStage,
)

# Pipeline kind classification
RecipeKind = Literal["ingestion", "graphs", "analytics", "full", "custom"]
"""Classification of recipe pipeline type.

- ``ingestion``: Source code scanning and indexing
- ``graphs``: Graph construction (call graph, import graph, etc.)
- ``analytics``: Analytics computation (metrics, profiles, etc.)
- ``full``: Complete pipeline (ingest + graphs + analytics)
- ``custom``: User-defined recipe combination
"""

# Module classification for stage dispatch
StageModule = Literal["ingestion", "graphs", "analytics", "export"]
"""Module that should execute a stage.

- ``ingestion``: codeintel.ingestion plugins
- ``graphs``: codeintel.graphs plugins
- ``analytics``: codeintel.analytics plugins
- ``export``: codeintel.export operations
"""


@dataclass(frozen=True)
class UnifiedStage(BaseRecipeStage):
    """Stage within a unified recipe with module classification.

    Extends BaseRecipeStage with module assignment for proper dispatch.

    Attributes
    ----------
    name
        Stage identifier.
    plugins
        Plugin names to execute in this stage.
    module
        Target module for plugin dispatch.
    parallel
        Whether plugins within the stage can run in parallel.
    fail_fast
        Whether to abort stage on first plugin failure.
    required
        Whether this stage must succeed for the recipe to continue.
    description
        Human-readable description.
    """

    module: StageModule = "analytics"
    required: bool = True
    description: str = ""


@dataclass(frozen=True)
class UnifiedRecipeOptions(BaseRecipeOptions):
    """Options for unified recipe execution.

    Extends BaseRecipeOptions with unified pipeline options.

    Attributes
    ----------
    dry_run
        Whether to simulate execution without side effects.
    max_parallel
        Maximum concurrent plugin executions.
    fail_fast
        Whether to stop on first plugin failure.
    skip_on_unchanged
        Whether to skip plugins when inputs are unchanged.
    timeout_ms
        Default timeout per plugin in milliseconds.
    max_duration_ms
        Maximum total recipe execution time.
    enable_incremental
        Whether to enable incremental processing.
    validate_contracts
        Whether to validate plugin contracts.
    """

    skip_on_unchanged: bool = False
    timeout_ms: int | None = None
    max_duration_ms: int | None = None
    enable_incremental: bool = True
    validate_contracts: bool = True


@dataclass(frozen=True)
class UnifiedRecipe:
    """Unified recipe for any CodeIntel pipeline.

    This is the top-level recipe abstraction that can represent any pipeline
    type. It provides a consistent interface for:

    - Full pipelines (replacing PipelineSpec)
    - Domain-specific recipes (ingestion, graphs, analytics)
    - Custom composed workflows

    Attributes
    ----------
    name
        Unique recipe identifier.
    kind
        Pipeline type classification.
    stages
        Ordered stages with module classification.
    options
        Execution options.
    description
        Human-readable description.
    version
        Recipe version for cache invalidation.
    tags
        Free-form tags for categorization.
    default_configs
        Configuration overrides keyed by plugin name.
    scope
        Optional scope constraints.

    Examples
    --------
    >>> full_pipeline = UnifiedRecipe(
    ...     name="full_pipeline",
    ...     kind="full",
    ...     stages=(
    ...         UnifiedStage(name="ingest", plugins=("repo_scan",), module="ingestion"),
    ...         UnifiedStage(name="graphs", plugins=("callgraph",), module="graphs"),
    ...         UnifiedStage(name="analytics", plugins=("metrics",), module="analytics"),
    ...     ),
    ... )
    """

    name: str
    kind: RecipeKind
    stages: tuple[UnifiedStage, ...] = ()
    options: UnifiedRecipeOptions = field(default_factory=UnifiedRecipeOptions)
    description: str = ""
    version: str = "1.0.0"
    tags: tuple[str, ...] = ()
    default_configs: Mapping[str, Mapping[str, object]] = field(default_factory=dict)
    scope: RecipeScope | None = None

    @property
    def all_plugins(self) -> tuple[str, ...]:
        """Return all plugin names across all stages.

        Returns
        -------
        tuple[str, ...]
            Unique plugin names in execution order.
        """
        plugins: list[str] = []
        for stage in self.stages:
            for plugin in stage.plugins:
                if plugin not in plugins:
                    plugins.append(plugin)
        return tuple(plugins)

    @property
    def stage_names(self) -> tuple[str, ...]:
        """Return all stage names.

        Returns
        -------
        tuple[str, ...]
            Stage names in order.
        """
        return tuple(s.name for s in self.stages)

    def get_stage(self, name: str) -> UnifiedStage | None:
        """Get a stage by name.

        Parameters
        ----------
        name
            Stage name to look up.

        Returns
        -------
        UnifiedStage | None
            Stage if found, None otherwise.
        """
        for stage in self.stages:
            if stage.name == name:
                return stage
        return None

    def stages_for_module(self, module: StageModule) -> tuple[UnifiedStage, ...]:
        """Return stages for a specific module.

        Parameters
        ----------
        module
            Target module.

        Returns
        -------
        tuple[UnifiedStage, ...]
            Stages assigned to that module.
        """
        return tuple(s for s in self.stages if s.module == module)

    def to_core_recipe(self) -> Recipe:
        """Convert to a core Recipe for domain-specific execution.

        Returns
        -------
        Recipe
            Core Recipe instance.
        """
        core_stages = tuple(
            RecipeStage(
                name=s.name,
                plugins=s.plugins,
                parallel=s.parallel,
                fail_fast=s.fail_fast,
                optional=not s.required,
            )
            for s in self.stages
        )

        return Recipe(
            name=self.name,
            description=self.description,
            stages=core_stages,
            plugins=(),
            options=RecipeOptions(
                dry_run=self.options.dry_run,
                skip_on_unchanged=self.options.skip_on_unchanged,
                max_parallel=self.options.max_parallel,
                timeout_ms=self.options.timeout_ms,
                fail_fast=self.options.fail_fast,
                max_duration_ms=self.options.max_duration_ms,
            ),
            default_configs=self.default_configs,
            tags=self.tags,
            version=self.version,
        )

    @classmethod
    def from_core_recipe(
        cls,
        recipe: Recipe,
        *,
        kind: RecipeKind = "custom",
        module: StageModule = "analytics",
    ) -> UnifiedRecipe:
        """Create from a core Recipe.

        Parameters
        ----------
        recipe
            Core Recipe to convert.
        kind
            Pipeline kind classification.
        module
            Default module for stages.

        Returns
        -------
        UnifiedRecipe
            Unified recipe instance.
        """
        stages = tuple(
            UnifiedStage(
                name=s.name,
                plugins=s.plugins,
                module=module,
                parallel=s.parallel,
                fail_fast=s.fail_fast,
                required=not s.optional,
            )
            for s in recipe.stages
        )

        # If no stages but plugins defined, create a single stage
        if not stages and recipe.plugins:
            stages = (
                UnifiedStage(
                    name="default",
                    plugins=recipe.plugins,
                    module=module,
                ),
            )

        return cls(
            name=recipe.name,
            kind=kind,
            stages=stages,
            options=UnifiedRecipeOptions(
                dry_run=recipe.options.dry_run,
                skip_on_unchanged=recipe.options.skip_on_unchanged,
                max_parallel=recipe.options.max_parallel,
                timeout_ms=recipe.options.timeout_ms,
                fail_fast=recipe.options.fail_fast,
                max_duration_ms=recipe.options.max_duration_ms,
            ),
            description=recipe.description,
            version=recipe.version,
            tags=recipe.tags,
            default_configs=recipe.default_configs,
        )


# -----------------------------------------------------------------------------
# Builder helpers
# -----------------------------------------------------------------------------


def unified_stage(
    name: str,
    plugins: Sequence[str],
    *,
    module: StageModule = "analytics",
    **kwargs: object,
) -> UnifiedStage:
    """Create a unified stage.

    For full customization, use the UnifiedStage dataclass directly.

    Parameters
    ----------
    name
        Stage identifier.
    plugins
        Plugin names for this stage.
    module
        Target module for dispatch.
    **kwargs
        Additional UnifiedStage fields (parallel, fail_fast, required, description).

    Returns
    -------
    UnifiedStage
        New stage instance.
    """
    parallel = bool(kwargs.get("parallel"))
    fail_fast = bool(kwargs.get("fail_fast", True))
    required = bool(kwargs.get("required", True))
    description = str(kwargs.get("description", ""))

    return UnifiedStage(
        name=name,
        plugins=tuple(plugins),
        module=module,
        parallel=parallel,
        fail_fast=fail_fast,
        required=required,
        description=description,
    )


def unified_recipe(
    name: str,
    *,
    kind: RecipeKind = "custom",
    stages: Sequence[UnifiedStage] | None = None,
    **kwargs: object,
) -> UnifiedRecipe:
    """Create a unified recipe.

    For full customization, use the UnifiedRecipe dataclass directly.

    Parameters
    ----------
    name
        Recipe identifier.
    kind
        Pipeline type classification.
    stages
        Ordered execution stages.
    **kwargs
        Additional UnifiedRecipe fields (description, version, options, tags).

    Returns
    -------
    UnifiedRecipe
        New recipe instance.
    """
    description = str(kwargs.get("description", ""))
    version = str(kwargs.get("version", "1.0.0"))
    options_val = kwargs.get("options")
    options = (
        options_val if isinstance(options_val, UnifiedRecipeOptions) else UnifiedRecipeOptions()
    )
    tags_val = kwargs.get("tags")
    tags = tuple(tags_val) if isinstance(tags_val, Sequence) else ()

    return UnifiedRecipe(
        name=name,
        kind=kind,
        stages=tuple(stages) if stages else (),
        options=options,
        description=description,
        version=version,
        tags=tags,
    )


__all__ = [
    "RecipeKind",
    "StageModule",
    "UnifiedRecipe",
    "UnifiedRecipeOptions",
    "UnifiedStage",
    "unified_recipe",
    "unified_stage",
]
