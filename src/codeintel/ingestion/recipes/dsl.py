"""Recipe DSL for declarative ingestion pipeline composition.

This module provides the data structures for defining ingestion recipes,
which are declarative compositions of plugins organized into stages.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class RecipeStage:
    """A stage in the ingestion recipe.

    Stages group plugins that can potentially run together and define
    execution semantics like parallelism and failure handling.

    Attributes
    ----------
    name
        Stage identifier (e.g., "scan", "parse", "enrich").
    plugins
        Plugin names to execute in this stage.
    parallel
        Whether plugins in this stage can run in parallel.
    required
        Whether this stage must succeed for the recipe to continue.
    timeout_s
        Maximum execution time for this stage in seconds.
    description
        Human-readable description of the stage.
    """

    name: str
    plugins: tuple[str, ...]
    parallel: bool = False
    required: bool = True
    timeout_s: int | None = None
    description: str = ""

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> RecipeStage:
        """Create a stage from a dictionary.

        Parameters
        ----------
        data
            Dictionary with stage configuration.

        Returns
        -------
        RecipeStage
            Parsed stage instance.
        """
        plugins_raw = data.get("plugins", [])
        plugins = tuple(plugins_raw) if isinstance(plugins_raw, (list, tuple)) else ()

        return cls(
            name=str(data.get("name", "")),
            plugins=plugins,
            parallel=bool(data.get("parallel", False)),
            required=bool(data.get("required", True)),
            timeout_s=int(str(data["timeout_s"])) if data.get("timeout_s") else None,
            description=str(data.get("description", "")),
        )

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary representation.

        Returns
        -------
        dict[str, object]
            Dictionary suitable for serialization.
        """
        result: dict[str, object] = {
            "name": self.name,
            "plugins": list(self.plugins),
        }
        if self.parallel:
            result["parallel"] = True
        if not self.required:
            result["required"] = False
        if self.timeout_s is not None:
            result["timeout_s"] = self.timeout_s
        if self.description:
            result["description"] = self.description
        return result


@dataclass(frozen=True)
class RecipeOptions:
    """Global options for recipe execution.

    Attributes
    ----------
    enable_incremental
        Whether to enable incremental ingestion.
    enable_contracts
        Whether to validate output contracts.
    max_parallel_plugins
        Maximum number of plugins to run in parallel.
    fail_fast
        Stop on first plugin failure.
    continue_on_soft_fail
        Continue execution after soft failures.
    dry_run
        Validate recipe without executing.
    """

    enable_incremental: bool = True
    enable_contracts: bool = True
    max_parallel_plugins: int = 4
    fail_fast: bool = True
    continue_on_soft_fail: bool = True
    dry_run: bool = False

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> RecipeOptions:
        """Create options from a dictionary.

        Parameters
        ----------
        data
            Dictionary with options configuration.

        Returns
        -------
        RecipeOptions
            Parsed options instance.
        """
        max_parallel_raw = data.get("max_parallel_plugins", 4)
        max_parallel = int(str(max_parallel_raw)) if max_parallel_raw is not None else 4
        return cls(
            enable_incremental=bool(data.get("enable_incremental", True)),
            enable_contracts=bool(data.get("enable_contracts", True)),
            max_parallel_plugins=max_parallel,
            fail_fast=bool(data.get("fail_fast", True)),
            continue_on_soft_fail=bool(data.get("continue_on_soft_fail", True)),
            dry_run=bool(data.get("dry_run", False)),
        )

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary representation.

        Returns
        -------
        dict[str, object]
            Dictionary suitable for serialization.
        """
        return {
            "enable_incremental": self.enable_incremental,
            "enable_contracts": self.enable_contracts,
            "max_parallel_plugins": self.max_parallel_plugins,
            "fail_fast": self.fail_fast,
            "continue_on_soft_fail": self.continue_on_soft_fail,
            "dry_run": self.dry_run,
        }


@dataclass(frozen=True)
class IngestRecipe:
    """Declarative recipe for ingestion pipeline composition.

    Recipes define which plugins run in what order, with options
    for parallelism, failure handling, and conditional execution.

    Attributes
    ----------
    name
        Unique recipe identifier.
    description
        Human-readable description.
    version
        Recipe version string.
    stages
        Ordered list of execution stages.
    options
        Global execution options.
    disabled_plugins
        Plugins to exclude from execution.
    enabled_plugins
        Explicit list of plugins to enable (overrides defaults).
    includes
        Other recipes to include before this one.
    tags
        Classification tags for the recipe.
    """

    name: str
    description: str = ""
    version: str = "1.0.0"
    stages: tuple[RecipeStage, ...] = ()
    options: RecipeOptions = field(default_factory=RecipeOptions)
    disabled_plugins: tuple[str, ...] = ()
    enabled_plugins: tuple[str, ...] | None = None
    includes: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    @property
    def all_plugins(self) -> tuple[str, ...]:
        """Return all plugin names referenced in stages.

        Returns
        -------
        tuple[str, ...]
            All plugin names in stage order.
        """
        plugins: list[str] = []
        for stage in self.stages:
            plugins.extend(stage.plugins)
        return tuple(plugins)

    @property
    def stage_names(self) -> tuple[str, ...]:
        """Return all stage names.

        Returns
        -------
        tuple[str, ...]
            Stage names in order.
        """
        return tuple(stage.name for stage in self.stages)

    def get_stage(self, name: str) -> RecipeStage | None:
        """Get a stage by name.

        Parameters
        ----------
        name
            Stage name to look up.

        Returns
        -------
        RecipeStage | None
            Stage if found, None otherwise.
        """
        for stage in self.stages:
            if stage.name == name:
                return stage
        return None

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> IngestRecipe:
        """Create a recipe from a dictionary.

        Parameters
        ----------
        data
            Dictionary with recipe configuration.

        Returns
        -------
        IngestRecipe
            Parsed recipe instance.
        """
        stages_raw = data.get("stages", [])
        stages: list[RecipeStage] = [
            RecipeStage.from_dict(stage_data)
            for stage_data in (stages_raw if isinstance(stages_raw, (list, tuple)) else [])
            if isinstance(stage_data, Mapping)
        ]

        options_raw = data.get("options", {})
        options = (
            RecipeOptions.from_dict(options_raw)
            if isinstance(options_raw, Mapping)
            else RecipeOptions()
        )

        disabled_raw = data.get("disabled_plugins", [])
        disabled = tuple(disabled_raw) if isinstance(disabled_raw, (list, tuple)) else ()

        enabled_raw = data.get("enabled_plugins")
        enabled = tuple(enabled_raw) if isinstance(enabled_raw, (list, tuple)) else None

        includes_raw = data.get("includes", [])
        includes = tuple(includes_raw) if isinstance(includes_raw, (list, tuple)) else ()

        tags_raw = data.get("tags", [])
        tags = tuple(tags_raw) if isinstance(tags_raw, (list, tuple)) else ()

        return cls(
            name=str(data.get("name", "")),
            description=str(data.get("description", "")),
            version=str(data.get("version", "1.0.0")),
            stages=tuple(stages),
            options=options,
            disabled_plugins=disabled,
            enabled_plugins=enabled,
            includes=includes,
            tags=tags,
        )

    @classmethod
    def from_yaml(cls, path: Path) -> IngestRecipe:
        """Load a recipe from a YAML file.

        Parameters
        ----------
        path
            Path to the YAML file.

        Returns
        -------
        IngestRecipe
            Parsed recipe instance.
        """
        content = path.read_text(encoding="utf-8")
        data = yaml.safe_load(content) or {}
        return cls.from_dict(data)

    @classmethod
    def from_yaml_str(cls, content: str) -> IngestRecipe:
        """Load a recipe from a YAML string.

        Parameters
        ----------
        content
            YAML content.

        Returns
        -------
        IngestRecipe
            Parsed recipe instance.
        """
        data = yaml.safe_load(content) or {}
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary representation.

        Returns
        -------
        dict[str, object]
            Dictionary suitable for serialization.
        """
        result: dict[str, object] = {
            "name": self.name,
            "version": self.version,
            "stages": [stage.to_dict() for stage in self.stages],
        }
        if self.description:
            result["description"] = self.description
        if self.options != RecipeOptions():
            result["options"] = self.options.to_dict()
        if self.disabled_plugins:
            result["disabled_plugins"] = list(self.disabled_plugins)
        if self.enabled_plugins is not None:
            result["enabled_plugins"] = list(self.enabled_plugins)
        if self.includes:
            result["includes"] = list(self.includes)
        if self.tags:
            result["tags"] = list(self.tags)
        return result

    def to_yaml(self) -> str:
        """Convert to YAML string.

        Returns
        -------
        str
            YAML representation of the recipe.
        """
        result = yaml.safe_dump(self.to_dict(), default_flow_style=False, sort_keys=False)
        return str(result) if result is not None else ""

    def with_options(self, **kwargs: object) -> IngestRecipe:
        """Create a copy with modified options.

        Parameters
        ----------
        **kwargs
            Option overrides.

        Returns
        -------
        IngestRecipe
            New recipe with modified options.
        """
        current = self.options.to_dict()
        current.update(kwargs)
        new_options = RecipeOptions.from_dict(current)
        return IngestRecipe(
            name=self.name,
            description=self.description,
            version=self.version,
            stages=self.stages,
            options=new_options,
            disabled_plugins=self.disabled_plugins,
            enabled_plugins=self.enabled_plugins,
            includes=self.includes,
            tags=self.tags,
        )

    def with_disabled(self, *plugins: str) -> IngestRecipe:
        """Create a copy with additional disabled plugins.

        Parameters
        ----------
        *plugins
            Plugin names to disable.

        Returns
        -------
        IngestRecipe
            New recipe with additional disabled plugins.
        """
        return IngestRecipe(
            name=self.name,
            description=self.description,
            version=self.version,
            stages=self.stages,
            options=self.options,
            disabled_plugins=self.disabled_plugins + plugins,
            enabled_plugins=self.enabled_plugins,
            includes=self.includes,
            tags=self.tags,
        )


@dataclass(frozen=True)
class RecipeStageResult:
    """Result of executing a recipe stage.

    Attributes
    ----------
    stage
        The stage that was executed.
    success
        Whether all plugins in the stage succeeded.
    plugin_results
        Mapping of plugin name to result.
    duration_s
        Execution duration in seconds.
    """

    stage: RecipeStage
    success: bool
    plugin_results: Mapping[str, object]
    duration_s: float = 0.0


@dataclass(frozen=True)
class RecipeExecutionResult:
    """Result of executing a complete recipe.

    Attributes
    ----------
    recipe
        The recipe that was executed.
    success
        Whether all required stages succeeded.
    stage_results
        Results for each executed stage.
    skipped_stages
        Stages that were skipped.
    duration_s
        Total execution duration in seconds.
    error
        Error message if execution failed.
    """

    recipe: IngestRecipe
    success: bool
    stage_results: tuple[RecipeStageResult, ...] = ()
    skipped_stages: tuple[str, ...] = ()
    duration_s: float = 0.0
    error: str | None = None


# Recipe builder helpers


def stage(  # noqa: PLR0913
    name: str,
    plugins: Sequence[str],
    *,
    parallel: bool = False,
    required: bool = True,
    timeout_s: int | None = None,
    description: str = "",
) -> RecipeStage:
    """Create a recipe stage.

    Parameters
    ----------
    name
        Stage identifier.
    plugins
        Plugin names for this stage.
    parallel
        Whether plugins can run in parallel.
    required
        Whether stage must succeed.
    timeout_s
        Maximum execution time.
    description
        Stage description.

    Returns
    -------
    RecipeStage
        New stage instance.
    """
    return RecipeStage(
        name=name,
        plugins=tuple(plugins),
        parallel=parallel,
        required=required,
        timeout_s=timeout_s,
        description=description,
    )


def recipe(  # noqa: PLR0913
    name: str,
    stages: Sequence[RecipeStage],
    *,
    description: str = "",
    version: str = "1.0.0",
    options: RecipeOptions | None = None,
    disabled_plugins: Sequence[str] = (),
    enabled_plugins: Sequence[str] | None = None,
    includes: Sequence[str] = (),
    tags: Sequence[str] = (),
) -> IngestRecipe:
    """Create an ingestion recipe.

    Parameters
    ----------
    name
        Recipe identifier.
    stages
        Execution stages.
    description
        Recipe description.
    version
        Recipe version.
    options
        Execution options.
    disabled_plugins
        Plugins to exclude.
    enabled_plugins
        Plugins to enable explicitly.
    includes
        Recipes to include.
    tags
        Classification tags.

    Returns
    -------
    IngestRecipe
        New recipe instance.
    """
    return IngestRecipe(
        name=name,
        description=description,
        version=version,
        stages=tuple(stages),
        options=options or RecipeOptions(),
        disabled_plugins=tuple(disabled_plugins),
        enabled_plugins=tuple(enabled_plugins) if enabled_plugins is not None else None,
        includes=tuple(includes),
        tags=tuple(tags),
    )


__all__ = [
    "IngestRecipe",
    "RecipeExecutionResult",
    "RecipeOptions",
    "RecipeStage",
    "RecipeStageResult",
    "recipe",
    "stage",
]
