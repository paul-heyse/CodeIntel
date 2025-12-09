"""Graph plugin registry with dependency resolution.

This module provides the registry for graph plugins, extending the base
registry infrastructure from codeintel.core.plugins. It supports
decorator-based registration, dependency resolution, topological ordering,
and discovery via Python entry points.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Literal, TypeGuard

from codeintel.core.execution.ids import new_run_id
from codeintel.core.plugins.registry.base import BasePluginRegistry, RegistryHooks
from codeintel.core.plugins.registry.sorting import (
    build_provider_index_from_metadata,
    topological_sort,
)
from codeintel.core.plugins.types.result import PluginResult
from codeintel.core.singleton import SingletonHolder
from codeintel.graphs.core.protocol import (
    DEFAULT_GRAPH_PLUGINS,
    GraphPluginMetadata,
    GraphPluginMetadataConfig,
    GraphPluginPlan,
    GraphPluginProtocol,
    GraphPluginSkip,
    create_graph_metadata,
)

log = logging.getLogger(__name__)


class _GraphPluginRegistryHolder(SingletonHolder["GraphPluginRegistry"]):
    """Singleton holder for GraphPluginRegistry.

    Uses the thread-safe SingletonHolder pattern from core.
    """


class GraphRegistryHooks(RegistryHooks[GraphPluginProtocol]):
    """Graph-specific registry hooks."""

    def __init__(self) -> None:
        """Initialize graph registry hooks."""
        self._entrypoint_group = "codeintel.graph_plugins"

    @property
    def entrypoint_group(self) -> str:
        """Entrypoint group name for graph plugins."""
        return self._entrypoint_group

    def load_builtins(self) -> None:
        """Import built-in graph plugins to guarantee registration."""
        _ = self._entrypoint_group
        try:
            importlib.import_module("codeintel.graphs.plugins")
        except ImportError as exc:
            log.warning("Failed to import built-in graph plugins: %s", exc)

    def resolve_entrypoint(self, loaded: object) -> GraphPluginProtocol | None:
        """Resolve an entry point to a GraphPluginProtocol instance.

        Returns
        -------
        GraphPluginProtocol | None
            Valid plugin instance or None if resolution fails.
        """
        if self.is_valid_plugin(loaded):
            return loaded

        if isinstance(loaded, type) or (callable(loaded) and not hasattr(loaded, "metadata")):
            instance = loaded()
            if self.is_valid_plugin(instance):
                return instance
        return None

    def is_valid_plugin(self, obj: object) -> TypeGuard[GraphPluginProtocol]:
        """Check if an object is a valid graph plugin.

        Returns
        -------
        TypeGuard[GraphPluginProtocol]
            True when the object satisfies the graph plugin protocol.
        """
        _ = self._entrypoint_group
        return isinstance(obj, GraphPluginProtocol)


class DependencyPolicy(Enum):
    """Dependency resolution policy for planning."""

    STRICT = "strict"
    SKIP = "skip"


class SelectionPolicy(Enum):
    """Selection handling policy for requested plugin names."""

    LENIENT = "lenient"
    STRICT = "strict"


class _StubGraphPlugin(GraphPluginProtocol):
    """Minimal stub plugin used to satisfy ingestion dependencies."""

    def __init__(self, *, name: str, depends_on: tuple[str, ...]) -> None:
        config = GraphPluginMetadataConfig(
            severity="soft_fail",
            enabled_by_default=False,
            produces_tables=(),
            depends_on=depends_on,
        )
        self._metadata = create_graph_metadata(
            name=name,
            description="Stub plugin for dependency resolution",
            kind="builder",
            stage="goid",
            config=config,
        )

    @property
    def metadata(self) -> GraphPluginMetadata:
        return self._metadata

    def execute(self, ctx: object) -> PluginResult:  # pragma: no cover - not executed
        _ = ctx
        return PluginResult.ok(meta={"stub": self._metadata.name})


@dataclass(frozen=True)
class PlanningOptions:
    """Options controlling graph planning and dependency handling.

    ``requested_required`` controls whether explicitly requested plugins must be
    present even when ``selection_policy`` is lenient. When left as ``None``,
    it is derived from dependency strictness and stub usage so that strict
    dependency plans treat requested plugins as required by default.
    """

    allow_missing_dependencies: bool = False
    dependency_policy: DependencyPolicy = DependencyPolicy.STRICT
    selection_policy: SelectionPolicy = SelectionPolicy.LENIENT
    use_stubs: bool = True
    requested_required: bool | None = None

    def __post_init__(self) -> None:
        """Derive ``requested_required`` from other fields when not set explicitly."""
        if self.requested_required is not None:
            return
        auto_required = self.dependency_policy is DependencyPolicy.STRICT or not self.use_stubs
        object.__setattr__(self, "requested_required", auto_required)

    @classmethod
    def for_required_requests(
        cls,
        *,
        allow_missing_dependencies: bool = False,
        dependency_policy: DependencyPolicy = DependencyPolicy.STRICT,
        selection_policy: SelectionPolicy = SelectionPolicy.LENIENT,
        use_stubs: bool = True,
    ) -> PlanningOptions:
        """Build planning options that treat explicit requests as required.

        Parameters
        ----------
        allow_missing_dependencies
            Whether to allow missing dependencies in plans.
        dependency_policy
            Policy for handling dependencies.
        selection_policy
            Policy for plugin selection.
        use_stubs
            Whether to use stub plugins for missing dependencies.

        Returns
        -------
        PlanningOptions
            Configured planning options with ``requested_required=True``.
        """
        return cls(
            requested_required=True,
            allow_missing_dependencies=allow_missing_dependencies,
            dependency_policy=dependency_policy,
            selection_policy=selection_policy,
            use_stubs=use_stubs,
        )

    @classmethod
    def for_lenient_requests(
        cls,
        *,
        allow_missing_dependencies: bool = False,
        dependency_policy: DependencyPolicy = DependencyPolicy.STRICT,
        selection_policy: SelectionPolicy = SelectionPolicy.LENIENT,
        use_stubs: bool = True,
    ) -> PlanningOptions:
        """Build planning options that treat explicit requests as optional.

        Parameters
        ----------
        allow_missing_dependencies
            Whether to allow missing dependencies in plans.
        dependency_policy
            Policy for handling dependencies.
        selection_policy
            Policy for plugin selection.
        use_stubs
            Whether to use stub plugins for missing dependencies.

        Returns
        -------
        PlanningOptions
            Configured planning options with ``requested_required=False``.
        """
        return cls(
            requested_required=False,
            allow_missing_dependencies=allow_missing_dependencies,
            dependency_policy=dependency_policy,
            selection_policy=selection_policy,
            use_stubs=use_stubs,
        )


GraphSkipReason = Literal[
    "disabled",
    "missing_dependency",
    "missing_graph",
    "config_error",
    "incremental_skip",
    "unchanged",
]


@dataclass
class _DependencyContext:
    """Mutable context used during dependency expansion."""

    selected: dict[str, GraphPluginProtocol]
    disabled_set: set[str]
    allow_missing: bool
    dependency_policy: DependencyPolicy
    skipped: list[GraphPluginSkip]


class GraphPluginRegistry(BasePluginRegistry[GraphPluginProtocol]):
    """Central registry for graph plugins.

    Extends BasePluginRegistry with graph-specific functionality
    including GraphPluginPlan and GraphPluginSkip types.

    For singleton access, use :func:`get_graph_registry` rather than
    instantiating directly. Direct instantiation is useful for testing.
    """

    def __init__(self, hooks: RegistryHooks[GraphPluginProtocol] | None = None) -> None:
        """Initialize the graph registry with hooks."""
        super().__init__(hooks=hooks or GraphRegistryHooks())

    @staticmethod
    def _get_default_plugins() -> Sequence[str]:
        """Return the default graph plugin names.

        Returns
        -------
        Sequence[str]
            Default graph plugin names.
        """
        return DEFAULT_GRAPH_PLUGINS

    def plan(
        self,
        plugin_names: Sequence[str] | None = None,
        *,
        enabled: Sequence[str] | None = None,
        disabled: Sequence[str] | None = None,
        defaults: Sequence[str] | None = None,
        plan_options: PlanningOptions | None = None,
    ) -> GraphPluginPlan:
        """Build an execution plan with dependency resolution.

        Override base implementation to return GraphPluginPlan with
        graph-specific skip reasons. May raise ValueError from helper
        methods if plugins are listed more than once or dependencies
        are missing/cyclic.

        Parameters
        ----------
        plugin_names
            Explicit plugin names to include.
        enabled
            Override list of enabled plugins.
        disabled
            Plugins to exclude from the plan.
        defaults
            Default plugins if no explicit list provided.
        plan_options
            Planning options controlling dependency handling. Selection
            defaults to ``SelectionPolicy.LENIENT`` and explicit requests
            can still be treated as required via ``requested_required``.

        Returns
        -------
        GraphPluginPlan
            Ordered execution plan with graph-specific metadata.
        """
        plan_opts = plan_options or PlanningOptions()

        self._ensure_loaded()
        if plan_opts.use_stubs:
            self._ensure_stub_plugins()

        if (
            plan_opts.selection_policy is not SelectionPolicy.LENIENT
            or plan_opts.dependency_policy is not DependencyPolicy.STRICT
            or plan_opts.allow_missing_dependencies
            or not plan_opts.use_stubs
            or plan_opts.requested_required
        ):
            log.info(
                "Planning with selection_policy=%s dependency_policy=%s "
                "allow_missing=%s use_stubs=%s requested_required=%s",
                plan_opts.selection_policy.value,
                plan_opts.dependency_policy.value,
                plan_opts.allow_missing_dependencies,
                plan_opts.use_stubs,
                plan_opts.requested_required,
            )

        # Resolve which plugins to include
        selected, skipped = self._resolve_graph_selection(
            plugin_names=plugin_names,
            enabled=enabled,
            disabled=disabled,
            defaults=defaults or self._get_default_plugins(),
            plan_opts=plan_opts,
        )
        skip_buffer = list(skipped)

        # Build dependency graph
        dependencies = self._resolve_graph_dependencies(
            selected,
            skipped=skip_buffer,
            plan_opts=plan_opts,
        )

        # Topological sort using shared utility
        ordered = topological_sort(selected, dependencies)

        dep_graph_out: dict[str, tuple[str, ...]] = {
            name: tuple(sorted(deps)) for name, deps in dependencies.items()
        }
        for skip in skip_buffer:
            if skip.name not in dep_graph_out:
                dep_graph_out[skip.name] = ()

        return GraphPluginPlan(
            plugins=tuple(ordered),
            plan_id=new_run_id("plan"),
            skipped_plugins=tuple(skip_buffer),
            dep_graph=dep_graph_out,
        )

    def _resolve_graph_selection(
        self,
        *,
        plugin_names: Sequence[str] | None,
        enabled: Sequence[str] | None,
        disabled: Sequence[str] | None,
        defaults: Sequence[str],
        plan_opts: PlanningOptions,
    ) -> tuple[dict[str, GraphPluginProtocol], list[GraphPluginSkip]]:
        """Resolve which plugins to include in the plan.

        Returns
        -------
        tuple[dict[str, GraphPluginProtocol], tuple[GraphPluginSkip, ...]]
            Selected plugins keyed by name and plugins that were skipped.

        Raises
        ------
        ValueError
            If a plugin name appears more than once or required dependencies are disabled.
        """
        # Determine base selection
        if enabled:
            names = list(enabled)
        elif plugin_names:
            names = list(plugin_names)
        else:
            names = list(defaults)

        requested_names = set(plugin_names or ())
        disabled_set = set(disabled or ())
        selected: dict[str, GraphPluginProtocol] = {}
        skipped: list[GraphPluginSkip] = []

        for name in names:
            if name in disabled_set:
                self._add_skip_once(skipped, name, reason="disabled")
                continue

            if name in selected:
                message = f"Graph plugin '{name}' listed more than once"
                raise ValueError(message)

            try:
                plugin = self.get(name)
            except KeyError:
                if plan_opts.selection_policy is SelectionPolicy.STRICT or (
                    plan_opts.requested_required and name in requested_names
                ):
                    message = self._unknown_plugin_message(name)
                    raise ValueError(message) from None
                self._add_skip_once(skipped, name, reason="missing_graph")
                log.warning("Skipping unknown graph plugin: %s", name)
                continue

            selected[name] = plugin

        self._expand_dependencies(
            selected,
            disabled_set=disabled_set,
            allow_missing=plan_opts.allow_missing_dependencies,
            dependency_policy=plan_opts.dependency_policy,
            skipped=skipped,
        )

        return selected, skipped

    def _expand_dependencies(
        self,
        selected: dict[str, GraphPluginProtocol],
        *,
        disabled_set: set[str],
        allow_missing: bool,
        dependency_policy: DependencyPolicy,
        skipped: list[GraphPluginSkip],
    ) -> None:
        """Ensure all explicit depends_on plugins are included.

        Raises
        ------
        ValueError
            If a dependency is missing or disabled and allow_missing is False.
        """
        ctx = _DependencyContext(
            selected=selected,
            disabled_set=disabled_set,
            allow_missing=allow_missing,
            dependency_policy=dependency_policy,
            skipped=skipped,
        )
        added = True
        while added:
            added = False
            for name, plugin in list(selected.items()):
                for dep in plugin.metadata.depends_on:
                    outcome, message = self._process_dependency(
                        requester=name,
                        dependency=dep,
                        ctx=ctx,
                    )
                    if outcome == "error" and message is not None:
                        raise ValueError(message)
                    if outcome == "added":
                        added = True

    def _resolve_graph_dependencies(
        self,
        selected: Mapping[str, GraphPluginProtocol],
        *,
        skipped: list[GraphPluginSkip],
        plan_opts: PlanningOptions,
    ) -> dict[str, set[str]]:
        """Build dependency graph for selected plugins.

        Parameters
        ----------
        selected
            Selected plugins keyed by name.
        skipped
            Mutable buffer of skip records to append missing dependencies to.
        plan_opts
            Planning options controlling dependency handling.

        Returns
        -------
        dict[str, set[str]]
            Mapping of plugin name to its dependency names.
        """
        dependencies: dict[str, set[str]] = {name: set() for name in selected}

        self._add_explicit_dependencies(
            selected,
            dependencies,
            skipped=skipped,
            plan_opts=plan_opts,
        )
        self._add_capability_dependencies(
            selected,
            dependencies,
            skipped=skipped,
            plan_opts=plan_opts,
        )

        return dependencies

    @staticmethod
    def _add_explicit_dependencies(
        selected: Mapping[str, GraphPluginProtocol],
        dependencies: dict[str, set[str]],
        *,
        skipped: list[GraphPluginSkip],
        plan_opts: PlanningOptions,
    ) -> None:
        for name, plugin in selected.items():
            for dep in plugin.metadata.depends_on:
                if dep not in selected:
                    if (
                        plan_opts.allow_missing_dependencies
                        or plan_opts.dependency_policy is DependencyPolicy.SKIP
                    ):
                        GraphPluginRegistry._add_skip_once(
                            skipped, dep, reason="missing_dependency"
                        )
                        log.warning(
                            "Skipping missing dependency %s for plugin %s due to allow_missing_dependencies",
                            dep,
                            name,
                        )
                        continue
                    message = GraphPluginRegistry._missing_in_selection_message(name, dep)
                    raise ValueError(message)
                dependencies[name].add(dep)

    @staticmethod
    def _add_capability_dependencies(
        selected: Mapping[str, GraphPluginProtocol],
        dependencies: dict[str, set[str]],
        *,
        skipped: list[GraphPluginSkip],
        plan_opts: PlanningOptions,
    ) -> None:
        provider_index = build_provider_index_from_metadata(
            selected,
            get_provides=lambda p: p.metadata.provides,
        )
        for name, plugin in selected.items():
            for requirement in plugin.metadata.requires:
                providers = provider_index.get(requirement, set())
                if not providers:
                    if (
                        plan_opts.allow_missing_dependencies
                        or plan_opts.dependency_policy is DependencyPolicy.SKIP
                    ):
                        GraphPluginRegistry._add_skip_once(
                            skipped, requirement, reason="missing_dependency"
                        )
                        log.warning(
                            "No providers for capability %s required by %s; skipping due to allow_missing_dependencies",
                            requirement,
                            name,
                        )
                        continue
                    message = (
                        f"Graph plugin '{name}' requires capability '{requirement}', "
                        "but no provider plugin is selected"
                    )
                    raise ValueError(message)
                if name in providers:
                    continue
                explicit_deps = dependencies[name]
                if providers.intersection(explicit_deps):
                    continue
                if len(providers) > 1:
                    provider_list = ", ".join(sorted(providers))
                    message = (
                        f"Graph plugin '{name}' requires capability '{requirement}', "
                        f"but multiple providers are available ({provider_list}). "
                        "Add an explicit depends_on entry to disambiguate."
                    )
                    raise ValueError(message)
                dependencies[name].add(next(iter(providers)))

    @staticmethod
    def _unknown_plugin_message(name: str) -> str:
        return f"Graph plugin '{name}' is not registered"

    @staticmethod
    def _add_skip_once(
        skipped: list[GraphPluginSkip],
        name: str,
        *,
        reason: GraphSkipReason = "missing_dependency",
    ) -> None:
        if any(skip.name == name for skip in skipped):
            return
        skipped.append(GraphPluginSkip(name=name, reason=reason))

    def _process_dependency(
        self,
        *,
        requester: str,
        dependency: str,
        ctx: _DependencyContext,
    ) -> tuple[Literal["present", "added", "skipped", "error"], str | None]:
        outcome: Literal["present", "added", "skipped", "error"] = "present"
        message: str | None = None

        if dependency in ctx.selected:
            return outcome, message

        if dependency in ctx.disabled_set:
            if ctx.allow_missing or ctx.dependency_policy is DependencyPolicy.SKIP:
                self._add_skip_once(ctx.skipped, dependency, reason="missing_dependency")
                log.warning(
                    "Dependency %s for plugin %s is disabled; skipping inclusion",
                    dependency,
                    requester,
                )
                outcome = "skipped"
            else:
                outcome = "error"
                message = self._disabled_dependency_message(requester, dependency)
            return outcome, message

        if not self.contains(dependency):
            if ctx.allow_missing or ctx.dependency_policy is DependencyPolicy.SKIP:
                self._add_skip_once(ctx.skipped, dependency, reason="missing_dependency")
                log.warning(
                    "Skipping missing dependency %s for plugin %s",
                    dependency,
                    requester,
                )
                outcome = "skipped"
            else:
                outcome = "error"
                message = self._missing_dependency_message(requester, dependency)
            return outcome, message

        try:
            ctx.selected[dependency] = self.get(dependency)
        except KeyError:
            if ctx.allow_missing or ctx.dependency_policy is DependencyPolicy.SKIP:
                self._add_skip_once(ctx.skipped, dependency, reason="missing_dependency")
                log.warning(
                    "Skipping missing dependency %s for plugin %s",
                    dependency,
                    requester,
                )
                outcome = "skipped"
            else:
                outcome = "error"
                message = self._missing_dependency_message(requester, dependency)
        else:
            outcome = "added"

        return outcome, message

    @staticmethod
    def _missing_dependency_message(requester: str, dependency: str) -> str:
        return f"Graph plugin '{requester}' depends on '{dependency}', which is not registered"

    @staticmethod
    def _missing_in_selection_message(requester: str, dependency: str) -> str:
        return (
            f"Graph plugin '{requester}' depends on '{dependency}', "
            "which is not in the selected plugin set"
        )

    @staticmethod
    def _disabled_dependency_message(requester: str, dependency: str) -> str:
        return (
            f"Graph plugin '{requester}' depends on '{dependency}', "
            "which is disabled or missing. Re-enable it or set allow_missing_dependencies."
        )

    def _ensure_stub_plugins(self) -> None:
        """Register stub ingestion plugins when missing."""
        known_deps = {
            "repo_scan": (),
            "ast_extract": (),
            "scip_ingest": (),
            "goid_builder": ("repo_scan", "ast_extract", "scip_ingest"),
        }
        for name, depends_on in known_deps.items():
            if self.contains(name):
                continue
            self.register(_StubGraphPlugin(name=name, depends_on=depends_on))


def get_graph_registry() -> GraphPluginRegistry:
    """Return the global graph plugin registry.

    Returns
    -------
    GraphPluginRegistry
        The singleton registry instance.
    """
    return _GraphPluginRegistryHolder.get(GraphPluginRegistry)


def reset_graph_registry() -> None:
    """Reset the global registry for testing.

    This clears the global registry, allowing tests to start fresh.
    """
    _GraphPluginRegistryHolder.reset()


def register_graph_plugin(plugin: GraphPluginProtocol) -> None:
    """Register a plugin with the global registry.

    Parameters
    ----------
    plugin
        Plugin instance to register.
    """
    get_graph_registry().register(plugin)


def unregister_graph_plugin(name: str) -> None:
    """Remove a plugin from the global registry.

    Parameters
    ----------
    name
        Plugin name to remove.
    """
    get_graph_registry().unregister(name)


def list_graph_plugins() -> tuple[GraphPluginProtocol, ...]:
    """Return all registered graph plugins.

    Returns
    -------
    tuple[GraphPluginProtocol, ...]
        All registered graph plugins.
    """
    return get_graph_registry().list_all()


def plan_graph_plugins(
    plugin_names: Sequence[str] | None = None,
    *,
    enabled: Sequence[str] | None = None,
    disabled: Sequence[str] | None = None,
    defaults: Sequence[str] | None = None,
    plan_options: PlanningOptions | None = None,
) -> GraphPluginPlan:
    """Build an execution plan for graph plugins.

    Parameters
    ----------
    plugin_names
        Explicit plugin names to include.
    enabled
        Override list of enabled plugins.
    disabled
        Plugins to exclude.
    defaults
        Default plugins if no explicit list provided.
    plan_options
        Planning options controlling dependency handling. Selection defaults
        to ``SelectionPolicy.LENIENT`` to preserve backward-compatible
        skipping of unknown plugins.

    Returns
    -------
    GraphPluginPlan
        Ordered execution plan.
    """
    resolved_plan_options = plan_options or PlanningOptions()
    return get_graph_registry().plan(
        plugin_names=plugin_names,
        enabled=enabled,
        disabled=disabled,
        defaults=defaults,
        plan_options=resolved_plan_options,
    )


__all__ = [
    "DependencyPolicy",
    "GraphPluginRegistry",
    "GraphRegistryHooks",
    "PlanningOptions",
    "SelectionPolicy",
    "get_graph_registry",
    "list_graph_plugins",
    "plan_graph_plugins",
    "register_graph_plugin",
    "reset_graph_registry",
    "unregister_graph_plugin",
]
