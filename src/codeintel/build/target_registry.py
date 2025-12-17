"""Unified registry for build targets with Hamilton-derived dependencies.

This module defines TargetRegistry, a thin wrapper that bundles:

- A TargetGraph containing OutputTargets with dependency edges derived from the
  executable Hamilton DAG.
- The raw derived dependency mapping produced by Hamilton graph introspection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.build.hamilton.introspect import (
    derive_target_dependencies,
    target_graph_from_hamilton,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from codeintel.build.hamilton.driver_factory import HamiltonRuntime
    from codeintel.build.targets import OutputTarget, TargetGraph


@dataclass(frozen=True)
class TargetRegistry:
    """Registry of OutputTargets whose dependencies match the Hamilton DAG.

    Parameters
    ----------
    graph
        TargetGraph containing OutputTargets with Hamilton-derived dependencies.
    derived_dependencies
        Mapping from target name to a sorted tuple of direct dependency target names.
    """

    graph: TargetGraph
    derived_dependencies: dict[str, tuple[str, ...]]

    @classmethod
    def from_hamilton(
        cls,
        runtime: HamiltonRuntime,
        *,
        base_graph: TargetGraph | None = None,
        strict: bool = False,
    ) -> TargetRegistry:
        """Build a registry from a configured Hamilton runtime.

        Parameters
        ----------
        runtime
            Hamilton runtime containing a configured Driver.
        base_graph
            Optional TargetGraph providing OutputTarget metadata. Defaults to ``runtime.graph``.
        strict
            When True, raise if the Hamilton graph is missing materialize nodes for any targets
            in ``base_graph``.

        Returns
        -------
        TargetRegistry
            Registry containing a graph with Hamilton-derived dependencies.
        """
        base = runtime.graph if base_graph is None else base_graph
        derived = derive_target_dependencies(runtime)
        graph = target_graph_from_hamilton(
            runtime,
            base_graph=base,
            derived_deps=derived,
            strict=strict,
        )

        return cls(graph=graph, derived_dependencies=derived)

    def get(self, name: str) -> OutputTarget | None:
        """Return target metadata by name.

        Parameters
        ----------
        name
            Target name.

        Returns
        -------
        OutputTarget | None
            Target metadata, or None when target is not registered.
        """
        try:
            return self.graph.get(name)
        except KeyError:
            return None

    def dependencies(self, name: str) -> tuple[str, ...]:
        """Return direct dependency target names.

        Parameters
        ----------
        name
            Target name.

        Returns
        -------
        tuple[str, ...]
            Direct dependencies for the target, or empty tuple if unknown.
        """
        deps = self.derived_dependencies.get(name)
        if deps is not None:
            return deps

        target = self.get(name)
        return target.dependencies if target is not None else ()

    def closure(self, names: Iterable[str]) -> tuple[str, ...]:
        """Compute a dependency closure in topological order.

        Parameters
        ----------
        names
            Target names to compute the closure for.

        Returns
        -------
        tuple[str, ...]
            Closure in dependency order (dependencies first).
        """
        return self.graph.topological_order(names)

    @property
    def all_targets(self) -> tuple[OutputTarget, ...]:
        """Return all targets in the registry.

        Returns
        -------
        tuple[OutputTarget, ...]
            All registered targets.
        """
        return self.graph.all_targets


__all__ = [
    "TargetRegistry",
]
