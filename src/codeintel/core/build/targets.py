"""Target model and dependency graph for the build system.

This module defines the core abstractions for tracking what outputs
the build system can produce and their interdependencies.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Literal

TargetModule = Literal["ingestion", "graphs", "analytics"]
"""Classification of which pipeline module produces a target."""


@dataclass(frozen=True)
class OutputTarget:
    """A discrete output that can be requested and validated.

    Each target represents a logical output that the build system can
    produce. Targets have dependencies on other targets, forming a DAG.

    Attributes
    ----------
    name
        Canonical target identifier (e.g., "function.metrics").
    module
        Which pipeline module produces this target.
    plugin
        Plugin name that produces this target.
    tables
        DuckDB tables this target writes to.
    dependencies
        Other OutputTarget names that must be computed first.
    description
        Human-readable description.
    estimated_duration_ms
        Typical execution time in milliseconds (for planning).

    Examples
    --------
    >>> target = OutputTarget(
    ...     name="risk_factors",
    ...     module="analytics",
    ...     plugin="risk_factors_plugin",
    ...     tables=("analytics.goid_risk_factors",),
    ...     dependencies=("function_metrics", "coverage"),
    ...     description="Risk factors per function",
    ... )
    """

    name: str
    module: TargetModule
    plugin: str
    tables: tuple[str, ...]
    dependencies: tuple[str, ...] = ()
    description: str = ""
    estimated_duration_ms: int | None = None


@dataclass
class TargetGraph:
    """Complete dependency graph of all output targets.

    Provides methods for dependency resolution, topological sorting,
    and target lookup. This is the core data structure for computing
    minimal execution plans.

    Attributes
    ----------
    _targets
        Internal mapping of target names to OutputTarget instances.
    _dependents
        Internal mapping of target names to their dependents.

    Examples
    --------
    >>> graph = TargetGraph()
    >>> graph.register(OutputTarget(
    ...     name="modules",
    ...     module="ingestion",
    ...     plugin="repo_scan",
    ...     tables=("core.modules",),
    ... ))
    >>> "modules" in graph
    True
    """

    _targets: dict[str, OutputTarget] = field(default_factory=dict)
    _dependents: dict[str, set[str]] = field(default_factory=dict)

    def register(self, target: OutputTarget) -> None:
        """Register a target in the graph.

        Parameters
        ----------
        target
            The OutputTarget to register.

        Raises
        ------
        ValueError
            If a target with the same name already exists.
        """
        if target.name in self._targets:
            msg = f"Target '{target.name}' is already registered"
            raise ValueError(msg)
        self._targets[target.name] = target
        # Initialize dependents set for this target
        if target.name not in self._dependents:
            self._dependents[target.name] = set()
        # Register as dependent for each of its dependencies
        for dep in target.dependencies:
            if dep not in self._dependents:
                self._dependents[dep] = set()
            self._dependents[dep].add(target.name)

    def get(self, name: str) -> OutputTarget:
        """Get a target by name.

        Parameters
        ----------
        name
            Target name to look up.

        Returns
        -------
        OutputTarget
            The target with the given name.

        Raises
        ------
        KeyError
            If the target is not found.
        """
        if name not in self._targets:
            msg = f"Target '{name}' not found in graph"
            raise KeyError(msg)
        return self._targets[name]

    def __contains__(self, name: str) -> bool:
        """Check if a target exists in the graph.

        Parameters
        ----------
        name
            Target name to check.

        Returns
        -------
        bool
            True if the target exists.
        """
        return name in self._targets

    def __len__(self) -> int:
        """Return the number of targets in the graph.

        Returns
        -------
        int
            Number of registered targets.
        """
        return len(self._targets)

    def __iter__(self) -> Iterator[str]:
        """Iterate over target names.

        Returns
        -------
        Iterator[str]
            Iterator over target names in arbitrary order.
        """
        return iter(self._targets)

    @property
    def all_targets(self) -> tuple[OutputTarget, ...]:
        """Return all registered targets.

        Returns
        -------
        tuple[OutputTarget, ...]
            All targets in arbitrary order.
        """
        return tuple(self._targets.values())

    def dependencies_of(self, name: str) -> tuple[str, ...]:
        """Return direct dependencies of a target.

        Parameters
        ----------
        name
            Target name to get dependencies for.

        Returns
        -------
        tuple[str, ...]
            Names of direct dependencies.
        """
        return self.get(name).dependencies

    def transitive_deps(self, name: str) -> frozenset[str]:
        """Return all transitive dependencies (not including the target itself).

        Parameters
        ----------
        name
            Target name to get transitive dependencies for.

        Returns
        -------
        frozenset[str]
            Names of all transitive dependencies.
        """
        result: set[str] = set()
        stack = list(self.get(name).dependencies)
        while stack:
            dep = stack.pop()
            if dep in result:
                continue
            result.add(dep)
            if dep in self._targets:
                stack.extend(self._targets[dep].dependencies)
        return frozenset(result)

    def dependents_of(self, name: str) -> tuple[str, ...]:
        """Return targets that depend on this one.

        Parameters
        ----------
        name
            Target name to get dependents for.

        Returns
        -------
        tuple[str, ...]
            Names of targets that directly depend on this target.
        """
        return tuple(sorted(self._dependents.get(name, set())))

    def topological_order(self, names: Iterable[str]) -> tuple[str, ...]:
        """Sort targets in dependency order (dependencies first).

        Uses Kahn's algorithm for topological sorting.

        Parameters
        ----------
        names
            Target names to sort.

        Returns
        -------
        tuple[str, ...]
            Target names in topological order.

        Raises
        ------
        ValueError
            If a cycle is detected in the dependencies.
        """
        # Expand to include all transitive dependencies
        all_names: set[str] = set()
        for name in names:
            all_names.add(name)
            all_names.update(self.transitive_deps(name))

        # Build in-degree map for the subgraph
        in_degree: dict[str, int] = dict.fromkeys(all_names, 0)
        for name in all_names:
            for dep in self.get(name).dependencies:
                if dep in all_names:
                    in_degree[name] += 1

        # Kahn's algorithm
        queue = [name for name, degree in in_degree.items() if degree == 0]
        result: list[str] = []

        while queue:
            # Sort queue for deterministic output
            queue.sort()
            current = queue.pop(0)
            result.append(current)

            for dependent in self._dependents.get(current, set()):
                if dependent in in_degree:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)

        if len(result) != len(all_names):
            computed = set(result)
            remaining = all_names - computed
            msg = f"Cycle detected in dependencies involving: {sorted(remaining)}"
            raise ValueError(msg)

        return tuple(result)

    def targets_for_module(self, module: TargetModule) -> tuple[OutputTarget, ...]:
        """Return all targets for a specific module.

        Parameters
        ----------
        module
            Module type to filter by.

        Returns
        -------
        tuple[OutputTarget, ...]
            Targets belonging to the specified module.
        """
        return tuple(t for t in self._targets.values() if t.module == module)

    def validate(self) -> tuple[str, ...]:
        """Validate graph integrity.

        Check for:
        - Missing dependencies (references to non-existent targets)
        - Cycles in the dependency graph

        Returns
        -------
        tuple[str, ...]
            Error messages (empty if valid).
        """
        # Check for missing dependencies
        errors: list[str] = [
            f"Target '{target.name}' depends on unknown target '{dep}'"
            for target in self._targets.values()
            for dep in target.dependencies
            if dep not in self._targets
        ]

        # Check for cycles using topological sort
        if not errors:
            try:
                self.topological_order(self._targets.keys())
            except ValueError as e:
                errors.append(str(e))

        return tuple(errors)


__all__ = [
    "OutputTarget",
    "TargetGraph",
    "TargetModule",
]
