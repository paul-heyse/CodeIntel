"""Target model and dependency graph for the build system.

This module defines the core abstractions for tracking what outputs
the build system can produce and their interdependencies.

The OutputTarget is now the single source of truth for:
- What tables/artifacts a target produces (contract)
- What resources it needs (resources)
- How it should be executed (execution)
- Tuning parameters (parameters)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from codeintel.build.contracts import EMPTY_CONTRACT, OutputContract
from codeintel.build.errors import CycleDetectedError
from codeintel.build.parameters import EMPTY_PARAMETERS
from codeintel.build.resources import (
    DEFAULT_EXECUTION,
    DEFAULT_RESOURCES,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from codeintel.build.contracts import ArtifactSpec
    from codeintel.build.parameters import TargetParameters
    from codeintel.build.resources import (
        TargetExecution,
        TargetResources,
    )

TargetModule = Literal["ingestion", "graphs", "analytics", "export"]
"""Classification of which target module produces an output."""


@dataclass(frozen=True)
class TargetOptions:
    """Optional configuration overrides for OutputTarget factories."""

    artifacts: tuple[ArtifactSpec, ...] = ()
    dependencies: tuple[str, ...] = ()
    resources: TargetResources = DEFAULT_RESOURCES
    execution: TargetExecution = DEFAULT_EXECUTION
    parameters: TargetParameters = EMPTY_PARAMETERS
    description: str = ""


@dataclass(frozen=True)
class OutputTarget:
    """A discrete output that can be requested and validated.

    Each target represents a logical output that the build system can
    produce. Targets have dependencies on other targets, forming a DAG.

    The OutputTarget is the single source of truth for what a target
    produces and how it should be executed. Plugins receive all their
    configuration from the target via TargetExecutionContext.

    Attributes
    ----------
    name
        Canonical target identifier (e.g., "function_metrics").
    module
        Which target module produces this output.
    plugin
        Plugin name that produces this target.
    contract
        Output contract defining tables and artifacts produced.
        This is authoritative; prefer `contract.table_keys` over legacy
        shortcuts when referring to outputs.
    dependencies
        Other OutputTarget names that must be computed first.
    resources
        Resources required for execution (tracker, tools, etc.).
    execution
        Execution configuration (isolation, timeouts, etc.).
    parameters
        Tuning parameters for this target.
    description
        Human-readable description.

    Examples
    --------
    >>> from codeintel.build.contracts import OutputContract, ArtifactSpec
    >>> from codeintel.build.resources import TargetResources, TargetExecution
    >>> from codeintel.config.datasets.primitives import TableSchema, Column
    >>> target = OutputTarget(
    ...     name="scip",
    ...     module="ingestion",
    ...     plugin="scip_ingest",
    ...     contract=OutputContract(
    ...         tables=(TableSchema("core", "goids", [Column("goid_h128", "DECIMAL(38,0)")]),),
    ...         artifacts=(ArtifactSpec("scip_index", "{scip_dir}/index.scip"),),
    ...     ),
    ...     dependencies=("modules",),
    ...     resources=TargetResources(tracker=True, tools=("scip-python",)),
    ...     execution=TargetExecution(cpu_intensive=True, isolation="process"),
    ...     description="SCIP index ingestion",
    ... )
    """

    name: str
    module: TargetModule
    plugin: str

    contract: OutputContract = field(default_factory=lambda: EMPTY_CONTRACT)
    dependencies: tuple[str, ...] = ()

    resources: TargetResources = field(default_factory=lambda: DEFAULT_RESOURCES)

    execution: TargetExecution = field(default_factory=lambda: DEFAULT_EXECUTION)

    parameters: TargetParameters = field(default_factory=lambda: EMPTY_PARAMETERS)
    description: str = ""

    def __post_init__(self) -> None:
        """Validate contract structure after initialization.

        Raises
        ------
        ValueError
            If the contract contains structural validation errors.
        """
        errors = self.contract.validate()
        if errors:
            message = "; ".join(errors)
            raise ValueError(message)

    @classmethod
    def from_tables(
        cls,
        *,
        name: str,
        module: TargetModule,
        plugin: str,
        tables: Iterable[str],
        options: TargetOptions | None = None,
    ) -> OutputTarget:
        """Create an OutputTarget from table keys and optional artifacts.

        This factory provides compatibility for legacy call sites that
        previously passed ``tables=...`` directly to the constructor.

        Returns
        -------
        OutputTarget
            Target configured with a simple contract derived from table keys.
        """
        opts = options or TargetOptions()
        return cls(
            name=name,
            module=module,
            plugin=plugin,
            contract=OutputContract.simple(table_keys=tables, artifacts=opts.artifacts),
            dependencies=opts.dependencies,
            resources=opts.resources,
            execution=opts.execution,
            parameters=opts.parameters,
            description=opts.description,
        )

    @property
    def table_keys(self) -> tuple[str, ...]:
        """Return table keys from contract.

        Returns
        -------
        tuple[str, ...]
            Fully-qualified table names this target writes to.
            Returns empty tuple for artifact-only targets.
        """
        return self.contract.table_keys

    @property
    def estimated_duration_ms(self) -> int:
        """Return estimated duration from execution config.

        Returns
        -------
        int
            Estimated execution duration in milliseconds.
        """
        return self.execution.estimated_duration_ms()


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
    >>> from codeintel.config.datasets.primitives import TableSchema, Column
    >>> graph.register(
    ...     OutputTarget(
    ...         name="modules",
    ...         module="ingestion",
    ...         plugin="repo_scan",
    ...         contract=OutputContract(
    ...             tables=(TableSchema("core", "modules", [Column("module", "VARCHAR")]),)
    ...         ),
    ...     )
    ... )
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

        if target.name not in self._dependents:
            self._dependents[target.name] = set()

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
        CycleDetectedError
            If a cycle is detected in the dependencies.
        """
        all_names: set[str] = set()
        for name in names:
            all_names.add(name)
            all_names.update(self.transitive_deps(name))

        in_degree: dict[str, int] = dict.fromkeys(all_names, 0)
        for name in all_names:
            for dep in self.get(name).dependencies:
                if dep in all_names:
                    in_degree[name] += 1

        queue = [name for name, degree in in_degree.items() if degree == 0]
        result: list[str] = []

        while queue:
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
            raise CycleDetectedError(sorted(remaining))

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
        errors: list[str] = [
            f"Target '{target.name}' depends on unknown target '{dep}'"
            for target in self._targets.values()
            for dep in target.dependencies
            if dep not in self._targets
        ]

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
