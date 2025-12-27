"""Immutable catalog derived from the Hamilton FunctionGraph."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, cast

from codeintel.build.hamilton.tag_spec import TagSpec

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from codeintel.build.parameters import TargetParameters
    from codeintel.build.resources import TargetExecution, TargetResources
    from codeintel.build.targets import TargetModule


OutputKind = Literal["table", "artifact"]
OutputRole = Literal["contract", "internal"]


@dataclass(frozen=True, slots=True)
class NodeDescriptor:
    """Hamilton node descriptor with parsed tags and dependencies."""

    name: str
    deps: tuple[str, ...]
    tags: Mapping[str, object]
    tag_spec: TagSpec | None = None


@dataclass(frozen=True, slots=True)
class OutputDescriptor:
    """Single output declared by a data saver node."""

    kind: OutputKind
    key: str
    role: OutputRole
    producer_target: str
    saver_node: str
    sink: str
    artifact_path_template: str | None = None
    tags: Mapping[str, object] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class TableRead:
    """Table read derived from loader/dataset nodes."""

    table_key: str
    producer_target: str | None
    loader_node: str
    loader_type: str


@dataclass(frozen=True, slots=True)
class TableWrite:
    """Table write derived from data saver nodes."""

    table_key: str
    sink: str
    saver_node: str


@dataclass(frozen=True, slots=True)
class ArtifactWrite:
    """Artifact write derived from data saver nodes."""

    artifact_name: str
    sink: str
    saver_node: str


@dataclass(frozen=True, slots=True)
class IOSurface:
    """Read/write surface for a target."""

    target: str
    reads: tuple[TableRead, ...]
    table_writes: tuple[TableWrite, ...]
    artifact_writes: tuple[ArtifactWrite, ...]


@dataclass(frozen=True, slots=True)
class TargetDescriptor:
    """Immutable target descriptor derived from Hamilton tags."""

    name: str
    module: TargetModule
    anchor_node: str
    dependencies: tuple[str, ...]
    resources: TargetResources
    execution: TargetExecution
    parameters: TargetParameters
    description: str
    spec_version: str
    table_keys: tuple[str, ...] = ()
    artifact_names: tuple[str, ...] = ()

    @property
    def estimated_duration_ms(self) -> int:
        """Return estimated execution duration in milliseconds."""
        return self.execution.estimated_duration_ms()


@dataclass(frozen=True, slots=True)
class DagCatalog:
    """Compiled view over the Hamilton FunctionGraph."""

    nodes: Mapping[str, NodeDescriptor]
    targets: Mapping[str, TargetDescriptor]
    target_nodes: Mapping[str, str]
    node_to_target: Mapping[str, str]
    target_dependencies: Mapping[str, tuple[str, ...]]
    target_dependents: Mapping[str, tuple[str, ...]]
    table_outputs: Mapping[str, OutputDescriptor]
    artifact_outputs: Mapping[str, OutputDescriptor]
    table_outputs_by_target: Mapping[str, tuple[OutputDescriptor, ...]]
    artifact_outputs_by_target: Mapping[str, tuple[OutputDescriptor, ...]]
    io_surfaces: Mapping[str, IOSurface]

    def __contains__(self, target_name: str) -> bool:
        """Return True if a target is present in the catalog.

        Parameters
        ----------
        target_name
            Target name to check.

        Returns
        -------
        bool
            True when the catalog contains the target.
        """
        return target_name in self.targets

    def __iter__(self) -> Iterator[str]:
        """Return an iterator over target names.

        Returns
        -------
        Iterator[str]
            Iterator of target names in the catalog.
        """
        return iter(self.targets)

    def __len__(self) -> int:
        """Return the number of targets in the catalog.

        Returns
        -------
        int
            Count of targets.
        """
        return len(self.targets)

    @property
    def all_targets(self) -> tuple[TargetDescriptor, ...]:
        """Return all targets in deterministic order.

        Returns
        -------
        tuple[TargetDescriptor, ...]
            Targets sorted by name.
        """
        return tuple(self.targets[name] for name in sorted(self.targets))

    def target_node(self, target_name: str) -> str:
        """Return the anchor node name for a target.

        Parameters
        ----------
        target_name
            Target name to resolve.

        Returns
        -------
        str
            Hamilton anchor node name.

        Raises
        ------
        KeyError
            If the target is not present in the catalog.
        """
        node = self.target_nodes.get(target_name)
        if node is None:
            msg = f"Target not found in catalog: {target_name}"
            raise KeyError(msg)
        return node

    def dependencies_of(self, target_name: str) -> tuple[str, ...]:
        """Return direct dependencies for a target.

        Parameters
        ----------
        target_name
            Target name to resolve.

        Returns
        -------
        tuple[str, ...]
            Direct dependencies for the target.

        Raises
        ------
        KeyError
            If the target is not present in the catalog.
        """
        if target_name not in self.targets:
            msg = f"Target not found in catalog: {target_name}"
            raise KeyError(msg)
        return self.target_dependencies.get(target_name, ())

    def get(self, target_name: str) -> TargetDescriptor:
        """Return a target descriptor by name.

        Parameters
        ----------
        target_name
            Target name to resolve.

        Returns
        -------
        TargetDescriptor
            Target metadata for the requested name.

        Raises
        ------
        KeyError
            If the target is not present in the catalog.
        """
        if target_name not in self.targets:
            msg = f"Target not found in catalog: {target_name}"
            raise KeyError(msg)
        return self.targets[target_name]

    def get_target(self, target_name: str) -> TargetDescriptor | None:
        """Return a target descriptor by name, or None if missing.

        Parameters
        ----------
        target_name
            Target name to resolve.

        Returns
        -------
        TargetDescriptor | None
            Target descriptor if present, otherwise None.
        """
        return self.targets.get(target_name)

    def dependents_of(self, target_name: str) -> tuple[str, ...]:
        """Return direct dependents for a target.

        Parameters
        ----------
        target_name
            Target name to resolve.

        Returns
        -------
        tuple[str, ...]
            Direct dependents for the target.

        Raises
        ------
        KeyError
            If the target is not present in the catalog.
        """
        if target_name not in self.targets:
            msg = f"Target not found in catalog: {target_name}"
            raise KeyError(msg)
        return self.target_dependents.get(target_name, ())

    def closure(self, targets: Sequence[str]) -> tuple[str, ...]:
        """Return dependency closure in topological order.

        Parameters
        ----------
        targets
            Target names to compute the closure for.

        Returns
        -------
        tuple[str, ...]
            Dependency closure ordered by dependencies-first.

        Raises
        ------
        ValueError
            If a cycle is detected in the target dependencies.
        """
        all_names: set[str] = set()
        for name in targets:
            all_names.add(name)
            all_names.update(self._transitive_deps(name))

        in_degree: dict[str, int] = dict.fromkeys(all_names, 0)
        for name in all_names:
            for dep in self.target_dependencies.get(name, ()):
                if dep in all_names:
                    in_degree[name] += 1

        queue = [name for name, degree in in_degree.items() if degree == 0]
        result: list[str] = []

        while queue:
            queue.sort()
            current = queue.pop(0)
            result.append(current)
            for dependent in self.target_dependents.get(current, ()):
                if dependent in in_degree:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)

        if len(result) != len(all_names):
            remaining = sorted(all_names - set(result))
            msg = f"Cycle detected in target dependencies: {remaining}"
            raise ValueError(msg)

        return tuple(result)

    def validate(self) -> tuple[str, ...]:
        """Validate catalog integrity.

        Returns
        -------
        tuple[str, ...]
            Validation error messages (empty when valid).
        """
        errors = [
            f"Target '{target_name}' depends on unknown target '{dep}'"
            for target_name, deps in self.target_dependencies.items()
            for dep in deps
            if dep not in self.targets
        ]
        if not errors:
            try:
                self.closure(tuple(self.targets))
            except ValueError as exc:
                errors.append(str(exc))
        return tuple(errors)

    def producer_of_table(self, table_key: str) -> str | None:
        """Return producer target for a table key.

        Parameters
        ----------
        table_key
            Fully-qualified table key.

        Returns
        -------
        str | None
            Producer target name when known, otherwise None.
        """
        output = self.table_outputs.get(table_key)
        if output is None:
            return None
        return output.producer_target

    def producer_of_artifact(self, artifact_name: str) -> str | None:
        """Return producer target for an artifact name.

        Parameters
        ----------
        artifact_name
            Artifact name to resolve.

        Returns
        -------
        str | None
            Producer target name when known, otherwise None.
        """
        output = self.artifact_outputs.get(artifact_name)
        if output is None:
            return None
        return output.producer_target

    def outputs_for_target(self, target_name: str) -> tuple[OutputDescriptor, ...]:
        """Return contract outputs for a target.

        Parameters
        ----------
        target_name
            Target name to resolve.

        Returns
        -------
        tuple[OutputDescriptor, ...]
            Contract outputs for the target.
        """
        tables = self.table_outputs_by_target.get(target_name, ())
        artifacts = self.artifact_outputs_by_target.get(target_name, ())
        return (*tables, *artifacts)

    def find_nodes(
        self,
        tag_key: str,
        tag_value: object | None = None,
    ) -> tuple[NodeDescriptor, ...]:
        """Find nodes by tag key/value.

        Parameters
        ----------
        tag_key
            Tag key to search for.
        tag_value
            Optional tag value to match exactly.

        Returns
        -------
        tuple[NodeDescriptor, ...]
            Nodes matching the tag criteria.
        """
        matches: list[NodeDescriptor] = []
        for node in self.nodes.values():
            if tag_key not in node.tags:
                continue
            if tag_value is None or node.tags.get(tag_key) == tag_value:
                matches.append(node)
        return tuple(matches)

    def targets_for_module(self, module: TargetModule) -> tuple[TargetDescriptor, ...]:
        """Return targets for a specific module.

        Parameters
        ----------
        module
            Target module name to filter by.

        Returns
        -------
        tuple[TargetDescriptor, ...]
            Targets in the requested module.
        """
        return tuple(t for t in self.all_targets if t.module == module)

    def all_table_keys(self) -> frozenset[str]:
        """Return all contract table keys.

        Returns
        -------
        frozenset[str]
            Set of all contract table keys.
        """
        return frozenset(self.table_outputs)

    def all_artifact_names(self) -> frozenset[str]:
        """Return all contract artifact names.

        Returns
        -------
        frozenset[str]
            Set of all contract artifact names.
        """
        return frozenset(self.artifact_outputs)

    def _transitive_deps(self, target_name: str) -> frozenset[str]:
        result: set[str] = set()
        stack = list(self.target_dependencies.get(target_name, ()))
        while stack:
            dep = stack.pop()
            if dep in result:
                continue
            result.add(dep)
            stack.extend(self.target_dependencies.get(dep, ()))
        return frozenset(result)


def freeze_mapping[T](mapping: Mapping[str, T]) -> Mapping[str, T]:
    """Return an immutable view of a mapping.

    Returns
    -------
    Mapping[str, T]
        Frozen mapping view.
    """
    return cast("Mapping[str, T]", MappingProxyType(dict(mapping)))


__all__ = [
    "ArtifactWrite",
    "DagCatalog",
    "IOSurface",
    "NodeDescriptor",
    "OutputDescriptor",
    "OutputKind",
    "OutputRole",
    "TableRead",
    "TableWrite",
    "TargetDescriptor",
    "freeze_mapping",
]
