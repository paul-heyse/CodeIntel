"""Canonical build target entrypoint.

This module defines :class:`TargetSystem`, the single entrypoint for accessing:

- The Hamilton runtime (Driver + node mappings)
- The Hamilton-derived dependency graph (TargetGraph)
- Indexes for resolving targets by name/table_key/artifact_name

It replaces overlapping concepts previously spread across registry/catalog wrappers.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence
    from types import ModuleType

    from codeintel.build.hamilton.driver_factory import HamiltonRuntime
    from codeintel.build.targets import OutputTarget, TargetGraph


@dataclass(frozen=True, slots=True)
class TargetSystem:
    """Bundle runtime, graph, and target lookup indexes."""

    runtime: HamiltonRuntime
    graph: TargetGraph
    by_name: Mapping[str, OutputTarget]
    by_table_key: Mapping[str, OutputTarget]
    by_artifact_name: Mapping[str, OutputTarget]

    def get_target(self, name: str) -> OutputTarget | None:
        """Return target metadata by name.

        Parameters
        ----------
        name
            Target name.

        Returns
        -------
        OutputTarget | None
            Target metadata if present, otherwise None.
        """
        return self.by_name.get(name)

    def closure(self, targets: Iterable[str]) -> tuple[str, ...]:
        """Return dependency closure in topological order.

        Parameters
        ----------
        targets
            Target names to compute the closure for.

        Returns
        -------
        tuple[str, ...]
            Dependency closure in topological order.
        """
        return self.graph.topological_order(targets)

    def target_for_table_key(self, table_key: str) -> OutputTarget | None:
        """Return producing target for a table key.

        Parameters
        ----------
        table_key
            Fully-qualified table key (schema.table).

        Returns
        -------
        OutputTarget | None
            Producing target if present, otherwise None.
        """
        return self.by_table_key.get(table_key)

    def target_for_artifact(self, artifact_name: str) -> OutputTarget | None:
        """Return producing target for an artifact name.

        Parameters
        ----------
        artifact_name
            Artifact name declared in a target contract.

        Returns
        -------
        OutputTarget | None
            Producing target if present, otherwise None.
        """
        return self.by_artifact_name.get(artifact_name)

    @property
    def all_table_keys(self) -> frozenset[str]:
        """Return all produced table keys declared by the target graph."""
        return frozenset(self.by_table_key)

    @property
    def all_artifact_names(self) -> frozenset[str]:
        """Return all produced artifact names declared by the target graph."""
        return frozenset(self.by_artifact_name)


def _build_indexes(
    targets: Sequence[OutputTarget],
) -> tuple[Mapping[str, OutputTarget], Mapping[str, OutputTarget], Mapping[str, OutputTarget]]:
    by_name: dict[str, OutputTarget] = {}
    by_table_key: dict[str, OutputTarget] = {}
    by_artifact_name: dict[str, OutputTarget] = {}

    for target in targets:
        by_name[target.name] = target
        for table_key in target.contract.table_keys:
            existing = by_table_key.get(table_key)
            if existing is not None and existing.name != target.name:
                msg = (
                    "Duplicate table_key declared by multiple targets: "
                    f"{table_key} ({existing.name}, {target.name})"
                )
                raise ValueError(msg)
            by_table_key[table_key] = target

        for artifact_name in target.contract.artifact_names:
            existing = by_artifact_name.get(artifact_name)
            if existing is not None and existing.name != target.name:
                msg = (
                    "Duplicate artifact name declared by multiple targets: "
                    f"{artifact_name} ({existing.name}, {target.name})"
                )
                raise ValueError(msg)
            by_artifact_name[artifact_name] = target

    return (
        MappingProxyType(by_name),
        MappingProxyType(by_table_key),
        MappingProxyType(by_artifact_name),
    )


@lru_cache(maxsize=1)
def load_target_system() -> TargetSystem:
    """Load the singleton TargetSystem for the current process.

    Returns
    -------
    TargetSystem
        Loaded TargetSystem (runtime + graph + indexes).

    Raises
    ------
    TypeError
        If the Hamilton driver factory does not expose a callable ``build_driver`` function.
    """
    driver_factory_mod: ModuleType = importlib.import_module(
        "codeintel.build.hamilton.driver_factory"
    )
    build_driver_fn_raw = getattr(driver_factory_mod, "build_driver", None)
    if not callable(build_driver_fn_raw):
        msg = "codeintel.build.hamilton.driver_factory.build_driver is missing or not callable"
        raise TypeError(msg)

    build_driver_fn = cast("Callable[[], HamiltonRuntime]", build_driver_fn_raw)
    runtime = build_driver_fn()
    graph = runtime.graph

    by_name, by_table_key, by_artifact_name = _build_indexes(graph.all_targets)

    return TargetSystem(
        runtime=runtime,
        graph=graph,
        by_name=by_name,
        by_table_key=by_table_key,
        by_artifact_name=by_artifact_name,
    )


__all__ = [
    "TargetSystem",
    "load_target_system",
]
