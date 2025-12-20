"""Canonical target metadata service for build target access."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, cast

from codeintel.build.hamilton.introspect import derive_target_outputs
from codeintel.build.hamilton.tag_index import TagIndex
from codeintel.core.imports.lazy import lazy_getattr
from codeintel.core.schemas.declared import source_declared_schema_provider

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from types import ModuleType

    from codeintel.build.hamilton.driver_factory import HamiltonRuntime
    from codeintel.build.schemas.inference_service import SchemaInferenceService
    from codeintel.build.schemas.schema_index import SchemaIndex
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

    def closure(self, targets: Sequence[str]) -> tuple[str, ...]:
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
def _load_target_system() -> TargetSystem:
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


@dataclass(frozen=True, slots=True)
class OutputInventory:
    """Derived output inventory for build targets."""

    datasets_by_target: Mapping[str, tuple[str, ...]]
    artifacts_by_target: Mapping[str, tuple[str, ...]]

    def datasets_for(self, target_name: str) -> tuple[str, ...]:
        """Return dataset table keys for a target.

        Parameters
        ----------
        target_name
            Target name to query.

        Returns
        -------
        tuple[str, ...]
            Dataset table keys for the target.
        """
        return self.datasets_by_target.get(target_name, ())

    def artifacts_for(self, target_name: str) -> tuple[str, ...]:
        """Return artifact names for a target.

        Parameters
        ----------
        target_name
            Target name to query.

        Returns
        -------
        tuple[str, ...]
            Artifact names for the target.
        """
        return self.artifacts_by_target.get(target_name, ())

    @property
    def all_dataset_keys(self) -> frozenset[str]:
        """Return all dataset table keys across targets.

        Returns
        -------
        frozenset[str]
            Unique dataset table keys.
        """
        return frozenset(key for keys in self.datasets_by_target.values() for key in keys)

    @property
    def all_artifact_names(self) -> frozenset[str]:
        """Return all artifact names across targets.

        Returns
        -------
        frozenset[str]
            Unique artifact names.
        """
        return frozenset(name for names in self.artifacts_by_target.values() for name in names)


@dataclass(frozen=True, slots=True)
class TargetMetadataService:
    """Bundle of target system, outputs, and tag index."""

    system: TargetSystem
    outputs: OutputInventory
    tag_index: TagIndex
    schema_index: SchemaIndex

    def get_target(self, name: str) -> OutputTarget | None:
        """Return target metadata by name.

        Parameters
        ----------
        name
            Target name to resolve.

        Returns
        -------
        OutputTarget | None
            Target metadata if found.
        """
        return self.system.get_target(name)

    def target_for_table_key(self, table_key: str) -> OutputTarget | None:
        """Return the target that produces a dataset table key.

        Parameters
        ----------
        table_key
            Fully qualified table key (schema.table).

        Returns
        -------
        OutputTarget | None
            Target metadata if found.
        """
        return self.system.target_for_table_key(table_key)

    def target_for_artifact(self, artifact_name: str) -> OutputTarget | None:
        """Return the target that produces an artifact name.

        Parameters
        ----------
        artifact_name
            Artifact name to resolve.

        Returns
        -------
        OutputTarget | None
            Target metadata if found.
        """
        return self.system.target_for_artifact(artifact_name)


class TargetMetadataProvider(Protocol):
    """Protocol for resolving target metadata."""

    def get_target(self, name: str) -> OutputTarget | None:
        """Return target metadata by name."""
        ...

    def target_for_table_key(self, table_key: str) -> OutputTarget | None:
        """Return target metadata for a table key."""
        ...

    def target_for_artifact(self, artifact_name: str) -> OutputTarget | None:
        """Return target metadata for an artifact name."""
        ...


@dataclass(slots=True)
class LazyTargetMetadataProvider:
    """Lazy provider that loads the target metadata service on demand."""

    factory: Callable[[], TargetMetadataService]
    _service: TargetMetadataService | None = None

    def _resolve(self) -> TargetMetadataService:
        if self._service is None:
            self._service = self.factory()
        return self._service

    def get_target(self, name: str) -> OutputTarget | None:
        """Return target metadata by name."""
        return self._resolve().get_target(name)

    def target_for_table_key(self, table_key: str) -> OutputTarget | None:
        """Return target metadata for a table key."""
        return self._resolve().target_for_table_key(table_key)

    def target_for_artifact(self, artifact_name: str) -> OutputTarget | None:
        """Return target metadata for an artifact name."""
        return self._resolve().target_for_artifact(artifact_name)


@lru_cache(maxsize=1)
def get_target_metadata_service() -> TargetMetadataService:
    """Return the canonical target metadata service.

    Returns
    -------
    TargetMetadataService
        Singleton target metadata service.
    """
    system = _load_target_system()
    build_schema_index = cast(
        "Callable[..., SchemaIndex]",
        lazy_getattr("codeintel.build.schemas.schema_index", "build_schema_index"),
    )
    get_schema_inference_service = cast(
        "Callable[[], SchemaInferenceService]",
        lazy_getattr("codeintel.build.schemas.inference_service", "get_schema_inference_service"),
    )
    schema_index = build_schema_index(
        system=system,
        declared_provider=source_declared_schema_provider(
            exclude_table_keys=system.all_table_keys,
        ),
        inference_service=get_schema_inference_service(),
    )
    derived = derive_target_outputs(system.runtime)
    inventory = OutputInventory(
        datasets_by_target=derived.datasets_by_target,
        artifacts_by_target=derived.artifacts_by_target,
    )
    tag_index = TagIndex.from_runtime(system.runtime)
    return TargetMetadataService(
        system=system,
        outputs=inventory,
        tag_index=tag_index,
        schema_index=schema_index,
    )


def get_target_metadata_provider() -> TargetMetadataProvider:
    """Return a lazy target metadata provider."""
    return LazyTargetMetadataProvider(get_target_metadata_service)


def is_target_metadata_loaded() -> bool:
    """Return True if the target metadata service has been initialized."""
    return (
        _load_target_system.cache_info().currsize > 0
        or get_target_metadata_service.cache_info().currsize > 0
    )


def clear_target_metadata_cache() -> None:
    """Clear cached target metadata services."""
    _load_target_system.cache_clear()
    get_target_metadata_service.cache_clear()


__all__ = [
    "OutputInventory",
    "TargetMetadataProvider",
    "TargetMetadataService",
    "TargetSystem",
    "clear_target_metadata_cache",
    "get_target_metadata_service",
    "get_target_metadata_provider",
    "is_target_metadata_loaded",
]
