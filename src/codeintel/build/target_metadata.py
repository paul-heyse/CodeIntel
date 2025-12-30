"""Canonical target metadata service for build target access."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol

from codeintel.build.hamilton.dag_catalog import DagCatalog, OutputDescriptor, TargetDescriptor
from codeintel.build.schemas.schema_index import SchemaIndex
from codeintel.runtime.runtime_bundle import RuntimeBundle

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class TargetSystem:
    """Bundle runtime, catalog, and target lookup indexes."""

    runtime: RuntimeBundle
    catalog: DagCatalog
    by_name: Mapping[str, TargetDescriptor]
    by_table_key: Mapping[str, TargetDescriptor]
    by_artifact_name: Mapping[str, TargetDescriptor]

    def get_target(self, name: str) -> TargetDescriptor | None:
        """Return target metadata by name.

        Parameters
        ----------
        name
            Target name.

        Returns
        -------
        TargetDescriptor | None
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
        return self.catalog.closure(targets)

    def target_for_table_key(self, table_key: str) -> TargetDescriptor | None:
        """Return producing target for a table key.

        Parameters
        ----------
        table_key
            Fully-qualified table key (schema.table).

        Returns
        -------
        TargetDescriptor | None
            Producing target if present, otherwise None.
        """
        return self.by_table_key.get(table_key)

    def output_for_table_key(self, table_key: str) -> OutputDescriptor | None:
        """Return output descriptor for a table key.

        Parameters
        ----------
        table_key
            Fully-qualified table key (schema.table).

        Returns
        -------
        OutputDescriptor | None
            Output descriptor if present, otherwise None.
        """
        return self.catalog.table_outputs.get(table_key)

    def target_for_artifact(self, artifact_name: str) -> TargetDescriptor | None:
        """Return producing target for an artifact name.

        Parameters
        ----------
        artifact_name
            Artifact name declared in a target contract.

        Returns
        -------
        TargetDescriptor | None
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
    catalog: DagCatalog,
) -> tuple[
    Mapping[str, TargetDescriptor],
    Mapping[str, TargetDescriptor],
    Mapping[str, TargetDescriptor],
]:
    by_name: dict[str, TargetDescriptor] = {}
    by_table_key: dict[str, TargetDescriptor] = {}
    by_artifact_name: dict[str, TargetDescriptor] = {}

    for target in catalog.all_targets:
        by_name[target.name] = target

    for table_key, output in catalog.table_outputs.items():
        target = catalog.targets.get(output.producer_target)
        if target is None:
            msg = f"Unknown target for table output {table_key}: {output.producer_target}"
            raise ValueError(msg)
        by_table_key[table_key] = target

    for artifact_name, output in catalog.artifact_outputs.items():
        target = catalog.targets.get(output.producer_target)
        if target is None:
            msg = f"Unknown target for artifact output {artifact_name}: {output.producer_target}"
            raise ValueError(msg)
        by_artifact_name[artifact_name] = target

    return (
        MappingProxyType(by_name),
        MappingProxyType(by_table_key),
        MappingProxyType(by_artifact_name),
    )


def build_target_system(*, runtime: RuntimeBundle) -> TargetSystem:
    """Build a TargetSystem from a runtime bundle.

    Returns
    -------
    TargetSystem
        Target system derived from the runtime.
    """
    catalog = runtime.catalog
    by_name, by_table_key, by_artifact_name = _build_indexes(catalog)
    return TargetSystem(
        runtime=runtime,
        catalog=catalog,
        by_name=by_name,
        by_table_key=by_table_key,
        by_artifact_name=by_artifact_name,
    )


@dataclass(frozen=True, slots=True)
class TargetMetadataService:
    """Bundle of target system and schema index."""

    system: TargetSystem
    schema_index: SchemaIndex

    def get_target(self, name: str) -> TargetDescriptor | None:
        """Return target metadata by name.

        Parameters
        ----------
        name
            Target name to resolve.

        Returns
        -------
        TargetDescriptor | None
            Target metadata if found.
        """
        return self.system.get_target(name)

    def target_for_table_key(self, table_key: str) -> TargetDescriptor | None:
        """Return the target that produces a dataset table key.

        Parameters
        ----------
        table_key
            Fully qualified table key (schema.table).

        Returns
        -------
        TargetDescriptor | None
            Target metadata if found.
        """
        return self.system.target_for_table_key(table_key)

    def output_for_table_key(self, table_key: str) -> OutputDescriptor | None:
        """Return the output descriptor for a dataset table key.

        Parameters
        ----------
        table_key
            Fully qualified table key (schema.table).

        Returns
        -------
        OutputDescriptor | None
            Output descriptor if found.
        """
        return self.system.output_for_table_key(table_key)

    def target_for_artifact(self, artifact_name: str) -> TargetDescriptor | None:
        """Return the target that produces an artifact name.

        Parameters
        ----------
        artifact_name
            Artifact name to resolve.

        Returns
        -------
        TargetDescriptor | None
            Target metadata if found.
        """
        return self.system.target_for_artifact(artifact_name)


class TargetMetadataProvider(Protocol):
    """Protocol for resolving target metadata."""

    def get_target(self, name: str) -> TargetDescriptor | None:
        """Return target metadata by name."""
        ...

    def target_for_table_key(self, table_key: str) -> TargetDescriptor | None:
        """Return target metadata for a table key."""
        ...

    def output_for_table_key(self, table_key: str) -> OutputDescriptor | None:
        """Return output descriptor for a table key."""
        ...

    def target_for_artifact(self, artifact_name: str) -> TargetDescriptor | None:
        """Return target metadata for an artifact name."""
        ...


@dataclass(slots=True)
class LazyTargetMetadataProvider:
    """Lazy provider that loads the target metadata service on demand."""

    runtime: RuntimeBundle
    _service: TargetMetadataService | None = None

    def _resolve(self) -> TargetMetadataService:
        if self._service is None:
            self._service = get_target_metadata_service(runtime=self.runtime)
        return self._service

    def get_target(self, name: str) -> TargetDescriptor | None:
        """Return target metadata by name.

        Returns
        -------
        TargetDescriptor | None
            Target metadata if present, otherwise None.
        """
        return self._resolve().get_target(name)

    def target_for_table_key(self, table_key: str) -> TargetDescriptor | None:
        """Return target metadata for a table key.

        Returns
        -------
        TargetDescriptor | None
            Target metadata if present, otherwise None.
        """
        return self._resolve().target_for_table_key(table_key)

    def output_for_table_key(self, table_key: str) -> OutputDescriptor | None:
        """Return output descriptor for a table key.

        Returns
        -------
        OutputDescriptor | None
            Output descriptor if present, otherwise None.
        """
        return self._resolve().output_for_table_key(table_key)

    def target_for_artifact(self, artifact_name: str) -> TargetDescriptor | None:
        """Return target metadata for an artifact name.

        Returns
        -------
        TargetDescriptor | None
            Target metadata if present, otherwise None.
        """
        return self._resolve().target_for_artifact(artifact_name)

    def reset(self) -> None:
        """Clear any cached target metadata service."""
        self._service = None

    def is_loaded(self) -> bool:
        """Return True if the target metadata service has been loaded.

        Returns
        -------
        bool
            True when the provider has initialized its service.
        """
        return self._service is not None


_TARGET_METADATA_PROVIDERS: list[LazyTargetMetadataProvider] = []


def get_target_metadata_service(*, runtime: RuntimeBundle) -> TargetMetadataService:
    """Return the canonical target metadata service for a runtime.

    Returns
    -------
    TargetMetadataService
        Target metadata service scoped to the runtime bundle.

    Raises
    ------
    ValueError
        If the runtime bundle lacks a schema index.
    """
    schema_index = runtime.schema_index
    if schema_index is None:
        msg = "RuntimeBundle.schema_index is required to build TargetMetadataService"
        raise ValueError(msg)
    return TargetMetadataService(
        system=build_target_system(runtime=runtime),
        schema_index=schema_index,
    )


def get_target_system(*, runtime: RuntimeBundle) -> TargetSystem:
    """Return the target system for a runtime bundle.

    Returns
    -------
    TargetSystem
        Target system derived from the runtime.
    """
    return build_target_system(runtime=runtime)


def get_target_metadata_provider(*, runtime: RuntimeBundle) -> TargetMetadataProvider:
    """Return a lazy target metadata provider bound to a runtime bundle.

    Returns
    -------
    TargetMetadataProvider
        Lazy target metadata provider.
    """
    provider = LazyTargetMetadataProvider(runtime=runtime)
    _TARGET_METADATA_PROVIDERS.append(provider)
    return provider


def is_target_metadata_loaded() -> bool:
    """Return True if any target metadata providers have been loaded.

    Returns
    -------
    bool
        True when any provider has initialized its service.
    """
    return any(provider.is_loaded() for provider in _TARGET_METADATA_PROVIDERS)


def clear_target_metadata_cache() -> None:
    """Clear cached target metadata services."""
    for provider in _TARGET_METADATA_PROVIDERS:
        provider.reset()


def reset_target_metadata_state() -> None:
    """Reset cached target metadata providers.

    Intended for tests that need a fresh provider registration state.
    """
    for provider in _TARGET_METADATA_PROVIDERS:
        provider.reset()
    _TARGET_METADATA_PROVIDERS.clear()


__all__ = [
    "LazyTargetMetadataProvider",
    "TargetMetadataProvider",
    "TargetMetadataService",
    "TargetSystem",
    "build_target_system",
    "clear_target_metadata_cache",
    "get_target_metadata_provider",
    "get_target_metadata_service",
    "get_target_system",
    "is_target_metadata_loaded",
    "reset_target_metadata_state",
]
