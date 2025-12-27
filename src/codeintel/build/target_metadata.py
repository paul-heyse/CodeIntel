"""Canonical target metadata service for build target access."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, cast

from codeintel.build.hamilton.dag_catalog import DagCatalog, OutputDescriptor, TargetDescriptor
from codeintel.core.imports.lazy import lazy_getattr
from codeintel.core.schemas.declared import source_declared_schema_provider

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from types import ModuleType

    from codeintel.build.hamilton.driver_factory import HamiltonRuntime
    from codeintel.build.schemas.inference_service import SchemaInferenceService
    from codeintel.build.schemas.schema_index import SchemaIndex


@dataclass(frozen=True, slots=True)
class TargetSystem:
    """Bundle runtime, catalog, and target lookup indexes."""

    runtime: HamiltonRuntime
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

    factory: Callable[[], TargetMetadataService]
    _service: TargetMetadataService | None = None

    def _resolve(self) -> TargetMetadataService:
        if self._service is None:
            self._service = self.factory()
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


_TARGET_METADATA_PROVIDERS: list[LazyTargetMetadataProvider] = []


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
    inference_service = get_schema_inference_service()
    inferable_table_keys = inference_service.inferable_table_keys(catalog=system.catalog)
    schema_index = build_schema_index(
        system=system,
        declared_provider=source_declared_schema_provider(
            exclude_table_keys=inferable_table_keys,
        ),
        inference_service=inference_service,
    )
    return TargetMetadataService(
        system=system,
        schema_index=schema_index,
    )


def get_target_system() -> TargetSystem:
    """Return the cached TargetSystem without loading schema inventory helpers.

    Returns
    -------
    TargetSystem
        Cached target system with runtime and graph metadata.
    """
    return _load_target_system()


def get_target_metadata_provider() -> TargetMetadataProvider:
    """Return a lazy target metadata provider.

    Returns
    -------
    TargetMetadataProvider
        Lazy provider that resolves metadata on demand.
    """
    provider = LazyTargetMetadataProvider(get_target_metadata_service)
    _TARGET_METADATA_PROVIDERS.append(provider)
    return provider


def is_target_metadata_loaded() -> bool:
    """Return True if the target metadata service has been initialized.

    Returns
    -------
    bool
        True when the metadata service has been initialized.
    """
    return (
        _load_target_system.cache_info().currsize > 0
        or get_target_metadata_service.cache_info().currsize > 0
    )


def clear_target_metadata_cache() -> None:
    """Clear cached target metadata services."""
    _load_target_system.cache_clear()
    get_target_metadata_service.cache_clear()
    for provider in _TARGET_METADATA_PROVIDERS:
        provider.reset()


__all__ = [
    "TargetMetadataProvider",
    "TargetMetadataService",
    "TargetSystem",
    "clear_target_metadata_cache",
    "get_target_metadata_provider",
    "get_target_metadata_service",
    "get_target_system",
    "is_target_metadata_loaded",
]
