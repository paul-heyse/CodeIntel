"""Canonical build target catalog derived from native Hamilton modules.

This module defines the single source of truth for OutputTarget metadata used by
the build system. The catalog is compiled from native Hamilton modules, which
declare `TARGET_SPECS` alongside the materialize nodes they implement.

Design notes
------------
- Dependencies are derived from the Hamilton DAG at runtime.
- Metadata (contracts, resources, execution policy, descriptions) lives next to
  native Hamilton implementations and is collected into a deterministic catalog.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import TYPE_CHECKING

from codeintel.build.hamilton.native.discovery import load_native_modules
from codeintel.build.targets import OutputTarget

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from types import ModuleType


def _iter_module_target_specs(module: ModuleType) -> Iterable[OutputTarget]:
    specs_obj = getattr(module, "TARGET_SPECS", None)
    if specs_obj is None:
        return ()

    if isinstance(specs_obj, (tuple, list)):
        specs: list[OutputTarget] = []
        for item in specs_obj:
            if not isinstance(item, OutputTarget):
                msg = (
                    f"{module.__name__}.TARGET_SPECS contains non-OutputTarget element: "
                    f"{type(item)}"
                )
                raise TypeError(msg)
            specs.append(item)
        return tuple(specs)

    msg = (
        f"{module.__name__}.TARGET_SPECS must be a tuple/list of OutputTarget, got {type(specs_obj)}"
    )
    raise TypeError(msg)


def _validate_specs(specs: Iterable[OutputTarget]) -> tuple[OutputTarget, ...]:
    by_name: dict[str, OutputTarget] = {}
    for target in specs:
        if target.name in by_name:
            msg = f"Duplicate target spec name: {target.name}"
            raise ValueError(msg)
        if target.dependencies:
            msg = (
                "Target specs must not declare dependencies; Hamilton is the single source of "
                f"truth. Found dependencies for {target.name}: {target.dependencies!r}"
            )
            raise ValueError(msg)
        if target.plugin:
            msg = (
                "Target specs must not declare plugin implementations in Hamilton-first "
                f"execution. Found plugin for {target.name}: {target.plugin!r}"
            )
            raise ValueError(msg)
        by_name[target.name] = target

    return tuple(by_name[name] for name in sorted(by_name))


@lru_cache(maxsize=1)
def load_target_specs() -> tuple[OutputTarget, ...]:
    """Load the canonical OutputTarget specs from native Hamilton modules.

    Returns
    -------
    tuple[OutputTarget, ...]
        Deterministically ordered OutputTarget specifications.
    """
    specs: list[OutputTarget] = []
    for module in load_native_modules():
        specs.extend(_iter_module_target_specs(module))
    return _validate_specs(specs)


@dataclass(frozen=True, slots=True)
class TargetCatalog:
    """Indexed view over the canonical OutputTarget specifications.

    Attributes
    ----------
    targets
        Deterministically ordered target specs.
    by_name
        Mapping of target name to OutputTarget.
    by_table_key
        Mapping of produced table_key to OutputTarget.
    by_artifact_name
        Mapping of produced artifact name to OutputTarget.
    """

    targets: tuple[OutputTarget, ...]
    by_name: Mapping[str, OutputTarget]
    by_table_key: Mapping[str, OutputTarget]
    by_artifact_name: Mapping[str, OutputTarget]

    def get(self, name: str) -> OutputTarget | None:
        """Lookup a target by name.

        Parameters
        ----------
        name
            Target name.

        Returns
        -------
        OutputTarget | None
            Target specification if present, otherwise None.
        """
        target = self.by_name.get(name)
        return target if isinstance(target, OutputTarget) else None

    def target_for_table_key(self, table_key: str) -> OutputTarget | None:
        """Lookup the producing target for a table key.

        Parameters
        ----------
        table_key
            Fully qualified table key (schema.table).

        Returns
        -------
        OutputTarget | None
            Producing target specification if present, otherwise None.
        """
        target = self.by_table_key.get(table_key)
        return target if isinstance(target, OutputTarget) else None

    def target_for_artifact(self, artifact_name: str) -> OutputTarget | None:
        """Lookup the producing target for an artifact name.

        Parameters
        ----------
        artifact_name
            Artifact name declared in the target contract.

        Returns
        -------
        OutputTarget | None
            Producing target specification if present, otherwise None.
        """
        target = self.by_artifact_name.get(artifact_name)
        return target if isinstance(target, OutputTarget) else None

    @property
    def all_table_keys(self) -> frozenset[str]:
        """Return all produced table keys."""
        return frozenset(self.by_table_key)

    @property
    def all_artifact_names(self) -> frozenset[str]:
        """Return all produced artifact names."""
        return frozenset(self.by_artifact_name)


def _build_catalog(specs: tuple[OutputTarget, ...]) -> TargetCatalog:
    by_name: dict[str, OutputTarget] = {target.name: target for target in specs}

    by_table_key: dict[str, OutputTarget] = {}
    by_artifact_name: dict[str, OutputTarget] = {}

    for target in specs:
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

    return TargetCatalog(
        targets=specs,
        by_name=MappingProxyType(by_name),
        by_table_key=MappingProxyType(by_table_key),
        by_artifact_name=MappingProxyType(by_artifact_name),
    )


@lru_cache(maxsize=1)
def load_target_catalog() -> TargetCatalog:
    """Load the canonical TargetCatalog (specs + indexes).

    Returns
    -------
    TargetCatalog
        Catalog containing the canonical target specs and indexes.
    """
    return _build_catalog(load_target_specs())


__all__ = [
    "TargetCatalog",
    "load_target_catalog",
    "load_target_specs",
]
