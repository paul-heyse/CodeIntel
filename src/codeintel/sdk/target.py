"""Stable target anchor helpers for plugins."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import ParamSpec, TypeVar, cast

from codeintel.build.hamilton.boundary_types import MaterializationResult
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.materialization_records import (
    MaterializationRecordContext,
    record_from_materializations,
)
from codeintel.build.hamilton.native.target_decorators import TargetSpecDescriptor, codeintel_target
from codeintel.core.hamilton.records import TargetRunRecord

P = ParamSpec("P")
R = TypeVar("R")
Decorator = Callable[[Callable[P, R]], Callable[P, R]]


@dataclass(frozen=True, slots=True)
class MaterializationBundle:
    """Materialization collections for a target run."""

    table_materializations: Mapping[str, MaterializationResult] | None = None
    artifact_materializations: Mapping[str, MaterializationResult] | None = None


def target_anchor(
    *,
    domain: str,
    target: str,
    spec: TargetSpecDescriptor | None = None,
) -> Decorator[P, R]:
    """Return a target anchor decorator with canonical tags.

    Parameters
    ----------
    domain
        Target domain name.
    target
        Target identifier.
    spec
        Optional target specification descriptor.

    Returns
    -------
    Decorator[P, R]
        Decorator that marks a target anchor function.
    """
    return cast("Decorator[P, R]", codeintel_target(domain=domain, target=target, spec=spec))


def finalize_materializations(
    *,
    env: BuildEnv,
    catalog: DagCatalog,
    target_name: str,
    materializations: MaterializationBundle | None = None,
    change_delta: Mapping[str, object] | None = None,
) -> TargetRunRecord:
    """Create a TargetRunRecord from materialization results.

    Parameters
    ----------
    env
        Build environment for this target run.
    catalog
        DAG catalog for resolving target metadata.
    target_name
        Target identifier.
    materializations
        Collected table and artifact materialization results.
    change_delta
        Optional change metadata payload.

    Returns
    -------
    TargetRunRecord
        Record describing the target materialization run.
    """
    materializations = materializations or MaterializationBundle()
    context = MaterializationRecordContext(
        env=env,
        catalog=catalog,
        target_name=target_name,
        change_delta=change_delta,
    )
    return record_from_materializations(
        context=context,
        artifact_materializations=materializations.artifact_materializations,
        table_materializations=materializations.table_materializations,
    )


__all__ = [
    "BuildEnv",
    "DagCatalog",
    "MaterializationBundle",
    "MaterializationResult",
    "TargetRunRecord",
    "TargetSpecDescriptor",
    "finalize_materializations",
    "target_anchor",
]
