"""Hamilton build planner for actionable dry-run output.

This module provides planning infrastructure for the Hamilton build system,
showing the dependency closure and structural execution order.

Design Principles
-----------------
1. PlanEntry captures structural information about the target closure.
2. HamiltonBuildPlan provides structured access to the full build plan.
3. compute_plan() computes the plan without executing anything.
4. Plans are useful for dry-run output and dependency inspection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from codeintel.build.hamilton.driver_factory import build_driver
from codeintel.build.hamilton.impl_kind import ImplKind, native_target_names
from codeintel.build.hamilton.naming import target_node
from codeintel.build.hamilton.native.outputs import (
    expected_artifact_names_for_target,
    expected_table_keys_for_target,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.build.hamilton.dag_catalog import DagCatalog, TargetDescriptor
    from codeintel.build.hamilton.env import BuildEnv
    from codeintel.build.targets import TargetModule


PlanStatus = Literal["compute", "missing", "blocked"]


PlanReason = Literal[
    "scheduled",
    "upstream_missing",
    "no_impl",
]


@dataclass(frozen=True)
class PlanEntry:
    """Plan entry describing why a target will or won't run.

    Each PlanEntry captures the structural planning context for a target:
    where it sits in the closure and why it may be blocked or missing.

    Attributes
    ----------
    target
        Target name being planned.
    node
        Hamilton node name (e.g., "t__function_metrics").
    module
        Target module (ingestion, graphs, analytics, export).
        May be ``"unknown"`` for missing targets.
    status
        Plan status: "compute", "missing", or "blocked".
    reason
        Reason for the status:
        - "scheduled": Target is scheduled for execution
        - "upstream_missing": An upstream dependency is missing
        - "no_impl": No implementation registered for this target
    dependencies
        Tuple of target names this target depends on.
    table_keys
        Tuple of table keys this target produces.
    artifact_keys
        Tuple of artifact keys this target produces (future use).
    impl_kind
        Implementation kind. Native Hamilton pipelines are required.

    Examples
    --------
    >>> entry = PlanEntry(
    ...     target="function_metrics",
    ...     node="t__function_metrics",
    ...     module="analytics",
    ...     status="compute",
    ...     reason="scheduled",
    ...     dependencies=("goids", "ast"),
    ...     table_keys=("analytics.function_metrics",),
    ...     artifact_keys=(),
    ... )
    """

    target: str
    node: str
    module: TargetModule | Literal["unknown"]
    status: PlanStatus
    reason: PlanReason
    dependencies: tuple[str, ...]
    table_keys: tuple[str, ...]
    artifact_keys: tuple[str, ...] = ()
    impl_kind: ImplKind = "native"

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation of the plan entry.
        """
        return {
            "target": self.target,
            "node": self.node,
            "module": self.module,
            "status": self.status,
            "reason": self.reason,
            "dependencies": list(self.dependencies),
            "table_keys": list(self.table_keys),
            "artifact_keys": list(self.artifact_keys),
            "impl_kind": self.impl_kind,
        }


@dataclass(frozen=True)
class HamiltonBuildPlan:
    """Complete build plan for Hamilton execution.

    The plan contains all information needed to understand what will happen
    during a build, without actually executing anything.

    Attributes
    ----------
    requested
        Tuple of target names originally requested by the user.
    closure
        Tuple of target names in dependency closure (topological order).
    entries
        Tuple of PlanEntry objects, one per target in closure.

    Examples
    --------
    >>> plan = compute_plan(env=env, requested=("risk_factors",))
    >>> plan.to_compute
    ('modules', 'scip', 'ast', 'goids', 'function_metrics', 'risk_factors')
    >>> plan.to_skip
    ()
    """

    requested: tuple[str, ...]
    closure: tuple[str, ...]
    entries: tuple[PlanEntry, ...] = field(default_factory=tuple)

    @property
    def to_compute(self) -> tuple[str, ...]:
        """Return targets that will be computed.

        Returns
        -------
        tuple[str, ...]
            Target names with status="compute".
        """
        return tuple(e.target for e in self.entries if e.status == "compute")

    @property
    def to_skip(self) -> tuple[str, ...]:
        """Return targets that will be skipped.

        Returns
        -------
        tuple[str, ...]
            Always empty for cache-driven planning.
        """
        return ()

    @property
    def blocked(self) -> tuple[str, ...]:
        """Return targets that are blocked.

        Returns
        -------
        tuple[str, ...]
            Target names with status="blocked".
        """
        return tuple(e.target for e in self.entries if e.status == "blocked")

    @property
    def missing(self) -> tuple[str, ...]:
        """Return targets that are missing.

        Returns
        -------
        tuple[str, ...]
            Target names with status="missing".
        """
        return tuple(e.target for e in self.entries if e.status == "missing")

    def get_entry(self, target_name: str) -> PlanEntry | None:
        """Get plan entry for a specific target.

        Parameters
        ----------
        target_name
            Target name to look up.

        Returns
        -------
        PlanEntry | None
            The plan entry if found, None otherwise.
        """
        for entry in self.entries:
            if entry.target == target_name:
                return entry
        return None

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation of the build plan.
        """
        return {
            "requested": list(self.requested),
            "closure": list(self.closure),
            "entries": [e.to_dict() for e in self.entries],
            "to_compute": list(self.to_compute),
            "to_skip": list(self.to_skip),
            "blocked": list(self.blocked),
            "missing": list(self.missing),
        }


@dataclass(frozen=True, slots=True)
class _PlanEntryInputs:
    upstream_status: Mapping[str, PlanStatus]
    native_names: frozenset[str]
    catalog: DagCatalog


def _compute_entry_for_target(
    target: TargetDescriptor,
    *,
    inputs: _PlanEntryInputs,
) -> PlanEntry:
    """Compute plan entry for a single target.

    Parameters
    ----------
    target
        Target metadata from the graph.
    inputs
        Shared planning inputs for upstream status and catalog access.

    Returns
    -------
    PlanEntry
        Computed plan entry for this target.

    """
    target_name = target.name
    node = target_node(target_name)
    module = target.module

    table_keys = expected_table_keys_for_target(target.name, outputs=inputs.catalog)
    artifact_keys = expected_artifact_names_for_target(target.name, outputs=inputs.catalog)
    blocked_deps = [
        dep
        for dep in target.dependencies
        if inputs.upstream_status.get(dep) in {"missing", "blocked"}
    ]
    if target_name not in inputs.native_names:
        return PlanEntry(
            target=target_name,
            node=node,
            module=module,
            status="missing",
            reason="no_impl",
            dependencies=tuple(target.dependencies),
            table_keys=tuple(table_keys),
            artifact_keys=tuple(artifact_keys),
            impl_kind="native",
        )
    impl_kind: ImplKind = "native"

    if blocked_deps:
        return PlanEntry(
            target=target_name,
            node=node,
            module=module,
            status="blocked",
            reason="upstream_missing",
            dependencies=tuple(target.dependencies),
            table_keys=tuple(table_keys),
            artifact_keys=tuple(artifact_keys),
            impl_kind=impl_kind,
        )

    return PlanEntry(
        target=target_name,
        node=node,
        module=module,
        status="compute",
        reason="scheduled",
        dependencies=tuple(target.dependencies),
        table_keys=tuple(table_keys),
        artifact_keys=tuple(artifact_keys),
        impl_kind=impl_kind,
    )


def compute_plan(
    *,
    env: BuildEnv,
    catalog: DagCatalog | None = None,
    requested: tuple[str, ...],
) -> HamiltonBuildPlan:
    """Compute build plan for requested targets.

    Analyzes the target graph to produce a structural plan for the closure.

    Parameters
    ----------
    env
        Build environment with gateway, snapshot, and configuration.
    catalog
        DAG catalog to use. If None, uses the catalog from the Hamilton runtime.
    requested
        Tuple of target names requested by the user.

    Returns
    -------
    HamiltonBuildPlan
        Complete build plan with entries for all targets in closure.

    Examples
    --------
    >>> plan = compute_plan(
    ...     env=env,
    ...     requested=("risk_factors",),
    ... )
    >>> len(plan.to_compute)
    7
    """
    _ = env
    runtime = build_driver()
    if catalog is None:
        catalog = runtime.catalog
    native_names = native_target_names(runtime)

    closure = catalog.closure(requested)

    entries: list[PlanEntry] = []
    upstream_status: dict[str, PlanStatus] = {}
    inputs = _PlanEntryInputs(
        upstream_status=upstream_status,
        native_names=native_names,
        catalog=catalog,
    )

    for target_name in closure:
        try:
            target = catalog.get(target_name)
        except KeyError:
            entry = PlanEntry(
                target=target_name,
                node=target_node(target_name),
                module="unknown",
                status="missing",
                reason="no_impl",
                dependencies=(),
                table_keys=(),
            )
            entries.append(entry)
            upstream_status[target_name] = "missing"
            continue

        entry = _compute_entry_for_target(target, inputs=inputs)
        entries.append(entry)
        upstream_status[target_name] = entry.status

    return HamiltonBuildPlan(
        requested=requested,
        closure=closure,
        entries=tuple(entries),
    )


__all__ = [
    "HamiltonBuildPlan",
    "ImplKind",
    "PlanEntry",
    "PlanReason",
    "PlanStatus",
    "compute_plan",
]
