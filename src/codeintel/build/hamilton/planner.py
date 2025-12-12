"""Hamilton build planner for actionable dry-run output.

This module provides planning infrastructure for the Hamilton build system,
enabling best-in-class DX with real planning that shows what will run and why.

Design Principles
-----------------
1. PlanEntry captures complete information about why a target will/won't run.
2. HamiltonBuildPlan provides structured access to the full build plan.
3. compute_plan() computes the plan without executing anything.
4. Plans are useful for both dry-run output and incremental build optimization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from codeintel.build.hamilton.manifest_hook import (
    compute_target_input_hash,
    compute_target_options_hash,
)
from codeintel.build.hamilton.naming import target_node
from codeintel.build.registry import get_target_graph

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.build.hamilton.driver_factory import HamiltonNodeMode
    from codeintel.build.hamilton.env import BuildEnv
    from codeintel.build.manifest import OutputManifest
    from codeintel.build.targets import OutputTarget, TargetGraph

# Plan entry status values
PlanStatus = Literal["compute", "skip", "missing", "blocked"]

# Plan entry reason values
PlanReason = Literal[
    "forced",
    "no_manifest",
    "hash_changed",
    "up_to_date",
    "upstream_missing",
    "no_plugin",
]


@dataclass(frozen=True)
class PlanEntry:
    """Plan entry describing why a target will or won't run.

    Each PlanEntry captures the complete decision context for a target:
    what its status is, why, and all relevant metadata for debugging.

    Attributes
    ----------
    target
        Target name being planned.
    node
        Hamilton node name (e.g., "t__function_metrics").
    module
        Target module (ingestion, graphs, analytics).
    status
        Plan status: "compute", "skip", "missing", or "blocked".
    reason
        Reason for the status:
        - "forced": Target is in force set, will recompute
        - "no_manifest": No prior manifest exists, must compute
        - "hash_changed": Input hash differs from manifest, must recompute
        - "up_to_date": Input hash matches manifest, can skip
        - "upstream_missing": An upstream dependency is missing
        - "no_plugin": No plugin registered for this target
    input_hash
        Current computed input hash for the target.
    options_hash
        Current computed options hash from configuration.
    prior_input_hash
        Input hash from prior manifest, if available.
    dependencies
        Tuple of target names this target depends on.
    table_keys
        Tuple of table keys this target produces.
    artifact_keys
        Tuple of artifact keys this target produces (future use).

    Examples
    --------
    >>> entry = PlanEntry(
    ...     target="function_metrics",
    ...     node="t__function_metrics",
    ...     module="analytics",
    ...     status="compute",
    ...     reason="hash_changed",
    ...     input_hash="abc123",
    ...     options_hash=None,
    ...     prior_input_hash="def456",
    ...     dependencies=("goids", "ast"),
    ...     table_keys=("analytics.function_metrics",),
    ...     artifact_keys=(),
    ... )
    """

    target: str
    node: str
    module: str
    status: PlanStatus
    reason: PlanReason
    input_hash: str | None
    options_hash: str | None
    prior_input_hash: str | None
    dependencies: tuple[str, ...]
    table_keys: tuple[str, ...]
    artifact_keys: tuple[str, ...] = ()
    dep_hashes: dict[str, str] = field(default_factory=dict)
    prior_dep_hashes: dict[str, str] = field(default_factory=dict)

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
            "input_hash": self.input_hash,
            "options_hash": self.options_hash,
            "prior_input_hash": self.prior_input_hash,
            "dependencies": list(self.dependencies),
            "table_keys": list(self.table_keys),
            "artifact_keys": list(self.artifact_keys),
            "dep_hashes": dict(self.dep_hashes),
            "prior_dep_hashes": dict(self.prior_dep_hashes),
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
    >>> plan = compute_plan(env=env, graph=graph, requested=("risk_factors",))
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
            Target names with status="skip".
        """
        return tuple(e.target for e in self.entries if e.status == "skip")

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


def _compute_entry_for_target(
    target: OutputTarget,
    env: BuildEnv,
    manifests: Mapping[str, OutputManifest],
    upstream_status: dict[str, PlanStatus],
) -> PlanEntry:
    """Compute plan entry for a single target.

    Parameters
    ----------
    target
        Target metadata from the graph.
    env
        Build environment with configuration and snapshot.
    manifests
        Pre-loaded manifest index.
    upstream_status
        Status of upstream targets computed so far.

    Returns
    -------
    PlanEntry
        Computed plan entry for this target.
    """
    target_name = target.name
    node = target_node(target_name)
    module = target.module

    # Get table keys
    table_keys = target.contract.table_keys or target.table_keys

    # Check for upstream issues
    blocked_deps = [
        dep for dep in target.dependencies if upstream_status.get(dep) in {"missing", "blocked"}
    ]
    if blocked_deps:
        return PlanEntry(
            target=target_name,
            node=node,
            module=module,
            status="blocked",
            reason="upstream_missing",
            input_hash=None,
            options_hash=None,
            prior_input_hash=None,
            dependencies=tuple(target.dependencies),
            table_keys=tuple(table_keys),
        )

    # Check if forced
    if env.is_forced(target_name):
        # Still compute hash for reference
        raw_params = env.config.parameters_for(target_name)
        options_hash = compute_target_options_hash(raw_params) if raw_params else None
        input_hash = compute_target_input_hash(
            target=target,
            snapshot=env.snapshot,
            gateway=env.gateway,
            options_hash=options_hash,
            manifests=manifests,
        )
        prior = manifests.get(target_name)
        return PlanEntry(
            target=target_name,
            node=node,
            module=module,
            status="compute",
            reason="forced",
            input_hash=input_hash,
            options_hash=options_hash,
            prior_input_hash=prior.input_hash if prior else None,
            dependencies=tuple(target.dependencies),
            table_keys=tuple(table_keys),
        )

    # Compute hashes
    raw_params = env.config.parameters_for(target_name)
    options_hash = compute_target_options_hash(raw_params) if raw_params else None
    input_hash = compute_target_input_hash(
        target=target,
        snapshot=env.snapshot,
        gateway=env.gateway,
        options_hash=options_hash,
        manifests=manifests,
    )

    # Check manifest
    prior = manifests.get(target_name)
    if prior is None:
        return PlanEntry(
            target=target_name,
            node=node,
            module=module,
            status="compute",
            reason="no_manifest",
            input_hash=input_hash,
            options_hash=options_hash,
            prior_input_hash=None,
            dependencies=tuple(target.dependencies),
            table_keys=tuple(table_keys),
        )

    # Compare hashes
    if prior.input_hash != input_hash:
        return PlanEntry(
            target=target_name,
            node=node,
            module=module,
            status="compute",
            reason="hash_changed",
            input_hash=input_hash,
            options_hash=options_hash,
            prior_input_hash=prior.input_hash,
            dependencies=tuple(target.dependencies),
            table_keys=tuple(table_keys),
        )

    # Up to date - can skip
    return PlanEntry(
        target=target_name,
        node=node,
        module=module,
        status="skip",
        reason="up_to_date",
        input_hash=input_hash,
        options_hash=options_hash,
        prior_input_hash=prior.input_hash,
        dependencies=tuple(target.dependencies),
        table_keys=tuple(table_keys),
    )


def compute_plan(
    *,
    env: BuildEnv,
    graph: TargetGraph | None = None,
    requested: tuple[str, ...],
    mode: HamiltonNodeMode = "generated",
) -> HamiltonBuildPlan:
    """Compute build plan for requested targets.

    Analyzes the target graph and manifest state to produce a complete plan
    showing what will run, what will be skipped, and why.

    Parameters
    ----------
    env
        Build environment with gateway, snapshot, and configuration.
    graph
        Target graph to use. If None, fetches from registry.
    requested
        Tuple of target names requested by the user.
    mode
        Hamilton node mode (for metadata only, doesn't affect planning).

    Returns
    -------
    HamiltonBuildPlan
        Complete build plan with entries for all targets in closure.

    Examples
    --------
    >>> plan = compute_plan(
    ...     env=env,
    ...     requested=("risk_factors",),
    ...     mode="generated",
    ... )
    >>> len(plan.to_compute)
    7
    """
    # Use mode parameter for potential future expansion
    _ = mode

    if graph is None:
        graph = get_target_graph()

    # Compute closure
    closure = graph.topological_order(list(requested))

    # Load manifests (use env.manifest_index if available)
    if env.manifest_index is not None:
        manifests: Mapping[str, OutputManifest] = env.manifest_index
    else:
        manifest_list = env.gateway.build.list_manifests(
            repo=env.snapshot.repo,
            commit=env.snapshot.commit,
        )
        manifests = {m.target: m for m in manifest_list}

    # Compute entries in topological order
    entries: list[PlanEntry] = []
    upstream_status: dict[str, PlanStatus] = {}

    for target_name in closure:
        try:
            target = graph.get(target_name)
        except KeyError:
            # Target not in graph - mark as missing
            entry = PlanEntry(
                target=target_name,
                node=target_node(target_name),
                module="unknown",
                status="missing",
                reason="no_plugin",
                input_hash=None,
                options_hash=None,
                prior_input_hash=None,
                dependencies=(),
                table_keys=(),
            )
            entries.append(entry)
            upstream_status[target_name] = "missing"
            continue

        entry = _compute_entry_for_target(target, env, manifests, upstream_status)
        entries.append(entry)
        upstream_status[target_name] = entry.status

    return HamiltonBuildPlan(
        requested=requested,
        closure=closure,
        entries=tuple(entries),
    )


__all__ = [
    "HamiltonBuildPlan",
    "PlanEntry",
    "PlanReason",
    "PlanStatus",
    "compute_plan",
]
