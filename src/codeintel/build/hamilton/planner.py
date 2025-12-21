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

from codeintel.build.hamilton.driver_factory import build_driver
from codeintel.build.hamilton.impl_kind import ImplKind, native_target_names
from codeintel.build.hamilton.introspect import target_graph_from_hamilton
from codeintel.build.hamilton.naming import target_node
from codeintel.build.hash_evaluator import compute_hash_evaluation
from codeintel.build.hashing import InputHashOptions, compute_target_options_hash

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.build.hamilton.env import BuildEnv
    from codeintel.build.hamilton.introspect import GraphSource
    from codeintel.build.targets import OutputTarget, TargetGraph, TargetModule
    from codeintel.core.build_manifest import OutputManifest


PlanStatus = Literal["compute", "skip", "missing", "blocked"]


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
        Target module (ingestion, graphs, analytics, export).
        May be ``"unknown"`` for missing targets.
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
    impl_kind
        Implementation kind. Native Hamilton pipelines are required.

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
    module: TargetModule | Literal["unknown"]
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
            "input_hash": self.input_hash,
            "options_hash": self.options_hash,
            "prior_input_hash": self.prior_input_hash,
            "dependencies": list(self.dependencies),
            "table_keys": list(self.table_keys),
            "artifact_keys": list(self.artifact_keys),
            "dep_hashes": dict(self.dep_hashes),
            "prior_dep_hashes": dict(self.prior_dep_hashes),
            "impl_kind": self.impl_kind,
        }

    def explain_staleness(self) -> StalenessExplanation:
        """Explain why this target is stale.

        Compares current dep_hashes to prior_dep_hashes to identify which
        dependencies changed and caused this target to be recomputed.

        Returns
        -------
        StalenessExplanation
            Detailed explanation of staleness, including changed dependencies.
        """
        added_deps = [dep for dep in self.dep_hashes if dep not in self.prior_dep_hashes]
        changed_deps = [
            dep
            for dep, current_hash in self.dep_hashes.items()
            if dep in self.prior_dep_hashes and self.prior_dep_hashes[dep] != current_hash
        ]
        removed_deps = [dep for dep in self.prior_dep_hashes if dep not in self.dep_hashes]

        return StalenessExplanation(
            target=self.target,
            status=self.status,
            reason=self.reason,
            input_hash_current=self.input_hash,
            input_hash_prior=self.prior_input_hash,
            changed_deps=tuple(sorted(changed_deps)),
            added_deps=tuple(sorted(added_deps)),
            removed_deps=tuple(sorted(removed_deps)),
            dep_hashes=dict(self.dep_hashes),
            prior_dep_hashes=dict(self.prior_dep_hashes),
        )


@dataclass(frozen=True)
class StalenessExplanation:
    """Detailed explanation of why a target is stale.

    Provides a breakdown of what changed between the prior computation
    and the current state, enabling users to understand incremental builds.

    Attributes
    ----------
    target
        Target name.
    status
        Plan status (compute, skip, blocked, missing).
    reason
        Reason for the status.
    input_hash_current
        Current computed input hash.
    input_hash_prior
        Prior input hash from manifest (if any).
    changed_deps
        Dependencies whose hashes changed.
    added_deps
        Dependencies that were added since prior computation.
    removed_deps
        Dependencies that were removed since prior computation.
    dep_hashes
        Current dependency hash mapping.
    prior_dep_hashes
        Prior dependency hash mapping.

    Examples
    --------
    >>> entry = plan.get_entry("risk_factors")
    >>> explanation = entry.explain_staleness()
    >>> explanation.changed_deps
    ('function_metrics',)
    """

    target: str
    status: PlanStatus
    reason: PlanReason
    input_hash_current: str | None
    input_hash_prior: str | None
    changed_deps: tuple[str, ...]
    added_deps: tuple[str, ...]
    removed_deps: tuple[str, ...]
    dep_hashes: dict[str, str]
    prior_dep_hashes: dict[str, str]

    @property
    def is_stale(self) -> bool:
        """Check if target is stale (needs recomputation).

        Returns
        -------
        bool
            True if target will be computed (stale).
        """
        return self.status == "compute"

    @property
    def has_changes(self) -> bool:
        """Check if there are any dependency changes.

        Returns
        -------
        bool
            True if any deps changed, added, or removed.
        """
        return bool(self.changed_deps or self.added_deps or self.removed_deps)

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "target": self.target,
            "status": self.status,
            "reason": self.reason,
            "is_stale": self.is_stale,
            "input_hash_current": self.input_hash_current,
            "input_hash_prior": self.input_hash_prior,
            "changed_deps": list(self.changed_deps),
            "added_deps": list(self.added_deps),
            "removed_deps": list(self.removed_deps),
            "dep_hashes": self.dep_hashes,
            "prior_dep_hashes": self.prior_dep_hashes,
        }

    def summary(self) -> str:
        """Generate human-readable summary.

        Returns
        -------
        str
            Human-readable staleness summary.
        """
        if self.status == "skip":
            return f"{self.target}: up-to-date (hash {self.input_hash_current})"
        if self.status == "blocked":
            return f"{self.target}: blocked on upstream dependencies"
        if self.reason == "no_manifest":
            return f"{self.target}: no prior manifest (first run)"
        if self.reason == "forced":
            return f"{self.target}: forced recomputation"

        parts: list[str] = [f"{self.target}: stale"]
        if self.changed_deps:
            parts.append(f"changed deps: {', '.join(self.changed_deps)}")
        if self.added_deps:
            parts.append(f"added deps: {', '.join(self.added_deps)}")
        if self.removed_deps:
            parts.append(f"removed deps: {', '.join(self.removed_deps)}")
        return " - ".join(parts)


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
    *,
    native_names: frozenset[str],
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
    native_names
        Set of target names whose `t__*` node originates from a native module.

    Returns
    -------
    PlanEntry
        Computed plan entry for this target.

    Raises
    ------
    RuntimeError
        If the target does not resolve to a native implementation.
    """
    target_name = target.name
    node = target_node(target_name)
    module = target.module

    table_keys = target.contract.table_keys
    blocked_deps = [
        dep for dep in target.dependencies if upstream_status.get(dep) in {"missing", "blocked"}
    ]
    if target_name not in native_names:
        msg = f"Target '{target_name}' lacks a native implementation"
        raise RuntimeError(msg)
    impl_kind: ImplKind = "native"

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
            impl_kind=impl_kind,
        )

    params = env.config.parameters_for(target_name)
    options_hash = compute_target_options_hash(params)
    hash_options = InputHashOptions(options_hash=options_hash, manifests=manifests)
    evaluation = compute_hash_evaluation(
        target=target,
        snapshot=env.snapshot,
        gateway=env.gateway,
        settings=env.settings,
        options=hash_options,
    )

    if env.is_forced(target_name):
        return PlanEntry(
            target=target_name,
            node=node,
            module=module,
            status="compute",
            reason="forced",
            input_hash=evaluation.input_hash,
            options_hash=evaluation.options_hash,
            prior_input_hash=evaluation.stored_hash,
            dependencies=tuple(target.dependencies),
            table_keys=tuple(table_keys),
            dep_hashes=evaluation.dep_hashes,
            prior_dep_hashes=evaluation.prior_dep_hashes,
            impl_kind=impl_kind,
        )

    if evaluation.status == "missing":
        return PlanEntry(
            target=target_name,
            node=node,
            module=module,
            status="compute",
            reason="no_manifest",
            input_hash=evaluation.input_hash,
            options_hash=evaluation.options_hash,
            prior_input_hash=None,
            dependencies=tuple(target.dependencies),
            table_keys=tuple(table_keys),
            dep_hashes=evaluation.dep_hashes,
            prior_dep_hashes=evaluation.prior_dep_hashes,
            impl_kind=impl_kind,
        )

    if evaluation.status == "stale":
        return PlanEntry(
            target=target_name,
            node=node,
            module=module,
            status="compute",
            reason="hash_changed",
            input_hash=evaluation.input_hash,
            options_hash=evaluation.options_hash,
            prior_input_hash=evaluation.stored_hash,
            dependencies=tuple(target.dependencies),
            table_keys=tuple(table_keys),
            dep_hashes=evaluation.dep_hashes,
            prior_dep_hashes=evaluation.prior_dep_hashes,
            impl_kind=impl_kind,
        )

    return PlanEntry(
        target=target_name,
        node=node,
        module=module,
        status="skip",
        reason="up_to_date",
        input_hash=evaluation.input_hash,
        options_hash=evaluation.options_hash,
        prior_input_hash=evaluation.stored_hash,
        dependencies=tuple(target.dependencies),
        table_keys=tuple(table_keys),
        dep_hashes=evaluation.dep_hashes,
        prior_dep_hashes=evaluation.prior_dep_hashes,
        impl_kind=impl_kind,
    )


def compute_plan(
    *,
    env: BuildEnv,
    graph: TargetGraph | None = None,
    requested: tuple[str, ...],
    graph_source: GraphSource = "hamilton",
) -> HamiltonBuildPlan:
    """Compute build plan for requested targets.

    Analyzes the target graph and manifest state to produce a complete plan
    showing what will run, what will be skipped, and why.

    Parameters
    ----------
    env
        Build environment with gateway, snapshot, and configuration.
    graph
        Target graph to use. If None, uses the graph from the target metadata service.
    requested
        Tuple of target names requested by the user.
    graph_source
        Source of dependency edges (only "hamilton" is supported).

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
    _ = graph_source
    runtime = build_driver()
    if graph is None:
        graph = runtime.graph
        native_names = native_target_names(runtime)
    else:
        graph = target_graph_from_hamilton(runtime, base_graph=graph)
        native_names = frozenset(target.name for target in graph.all_targets)

    closure = graph.topological_order(list(requested))

    if env.manifest_index is not None:
        manifests: Mapping[str, OutputManifest] = env.manifest_index
    else:
        manifest_list = env.gateway.build.list_manifests(
            repo=env.snapshot.repo,
            commit=env.snapshot.commit,
        )
        manifests = {m.target: m for m in manifest_list}

    entries: list[PlanEntry] = []
    upstream_status: dict[str, PlanStatus] = {}

    for target_name in closure:
        try:
            target = graph.get(target_name)
        except KeyError:
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

        entry = _compute_entry_for_target(
            target, env, manifests, upstream_status, native_names=native_names
        )
        entries.append(entry)
        upstream_status[target_name] = entry.status

    return HamiltonBuildPlan(
        requested=requested,
        closure=closure,
        entries=tuple(entries),
    )


def explain_plan(plan: HamiltonBuildPlan) -> list[StalenessExplanation]:
    """Generate staleness explanations for all targets in a plan.

    Provides detailed explanations for each target, useful for debugging
    incremental builds and understanding cache behavior.

    Parameters
    ----------
    plan
        The build plan to explain.

    Returns
    -------
    list[StalenessExplanation]
        List of explanations, one per target in the plan's closure.

    Examples
    --------
    >>> plan = compute_plan(env=env, requested=("risk_factors",))
    >>> explanations = explain_plan(plan)
    >>> for exp in explanations:
    ...     print(exp.summary())
    goids: stale - changed deps: ast
    """
    return [entry.explain_staleness() for entry in plan.entries]


__all__ = [
    "HamiltonBuildPlan",
    "ImplKind",
    "PlanEntry",
    "PlanReason",
    "PlanStatus",
    "StalenessExplanation",
    "compute_plan",
    "explain_plan",
]
