"""Planning DAG nodes for plan/explain outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import cast

from hamilton.function_modifiers import cache

from codeintel.build.hamilton.cache_index import CacheIndex
from codeintel.build.hamilton.cache_key_resolver import CacheKeyResolver
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.planning.model import (
    BuildPlan,
    PlanCacheStatus,
    PlanPredictedAction,
    PlanRequest,
    PlanTargetEntry,
)

UNKNOWN_CACHE_STATUS: PlanCacheStatus = "unknown"


@dataclass(frozen=True, slots=True)
class PlanContext:
    """Bundled plan inputs shared across planning nodes."""

    catalog: DagCatalog
    env: BuildEnv
    plan_request: PlanRequest
    runtime_fingerprint: str


@dataclass(frozen=True, slots=True)
class PlanGraphInputs:
    """Bundled graph inputs for plan entry construction."""

    plan_target_closure: tuple[str, ...]
    plan_target_subgraph_nodes: Mapping[str, tuple[str, ...]]
    plan_cache_probe: Mapping[str, PlanCacheStatus]
    preflight_block_map: Mapping[str, tuple[str, ...]]


@cache(behavior="ignore")
def plan_context(
    catalog: DagCatalog,
    env: BuildEnv,
    runtime_fingerprint: str,
    plan_request: PlanRequest | None = None,
) -> PlanContext:
    """Bundle plan inputs shared across nodes.

    Returns
    -------
    PlanContext
        Bundled planning inputs for downstream nodes.
    """
    resolved_request = _resolve_plan_request(plan_request)
    return PlanContext(
        catalog=catalog,
        env=env,
        plan_request=resolved_request,
        runtime_fingerprint=runtime_fingerprint,
    )


def plan_target_closure(
    catalog: DagCatalog,
    plan_request: PlanRequest | None = None,
) -> tuple[str, ...]:
    """Return the dependency closure for requested targets.

    Returns
    -------
    tuple[str, ...]
        Closure of target names.
    """
    resolved_request = _resolve_plan_request(plan_request)
    if not resolved_request.requested_targets:
        return ()
    return catalog.closure(resolved_request.requested_targets)


def plan_target_subgraph_nodes(
    catalog: DagCatalog,
    plan_target_closure: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    """Compute per-target node cones used for cache probing.

    Returns
    -------
    dict[str, tuple[str, ...]]
        Mapping of target names to their subgraph node lists.
    """
    subgraphs: dict[str, tuple[str, ...]] = {}
    for target in plan_target_closure:
        anchor = catalog.target_nodes.get(target)
        if anchor is None:
            subgraphs[target] = ()
            continue
        nodes = _collect_target_nodes(catalog=catalog, anchor=anchor, target=target)
        subgraphs[target] = _topo_sort_nodes(catalog=catalog, nodes=nodes)
    return subgraphs


def plan_node_versions(
    cache_key_resolver: CacheKeyResolver | None,
    plan_context: PlanContext,
    plan_target_subgraph_nodes: Mapping[str, tuple[str, ...]],
) -> dict[str, str]:
    """Compute cache key versions for nodes in the planning subgraph.

    Returns
    -------
    dict[str, str]
        Mapping of node names to cache key versions.
    """
    if cache_key_resolver is None:
        return {}

    nodes = {node for subgraph in plan_target_subgraph_nodes.values() for node in subgraph}
    if not nodes:
        return {}

    input_values: dict[str, object] = {
        "env": plan_context.env,
        "catalog": plan_context.catalog,
        "plan_request": plan_context.plan_request,
        "runtime_fingerprint": plan_context.runtime_fingerprint,
    }
    return cache_key_resolver.resolve_node_versions(nodes=nodes, input_values=input_values)


def plan_cache_probe(
    cache_index: CacheIndex | None,
    plan_node_versions: Mapping[str, str],
    plan_target_subgraph_nodes: Mapping[str, tuple[str, ...]],
) -> dict[str, PlanCacheStatus]:
    """Probe cache status for node versions.

    Returns
    -------
    dict[str, PlanCacheStatus]
        Mapping of node names to cache statuses.
    """
    all_nodes = {node for subgraph in plan_target_subgraph_nodes.values() for node in subgraph}
    if not all_nodes:
        return {}
    if cache_index is None:
        return dict.fromkeys(all_nodes, "unknown")

    pairs = [(node, version) for node, version in plan_node_versions.items() if version]
    results = cache_index.batch_has(pairs)
    statuses: dict[str, PlanCacheStatus] = {
        result.node: "hit" if result.hit else "miss" for result in results
    }
    for node in all_nodes:
        statuses.setdefault(node, "unknown")
    return statuses


def plan_graph_inputs(
    plan_target_closure: tuple[str, ...],
    plan_target_subgraph_nodes: Mapping[str, tuple[str, ...]],
    plan_cache_probe: Mapping[str, PlanCacheStatus],
    preflight_block_map: Mapping[str, tuple[str, ...]],
) -> PlanGraphInputs:
    """Bundle graph inputs for plan entry construction.

    Returns
    -------
    PlanGraphInputs
        Bundled graph inputs for plan entries.
    """
    return PlanGraphInputs(
        plan_target_closure=plan_target_closure,
        plan_target_subgraph_nodes=plan_target_subgraph_nodes,
        plan_cache_probe=plan_cache_probe,
        preflight_block_map=preflight_block_map,
    )


def plan(
    plan_context: PlanContext,
    plan_graph_inputs: PlanGraphInputs,
) -> BuildPlan:
    """Build a deterministic plan from catalog structure and cache probes.

    Returns
    -------
    BuildPlan
        Deterministic plan for the requested targets.
    """
    created_at = datetime.now(tz=UTC).isoformat()
    entries: list[PlanTargetEntry] = []

    catalog = plan_context.catalog
    plan_request = plan_context.plan_request

    for target in plan_graph_inputs.plan_target_closure:
        target_desc = catalog.get_target(target)
        if target_desc is None:
            entries.append(
                PlanTargetEntry(
                    target=target,
                    domain="unknown",
                    deps=(),
                    reads=(),
                    writes_tables=(),
                    writes_artifacts=(),
                    predicted_action="blocked",
                    block_reasons=("missing_target",),
                    cache_hit_ratio=None,
                    miss_nodes=(),
                )
            )
            continue

        reads, writes_tables, writes_artifacts = _resolve_io_details(
            catalog=catalog,
            target=target,
            include_io_details=plan_request.include_io_details,
        )
        block_reasons = plan_graph_inputs.preflight_block_map.get(target, ())
        nodes = plan_graph_inputs.plan_target_subgraph_nodes.get(target, ())
        node_statuses = cast(
            "list[PlanCacheStatus]",
            [plan_graph_inputs.plan_cache_probe.get(node, UNKNOWN_CACHE_STATUS) for node in nodes],
        )
        cache_hit_ratio = _cache_hit_ratio(
            statuses=node_statuses,
            include_cache_details=plan_request.include_cache_details,
        )
        miss_nodes = _miss_nodes(
            nodes=nodes,
            statuses=node_statuses,
            include_node_details=plan_request.include_node_details,
        )
        predicted_action = _predicted_action(block_reasons, node_statuses)

        entries.append(
            PlanTargetEntry(
                target=target,
                domain=target_desc.domain,
                deps=target_desc.dependencies,
                reads=reads,
                writes_tables=writes_tables,
                writes_artifacts=writes_artifacts,
                predicted_action=predicted_action,
                block_reasons=block_reasons,
                cache_hit_ratio=cache_hit_ratio,
                miss_nodes=miss_nodes,
            )
        )

    return BuildPlan(
        request=plan_request,
        closure=plan_graph_inputs.plan_target_closure,
        entries=tuple(entries),
        created_at_utc=created_at,
        build_fingerprint=plan_context.runtime_fingerprint,
    )


def _resolve_io_details(
    *,
    catalog: DagCatalog,
    target: str,
    include_io_details: bool,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    if not include_io_details:
        return (), (), ()
    surface = catalog.io_surfaces.get(target)
    if surface is None:
        return (), (), ()
    reads = tuple(sorted({read.table_key for read in surface.reads}))
    writes_tables = tuple(sorted({write.table_key for write in surface.table_writes}))
    writes_artifacts = tuple(sorted({write.artifact_name for write in surface.artifact_writes}))
    return reads, writes_tables, writes_artifacts


def _cache_hit_ratio(
    *,
    statuses: Sequence[PlanCacheStatus],
    include_cache_details: bool,
) -> float | None:
    if not include_cache_details:
        return None
    if not statuses:
        return None
    hits = sum(1 for status in statuses if status == "hit")
    return hits / len(statuses)


def _miss_nodes(
    *,
    nodes: Sequence[str],
    statuses: Sequence[PlanCacheStatus],
    include_node_details: bool,
) -> tuple[str, ...]:
    if not include_node_details:
        return ()
    misses: list[str] = []
    for node, status in zip(nodes, statuses, strict=False):
        if status != "hit":
            misses.append(node)
    return tuple(misses)


def _predicted_action(
    block_reasons: Sequence[str],
    statuses: Sequence[PlanCacheStatus],
) -> PlanPredictedAction:
    if block_reasons:
        return "blocked"
    if statuses and all(status == "hit" for status in statuses):
        return "reuse"
    return "compute"


def _resolve_plan_request(plan_request: PlanRequest | None) -> PlanRequest:
    if plan_request is not None:
        return plan_request
    return PlanRequest(
        requested_targets=(),
        mode="predict",
        include_node_details=False,
        include_io_details=False,
        include_cache_details=False,
    )


def _collect_target_nodes(
    *,
    catalog: DagCatalog,
    anchor: str,
    target: str,
) -> tuple[str, ...]:
    nodes: set[str] = set()
    stack = [anchor]
    while stack:
        node = stack.pop()
        if node in nodes:
            continue
        nodes.add(node)
        desc = catalog.nodes.get(node)
        if desc is None:
            continue
        for dep in desc.deps:
            dep_target = catalog.node_to_target.get(dep)
            if dep_target is not None and dep_target != target:
                nodes.add(dep)
                continue
            stack.append(dep)
    return tuple(nodes)


def _topo_sort_nodes(*, catalog: DagCatalog, nodes: Sequence[str]) -> tuple[str, ...]:
    node_set = set(nodes)
    in_degree: dict[str, int] = dict.fromkeys(node_set, 0)
    graph: dict[str, list[str]] = {node: [] for node in node_set}

    for node in node_set:
        desc = catalog.nodes.get(node)
        if desc is None:
            continue
        for dep in desc.deps:
            if dep not in node_set:
                continue
            graph[dep].append(node)
            in_degree[node] += 1

    ready = sorted([node for node, degree in in_degree.items() if degree == 0])
    ordered: list[str] = []

    while ready:
        current = ready.pop(0)
        ordered.append(current)
        for neighbor in sorted(graph[current]):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                ready.append(neighbor)
        ready.sort()

    if len(ordered) != len(node_set):
        return tuple(sorted(node_set))
    return tuple(ordered)


__all__ = [
    "plan",
    "plan_cache_probe",
    "plan_node_versions",
    "plan_target_closure",
    "plan_target_subgraph_nodes",
]
