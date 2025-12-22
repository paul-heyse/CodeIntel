"""Subsystem analytics orchestration."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from codeintel.analytics.subsystems.affinity import (
    build_weighted_graph,
    clusters_from_labels,
    graph_to_adjacency,
    label_propagation_nx,
    limit_clusters,
    load_modules,
    reassign_small_clusters,
    seed_labels_from_tags,
)
from codeintel.analytics.subsystems.edge_stats import (
    compute_subsystem_edge_stats,
)
from codeintel.analytics.subsystems.risk import SubsystemRisk, aggregate_risk
from codeintel.graphs.runtime import GraphRuntime, GraphRuntimeOptions, resolve_graph_runtime

if TYPE_CHECKING:
    import networkx as nx

    from codeintel.analytics.subsystems.affinity import (
        AffinityWeights,
    )
    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)

HASH_PREFIX_LENGTH = 16
DEFAULT_MIN_MODULES = 3


@dataclass(frozen=True)
class SubsystemOptions:
    """Options for subsystem inference."""

    min_modules: int = DEFAULT_MIN_MODULES
    max_subsystems: int | None = None
    weights: AffinityWeights | None = None


ROLE_TAGS = {
    "api": "api",
    "endpoint": "api",
    "routes": "api",
    "core": "core",
    "domain": "domain",
    "service": "core",
    "services": "core",
    "infra": "infra",
    "ops": "infra",
    "platform": "platform",
    "data": "data",
    "ml": "ml",
    "ai": "ml",
    "etl": "data",
    "cli": "cli",
    "tool": "cli",
    "tests": "tests",
    "test": "tests",
}


@dataclass(frozen=True)
class SubsystemBuildContext:
    """Reusable context for assembling subsystem rows."""

    snapshot: SnapshotRef
    labels: dict[str, str]
    tags_by_module: dict[str, list[str]]
    import_graph: nx.DiGraph
    risk_stats: dict[str, SubsystemRisk]
    now: datetime


@dataclass(frozen=True)
class SubsystemRows:
    """Container for subsystem and membership rows."""

    subsystem_rows: list[tuple[Any, ...]]
    membership_rows: list[tuple[Any, ...]]


def build_subsystem_rows(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    *,
    runtime: GraphRuntime | GraphRuntimeOptions | None = None,
    options: SubsystemOptions | None = None,
) -> SubsystemRows:
    """
    Build analytics.subsystems and analytics.subsystem_modules rows for a repo/commit.

    Parameters
    ----------
    gateway :
        Storage gateway backing the analytics tables.
    snapshot :
        Repository and commit identifiers.
    runtime :
        Shared graph runtime or options describing how to build one.
    options :
        Subsystem inference options.

    Returns
    -------
    SubsystemRows
        Container holding subsystem and membership rows.
    """
    opts = options or SubsystemOptions()

    modules, tags_by_module = load_modules(gateway, snapshot)
    if not modules:
        log.info("No modules available for subsystem inference; skipping.")
        return SubsystemRows(subsystem_rows=[], membership_rows=[])

    affinity_graph = build_weighted_graph(gateway, snapshot, modules, weights=opts.weights)
    adjacency = graph_to_adjacency(affinity_graph)
    labels = label_propagation_nx(affinity_graph, seed_labels_from_tags(tags_by_module))
    labels = reassign_small_clusters(labels, adjacency, opts.min_modules)
    labels = limit_clusters(labels, adjacency, opts.max_subsystems)

    runtime_opts: GraphRuntimeOptions
    if isinstance(runtime, GraphRuntime):
        runtime_opts = runtime.options
    else:
        runtime_opts = runtime or GraphRuntimeOptions()

    resolved_runtime = resolve_graph_runtime(
        gateway,
        snapshot,
        runtime_opts,
    )
    ctx = SubsystemBuildContext(
        snapshot=snapshot,
        labels=labels,
        tags_by_module=tags_by_module,
        import_graph=resolved_runtime.ensure_import_graph(),
        risk_stats=aggregate_risk(gateway, snapshot, labels),
        now=datetime.now(UTC),
    )
    subsystem_rows, membership_rows = _build_rows(clusters_from_labels(labels), ctx)

    log.info(
        "subsystems rows built: %d subsystems, %d memberships for %s@%s",
        len(subsystem_rows),
        len(membership_rows),
        snapshot.repo,
        snapshot.commit,
    )
    return SubsystemRows(subsystem_rows=subsystem_rows, membership_rows=membership_rows)


def _build_rows(
    clusters: dict[str, list[str]],
    ctx: SubsystemBuildContext,
) -> tuple[list[tuple[Any, ...]], list[tuple[Any, ...]]]:
    subsystem_rows: list[tuple[Any, ...]] = []
    membership_rows: list[tuple[Any, ...]] = []
    default_risk = SubsystemRisk(0, 0.0, None, 0, "low")

    for label, members in clusters.items():
        member_list = sorted(members)
        subsystem_id = _subsystem_id(ctx.snapshot.repo, member_list)
        dominant_role = _dominant_role(member_list, ctx.tags_by_module)
        name = _derive_name(member_list, subsystem_id, dominant_role)
        description = _describe_subsystem(member_list, name, dominant_role)
        entrypoints = _entrypoints_for_cluster(member_list, ctx.tags_by_module)
        edge_stats = compute_subsystem_edge_stats(member_list, ctx.labels, ctx.import_graph)
        risk = ctx.risk_stats.get(label, default_risk)

        subsystem_rows.append(
            (
                ctx.snapshot.repo,
                ctx.snapshot.commit,
                subsystem_id,
                name,
                description,
                len(member_list),
                member_list,
                entrypoints if entrypoints else None,
                edge_stats.internal_edges,
                edge_stats.external_edges,
                len(edge_stats.fan_in),
                len(edge_stats.fan_out),
                risk.function_count,
                risk.avg_risk,
                risk.max_risk,
                risk.high_risk,
                risk.level,
                ctx.now,
            )
        )

        membership_rows.extend(
            (
                ctx.snapshot.repo,
                ctx.snapshot.commit,
                subsystem_id,
                module,
                _role_from_tags(ctx.tags_by_module.get(module)),
            )
            for module in member_list
        )

    return subsystem_rows, membership_rows




def _subsystem_id(repo: str, modules: list[str]) -> str:
    raw = f"{repo}:{','.join(sorted(modules))}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return digest[:HASH_PREFIX_LENGTH]


def _derive_name(modules: list[str], subsystem_id: str, dominant_role: str | None) -> str:
    prefix = _common_prefix(modules)
    if prefix:
        base = prefix.replace(".", "_")
        if dominant_role and not base.startswith(f"{dominant_role}_"):
            return f"{dominant_role}_{base}"
        return base
    if dominant_role:
        return f"{dominant_role}_subsys_{subsystem_id[:6]}"
    return f"subsys_{subsystem_id[:8]}"


def _common_prefix(modules: list[str]) -> str | None:
    if not modules:
        return None
    parts = [module.split(".") for module in modules]
    prefix: list[str] = []
    for segment in zip(*parts, strict=False):
        if len(set(segment)) == 1:
            prefix.append(segment[0])
        else:
            break
    if prefix:
        return ".".join(prefix[:3])
    return None


def _describe_subsystem(modules: list[str], name: str, dominant_role: str | None) -> str:
    examples = ", ".join(modules[:3])
    role_hint = f" ({dominant_role})" if dominant_role else ""
    return f"Subsystem {name}{role_hint} covering {len(modules)} modules (e.g., {examples})."


def _role_from_tags(tags: list[str] | None) -> str | None:
    if not tags:
        return None
    for tag in tags:
        tag_lower = str(tag).lower()
        role = ROLE_TAGS.get(tag_lower)
        if role:
            return role
    return str(tags[0]).lower()


def _dominant_role(
    members: list[str],
    tags_by_module: dict[str, list[str]],
) -> str | None:
    role_counts: dict[str, int] = {}
    for module in members:
        parts = module.split(".")
        for idx, segment in enumerate(parts):
            role = ROLE_TAGS.get(segment.lower())
            if role:
                weight = 2 if idx == 0 else 1
                role_counts[role] = role_counts.get(role, 0) + weight
        for tag in tags_by_module.get(module, []):
            role = ROLE_TAGS.get(str(tag).lower())
            if role:
                role_counts[role] = role_counts.get(role, 0) + 3

    if not role_counts:
        return None
    return max(role_counts.items(), key=lambda item: (item[1], item[0]))[0]


def _entrypoints_for_cluster(
    modules: list[str], tags_by_module: dict[str, list[str]]
) -> list[dict[str, str]]:
    return [
        {"kind": "tag", "tag": str(tag), "module": module}
        for module in modules
        for tag in tags_by_module.get(module, ())
    ]
