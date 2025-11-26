"""Subsystem analytics orchestration."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import networkx as nx

from codeintel.analytics.context import AnalyticsContext
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
from codeintel.config import SubsystemsStepConfig
from codeintel.config.schemas.sql_builder import ensure_schema
from codeintel.graphs.engine import GraphEngine, GraphKind, NxGraphEngine
from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)

HASH_PREFIX_LENGTH = 16
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

    cfg: SubsystemsStepConfig
    labels: dict[str, str]
    tags_by_module: dict[str, list[str]]
    import_graph: nx.DiGraph
    risk_stats: dict[str, SubsystemRisk]
    now: datetime


def build_subsystems(
    gateway: StorageGateway,
    cfg: SubsystemsStepConfig,
    *,
    context: AnalyticsContext | None = None,
    engine: GraphEngine | None = None,
) -> None:
    """Populate analytics.subsystems and analytics.subsystem_modules for a repo/commit."""
    con = gateway.con
    ensure_schema(con, "analytics.subsystems")
    ensure_schema(con, "analytics.subsystem_modules")

    con.execute(
        "DELETE FROM analytics.subsystems WHERE repo = ? AND commit = ?",
        [cfg.repo, cfg.commit],
    )
    con.execute(
        "DELETE FROM analytics.subsystem_modules WHERE repo = ? AND commit = ?",
        [cfg.repo, cfg.commit],
    )

    modules, tags_by_module = load_modules(gateway, cfg)
    if not modules:
        log.info("No modules available for subsystem inference; skipping.")
        return

    affinity_graph = build_weighted_graph(gateway, cfg, modules)
    adjacency = graph_to_adjacency(affinity_graph)
    labels = label_propagation_nx(affinity_graph, seed_labels_from_tags(tags_by_module))
    labels = reassign_small_clusters(labels, adjacency, cfg.min_modules)
    labels = limit_clusters(labels, adjacency, cfg.max_subsystems)

    if context is not None and (context.repo != cfg.repo or context.commit != cfg.commit):
        log.warning(
            "subsystems context mismatch: context=%s@%s cfg=%s@%s",
            context.repo,
            context.commit,
            cfg.repo,
            cfg.commit,
        )
    graph_engine = engine
    if graph_engine is None:
        graph_engine = NxGraphEngine(
            gateway=gateway,
            repo=cfg.repo,
            commit=cfg.commit,
            use_gpu=context.use_gpu if context is not None else False,
        )
        if context is not None and context.repo == cfg.repo and context.commit == cfg.commit:
            graph_engine.seed(GraphKind.IMPORT_GRAPH, context.import_graph)
    ctx = SubsystemBuildContext(
        cfg=cfg,
        labels=labels,
        tags_by_module=tags_by_module,
        import_graph=graph_engine.import_graph(),
        risk_stats=aggregate_risk(gateway, cfg, labels),
        now=datetime.now(UTC),
    )
    subsystem_rows, membership_rows = _build_rows(clusters_from_labels(labels), ctx)

    if membership_rows:
        con.executemany(
            """
            INSERT INTO analytics.subsystem_modules (
                repo, commit, subsystem_id, module, role
            ) VALUES (?, ?, ?, ?, ?)
            """,
            membership_rows,
        )

    if subsystem_rows:
        con.executemany(
            """
            INSERT INTO analytics.subsystems (
                repo, commit, subsystem_id, name, description, module_count, modules_json,
                entrypoints_json, internal_edge_count, external_edge_count, fan_in, fan_out,
                function_count, avg_risk_score, max_risk_score, high_risk_function_count,
                risk_level, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            subsystem_rows,
        )
        log.info(
            "subsystems populated: %d subsystems, %d memberships for %s@%s",
            len(subsystem_rows),
            len(membership_rows),
            cfg.repo,
            cfg.commit,
        )


def _build_rows(
    clusters: dict[str, list[str]],
    ctx: SubsystemBuildContext,
) -> tuple[list[tuple[Any, ...]], list[tuple[Any, ...]]]:
    subsystem_rows: list[tuple[Any, ...]] = []
    membership_rows: list[tuple[Any, ...]] = []
    default_risk = SubsystemRisk(0, 0.0, None, 0, "low")

    for label, members in clusters.items():
        member_list = sorted(members)
        subsystem_id = _subsystem_id(ctx.cfg.repo, member_list)
        dominant_role = _dominant_role(member_list, ctx.tags_by_module)
        name = _derive_name(member_list, subsystem_id, dominant_role)
        description = _describe_subsystem(member_list, name, dominant_role)
        entrypoints = _entrypoints_for_cluster(member_list, ctx.tags_by_module)
        edge_stats = compute_subsystem_edge_stats(member_list, ctx.labels, ctx.import_graph)
        risk = ctx.risk_stats.get(label, default_risk)

        subsystem_rows.append(
            (
                ctx.cfg.repo,
                ctx.cfg.commit,
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
                ctx.cfg.repo,
                ctx.cfg.commit,
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
