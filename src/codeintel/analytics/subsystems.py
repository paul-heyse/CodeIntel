"""
Infer higher-level subsystems from module coupling signals.

Subsystems cluster modules using import edges, symbol-use coupling, and shared
configuration keys. The resulting tables summarize membership and risk so
agents can navigate the architecture quickly.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import duckdb

from codeintel.config.models import SubsystemsConfig
from codeintel.config.schemas.sql_builder import ensure_schema

log = logging.getLogger(__name__)

MIN_SHARED_MODULES = 2
HASH_PREFIX_LENGTH = 16
MEDIUM_RISK_THRESHOLD = 0.4
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
class SubsystemEdgeStats:
    """Edge counts and fan-in/out sets for a subsystem."""

    internal_edges: int
    external_edges: int
    fan_in: set[str]
    fan_out: set[str]


@dataclass(frozen=True)
class SubsystemRisk:
    """Aggregated risk signals for a subsystem."""

    function_count: int
    total_risk: float
    max_risk: float | None
    high_risk: int
    level: str

    @property
    def avg_risk(self) -> float | None:
        """Average risk score across subsystem functions."""
        if self.function_count == 0:
            return None
        return self.total_risk / self.function_count


@dataclass(frozen=True)
class SubsystemBuildContext:
    """Reusable context for assembling subsystem rows."""

    cfg: SubsystemsConfig
    labels: dict[str, str]
    tags_by_module: dict[str, list[str]]
    import_edges: dict[tuple[str, str], int]
    risk_stats: dict[str, SubsystemRisk]
    now: datetime


@dataclass
class RiskTally:
    """Mutable accumulator for subsystem risk."""

    count: int = 0
    total: float = 0.0
    max_score: float | None = None
    high: int = 0

    def add(self, score: float, *, is_high: bool) -> None:
        """Update the tally with a new score."""
        self.count += 1
        self.total += score
        self.max_score = score if self.max_score is None else max(self.max_score, score)
        if is_high:
            self.high += 1


def build_subsystems(con: duckdb.DuckDBPyConnection, cfg: SubsystemsConfig) -> None:
    """Populate analytics.subsystems and analytics.subsystem_modules for a repo/commit."""
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

    modules, tags_by_module = _load_modules(con, cfg)
    if not modules:
        log.info("No modules available for subsystem inference; skipping.")
        return

    adjacency = _build_weighted_adjacency(con, cfg, modules)
    seed_labels = _seed_labels_from_tags(tags_by_module)
    labels = _label_propagation(modules, adjacency, seed_labels)
    labels = _reassign_small_clusters(labels, adjacency, cfg.min_modules)
    labels = _limit_clusters(labels, adjacency, cfg.max_subsystems)
    clusters = _clusters_from_labels(labels)

    import_edge_counts = _load_import_edge_counts(con, cfg)
    risk_stats = _aggregate_risk(con, cfg, labels)

    now = datetime.now(UTC)
    ctx = SubsystemBuildContext(
        cfg=cfg,
        labels=labels,
        tags_by_module=tags_by_module,
        import_edges=import_edge_counts,
        risk_stats=risk_stats,
        now=now,
    )
    subsystem_rows, membership_rows = _build_rows(clusters, ctx)

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
        edge_stats = _subsystem_edge_stats(member_list, ctx.labels, ctx.import_edges)
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


def _load_modules(
    con: duckdb.DuckDBPyConnection, cfg: SubsystemsConfig
) -> tuple[set[str], dict[str, list[str]]]:
    rows = con.execute(
        "SELECT module, tags FROM core.modules WHERE repo = ? AND commit = ?",
        [cfg.repo, cfg.commit],
    ).fetchall()
    if not rows:
        rows = con.execute("SELECT module, tags FROM core.modules").fetchall()

    modules: set[str] = set()
    tags_by_module: dict[str, list[str]] = {}
    for module, tags in rows:
        if module is None:
            continue
        module_name = str(module)
        modules.add(module_name)
        parsed_tags = _parse_tags(tags)
        if parsed_tags:
            tags_by_module[module_name] = parsed_tags
    return modules, tags_by_module


def _parse_tags(raw: object) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(tag) for tag in parsed]
            return [str(parsed)]
        except json.JSONDecodeError:
            return [raw]
    if isinstance(raw, list):
        return [str(tag) for tag in raw]
    return [str(raw)]


def _add_weight(
    adjacency: dict[str, dict[str, float]],
    a: str,
    b: str,
    weight: float,
) -> None:
    if a == b or weight <= 0:
        return
    adjacency[a][b] = adjacency[a].get(b, 0.0) + weight
    adjacency[b][a] = adjacency[b].get(a, 0.0) + weight


def _build_weighted_adjacency(
    con: duckdb.DuckDBPyConnection, cfg: SubsystemsConfig, modules: set[str]
) -> dict[str, dict[str, float]]:
    adjacency: dict[str, dict[str, float]] = defaultdict(dict)

    rows = con.execute(
        "SELECT src_module, dst_module FROM graph.import_graph_edges WHERE repo = ? AND commit = ?",
        [cfg.repo, cfg.commit],
    ).fetchall()
    for src, dst in rows:
        if src is None or dst is None:
            continue
        src_mod = str(src)
        dst_mod = str(dst)
        if src_mod in modules and dst_mod in modules:
            _add_weight(adjacency, src_mod, dst_mod, cfg.import_weight)

    rows = con.execute(
        """
        SELECT m_use.module, m_def.module
        FROM graph.symbol_use_edges su
        LEFT JOIN core.modules m_def ON m_def.path = su.def_path
        LEFT JOIN core.modules m_use ON m_use.path = su.use_path
        WHERE m_def.module IS NOT NULL AND m_use.module IS NOT NULL
        """
    ).fetchall()
    for use_module, def_module in rows:
        src_mod = str(use_module)
        dst_mod = str(def_module)
        if src_mod in modules and dst_mod in modules:
            _add_weight(adjacency, src_mod, dst_mod, cfg.symbol_weight)

    rows = con.execute("SELECT reference_modules FROM analytics.config_values").fetchall()
    for (mods_raw,) in rows:
        modules_list = _parse_tags(mods_raw)
        filtered = [m for m in modules_list if m in modules]
        if len(filtered) < MIN_SHARED_MODULES:
            continue
        weight = cfg.config_weight / max(len(filtered) - 1, 1)
        for idx, left in enumerate(filtered):
            for right in filtered[idx + 1 :]:
                _add_weight(adjacency, left, right, weight)

    return adjacency


def _seed_labels_from_tags(tags_by_module: dict[str, list[str]]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for module, tags in tags_by_module.items():
        if tags:
            labels[module] = str(tags[0]).lower()
    return labels


def _label_propagation(
    modules: set[str],
    adjacency: dict[str, dict[str, float]],
    seed_labels: dict[str, str],
    max_iters: int = 20,
) -> dict[str, str]:
    labels: dict[str, str] = {module: seed_labels.get(module, module) for module in modules}
    frozen: set[str] = set(seed_labels)
    ordered_nodes = sorted(modules)

    for _ in range(max_iters):
        changed = False
        for node in ordered_nodes:
            if node in frozen:
                continue
            weights: dict[str, float] = defaultdict(float)
            for neighbor, weight in adjacency.get(node, {}).items():
                neighbor_label = labels.get(neighbor)
                if neighbor_label is None:
                    continue
                weights[neighbor_label] += weight
            if not weights:
                continue
            best_label = max(weights.items(), key=lambda item: (item[1], item[0]))[0]
            if labels[node] != best_label:
                labels[node] = best_label
                changed = True
        if not changed:
            break
    return labels


def _reassign_small_clusters(
    labels: dict[str, str],
    adjacency: dict[str, dict[str, float]],
    min_size: int,
) -> dict[str, str]:
    if min_size <= 1:
        return labels
    cluster_sizes = _cluster_sizes(labels)
    stable_labels = {label for label, size in cluster_sizes.items() if size >= min_size}
    if len(stable_labels) == len(cluster_sizes):
        return labels

    new_labels = dict(labels)
    for node, label in labels.items():
        if cluster_sizes.get(label, 0) >= min_size:
            continue
        best_label = _best_neighbor_label(node, adjacency, new_labels, stable_labels)
        if best_label is not None:
            new_labels[node] = best_label
    return new_labels


def _best_neighbor_label(
    node: str,
    adjacency: dict[str, dict[str, float]],
    labels: dict[str, str],
    allowed_labels: set[str],
) -> str | None:
    weights: dict[str, float] = defaultdict(float)
    for neighbor, weight in adjacency.get(node, {}).items():
        label = labels.get(neighbor)
        if label is None or label not in allowed_labels:
            continue
        current = weights.get(label, 0.0)
        weights[label] = current + weight
    if not weights:
        return None
    return max(weights.items(), key=lambda item: (item[1], item[0]))[0]


def _limit_clusters(
    labels: dict[str, str],
    adjacency: dict[str, dict[str, float]],
    max_clusters: int | None,
) -> dict[str, str]:
    if max_clusters is None:
        return labels
    clusters = _clusters_from_labels(labels)
    if len(clusters) <= max_clusters:
        return labels

    kept = sorted(clusters.items(), key=lambda item: (-len(item[1]), item[0]))[:max_clusters]
    kept_labels = {label for label, _ in kept}
    new_labels = dict(labels)
    for node, label in labels.items():
        if label in kept_labels:
            continue
        best_label = _best_neighbor_label(node, adjacency, new_labels, kept_labels)
        if best_label is None:
            best_label = sorted(kept_labels)[0]
        new_labels[node] = best_label
    return new_labels


def _clusters_from_labels(labels: dict[str, str]) -> dict[str, list[str]]:
    clusters: dict[str, list[str]] = defaultdict(list)
    for module, label in labels.items():
        clusters[label].append(module)
    for mods in clusters.values():
        mods.sort()
    return clusters


def _cluster_sizes(labels: dict[str, str]) -> dict[str, int]:
    sizes: dict[str, int] = defaultdict(int)
    for label in labels.values():
        sizes[label] += 1
    return sizes


def _subsystem_id(repo: str, modules: Iterable[str]) -> str:
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
    role_counts: dict[str, int] = defaultdict(int)
    for module in members:
        parts = module.split(".")
        for idx, segment in enumerate(parts):
            role = ROLE_TAGS.get(segment.lower())
            if role:
                weight = 2 if idx == 0 else 1
                role_counts[role] += weight
        for tag in tags_by_module.get(module, []):
            role = ROLE_TAGS.get(str(tag).lower())
            if role:
                role_counts[role] += 3

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


def _aggregate_risk(
    con: duckdb.DuckDBPyConnection, cfg: SubsystemsConfig, labels: dict[str, str]
) -> dict[str, SubsystemRisk]:
    risk_by_label: dict[str, SubsystemRisk] = {}
    stats: dict[str, RiskTally] = defaultdict(RiskTally)
    rows = con.execute(
        """
        SELECT rf.risk_score, rf.risk_level, m.module
        FROM analytics.goid_risk_factors rf
        LEFT JOIN analytics.function_metrics fm
          ON fm.function_goid_h128 = rf.function_goid_h128
        LEFT JOIN core.modules m
          ON m.path = fm.rel_path
        WHERE rf.repo = ? AND rf.commit = ?
        """,
        [cfg.repo, cfg.commit],
    ).fetchall()
    for risk_score, risk_level, module in rows:
        if module is None:
            continue
        label = labels.get(str(module))
        if label is None:
            continue
        score = float(risk_score) if risk_score is not None else 0.0
        entry = stats[label]
        entry.add(score, is_high=risk_level == "high")

    for label, entry in stats.items():
        count = entry.count
        total = entry.total
        max_score = entry.max_score
        high = entry.high
        risk_level = "low"
        if high > 0:
            risk_level = "high"
        elif count > 0 and (total / count) >= MEDIUM_RISK_THRESHOLD:
            risk_level = "medium"
        risk_by_label[label] = SubsystemRisk(
            function_count=count,
            total_risk=total,
            max_risk=max_score,
            high_risk=high,
            level=risk_level,
        )
    return risk_by_label


def _load_import_edge_counts(
    con: duckdb.DuckDBPyConnection, cfg: SubsystemsConfig
) -> dict[tuple[str, str], int]:
    edge_counts: dict[tuple[str, str], int] = defaultdict(int)
    rows = con.execute(
        "SELECT src_module, dst_module FROM graph.import_graph_edges WHERE repo = ? AND commit = ?",
        [cfg.repo, cfg.commit],
    ).fetchall()
    for src, dst in rows:
        if src is None or dst is None:
            continue
        edge_counts[str(src), str(dst)] += 1
    return edge_counts


def _subsystem_edge_stats(
    members: list[str],
    labels: dict[str, str],
    import_edges: dict[tuple[str, str], int],
) -> SubsystemEdgeStats:
    member_set = set(members)
    label = labels[members[0]]
    internal_edges = 0
    external_edges = 0
    fan_in: set[str] = set()
    fan_out: set[str] = set()

    for (src, dst), count in import_edges.items():
        src_label = labels.get(src)
        dst_label = labels.get(dst)
        if src_label is None or dst_label is None:
            continue
        if src in member_set and dst in member_set:
            internal_edges += count
        elif src_label == label and dst_label != label:
            external_edges += count
            fan_out.add(dst_label)
        elif dst_label == label and src_label != label:
            external_edges += count
            fan_in.add(src_label)

    return SubsystemEdgeStats(
        internal_edges=internal_edges,
        external_edges=external_edges,
        fan_in=fan_in,
        fan_out=fan_out,
    )
