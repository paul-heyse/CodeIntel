"""Lightweight validations for graph construction outputs."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import networkx as nx

from codeintel.analytics.graph_runtime import (
    GraphRuntime,
    GraphRuntimeOptions,
    resolve_graph_runtime,
)
from codeintel.analytics.parsing.validation import GraphValidationReporter
from codeintel.config.primitives import SnapshotRef
from codeintel.graphs.engine import GraphEngine
from codeintel.graphs.function_catalog import FunctionCatalog, load_function_catalog
from codeintel.graphs.function_catalog_service import FunctionCatalogProvider
from codeintel.storage.gateway import DuckDBError, StorageGateway
from codeintel.storage.sql_helpers import ensure_schema

SAMPLE_LIMIT = 5
SYMBOL_COMMUNITY_MIN = 2
CONFIG_KEY_MIN_THRESHOLD = 2
HUB_MIN_DEGREE_FLOOR = 10
HUB_DEGREE_RATIO = 0.1
CALL_SCC_MIN = 5


@dataclass(frozen=True)
class GraphValidationOptions:
    """Optional controls for graph validation behavior."""

    severity_overrides: Mapping[str, Literal["info", "warning", "error"]] | None = None
    hard_fail: bool = False
    max_findings_per_rule: int | None = None


def _hub_threshold(node_count: int) -> int:
    """
    Compute a hub threshold that scales with graph size.

    Parameters
    ----------
    node_count : int
        Number of nodes in the graph.

    Returns
    -------
    int
        Degree threshold used to flag hubs.
    """
    return max(HUB_MIN_DEGREE_FLOOR, int(node_count * HUB_DEGREE_RATIO))


def run_graph_validations(
    gateway: StorageGateway,
    *,
    snapshot: SnapshotRef,
    catalog_provider: FunctionCatalogProvider | None = None,
    runtime: GraphRuntime | GraphRuntimeOptions,
    options: GraphValidationOptions | None = None,
) -> None:
    """
    Emit warnings for common graph integrity issues.

    Checks include:
    - Files with functions in AST that are missing GOIDs.
    - Call graph edges whose callsites lie outside caller spans.
    - Modules with no GOIDs (orphans).

    Raises
    ------
    RuntimeError
        When hard_fail is enabled and error-level findings are present.
    """
    validation_opts = options or GraphValidationOptions()
    active_log = logging.getLogger(__name__)
    repo = snapshot.repo
    commit = snapshot.commit
    _log_db_snapshot(gateway, repo, commit, active_log)
    catalog = (
        catalog_provider.catalog()
        if catalog_provider is not None
        else load_function_catalog(gateway, repo=snapshot.repo, commit=snapshot.commit)
    )
    resolved_runtime = _resolve_validation_runtime(
        gateway,
        snapshot=snapshot,
        runtime=runtime,
    )
    engine: GraphEngine = resolved_runtime.engine
    findings = []
    findings.extend(_warn_missing_function_goids(gateway, repo, commit, active_log))
    findings.extend(_warn_callsite_span_mismatches(gateway, catalog, repo, commit, active_log))
    findings.extend(_warn_orphan_modules(gateway, repo, commit, active_log, catalog))
    findings.extend(warn_graph_structure(engine, repo, commit, active_log))
    normalized_findings = _apply_severity_overrides(
        findings, validation_opts.severity_overrides
    )
    capped_findings = _cap_findings(normalized_findings, validation_opts.max_findings_per_rule)
    _persist_findings(gateway, capped_findings, repo, commit)
    active_log.info(
        "Graph validation completed for %s@%s: %d finding(s)",
        repo,
        commit,
        len(capped_findings),
    )
    if validation_opts.hard_fail and _has_error_findings(capped_findings):
        message = "Graph validation failed with error-level findings"
        raise RuntimeError(message)


def _resolve_validation_runtime(
    gateway: StorageGateway,
    *,
    snapshot: SnapshotRef,
    runtime: GraphRuntime | GraphRuntimeOptions,
) -> GraphRuntime:
    runtime_snapshot = (
        runtime.options.snapshot if isinstance(runtime, GraphRuntime) else runtime.snapshot
    )
    if runtime_snapshot is not None and (
        runtime_snapshot.repo != snapshot.repo or runtime_snapshot.commit != snapshot.commit
    ):
        message = "GraphRuntime snapshot mismatch for validation run"
        raise ValueError(message)

    if isinstance(runtime, GraphRuntime):
        return runtime

    options = runtime if runtime.snapshot is not None else replace(runtime, snapshot=snapshot)
    return resolve_graph_runtime(gateway, snapshot, options)


def _apply_severity_overrides(
    findings: list[dict[str, object]],
    overrides: Mapping[str, Literal["info", "warning", "error"]] | None,
) -> list[dict[str, object]]:
    if not overrides:
        return findings
    normalized: list[dict[str, object]] = []
    for finding in findings:
        check = str(finding.get("check_name") or "")
        override = overrides.get(check)
        if override is None:
            normalized.append(finding)
            continue
        updated = dict(finding)
        updated["severity"] = override
        normalized.append(updated)
    return normalized


def _cap_findings(
    findings: list[dict[str, object]], max_per_rule: int | None
) -> list[dict[str, object]]:
    if max_per_rule is None or max_per_rule <= 0:
        return findings
    counts: dict[str, int] = {}
    capped: list[dict[str, object]] = []
    for finding in findings:
        check = str(finding.get("check_name") or "")
        seen = counts.get(check, 0)
        if seen >= max_per_rule:
            continue
        counts[check] = seen + 1
        capped.append(finding)
    return capped


def _has_error_findings(findings: list[dict[str, object]]) -> bool:
    return any(finding.get("severity") == "error" for finding in findings)


def _warn_missing_function_goids(
    gateway: StorageGateway, repo: str, commit: str, log: logging.Logger
) -> list[dict[str, object]]:
    try:
        rows = gateway.con.execute(
            """
            WITH funcs AS (
                SELECT path AS rel_path, COUNT(*) AS function_count
                FROM core.ast_nodes
                WHERE repo = ? AND commit = ? AND node_type IN ('FunctionDef', 'AsyncFunctionDef')
                GROUP BY path
            ),
            goids AS (
                SELECT rel_path, COUNT(*) AS goid_count
                FROM core.goids
                WHERE repo = ? AND commit = ? AND kind IN ('function', 'method')
                GROUP BY rel_path
            )
            SELECT f.rel_path, f.function_count, COALESCE(g.goid_count, 0) AS goid_count
            FROM funcs f
            LEFT JOIN goids g ON g.rel_path = f.rel_path
            WHERE COALESCE(g.goid_count, 0) < f.function_count
            ORDER BY f.rel_path
            """,
            [repo, commit, repo, commit],
        ).fetchall()
    except DuckDBError:
        return []

    if not rows:
        return []
    sample = ", ".join(path for path, _, _ in rows[:5])
    log.warning(
        "Validation: %d file(s) have functions without GOIDs (sample: %s)",
        len(rows),
        sample,
    )
    return [
        {
            "repo": repo,
            "commit": commit,
            "check_name": "missing_function_goids",
            "severity": "warning",
            "path": path,
            "detail": f"{function_count} functions, {goid_count} GOIDs",
            "context": {"function_count": function_count, "goid_count": goid_count},
        }
        for path, function_count, goid_count in rows
    ]


def _warn_callsite_span_mismatches(
    gateway: StorageGateway,
    catalog: FunctionCatalog,
    repo: str,
    commit: str,
    log: logging.Logger,
) -> list[dict[str, object]]:
    spans_by_goid = {span.goid: span for span in catalog.function_spans}
    try:
        rows = gateway.con.execute(
            """
            SELECT
                e.caller_goid_h128,
                e.callsite_path,
                e.callsite_line
            FROM graph.call_graph_edges e
            WHERE e.callsite_line IS NOT NULL
              AND e.repo = ? AND e.commit = ?
            """,
            [repo, commit],
        ).fetchall()
    except DuckDBError:
        return []

    mismatches = []
    for goid, path, line in rows:
        span = spans_by_goid.get(int(goid)) if goid is not None else None
        if span is None:
            continue
        if line < span.start_line or line > span.end_line:
            mismatches.append((path, line, span.start_line, span.end_line))

    if not mismatches:
        return []
    sample = ", ".join(f"{path}:{line}" for path, line, _, _ in mismatches[:5])
    log.warning(
        "Validation: %d call graph edges fall outside caller spans (sample: %s)",
        len(mismatches),
        sample,
    )
    return [
        {
            "repo": repo,
            "commit": commit,
            "check_name": "callsite_span_mismatch",
            "severity": "warning",
            "path": path,
            "detail": f"callsite {line} outside span {start}-{end}",
            "context": {"callsite_line": line, "start_line": start, "end_line": end},
        }
        for path, line, start, end in mismatches
    ]


def _call_graph_findings(
    call_graph: nx.DiGraph, repo: str, commit: str, log: logging.Logger
) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    call_graph_any: Any = call_graph
    kinds = nx.get_node_attributes(call_graph, "kind")
    isolated = [
        node
        for node in call_graph.nodes
        if kinds.get(node) not in {"module", "class"} and int(call_graph_any.degree(node)) == 0
    ]
    if isolated:
        isolated_sample = ", ".join(str(node) for node in isolated[:SAMPLE_LIMIT])
        log.warning(
            "Validation: %d isolated call graph node(s) (sample: %s)",
            len(isolated),
            isolated_sample,
        )
        findings.append(
            {
                "repo": repo,
                "commit": commit,
                "check_name": "call_graph_isolated_nodes",
                "severity": "warning",
                "path": None,
                "detail": f"{len(isolated)} isolated functions (sample: {isolated_sample})",
                "context": {"isolated_nodes": isolated[: SAMPLE_LIMIT * 4]},
            }
        )

    sccs = [
        comp for comp in nx.strongly_connected_components(call_graph) if len(comp) >= CALL_SCC_MIN
    ]
    if sccs:
        largest = max(sccs, key=len)
        log.warning(
            "Validation: %d recursive call cluster(s) detected (largest size %d)",
            len(sccs),
            len(largest),
        )
        findings.append(
            {
                "repo": repo,
                "commit": commit,
                "check_name": "call_graph_large_scc",
                "severity": "warning",
                "path": None,
                "detail": f"{len(sccs)} recursion cluster(s), largest size {len(largest)}",
                "context": {"largest_cluster": sorted(largest)[: SAMPLE_LIMIT * 4]},
            }
        )

    degree_threshold = max(
        HUB_MIN_DEGREE_FLOOR, int(call_graph.number_of_nodes() * HUB_DEGREE_RATIO)
    )
    degree_map = {node: int(call_graph_any.degree(node)) for node in call_graph.nodes}
    hubs = [node for node, deg in degree_map.items() if deg > degree_threshold]
    if hubs:
        sample = ", ".join(str(node) for node in hubs[:SAMPLE_LIMIT])
        log.warning("Validation: %d high-degree call graph hub(s) (sample: %s)", len(hubs), sample)
        findings.append(
            {
                "repo": repo,
                "commit": commit,
                "check_name": "call_graph_degree_hubs",
                "severity": "info",
                "path": None,
                "detail": f"{len(hubs)} hubs above degree {degree_threshold} (sample: {sample})",
                "context": {"hubs": hubs[: SAMPLE_LIMIT * 4]},
            }
        )

    return findings


def _import_graph_findings(
    import_graph: nx.DiGraph, repo: str, commit: str, log: logging.Logger
) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    sccs = list(nx.strongly_connected_components(import_graph))
    findings.extend(_import_cycle_findings(sccs, repo, commit, log))
    findings.extend(_import_hub_findings(import_graph, repo, commit, log))
    findings.extend(_import_upward_findings(import_graph, repo, commit, log))
    findings.extend(_import_bridge_findings(import_graph, repo, commit, log))
    return findings


def _import_cycle_findings(
    sccs: list[set[str]], repo: str, commit: str, log: logging.Logger
) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    large_sccs = [comp for comp in sccs if len(comp) > HUB_MIN_DEGREE_FLOOR // 2]
    if large_sccs:
        largest = max(large_sccs, key=len)
        log.warning(
            "Validation: %d import cycles detected (largest size %d)",
            len(large_sccs),
            len(largest),
        )
        findings.append(
            {
                "repo": repo,
                "commit": commit,
                "check_name": "import_graph_large_scc",
                "severity": "warning",
                "path": None,
                "detail": f"{len(large_sccs)} import cycles, largest size {len(largest)}",
                "context": {"largest_cycle": sorted(largest)[: SAMPLE_LIMIT * 4]},
            }
        )

    cross_package_cycles = [
        comp
        for comp in sccs
        if len(comp) > 1 and len({str(module).split(".")[0] for module in comp}) > 1
    ]
    if cross_package_cycles:
        sample_cycle = sorted(cross_package_cycles[0])[: SAMPLE_LIMIT * 4]
        log.warning(
            "Validation: %d import cycle(s) cross package boundaries", len(cross_package_cycles)
        )
        findings.append(
            {
                "repo": repo,
                "commit": commit,
                "check_name": "import_graph_cross_package_cycles",
                "severity": "warning",
                "path": None,
                "detail": f"{len(cross_package_cycles)} cycles cross package boundaries",
                "context": {"sample_cycle": sample_cycle},
            }
        )
    return findings


def _import_hub_findings(
    import_graph: nx.DiGraph, repo: str, commit: str, log: logging.Logger
) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    degree_threshold = _hub_threshold(import_graph.number_of_nodes())
    degree_map = {}
    for node in import_graph.nodes:
        out_deg_raw = import_graph.out_degree(node)
        in_deg_raw = import_graph.in_degree(node)
        out_deg = int(cast("int", out_deg_raw))
        in_deg = int(cast("int", in_deg_raw))
        degree_map[node] = out_deg + in_deg
    hubs = [node for node, deg in degree_map.items() if deg > degree_threshold]
    if hubs:
        sample = ", ".join(sorted(hubs)[:SAMPLE_LIMIT])
        log.warning("Validation: %d import graph hub(s) (sample: %s)", len(hubs), sample)
        findings.append(
            {
                "repo": repo,
                "commit": commit,
                "check_name": "import_graph_degree_hubs",
                "severity": "info",
                "path": None,
                "detail": f"{len(hubs)} hubs above degree {degree_threshold} (sample: {sample})",
                "context": {"hubs": sorted(hubs)[: SAMPLE_LIMIT * 4]},
            }
        )
    return findings


def _import_upward_findings(
    import_graph: nx.DiGraph, repo: str, commit: str, log: logging.Logger
) -> list[dict[str, object]]:
    upward_edges = [
        (src, dst)
        for src, dst in import_graph.edges
        if import_graph.nodes.get(src, {}).get("layer") is not None
        and import_graph.nodes.get(dst, {}).get("layer") is not None
        and import_graph.nodes[src]["layer"] > import_graph.nodes[dst]["layer"]
    ]
    if not upward_edges:
        return []
    sample_edges = [f"{s}->{d}" for s, d in upward_edges[:SAMPLE_LIMIT]]
    log.warning(
        "Validation: %d upward import edge(s) against layering (sample: %s)",
        len(upward_edges),
        ", ".join(sample_edges),
    )
    return [
        {
            "repo": repo,
            "commit": commit,
            "check_name": "import_graph_upward_edges",
            "severity": "info",
            "path": None,
            "detail": f"{len(upward_edges)} edges go from deeper to shallower layer",
            "context": {"sample_edges": sample_edges},
        }
    ]


def _import_bridge_findings(
    import_graph: nx.DiGraph, repo: str, commit: str, log: logging.Logger
) -> list[dict[str, object]]:
    betweenness: dict[str, float] = {}
    if import_graph.number_of_nodes() > 0:
        sample_size = min(200, import_graph.number_of_nodes())
        betweenness = nx.betweenness_centrality(
            import_graph,
            k=sample_size if sample_size < import_graph.number_of_nodes() else None,
        )
    if not betweenness:
        return []
    max_score = max(betweenness.values())
    threshold = max_score * 0.25 if max_score > 0 else 0.0
    bridges = [node for node, score in betweenness.items() if score >= threshold and score > 0]
    if not bridges:
        return []
    sample = ", ".join(sorted(bridges)[:SAMPLE_LIMIT])
    log.warning(
        "Validation: %d bridge-like import modules (sample: %s)",
        len(bridges),
        sample,
    )
    return [
        {
            "repo": repo,
            "commit": commit,
            "check_name": "import_graph_bridges",
            "severity": "info",
            "path": None,
            "detail": f"{len(bridges)} modules with high betweenness (sample: {sample})",
            "context": {"bridges": sorted(bridges)[: SAMPLE_LIMIT * 4]},
        }
    ]


def _symbol_graph_findings(
    symbol_graph: nx.Graph, repo: str, commit: str, log: logging.Logger
) -> list[dict[str, object]]:
    if symbol_graph.number_of_nodes() == 0:
        return []
    symbol_graph_any: Any = symbol_graph
    degree_map = {node: int(symbol_graph_any.degree(node)) for node in symbol_graph.nodes}
    threshold = max(HUB_MIN_DEGREE_FLOOR, int(symbol_graph.number_of_nodes() * HUB_DEGREE_RATIO))
    high_degree = [node for node, deg in degree_map.items() if deg > threshold]
    if not high_degree:
        return []
    sample = ", ".join(str(node) for node in high_degree[:SAMPLE_LIMIT])
    log.warning(
        "Validation: %d symbol graph hubs detected (sample: %s)",
        len(high_degree),
        sample,
    )
    return [
        {
            "repo": repo,
            "commit": commit,
            "check_name": "symbol_graph_hubs",
            "severity": "warning",
            "path": None,
            "detail": f"{len(high_degree)} high-degree symbol hubs (sample: {sample})",
            "context": {"hubs": high_degree[: SAMPLE_LIMIT * 4]},
        }
    ]


def _symbol_community_findings(
    gateway: StorageGateway, repo: str, commit: str, log: logging.Logger
) -> list[dict[str, object]]:
    comm_counts = gateway.con.execute(
        """
        SELECT symbol_community_id, COUNT(*) AS count
        FROM analytics.symbol_graph_metrics_modules
        WHERE repo = ? AND commit = ? AND symbol_community_id IS NOT NULL
        GROUP BY symbol_community_id
        HAVING count > ?
        """,
        [repo, commit, SYMBOL_COMMUNITY_MIN],
    ).fetchall()
    if not comm_counts:
        return []
    largest = max(comm_counts, key=lambda row: row[1])
    log.warning("Validation: large symbol communities detected (largest size %d)", largest[1])
    return [
        {
            "repo": repo,
            "commit": commit,
            "check_name": "symbol_graph_large_community",
            "severity": "info",
            "path": None,
            "detail": f"{len(comm_counts)} communities exceed threshold; largest {largest[1]}",
            "context": {"communities": comm_counts[: SAMPLE_LIMIT * 4]},
        }
    ]


def _config_key_findings(
    cfg_bipartite: nx.Graph, repo: str, commit: str, log: logging.Logger
) -> list[dict[str, object]]:
    if cfg_bipartite.number_of_nodes() == 0:
        return []
    keys = [n for n, d in cfg_bipartite.nodes(data=True) if d.get("bipartite") == 0]
    cfg_bipartite_any: Any = cfg_bipartite
    degs = {node: int(cfg_bipartite_any.degree(node)) for node in keys}
    key_threshold = max(CONFIG_KEY_MIN_THRESHOLD, int(len(keys) * 0.05))
    high_keys = [str(k[1]) for k, deg in degs.items() if deg > key_threshold]
    if not high_keys:
        return []
    sample = ", ".join(high_keys[:SAMPLE_LIMIT])
    log.warning(
        "Validation: %d config keys referenced broadly (sample: %s)",
        len(high_keys),
        sample,
    )
    return [
        {
            "repo": repo,
            "commit": commit,
            "check_name": "config_keys_broad_usage",
            "severity": "info",
            "path": None,
            "detail": f"{len(high_keys)} keys used widely (sample: {sample})",
            "context": {"keys": high_keys[: SAMPLE_LIMIT * 4]},
        }
    ]


def _subsystem_disagreement_findings(
    gateway: StorageGateway, repo: str, commit: str, log: logging.Logger
) -> list[dict[str, object]]:
    disagreements = gateway.con.execute(
        """
        SELECT module, subsystem_id, import_community_id
        FROM analytics.subsystem_agreement
        WHERE repo = ? AND commit = ? AND agrees = false
        """,
        [repo, commit],
    ).fetchall()
    if not disagreements:
        return []
    sample = ", ".join(str(row[0]) for row in disagreements[:SAMPLE_LIMIT])
    log.warning(
        "Validation: %d module(s) disagree on subsystem vs import community (sample: %s)",
        len(disagreements),
        sample,
    )
    return [
        {
            "repo": repo,
            "commit": commit,
            "check_name": "subsystem_community_disagreement",
            "severity": "warning",
            "path": None,
            "detail": f"{len(disagreements)} modules disagree (sample: {sample})",
            "context": {"modules": disagreements[: SAMPLE_LIMIT * 4]},
        }
    ]


def warn_graph_structure(
    engine: GraphEngine,
    repo: str,
    commit: str,
    log: logging.Logger | None = None,
) -> list[dict[str, object]]:
    """
    Emit warnings for common graph structure anomalies.

    Returns
    -------
    list[dict[str, object]]
        Findings describing graph hotspots and anomalies.
    """
    findings: list[dict[str, object]] = []
    active_log = log or logging.getLogger(__name__)

    call_graph = engine.call_graph()
    findings.extend(_call_graph_findings(call_graph, repo, commit, active_log))

    import_graph = engine.import_graph()
    findings.extend(_import_graph_findings(import_graph, repo, commit, active_log))

    symbol_graph = engine.symbol_module_graph()
    findings.extend(_symbol_graph_findings(symbol_graph, repo, commit, active_log))
    findings.extend(_symbol_community_findings(engine.gateway, repo, commit, active_log))

    cfg_bipartite = engine.config_module_bipartite()
    findings.extend(_config_key_findings(cfg_bipartite, repo, commit, active_log))

    findings.extend(_subsystem_disagreement_findings(engine.gateway, repo, commit, active_log))
    return findings


def _warn_orphan_modules(
    gateway: StorageGateway,
    repo: str,
    commit: str,
    log: logging.Logger,
    catalog: FunctionCatalog,
) -> list[dict[str, object]]:
    query_failed = False
    try:
        con = gateway.con
        rows = con.execute(
            """
            SELECT m.path
            FROM core.modules m
            LEFT JOIN core.goids g
              ON g.rel_path = m.path AND g.repo = ? AND g.commit = ? AND g.kind = 'module'
            WHERE m.repo = ? AND m.commit = ? AND g.goid_h128 IS NULL
            """,
            [repo, commit, repo, commit],
        ).fetchall()
        if rows:
            stats = con.execute(
                """
                WITH module_goids AS (
                    SELECT rel_path, COUNT(*) AS cnt
                    FROM core.goids
                    WHERE repo = ? AND commit = ? AND kind = 'module'
                    GROUP BY rel_path
                )
                SELECT m.path, COALESCE(g.cnt, 0) AS module_goids
                FROM core.modules m
                LEFT JOIN module_goids g ON g.rel_path = m.path
                WHERE m.repo = ? AND m.commit = ?
                ORDER BY module_goids ASC, m.path
                LIMIT 5
                """,
                [repo, commit, repo, commit],
            ).fetchall()
            sample_detail = ", ".join(f"{path} (module_goids={cnt})" for path, cnt in stats)
            log.info(
                "Orphan module debug: repo=%s commit=%s sample=%s",
                repo,
                commit,
                sample_detail,
            )
    except DuckDBError:
        query_failed = True
        rows = []

    if query_failed and catalog.module_by_path:
        rows = [(path,) for path in catalog.module_by_path]

    if not rows:
        return []
    sample = ", ".join(path for (path,) in rows[:5])
    log.warning("Validation: %d module(s) have no GOIDs (sample: %s)", len(rows), sample)
    return [
        {
            "repo": repo,
            "commit": commit,
            "check_name": "orphan_module",
            "severity": "warning",
            "path": path,
            "detail": "module has no GOIDs",
            "context": {},
        }
        for (path,) in rows
    ]


def _persist_findings(
    gateway: StorageGateway, findings: list[dict[str, object]], repo: str, commit: str
) -> None:
    if not findings:
        return
    con = gateway.con
    ensure_schema(con, "analytics.graph_validation")
    con.execute(
        "DELETE FROM analytics.graph_validation WHERE repo = ? AND commit = ?",
        [repo, commit],
    )
    reporter = GraphValidationReporter(repo=repo, commit=commit)
    for finding in findings:
        graph_name = str(finding.get("check_name") or "graph_validation")
        entity_ref = finding.get("path") or finding.get("entity_id") or finding.get("graph_name")
        entity_id = str(entity_ref) if entity_ref is not None else graph_name
        issue = str(finding.get("issue") or finding.get("severity") or graph_name)
        severity = str(finding.get("severity") or "info")
        rel_path = finding.get("path")
        detail = str(finding.get("detail") or "")
        metadata = finding.get("context")
        extras = {
            "severity": severity,
            "rel_path": str(rel_path) if rel_path is not None else None,
            "metadata": metadata,
        }
        reporter.record(
            graph_name=graph_name,
            entity_id=entity_id,
            issue=issue,
            detail=detail,
            extras=extras,
        )
    reporter.flush(gateway)


def _log_db_snapshot(gateway: StorageGateway, repo: str, commit: str, log: logging.Logger) -> None:
    """Record table counts to aid debugging validation state."""
    con = gateway.con

    def _count(query: str, *, use_params: bool) -> int:
        try:
            row = (
                con.execute(query, [repo, commit]).fetchone()
                if use_params
                else con.execute(query).fetchone()
            )
        except DuckDBError as exc:  # pragma: no cover - defensive logging
            log.warning("Validation snapshot count failed for %s: %s", query, exc)
            return -1
        if row is None:
            return 0
        value = row[0]
        return int(value) if value is not None else 0

    counts = {
        "modules": _count(
            "SELECT COUNT(*) FROM core.modules WHERE repo = ? AND commit = ?", use_params=True
        ),
        "goids": _count(
            "SELECT COUNT(*) FROM core.goids WHERE repo = ? AND commit = ?", use_params=True
        ),
        "module_goids": _count(
            "SELECT COUNT(*) FROM core.goids WHERE repo = ? AND commit = ? AND kind = 'module'",
            use_params=True,
        ),
        "class_goids": _count(
            "SELECT COUNT(*) FROM core.goids WHERE repo = ? AND commit = ? AND kind = 'class'",
            use_params=True,
        ),
        "function_goids": _count(
            """
            SELECT COUNT(*) FROM core.goids
            WHERE repo = ? AND commit = ? AND kind IN ('function', 'method')
            """,
            use_params=True,
        ),
        "call_nodes": _count("SELECT COUNT(*) FROM graph.call_graph_nodes", use_params=False),
        "call_edges": _count("SELECT COUNT(*) FROM graph.call_graph_edges", use_params=False),
    }
    snapshot = (
        f"[graph_validation] repo={repo} commit={commit} "
        f"modules={counts['modules']} goids={counts['goids']} "
        f"module_goids={counts['module_goids']} class_goids={counts['class_goids']} "
        f"function_goids={counts['function_goids']} "
        f"call_nodes={counts['call_nodes']} call_edges={counts['call_edges']}"
    )
    log.info(snapshot)
    _append_log(snapshot)


def _append_log(message: str) -> None:
    """Append a timestamped line to build/logs/pipeline.log for offline inspection."""
    log_path = Path("build/logs/pipeline.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=UTC).isoformat()
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"{timestamp} {message}\n")


# Backwards-compatible alias for legacy imports.
_warn_graph_structure = warn_graph_structure
