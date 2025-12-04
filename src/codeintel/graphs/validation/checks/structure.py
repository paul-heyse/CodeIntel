"""Graph structure validation checks.

This module contains validation checks that analyze graph structure
for anomalies like cycles, hubs, and connectivity issues.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import networkx as nx

from codeintel.graphs.validation.checks.anomaly import (
    subsystem_disagreement_findings,
    symbol_community_findings,
)
from codeintel.graphs.validation.findings import (
    CALL_SCC_MIN,
    CONFIG_KEY_MIN_THRESHOLD,
    HUB_DEGREE_RATIO,
    HUB_MIN_DEGREE_FLOOR,
    SAMPLE_LIMIT,
    hub_threshold,
)

if TYPE_CHECKING:
    from codeintel.graphs.engine import GraphEngine


# =============================================================================
# Call Graph Checks
# =============================================================================


def call_graph_findings(
    call_graph: nx.DiGraph, repo: str, commit: str, log: logging.Logger
) -> list[dict[str, object]]:
    """Check for call graph structural anomalies.

    Parameters
    ----------
    call_graph
        Call graph to analyze.
    repo
        Repository identifier.
    commit
        Commit identifier.
    log
        Logger for output.

    Returns
    -------
    list[dict[str, object]]
        Findings for call graph anomalies.
    """
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


# =============================================================================
# Import Graph Checks
# =============================================================================


def import_graph_findings(
    import_graph: nx.DiGraph, repo: str, commit: str, log: logging.Logger
) -> list[dict[str, object]]:
    """Check for import graph structural anomalies.

    Parameters
    ----------
    import_graph
        Import graph to analyze.
    repo
        Repository identifier.
    commit
        Commit identifier.
    log
        Logger for output.

    Returns
    -------
    list[dict[str, object]]
        Findings for import graph anomalies.
    """
    findings: list[dict[str, object]] = []
    sccs = list(nx.strongly_connected_components(import_graph))
    findings.extend(import_cycle_findings(sccs, repo, commit, log))
    findings.extend(import_hub_findings(import_graph, repo, commit, log))
    findings.extend(import_upward_findings(import_graph, repo, commit, log))
    findings.extend(import_bridge_findings(import_graph, repo, commit, log))
    return findings


def import_cycle_findings(
    sccs: list[set[str]], repo: str, commit: str, log: logging.Logger
) -> list[dict[str, object]]:
    """Check for import cycles.

    Returns
    -------
    list[dict[str, object]]
        Findings for import cycle anomalies.
    """
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


def import_hub_findings(
    import_graph: nx.DiGraph, repo: str, commit: str, log: logging.Logger
) -> list[dict[str, object]]:
    """Check for import graph hubs.

    Returns
    -------
    list[dict[str, object]]
        Findings for import hub anomalies.
    """
    findings: list[dict[str, object]] = []
    degree_threshold = hub_threshold(import_graph.number_of_nodes())
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


def import_upward_findings(
    import_graph: nx.DiGraph, repo: str, commit: str, log: logging.Logger
) -> list[dict[str, object]]:
    """Check for upward imports against layering.

    Returns
    -------
    list[dict[str, object]]
        Findings for upward import anomalies.
    """
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


def import_bridge_findings(
    import_graph: nx.DiGraph, repo: str, commit: str, log: logging.Logger
) -> list[dict[str, object]]:
    """Check for bridge-like import modules.

    Returns
    -------
    list[dict[str, object]]
        Findings for bridge module anomalies.
    """
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


# =============================================================================
# Symbol Graph Checks
# =============================================================================


def symbol_graph_findings(
    symbol_graph: nx.Graph, repo: str, commit: str, log: logging.Logger
) -> list[dict[str, object]]:
    """Check for symbol graph structural anomalies.

    Returns
    -------
    list[dict[str, object]]
        Findings for symbol graph anomalies.
    """
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


# =============================================================================
# Config Key Checks
# =============================================================================


def config_key_findings(
    cfg_bipartite: nx.Graph, repo: str, commit: str, log: logging.Logger
) -> list[dict[str, object]]:
    """Check for broadly-used config keys.

    Returns
    -------
    list[dict[str, object]]
        Findings for config key usage anomalies.
    """
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


# =============================================================================
# Composite Check
# =============================================================================


def warn_graph_structure(
    engine: GraphEngine,
    repo: str,
    commit: str,
    log: logging.Logger | None = None,
) -> list[dict[str, object]]:
    """Emit warnings for common graph structure anomalies.

    Parameters
    ----------
    engine
        Graph engine providing access to graphs.
    repo
        Repository identifier.
    commit
        Commit identifier.
    log
        Optional logger; uses module logger if not provided.

    Returns
    -------
    list[dict[str, object]]
        Findings describing graph hotspots and anomalies.
    """
    findings: list[dict[str, object]] = []
    active_log = log or logging.getLogger(__name__)

    call_graph = engine.call_graph()
    findings.extend(call_graph_findings(call_graph, repo, commit, active_log))

    import_graph = engine.import_graph()
    findings.extend(import_graph_findings(import_graph, repo, commit, active_log))

    symbol_graph = engine.symbol_module_graph()
    findings.extend(symbol_graph_findings(symbol_graph, repo, commit, active_log))
    findings.extend(symbol_community_findings(engine.gateway, repo, commit, active_log))

    cfg_bipartite = engine.config_module_bipartite()
    findings.extend(config_key_findings(cfg_bipartite, repo, commit, active_log))

    findings.extend(subsystem_disagreement_findings(engine.gateway, repo, commit, active_log))
    return findings


__all__ = [
    "call_graph_findings",
    "config_key_findings",
    "import_bridge_findings",
    "import_cycle_findings",
    "import_graph_findings",
    "import_hub_findings",
    "import_upward_findings",
    "symbol_graph_findings",
    "warn_graph_structure",
]
