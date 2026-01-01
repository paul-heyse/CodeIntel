"""Graph structure validation checks.

This module contains validation checks that analyze graph structure
for anomalies like cycles, hubs, and connectivity issues.

Check classes implement CheckProtocol from core/validation.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING, Any, ClassVar

import networkx as nx

from codeintel.build.graphs.validation.base import GraphCheckBase
from codeintel.build.graphs.validation.findings import (
    CALL_SCC_MIN,
    CONFIG_KEY_MIN_THRESHOLD,
    HUB_DEGREE_RATIO,
    HUB_MIN_DEGREE_FLOOR,
    SAMPLE_LIMIT,
    hub_threshold,
)
from codeintel.core.data_models.ids import as_int

if TYPE_CHECKING:
    import logging

    from codeintel.build.graphs.validation.context import GraphValidationContext
    from codeintel.core.validation import ValidationSeverity


# =============================================================================
# Check Classes (CheckProtocol-compliant)
# =============================================================================


class CallGraphStructureCheck(GraphCheckBase):
    """Check for call graph structural anomalies.

    Detects isolated nodes, large strongly connected components (recursion),
    and high-degree hubs.
    """

    check_name: ClassVar[str] = "call_graph_structure"
    check_description: ClassVar[str] = "Detect call graph anomalies"
    default_severity: ClassVar[ValidationSeverity] = "warning"

    def execute(self, ctx: GraphValidationContext) -> list[dict[str, object]]:
        """Execute call graph structure checks.

        Parameters
        ----------
        ctx
            Graph validation context with call_graph or engine.

        Returns
        -------
        list[dict[str, object]]
            Findings for call graph anomalies.
        """
        _ = self  # Instance method required for CheckProtocol
        call_graph = ctx.call_graph
        if call_graph is None and ctx.engine is not None:
            call_graph = ctx.engine.call_graph()
        if call_graph is None:
            return []

        return _call_graph_findings_impl(call_graph, ctx.repo, ctx.commit, ctx.logger)


class ImportGraphStructureCheck(GraphCheckBase):
    """Check for import graph structural anomalies.

    Orchestrates import cycle, hub, upward, and bridge checks.
    """

    check_name: ClassVar[str] = "import_graph_structure"
    check_description: ClassVar[str] = "Detect import graph anomalies"
    default_severity: ClassVar[ValidationSeverity] = "warning"

    def execute(self, ctx: GraphValidationContext) -> list[dict[str, object]]:
        """Execute import graph structure checks.

        Parameters
        ----------
        ctx
            Graph validation context with import_graph or engine.

        Returns
        -------
        list[dict[str, object]]
            Findings for import graph anomalies.
        """
        _ = self  # Instance method required for CheckProtocol
        import_graph = ctx.import_graph
        if import_graph is None and ctx.engine is not None:
            import_graph = ctx.engine.import_graph()
        if import_graph is None:
            return []

        return _import_graph_findings_impl(import_graph, ctx.repo, ctx.commit, ctx.logger)


class ImportCycleCheck(GraphCheckBase):
    """Check for import cycles."""

    check_name: ClassVar[str] = "import_cycles"
    check_description: ClassVar[str] = "Detect import cycle anomalies"
    default_severity: ClassVar[ValidationSeverity] = "warning"

    def execute(self, ctx: GraphValidationContext) -> list[dict[str, object]]:
        """Execute import cycle check.

        Parameters
        ----------
        ctx
            Graph validation context with import_graph or engine.

        Returns
        -------
        list[dict[str, object]]
            Findings for import cycle anomalies.
        """
        _ = self  # Instance method required for CheckProtocol
        import_graph = ctx.import_graph
        if import_graph is None and ctx.engine is not None:
            import_graph = ctx.engine.import_graph()
        if import_graph is None:
            return []

        sccs = list(nx.strongly_connected_components(import_graph))
        return _import_cycle_findings_impl(sccs, ctx.repo, ctx.commit, ctx.logger)


class ImportHubCheck(GraphCheckBase):
    """Check for import graph hubs."""

    check_name: ClassVar[str] = "import_hubs"
    check_description: ClassVar[str] = "Detect import graph hubs"
    default_severity: ClassVar[ValidationSeverity] = "info"

    def execute(self, ctx: GraphValidationContext) -> list[dict[str, object]]:
        """Execute import hub check.

        Parameters
        ----------
        ctx
            Graph validation context with import_graph or engine.

        Returns
        -------
        list[dict[str, object]]
            Findings for import hub anomalies.
        """
        _ = self  # Instance method required for CheckProtocol
        import_graph = ctx.import_graph
        if import_graph is None and ctx.engine is not None:
            import_graph = ctx.engine.import_graph()
        if import_graph is None:
            return []

        return _import_hub_findings_impl(import_graph, ctx.repo, ctx.commit, ctx.logger)


class ImportUpwardCheck(GraphCheckBase):
    """Check for upward imports against layering."""

    check_name: ClassVar[str] = "import_upward"
    check_description: ClassVar[str] = "Detect upward import violations"
    default_severity: ClassVar[ValidationSeverity] = "info"

    def execute(self, ctx: GraphValidationContext) -> list[dict[str, object]]:
        """Execute upward import check.

        Parameters
        ----------
        ctx
            Graph validation context with import_graph or engine.

        Returns
        -------
        list[dict[str, object]]
            Findings for upward import anomalies.
        """
        _ = self  # Instance method required for CheckProtocol
        import_graph = ctx.import_graph
        if import_graph is None and ctx.engine is not None:
            import_graph = ctx.engine.import_graph()
        if import_graph is None:
            return []

        return _import_upward_findings_impl(import_graph, ctx.repo, ctx.commit, ctx.logger)


class ImportBridgeCheck(GraphCheckBase):
    """Check for bridge-like import modules."""

    check_name: ClassVar[str] = "import_bridges"
    check_description: ClassVar[str] = "Detect import bridge modules"
    default_severity: ClassVar[ValidationSeverity] = "info"

    def execute(self, ctx: GraphValidationContext) -> list[dict[str, object]]:
        """Execute import bridge check.

        Parameters
        ----------
        ctx
            Graph validation context with import_graph or engine.

        Returns
        -------
        list[dict[str, object]]
            Findings for import bridge anomalies.
        """
        _ = self  # Instance method required for CheckProtocol
        import_graph = ctx.import_graph
        if import_graph is None and ctx.engine is not None:
            import_graph = ctx.engine.import_graph()
        if import_graph is None:
            return []

        return _import_bridge_findings_impl(import_graph, ctx.repo, ctx.commit, ctx.logger)


class SymbolGraphCheck(GraphCheckBase):
    """Check for symbol graph structural anomalies."""

    check_name: ClassVar[str] = "symbol_graph_structure"
    check_description: ClassVar[str] = "Detect symbol graph anomalies"
    default_severity: ClassVar[ValidationSeverity] = "warning"

    def execute(self, ctx: GraphValidationContext) -> list[dict[str, object]]:
        """Execute symbol graph structure check.

        Parameters
        ----------
        ctx
            Graph validation context with symbol_graph or engine.

        Returns
        -------
        list[dict[str, object]]
            Findings for symbol graph anomalies.
        """
        _ = self  # Instance method required for CheckProtocol
        symbol_graph = ctx.symbol_graph
        if symbol_graph is None and ctx.engine is not None:
            symbol_graph = ctx.engine.symbol_module_graph()
        if symbol_graph is None:
            return []

        return _symbol_graph_findings_impl(symbol_graph, ctx.repo, ctx.commit, ctx.logger)


class ConfigKeyCheck(GraphCheckBase):
    """Check for broadly-used config keys."""

    check_name: ClassVar[str] = "config_key_usage"
    check_description: ClassVar[str] = "Detect widely-used config keys"
    default_severity: ClassVar[ValidationSeverity] = "info"

    def execute(self, ctx: GraphValidationContext) -> list[dict[str, object]]:
        """Execute config key usage check.

        Parameters
        ----------
        ctx
            Graph validation context with engine.

        Returns
        -------
        list[dict[str, object]]
            Findings for config key usage anomalies.
        """
        _ = self  # Instance method required for CheckProtocol
        if ctx.engine is None:
            return []

        cfg_bipartite = ctx.engine.config_module_bipartite()
        return _config_key_findings_impl(cfg_bipartite, ctx.repo, ctx.commit, ctx.logger)


# =============================================================================
# Implementation Functions (internal)
# =============================================================================


def _call_graph_findings_impl(
    call_graph: nx.DiGraph, repo: str, commit: str, log: logging.Logger
) -> list[dict[str, object]]:
    """Check for call graph structural anomalies (implementation).

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
                "context": {
                    "largest_cluster": sorted(str(node) for node in largest)[: SAMPLE_LIMIT * 4]
                },
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


def _import_graph_findings_impl(
    import_graph: nx.DiGraph, repo: str, commit: str, log: logging.Logger
) -> list[dict[str, object]]:
    """Check for import graph structural anomalies (implementation).

    Returns
    -------
    list[dict[str, object]]
        Findings for import graph anomalies.
    """
    findings: list[dict[str, object]] = []
    sccs = list(nx.strongly_connected_components(import_graph))
    findings.extend(_import_cycle_findings_impl(sccs, repo, commit, log))
    findings.extend(_import_hub_findings_impl(import_graph, repo, commit, log))
    findings.extend(_import_upward_findings_impl(import_graph, repo, commit, log))
    findings.extend(_import_bridge_findings_impl(import_graph, repo, commit, log))
    return findings


def _import_cycle_findings_impl(
    sccs: list[set[Hashable]], repo: str, commit: str, log: logging.Logger
) -> list[dict[str, object]]:
    """Check for import cycles (implementation).

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
                "context": {
                    "largest_cycle": sorted(str(node) for node in largest)[: SAMPLE_LIMIT * 4]
                },
            }
        )

    cross_package_cycles = [
        comp
        for comp in sccs
        if len(comp) > 1 and len({str(module).split(".")[0] for module in comp}) > 1
    ]
    if cross_package_cycles:
        sample_cycle = sorted(str(node) for node in cross_package_cycles[0])[: SAMPLE_LIMIT * 4]
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


def _import_hub_findings_impl(
    import_graph: nx.DiGraph, repo: str, commit: str, log: logging.Logger
) -> list[dict[str, object]]:
    """Check for import graph hubs (implementation).

    Returns
    -------
    list[dict[str, object]]
        Findings for import hub anomalies.
    """
    findings: list[dict[str, object]] = []
    degree_threshold = hub_threshold(import_graph.number_of_nodes())
    degree_map: dict[str, int] = {}
    for node in import_graph.nodes:
        out_deg_raw = import_graph.out_degree(node)
        in_deg_raw = import_graph.in_degree(node)
        out_deg = int(out_deg_raw)
        in_deg = int(in_deg_raw)
        degree_map[str(node)] = out_deg + in_deg
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


def _import_upward_findings_impl(
    import_graph: nx.DiGraph, repo: str, commit: str, log: logging.Logger
) -> list[dict[str, object]]:
    """Check for upward imports against layering (implementation).

    Returns
    -------
    list[dict[str, object]]
        Findings for upward import anomalies.
    """
    upward_edges = []
    for src, dst in import_graph.edges:
        src_layer = as_int(import_graph.nodes.get(src, {}).get("layer"))
        dst_layer = as_int(import_graph.nodes.get(dst, {}).get("layer"))
        if src_layer is None or dst_layer is None:
            continue
        if src_layer > dst_layer:
            upward_edges.append((src, dst))
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


def _import_bridge_findings_impl(
    import_graph: nx.DiGraph, repo: str, commit: str, log: logging.Logger
) -> list[dict[str, object]]:
    """Check for bridge-like import modules (implementation).

    Returns
    -------
    list[dict[str, object]]
        Findings for import bridge anomalies.
    """
    betweenness: dict[str, float] = {}
    if import_graph.number_of_nodes() > 0:
        sample_size = min(200, import_graph.number_of_nodes())
        raw_betweenness = nx.betweenness_centrality(
            import_graph,
            k=sample_size if sample_size < import_graph.number_of_nodes() else None,
        )
        betweenness = {str(node): float(score) for node, score in raw_betweenness.items()}
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


def _symbol_graph_findings_impl(
    symbol_graph: nx.Graph, repo: str, commit: str, log: logging.Logger
) -> list[dict[str, object]]:
    """Check for symbol graph structural anomalies (implementation).

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


def _config_key_findings_impl(
    cfg_bipartite: nx.Graph, repo: str, commit: str, log: logging.Logger
) -> list[dict[str, object]]:
    """Check for broadly-used config keys (implementation).

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
    high_keys: list[str] = []
    for node, deg in degs.items():
        if deg <= key_threshold:
            continue
        if isinstance(node, tuple) and len(node) > 1:
            high_keys.append(str(node[1]))
        else:
            high_keys.append(str(node))
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
# All Check Classes (for runner registration)
# =============================================================================

ALL_STRUCTURE_CHECKS: tuple[type[GraphCheckBase], ...] = (
    CallGraphStructureCheck,
    ImportGraphStructureCheck,
    ImportCycleCheck,
    ImportHubCheck,
    ImportUpwardCheck,
    ImportBridgeCheck,
    SymbolGraphCheck,
    ConfigKeyCheck,
)

__all__ = [
    # Check classes
    "ALL_STRUCTURE_CHECKS",
    "CallGraphStructureCheck",
    "ConfigKeyCheck",
    "ImportBridgeCheck",
    "ImportCycleCheck",
    "ImportGraphStructureCheck",
    "ImportHubCheck",
    "ImportUpwardCheck",
    "SymbolGraphCheck",
]
