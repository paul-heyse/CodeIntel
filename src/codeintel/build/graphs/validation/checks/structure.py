"""Graph structure validation checks.

This module contains validation checks that analyze graph structure
for anomalies like cycles, hubs, and connectivity issues.

Check classes implement CheckProtocol from core/validation.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING, ClassVar

from codeintel.build.graphs.compute.metrics.components import find_strongly_connected
from codeintel.build.graphs.engine.datasets import dataset_snapshot_exists
from codeintel.build.graphs.rx.algos import (
    BetweennessOptions,
    GraphInput,
    ensure_directed_store,
    graph_to_store,
    total_degree_by_id,
)
from codeintel.build.graphs.rx.iterators import iter_edge_id_payloads
from codeintel.build.graphs.rx.normalize import stable_key
from codeintel.build.graphs.validation.base import GraphCheckBase
from codeintel.build.graphs.validation.findings import (
    CALL_SCC_MIN,
    CONFIG_KEY_MIN_THRESHOLD,
    HUB_DEGREE_RATIO,
    HUB_MIN_DEGREE_FLOOR,
    SAMPLE_LIMIT,
    hub_threshold,
)
from codeintel.core.compute.centrality import compute_betweenness
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
        if not _ensure_parquet_tables(ctx, ("graph.call_graph_edges", "graph.call_graph_nodes")):
            return []
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
        if not _ensure_parquet_tables(ctx, ("graph.import_graph_edges", "graph.import_modules")):
            return []
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
        if not _ensure_parquet_tables(ctx, ("graph.import_graph_edges", "graph.import_modules")):
            return []
        import_graph = ctx.import_graph
        if import_graph is None and ctx.engine is not None:
            import_graph = ctx.engine.import_graph()
        if import_graph is None:
            return []

        sccs = _strongly_connected_sets(import_graph)
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
        if not _ensure_parquet_tables(ctx, ("graph.import_graph_edges", "graph.import_modules")):
            return []
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
        if not _ensure_parquet_tables(ctx, ("graph.import_graph_edges", "graph.import_modules")):
            return []
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
        if not _ensure_parquet_tables(ctx, ("graph.import_graph_edges", "graph.import_modules")):
            return []
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
        if not _ensure_parquet_tables(ctx, ("graph.symbol_use_edges", "core.modules")):
            return []
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
        if not _ensure_parquet_tables(ctx, ("analytics.config_values", "core.modules")):
            return []
        if ctx.engine is None:
            return []

        cfg_bipartite = ctx.engine.config_module_bipartite()
        return _config_key_findings_impl(cfg_bipartite, ctx.repo, ctx.commit, ctx.logger)


# =============================================================================
# Implementation Functions (internal)
# =============================================================================


def _strongly_connected_sets(graph: GraphInput) -> list[set[Hashable]]:
    result = find_strongly_connected(graph)
    return [set(component.nodes) for component in result.components]


def _call_graph_findings_impl(
    call_graph: GraphInput,
    repo: str,
    commit: str,
    log: logging.Logger,
) -> list[dict[str, object]]:
    """Check for call graph structural anomalies (implementation).

    Returns
    -------
    list[dict[str, object]]
        Findings for call graph anomalies.
    """
    findings: list[dict[str, object]] = []
    store = graph_to_store(call_graph)
    kinds = {node: store.get_node_attrs(node).get("kind") for node in store.node_ids()}
    degree_map = total_degree_by_id(store)
    isolated: list[Hashable] = []
    for node_id in store.node_ids():
        if kinds.get(node_id) in {"module", "class"}:
            continue
        if degree_map.get(node_id, 0) == 0:
            isolated.append(node_id)
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

    sccs = [comp for comp in _strongly_connected_sets(call_graph) if len(comp) >= CALL_SCC_MIN]
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
                    "largest_cluster": [str(node) for node in sorted(largest, key=stable_key)][
                        : SAMPLE_LIMIT * 4
                    ]
                },
            }
        )

    degree_threshold = max(HUB_MIN_DEGREE_FLOOR, int(store.graph.num_nodes() * HUB_DEGREE_RATIO))
    degree_map = total_degree_by_id(store)
    hubs = [node for node, deg in degree_map.items() if deg > degree_threshold]
    if hubs:
        hubs_sorted = sorted(hubs, key=stable_key)
        sample = ", ".join(str(node) for node in hubs_sorted[:SAMPLE_LIMIT])
        log.warning("Validation: %d high-degree call graph hub(s) (sample: %s)", len(hubs), sample)
        findings.append(
            {
                "repo": repo,
                "commit": commit,
                "check_name": "call_graph_degree_hubs",
                "severity": "info",
                "path": None,
                "detail": f"{len(hubs)} hubs above degree {degree_threshold} (sample: {sample})",
                "context": {"hubs": [str(node) for node in hubs_sorted[: SAMPLE_LIMIT * 4]]},
            }
        )

    return findings


def _import_graph_findings_impl(
    import_graph: GraphInput,
    repo: str,
    commit: str,
    log: logging.Logger,
) -> list[dict[str, object]]:
    """Check for import graph structural anomalies (implementation).

    Returns
    -------
    list[dict[str, object]]
        Findings for import graph anomalies.
    """
    findings: list[dict[str, object]] = []
    sccs = _strongly_connected_sets(import_graph)
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
    import_graph: GraphInput,
    repo: str,
    commit: str,
    log: logging.Logger,
) -> list[dict[str, object]]:
    """Check for import graph hubs (implementation).

    Returns
    -------
    list[dict[str, object]]
        Findings for import hub anomalies.
    """
    findings: list[dict[str, object]] = []
    store = ensure_directed_store(import_graph)
    degree_threshold = hub_threshold(store.graph.num_nodes())
    degree_map = {str(node_id): degree for node_id, degree in total_degree_by_id(store).items()}
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
    import_graph: GraphInput,
    repo: str,
    commit: str,
    log: logging.Logger,
) -> list[dict[str, object]]:
    """Check for upward imports against layering (implementation).

    Returns
    -------
    list[dict[str, object]]
        Findings for upward import anomalies.
    """
    upward_edges: list[tuple[Hashable, Hashable]] = []
    store = graph_to_store(import_graph)
    for src_id, dst_id, _payload in iter_edge_id_payloads(store):
        src_layer = as_int(store.get_node_attrs(src_id).get("layer"))
        dst_layer = as_int(store.get_node_attrs(dst_id).get("layer"))
        if src_layer is None or dst_layer is None:
            continue
        if src_layer > dst_layer:
            upward_edges.append((src_id, dst_id))
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
    import_graph: GraphInput,
    repo: str,
    commit: str,
    log: logging.Logger,
) -> list[dict[str, object]]:
    """Check for bridge-like import modules (implementation).

    Returns
    -------
    list[dict[str, object]]
        Findings for import bridge anomalies.
    """
    betweenness: dict[str, float] = {}
    store = graph_to_store(import_graph)
    node_count = store.graph.num_nodes()
    if node_count > 0:
        sample_size = min(200, node_count)
        raw_betweenness = compute_betweenness(
            import_graph,
            options=BetweennessOptions(
                k=sample_size if sample_size < node_count else None,
                seed=0,
            ),
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
    symbol_graph: GraphInput,
    repo: str,
    commit: str,
    log: logging.Logger,
) -> list[dict[str, object]]:
    """Check for symbol graph structural anomalies (implementation).

    Returns
    -------
    list[dict[str, object]]
        Findings for symbol graph anomalies.
    """
    store = graph_to_store(symbol_graph)
    if store.graph.num_nodes() == 0:
        return []
    degree_map = total_degree_by_id(store)
    threshold = max(HUB_MIN_DEGREE_FLOOR, int(store.graph.num_nodes() * HUB_DEGREE_RATIO))
    high_degree = [node for node, deg in degree_map.items() if deg > threshold]
    if not high_degree:
        return []
    high_sorted = sorted(high_degree, key=stable_key)
    sample = ", ".join(str(node) for node in high_sorted[:SAMPLE_LIMIT])
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
            "context": {"hubs": [str(node) for node in high_sorted[: SAMPLE_LIMIT * 4]]},
        }
    ]


def _config_key_findings_impl(
    cfg_bipartite: GraphInput,
    repo: str,
    commit: str,
    log: logging.Logger,
) -> list[dict[str, object]]:
    """Check for broadly-used config keys (implementation).

    Returns
    -------
    list[dict[str, object]]
        Findings for config key usage anomalies.
    """
    store = graph_to_store(cfg_bipartite)
    if store.graph.num_nodes() == 0:
        return []
    keys = [node for node in store.node_ids() if store.get_node_attrs(node).get("bipartite") == 0]
    degree_map = total_degree_by_id(store)
    degs = {node: degree_map.get(node, 0) for node in keys}
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


def _ensure_parquet_tables(
    ctx: GraphValidationContext,
    table_keys: tuple[str, ...],
) -> bool:
    dataset_root = ctx.dataset_root_dir
    if dataset_root is None:
        return True
    for table_key in table_keys:
        if not dataset_snapshot_exists(dataset_root, table_key, ctx.commit):
            ctx.logger.warning("Validation table missing: %s", table_key)
            return False
    return True


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
