"""Operation-driven pipeline planning and orchestration.

This module provides functions to determine the minimal pipeline execution
required for a given serving operation, based on the operation's declared
requirements (datasets and graph runtimes) and the dataset contract metadata.

The core workflow:
1. Look up operation requirements from the operation catalog
2. Expand dataset dependencies transitively via dataset contracts
3. Partition tables by owner_package to identify which stages are needed
4. Choose the appropriate canonical PipelineSpec
5. Optionally execute the spec via the unified executor
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from codeintel.config.datasets import (
    DATASET_CONTRACTS,
    DATASET_CONTRACTS_BY_TABLE_KEY,
    DatasetContract,
)
from codeintel.pipeline.spec import (
    ANALYTICS_ONLY,
    FULL_PIPELINE,
    GRAPHS_ONLY,
    INGEST_ONLY,
    NOOP_PIPELINE,
    PipelineSpec,
)
from codeintel.serving.operations.catalog import Operation, get_operation

if TYPE_CHECKING:
    from codeintel.config.models import ToolsConfig
    from codeintel.config.primitives import BuildPaths, SnapshotRef
    from codeintel.runtime import TriggerKind
    from codeintel.storage.gateway import StorageGateway
    from codeintel.storage.run_tracking import PipelineRunRecord

log = logging.getLogger(__name__)

OwnerPackage = Literal["core", "analytics", "graphs", "qa", "docs"]


# -----------------------------------------------------------------------------
# Data Structures
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class OpPrereqSummary:
    """Summary of prerequisites computed for an operation.

    This dataclass provides introspection into the planning decisions made
    for an operation, useful for debugging and observability.

    Attributes
    ----------
    op
        The operation being planned for.
    required_tables
        Direct dataset table_keys from Operation.required_datasets.
    expanded_tables
        All dataset table_keys after transitive dependency expansion.
    core_tables
        Tables owned by the core package (ingestion).
    graph_tables
        Tables owned by the graphs package.
    analytics_tables
        Tables owned by the analytics package.
    required_graphs
        Graph runtimes required by the operation.
    """

    op: Operation
    required_tables: frozenset[str]
    expanded_tables: frozenset[str]
    core_tables: frozenset[str]
    graph_tables: frozenset[str]
    analytics_tables: frozenset[str]
    required_graphs: frozenset[str]


# -----------------------------------------------------------------------------
# Contract Index Building
# -----------------------------------------------------------------------------


def _build_contract_index() -> tuple[dict[str, DatasetContract], dict[str, DatasetContract]]:
    """Build lookup dictionaries for dataset contracts.

    Returns
    -------
    tuple[dict[str, DatasetContract], dict[str, DatasetContract]]
        A tuple of (by_table_key, by_name) dictionaries for contract lookup.
    """
    by_table = DATASET_CONTRACTS_BY_TABLE_KEY
    by_name = {contract.name: contract for contract in DATASET_CONTRACTS.values()}
    return by_table, by_name


# -----------------------------------------------------------------------------
# Operation Lookup
# -----------------------------------------------------------------------------


def _get_required_from_operation(op_id: str) -> tuple[Operation, set[str], set[str]]:
    """Look up an operation and extract its direct requirements.

    Parameters
    ----------
    op_id
        Operation identifier (e.g., "function.summary", "datasets.list").

    Returns
    -------
    tuple[Operation, set[str], set[str]]
        A tuple of (operation, required_tables, required_graphs).

    Raises
    ------
    ValueError
        If the operation ID is not found in the catalog.
    """
    op = get_operation(op_id)
    if op is None:
        message = f"Unknown operation id: {op_id}"
        raise ValueError(message)

    required_tables = set(op.required_datasets)
    required_graphs = set(op.required_graphs)
    return op, required_tables, required_graphs


# -----------------------------------------------------------------------------
# Dataset Dependency Expansion
# -----------------------------------------------------------------------------


def _expand_dataset_dependencies(required_tables: set[str]) -> set[str]:
    """Expand required dataset table_keys by following upstream_dependencies.

    Performs a BFS traversal to compute the transitive closure of all
    upstream dependencies for the given table_keys.

    Parameters
    ----------
    required_tables
        Directly required dataset table_keys from Operation.required_datasets.

    Returns
    -------
    set[str]
        All dataset table_keys needed, including upstream dependencies.
    """
    by_table, by_name = _build_contract_index()

    needed: set[str] = set(required_tables)
    queue: list[str] = list(required_tables)

    while queue:
        table_key = queue.pop()
        contract = by_table.get(table_key)
        if contract is None:
            continue

        deps = contract.upstream_dependencies or ()
        for upstream_name in deps:
            upstream_contract = by_name.get(upstream_name)
            if upstream_contract is None:
                continue
            upstream_table = upstream_contract.table_key
            if upstream_table not in needed:
                needed.add(upstream_table)
                queue.append(upstream_table)

    return needed


# -----------------------------------------------------------------------------
# Owner Package Partitioning
# -----------------------------------------------------------------------------


def _partition_by_owner_package(
    table_keys: Iterable[str],
) -> tuple[set[str], set[str], set[str]]:
    """Partition dataset table_keys into core / graphs / analytics buckets.

    Uses the owner_package field from dataset contracts to classify tables.
    Tables with unknown or None owner_package are attributed to core.

    Parameters
    ----------
    table_keys
        Iterable of dataset table_keys to partition.

    Returns
    -------
    tuple[set[str], set[str], set[str]]
        A tuple of (core_tables, graph_tables, analytics_tables).
    """
    by_table, _ = _build_contract_index()

    core_tables: set[str] = set()
    graph_tables: set[str] = set()
    analytics_tables: set[str] = set()

    for table_key in table_keys:
        contract = by_table.get(table_key)
        if contract is None:
            # Unknown table, attribute to core by default
            core_tables.add(table_key)
            continue

        owner = contract.owner_package or ""
        if owner == "core":
            core_tables.add(table_key)
        elif owner == "graphs":
            graph_tables.add(table_key)
        elif owner == "analytics":
            analytics_tables.add(table_key)
        else:
            # Unknown owner_package, attribute to core
            core_tables.add(table_key)

    return core_tables, graph_tables, analytics_tables


# -----------------------------------------------------------------------------
# Stage Flag Computation
# -----------------------------------------------------------------------------


def _compute_stage_flags(
    *,
    core_tables: set[str],
    graph_tables: set[str],
    analytics_tables: set[str],
    required_graphs: set[str],
    include_analytics: bool,
) -> tuple[bool, bool, bool]:
    """Determine which pipeline stages are needed based on requirements.

    Parameters
    ----------
    core_tables
        Tables owned by the core package.
    graph_tables
        Tables owned by the graphs package.
    analytics_tables
        Tables owned by the analytics package.
    required_graphs
        Graph runtimes required by the operation.
    include_analytics
        If True, include analytics stage when any datasets/graphs are needed.

    Returns
    -------
    tuple[bool, bool, bool]
        A tuple of (need_ingestion, need_graphs, need_analytics).
    """
    # Ingestion is needed if we have core tables, graph tables, or need graphs
    need_ingestion = bool(core_tables) or bool(graph_tables) or bool(required_graphs)

    # Graphs stage is needed if we have graph-owned tables or require graph runtimes
    need_graphs = bool(graph_tables) or bool(required_graphs)

    # Analytics stage is needed if we have analytics tables or caller opted in
    need_analytics = bool(analytics_tables)
    if include_analytics and (need_ingestion or need_graphs):
        need_analytics = True

    return need_ingestion, need_graphs, need_analytics


# -----------------------------------------------------------------------------
# Spec Selection
# -----------------------------------------------------------------------------


def _choose_spec(
    *,
    need_ingestion: bool,
    need_graphs: bool,
    need_analytics: bool,
) -> PipelineSpec:
    """Map stage flags to one of the canonical PipelineSpecs.

    Parameters
    ----------
    need_ingestion
        Whether ingestion stage is required.
    need_graphs
        Whether graphs stage is required.
    need_analytics
        Whether analytics stage is required.

    Returns
    -------
    PipelineSpec
        The appropriate canonical pipeline spec.
    """
    if not (need_ingestion or need_graphs or need_analytics):
        return NOOP_PIPELINE

    # All three stages needed
    if need_ingestion and need_graphs and need_analytics:
        return FULL_PIPELINE

    # Single-stage cases
    if need_ingestion and not need_graphs and not need_analytics:
        return INGEST_ONLY
    if need_graphs and not need_ingestion and not need_analytics:
        return GRAPHS_ONLY
    if need_analytics and not need_ingestion and not need_graphs:
        return ANALYTICS_ONLY

    # Mixed but not all three - default to full pipeline
    # This handles cases like ingest+graphs, ingest+analytics, graphs+analytics
    return FULL_PIPELINE


# -----------------------------------------------------------------------------
# Main Planning Function
# -----------------------------------------------------------------------------


def build_pipeline_for_operation(
    op_id: str,
    _snapshot: SnapshotRef,
    *,
    include_analytics: bool = True,
) -> PipelineSpec:
    """Build a PipelineSpec representing the minimal stages needed for an operation.

    This function:
    - Looks up the operation from the canonical catalog.
    - Expands required datasets via dataset contracts and upstream dependencies.
    - Partitions tables by owner_package into ingestion / graphs / analytics.
    - Uses required_graphs to force graph stage when necessary.
    - Chooses a canonical PipelineSpec based on the inferred stage needs.

    Parameters
    ----------
    op_id
        Operation identifier (e.g., "function.summary", "datasets.list").
    _snapshot
        Repository snapshot reference (reserved for future incremental hints).
    include_analytics
        If True, include analytics stage when any datasets/graphs are needed.
        Defaults to True.

    Returns
    -------
    PipelineSpec
        Canonical pipeline spec to execute for prerequisites.
    """
    _op, required_tables, required_graphs = _get_required_from_operation(op_id)

    expanded_tables = _expand_dataset_dependencies(required_tables)

    core_tables, graph_tables, analytics_tables = _partition_by_owner_package(
        expanded_tables,
    )

    need_ingestion, need_graphs, need_analytics = _compute_stage_flags(
        core_tables=core_tables,
        graph_tables=graph_tables,
        analytics_tables=analytics_tables,
        required_graphs=required_graphs,
        include_analytics=include_analytics,
    )

    spec = _choose_spec(
        need_ingestion=need_ingestion,
        need_graphs=need_graphs,
        need_analytics=need_analytics,
    )

    log.info(
        "op_planner.build op=%s tables=%d graphs=%d spec=%s",
        op_id,
        len(expanded_tables),
        len(required_graphs),
        spec.id,
    )

    return spec


def build_prereq_summary(
    op_id: str,
    _snapshot: SnapshotRef,
) -> OpPrereqSummary:
    """Build a detailed prerequisite summary for introspection.

    Parameters
    ----------
    op_id
        Operation identifier (e.g., "function.summary").
    _snapshot
        Repository snapshot reference (reserved for future hints).

    Returns
    -------
    OpPrereqSummary
        Detailed summary of prerequisites for the operation.
    """
    op, required_tables, required_graphs = _get_required_from_operation(op_id)
    expanded_tables = _expand_dataset_dependencies(required_tables)
    core_tables, graph_tables, analytics_tables = _partition_by_owner_package(
        expanded_tables,
    )

    return OpPrereqSummary(
        op=op,
        required_tables=frozenset(required_tables),
        expanded_tables=frozenset(expanded_tables),
        core_tables=frozenset(core_tables),
        graph_tables=frozenset(graph_tables),
        analytics_tables=frozenset(analytics_tables),
        required_graphs=frozenset(required_graphs),
    )


# -----------------------------------------------------------------------------
# Orchestrator API
# -----------------------------------------------------------------------------


def ensure_prerequisites_for_operation(  # noqa: PLR0913
    *,
    op_id: str,
    snapshot: SnapshotRef,
    paths: BuildPaths,
    gateway: StorageGateway,
    tools: ToolsConfig,
    include_analytics: bool = True,
    trigger: TriggerKind = "api",
) -> PipelineRunRecord:
    """Run whatever work is necessary before serving operation `op_id`.

    This function:
    - Builds an operation-driven PipelineSpec via build_pipeline_for_operation.
    - Executes that spec using the unified pipeline executor.
    - Records run + step metadata to pipeline_runs / pipeline_steps.

    For NOOP_PIPELINE (operations with no prerequisites), this function
    still creates a run record but executes no stages.

    Parameters
    ----------
    op_id
        Operation identifier (e.g., "function.summary", "datasets.list").
    snapshot
        Repository snapshot reference.
    paths
        Build paths for this run.
    gateway
        Storage gateway for DuckDB and metadata tables.
    tools
        Tools configuration (used by ingestion and analytics where relevant).
    include_analytics
        Whether to include analytics stage even if not strictly required
        by contracts. Defaults to True.
    trigger
        How this run was triggered. Defaults to "api".

    Returns
    -------
    PipelineRunRecord
        Run record describing the prereq pipeline execution.
    """
    # Avoid circular import by importing here
    from codeintel.pipeline.executor import run_pipeline  # noqa: PLC0415

    spec = build_pipeline_for_operation(
        op_id,
        snapshot,
        include_analytics=include_analytics,
    )

    log.info(
        "op_planner.ensure_prereqs op=%s spec=%s trigger=%s kind=op_prereqs",
        op_id,
        spec.id,
        trigger,
    )

    return run_pipeline(
        spec=spec,
        snapshot=snapshot,
        paths=paths,
        gateway=gateway,
        tools=tools,
        trigger=trigger,
        run_kind_override="op_prereqs",
    )


__all__ = [
    "OpPrereqSummary",
    "build_pipeline_for_operation",
    "build_prereq_summary",
    "ensure_prerequisites_for_operation",
]
