"""Result type definitions for CLI handlers.

Each handler that returns structured data should have a corresponding
result type defined here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class OperationListResult:
    """Result from op list command.

    Parameters
    ----------
    operations
        List of operation metadata dictionaries.
    count
        Total number of operations.
    """

    operations: list[dict[str, str | None]]
    count: int

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {"operations": self.operations, "count": self.count}


@dataclass(frozen=True)
class OperationCallResult:
    """Result from op call command.

    Parameters
    ----------
    operation_id
        The operation that was invoked.
    result
        The operation result data.
    """

    operation_id: str
    result: dict[str, Any]

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {"operation_id": self.operation_id, "result": self.result}


@dataclass(frozen=True)
class DatasetListResult:
    """Result from dataset list command.

    Parameters
    ----------
    datasets
        List of dataset metadata dictionaries.
    count
        Total number of datasets.
    """

    datasets: list[dict[str, str | None]]
    count: int

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {"datasets": self.datasets, "count": self.count}


@dataclass(frozen=True)
class DatasetDescribeResult:
    """Result from dataset describe command.

    Parameters
    ----------
    table_key
        The dataset table key.
    columns
        List of column metadata dictionaries.
    row_count
        Number of rows (if available).
    name
        Dataset name.
    description
        Dataset description.
    owner_package
        Owner package name.
    upstream_dependencies
        List of upstream dependencies.
    """

    table_key: str
    columns: list[dict[str, str | bool]]
    row_count: int | None
    name: str
    description: str | None = None
    owner_package: str | None = None
    upstream_dependencies: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        result: dict[str, object] = {
            "table_key": self.table_key,
            "name": self.name,
            "columns": self.columns,
        }
        if self.description:
            result["description"] = self.description
        if self.owner_package:
            result["owner_package"] = self.owner_package
        if self.row_count is not None:
            result["row_count"] = self.row_count
        if self.upstream_dependencies:
            result["upstream_dependencies"] = self.upstream_dependencies
        return result


@dataclass(frozen=True)
class DatasetVerifyResult:
    """Result from dataset verify command.

    Parameters
    ----------
    verified
        Whether verification passed.
    issues
        List of issues found (empty if verified).
    """

    verified: bool
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {"verified": self.verified, "issues": self.issues}


@dataclass(frozen=True)
class BuildStatusResult:
    """Result from build status command.

    Parameters
    ----------
    targets
        List of target status dictionaries.
    stale_count
        Number of stale targets.
    fresh_count
        Number of fresh targets.
    """

    targets: list[dict[str, Any]]
    stale_count: int
    fresh_count: int

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "targets": self.targets,
            "stale_count": self.stale_count,
            "fresh_count": self.fresh_count,
        }


@dataclass(frozen=True)
class BuildRunResult:
    """Result from build run command.

    Parameters
    ----------
    executed
        List of executed target names.
    skipped
        List of skipped target names.
    failed
        List of failed target names.
    duration_seconds
        Total duration in seconds.
    """

    executed: list[str]
    skipped: list[str]
    failed: list[str]
    duration_seconds: float

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "executed": self.executed,
            "skipped": self.skipped,
            "failed": self.failed,
            "duration_seconds": self.duration_seconds,
        }


@dataclass(frozen=True)
class SubsystemListResult:
    """Result from subsystem list command.

    Parameters
    ----------
    subsystems
        List of subsystem metadata dictionaries.
    count
        Total number of subsystems.
    """

    subsystems: list[dict[str, Any]]
    count: int

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {"subsystems": self.subsystems, "count": self.count}


@dataclass(frozen=True)
class ConfigShowResult:
    """Result from config show command.

    Parameters
    ----------
    config
        Configuration dictionary.
    sources
        Map of config keys to their sources.
    """

    config: dict[str, Any]
    sources: dict[str, list[str]]

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {"config": self.config, "sources": self.sources}


@dataclass(frozen=True)
class GraphPluginsResult:
    """Result from graph plugins command.

    Parameters
    ----------
    plugins
        List of plugin metadata dictionaries.
    count
        Total number of plugins.
    """

    plugins: list[dict[str, Any]]
    count: int

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {"plugins": self.plugins, "count": self.count}


@dataclass(frozen=True)
class DryRunStep:
    """A single step in a dry-run plan.

    Parameters
    ----------
    operation_id
        Operation identifier.
    description
        Human-readable description.
    params
        Parameters for this step.
    is_prereq
        Whether this step is a prerequisite.
    """

    operation_id: str
    description: str
    params: dict[str, Any]
    is_prereq: bool = False

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "operation_id": self.operation_id,
            "description": self.description,
            "params": self.params,
            "is_prereq": self.is_prereq,
        }


@dataclass(frozen=True)
class DryRunResult:
    """Result from dry-run execution.

    Parameters
    ----------
    target_operation
        The target operation that would be executed.
    steps
        List of steps in the execution plan.
    estimated_duration
        Human-readable duration estimate (if available).
    warnings
        List of warnings about the plan.
    """

    target_operation: str
    steps: list[DryRunStep]
    estimated_duration: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        result: dict[str, object] = {
            "target_operation": self.target_operation,
            "steps": [step.to_dict() for step in self.steps],
        }
        if self.estimated_duration:
            result["estimated_duration"] = self.estimated_duration
        if self.warnings:
            result["warnings"] = self.warnings
        return result


@dataclass(frozen=True)
class BuildTargetInfo:
    """Information about a build target.

    Parameters
    ----------
    name
        Target name.
    status
        Current status (fresh, stale, error).
    last_run
        Timestamp of last successful run.
    dependencies
        List of dependency target names.
    outputs
        List of output paths.
    """

    name: str
    status: str
    last_run: str | None
    dependencies: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "name": self.name,
            "status": self.status,
            "last_run": self.last_run,
            "dependencies": self.dependencies,
            "outputs": self.outputs,
        }


@dataclass(frozen=True)
class BuildExecutionResult:
    """Result from executing a build.

    Parameters
    ----------
    targets_executed
        List of targets that were executed.
    targets_skipped
        List of targets that were skipped (already fresh).
    targets_failed
        List of targets that failed.
    total_duration_seconds
        Total execution time.
    success
        Whether all targets succeeded.
    """

    targets_executed: list[str]
    targets_skipped: list[str]
    targets_failed: list[str]
    total_duration_seconds: float
    success: bool = True

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "targets_executed": self.targets_executed,
            "targets_skipped": self.targets_skipped,
            "targets_failed": self.targets_failed,
            "total_duration_seconds": self.total_duration_seconds,
            "success": self.success,
        }


@dataclass(frozen=True)
class BuildHistoryResult:
    """Result from build history command.

    Parameters
    ----------
    runs
        List of build run records.
    count
        Total number of runs returned.
    """

    runs: list[dict[str, Any]]
    count: int

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {"runs": self.runs, "count": self.count}


@dataclass(frozen=True)
class GraphStatsResult:
    """Statistics about a graph.

    Parameters
    ----------
    node_count
        Number of nodes.
    edge_count
        Number of edges.
    density
        Graph density (0.0 to 1.0).
    components
        Number of connected components.
    avg_degree
        Average node degree.
    """

    node_count: int
    edge_count: int
    density: float
    components: int
    avg_degree: float

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "density": self.density,
            "components": self.components,
            "avg_degree": self.avg_degree,
        }


@dataclass(frozen=True)
class GraphQueryResult:
    """Result from a graph query.

    Parameters
    ----------
    nodes
        Matching nodes with their attributes.
    edges
        Edges between matching nodes.
    query
        The query that was executed.
    """

    nodes: list[dict[str, object]]
    edges: list[dict[str, object]]
    query: str

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "query": self.query,
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
        }


@dataclass(frozen=True)
class GraphPluginInfo:
    """Information about a graph plugin.

    Parameters
    ----------
    name
        Plugin name.
    stage
        Plugin stage.
    output_tables
        Output tables produced by the plugin.
    enabled
        Whether the plugin is enabled.
    """

    name: str
    stage: str
    output_tables: tuple[str, ...]
    enabled: bool = True

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "name": self.name,
            "stage": self.stage,
            "output_tables": list(self.output_tables),
            "enabled": self.enabled,
        }


@dataclass(frozen=True)
class GraphPlanResult:
    """Result from graph plugin planning.

    Parameters
    ----------
    plan_id
        Unique identifier for the plan.
    plugins
        Ordered list of plugins to execute.
    skipped
        List of skipped plugins with reasons.
    """

    plan_id: str
    plugins: list[GraphPluginInfo]
    skipped: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "plan_id": self.plan_id,
            "plugins": [p.to_dict() for p in self.plugins],
            "skipped": self.skipped,
            "plugin_count": len(self.plugins),
        }


__all__ = [
    "BuildExecutionResult",
    "BuildHistoryResult",
    "BuildRunResult",
    "BuildStatusResult",
    "BuildTargetInfo",
    "ConfigShowResult",
    "DatasetDescribeResult",
    "DatasetListResult",
    "DatasetVerifyResult",
    "DryRunResult",
    "DryRunStep",
    "GraphPlanResult",
    "GraphPluginInfo",
    "GraphPluginsResult",
    "GraphQueryResult",
    "GraphStatsResult",
    "OperationCallResult",
    "OperationListResult",
    "SubsystemListResult",
]
