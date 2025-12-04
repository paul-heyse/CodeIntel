"""Configuration dataclasses for test provisioning environments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from codeintel.config import GraphMetricsStepConfig
    from codeintel.config.models import ToolsConfig
    from codeintel.config.primitives import BuildPaths
    from codeintel.ingestion import (
        DuckDBStorageAdapter,
        FilesystemDiscoveryAdapter,
        HashChangeDetectionAdapter,
        ToolRunnerAdapter,
    )
    from codeintel.ingestion.infrastructure_utilities.tool_runner import ToolRunner
    from codeintel.ingestion.tool_service import ToolService
    from codeintel.storage.gateway import StorageGateway

# Default constants for provisioning
DEFAULT_REPO = "demo/repo"
DEFAULT_COMMIT = "deadbeef"


@dataclass(frozen=True)
class ProvisionedGateway:
    """Container for an ingested gateway and associated filesystem context."""

    repo: str
    commit: str
    repo_root: Path
    build_dir: Path
    db_path: Path
    document_output_dir: Path
    coverage_file: Path
    gateway: StorageGateway
    runner: ToolRunner

    def close(self) -> None:
        """Close the underlying gateway connection."""
        self.gateway.close()

    def __enter__(self) -> Self:
        """Enter context manager scope.

        Returns
        -------
        ProvisionedGateway
            Self reference for use within a context block.
        """
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        """Close gateway on context exit."""
        self.close()


@dataclass(frozen=True)
class RepoContext:
    """Paths and identifiers for a test repository."""

    repo: str
    commit: str
    repo_root: Path
    build_dir: Path
    db_path: Path
    document_output_dir: Path


@dataclass(frozen=True)
class ProvisionOptions:
    """Options controlling ingestion seeds for provisioned gateways."""

    include_typing: bool = True
    include_coverage: bool = True
    build_graph_metrics: bool = False
    file_backed: bool = False
    db_path: Path | None = None
    include_seed_goid: bool = True


@dataclass(frozen=True)
class GatewayOptions:
    """Options controlling gateway setup without ingestion."""

    db_path: Path | None = None
    apply_schema: bool = True
    ensure_views: bool = True
    validate_schema: bool = True
    file_backed: bool = True
    strict_schema: bool = True


@dataclass
class ProvisioningSetup:
    """Container for provisioning setup components.

    This dataclass consolidates all the components needed during repo
    provisioning, reducing the number of local variables in provisioning
    functions while improving clarity and reusability.
    """

    ctx: RepoContext
    build_paths: BuildPaths
    coverage_file: Path
    tools_cfg: ToolsConfig
    runner: ToolRunner
    tool_service: ToolService
    gateway: StorageGateway
    storage: DuckDBStorageAdapter
    discovery: FilesystemDiscoveryAdapter
    change_detection: HashChangeDetectionAdapter
    tool_adapter: ToolRunnerAdapter


@dataclass(frozen=True)
class ProvisioningConfig:
    """Configuration for context-managed gateway provisioning."""

    repo: str = DEFAULT_REPO
    commit: str = DEFAULT_COMMIT
    provision_options: ProvisionOptions | None = None
    gateway_options: GatewayOptions | None = None
    run_ingestion: bool = True


@dataclass(frozen=True)
class GraphMetricsGatewayOptions:
    """Options for provisioning graph-metrics-ready gateways."""

    repo: str = DEFAULT_REPO
    commit: str = DEFAULT_COMMIT
    graph_cfg: GraphMetricsStepConfig | None = None
    include_symbol_edges: bool = True
    file_backed: bool = False
    db_path: Path | None = None
    run_metrics: bool = True
    build_callgraph_enabled: bool = True


@dataclass(frozen=True)
class CallgraphFixtureOptions:
    """Options for provisioning callgraph fixture repos."""

    repo: str = DEFAULT_REPO
    commit: str = DEFAULT_COMMIT
    file_backed: bool = False
    db_path: Path | None = None
    goid_entries: list[tuple[int, str, str, int, int, str]] | None = None


__all__ = [
    "DEFAULT_COMMIT",
    "DEFAULT_REPO",
    "CallgraphFixtureOptions",
    "GatewayOptions",
    "GraphMetricsGatewayOptions",
    "ProvisionOptions",
    "ProvisionedGateway",
    "ProvisioningConfig",
    "ProvisioningSetup",
    "RepoContext",
]
