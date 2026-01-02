"""Configuration dataclasses for test provisioning environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.build.graphs.runtime import GraphMetricsOptions
    from codeintel.config.models import ToolsConfig
    from codeintel.config.primitives import BuildPaths
    from codeintel.ingestion.adapters import (
        DuckDBStorageAdapter,
        FilesystemDiscoveryAdapter,
        HashChangeDetectionAdapter,
    )
    from codeintel.ingestion.adapters.tool_runner import ToolRunnerAdapter
    from codeintel.ingestion.engine.infrastructure import ToolRunner
    from codeintel.ingestion.engine.service import ToolService
    from codeintel.storage.gateway import StorageGateway


from tests._helpers.env_options import GatewayOptions
from tests._helpers.fixtures.snapshots import DEFAULT_VARIANT, SnapshotVariant


@dataclass(frozen=True)
class ProvisionedGateway:
    """Container for an ingested gateway and associated filesystem context."""

    repo: str
    commit: str
    repo_root: Path
    build_dir: Path
    db_path: Path
    document_output_dir: Path
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
    build_graph_metrics: bool = False
    file_backed: bool = True
    db_path: Path | None = None
    include_seed_goid: bool = True


@dataclass
class ProvisioningGatewayOverrides:
    """Overrides for gateway provisioning defaults."""

    db_path: Path | None = None
    apply_schema: bool = True
    ensure_views: bool = True
    validate_schema: bool = True
    file_backed: bool = True
    strict_schema: bool = True


def provisioning_gateway_options(
    overrides: ProvisioningGatewayOverrides | None = None,
    **kwargs: object,
) -> GatewayOptions:
    """Create GatewayOptions with provisioning defaults (file_backed=True).

    This factory provides the default options used by provisioning functions,
    which typically use file-backed databases for persistence across steps.

    Parameters
    ----------
    overrides
        Optional overrides bundle for gateway settings.
    **kwargs
        Backwards-compatible override values (e.g., apply_schema=False).

    Returns
    -------
    GatewayOptions
        Configured options for gateway creation.

    Raises
    ------
    ValueError
        If an unknown override key is provided.
    """
    settings = overrides or ProvisioningGatewayOverrides()
    for key, value in kwargs.items():
        if not hasattr(settings, key):
            message = f"Unknown provisioning override: {key}"
            raise ValueError(message)
        setattr(settings, key, value)

    return GatewayOptions(
        db_path=settings.db_path,
        apply_schema=settings.apply_schema,
        ensure_views=settings.ensure_views,
        validate_schema=settings.validate_schema,
        file_backed=settings.file_backed,
        strict_schema=settings.strict_schema,
    )


@dataclass
class ProvisioningSetup:
    """Container for provisioning setup components.

    This dataclass consolidates all the components needed during repo
    provisioning, reducing the number of local variables in provisioning
    functions while improving clarity and reusability.
    """

    ctx: RepoContext
    build_paths: BuildPaths
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

    snapshot_variant: SnapshotVariant = DEFAULT_VARIANT
    provision_options: ProvisionOptions | None = None
    gateway_options: GatewayOptions | None = None
    run_ingestion: bool = True

    @property
    def repo(self) -> str:
        return self.snapshot_variant.repo

    @property
    def commit(self) -> str:
        return self.snapshot_variant.commit


@dataclass(frozen=True)
class GraphMetricsGatewayOptions:
    """Options for provisioning graph-metrics-ready gateways."""

    snapshot_variant: SnapshotVariant = DEFAULT_VARIANT
    metrics_options: GraphMetricsOptions | None = None
    include_symbol_edges: bool = True
    file_backed: bool = False
    db_path: Path | None = None
    run_metrics: bool = True
    build_callgraph_enabled: bool = True

    @property
    def repo(self) -> str:
        return self.snapshot_variant.repo

    @property
    def commit(self) -> str:
        return self.snapshot_variant.commit


@dataclass(frozen=True)
class CallgraphFixtureOptions:
    """Options for provisioning callgraph fixture repos."""

    snapshot_variant: SnapshotVariant = DEFAULT_VARIANT
    file_backed: bool = False
    db_path: Path | None = None
    goid_entries: list[tuple[int, str, str, int, int, str]] | None = None

    @property
    def repo(self) -> str:
        return self.snapshot_variant.repo

    @property
    def commit(self) -> str:
        return self.snapshot_variant.commit


__all__ = [
    "DEFAULT_VARIANT",
    "CallgraphFixtureOptions",
    "GatewayOptions",
    "GraphMetricsGatewayOptions",
    "ProvisionOptions",
    "ProvisionedGateway",
    "ProvisioningConfig",
    "ProvisioningGatewayOverrides",
    "ProvisioningSetup",
    "RepoContext",
    "SnapshotVariant",
    "provisioning_gateway_options",
]
