"""Asset impact analysis for build planning.

Analyze downstream impact of asset changes by traversing the lineage graph.
Used to identify which targets need to re-run when an upstream asset changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.storage.gateway.protocol import StorageGateway
    from codeintel.storage.tracking.asset_tracking import AssetTracking


@dataclass(frozen=True)
class ImpactedAsset:
    """An asset impacted by an upstream change.

    Attributes
    ----------
    asset_kind
        Kind of asset (table, artifact).
    asset_key
        Unique asset identifier.
    version_hash
        Specific version hash if known, None for logical-level impact.
    target
        Target that produces this asset, if known.
    depth
        Distance from the source asset in the lineage graph.
    """

    asset_kind: str
    asset_key: str
    version_hash: str | None = None
    target: str | None = None
    depth: int = 0


@dataclass(frozen=True)
class ImpactResult:
    """Result of impact analysis.

    Attributes
    ----------
    source_kind
        Kind of the source asset.
    source_key
        Key of the source asset.
    source_version
        Version hash of the source asset if specified.
    impacted_assets
        List of assets impacted by changes to the source.
    impacted_targets
        Set of target names that would need to re-run.
    """

    source_kind: str
    source_key: str
    source_version: str | None
    impacted_assets: list[ImpactedAsset] = field(default_factory=list)
    impacted_targets: set[str] = field(default_factory=set)


def compute_impact(
    gateway: StorageGateway,
    *,
    asset_kind: str,
    asset_key: str,
    version_hash: str | None = None,
    max_depth: int = 10,
) -> ImpactResult:
    """Compute downstream impact of changes to an asset.

    Performs a BFS traversal of the asset lineage graph starting from
    the specified asset, collecting all downstream dependencies.

    Parameters
    ----------
    gateway
        Storage gateway for database access.
    asset_kind
        Kind of source asset (table, artifact).
    asset_key
        Key of source asset.
    version_hash
        Specific version to analyze, or None for all versions.
    max_depth
        Maximum traversal depth to prevent runaway expansion.

    Returns
    -------
    ImpactResult
        Analysis result with impacted assets and targets.
    """
    tracking = gateway.assets

    # BFS state
    visited: set[tuple[str, str]] = set()
    impacted: list[ImpactedAsset] = []
    targets: set[str] = set()

    queue: list[tuple[str, str, str | None, int]] = [(asset_kind, asset_key, version_hash, 0)]

    while queue:
        kind, key, version, depth = queue.pop(0)

        if depth >= max_depth:
            continue

        # Get downstream edges from lineage table
        downstream = _get_downstream_assets(tracking, kind, key, version)

        for edge in downstream:
            asset_tuple = (edge.downstream_kind, edge.downstream_key)
            if asset_tuple in visited:
                continue

            visited.add(asset_tuple)

            # Look up target for this asset
            target = tracking.get_asset_target(edge.downstream_kind, edge.downstream_key)

            impacted_asset = ImpactedAsset(
                asset_kind=edge.downstream_kind,
                asset_key=edge.downstream_key,
                version_hash=edge.downstream_version,
                target=target,
                depth=depth + 1,
            )
            impacted.append(impacted_asset)

            if target:
                targets.add(target)

            # Enqueue for further traversal
            queue.append(
                (edge.downstream_kind, edge.downstream_key, edge.downstream_version, depth + 1)
            )

    return ImpactResult(
        source_kind=asset_kind,
        source_key=asset_key,
        source_version=version_hash,
        impacted_assets=impacted,
        impacted_targets=targets,
    )


@dataclass(frozen=True)
class _DownstreamEdge:
    """Internal edge record from lineage query."""

    downstream_kind: str
    downstream_key: str
    downstream_version: str


def _get_downstream_assets(
    tracking: AssetTracking,
    upstream_kind: str,
    upstream_key: str,
    upstream_version: str | None,
) -> list[_DownstreamEdge]:
    """Query lineage table for downstream dependencies.

    Returns
    -------
    list[_DownstreamEdge]
        Edges where the specified asset is upstream.
    """
    edges = tracking.get_downstream_edges(
        upstream_kind=upstream_kind,
        upstream_key=upstream_key,
        upstream_version=upstream_version,
    )
    return [
        _DownstreamEdge(
            downstream_kind=edge.downstream_kind,
            downstream_key=edge.downstream_key,
            downstream_version=edge.downstream_version,
        )
        for edge in edges
    ]


__all__ = [
    "ImpactResult",
    "ImpactedAsset",
    "compute_impact",
]
