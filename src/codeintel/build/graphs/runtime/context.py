"""Graph runtime context helpers and normalization.

This module provides core utilities for graph metric computations, including
context specification and normalization. Graph contexts are anchored to
Parquet-backed snapshots (repo/commit) to ensure graph inputs are dataset-derived.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from codeintel.config.primitives import SnapshotRef

DEFAULT_BETWEENNESS_SAMPLE = 500


@dataclass(frozen=True)
class GraphMetricsOptions:
    """Configuration options for graph metrics computation.

    Parameters
    ----------
    max_betweenness_sample
        Maximum sample size for betweenness centrality estimation.
    eigen_max_iter
        Maximum iterations for eigenvector centrality convergence.
    seed
        Random seed for reproducible computations.
    pagerank_weight
        Edge attribute to use as weight for PageRank (None for unweighted).
    betweenness_weight
        Edge attribute to use as weight for betweenness (None for unweighted).
    """

    max_betweenness_sample: int | None = 200
    eigen_max_iter: int = 200
    seed: int = 0
    pagerank_weight: str | None = "weight"
    betweenness_weight: str | None = "weight"


@dataclass(frozen=True)
class GraphContext:
    """Describe runtime parameters for graph computations.

    Graph contexts are anchored to repo/commit partitions so graph inputs
    resolve to Parquet-backed datasets.
    """

    repo: str
    commit: str
    now: datetime | None = None
    betweenness_sample: int = DEFAULT_BETWEENNESS_SAMPLE
    eigen_max_iter: int = 200
    seed: int = 0
    pagerank_weight: str | None = "weight"
    betweenness_weight: str | None = "weight"
    use_gpu: bool = False
    community_detection_limit: int | None = None

    def resolved_now(self) -> datetime:
        """Return a concrete timestamp, defaulting to current UTC time.

        Returns
        -------
        datetime
            Provided timestamp or current UTC time.
        """
        return self.now or datetime.now(tz=UTC)


@dataclass(frozen=True)
class GraphContextSpec:
    """Specification for deriving a normalized GraphContext."""

    repo: str
    commit: str
    use_gpu: bool
    options: GraphMetricsOptions | None = None
    ctx: GraphContext | None = None
    now: datetime | None = None
    betweenness_cap: int | None = None
    eigen_cap: int | None = None
    pagerank_weight: str | None = None
    betweenness_weight: str | None = None
    seed: int | None = None
    community_detection_limit: int | None = None


@dataclass(frozen=True)
class GraphContextCaps:
    """Optional caps for graph context derivation."""

    betweenness_cap: int | None = None
    eigen_cap: int | None = None
    community_detection_limit: int | None = None


def build_graph_context(
    snapshot: SnapshotRef,
    *,
    options: GraphMetricsOptions | None = None,
    now: datetime | None = None,
    caps: GraphContextCaps | None = None,
    use_gpu: bool = False,
) -> GraphContext:
    """Construct a GraphContext from SnapshotRef and GraphMetricsOptions.

    Parameters
    ----------
    snapshot
        Repository snapshot reference (repo, commit, repo_root).
    options
        Graph metrics configuration options.
    now
        Optional timestamp; defaults to UTC now when omitted.
    caps
        Optional container for sampling caps and community detection limit.
    use_gpu
        Whether to prefer GPU-backed NetworkX execution when available.

    Returns
    -------
    GraphContext
        Graph context with caps and seeds applied.
    """
    _validate_snapshot_identity(snapshot.repo, snapshot.commit)
    opts = options or GraphMetricsOptions()
    resolved_caps = caps or GraphContextCaps()
    betweenness_sample = opts.max_betweenness_sample or DEFAULT_BETWEENNESS_SAMPLE
    if resolved_caps.betweenness_cap is not None:
        betweenness_sample = min(betweenness_sample, resolved_caps.betweenness_cap)
    eigen_max_iter = (
        opts.eigen_max_iter
        if resolved_caps.eigen_cap is None
        else min(opts.eigen_max_iter, resolved_caps.eigen_cap)
    )
    return GraphContext(
        repo=snapshot.repo,
        commit=snapshot.commit,
        now=now,
        betweenness_sample=betweenness_sample,
        eigen_max_iter=eigen_max_iter,
        seed=opts.seed,
        pagerank_weight=opts.pagerank_weight,
        betweenness_weight=opts.betweenness_weight,
        use_gpu=use_gpu,
        community_detection_limit=resolved_caps.community_detection_limit,
    )


def resolve_graph_context(spec: GraphContextSpec) -> GraphContext:
    """Normalize a GraphContext to the target repo/commit and backend preferences.

    Parameters
    ----------
    spec
        Context specification describing the repo/commit, backend preference, and
        optional overrides.

    Returns
    -------
    GraphContext
        Context aligned to the provided repo, commit, and backend preferences.
    """
    _validate_snapshot_identity(spec.repo, spec.commit)
    base_now = spec.now or datetime.now(tz=UTC)
    resolved = _base_context(spec, base_now)
    return _normalize_context(spec, resolved, base_now)


def _validate_snapshot_identity(repo: str, commit: str) -> None:
    if not isinstance(repo, str) or not repo.strip():
        msg = "Graph context repo is required for Parquet-backed graph inputs"
        raise ValueError(msg)
    if not isinstance(commit, str) or not commit.strip():
        msg = "Graph context commit is required for Parquet-backed graph inputs"
        raise ValueError(msg)


def _base_context(spec: GraphContextSpec, base_now: datetime) -> GraphContext:
    if spec.ctx is not None:
        return spec.ctx
    if spec.options is not None:
        snapshot = SnapshotRef(repo=spec.repo, commit=spec.commit, repo_root=Path())
        caps = GraphContextCaps(
            betweenness_cap=spec.betweenness_cap,
            eigen_cap=spec.eigen_cap,
            community_detection_limit=spec.community_detection_limit,
        )
        return build_graph_context(
            snapshot,
            options=spec.options,
            now=base_now,
            caps=caps,
            use_gpu=spec.use_gpu,
        )
    return GraphContext(
        repo=spec.repo,
        commit=spec.commit,
        now=base_now,
        betweenness_sample=spec.betweenness_cap or DEFAULT_BETWEENNESS_SAMPLE,
        eigen_max_iter=spec.eigen_cap or DEFAULT_BETWEENNESS_SAMPLE,
        seed=spec.seed or 0,
        pagerank_weight=spec.pagerank_weight or "weight",
        betweenness_weight=spec.betweenness_weight or "weight",
        use_gpu=spec.use_gpu,
        community_detection_limit=spec.community_detection_limit,
    )


def _normalize_context(
    spec: GraphContextSpec,
    ctx: GraphContext,
    base_now: datetime,
) -> GraphContext:
    normalized = ctx
    if ctx.repo != spec.repo or ctx.commit != spec.commit:
        normalized = replace(normalized, repo=spec.repo, commit=spec.commit)
    if normalized.use_gpu != spec.use_gpu:
        normalized = replace(normalized, use_gpu=spec.use_gpu)
    if spec.betweenness_cap is not None and normalized.betweenness_sample > spec.betweenness_cap:
        normalized = replace(normalized, betweenness_sample=spec.betweenness_cap)
    if spec.eigen_cap is not None and normalized.eigen_max_iter > spec.eigen_cap:
        normalized = replace(normalized, eigen_max_iter=spec.eigen_cap)
    if spec.pagerank_weight is not None and normalized.pagerank_weight != spec.pagerank_weight:
        normalized = replace(normalized, pagerank_weight=spec.pagerank_weight)
    if (
        spec.betweenness_weight is not None
        and normalized.betweenness_weight != spec.betweenness_weight
    ):
        normalized = replace(normalized, betweenness_weight=spec.betweenness_weight)
    if spec.seed is not None and normalized.seed != spec.seed:
        normalized = replace(normalized, seed=spec.seed)
    if normalized.now is None:
        normalized = replace(normalized, now=base_now)
    if (
        spec.community_detection_limit is not None
        and normalized.community_detection_limit != spec.community_detection_limit
    ):
        normalized = replace(normalized, community_detection_limit=spec.community_detection_limit)
    return normalized


def load_prior_manifest(path: Path | None) -> dict[str, dict[str, object]] | None:
    """Load the prior manifest and normalize records for unchanged detection.

    Parameters
    ----------
    path
        Path to the manifest JSON file.

    Returns
    -------
    dict[str, dict[str, object]] | None
        Mapping of plugin name to normalized manifest record.
    """
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    records = payload.get("records")
    if not isinstance(records, list):
        return None

    normalized: dict[str, dict[str, object]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        name = record.get("name")
        if not isinstance(name, str):
            continue
        merged: dict[str, object] = dict(record)
        meta = record.get("meta")
        if isinstance(meta, dict):
            merged.update(meta)
        normalized[name] = merged
    return normalized


__all__ = [
    "DEFAULT_BETWEENNESS_SAMPLE",
    "GraphContext",
    "GraphContextCaps",
    "GraphContextSpec",
    "GraphMetricsOptions",
    "build_graph_context",
    "load_prior_manifest",
    "resolve_graph_context",
]
