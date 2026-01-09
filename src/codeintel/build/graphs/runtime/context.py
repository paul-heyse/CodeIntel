"""Graph runtime context helpers and normalization.

This module provides core utilities for graph metric computations, including
context specification and normalization. Graph contexts are anchored to
Parquet-backed snapshots (repo/commit) to ensure graph inputs are dataset-derived.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.build.graphs.rx.weights import WeightSemantics
from codeintel.config.primitives import SnapshotRef
from codeintel.core.columnar.execution_context import (
    resolve_runtime_profile,
    runtime_profile_from_settings,
)
from codeintel.core.runtime.loader import load_runtime_settings

if TYPE_CHECKING:
    from codeintel.core.columnar.dedupe_ops import DedupeTier
    from codeintel.core.columnar.profiles import RuntimeProfile

DEFAULT_BETWEENNESS_SAMPLE = 500
DEFAULT_PARALLEL_THRESHOLD = 50


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
    parallel_threshold
        Node-count threshold for rustworkx parallel algorithms.
    rayon_threads
        Optional thread count override for rustworkx rayon execution.
    weight_semantics
        Whether weights represent strength or cost semantics.
    """

    max_betweenness_sample: int | None = 200
    eigen_max_iter: int = 200
    seed: int = 0
    pagerank_weight: str | None = "weight"
    betweenness_weight: str | None = "weight"
    parallel_threshold: int | None = DEFAULT_PARALLEL_THRESHOLD
    rayon_threads: int | None = None
    weight_semantics: WeightSemantics = WeightSemantics.STRENGTH


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
    parallel_threshold: int = DEFAULT_PARALLEL_THRESHOLD
    rayon_threads: int | None = None
    weight_semantics: WeightSemantics = WeightSemantics.STRENGTH
    runtime_profile: str | None = None
    scan_profile: str | None = None
    determinism_tier: DedupeTier | None = None

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
    parallel_threshold: int | None = None
    rayon_threads: int | None = None
    weight_semantics: WeightSemantics | None = None
    runtime_profile: str | None = None


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
        Whether to prefer GPU execution (ignored; rustworkx is CPU-only).

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
    parallel_threshold = (
        opts.parallel_threshold
        if opts.parallel_threshold is not None
        else DEFAULT_PARALLEL_THRESHOLD
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
        parallel_threshold=parallel_threshold,
        rayon_threads=opts.rayon_threads,
        weight_semantics=opts.weight_semantics,
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
    parallel_threshold = (
        spec.parallel_threshold
        if spec.parallel_threshold is not None
        else DEFAULT_PARALLEL_THRESHOLD
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
        parallel_threshold=parallel_threshold,
        rayon_threads=spec.rayon_threads,
        weight_semantics=spec.weight_semantics or WeightSemantics.STRENGTH,
    )


def _normalize_context(
    spec: GraphContextSpec,
    ctx: GraphContext,
    base_now: datetime,
) -> GraphContext:
    normalized = _apply_repo_commit(spec, ctx)
    normalized = _apply_use_gpu(spec, normalized)
    normalized = _apply_caps(spec, normalized)
    normalized = _apply_weights(spec, normalized)
    normalized = _apply_seed(spec, normalized)
    normalized = _apply_runtime_profile(spec, normalized)
    normalized = _apply_parallel_threshold(spec, normalized)
    normalized = _apply_rayon_threads(spec, normalized)
    normalized = _apply_weight_semantics(spec, normalized)
    normalized = _apply_now(normalized, base_now)
    return _apply_community_limit(spec, normalized)


def _apply_repo_commit(spec: GraphContextSpec, ctx: GraphContext) -> GraphContext:
    if ctx.repo == spec.repo and ctx.commit == spec.commit:
        return ctx
    return replace(ctx, repo=spec.repo, commit=spec.commit)


def _apply_use_gpu(spec: GraphContextSpec, ctx: GraphContext) -> GraphContext:
    if ctx.use_gpu == spec.use_gpu:
        return ctx
    return replace(ctx, use_gpu=spec.use_gpu)


def _apply_caps(spec: GraphContextSpec, ctx: GraphContext) -> GraphContext:
    updated = ctx
    if spec.betweenness_cap is not None and updated.betweenness_sample > spec.betweenness_cap:
        updated = replace(updated, betweenness_sample=spec.betweenness_cap)
    if spec.eigen_cap is not None and updated.eigen_max_iter > spec.eigen_cap:
        updated = replace(updated, eigen_max_iter=spec.eigen_cap)
    return updated


def _apply_weights(spec: GraphContextSpec, ctx: GraphContext) -> GraphContext:
    updated = ctx
    if spec.pagerank_weight is not None and updated.pagerank_weight != spec.pagerank_weight:
        updated = replace(updated, pagerank_weight=spec.pagerank_weight)
    if (
        spec.betweenness_weight is not None
        and updated.betweenness_weight != spec.betweenness_weight
    ):
        updated = replace(updated, betweenness_weight=spec.betweenness_weight)
    return updated


def _apply_seed(spec: GraphContextSpec, ctx: GraphContext) -> GraphContext:
    if spec.seed is None or ctx.seed == spec.seed:
        return ctx
    return replace(ctx, seed=spec.seed)


def _apply_parallel_threshold(spec: GraphContextSpec, ctx: GraphContext) -> GraphContext:
    if spec.parallel_threshold is None or ctx.parallel_threshold == spec.parallel_threshold:
        return ctx
    return replace(ctx, parallel_threshold=spec.parallel_threshold)


def _apply_rayon_threads(spec: GraphContextSpec, ctx: GraphContext) -> GraphContext:
    if spec.rayon_threads is None or ctx.rayon_threads == spec.rayon_threads:
        return ctx
    return replace(ctx, rayon_threads=spec.rayon_threads)


def _apply_weight_semantics(spec: GraphContextSpec, ctx: GraphContext) -> GraphContext:
    if spec.weight_semantics is None or ctx.weight_semantics == spec.weight_semantics:
        return ctx
    return replace(ctx, weight_semantics=spec.weight_semantics)


def _apply_runtime_profile(spec: GraphContextSpec, ctx: GraphContext) -> GraphContext:
    profile = _normalize_runtime_profile(
        _resolve_runtime_profile(spec.runtime_profile),
        default_name="graph_metrics",
    )
    if profile is None:
        return ctx
    parallel_threshold = _resolve_parallel_threshold(profile, ctx.parallel_threshold)
    rayon_threads = _resolve_rayon_threads(profile, ctx.rayon_threads)
    determinism_tier = profile.determinism or ctx.determinism_tier
    runtime_name = profile.name
    scan_profile = profile.scan_profile
    if (
        parallel_threshold == ctx.parallel_threshold
        and rayon_threads == ctx.rayon_threads
        and runtime_name == ctx.runtime_profile
        and scan_profile == ctx.scan_profile
        and determinism_tier == ctx.determinism_tier
    ):
        return ctx
    return replace(
        ctx,
        parallel_threshold=parallel_threshold,
        rayon_threads=rayon_threads,
        runtime_profile=runtime_name,
        scan_profile=scan_profile,
        determinism_tier=determinism_tier,
    )


def _resolve_runtime_profile(profile_name: str | None) -> RuntimeProfile | None:
    if profile_name is not None:
        return resolve_runtime_profile(profile_name)
    settings = load_runtime_settings()
    return runtime_profile_from_settings(settings.columnar)


def _normalize_runtime_profile(
    profile: RuntimeProfile | None,
    *,
    default_name: str,
) -> RuntimeProfile | None:
    if profile is None:
        return None
    scan_settings = load_runtime_settings().build.arrow_scan
    scan_profile = profile.scan_profile or scan_settings.profile
    use_threads = (
        profile.use_threads if profile.use_threads is not None else scan_settings.use_threads
    )
    implicit_ordering = (
        True if profile.implicit_ordering is None else profile.implicit_ordering
    )
    require_sequenced_output = (
        True
        if profile.require_sequenced_output is None
        else profile.require_sequenced_output
    )
    return replace(
        profile,
        name=profile.name or default_name,
        scan_profile=scan_profile,
        implicit_ordering=implicit_ordering,
        require_sequenced_output=require_sequenced_output,
        use_threads=use_threads,
    )


def _resolve_parallel_threshold(
    profile: RuntimeProfile,
    current: int | None,
) -> int:
    use_threads = profile.use_threads
    if use_threads is False:
        return sys.maxsize
    if current is None:
        return DEFAULT_PARALLEL_THRESHOLD
    return current


def _resolve_rayon_threads(
    profile: RuntimeProfile,
    current: int | None,
) -> int | None:
    resolved = profile.resolve_cpu_threads(default=current)
    if profile.use_threads is False and (resolved is None or resolved > 1):
        return 1
    return resolved


def _apply_now(ctx: GraphContext, base_now: datetime) -> GraphContext:
    if ctx.now is not None:
        return ctx
    return replace(ctx, now=base_now)


def _apply_community_limit(spec: GraphContextSpec, ctx: GraphContext) -> GraphContext:
    if (
        spec.community_detection_limit is None
        or ctx.community_detection_limit == spec.community_detection_limit
    ):
        return ctx
    return replace(ctx, community_detection_limit=spec.community_detection_limit)


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
