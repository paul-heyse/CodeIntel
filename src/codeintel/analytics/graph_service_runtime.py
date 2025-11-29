"""Graph runtime context utilities shared across analytics graph metrics."""

from __future__ import annotations

import importlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from codeintel.config import GraphMetricsStepConfig

if TYPE_CHECKING:
    from codeintel.analytics.context import AnalyticsContext
    from codeintel.analytics.graph_runtime import GraphRuntime
    from codeintel.graphs.function_catalog_service import FunctionCatalogProvider
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)

DEFAULT_BETWEENNESS_SAMPLE = 500


@dataclass(frozen=True)
class GraphContext:
    """Execution context for graph computations."""

    repo: str
    commit: str
    now: datetime | None = None
    betweenness_sample: int = 500
    eigen_max_iter: int = 200
    seed: int = 0
    pagerank_weight: str | None = "weight"
    betweenness_weight: str | None = "weight"
    use_gpu: bool = False
    community_detection_limit: int | None = None

    def resolved_now(self) -> datetime:
        """
        Return a concrete timestamp, defaulting to UTC now when unset.

        Returns
        -------
        datetime
            Existing timestamp or the current UTC time.
        """
        return self.now or datetime.now(tz=UTC)


@dataclass(frozen=True)
class GraphContextSpec:
    """Specification for normalizing graph contexts."""

    repo: str
    commit: str
    use_gpu: bool
    metrics_cfg: GraphMetricsStepConfig | None = None
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


@dataclass
class GraphServiceRuntime:
    """Lightweight orchestrator for graph analytics using a shared runtime."""

    gateway: StorageGateway
    runtime: GraphRuntime
    analytics_context: AnalyticsContext | None = None
    catalog_provider: FunctionCatalogProvider | None = None

    def compute_graph_metrics(
        self, cfg: GraphMetricsStepConfig, *, filters: object | None = None
    ) -> None:
        """Compute core function/module graph metrics."""
        module = importlib.import_module("codeintel.analytics.graphs.graph_metrics")
        log.info(
            "graph_runtime.compute_graph_metrics repo=%s commit=%s filters=%s",
            cfg.repo,
            cfg.commit,
            "provided" if filters is not None else "default",
        )
        self._observe(
            "graph_metrics",
            lambda: module.compute_graph_metrics(
                self.gateway,
                cfg,
                deps=module.GraphMetricsDeps(
                    catalog_provider=self.catalog_provider,
                    runtime=self.runtime,
                    analytics_context=self.analytics_context,
                    filters=filters,
                ),
            ),
        )

    def compute_graph_metrics_ext(self, *, repo: str, commit: str) -> None:
        """Compute extended function and module graph metrics."""
        funcs_module = importlib.import_module("codeintel.analytics.graphs.graph_metrics_ext")
        modules_module = importlib.import_module(
            "codeintel.analytics.graphs.module_graph_metrics_ext"
        )
        self._observe(
            "graph_metrics_ext_functions",
            lambda: funcs_module.compute_graph_metrics_functions_ext(
                self.gateway,
                repo=repo,
                commit=commit,
                runtime=self.runtime,
            ),
        )
        self._observe(
            "graph_metrics_ext_modules",
            lambda: modules_module.compute_graph_metrics_modules_ext(
                self.gateway,
                repo=repo,
                commit=commit,
                runtime=self.runtime,
            ),
        )

    def compute_symbol_metrics(self, *, repo: str, commit: str) -> None:
        """Compute symbol graph metrics for modules and functions."""
        module = importlib.import_module("codeintel.analytics.graphs.symbol_graph_metrics")
        self._observe(
            "symbol_graph_metrics_modules",
            lambda: module.compute_symbol_graph_metrics_modules(
                self.gateway,
                repo=repo,
                commit=commit,
                runtime=self.runtime,
            ),
        )
        self._observe(
            "symbol_graph_metrics_functions",
            lambda: module.compute_symbol_graph_metrics_functions(
                self.gateway,
                repo=repo,
                commit=commit,
                runtime=self.runtime,
            ),
        )

    def compute_subsystem_metrics(self, *, repo: str, commit: str) -> None:
        """Compute subsystem-level graph metrics."""
        module = importlib.import_module("codeintel.analytics.graphs.subsystem_graph_metrics")
        self._observe(
            "subsystem_graph_metrics",
            lambda: module.compute_subsystem_graph_metrics(
                self.gateway,
                repo=repo,
                commit=commit,
                runtime=self.runtime,
            ),
        )

    def compute_graph_stats(self, *, repo: str, commit: str) -> None:
        """Compute global graph statistics."""
        module = importlib.import_module("codeintel.analytics.graphs.graph_stats")
        self._observe(
            "graph_stats",
            lambda: module.compute_graph_stats(
                self.gateway,
                repo=repo,
                commit=commit,
                runtime=self.runtime,
            ),
        )

    def _observe(self, name: str, func: Callable[[], None]) -> None:
        start = time.perf_counter()
        func()
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        log.info(
            "graph_runtime.%s completed in %.2f ms",
            name,
            duration_ms,
            extra={
                "metric": "graph_runtime",
                "op": name,
                "duration_ms": duration_ms,
                "use_gpu": self.runtime.use_gpu,
                "features": self.runtime.options.features,
            },
        )


def build_graph_context(
    cfg: GraphMetricsStepConfig,
    *,
    now: datetime | None = None,
    caps: GraphContextCaps | None = None,
    use_gpu: bool = False,
) -> GraphContext:
    """
    Construct a GraphContext from GraphMetricsStepConfig with optional caps.

    Parameters
    ----------
    cfg :
        Graph metrics configuration values.
    now :
        Optional timestamp; defaults to UTC now when omitted.
    caps :
        Optional container for sampling caps and community detection limit.
    use_gpu :
        Whether to prefer GPU-backed NetworkX execution when available.

    Returns
    -------
    GraphContext
        Graph context with caps and seeds applied.
    """
    resolved_caps = caps or GraphContextCaps()
    betweenness_sample = cfg.max_betweenness_sample or DEFAULT_BETWEENNESS_SAMPLE
    if resolved_caps.betweenness_cap is not None:
        betweenness_sample = min(betweenness_sample, resolved_caps.betweenness_cap)
    eigen_max_iter = (
        cfg.eigen_max_iter
        if resolved_caps.eigen_cap is None
        else min(cfg.eigen_max_iter, resolved_caps.eigen_cap)
    )
    return GraphContext(
        repo=cfg.repo,
        commit=cfg.commit,
        now=now,
        betweenness_sample=betweenness_sample,
        eigen_max_iter=eigen_max_iter,
        seed=cfg.seed,
        pagerank_weight=cfg.pagerank_weight,
        betweenness_weight=cfg.betweenness_weight,
        use_gpu=use_gpu,
        community_detection_limit=resolved_caps.community_detection_limit,
    )


def resolve_graph_context(
    spec: GraphContextSpec,
) -> GraphContext:
    """
    Normalize a GraphContext to the target repo/commit and backend preferences.

    Parameters
    ----------
    spec :
        Context specification describing the repo/commit, backend preference, and
        optional overrides.

    Returns
    -------
    GraphContext
        Context aligned to the provided repo, commit, and backend preferences.
    """
    base_now = spec.now or datetime.now(tz=UTC)
    resolved = _base_context(spec, base_now)
    return _normalize_context(spec, resolved, base_now)


def _base_context(spec: GraphContextSpec, base_now: datetime) -> GraphContext:
    if spec.ctx is not None:
        return spec.ctx
    if spec.metrics_cfg is not None:
        caps = GraphContextCaps(
            betweenness_cap=spec.betweenness_cap,
            eigen_cap=spec.eigen_cap,
            community_detection_limit=spec.community_detection_limit,
        )
        return build_graph_context(
            spec.metrics_cfg,
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


__all__ = [
    "DEFAULT_BETWEENNESS_SAMPLE",
    "GraphContext",
    "GraphContextCaps",
    "GraphContextSpec",
    "GraphServiceRuntime",
    "build_graph_context",
    "resolve_graph_context",
]
