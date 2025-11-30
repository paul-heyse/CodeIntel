"""Unit tests for graph runtime context helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from codeintel.analytics.graphs.runtime.context import (
    DEFAULT_BETWEENNESS_SAMPLE,
    GraphContextCaps,
    GraphContextSpec,
    build_graph_context,
    resolve_graph_context,
)
from codeintel.config import GraphMetricsStepConfig
from codeintel.config.primitives import SnapshotRef

BETWEENNESS_CAP = 50
EIGEN_CAP = 200
COMMUNITY_CAP = 3
BETWEENNESS_OVERRIDE = 25
EIGEN_OVERRIDE = 150
SEED_OVERRIDE = 7


def test_build_graph_context_applies_caps() -> None:
    """Caps should clamp config-derived sampling values."""
    snapshot = SnapshotRef(repo="demo/repo", commit="deadbeef", repo_root=Path())
    cfg = GraphMetricsStepConfig(
        snapshot=snapshot,
        max_betweenness_sample=1000,
        eigen_max_iter=400,
        pagerank_weight="pr",
        betweenness_weight="bw",
        seed=42,
    )
    caps = GraphContextCaps(
        betweenness_cap=BETWEENNESS_CAP,
        eigen_cap=EIGEN_CAP,
        community_detection_limit=COMMUNITY_CAP,
    )
    ctx = build_graph_context(cfg, caps=caps, use_gpu=True, now=datetime.now(tz=UTC))
    if ctx.betweenness_sample != BETWEENNESS_CAP:
        pytest.fail("Betweenness sample should respect cap")
    if ctx.eigen_max_iter != EIGEN_CAP:
        pytest.fail("Eigen max iter should respect cap")
    if ctx.use_gpu is not True:
        pytest.fail("use_gpu should propagate")
    if ctx.community_detection_limit != COMMUNITY_CAP:
        pytest.fail("Community detection cap should propagate")


def test_resolve_graph_context_normalizes_spec() -> None:
    """GraphContextSpec should normalize overrides for repo/commit and weights."""
    snapshot = SnapshotRef(repo="demo/repo", commit="deadbeef", repo_root=Path())
    cfg = GraphMetricsStepConfig(snapshot=snapshot, pagerank_weight="w1", betweenness_weight="w2")
    spec = GraphContextSpec(
        repo="demo/repo",
        commit="beadfeed",
        use_gpu=True,
        metrics_cfg=cfg,
        betweenness_cap=BETWEENNESS_OVERRIDE,
        eigen_cap=EIGEN_OVERRIDE,
        pagerank_weight="w3",
        betweenness_weight="w4",
        seed=SEED_OVERRIDE,
    )
    ctx = resolve_graph_context(spec)
    if ctx.repo != "demo/repo" or ctx.commit != "beadfeed":
        pytest.fail("Context should normalize repo/commit")
    if ctx.use_gpu is not True:
        pytest.fail("use_gpu should be normalized")
    if ctx.betweenness_sample != BETWEENNESS_OVERRIDE or ctx.eigen_max_iter != EIGEN_OVERRIDE:
        pytest.fail("Caps should clamp betweenness and eigen values")
    if ctx.pagerank_weight != "w3" or ctx.betweenness_weight != "w4":
        pytest.fail("Weights should be overridden")
    if ctx.seed != SEED_OVERRIDE:
        pytest.fail("Seed override should be applied")
    if ctx.now is None:
        pytest.fail("Context should assign current timestamp when missing")
    if ctx.betweenness_sample == DEFAULT_BETWEENNESS_SAMPLE:
        pytest.fail("Betweenness sample should differ from default when capped")
