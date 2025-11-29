"""Tests for graph feature flag validation and propagation."""

from __future__ import annotations

import pytest

from codeintel.analytics.graph_runtime import GraphRuntimeOptions
from codeintel.config.primitives import GraphFeatureFlags, SnapshotRef


def _expect(*, condition: bool, detail: str) -> None:
    if condition:
        return
    raise AssertionError(detail)


def test_graph_feature_flags_validate_limit() -> None:
    """community_detection_limit must be positive when provided."""
    flags = GraphFeatureFlags(community_detection_limit=-1)
    with pytest.raises(ValueError, match="community_detection_limit"):
        flags.validate()


def test_graph_runtime_options_resolve_eager_flag(tmp_path: object) -> None:
    """Resolved eager flag should honor feature override when set."""
    snapshot = SnapshotRef(repo="r", commit="c", repo_root=tmp_path)  # type: ignore[arg-type]
    options = GraphRuntimeOptions(snapshot=snapshot, eager=False)
    _expect(
        condition=options.resolved_eager is False,
        detail="resolved eager should reflect default flag",
    )

    override = GraphFeatureFlags(eager_hydration=True)
    options_override = GraphRuntimeOptions(snapshot=snapshot, eager=False, features=override)
    _expect(
        condition=options_override.resolved_eager is True,
        detail="feature override should take precedence",
    )
