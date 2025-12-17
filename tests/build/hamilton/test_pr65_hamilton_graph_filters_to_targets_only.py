"""PR-65: Hamilton-derived TargetGraph filters to targets only."""

from __future__ import annotations

import pytest

from codeintel.build.hamilton.driver_factory import build_driver
from codeintel.build.hamilton.introspect import derive_target_dependencies


def test_pr65_hamilton_graph_filters_to_targets_only() -> None:
    """Derived dependencies should reference target names only (no loader/dataset nodes)."""
    runtime = build_driver()

    deps = derive_target_dependencies(runtime)
    all_targets = {t.name for t in runtime.graph.all_targets}

    if not set(deps).issubset(all_targets):
        pytest.fail("Derived dependency mapping contains non-target keys")
    for target_name, dep_targets in deps.items():
        if target_name not in all_targets:
            pytest.fail(f"Derived dependency key is not a target: {target_name}")
        if not set(dep_targets).issubset(all_targets):
            pytest.fail(f"{target_name} dependencies include non-targets: {dep_targets}")
