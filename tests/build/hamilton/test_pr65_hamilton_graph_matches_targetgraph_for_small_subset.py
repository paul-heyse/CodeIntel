"""PR-65: Runtime target graph uses Hamilton-derived dependencies."""

from __future__ import annotations

import pytest

from codeintel.build.hamilton.driver_factory import build_driver
from codeintel.build.hamilton.introspect import derive_target_dependencies


def test_pr65_runtime_graph_dependencies_match_introspection() -> None:
    """Runtime.graph dependencies should match Hamilton introspection output."""
    runtime = build_driver()
    derived = derive_target_dependencies(runtime)

    missing: list[str] = []
    for target_name in sorted(t.name for t in runtime.graph.all_targets):
        deps = derived.get(target_name)
        if deps is None:
            missing.append(target_name)
            continue
        target = runtime.graph.get(target_name)
        if target.dependencies != deps:
            message = f"Target '{target_name}' dependencies differ: {target.dependencies} != {deps}"
            pytest.fail(message)

    if missing:
        pytest.fail("Introspection missing derived dependencies for targets: " + ", ".join(missing))
