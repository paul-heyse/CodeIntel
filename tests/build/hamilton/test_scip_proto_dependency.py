"""Tests for SCIP target dependencies."""

from __future__ import annotations

import pytest

from codeintel.build.hamilton.driver_factory import build_driver
from codeintel.build.hamilton.introspect import derive_target_dependencies


def test_scip_depends_on_proto_codegen() -> None:
    """SCIP target should depend on scip_proto for protobuf codegen."""
    runtime = build_driver()
    derived = derive_target_dependencies(runtime)
    deps = derived.get("scip")
    if deps is None:
        pytest.fail("Expected derived dependencies for scip target")
    if "scip_proto" not in deps:
        pytest.fail("Expected scip target to depend on scip_proto")
