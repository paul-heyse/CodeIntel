"""Tests for SCIP target dependencies."""

from __future__ import annotations

import pytest

from codeintel.runtime.runtime_bundle import HamiltonRuntimeBundle


def test_scip_depends_on_proto_codegen(hamilton_runtime: HamiltonRuntimeBundle) -> None:
    """SCIP target should depend on scip_proto for protobuf codegen."""
    deps = hamilton_runtime.catalog.target_dependencies.get("scip")
    if deps is None:
        pytest.fail("Expected derived dependencies for scip target")
    if "scip_proto" not in deps:
        pytest.fail("Expected scip target to depend on scip_proto")
