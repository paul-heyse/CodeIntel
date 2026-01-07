"""Tests for PR-76: BuildSpec compiler is deterministic."""

from __future__ import annotations

import pytest

from codeintel.build.schemas import get_schema_provider
from codeintel.build.spec import compile_buildspec
from codeintel.runtime.runtime_bundle import HamiltonRuntimeBundle


def test_buildspec_compiler_is_stable(hamilton_runtime: HamiltonRuntimeBundle) -> None:
    """Compile twice and require identical buildspec_hash."""
    spec1 = compile_buildspec(
        catalog=hamilton_runtime.catalog,
        provider=get_schema_provider(),
    )
    spec2 = compile_buildspec(
        catalog=hamilton_runtime.catalog,
        provider=get_schema_provider(),
    )

    if spec1.buildspec_hash != spec2.buildspec_hash:
        pytest.fail(
            f"buildspec_hash mismatch across two compiles: {spec1.buildspec_hash} != {spec2.buildspec_hash}"
        )
