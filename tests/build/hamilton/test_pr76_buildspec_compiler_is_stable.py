"""Tests for PR-76: BuildSpec compiler is deterministic."""

from __future__ import annotations

import pytest

from codeintel.build.spec import compile_buildspec


def test_buildspec_compiler_is_stable() -> None:
    """Compile twice and require identical buildspec_hash."""
    spec1 = compile_buildspec()
    spec2 = compile_buildspec()

    if spec1.buildspec_hash != spec2.buildspec_hash:
        pytest.fail(
            f"buildspec_hash mismatch across two compiles: {spec1.buildspec_hash} != {spec2.buildspec_hash}"
        )
