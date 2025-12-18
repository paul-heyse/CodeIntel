"""Regression test: static OutputTarget dependencies are decommissioned.

The Hamilton DAG is the single source of truth for target dependencies. The
static OutputTarget specs in ``codeintel.build.target_catalog`` must not encode
dependency tuples, to avoid drift and accidental reintroduction of a second
dependency source of truth.
"""

from __future__ import annotations

from codeintel.build.target_catalog import load_target_specs
from tests._helpers.assertions import expect_true


def test_all_targets_have_no_static_dependencies() -> None:
    """Ensure every static OutputTarget declares empty dependencies."""
    non_empty = {t.name: t.dependencies for t in load_target_specs() if t.dependencies}
    expect_true(
        len(non_empty) == 0,
        message="Targets still declare static dependencies: " + str(sorted(non_empty.items())),
    )
