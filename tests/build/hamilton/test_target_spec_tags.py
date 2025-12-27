"""Tests for DAG-derived target specs and tag invariants."""

from __future__ import annotations

import pytest

from codeintel.build.hamilton.dag_catalog_compiler import compile_dag_catalog
from codeintel.build.hamilton.driver_factory import build_driver
from codeintel.core.hamilton import tags as ht


def _variable_name(variable: object) -> str:
    if isinstance(variable, str):
        return variable
    name = getattr(variable, "name", None)
    return str(name) if name is not None else str(variable)


def test_all_targets_compile_from_dag() -> None:
    """Ensure DAG-derived target compilation succeeds and produces targets."""
    runtime = build_driver()
    catalog = compile_dag_catalog(runtime.dr, strict=True)
    if not catalog.all_targets:
        pytest.fail("No build targets compiled from DAG tags")


def test_target_anchors_have_docstrings() -> None:
    """Ensure every target anchor has a docstring summary."""
    runtime = build_driver()
    catalog = compile_dag_catalog(runtime.dr, strict=False)
    missing = [target.name for target in catalog.all_targets if not target.description.strip()]
    if missing:
        pytest.fail("Targets missing docstring summaries:\n" + "\n".join(sorted(missing)))


def test_target_anchors_have_spec_version() -> None:
    """Ensure target anchors carry the canonical spec version tag."""
    runtime = build_driver()
    missing: list[str] = []
    variables = runtime.tag_query.query({ht.TAG_NODE_TYPE: ht.NODE_TYPE_MATERIALIZE})
    for variable in variables:
        tags = getattr(variable, "tags", None)
        if not isinstance(tags, dict):
            continue
        if tags.get(ht.TAG_TARGET_SPEC_VERSION) != "1":
            missing.append(_variable_name(variable))
    if missing:
        pytest.fail("Target anchors missing spec version tag:\n" + "\n".join(sorted(missing)))
