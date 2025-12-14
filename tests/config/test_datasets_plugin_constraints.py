"""Tests for plugin-based constraint extraction.

Tests the PluginTableRelation dataclass, plugin lookup functions,
and extract_constraints_from_plugins function.
"""

from __future__ import annotations

import pytest

from codeintel.build.hamilton.contracts.schemas.constraints import (
    Constraint,
    ConstraintKind,
    ConstraintSet,
)
from codeintel.build.hamilton.contracts.schemas.plugin_constraints import (
    PluginTableRelation,
    extract_constraints_from_plugins,
    get_consumer_plugins,
    get_producer_plugins,
    get_table_plugin_relations,
    merge_constraint_sets,
)


def _require(*, condition: bool, message: str) -> None:
    """Assert a condition using pytest.fail for S101 compliance."""
    if not condition:
        pytest.fail(message)


def _expect_equal(actual: object, expected: object, label: str) -> None:
    """Check equality with clear failure message."""
    if actual != expected:
        pytest.fail(f"{label}: expected {expected!r}, got {actual!r}")


def test_plugin_table_relation_creation() -> None:
    """Create PluginTableRelation with all fields."""
    rel = PluginTableRelation(
        plugin_name="analytics.function_metrics",
        plugin_version="3.0.0",
        table_key="analytics.function_metrics",
        relation_type="produces",
        domain="analytics",
    )
    _expect_equal(rel.plugin_name, "analytics.function_metrics", "plugin_name")
    _expect_equal(rel.plugin_version, "3.0.0", "plugin_version")
    _expect_equal(rel.table_key, "analytics.function_metrics", "table_key")
    _expect_equal(rel.relation_type, "produces", "relation_type")
    _expect_equal(rel.domain, "analytics", "domain")


def test_plugin_table_relation_is_producer() -> None:
    """Verify is_producer property for producer relations."""
    rel = PluginTableRelation(
        plugin_name="test",
        plugin_version="1.0",
        table_key="test.table",
        relation_type="produces",
        domain="analytics",
    )
    _require(condition=rel.is_producer, message="should be producer")
    _require(condition=not rel.is_consumer, message="should not be consumer")


def test_plugin_table_relation_is_consumer() -> None:
    """Verify is_consumer property for consumer relations."""
    rel = PluginTableRelation(
        plugin_name="test",
        plugin_version="1.0",
        table_key="test.table",
        relation_type="consumes",
        domain="analytics",
    )
    _require(condition=rel.is_consumer, message="should be consumer")
    _require(condition=not rel.is_producer, message="should not be producer")


def test_plugin_table_relation_immutability() -> None:
    """Verify PluginTableRelation is frozen/immutable."""
    rel = PluginTableRelation(
        plugin_name="test",
        plugin_version="1.0",
        table_key="test.table",
        relation_type="produces",
        domain="analytics",
    )
    frozen_rel: object = rel
    attr = "plugin_name"
    with pytest.raises(AttributeError):
        setattr(frozen_rel, attr, "modified")


def test_get_producer_plugins_returns_list() -> None:
    """Verify get_producer_plugins returns a list."""
    result = get_producer_plugins("analytics.function_metrics")
    _require(condition=isinstance(result, list), message="should return list")


def test_get_consumer_plugins_returns_list() -> None:
    """Verify get_consumer_plugins returns a list."""
    result = get_consumer_plugins("analytics.function_metrics")
    _require(condition=isinstance(result, list), message="should return list")


def test_get_table_plugin_relations_returns_list() -> None:
    """Verify get_table_plugin_relations returns a list."""
    result = get_table_plugin_relations("analytics.function_metrics")
    _require(condition=isinstance(result, list), message="should return list")


def test_get_producer_plugins_unknown_table() -> None:
    """Verify get_producer_plugins returns empty for unknown table."""
    result = get_producer_plugins("nonexistent.table")
    _expect_equal(len(result), 0, "should be empty for unknown table")


def test_get_consumer_plugins_unknown_table() -> None:
    """Verify get_consumer_plugins returns empty for unknown table."""
    result = get_consumer_plugins("nonexistent.table")
    _expect_equal(len(result), 0, "should be empty for unknown table")


def test_extract_constraints_from_plugins_returns_constraintset() -> None:
    """Verify extract_constraints_from_plugins returns ConstraintSet."""
    result = extract_constraints_from_plugins("analytics.function_metrics")
    _require(condition=isinstance(result, ConstraintSet), message="should return ConstraintSet")
    _expect_equal(result.table_key, "analytics.function_metrics", "table_key")


def test_extract_constraints_from_plugins_empty_for_unknown() -> None:
    """Verify empty constraints for unknown table."""
    result = extract_constraints_from_plugins("nonexistent.table")
    _expect_equal(len(result.constraints), 0, "should have no constraints")


def test_extract_constraints_from_plugins_constraint_kinds() -> None:
    """Verify constraint kinds are COMPUTATION or FOREIGN_KEY."""
    result = extract_constraints_from_plugins("analytics.function_metrics")
    valid_kinds = {ConstraintKind.COMPUTATION, ConstraintKind.FOREIGN_KEY}
    for c in result.constraints:
        _require(
            condition=c.kind in valid_kinds,
            message=f"unexpected constraint kind: {c.kind}",
        )


def test_merge_constraint_sets_single() -> None:
    """Verify merging a single ConstraintSet returns equivalent set."""
    cs = ConstraintSet(table_key="test.table")
    cs.add(
        Constraint(
            kind=ConstraintKind.TYPE,
            column="col",
            expression="col: int",
        )
    )

    merged = merge_constraint_sets(cs)
    _expect_equal(merged.table_key, "test.table", "table_key")
    _expect_equal(len(merged.constraints), 1, "constraint count")


def test_merge_constraint_sets_multiple() -> None:
    """Verify merging multiple ConstraintSets combines constraints."""
    cs1 = ConstraintSet(table_key="test.table")
    cs1.add(
        Constraint(
            kind=ConstraintKind.TYPE,
            column="col1",
            expression="col1: int",
        )
    )

    cs2 = ConstraintSet(table_key="test.table")
    cs2.add(
        Constraint(
            kind=ConstraintKind.RANGE,
            column="col2",
            expression="col2 >= 0",
        )
    )

    merged = merge_constraint_sets(cs1, cs2)
    _expect_equal(merged.table_key, "test.table", "table_key")
    _expect_equal(len(merged.constraints), 2, "constraint count")


def test_merge_constraint_sets_different_keys_raises() -> None:
    """Verify merging ConstraintSets with different keys raises ValueError."""
    cs1 = ConstraintSet(table_key="test.table1")
    cs2 = ConstraintSet(table_key="test.table2")

    try:
        merge_constraint_sets(cs1, cs2)
        pytest.fail("Should have raised ValueError")
    except ValueError:
        pass


def test_merge_constraint_sets_empty_raises() -> None:
    """Verify merging no ConstraintSets raises ValueError."""
    try:
        merge_constraint_sets()
        pytest.fail("Should have raised ValueError")
    except ValueError:
        pass
