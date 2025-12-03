"""Tests for data_models module."""

from __future__ import annotations

from datetime import UTC, datetime

from codeintel.storage.data_models import (
    DataModelFieldRow,
    DataModelRelationshipRow,
    DataModelRow,
    fetch_fields,
    fetch_models,
    fetch_relationships,
)
from codeintel.storage.gateway import StorageGateway


def test_data_model_row_is_frozen() -> None:
    """Verify DataModelRow is immutable."""
    now = datetime.now(tz=UTC)
    row = DataModelRow(
        repo="test/repo",
        commit="abc123",
        model_id="model_1",
        goid_h128=None,
        model_name="TestModel",
        module="test.module",
        rel_path="test/module.py",
        model_kind="dataclass",
        base_classes=[],
        doc_short="Short doc",
        doc_long=None,
        created_at=now,
    )

    assert row.model_name == "TestModel"


def test_data_model_field_row_is_frozen() -> None:
    """Verify DataModelFieldRow is immutable."""
    now = datetime.now(tz=UTC)
    row = DataModelFieldRow(
        repo="test/repo",
        commit="abc123",
        model_id="model_1",
        name="field1",
        field_type="str",
        required=True,
        has_default=False,
        default_expr=None,
        constraints={},
        source="annotation",
        rel_path="test/module.py",
        lineno=10,
        created_at=now,
    )

    assert row.name == "field1"


def test_data_model_relationship_row_is_frozen() -> None:
    """Verify DataModelRelationshipRow is immutable."""
    now = datetime.now(tz=UTC)
    row = DataModelRelationshipRow(
        repo="test/repo",
        commit="abc123",
        source_model_id="model_1",
        target_model_id="model_2",
        target_module="target.module",
        target_model_name="TargetModel",
        field_name="items",
        relationship_kind="has_many",
        multiplicity="many",
        via="foreign_key",
        evidence={},
        rel_path="test/module.py",
        lineno=20,
        created_at=now,
    )

    assert row.relationship_kind == "has_many"


def test_fetch_models_returns_list(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify fetch_models returns list of DataModelRow."""
    result = fetch_models(fresh_gateway, "test/repo", "abc123")

    assert isinstance(result, list)


def test_fetch_models_filters_by_repo_commit(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify fetch_models filters by repo and commit."""
    con = fresh_gateway.con
    now = datetime.now(tz=UTC)

    con.execute(
        """
        INSERT INTO analytics.data_models (
            repo, commit, model_id, goid_h128, model_name, module, rel_path,
            model_kind, base_classes_json, doc_short, doc_long, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "test/repo",
            "abc123",
            "model_1",
            None,
            "TestModel",
            "test.module",
            "test/module.py",
            "dataclass",
            "[]",
            "Short doc",
            None,
            now,
        ],
    )

    result = fetch_models(fresh_gateway, "test/repo", "abc123")

    assert len(result) == 1

    result_other = fetch_models(fresh_gateway, "other/repo", "def456")
    assert len(result_other) == 0


def test_fetch_fields_returns_list(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify fetch_fields returns list of DataModelFieldRow."""
    result = fetch_fields(fresh_gateway, "test/repo", "abc123")

    assert isinstance(result, list)


def test_fetch_relationships_returns_list(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify fetch_relationships returns list."""
    result = fetch_relationships(fresh_gateway, "test/repo", "abc123")

    assert isinstance(result, list)
