"""Comprehensive tests for data_models repository module.

This module tests all data model functions and dataclasses in
codeintel.storage.repositories.data_models, following the Testing Charter.
"""

from __future__ import annotations

from datetime import UTC, datetime

from codeintel.storage.gateway import StorageGateway
from codeintel.storage.repositories.data_models import (
    DataModelFieldRow,
    DataModelRelationshipRow,
    DataModelRow,
    NormalizedDataModel,
    fetch_fields,
    fetch_models,
    fetch_models_normalized,
    fetch_relationships,
)
from tests._helpers import assert_frozen

# Test constants to avoid magic value warnings
EXPECTED_INT_42 = 42
EXPECTED_INT_123 = 123
EXPECTED_GOID = 1001
EXPECTED_LINENO_10 = 10
EXPECTED_LINENO_25 = 25
EXPECTED_COUNT_2 = 2
EXPECTED_COUNT_3 = 3


# =============================================================================
# DataClass Tests
# =============================================================================


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
    assert_frozen(row, "model_name", "Other")


def test_data_model_row_stores_all_fields() -> None:
    """Verify DataModelRow stores all fields correctly."""
    now = datetime.now(tz=UTC)
    base_classes = [{"name": "BaseClass", "qualname": "mod.BaseClass"}]
    row = DataModelRow(
        repo="test/repo",
        commit="abc123",
        model_id="model_1",
        goid_h128=EXPECTED_GOID,
        model_name="TestModel",
        module="test.module",
        rel_path="test/module.py",
        model_kind="dataclass",
        base_classes=base_classes,
        doc_short="Short doc",
        doc_long="Long documentation",
        created_at=now,
    )
    assert row.repo == "test/repo"
    assert row.commit == "abc123"
    assert row.model_id == "model_1"
    assert row.goid_h128 == EXPECTED_GOID
    assert row.model_name == "TestModel"
    assert row.module == "test.module"
    assert row.rel_path == "test/module.py"
    assert row.model_kind == "dataclass"
    assert row.base_classes == base_classes
    assert row.doc_short == "Short doc"
    assert row.doc_long == "Long documentation"
    assert row.created_at == now


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
        lineno=EXPECTED_LINENO_10,
        created_at=now,
    )
    assert row.name == "field1"
    assert_frozen(row, "name", "other")


def test_data_model_field_row_stores_all_fields() -> None:
    """Verify DataModelFieldRow stores all fields correctly."""
    now = datetime.now(tz=UTC)
    constraints: dict[str, object] = {"max_length": 100}
    row = DataModelFieldRow(
        repo="test/repo",
        commit="abc123",
        model_id="model_1",
        name="field1",
        field_type="str",
        required=True,
        has_default=True,
        default_expr='"default"',
        constraints=constraints,
        source="annotation",
        rel_path="test/module.py",
        lineno=EXPECTED_LINENO_10,
        created_at=now,
    )
    assert row.repo == "test/repo"
    assert row.commit == "abc123"
    assert row.model_id == "model_1"
    assert row.name == "field1"
    assert row.field_type == "str"
    assert row.required is True
    assert row.has_default is True
    assert row.default_expr == '"default"'
    assert row.constraints == constraints
    assert row.source == "annotation"
    assert row.rel_path == "test/module.py"
    assert row.lineno == EXPECTED_LINENO_10
    assert row.created_at == now


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
    assert_frozen(row, "relationship_kind", "other")


def test_data_model_relationship_row_stores_all_fields() -> None:
    """Verify DataModelRelationshipRow stores all fields correctly."""
    now = datetime.now(tz=UTC)
    evidence: dict[str, object] = {"confidence": 0.9}
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
        evidence=evidence,
        rel_path="test/module.py",
        lineno=EXPECTED_LINENO_25,
        created_at=now,
    )
    assert row.repo == "test/repo"
    assert row.commit == "abc123"
    assert row.source_model_id == "model_1"
    assert row.target_model_id == "model_2"
    assert row.target_module == "target.module"
    assert row.target_model_name == "TargetModel"
    assert row.field_name == "items"
    assert row.relationship_kind == "has_many"
    assert row.multiplicity == "many"
    assert row.via == "foreign_key"
    assert row.evidence == evidence
    assert row.rel_path == "test/module.py"
    assert row.lineno == EXPECTED_LINENO_25
    assert row.created_at == now


def test_normalized_data_model_is_frozen() -> None:
    """Verify NormalizedDataModel is immutable."""
    now = datetime.now(tz=UTC)
    model = NormalizedDataModel(
        repo="test/repo",
        commit="abc123",
        model_id="model_1",
        goid_h128=EXPECTED_GOID,
        model_name="TestModel",
        module="test.module",
        rel_path="test/module.py",
        model_kind="dataclass",
        base_classes=[],
        fields=[],
        relationships=[],
        doc_short="Short doc",
        doc_long="Long doc",
        created_at=now,
    )
    assert model.model_name == "TestModel"
    assert_frozen(model, "model_name", "Other")


def test_normalized_data_model_stores_nested_data() -> None:
    """Verify NormalizedDataModel stores nested fields and relationships."""
    now = datetime.now(tz=UTC)
    field = DataModelFieldRow(
        repo="test/repo",
        commit="abc123",
        model_id="model_1",
        name="field",
        field_type="str",
        required=True,
        has_default=False,
        default_expr=None,
        constraints={},
        source="annotation",
        rel_path="test.py",
        lineno=1,
        created_at=now,
    )
    rel = DataModelRelationshipRow(
        repo="test/repo",
        commit="abc123",
        source_model_id="model_1",
        target_model_id="model_2",
        target_module=None,
        target_model_name=None,
        field_name="ref",
        relationship_kind="has_one",
        multiplicity="one",
        via=None,
        evidence={},
        rel_path="test.py",
        lineno=1,
        created_at=now,
    )
    model = NormalizedDataModel(
        repo="test/repo",
        commit="abc123",
        model_id="model_1",
        goid_h128=EXPECTED_GOID,
        model_name="TestModel",
        module="test.module",
        rel_path="test.py",
        model_kind="dataclass",
        base_classes=[],
        fields=[field],
        relationships=[rel],
        doc_short=None,
        doc_long=None,
        created_at=now,
    )
    assert len(model.fields) == 1
    assert model.fields[0].name == "field"
    assert len(model.relationships) == 1
    assert model.relationships[0].target_model_id == "model_2"


# =============================================================================
# fetch_models Tests
# =============================================================================


def test_fetch_models_returns_empty_list(fresh_gateway: StorageGateway) -> None:
    """Verify fetch_models returns empty list when no data."""
    result = fetch_models(fresh_gateway, "test/repo", "abc123")
    assert isinstance(result, list)
    assert len(result) == 0


def test_fetch_models_filters_by_repo_commit(fresh_gateway: StorageGateway) -> None:
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


def test_fetch_models_parses_all_fields(fresh_gateway: StorageGateway) -> None:
    """Verify fetch_models correctly parses all fields."""
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
            EXPECTED_GOID,
            "TestModel",
            "test.module",
            "test/module.py",
            "pydantic",
            '[{"name": "BaseModel", "qualname": "pydantic.BaseModel"}]',
            "Short doc",
            "Long documentation",
            now,
        ],
    )

    result = fetch_models(fresh_gateway, "test/repo", "abc123")
    assert len(result) == 1

    model = result[0]
    assert model.repo == "test/repo"
    assert model.commit == "abc123"
    assert model.model_id == "model_1"
    assert model.goid_h128 == EXPECTED_GOID
    assert model.model_name == "TestModel"
    assert model.module == "test.module"
    assert model.rel_path == "test/module.py"
    assert model.model_kind == "pydantic"
    assert len(model.base_classes) == 1
    assert model.base_classes[0]["name"] == "BaseModel"
    assert model.doc_short == "Short doc"
    assert model.doc_long == "Long documentation"


def test_fetch_models_parses_base_classes_from_json(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify fetch_models correctly parses base_classes_json."""
    con = fresh_gateway.con
    now = datetime.now(tz=UTC)

    # Multiple base classes
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
            '[{"name": "Base1", "qualname": "m.Base1"}, {"name": "Base2", "qualname": "m.Base2"}]',
            None,
            None,
            now,
        ],
    )

    result = fetch_models(fresh_gateway, "test/repo", "abc123")
    model = result[0]
    assert len(model.base_classes) == EXPECTED_COUNT_2
    assert model.base_classes[0]["name"] == "Base1"
    assert model.base_classes[1]["name"] == "Base2"


def test_fetch_models_handles_goid_as_decimal(fresh_gateway: StorageGateway) -> None:
    """Verify fetch_models handles goid_h128 Decimal conversion."""
    con = fresh_gateway.con
    now = datetime.now(tz=UTC)

    # Insert with a large goid value
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
            123456789012345,  # Large number that might be stored as Decimal
            "TestModel",
            "test.module",
            "test/module.py",
            "dataclass",
            "[]",
            None,
            None,
            now,
        ],
    )

    result = fetch_models(fresh_gateway, "test/repo", "abc123")
    model = result[0]
    assert model.goid_h128 is not None
    assert isinstance(model.goid_h128, int)


def test_fetch_models_returns_multiple(fresh_gateway: StorageGateway) -> None:
    """Verify fetch_models returns multiple models."""
    con = fresh_gateway.con
    now = datetime.now(tz=UTC)

    for i in range(EXPECTED_COUNT_3):
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
                f"model_{i}",
                1000 + i,
                f"Model{i}",
                "test.module",
                "test/module.py",
                "dataclass",
                "[]",
                None,
                None,
                now,
            ],
        )

    result = fetch_models(fresh_gateway, "test/repo", "abc123")
    assert len(result) == EXPECTED_COUNT_3


# =============================================================================
# fetch_fields Tests
# =============================================================================


def test_fetch_fields_returns_empty_list(fresh_gateway: StorageGateway) -> None:
    """Verify fetch_fields returns empty list when no data."""
    result = fetch_fields(fresh_gateway, "test/repo", "abc123")
    assert isinstance(result, list)
    assert len(result) == 0


def test_fetch_fields_parses_all_fields(fresh_gateway: StorageGateway) -> None:
    """Verify fetch_fields correctly parses all fields."""
    con = fresh_gateway.con
    now = datetime.now(tz=UTC)

    con.execute(
        """
        INSERT INTO analytics.data_model_fields (
            repo, commit, model_id, field_name, field_type, required, has_default,
            default_expr, constraints_json, source, rel_path, lineno, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "test/repo",
            "abc123",
            "model_1",
            "name",
            "str",
            True,
            False,
            None,
            '{"max_length": 100}',
            "annotation",
            "test/module.py",
            EXPECTED_LINENO_10,
            now,
        ],
    )

    result = fetch_fields(fresh_gateway, "test/repo", "abc123")
    assert len(result) == 1

    field = result[0]
    assert field.repo == "test/repo"
    assert field.commit == "abc123"
    assert field.model_id == "model_1"
    assert field.name == "name"
    assert field.field_type == "str"
    assert field.required is True
    assert field.has_default is False
    assert field.default_expr is None
    assert field.constraints == {"max_length": 100}
    assert field.source == "annotation"
    assert field.rel_path == "test/module.py"
    assert field.lineno == EXPECTED_LINENO_10


def test_fetch_fields_filters_by_model_ids(fresh_gateway: StorageGateway) -> None:
    """Verify fetch_fields filters by model_ids when provided."""
    con = fresh_gateway.con
    now = datetime.now(tz=UTC)

    # Insert fields for multiple models
    for model_id in ["model_1", "model_2", "model_3"]:
        con.execute(
            """
            INSERT INTO analytics.data_model_fields (
                repo, commit, model_id, field_name, field_type, required, has_default,
                default_expr, constraints_json, source, rel_path, lineno, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                "test/repo",
                "abc123",
                model_id,
                "field",
                "str",
                True,
                False,
                None,
                "{}",
                "annotation",
                "test.py",
                1,
                now,
            ],
        )

    # Fetch only for model_1 and model_2
    result = fetch_fields(fresh_gateway, "test/repo", "abc123", model_ids=["model_1", "model_2"])
    assert len(result) == EXPECTED_COUNT_2

    model_ids = {f.model_id for f in result}
    assert "model_1" in model_ids
    assert "model_2" in model_ids
    assert "model_3" not in model_ids


def test_fetch_fields_without_model_ids_returns_all(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify fetch_fields returns all fields when model_ids is None."""
    con = fresh_gateway.con
    now = datetime.now(tz=UTC)

    for model_id in ["model_1", "model_2"]:
        con.execute(
            """
            INSERT INTO analytics.data_model_fields (
                repo, commit, model_id, field_name, field_type, required, has_default,
                default_expr, constraints_json, source, rel_path, lineno, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                "test/repo",
                "abc123",
                model_id,
                "field",
                "str",
                True,
                False,
                None,
                "{}",
                "annotation",
                "test.py",
                1,
                now,
            ],
        )

    result = fetch_fields(fresh_gateway, "test/repo", "abc123")
    assert len(result) == EXPECTED_COUNT_2


def test_fetch_fields_parses_constraints_json(fresh_gateway: StorageGateway) -> None:
    """Verify fetch_fields correctly parses constraints_json."""
    con = fresh_gateway.con
    now = datetime.now(tz=UTC)

    con.execute(
        """
        INSERT INTO analytics.data_model_fields (
            repo, commit, model_id, field_name, field_type, required, has_default,
            default_expr, constraints_json, source, rel_path, lineno, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "test/repo",
            "abc123",
            "model_1",
            "email",
            "str",
            True,
            False,
            None,
            '{"pattern": "^.+@.+$", "max_length": 255}',
            "annotation",
            "test.py",
            1,
            now,
        ],
    )

    result = fetch_fields(fresh_gateway, "test/repo", "abc123")
    field = result[0]
    assert field.constraints == {"pattern": "^.+@.+$", "max_length": 255}


# =============================================================================
# fetch_relationships Tests
# =============================================================================


def test_fetch_relationships_returns_empty_list(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify fetch_relationships returns empty list when no data."""
    result = fetch_relationships(fresh_gateway, "test/repo", "abc123")
    assert isinstance(result, list)
    assert len(result) == 0


def test_fetch_relationships_parses_all_fields(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify fetch_relationships correctly parses all fields."""
    con = fresh_gateway.con
    now = datetime.now(tz=UTC)

    con.execute(
        """
        INSERT INTO analytics.data_model_relationships (
            repo, commit, source_model_id, target_model_id, target_module,
            target_model_name, field_name, relationship_kind, multiplicity, via,
            evidence_json, rel_path, lineno, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "test/repo",
            "abc123",
            "model_1",
            "model_2",
            "target.module",
            "TargetModel",
            "items",
            "has_many",
            "many",
            "foreign_key",
            '{"confidence": 0.9}',
            "test/module.py",
            EXPECTED_LINENO_25,
            now,
        ],
    )

    result = fetch_relationships(fresh_gateway, "test/repo", "abc123")
    assert len(result) == 1

    rel = result[0]
    assert rel.repo == "test/repo"
    assert rel.commit == "abc123"
    assert rel.source_model_id == "model_1"
    assert rel.target_model_id == "model_2"
    assert rel.target_module == "target.module"
    assert rel.target_model_name == "TargetModel"
    assert rel.field_name == "items"
    assert rel.relationship_kind == "has_many"
    assert rel.multiplicity == "many"
    assert rel.via == "foreign_key"
    assert rel.evidence == {"confidence": 0.9}
    assert rel.rel_path == "test/module.py"
    assert rel.lineno == EXPECTED_LINENO_25


def test_fetch_relationships_filters_by_model_ids(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify fetch_relationships filters by source model_ids when provided."""
    con = fresh_gateway.con
    now = datetime.now(tz=UTC)

    for src_model in ["model_1", "model_2", "model_3"]:
        con.execute(
            """
            INSERT INTO analytics.data_model_relationships (
                repo, commit, source_model_id, target_model_id, target_module,
                target_model_name, field_name, relationship_kind, multiplicity, via,
                evidence_json, rel_path, lineno, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                "test/repo",
                "abc123",
                src_model,
                "target",
                "target.mod",
                "Target",
                "ref",
                "has_one",
                "one",
                None,
                "{}",
                "test.py",
                1,
                now,
            ],
        )

    result = fetch_relationships(fresh_gateway, "test/repo", "abc123", model_ids=["model_1"])
    assert len(result) == 1
    assert result[0].source_model_id == "model_1"


def test_fetch_relationships_handles_nullable_fields(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify fetch_relationships handles null optional fields."""
    con = fresh_gateway.con
    now = datetime.now(tz=UTC)

    con.execute(
        """
        INSERT INTO analytics.data_model_relationships (
            repo, commit, source_model_id, target_model_id, target_module,
            target_model_name, field_name, relationship_kind, multiplicity, via,
            evidence_json, rel_path, lineno, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "test/repo",
            "abc123",
            "model_1",
            "model_2",
            None,  # nullable
            None,  # nullable
            "field_ref",  # NOT NULL - must provide a value
            "references",
            None,  # nullable
            None,  # nullable
            "{}",
            "test.py",
            None,  # nullable
            now,
        ],
    )

    result = fetch_relationships(fresh_gateway, "test/repo", "abc123")
    rel = result[0]
    assert rel.target_module is None
    assert rel.target_model_name is None
    assert rel.field_name == "field_ref"
    assert rel.multiplicity is None
    assert rel.via is None
    assert rel.lineno is None


# =============================================================================
# fetch_models_normalized Tests
# =============================================================================


def test_fetch_models_normalized_returns_empty_list(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify fetch_models_normalized returns empty list when no data."""
    result = fetch_models_normalized(fresh_gateway, "test/repo", "abc123")
    assert isinstance(result, list)
    assert len(result) == 0


def test_fetch_models_normalized_joins_data(fresh_gateway: StorageGateway) -> None:
    """Verify fetch_models_normalized returns joined fields and relationships."""
    con = fresh_gateway.con
    now = datetime.now(tz=UTC)

    # Insert a data model
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
            EXPECTED_GOID,
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

    # Insert a field
    con.execute(
        """
        INSERT INTO analytics.data_model_fields (
            repo, commit, model_id, field_name, field_type, required, has_default,
            default_expr, constraints_json, source, rel_path, lineno, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "test/repo",
            "abc123",
            "model_1",
            "name",
            "str",
            True,
            False,
            None,
            "{}",
            "annotation",
            "test/module.py",
            EXPECTED_LINENO_10,
            now,
        ],
    )

    # Insert a relationship
    con.execute(
        """
        INSERT INTO analytics.data_model_relationships (
            repo, commit, source_model_id, target_model_id, target_module,
            target_model_name, field_name, relationship_kind, multiplicity, via,
            evidence_json, rel_path, lineno, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            "test/repo",
            "abc123",
            "model_1",
            "model_2",
            "target.mod",
            "Target",
            "ref",
            "has_one",
            "one",
            None,
            "{}",
            "test/module.py",
            20,
            now,
        ],
    )

    result = fetch_models_normalized(fresh_gateway, "test/repo", "abc123")
    assert len(result) == 1

    model = result[0]
    assert model.model_id == "model_1"
    assert model.model_name == "TestModel"
    assert len(model.fields) == 1
    assert model.fields[0].name == "name"
    assert len(model.relationships) == 1
    assert model.relationships[0].target_model_id == "model_2"


def test_fetch_models_normalized_filters_by_model_ids(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify fetch_models_normalized filters by model_ids."""
    con = fresh_gateway.con
    now = datetime.now(tz=UTC)

    for i in range(EXPECTED_COUNT_3):
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
                f"model_{i}",
                1000 + i,
                f"Model{i}",
                "test.module",
                "test/module.py",
                "dataclass",
                "[]",
                None,
                None,
                now,
            ],
        )

    result = fetch_models_normalized(
        fresh_gateway, "test/repo", "abc123", model_ids=["model_0", "model_1"]
    )
    assert len(result) == EXPECTED_COUNT_2

    model_ids = {m.model_id for m in result}
    assert "model_0" in model_ids
    assert "model_1" in model_ids
    assert "model_2" not in model_ids
