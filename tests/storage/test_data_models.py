"""Comprehensive tests for data_models repository module.

This module tests all data model functions and dataclasses in
codeintel.storage.repositories.data_models, following the Testing Charter.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from codeintel.storage.repositories.data_models import (
    DataModelFieldRow,
    DataModelRelationshipRow,
    DataModelRow,
    DataModelsRepository,
    NormalizedDataModel,
)
from tests._helpers import assert_frozen
from tests._helpers.assertions import (
    expect_equal,
    expect_false,
    expect_in,
    expect_is_instance,
    expect_is_none,
    expect_is_not_none,
    expect_length,
    expect_not_in,
    expect_true,
)
from tests._helpers.rows import (
    DataModelFieldSeed,
    DataModelRelationshipSeed,
    DataModelSeed,
    data_model_field_row,
    data_model_relationship_row,
    data_model_row,
)
from tests._helpers.seeds import DATA_MODELS_PACK

if TYPE_CHECKING:
    from tests._helpers.context import TestContext


@pytest.fixture
def data_models_ctx(test_ctx: TestContext) -> TestContext:
    """Provide a TestContext seeded with data models pack for realistic layout.

    Returns
    -------
    TestContext
        Context populated with data model seeds for repository tests.
    """
    test_ctx.require(DATA_MODELS_PACK)
    return test_ctx


def _insert_models(ctx: TestContext, seeds: list[DataModelSeed]) -> None:
    """Insert data model rows using canonical seed helpers."""
    ctx.gateway.con.executemany(
        """
        INSERT INTO analytics.data_models (
            repo, commit, model_id, goid_h128, model_name, module, rel_path,
            model_kind, base_classes_json, doc_short, doc_long, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [data_model_row(seed) for seed in seeds],
    )


def _insert_fields(ctx: TestContext, seeds: list[DataModelFieldSeed]) -> None:
    """Insert data model field rows using canonical seed helpers."""
    ctx.gateway.con.executemany(
        """
        INSERT INTO analytics.data_model_fields (
            repo, commit, model_id, field_name, field_type, required, has_default,
            default_expr, constraints_json, source, rel_path, lineno, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [data_model_field_row(seed) for seed in seeds],
    )


def _insert_relationships(ctx: TestContext, seeds: list[DataModelRelationshipSeed]) -> None:
    """Insert data model relationship rows using canonical seed helpers."""
    ctx.gateway.con.executemany(
        """
        INSERT INTO analytics.data_model_relationships (
            repo, commit, source_model_id, target_model_id, target_module,
            target_model_name, field_name, relationship_kind, multiplicity, via,
            evidence_json, rel_path, lineno, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [data_model_relationship_row(seed) for seed in seeds],
    )


EXPECTED_INT_42 = 42
EXPECTED_INT_123 = 123
EXPECTED_GOID = 1001
EXPECTED_LINENO_10 = 10
EXPECTED_LINENO_25 = 25
EXPECTED_COUNT_2 = 2
EXPECTED_COUNT_3 = 3


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
    expect_equal(row.model_name, "TestModel")
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
    expect_equal(row.repo, "test/repo")
    expect_equal(row.commit, "abc123")
    expect_equal(row.model_id, "model_1")
    expect_equal(row.goid_h128, EXPECTED_GOID)
    expect_equal(row.model_name, "TestModel")
    expect_equal(row.module, "test.module")
    expect_equal(row.rel_path, "test/module.py")
    expect_equal(row.model_kind, "dataclass")
    expect_equal(row.base_classes, base_classes)
    expect_equal(row.doc_short, "Short doc")
    expect_equal(row.doc_long, "Long documentation")
    expect_equal(row.created_at, now)


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
    expect_equal(row.name, "field1")
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
    expect_equal(row.repo, "test/repo")
    expect_equal(row.commit, "abc123")
    expect_equal(row.model_id, "model_1")
    expect_equal(row.name, "field1")
    expect_equal(row.field_type, "str")
    expect_true(row.required)
    expect_true(row.has_default)
    expect_equal(row.default_expr, '"default"')
    expect_equal(row.constraints, constraints)
    expect_equal(row.source, "annotation")
    expect_equal(row.rel_path, "test/module.py")
    expect_equal(row.lineno, EXPECTED_LINENO_10)
    expect_equal(row.created_at, now)


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
    expect_equal(row.relationship_kind, "has_many")
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
    expect_equal(row.repo, "test/repo")
    expect_equal(row.commit, "abc123")
    expect_equal(row.source_model_id, "model_1")
    expect_equal(row.target_model_id, "model_2")
    expect_equal(row.target_module, "target.module")
    expect_equal(row.target_model_name, "TargetModel")
    expect_equal(row.field_name, "items")
    expect_equal(row.relationship_kind, "has_many")
    expect_equal(row.multiplicity, "many")
    expect_equal(row.via, "foreign_key")
    expect_equal(row.evidence, evidence)
    expect_equal(row.rel_path, "test/module.py")
    expect_equal(row.lineno, EXPECTED_LINENO_25)
    expect_equal(row.created_at, now)


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
    expect_equal(model.model_name, "TestModel")
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
    expect_length(model.fields, 1)
    expect_equal(model.fields[0].name, "field")
    expect_length(model.relationships, 1)
    expect_equal(model.relationships[0].target_model_id, "model_2")


def test_fetch_models_returns_empty_list(data_models_ctx: TestContext) -> None:
    """Verify fetch_models returns empty list when no data."""
    result = DataModelsRepository(
        data_models_ctx.gateway, data_models_ctx.repo, data_models_ctx.commit
    ).list_models()
    expect_is_instance(result, list)
    expect_length(result, 0)


def test_fetch_models_filters_by_repo_commit(data_models_ctx: TestContext) -> None:
    """Verify fetch_models filters by repo and commit."""
    now = datetime.now(tz=UTC)

    _insert_models(
        data_models_ctx,
        [
            DataModelSeed(
                model_id="model_1",
                model_name="TestModel",
                module="test.module",
                rel_path="test/module.py",
                model_kind="dataclass",
                goid=None,
                doc_short="Short doc",
                created_at=now,
                repo=data_models_ctx.repo,
                commit=data_models_ctx.commit,
            ),
            DataModelSeed(
                model_id="model_other",
                model_name="Other",
                module="other.module",
                rel_path="other/module.py",
                model_kind="pydantic",
                repo="other/repo",
                commit="def456",
                created_at=now,
            ),
        ],
    )

    result = DataModelsRepository(
        data_models_ctx.gateway, data_models_ctx.repo, data_models_ctx.commit
    ).list_models()
    expect_length(result, 1)

    result_other = DataModelsRepository(data_models_ctx.gateway, "other/repo", "def456").list_models()
    expect_length(result_other, 1)


def test_fetch_models_parses_all_fields(data_models_ctx: TestContext) -> None:
    """Verify fetch_models correctly parses all fields."""
    now = datetime.now(tz=UTC)

    _insert_models(
        data_models_ctx,
        [
            DataModelSeed(
                model_id="model_1",
                model_name="TestModel",
                module="test.module",
                rel_path="test/module.py",
                model_kind="pydantic",
                goid=EXPECTED_GOID,
                base_classes_json=[{"name": "BaseModel", "qualname": "pydantic.BaseModel"}],
                doc_short="Short doc",
                doc_long="Long documentation",
                created_at=now,
                repo=data_models_ctx.repo,
                commit=data_models_ctx.commit,
            )
        ],
    )

    result = DataModelsRepository(
        data_models_ctx.gateway, data_models_ctx.repo, data_models_ctx.commit
    ).list_models()
    expect_length(result, 1)

    model = result[0]
    expect_equal(model.repo, data_models_ctx.repo)
    expect_equal(model.commit, data_models_ctx.commit)
    expect_equal(model.model_id, "model_1")
    expect_equal(model.goid_h128, EXPECTED_GOID)
    expect_equal(model.model_name, "TestModel")
    expect_equal(model.module, "test.module")
    expect_equal(model.rel_path, "test/module.py")
    expect_equal(model.model_kind, "pydantic")
    expect_length(model.base_classes, 1)
    expect_equal(model.base_classes[0]["name"], "BaseModel")
    expect_equal(model.doc_short, "Short doc")
    expect_equal(model.doc_long, "Long documentation")


def test_fetch_models_parses_base_classes_from_json(
    data_models_ctx: TestContext,
) -> None:
    """Verify fetch_models correctly parses base_classes_json."""
    now = datetime.now(tz=UTC)

    _insert_models(
        data_models_ctx,
        [
            DataModelSeed(
                model_id="model_1",
                model_name="TestModel",
                module="test.module",
                rel_path="test/module.py",
                model_kind="dataclass",
                base_classes_json=[
                    {"name": "Base1", "qualname": "m.Base1"},
                    {"name": "Base2", "qualname": "m.Base2"},
                ],
                created_at=now,
                repo=data_models_ctx.repo,
                commit=data_models_ctx.commit,
            )
        ],
    )

    result = DataModelsRepository(
        data_models_ctx.gateway, data_models_ctx.repo, data_models_ctx.commit
    ).list_models()
    model = result[0]
    expect_length(model.base_classes, EXPECTED_COUNT_2)
    expect_equal(model.base_classes[0]["name"], "Base1")
    expect_equal(model.base_classes[1]["name"], "Base2")


def test_fetch_models_handles_goid_as_decimal(data_models_ctx: TestContext) -> None:
    """Verify fetch_models handles goid_h128 Decimal conversion."""
    now = datetime.now(tz=UTC)

    _insert_models(
        data_models_ctx,
        [
            DataModelSeed(
                model_id="model_1",
                model_name="TestModel",
                module="test.module",
                rel_path="test/module.py",
                model_kind="dataclass",
                goid=123456789012345,
                created_at=now,
                repo=data_models_ctx.repo,
                commit=data_models_ctx.commit,
            )
        ],
    )

    result = DataModelsRepository(
        data_models_ctx.gateway, data_models_ctx.repo, data_models_ctx.commit
    ).list_models()
    model = result[0]
    expect_is_not_none(model.goid_h128)
    expect_is_instance(model.goid_h128, int)


def test_fetch_models_returns_multiple(data_models_ctx: TestContext) -> None:
    """Verify fetch_models returns multiple models."""
    now = datetime.now(tz=UTC)

    _insert_models(
        data_models_ctx,
        [
            DataModelSeed(
                model_id=f"model_{i}",
                model_name=f"Model{i}",
                module="test.module",
                rel_path="test/module.py",
                model_kind="dataclass",
                goid=1000 + i,
                created_at=now,
                repo=data_models_ctx.repo,
                commit=data_models_ctx.commit,
            )
            for i in range(EXPECTED_COUNT_3)
        ],
    )

    result = DataModelsRepository(
        data_models_ctx.gateway, data_models_ctx.repo, data_models_ctx.commit
    ).list_models()
    expect_length(result, EXPECTED_COUNT_3)


def test_fetch_fields_returns_empty_list(data_models_ctx: TestContext) -> None:
    """Verify fetch_fields returns empty list when no data."""
    result = DataModelsRepository(
        data_models_ctx.gateway, data_models_ctx.repo, data_models_ctx.commit
    ).list_fields()
    expect_is_instance(result, list)
    expect_length(result, 0)


def test_fetch_fields_parses_all_fields(data_models_ctx: TestContext) -> None:
    """Verify fetch_fields correctly parses all fields."""
    now = datetime.now(tz=UTC)

    _insert_fields(
        data_models_ctx,
        [
            DataModelFieldSeed(
                model_id="model_1",
                field_name="name",
                field_type="str",
                required=True,
                has_default=False,
                constraints_json={"max_length": 100},
                source="annotation",
                rel_path="test/module.py",
                lineno=EXPECTED_LINENO_10,
                created_at=now,
                repo=data_models_ctx.repo,
                commit=data_models_ctx.commit,
            )
        ],
    )

    result = DataModelsRepository(
        data_models_ctx.gateway, data_models_ctx.repo, data_models_ctx.commit
    ).list_fields()
    expect_length(result, 1)

    field = result[0]
    expect_equal(field.repo, data_models_ctx.repo)
    expect_equal(field.commit, data_models_ctx.commit)
    expect_equal(field.model_id, "model_1")
    expect_equal(field.name, "name")
    expect_equal(field.field_type, "str")
    expect_true(field.required)
    expect_false(field.has_default)
    expect_is_none(field.default_expr)
    expect_equal(field.constraints, {"max_length": 100})
    expect_equal(field.source, "annotation")
    expect_equal(field.rel_path, "test/module.py")
    expect_equal(field.lineno, EXPECTED_LINENO_10)


def test_fetch_fields_filters_by_model_ids(data_models_ctx: TestContext) -> None:
    """Verify fetch_fields filters by model_ids when provided."""
    now = datetime.now(tz=UTC)

    _insert_fields(
        data_models_ctx,
        [
            DataModelFieldSeed(
                model_id=model_id,
                field_name="field",
                field_type="str",
                required=True,
                has_default=False,
                rel_path="test.py",
                lineno=1,
                created_at=now,
                repo=data_models_ctx.repo,
                commit=data_models_ctx.commit,
            )
            for model_id in ["model_1", "model_2", "model_3"]
        ],
    )

    result = DataModelsRepository(
        data_models_ctx.gateway, data_models_ctx.repo, data_models_ctx.commit
    ).list_fields(model_ids=["model_1", "model_2"])
    expect_length(result, EXPECTED_COUNT_2)

    model_ids = {f.model_id for f in result}
    expect_in("model_1", model_ids)
    expect_in("model_2", model_ids)
    expect_not_in("model_3", model_ids)


def test_fetch_fields_without_model_ids_returns_all(
    data_models_ctx: TestContext,
) -> None:
    """Verify fetch_fields returns all fields when model_ids is None."""
    now = datetime.now(tz=UTC)

    _insert_fields(
        data_models_ctx,
        [
            DataModelFieldSeed(
                model_id=model_id,
                field_name="field",
                field_type="str",
                required=True,
                has_default=False,
                rel_path="test.py",
                lineno=1,
                created_at=now,
                repo=data_models_ctx.repo,
                commit=data_models_ctx.commit,
            )
            for model_id in ["model_1", "model_2"]
        ],
    )

    result = DataModelsRepository(
        data_models_ctx.gateway, data_models_ctx.repo, data_models_ctx.commit
    ).list_fields()
    expect_length(result, EXPECTED_COUNT_2)


def test_fetch_fields_parses_constraints_json(data_models_ctx: TestContext) -> None:
    """Verify fetch_fields correctly parses constraints_json."""
    now = datetime.now(tz=UTC)

    _insert_fields(
        data_models_ctx,
        [
            DataModelFieldSeed(
                model_id="model_1",
                field_name="email",
                field_type="str",
                required=True,
                has_default=False,
                constraints_json={"pattern": "^.+@.+$", "max_length": 255},
                source="annotation",
                rel_path="test.py",
                lineno=1,
                created_at=now,
                repo=data_models_ctx.repo,
                commit=data_models_ctx.commit,
            )
        ],
    )

    result = DataModelsRepository(
        data_models_ctx.gateway, data_models_ctx.repo, data_models_ctx.commit
    ).list_fields()
    field = result[0]
    expect_equal(field.constraints, {"pattern": "^.+@.+$", "max_length": 255})


def test_fetch_relationships_returns_empty_list(
    data_models_ctx: TestContext,
) -> None:
    """Verify fetch_relationships returns empty list when no data."""
    result = DataModelsRepository(
        data_models_ctx.gateway, data_models_ctx.repo, data_models_ctx.commit
    ).list_relationships()
    expect_is_instance(result, list)
    expect_length(result, 0)


def test_fetch_relationships_parses_all_fields(
    data_models_ctx: TestContext,
) -> None:
    """Verify fetch_relationships correctly parses all fields."""
    now = datetime.now(tz=UTC)

    _insert_relationships(
        data_models_ctx,
        [
            DataModelRelationshipSeed(
                source_model_id="model_1",
                target_model_id="model_2",
                target_module="target.module",
                target_model_name="TargetModel",
                field_name="items",
                relationship_kind="has_many",
                multiplicity="many",
                via="foreign_key",
                evidence_json={"confidence": 0.9},
                rel_path="test/module.py",
                lineno=EXPECTED_LINENO_25,
                created_at=now,
                repo=data_models_ctx.repo,
                commit=data_models_ctx.commit,
            )
        ],
    )

    result = DataModelsRepository(
        data_models_ctx.gateway, data_models_ctx.repo, data_models_ctx.commit
    ).list_relationships()
    expect_length(result, 1)

    rel = result[0]
    expect_equal(rel.repo, data_models_ctx.repo)
    expect_equal(rel.commit, data_models_ctx.commit)
    expect_equal(rel.source_model_id, "model_1")
    expect_equal(rel.target_model_id, "model_2")
    expect_equal(rel.target_module, "target.module")
    expect_equal(rel.target_model_name, "TargetModel")
    expect_equal(rel.field_name, "items")
    expect_equal(rel.relationship_kind, "has_many")
    expect_equal(rel.multiplicity, "many")
    expect_equal(rel.via, "foreign_key")
    expect_equal(rel.evidence, {"confidence": 0.9})
    expect_equal(rel.rel_path, "test/module.py")
    expect_equal(rel.lineno, EXPECTED_LINENO_25)


def test_fetch_relationships_filters_by_model_ids(
    data_models_ctx: TestContext,
) -> None:
    """Verify fetch_relationships filters by source model_ids when provided."""
    now = datetime.now(tz=UTC)

    _insert_relationships(
        data_models_ctx,
        [
            DataModelRelationshipSeed(
                source_model_id=src_model,
                target_model_id="target",
                target_module="target.mod",
                target_model_name="Target",
                field_name="ref",
                relationship_kind="has_one",
                multiplicity="one",
                evidence_json={},
                rel_path="test.py",
                lineno=1,
                created_at=now,
                repo=data_models_ctx.repo,
                commit=data_models_ctx.commit,
            )
            for src_model in ["model_1", "model_2", "model_3"]
        ],
    )

    result = DataModelsRepository(
        data_models_ctx.gateway, data_models_ctx.repo, data_models_ctx.commit
    ).list_relationships(model_ids=["model_1"])
    expect_length(result, 1)
    expect_equal(result[0].source_model_id, "model_1")


def test_fetch_relationships_handles_nullable_fields(
    data_models_ctx: TestContext,
) -> None:
    """Verify fetch_relationships handles null optional fields."""
    now = datetime.now(tz=UTC)

    _insert_relationships(
        data_models_ctx,
        [
            DataModelRelationshipSeed(
                source_model_id="model_1",
                target_model_id="model_2",
                target_module=None,
                target_model_name=None,
                field_name="field_ref",
                relationship_kind="references",
                multiplicity=None,
                via=None,
                evidence_json={},
                rel_path="test.py",
                lineno=None,
                created_at=now,
                repo=data_models_ctx.repo,
                commit=data_models_ctx.commit,
            )
        ],
    )

    result = DataModelsRepository(
        data_models_ctx.gateway, data_models_ctx.repo, data_models_ctx.commit
    ).list_relationships()
    rel = result[0]
    expect_is_none(rel.target_module)
    expect_is_none(rel.target_model_name)
    expect_equal(rel.field_name, "field_ref")
    expect_is_none(rel.multiplicity)
    expect_is_none(rel.via)
    expect_is_none(rel.lineno)


def test_fetch_models_normalized_returns_empty_list(
    data_models_ctx: TestContext,
) -> None:
    """Verify fetch_models_normalized returns empty list when no data."""
    result = DataModelsRepository(
        data_models_ctx.gateway, data_models_ctx.repo, data_models_ctx.commit
    ).list_models_normalized()
    expect_is_instance(result, list)
    expect_length(result, 0)


def test_fetch_models_normalized_joins_data(data_models_ctx: TestContext) -> None:
    """Verify fetch_models_normalized returns joined fields and relationships."""
    now = datetime.now(tz=UTC)

    _insert_models(
        data_models_ctx,
        [
            DataModelSeed(
                model_id="model_1",
                model_name="TestModel",
                module="test.module",
                rel_path="test/module.py",
                model_kind="dataclass",
                goid=EXPECTED_GOID,
                doc_short="Short doc",
                created_at=now,
                repo=data_models_ctx.repo,
                commit=data_models_ctx.commit,
            )
        ],
    )
    _insert_fields(
        data_models_ctx,
        [
            DataModelFieldSeed(
                model_id="model_1",
                field_name="name",
                field_type="str",
                required=True,
                has_default=False,
                constraints_json={},
                source="annotation",
                rel_path="test/module.py",
                lineno=EXPECTED_LINENO_10,
                created_at=now,
                repo=data_models_ctx.repo,
                commit=data_models_ctx.commit,
            )
        ],
    )
    _insert_relationships(
        data_models_ctx,
        [
            DataModelRelationshipSeed(
                source_model_id="model_1",
                target_model_id="model_2",
                target_module="target.mod",
                target_model_name="Target",
                field_name="ref",
                relationship_kind="has_one",
                multiplicity="one",
                evidence_json={},
                rel_path="test/module.py",
                lineno=20,
                created_at=now,
                repo=data_models_ctx.repo,
                commit=data_models_ctx.commit,
            )
        ],
    )

    result = DataModelsRepository(
        data_models_ctx.gateway, data_models_ctx.repo, data_models_ctx.commit
    ).list_models_normalized()
    expect_length(result, 1)

    model = result[0]
    expect_equal(model.model_id, "model_1")
    expect_equal(model.model_name, "TestModel")
    expect_length(model.fields, 1)
    expect_equal(model.fields[0].name, "name")
    expect_length(model.relationships, 1)
    expect_equal(model.relationships[0].target_model_id, "model_2")


def test_fetch_models_normalized_filters_by_model_ids(
    data_models_ctx: TestContext,
) -> None:
    """Verify fetch_models_normalized filters by model_ids."""
    now = datetime.now(tz=UTC)

    _insert_models(
        data_models_ctx,
        [
            DataModelSeed(
                model_id=f"model_{i}",
                model_name=f"Model{i}",
                module="test.module",
                rel_path="test/module.py",
                model_kind="dataclass",
                goid=1000 + i,
                created_at=now,
                repo=data_models_ctx.repo,
                commit=data_models_ctx.commit,
            )
            for i in range(EXPECTED_COUNT_3)
        ],
    )

    result = DataModelsRepository(
        data_models_ctx.gateway, data_models_ctx.repo, data_models_ctx.commit
    ).list_models_normalized(model_ids=["model_0", "model_1"])
    expect_length(result, EXPECTED_COUNT_2)

    model_ids = {m.model_id for m in result}
    expect_in("model_0", model_ids)
    expect_in("model_1", model_ids)
    expect_not_in("model_2", model_ids)
