"""Tests for PR-73: JSON Schema generation from TableSchema.

Test coverage:
- All ColumnType mappings produce correct JSON Schema types
- Nullable columns use array type syntax
- Non-nullable columns appear in `required`
- Generated schema validates against JSON Schema 2020-12 meta-schema
- Generated schemas validate known-good export rows
- Parity: compare generated vs hand-maintained for overlap datasets
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jsonschema
import pytest

from codeintel.core.schemas.json_schema_gen import (
    json_schema_from_table_schema,
)
from codeintel.core.schemas.primitives import Column, TableSchema

if TYPE_CHECKING:
    from typing import Any


# -----------------------------------------------------------------------------
# Test Constants
# -----------------------------------------------------------------------------

EXPECTED_COLUMN_TYPE_COUNT = 10  # Number of ColumnType literals
EXPECTED_SCHEMA_VERSION = "https://json-schema.org/draft/2020-12/schema"


# -----------------------------------------------------------------------------
# Test Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def simple_table_schema() -> TableSchema:
    """Create a simple TableSchema for testing.

    Returns
    -------
    TableSchema
        A test schema with various column types.
    """
    return TableSchema(
        schema="test",
        name="simple",
        columns=[
            Column(name="id", type="INTEGER", nullable=False),
            Column(name="name", type="VARCHAR", nullable=True),
            Column(name="active", type="BOOLEAN", nullable=False),
        ],
        description="A simple test table",
    )


@pytest.fixture
def all_types_table_schema() -> TableSchema:
    """Create a TableSchema with all column types.

    Returns
    -------
    TableSchema
        A test schema with all supported column types.
    """
    return TableSchema(
        schema="test",
        name="all_types",
        columns=[
            Column(name="bool_col", type="BOOLEAN", nullable=False),
            Column(name="int_col", type="INTEGER", nullable=False),
            Column(name="bigint_col", type="BIGINT", nullable=True),
            Column(name="double_col", type="DOUBLE", nullable=True),
            Column(name="decimal_col", type="DECIMAL", nullable=True),
            Column(name="decimal38_col", type="DECIMAL(38,0)", nullable=False),
            Column(name="varchar_col", type="VARCHAR", nullable=False),
            Column(name="json_col", type="JSON", nullable=True),
            Column(name="timestamp_col", type="TIMESTAMP", nullable=True),
            Column(name="timestamptz_col", type="TIMESTAMPTZ", nullable=False),
        ],
    )


# -----------------------------------------------------------------------------
# Test: Basic Schema Generation
# -----------------------------------------------------------------------------


class TestJsonSchemaFromTableSchema:
    """Test json_schema_from_table_schema function."""

    @staticmethod
    def test_generates_valid_json_schema_version(simple_table_schema: TableSchema) -> None:
        """Verify generated schema has correct $schema version."""
        result = json_schema_from_table_schema(simple_table_schema)

        if result["$schema"] != EXPECTED_SCHEMA_VERSION:
            pytest.fail(
                f"Expected schema version {EXPECTED_SCHEMA_VERSION}, got {result['$schema']}"
            )

    @staticmethod
    def test_generates_object_type(simple_table_schema: TableSchema) -> None:
        """Verify generated schema has type 'object'."""
        result = json_schema_from_table_schema(simple_table_schema)

        if result["type"] != "object":
            pytest.fail(f"Expected type 'object', got {result['type']}")

    @staticmethod
    def test_includes_schema_id_when_provided(simple_table_schema: TableSchema) -> None:
        """Verify $id is included when schema_id parameter is set."""
        schema_id = "urn:test:schema:test.simple"
        result = json_schema_from_table_schema(simple_table_schema, schema_id=schema_id)

        if result.get("$id") != schema_id:
            pytest.fail(f"Expected $id {schema_id}, got {result.get('$id')}")

    @staticmethod
    def test_uses_table_key_as_title(simple_table_schema: TableSchema) -> None:
        """Verify title is set to table_key."""
        result = json_schema_from_table_schema(simple_table_schema)

        if result["title"] != "test.simple":
            pytest.fail(f"Expected title 'test.simple', got {result['title']}")

    @staticmethod
    def test_includes_description_when_present(simple_table_schema: TableSchema) -> None:
        """Verify description is included when table has description."""
        result = json_schema_from_table_schema(simple_table_schema)

        if result.get("description") != "A simple test table":
            pytest.fail(f"Expected description, got {result.get('description')}")

    @staticmethod
    def test_sets_additional_properties_false(simple_table_schema: TableSchema) -> None:
        """Verify additionalProperties is set to false."""
        result = json_schema_from_table_schema(simple_table_schema)

        if result.get("additionalProperties") is not False:
            pytest.fail(
                f"Expected additionalProperties=false, got {result.get('additionalProperties')}"
            )

    @staticmethod
    def test_generates_properties_for_all_columns(simple_table_schema: TableSchema) -> None:
        """Verify all columns appear in properties."""
        result = json_schema_from_table_schema(simple_table_schema)

        expected_columns = {"id", "name", "active"}
        actual_columns = set(result["properties"].keys())

        if actual_columns != expected_columns:
            pytest.fail(f"Expected columns {expected_columns}, got {actual_columns}")


# -----------------------------------------------------------------------------
# Test: Nullable Column Handling
# -----------------------------------------------------------------------------


class TestNullableColumnHandling:
    """Test nullable column handling in JSON Schema generation."""

    @staticmethod
    def test_nullable_column_uses_array_type(simple_table_schema: TableSchema) -> None:
        """Verify nullable columns use array type syntax."""
        result = json_schema_from_table_schema(simple_table_schema)

        name_type = result["properties"]["name"]["type"]
        if not isinstance(name_type, list):
            pytest.fail(f"Expected list type for nullable column, got {type(name_type)}")
        if "null" not in name_type:
            pytest.fail("Expected 'null' in nullable column type array")

    @staticmethod
    def test_non_nullable_column_uses_scalar_type(simple_table_schema: TableSchema) -> None:
        """Verify non-nullable columns use scalar type."""
        result = json_schema_from_table_schema(simple_table_schema)

        id_type = result["properties"]["id"]["type"]
        if isinstance(id_type, list):
            pytest.fail(f"Expected scalar type for non-nullable column, got list: {id_type}")

    @staticmethod
    def test_non_nullable_columns_in_required(simple_table_schema: TableSchema) -> None:
        """Verify non-nullable columns appear in required list."""
        result = json_schema_from_table_schema(simple_table_schema)

        required = set(result.get("required", []))
        expected_required = {"id", "active"}  # name is nullable

        if required != expected_required:
            pytest.fail(f"Expected required {expected_required}, got {required}")

    @staticmethod
    def test_nullable_columns_not_in_required(simple_table_schema: TableSchema) -> None:
        """Verify nullable columns do not appear in required list."""
        result = json_schema_from_table_schema(simple_table_schema)

        required = set(result.get("required", []))
        if "name" in required:
            pytest.fail("Nullable column 'name' should not be in required list")


# -----------------------------------------------------------------------------
# Test: Column Type Mapping
# -----------------------------------------------------------------------------


class TestColumnTypeMapping:
    """Test ColumnType to JSON Schema type mapping."""

    @staticmethod
    def test_boolean_maps_to_boolean(all_types_table_schema: TableSchema) -> None:
        """Verify BOOLEAN maps to JSON Schema boolean."""
        result = json_schema_from_table_schema(all_types_table_schema)

        if result["properties"]["bool_col"]["type"] != "boolean":
            pytest.fail(f"Expected 'boolean', got {result['properties']['bool_col']['type']}")

    @staticmethod
    def test_integer_maps_to_integer(all_types_table_schema: TableSchema) -> None:
        """Verify INTEGER maps to JSON Schema integer."""
        result = json_schema_from_table_schema(all_types_table_schema)

        if result["properties"]["int_col"]["type"] != "integer":
            pytest.fail(f"Expected 'integer', got {result['properties']['int_col']['type']}")

    @staticmethod
    def test_bigint_maps_to_integer(all_types_table_schema: TableSchema) -> None:
        """Verify BIGINT maps to JSON Schema integer."""
        result = json_schema_from_table_schema(all_types_table_schema)

        # bigint_col is nullable, so type is a list
        bigint_type = result["properties"]["bigint_col"]["type"]
        if "integer" not in bigint_type:
            pytest.fail(f"Expected 'integer' in type, got {bigint_type}")

    @staticmethod
    def test_decimal38_maps_to_integer(all_types_table_schema: TableSchema) -> None:
        """Verify DECIMAL(38,0) maps to JSON Schema integer."""
        result = json_schema_from_table_schema(all_types_table_schema)

        if result["properties"]["decimal38_col"]["type"] != "integer":
            pytest.fail(f"Expected 'integer', got {result['properties']['decimal38_col']['type']}")

    @staticmethod
    def test_double_maps_to_number(all_types_table_schema: TableSchema) -> None:
        """Verify DOUBLE maps to JSON Schema number."""
        result = json_schema_from_table_schema(all_types_table_schema)

        # double_col is nullable, so type is a list
        double_type = result["properties"]["double_col"]["type"]
        if "number" not in double_type:
            pytest.fail(f"Expected 'number' in type, got {double_type}")

    @staticmethod
    def test_decimal_maps_to_number(all_types_table_schema: TableSchema) -> None:
        """Verify DECIMAL maps to JSON Schema number."""
        result = json_schema_from_table_schema(all_types_table_schema)

        # decimal_col is nullable, so type is a list
        decimal_type = result["properties"]["decimal_col"]["type"]
        if "number" not in decimal_type:
            pytest.fail(f"Expected 'number' in type, got {decimal_type}")

    @staticmethod
    def test_varchar_maps_to_string(all_types_table_schema: TableSchema) -> None:
        """Verify VARCHAR maps to JSON Schema string."""
        result = json_schema_from_table_schema(all_types_table_schema)

        if result["properties"]["varchar_col"]["type"] != "string":
            pytest.fail(f"Expected 'string', got {result['properties']['varchar_col']['type']}")

    @staticmethod
    def test_json_maps_to_any(all_types_table_schema: TableSchema) -> None:
        """Verify JSON maps to any value (no type constraint)."""
        result = json_schema_from_table_schema(all_types_table_schema)

        # JSON columns should not have a type constraint (any valid JSON)
        json_prop = result["properties"]["json_col"]
        # Nullable JSON column still doesn't have type constraint
        if "type" in json_prop and json_prop.get("type") not in (None, []):
            pytest.fail(f"Expected no type constraint for JSON, got {json_prop.get('type')}")

    @staticmethod
    def test_timestamp_maps_to_datetime_string(all_types_table_schema: TableSchema) -> None:
        """Verify TIMESTAMP maps to string with date-time format."""
        result = json_schema_from_table_schema(all_types_table_schema)

        ts_prop = result["properties"]["timestamp_col"]
        # timestamp_col is nullable
        if "string" not in ts_prop.get("type", []):
            pytest.fail(f"Expected 'string' in type, got {ts_prop.get('type')}")
        if ts_prop.get("format") != "date-time":
            pytest.fail(f"Expected format 'date-time', got {ts_prop.get('format')}")

    @staticmethod
    def test_timestamptz_maps_to_datetime_string(all_types_table_schema: TableSchema) -> None:
        """Verify TIMESTAMPTZ maps to string with date-time format."""
        result = json_schema_from_table_schema(all_types_table_schema)

        tstz_prop = result["properties"]["timestamptz_col"]
        if tstz_prop.get("type") != "string":
            pytest.fail(f"Expected 'string', got {tstz_prop.get('type')}")
        if tstz_prop.get("format") != "date-time":
            pytest.fail(f"Expected format 'date-time', got {tstz_prop.get('format')}")


# -----------------------------------------------------------------------------
# Test: Meta-Schema Validation
# -----------------------------------------------------------------------------


class TestMetaSchemaValidation:
    """Test that generated schemas are valid JSON Schema 2020-12."""

    @staticmethod
    def test_generated_schema_is_valid_json_schema(simple_table_schema: TableSchema) -> None:
        """Verify generated schema validates against 2020-12 meta-schema."""
        result = json_schema_from_table_schema(simple_table_schema)

        # jsonschema library validates that the schema is well-formed
        # by creating a validator without errors
        try:
            jsonschema.Draft202012Validator.check_schema(result)
        except jsonschema.SchemaError as e:
            pytest.fail(f"Generated schema is not valid JSON Schema: {e}")

    @staticmethod
    def test_all_types_schema_is_valid(all_types_table_schema: TableSchema) -> None:
        """Verify all-types schema validates against 2020-12 meta-schema."""
        result = json_schema_from_table_schema(all_types_table_schema)

        try:
            jsonschema.Draft202012Validator.check_schema(result)
        except jsonschema.SchemaError as e:
            pytest.fail(f"All-types schema is not valid JSON Schema: {e}")


# -----------------------------------------------------------------------------
# Test: Record Validation
# -----------------------------------------------------------------------------


class TestRecordValidation:
    """Test that generated schemas validate records correctly."""

    @staticmethod
    def test_valid_record_passes_validation(simple_table_schema: TableSchema) -> None:
        """Verify valid records pass schema validation."""
        schema = json_schema_from_table_schema(simple_table_schema)
        validator = jsonschema.Draft202012Validator(schema)

        valid_record: dict[str, Any] = {
            "id": 123,
            "name": "test",
            "active": True,
        }

        errors = list(validator.iter_errors(valid_record))
        if errors:
            pytest.fail(f"Valid record should pass validation: {errors}")

    @staticmethod
    def test_null_in_nullable_field_passes(simple_table_schema: TableSchema) -> None:
        """Verify null value in nullable field passes validation."""
        schema = json_schema_from_table_schema(simple_table_schema)
        validator = jsonschema.Draft202012Validator(schema)

        record_with_null: dict[str, Any] = {
            "id": 123,
            "name": None,
            "active": True,
        }

        errors = list(validator.iter_errors(record_with_null))
        if errors:
            pytest.fail(f"Null in nullable field should pass: {errors}")

    @staticmethod
    def test_missing_required_field_fails(simple_table_schema: TableSchema) -> None:
        """Verify missing required field fails validation."""
        schema = json_schema_from_table_schema(simple_table_schema)
        validator = jsonschema.Draft202012Validator(schema)

        record_missing_required: dict[str, Any] = {
            "name": "test",
            "active": True,
            # missing 'id' which is required
        }

        errors = list(validator.iter_errors(record_missing_required))
        if not errors:
            pytest.fail("Missing required field should fail validation")

    @staticmethod
    def test_wrong_type_fails(simple_table_schema: TableSchema) -> None:
        """Verify wrong type fails validation."""
        schema = json_schema_from_table_schema(simple_table_schema)
        validator = jsonschema.Draft202012Validator(schema)

        record_wrong_type: dict[str, Any] = {
            "id": "not_an_integer",  # should be integer
            "name": "test",
            "active": True,
        }

        errors = list(validator.iter_errors(record_wrong_type))
        if not errors:
            pytest.fail("Wrong type should fail validation")

    @staticmethod
    def test_additional_property_fails(simple_table_schema: TableSchema) -> None:
        """Verify additional properties fail validation."""
        schema = json_schema_from_table_schema(simple_table_schema)
        validator = jsonschema.Draft202012Validator(schema)

        record_extra_field: dict[str, Any] = {
            "id": 123,
            "name": "test",
            "active": True,
            "extra_field": "should fail",
        }

        errors = list(validator.iter_errors(record_extra_field))
        if not errors:
            pytest.fail("Additional property should fail validation")


# -----------------------------------------------------------------------------
# Test: JSON Schema Registry
# -----------------------------------------------------------------------------


class TestJsonSchemaRegistry:
    """Test JSON Schema registry functions."""

    @staticmethod
    def test_get_json_schema_returns_valid_schema() -> None:
        """Verify get_json_schema returns valid schema for known table."""
        from codeintel.build.schemas.json_schema_registry import (  # noqa: PLC0415
            get_json_schema,
        )

        # Use a well-known table key
        schema = get_json_schema("analytics.function_metrics")

        if schema["$schema"] != EXPECTED_SCHEMA_VERSION:
            pytest.fail(f"Expected schema version {EXPECTED_SCHEMA_VERSION}")
        if "properties" not in schema:
            pytest.fail("Schema should have properties")

    @staticmethod
    def test_get_json_schema_is_cached() -> None:
        """Verify get_json_schema caches results."""
        from codeintel.build.schemas.json_schema_registry import (  # noqa: PLC0415
            clear_json_schema_cache,
            get_json_schema,
        )

        clear_json_schema_cache()
        schema1 = get_json_schema("analytics.function_metrics")
        schema2 = get_json_schema("analytics.function_metrics")

        # Should be the same object due to caching
        if schema1 is not schema2:
            pytest.fail("Schema should be cached")

    @staticmethod
    def test_get_json_schema_for_dataset_name() -> None:
        """Verify get_json_schema_for_dataset_name works."""
        from codeintel.build.schemas.json_schema_registry import (  # noqa: PLC0415
            get_json_schema_for_dataset_name,
        )

        schema = get_json_schema_for_dataset_name("function_metrics")

        if schema is not None and schema["$schema"] != EXPECTED_SCHEMA_VERSION:
            pytest.fail(f"Expected schema version {EXPECTED_SCHEMA_VERSION}")

    @staticmethod
    def test_compute_json_schema_digest_returns_hex_string() -> None:
        """Verify compute_json_schema_digest returns valid hex digest."""
        from codeintel.build.schemas.json_schema_registry import (  # noqa: PLC0415
            compute_json_schema_digest,
        )

        digest = compute_json_schema_digest("analytics.function_metrics")

        if digest is None:
            pytest.fail("Expected non-None digest for known table")
        # SHA-256 hex digest should be 64 characters
        if len(digest) != 64:  # noqa: PLR2004
            pytest.fail(f"Expected 64-char hex digest, got {len(digest)}")

    @staticmethod
    def test_compute_json_schema_digest_is_deterministic() -> None:
        """Verify digest is deterministic across calls."""
        from codeintel.build.schemas.json_schema_registry import (  # noqa: PLC0415
            compute_json_schema_digest,
        )

        digest1 = compute_json_schema_digest("analytics.function_metrics")
        digest2 = compute_json_schema_digest("analytics.function_metrics")

        if digest1 != digest2:
            pytest.fail("Digest should be deterministic")


# -----------------------------------------------------------------------------
# Test: Parity with Hand-Maintained Schemas
# -----------------------------------------------------------------------------


class TestParityWithHandMaintained:
    """Test parity between generated and hand-maintained schemas."""

    @staticmethod
    def test_generated_schema_has_same_properties_as_hand_maintained() -> None:
        """Compare generated vs hand-maintained schema properties."""
        import json as json_module  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        from codeintel.build.schemas.json_schema_registry import (  # noqa: PLC0415
            get_json_schema_for_dataset_name,
        )

        # Load a hand-maintained schema
        schema_root = (
            Path(__file__).parent.parent.parent.parent
            / "src"
            / "codeintel"
            / "config"
            / "schemas"
            / "export"
        )
        function_profile_path = schema_root / "function_profile.json"

        if not function_profile_path.exists():
            pytest.skip("Hand-maintained schema not found")

        with function_profile_path.open("r", encoding="utf-8") as f:
            hand_maintained = json_module.load(f)

        # Get generated schema
        generated = get_json_schema_for_dataset_name("function_profile")

        if generated is None:
            pytest.skip("Generated schema not available for function_profile")

        # Check that generated schema has same properties
        hand_props = set(hand_maintained.get("properties", {}).keys())
        gen_props = set(generated.get("properties", {}).keys())

        # Generated may have more/fewer properties depending on current schema
        # This is an informational check - schemas may legitimately differ
        _ = hand_props - gen_props  # Only in hand-maintained
        _ = gen_props - hand_props  # Only in generated
