"""Schema smoke tests for normalized data model export datasets."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.docs_export.validate_exports import validate_files


def test_data_model_field_schema_validates_fixture(tmp_path: Path) -> None:
    """Ensure a representative field record passes schema validation."""
    path = tmp_path / "data_model_fields.jsonl"
    path.write_text(
        (
            '{"repo":"r","commit":"c","model_id":"m1","field_name":"name","field_type":"str",'
            '"required":true,"has_default":false,"default_expr":null,'
            '"constraints_json":{},"source":"pydantic_field","rel_path":"pkg/mod.py",'
            '"lineno":10,"created_at":"2025-01-01T00:00:00Z"}'
        ),
        encoding="utf-8",
    )
    exit_code = validate_files("data_model_fields", [path])
    if exit_code != 0:
        pytest.fail("data_model_fields schema validation failed for fixture")


def test_data_model_relationship_schema_validates_fixture(tmp_path: Path) -> None:
    """Ensure a representative relationship record passes schema validation."""
    path = tmp_path / "data_model_relationships.jsonl"
    path.write_text(
        (
            '{"repo":"r","commit":"c","source_model_id":"m1","target_model_id":"m2",'
            '"target_module":"pkg.mod","target_model_name":"User","field_name":"user",'
            '"relationship_kind":"reference","multiplicity":"one","via":"annotation",'
            '"evidence_json":{"source":"hint"},"rel_path":"pkg/mod.py","lineno":20,'
            '"created_at":"2025-01-01T00:00:00Z"}'
        ),
        encoding="utf-8",
    )
    exit_code = validate_files("data_model_relationships", [path])
    if exit_code != 0:
        pytest.fail("data_model_relationships schema validation failed for fixture")
