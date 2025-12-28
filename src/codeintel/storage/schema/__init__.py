"""Schema management utilities for DuckDB and JSON Schema.

This package intentionally avoids eager imports to prevent circular import
issues during storage bootstrap. Import from submodules directly for most
internal call sites, or rely on the lazy exports defined here.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.storage.schema.arrow_schema import arrow_schema_for_table_key, arrow_schema_hash
    from codeintel.storage.schema.ddl import (
        SCHEMAS,
        apply_all_schemas,
        assert_schema_alignment,
        create_schemas,
        ensure_schemas_preserve,
    )
    from codeintel.storage.schema.json_schema import (
        build_validator,
        generate_export_schemas,
        json_schema_from_typeddict,
        validate_row_with_schema,
    )

    _TYPE_CHECKING_EXPORTS = (
        arrow_schema_for_table_key,
        arrow_schema_hash,
        SCHEMAS,
        apply_all_schemas,
        assert_schema_alignment,
        create_schemas,
        ensure_schemas_preserve,
        build_validator,
        generate_export_schemas,
        json_schema_from_typeddict,
        validate_row_with_schema,
    )

_EXPORTS: dict[str, tuple[str, str]] = {
    "arrow_schema_for_table_key": (
        "codeintel.storage.schema.arrow_schema",
        "arrow_schema_for_table_key",
    ),
    "arrow_schema_hash": (
        "codeintel.storage.schema.arrow_schema",
        "arrow_schema_hash",
    ),
    "SCHEMAS": ("codeintel.storage.schema.ddl", "SCHEMAS"),
    "apply_all_schemas": ("codeintel.storage.schema.ddl", "apply_all_schemas"),
    "assert_schema_alignment": ("codeintel.storage.schema.ddl", "assert_schema_alignment"),
    "create_schemas": ("codeintel.storage.schema.ddl", "create_schemas"),
    "ensure_schemas_preserve": ("codeintel.storage.schema.ddl", "ensure_schemas_preserve"),
    "build_validator": ("codeintel.storage.schema.json_schema", "build_validator"),
    "generate_export_schemas": ("codeintel.storage.schema.json_schema", "generate_export_schemas"),
    "json_schema_from_typeddict": (
        "codeintel.storage.schema.json_schema",
        "json_schema_from_typeddict",
    ),
    "validate_row_with_schema": (
        "codeintel.storage.schema.json_schema",
        "validate_row_with_schema",
    ),
}


def __getattr__(name: str) -> object:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attr_name = target
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))


__all__ = (
    "SCHEMAS",
    "apply_all_schemas",
    "arrow_schema_for_table_key",
    "arrow_schema_hash",
    "assert_schema_alignment",
    "build_validator",
    "create_schemas",
    "ensure_schemas_preserve",
    "generate_export_schemas",
    "json_schema_from_typeddict",
    "validate_row_with_schema",
)
