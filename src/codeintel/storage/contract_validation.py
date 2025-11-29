"""Validation helpers for the dataset contract."""

from __future__ import annotations

from pathlib import Path

from duckdb import DuckDBPyConnection

from codeintel.config.schemas.tables import TABLE_SCHEMAS
from codeintel.storage.datasets import (
    DEPENDENCIES_BY_DATASET_NAME,
    JSON_SCHEMA_BY_DATASET_NAME,
    DatasetRegistry,
    build_dataset_dependency_graph,
    load_dataset_registry,
)


def _schema_path(schema_id: str, *, base_dir: Path | None = None) -> Path:
    root = base_dir or Path("config/config/schemas/export")
    return root / f"{schema_id}.json"


def _validate_schema_files(registry: DatasetRegistry, *, base_dir: Path | None = None) -> list[str]:
    return [
        f"Missing JSON Schema for dataset {name}: {_schema_path(ds.json_schema_id, base_dir=base_dir)}"
        for name, ds in registry.by_name.items()
        if ds.json_schema_id is not None
        and not _schema_path(ds.json_schema_id, base_dir=base_dir).exists()
    ]


def _validate_row_bindings(registry: DatasetRegistry) -> list[str]:
    return [
        f"Dataset {name} missing row binding"
        for name, ds in registry.by_name.items()
        if not ds.is_view and ds.row_binding is None
    ]


def _validate_schema_alignment(registry: DatasetRegistry) -> list[str]:
    missing_schema = [
        f"Dataset {name} missing TableSchema definition"
        for name, ds in registry.by_name.items()
        if not ds.is_view and ds.schema is None
    ]
    unnamed_columns = [
        f"Dataset {name} has unnamed column in schema"
        for name, ds in registry.by_name.items()
        if not ds.is_view
        for column in (ds.schema.columns if ds.schema is not None else ())
        if column.name is None
    ]
    missing_in_registry = [
        key
        for key in TABLE_SCHEMAS
        if key not in registry.by_table_key and not key.startswith("tmp_")
    ]
    registry_errors = (
        [
            "Table schemas missing from metadata registry: "
            f"{', '.join(sorted(missing_in_registry))}"
        ]
        if missing_in_registry
        else []
    )
    return [*missing_schema, *unnamed_columns, *registry_errors]


def _validate_dependencies(registry: DatasetRegistry) -> list[str]:
    known = set(registry.by_name)
    graph = build_dataset_dependency_graph(registry)
    errors = [
        f"Dataset {name} depends on unknown dataset {dep}"
        for name, deps in graph.items()
        for dep in deps
        if dep not in known
    ]
    errors.extend(
        f"Dependency mapping references unknown dataset {dep_name}"
        for dep_name in DEPENDENCIES_BY_DATASET_NAME
        if dep_name not in known
    )
    return errors


def collect_contract_issues(
    con: DuckDBPyConnection,
    *,
    schema_base_dir: Path | None = None,
) -> list[str]:
    """
    Collect contract inconsistencies for the active database.

    Returns
    -------
    list[str]
        Human-readable list of problems. Empty when the contract is healthy.
    """
    registry = load_dataset_registry(con)
    issues: list[str] = []
    issues.extend(_validate_schema_files(registry, base_dir=schema_base_dir))
    issues.extend(_validate_row_bindings(registry))
    issues.extend(_validate_schema_alignment(registry))
    issues.extend(_validate_dependencies(registry))
    missing_json_schema = [
        name for name in JSON_SCHEMA_BY_DATASET_NAME if name not in registry.by_name
    ]
    if missing_json_schema:
        issues.append(
            "Datasets with configured schemas missing from registry: "
            f"{', '.join(sorted(missing_json_schema))}"
        )
    return issues


def validate_contract_or_raise(
    con: DuckDBPyConnection,
    *,
    schema_base_dir: Path | None = None,
) -> None:
    """
    Validate dataset contract and raise on any issues.

    Raises
    ------
    ValueError
        When any contract problems are detected.
    """
    issues = collect_contract_issues(con, schema_base_dir=schema_base_dir)
    if issues:
        message = "Dataset contract validation failed:\n" + "\n".join(f"- {i}" for i in issues)
        raise ValueError(message)
