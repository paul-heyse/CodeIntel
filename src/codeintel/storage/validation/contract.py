"""Validation helpers for the dataset contract."""

from __future__ import annotations

from pathlib import Path

from duckdb import DuckDBPyConnection

from codeintel.config.datasets import (
    DATASET_CONTRACTS,
    DATASET_CONTRACTS_BY_TABLE_KEY,
    JSON_SCHEMA_BY_DATASET_NAME,
)
from codeintel.storage.datasets.registry import (
    DatasetRegistry,
    build_dataset_dependency_graph,
    load_dataset_registry,
)
from codeintel.storage.metadata import NORMALIZED_MACROS

BINDING_REQUIRED_DATASETS: set[str] = {
    name
    for name, contract in DATASET_CONTRACTS.items()
    if contract.json_schema_id is not None
    and name not in {"data_model_fields", "data_model_relationships"}
}

__all__ = [
    "BINDING_REQUIRED_DATASETS",
    "collect_contract_issues",
    "validate_contract_or_raise",
]


def _schema_path(schema_id: str, *, base_dir: Path | None = None) -> Path:
    root = base_dir or Path("src/codeintel/config/schemas/export")
    return root / f"{schema_id}.json"


def _validate_schema_files(registry: DatasetRegistry, *, base_dir: Path | None = None) -> list[str]:
    return [
        f"Missing JSON Schema for dataset {name}: {_schema_path(ds.json_schema_id, base_dir=base_dir)}"
        for name, ds in registry.by_name.items()
        if ds.json_schema_id is not None
        and not _schema_path(ds.json_schema_id, base_dir=base_dir).exists()
    ]


def _validate_row_bindings(registry: DatasetRegistry) -> list[str]:
    binding_targets = {
        name for name, ds in registry.by_name.items() if name in BINDING_REQUIRED_DATASETS
    }
    return [
        f"Dataset {name} missing row binding"
        for name, ds in registry.by_name.items()
        if name in binding_targets and not ds.is_view and ds.row_binding is None
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
        for key, contract in DATASET_CONTRACTS_BY_TABLE_KEY.items()
        if key not in registry.by_table_key and not key.startswith("tmp_")
    ]
    registry_errors = (
        [f"Table schemas missing from metadata registry: {', '.join(sorted(missing_in_registry))}"]
        if missing_in_registry
        else []
    )
    return [*missing_schema, *unnamed_columns, *registry_errors]


def _validate_table_columns(con: DuckDBPyConnection, registry: DatasetRegistry) -> list[str]:
    errors: list[str] = []
    for name, ds in registry.by_name.items():
        if ds.is_view or ds.schema is None:
            continue
        if "dataset_rows_only" in ds.tags and ds.table_key in NORMALIZED_MACROS:
            errors.append(
                f"Dataset {name} is marked dataset_rows_only but has a normalized macro "
                f"registered: {NORMALIZED_MACROS[ds.table_key]}"
            )
        schema_name, table_name = ds.table_key.split(".", maxsplit=1)
        info = con.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = ? AND table_name = ?
            ORDER BY ordinal_position
            """,
            [schema_name, table_name],
        ).fetchall()
        actual_columns = [row[0] for row in info]
        expected_columns = [col.name for col in ds.schema.columns if col.name is not None]
        if actual_columns != expected_columns:
            errors.append(
                f"Table column mismatch for {name}: expected {expected_columns}, found {actual_columns}"
            )
    return errors


def _validate_schemas_match_contracts() -> list[str]:
    issues: list[str] = []
    for table_key, contract in DATASET_CONTRACTS_BY_TABLE_KEY.items():
        if table_key.startswith("tmp_") or contract.is_view:
            continue
        if contract.schema is None:
            issues.append(f"DatasetContract missing schema for table {table_key}")
    return issues


def _validate_dependencies(registry: DatasetRegistry) -> list[str]:
    issues: list[str] = []
    known = set(registry.by_name)
    graph = build_dataset_dependency_graph(registry)

    issues.extend(
        f"Dataset {name} depends on unknown dataset {dep}"
        for name, deps in graph.items()
        for dep in deps
        if dep not in known
    )

    for name, contract in DATASET_CONTRACTS.items():
        expected = set(contract.upstream_dependencies)
        actual = set(graph.get(name, ()))
        if expected != actual:
            issues.append(
                f"Dataset {name} dependency mismatch: expected {sorted(expected)}, "
                f"got {sorted(actual)}"
            )

    return issues


def collect_contract_issues(
    con: DuckDBPyConnection,
    *,
    schema_base_dir: Path | None = None,
) -> list[str]:
    """Collect contract inconsistencies for the active database.

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
    issues.extend(_validate_schemas_match_contracts())
    issues.extend(_validate_table_columns(con, registry))
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
    """Validate dataset contract and raise on any issues.

    Raises
    ------
    ValueError
        When any contract problems are detected.
    """
    issues = collect_contract_issues(con, schema_base_dir=schema_base_dir)
    if issues:
        message = "Dataset contract validation failed:\n" + "\n".join(f"- {i}" for i in issues)
        raise ValueError(message)
