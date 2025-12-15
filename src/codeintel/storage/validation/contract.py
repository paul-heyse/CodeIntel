"""Validation helpers for the dataset contract."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from codeintel.build.schemas import iter_contracts, iter_contracts_by_table_key
from codeintel.storage.datasets.registry import (
    build_dataset_dependency_graph,
    load_dataset_registry,
)

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

    from codeintel.core.schemas.contract_primitives import DatasetContract
    from codeintel.storage.datasets.registry import (
        DatasetRegistry,
    )


@lru_cache(maxsize=1)
def _get_contracts_by_name() -> dict[str, DatasetContract]:
    """Build name-to-contract mapping from contract provider.

    Returns
    -------
    dict[str, DatasetContract]
        Mapping from contract name to DatasetContract.
    """
    return {c.name: c for c in iter_contracts()}


@lru_cache(maxsize=1)
def _get_contracts_by_table_key() -> dict[str, DatasetContract]:
    """Build table_key-to-contract mapping from contract provider.

    Returns
    -------
    dict[str, DatasetContract]
        Mapping from table_key to DatasetContract.
    """
    return dict(iter_contracts_by_table_key())


@lru_cache(maxsize=1)
def get_binding_required_datasets() -> frozenset[str]:
    """Return set of dataset names that require row bindings.

    This function is lazily evaluated and cached to avoid circular imports
    at module load time.

    Returns
    -------
    frozenset[str]
        Dataset names with JSON schema IDs (excluding data model tables).
    """
    return frozenset(
        contract.name
        for contract in iter_contracts()
        if contract.json_schema_id is not None
        and contract.name not in {"data_model_fields", "data_model_relationships"}
    )


__all__ = [
    "collect_contract_issues",
    "get_binding_required_datasets",
    "validate_contract_or_raise",
]


def _validate_row_bindings(registry: DatasetRegistry) -> list[str]:
    binding_required = get_binding_required_datasets()
    binding_targets = {name for name, ds in registry.by_name.items() if name in binding_required}
    return [
        f"Dataset {name} missing row binding"
        for name, ds in registry.by_name.items()
        if name in binding_targets and not ds.is_view and ds.row_binding is None
    ]


def _validate_schema_alignment(registry: DatasetRegistry, *, include_views: bool) -> list[str]:
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
    contracts_by_table = _get_contracts_by_table_key()
    missing_in_registry = [
        key
        for key, contract in contracts_by_table.items()
        if key not in registry.by_table_key
        and not key.startswith("tmp_")
        and (include_views or not contract.is_view)
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
    contracts_by_table = _get_contracts_by_table_key()
    for table_key, contract in contracts_by_table.items():
        if table_key.startswith("tmp_") or contract.is_view:
            continue
        if contract.schema is None:
            issues.append(f"DatasetContract missing schema for table {table_key}")
    return issues


def _validate_dependencies(registry: DatasetRegistry, *, include_views: bool) -> list[str]:
    issues: list[str] = []
    known = set(registry.by_name)
    graph = build_dataset_dependency_graph(registry)

    issues.extend(
        f"Dataset {name} depends on unknown dataset {dep}"
        for name, deps in graph.items()
        for dep in deps
        if dep not in known
    )

    contracts_by_name = _get_contracts_by_name()
    for name, contract in contracts_by_name.items():
        if contract.is_view and not include_views:
            continue
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
    include_views: bool = True,
) -> list[str]:
    """Collect contract inconsistencies for the active database.

    JSON schema validation is performed using generated schemas from TableSchema
    definitions, not file-based schemas (removed in PR-73).

    Returns
    -------
    list[str]
        Human-readable list of problems. Empty when the contract is healthy.
    """
    registry = load_dataset_registry(con)
    issues: list[str] = []
    issues.extend(_validate_row_bindings(registry))
    issues.extend(_validate_schema_alignment(registry, include_views=include_views))
    issues.extend(_validate_schemas_match_contracts())
    issues.extend(_validate_table_columns(con, registry))
    issues.extend(_validate_dependencies(registry, include_views=include_views))

    # Check for datasets with JSON schemas missing from registry
    json_schema_datasets = {c.name for c in iter_contracts() if c.json_schema_id is not None}
    contracts_by_name = _get_contracts_by_name()
    missing_json_schema = [
        name
        for name in json_schema_datasets
        if name not in registry.by_name
        and (include_views or not contracts_by_name[name].is_view)
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
    include_views: bool = True,
) -> None:
    """Validate dataset contract and raise on any issues.

    Raises
    ------
    ValueError
        When any contract problems are detected.
    """
    issues = collect_contract_issues(
        con,
        include_views=include_views,
    )
    if issues:
        message = "Dataset contract validation failed:\n" + "\n".join(f"- {i}" for i in issues)
        raise ValueError(message)
