"""Shared dataset contract validation helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from codeintel.core.schemas.contract_primitives import DatasetContract

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


@dataclass(frozen=True)
class ContractRegistry:
    """Contract registry indexed by name and table key."""

    by_name: Mapping[str, DatasetContract]
    by_table_key: Mapping[str, DatasetContract]


class DatasetRegistryLike(Protocol):
    """Protocol for dataset registries used in contract validation."""

    @property
    def by_name(self) -> Mapping[str, DatasetContract]:
        """Return dataset contracts keyed by name."""
        ...

    @property
    def by_table_key(self) -> Mapping[str, DatasetContract]:
        """Return dataset contracts keyed by table key."""
        ...


TableColumnsLookup = Callable[[str], Sequence[str] | None]


def build_contract_registry(contracts: Iterable[DatasetContract]) -> ContractRegistry:
    """Build a ContractRegistry from DatasetContract entries.

    Parameters
    ----------
    contracts
        Iterable of dataset contracts.

    Returns
    -------
    ContractRegistry
        Registry indexed by dataset name and table key.
    """
    by_name: dict[str, DatasetContract] = {}
    by_table: dict[str, DatasetContract] = {}
    for contract in contracts:
        by_name[contract.name] = contract
        by_table[contract.table_key] = contract
    return ContractRegistry(by_name=by_name, by_table_key=by_table)


def _binding_required_datasets(contracts: Mapping[str, DatasetContract]) -> frozenset[str]:
    return frozenset(
        contract.name
        for contract in contracts.values()
        if contract.json_schema_id is not None
        and contract.name not in {"data_model_fields", "data_model_relationships"}
    )


def _validate_row_bindings(
    registry: DatasetRegistryLike,
    contracts_by_name: Mapping[str, DatasetContract],
) -> list[str]:
    binding_required = _binding_required_datasets(contracts_by_name)
    binding_targets = {name for name in registry.by_name if name in binding_required}
    return [
        f"Dataset {name} missing row binding"
        for name, ds in registry.by_name.items()
        if name in binding_targets and not ds.is_view and ds.row_binding is None
    ]


def _validate_schema_alignment(
    registry: DatasetRegistryLike,
    contracts_by_table_key: Mapping[str, DatasetContract],
    *,
    include_views: bool,
) -> list[str]:
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
        for key, contract in contracts_by_table_key.items()
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


def _validate_schemas_match_contracts(
    contracts_by_table_key: Mapping[str, DatasetContract],
) -> list[str]:
    issues: list[str] = []
    for table_key, contract in contracts_by_table_key.items():
        if table_key.startswith("tmp_") or contract.is_view:
            continue
        if contract.schema is None:
            issues.append(f"DatasetContract missing schema for table {table_key}")
    return issues


def _validate_dependencies(
    registry: DatasetRegistryLike,
    contracts_by_name: Mapping[str, DatasetContract],
    *,
    include_views: bool,
) -> list[str]:
    issues: list[str] = []
    known = set(registry.by_name)
    graph = {
        name: contract.upstream_dependencies
        for name, contract in registry.by_name.items()
        if contract.upstream_dependencies
    }

    issues.extend(
        f"Dataset {name} depends on unknown dataset {dep}"
        for name, deps in graph.items()
        for dep in deps
        if dep not in known
    )

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


def _validate_table_columns(
    registry: DatasetRegistryLike,
    table_columns_lookup: TableColumnsLookup | None,
) -> list[str]:
    if table_columns_lookup is None:
        return []
    errors: list[str] = []
    for name, ds in registry.by_name.items():
        if ds.is_view or ds.schema is None:
            continue
        actual_columns = table_columns_lookup(ds.table_key)
        if actual_columns is None:
            continue
        expected_columns = [col.name for col in ds.schema.columns if col.name is not None]
        if actual_columns != expected_columns:
            errors.append(
                f"Table column mismatch for {name}: expected {expected_columns}, "
                f"found {actual_columns}"
            )
    return errors


def _validate_missing_json_schemas(
    registry: DatasetRegistryLike,
    contracts_by_name: Mapping[str, DatasetContract],
    *,
    include_views: bool,
) -> list[str]:
    json_schema_datasets = {c.name for c in contracts_by_name.values() if c.json_schema_id}
    missing_json_schema = [
        name
        for name in json_schema_datasets
        if name not in registry.by_name and (include_views or not contracts_by_name[name].is_view)
    ]
    if not missing_json_schema:
        return []
    missing_list = ", ".join(sorted(missing_json_schema))
    return [
        f"Datasets with configured schemas missing from registry: {missing_list}"
    ]


def collect_contract_issues(
    registry: DatasetRegistryLike,
    *,
    contracts_by_table_key: Mapping[str, DatasetContract],
    contracts_by_name: Mapping[str, DatasetContract],
    include_views: bool = True,
    table_columns_lookup: TableColumnsLookup | None = None,
) -> list[str]:
    """Collect contract inconsistencies for the provided registry.

    Parameters
    ----------
    registry
        Dataset registry to validate (metadata-backed or in-memory).
    contracts_by_table_key
        Mapping of table key to DatasetContract.
    contracts_by_name
        Mapping of dataset name to DatasetContract.
    include_views
        Whether to include views in validation checks.
    table_columns_lookup
        Optional callable returning actual column order for a table key.

    Returns
    -------
    list[str]
        Human-readable list of problems. Empty when the contract is healthy.
    """
    issues: list[str] = []
    issues.extend(_validate_row_bindings(registry, contracts_by_name))
    issues.extend(
        _validate_schema_alignment(
            registry,
            contracts_by_table_key,
            include_views=include_views,
        )
    )
    issues.extend(_validate_schemas_match_contracts(contracts_by_table_key))
    issues.extend(_validate_table_columns(registry, table_columns_lookup))
    issues.extend(
        _validate_dependencies(
            registry,
            contracts_by_name,
            include_views=include_views,
        )
    )
    issues.extend(
        _validate_missing_json_schemas(
            registry,
            contracts_by_name,
            include_views=include_views,
        )
    )
    return issues


def validate_contract_or_raise(
    registry: DatasetRegistryLike,
    *,
    contracts_by_table_key: Mapping[str, DatasetContract],
    contracts_by_name: Mapping[str, DatasetContract],
    include_views: bool = True,
    table_columns_lookup: TableColumnsLookup | None = None,
) -> None:
    """Validate dataset contract and raise on any issues.

    Parameters
    ----------
    registry
        Dataset registry to validate.
    contracts_by_table_key
        Mapping of table key to DatasetContract.
    contracts_by_name
        Mapping of dataset name to DatasetContract.
    include_views
        Whether to include views in validation checks.
    table_columns_lookup
        Optional callable returning actual column order for a table key.

    Raises
    ------
    ValueError
        When any contract problems are detected.
    """
    issues = collect_contract_issues(
        registry,
        contracts_by_table_key=contracts_by_table_key,
        contracts_by_name=contracts_by_name,
        include_views=include_views,
        table_columns_lookup=table_columns_lookup,
    )
    if issues:
        message = "Dataset contract validation failed:\n" + "\n".join(f"- {i}" for i in issues)
        raise ValueError(message)


__all__ = [
    "ContractRegistry",
    "DatasetRegistryLike",
    "build_contract_registry",
    "collect_contract_issues",
    "validate_contract_or_raise",
]
