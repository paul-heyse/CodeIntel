"""Dataset schema service helpers for build-first contract access."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from codeintel.build.schemas import (
    ContractResolutionMode,
    ContractResolutionSettings,
    extract_constraints_from_table_schema,
    get_contract_for_table_key,
    get_schema_service,
    iter_contracts,
)
from codeintel.build.schemas.inference_service import (
    inferability_inventory as _inferability_inventory,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from hamilton.driver import Driver

    from codeintel.build.hamilton.dag_catalog import DagCatalog
    from codeintel.build.schemas.constraints import ConstraintSet
    from codeintel.build.schemas.inference_service import InferabilityRecord
    from codeintel.core.schemas.contract_primitives import DatasetContract


DocsFilterMode = Literal["include", "only", "exclude"]
ReadOnlyFilterMode = Literal["include", "only", "exclude"]


def list_datasets(
    *,
    docs_view: DocsFilterMode = "include",
    read_only: ReadOnlyFilterMode = "include",
) -> list[DatasetContract]:
    """List dataset contracts with optional capability filters.

    Parameters
    ----------
    docs_view
        Filter for docs view datasets.
    read_only
        Filter for read-only datasets.

    Returns
    -------
    list[DatasetContract]
        Filtered dataset contracts.
    """
    settings = ContractResolutionSettings(mode=ContractResolutionMode.FULL)
    contracts = list(iter_contracts(settings=settings))
    return _apply_contract_filters(
        contracts,
        docs_view=docs_view,
        read_only=read_only,
    )


def describe_dataset(table_key: str) -> DatasetContract | None:
    """Return the contract for a specific dataset table key.

    Parameters
    ----------
    table_key
        Dataset table key.

    Returns
    -------
    DatasetContract | None
        Dataset contract or None when not found.
    """
    try:
        return get_contract_for_table_key(table_key)
    except KeyError:
        return None


def constraints_summary(table_key: str) -> ConstraintSet | None:
    """Return constraint summary for a dataset.

    Parameters
    ----------
    table_key
        Dataset table key.

    Returns
    -------
    ConstraintSet | None
        Constraint set when a schema is registered, otherwise None.
    """
    schema_service = get_schema_service()
    table_schema = schema_service.get_table_schema(table_key)
    if table_schema is None:
        return None
    return extract_constraints_from_table_schema(table_schema)


def flow(table_key: str, *, catalog: DagCatalog) -> tuple[list[str], list[str]]:
    """Return producer/consumer targets for a dataset table key.

    Parameters
    ----------
    table_key
        Dataset table key.
    catalog
        DAG catalog with IO surface details.

    Returns
    -------
    tuple[list[str], list[str]]
        Producer targets, consumer targets.
    """
    producers: set[str] = set()
    consumers: set[str] = set()
    for target_name, surface in catalog.io_surfaces.items():
        for write in surface.table_writes:
            if write.table_key == table_key:
                producers.add(target_name)
        for read in surface.reads:
            if read.table_key == table_key:
                consumers.add(target_name)
    return sorted(producers), sorted(consumers)


def inferability_inventory(
    driver: Driver,
    *,
    catalog: DagCatalog,
) -> list[InferabilityRecord]:
    """Return inferability inventory for DAG-produced outputs.

    Parameters
    ----------
    driver
        Hamilton driver instance.
    catalog
        DAG catalog with output metadata.

    Returns
    -------
    list[InferabilityRecord]
        Inferability records for the catalog outputs.
    """
    return list(_inferability_inventory(driver=driver, catalog=catalog))


def _apply_contract_filters(
    contracts: Iterable[DatasetContract],
    *,
    docs_view: DocsFilterMode,
    read_only: ReadOnlyFilterMode,
) -> list[DatasetContract]:
    filtered: list[DatasetContract] = []
    for contract in contracts:
        capabilities = contract.capabilities()
        is_docs_view = capabilities.get("docs_view", False)
        is_read_only = capabilities.get("read_only", False)
        if docs_view == "only" and not is_docs_view:
            continue
        if docs_view == "exclude" and is_docs_view:
            continue
        if read_only == "only" and not is_read_only:
            continue
        if read_only == "exclude" and is_read_only:
            continue
        filtered.append(contract)
    filtered.sort(key=lambda entry: entry.name)
    return filtered


__all__ = [
    "DocsFilterMode",
    "ReadOnlyFilterMode",
    "constraints_summary",
    "describe_dataset",
    "flow",
    "inferability_inventory",
    "list_datasets",
]
