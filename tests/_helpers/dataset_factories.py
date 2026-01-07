"""Factories for dataset registries used in tests."""

from __future__ import annotations

from pathlib import Path

from codeintel.storage.contracts.provider import get_contract_for_table_key
from codeintel.storage.datasets.registry import DatasetRegistry
from tests._helpers.schemas import ensure_storage_contract_catalog


def sample_dataset_registry(tmp_path: Path | None = None) -> DatasetRegistry:
    """
    Build a minimal DatasetRegistry for catalog/scaffold tests.

    Parameters
    ----------
    tmp_path
        Optional base path for file outputs; defaults to cwd.

    Returns
    -------
    DatasetRegistry
        Registry containing a single ast_nodes dataset with file bindings.
    """
    base_path = tmp_path if tmp_path is not None else Path.cwd()
    table_key = "core.ast_nodes"
    ensure_storage_contract_catalog()
    contract = get_contract_for_table_key(table_key)
    jsonl_name = contract.jsonl_filename or f"{contract.name}.jsonl"
    parquet_name = contract.parquet_filename or f"{contract.name}.parquet"
    return DatasetRegistry(
        by_name={contract.name: contract},
        by_table_key={table_key: contract},
        jsonl_datasets={table_key: str(base_path / jsonl_name)},
        parquet_datasets={table_key: str(base_path / parquet_name)},
        dataset_root_dir=base_path,
    )


__all__ = [
    "sample_dataset_registry",
]
