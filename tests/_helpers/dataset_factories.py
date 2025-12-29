"""Factories for dataset registries used in tests."""

from __future__ import annotations

from pathlib import Path

from codeintel.core.schemas.contract_primitives import DatasetContract
from codeintel.storage.contracts.schema_provider import get_schema_provider
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

    Raises
    ------
    ValueError
        If the schema registry is missing the requested table schema.
    """
    base_path = tmp_path if tmp_path is not None else Path.cwd()
    table_key = "core.ast_nodes"
    ensure_storage_contract_catalog()
    provider = get_schema_provider()
    table_schema = provider.get_table_schema(table_key)
    if table_schema is None:
        msg = f"Missing schema for {table_key}"
        raise ValueError(msg)
    contract = DatasetContract(
        table_key=table_key,
        name="ast_nodes",
        schema=table_schema,
        json_schema_id="ast_nodes",
        jsonl_filename="ast_nodes.jsonl",
        parquet_filename="ast_nodes.parquet",
        owner="team-data",
        freshness_sla="daily",
        retention_policy="90d",
        schema_version="1",
        stable_id="ast_nodes",
        upstream_dependencies=("core.modules",),
        validation_profile="strict",
    )
    return DatasetRegistry(
        by_name={"ast_nodes": contract},
        by_table_key={table_key: contract},
        jsonl_datasets={table_key: str(base_path / "ast_nodes.jsonl")},
        parquet_datasets={table_key: str(base_path / "ast_nodes.parquet")},
    )


__all__ = [
    "sample_dataset_registry",
]
