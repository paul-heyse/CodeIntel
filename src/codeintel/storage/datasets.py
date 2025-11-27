"""Dataset metadata registry backed by DuckDB's metadata.datasets table."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import cast

from duckdb import DuckDBPyConnection

from codeintel.config.schemas.tables import TABLE_SCHEMAS, TableSchema
from codeintel.storage import rows as row_models

RowToTuple = Callable[[Mapping[str, object]], tuple[object, ...]]
RowDictType = type[object]


@dataclass(frozen=True)
class RowBinding:
    """Connect a DuckDB table key to a TypedDict row model and serializer."""

    row_type: RowDictType
    to_tuple: RowToTuple


@dataclass(frozen=True)
class Dataset:
    """Metadata describing a logical dataset backed by a DuckDB table or view."""

    table_key: str
    name: str
    schema: TableSchema | None
    row_binding: RowBinding | None = None
    jsonl_filename: str | None = None
    parquet_filename: str | None = None
    is_view: bool = False


@dataclass(frozen=True)
class DatasetRegistry:
    """In-memory view of metadata.datasets plus Python row bindings."""

    by_name: Mapping[str, Dataset]
    by_table_key: Mapping[str, Dataset]
    jsonl_datasets: Mapping[str, str]
    parquet_datasets: Mapping[str, str]

    @property
    def all_datasets(self) -> tuple[str, ...]:
        """Return all dataset names."""
        return tuple(self.by_name.keys())

    def resolve_table_key(self, name: str) -> str:
        """
        Resolve dataset name into fully qualified table or view key.

        Returns
        -------
        str
            Fully qualified table or view identifier.

        Raises
        ------
        KeyError
            If the dataset name is unknown.
        """
        ds = self.by_name.get(name)
        if ds is None:
            message = f"Unknown dataset: {name}"
            raise KeyError(message)
        return ds.table_key


def _row_binding(
    row_type: RowDictType,
    to_tuple: Callable[..., tuple[object, ...]],
) -> RowBinding:
    return RowBinding(row_type=row_type, to_tuple=cast("RowToTuple", to_tuple))


ROW_BINDINGS_BY_TABLE_KEY: dict[str, RowBinding] = {
    "analytics.coverage_lines": _row_binding(
        row_type=row_models.CoverageLineRow,
        to_tuple=row_models.coverage_line_to_tuple,
    ),
    "analytics.config_values": _row_binding(
        row_type=row_models.ConfigValueRow,
        to_tuple=row_models.config_value_to_tuple,
    ),
    "analytics.typedness": _row_binding(
        row_type=row_models.TypednessRow,
        to_tuple=row_models.typedness_row_to_tuple,
    ),
    "analytics.static_diagnostics": _row_binding(
        row_type=row_models.StaticDiagnosticRow,
        to_tuple=row_models.static_diagnostic_to_tuple,
    ),
    "analytics.function_validation": _row_binding(
        row_type=row_models.FunctionValidationRow,
        to_tuple=row_models.function_validation_row_to_tuple,
    ),
    "analytics.graph_validation": _row_binding(
        row_type=row_models.GraphValidationRow,
        to_tuple=row_models.graph_validation_row_to_tuple,
    ),
    "analytics.hotspots": _row_binding(
        row_type=row_models.HotspotRow,
        to_tuple=row_models.hotspot_row_to_tuple,
    ),
    "analytics.test_catalog": _row_binding(
        row_type=row_models.TestCatalogRowModel,
        to_tuple=row_models.test_catalog_row_to_tuple,
    ),
    "analytics.test_coverage_edges": _row_binding(
        row_type=row_models.TestCoverageEdgeRow,
        to_tuple=row_models.test_coverage_edge_to_tuple,
    ),
    "core.docstrings": _row_binding(
        row_type=row_models.DocstringRow,
        to_tuple=row_models.docstring_row_to_tuple,
    ),
    "core.goids": _row_binding(
        row_type=row_models.GoidRow,
        to_tuple=row_models.goid_to_tuple,
    ),
    "core.goid_crosswalk": _row_binding(
        row_type=row_models.GoidCrosswalkRow,
        to_tuple=row_models.goid_crosswalk_to_tuple,
    ),
    "graph.call_graph_nodes": _row_binding(
        row_type=row_models.CallGraphNodeRow,
        to_tuple=row_models.call_graph_node_to_tuple,
    ),
    "graph.call_graph_edges": _row_binding(
        row_type=row_models.CallGraphEdgeRow,
        to_tuple=row_models.call_graph_edge_to_tuple,
    ),
    "graph.import_graph_edges": _row_binding(
        row_type=row_models.ImportEdgeRow,
        to_tuple=row_models.import_edge_to_tuple,
    ),
    "graph.import_modules": _row_binding(
        row_type=row_models.ImportModuleRow,
        to_tuple=row_models.import_module_to_tuple,
    ),
    "graph.cfg_blocks": _row_binding(
        row_type=row_models.CFGBlockRow,
        to_tuple=row_models.cfg_block_to_tuple,
    ),
    "graph.cfg_edges": _row_binding(
        row_type=row_models.CFGEdgeRow,
        to_tuple=row_models.cfg_edge_to_tuple,
    ),
    "graph.dfg_edges": _row_binding(
        row_type=row_models.DFGEdgeRow,
        to_tuple=row_models.dfg_edge_to_tuple,
    ),
    "graph.symbol_use_edges": _row_binding(
        row_type=row_models.SymbolUseRow,
        to_tuple=row_models.symbol_use_to_tuple,
    ),
}


def load_dataset_registry(con: DuckDBPyConnection) -> DatasetRegistry:
    """
    Load dataset metadata from DuckDB's metadata.datasets table.

    Assumes metadata_bootstrap.bootstrap_metadata_datasets() has run on this database.

    Returns
    -------
    DatasetRegistry
        Registry containing dataset metadata mirrored from DuckDB.
    """
    rows = con.execute(
        """
        SELECT table_key, name, is_view, jsonl_filename, parquet_filename
        FROM metadata.datasets
        ORDER BY table_key
        """
    ).fetchall()

    by_name: dict[str, Dataset] = {}
    by_table: dict[str, Dataset] = {}
    jsonl_map: dict[str, str] = {}
    parquet_map: dict[str, str] = {}

    for table_key, name, is_view, jsonl_filename, parquet_filename in rows:
        schema: TableSchema | None = None if is_view else TABLE_SCHEMAS.get(table_key)
        row_binding = ROW_BINDINGS_BY_TABLE_KEY.get(table_key)
        ds = Dataset(
            table_key=table_key,
            name=name,
            schema=schema,
            row_binding=row_binding,
            jsonl_filename=jsonl_filename,
            parquet_filename=parquet_filename,
            is_view=bool(is_view),
        )
        by_name[name] = ds
        by_table[table_key] = ds
        if jsonl_filename:
            jsonl_map[table_key] = jsonl_filename
        if parquet_filename:
            parquet_map[table_key] = parquet_filename

    return DatasetRegistry(
        by_name=by_name,
        by_table_key=by_table,
        jsonl_datasets=jsonl_map,
        parquet_datasets=parquet_map,
    )


def dataset_for_name(registry: DatasetRegistry, name: str) -> Dataset:
    """
    Return dataset metadata for a dataset name.

    Returns
    -------
    Dataset
        Dataset metadata resolved from the registry.

    Raises
    ------
    KeyError
        If the dataset name is not present.
    """
    ds = registry.by_name.get(name)
    if ds is None:
        message = f"Unknown dataset name: {name}"
        raise KeyError(message)
    return ds


def dataset_for_table(registry: DatasetRegistry, table_key: str) -> Dataset:
    """
    Return dataset metadata for a fully qualified table or view key.

    Returns
    -------
    Dataset
        Dataset metadata resolved from the registry.

    Raises
    ------
    KeyError
        If the table key is not present.
    """
    ds = registry.by_table_key.get(table_key)
    if ds is None:
        message = f"Unknown dataset table key: {table_key}"
        raise KeyError(message)
    return ds
