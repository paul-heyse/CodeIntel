"""Shared columnar materialization helpers for Hamilton savers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

import polars as pl
import pyarrow as pa

from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.schemas import get_schema_provider
from codeintel.build.schemas.observations import (
    SchemaHints,
    merge_table_schema_hints,
)
from codeintel.core.columnar.schema_alignment import (
    align_reader_to_contract,
    extras_policy_from_schema,
)
from codeintel.core.hamilton import tags as hamilton_tags
from codeintel.core.schemas.contracts import (
    table_schema_from_arrow_schema,
    table_schema_from_polars_lazyframe,
)
from codeintel.core.schemas.primitives import TableSchema
from codeintel.storage.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.storage.duckdb_types import DuckDBRelation

if TYPE_CHECKING:
    from pyarrow import RecordBatchReader

    type ColumnarInput = RecordBatchReader | pa.Table | pl.LazyFrame | DuckDBRelation
else:
    type ColumnarInput = object

_SCHEMA_OUTPUT_TAG = "hamilton.internal.schema_output"


def schema_tag_sets_for_table(
    *,
    catalog: DagCatalog,
    table_key: str,
) -> tuple[Mapping[str, object], ...]:
    """Collect schema-related tag sets for a table key.

    Returns
    -------
    tuple[Mapping[str, object], ...]
        Tag sets associated with schema-related nodes for the table key.
    """
    tag_sets: list[Mapping[str, object]] = []
    output = catalog.table_outputs.get(table_key)
    if output is not None:
        tag_sets.append(output.tags)
        tag_sets.extend(_schema_output_tag_sets(catalog=catalog, saver_node=output.saver_node))
    tag_sets.extend(
        node.tags
        for node in catalog.nodes.values()
        if node.tags.get(hamilton_tags.TAG_TABLE_KEY) == table_key
    )
    return tuple(tag_sets)


def declared_schema_hint(table_key: str) -> TableSchema | None:
    """Return declared schema hints for a table key when available.

    Returns
    -------
    TableSchema | None
        Declared schema when available; otherwise None.
    """
    try:
        provider = get_schema_provider()
    except RuntimeError:
        return None
    return provider.get_table_schema(table_key)


def table_schema_for_data(
    *,
    table_key: str,
    data: ColumnarInput,
    declared_schema: TableSchema | None,
    schema_hints: SchemaHints | None,
) -> TableSchema:
    """Infer a table schema from columnar data and merge hints.

    Returns
    -------
    TableSchema
        Inferred table schema with declared and observed hints merged.

    Raises
    ------
    TypeError
        If the input data type is unsupported.
    """
    if isinstance(data, pl.LazyFrame):
        inferred = table_schema_from_polars_lazyframe(frame=data, table_key=table_key)
        return merge_table_schema_hints(
            inferred,
            declared_schema,
            schema_hints=schema_hints,
        )
    if isinstance(data, pa.RecordBatchReader):
        arrow_reader = cast("RecordBatchReader", data)
        inferred = table_schema_from_arrow_schema(
            arrow_schema=arrow_reader.schema,
            table_key=table_key,
        )
        return merge_table_schema_hints(
            inferred,
            declared_schema,
            schema_hints=schema_hints,
        )
    if isinstance(data, pa.Table):
        inferred = table_schema_from_arrow_schema(
            arrow_schema=data.schema,
            table_key=table_key,
        )
        return merge_table_schema_hints(
            inferred,
            declared_schema,
            schema_hints=schema_hints,
        )
    if isinstance(data, DuckDBRelation):
        reader = _relation_arrow_reader(data)
        inferred = table_schema_from_arrow_schema(
            arrow_schema=reader.schema,
            table_key=table_key,
        )
        return merge_table_schema_hints(
            inferred,
            declared_schema,
            schema_hints=schema_hints,
        )
    msg = f"Unsupported columnar input type: {type(data).__name__}"
    raise TypeError(msg)


def arrow_schema_for_data(*, data: ColumnarInput) -> pa.Schema:
    """Return an Arrow schema for the given columnar input.

    Returns
    -------
    pyarrow.Schema
        Arrow schema derived from the input data.

    Raises
    ------
    TypeError
        If the input data type is unsupported.
    """
    if isinstance(data, pl.LazyFrame):
        return data.collect_schema().to_arrow()
    if isinstance(data, pa.RecordBatchReader):
        return data.schema
    if isinstance(data, pa.Table):
        return data.schema
    if isinstance(data, DuckDBRelation):
        return _relation_arrow_reader(data).schema
    msg = f"Unsupported columnar input type: {type(data).__name__}"
    raise TypeError(msg)


def align_reader_to_contract_schema(
    reader: RecordBatchReader,
    *,
    contract_schema: pa.Schema | None,
) -> RecordBatchReader:
    """Align a RecordBatchReader to an Arrow contract schema.

    Returns
    -------
    pyarrow.RecordBatchReader
        Reader aligned to the contract schema.
    """
    if contract_schema is None:
        return reader
    return align_reader_to_contract(
        reader,
        contract_schema,
        extras_policy=extras_policy_from_schema(contract_schema),
    )


def _relation_arrow_reader(relation: DuckDBRelation) -> RecordBatchReader:
    fetcher = getattr(relation, "fetch_arrow_reader", None)
    if callable(fetcher):
        try:
            return fetcher(DEFAULT_ARROW_BATCH_SIZE)
        except TypeError:
            return fetcher()
    return relation.fetch_record_batch(DEFAULT_ARROW_BATCH_SIZE)


def _schema_output_tag_sets(
    *,
    catalog: DagCatalog,
    saver_node: str,
) -> list[Mapping[str, object]]:
    node = catalog.nodes.get(saver_node)
    if node is None:
        return []
    visited: set[str] = set()
    stack = list(node.deps)
    tag_sets: list[Mapping[str, object]] = []
    while stack:
        node_name = stack.pop()
        if node_name in visited:
            continue
        visited.add(node_name)
        candidate = catalog.nodes.get(node_name)
        if candidate is None:
            continue
        if _SCHEMA_OUTPUT_TAG in candidate.tags:
            tag_sets.append(candidate.tags)
        stack.extend(candidate.deps)
    return tag_sets


__all__ = [
    "align_reader_to_contract_schema",
    "arrow_schema_for_data",
    "declared_schema_hint",
    "schema_tag_sets_for_table",
    "table_schema_for_data",
]
