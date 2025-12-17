"""Reusable utilities for multi-table row materialization targets.

This module provides utilities for targets that produce multiple tables from a
single compute result. Each table is materialized via ``DuckDBRowsSaver`` and
the results are combined into a single ``TargetRunRecord``.

Targets using this pattern: function_metrics (3 tables), config_data_flow,
cfg_dfg_metrics, external_deps, data_models, and more.

Notes
-----
This template provides:
1. ``multi_table_record`` - Combine multiple materialization metadata dicts
2. ``create_row_extractor`` - Factory for creating row extraction functions

The row extraction factory is particularly useful for creating the extraction
functions that sit between a compute node and multiple DuckDBRowsSaver nodes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from hamilton.function_modifiers import tag

from codeintel.build.hamilton.native.materialization_records import (
    record_from_duckdb_materializations,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from codeintel.build.hamilton.env import BuildEnv
    from codeintel.build.hamilton.hooks.manifest_hook import TargetRunRecord
    from codeintel.build.targets import TargetGraph


@tag(node_type="materialize")
def multi_table_record(
    env: BuildEnv,
    graph: TargetGraph,
    target_name: str,
    materializations: dict[str, dict[str, Any]],
) -> TargetRunRecord:
    """Combine multiple table materializations into a single TargetRunRecord.

    This function aggregates materialization metadata from multiple DuckDB
    table writes into a single TargetRunRecord. Use this as the final
    materialize node when a target produces multiple tables.

    Parameters
    ----------
    env
        Build environment for manifest persistence and expected output refs.
    graph
        Target graph used to resolve the OutputTarget contract.
    target_name
        Target name for which the record is being produced.
    materializations
        Mapping of table_key to materialization metadata dict returned by
        the Hamilton saver nodes.

    Returns
    -------
    TargetRunRecord
        Combined execution record with aggregated status, row counts, and
        duration from all table writes.

    Examples
    --------
    Use with multiple DuckDBRowsSaver materialization outputs:

    >>> @tag(node_type="materialize")
    ... def t__function_metrics(
    ...     env: BuildEnv,
    ...     graph: TargetGraph,
    ...     m__analytics__function_metrics: dict[str, Any],
    ...     m__analytics__function_types: dict[str, Any],
    ... ) -> TargetRunRecord:
    ...     return multi_table_record(
    ...         env=env,
    ...         graph=graph,
    ...         target_name="function_metrics",
    ...         materializations={
    ...             "analytics.function_metrics": m__analytics__function_metrics,
    ...             "analytics.function_types": m__analytics__function_types,
    ...         },
    ...     )

    Notes
    -----
    The combined record status is determined as follows:
    - If any table failed: status="failed"
    - If all tables skipped: status="skipped"
    - Otherwise: status="succeeded"
    """
    return record_from_duckdb_materializations(
        env=env,
        graph=graph,
        target_name=target_name,
        materializations=materializations,
    )


@tag(node_type="materialize")
def record(
    env: BuildEnv,
    graph: TargetGraph,
    target_name: str,
    materializations: dict[str, dict[str, Any]],
) -> TargetRunRecord:
    """Alias for multi_table_record for subDAG composition.

    This function provides the same functionality as ``multi_table_record``
    but with a shorter name for use in subDAG wiring where the output node
    name matters.

    Parameters
    ----------
    env
        Build environment for manifest persistence and expected output refs.
    graph
        Target graph used to resolve the OutputTarget contract.
    target_name
        Target name for which the record is being produced.
    materializations
        Mapping of table_key to materialization metadata dict.

    Returns
    -------
    TargetRunRecord
        Combined execution record.
    """
    return multi_table_record(env, graph, target_name, materializations)


def create_row_extractor(
    result_attr: str,
    columns: tuple[str, ...] | None = None,
    row_converter: Callable[[Any], tuple[object, ...]] | None = None,
) -> Callable[[object], tuple[tuple[object, ...], ...] | None]:
    """Create a row extraction function from a Result dataclass attribute.

    This factory creates a function that extracts rows from a compute result's
    attribute and converts them to tuple format suitable for DuckDBRowsSaver.

    Parameters
    ----------
    result_attr
        Name of the attribute on the Result dataclass containing the rows.
    columns
        Column names in the desired order for tuple conversion. If provided
        and row_converter is None, rows are assumed to be mappings and will
        be converted using these column names.
    row_converter
        Optional function to convert each row to a tuple. If provided, this
        takes precedence over column-based conversion.

    Returns
    -------
    Callable[[object], tuple[tuple[object, ...], ...] | None]
        Row extraction function that takes a compute result (or None) and
        returns a tuple of row tuples (or None if input is None or attribute
        is empty).

    Examples
    --------
    Create an extractor for metrics rows using column order:

    >>> COLS = ("id", "name", "value")
    >>> extract_metrics = create_row_extractor("metrics_rows", columns=COLS)
    >>> # Assuming result.metrics_rows = [{"id": 1, "name": "a", "value": 10}]
    >>> rows = extract_metrics(result)  # Returns ((1, "a", 10),)

    Create an extractor with a custom converter:

    >>> extract_custom = create_row_extractor(
    ...     "items",
    ...     row_converter=lambda item: (item.id, item.name),
    ... )
    """

    def extractor(compute_result: object) -> tuple[tuple[object, ...], ...] | None:
        """Extract rows from compute result.

        Parameters
        ----------
        compute_result
            Result from compute node, or None if skipped.

        Returns
        -------
        tuple[tuple[object, ...], ...] | None
            Extracted rows as tuples, or None if compute_result is None
            or the attribute is empty/None.
        """
        if compute_result is None:
            return None

        rows = getattr(compute_result, result_attr, None)
        if rows is None:
            return None

        # Handle empty sequences
        if isinstance(rows, Sequence) and len(rows) == 0:
            return None

        if row_converter is not None:
            return tuple(row_converter(row) for row in rows)

        if columns is not None:
            # Assume rows are mappings (dict or TypedDict)
            return tuple(_row_to_tuple(row, columns) for row in rows)

        # Assume rows are already tuples
        return tuple(rows)

    return extractor


def _row_to_tuple(row: Mapping[str, object], columns: tuple[str, ...]) -> tuple[object, ...]:
    """Convert a mapping row to a tuple in column order.

    Parameters
    ----------
    row
        Row mapping from column name to value.
    columns
        Column names in the desired order.

    Returns
    -------
    tuple[object, ...]
        Values in column order.
    """
    return tuple(row.get(col) for col in columns)


__all__ = [
    "create_row_extractor",
    "multi_table_record",
    "record",
]
