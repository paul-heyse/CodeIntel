"""Anchor map helpers for CPG assembly."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pyarrow as pa
import pyarrow.compute as pc

from codeintel.build.graphs.assembly import rename_table_columns, select_table_columns, table_rows
from codeintel.build.hamilton.native.graphs.cpg2.ids import cpg_node_id, cpg_source_pk_json

IDENTITY_KEY_REGISTRY: dict[str, tuple[str, ...]] = {
    "core.syntax_nodes": ("repo", "commit", "rel_path", "producer", "node_id"),
    "core.ast_nodes": ("hash",),
    "core.scip_symbol_information": ("repo", "commit", "symbol"),
    "core.goids": ("goid_h128",),
    "graph.cfg_blocks": ("function_goid_h128", "block_idx"),
    "graph.import_modules": ("repo", "commit", "module"),
    "core.ts_tokens": ("repo", "commit", "rel_path", "language", "token_id"),
    "core.ts_trivia": ("repo", "commit", "rel_path", "language", "trivia_id"),
    "core.py_sym_scopes": ("repo", "commit", "rel_path", "scope_id"),
    "core.py_sym_bindings": ("repo", "commit", "rel_path", "binding_id"),
    "core.py_bc_code_units": ("repo", "commit", "rel_path", "code_unit_id"),
    "core.py_bc_instructions": ("repo", "commit", "rel_path", "code_unit_id", "instr_id"),
    "core.py_bc_blocks": ("repo", "commit", "rel_path", "block_id"),
    "core.py_inspect_objects": ("repo", "commit", "object_id"),
    "core.py_inspect_signatures": ("repo", "commit", "signature_id"),
    "core.py_inspect_signature_params": ("repo", "commit", "signature_id", "param_index"),
}

LOOKUP_KEY_REGISTRY: dict[str, dict[str, tuple[str, ...]]] = {
    "core.syntax_nodes": {
        "full": ("repo", "commit", "rel_path", "producer", "node_id"),
        "node_id": ("repo", "commit", "node_id"),
    },
    "core.ast_nodes": {"hash": ("hash",)},
    "core.scip_symbol_information": {"symbol": ("repo", "commit", "symbol")},
    "core.goids": {"goid_h128": ("goid_h128",)},
    "graph.cfg_blocks": {
        "full": ("function_goid_h128", "block_idx"),
        "block_id": ("block_id",),
    },
    "graph.import_modules": {"module": ("repo", "commit", "module")},
    "core.ts_tokens": {"token_id": ("repo", "commit", "rel_path", "language", "token_id")},
    "core.ts_trivia": {"trivia_id": ("repo", "commit", "rel_path", "language", "trivia_id")},
    "core.py_sym_scopes": {"scope_id": ("repo", "commit", "rel_path", "scope_id")},
    "core.py_sym_bindings": {
        "binding_id": ("repo", "commit", "rel_path", "binding_id"),
        "scope_name": ("repo", "commit", "rel_path", "scope_id", "name"),
    },
    "core.py_bc_code_units": {"code_unit_id": ("repo", "commit", "rel_path", "code_unit_id")},
    "core.py_bc_instructions": {
        "instruction_id": ("repo", "commit", "rel_path", "code_unit_id", "instr_id")
    },
    "core.py_bc_blocks": {"block_id": ("repo", "commit", "rel_path", "block_id")},
    "core.py_inspect_objects": {"object_id": ("repo", "commit", "object_id")},
    "core.py_inspect_signatures": {"signature_id": ("repo", "commit", "signature_id")},
    "core.py_inspect_signature_params": {
        "param_id": ("repo", "commit", "signature_id", "param_index")
    },
}

CANONICAL_CASTS: dict[str, Mapping[str, pa.DataType]] = {
    "core.scip_symbol_information": {
        "repo": pa.string(),
        "commit": pa.string(),
        "symbol": pa.string(),
    },
    "core.goids": {"goid_h128": pa.decimal128(38, 0)},
    "graph.import_modules": {"repo": pa.string(), "commit": pa.string(), "module": pa.string()},
    "graph.call_graph_edges": {
        "caller_goid_h128": pa.decimal128(38, 0),
        "callee_goid_h128": pa.decimal128(38, 0),
    },
    "graph.import_graph_edges": {"repo": pa.string(), "commit": pa.string()},
}


def identity_keys(table_key: str) -> tuple[str, ...]:
    """Return the identity (primary) key columns for a table key.

    Parameters
    ----------
    table_key
        Table key to look up in the identity registry.

    Returns
    -------
    tuple[str, ...]
        Identity key columns for the table.

    Raises
    ------
    KeyError
        Raised when the table key is not registered.
    """
    keys = IDENTITY_KEY_REGISTRY.get(table_key)
    if keys is None:
        msg = f"Missing identity key registry for table: {table_key}"
        raise KeyError(msg)
    return keys


def lookup_keys(table_key: str, lookup_name: str) -> tuple[str, ...]:
    """Return the lookup key columns for a table key and lookup name.

    Parameters
    ----------
    table_key
        Table key to look up in the registry.
    lookup_name
        Named lookup key set for the table.

    Returns
    -------
    tuple[str, ...]
        Lookup key columns for the table and lookup name.

    Raises
    ------
    KeyError
        Raised when the table or lookup name is not registered.
    """
    table_lookups = LOOKUP_KEY_REGISTRY.get(table_key)
    if table_lookups is None or lookup_name not in table_lookups:
        msg = f"Missing lookup key registry for {table_key}.{lookup_name}"
        raise KeyError(msg)
    return table_lookups[lookup_name]


def canonicalize_for_table(
    table: pa.Table,
    *,
    table_key: str,
    renames: Mapping[str, str] | None = None,
    casts: Mapping[str, pa.DataType] | None = None,
) -> pa.Table:
    """Normalize a table using registry-backed renames and casts.

    Parameters
    ----------
    table
        Input table to normalize.
    table_key
        Table key used to pull default casts.
    renames
        Optional column renames to apply.
    casts
        Optional column casts to apply after defaults.

    Returns
    -------
    pyarrow.Table
        Normalized table with applied renames and casts.
    """
    merged_casts: dict[str, pa.DataType] = {}
    default_casts = CANONICAL_CASTS.get(table_key)
    if default_casts:
        merged_casts.update(default_casts)
    if casts:
        merged_casts.update(casts)
    return canonicalize_table(
        table,
        renames=renames,
        casts=merged_casts if merged_casts else None,
    )


def pk_from_row(
    row: Mapping[str, object],
    *,
    table_key: str,
) -> dict[str, object]:
    """Build a primary-key payload from a row using registry identity keys.

    Parameters
    ----------
    row
        Row mapping containing key columns.
    table_key
        Table key used to resolve identity columns.

    Returns
    -------
    dict[str, object]
        Primary key mapping for the row.
    """
    return {column: row.get(column) for column in identity_keys(table_key)}

CPG_NODE_ID_TYPE = pa.decimal128(38, 0)
SOURCE_PK_JSON_TYPE = pa.binary()


def canonicalize_table(
    table: pa.Table,
    *,
    renames: Mapping[str, str] | None = None,
    casts: Mapping[str, pa.DataType] | None = None,
) -> pa.Table:
    """Normalize column names and Arrow types for consistent joins.

    Returns
    -------
    pyarrow.Table
        Normalized table with applied renames and casts.
    """
    normalized = table
    if renames:
        normalized = rename_table_columns(normalized, dict(renames))
    if casts:
        for column_name, target_type in casts.items():
            if column_name not in normalized.column_names:
                continue
            index = normalized.schema.get_field_index(column_name)
            if index < 0:
                continue
            casted = pc.cast(normalized[column_name], target_type)
            normalized = normalized.set_column(index, column_name, casted)
    return normalized


def build_anchor_map(
    table: pa.Table,
    *,
    table_key: str,
    pk_columns: Sequence[str],
    include_source_pk_json: bool = True,
) -> pa.Table:
    """Build an anchor map for a source table keyed by primary-key columns.

    Returns
    -------
    pyarrow.Table
        Anchor map containing primary key columns, CPG node IDs, and payload bytes.
    """
    if table.num_rows == 0:
        return _empty_anchor_table(table, pk_columns, include_source_pk_json=include_source_pk_json)
    missing = [column for column in pk_columns if column not in table.column_names]
    if missing:
        return _empty_anchor_table(table, pk_columns, include_source_pk_json=include_source_pk_json)
    pk_table = select_table_columns(table, pk_columns)
    columnar: dict[str, list[object]] = {column: [] for column in pk_columns}
    cpg_ids: list[int] = []
    pk_payloads: list[bytes] = []
    for row in table_rows(pk_table):
        pk_values = {column: row.get(column) for column in pk_columns}
        for column in pk_columns:
            columnar[column].append(pk_values[column])
        cpg_ids.append(cpg_node_id(table_key, pk_values))
        if include_source_pk_json:
            pk_payloads.append(cpg_source_pk_json(pk_values))
    arrays: dict[str, pa.Array] = {}
    for column in pk_columns:
        arrays[column] = pa.array(columnar[column], type=pk_table.schema.field(column).type)
    arrays["cpg_node_id"] = pa.array(cpg_ids, type=CPG_NODE_ID_TYPE)
    if include_source_pk_json:
        arrays["source_pk_json"] = pa.array(pk_payloads, type=SOURCE_PK_JSON_TYPE)
    return pa.Table.from_pydict(arrays)


def _empty_anchor_table(
    table: pa.Table,
    pk_columns: Sequence[str],
    *,
    include_source_pk_json: bool,
) -> pa.Table:
    arrays: dict[str, pa.Array] = {}
    for column in pk_columns:
        dtype = table.schema.field(column).type if column in table.column_names else pa.string()
        arrays[column] = pa.array([], type=dtype)
    arrays["cpg_node_id"] = pa.array([], type=CPG_NODE_ID_TYPE)
    if include_source_pk_json:
        arrays["source_pk_json"] = pa.array([], type=SOURCE_PK_JSON_TYPE)
    return pa.Table.from_pydict(arrays)


__all__ = [
    "CANONICAL_CASTS",
    "CPG_NODE_ID_TYPE",
    "IDENTITY_KEY_REGISTRY",
    "LOOKUP_KEY_REGISTRY",
    "SOURCE_PK_JSON_TYPE",
    "build_anchor_map",
    "canonicalize_for_table",
    "canonicalize_table",
    "identity_keys",
    "lookup_keys",
    "pk_from_row",
]
