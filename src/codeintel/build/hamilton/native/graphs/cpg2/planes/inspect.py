"""Python inspect plane CPG nodes."""

from __future__ import annotations

import pyarrow as pa

from codeintel.build.graphs.assembly import table_rows
from codeintel.build.hamilton.native.graphs.cpg2.anchors import pk_from_row
from codeintel.build.hamilton.native.graphs.cpg2.ids import cpg_node_id, cpg_source_pk_json
from codeintel.build.tabular.extras_ops import extras_kv_from_mapping
from codeintel.core.columnar.rows import empty_table_for_table, table_for_rows

CPG_NODES_TABLE_KEY = "graph.cpg_nodes"
PY_INSPECT_OBJECTS_TABLE_KEY = "core.py_inspect_objects"
PY_INSPECT_SIGNATURES_TABLE_KEY = "core.py_inspect_signatures"
PY_INSPECT_SIGNATURE_PARAMS_TABLE_KEY = "core.py_inspect_signature_params"


def cpg2_nodes__py_inspect_objects(objects: pa.Table) -> pa.Table:
    """Build CPG nodes from inspect objects.

    Returns
    -------
    pyarrow.Table
        CPG node table for inspect objects.
    """
    required = {"repo", "commit", "object_id", "kind"}
    if not required.issubset(set(objects.column_names)):
        return empty_table_for_table(CPG_NODES_TABLE_KEY)
    rows: list[dict[str, object]] = []
    for row in table_rows(objects):
        pk_values = pk_from_row(row, table_key=PY_INSPECT_OBJECTS_TABLE_KEY)
        extras_values = {
            "kind": row.get("kind"),
            "module_name": row.get("module_name"),
            "qualname": row.get("qualname"),
            "name": row.get("name"),
            "type_qualname": row.get("type_qualname"),
            "object_addr": row.get("object_addr"),
            "is_builtin": row.get("is_builtin"),
            "is_callable": row.get("is_callable"),
            "is_descriptor": row.get("is_descriptor"),
            "has_wrapped": row.get("has_wrapped"),
            "has_signature_override": row.get("has_signature_override"),
            "has_annotations": row.get("has_annotations"),
            "status": row.get("status"),
        }
        extras_kv = extras_kv_from_mapping(extras_values)
        rows.append(
            {
                "repo": row.get("repo"),
                "commit": row.get("commit"),
                "cpg_node_id": cpg_node_id(PY_INSPECT_OBJECTS_TABLE_KEY, pk_values),
                "node_kind": "INSPECT_OBJECT",
                "source_table_key": PY_INSPECT_OBJECTS_TABLE_KEY,
                "source_pk_json": cpg_source_pk_json(pk_values),
                "rel_path": None,
                "start_byte": None,
                "end_byte": None,
                "extras": None,
                "extras_kv": extras_kv,
            }
        )
    table, _ = table_for_rows(CPG_NODES_TABLE_KEY, rows)
    return table


def cpg2_nodes__py_inspect_signatures(signatures: pa.Table) -> pa.Table:
    """Build CPG nodes from inspect signatures.

    Returns
    -------
    pyarrow.Table
        CPG node table for inspect signatures.
    """
    required = {"repo", "commit", "signature_id", "object_id"}
    if not required.issubset(set(signatures.column_names)):
        return empty_table_for_table(CPG_NODES_TABLE_KEY)
    rows: list[dict[str, object]] = []
    for row in table_rows(signatures):
        pk_values = pk_from_row(row, table_key=PY_INSPECT_SIGNATURES_TABLE_KEY)
        extras_values = {
            "object_id": row.get("object_id"),
            "mode": row.get("mode"),
            "variant": row.get("variant"),
            "follow_wrapped": row.get("follow_wrapped"),
            "eval_str": row.get("eval_str"),
            "effective_object_id": row.get("effective_object_id"),
            "sig_text": row.get("sig_text"),
            "sig_format": row.get("sig_format"),
            "has_varargs": row.get("has_varargs"),
            "has_varkw": row.get("has_varkw"),
            "status": row.get("status"),
        }
        extras_kv = extras_kv_from_mapping(extras_values)
        rows.append(
            {
                "repo": row.get("repo"),
                "commit": row.get("commit"),
                "cpg_node_id": cpg_node_id(PY_INSPECT_SIGNATURES_TABLE_KEY, pk_values),
                "node_kind": "INSPECT_SIGNATURE",
                "source_table_key": PY_INSPECT_SIGNATURES_TABLE_KEY,
                "source_pk_json": cpg_source_pk_json(pk_values),
                "rel_path": None,
                "start_byte": None,
                "end_byte": None,
                "extras": None,
                "extras_kv": extras_kv,
            }
        )
    table, _ = table_for_rows(CPG_NODES_TABLE_KEY, rows)
    return table


def cpg2_nodes__py_inspect_signature_params(params: pa.Table) -> pa.Table:
    """Build CPG nodes from inspect signature parameters.

    Returns
    -------
    pyarrow.Table
        CPG node table for inspect signature params.
    """
    required = {"repo", "commit", "signature_id", "param_index"}
    if not required.issubset(set(params.column_names)):
        return empty_table_for_table(CPG_NODES_TABLE_KEY)
    rows: list[dict[str, object]] = []
    for row in table_rows(params):
        pk_values = pk_from_row(row, table_key=PY_INSPECT_SIGNATURE_PARAMS_TABLE_KEY)
        extras_values = {
            "mode": row.get("mode"),
            "name": row.get("name"),
            "kind": row.get("kind"),
            "default_present": row.get("default_present"),
            "default_value": row.get("default_value"),
            "annotation_present": row.get("annotation_present"),
            "annotation_value": row.get("annotation_value"),
            "status": row.get("status"),
        }
        extras_kv = extras_kv_from_mapping(extras_values)
        rows.append(
            {
                "repo": row.get("repo"),
                "commit": row.get("commit"),
                "cpg_node_id": cpg_node_id(PY_INSPECT_SIGNATURE_PARAMS_TABLE_KEY, pk_values),
                "node_kind": "INSPECT_SIGNATURE_PARAM",
                "source_table_key": PY_INSPECT_SIGNATURE_PARAMS_TABLE_KEY,
                "source_pk_json": cpg_source_pk_json(pk_values),
                "rel_path": None,
                "start_byte": None,
                "end_byte": None,
                "extras": None,
                "extras_kv": extras_kv,
            }
        )
    table, _ = table_for_rows(CPG_NODES_TABLE_KEY, rows)
    return table


__all__ = [
    "cpg2_nodes__py_inspect_objects",
    "cpg2_nodes__py_inspect_signature_params",
    "cpg2_nodes__py_inspect_signatures",
]
