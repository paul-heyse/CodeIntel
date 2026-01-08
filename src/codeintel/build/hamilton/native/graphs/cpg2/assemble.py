"""CPG assembly helpers and diagnostics emission."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa

from codeintel.build.graphs.assembly import ensure_table_columns, tabular_to_table
from codeintel.build.hamilton.diagnostics import diagnostics_dir
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.graphs.cpg.constants import CPG_TARGET_NAME
from codeintel.build.hamilton.native.graphs.cpg2.planes.ast import cpg2_nodes__ast_nodes
from codeintel.build.hamilton.native.graphs.cpg2.planes.bytecode import (
    cpg2_nodes__py_bc_blocks,
    cpg2_nodes__py_bc_code_units,
    cpg2_nodes__py_bc_instructions,
)
from codeintel.build.hamilton.native.graphs.cpg2.planes.call_wiring import (
    cpg2_edges__call_wiring_arg_to_param,
    cpg2_edges__call_wiring_calls,
    cpg2_edges__call_wiring_ret_to_call,
)
from codeintel.build.hamilton.native.graphs.cpg2.planes.flow import (
    cpg2_edges__cdg_edges,
    cpg2_edges__cfg_edges,
    cpg2_edges__dfg_edges,
    cpg2_nodes__cfg_blocks,
)
from codeintel.build.hamilton.native.graphs.cpg2.planes.goids import cpg2_nodes__goids
from codeintel.build.hamilton.native.graphs.cpg2.planes.inspect import (
    cpg2_nodes__py_inspect_objects,
    cpg2_nodes__py_inspect_signature_params,
    cpg2_nodes__py_inspect_signatures,
)
from codeintel.build.hamilton.native.graphs.cpg2.planes.link import (
    cpg2_edges__call_graph_edges,
    cpg2_edges__import_graph_edges,
    cpg2_nodes__import_modules,
)
from codeintel.build.hamilton.native.graphs.cpg2.planes.overlays_bytecode import (
    PyBcReachesInputs,
    cpg2_edges__py_bc_callsite,
    cpg2_edges__py_bc_callsite_symbol,
    cpg2_edges__py_bc_cfg,
    cpg2_edges__py_bc_defuse_binding,
    cpg2_edges__py_bc_instruction_ast,
    cpg2_edges__py_bc_memory,
    cpg2_edges__py_bc_reaches,
    cpg2_edges__py_bc_stack,
)
from codeintel.build.hamilton.native.graphs.cpg2.planes.overlays_inspect import (
    InspectArgToParamInputs,
    cpg2_edges__inspect_arg_to_param,
    cpg2_edges__inspect_to_ast,
    cpg2_edges__inspect_to_scip,
    cpg2_edges__py_inspect_class_attr,
    cpg2_edges__py_inspect_class_mro,
    cpg2_edges__py_inspect_runtime_state,
    cpg2_edges__py_inspect_signature,
    cpg2_edges__py_inspect_unwrap,
)
from codeintel.build.hamilton.native.graphs.cpg2.planes.overlays_symtable import (
    cpg2_edges__ast_binding_edges,
    cpg2_edges__py_sym_binding_edges,
    cpg2_edges__py_sym_binding_symbol_edges,
    cpg2_edges__py_sym_namespace_edges,
    cpg2_edges__py_sym_resolution_edges,
    cpg2_edges__py_sym_scope_edges,
)
from codeintel.build.hamilton.native.graphs.cpg2.planes.py_sym import (
    cpg2_nodes__py_sym_bindings,
    cpg2_nodes__py_sym_scopes,
    cpg2_nodes__py_sym_unresolved_bindings,
)
from codeintel.build.hamilton.native.graphs.cpg2.planes.scip import (
    cpg2_edges__scip_occurrences,
    cpg2_nodes__scip_external_symbols,
    cpg2_nodes__scip_symbols,
)
from codeintel.build.hamilton.native.graphs.cpg2.planes.symbol import (
    cpg2_edges__scip_symbol_goid_xref,
    cpg2_edges__scip_symbol_relationships,
)
from codeintel.build.hamilton.native.graphs.cpg2.planes.syntax import (
    cpg2_edges__syntax_edges,
    cpg2_nodes__syntax_nodes,
)
from codeintel.build.hamilton.native.graphs.cpg2.planes.treesitter import (
    cpg2_nodes__ts_tokens,
    cpg2_nodes__ts_trivia,
)
from codeintel.build.hamilton.native.graphs.cpg2.types import (
    CpgEdgeConfig,
    _CpgEdgeCoreInputs,
    _CpgNodeCoreInputs,
    _CpgNodeCoreLazyFrames,
    _CpgNodeGraphInputs,
    _CpgNodeGraphLazyFrames,
    _CpgNodeInputs,
    _CpgOverlayEdgeInputs,
)
from codeintel.build.tabular.compute_helpers import cast_array, scalar_from_compute
from codeintel.build.tabular.compute_masks import (
    and_kleene,
    invert_mask,
    is_in_mask,
    is_valid_mask,
)
from codeintel.build.tabular.finalize_ops import FinalizeSpec, finalize_table
from codeintel.core.columnar.rows import empty_table_for_table
from codeintel.core.columnar.schema_ops import concat_tables_unified
from codeintel.core.schemas.arrow_gen import arrow_contract_for_table_schema
from codeintel.core.schemas.row_models import columns_for_table_key
from codeintel.core.schemas.service import get_schema_service

LOG = logging.getLogger(__name__)

CPG_NODES_TABLE_KEY = "graph.cpg_nodes"
CPG_EDGES_TABLE_KEY = "graph.cpg_edges"


@dataclass(frozen=True, slots=True)
class CpgDiagnosticsPayload:
    """Optional diagnostics payloads emitted during CPG assembly."""

    plane_row_counts: Mapping[str, object] | None = None
    anchor_resolution: Mapping[str, object] | None = None
    join_drop_rates: Mapping[str, object] | None = None
    contract_mismatches: Mapping[str, object] | None = None
    edge_integrity: Mapping[str, object] | None = None

    def iter_payloads(self) -> Sequence[tuple[str, Mapping[str, object] | None]]:
        """Return filename/payload pairs for diagnostics emission.

        Returns
        -------
        Sequence[tuple[str, Mapping[str, object] | None]]
            Filename and payload pairs for diagnostics output.
        """
        return (
            ("cpg_plane_row_counts.json", self.plane_row_counts),
            ("cpg_anchor_resolution.json", self.anchor_resolution),
            ("cpg_join_drop_rates.json", self.join_drop_rates),
            ("cpg_contract_mismatches.json", self.contract_mismatches),
            ("cpg_edge_integrity.json", self.edge_integrity),
        )


def emit_cpg_diagnostics(
    env: BuildEnv,
    *,
    payloads: CpgDiagnosticsPayload,
) -> None:
    """Emit CPG diagnostics under build/diagnostics without blocking execution.

    Parameters
    ----------
    env
        Build environment with the output directory paths.
    payloads
        Diagnostics payloads to serialize.
    """
    try:
        diag_dir = diagnostics_dir(env.paths.build_dir)
        diag_dir.mkdir(parents=True, exist_ok=True)
        for filename, payload in payloads.iter_payloads():
            _merge_json(diag_dir / filename, payload)
    except (OSError, ValueError, TypeError) as exc:
        LOG.warning("build.cpg.diagnostics_failed error=%s", exc)


def assemble_cpg_nodes(tables: Sequence[pa.Table]) -> pa.Table:
    """Assemble CPG nodes from per-plane tables.

    Returns
    -------
    pyarrow.Table
        Contract-aligned CPG nodes table.
    """
    tables = [table for table in tables if table.num_rows > 0]
    if not tables:
        return empty_table_for_table(CPG_NODES_TABLE_KEY)
    combined = concat_tables_unified(tables)
    combined = _ensure_contract_columns(CPG_NODES_TABLE_KEY, combined)
    combined = _cast_to_contract_types(CPG_NODES_TABLE_KEY, combined)
    result = finalize_table(
        combined,
        spec=FinalizeSpec(
            table_key=CPG_NODES_TABLE_KEY,
            mode="strict",
            target_name=CPG_TARGET_NAME,
        ),
    )
    return result.good


def assemble_cpg_edges(tables: Sequence[pa.Table]) -> pa.Table:
    """Assemble CPG edges from per-plane tables.

    Returns
    -------
    pyarrow.Table
        Contract-aligned CPG edges table.
    """
    tables = [table for table in tables if table.num_rows > 0]
    if not tables:
        return empty_table_for_table(CPG_EDGES_TABLE_KEY)
    combined = concat_tables_unified(tables)
    combined = _ensure_contract_columns(CPG_EDGES_TABLE_KEY, combined)
    combined = _cast_to_contract_types(CPG_EDGES_TABLE_KEY, combined)
    result = finalize_table(
        combined,
        spec=FinalizeSpec(
            table_key=CPG_EDGES_TABLE_KEY,
            mode="strict",
            target_name=CPG_TARGET_NAME,
        ),
    )
    return result.good


def edge_integrity_report(
    edges: pa.Table,
    *,
    nodes: pa.Table | None = None,
) -> dict[str, object]:
    """Return edge referential integrity metrics for diagnostics.

    Parameters
    ----------
    edges
        CPG edges table to analyze.
    nodes
        Optional CPG nodes table for referential checks.

    Returns
    -------
    dict[str, object]
        Edge integrity counts keyed by metric name.
    """
    report: dict[str, object] = {"edge_rows": edges.num_rows}
    if edges.num_rows == 0:
        return report
    src_col = edges.column("src_cpg_node_id") if "src_cpg_node_id" in edges.column_names else None
    dst_col = edges.column("dst_cpg_node_id") if "dst_cpg_node_id" in edges.column_names else None
    ordinal_col = edges.column("ordinal") if "ordinal" in edges.column_names else None
    if src_col is not None:
        report["src_null"] = _count_mask(invert_mask(is_valid_mask(src_col)))
    if dst_col is not None:
        report["dst_null"] = _count_mask(invert_mask(is_valid_mask(dst_col)))
    if ordinal_col is not None:
        report["ordinal_null"] = _count_mask(invert_mask(is_valid_mask(ordinal_col)))
    if nodes is None or nodes.num_rows == 0:
        return report
    if "cpg_node_id" not in nodes.column_names:
        return report
    node_ids = nodes.column("cpg_node_id")
    if src_col is not None:
        src_in = is_in_mask(src_col, value_set=node_ids)
        src_valid = is_valid_mask(src_col)
        report["src_missing"] = _count_mask(and_kleene(src_valid, invert_mask(src_in)))
    if dst_col is not None:
        dst_in = is_in_mask(dst_col, value_set=node_ids)
        dst_valid = is_valid_mask(dst_col)
        report["dst_missing"] = _count_mask(and_kleene(dst_valid, invert_mask(dst_in)))
    return report


def _core_lazyframes(inputs: _CpgNodeCoreInputs) -> _CpgNodeCoreLazyFrames:
    return _CpgNodeCoreLazyFrames(
        syntax_nodes=tabular_to_table(inputs.syntax_nodes),
        ast_nodes=tabular_to_table(inputs.ast_nodes),
        scip_symbol_information=tabular_to_table(inputs.scip_symbol_information),
        scip_external_symbols=tabular_to_table(inputs.scip_external_symbols),
        goids=tabular_to_table(inputs.goids),
        py_sym_scopes=tabular_to_table(inputs.py_sym_scopes),
        py_sym_bindings=tabular_to_table(inputs.py_sym_bindings),
        py_sym_unresolved_bindings=tabular_to_table(inputs.py_sym_unresolved_bindings),
        py_bc_code_units=tabular_to_table(inputs.py_bc_code_units),
        py_bc_instructions=tabular_to_table(inputs.py_bc_instructions),
        py_bc_blocks=tabular_to_table(inputs.py_bc_blocks),
        py_inspect_objects=tabular_to_table(inputs.py_inspect_objects),
        py_inspect_signatures=tabular_to_table(inputs.py_inspect_signatures),
        py_inspect_signature_params=tabular_to_table(inputs.py_inspect_signature_params),
        ts_tokens=tabular_to_table(inputs.ts_tokens),
        ts_trivia=tabular_to_table(inputs.ts_trivia),
    )


def _graph_lazyframes(inputs: _CpgNodeGraphInputs) -> _CpgNodeGraphLazyFrames:
    return _CpgNodeGraphLazyFrames(
        cfg_blocks=tabular_to_table(inputs.cfg_blocks),
        import_modules=tabular_to_table(inputs.import_modules),
    )


def cpg2_nodes__frames(
    env: BuildEnv,
    cpg_nodes__inputs: _CpgNodeInputs,
) -> pa.Table:
    """Build and assemble CPG node frames (prefixed internal aggregator).

    Returns
    -------
    pyarrow.Table
        Contract-aligned CPG nodes table.
    """
    core = _core_lazyframes(cpg_nodes__inputs.core)
    graph = _graph_lazyframes(cpg_nodes__inputs.graph)
    frames = [
        cpg2_nodes__syntax_nodes(core.syntax_nodes),
        cpg2_nodes__ast_nodes(core.ast_nodes, env),
        cpg2_nodes__scip_symbols(core.scip_symbol_information),
        cpg2_nodes__scip_external_symbols(core.scip_external_symbols),
        cpg2_nodes__goids(core.goids),
        cpg2_nodes__py_sym_scopes(core.py_sym_scopes),
        cpg2_nodes__py_sym_bindings(core.py_sym_bindings),
        cpg2_nodes__py_sym_unresolved_bindings(core.py_sym_unresolved_bindings),
        cpg2_nodes__py_bc_code_units(core.py_bc_code_units),
        cpg2_nodes__py_bc_instructions(core.py_bc_instructions),
        cpg2_nodes__py_bc_blocks(core.py_bc_blocks),
        cpg2_nodes__py_inspect_objects(core.py_inspect_objects),
        cpg2_nodes__py_inspect_signatures(core.py_inspect_signatures),
        cpg2_nodes__py_inspect_signature_params(core.py_inspect_signature_params),
        cpg2_nodes__ts_tokens(core.ts_tokens),
        cpg2_nodes__ts_trivia(core.ts_trivia),
        cpg2_nodes__cfg_blocks(graph.cfg_blocks, core.goids),
        cpg2_nodes__import_modules(graph.import_modules),
    ]
    return assemble_cpg_nodes(frames)


def cpg2_edges__frames(
    env: BuildEnv,
    cpg_edge_core_inputs: _CpgEdgeCoreInputs,
    cpg_edge_overlay_inputs: _CpgOverlayEdgeInputs,
    cpg__edge_config: CpgEdgeConfig,
    cpg2_nodes__frames: pa.Table,
) -> pa.Table:
    """Build and assemble CPG edge frames (prefixed internal aggregator).

    Returns
    -------
    pyarrow.Table
        Contract-aligned CPG edges table.
    """
    core = cpg_edge_core_inputs
    overlay = cpg_edge_overlay_inputs
    frames = [
        cpg2_edges__syntax_edges(core.symbol.syntax_edges, core.syntax_nodes.syntax_nodes),
        cpg2_edges__scip_occurrences(
            core.symbol.occ_syntax,
            core.symbol.occ_span,
            core.symbol.scip_symbols,
            core.symbol.scip_external_symbols,
        ),
        cpg2_edges__scip_symbol_relationships(
            core.symbol.symbol_rels,
            core.symbol.scip_symbols,
            core.symbol.scip_external_symbols,
        ),
        cpg2_edges__scip_symbol_goid_xref(
            core.symbol.symbol_goid,
            core.symbol.scip_symbols,
            core.flow.goids,
        ),
        cpg2_edges__call_graph_edges(core.link.call_edges, core.flow.goids),
        cpg2_edges__import_graph_edges(core.link.import_edges, core.link.import_modules),
        cpg2_edges__cfg_edges(core.flow.cfg_edges, core.flow.cfg_blocks, core.flow.goids),
        cpg2_edges__dfg_edges(core.flow.dfg_edges, core.flow.cfg_blocks, core.flow.goids),
        cpg2_edges__cdg_edges(core.flow.cdg_edges, core.flow.cfg_blocks, core.flow.goids),
        cpg2_edges__call_wiring_calls(
            core.call_wiring.call_edges,
            core.flow.cfg_blocks,
            core.syntax_nodes.syntax_nodes,
        ),
        cpg2_edges__call_wiring_arg_to_param(
            core.call_wiring.arg_to_param_edges,
            core.syntax_nodes.syntax_nodes,
        ),
        cpg2_edges__call_wiring_ret_to_call(
            core.call_wiring.ret_to_call_edges,
            core.flow.cfg_blocks,
            core.syntax_nodes.syntax_nodes,
        ),
    ]
    if cpg__edge_config.overlay_options.enable_symtable:
        frames.extend(
            [
                cpg2_edges__py_sym_scope_edges(overlay.py_sym_scope_edges),
                cpg2_edges__py_sym_namespace_edges(
                    overlay.py_sym_namespace_edges,
                    overlay.py_sym_bindings,
                ),
                cpg2_edges__py_sym_binding_edges(overlay.py_sym_bindings),
                cpg2_edges__py_sym_resolution_edges(overlay.py_sym_resolution_edges),
                cpg2_edges__py_sym_binding_symbol_edges(
                    overlay.py_sym_bindings,
                    overlay.py_sym_scopes,
                    overlay.scip_symbols,
                ),
                cpg2_edges__ast_binding_edges(
                    overlay.ast_nodes,
                    overlay.py_sym_scopes,
                    overlay.py_sym_bindings,
                    overlay.py_sym_resolution_edges,
                ),
            ]
        )
    if cpg__edge_config.overlay_options.enable_bytecode:
        bytecode_inputs = PyBcReachesInputs(
            defuse_events=overlay.py_bc_defuse_events,
            code_units=overlay.py_bc_code_units,
            scopes=overlay.py_sym_scopes,
            bindings=overlay.py_sym_bindings,
            resolution_edges=overlay.py_sym_resolution_edges,
            blocks=overlay.py_bc_blocks,
            cfg_edges=overlay.py_bc_cfg_edges,
        )
        frames.extend(
            [
                cpg2_edges__py_bc_instruction_ast(overlay.py_bc_instructions, overlay.ast_nodes),
                cpg2_edges__py_bc_callsite(overlay.py_bc_instructions, overlay.syntax_calls),
                cpg2_edges__py_bc_callsite_symbol(
                    overlay.py_bc_instructions,
                    overlay.syntax_calls,
                    overlay.scip_symbols,
                ),
                cpg2_edges__py_bc_cfg(overlay.py_bc_cfg_edges),
                cpg2_edges__py_bc_defuse_binding(bytecode_inputs),
                cpg2_edges__py_bc_memory(
                    overlay.py_bc_defuse_events,
                    overlay.py_bc_instructions,
                    overlay.ast_nodes,
                ),
                cpg2_edges__py_bc_stack(overlay.py_bc_instructions, overlay.py_bc_blocks),
            ]
        )
        if cpg__edge_config.options.enable_reaches:
            frames.append(cpg2_edges__py_bc_reaches(bytecode_inputs))
    if cpg__edge_config.overlay_options.enable_inspect:
        inspect_inputs = InspectArgToParamInputs(
            syntax_calls=overlay.syntax_calls,
            syntax_call_args=overlay.syntax_call_args,
            inspect_objects=overlay.py_inspect_objects,
            inspect_signatures=overlay.py_inspect_signatures,
            inspect_signature_params=overlay.py_inspect_signature_params,
        )
        frames.extend(
            [
                cpg2_edges__py_inspect_signature(
                    overlay.py_inspect_signatures,
                    overlay.py_inspect_signature_params,
                ),
                cpg2_edges__inspect_arg_to_param(inspect_inputs),
                cpg2_edges__py_inspect_unwrap(overlay.py_inspect_unwrap_hops),
                cpg2_edges__py_inspect_class_mro(overlay.py_inspect_class_mro),
                cpg2_edges__py_inspect_class_attr(overlay.py_inspect_class_attrs),
                cpg2_edges__py_inspect_runtime_state(
                    overlay.py_inspect_runtime_state,
                    overlay.py_bc_code_units,
                    overlay.py_bc_instructions,
                ),
                cpg2_edges__inspect_to_ast(
                    overlay.py_inspect_objects,
                    overlay.py_inspect_source,
                    overlay.ast_nodes,
                ),
                cpg2_edges__inspect_to_scip(overlay.py_inspect_objects, overlay.scip_symbols),
            ]
        )
    assembled = assemble_cpg_edges(frames)
    emit_cpg_diagnostics(
        env,
        payloads=CpgDiagnosticsPayload(
            edge_integrity=edge_integrity_report(assembled, nodes=cpg2_nodes__frames),
        ),
    )
    return assembled


def _merge_json(path: Path, payload: Mapping[str, object] | None) -> None:
    if payload is None:
        return
    if not payload:
        return
    existing = _read_json(path)
    merged = _merge_mapping(existing, payload)
    path.write_text(json.dumps(merged, indent=2, sort_keys=True), encoding="utf-8")


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        LOG.warning("build.cpg.diagnostics_read_failed path=%s error=%s", path, exc)
        return {}
    if isinstance(parsed, dict):
        return parsed
    return {}


def _merge_mapping(
    left: Mapping[str, object] | None,
    right: Mapping[str, object],
) -> dict[str, object]:
    merged: dict[str, object] = {}
    if isinstance(left, Mapping):
        merged.update(left)
    for key, value in right.items():
        existing_value = merged.get(key)
        if isinstance(value, Mapping) and isinstance(existing_value, Mapping):
            merged[key] = _merge_mapping(existing_value, value)
        else:
            merged[key] = value
    return merged


def _ensure_contract_columns(table_key: str, table: pa.Table) -> pa.Table:
    columns = columns_for_table_key(table_key)
    if columns is None:
        return table
    return ensure_table_columns(table, columns)


def _cast_to_contract_types(table_key: str, table: pa.Table) -> pa.Table:
    schema_service = get_schema_service()
    table_schema = schema_service.get_table_schema(table_key)
    if table_schema is None:
        return table
    contract = arrow_contract_for_table_schema(table_schema=table_schema, metadata=None)
    type_map = {field.name: field.type for field in contract}
    arrays: list[pa.Array | pa.ChunkedArray] = []
    changed = False
    for name in table.column_names:
        column = table[name]
        target_type = type_map.get(name)
        if target_type is None or column.type == target_type:
            arrays.append(column)
            continue
        if pa.types.is_null(column.type):
            casted = pa.nulls(table.num_rows, type=target_type)
        else:
            casted = cast_array(column, target_type, safe=False)
        arrays.append(casted)
        changed = True
    if not changed:
        return table
    fields: list[pa.Field] = []
    for name, array in zip(table.column_names, arrays, strict=True):
        field = table.schema.field(name)
        if field.type != array.type:
            field = field.with_type(array.type)
        fields.append(field)
    return pa.Table.from_arrays(arrays, schema=pa.schema(fields, metadata=table.schema.metadata))


def _count_mask(mask: pa.Array | pa.ChunkedArray) -> int:
    total = scalar_from_compute("sum", [mask])
    if isinstance(total, (int, float)):
        return int(total)
    return 0


__all__ = [
    "CPG_EDGES_TABLE_KEY",
    "CPG_NODES_TABLE_KEY",
    "CpgDiagnosticsPayload",
    "assemble_cpg_edges",
    "assemble_cpg_nodes",
    "cpg2_edges__frames",
    "cpg2_nodes__frames",
    "edge_integrity_report",
    "emit_cpg_diagnostics",
]
