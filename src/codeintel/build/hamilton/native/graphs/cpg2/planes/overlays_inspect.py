"""Inspect overlay CPG edges."""

from __future__ import annotations

import inspect
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import pyarrow as pa

from codeintel.build.graphs.assembly import ensure_table_columns, table_rows
from codeintel.build.hamilton.native.graphs.cpg2.ids import cpg_edge_ordinal, cpg_node_id
from codeintel.build.tabular.arrow_ops import concat_tables_unified
from codeintel.core.columnar.rows import empty_table_for_table, table_for_rows
from codeintel.core.schemas.row_models import columns_for_table_key
from codeintel.core.serialization.payload import decode_payload, encode_payload

CPG_EDGES_TABLE_KEY = "graph.cpg_edges"
AST_NODES_TABLE_KEY = "core.ast_nodes"
SCIP_SYMBOLS_TABLE_KEY = "core.scip_symbol_information"
SYNTAX_NODES_TABLE_KEY = "core.syntax_nodes"
SYNTAX_CALLS_TABLE_KEY = "core.syntax_calls"
SYNTAX_CALL_ARGS_TABLE_KEY = "core.syntax_call_args"
PY_INSPECT_OBJECTS_TABLE_KEY = "core.py_inspect_objects"
PY_INSPECT_CLASS_MRO_TABLE_KEY = "core.py_inspect_class_mro"
PY_INSPECT_CLASS_ATTRS_TABLE_KEY = "core.py_inspect_class_attrs"
PY_INSPECT_UNWRAP_TABLE_KEY = "core.py_inspect_unwrap_hops"
PY_INSPECT_SIGNATURES_TABLE_KEY = "core.py_inspect_signatures"
PY_INSPECT_SIGNATURE_PARAMS_TABLE_KEY = "core.py_inspect_signature_params"
PY_INSPECT_SOURCE_TABLE_KEY = "core.py_inspect_source"
PY_INSPECT_RUNTIME_STATE_TABLE_KEY = "core.py_inspect_runtime_state"
PY_BC_CODE_UNITS_TABLE_KEY = "core.py_bc_code_units"
PY_BC_INSTRUCTIONS_TABLE_KEY = "core.py_bc_instructions"


@dataclass(frozen=True)
class OverlayEdgeDiagnostics:
    """Diagnostics for overlay edge resolution."""

    expected_edges: int
    produced_edges: int
    dropped_edges: int


@dataclass(frozen=True)
class _ParamGroups:
    positional: list[dict[str, object]]
    keyword: dict[str, dict[str, object]]
    var_positional: dict[str, object] | None
    var_keyword: dict[str, object] | None


@dataclass(frozen=True)
class _InspectArgToParamContext:
    repo: str
    commit: str
    rel_path: str
    producer: str
    call_id: str
    signature_id: str


@dataclass(frozen=True, slots=True)
class InspectArgToParamInputs:
    """Inputs for inspect arg-to-param edge emission."""

    syntax_calls: pa.Table
    syntax_call_args: pa.Table
    inspect_objects: pa.Table
    inspect_signatures: pa.Table
    inspect_signature_params: pa.Table


@dataclass(frozen=True, slots=True)
class _InspectAstIndex:
    by_qualname: dict[str, list[dict[str, object]]]
    by_norm_path: dict[str, list[dict[str, object]]]
    paths: list[str]


@dataclass(frozen=True, slots=True)
class _InspectAstContext:
    repo: str
    commit: str
    object_id: str


@dataclass(frozen=True)
class _InspectClassAttrContext:
    repo: str
    commit: str
    class_object_id: str
    attr_name: str
    defining_object_id: str | None
    value_object_id: str | None
    extras: dict[str, object]
    src_cpg_node_id: int
    is_descriptor: bool


@dataclass(frozen=True, slots=True)
class _RuntimeStateInstrContext:
    repo: str
    commit: str
    object_id: str
    state_kind: str | None
    frame_qualpath: str
    frame_offset: int
    extras: dict[str, object]


def cpg2_edges__py_inspect_signature(
    signatures: pa.Table,
    params: pa.Table,
    *,
    diagnostics: dict[str, object] | None = None,
) -> pa.Table:
    """Build inspect signature + param edges.

    Returns
    -------
    pyarrow.Table
        CPG edges for inspect signatures.
    """
    edges = _py_inspect_signature_edges_to_rows(signatures, params)
    table, row_count = table_for_rows(CPG_EDGES_TABLE_KEY, edges)
    _record_diagnostics(
        diagnostics,
        "overlay_inspect_signature",
        expected_edges=signatures.num_rows + params.num_rows,
        produced_edges=row_count,
    )
    return table


def cpg2_edges__inspect_arg_to_param(
    inputs: InspectArgToParamInputs,
    *,
    diagnostics: dict[str, object] | None = None,
) -> pa.Table:
    """Build inspect arg -> param edges.

    Returns
    -------
    pyarrow.Table
        CPG edges mapping call args to inspect params.
    """
    edges = _inspect_arg_to_param_edges_to_rows(
        inputs.syntax_calls,
        inputs.syntax_call_args,
        inputs.inspect_objects,
        inputs.inspect_signatures,
        inputs.inspect_signature_params,
    )
    table, row_count = table_for_rows(CPG_EDGES_TABLE_KEY, edges)
    _record_diagnostics(
        diagnostics,
        "overlay_inspect_arg_to_param",
        expected_edges=inputs.syntax_call_args.num_rows,
        produced_edges=row_count,
    )
    return table


def cpg2_edges__py_inspect_unwrap(
    unwrap_hops: pa.Table,
    *,
    diagnostics: dict[str, object] | None = None,
) -> pa.Table:
    """Build inspect unwrap edges.

    Returns
    -------
    pyarrow.Table
        CPG edges for unwrap hops.
    """
    edges = _py_inspect_unwrap_edges_to_rows(unwrap_hops)
    table, row_count = table_for_rows(CPG_EDGES_TABLE_KEY, edges)
    _record_diagnostics(
        diagnostics,
        "overlay_inspect_unwrap",
        expected_edges=unwrap_hops.num_rows,
        produced_edges=row_count,
    )
    return table


def cpg2_edges__py_inspect_class_mro(
    class_mro: pa.Table,
    *,
    diagnostics: dict[str, object] | None = None,
) -> pa.Table:
    """Build inspect class MRO edges.

    Returns
    -------
    pyarrow.Table
        CPG edges for class inheritance.
    """
    edges = _py_inspect_class_mro_edges_to_rows(class_mro)
    table, row_count = table_for_rows(CPG_EDGES_TABLE_KEY, edges)
    _record_diagnostics(
        diagnostics,
        "overlay_inspect_class_mro",
        expected_edges=class_mro.num_rows,
        produced_edges=row_count,
    )
    return table


def cpg2_edges__py_inspect_class_attr(
    class_attrs: pa.Table,
    *,
    diagnostics: dict[str, object] | None = None,
) -> pa.Table:
    """Build inspect class attribute edges.

    Returns
    -------
    pyarrow.Table
        CPG edges for class attributes.
    """
    edges = _py_inspect_class_attr_edges_to_rows(class_attrs)
    table, row_count = table_for_rows(CPG_EDGES_TABLE_KEY, edges)
    _record_diagnostics(
        diagnostics,
        "overlay_inspect_class_attr",
        expected_edges=class_attrs.num_rows,
        produced_edges=row_count,
    )
    return table


def cpg2_edges__py_inspect_runtime_state(
    runtime_state: pa.Table,
    code_units: pa.Table,
    instructions: pa.Table,
    *,
    diagnostics: dict[str, object] | None = None,
) -> pa.Table:
    """Build inspect runtime state edges.

    Returns
    -------
    pyarrow.Table
        CPG edges for runtime state references.
    """
    table = _py_inspect_runtime_state_edges_to_table(runtime_state, code_units, instructions)
    _record_diagnostics(
        diagnostics,
        "overlay_inspect_runtime_state",
        expected_edges=runtime_state.num_rows,
        produced_edges=table.num_rows,
    )
    return table


def cpg2_edges__inspect_to_ast(
    inspect_objects: pa.Table,
    inspect_source: pa.Table,
    ast_nodes: pa.Table,
    *,
    diagnostics: dict[str, object] | None = None,
) -> pa.Table:
    """Build inspect -> AST anchor edges.

    Returns
    -------
    pyarrow.Table
        CPG edges linking inspect objects to AST nodes.
    """
    edges = _inspect_to_ast_edges_to_rows(inspect_objects, inspect_source, ast_nodes)
    table, row_count = table_for_rows(CPG_EDGES_TABLE_KEY, edges)
    _record_diagnostics(
        diagnostics,
        "overlay_inspect_to_ast",
        expected_edges=inspect_objects.num_rows,
        produced_edges=row_count,
    )
    return table


def cpg2_edges__inspect_to_scip(
    inspect_objects: pa.Table,
    scip_symbols: pa.Table,
    *,
    diagnostics: dict[str, object] | None = None,
) -> pa.Table:
    """Build inspect -> SCIP symbol edges.

    Returns
    -------
    pyarrow.Table
        CPG edges linking inspect objects to SCIP symbols.
    """
    edges = _inspect_to_scip_edges_to_rows(inspect_objects, scip_symbols)
    table, row_count = table_for_rows(CPG_EDGES_TABLE_KEY, edges)
    _record_diagnostics(
        diagnostics,
        "overlay_inspect_to_scip",
        expected_edges=inspect_objects.num_rows,
        produced_edges=row_count,
    )
    return table


def _inspect_full_qualname(module_name: str | None, qualname: str | None) -> str | None:
    if module_name is None:
        return None
    if qualname is None:
        return module_name
    if qualname == module_name:
        return module_name
    if qualname.startswith(f"{module_name}."):
        return qualname
    return f"{module_name}.{qualname}"


def _inspect_status_ok(status: object) -> bool:
    if isinstance(status, dict):
        return status.get("ok") is True
    return False


def _callee_qname_priority(source: object) -> int:
    if isinstance(source, str):
        normalized = source.upper()
        if normalized == "IMPORT":
            return 0
        if normalized == "LOCAL":
            return 1
        if normalized == "BUILTIN":
            return 2
    return 3


def _call_callee_candidates(extras: object, callee_text: str | None) -> list[str]:
    decoded = decode_payload(extras)
    candidates: list[tuple[int, str]] = []
    if isinstance(decoded, dict):
        qnames = decoded.get("callee_qnames")
        if isinstance(qnames, list):
            for item in qnames:
                if not isinstance(item, dict):
                    continue
                name = item.get("name")
                if not isinstance(name, str):
                    continue
                priority = _callee_qname_priority(item.get("source"))
                candidates.append((priority, name))
    if not candidates and isinstance(callee_text, str) and "." in callee_text:
        candidates.append((4, callee_text))
    candidates.sort(key=lambda item: (item[0], item[1]))
    return [name for _, name in candidates]


def _sorted_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    key_field: str,
) -> list[dict[str, object]]:
    return sorted(
        (dict(row) for row in rows),
        key=lambda item: _coerce_int(item.get(key_field)) or 0,
    )


def _param_groups(params: Sequence[dict[str, object]]) -> _ParamGroups:
    positional: list[dict[str, object]] = []
    keyword: dict[str, dict[str, object]] = {}
    var_positional: dict[str, object] | None = None
    var_keyword: dict[str, object] | None = None
    for param in params:
        kind = _coerce_str(param.get("kind"))
        if kind in {"POSITIONAL_ONLY", "POSITIONAL_OR_KEYWORD"}:
            positional.append(param)
        if kind in {"POSITIONAL_OR_KEYWORD", "KEYWORD_ONLY"}:
            name = _coerce_str(param.get("name"))
            if name is not None:
                keyword[name] = param
        if kind == "VAR_POSITIONAL":
            var_positional = param
        elif kind == "VAR_KEYWORD":
            var_keyword = param
    return _ParamGroups(
        positional=positional,
        keyword=keyword,
        var_positional=var_positional,
        var_keyword=var_keyword,
    )


def _next_positional_param(
    params: Sequence[dict[str, object]],
    assigned: set[int],
    start_index: int,
) -> tuple[dict[str, object] | None, int]:
    index = start_index
    while index < len(params):
        param = params[index]
        param_index = _coerce_int(param.get("param_index"))
        if param_index is not None and param_index not in assigned:
            return param, index + 1
        index += 1
    return None, index


def _map_positional_arg(
    arg: dict[str, object],
    *,
    groups: _ParamGroups,
    assigned: set[int],
    pos_index: int,
) -> tuple[list[tuple[dict[str, object], dict[str, object], str]], int]:
    mappings: list[tuple[dict[str, object], dict[str, object], str]] = []
    param, next_index = _next_positional_param(groups.positional, assigned, pos_index)
    if param is not None:
        param_index = _coerce_int(param.get("param_index"))
        if param_index is not None:
            assigned.add(param_index)
            mappings.append((arg, param, "positional"))
        return mappings, next_index
    if groups.var_positional is not None:
        mappings.append((arg, groups.var_positional, "varargs"))
    return mappings, next_index


def _map_keyword_arg(
    arg: dict[str, object],
    *,
    groups: _ParamGroups,
    assigned: set[int],
) -> list[tuple[dict[str, object], dict[str, object], str]]:
    arg_name = _coerce_str(arg.get("arg_name"))
    if arg_name is None:
        return []
    param = groups.keyword.get(arg_name)
    if param is None:
        if groups.var_keyword is None:
            return []
        return [(arg, groups.var_keyword, "varkw")]
    param_index = _coerce_int(param.get("param_index"))
    if param_index is None or param_index in assigned:
        if groups.var_keyword is None:
            return []
        return [(arg, groups.var_keyword, "varkw")]
    assigned.add(param_index)
    return [(arg, param, "keyword")]


def _map_starargs_arg(
    arg: dict[str, object],
    *,
    groups: _ParamGroups,
) -> list[tuple[dict[str, object], dict[str, object], str]]:
    if groups.var_positional is None:
        return []
    return [(arg, groups.var_positional, "varargs")]


def _map_kwargs_arg(
    arg: dict[str, object],
    *,
    groups: _ParamGroups,
) -> list[tuple[dict[str, object], dict[str, object], str]]:
    if groups.var_keyword is None:
        return []
    return [(arg, groups.var_keyword, "varkw")]


def _arg_mappings_for_arg(
    arg: dict[str, object],
    *,
    groups: _ParamGroups,
    assigned: set[int],
    pos_index: int,
) -> tuple[list[tuple[dict[str, object], dict[str, object], str]], int]:
    arg_kind = _coerce_str(arg.get("arg_kind"))
    if arg_kind == "positional":
        return _map_positional_arg(arg, groups=groups, assigned=assigned, pos_index=pos_index)
    if arg_kind == "keyword":
        return _map_keyword_arg(arg, groups=groups, assigned=assigned), pos_index
    if arg_kind == "starargs":
        return _map_starargs_arg(arg, groups=groups), pos_index
    if arg_kind == "kwargs":
        return _map_kwargs_arg(arg, groups=groups), pos_index
    return [], pos_index


def _arg_identity(arg: Mapping[str, object]) -> tuple[object, object, object, object]:
    return (
        arg.get("arg_expr_node_id"),
        arg.get("arg_ordinal"),
        arg.get("arg_kind"),
        arg.get("arg_name"),
    )


def _bound_arg_mappings(
    args: Sequence[Mapping[str, object]],
    params: Sequence[Mapping[str, object]],
) -> list[tuple[dict[str, object], dict[str, object], str]]:
    signature = _signature_from_params(params)
    if signature is None:
        return []
    ordered_args = _sorted_rows(args, key_field="arg_ordinal")
    if any(_coerce_str(arg.get("arg_kind")) in {"starargs", "kwargs"} for arg in ordered_args):
        return []
    param_by_name = {
        cast("str", _coerce_str(row.get("name"))): dict(row)
        for row in params
        if _coerce_str(row.get("name")) is not None
    }
    token_result = _arg_tokens(ordered_args)
    if token_result is None:
        return []
    tokens_by_arg, positional_tokens, keyword_tokens = token_result
    try:
        bound = signature.bind_partial(*positional_tokens, **keyword_tokens)
    except TypeError:
        return []
    mappings: list[tuple[dict[str, object], dict[str, object], str]] = []
    for param_name, value in bound.arguments.items():
        param = param_by_name.get(param_name)
        if param is None:
            continue
        _append_bound_mappings(
            value=value,
            tokens_by_arg=tokens_by_arg,
            param=param,
            mappings=mappings,
        )
    return mappings


def _arg_tokens(
    ordered_args: Sequence[Mapping[str, object]],
) -> tuple[dict[object, dict[str, object]], list[object], dict[str, object]] | None:
    tokens_by_arg: dict[object, dict[str, object]] = {}
    positional_tokens: list[object] = []
    keyword_tokens: dict[str, object] = {}
    for arg in ordered_args:
        arg_kind = _coerce_str(arg.get("arg_kind"))
        token = object()
        tokens_by_arg[token] = dict(arg)
        if arg_kind == "positional":
            positional_tokens.append(token)
            continue
        if arg_kind == "keyword":
            arg_name = _coerce_str(arg.get("arg_name"))
            if arg_name is None:
                return None
            keyword_tokens[arg_name] = token
            continue
        return None
    return tokens_by_arg, positional_tokens, keyword_tokens


def _append_bound_mappings(
    *,
    value: object,
    tokens_by_arg: Mapping[object, dict[str, object]],
    param: dict[str, object],
    mappings: list[tuple[dict[str, object], dict[str, object], str]],
) -> None:
    if isinstance(value, tuple):
        _append_token_mappings(
            tokens=value,
            tokens_by_arg=tokens_by_arg,
            param=param,
            mapping_kind="bound_varargs",
            mappings=mappings,
        )
        return
    if isinstance(value, dict):
        _append_token_mappings(
            tokens=value.values(),
            tokens_by_arg=tokens_by_arg,
            param=param,
            mapping_kind="bound_varkw",
            mappings=mappings,
        )
        return
    arg = tokens_by_arg.get(value)
    if arg is None:
        return
    arg_kind = _coerce_str(arg.get("arg_kind"))
    mapping_kind = "bound_keyword" if arg_kind == "keyword" else "bound_positional"
    mappings.append((arg, param, mapping_kind))


def _append_token_mappings(
    *,
    tokens: Iterable[object],
    tokens_by_arg: Mapping[object, dict[str, object]],
    param: dict[str, object],
    mapping_kind: str,
    mappings: list[tuple[dict[str, object], dict[str, object], str]],
) -> None:
    for item in tokens:
        arg = tokens_by_arg.get(item)
        if arg is not None:
            mappings.append((arg, param, mapping_kind))


def _assign_args_to_params(
    args: Sequence[Mapping[str, object]],
    params: Sequence[Mapping[str, object]],
) -> list[tuple[dict[str, object], dict[str, object], str]]:
    ordered_args = _sorted_rows(args, key_field="arg_ordinal")
    ordered_params = _sorted_rows(params, key_field="param_index")
    groups = _param_groups(ordered_params)
    assigned: set[int] = set()
    pos_index = 0
    mappings: list[tuple[dict[str, object], dict[str, object], str]] = []
    for arg in ordered_args:
        new_mappings, pos_index = _arg_mappings_for_arg(
            arg,
            groups=groups,
            assigned=assigned,
            pos_index=pos_index,
        )
        mappings.extend(new_mappings)
    return mappings


def _signature_from_params(
    params: Sequence[Mapping[str, object]],
) -> inspect.Signature | None:
    kind_map = {
        "POSITIONAL_ONLY": inspect.Parameter.POSITIONAL_ONLY,
        "POSITIONAL_OR_KEYWORD": inspect.Parameter.POSITIONAL_OR_KEYWORD,
        "VAR_POSITIONAL": inspect.Parameter.VAR_POSITIONAL,
        "KEYWORD_ONLY": inspect.Parameter.KEYWORD_ONLY,
        "VAR_KEYWORD": inspect.Parameter.VAR_KEYWORD,
    }
    parameters: list[inspect.Parameter] = []
    for row in _sorted_rows(params, key_field="param_index"):
        name = _coerce_str(row.get("name"))
        kind_name = _coerce_str(row.get("kind"))
        if name is None or kind_name is None:
            return None
        kind = kind_map.get(kind_name)
        if kind is None:
            return None
        parameters.append(inspect.Parameter(name, kind))
    if not parameters:
        return None
    try:
        return inspect.Signature(parameters)
    except ValueError:
        return None


def _inspect_arg_to_param_edge_row(
    *,
    arg: Mapping[str, object],
    param: Mapping[str, object],
    mapping_kind: str,
    context: _InspectArgToParamContext,
) -> dict[str, object]:
    arg_node_id = _coerce_str(arg.get("arg_expr_node_id"))
    if arg_node_id is None:
        return {}
    param_index = _coerce_int(param.get("param_index"))
    if param_index is None:
        return {}
    src_cpg_node_id = cpg_node_id(
        SYNTAX_NODES_TABLE_KEY,
        {
            "repo": context.repo,
            "commit": context.commit,
            "rel_path": context.rel_path,
            "producer": context.producer,
            "node_id": arg_node_id,
        },
    )
    dst_cpg_node_id = _inspect_signature_param_cpg_id(
        repo=context.repo,
        commit=context.commit,
        signature_id=context.signature_id,
        param_index=param_index,
    )
    extras = {
        "call_id": context.call_id,
        "signature_id": context.signature_id,
        "arg_kind": arg.get("arg_kind"),
        "arg_name": arg.get("arg_name"),
        "param_name": param.get("name"),
        "param_kind": param.get("kind"),
        "mapping_kind": mapping_kind,
    }
    ordinal = cpg_edge_ordinal(
        "graph.cpg_edges_arg_to_param_inspect",
        {
            "call_id": context.call_id,
            "signature_id": context.signature_id,
            "arg_ordinal": arg.get("arg_ordinal"),
            "param_index": param_index,
            "mapping_kind": mapping_kind,
        },
    )
    return {
        "repo": context.repo,
        "commit": context.commit,
        "src_cpg_node_id": src_cpg_node_id,
        "dst_cpg_node_id": dst_cpg_node_id,
        "edge_kind": "ARG_TO_PARAM",
        "edge_layer": "FLOW",
        "rel_path": context.rel_path,
        "ordinal": ordinal,
        "extras_json": _row_to_payload(extras),
    }


def _inspect_arg_context_for_row(
    row: Mapping[str, object],
    *,
    signature_id: str,
) -> _InspectArgToParamContext | None:
    repo = _coerce_str(row.get("repo"))
    commit = _coerce_str(row.get("commit"))
    rel_path = _coerce_str(row.get("rel_path"))
    producer = _coerce_str(row.get("producer"))
    call_id = _coerce_str(row.get("call_id"))
    if None in {repo, commit, rel_path, producer, call_id}:
        return None
    return _InspectArgToParamContext(
        repo=cast("str", repo),
        commit=cast("str", commit),
        rel_path=cast("str", rel_path),
        producer=cast("str", producer),
        call_id=cast("str", call_id),
        signature_id=signature_id,
    )


def _inspect_arg_to_param_edges_for_call(
    args: Sequence[dict[str, object]],
    *,
    signature_id: str,
    params: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    edges: list[dict[str, object]] = []
    bound_mappings = _bound_arg_mappings(args, params)
    assigned = {_arg_identity(arg) for arg, _, _ in bound_mappings}
    remaining_args = [arg for arg in args if _arg_identity(arg) not in assigned]
    fallback_mappings = _assign_args_to_params(remaining_args, params) if remaining_args else []
    mappings = [*bound_mappings, *fallback_mappings]
    for arg, param, mapping_kind in mappings:
        context = _inspect_arg_context_for_row(arg, signature_id=signature_id)
        if context is None:
            continue
        edge = _inspect_arg_to_param_edge_row(
            arg=arg,
            param=param,
            mapping_kind=mapping_kind,
            context=context,
        )
        if edge:
            edges.append(edge)
    return edges


def _inspect_arg_to_param_edges_to_rows(
    syntax_calls: pa.Table,
    syntax_call_args: pa.Table,
    inspect_objects: pa.Table,
    inspect_signatures: pa.Table,
    inspect_signature_params: pa.Table,
) -> list[dict[str, object]]:
    call_rows = _collect_rows(
        syntax_calls,
        columns=("repo", "commit", "rel_path", "producer", "call_id", "callee_text", "extras_json"),
    )
    arg_rows = _collect_rows(
        syntax_call_args,
        columns=(
            "repo",
            "commit",
            "rel_path",
            "producer",
            "call_id",
            "arg_ordinal",
            "arg_kind",
            "arg_name",
            "arg_expr_node_id",
        ),
    )
    inspect_rows = _collect_rows(
        inspect_objects,
        columns=("repo", "commit", "object_id", "module_name", "qualname"),
    )
    signature_rows = _collect_rows(
        inspect_signatures,
        columns=("repo", "commit", "signature_id", "object_id", "variant", "status"),
    )
    param_rows = _collect_rows(
        inspect_signature_params,
        columns=("repo", "commit", "signature_id", "param_index", "name", "kind", "status"),
    )
    if not call_rows or not arg_rows or not inspect_rows or not signature_rows or not param_rows:
        return []
    object_by_name = _inspect_object_by_name(inspect_rows)
    signature_by_object = _signature_by_object(signature_rows)
    params_by_signature = _params_by_signature(param_rows)
    signature_by_call = _signature_by_call(
        call_rows,
        object_by_name=object_by_name,
        signature_by_object=signature_by_object,
    )
    args_by_call = _args_by_call(arg_rows)
    edges: list[dict[str, object]] = []
    for call_key, args in args_by_call.items():
        signature_id = signature_by_call.get(call_key)
        if signature_id is None:
            continue
        params = params_by_signature.get(signature_id)
        if not params:
            continue
        edges.extend(
            _inspect_arg_to_param_edges_for_call(
                args,
                signature_id=signature_id,
                params=params,
            )
        )
    return edges


def _inspect_object_by_name(
    rows: Sequence[Mapping[str, object]],
) -> dict[tuple[str, str, str], str]:
    entries: list[tuple[tuple[str, str, str], str]] = []
    for row in rows:
        repo = _coerce_str(row.get("repo"))
        commit = _coerce_str(row.get("commit"))
        object_id = _coerce_str(row.get("object_id"))
        full_name = _inspect_full_qualname(
            _coerce_str(row.get("module_name")),
            _coerce_str(row.get("qualname")),
        )
        if None in {repo, commit, object_id, full_name}:
            continue
        repo_value = cast("str", repo)
        commit_value = cast("str", commit)
        object_id_value = cast("str", object_id)
        full_name_value = cast("str", full_name)
        entries.append(((repo_value, commit_value, full_name_value), object_id_value))
    entries.sort(key=lambda item: item[0])
    return dict(entries)


def _signature_by_object(
    rows: Sequence[Mapping[str, object]],
) -> dict[tuple[str, str, str], str]:
    signature_by_object: dict[tuple[str, str, str], str] = {}
    for row in rows:
        if not _inspect_status_ok(row.get("status")):
            continue
        variant = _coerce_str(row.get("variant"))
        if variant is not None and variant != "primary":
            continue
        repo = _coerce_str(row.get("repo"))
        commit = _coerce_str(row.get("commit"))
        object_id = _coerce_str(row.get("object_id"))
        signature_id = _coerce_str(row.get("signature_id"))
        if None in {repo, commit, object_id, signature_id}:
            continue
        signature_by_object[
            cast("str", repo),
            cast("str", commit),
            cast("str", object_id),
        ] = cast("str", signature_id)
    return signature_by_object


def _params_by_signature(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, list[dict[str, object]]]:
    params_by_signature: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if not _inspect_status_ok(row.get("status")):
            continue
        signature_id = _coerce_str(row.get("signature_id"))
        if signature_id is None:
            continue
        params_by_signature[signature_id].append(dict(row))
    for params in params_by_signature.values():
        params.sort(key=lambda item: _coerce_int(item.get("param_index")) or 0)
    return params_by_signature


def _signature_by_call(
    call_rows: Sequence[Mapping[str, object]],
    *,
    object_by_name: Mapping[tuple[str, str, str], str],
    signature_by_object: Mapping[tuple[str, str, str], str],
) -> dict[tuple[str, str, str], str]:
    signature_by_call: dict[tuple[str, str, str], str] = {}
    for row in call_rows:
        repo = _coerce_str(row.get("repo"))
        commit = _coerce_str(row.get("commit"))
        call_id = _coerce_str(row.get("call_id"))
        if None in {repo, commit, call_id}:
            continue
        repo_value = cast("str", repo)
        commit_value = cast("str", commit)
        call_id_value = cast("str", call_id)
        candidates = _call_callee_candidates(
            row.get("extras_json"),
            _coerce_str(row.get("callee_text")),
        )
        for candidate in candidates:
            object_id = object_by_name.get((repo_value, commit_value, candidate))
            if object_id is None:
                continue
            signature_id = signature_by_object.get((repo_value, commit_value, object_id))
            if signature_id is None:
                continue
            signature_by_call[repo_value, commit_value, call_id_value] = signature_id
            break
    return signature_by_call


def _args_by_call(
    rows: Sequence[Mapping[str, object]],
) -> dict[tuple[str, str, str], list[dict[str, object]]]:
    args_by_call: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        repo = _coerce_str(row.get("repo"))
        commit = _coerce_str(row.get("commit"))
        call_id = _coerce_str(row.get("call_id"))
        if None in {repo, commit, call_id}:
            continue
        args_by_call[cast("str", repo), cast("str", commit), cast("str", call_id)].append(dict(row))
    return args_by_call


def _py_inspect_signature_edges_to_rows(
    signatures: pa.Table,
    params: pa.Table,
) -> list[dict[str, object]]:
    required_signatures = {"repo", "commit", "signature_id", "object_id"}
    required_params = {"repo", "commit", "signature_id", "param_index"}
    if not required_signatures.issubset(signatures.column_names) or not required_params.issubset(
        params.column_names
    ):
        return []
    edges: list[dict[str, object]] = []
    for row in table_rows(signatures):
        repo = _coerce_str(row.get("repo"))
        commit = _coerce_str(row.get("commit"))
        signature_id = _coerce_str(row.get("signature_id"))
        object_id = _coerce_str(row.get("object_id"))
        if _has_missing(repo, commit, signature_id, object_id):
            continue
        src_cpg_node_id = cpg_node_id(
            PY_INSPECT_OBJECTS_TABLE_KEY,
            {"repo": repo, "commit": commit, "object_id": object_id},
        )
        dst_cpg_node_id = cpg_node_id(
            PY_INSPECT_SIGNATURES_TABLE_KEY,
            {"repo": repo, "commit": commit, "signature_id": signature_id},
        )
        extras = {
            "variant": row.get("variant"),
            "follow_wrapped": row.get("follow_wrapped"),
            "eval_str": row.get("eval_str"),
            "status": row.get("status"),
        }
        ordinal = cpg_edge_ordinal(
            "graph.cpg_edges_inspect_signature",
            {"signature_id": signature_id},
        )
        edges.append(
            {
                "repo": repo,
                "commit": commit,
                "src_cpg_node_id": src_cpg_node_id,
                "dst_cpg_node_id": dst_cpg_node_id,
                "edge_kind": "HAS_SIGNATURE",
                "edge_layer": "SYMBOL",
                "rel_path": None,
                "ordinal": ordinal,
                "extras_json": _row_to_payload(extras),
            }
        )
    for row in table_rows(params):
        repo = _coerce_str(row.get("repo"))
        commit = _coerce_str(row.get("commit"))
        signature_id = _coerce_str(row.get("signature_id"))
        param_index = _coerce_int(row.get("param_index"))
        if _has_missing(repo, commit, signature_id, param_index):
            continue
        src_cpg_node_id = cpg_node_id(
            PY_INSPECT_SIGNATURES_TABLE_KEY,
            {"repo": repo, "commit": commit, "signature_id": signature_id},
        )
        dst_cpg_node_id = _inspect_signature_param_cpg_id(
            repo=cast("str", repo),
            commit=cast("str", commit),
            signature_id=cast("str", signature_id),
            param_index=cast("int", param_index),
        )
        extras = {
            "param_index": param_index,
            "name": row.get("name"),
            "kind": row.get("kind"),
            "status": row.get("status"),
        }
        ordinal = cpg_edge_ordinal(
            "graph.cpg_edges_inspect_signature_param",
            {"signature_id": signature_id, "param_index": param_index},
        )
        edges.append(
            {
                "repo": repo,
                "commit": commit,
                "src_cpg_node_id": src_cpg_node_id,
                "dst_cpg_node_id": dst_cpg_node_id,
                "edge_kind": "HAS_PARAM",
                "edge_layer": "SYMBOL",
                "rel_path": None,
                "ordinal": ordinal,
                "extras_json": _row_to_payload(extras),
            }
        )
    return edges


def _inspect_ast_indices(ast_rows: Sequence[dict[str, object]]) -> _InspectAstIndex:
    ast_by_qualname: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in ast_rows:
        qualname = _coerce_str(row.get("qualname"))
        if qualname is None:
            continue
        ast_by_qualname[qualname].append(row)
    ast_by_path = _ast_nodes_by_path(ast_rows)
    ast_by_norm_path = {_normalize_path(path): rows for path, rows in ast_by_path.items()}
    ast_paths = sorted(ast_by_norm_path.keys(), key=len, reverse=True)
    return _InspectAstIndex(
        by_qualname=ast_by_qualname,
        by_norm_path=ast_by_norm_path,
        paths=ast_paths,
    )


def _inspect_sources_by_object(
    source_rows: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    sources_by_object: dict[str, dict[str, object]] = {}
    for row in source_rows:
        object_id = _coerce_str(row.get("object_id"))
        file_name = _coerce_str(row.get("file_name"))
        start_line = _coerce_int(row.get("start_line"))
        line_count = _coerce_int(row.get("line_count"))
        if _has_missing(object_id, file_name, start_line):
            continue
        start_line_value = cast("int", start_line)
        end_line = (
            start_line_value
            if line_count is None or line_count <= 0
            else start_line_value + line_count - 1
        )
        sources_by_object[cast("str", object_id)] = {
            "file_name": cast("str", file_name),
            "start_line": start_line_value,
            "end_line": end_line,
        }
    return sources_by_object


def _inspect_ast_edge_row(
    context: _InspectAstContext,
    *,
    node_hash: str,
    rel_path: object,
    extras: Mapping[str, object],
) -> dict[str, object]:
    ordinal = cpg_edge_ordinal(
        "graph.cpg_edges_inspect_ast",
        {"object_id": context.object_id, "ast_hash": node_hash},
    )
    return {
        "repo": context.repo,
        "commit": context.commit,
        "src_cpg_node_id": cpg_node_id(
            PY_INSPECT_OBJECTS_TABLE_KEY,
            {"repo": context.repo, "commit": context.commit, "object_id": context.object_id},
        ),
        "dst_cpg_node_id": cpg_node_id(AST_NODES_TABLE_KEY, {"hash": node_hash}),
        "edge_kind": "INSPECT_ANCHORS_AST",
        "edge_layer": "SYMBOL",
        "rel_path": rel_path,
        "ordinal": ordinal,
        "extras_json": _row_to_payload(extras),
    }


def _inspect_ast_edges_for_source(
    context: _InspectAstContext,
    *,
    source: Mapping[str, object],
    ast_index: _InspectAstIndex,
    seen: set[tuple[str, str]],
) -> list[dict[str, object]]:
    file_name = cast("str", source["file_name"])
    path = _best_source_path(file_name, ast_index.paths)
    if path is None:
        return []
    nodes = ast_index.by_norm_path.get(path, [])
    match = _select_ast_anchor_for_source(
        nodes,
        source_start=cast("int", source["start_line"]),
        source_end=cast("int", source["end_line"]),
    )
    if match is None:
        return []
    node, confidence, match_kind = match
    node_hash = _coerce_str(node.get("hash"))
    if node_hash is None:
        return []
    key = (context.object_id, node_hash)
    if key in seen:
        return []
    extras = {
        "match_kind": match_kind,
        "ast_kind": node.get("node_type"),
        "match_confidence": confidence,
        "source_start_line": source["start_line"],
        "source_end_line": source["end_line"],
    }
    seen.add(key)
    return [
        _inspect_ast_edge_row(
            context,
            node_hash=node_hash,
            rel_path=node.get("path"),
            extras=extras,
        )
    ]


def _inspect_ast_edges_for_qualname(
    context: _InspectAstContext,
    *,
    ast_rows: Sequence[Mapping[str, object]],
    seen: set[tuple[str, str]],
) -> list[dict[str, object]]:
    edges: list[dict[str, object]] = []
    for ast_row in ast_rows:
        node_hash = _coerce_str(ast_row.get("hash"))
        if node_hash is None:
            continue
        key = (context.object_id, node_hash)
        if key in seen:
            continue
        extras = {
            "match_kind": "QUALNAME",
            "ast_kind": ast_row.get("node_type"),
            "match_confidence": 0.6,
        }
        edges.append(
            _inspect_ast_edge_row(
                context,
                node_hash=node_hash,
                rel_path=ast_row.get("path"),
                extras=extras,
            )
        )
        seen.add(key)
    return edges


def _inspect_to_ast_edges_to_rows(
    inspect_objects: pa.Table,
    inspect_source: pa.Table,
    ast_nodes: pa.Table,
) -> list[dict[str, object]]:
    if (
        not _INSPECT_REQUIRED_COLUMNS.issubset(inspect_objects.column_names)
        or not _INSPECT_SOURCE_REQUIRED_COLUMNS.issubset(inspect_source.column_names)
        or not _INSPECT_AST_REQUIRED_COLUMNS.issubset(ast_nodes.column_names)
    ):
        return []
    ast_rows = _collect_rows(
        ast_nodes,
        columns=(
            "path",
            "hash",
            "node_type",
            "qualname",
            "lineno",
            "end_lineno",
            "decorator_start_line",
            "decorator_end_line",
        ),
    )
    if not ast_rows:
        return []
    ast_index = _inspect_ast_indices(ast_rows)
    source_rows = _collect_rows(
        inspect_source,
        columns=("object_id", "file_name", "start_line", "line_count"),
    )
    sources_by_object = _inspect_sources_by_object(source_rows)
    edges: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for row in table_rows(inspect_objects):
        repo = _coerce_str(row.get("repo"))
        commit = _coerce_str(row.get("commit"))
        object_id = _coerce_str(row.get("object_id"))
        if _has_missing(repo, commit, object_id):
            continue
        context = _InspectAstContext(
            repo=cast("str", repo),
            commit=cast("str", commit),
            object_id=cast("str", object_id),
        )
        source = sources_by_object.get(context.object_id)
        if source is not None:
            edges.extend(
                _inspect_ast_edges_for_source(
                    context,
                    source=source,
                    ast_index=ast_index,
                    seen=seen,
                )
            )
        full_qualname = _inspect_full_qualname(
            _coerce_str(row.get("module_name")),
            _coerce_str(row.get("qualname")),
        )
        if full_qualname is None:
            continue
        matches = ast_index.by_qualname.get(full_qualname)
        if not matches:
            continue
        edges.extend(
            _inspect_ast_edges_for_qualname(
                context,
                ast_rows=matches,
                seen=seen,
            )
        )
    return edges


def _inspect_to_scip_edges_to_rows(
    inspect_objects: pa.Table,
    scip_symbols: pa.Table,
) -> list[dict[str, object]]:
    required_inspect = {"repo", "commit", "object_id", "module_name", "qualname"}
    required_symbols = {"repo", "commit", "symbol", "display_name"}
    if not required_inspect.issubset(inspect_objects.column_names) or not required_symbols.issubset(
        scip_symbols.column_names
    ):
        return []
    symbols_by_key: dict[tuple[str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in table_rows(scip_symbols):
        repo = _coerce_str(row.get("repo"))
        commit = _coerce_str(row.get("commit"))
        display_name = _coerce_str(row.get("display_name"))
        if _has_missing(repo, commit, display_name):
            continue
        symbols_by_key[cast("str", repo), cast("str", commit), cast("str", display_name)].append(
            row
        )
    edges: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for row in table_rows(inspect_objects):
        repo = _coerce_str(row.get("repo"))
        commit = _coerce_str(row.get("commit"))
        object_id = _coerce_str(row.get("object_id"))
        if _has_missing(repo, commit, object_id):
            continue
        full_qualname = _inspect_full_qualname(
            _coerce_str(row.get("module_name")),
            _coerce_str(row.get("qualname")),
        )
        if full_qualname is None:
            continue
        symbol_key = (cast("str", repo), cast("str", commit), full_qualname)
        for symbol in symbols_by_key.get(symbol_key, []):
            symbol_id = _coerce_str(symbol.get("symbol"))
            if symbol_id is None:
                continue
            key = (cast("str", object_id), symbol_id)
            if key in seen:
                continue
            extras = {
                "match_kind": "QUALNAME",
                "symbol_display_name": symbol.get("display_name"),
            }
            ordinal = cpg_edge_ordinal(
                "graph.cpg_edges_inspect_symbol",
                {"object_id": object_id, "symbol": symbol_id},
            )
            edges.append(
                {
                    "repo": repo,
                    "commit": commit,
                    "src_cpg_node_id": cpg_node_id(
                        PY_INSPECT_OBJECTS_TABLE_KEY,
                        {"repo": repo, "commit": commit, "object_id": object_id},
                    ),
                    "dst_cpg_node_id": cpg_node_id(
                        SCIP_SYMBOLS_TABLE_KEY,
                        {"repo": repo, "commit": commit, "symbol": symbol_id},
                    ),
                    "edge_kind": "INSPECT_SYMBOL",
                    "edge_layer": "SYMBOL",
                    "rel_path": None,
                    "ordinal": ordinal,
                    "extras_json": _row_to_payload(extras),
                }
            )
            seen.add(key)
    return edges


def _py_inspect_class_mro_edges_to_rows(class_mro: pa.Table) -> list[dict[str, object]]:
    rows = _collect_rows(
        class_mro,
        columns=(
            "repo",
            "commit",
            "class_object_id",
            "base_object_id",
            "mro_index",
            "status",
        ),
    )
    if not rows:
        return []
    edges: list[dict[str, object]] = []
    for row in rows:
        repo = _coerce_str(row.get("repo"))
        commit = _coerce_str(row.get("commit"))
        class_object_id = _coerce_str(row.get("class_object_id"))
        base_object_id = _coerce_str(row.get("base_object_id"))
        mro_index = _coerce_int(row.get("mro_index"))
        if _has_missing(repo, commit, class_object_id, base_object_id, mro_index):
            continue
        src_cpg_node_id = cpg_node_id(
            PY_INSPECT_OBJECTS_TABLE_KEY,
            {"repo": repo, "commit": commit, "object_id": class_object_id},
        )
        dst_cpg_node_id = cpg_node_id(
            PY_INSPECT_OBJECTS_TABLE_KEY,
            {"repo": repo, "commit": commit, "object_id": base_object_id},
        )
        extras = {"mro_index": mro_index, "status": row.get("status")}
        ordinal = cpg_edge_ordinal(
            "graph.cpg_edges_inspect_class_mro",
            {
                "class_object_id": class_object_id,
                "base_object_id": base_object_id,
                "mro_index": mro_index,
            },
        )
        edges.append(
            {
                "repo": repo,
                "commit": commit,
                "src_cpg_node_id": src_cpg_node_id,
                "dst_cpg_node_id": dst_cpg_node_id,
                "edge_kind": "INHERITS",
                "edge_layer": "SYMBOL",
                "rel_path": None,
                "ordinal": ordinal,
                "extras_json": _row_to_payload(extras),
            }
        )
    return edges


def _py_inspect_class_attr_edges_to_rows(class_attrs: pa.Table) -> list[dict[str, object]]:
    rows = _collect_rows(
        class_attrs,
        columns=(
            "repo",
            "commit",
            "class_object_id",
            "attr_name",
            "attr_kind",
            "defining_object_id",
            "value_kind",
            "value_object_id",
            "desc_is_data",
            "desc_is_methoddesc",
            "desc_is_getset",
            "desc_is_member",
            "status",
        ),
    )
    if not rows:
        return []
    edges: list[dict[str, object]] = []
    for row in rows:
        context = _inspect_class_attr_context(row)
        if context is None:
            continue
        edges.extend(_inspect_class_attr_edges(context))
    return edges


def _inspect_class_attr_context(row: Mapping[str, object]) -> _InspectClassAttrContext | None:
    repo = _coerce_str(row.get("repo"))
    commit = _coerce_str(row.get("commit"))
    class_object_id = _coerce_str(row.get("class_object_id"))
    attr_name = _coerce_str(row.get("attr_name"))
    defining_object_id = _coerce_str(row.get("defining_object_id"))
    value_object_id = _coerce_str(row.get("value_object_id"))
    if _has_missing(repo, commit, class_object_id, attr_name):
        return None
    extras = {
        "attr_name": attr_name,
        "attr_kind": row.get("attr_kind"),
        "defining_object_id": defining_object_id,
        "value_kind": row.get("value_kind"),
        "desc_is_data": row.get("desc_is_data"),
        "desc_is_methoddesc": row.get("desc_is_methoddesc"),
        "desc_is_getset": row.get("desc_is_getset"),
        "desc_is_member": row.get("desc_is_member"),
        "status": row.get("status"),
    }
    src_cpg_node_id = cpg_node_id(
        PY_INSPECT_OBJECTS_TABLE_KEY,
        {"repo": repo, "commit": commit, "object_id": class_object_id},
    )
    is_descriptor = any(
        _coerce_bool(row.get(flag))
        for flag in ("desc_is_data", "desc_is_methoddesc", "desc_is_getset", "desc_is_member")
    )
    return _InspectClassAttrContext(
        repo=cast("str", repo),
        commit=cast("str", commit),
        class_object_id=cast("str", class_object_id),
        attr_name=cast("str", attr_name),
        defining_object_id=defining_object_id,
        value_object_id=value_object_id,
        extras=extras,
        src_cpg_node_id=src_cpg_node_id,
        is_descriptor=is_descriptor,
    )


def _inspect_class_attr_edge(
    context: _InspectClassAttrContext,
    *,
    target_object_id: str,
    edge_kind: str,
    ordinal_values: Mapping[str, object],
) -> dict[str, object]:
    dst_cpg_node_id = cpg_node_id(
        PY_INSPECT_OBJECTS_TABLE_KEY,
        {"repo": context.repo, "commit": context.commit, "object_id": target_object_id},
    )
    ordinal = cpg_edge_ordinal("graph.cpg_edges_inspect_class_attr", ordinal_values)
    return {
        "repo": context.repo,
        "commit": context.commit,
        "src_cpg_node_id": context.src_cpg_node_id,
        "dst_cpg_node_id": dst_cpg_node_id,
        "edge_kind": edge_kind,
        "edge_layer": "SYMBOL",
        "rel_path": None,
        "ordinal": ordinal,
        "extras_json": _row_to_payload(context.extras),
    }


def _inspect_class_attr_edges(context: _InspectClassAttrContext) -> list[dict[str, object]]:
    edges: list[dict[str, object]] = []
    if context.value_object_id is not None:
        edges.append(
            _inspect_class_attr_edge(
                context,
                target_object_id=context.value_object_id,
                edge_kind="DECLARES_ATTR",
                ordinal_values={
                    "class_object_id": context.class_object_id,
                    "attr_name": context.attr_name,
                    "value_object_id": context.value_object_id,
                    "edge_kind": "DECLARES_ATTR",
                },
            )
        )
    if (
        context.defining_object_id is not None
        and context.defining_object_id != context.class_object_id
    ):
        edges.append(
            _inspect_class_attr_edge(
                context,
                target_object_id=context.defining_object_id,
                edge_kind="OVERRIDES",
                ordinal_values={
                    "class_object_id": context.class_object_id,
                    "attr_name": context.attr_name,
                    "defining_object_id": context.defining_object_id,
                    "edge_kind": "OVERRIDES",
                },
            )
        )
    if context.value_object_id is not None and context.is_descriptor:
        edges.append(
            _inspect_class_attr_edge(
                context,
                target_object_id=context.value_object_id,
                edge_kind="DESCRIPTOR",
                ordinal_values={
                    "class_object_id": context.class_object_id,
                    "attr_name": context.attr_name,
                    "value_object_id": context.value_object_id,
                    "edge_kind": "DESCRIPTOR",
                },
            )
        )
    return edges


def _runtime_state_extras(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "state_kind": row.get("state_kind"),
        "state": row.get("state"),
        "object_kind": row.get("object_kind"),
        "frame_file": row.get("frame_file"),
        "frame_module": row.get("frame_module"),
        "frame_line": row.get("frame_line"),
        "frame_offset": row.get("frame_offset"),
        "status": row.get("status"),
    }


def _runtime_state_has_state_edges(runtime_state: pa.Table) -> pa.Table:
    edges: list[dict[str, object]] = []
    for row in table_rows(runtime_state):
        repo = _coerce_str(row.get("repo"))
        commit = _coerce_str(row.get("commit"))
        object_id = _coerce_str(row.get("object_id"))
        frame_object_id = _coerce_str(row.get("frame_object_id"))
        if _has_missing(repo, commit, object_id, frame_object_id):
            continue
        extras = _runtime_state_extras(row)
        ordinal = cpg_edge_ordinal(
            "graph.cpg_edges_inspect_runtime_state",
            {
                "object_id": object_id,
                "state_kind": row.get("state_kind"),
                "frame_object_id": frame_object_id,
            },
        )
        edges.append(
            {
                "repo": repo,
                "commit": commit,
                "src_cpg_node_id": cpg_node_id(
                    PY_INSPECT_OBJECTS_TABLE_KEY,
                    {"repo": repo, "commit": commit, "object_id": object_id},
                ),
                "dst_cpg_node_id": cpg_node_id(
                    PY_INSPECT_OBJECTS_TABLE_KEY,
                    {"repo": repo, "commit": commit, "object_id": frame_object_id},
                ),
                "edge_kind": "HAS_STATE",
                "edge_layer": "FLOW",
                "rel_path": None,
                "ordinal": ordinal,
                "extras_json": _row_to_payload(extras),
            }
        )
    table, _ = table_for_rows(CPG_EDGES_TABLE_KEY, edges)
    return table


def _runtime_state_frame_name(row: Mapping[str, object]) -> str | None:
    return _coerce_str(row.get("frame_code_qualname")) or _coerce_str(row.get("frame_code_name"))


def _runtime_state_context(row: Mapping[str, object]) -> _RuntimeStateInstrContext | None:
    repo = _coerce_str(row.get("repo"))
    commit = _coerce_str(row.get("commit"))
    object_id = _coerce_str(row.get("object_id"))
    if _has_missing(repo, commit, object_id):
        return None
    frame_module = _coerce_str(row.get("frame_module"))
    frame_name = _runtime_state_frame_name(row)
    frame_offset = _coerce_int(row.get("frame_offset"))
    if _has_missing(frame_module, frame_name, frame_offset):
        return None
    frame_offset_value = cast("int", frame_offset)
    if frame_offset_value < 0:
        return None
    frame_qualpath = f"{frame_module}::{frame_name}"
    return _RuntimeStateInstrContext(
        repo=cast("str", repo),
        commit=cast("str", commit),
        object_id=cast("str", object_id),
        state_kind=_coerce_str(row.get("state_kind")),
        frame_qualpath=frame_qualpath,
        frame_offset=frame_offset_value,
        extras=_runtime_state_extras(row),
    )


def _runtime_state_instr_edges(
    runtime_state: pa.Table,
    code_units: pa.Table,
    instructions: pa.Table,
) -> pa.Table:
    code_unit_rows = _collect_rows(
        code_units,
        columns=("repo", "commit", "rel_path", "code_unit_id", "qualpath"),
    )
    instr_rows = _collect_rows(
        instructions,
        columns=("repo", "commit", "rel_path", "code_unit_id", "instr_id", "offset"),
    )
    if not code_unit_rows or not instr_rows:
        return empty_table_for_table(CPG_EDGES_TABLE_KEY)
    units_by_key = _runtime_units_index(code_unit_rows)
    instr_by_key = _runtime_instr_index(instr_rows)
    edges: list[dict[str, object]] = []
    for row in table_rows(runtime_state):
        edges.extend(_runtime_state_instr_edge_rows(row, units_by_key, instr_by_key))
    table, _ = table_for_rows(CPG_EDGES_TABLE_KEY, edges)
    return table


def _runtime_units_index(
    code_unit_rows: Sequence[Mapping[str, object]],
) -> dict[tuple[str, str, str], list[tuple[str, str]]]:
    units_by_key: dict[tuple[str, str, str], list[tuple[str, str]]] = defaultdict(list)
    for row in code_unit_rows:
        repo = _coerce_str(row.get("repo"))
        commit = _coerce_str(row.get("commit"))
        qualpath = _coerce_str(row.get("qualpath"))
        rel_path = _coerce_str(row.get("rel_path"))
        code_unit_id = _coerce_str(row.get("code_unit_id"))
        if _has_missing(repo, commit, qualpath, rel_path, code_unit_id):
            continue
        units_by_key[cast("str", repo), cast("str", commit), cast("str", qualpath)].append(
            (cast("str", rel_path), cast("str", code_unit_id))
        )
    return units_by_key


def _runtime_instr_index(
    instr_rows: Sequence[Mapping[str, object]],
) -> dict[tuple[str, str, str, str, int], list[str]]:
    instr_by_key: dict[tuple[str, str, str, str, int], list[str]] = defaultdict(list)
    for row in instr_rows:
        repo = _coerce_str(row.get("repo"))
        commit = _coerce_str(row.get("commit"))
        rel_path = _coerce_str(row.get("rel_path"))
        code_unit_id = _coerce_str(row.get("code_unit_id"))
        instr_id = _coerce_str(row.get("instr_id"))
        offset = _coerce_int(row.get("offset"))
        if _has_missing(repo, commit, rel_path, code_unit_id, instr_id, offset):
            continue
        key = (
            cast("str", repo),
            cast("str", commit),
            cast("str", rel_path),
            cast("str", code_unit_id),
            cast("int", offset),
        )
        instr_by_key[key].append(cast("str", instr_id))
    return instr_by_key


def _runtime_state_instr_edge_rows(
    row: Mapping[str, object],
    units_by_key: Mapping[tuple[str, str, str], Sequence[tuple[str, str]]],
    instr_by_key: Mapping[tuple[str, str, str, str, int], Sequence[str]],
) -> list[dict[str, object]]:
    context = _runtime_state_context(row)
    if context is None:
        return []
    unit_matches = units_by_key.get((context.repo, context.commit, context.frame_qualpath))
    if not unit_matches:
        return []
    return _runtime_state_edges_for_units(context, unit_matches, instr_by_key)


def _runtime_state_edges_for_units(
    context: _RuntimeStateInstrContext,
    unit_matches: Sequence[tuple[str, str]],
    instr_by_key: Mapping[tuple[str, str, str, str, int], Sequence[str]],
) -> list[dict[str, object]]:
    edge_kind = "TRACEBACK_AT_INSTR" if context.state_kind == "traceback" else "FRAME_AT_INSTR"
    edges: list[dict[str, object]] = []
    for rel_path, code_unit_id in unit_matches:
        instr_matches = instr_by_key.get(
            (context.repo, context.commit, rel_path, code_unit_id, context.frame_offset)
        )
        if not instr_matches:
            continue
        edges.extend(
            _runtime_state_edges_for_instr(
                context,
                edge_kind=edge_kind,
                rel_path=rel_path,
                code_unit_id=code_unit_id,
                instr_matches=instr_matches,
            )
        )
    return edges


def _runtime_state_edges_for_instr(
    context: _RuntimeStateInstrContext,
    *,
    edge_kind: str,
    rel_path: str,
    code_unit_id: str,
    instr_matches: Sequence[str],
) -> list[dict[str, object]]:
    edges: list[dict[str, object]] = []
    for instr_id in instr_matches:
        ordinal = cpg_edge_ordinal(
            "graph.cpg_edges_inspect_runtime_state",
            {
                "object_id": context.object_id,
                "state_kind": context.state_kind,
                "instr_id": instr_id,
                "frame_offset": context.frame_offset,
            },
        )
        edges.append(
            {
                "repo": context.repo,
                "commit": context.commit,
                "src_cpg_node_id": cpg_node_id(
                    PY_INSPECT_OBJECTS_TABLE_KEY,
                    {
                        "repo": context.repo,
                        "commit": context.commit,
                        "object_id": context.object_id,
                    },
                ),
                "dst_cpg_node_id": _instruction_cpg_id(
                    repo=context.repo,
                    commit=context.commit,
                    rel_path=rel_path,
                    code_unit_id=code_unit_id,
                    instr_id=instr_id,
                ),
                "edge_kind": edge_kind,
                "edge_layer": "FLOW",
                "rel_path": rel_path,
                "ordinal": ordinal,
                "extras_json": _row_to_payload(context.extras),
            }
        )
    return edges


def _py_inspect_runtime_state_edges_to_table(
    runtime_state: pa.Table,
    code_units: pa.Table,
    instructions: pa.Table,
) -> pa.Table:
    required_state = {
        "repo",
        "commit",
        "object_id",
        "object_kind",
        "state_kind",
        "state",
        "frame_object_id",
        "frame_file",
        "frame_module",
        "frame_code_qualname",
        "frame_code_name",
        "frame_line",
        "frame_offset",
        "status",
    }
    required_units = {"repo", "commit", "rel_path", "code_unit_id", "qualpath"}
    required_instr = {"repo", "commit", "rel_path", "code_unit_id", "instr_id", "offset"}
    if (
        not required_state.issubset(runtime_state.column_names)
        or not required_units.issubset(code_units.column_names)
        or not required_instr.issubset(instructions.column_names)
    ):
        return empty_table_for_table(CPG_EDGES_TABLE_KEY)
    has_state_edges = _runtime_state_has_state_edges(runtime_state)
    instr_edges = _runtime_state_instr_edges(runtime_state, code_units, instructions)
    tables = [table for table in (has_state_edges, instr_edges) if table.num_rows > 0]
    if not tables:
        return empty_table_for_table(CPG_EDGES_TABLE_KEY)
    combined = concat_tables_unified(tables)
    return _select_edge_columns(combined)


def _py_inspect_unwrap_edges_to_rows(unwrap_hops: pa.Table) -> list[dict[str, object]]:
    rows = _collect_rows(
        unwrap_hops,
        columns=(
            "repo",
            "commit",
            "root_object_id",
            "hop",
            "object_id",
            "has_wrapped",
            "has_signature_override",
            "stop_reason",
        ),
    )
    if not rows:
        return []
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        repo = _coerce_str(row.get("repo"))
        commit = _coerce_str(row.get("commit"))
        root_object_id = _coerce_str(row.get("root_object_id"))
        hop = _coerce_int(row.get("hop"))
        obj_id = _coerce_str(row.get("object_id"))
        if _has_missing(repo, commit, root_object_id, hop, obj_id):
            continue
        grouped[cast("str", repo), cast("str", commit), cast("str", root_object_id)].append(row)
    edges: list[dict[str, object]] = []
    for (repo, commit, root_object_id), items in grouped.items():
        items.sort(key=lambda item: _coerce_int(item.get("hop")) or 0)
        for idx in range(len(items) - 1):
            src_id = _coerce_str(items[idx].get("object_id"))
            dst_id = _coerce_str(items[idx + 1].get("object_id"))
            if _has_missing(src_id, dst_id):
                continue
            edge_kind = "DECORATES" if idx == 0 else "WRAPS"
            src_pk = {"repo": repo, "commit": commit, "object_id": src_id}
            dst_pk = {"repo": repo, "commit": commit, "object_id": dst_id}
            extras = {
                "root_object_id": root_object_id,
                "hop": items[idx].get("hop"),
                "has_wrapped": items[idx].get("has_wrapped"),
                "has_signature_override": items[idx].get("has_signature_override"),
                "stop_reason": items[idx].get("stop_reason"),
                "edge_kind": edge_kind,
            }
            hop_value = _coerce_int(items[idx].get("hop"))
            ordinal = cpg_edge_ordinal(
                "graph.cpg_edges_inspect_wraps",
                {
                    "root_object_id": root_object_id,
                    "hop": hop_value,
                    "edge_kind": edge_kind,
                },
            )
            edges.append(
                {
                    "repo": repo,
                    "commit": commit,
                    "src_cpg_node_id": cpg_node_id(PY_INSPECT_OBJECTS_TABLE_KEY, src_pk),
                    "dst_cpg_node_id": cpg_node_id(PY_INSPECT_OBJECTS_TABLE_KEY, dst_pk),
                    "edge_kind": edge_kind,
                    "edge_layer": "SYMBOL",
                    "rel_path": None,
                    "ordinal": ordinal,
                    "extras_json": _row_to_payload(extras),
                }
            )
    return edges


def _inspect_signature_param_cpg_id(
    *,
    repo: str,
    commit: str,
    signature_id: str,
    param_index: int,
) -> int:
    pk_values = {
        "repo": repo,
        "commit": commit,
        "signature_id": signature_id,
        "param_index": param_index,
    }
    return cpg_node_id(PY_INSPECT_SIGNATURE_PARAMS_TABLE_KEY, pk_values)


def _instruction_cpg_id(
    *,
    repo: str,
    commit: str,
    rel_path: str,
    code_unit_id: str,
    instr_id: str,
) -> int:
    pk_values = {
        "repo": repo,
        "commit": commit,
        "rel_path": rel_path,
        "code_unit_id": code_unit_id,
        "instr_id": instr_id,
    }
    return cpg_node_id(PY_BC_INSTRUCTIONS_TABLE_KEY, pk_values)


def _normalize_path(value: str) -> str:
    return value.replace("\\", "/")


def _best_source_path(file_name: str, ast_paths: Sequence[str]) -> str | None:
    normalized = _normalize_path(file_name)
    best_match: str | None = None
    for path in ast_paths:
        if normalized.endswith(path) and (best_match is None or len(path) > len(best_match)):
            best_match = path
    return best_match


def _ast_span_for_source(node: Mapping[str, object]) -> tuple[int | None, int | None]:
    decorator_start = _coerce_int(node.get("decorator_start_line"))
    start_line = decorator_start if decorator_start is not None else _coerce_int(node.get("lineno"))
    end_line = _coerce_int(node.get("end_lineno")) or start_line
    return start_line, end_line


def _select_ast_anchor_for_source(
    nodes: list[dict[str, object]],
    *,
    source_start: int | None,
    source_end: int | None,
) -> tuple[dict[str, object], float, str] | None:
    if source_start is None or source_end is None:
        return None
    candidates: list[tuple[int, dict[str, object]]] = []
    for node in nodes:
        start_line, end_line = _ast_span_for_source(node)
        if start_line is None or end_line is None:
            continue
        if source_start < start_line or source_end > end_line:
            continue
        span_len = end_line - start_line
        candidates.append((span_len, node))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    node = candidates[0][1]
    start_line, end_line = _ast_span_for_source(node)
    if start_line is None or end_line is None:
        return None
    confidence = 0.9
    if source_start == start_line and source_end == end_line:
        confidence = 0.95
    return node, confidence, "SOURCE_SPAN"


def _ast_nodes_by_path(
    ast_rows: Sequence[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    by_path: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in ast_rows:
        rel_path = _coerce_str(row.get("path"))
        if rel_path is None:
            continue
        by_path[rel_path].append(row)
    return by_path


def _select_edge_columns(table: pa.Table) -> pa.Table:
    columns = columns_for_table_key(CPG_EDGES_TABLE_KEY)
    if columns is None:
        return table
    return ensure_table_columns(table, columns)


def _record_diagnostics(
    diagnostics: dict[str, object] | None,
    key: str,
    *,
    expected_edges: int,
    produced_edges: int,
) -> None:
    if diagnostics is None:
        return
    dropped = max(expected_edges - produced_edges, 0)
    diagnostics[key] = OverlayEdgeDiagnostics(
        expected_edges=expected_edges,
        produced_edges=produced_edges,
        dropped_edges=dropped,
    )


def _collect_rows(frame: pa.Table, *, columns: Sequence[str]) -> list[dict[str, object]]:
    if not set(columns).issubset(frame.column_names):
        return []
    return [{column: row.get(column) for column in columns} for row in table_rows(frame)]


def _row_to_payload(values: Mapping[str, object]) -> bytes:
    payload = encode_payload(dict(values))
    if payload is None:
        msg = "Expected payload encoding to return bytes"
        raise ValueError(msg)
    return payload


def _coerce_str(value: object) -> str | None:
    if isinstance(value, str):
        return value
    return None


def _coerce_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    return None


def _coerce_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _has_missing(*values: object) -> bool:
    return any(value is None for value in values)


_INSPECT_REQUIRED_COLUMNS = frozenset({"repo", "commit", "object_id", "module_name", "qualname"})
_INSPECT_SOURCE_REQUIRED_COLUMNS = frozenset({"object_id", "file_name", "start_line", "line_count"})
_INSPECT_AST_REQUIRED_COLUMNS = frozenset(
    {
        "hash",
        "qualname",
        "node_type",
        "path",
        "lineno",
        "end_lineno",
        "decorator_start_line",
        "decorator_end_line",
    }
)


__all__ = [
    "InspectArgToParamInputs",
    "OverlayEdgeDiagnostics",
    "cpg2_edges__inspect_arg_to_param",
    "cpg2_edges__inspect_to_ast",
    "cpg2_edges__inspect_to_scip",
    "cpg2_edges__py_inspect_class_attr",
    "cpg2_edges__py_inspect_class_mro",
    "cpg2_edges__py_inspect_runtime_state",
    "cpg2_edges__py_inspect_signature",
    "cpg2_edges__py_inspect_unwrap",
]
