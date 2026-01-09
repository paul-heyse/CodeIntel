"""Call wiring relation sources for CPG interprocedural edges."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Literal

import pyarrow as pa
from intervaltree import IntervalTree

from codeintel.build.graphs.assembly import (
    drop_table_columns as _drop_table_columns,
)
from codeintel.build.graphs.assembly import (
    empty_reader,
    stable_decimal_id,
    table_to_reader,
    tabular_to_table,
)
from codeintel.build.graphs.assembly import (
    rename_table_columns as _rename_table_columns,
)
from codeintel.build.graphs.assembly import (
    select_table_columns as _select_table_columns,
)
from codeintel.build.graphs.assembly import (
    table_rows as _table_rows,
)
from codeintel.build.schemas.service import get_schema_service
from codeintel.build.tabular.arrow_ops import (
    AlignmentReport,
    dedupe_table_for_table,
    emit_alignment_report,
    normalize_table_for_join,
)
from codeintel.build.tabular.compute_columns import empty_table as _empty_table
from codeintel.build.tabular.compute_helpers import cast_array
from codeintel.build.tabular.expr_vocab import E, Expression
from codeintel.build.tabular.finalize_ops import (
    FinalizeResult,
    finalize_join_keys,
    finalize_reader,
    finalize_spec_for_table,
    finalize_table,
    record_join_precheck_errors,
)
from codeintel.build.tabular.plan_ops import HashJoinSpec
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.columnar.arrowdsl import ExecutionPlan, join_safe_projection
from codeintel.core.columnar.conversion import reader_to_table
from codeintel.core.columnar.execution_context import resolve_execution_context
from codeintel.core.columnar.iter import iter_tuples
from codeintel.core.columnar.plan_builder import TablePlanOptions, build_table_plan
from codeintel.core.columnar.plan_kernels import (
    ExplodeSpec,
    GroupedRollupSpec,
    StableDedupeSpec,
    explode_edges_for_join,
    grouped_rollup_table,
    stable_dedupe_with_ties,
)
from codeintel.core.columnar.plan_ops import Plan
from codeintel.core.columnar.rows import table_for_rows
from codeintel.core.intervals.span_resolver import MatchKind, SpanResolver
from codeintel.core.schemas.primitives import resolve_join_safe_columns
from codeintel.core.serialization.payload import PayloadValue, decode_payload, encode_payload

if TYPE_CHECKING:
    from codeintel.core.columnar.kernels import SortKey

CALL_WIRING_TARGET_NAME = "call_wiring"
CPG_CALL_TARGETS_TABLE_KEY = "graph.cpg_call_targets"
CPG_CALL_CANDIDATES_TABLE_KEY = "graph.cpg_call_candidates"
CPG_CALL_EDGES_TABLE_KEY = "graph.cpg_edges_calls"
CPG_ARG_TO_PARAM_EDGES_TABLE_KEY = "graph.cpg_edges_arg_to_param"
CPG_RET_TO_CALL_EDGES_TABLE_KEY = "graph.cpg_edges_ret_to_call"
CFG_BLOCKS_TABLE_KEY = "graph.cfg_blocks"

_GOID_ARROW_TYPE = pa.decimal128(38, 0)
_GOID_CAST_TYPE = "decimal128(38,0)"
_BLOCK_ID_ARROW_TYPE = pa.string()
_ROLE_DEFINITION = 0x1
_OVERLAP_CONFIDENCE_THRESHOLD = 3
_IMPLICIT_CALL_ID_SEP = "#"
_MAX_ANCESTOR_DEPTH = 6
_AUGOP_PATTERN = re.compile(r"([+\-*/%&|^]=)")
_ASSIGN_CALL_PATTERN = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?::[^=]+)?=\s*([A-Za-z_][A-Za-z0-9_]*)\s*\("
)
_BASE_CALL_PATTERN = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\(.*\)$")

_TARGET_ROLE_PRIMARY = "primary"
_ASCENDING: Literal["ascending"] = "ascending"
_TARGET_ROLE_INIT = "init"

_BINDING_CONSTRUCTOR = "constructor"
_BINDING_INIT = "init"
_BINDING_BOUND_METHOD = "bound_method"
_BINDING_UNBOUND_METHOD = "unbound_method"
_BINDING_CLASSMETHOD = "classmethod"
_BINDING_STATICMETHOD = "staticmethod"
_BINDING_PROPERTY_GET = "property_get"
_BINDING_PROPERTY_SET = "property_set"
_BINDING_DESCRIPTOR_GET = "descriptor_get"
_BINDING_DESCRIPTOR_SET = "descriptor_set"
_BINDING_DESCRIPTOR_SET_AUG = "descriptor_set_augassign"
_BINDING_OPERATOR_DUNDER = "operator_dunder"
_BINDING_UNKNOWN = "unknown"

_ORIGIN_SYNTAX_CALL = "syntax_call"
_ORIGIN_DESCRIPTOR = "descriptor_desugar"
_ORIGIN_AUGASSIGN = "augassign_desugar"

_CALL_KIND_EXPLICIT = "explicit"
_CALL_KIND_IMPLICIT_GET = "implicit_descriptor_get"
_CALL_KIND_IMPLICIT_SET = "implicit_descriptor_set"
_CALL_KIND_IMPLICIT_SET_AUG = "implicit_descriptor_set_augassign"

_AUGOP_CALL_KIND = {
    "+=": "implicit_augassign_iadd",
    "-=": "implicit_augassign_isub",
    "*=": "implicit_augassign_imul",
    "/=": "implicit_augassign_itruediv",
    "%=": "implicit_augassign_imod",
    "&=": "implicit_augassign_iand",
    "|=": "implicit_augassign_ior",
    "^=": "implicit_augassign_ixor",
}


@dataclass(frozen=True)
class _OccurrenceCandidate:
    start: int
    end: int
    symbol: str
    roles: int


@dataclass(frozen=True, slots=True)
class _DefInfo:
    def_id: str
    def_kind: str
    name: str
    class_name: str | None
    scip_symbol: str | None
    goid_h128: int | None
    syntax_node_id: str | None
    method_kind: str
    container_def_id: str | None
    start_line: int | None
    end_line: int | None
    start_byte: int | None
    end_byte: int | None


@dataclass(frozen=True, slots=True)
class _ClassInfo:
    def_id: str
    name: str
    start_line: int | None
    end_line: int | None
    start_byte: int | None
    end_byte: int | None


@dataclass(frozen=True, slots=True)
class _DescriptorInfo:
    class_name: str
    class_def_id: str
    get_def: _DefInfo | None
    set_def: _DefInfo | None


@dataclass(frozen=True, slots=True)
class _SpanRange:
    start_byte: int | None
    end_byte: int | None
    start_line: int
    end_line: int


@dataclass(frozen=True, slots=True)
class _CallSite:
    rel_path: str
    start_line: int
    start_col: int
    end_line: int
    end_col: int


@dataclass(frozen=True, slots=True)
class _CallTargetRecordContext:
    repo: str
    commit: str
    rel_path: str
    call_id: str
    call_node_id: str | None
    callee_symbol: str | None
    callee_def: _DefInfo | None
    target_role: str
    binding_kind: str
    origin: str
    call_kind: str | None
    augop: str | None
    resolution_kind: str
    confidence: float
    candidate_count: int
    extras: PayloadValue | bytes | bytearray | memoryview | None


@dataclass(frozen=True, slots=True)
class _DescriptorAssignmentContext:
    class_tree: IntervalTree
    class_line_spans: Sequence[tuple[int | None, int | None, str]]
    func_tree: IntervalTree
    func_line_spans: Sequence[tuple[int | None, int | None]]
    descriptor_classes: set[str]


@dataclass(frozen=True, slots=True)
class _ExplicitCallRow:
    repo: str
    commit: str
    rel_path: str
    call_id: str
    call_node_id: str | None
    callee_symbol: str | None
    callee_text: str | None
    resolution_kind: str
    confidence: float
    candidate_count: int
    extras: PayloadValue | bytes | bytearray | memoryview | None


@dataclass(frozen=True, slots=True)
class _SpanIndex:
    class_tree: IntervalTree
    class_line_spans: Sequence[tuple[int | None, int | None, str]]
    func_tree: IntervalTree
    func_line_spans: Sequence[tuple[int | None, int | None]]
    class_names: set[str]
    descriptor_classes: set[str]


@dataclass(frozen=True, slots=True)
class _ImplicitResolutionContext:
    catalog: _DefCatalog
    descriptor_assignments: Mapping[str, Mapping[str, str]]
    property_classes_by_attr: Mapping[str, set[str]]
    descriptor_classes_by_attr: Mapping[str, set[str]]
    class_names: set[str]


@dataclass(frozen=True, slots=True)
class _AugassignResolution:
    operator: str
    class_name: str
    attr_text: str
    prop_get: _DefInfo | None
    prop_set: _DefInfo | None
    assigned: str | None
    descriptor: _DescriptorInfo | None


@dataclass(frozen=True, slots=True)
class _AttributeAccess:
    repo: str
    commit: str
    rel_path: str
    node_id: str
    text: str
    attribute: str
    ctx: str | None
    start_line: int
    start_col: int
    end_line: int
    end_col: int
    start_byte: int | None
    end_byte: int | None


@dataclass(frozen=True, slots=True)
class _AugAssignAccess:
    repo: str
    commit: str
    rel_path: str
    node_id: str
    text: str
    start_line: int
    start_col: int
    end_line: int
    end_col: int
    start_byte: int | None
    end_byte: int | None


@dataclass(frozen=True, slots=True)
class _DefCatalog:
    defs_by_symbol: dict[str, _DefInfo]
    defs_by_id: dict[str, _DefInfo]
    classes_by_id: dict[str, _ClassInfo]
    classes_by_name: dict[str, _ClassInfo]
    init_by_class_id: dict[str, _DefInfo]
    property_get: dict[tuple[str, str], _DefInfo]
    property_set: dict[tuple[str, str], _DefInfo]
    descriptors: dict[str, _DescriptorInfo]


class _DefCatalogBuilder:
    def __init__(self) -> None:
        self.defs_by_symbol: dict[str, _DefInfo] = {}
        self.defs_by_id: dict[str, _DefInfo] = {}
        self.classes_by_id: dict[str, _ClassInfo] = {}
        self.classes_by_name: dict[str, _ClassInfo] = {}
        self.init_by_class_id: dict[str, _DefInfo] = {}
        self.property_get: dict[tuple[str, str], _DefInfo] = {}
        self.property_set: dict[tuple[str, str], _DefInfo] = {}
        self.descriptor_get: dict[str, _DefInfo] = {}
        self.descriptor_set: dict[str, _DefInfo] = {}
        self.pending_def_rows: list[tuple[_DefInfo, str | None, str | None]] = []

    def add_row(self, row: Mapping[str, object]) -> None:
        def_id = row.get("def_id")
        def_kind = row.get("def_kind")
        name = row.get("name")
        if (
            not isinstance(def_id, str)
            or not isinstance(def_kind, str)
            or not isinstance(name, str)
        ):
            return
        info, setter_name, container_def_id = _extract_def_info(row)
        if def_kind == "class":
            class_info = _ClassInfo(
                def_id=info.def_id,
                name=info.name,
                start_line=info.start_line,
                end_line=info.end_line,
                start_byte=info.start_byte,
                end_byte=info.end_byte,
            )
            self.classes_by_id[class_info.def_id] = class_info
            self.classes_by_name[class_info.name] = class_info
        self.pending_def_rows.append((info, setter_name, container_def_id))

    def _resolve_class_name(self, container_def_id: str | None) -> str | None:
        if container_def_id is None:
            return None
        class_info = self.classes_by_id.get(container_def_id)
        return class_info.name if class_info is not None else None

    @staticmethod
    def _normalize_method_kind(class_name: str | None, method_kind: str) -> str:
        if class_name is not None:
            return method_kind
        if method_kind in {
            "instance_method",
            _BINDING_CLASSMETHOD,
            _BINDING_STATICMETHOD,
            _BINDING_PROPERTY_GET,
            _BINDING_PROPERTY_SET,
        }:
            return "function"
        return method_kind

    def _record_descriptor(self, enriched: _DefInfo, setter_name: str | None) -> None:
        if enriched.class_name is None:
            return
        if enriched.method_kind == _BINDING_PROPERTY_GET:
            self.property_get[enriched.class_name, enriched.name] = enriched
        if enriched.method_kind == _BINDING_PROPERTY_SET:
            prop_name = setter_name or enriched.name
            self.property_set[enriched.class_name, prop_name] = enriched
        if enriched.name == "__init__" and enriched.container_def_id is not None:
            self.init_by_class_id[enriched.container_def_id] = enriched
        if enriched.name == "__get__":
            self.descriptor_get[enriched.class_name] = enriched
        if enriched.name == "__set__":
            self.descriptor_set[enriched.class_name] = enriched

    def _add_enriched_def(
        self, info: _DefInfo, setter_name: str | None, container_def_id: str | None
    ) -> None:
        class_name = self._resolve_class_name(container_def_id)
        method_kind = self._normalize_method_kind(class_name, info.method_kind)
        enriched = _DefInfo(
            def_id=info.def_id,
            def_kind=info.def_kind,
            name=info.name,
            class_name=class_name,
            scip_symbol=info.scip_symbol,
            goid_h128=info.goid_h128,
            syntax_node_id=info.syntax_node_id,
            method_kind=method_kind,
            container_def_id=container_def_id,
            start_line=info.start_line,
            end_line=info.end_line,
            start_byte=info.start_byte,
            end_byte=info.end_byte,
        )
        self.defs_by_id[enriched.def_id] = enriched
        if enriched.scip_symbol is not None:
            self.defs_by_symbol[enriched.scip_symbol] = enriched
        self._record_descriptor(enriched, setter_name)

    def finalize(self) -> _DefCatalog:
        for info, setter_name, container_def_id in self.pending_def_rows:
            self._add_enriched_def(info, setter_name, container_def_id)

        descriptors: dict[str, _DescriptorInfo] = {}
        for class_name, class_info in self.classes_by_name.items():
            get_def = self.descriptor_get.get(class_name)
            set_def = self.descriptor_set.get(class_name)
            if get_def is None and set_def is None:
                continue
            descriptors[class_name] = _DescriptorInfo(
                class_name=class_name,
                class_def_id=class_info.def_id,
                get_def=get_def,
                set_def=set_def,
            )

        return _DefCatalog(
            defs_by_symbol=self.defs_by_symbol,
            defs_by_id=self.defs_by_id,
            classes_by_id=self.classes_by_id,
            classes_by_name=self.classes_by_name,
            init_by_class_id=self.init_by_class_id,
            property_get=self.property_get,
            property_set=self.property_set,
            descriptors=descriptors,
        )


def _score_occurrence(candidate: _OccurrenceCandidate, callee_end: int) -> tuple[int, int, int]:
    is_def = 1 if (candidate.roles & _ROLE_DEFINITION) else 0
    span_len = candidate.end - candidate.start
    dist = abs(callee_end - candidate.end)
    return is_def, span_len, dist


def _pick_best_symbol(
    candidates: list[_OccurrenceCandidate],
    callee_end: int,
) -> tuple[str | None, float, list[str]]:
    if not candidates:
        return None, 0.0, []
    best = min(candidates, key=lambda item: _score_occurrence(item, callee_end))
    confidence = 1.0
    if best.roles & _ROLE_DEFINITION:
        confidence *= 0.4
    if len(candidates) > _OVERLAP_CONFIDENCE_THRESHOLD:
        confidence *= 0.7
    symbols = sorted({candidate.symbol for candidate in candidates})
    return best.symbol, confidence, symbols


def _stable_id(*parts: object) -> str:
    payload = {"parts": list(parts)}
    value = stable_decimal_id(payload, digest_size=16)
    return f"{value:032x}"


def _normalize_decorators(raw: object) -> tuple[str, ...]:
    if not isinstance(raw, Sequence):
        return ()
    return tuple(item for item in raw if isinstance(item, str) and item)


def _decorator_setter_name(decorators: Sequence[str]) -> str | None:
    for dec in decorators:
        if not dec.endswith(".setter"):
            continue
        base = dec[: -len(".setter")]
        if not base:
            continue
        return base.split(".")[-1]
    return None


def _method_kind(
    decorators: Sequence[str],
    *,
    has_class_container: bool,
) -> tuple[str, str | None]:
    if "classmethod" in decorators:
        return _BINDING_CLASSMETHOD, None
    if "staticmethod" in decorators:
        return _BINDING_STATICMETHOD, None
    setter_name = _decorator_setter_name(decorators)
    if setter_name is not None:
        return _BINDING_PROPERTY_SET, setter_name
    if "property" in decorators:
        return _BINDING_PROPERTY_GET, None
    if has_class_container:
        return "instance_method", None
    return "function", None


def _extras_struct(row: Mapping[str, object], key: str) -> Mapping[str, object] | None:
    value = row.get(key)
    if isinstance(value, Mapping):
        return dict(value)
    return None


def _extras_kv_value(value: object) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def _extras_kv_from_mapping(values: Mapping[object, object]) -> dict[str, str] | None:
    if not values:
        return None
    extras: dict[str, str] = {}
    for key, item in values.items():
        if item is None:
            continue
        extras[str(key)] = _extras_kv_value(item)
    return extras or None


def _extras_kv_from_payload(value: object) -> dict[str, str] | None:
    decoded = decode_payload(value)
    if decoded is None:
        return None
    if isinstance(decoded, Mapping):
        return _extras_kv_from_mapping(decoded)
    return {"value": _extras_kv_value(decoded)}


def _coerce_extras_kv(value: object) -> dict[str, str] | None:
    if isinstance(value, Mapping):
        return _extras_kv_from_mapping(value)
    return None


def _extract_def_info(row: Mapping[str, object]) -> tuple[_DefInfo, str | None, str | None]:
    extras = _extras_struct(row, "extras") or {}
    container_def_id = extras.get("container_def_id")
    if not isinstance(container_def_id, str):
        container_def_id = None
    decorators = _normalize_decorators(extras.get("decorators"))
    has_class_container = container_def_id is not None
    method_kind, setter_name = _method_kind(decorators, has_class_container=has_class_container)
    return (
        _DefInfo(
            def_id=str(row["def_id"]),
            def_kind=str(row["def_kind"]),
            name=str(row["name"]),
            class_name=None,
            scip_symbol=_coerce_str(row.get("scip_symbol")),
            goid_h128=_coerce_int(row.get("goid_h128")),
            syntax_node_id=_coerce_str(row.get("syntax_node_id")),
            method_kind=method_kind,
            container_def_id=container_def_id,
            start_line=_coerce_int(row.get("start_line")),
            end_line=_coerce_int(row.get("end_line")),
            start_byte=_coerce_int(row.get("start_byte")),
            end_byte=_coerce_int(row.get("end_byte")),
        ),
        setter_name,
        container_def_id,
    )


def _coerce_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, Decimal):
        return int(value)
    return None


def _coerce_str(value: object) -> str | None:
    if isinstance(value, str):
        return value
    return None


def _coerce_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _coerce_payload_value(
    value: object,
) -> PayloadValue | bytes | bytearray | memoryview | None:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return value
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, Sequence) and not isinstance(value, str):
        return list(value)
    return None


def _pick_int(primary: int | None, fallback: int | None) -> int | None:
    return primary if primary is not None else fallback


def _extras_descriptor_obj_is_none(value: object) -> bool | None:
    if isinstance(value, Mapping):
        raw = value.get("descriptor_obj_is_none")
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, str):
            normalized = raw.strip().lower()
            if normalized in {"true", "false"}:
                return normalized == "true"
            if normalized in {"1", "0"}:
                return normalized == "1"
    return None


def _explicit_binding_kind(def_info: _DefInfo, callee_text: str | None) -> str:
    direct_map = {
        _BINDING_CLASSMETHOD: _BINDING_CLASSMETHOD,
        _BINDING_STATICMETHOD: _BINDING_STATICMETHOD,
        _BINDING_PROPERTY_GET: _BINDING_PROPERTY_GET,
        _BINDING_PROPERTY_SET: _BINDING_PROPERTY_SET,
    }
    if def_info.method_kind in direct_map:
        return direct_map[def_info.method_kind]
    if def_info.method_kind != "instance_method":
        return _BINDING_UNKNOWN
    if callee_text and def_info.class_name:
        prefix = f"{def_info.class_name}."
        if callee_text.startswith(prefix):
            return _BINDING_UNBOUND_METHOD
    return _BINDING_BOUND_METHOD


def _build_def_catalog(defs_rows: Sequence[Mapping[str, object]]) -> _DefCatalog:
    builder = _DefCatalogBuilder()
    for row in defs_rows:
        builder.add_row(row)
    return builder.finalize()


def _string_list(value: object) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(item for item in value if isinstance(item, str))
    return ()


def _emit_alignment_report_from_finalize(result: FinalizeResult) -> None:
    if result.alignment.num_rows == 0:
        return
    columns = [
        "table_key",
        "target_name",
        "missing_columns",
        "extra_columns",
        "coerced_columns",
        "row_count",
    ]
    values = next(iter_tuples(table_to_reader(result.alignment), columns=columns), None)
    if values is None:
        return
    (
        table_key_value,
        target_name_value,
        missing_columns_value,
        extra_columns_value,
        coerced_columns_value,
        row_count_value,
    ) = values
    report = AlignmentReport(
        table_key=table_key_value if isinstance(table_key_value, str) else "",
        target_name=target_name_value if isinstance(target_name_value, str) else None,
        missing_columns=_string_list(missing_columns_value),
        extra_columns=_string_list(extra_columns_value),
        coerced_columns=_string_list(coerced_columns_value),
        row_count=row_count_value if isinstance(row_count_value, int) else None,
    )
    emit_alignment_report(report)


def _key_fields_for_table(table_key: str) -> tuple[str, ...]:
    try:
        schema = get_schema_service().get_table_schema(table_key)
    except (KeyError, RuntimeError, TypeError):
        return ()
    if schema is None or not schema.primary_key:
        return ()
    return tuple(schema.primary_key)


def _order_by_for_table(table_key: str) -> tuple[SortKey, ...]:
    key_fields = _key_fields_for_table(table_key)
    if not key_fields:
        return ()
    return tuple((field, _ASCENDING) for field in key_fields)


def _table_to_reader(table_key: str, table: pa.Table) -> pa.Table:
    result = finalize_table(
        table,
        spec=finalize_spec_for_table(
            table_key,
            mode="strict",
            key_fields=_key_fields_for_table(table_key),
            order_by=_order_by_for_table(table_key),
            emit_artifacts=True,
            target_name=CALL_WIRING_TARGET_NAME,
        ),
    )
    _emit_alignment_report_from_finalize(result)
    return result.good


def _cast_table_column(
    table: pa.Table,
    column_name: str,
    target_type: pa.DataType,
) -> pa.Table:
    index = table.schema.get_field_index(column_name)
    if index < 0:
        return table
    column = table.column(index)
    if column.type == target_type:
        return table
    if pa.types.is_null(column.type):
        casted = pa.nulls(table.num_rows, type=target_type)
    else:
        casted = cast_array(column, target_type, safe=False)
    return table.set_column(index, column_name, casted)


def _call_candidate_extras_kv(row: Mapping[str, object]) -> dict[str, str] | None:
    base: dict[str, str] = {}
    for key in ("binding_kind", "target_role", "call_kind", "origin", "augop"):
        value = row.get(key)
        if value is None:
            continue
        base[key] = _extras_kv_value(value)
    extra = _coerce_extras_kv(row.get("extras_kv"))
    if not base and not extra:
        return None
    merged = dict(base)
    if extra:
        merged.update(extra)
    return merged


def _rel_path_key(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, tuple) and value and isinstance(value[0], str):
        return value[0]
    return None


def _build_occurrence_resolver(
    *,
    rel_path: str,
    occ_rows: Sequence[Mapping[str, object]] | None,
) -> SpanResolver[_OccurrenceCandidate]:
    resolver = SpanResolver.for_bytes(path_normalizer=lambda value: value)
    if not occ_rows:
        return resolver
    spans: list[tuple[str, int, int, _OccurrenceCandidate]] = []
    for row in occ_rows:
        symbol = row.get("scip_symbol")
        start = row.get("start_byte")
        end = row.get("end_byte")
        roles = _coerce_int(row.get("roles")) or 0
        if not isinstance(symbol, str) or not isinstance(start, int) or not isinstance(end, int):
            continue
        if end <= start:
            continue
        spans.append(
            (
                rel_path,
                start,
                end,
                _OccurrenceCandidate(start, end, symbol, roles),
            )
        )
    resolver.add_spans(spans)
    return resolver


def _resolution_kind(match_kind: MatchKind) -> str:
    mapping = {
        "EXACT": "scip_exact",
        "POINT": "scip_point",
        "POINT_ADJACENT": "scip_point_adjacent",
        "CONTAINS": "scip_contains",
        "OVERLAP": "scip_overlap",
        "NONE": "scip_none",
    }
    return mapping.get(match_kind, "scip_none")


def _call_target_row(
    call_row: Mapping[str, object],
    *,
    rel_path: str,
    resolver: SpanResolver[_OccurrenceCandidate],
) -> dict[str, object]:
    call_id = call_row.get("call_id")
    callee_start = call_row.get("callee_start_byte")
    callee_end = call_row.get("callee_end_byte")
    if (
        not isinstance(call_id, str)
        or not isinstance(callee_start, int)
        or not isinstance(callee_end, int)
    ):
        return {
            "repo": call_row.get("repo"),
            "commit": call_row.get("commit"),
            "rel_path": rel_path,
            "call_id": call_id,
            "call_node_id": call_row.get("call_node_id"),
            "callee_text": call_row.get("callee_text"),
            "callee_symbol": None,
            "resolution_kind": "scip_none",
            "confidence": 0.0,
            "candidate_count": 0,
            "extras": encode_payload(None),
        }
    candidates, match_kind = resolver.resolve_candidates(
        rel_path,
        callee_start,
        callee_end,
        allow_adjacent_point=True,
    )
    symbol, confidence, candidate_symbols = _pick_best_symbol(candidates, callee_end)
    resolution_kind = _resolution_kind(match_kind)
    candidate_count = len(candidates)
    extras = {"candidate_symbols": candidate_symbols} if candidate_symbols else None
    return {
        "repo": call_row.get("repo"),
        "commit": call_row.get("commit"),
        "rel_path": rel_path,
        "call_id": call_id,
        "call_node_id": call_row.get("call_node_id"),
        "callee_text": call_row.get("callee_text"),
        "callee_symbol": symbol,
        "resolution_kind": resolution_kind,
        "confidence": confidence,
        "candidate_count": candidate_count,
        "extras": encode_payload(extras),
    }


def _resolve_call_targets(
    calls: Sequence[Mapping[str, object]],
    occurrences: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    out_rows: list[dict[str, object]] = []
    calls_by_path: dict[str, list[Mapping[str, object]]] = {}
    occs_by_path: dict[str, list[Mapping[str, object]]] = {}
    for row in calls:
        rel_path = _rel_path_key(row.get("rel_path"))
        if rel_path is None:
            continue
        calls_by_path.setdefault(rel_path, []).append(row)
    for row in occurrences:
        rel_path = _rel_path_key(row.get("rel_path"))
        if rel_path is None:
            continue
        occs_by_path.setdefault(rel_path, []).append(row)

    for rel_path, calls_rows in calls_by_path.items():
        resolver = _build_occurrence_resolver(
            rel_path=rel_path, occ_rows=occs_by_path.get(rel_path)
        )
        out_rows.extend(
            [
                _call_target_row(call_row, rel_path=rel_path, resolver=resolver)
                for call_row in calls_rows
            ]
        )

    return out_rows


def _ast_nodes_from_extras(extras: Mapping[str, object] | None) -> list[Mapping[str, object]]:
    if extras is None:
        return []
    nodes = extras.get("ast_nodes")
    if not isinstance(nodes, list):
        return []
    return [node for node in nodes if isinstance(node, Mapping)]


def _has_ast_kind(ast_nodes: Sequence[Mapping[str, object]], kind: str) -> bool:
    return any(node.get("ast_kind") == kind for node in ast_nodes)


def _select_attribute_payload(
    ast_nodes: Sequence[Mapping[str, object]],
) -> Mapping[str, object] | None:
    for node in ast_nodes:
        if node.get("ast_kind") == "Attribute" and node.get("match_kind") == "exact":
            return node
    for node in ast_nodes:
        if node.get("ast_kind") == "Attribute":
            return node
    return None


def _name_nodes_from_ast(
    ast_nodes: Sequence[Mapping[str, object]],
) -> list[tuple[int, int, str, str, str | None]]:
    name_nodes: list[tuple[int, int, str, str, str | None]] = []
    for node in ast_nodes:
        kind = _coerce_str(node.get("ast_kind"))
        if kind is None or kind not in {"Name", "Attribute"}:
            continue
        start = _coerce_int(node.get("ast_start_byte"))
        end = _coerce_int(node.get("ast_end_byte"))
        if start is None or end is None:
            continue
        identifier = _coerce_str(node.get("identifier"))
        if identifier is None:
            identifier = _coerce_str(node.get("attribute"))
        if identifier is None:
            continue
        ctx = _coerce_str(node.get("ctx"))
        name_nodes.append((start, end, kind, identifier, ctx))
    return name_nodes


def _lhs_name_node(
    name_nodes: list[tuple[int, int, str, str, str | None]],
) -> tuple[int, int, str, str, str | None] | None:
    if not name_nodes:
        return None
    name_nodes.sort(key=lambda item: (item[0], item[1], item[2]))
    return next((node for node in name_nodes if node[4] == "store"), name_nodes[0])


def _call_nodes_after(
    ast_nodes: Sequence[Mapping[str, object]],
    after_end: int,
) -> list[tuple[int, int]]:
    call_nodes: list[tuple[int, int]] = []
    for node in ast_nodes:
        if node.get("ast_kind") != "Call":
            continue
        start = _coerce_int(node.get("ast_start_byte"))
        end = _coerce_int(node.get("ast_end_byte"))
        if start is None or end is None or start < after_end:
            continue
        call_nodes.append((start, end))
    return call_nodes


def _rhs_from_call(
    name_nodes: Sequence[tuple[int, int, str, str, str | None]],
    *,
    lhs: tuple[int, int, str, str, str | None],
    call_start: int,
    call_end: int,
) -> str | None:
    call_candidates = [
        node for node in name_nodes if node != lhs and node[0] == call_start and node[1] <= call_end
    ]
    if not call_candidates:
        return None
    call_candidates.sort(key=lambda item: (0 if item[2] == "Attribute" else 1, item[1] - item[0]))
    return call_candidates[0][3]


def _assignment_from_ast_nodes(
    ast_nodes: Sequence[Mapping[str, object]],
) -> tuple[str, str] | None:
    name_nodes = _name_nodes_from_ast(ast_nodes)
    lhs = _lhs_name_node(name_nodes)
    if lhs is None:
        return None
    call_nodes = _call_nodes_after(ast_nodes, lhs[1])
    if not call_nodes:
        return None
    call_nodes.sort(key=lambda item: (item[0], item[1]))
    call_start, call_end = call_nodes[0]
    rhs = _rhs_from_call(name_nodes, lhs=lhs, call_start=call_start, call_end=call_end)
    if rhs is None:
        return None
    return lhs[3], rhs


def _row_identity(row: Mapping[str, object]) -> tuple[str, str, str, str] | None:
    repo = _coerce_str(row.get("repo"))
    commit = _coerce_str(row.get("commit"))
    rel_path = _coerce_str(row.get("rel_path"))
    node_id = _coerce_str(row.get("node_id"))
    if repo is None or commit is None or rel_path is None or node_id is None:
        return None
    return repo, commit, rel_path, node_id


def _attribute_payload_from_row(row: Mapping[str, object]) -> Mapping[str, object] | None:
    extras = _extras_struct(row, "extras")
    if extras is None:
        return None
    ast_nodes = _ast_nodes_from_extras(extras)
    if not ast_nodes or _has_ast_kind(ast_nodes, "AugAssign"):
        return None
    return _select_attribute_payload(ast_nodes)


def _attribute_span(
    row: Mapping[str, object],
    attr_payload: Mapping[str, object],
) -> tuple[int, int, int, int, int | None, int | None] | None:
    start_line = _pick_int(
        _coerce_int(attr_payload.get("ast_start_line")),
        _coerce_int(row.get("start_line")),
    )
    end_line = _pick_int(
        _coerce_int(attr_payload.get("ast_end_line")),
        _coerce_int(row.get("end_line")),
    )
    start_col = _pick_int(
        _coerce_int(attr_payload.get("ast_start_col_utf8")),
        _coerce_int(row.get("start_col")),
    )
    end_col = _pick_int(
        _coerce_int(attr_payload.get("ast_end_col_utf8")),
        _coerce_int(row.get("end_col")),
    )
    if start_line is None or end_line is None or start_col is None or end_col is None:
        return None
    start_byte = _pick_int(
        _coerce_int(attr_payload.get("ast_start_byte")),
        _coerce_int(row.get("start_byte")),
    )
    end_byte = _pick_int(
        _coerce_int(attr_payload.get("ast_end_byte")),
        _coerce_int(row.get("end_byte")),
    )
    return start_line, start_col, end_line, end_col, start_byte, end_byte


def _attribute_access_from_row(row: Mapping[str, object]) -> _AttributeAccess | None:
    identity = _row_identity(row)
    if identity is None:
        return None
    repo, commit, rel_path, node_id = identity
    attr_payload = _attribute_payload_from_row(row)
    if attr_payload is None:
        return None
    attribute = _coerce_str(attr_payload.get("attribute"))
    if attribute is None or not attribute:
        return None
    text = _coerce_str(row.get("text_preview"))
    if text is None or not text.strip():
        return None
    span = _attribute_span(row, attr_payload)
    if span is None:
        return None
    start_line, start_col, end_line, end_col, start_byte, end_byte = span
    return _AttributeAccess(
        repo=repo,
        commit=commit,
        rel_path=rel_path,
        node_id=node_id,
        text=text,
        attribute=attribute,
        ctx=_coerce_str(attr_payload.get("ctx")),
        start_line=start_line,
        start_col=start_col,
        end_line=end_line,
        end_col=end_col,
        start_byte=start_byte,
        end_byte=end_byte,
    )


def _extract_attribute_accesses(
    syntax_nodes: Sequence[Mapping[str, object]],
) -> list[_AttributeAccess]:
    accesses: list[_AttributeAccess] = []
    for row in syntax_nodes:
        access = _attribute_access_from_row(row)
        if access is not None:
            accesses.append(access)
    return accesses


def _extract_augassigns(
    syntax_nodes: Sequence[Mapping[str, object]],
) -> list[_AugAssignAccess]:
    records: list[_AugAssignAccess] = []
    for row in syntax_nodes:
        repo = row.get("repo")
        commit = row.get("commit")
        rel_path = row.get("rel_path")
        node_id = row.get("node_id")
        if (
            not isinstance(repo, str)
            or not isinstance(commit, str)
            or not isinstance(rel_path, str)
            or not isinstance(node_id, str)
        ):
            continue
        extras = _extras_struct(row, "extras")
        if extras is None:
            continue
        ast_nodes = _ast_nodes_from_extras(extras)
        if not ast_nodes or not _has_ast_kind(ast_nodes, "AugAssign"):
            continue
        text = row.get("text_preview")
        if not isinstance(text, str) or not text.strip():
            continue
        start_line = _coerce_int(row.get("start_line"))
        end_line = _coerce_int(row.get("end_line"))
        start_col = _coerce_int(row.get("start_col"))
        end_col = _coerce_int(row.get("end_col"))
        if start_line is None or end_line is None or start_col is None or end_col is None:
            continue
        records.append(
            _AugAssignAccess(
                repo=repo,
                commit=commit,
                rel_path=rel_path,
                node_id=node_id,
                text=text,
                start_line=start_line,
                start_col=start_col,
                end_line=end_line,
                end_col=end_col,
                start_byte=_coerce_int(row.get("start_byte")),
                end_byte=_coerce_int(row.get("end_byte")),
            )
        )
    return records


def _build_span_tree(
    spans: Iterable[tuple[int | None, int | None, str]],
) -> IntervalTree:
    entries = [
        (start, end, label)
        for start, end, label in spans
        if start is not None and end is not None and end > start
    ]
    if not entries:
        return IntervalTree()
    return IntervalTree.from_tuples(entries)


def _class_for_span(
    class_tree: IntervalTree,
    class_line_spans: Sequence[tuple[int | None, int | None, str]],
    span: _SpanRange,
) -> str | None:
    if span.start_byte is not None and span.end_byte is not None:
        if span.start_byte == span.end_byte:
            candidates = list(class_tree.at(span.start_byte))
        else:
            candidates = [
                interval
                for interval in class_tree.overlap(span.start_byte, span.end_byte)
                if interval.begin <= span.start_byte and interval.end >= span.end_byte
            ]
        if candidates:
            chosen = min(candidates, key=lambda item: item.end - item.begin)
            if isinstance(chosen.data, str):
                return chosen.data
    for span_start, span_end, class_name in class_line_spans:
        if span_start is None or span_end is None:
            continue
        if span_start <= span.start_line <= span_end and span.end_line <= span_end:
            return class_name
    return None


def _is_within_function_span(
    func_tree: IntervalTree,
    func_line_spans: Sequence[tuple[int | None, int | None]],
    span: _SpanRange,
) -> bool:
    if span.start_byte is not None and span.end_byte is not None:
        if span.start_byte == span.end_byte:
            overlaps = list(func_tree.at(span.start_byte))
        else:
            overlaps = [
                interval
                for interval in func_tree.overlap(span.start_byte, span.end_byte)
                if interval.begin <= span.start_byte and interval.end >= span.end_byte
            ]
        if overlaps:
            return True
    for span_start, span_end in func_line_spans:
        if span_start is None or span_end is None:
            continue
        if span_start <= span.start_line <= span_end and span.end_line <= span_end:
            return True
    return False


def _descriptor_assignment_candidate(
    row: Mapping[str, object],
    context: _DescriptorAssignmentContext,
) -> tuple[str, str, str] | None:
    extras = _extras_struct(row, "extras")
    ast_nodes = _ast_nodes_from_extras(extras) if extras is not None else []
    text = _coerce_str(row.get("text_preview"))
    start_line = _coerce_int(row.get("start_line"))
    end_line = _coerce_int(row.get("end_line"))
    if start_line is None or end_line is None:
        return None
    if ast_nodes and not (
        _has_ast_kind(ast_nodes, "Assign") or _has_ast_kind(ast_nodes, "AnnAssign")
    ):
        return None
    span = _SpanRange(
        start_byte=_coerce_int(row.get("start_byte")),
        end_byte=_coerce_int(row.get("end_byte")),
        start_line=start_line,
        end_line=end_line,
    )
    if _is_within_function_span(context.func_tree, context.func_line_spans, span):
        return None
    class_name = _class_for_span(context.class_tree, context.class_line_spans, span)
    assignment = _assignment_from_ast_nodes(ast_nodes) if ast_nodes else None
    if assignment is None:
        assignment = _assignment_from_text(text)
    if class_name is None or assignment is None:
        return None
    lhs_name, rhs_name = assignment
    if rhs_name not in context.descriptor_classes:
        return None
    return class_name, lhs_name, rhs_name


def _assignment_from_text(text: str | None) -> tuple[str, str] | None:
    if text is None or not text.strip():
        return None
    match = _ASSIGN_CALL_PATTERN.match(text.strip())
    if match is None:
        return None
    lhs_name, rhs_name = match.groups()
    return lhs_name, rhs_name


def _parse_descriptor_assignments(
    syntax_nodes: Sequence[Mapping[str, object]],
    context: _DescriptorAssignmentContext,
) -> dict[str, dict[str, str]]:
    assignments: dict[str, dict[str, str]] = {}
    for row in syntax_nodes:
        candidate = _descriptor_assignment_candidate(row, context)
        if candidate is None:
            continue
        class_name, lhs_name, rhs_name = candidate
        assignments.setdefault(class_name, {})[lhs_name] = rhs_name
    return assignments


def _split_attribute_text(text: str) -> tuple[str, str] | None:
    cleaned = text.strip()
    if "." not in cleaned:
        return None
    base, attr = cleaned.rsplit(".", maxsplit=1)
    base = base.strip()
    attr = attr.strip()
    if not base or not attr:
        return None
    return base, attr


def _base_class_from_text(base_text: str, class_names: set[str]) -> tuple[str | None, bool]:
    if base_text in class_names:
        return base_text, True
    match = _BASE_CALL_PATTERN.match(base_text)
    if match is not None:
        candidate = match.group(1)
        if candidate in class_names:
            return candidate, False
    return None, False


def _resolve_class_for_attribute(
    attr_name: str,
    base_text: str,
    *,
    class_names: set[str],
    property_classes: Mapping[str, set[str]],
    descriptor_classes: Mapping[str, set[str]],
) -> tuple[str | None, bool]:
    class_name, is_class_access = _base_class_from_text(base_text, class_names)
    if class_name is not None:
        return class_name, is_class_access
    candidates: set[str] = set()
    candidates.update(property_classes.get(attr_name, set()))
    candidates.update(descriptor_classes.get(attr_name, set()))
    if len(candidates) == 1:
        return next(iter(candidates)), False
    return None, False


def _implicit_call_id(site: _CallSite, *, suffix: str) -> str:
    return _stable_id(
        "implicit_call",
        site.rel_path,
        site.start_line,
        site.start_col,
        site.end_line,
        site.end_col,
        suffix,
    )


def _call_site_from_access(access: _AttributeAccess | _AugAssignAccess) -> _CallSite:
    return _CallSite(
        rel_path=access.rel_path,
        start_line=access.start_line,
        start_col=access.start_col,
        end_line=access.end_line,
        end_col=access.end_col,
    )


def _dedupe_block_table(
    table: pa.Table,
    *,
    output_column: str,
) -> pa.Table:
    if table.num_rows == 0:
        return _empty_table(["function_goid_h128", output_column])
    try:
        grouped = grouped_rollup_table(
            table,
            spec=GroupedRollupSpec(
                keys=("function_goid_h128",),
                aggregates=[("block_id", "min", None, "block_id_min")],
            ),
        )
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError, TypeError, ValueError):
        return _dedupe_block_rows(table, output_column=output_column)
    block_column = "block_id_min"
    if block_column not in grouped.column_names:
        return _dedupe_block_rows(table, output_column=output_column)
    return grouped.rename_columns(["function_goid_h128", output_column])


def _table_from_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    columns: Sequence[str],
) -> pa.Table:
    if not rows:
        return pa.table({name: [] for name in columns})
    data: dict[str, list[object]] = {name: [] for name in columns}
    for row in rows:
        for name in columns:
            data[name].append(row.get(name))
    return pa.table(data)


def _dedupe_block_rows(table: pa.Table, *, output_column: str) -> pa.Table:
    required = {"function_goid_h128", "block_id"}
    if table.num_rows == 0 or not required.issubset(set(table.column_names)):
        return _empty_table(["function_goid_h128", output_column])
    projected = table.select(["function_goid_h128", "block_id"])
    deduped = stable_dedupe_with_ties(
        projected,
        spec=StableDedupeSpec(
            key_columns=("function_goid_h128",),
            order_by=(("block_id", "ascending"),),
            hash_tiebreaker=True,
        ),
    )
    return deduped.rename_columns(["function_goid_h128", output_column])


def _entry_blocks(cfg_blocks: pa.Table) -> pa.Table:
    required = {"kind", "function_goid_h128", "block_id"}
    if cfg_blocks.num_rows == 0 or not required.issubset(set(cfg_blocks.column_names)):
        return _empty_table(["function_goid_h128", "entry_block_id"])
    try:
        filtered_plan = build_table_plan(
            table=cfg_blocks,
            options=TablePlanOptions(filter_expr=E.field("kind") == E.scalar("entry")),
        )
        filtered = _plan_to_table(filtered_plan)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError, TypeError, ValueError):
        return _empty_table(["function_goid_h128", "entry_block_id"])
    if filtered.num_rows == 0:
        return _empty_table(["function_goid_h128", "entry_block_id"])
    table = filtered.select(["function_goid_h128", "block_id"])
    return _dedupe_block_table(table, output_column="entry_block_id")


def _exit_blocks(cfg_blocks: pa.Table) -> pa.Table:
    required = {"kind", "function_goid_h128", "block_id"}
    if cfg_blocks.num_rows == 0 or not required.issubset(set(cfg_blocks.column_names)):
        return _empty_table(["function_goid_h128", "exit_block_id"])
    try:
        filtered_plan = build_table_plan(
            table=cfg_blocks,
            options=TablePlanOptions(filter_expr=E.field("kind") == E.scalar("exit")),
        )
        filtered = _plan_to_table(filtered_plan)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError, TypeError, ValueError):
        return _empty_table(["function_goid_h128", "exit_block_id"])
    if filtered.num_rows == 0:
        return _empty_table(["function_goid_h128", "exit_block_id"])
    table = filtered.select(["function_goid_h128", "block_id"])
    return _dedupe_block_table(table, output_column="exit_block_id")


def _project_with_goid_cast(table: pa.Table, *, key: str) -> dict[str, Expression]:
    exprs: dict[str, Expression] = {}
    for name in table.column_names:
        if name == key:
            exprs[name] = E.cast(E.field(name), _GOID_CAST_TYPE)
        else:
            exprs[name] = E.field(name)
    return exprs


def _join_safe_allowlist(table_key: str | None) -> tuple[str, ...]:
    if table_key is None:
        return ()
    try:
        schema = get_schema_service().get_table_schema(table_key)
    except RuntimeError:
        return ()
    return resolve_join_safe_columns(schema)


def _hash_join_block_targets(
    targets: pa.Table,
    blocks: pa.Table,
    *,
    right_key: str,
    output_column: str,
    left_columns: Sequence[str] | None = None,
) -> pa.Table:
    if targets.num_rows == 0 or blocks.num_rows == 0:
        return targets
    if "callee_goid_h128" not in targets.column_names:
        return targets
    if right_key not in blocks.column_names or output_column not in blocks.column_names:
        return targets
    resolved_columns = list(targets.column_names) if left_columns is None else list(left_columns)
    if "callee_goid_h128" not in resolved_columns:
        resolved_columns.append("callee_goid_h128")
    resolved_columns = [name for name in resolved_columns if name in targets.column_names]
    if not resolved_columns:
        return targets
    target_allowlist = _join_safe_allowlist(CPG_CALL_TARGETS_TABLE_KEY)
    block_allowlist = _join_safe_allowlist(CFG_BLOCKS_TABLE_KEY)
    target_columns = (*resolved_columns, *target_allowlist)
    block_columns = (right_key, output_column, *block_allowlist)
    targets = join_safe_projection(targets, allowed_columns=target_columns)
    blocks = join_safe_projection(blocks, allowed_columns=block_columns)
    targets = normalize_table_for_join(targets, allowed_columns=target_columns)
    blocks = normalize_table_for_join(blocks, allowed_columns=block_columns)
    left_precheck = finalize_join_keys(
        targets,
        required_non_null=["callee_goid_h128"],
        key_fields=["callee_goid_h128"],
        stage="join_precheck",
    )
    record_join_precheck_errors(
        left_precheck,
        table_key=CPG_CALL_TARGETS_TABLE_KEY,
        target_name=CALL_WIRING_TARGET_NAME,
        join_keys=["callee_goid_h128"],
    )
    targets = left_precheck.good
    right_precheck = finalize_join_keys(
        blocks,
        required_non_null=[right_key],
        key_fields=[right_key],
        stage="join_precheck",
    )
    record_join_precheck_errors(
        right_precheck,
        table_key=CFG_BLOCKS_TABLE_KEY,
        target_name=CALL_WIRING_TARGET_NAME,
        join_keys=[right_key],
    )
    blocks = right_precheck.good
    left_exprs = _project_with_goid_cast(targets, key="callee_goid_h128")
    right_exprs = {
        right_key: E.cast(E.field(right_key), _GOID_CAST_TYPE),
        output_column: E.field(output_column),
    }
    left_plan = build_table_plan(
        table=targets,
        options=TablePlanOptions(
            projection=left_exprs,
            filter_expr=E.is_valid("callee_goid_h128"),
        ),
    )
    right_plan = build_table_plan(
        table=blocks,
        options=TablePlanOptions(
            projection=right_exprs,
            filter_expr=E.is_valid(right_key),
        ),
    )
    joined = left_plan.hash_join(
        right=right_plan,
        spec=HashJoinSpec(
            left_keys=["callee_goid_h128"],
            right_keys=[right_key],
            how="left outer",
            left_output=list(left_exprs.keys()),
            right_output=[output_column],
        ),
    )
    sort_base = ("repo", "commit", "rel_path", "call_id", "callee_goid_h128")
    sort_keys = [key for key in sort_base if key in left_exprs]
    if sort_keys:
        joined = joined.order_by(sort_keys=[(key, "ascending") for key in sort_keys])
    return _plan_to_table(joined)


def _plan_to_table(plan: Plan) -> pa.Table:
    execution_ctx = resolve_execution_context(None)
    reader = ExecutionPlan.from_plan(plan).to_reader(ctx=execution_ctx)
    return reader_to_table(reader)


def _call_target_record(context: _CallTargetRecordContext) -> dict[str, object]:
    extras_kv = _extras_kv_from_payload(context.extras)
    return {
        "repo": context.repo,
        "commit": context.commit,
        "rel_path": context.rel_path,
        "call_id": context.call_id,
        "call_node_id": context.call_node_id,
        "callee_symbol": context.callee_symbol,
        "callee_def_id": context.callee_def.def_id if context.callee_def is not None else None,
        "callee_def_node_id": context.callee_def.syntax_node_id
        if context.callee_def is not None
        else None,
        "callee_goid_h128": context.callee_def.goid_h128
        if context.callee_def is not None
        else None,
        "callee_entry_block_id": None,
        "callee_exit_block_id": None,
        "target_role": context.target_role,
        "binding_kind": context.binding_kind,
        "origin": context.origin,
        "call_kind": context.call_kind,
        "augop": context.augop,
        "resolution_kind": context.resolution_kind,
        "confidence": context.confidence,
        "candidate_count": context.candidate_count,
        "extras": None,
        "extras_kv": extras_kv,
    }


def _parse_explicit_call_row(row: Mapping[str, object]) -> _ExplicitCallRow | None:
    repo = _coerce_str(row.get("repo"))
    commit = _coerce_str(row.get("commit"))
    rel_path = _coerce_str(row.get("rel_path"))
    call_id = _coerce_str(row.get("call_id"))
    if repo is None or commit is None or rel_path is None or call_id is None:
        return None
    call_node_id = _coerce_str(row.get("call_node_id"))
    callee_symbol = _coerce_str(row.get("callee_symbol"))
    callee_text = _coerce_str(row.get("callee_text"))
    resolution_kind = _coerce_str(row.get("resolution_kind")) or "scip_none"
    confidence = _coerce_float(row.get("confidence")) or 0.0
    candidate_count = _coerce_int(row.get("candidate_count")) or 0
    extras = _coerce_payload_value(row.get("extras"))
    return _ExplicitCallRow(
        repo=repo,
        commit=commit,
        rel_path=rel_path,
        call_id=call_id,
        call_node_id=call_node_id,
        callee_symbol=callee_symbol,
        callee_text=callee_text,
        resolution_kind=resolution_kind,
        confidence=confidence,
        candidate_count=candidate_count,
        extras=extras,
    )


def _explicit_rows_for_entry(
    entry: _ExplicitCallRow,
    catalog: _DefCatalog,
) -> list[dict[str, object]]:
    def_info = (
        catalog.defs_by_symbol.get(entry.callee_symbol) if entry.callee_symbol is not None else None
    )
    if def_info is None:
        return [
            _call_target_record(
                _CallTargetRecordContext(
                    repo=entry.repo,
                    commit=entry.commit,
                    rel_path=entry.rel_path,
                    call_id=entry.call_id,
                    call_node_id=entry.call_node_id,
                    callee_symbol=entry.callee_symbol,
                    callee_def=None,
                    target_role=_TARGET_ROLE_PRIMARY,
                    binding_kind=_BINDING_UNKNOWN,
                    origin=_ORIGIN_SYNTAX_CALL,
                    call_kind=_CALL_KIND_EXPLICIT,
                    augop=None,
                    resolution_kind=entry.resolution_kind,
                    confidence=entry.confidence,
                    candidate_count=entry.candidate_count,
                    extras=entry.extras,
                )
            )
        ]
    if def_info.def_kind == "class":
        rows = [
            _call_target_record(
                _CallTargetRecordContext(
                    repo=entry.repo,
                    commit=entry.commit,
                    rel_path=entry.rel_path,
                    call_id=entry.call_id,
                    call_node_id=entry.call_node_id,
                    callee_symbol=def_info.scip_symbol,
                    callee_def=def_info,
                    target_role=_TARGET_ROLE_PRIMARY,
                    binding_kind=_BINDING_CONSTRUCTOR,
                    origin=_ORIGIN_SYNTAX_CALL,
                    call_kind=_CALL_KIND_EXPLICIT,
                    augop=None,
                    resolution_kind=entry.resolution_kind,
                    confidence=entry.confidence,
                    candidate_count=entry.candidate_count,
                    extras=entry.extras,
                )
            )
        ]
        init_def = catalog.init_by_class_id.get(def_info.def_id)
        if init_def is not None:
            rows.append(
                _call_target_record(
                    _CallTargetRecordContext(
                        repo=entry.repo,
                        commit=entry.commit,
                        rel_path=entry.rel_path,
                        call_id=entry.call_id,
                        call_node_id=entry.call_node_id,
                        callee_symbol=init_def.scip_symbol,
                        callee_def=init_def,
                        target_role=_TARGET_ROLE_INIT,
                        binding_kind=_BINDING_INIT,
                        origin=_ORIGIN_SYNTAX_CALL,
                        call_kind=_CALL_KIND_EXPLICIT,
                        augop=None,
                        resolution_kind=entry.resolution_kind,
                        confidence=entry.confidence,
                        candidate_count=entry.candidate_count,
                        extras=entry.extras,
                    )
                )
            )
        return rows
    binding_kind = _explicit_binding_kind(def_info, entry.callee_text)
    return [
        _call_target_record(
            _CallTargetRecordContext(
                repo=entry.repo,
                commit=entry.commit,
                rel_path=entry.rel_path,
                call_id=entry.call_id,
                call_node_id=entry.call_node_id,
                callee_symbol=def_info.scip_symbol,
                callee_def=def_info,
                target_role=_TARGET_ROLE_PRIMARY,
                binding_kind=binding_kind,
                origin=_ORIGIN_SYNTAX_CALL,
                call_kind=_CALL_KIND_EXPLICIT,
                augop=None,
                resolution_kind=entry.resolution_kind,
                confidence=entry.confidence,
                candidate_count=entry.candidate_count,
                extras=entry.extras,
            )
        )
    ]


def _explicit_call_rows(
    resolved: Sequence[Mapping[str, object]],
    catalog: _DefCatalog,
) -> list[dict[str, object]]:
    explicit_rows: list[dict[str, object]] = []
    for row in resolved:
        entry = _parse_explicit_call_row(row)
        if entry is None:
            continue
        explicit_rows.extend(_explicit_rows_for_entry(entry, catalog))
    return explicit_rows


def _span_index_from_catalog(catalog: _DefCatalog) -> _SpanIndex:
    class_spans = [
        (cls.start_byte, cls.end_byte, cls.name)
        for cls in catalog.classes_by_name.values()
        if cls.name
    ]
    class_line_spans = [
        (cls.start_line, cls.end_line, cls.name)
        for cls in catalog.classes_by_name.values()
        if cls.name
    ]
    func_spans = [
        (info.start_byte, info.end_byte, info.def_id)
        for info in catalog.defs_by_id.values()
        if info.def_kind in {"function", "async_function"}
    ]
    func_line_spans = [
        (info.start_line, info.end_line)
        for info in catalog.defs_by_id.values()
        if info.def_kind in {"function", "async_function"}
    ]
    return _SpanIndex(
        class_tree=_build_span_tree(class_spans),
        class_line_spans=class_line_spans,
        func_tree=_build_span_tree(func_spans),
        func_line_spans=func_line_spans,
        class_names=set(catalog.classes_by_name),
        descriptor_classes=set(catalog.descriptors),
    )


def _descriptor_assignments_for_syntax(
    syntax_nodes_df: Sequence[Mapping[str, object]],
    span_index: _SpanIndex,
) -> dict[str, dict[str, str]]:
    assignment_context = _DescriptorAssignmentContext(
        class_tree=span_index.class_tree,
        class_line_spans=span_index.class_line_spans,
        func_tree=span_index.func_tree,
        func_line_spans=span_index.func_line_spans,
        descriptor_classes=span_index.descriptor_classes,
    )
    return _parse_descriptor_assignments(syntax_nodes_df, assignment_context)


def _implicit_context_from_assignments(
    catalog: _DefCatalog,
    descriptor_assignments: Mapping[str, Mapping[str, str]],
    span_index: _SpanIndex,
) -> _ImplicitResolutionContext:
    property_classes_by_attr: dict[str, set[str]] = {}
    for class_name, prop_name in catalog.property_get:
        property_classes_by_attr.setdefault(prop_name, set()).add(class_name)
    for class_name, prop_name in catalog.property_set:
        property_classes_by_attr.setdefault(prop_name, set()).add(class_name)

    descriptor_classes_by_attr: dict[str, set[str]] = {}
    for class_name, attr_map in descriptor_assignments.items():
        for attr_name in attr_map:
            descriptor_classes_by_attr.setdefault(attr_name, set()).add(class_name)

    return _ImplicitResolutionContext(
        catalog=catalog,
        descriptor_assignments=descriptor_assignments,
        property_classes_by_attr=property_classes_by_attr,
        descriptor_classes_by_attr=descriptor_classes_by_attr,
        class_names=span_index.class_names,
    )


def _resolve_attribute_access(
    access: _AttributeAccess,
    context: _ImplicitResolutionContext,
) -> tuple[str, bool, str] | None:
    split = _split_attribute_text(access.text)
    if split is None:
        return None
    base_text, _attr_text = split
    class_name, is_class_access = _resolve_class_for_attribute(
        access.attribute,
        base_text,
        class_names=context.class_names,
        property_classes=context.property_classes_by_attr,
        descriptor_classes=context.descriptor_classes_by_attr,
    )
    if class_name is None:
        return None
    ctx_value = access.ctx or "load"
    return class_name, is_class_access, ctx_value


def _attribute_load_rows(
    access: _AttributeAccess,
    *,
    class_name: str,
    is_class_access: bool,
    context: _ImplicitResolutionContext,
) -> list[dict[str, object]]:
    prop_def = context.catalog.property_get.get((class_name, access.attribute))
    if prop_def is not None:
        return [
            _call_target_record(
                _CallTargetRecordContext(
                    repo=access.repo,
                    commit=access.commit,
                    rel_path=access.rel_path,
                    call_id=_implicit_call_id(
                        _call_site_from_access(access),
                        suffix="property_get",
                    ),
                    call_node_id=access.node_id,
                    callee_symbol=prop_def.scip_symbol,
                    callee_def=prop_def,
                    target_role=_TARGET_ROLE_PRIMARY,
                    binding_kind=_BINDING_PROPERTY_GET,
                    origin=_ORIGIN_DESCRIPTOR,
                    call_kind=_CALL_KIND_IMPLICIT_GET,
                    augop=None,
                    resolution_kind="implicit",
                    confidence=1.0,
                    candidate_count=0,
                    extras={
                        "descriptor_obj_is_none": False,
                        "attribute_name": access.attribute,
                        "owner_class": class_name,
                    },
                )
            )
        ]
    assigned = context.descriptor_assignments.get(class_name, {}).get(access.attribute)
    if assigned is None:
        return []
    descriptor = context.catalog.descriptors.get(assigned)
    if descriptor is None or descriptor.get_def is None:
        return []
    return [
        _call_target_record(
            _CallTargetRecordContext(
                repo=access.repo,
                commit=access.commit,
                rel_path=access.rel_path,
                call_id=_implicit_call_id(
                    _call_site_from_access(access),
                    suffix="descriptor_get",
                ),
                call_node_id=access.node_id,
                callee_symbol=descriptor.get_def.scip_symbol,
                callee_def=descriptor.get_def,
                target_role=_TARGET_ROLE_PRIMARY,
                binding_kind=_BINDING_DESCRIPTOR_GET,
                origin=_ORIGIN_DESCRIPTOR,
                call_kind=_CALL_KIND_IMPLICIT_GET,
                augop=None,
                resolution_kind="implicit",
                confidence=1.0,
                candidate_count=0,
                extras={
                    "descriptor_obj_is_none": is_class_access,
                    "attribute_name": access.attribute,
                    "owner_class": class_name,
                    "descriptor_class": assigned,
                },
            )
        )
    ]


def _attribute_store_rows(
    access: _AttributeAccess,
    *,
    class_name: str,
    context: _ImplicitResolutionContext,
) -> list[dict[str, object]]:
    prop_def = context.catalog.property_set.get((class_name, access.attribute))
    if prop_def is not None:
        return [
            _call_target_record(
                _CallTargetRecordContext(
                    repo=access.repo,
                    commit=access.commit,
                    rel_path=access.rel_path,
                    call_id=_implicit_call_id(
                        _call_site_from_access(access),
                        suffix="property_set",
                    ),
                    call_node_id=access.node_id,
                    callee_symbol=prop_def.scip_symbol,
                    callee_def=prop_def,
                    target_role=_TARGET_ROLE_PRIMARY,
                    binding_kind=_BINDING_PROPERTY_SET,
                    origin=_ORIGIN_DESCRIPTOR,
                    call_kind=_CALL_KIND_IMPLICIT_SET,
                    augop=None,
                    resolution_kind="implicit",
                    confidence=1.0,
                    candidate_count=0,
                    extras={
                        "descriptor_obj_is_none": False,
                        "attribute_name": access.attribute,
                        "owner_class": class_name,
                    },
                )
            )
        ]
    assigned = context.descriptor_assignments.get(class_name, {}).get(access.attribute)
    if assigned is None:
        return []
    descriptor = context.catalog.descriptors.get(assigned)
    if descriptor is None or descriptor.set_def is None:
        return []
    return [
        _call_target_record(
            _CallTargetRecordContext(
                repo=access.repo,
                commit=access.commit,
                rel_path=access.rel_path,
                call_id=_implicit_call_id(
                    _call_site_from_access(access),
                    suffix="descriptor_set",
                ),
                call_node_id=access.node_id,
                callee_symbol=descriptor.set_def.scip_symbol,
                callee_def=descriptor.set_def,
                target_role=_TARGET_ROLE_PRIMARY,
                binding_kind=_BINDING_DESCRIPTOR_SET,
                origin=_ORIGIN_DESCRIPTOR,
                call_kind=_CALL_KIND_IMPLICIT_SET,
                augop=None,
                resolution_kind="implicit",
                confidence=1.0,
                candidate_count=0,
                extras={
                    "descriptor_obj_is_none": False,
                    "attribute_name": access.attribute,
                    "owner_class": class_name,
                    "descriptor_class": assigned,
                },
            )
        )
    ]


def _attribute_rows_for_access(
    access: _AttributeAccess,
    context: _ImplicitResolutionContext,
) -> list[dict[str, object]]:
    resolved = _resolve_attribute_access(access, context)
    if resolved is None:
        return []
    class_name, is_class_access, ctx_value = resolved
    if ctx_value == "load":
        return _attribute_load_rows(
            access,
            class_name=class_name,
            is_class_access=is_class_access,
            context=context,
        )
    if ctx_value == "store" and not is_class_access:
        return _attribute_store_rows(access, class_name=class_name, context=context)
    return []


def _implicit_rows_from_attributes(
    attributes: Sequence[_AttributeAccess],
    context: _ImplicitResolutionContext,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for access in attributes:
        rows.extend(_attribute_rows_for_access(access, context))
    return rows


def _augassign_resolution(
    access: _AugAssignAccess,
    context: _ImplicitResolutionContext,
) -> _AugassignResolution | None:
    match = _AUGOP_PATTERN.search(access.text)
    if match is None:
        return None
    operator = match.group(1)
    left = access.text[: match.start()].strip()
    split = _split_attribute_text(left)
    if split is None:
        return None
    base_text, attr_text = split
    class_name, is_class_access = _resolve_class_for_attribute(
        attr_text,
        base_text,
        class_names=context.class_names,
        property_classes=context.property_classes_by_attr,
        descriptor_classes=context.descriptor_classes_by_attr,
    )
    if class_name is None or is_class_access:
        return None
    prop_get = context.catalog.property_get.get((class_name, attr_text))
    prop_set = context.catalog.property_set.get((class_name, attr_text))
    assigned = context.descriptor_assignments.get(class_name, {}).get(attr_text)
    descriptor = context.catalog.descriptors.get(assigned) if assigned is not None else None
    if prop_get is None and descriptor is None:
        return None
    if prop_set is None and (descriptor is None or descriptor.set_def is None):
        return None
    return _AugassignResolution(
        operator=operator,
        class_name=class_name,
        attr_text=attr_text,
        prop_get=prop_get,
        prop_set=prop_set,
        assigned=assigned,
        descriptor=descriptor,
    )


def _augassign_rows_for_access(
    access: _AugAssignAccess,
    context: _ImplicitResolutionContext,
) -> list[dict[str, object]]:
    resolution = _augassign_resolution(access, context)
    if resolution is None:
        return []
    rows: list[dict[str, object]] = []
    get_def = resolution.prop_get or (
        resolution.descriptor.get_def if resolution.descriptor is not None else None
    )
    if get_def is not None:
        rows.append(
            _call_target_record(
                _CallTargetRecordContext(
                    repo=access.repo,
                    commit=access.commit,
                    rel_path=access.rel_path,
                    call_id=_implicit_call_id(
                        _call_site_from_access(access),
                        suffix="augassign_get",
                    ),
                    call_node_id=access.node_id,
                    callee_symbol=get_def.scip_symbol,
                    callee_def=get_def,
                    target_role=_TARGET_ROLE_PRIMARY,
                    binding_kind=_BINDING_PROPERTY_GET
                    if resolution.prop_get is not None
                    else _BINDING_DESCRIPTOR_GET,
                    origin=_ORIGIN_AUGASSIGN,
                    call_kind=_CALL_KIND_IMPLICIT_GET,
                    augop=resolution.operator,
                    resolution_kind="implicit",
                    confidence=1.0,
                    candidate_count=0,
                    extras={
                        "descriptor_obj_is_none": False,
                        "attribute_name": resolution.attr_text,
                        "owner_class": resolution.class_name,
                        "descriptor_class": resolution.assigned,
                    },
                )
            )
        )
    set_def = resolution.prop_set or (
        resolution.descriptor.set_def if resolution.descriptor is not None else None
    )
    if set_def is not None:
        rows.append(
            _call_target_record(
                _CallTargetRecordContext(
                    repo=access.repo,
                    commit=access.commit,
                    rel_path=access.rel_path,
                    call_id=_implicit_call_id(
                        _call_site_from_access(access),
                        suffix="augassign_set",
                    ),
                    call_node_id=access.node_id,
                    callee_symbol=set_def.scip_symbol,
                    callee_def=set_def,
                    target_role=_TARGET_ROLE_PRIMARY,
                    binding_kind=_BINDING_PROPERTY_SET
                    if resolution.prop_set is not None
                    else _BINDING_DESCRIPTOR_SET_AUG,
                    origin=_ORIGIN_AUGASSIGN,
                    call_kind=_CALL_KIND_IMPLICIT_SET_AUG,
                    augop=resolution.operator,
                    resolution_kind="implicit",
                    confidence=1.0,
                    candidate_count=0,
                    extras={
                        "descriptor_obj_is_none": False,
                        "attribute_name": resolution.attr_text,
                        "owner_class": resolution.class_name,
                        "descriptor_class": resolution.assigned,
                    },
                )
            )
        )
    call_kind = _AUGOP_CALL_KIND.get(resolution.operator, "implicit_augassign_op")
    rows.append(
        _call_target_record(
            _CallTargetRecordContext(
                repo=access.repo,
                commit=access.commit,
                rel_path=access.rel_path,
                call_id=_implicit_call_id(
                    _call_site_from_access(access),
                    suffix="operator_dunder",
                ),
                call_node_id=access.node_id,
                callee_symbol=None,
                callee_def=None,
                target_role=_TARGET_ROLE_PRIMARY,
                binding_kind=_BINDING_OPERATOR_DUNDER,
                origin=_ORIGIN_AUGASSIGN,
                call_kind=call_kind,
                augop=resolution.operator,
                resolution_kind="implicit",
                confidence=0.5,
                candidate_count=0,
                extras={
                    "attribute_name": resolution.attr_text,
                    "owner_class": resolution.class_name,
                },
            )
        )
    )
    return rows


def _implicit_rows_from_augassigns(
    augassigns: Sequence[_AugAssignAccess],
    context: _ImplicitResolutionContext,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for access in augassigns:
        rows.extend(_augassign_rows_for_access(access, context))
    return rows


def _implicit_call_rows(
    syntax_nodes_df: Sequence[Mapping[str, object]],
    context: _ImplicitResolutionContext,
) -> list[dict[str, object]]:
    if not syntax_nodes_df:
        return []
    attributes = _extract_attribute_accesses(syntax_nodes_df)
    augassigns = _extract_augassigns(syntax_nodes_df)
    rows = _implicit_rows_from_attributes(attributes, context)
    rows.extend(_implicit_rows_from_augassigns(augassigns, context))
    return rows


def _explicit_rows_for_call_targets(
    calls: pa.Table,
    occurrences: pa.Table,
    catalog: _DefCatalog,
) -> list[dict[str, object]]:
    calls_rows = _table_rows(calls)
    if not calls_rows:
        return []
    occ_rows = _table_rows(occurrences)
    resolved = _resolve_call_targets(calls_rows, occ_rows)
    return _explicit_call_rows(resolved, catalog)


def _implicit_rows_for_call_targets(
    syntax_nodes_df: Sequence[Mapping[str, object]],
    catalog: _DefCatalog,
) -> list[dict[str, object]]:
    span_index = _span_index_from_catalog(catalog)
    descriptor_assignments = _descriptor_assignments_for_syntax(syntax_nodes_df, span_index)
    implicit_context = _implicit_context_from_assignments(
        catalog,
        descriptor_assignments,
        span_index,
    )
    return _implicit_call_rows(syntax_nodes_df, implicit_context)


def _build_call_targets_table(
    q__core__syntax_calls: InferableTabularInput,
    q__core__scip_occurrence_span_xref: InferableTabularInput,
    q__core__syntax_defs_resolved: InferableTabularInput,
    q__core__syntax_nodes: InferableTabularInput,
) -> pa.Table | None:
    calls = _select_table_columns(
        tabular_to_table(q__core__syntax_calls),
        [
            "repo",
            "commit",
            "rel_path",
            "call_id",
            "call_node_id",
            "callee_start_byte",
            "callee_end_byte",
            "callee_text",
        ],
    )
    occurrences = _select_table_columns(
        tabular_to_table(q__core__scip_occurrence_span_xref),
        [
            "rel_path",
            "scip_symbol",
            "roles",
            "start_byte",
            "end_byte",
        ],
    )
    defs_table = _select_table_columns(
        tabular_to_table(q__core__syntax_defs_resolved),
        [
            "def_id",
            "def_kind",
            "name",
            "scip_symbol",
            "goid_h128",
            "syntax_node_id",
            "start_line",
            "end_line",
            "start_byte",
            "end_byte",
            "extras",
        ],
    )
    catalog = _build_def_catalog(_table_rows(defs_table))
    explicit_rows = _explicit_rows_for_call_targets(calls, occurrences, catalog)
    syntax_nodes_table = _select_table_columns(
        tabular_to_table(q__core__syntax_nodes),
        [
            "repo",
            "commit",
            "rel_path",
            "node_id",
            "start_line",
            "start_col",
            "end_line",
            "end_col",
            "start_byte",
            "end_byte",
            "text_preview",
            "extras",
        ],
    )
    syntax_nodes_rows = [
        row for row in _table_rows(syntax_nodes_table) if row.get("extras") is not None
    ]
    implicit_rows = _implicit_rows_for_call_targets(syntax_nodes_rows, catalog)
    all_rows = [*explicit_rows, *implicit_rows]
    if not all_rows:
        return None
    targets_table = table_for_rows(CPG_CALL_TARGETS_TABLE_KEY, all_rows)[0]
    return _drop_table_columns(
        targets_table,
        ["callee_entry_block_id", "callee_exit_block_id"],
    )


def _attach_call_target_blocks(
    targets_table: pa.Table,
    blocks: pa.Table,
) -> pa.Table:
    entry_table = _entry_blocks(blocks)
    entry_table = _cast_table_column(entry_table, "function_goid_h128", _GOID_ARROW_TYPE)
    entry_table = _cast_table_column(entry_table, "entry_block_id", _BLOCK_ID_ARROW_TYPE)
    exit_table = _exit_blocks(blocks)
    exit_table = _cast_table_column(exit_table, "function_goid_h128", _GOID_ARROW_TYPE)
    exit_table = _cast_table_column(exit_table, "exit_block_id", _BLOCK_ID_ARROW_TYPE)
    target_columns = [
        "repo",
        "commit",
        "rel_path",
        "call_id",
        "call_node_id",
        "callee_symbol",
        "callee_def_id",
        "callee_def_node_id",
        "callee_goid_h128",
        "target_role",
        "binding_kind",
        "origin",
        "call_kind",
        "augop",
        "resolution_kind",
        "confidence",
        "candidate_count",
        "extras",
        "extras_kv",
    ]
    joined = _hash_join_block_targets(
        targets_table,
        entry_table,
        right_key="function_goid_h128",
        output_column="entry_block_id",
        left_columns=target_columns,
    )
    joined = _hash_join_block_targets(
        joined,
        exit_table,
        right_key="function_goid_h128",
        output_column="exit_block_id",
        left_columns=target_columns,
    )
    joined = _rename_table_columns(
        joined,
        {
            "entry_block_id": "callee_entry_block_id",
            "exit_block_id": "callee_exit_block_id",
        },
    )
    return joined.select(
        [
            "repo",
            "commit",
            "rel_path",
            "call_id",
            "call_node_id",
            "callee_symbol",
            "callee_def_id",
            "callee_def_node_id",
            "callee_goid_h128",
            "callee_entry_block_id",
            "callee_exit_block_id",
            "target_role",
            "binding_kind",
            "origin",
            "call_kind",
            "augop",
            "resolution_kind",
            "confidence",
            "candidate_count",
            "extras",
            "extras_kv",
        ]
    )


def cpg_call_targets(
    q__core__syntax_calls: InferableTabularInput,
    q__core__scip_occurrence_span_xref: InferableTabularInput,
    q__core__syntax_defs_resolved: InferableTabularInput,
    q__graph__cfg_blocks: InferableTabularInput,
    q__core__syntax_nodes: InferableTabularInput,
) -> InferableTabularInput:
    """Resolve call targets by welding callee spans to SCIP occurrences.

    Returns
    -------
    InferableTabularInput
        Arrow table for graph.cpg_call_targets.
    """
    targets_table = _build_call_targets_table(
        q__core__syntax_calls,
        q__core__scip_occurrence_span_xref,
        q__core__syntax_defs_resolved,
        q__core__syntax_nodes,
    )
    if targets_table is None:
        return empty_reader(CPG_CALL_TARGETS_TABLE_KEY)
    blocks = tabular_to_table(q__graph__cfg_blocks)
    joined = _attach_call_target_blocks(targets_table, blocks)
    deduped = dedupe_table_for_table(CPG_CALL_TARGETS_TABLE_KEY, joined)
    return _table_to_reader(CPG_CALL_TARGETS_TABLE_KEY, deduped)


def cpg_call_candidates(cpg_call_targets: InferableTabularInput) -> InferableTabularInput:
    """Group call targets into per-call candidate lists.

    Returns
    -------
    InferableTabularInput
        Arrow table for graph.cpg_call_candidates.
    """
    call_targets = tabular_to_table(cpg_call_targets)
    grouped: dict[tuple[str, str, str, str, str | None], list[dict[str, object]]] = {}
    for row in _table_rows(call_targets):
        repo = _coerce_str(row.get("repo"))
        commit = _coerce_str(row.get("commit"))
        rel_path = _coerce_str(row.get("rel_path"))
        call_id = _coerce_str(row.get("call_id"))
        if repo is None or commit is None or rel_path is None or call_id is None:
            continue
        call_node_id = _coerce_str(row.get("call_node_id"))
        key = (repo, commit, rel_path, call_id, call_node_id)
        grouped.setdefault(key, []).append(
            {
                "callee_goid_h128": row.get("callee_goid_h128"),
                "callee_symbol": row.get("callee_symbol"),
                "callee_def_id": row.get("callee_def_id"),
                "callee_def_node_id": row.get("callee_def_node_id"),
                "target_role": row.get("target_role"),
                "binding_kind": row.get("binding_kind"),
                "origin": row.get("origin"),
                "call_kind": row.get("call_kind"),
                "augop": row.get("augop"),
                "resolution_kind": row.get("resolution_kind"),
                "confidence": _coerce_float(row.get("confidence")),
                "candidate_count": _coerce_int(row.get("candidate_count")),
                "extras_kv": _call_candidate_extras_kv(row),
            }
        )
    if not grouped:
        return empty_reader(CPG_CALL_CANDIDATES_TABLE_KEY)
    rows = [
        {
            "repo": repo,
            "commit": commit,
            "rel_path": rel_path,
            "call_id": call_id,
            "call_node_id": call_node_id,
            "extras": None,
            "extras_kv": None,
            "candidates": candidates,
        }
        for (repo, commit, rel_path, call_id, call_node_id), candidates in grouped.items()
    ]
    candidates_table, _ = table_for_rows(CPG_CALL_CANDIDATES_TABLE_KEY, rows)
    return _table_to_reader(CPG_CALL_CANDIDATES_TABLE_KEY, candidates_table)


def cpg_edges_calls(
    cpg_call_candidates: InferableTabularInput,
    q__graph__cfg_blocks: InferableTabularInput,
) -> InferableTabularInput:
    """Build CALLS edges from call candidates.

    Returns
    -------
    InferableTabularInput
        Arrow table for graph.cpg_edges_calls.
    """
    candidates = tabular_to_table(cpg_call_candidates)
    if candidates.num_rows == 0:
        return empty_reader(CPG_CALL_EDGES_TABLE_KEY)
    exploded = explode_edges_for_join(
        candidates,
        spec=ExplodeSpec(
            src_col="call_id",
            dst_list_col="candidates",
            repeat_cols=("repo", "commit", "call_node_id", "extras"),
            error_context_cols=("repo", "commit", "rel_path", "call_id"),
        ),
        table_key=CPG_CALL_CANDIDATES_TABLE_KEY,
        schema_service=get_schema_service(),
    )
    if exploded.good.num_rows == 0:
        return empty_reader(CPG_CALL_EDGES_TABLE_KEY)
    entry_blocks = _entry_blocks(tabular_to_table(q__graph__cfg_blocks))
    if entry_blocks.num_rows == 0:
        return empty_reader(CPG_CALL_EDGES_TABLE_KEY)
    entry_blocks = _cast_table_column(entry_blocks, "function_goid_h128", _GOID_ARROW_TYPE)
    entry_blocks = _cast_table_column(entry_blocks, "entry_block_id", _BLOCK_ID_ARROW_TYPE)
    candidate_allowlist = _join_safe_allowlist(CPG_CALL_CANDIDATES_TABLE_KEY)
    exploded_good = normalize_table_for_join(
        exploded.good,
        allowed_columns=candidate_allowlist,
    )
    entry_blocks = normalize_table_for_join(
        entry_blocks,
        allowed_columns=_join_safe_allowlist(CFG_BLOCKS_TABLE_KEY),
    )

    left_plan = build_table_plan(
        table=exploded_good,
        options=TablePlanOptions(
            projection={
                "repo": E.field("repo"),
                "commit": E.field("commit"),
                "call_id": E.field("call_id"),
                "call_node_id": E.field("call_node_id"),
                "extras": E.field("extras"),
                "callee_goid_h128": E.cast(
                    E.field(("candidates", "callee_goid_h128")),
                    "decimal128(38,0)",
                ),
                "confidence": E.field(("candidates", "confidence")),
                "extras_kv": E.field(("candidates", "extras_kv")),
            },
            filter_expr=E.is_valid("callee_goid_h128"),
        ),
    )
    right_plan = build_table_plan(
        table=entry_blocks,
        options=TablePlanOptions(
            projection={
                "function_goid_h128": E.cast(
                    E.field("function_goid_h128"),
                    "decimal128(38,0)",
                ),
                "entry_block_id": E.field("entry_block_id"),
            },
            filter_expr=E.is_valid("function_goid_h128"),
        ),
    )
    joined = left_plan.hash_join(
        right=right_plan,
        spec=HashJoinSpec(
            left_keys=["callee_goid_h128"],
            right_keys=["function_goid_h128"],
            how="left outer",
            left_output=[
                "repo",
                "commit",
                "call_id",
                "call_node_id",
                "extras",
                "confidence",
                "extras_kv",
            ],
            right_output=["entry_block_id"],
        ),
    )
    joined = joined.filter(E.is_valid("entry_block_id"))
    joined = joined.project(
        {
            "repo": E.field("repo"),
            "commit": E.field("commit"),
            "call_id": E.field("call_id"),
            "call_node_id": E.field("call_node_id"),
            "callee_entry_block_id": E.field("entry_block_id"),
            "edge_kind": E.scalar("CALLS"),
            "confidence": E.field("confidence"),
            "extras": E.field("extras"),
            "extras_kv": E.field("extras_kv"),
        }
    )
    ordered = joined.order_by(
        sort_keys=[
            ("repo", "ascending"),
            ("commit", "ascending"),
            ("call_id", "ascending"),
            ("call_node_id", "ascending"),
            ("callee_entry_block_id", "ascending"),
        ]
    )
    result = finalize_reader(
        ordered.to_reader(use_threads=True),
        spec=finalize_spec_for_table(
            CPG_CALL_EDGES_TABLE_KEY,
            mode="strict",
            key_fields=_key_fields_for_table(CPG_CALL_EDGES_TABLE_KEY),
            order_by=_order_by_for_table(CPG_CALL_EDGES_TABLE_KEY),
            target_name=CALL_WIRING_TARGET_NAME,
        ),
    )
    return result.good


@dataclass(frozen=True, slots=True)
class _ParamIndex:
    by_ordinal: dict[tuple[str, int], Mapping[str, object]]
    by_name: dict[tuple[str, str], Mapping[str, object]]
    by_kind: dict[tuple[str, str], Mapping[str, object]]


def _normalize_params_rows(params_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for row in params_rows:
        callee_def_id = _coerce_str(row.get("func_def_id") or row.get("callee_def_id"))
        if callee_def_id is None:
            continue
        normalized.append(
            {
                "callee_def_id": callee_def_id,
                "param_ordinal": _coerce_int(row.get("param_ordinal")),
                "param_kind": _coerce_str(row.get("param_kind")),
                "param_name": _coerce_str(row.get("param_name")),
                "param_node_id": row.get("param_node_id"),
            }
        )
    return normalized


def _build_param_index(params_rows: Sequence[Mapping[str, object]]) -> _ParamIndex:
    by_ordinal: dict[tuple[str, int], Mapping[str, object]] = {}
    by_name: dict[tuple[str, str], Mapping[str, object]] = {}
    by_kind: dict[tuple[str, str], Mapping[str, object]] = {}
    for row in params_rows:
        callee_def_id = _coerce_str(row.get("callee_def_id"))
        if callee_def_id is None:
            continue
        param_ordinal = _coerce_int(row.get("param_ordinal"))
        param_name = _coerce_str(row.get("param_name"))
        param_kind = _coerce_str(row.get("param_kind"))
        if param_ordinal is not None:
            by_ordinal[callee_def_id, param_ordinal] = row
        if param_name:
            by_name[callee_def_id, param_name] = row
        if param_kind:
            by_kind[callee_def_id, param_kind] = row
    return _ParamIndex(by_ordinal=by_ordinal, by_name=by_name, by_kind=by_kind)


def _explicit_targets_by_call(
    call_targets_rows: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], dict[tuple[str, str, str], list[dict[str, object]]]]:
    explicit_targets: list[dict[str, object]] = []
    by_call: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in call_targets_rows:
        if row.get("call_kind") != _CALL_KIND_EXPLICIT:
            continue
        target_role = row.get("target_role")
        binding_kind = row.get("binding_kind")
        if target_role != _TARGET_ROLE_INIT and binding_kind == _BINDING_CONSTRUCTOR:
            continue
        repo = _coerce_str(row.get("repo"))
        commit = _coerce_str(row.get("commit"))
        call_id = _coerce_str(row.get("call_id"))
        if repo is None or commit is None or call_id is None:
            continue
        target = {
            "repo": repo,
            "commit": commit,
            "call_id": call_id,
            "call_node_id": row.get("call_node_id"),
            "callee_def_id": _coerce_str(row.get("callee_def_id")),
            "binding_kind": _coerce_str(binding_kind),
            "target_role": _coerce_str(target_role),
            "call_kind": _coerce_str(row.get("call_kind")),
            "augop": row.get("augop"),
            "confidence": _coerce_float(row.get("confidence")) or 0.0,
        }
        explicit_targets.append(target)
        by_call.setdefault((repo, commit, call_id), []).append(target)
    return explicit_targets, by_call


def _arg_slot(arg_kind: str, arg_ordinal: int | None, arg_name: str | None) -> str | None:
    if arg_kind == "positional" and arg_ordinal is not None:
        return f"positional:{arg_ordinal}"
    if arg_kind == "keyword" and arg_name:
        return f"keyword:{arg_name}"
    if arg_kind == "starargs":
        return "positional:*"
    if arg_kind == "kwargs":
        return "keyword:**"
    return None


def _build_arg_rows(
    call_args_rows: Sequence[Mapping[str, object]],
    explicit_targets_by_call: Mapping[tuple[str, str, str], Sequence[Mapping[str, object]]],
) -> list[dict[str, object]]:
    args: list[dict[str, object]] = []
    for row in call_args_rows:
        repo = _coerce_str(row.get("repo"))
        commit = _coerce_str(row.get("commit"))
        call_id = _coerce_str(row.get("call_id"))
        if repo is None or commit is None or call_id is None:
            continue
        targets = explicit_targets_by_call.get((repo, commit, call_id))
        if not targets:
            continue
        arg_kind = _coerce_str(row.get("arg_kind"))
        if arg_kind is None:
            continue
        arg_ordinal = _coerce_int(row.get("arg_ordinal"))
        arg_name = _coerce_str(row.get("arg_name"))
        arg_expr_node_id = row.get("arg_expr_node_id")
        for target in targets:
            callee_def_id = _coerce_str(target.get("callee_def_id"))
            if callee_def_id is None:
                continue
            binding_kind = _coerce_str(target.get("binding_kind"))
            param_ordinal_hint = arg_ordinal
            if (
                binding_kind in {_BINDING_BOUND_METHOD, _BINDING_CLASSMETHOD}
                and arg_ordinal is not None
            ):
                param_ordinal_hint = arg_ordinal + 1
            args.append(
                {
                    "repo": repo,
                    "commit": commit,
                    "call_id": call_id,
                    "callee_def_id": callee_def_id,
                    "arg_kind": arg_kind,
                    "arg_ordinal": arg_ordinal,
                    "arg_name": arg_name,
                    "arg_expr_node_id": arg_expr_node_id,
                    "arg_slot": _arg_slot(arg_kind, arg_ordinal, arg_name),
                    "arg_role": "keyword" if arg_kind == "keyword" else "positional",
                    "arg_is_implicit": False,
                    "param_ordinal_hint": param_ordinal_hint,
                    "call_kind": _coerce_str(target.get("call_kind")),
                    "augop": target.get("augop"),
                    "confidence": _coerce_float(target.get("confidence")) or 0.0,
                }
            )
    return args


def _explicit_arg_edges(
    args: Sequence[Mapping[str, object]],
    params_index: _ParamIndex,
) -> list[dict[str, object]]:
    edges: list[dict[str, object]] = []
    for arg in args:
        callee_def_id = _coerce_str(arg.get("callee_def_id"))
        arg_kind = _coerce_str(arg.get("arg_kind"))
        if callee_def_id is None or arg_kind is None:
            continue
        param_row, confidence = _resolve_param_row(
            arg,
            callee_def_id=callee_def_id,
            arg_kind=arg_kind,
            params_index=params_index,
        )
        if param_row is None:
            continue
        edges.append(
            {
                "repo": arg.get("repo"),
                "commit": arg.get("commit"),
                "call_id": arg.get("call_id"),
                "src_arg_node_id": arg.get("arg_expr_node_id"),
                "dst_param_node_id": param_row.get("param_node_id"),
                "edge_kind": "ARG_TO_PARAM",
                "arg_ordinal": arg.get("arg_ordinal"),
                "param_ordinal": param_row.get("param_ordinal"),
                "arg_name": arg.get("arg_name"),
                "param_name": param_row.get("param_name"),
                "arg_slot": arg.get("arg_slot"),
                "arg_role": arg.get("arg_role"),
                "arg_is_implicit": arg.get("arg_is_implicit"),
                "call_kind": arg.get("call_kind"),
                "augop": arg.get("augop"),
                "confidence": confidence,
                "extras": None,
                "extras_kv": None,
            }
        )
    return edges


def _resolve_param_row(
    arg: Mapping[str, object],
    *,
    callee_def_id: str,
    arg_kind: str,
    params_index: _ParamIndex,
) -> tuple[Mapping[str, object] | None, float]:
    confidence = _coerce_float(arg.get("confidence")) or 0.0
    param_row: Mapping[str, object] | None = None
    if arg_kind == "positional":
        hint = _coerce_int(arg.get("param_ordinal_hint"))
        if hint is not None:
            candidate = params_index.by_ordinal.get((callee_def_id, hint))
            if candidate is not None and candidate.get("param_kind") not in {"varargs", "varkw"}:
                param_row = candidate
    elif arg_kind == "keyword":
        arg_name = _coerce_str(arg.get("arg_name"))
        if arg_name is not None:
            param_row = params_index.by_name.get((callee_def_id, arg_name))
    elif arg_kind == "starargs":
        param_row = params_index.by_kind.get((callee_def_id, "varargs"))
        confidence *= 0.7
    elif arg_kind == "kwargs":
        param_row = params_index.by_kind.get((callee_def_id, "varkw"))
        confidence *= 0.6
    return param_row, confidence


def _implicit_receiver_edges(
    explicit_targets: Sequence[Mapping[str, object]],
    params_index: _ParamIndex,
) -> list[dict[str, object]]:
    edges: list[dict[str, object]] = []
    for target in explicit_targets:
        binding_kind = _coerce_str(target.get("binding_kind"))
        if binding_kind not in {_BINDING_BOUND_METHOD, _BINDING_CLASSMETHOD}:
            continue
        callee_def_id = _coerce_str(target.get("callee_def_id"))
        if callee_def_id is None:
            continue
        param_row = params_index.by_ordinal.get((callee_def_id, 0))
        if param_row is None:
            continue
        edges.append(
            {
                "repo": target.get("repo"),
                "commit": target.get("commit"),
                "call_id": target.get("call_id"),
                "src_arg_node_id": target.get("call_node_id"),
                "dst_param_node_id": param_row.get("param_node_id"),
                "edge_kind": "ARG_TO_PARAM",
                "arg_ordinal": None,
                "param_ordinal": param_row.get("param_ordinal"),
                "arg_name": None,
                "param_name": param_row.get("param_name"),
                "arg_slot": "implicit:receiver",
                "arg_role": "receiver",
                "arg_is_implicit": True,
                "call_kind": target.get("call_kind"),
                "augop": target.get("augop"),
                "confidence": _coerce_float(target.get("confidence")) or 0.0,
                "extras": None,
                "extras_kv": None,
            }
        )
    return edges


def _descriptor_arg_templates(
    binding_kind: str,
    *,
    call_kind: str | None,
    descriptor_obj_is_none: bool | None,
) -> list[tuple[int, str, str]]:
    if binding_kind == _BINDING_PROPERTY_GET:
        return [(0, "implicit:receiver", "receiver")]
    if binding_kind == _BINDING_PROPERTY_SET:
        return [
            (0, "implicit:receiver", "receiver"),
            (1, "implicit:value", "descriptor_value"),
        ]
    if binding_kind == _BINDING_DESCRIPTOR_GET:
        slot = "implicit:none" if descriptor_obj_is_none else "implicit:obj"
        role = "descriptor_none" if descriptor_obj_is_none else "descriptor_obj"
        return [
            (0, "implicit:receiver", "receiver"),
            (1, slot, role),
            (2, "implicit:objtype", "descriptor_objtype"),
        ]
    if binding_kind in {_BINDING_DESCRIPTOR_SET, _BINDING_DESCRIPTOR_SET_AUG}:
        value_role = "descriptor_value"
        if call_kind == _CALL_KIND_IMPLICIT_SET_AUG:
            value_role = "augassign_value"
        return [
            (0, "implicit:receiver", "receiver"),
            (1, "implicit:obj", "descriptor_obj"),
            (2, "implicit:value", value_role),
        ]
    return []


def _implicit_descriptor_edges(
    call_targets: Sequence[Mapping[str, object]],
    params_index: _ParamIndex,
) -> list[dict[str, object]]:
    edges: list[dict[str, object]] = []
    for row in call_targets:
        call_kind = _coerce_str(row.get("call_kind"))
        if call_kind not in {
            _CALL_KIND_IMPLICIT_GET,
            _CALL_KIND_IMPLICIT_SET,
            _CALL_KIND_IMPLICIT_SET_AUG,
        }:
            continue
        binding_kind = _coerce_str(row.get("binding_kind"))
        if binding_kind is None:
            continue
        callee_def_id = _coerce_str(row.get("callee_def_id"))
        if callee_def_id is None:
            continue
        extras = _coerce_extras_kv(row.get("extras_kv"))
        descriptor_obj_is_none = _extras_descriptor_obj_is_none(extras)
        for param_ordinal, arg_slot, arg_role in _descriptor_arg_templates(
            binding_kind,
            call_kind=call_kind,
            descriptor_obj_is_none=descriptor_obj_is_none,
        ):
            param_row = params_index.by_ordinal.get((callee_def_id, param_ordinal))
            if param_row is None:
                continue
            edges.append(
                {
                    "repo": row.get("repo"),
                    "commit": row.get("commit"),
                    "call_id": row.get("call_id"),
                    "src_arg_node_id": row.get("call_node_id"),
                    "dst_param_node_id": param_row.get("param_node_id"),
                    "edge_kind": "ARG_TO_PARAM",
                    "arg_ordinal": None,
                    "param_ordinal": param_row.get("param_ordinal"),
                    "arg_name": None,
                    "param_name": param_row.get("param_name"),
                    "arg_slot": arg_slot,
                    "arg_role": arg_role,
                    "arg_is_implicit": True,
                    "call_kind": call_kind,
                    "augop": row.get("augop"),
                    "confidence": _coerce_float(row.get("confidence")) or 0.0,
                    "extras": None,
                    "extras_kv": None,
                }
            )
    return edges


def _group_edge_candidates_by_call(
    edges: Sequence[Mapping[str, object]],
    *,
    candidate_fields: Sequence[str],
    list_col: str,
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, object, object], list[dict[str, object]]] = {}
    for edge in edges:
        call_id = edge.get("call_id")
        if call_id is None:
            continue
        key = (edge.get("repo"), edge.get("commit"), call_id)
        candidate = {field: edge.get(field) for field in candidate_fields}
        grouped.setdefault(key, []).append(candidate)
    rows: list[dict[str, object]] = []
    for (repo, commit, call_id), candidates in grouped.items():
        if not candidates:
            continue
        rows.append(
            {
                "repo": repo,
                "commit": commit,
                "call_id": call_id,
                list_col: candidates,
            }
        )
    return rows


def cpg_edges_arg_to_param(
    cpg_call_targets: InferableTabularInput,
    q__core__syntax_call_args: InferableTabularInput,
    q__core__syntax_func_params: InferableTabularInput,
) -> InferableTabularInput:
    """Build ARG_TO_PARAM edges from call arguments and function params.

    Returns
    -------
    InferableTabularInput
        Arrow table for graph.cpg_edges_arg_to_param.
    """
    call_targets_rows = _table_rows(tabular_to_table(cpg_call_targets))
    explicit_targets, explicit_by_call = _explicit_targets_by_call(call_targets_rows)
    call_args_rows = _table_rows(tabular_to_table(q__core__syntax_call_args))
    args_rows = _build_arg_rows(call_args_rows, explicit_by_call)
    params_rows = _normalize_params_rows(_table_rows(tabular_to_table(q__core__syntax_func_params)))
    params_index = _build_param_index(params_rows)

    combined_rows = [
        *_explicit_arg_edges(args_rows, params_index),
        *_implicit_descriptor_edges(call_targets_rows, params_index),
        *_implicit_receiver_edges(explicit_targets, params_index),
    ]
    if not combined_rows:
        return empty_reader(CPG_ARG_TO_PARAM_EDGES_TABLE_KEY)
    candidates = _group_edge_candidates_by_call(
        combined_rows,
        candidate_fields=(
            "src_arg_node_id",
            "dst_param_node_id",
            "arg_ordinal",
            "param_ordinal",
            "arg_name",
            "param_name",
            "arg_slot",
            "arg_role",
            "arg_is_implicit",
            "call_kind",
            "augop",
            "confidence",
            "extras",
            "extras_kv",
        ),
        list_col="candidates",
    )
    if not candidates:
        return empty_reader(CPG_ARG_TO_PARAM_EDGES_TABLE_KEY)
    candidate_table = _table_from_rows(
        candidates,
        columns=("repo", "commit", "call_id", "candidates"),
    )
    exploded = explode_edges_for_join(
        candidate_table,
        spec=ExplodeSpec(
            src_col="call_id",
            dst_list_col="candidates",
            repeat_cols=("repo", "commit"),
            error_context_cols=("repo", "commit", "call_id"),
        ),
        table_key=CPG_CALL_CANDIDATES_TABLE_KEY,
        schema_service=get_schema_service(),
    )
    if exploded.good.num_rows == 0:
        return empty_reader(CPG_ARG_TO_PARAM_EDGES_TABLE_KEY)
    ordered = build_table_plan(
        table=exploded.good,
        options=TablePlanOptions(
            projection={
                "repo": E.field("repo"),
                "commit": E.field("commit"),
                "call_id": E.field("call_id"),
                "src_arg_node_id": E.field(("candidates", "src_arg_node_id")),
                "dst_param_node_id": E.field(("candidates", "dst_param_node_id")),
                "edge_kind": E.scalar("ARG_TO_PARAM"),
                "arg_ordinal": E.field(("candidates", "arg_ordinal")),
                "param_ordinal": E.field(("candidates", "param_ordinal")),
                "arg_name": E.field(("candidates", "arg_name")),
                "param_name": E.field(("candidates", "param_name")),
                "arg_slot": E.field(("candidates", "arg_slot")),
                "arg_role": E.field(("candidates", "arg_role")),
                "arg_is_implicit": E.field(("candidates", "arg_is_implicit")),
                "call_kind": E.field(("candidates", "call_kind")),
                "augop": E.field(("candidates", "augop")),
                "confidence": E.field(("candidates", "confidence")),
                "extras": E.field(("candidates", "extras")),
                "extras_kv": E.field(("candidates", "extras_kv")),
            },
            order_by=(
                ("repo", "ascending"),
                ("commit", "ascending"),
                ("call_id", "ascending"),
                ("src_arg_node_id", "ascending"),
                ("dst_param_node_id", "ascending"),
            ),
        ),
    )
    result = finalize_reader(
        ordered.to_reader(use_threads=True),
        spec=finalize_spec_for_table(
            CPG_ARG_TO_PARAM_EDGES_TABLE_KEY,
            mode="strict",
            key_fields=_key_fields_for_table(CPG_ARG_TO_PARAM_EDGES_TABLE_KEY),
            order_by=_order_by_for_table(CPG_ARG_TO_PARAM_EDGES_TABLE_KEY),
            target_name=CALL_WIRING_TARGET_NAME,
        ),
    )
    return result.good


def cpg_edges_ret_to_call(cpg_call_targets: InferableTabularInput) -> InferableTabularInput:
    """Build RET_TO_CALL edges using callee exit block summaries.

    Returns
    -------
    InferableTabularInput
        Arrow table for graph.cpg_edges_ret_to_call.
    """
    call_targets = tabular_to_table(cpg_call_targets)
    rows: list[dict[str, object]] = []
    for row in _table_rows(call_targets):
        if row.get("callee_exit_block_id") is None:
            continue
        if row.get("target_role") != _TARGET_ROLE_PRIMARY:
            continue
        binding_kind = row.get("binding_kind")
        if binding_kind in {
            _BINDING_PROPERTY_SET,
            _BINDING_DESCRIPTOR_SET,
            _BINDING_DESCRIPTOR_SET_AUG,
        }:
            continue
        confidence = _coerce_float(row.get("confidence")) or 0.0
        rows.append(
            {
                "repo": row.get("repo"),
                "commit": row.get("commit"),
                "call_id": row.get("call_id"),
                "exit_block_id": row.get("callee_exit_block_id"),
                "call_node_id": row.get("call_node_id"),
                "target_role": row.get("target_role"),
                "call_kind": row.get("call_kind"),
                "origin": row.get("origin"),
                "confidence": confidence * 0.9,
                "extras": None,
                "extras_kv": {"summary_kind": "exit_block"},
            }
        )
    if not rows:
        return empty_reader(CPG_RET_TO_CALL_EDGES_TABLE_KEY)
    candidates = _group_edge_candidates_by_call(
        rows,
        candidate_fields=(
            "exit_block_id",
            "call_node_id",
            "target_role",
            "call_kind",
            "origin",
            "confidence",
            "extras",
            "extras_kv",
        ),
        list_col="candidates",
    )
    if not candidates:
        return empty_reader(CPG_RET_TO_CALL_EDGES_TABLE_KEY)
    candidate_table = _table_from_rows(
        candidates,
        columns=("repo", "commit", "call_id", "candidates"),
    )
    exploded = explode_edges_for_join(
        candidate_table,
        spec=ExplodeSpec(
            src_col="call_id",
            dst_list_col="candidates",
            repeat_cols=("repo", "commit"),
            error_context_cols=("repo", "commit", "call_id"),
        ),
        table_key=CPG_CALL_CANDIDATES_TABLE_KEY,
        schema_service=get_schema_service(),
    )
    if exploded.good.num_rows == 0:
        return empty_reader(CPG_RET_TO_CALL_EDGES_TABLE_KEY)
    ordered = build_table_plan(
        table=exploded.good,
        options=TablePlanOptions(
            projection={
                "repo": E.field("repo"),
                "commit": E.field("commit"),
                "call_id": E.field("call_id"),
                "exit_block_id": E.field(("candidates", "exit_block_id")),
                "call_node_id": E.field(("candidates", "call_node_id")),
                "target_role": E.field(("candidates", "target_role")),
                "call_kind": E.field(("candidates", "call_kind")),
                "origin": E.field(("candidates", "origin")),
                "edge_kind": E.scalar("RET_TO_CALL"),
                "confidence": E.field(("candidates", "confidence")),
                "extras": E.field(("candidates", "extras")),
                "extras_kv": E.field(("candidates", "extras_kv")),
            },
            order_by=(
                ("repo", "ascending"),
                ("commit", "ascending"),
                ("call_id", "ascending"),
                ("exit_block_id", "ascending"),
                ("call_node_id", "ascending"),
            ),
        ),
    )
    result = finalize_reader(
        ordered.to_reader(use_threads=True),
        spec=finalize_spec_for_table(
            CPG_RET_TO_CALL_EDGES_TABLE_KEY,
            mode="strict",
            key_fields=_key_fields_for_table(CPG_RET_TO_CALL_EDGES_TABLE_KEY),
            order_by=_order_by_for_table(CPG_RET_TO_CALL_EDGES_TABLE_KEY),
            target_name=CALL_WIRING_TARGET_NAME,
        ),
    )
    return result.good


__all__ = [
    "CALL_WIRING_TARGET_NAME",
    "CPG_ARG_TO_PARAM_EDGES_TABLE_KEY",
    "CPG_CALL_CANDIDATES_TABLE_KEY",
    "CPG_CALL_EDGES_TABLE_KEY",
    "CPG_CALL_TARGETS_TABLE_KEY",
    "CPG_RET_TO_CALL_EDGES_TABLE_KEY",
    "cpg_call_candidates",
    "cpg_call_targets",
    "cpg_edges_arg_to_param",
    "cpg_edges_calls",
    "cpg_edges_ret_to_call",
]
