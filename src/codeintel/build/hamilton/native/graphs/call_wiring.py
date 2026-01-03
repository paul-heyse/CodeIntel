"""Call wiring relation sources for CPG interprocedural edges."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import polars as pl
from intervaltree import IntervalTree

from codeintel.build.tabular.conversion import tabular_to_lazyframe
from codeintel.build.tabular.frames import (
    JoinSpec,
    dedupe_frame_for_table,
    empty_frame_for_table,
    join_validated,
)
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.intervals.span_resolver import MatchKind, SpanResolver
from codeintel.core.serialization.payload import PayloadValue, decode_payload, encode_payload

CALL_WIRING_TARGET_NAME = "call_wiring"
CPG_CALL_TARGETS_TABLE_KEY = "graph.cpg_call_targets"
CPG_CALL_EDGES_TABLE_KEY = "graph.cpg_edges_calls"
CPG_ARG_TO_PARAM_EDGES_TABLE_KEY = "graph.cpg_edges_arg_to_param"
CPG_RET_TO_CALL_EDGES_TABLE_KEY = "graph.cpg_edges_ret_to_call"

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
    extras_json: PayloadValue | bytes | bytearray | memoryview | None


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
    extras_json: PayloadValue | bytes | bytearray | memoryview | None


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
    payload = json.dumps(parts, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=16)
    return digest.hexdigest()


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


def _extract_def_info(row: Mapping[str, object]) -> tuple[_DefInfo, str | None, str | None]:
    extras = _extras_struct(row, "extras_json") or {}
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


def _build_def_catalog(defs_df: pl.DataFrame) -> _DefCatalog:
    builder = _DefCatalogBuilder()
    for row in defs_df.iter_rows(named=True):
        builder.add_row(row)
    return builder.finalize()


def _payload_literal(value: PayloadValue | bytes | bytearray | memoryview | None) -> pl.Expr:
    return pl.lit(encode_payload(value)).cast(pl.Binary)


def _call_edge_extras(row: Mapping[str, object]) -> bytes:
    payload: dict[str, object] = {
        "binding_kind": row.get("binding_kind"),
        "target_role": row.get("target_role"),
        "call_kind": row.get("call_kind"),
        "origin": row.get("origin"),
        "augop": row.get("augop"),
    }
    extras = decode_payload(row.get("extras_json"))
    if extras is not None:
        payload["call_extras"] = extras
    encoded = encode_payload(payload)
    if encoded is None:
        msg = "Expected payload encoding to return bytes"
        raise ValueError(msg)
    return encoded


def _rel_path_key(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, tuple) and value and isinstance(value[0], str):
        return value[0]
    return None


def _build_occurrence_resolver(
    *,
    rel_path: str,
    occ_df: pl.DataFrame | None,
) -> SpanResolver[_OccurrenceCandidate]:
    resolver = SpanResolver.for_bytes(path_normalizer=lambda value: value)
    if occ_df is None or occ_df.is_empty():
        return resolver
    spans: list[tuple[str, int, int, _OccurrenceCandidate]] = []
    for row in occ_df.iter_rows(named=True):
        symbol = row.get("scip_symbol")
        start = row.get("start_byte")
        end = row.get("end_byte")
        roles = row.get("roles")
        if not isinstance(symbol, str) or not isinstance(start, int) or not isinstance(end, int):
            continue
        if end <= start:
            continue
        spans.append(
            (
                rel_path,
                start,
                end,
                _OccurrenceCandidate(start, end, symbol, int(roles or 0)),
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
    call_row: dict[str, object],
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
            "extras_json": encode_payload(None),
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
    extras_json = {"candidate_symbols": candidate_symbols} if candidate_symbols else None
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
        "extras_json": encode_payload(extras_json),
    }


def _resolve_call_targets(
    calls: pl.DataFrame,
    occurrences: pl.DataFrame,
) -> pl.DataFrame:
    out_rows: list[dict[str, object]] = []
    calls_by_path = calls.partition_by("rel_path", as_dict=True)
    occs_by_path = occurrences.partition_by("rel_path", as_dict=True)

    for rel_path_key, calls_df in calls_by_path.items():
        rel_path = _rel_path_key(rel_path_key)
        if rel_path is None:
            continue
        resolver = _build_occurrence_resolver(
            rel_path=rel_path,
            occ_df=occs_by_path.get(rel_path_key),
        )
        out_rows.extend(
            [
                _call_target_row(call_row, rel_path=rel_path, resolver=resolver)
                for call_row in calls_df.iter_rows(named=True)
            ]
        )

    if not out_rows:
        return pl.DataFrame()
    return pl.DataFrame(out_rows)


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
    extras = _extras_struct(row, "extras_json")
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


def _extract_attribute_accesses(syntax_nodes: pl.DataFrame) -> list[_AttributeAccess]:
    accesses: list[_AttributeAccess] = []
    for row in syntax_nodes.iter_rows(named=True):
        access = _attribute_access_from_row(row)
        if access is not None:
            accesses.append(access)
    return accesses


def _extract_augassigns(syntax_nodes: pl.DataFrame) -> list[_AugAssignAccess]:
    records: list[_AugAssignAccess] = []
    for row in syntax_nodes.iter_rows(named=True):
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
        extras = _extras_struct(row, "extras_json")
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
    extras = _extras_struct(row, "extras_json")
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
    syntax_nodes: pl.DataFrame,
    context: _DescriptorAssignmentContext,
) -> dict[str, dict[str, str]]:
    assignments: dict[str, dict[str, str]] = {}
    for row in syntax_nodes.iter_rows(named=True):
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


def _call_targets_defs(
    defs_resolved: pl.LazyFrame,
) -> pl.LazyFrame:
    return (
        defs_resolved.filter(pl.col("scip_symbol").is_not_null())
        .filter(pl.col("def_kind").is_in(["function", "async_function"]))
        .select(
            "repo",
            "commit",
            "scip_symbol",
            "def_id",
            "syntax_node_id",
            "goid_h128",
        )
        .group_by(["repo", "commit", "scip_symbol"])
        .agg(
            [
                pl.first("def_id").alias("def_id"),
                pl.first("syntax_node_id").alias("syntax_node_id"),
                pl.first("goid_h128").alias("goid_h128"),
            ]
        )
    )


def _entry_blocks(cfg_blocks: pl.LazyFrame) -> pl.LazyFrame:
    return (
        cfg_blocks.filter(pl.col("kind") == "entry")
        .select("function_goid_h128", pl.col("block_id").alias("entry_block_id"))
        .unique(subset=["function_goid_h128"])
    )


def _exit_blocks(cfg_blocks: pl.LazyFrame) -> pl.LazyFrame:
    return (
        cfg_blocks.filter(pl.col("kind") == "exit")
        .select("function_goid_h128", pl.col("block_id").alias("exit_block_id"))
        .unique(subset=["function_goid_h128"])
    )


def _call_target_record(context: _CallTargetRecordContext) -> dict[str, object]:
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
        "extras_json": encode_payload(context.extras_json),
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
    extras_json = _coerce_payload_value(row.get("extras_json"))
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
        extras_json=extras_json,
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
                    extras_json=entry.extras_json,
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
                    extras_json=entry.extras_json,
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
                        extras_json=entry.extras_json,
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
                extras_json=entry.extras_json,
            )
        )
    ]


def _explicit_call_rows(resolved: pl.DataFrame, catalog: _DefCatalog) -> list[dict[str, object]]:
    explicit_rows: list[dict[str, object]] = []
    for row in resolved.iter_rows(named=True):
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
    syntax_nodes_df: pl.DataFrame,
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
                    extras_json={
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
                extras_json={
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
                    extras_json={
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
                extras_json={
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
                    extras_json={
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
                    extras_json={
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
                extras_json={
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
    syntax_nodes_df: pl.DataFrame,
    context: _ImplicitResolutionContext,
) -> list[dict[str, object]]:
    if syntax_nodes_df.is_empty():
        return []
    attributes = _extract_attribute_accesses(syntax_nodes_df)
    augassigns = _extract_augassigns(syntax_nodes_df)
    rows = _implicit_rows_from_attributes(attributes, context)
    rows.extend(_implicit_rows_from_augassigns(augassigns, context))
    return rows


def _explicit_rows_for_call_targets(
    calls: pl.LazyFrame,
    occurrences: pl.LazyFrame,
    catalog: _DefCatalog,
) -> list[dict[str, object]]:
    calls_df = calls.collect()
    if calls_df.is_empty():
        return []
    occ_df = occurrences.collect()
    resolved = _resolve_call_targets(calls_df, occ_df)
    return _explicit_call_rows(resolved, catalog)


def _implicit_rows_for_call_targets(
    syntax_nodes_df: pl.DataFrame,
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


def cpg_call_targets(
    q__core__syntax_calls: InferableTabularInput,
    q__core__scip_occurrence_span_xref: InferableTabularInput,
    q__core__syntax_defs_resolved: InferableTabularInput,
    q__graph__cfg_blocks: InferableTabularInput,
    q__core__syntax_nodes: InferableTabularInput,
) -> pl.LazyFrame:
    """Resolve call targets by welding callee spans to SCIP occurrences.

    Returns
    -------
    polars.LazyFrame
        Lazy frame for graph.cpg_call_targets.
    """
    calls = tabular_to_lazyframe(q__core__syntax_calls).select(
        "repo",
        "commit",
        "rel_path",
        "call_id",
        "call_node_id",
        "callee_start_byte",
        "callee_end_byte",
        "callee_text",
    )
    occurrences = tabular_to_lazyframe(q__core__scip_occurrence_span_xref).select(
        "rel_path",
        "scip_symbol",
        "roles",
        "start_byte",
        "end_byte",
    )
    defs_df = (
        tabular_to_lazyframe(q__core__syntax_defs_resolved)
        .select(
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
                "extras_json",
            ]
        )
        .collect()
    )
    catalog = _build_def_catalog(defs_df)

    explicit_rows = _explicit_rows_for_call_targets(calls, occurrences, catalog)
    syntax_nodes_df = (
        tabular_to_lazyframe(q__core__syntax_nodes)
        .filter(pl.col("extras_json").is_not_null())
        .select(
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
                "extras_json",
            ]
        )
        .collect()
    )

    implicit_rows = _implicit_rows_for_call_targets(syntax_nodes_df, catalog)

    all_rows = [*explicit_rows, *implicit_rows]
    if not all_rows:
        return empty_frame_for_table(CPG_CALL_TARGETS_TABLE_KEY)
    targets = pl.DataFrame(all_rows).lazy()
    blocks = tabular_to_lazyframe(q__graph__cfg_blocks)

    targets = join_validated(
        targets,
        _entry_blocks(blocks),
        spec=JoinSpec(
            left_on=["callee_goid_h128"],
            right_on=["function_goid_h128"],
            how="left",
            validate="m:1",
        ),
    ).drop(["function_goid_h128"])
    targets = join_validated(
        targets,
        _exit_blocks(blocks),
        spec=JoinSpec(
            left_on=["callee_goid_h128"],
            right_on=["function_goid_h128"],
            how="left",
            validate="m:1",
        ),
    ).drop(["function_goid_h128"])
    targets = targets.rename(
        {
            "entry_block_id": "callee_entry_block_id",
            "exit_block_id": "callee_exit_block_id",
        }
    )

    targets = targets.select(
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
            "extras_json",
        ]
    )
    return dedupe_frame_for_table(targets, table_key=CPG_CALL_TARGETS_TABLE_KEY)


def cpg_edges_calls(cpg_call_targets: pl.LazyFrame) -> pl.LazyFrame:
    """Build CALLS edges from call targets.

    Returns
    -------
    polars.LazyFrame
        Lazy frame for graph.cpg_edges_calls.
    """
    required = [
        "binding_kind",
        "target_role",
        "call_kind",
        "origin",
        "augop",
        "extras_json",
    ]
    edges = cpg_call_targets.filter(pl.col("callee_entry_block_id").is_not_null())
    available_columns = set(edges.columns)
    missing = [column for column in required if column not in available_columns]
    if missing:
        edges = edges.with_columns([pl.lit(None).alias(column) for column in missing])
    return edges.with_columns(
        pl.lit("CALLS").alias("edge_kind"),
        pl.struct(
            [
                "binding_kind",
                "target_role",
                "call_kind",
                "origin",
                "augop",
                "extras_json",
            ]
        )
        .map_elements(_call_edge_extras, return_dtype=pl.Binary)
        .alias("extras_json"),
    ).select(
        [
            "repo",
            "commit",
            "call_id",
            "call_node_id",
            "callee_entry_block_id",
            "edge_kind",
            "confidence",
            "extras_json",
        ]
    )


def _arg_edges_positional(args: pl.LazyFrame, params: pl.LazyFrame) -> pl.LazyFrame:
    pos_args = args.filter(pl.col("arg_kind") == "positional")
    non_variadic = params.filter(~pl.col("param_kind").is_in(["varargs", "varkw"]))
    return (
        join_validated(
            pos_args,
            non_variadic,
            spec=JoinSpec(
                left_on=["callee_def_id", "param_ordinal_hint"],
                right_on=["callee_def_id", "param_ordinal"],
                how="inner",
                validate="m:1",
            ),
        )
        .with_columns(pl.lit("ARG_TO_PARAM").alias("edge_kind"))
        .select(
            [
                "repo",
                "commit",
                "call_id",
                pl.col("arg_expr_node_id").alias("src_arg_node_id"),
                pl.col("param_node_id").alias("dst_param_node_id"),
                "edge_kind",
                "arg_ordinal",
                "param_ordinal",
                pl.col("arg_name").alias("arg_name"),
                pl.col("param_name").alias("param_name"),
                "arg_slot",
                "arg_role",
                "arg_is_implicit",
                "call_kind",
                "augop",
                "confidence",
                _payload_literal(None).alias("extras_json"),
            ]
        )
    )


def _arg_edges_keyword(args: pl.LazyFrame, params: pl.LazyFrame) -> pl.LazyFrame:
    kw_args = args.filter(pl.col("arg_kind") == "keyword")
    return (
        join_validated(
            kw_args,
            params,
            spec=JoinSpec(
                left_on=["callee_def_id", "arg_name"],
                right_on=["callee_def_id", "param_name"],
                how="inner",
                validate="m:1",
            ),
        )
        .with_columns(pl.lit("ARG_TO_PARAM").alias("edge_kind"))
        .select(
            [
                "repo",
                "commit",
                "call_id",
                pl.col("arg_expr_node_id").alias("src_arg_node_id"),
                pl.col("param_node_id").alias("dst_param_node_id"),
                "edge_kind",
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
                _payload_literal(None).alias("extras_json"),
            ]
        )
    )


def _arg_edges_star(
    args: pl.LazyFrame,
    params: pl.LazyFrame,
    *,
    arg_kind: str,
    param_kind: str,
    confidence_scale: float,
) -> pl.LazyFrame:
    subset = args.filter(pl.col("arg_kind") == arg_kind)
    var_params = params.filter(pl.col("param_kind") == param_kind)
    return (
        join_validated(
            subset,
            var_params,
            spec=JoinSpec(on=["callee_def_id"], how="inner", validate="m:1"),
        )
        .with_columns(
            pl.lit("ARG_TO_PARAM").alias("edge_kind"),
            (pl.col("confidence") * pl.lit(confidence_scale)).alias("confidence"),
        )
        .select(
            [
                "repo",
                "commit",
                "call_id",
                pl.col("arg_expr_node_id").alias("src_arg_node_id"),
                pl.col("param_node_id").alias("dst_param_node_id"),
                "edge_kind",
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
                _payload_literal(None).alias("extras_json"),
            ]
        )
    )


@dataclass(frozen=True)
class _ArgToParamFrames:
    call_targets: pl.LazyFrame
    explicit_targets: pl.LazyFrame
    args: pl.LazyFrame
    params: pl.LazyFrame


def _arg_to_param_frames(
    cpg_call_targets: pl.LazyFrame,
    q__core__syntax_call_args: InferableTabularInput,
    q__core__syntax_func_params: InferableTabularInput,
) -> _ArgToParamFrames:
    call_targets = cpg_call_targets.select(
        [
            "repo",
            "commit",
            "call_id",
            "call_node_id",
            "callee_def_id",
            "binding_kind",
            "target_role",
            "call_kind",
            "augop",
            "confidence",
            "extras_json",
        ]
    )
    explicit_targets = call_targets.filter(pl.col("call_kind") == _CALL_KIND_EXPLICIT).filter(
        (pl.col("target_role") == _TARGET_ROLE_INIT)
        | (pl.col("binding_kind") != _BINDING_CONSTRUCTOR)
    )
    args = tabular_to_lazyframe(q__core__syntax_call_args).join(
        explicit_targets,
        on=["repo", "commit", "call_id"],
        how="left",
    )
    args = args.filter(pl.col("callee_def_id").is_not_null())
    args = args.with_columns(
        pl.col("confidence").fill_null(0.0),
        pl.when(pl.col("arg_kind") == "positional")
        .then(pl.format("positional:{}", pl.col("arg_ordinal")))
        .when(pl.col("arg_kind") == "keyword")
        .then(pl.format("keyword:{}", pl.col("arg_name")))
        .when(pl.col("arg_kind") == "starargs")
        .then(pl.lit("positional:*"))
        .when(pl.col("arg_kind") == "kwargs")
        .then(pl.lit("keyword:**"))
        .otherwise(pl.lit(None))
        .alias("arg_slot"),
        pl.when(pl.col("arg_kind") == "keyword")
        .then(pl.lit("keyword"))
        .otherwise(pl.lit("positional"))
        .alias("arg_role"),
        pl.lit(value=False).alias("arg_is_implicit"),
        pl.when(
            pl.col("binding_kind").is_in(
                [
                    _BINDING_BOUND_METHOD,
                    _BINDING_CLASSMETHOD,
                ]
            )
        )
        .then(pl.col("arg_ordinal") + pl.lit(1))
        .otherwise(pl.col("arg_ordinal"))
        .alias("param_ordinal_hint"),
    )
    params = tabular_to_lazyframe(q__core__syntax_func_params).select(
        [
            pl.col("func_def_id").alias("callee_def_id"),
            "param_ordinal",
            "param_kind",
            "param_name",
            "param_node_id",
        ]
    )
    return _ArgToParamFrames(
        call_targets=call_targets,
        explicit_targets=explicit_targets,
        args=args,
        params=params,
    )


def _explicit_arg_edges(args: pl.LazyFrame, params: pl.LazyFrame) -> pl.LazyFrame:
    frames = [
        _arg_edges_positional(args, params),
        _arg_edges_keyword(args, params),
        _arg_edges_star(
            args,
            params,
            arg_kind="starargs",
            param_kind="varargs",
            confidence_scale=0.7,
        ),
        _arg_edges_star(args, params, arg_kind="kwargs", param_kind="varkw", confidence_scale=0.6),
    ]
    return pl.concat(frames, how="vertical_relaxed")


def _implicit_receiver_edges(
    explicit_targets: pl.LazyFrame,
    params: pl.LazyFrame,
) -> pl.LazyFrame:
    implicit_receiver = explicit_targets.filter(
        pl.col("binding_kind").is_in([_BINDING_BOUND_METHOD, _BINDING_CLASSMETHOD])
    ).with_columns(
        pl.lit(0).alias("param_ordinal"),
        pl.lit("implicit:receiver").alias("arg_slot"),
        pl.lit("receiver").alias("arg_role"),
        pl.lit(value=True).alias("arg_is_implicit"),
        pl.lit(None).cast(pl.Int64).alias("arg_ordinal"),
        pl.lit(None).cast(pl.Utf8).alias("arg_name"),
        pl.col("call_node_id").alias("arg_expr_node_id"),
        pl.col("confidence").fill_null(0.0),
    )
    return (
        join_validated(
            implicit_receiver,
            params,
            spec=JoinSpec(
                on=["callee_def_id", "param_ordinal"],
                how="left",
                validate="m:1",
            ),
        )
        .with_columns(pl.lit("ARG_TO_PARAM").alias("edge_kind"))
        .select(
            [
                "repo",
                "commit",
                "call_id",
                pl.col("arg_expr_node_id").alias("src_arg_node_id"),
                pl.col("param_node_id").alias("dst_param_node_id"),
                "edge_kind",
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
                _payload_literal(None).alias("extras_json"),
            ]
        )
    )


def _implicit_descriptor_edges(
    call_targets: pl.LazyFrame,
    params: pl.LazyFrame,
) -> pl.LazyFrame:
    implicit_calls = call_targets.filter(
        pl.col("call_kind").is_in(
            [
                _CALL_KIND_IMPLICIT_GET,
                _CALL_KIND_IMPLICIT_SET,
                _CALL_KIND_IMPLICIT_SET_AUG,
            ]
        )
    ).with_columns(
        pl.col("extras_json").map_elements(decode_payload, return_dtype=pl.Object).alias("extras"),
    )
    implicit_calls = implicit_calls.with_columns(
        pl.col("extras")
        .map_elements(_extras_descriptor_obj_is_none, return_dtype=pl.Boolean)
        .alias("descriptor_obj_is_none")
    )
    implicit_prop_get = implicit_calls.filter(pl.col("binding_kind") == _BINDING_PROPERTY_GET)
    implicit_prop_set = implicit_calls.filter(pl.col("binding_kind") == _BINDING_PROPERTY_SET)
    implicit_desc_get = implicit_calls.filter(pl.col("binding_kind") == _BINDING_DESCRIPTOR_GET)
    implicit_desc_set = implicit_calls.filter(
        pl.col("binding_kind").is_in([_BINDING_DESCRIPTOR_SET, _BINDING_DESCRIPTOR_SET_AUG])
    )
    frames: list[pl.LazyFrame] = []
    frames.append(
        implicit_prop_get.with_columns(
            pl.lit(0).alias("param_ordinal"),
            pl.lit("implicit:receiver").alias("arg_slot"),
            pl.lit("receiver").alias("arg_role"),
            pl.lit(value=True).alias("arg_is_implicit"),
            pl.lit(None).cast(pl.Int64).alias("arg_ordinal"),
            pl.lit(None).cast(pl.Utf8).alias("arg_name"),
            pl.col("call_node_id").alias("arg_expr_node_id"),
            pl.col("confidence").fill_null(0.0),
        )
        .pipe(
            join_validated,
            params,
            spec=JoinSpec(
                on=["callee_def_id", "param_ordinal"],
                how="left",
                validate="m:1",
            ),
        )
        .with_columns(pl.lit("ARG_TO_PARAM").alias("edge_kind"))
        .select(
            [
                "repo",
                "commit",
                "call_id",
                pl.col("arg_expr_node_id").alias("src_arg_node_id"),
                pl.col("param_node_id").alias("dst_param_node_id"),
                "edge_kind",
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
                _payload_literal(None).alias("extras_json"),
            ]
        )
    )
    frames.append(
        implicit_prop_set.with_columns(
            pl.lit(0).alias("param_ordinal"),
            pl.lit("implicit:receiver").alias("arg_slot"),
            pl.lit("receiver").alias("arg_role"),
            pl.lit(value=True).alias("arg_is_implicit"),
            pl.lit(None).cast(pl.Int64).alias("arg_ordinal"),
            pl.lit(None).cast(pl.Utf8).alias("arg_name"),
            pl.col("call_node_id").alias("arg_expr_node_id"),
            pl.col("confidence").fill_null(0.0),
        )
        .pipe(
            join_validated,
            params,
            spec=JoinSpec(
                on=["callee_def_id", "param_ordinal"],
                how="left",
                validate="m:1",
            ),
        )
        .with_columns(pl.lit("ARG_TO_PARAM").alias("edge_kind"))
        .select(
            [
                "repo",
                "commit",
                "call_id",
                pl.col("arg_expr_node_id").alias("src_arg_node_id"),
                pl.col("param_node_id").alias("dst_param_node_id"),
                "edge_kind",
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
                _payload_literal(None).alias("extras_json"),
            ]
        )
    )
    frames.append(
        implicit_prop_set.with_columns(
            pl.lit(1).alias("param_ordinal"),
            pl.lit("implicit:value").alias("arg_slot"),
            pl.lit("descriptor_value").alias("arg_role"),
            pl.lit(value=True).alias("arg_is_implicit"),
            pl.lit(None).cast(pl.Int64).alias("arg_ordinal"),
            pl.lit(None).cast(pl.Utf8).alias("arg_name"),
            pl.col("call_node_id").alias("arg_expr_node_id"),
            pl.col("confidence").fill_null(0.0),
        )
        .pipe(
            join_validated,
            params,
            spec=JoinSpec(
                on=["callee_def_id", "param_ordinal"],
                how="left",
                validate="m:1",
            ),
        )
        .with_columns(pl.lit("ARG_TO_PARAM").alias("edge_kind"))
        .select(
            [
                "repo",
                "commit",
                "call_id",
                pl.col("arg_expr_node_id").alias("src_arg_node_id"),
                pl.col("param_node_id").alias("dst_param_node_id"),
                "edge_kind",
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
                _payload_literal(None).alias("extras_json"),
            ]
        )
    )
    frames.append(
        implicit_desc_get.with_columns(
            pl.lit(0).alias("param_ordinal"),
            pl.lit("implicit:receiver").alias("arg_slot"),
            pl.lit("receiver").alias("arg_role"),
            pl.lit(value=True).alias("arg_is_implicit"),
            pl.lit(None).cast(pl.Int64).alias("arg_ordinal"),
            pl.lit(None).cast(pl.Utf8).alias("arg_name"),
            pl.col("call_node_id").alias("arg_expr_node_id"),
            pl.col("confidence").fill_null(0.0),
        )
        .pipe(
            join_validated,
            params,
            spec=JoinSpec(
                on=["callee_def_id", "param_ordinal"],
                how="left",
                validate="m:1",
            ),
        )
        .with_columns(pl.lit("ARG_TO_PARAM").alias("edge_kind"))
        .select(
            [
                "repo",
                "commit",
                "call_id",
                pl.col("arg_expr_node_id").alias("src_arg_node_id"),
                pl.col("param_node_id").alias("dst_param_node_id"),
                "edge_kind",
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
                _payload_literal(None).alias("extras_json"),
            ]
        )
    )
    frames.append(
        implicit_desc_get.with_columns(
            pl.lit(1).alias("param_ordinal"),
            pl.when(pl.col("descriptor_obj_is_none"))
            .then(pl.lit("implicit:none"))
            .otherwise(pl.lit("implicit:obj"))
            .alias("arg_slot"),
            pl.when(pl.col("descriptor_obj_is_none"))
            .then(pl.lit("descriptor_none"))
            .otherwise(pl.lit("descriptor_obj"))
            .alias("arg_role"),
            pl.lit(value=True).alias("arg_is_implicit"),
            pl.lit(None).cast(pl.Int64).alias("arg_ordinal"),
            pl.lit(None).cast(pl.Utf8).alias("arg_name"),
            pl.col("call_node_id").alias("arg_expr_node_id"),
            pl.col("confidence").fill_null(0.0),
        )
        .pipe(
            join_validated,
            params,
            spec=JoinSpec(
                on=["callee_def_id", "param_ordinal"],
                how="left",
                validate="m:1",
            ),
        )
        .with_columns(pl.lit("ARG_TO_PARAM").alias("edge_kind"))
        .select(
            [
                "repo",
                "commit",
                "call_id",
                pl.col("arg_expr_node_id").alias("src_arg_node_id"),
                pl.col("param_node_id").alias("dst_param_node_id"),
                "edge_kind",
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
                _payload_literal(None).alias("extras_json"),
            ]
        )
    )
    frames.append(
        implicit_desc_get.with_columns(
            pl.lit(2).alias("param_ordinal"),
            pl.lit("implicit:objtype").alias("arg_slot"),
            pl.lit("descriptor_objtype").alias("arg_role"),
            pl.lit(value=True).alias("arg_is_implicit"),
            pl.lit(None).cast(pl.Int64).alias("arg_ordinal"),
            pl.lit(None).cast(pl.Utf8).alias("arg_name"),
            pl.col("call_node_id").alias("arg_expr_node_id"),
            pl.col("confidence").fill_null(0.0),
        )
        .pipe(
            join_validated,
            params,
            spec=JoinSpec(
                on=["callee_def_id", "param_ordinal"],
                how="left",
                validate="m:1",
            ),
        )
        .with_columns(pl.lit("ARG_TO_PARAM").alias("edge_kind"))
        .select(
            [
                "repo",
                "commit",
                "call_id",
                pl.col("arg_expr_node_id").alias("src_arg_node_id"),
                pl.col("param_node_id").alias("dst_param_node_id"),
                "edge_kind",
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
                _payload_literal(None).alias("extras_json"),
            ]
        )
    )
    frames.append(
        implicit_desc_set.with_columns(
            pl.lit(0).alias("param_ordinal"),
            pl.lit("implicit:receiver").alias("arg_slot"),
            pl.lit("receiver").alias("arg_role"),
            pl.lit(value=True).alias("arg_is_implicit"),
            pl.lit(None).cast(pl.Int64).alias("arg_ordinal"),
            pl.lit(None).cast(pl.Utf8).alias("arg_name"),
            pl.col("call_node_id").alias("arg_expr_node_id"),
            pl.col("confidence").fill_null(0.0),
        )
        .pipe(
            join_validated,
            params,
            spec=JoinSpec(
                on=["callee_def_id", "param_ordinal"],
                how="left",
                validate="m:1",
            ),
        )
        .with_columns(pl.lit("ARG_TO_PARAM").alias("edge_kind"))
        .select(
            [
                "repo",
                "commit",
                "call_id",
                pl.col("arg_expr_node_id").alias("src_arg_node_id"),
                pl.col("param_node_id").alias("dst_param_node_id"),
                "edge_kind",
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
                _payload_literal(None).alias("extras_json"),
            ]
        )
    )
    frames.append(
        implicit_desc_set.with_columns(
            pl.lit(1).alias("param_ordinal"),
            pl.lit("implicit:obj").alias("arg_slot"),
            pl.lit("descriptor_obj").alias("arg_role"),
            pl.lit(value=True).alias("arg_is_implicit"),
            pl.lit(None).cast(pl.Int64).alias("arg_ordinal"),
            pl.lit(None).cast(pl.Utf8).alias("arg_name"),
            pl.col("call_node_id").alias("arg_expr_node_id"),
            pl.col("confidence").fill_null(0.0),
        )
        .join(params, on=["callee_def_id", "param_ordinal"], how="left")
        .with_columns(pl.lit("ARG_TO_PARAM").alias("edge_kind"))
        .select(
            [
                "repo",
                "commit",
                "call_id",
                pl.col("arg_expr_node_id").alias("src_arg_node_id"),
                pl.col("param_node_id").alias("dst_param_node_id"),
                "edge_kind",
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
                _payload_literal(None).alias("extras_json"),
            ]
        )
    )
    frames.append(
        implicit_desc_set.with_columns(
            pl.lit(2).alias("param_ordinal"),
            pl.lit("implicit:value").alias("arg_slot"),
            pl.when(pl.col("call_kind") == _CALL_KIND_IMPLICIT_SET_AUG)
            .then(pl.lit("augassign_value"))
            .otherwise(pl.lit("descriptor_value"))
            .alias("arg_role"),
            pl.lit(value=True).alias("arg_is_implicit"),
            pl.lit(None).cast(pl.Int64).alias("arg_ordinal"),
            pl.lit(None).cast(pl.Utf8).alias("arg_name"),
            pl.col("call_node_id").alias("arg_expr_node_id"),
            pl.col("confidence").fill_null(0.0),
        )
        .join(params, on=["callee_def_id", "param_ordinal"], how="left")
        .with_columns(pl.lit("ARG_TO_PARAM").alias("edge_kind"))
        .select(
            [
                "repo",
                "commit",
                "call_id",
                pl.col("arg_expr_node_id").alias("src_arg_node_id"),
                pl.col("param_node_id").alias("dst_param_node_id"),
                "edge_kind",
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
                _payload_literal(None).alias("extras_json"),
            ]
        )
    )
    return pl.concat(frames, how="vertical_relaxed")


def cpg_edges_arg_to_param(
    cpg_call_targets: pl.LazyFrame,
    q__core__syntax_call_args: InferableTabularInput,
    q__core__syntax_func_params: InferableTabularInput,
) -> pl.LazyFrame:
    """Build ARG_TO_PARAM edges from call arguments and function params.

    Returns
    -------
    pl.LazyFrame
        ARG_TO_PARAM edges for graph.cpg_edges_arg_to_param.
    """
    frames = _arg_to_param_frames(
        cpg_call_targets,
        q__core__syntax_call_args,
        q__core__syntax_func_params,
    )
    explicit_edges = _explicit_arg_edges(frames.args, frames.params)
    implicit_receiver_edges = _implicit_receiver_edges(frames.explicit_targets, frames.params)
    implicit_descriptor_edges = _implicit_descriptor_edges(frames.call_targets, frames.params)
    implicit_edges = pl.concat(
        [implicit_descriptor_edges, implicit_receiver_edges],
        how="vertical_relaxed",
    )
    combined = pl.concat([explicit_edges, implicit_edges], how="vertical_relaxed")
    if not combined.columns:
        return empty_frame_for_table(CPG_ARG_TO_PARAM_EDGES_TABLE_KEY)
    return dedupe_frame_for_table(combined, table_key=CPG_ARG_TO_PARAM_EDGES_TABLE_KEY)


def cpg_edges_ret_to_call(cpg_call_targets: pl.LazyFrame) -> pl.LazyFrame:
    """Build RET_TO_CALL edges using callee exit block summaries.

    Returns
    -------
    pl.LazyFrame
        RET_TO_CALL edges for graph.cpg_edges_ret_to_call.
    """
    edges = (
        cpg_call_targets.filter(pl.col("callee_exit_block_id").is_not_null())
        .filter(pl.col("target_role") == _TARGET_ROLE_PRIMARY)
        .filter(
            ~pl.col("binding_kind").is_in(
                [
                    _BINDING_PROPERTY_SET,
                    _BINDING_DESCRIPTOR_SET,
                    _BINDING_DESCRIPTOR_SET_AUG,
                ]
            )
        )
        .with_columns(pl.lit("RET_TO_CALL").alias("edge_kind"))
        .select(
            [
                "repo",
                "commit",
                "call_id",
                pl.col("callee_exit_block_id").alias("exit_block_id"),
                "call_node_id",
                "target_role",
                "call_kind",
                "origin",
                "edge_kind",
                (pl.col("confidence") * pl.lit(0.9)).alias("confidence"),
                _payload_literal({"summary_kind": "exit_block"}).alias("extras_json"),
            ]
        )
    )
    return dedupe_frame_for_table(edges, table_key=CPG_RET_TO_CALL_EDGES_TABLE_KEY)


__all__ = [
    "CALL_WIRING_TARGET_NAME",
    "CPG_ARG_TO_PARAM_EDGES_TABLE_KEY",
    "CPG_CALL_EDGES_TABLE_KEY",
    "CPG_CALL_TARGETS_TABLE_KEY",
    "CPG_RET_TO_CALL_EDGES_TABLE_KEY",
    "cpg_call_targets",
    "cpg_edges_arg_to_param",
    "cpg_edges_calls",
    "cpg_edges_ret_to_call",
]
