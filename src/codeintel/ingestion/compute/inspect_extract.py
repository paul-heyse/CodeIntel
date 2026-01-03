"""Inspect extraction step with port injection."""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import inspect
import logging
import multiprocessing
import queue
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import (
    AsyncGeneratorType,
    CodeType,
    CoroutineType,
    FrameType,
    GeneratorType,
)
from typing import TYPE_CHECKING, cast

from codeintel.build.hamilton.execution_result import ExecutionResult
from codeintel.build.hamilton.native.options.ingestion import InspectExtractOptions
from codeintel.core.columnar.rows import (
    ColumnarRowBuffer,
    ColumnarRows,
    columnar_buffer_for_table_key,
    empty_reader_for_table,
    record_batch_reader_for_columnar_rows,
)
from codeintel.ingestion.compute.base import BaseExtractStep

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    import pyarrow as pa

    from codeintel.ingestion.ports.discovery import ModuleDiscoveryPort, ModuleRecord

type InspectableCallable = Callable[..., object] | type[object]

LOG = logging.getLogger(__name__)

PY_INSPECT_OBJECTS_TABLE_KEY = "core.py_inspect_objects"
PY_INSPECT_MEMBERS_TABLE_KEY = "core.py_inspect_members_static"
PY_INSPECT_CLASS_MRO_TABLE_KEY = "core.py_inspect_class_mro"
PY_INSPECT_CLASS_ATTRS_TABLE_KEY = "core.py_inspect_class_attrs"
PY_INSPECT_UNWRAP_TABLE_KEY = "core.py_inspect_unwrap_hops"
PY_INSPECT_SIGNATURES_TABLE_KEY = "core.py_inspect_signatures"
PY_INSPECT_SIGNATURE_PARAMS_TABLE_KEY = "core.py_inspect_signature_params"
PY_INSPECT_ANNOTATIONS_TABLE_KEY = "core.py_inspect_annotations_kv"
PY_INSPECT_SOURCE_TABLE_KEY = "core.py_inspect_source"
PY_INSPECT_RUNTIME_STATE_TABLE_KEY = "core.py_inspect_runtime_state"
_ALLOWLIST_PREVIEW_LIMIT = 5


@dataclass(frozen=True)
class InspectExtractResult:
    """Result bundle for inspect extraction."""

    result: ExecutionResult
    object_rows: ColumnarRows = field(default_factory=dict)
    member_rows: ColumnarRows = field(default_factory=dict)
    class_mro_rows: ColumnarRows = field(default_factory=dict)
    class_attr_rows: ColumnarRows = field(default_factory=dict)
    unwrap_rows: ColumnarRows = field(default_factory=dict)
    signature_rows: ColumnarRows = field(default_factory=dict)
    signature_param_rows: ColumnarRows = field(default_factory=dict)
    annotation_rows: ColumnarRows = field(default_factory=dict)
    source_rows: ColumnarRows = field(default_factory=dict)
    runtime_state_rows: ColumnarRows = field(default_factory=dict)
    object_rows_reader: pa.RecordBatchReader = field(
        default_factory=lambda: empty_reader_for_table(PY_INSPECT_OBJECTS_TABLE_KEY)
    )
    member_rows_reader: pa.RecordBatchReader = field(
        default_factory=lambda: empty_reader_for_table(PY_INSPECT_MEMBERS_TABLE_KEY)
    )
    class_mro_rows_reader: pa.RecordBatchReader = field(
        default_factory=lambda: empty_reader_for_table(PY_INSPECT_CLASS_MRO_TABLE_KEY)
    )
    class_attr_rows_reader: pa.RecordBatchReader = field(
        default_factory=lambda: empty_reader_for_table(PY_INSPECT_CLASS_ATTRS_TABLE_KEY)
    )
    unwrap_rows_reader: pa.RecordBatchReader = field(
        default_factory=lambda: empty_reader_for_table(PY_INSPECT_UNWRAP_TABLE_KEY)
    )
    signature_rows_reader: pa.RecordBatchReader = field(
        default_factory=lambda: empty_reader_for_table(PY_INSPECT_SIGNATURES_TABLE_KEY)
    )
    signature_param_rows_reader: pa.RecordBatchReader = field(
        default_factory=lambda: empty_reader_for_table(PY_INSPECT_SIGNATURE_PARAMS_TABLE_KEY)
    )
    annotation_rows_reader: pa.RecordBatchReader = field(
        default_factory=lambda: empty_reader_for_table(PY_INSPECT_ANNOTATIONS_TABLE_KEY)
    )
    source_rows_reader: pa.RecordBatchReader = field(
        default_factory=lambda: empty_reader_for_table(PY_INSPECT_SOURCE_TABLE_KEY)
    )
    runtime_state_rows_reader: pa.RecordBatchReader = field(
        default_factory=lambda: empty_reader_for_table(PY_INSPECT_RUNTIME_STATE_TABLE_KEY)
    )
    object_row_count: int = 0
    member_row_count: int = 0
    class_mro_row_count: int = 0
    class_attr_row_count: int = 0
    unwrap_row_count: int = 0
    signature_row_count: int = 0
    signature_param_row_count: int = 0
    annotation_row_count: int = 0
    source_row_count: int = 0
    runtime_state_row_count: int = 0


@dataclass(frozen=True, slots=True)
class _InspectContext:
    repo: str
    commit: str
    mode: str
    object_id: str
    follow_wrapped: bool
    eval_str: bool


@dataclass(frozen=True, slots=True)
class _InspectBuffers:
    objects: ColumnarRowBuffer
    members: ColumnarRowBuffer
    class_mro: ColumnarRowBuffer
    class_attrs: ColumnarRowBuffer
    unwrap: ColumnarRowBuffer
    signatures: ColumnarRowBuffer
    signature_params: ColumnarRowBuffer
    annotations: ColumnarRowBuffer
    sources: ColumnarRowBuffer
    runtime_state: ColumnarRowBuffer


@dataclass(slots=True)
class _InspectState:
    buffers: _InspectBuffers
    seen_objects: set[str]
    warnings: list[str]
    repo: str
    commit: str
    mode: str
    object_limit: int
    eval_str: bool
    follow_wrapped: bool
    object_count: int = 0


@dataclass(frozen=True, slots=True)
class _RuntimeFrameInfo:
    frame: FrameType | None
    frame_object_id: str | None
    frame_line: int | None
    frame_offset: int | None


@dataclass(frozen=True, slots=True)
class _RuntimeStateInfo:
    object_kind: str
    state_kind: str
    state: str | None
    status: dict[str, object]


@dataclass(frozen=True, slots=True)
class _InspectWorkerPayload:
    warnings: list[str]
    object_rows: ColumnarRows
    member_rows: ColumnarRows
    class_mro_rows: ColumnarRows
    class_attr_rows: ColumnarRows
    unwrap_rows: ColumnarRows
    signature_rows: ColumnarRows
    signature_param_rows: ColumnarRows
    annotation_rows: ColumnarRows
    source_rows: ColumnarRows
    runtime_state_rows: ColumnarRows
    object_row_count: int
    member_row_count: int
    class_mro_row_count: int
    class_attr_row_count: int
    unwrap_row_count: int
    signature_row_count: int
    signature_param_row_count: int
    annotation_row_count: int
    source_row_count: int
    runtime_state_row_count: int


@dataclass(frozen=True, slots=True)
class _InspectWorkerJob:
    modules: Sequence[ModuleRecord]
    repo: str
    commit: str
    options: InspectExtractOptions
    seed_warnings: list[str]


def _stable_id(*parts: object) -> str:
    payload = "|".join("" if part is None else str(part) for part in parts)
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=16)
    return digest.hexdigest()


def _build_inspect_buffers() -> _InspectBuffers:
    return _InspectBuffers(
        objects=columnar_buffer_for_table_key(PY_INSPECT_OBJECTS_TABLE_KEY),
        members=columnar_buffer_for_table_key(PY_INSPECT_MEMBERS_TABLE_KEY),
        class_mro=columnar_buffer_for_table_key(PY_INSPECT_CLASS_MRO_TABLE_KEY),
        class_attrs=columnar_buffer_for_table_key(PY_INSPECT_CLASS_ATTRS_TABLE_KEY),
        unwrap=columnar_buffer_for_table_key(PY_INSPECT_UNWRAP_TABLE_KEY),
        signatures=columnar_buffer_for_table_key(PY_INSPECT_SIGNATURES_TABLE_KEY),
        signature_params=columnar_buffer_for_table_key(PY_INSPECT_SIGNATURE_PARAMS_TABLE_KEY),
        annotations=columnar_buffer_for_table_key(PY_INSPECT_ANNOTATIONS_TABLE_KEY),
        sources=columnar_buffer_for_table_key(PY_INSPECT_SOURCE_TABLE_KEY),
        runtime_state=columnar_buffer_for_table_key(PY_INSPECT_RUNTIME_STATE_TABLE_KEY),
    )


def _ok_status() -> dict[str, object]:
    return {"ok": True}


def _error_status(exc: Exception) -> dict[str, object]:
    return {
        "ok": False,
        "error_type": type(exc).__name__,
        "error_msg": str(exc),
    }


def _truncate_repr(value: object, limit: int = 240) -> str:
    text = value if isinstance(value, str) else repr(value)
    if len(text) <= limit:
        return text
    return text[:limit]


def _value_ref(value: object) -> dict[str, object]:
    try:
        repr_text = repr(value)
    except (TypeError, ValueError, RecursionError):
        repr_text = "<unreprable>"
    repr_bytes = repr_text.encode("utf-8", errors="replace")
    return {
        "kind": type(value).__name__,
        "type_qualname": type(value).__qualname__,
        "repr_trunc": _truncate_repr(repr_text),
        "repr_len": len(repr_text),
        "repr_sha256": hashlib.sha256(repr_bytes).digest(),
        "is_callable": callable(value),
        "is_descriptor": inspect.isdatadescriptor(value)
        or inspect.ismemberdescriptor(value)
        or inspect.isgetsetdescriptor(value),
        "is_builtin": inspect.isbuiltin(value),
    }


def _object_kind(value: object) -> str:
    if inspect.ismodule(value):
        kind = "module"
    elif inspect.isclass(value):
        kind = "class"
    elif inspect.isfunction(value):
        kind = "function"
    elif inspect.ismethod(value):
        kind = "method"
    elif inspect.isbuiltin(value):
        kind = "builtin"
    elif inspect.isroutine(value):
        kind = "routine"
    else:
        kind = type(value).__name__
    return kind


def _runtime_frame(value: object) -> FrameType | None:
    if inspect.isframe(value):
        return value
    if inspect.istraceback(value):
        return value.tb_frame
    if inspect.isgenerator(value):
        return value.gi_frame
    if inspect.iscoroutine(value):
        return value.cr_frame
    if inspect.isasyncgen(value):
        return value.ag_frame
    return None


def _runtime_code(value: object) -> CodeType | None:
    if inspect.isframe(value):
        return value.f_code
    if inspect.istraceback(value):
        return value.tb_frame.f_code
    if inspect.isgenerator(value):
        return value.gi_code
    if inspect.iscoroutine(value):
        return value.cr_code
    if inspect.isasyncgen(value):
        return value.ag_code
    return None


def _frame_module_name(frame: FrameType | None) -> str | None:
    if frame is None:
        return None
    module_name = frame.f_globals.get("__name__")
    return module_name if isinstance(module_name, str) else None


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _object_module_name(value: object) -> str | None:
    if inspect.ismodule(value):
        return getattr(value, "__name__", None)
    runtime_frame = _runtime_frame(value)
    runtime_module = _frame_module_name(runtime_frame)
    if runtime_module is not None:
        return runtime_module
    module_name = getattr(value, "__module__", None)
    return module_name if isinstance(module_name, str) else None


def _object_qualname(value: object) -> str | None:
    qualname = getattr(value, "__qualname__", None)
    if isinstance(qualname, str):
        return qualname
    runtime_code = _runtime_code(value)
    if runtime_code is not None:
        runtime_qualname = getattr(runtime_code, "co_qualname", None)
        if isinstance(runtime_qualname, str) and runtime_qualname:
            return runtime_qualname
        runtime_name = getattr(runtime_code, "co_name", None)
        if isinstance(runtime_name, str) and runtime_name:
            return runtime_name
    name = getattr(value, "__name__", None)
    return name if isinstance(name, str) else None


def _object_name(value: object) -> str | None:
    name = getattr(value, "__name__", None)
    return name if isinstance(name, str) else None


def _object_id(value: object, kind: str) -> str:
    module_name = _object_module_name(value)
    qualname = _object_qualname(value)
    if (
        inspect.isframe(value)
        or inspect.istraceback(value)
        or inspect.isgenerator(value)
        or inspect.iscoroutine(value)
        or inspect.isasyncgen(value)
    ):
        return _stable_id("py_inspect_obj", module_name, qualname, kind, id(value))
    if module_name is None and qualname is None:
        return _stable_id("py_inspect_obj", kind, id(value))
    return _stable_id("py_inspect_obj", module_name, qualname, kind)


def _frame_position_info(
    frame: FrameType | None,
) -> tuple[int | None, int | None, int | None, int | None]:
    if frame is None:
        return None, None, None, None
    try:
        info = inspect.getframeinfo(frame)
    except (TypeError, ValueError):
        return None, None, None, None
    positions = getattr(info, "positions", None)
    if positions is None:
        return None, None, None, None
    return (
        _int_or_none(getattr(positions, "lineno", None)),
        _int_or_none(getattr(positions, "end_lineno", None)),
        _int_or_none(getattr(positions, "col_offset", None)),
        _int_or_none(getattr(positions, "end_col_offset", None)),
    )


def _frame_locals_payload(frame: FrameType | None) -> list[dict[str, object]]:
    if frame is None:
        return []
    try:
        locals_map = frame.f_locals
    except AttributeError:
        return []
    if isinstance(locals_map, dict):
        items = locals_map.items()
    else:
        try:
            items = locals_map.items()
        except AttributeError:
            return []
    payload: list[dict[str, object]] = []
    for key, value in items:
        if not isinstance(key, str):
            continue
        payload.append({"name": key, "value": _value_ref(value)})
    return payload


def _frame_line(value: object, frame: FrameType | None) -> int | None:
    if inspect.istraceback(value):
        return _int_or_none(value.tb_lineno)
    if frame is None:
        return None
    return _int_or_none(frame.f_lineno)


def _frame_offset(value: object, frame: FrameType | None) -> int | None:
    if inspect.istraceback(value):
        return _int_or_none(value.tb_lasti)
    if frame is None:
        return None
    return _int_or_none(frame.f_lasti)


def _runtime_state_row(
    *,
    context: _InspectContext,
    frame_info: _RuntimeFrameInfo,
    state_info: _RuntimeStateInfo,
) -> dict[str, object]:
    frame = frame_info.frame
    frame_file = None
    if frame is not None and isinstance(frame.f_code.co_filename, str):
        frame_file = frame.f_code.co_filename
    frame_module = _frame_module_name(frame)
    frame_code_qualname = None
    frame_code_name = None
    frame_firstlineno = None
    if frame is not None:
        if isinstance(frame.f_code.co_name, str):
            frame_code_name = frame.f_code.co_name
        qualname = getattr(frame.f_code, "co_qualname", None)
        if isinstance(qualname, str):
            frame_code_qualname = qualname
        frame_firstlineno = _int_or_none(frame.f_code.co_firstlineno)
    frame_start_line, frame_end_line, frame_start_col, frame_end_col = _frame_position_info(frame)
    return {
        "repo": context.repo,
        "commit": context.commit,
        "mode": context.mode,
        "object_id": context.object_id,
        "object_kind": state_info.object_kind,
        "state_kind": state_info.state_kind,
        "state": state_info.state,
        "frame_object_id": frame_info.frame_object_id,
        "frame_file": frame_file,
        "frame_module": frame_module,
        "frame_code_qualname": frame_code_qualname,
        "frame_code_name": frame_code_name,
        "frame_firstlineno": frame_firstlineno,
        "frame_line": frame_info.frame_line,
        "frame_start_line": frame_start_line,
        "frame_end_line": frame_end_line,
        "frame_start_col": frame_start_col,
        "frame_end_col": frame_end_col,
        "frame_offset": frame_info.frame_offset,
        "locals": _frame_locals_payload(frame),
        "status": state_info.status,
    }


def _is_runtime_object(value: object) -> bool:
    return (
        inspect.isframe(value)
        or inspect.istraceback(value)
        or inspect.isgenerator(value)
        or inspect.iscoroutine(value)
        or inspect.isasyncgen(value)
    )


def _has_signature_override(value: object) -> bool:
    return hasattr(value, "__signature__")


def _has_wrapped(value: object) -> bool:
    return getattr(value, "__wrapped__", None) is not None


def _annotation_payload(
    annotations: dict[str, object],
    *,
    context: _InspectContext,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key, value in annotations.items():
        if not isinstance(key, str):
            continue
        rows.append(
            {
                "repo": context.repo,
                "commit": context.commit,
                "mode": context.mode,
                "object_id": context.object_id,
                "eval_str": context.eval_str,
                "key": key,
                "value": _value_ref(value),
                "status": _ok_status(),
            }
        )
    return rows


def _signature_rows(
    value: InspectableCallable,
    *,
    context: _InspectContext,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    signature_rows: list[dict[str, object]] = []
    param_rows: list[dict[str, object]] = []
    signature_id = _stable_id(
        "py_inspect_sig",
        context.object_id,
        context.follow_wrapped,
        context.eval_str,
    )
    try:
        signature = inspect.signature(value, follow_wrapped=context.follow_wrapped)
    except (TypeError, ValueError) as exc:
        signature_rows.append(
            {
                "repo": context.repo,
                "commit": context.commit,
                "mode": context.mode,
                "signature_id": signature_id,
                "object_id": context.object_id,
                "variant": "primary",
                "follow_wrapped": context.follow_wrapped,
                "eval_str": context.eval_str,
                "effective_object_id": context.object_id,
                "sig_text": None,
                "sig_format": "inspect",
                "return_annotation": None,
                "has_varargs": None,
                "has_varkw": None,
                "status": _error_status(exc),
            }
        )
        return signature_rows, param_rows

    has_varargs = any(
        param.kind == inspect.Parameter.VAR_POSITIONAL for param in signature.parameters.values()
    )
    has_varkw = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()
    )
    return_annotation = None
    if signature.return_annotation is not inspect.Signature.empty:
        return_annotation = _value_ref(signature.return_annotation)
    signature_rows.append(
        {
            "repo": context.repo,
            "commit": context.commit,
            "mode": context.mode,
            "signature_id": signature_id,
            "object_id": context.object_id,
            "variant": "primary",
            "follow_wrapped": context.follow_wrapped,
            "eval_str": context.eval_str,
            "effective_object_id": context.object_id,
            "sig_text": str(signature),
            "sig_format": "inspect",
            "return_annotation": return_annotation,
            "has_varargs": has_varargs,
            "has_varkw": has_varkw,
            "status": _ok_status(),
        }
    )
    for idx, param in enumerate(signature.parameters.values()):
        default_present = param.default is not inspect.Parameter.empty
        annotation_present = param.annotation is not inspect.Parameter.empty
        param_rows.append(
            {
                "repo": context.repo,
                "commit": context.commit,
                "mode": context.mode,
                "signature_id": signature_id,
                "param_index": idx,
                "name": param.name,
                "kind": param.kind.name,
                "default_present": default_present,
                "default_value": _value_ref(param.default) if default_present else None,
                "annotation_present": annotation_present,
                "annotation_value": _value_ref(param.annotation) if annotation_present else None,
                "status": _ok_status(),
            }
        )
    return signature_rows, param_rows


def _unwrap_hops(
    value: InspectableCallable,
    *,
    context: _InspectContext,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    current = value
    stop_reason: str | None = None
    max_hops = 20
    for hop in range(max_hops):
        obj_id = _object_id(current, _object_kind(current))
        has_wrapped = _has_wrapped(current)
        rows.append(
            {
                "repo": context.repo,
                "commit": context.commit,
                "mode": context.mode,
                "root_object_id": context.object_id,
                "hop": hop,
                "object_id": obj_id,
                "has_wrapped": has_wrapped,
                "has_signature_override": _has_signature_override(current),
                "stop_reason": stop_reason,
                "status": _ok_status(),
            }
        )
        if not context.follow_wrapped:
            stop_reason = "follow_wrapped_disabled"
            break
        wrapped = getattr(current, "__wrapped__", None)
        if wrapped is None:
            stop_reason = "no_wrapped"
            break
        current = wrapped
    else:
        stop_reason = "max_hops"
    if rows:
        rows[-1]["stop_reason"] = stop_reason
    return rows


def _source_row(
    value: InspectableCallable,
    *,
    context: _InspectContext,
) -> dict[str, object] | None:
    try:
        source_lines, start_line = inspect.getsourcelines(value)
        file_name = inspect.getsourcefile(value)
    except (OSError, TypeError) as exc:
        return {
            "repo": context.repo,
            "commit": context.commit,
            "mode": context.mode,
            "object_id": context.object_id,
            "file_name": None,
            "start_line": None,
            "line_count": None,
            "source_sha256": None,
            "source_preview": None,
            "status": _error_status(exc),
        }
    source_text = "".join(source_lines)
    source_bytes = source_text.encode("utf-8", errors="replace")
    preview = source_text[:200]
    return {
        "repo": context.repo,
        "commit": context.commit,
        "mode": context.mode,
        "object_id": context.object_id,
        "file_name": file_name,
        "start_line": max(start_line - 1, 0),
        "line_count": len(source_lines),
        "source_sha256": hashlib.sha256(source_bytes).digest(),
        "source_preview": preview,
        "status": _ok_status(),
    }


def _inspect_context(state: _InspectState, *, object_id: str) -> _InspectContext:
    return _InspectContext(
        repo=state.repo,
        commit=state.commit,
        mode=state.mode,
        object_id=object_id,
        follow_wrapped=state.follow_wrapped,
        eval_str=state.eval_str,
    )


def _object_row(
    value: object,
    *,
    context: _InspectContext,
    kind: str,
) -> dict[str, object]:
    return {
        "repo": context.repo,
        "commit": context.commit,
        "mode": context.mode,
        "object_id": context.object_id,
        "object_addr": id(value),
        "kind": kind,
        "module_name": _object_module_name(value),
        "qualname": _object_qualname(value),
        "name": _object_name(value),
        "type_qualname": type(value).__qualname__,
        "is_builtin": inspect.isbuiltin(value),
        "is_callable": callable(value),
        "is_descriptor": inspect.isdatadescriptor(value),
        "has_wrapped": _has_wrapped(value),
        "has_signature_override": _has_signature_override(value),
        "has_annotations": bool(getattr(value, "__annotations__", None)),
        "status": _ok_status(),
    }


def _warn_object_limit(state: _InspectState) -> None:
    message = "Inspect object limit reached"
    if message not in state.warnings:
        state.warnings.append(message)


def _record_object(
    state: _InspectState,
    *,
    value: object,
    kind: str,
) -> str | None:
    if state.object_count >= state.object_limit:
        _warn_object_limit(state)
        return None
    object_id = _object_id(value, kind)
    if object_id in state.seen_objects:
        return object_id
    state.seen_objects.add(object_id)
    context = _inspect_context(state, object_id=object_id)
    state.buffers.objects.append(_object_row(value, context=context, kind=kind))
    state.object_count += 1
    if inspect.isclass(value):
        _inspect_class(state, cast("type[object]", value), object_id=object_id)
    return object_id


def _inspect_class(
    state: _InspectState,
    value: type[object],
    *,
    object_id: str,
) -> None:
    context = _inspect_context(state, object_id=object_id)
    if not _inspect_class_mro(state, value, context=context):
        return
    _inspect_class_attrs(state, value, context=context)


def _inspect_class_mro(
    state: _InspectState,
    value: type[object],
    *,
    context: _InspectContext,
) -> bool:
    try:
        mro_items = inspect.getmro(value)
    except (AttributeError, TypeError) as exc:
        state.buffers.class_mro.append(
            {
                "repo": context.repo,
                "commit": context.commit,
                "mode": context.mode,
                "class_object_id": context.object_id,
                "mro_index": 0,
                "base_object_id": None,
                "base_kind": None,
                "status": _error_status(exc),
            }
        )
        return False
    for index, base in enumerate(mro_items):
        base_kind = _object_kind(base)
        base_object_id = _record_object(state, value=base, kind=base_kind)
        if base_object_id is None:
            base_object_id = _object_id(base, base_kind)
        state.buffers.class_mro.append(
            {
                "repo": context.repo,
                "commit": context.commit,
                "mode": context.mode,
                "class_object_id": context.object_id,
                "mro_index": index,
                "base_object_id": base_object_id,
                "base_kind": base_kind,
                "status": _ok_status(),
            }
        )
    return True


def _inspect_class_attrs(
    state: _InspectState,
    value: type[object],
    *,
    context: _InspectContext,
) -> None:
    try:
        class_attrs = inspect.classify_class_attrs(value)
    except (AttributeError, TypeError) as exc:
        state.buffers.class_attrs.append(
            {
                "repo": context.repo,
                "commit": context.commit,
                "mode": context.mode,
                "class_object_id": context.object_id,
                "attr_name": "__inspect_error__",
                "attr_kind": None,
                "defining_object_id": None,
                "value_kind": None,
                "value_object_id": None,
                "value_ref": None,
                "desc_is_data": None,
                "desc_is_methoddesc": None,
                "desc_is_getset": None,
                "desc_is_member": None,
                "status": _error_status(exc),
            }
        )
        return
    for attr in class_attrs:
        row = _class_attr_row(state, context=context, attr=attr)
        if row is None:
            continue
        value_object_id = row["value_object_id"]
        state.buffers.class_attrs.append(row)
        if value_object_id is None:
            continue
        value_obj = getattr(attr, "object", None)
        if callable(value_obj):
            _inspect_callable(
                state,
                cast("InspectableCallable", value_obj),
                object_id=cast("str", value_object_id),
            )


def _class_attr_row(
    state: _InspectState,
    *,
    context: _InspectContext,
    attr: object,
) -> dict[str, object] | None:
    attr_name = getattr(attr, "name", None)
    if not isinstance(attr_name, str):
        return None
    attr_kind = getattr(attr, "kind", None)
    attr_kind_value = attr_kind if isinstance(attr_kind, str) else None
    defining_class = getattr(attr, "defining_class", None)
    defining_object_id: str | None = None
    if inspect.isclass(defining_class):
        defining_kind = _object_kind(defining_class)
        defining_object_id = _record_object(state, value=defining_class, kind=defining_kind)
        if defining_object_id is None:
            defining_object_id = _object_id(defining_class, defining_kind)
    value_obj = getattr(attr, "object", None)
    value_kind = _object_kind(value_obj)
    value_object_id: str | None = None
    if inspect.isroutine(value_obj) or inspect.isclass(value_obj) or inspect.ismodule(value_obj):
        value_object_id = _record_object(state, value=value_obj, kind=value_kind)
    return {
        "repo": context.repo,
        "commit": context.commit,
        "mode": context.mode,
        "class_object_id": context.object_id,
        "attr_name": attr_name,
        "attr_kind": attr_kind_value,
        "defining_object_id": defining_object_id,
        "value_kind": value_kind,
        "value_object_id": value_object_id,
        "value_ref": _value_ref(value_obj),
        "desc_is_data": inspect.isdatadescriptor(value_obj),
        "desc_is_methoddesc": inspect.ismethoddescriptor(value_obj),
        "desc_is_getset": inspect.isgetsetdescriptor(value_obj),
        "desc_is_member": inspect.ismemberdescriptor(value_obj),
        "status": _ok_status(),
    }


def _runtime_state_rows(
    state: _InspectState,
    *,
    value: object,
    object_id: str,
) -> list[dict[str, object]]:
    context = _inspect_context(state, object_id=object_id)
    object_kind = _object_kind(value)
    frame = _runtime_frame(value)
    frame_object_id: str | None = None
    if frame is not None:
        frame_kind = _object_kind(frame)
        frame_object_id = _record_object(state, value=frame, kind=frame_kind)
        if frame_object_id is None:
            frame_object_id = _object_id(frame, frame_kind)
    frame_info = _RuntimeFrameInfo(
        frame=frame,
        frame_object_id=frame_object_id,
        frame_line=_frame_line(value, frame),
        frame_offset=_frame_offset(value, frame),
    )
    state_info = _runtime_state_info(value, object_kind)
    if state_info is None:
        return []
    return [
        _runtime_state_row(
            context=context,
            frame_info=frame_info,
            state_info=state_info,
        )
    ]


def _runtime_state_info(
    value: object,
    object_kind: str,
) -> _RuntimeStateInfo | None:
    if inspect.isframe(value):
        return _RuntimeStateInfo(
            object_kind=object_kind,
            state_kind="frame",
            state=None,
            status=_ok_status(),
        )
    if inspect.istraceback(value):
        return _RuntimeStateInfo(
            object_kind=object_kind,
            state_kind="traceback",
            state=None,
            status=_ok_status(),
        )
    if inspect.isgenerator(value):
        return _runtime_state_from_generator(
            object_kind,
            cast("GeneratorType[object, object, object]", value),
        )
    if inspect.iscoroutine(value):
        return _runtime_state_from_coroutine(
            object_kind,
            cast("CoroutineType[object, object, object]", value),
        )
    if inspect.isasyncgen(value):
        return _runtime_state_from_asyncgen(
            object_kind,
            cast("AsyncGeneratorType[object, object]", value),
        )
    return None


def _runtime_state_from_generator(
    object_kind: str,
    value: GeneratorType[object, object, object],
) -> _RuntimeStateInfo:
    try:
        state_value = inspect.getgeneratorstate(value)
        status = _ok_status()
    except (ValueError, TypeError) as exc:
        state_value = None
        status = _error_status(exc)
    return _RuntimeStateInfo(
        object_kind=object_kind,
        state_kind="generator",
        state=state_value,
        status=status,
    )


def _runtime_state_from_coroutine(
    object_kind: str,
    value: CoroutineType[object, object, object],
) -> _RuntimeStateInfo:
    try:
        state_value = inspect.getcoroutinestate(value)
        status = _ok_status()
    except (ValueError, TypeError) as exc:
        state_value = None
        status = _error_status(exc)
    return _RuntimeStateInfo(
        object_kind=object_kind,
        state_kind="coroutine",
        state=state_value,
        status=status,
    )


def _runtime_state_from_asyncgen(
    object_kind: str,
    value: AsyncGeneratorType[object, object],
) -> _RuntimeStateInfo:
    try:
        state_value = inspect.getasyncgenstate(value)
        status = _ok_status()
    except (ValueError, TypeError) as exc:
        state_value = None
        status = _error_status(exc)
    return _RuntimeStateInfo(
        object_kind=object_kind,
        state_kind="asyncgen",
        state=state_value,
        status=status,
    )


def _inspect_runtime_state(
    state: _InspectState,
    *,
    value: object,
    object_id: str,
) -> None:
    rows = _runtime_state_rows(state, value=value, object_id=object_id)
    if rows:
        state.buffers.runtime_state.extend(rows)

def _inspect_callable(
    state: _InspectState,
    value: InspectableCallable,
    *,
    object_id: str,
) -> None:
    context = _inspect_context(state, object_id=object_id)
    state.buffers.unwrap.extend(_unwrap_hops(value, context=context))
    sig_rows, param_rows = _signature_rows(value, context=context)
    state.buffers.signatures.extend(sig_rows)
    state.buffers.signature_params.extend(param_rows)
    try:
        annotations = inspect.get_annotations(value, eval_str=state.eval_str)
    except (TypeError, ValueError, NameError) as exc:
        state.warnings.append(f"Inspect annotations failed: {exc}")
        annotations = {}
    state.buffers.annotations.extend(_annotation_payload(annotations, context=context))
    source_row = _source_row(value, context=context)
    if source_row is not None:
        state.buffers.sources.append(source_row)


def _inspect_member(
    state: _InspectState,
    *,
    owner_object_id: str,
    owner_kind: str,
    attr_name: str,
    value: object,
) -> None:
    if state.object_count >= state.object_limit:
        _warn_object_limit(state)
        return
    value_kind = _object_kind(value)
    value_object_id: str | None = None
    is_runtime = _is_runtime_object(value)
    if inspect.isroutine(value) or inspect.isclass(value) or inspect.ismodule(value) or is_runtime:
        value_object_id = _record_object(state, value=value, kind=value_kind)
    state.buffers.members.append(
        {
            "repo": state.repo,
            "commit": state.commit,
            "mode": state.mode,
            "owner_object_id": owner_object_id,
            "owner_kind": owner_kind,
            "attr_name": attr_name,
            "value_kind": value_kind,
            "value_object_id": value_object_id,
            "value_ref": _value_ref(value),
            "desc_kind": type(value).__name__,
            "desc_is_data": inspect.isdatadescriptor(value),
            "desc_is_methoddesc": inspect.ismethoddescriptor(value),
            "desc_is_getset": inspect.isgetsetdescriptor(value),
            "desc_is_member": inspect.ismemberdescriptor(value),
            "status": _ok_status(),
        }
    )
    if value_object_id is not None and is_runtime:
        _inspect_runtime_state(state, value=value, object_id=value_object_id)
    if value_object_id is None or not callable(value):
        return
    _inspect_callable(state, cast("InspectableCallable", value), object_id=value_object_id)


def _inspect_module(state: _InspectState, module: ModuleRecord) -> None:
    if state.object_count >= state.object_limit:
        _warn_object_limit(state)
        return
    repo_root = _repo_root_for_module(module)
    with _sys_path_prefix(repo_root):
        try:
            loaded_module = importlib.import_module(module.module_name)
        except ImportError as exc:
            message = f"Inspect import failed for {module.module_name}: {exc}"
            state.warnings.append(message)
            LOG.warning("%s", message)
            return
    module_kind = _object_kind(loaded_module)
    module_object_id = _record_object(state, value=loaded_module, kind=module_kind)
    if module_object_id is None:
        return
    try:
        members = inspect.getmembers_static(loaded_module)
    except (AttributeError, TypeError) as exc:
        state.warnings.append(f"Inspect members failed for {module.module_name}: {exc}")
        return
    for attr_name, value in members:
        if state.object_count >= state.object_limit:
            _warn_object_limit(state)
            break
        _inspect_member(
            state,
            owner_object_id=module_object_id,
            owner_kind=module_kind,
            attr_name=attr_name,
            value=value,
        )


def _process_module(state: _InspectState, module: ModuleRecord) -> bool:
    _inspect_module(state, module)
    return state.object_count >= state.object_limit


@contextlib.contextmanager
def _sys_path_prefix(root: Path) -> Iterator[None]:
    root_str = str(root)
    inserted = False
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
        inserted = True
    try:
        yield
    finally:
        if inserted:
            sys.path.remove(root_str)


def _repo_root_for_module(module: ModuleRecord) -> Path:
    root = module.file_path
    for _ in Path(module.rel_path).parts:
        root = root.parent
    return root


def _module_size_bytes(module: ModuleRecord) -> int | None:
    try:
        return module.file_path.stat().st_size
    except OSError:
        return None


def _filter_inspect_modules(
    modules: Sequence[ModuleRecord],
    *,
    allowlist: set[str],
    options: InspectExtractOptions,
) -> tuple[list[ModuleRecord], list[str]]:
    warnings: list[str] = []
    module_names = {module.module_name for module in modules}
    missing = sorted(allowlist - module_names)
    if missing:
        missing_preview = ", ".join(missing[:_ALLOWLIST_PREVIEW_LIMIT])
        suffix = "..." if len(missing) > _ALLOWLIST_PREVIEW_LIMIT else ""
        warnings.append(f"Inspect allowlist modules missing: {missing_preview}{suffix}")
    allowed = [module for module in modules if module.module_name in allowlist]
    max_modules = options.max_modules
    if max_modules is not None:
        if max_modules <= 0:
            warnings.append("Inspect max_modules is non-positive; skipping inspection")
            return [], warnings
        if len(allowed) > max_modules:
            warnings.append(f"Inspect module limit hit: {len(allowed)} -> {max_modules}")
            allowed = allowed[:max_modules]
    max_module_bytes = options.max_module_bytes
    if max_module_bytes is None:
        return allowed, warnings
    filtered: list[ModuleRecord] = []
    for module in allowed:
        size_bytes = _module_size_bytes(module)
        if size_bytes is not None and size_bytes > max_module_bytes:
            warnings.append(f"Skipping inspect for {module.module_name} (size {size_bytes} bytes)")
            continue
        filtered.append(module)
    return filtered, warnings


def _build_payload(state: _InspectState) -> _InspectWorkerPayload:
    return _InspectWorkerPayload(
        warnings=list(state.warnings),
        object_rows=state.buffers.objects.data,
        member_rows=state.buffers.members.data,
        class_mro_rows=state.buffers.class_mro.data,
        class_attr_rows=state.buffers.class_attrs.data,
        unwrap_rows=state.buffers.unwrap.data,
        signature_rows=state.buffers.signatures.data,
        signature_param_rows=state.buffers.signature_params.data,
        annotation_rows=state.buffers.annotations.data,
        source_rows=state.buffers.sources.data,
        runtime_state_rows=state.buffers.runtime_state.data,
        object_row_count=state.buffers.objects.row_count,
        member_row_count=state.buffers.members.row_count,
        class_mro_row_count=state.buffers.class_mro.row_count,
        class_attr_row_count=state.buffers.class_attrs.row_count,
        unwrap_row_count=state.buffers.unwrap.row_count,
        signature_row_count=state.buffers.signatures.row_count,
        signature_param_row_count=state.buffers.signature_params.row_count,
        annotation_row_count=state.buffers.annotations.row_count,
        source_row_count=state.buffers.sources.row_count,
        runtime_state_row_count=state.buffers.runtime_state.row_count,
    )


def _run_inspect_modules(
    modules: Sequence[ModuleRecord],
    *,
    repo: str,
    commit: str,
    options: InspectExtractOptions,
    warnings: list[str],
) -> _InspectWorkerPayload:
    buffers = _build_inspect_buffers()
    state = _InspectState(
        buffers=buffers,
        seen_objects=set(),
        warnings=warnings,
        repo=repo,
        commit=commit,
        mode="allowlist",
        object_limit=options.max_objects,
        eval_str=options.eval_str,
        follow_wrapped=options.follow_wrapped,
    )
    for module in modules:
        if state.object_count >= state.object_limit:
            _warn_object_limit(state)
            break
        _inspect_module(state, module)
    return _build_payload(state)


def _inspect_worker_entry(
    result_queue: multiprocessing.Queue[object],
    job: _InspectWorkerJob,
) -> None:
    payload = _run_inspect_modules(
        job.modules,
        repo=job.repo,
        commit=job.commit,
        options=job.options,
        warnings=list(job.seed_warnings),
    )
    result_queue.put({"ok": True, "payload": payload})


def _read_worker_payload(
    result_queue: multiprocessing.Queue[object],
) -> tuple[_InspectWorkerPayload | None, str | None]:
    try:
        message = result_queue.get_nowait()
    except queue.Empty:
        return None, "Inspect subprocess returned no results"
    if not isinstance(message, dict):
        return None, "Inspect subprocess returned invalid payload"
    ok = message.get("ok")
    if ok is True:
        payload = message.get("payload")
        if isinstance(payload, _InspectWorkerPayload):
            return payload, None
        return None, "Inspect subprocess payload type mismatch"
    error = message.get("error")
    if isinstance(error, str):
        return None, error
    return None, "Inspect subprocess failed without error message"


def _combine_warnings(base: list[str], extra: list[str]) -> tuple[str, ...]:
    combined = list(base)
    combined.extend(extra)
    return tuple(combined)


def _payload_to_result(
    payload: _InspectWorkerPayload,
    *,
    warnings: list[str] | None = None,
) -> InspectExtractResult:
    resolved_warnings = payload.warnings if warnings is None else warnings
    object_rows_reader, _ = record_batch_reader_for_columnar_rows(
        PY_INSPECT_OBJECTS_TABLE_KEY,
        payload.object_rows,
        extras_policy="retain",
    )
    member_rows_reader, _ = record_batch_reader_for_columnar_rows(
        PY_INSPECT_MEMBERS_TABLE_KEY,
        payload.member_rows,
        extras_policy="retain",
    )
    class_mro_rows_reader, _ = record_batch_reader_for_columnar_rows(
        PY_INSPECT_CLASS_MRO_TABLE_KEY,
        payload.class_mro_rows,
        extras_policy="retain",
    )
    class_attr_rows_reader, _ = record_batch_reader_for_columnar_rows(
        PY_INSPECT_CLASS_ATTRS_TABLE_KEY,
        payload.class_attr_rows,
        extras_policy="retain",
    )
    unwrap_rows_reader, _ = record_batch_reader_for_columnar_rows(
        PY_INSPECT_UNWRAP_TABLE_KEY,
        payload.unwrap_rows,
        extras_policy="retain",
    )
    signature_rows_reader, _ = record_batch_reader_for_columnar_rows(
        PY_INSPECT_SIGNATURES_TABLE_KEY,
        payload.signature_rows,
        extras_policy="retain",
    )
    signature_param_rows_reader, _ = record_batch_reader_for_columnar_rows(
        PY_INSPECT_SIGNATURE_PARAMS_TABLE_KEY,
        payload.signature_param_rows,
        extras_policy="retain",
    )
    annotation_rows_reader, _ = record_batch_reader_for_columnar_rows(
        PY_INSPECT_ANNOTATIONS_TABLE_KEY,
        payload.annotation_rows,
        extras_policy="retain",
    )
    source_rows_reader, _ = record_batch_reader_for_columnar_rows(
        PY_INSPECT_SOURCE_TABLE_KEY,
        payload.source_rows,
        extras_policy="retain",
    )
    runtime_state_rows_reader, _ = record_batch_reader_for_columnar_rows(
        PY_INSPECT_RUNTIME_STATE_TABLE_KEY,
        payload.runtime_state_rows,
        extras_policy="retain",
    )
    return InspectExtractResult(
        result=ExecutionResult.ok(warnings=tuple(resolved_warnings)),
        object_rows=payload.object_rows,
        member_rows=payload.member_rows,
        class_mro_rows=payload.class_mro_rows,
        class_attr_rows=payload.class_attr_rows,
        unwrap_rows=payload.unwrap_rows,
        signature_rows=payload.signature_rows,
        signature_param_rows=payload.signature_param_rows,
        annotation_rows=payload.annotation_rows,
        source_rows=payload.source_rows,
        runtime_state_rows=payload.runtime_state_rows,
        object_rows_reader=object_rows_reader,
        member_rows_reader=member_rows_reader,
        class_mro_rows_reader=class_mro_rows_reader,
        class_attr_rows_reader=class_attr_rows_reader,
        unwrap_rows_reader=unwrap_rows_reader,
        signature_rows_reader=signature_rows_reader,
        signature_param_rows_reader=signature_param_rows_reader,
        annotation_rows_reader=annotation_rows_reader,
        source_rows_reader=source_rows_reader,
        runtime_state_rows_reader=runtime_state_rows_reader,
        object_row_count=payload.object_row_count,
        member_row_count=payload.member_row_count,
        class_mro_row_count=payload.class_mro_row_count,
        class_attr_row_count=payload.class_attr_row_count,
        unwrap_row_count=payload.unwrap_row_count,
        signature_row_count=payload.signature_row_count,
        signature_param_row_count=payload.signature_param_row_count,
        annotation_row_count=payload.annotation_row_count,
        source_row_count=payload.source_row_count,
        runtime_state_row_count=payload.runtime_state_row_count,
    )


def _run_inspect_subprocess(
    *,
    modules: Sequence[ModuleRecord],
    repo: str,
    commit: str,
    options: InspectExtractOptions,
    seed_warnings: list[str],
) -> InspectExtractResult:
    ctx = multiprocessing.get_context("spawn")
    result_queue: multiprocessing.Queue[object] = ctx.Queue()
    job = _InspectWorkerJob(
        modules=modules,
        repo=repo,
        commit=commit,
        options=options,
        seed_warnings=seed_warnings,
    )
    process = ctx.Process(
        target=_inspect_worker_entry,
        args=(result_queue, job),
    )
    process.start()
    timeout = options.timeout_seconds if options.timeout_seconds > 0 else None
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join()
        timeout_warning = f"Inspect subprocess timed out after {options.timeout_seconds} seconds"
        warnings = _combine_warnings(seed_warnings, [timeout_warning])
        return InspectExtractResult(
            result=ExecutionResult.failed(
                "Inspect extraction timed out",
                warnings=warnings,
            )
        )
    payload, error = _read_worker_payload(result_queue)
    if payload is None:
        error_message = error or "Inspect subprocess failed"
        warnings = _combine_warnings(seed_warnings, [error_message])
        return InspectExtractResult(
            result=ExecutionResult.failed(
                "Inspect extraction failed",
                warnings=warnings,
            )
        )
    return _payload_to_result(payload)


class InspectExtractStep(BaseExtractStep):
    """Inspect extraction step with port injection."""

    def __init__(
        self,
        discovery: ModuleDiscoveryPort,
        *,
        options: InspectExtractOptions | None = None,
    ) -> None:
        super().__init__(discovery=discovery)
        self._options = options or InspectExtractOptions()

    def execute(
        self,
        modules: Sequence[ModuleRecord],
        *,
        repo: str,
        commit: str,
    ) -> InspectExtractResult:
        """Execute inspect extraction for the provided modules.

        Returns
        -------
        InspectExtractResult
            Result bundle with row payloads and execution status.
        """
        if not self._options.enable:
            return InspectExtractResult(
                result=ExecutionResult.skip("Inspect extraction disabled by options")
            )
        allowlist = set(self._options.module_allowlist)
        if not allowlist:
            return InspectExtractResult(
                result=ExecutionResult.skip("Inspect extraction disabled (no allowlist)")
            )
        filtered_modules, warnings = _filter_inspect_modules(
            modules,
            allowlist=allowlist,
            options=self._options,
        )
        if not filtered_modules:
            return InspectExtractResult(result=ExecutionResult.ok(warnings=tuple(warnings)))
        if self._options.use_subprocess:
            return _run_inspect_subprocess(
                modules=filtered_modules,
                repo=repo,
                commit=commit,
                options=self._options,
                seed_warnings=warnings,
            )
        try:
            payload = _run_inspect_modules(
                filtered_modules,
                repo=repo,
                commit=commit,
                options=self._options,
                warnings=warnings,
            )
        except (KeyError, RuntimeError) as exc:
            return InspectExtractResult(
                result=ExecutionResult.failed(str(exc), warnings=tuple(warnings))
            )
        return _payload_to_result(payload)


__all__ = ["InspectExtractResult", "InspectExtractStep"]
