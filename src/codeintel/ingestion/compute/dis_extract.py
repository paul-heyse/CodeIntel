"""Bytecode extraction step with port injection."""

from __future__ import annotations

import dis
import hashlib
import importlib.util
import inspect
import io
import logging
import sys
import tokenize
from dataclasses import dataclass, field
from types import CodeType
from typing import TYPE_CHECKING

from codeintel.build.hamilton.execution_result import ExecutionResult
from codeintel.build.hamilton.native.options.ingestion import BytecodeExtractOptions
from codeintel.core.columnar.rows import (
    ColumnarRowBuffer,
    ColumnarRows,
    columnar_buffer_for_table_key,
)
from codeintel.ingestion.compute.base import BaseExtractStep
from codeintel.ingestion.infrastructure.cst_utils import LineIndexedSource

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from codeintel.ingestion.ports.discovery import ModuleDiscoveryPort, ModuleRecord

LOG = logging.getLogger(__name__)

PY_BC_CODE_UNITS_TABLE_KEY = "core.py_bc_code_units"
PY_BC_INSTRUCTIONS_TABLE_KEY = "core.py_bc_instructions"
PY_BC_EXCEPTION_TABLE_KEY = "core.py_bc_exception_table"
PY_BC_BLOCKS_TABLE_KEY = "core.py_bc_blocks"
PY_BC_CFG_EDGES_TABLE_KEY = "core.py_bc_cfg_edges"
PY_BC_DEFUSE_EVENTS_TABLE_KEY = "core.py_bc_defuse_events"

_UNCONDITIONAL_JUMPS = {
    "JUMP",
    "JUMP_ABSOLUTE",
    "JUMP_FORWARD",
    "JUMP_BACKWARD",
    "JUMP_BACKWARD_NO_INTERRUPT",
    "JUMP_NO_INTERRUPT",
}
_TERMINATOR_PREFIXES = ("RETURN", "RAISE", "RERAISE")
_CACHE_INFO_ENTRY_LEN = 3


@dataclass(frozen=True)
class DisExtractResult:
    """Result bundle for bytecode extraction."""

    result: ExecutionResult
    code_unit_rows: ColumnarRows = field(default_factory=dict)
    instruction_rows: ColumnarRows = field(default_factory=dict)
    exception_rows: ColumnarRows = field(default_factory=dict)
    block_rows: ColumnarRows = field(default_factory=dict)
    cfg_edge_rows: ColumnarRows = field(default_factory=dict)
    defuse_event_rows: ColumnarRows = field(default_factory=dict)
    code_unit_row_count: int = 0
    instruction_row_count: int = 0
    exception_row_count: int = 0
    block_row_count: int = 0
    cfg_edge_row_count: int = 0
    defuse_event_row_count: int = 0


@dataclass(frozen=True, slots=True)
class _InstructionInfo:
    instr: dis.Instruction
    instr_id: str
    instr_index: int
    span_start_byte: int | None
    span_end_byte: int | None


@dataclass(frozen=True, slots=True)
class _ArgvalInfo:
    kind: str | None
    text: str | None
    int_value: int | None
    repr_text: str | None


@dataclass(frozen=True, slots=True)
class _OffsetInfo:
    ext_arg_len: int | None
    op_len: int | None
    cache_len: int | None


@dataclass(frozen=True, slots=True)
class _JumpInfo:
    target_offset: int | None
    target_label: str | None
    label: str | None


@dataclass(frozen=True, slots=True)
class _LineInfo:
    line_number: int | None
    starts_line: bool | None


@dataclass(frozen=True, slots=True)
class _ByteInfo:
    cache_bytes: bytes | None
    op_bytes: bytes | None


@dataclass(frozen=True, slots=True)
class _CodeUnitInfo:
    code: CodeType
    code_unit_id: str
    parent_code_unit_id: str | None
    qualpath: str
    ordinal: int


@dataclass(frozen=True, slots=True)
class _BytecodeContext:
    repo: str
    commit: str
    rel_path: str
    module_name: str
    source_index: LineIndexedSource
    options: BytecodeExtractOptions


@dataclass(frozen=True, slots=True)
class _CodeUnitContext:
    base: _BytecodeContext
    code: CodeType
    code_unit_id: str
    qualpath: str


@dataclass(frozen=True, slots=True)
class _DisBuffers:
    code_units: ColumnarRowBuffer
    instructions: ColumnarRowBuffer
    exceptions: ColumnarRowBuffer
    blocks: ColumnarRowBuffer
    cfg_edges: ColumnarRowBuffer
    defuse_events: ColumnarRowBuffer


@dataclass(frozen=True, slots=True)
class _BlockInfo:
    block_id: str
    start_offset: int
    end_offset: int
    first_instr_index: int
    last_instr_index: int
    last_instr_id: str
    last_instr: dis.Instruction
    anchor_span_start_byte: int | None
    anchor_span_end_byte: int | None


@dataclass(slots=True)
class _BlockBuildState:
    rows: list[dict[str, object]]
    blocks: list[_BlockInfo]
    offset_to_block_id: dict[int, str]
    label_map: dict[int, str]
    context: _CodeUnitContext


@dataclass(slots=True)
class _EdgeBuildState:
    edges: list[dict[str, object]]
    edge_keys: set[str]
    context: _CodeUnitContext


@dataclass(frozen=True, slots=True)
class _EdgeSpec:
    src_block_id: str
    dst_block_id: str
    kind: str
    cond_instr_id: str | None
    exc_entry_index: int | None


def _stable_id(*parts: object) -> str:
    payload = "|".join("" if part is None else str(part) for part in parts)
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=16)
    return digest.hexdigest()


def _decode_source_bytes(source_bytes: bytes) -> tuple[str, str]:
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(source_bytes).readline)
    except SyntaxError:
        encoding = "utf-8"
    try:
        return source_bytes.decode(encoding), encoding
    except UnicodeDecodeError:
        return source_bytes.decode(encoding, errors="replace"), encoding


def _build_source_index(source_bytes: bytes) -> tuple[str, LineIndexedSource]:
    source_text, encoding = _decode_source_bytes(source_bytes)
    source_index = LineIndexedSource(source_text, source_bytes, encoding=encoding)
    return source_text, source_index


def _build_dis_buffers() -> _DisBuffers:
    return _DisBuffers(
        code_units=columnar_buffer_for_table_key(PY_BC_CODE_UNITS_TABLE_KEY),
        instructions=columnar_buffer_for_table_key(PY_BC_INSTRUCTIONS_TABLE_KEY),
        exceptions=columnar_buffer_for_table_key(PY_BC_EXCEPTION_TABLE_KEY),
        blocks=columnar_buffer_for_table_key(PY_BC_BLOCKS_TABLE_KEY),
        cfg_edges=columnar_buffer_for_table_key(PY_BC_CFG_EDGES_TABLE_KEY),
        defuse_events=columnar_buffer_for_table_key(PY_BC_DEFUSE_EVENTS_TABLE_KEY),
    )


def _coerce_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    return None


def _normalize_line(value: int | None) -> int | None:
    if not isinstance(value, int):
        return None
    return max(value - 1, 0)


def _line_end_col_utf8(source_index: LineIndexedSource, line: int | None) -> int | None:
    if line is None or line < 0:
        return None
    snippet = source_index.line_snippet(line)
    if snippet is None:
        return None
    return len(snippet.encode("utf-8", errors="replace"))


def _positions_payload(
    positions: object,
    source_index: LineIndexedSource,
) -> tuple[dict[str, object] | None, int | None, int | None]:
    if positions is None:
        return None, None, None
    lineno = _normalize_line(getattr(positions, "lineno", None))
    end_lineno = _normalize_line(getattr(positions, "end_lineno", None))
    col = _coerce_int(getattr(positions, "col_offset", None))
    end_col = _coerce_int(getattr(positions, "end_col_offset", None))
    if lineno is None or col is None:
        return None, None, None
    if end_lineno is None:
        end_lineno = lineno
    if end_col is None:
        end_col = col
    start_byte = source_index.byte_offset_from_utf8(lineno, col)
    end_byte = source_index.byte_offset_from_utf8(end_lineno, end_col)
    payload: dict[str, object] = {
        "lineno": lineno,
        "end_lineno": end_lineno,
        "col": col,
        "end_col": end_col,
    }
    return payload, start_byte, end_byte


def _cache_info_payload(cache_info: Iterable[object] | None) -> list[dict[str, object]] | None:
    if not cache_info:
        return None
    payload: list[dict[str, object]] = []
    for entry in cache_info:
        if not isinstance(entry, tuple) or len(entry) < _CACHE_INFO_ENTRY_LEN:
            continue
        name, size, data = entry[:3]
        if not isinstance(name, str):
            continue
        payload.append(
            {
                "name": name,
                "size": size if isinstance(size, int) else None,
                "data": bytes(data) if isinstance(data, (bytes, bytearray)) else None,
            }
        )
    return payload or None


def _truncate_repr(value: object, limit: int = 240) -> str:
    text = repr(value)
    if len(text) <= limit:
        return text
    return text[:limit]


def _argval_fields(argval: object) -> _ArgvalInfo:
    if argval is None:
        return _ArgvalInfo(None, None, None, None)
    if isinstance(argval, str):
        return _ArgvalInfo("str", argval, None, None)
    if isinstance(argval, int):
        return _ArgvalInfo("int", None, argval, None)
    return _ArgvalInfo(type(argval).__name__, None, None, _truncate_repr(argval))


def _instruction_index(instr: dis.Instruction, fallback: int) -> int:
    return instr.index if isinstance(instr.index, int) else fallback


def _instruction_id(context: _CodeUnitContext, instr: dis.Instruction, instr_index: int) -> str:
    return _stable_id(
        "py_bc_instr",
        context.code_unit_id,
        instr_index,
        instr.offset,
        instr.opname,
        instr.arg,
    )


def _instruction_physical_id(
    code_unit_id: str,
    instr: dis.Instruction,
) -> str | None:
    if isinstance(instr.start_offset, int):
        return f"{code_unit_id}:{instr.start_offset}"
    return None


def _offset_info(instr: dis.Instruction) -> _OffsetInfo:
    ext_arg_len = (
        instr.offset - instr.start_offset
        if isinstance(instr.offset, int) and isinstance(instr.start_offset, int)
        else None
    )
    op_len = (
        instr.cache_offset - instr.start_offset
        if isinstance(instr.cache_offset, int) and isinstance(instr.start_offset, int)
        else None
    )
    cache_len = (
        instr.end_offset - instr.cache_offset
        if isinstance(instr.end_offset, int) and isinstance(instr.cache_offset, int)
        else None
    )
    return _OffsetInfo(ext_arg_len=ext_arg_len, op_len=op_len, cache_len=cache_len)


def _jump_info(instr: dis.Instruction, label_map: dict[int, str]) -> _JumpInfo:
    target_offset = instr.jump_target if isinstance(instr.jump_target, int) else None
    target_label = _label_for_offset(target_offset)
    label = label_map.get(instr.offset)
    return _JumpInfo(
        target_offset=target_offset,
        target_label=target_label,
        label=label,
    )


def _line_info(instr: dis.Instruction) -> _LineInfo:
    line_number = _normalize_line(instr.line_number)
    starts_line = instr.starts_line if isinstance(instr.starts_line, bool) else None
    return _LineInfo(line_number=line_number, starts_line=starts_line)


def _byte_info(instr: dis.Instruction, code_bytes: bytes) -> _ByteInfo:
    cache_bytes = _instruction_bytes(code_bytes, instr.cache_offset, instr.end_offset)
    op_bytes = _instruction_bytes(code_bytes, instr.start_offset, instr.cache_offset)
    return _ByteInfo(cache_bytes=cache_bytes, op_bytes=op_bytes)


def _instruction_bytes(
    code_bytes: bytes,
    start: int | None,
    end: int | None,
) -> bytes | None:
    if not isinstance(start, int) or not isinstance(end, int):
        return None
    if start < 0 or end < start:
        return None
    return code_bytes[start:end]


def _label_for_offset(offset: int | None) -> str | None:
    if not isinstance(offset, int):
        return None
    return f"L{offset}"


def _is_unconditional_jump(opname: str | None) -> bool:
    return opname in _UNCONDITIONAL_JUMPS


def _is_terminator(opname: str | None) -> bool:
    if opname is None:
        return False
    return opname.startswith(_TERMINATOR_PREFIXES) or opname == "RETURN_VALUE"


def _code_unit_id(
    context: _BytecodeContext,
    *,
    qualpath: str,
    firstlineno: int | None,
    ordinal: int,
) -> str:
    return _stable_id(
        "py_bc_code",
        context.repo,
        context.commit,
        context.rel_path,
        qualpath,
        firstlineno,
        ordinal,
    )


def _code_unit_qualpath(
    context: _BytecodeContext,
    *,
    code: CodeType,
    parent_qualpath: str | None,
) -> str:
    qualname = getattr(code, "co_qualname", None)
    if isinstance(qualname, str) and qualname:
        return f"{context.module_name}::{qualname}"
    if parent_qualpath:
        return f"{parent_qualpath}::{code.co_name}"
    return f"{context.module_name}::{code.co_name}"


def _infer_kind_from_line(
    code: CodeType,
    source_index: LineIndexedSource,
) -> str | None:
    first_line = _normalize_line(code.co_firstlineno)
    line = source_index.line_snippet(first_line or 0)
    if line is None:
        return None
    stripped = line.lstrip()
    if stripped.startswith("class "):
        return "CLASS"
    if stripped.startswith("async def "):
        return "ASYNC_FUNCTION"
    if stripped.startswith("def "):
        return "FUNCTION"
    return None


def _code_unit_kind(code: CodeType, source_index: LineIndexedSource) -> str:
    if code.co_name == "<module>":
        kind = "MODULE"
    elif code.co_name == "<lambda>":
        kind = "LAMBDA"
    elif code.co_name in {"<listcomp>", "<setcomp>", "<dictcomp>", "<genexpr>"}:
        kind = "COMPREHENSION"
    elif code.co_flags & inspect.CO_ASYNC_GENERATOR:
        kind = "ASYNC_GENERATOR"
    elif code.co_flags & inspect.CO_COROUTINE:
        kind = "ASYNC_FUNCTION"
    elif code.co_flags & inspect.CO_GENERATOR:
        kind = "GENERATOR"
    else:
        line_kind = _infer_kind_from_line(code, source_index)
        kind = line_kind if line_kind is not None else "FUNCTION"
    return kind


def _code_unit_span_from_positions(
    instructions: Sequence[_InstructionInfo],
    source_index: LineIndexedSource,
    code: CodeType,
) -> tuple[int | None, int | None]:
    starts = [info.span_start_byte for info in instructions if info.span_start_byte is not None]
    ends = [info.span_end_byte for info in instructions if info.span_end_byte is not None]
    if starts and ends:
        return min(starts), max(ends)
    first_line = _normalize_line(code.co_firstlineno)
    if first_line is None:
        return None, None
    last_line = _normalize_line(
        max(
            (line for _, line in dis.findlinestarts(code)),
            default=code.co_firstlineno,
        )
    )
    if last_line is None:
        last_line = first_line
    start_byte = source_index.byte_offset_from_utf8(first_line, 0)
    end_col = _line_end_col_utf8(source_index, last_line)
    end_byte = source_index.byte_offset_from_utf8(last_line, end_col or 0)
    return start_byte, end_byte


def _build_instruction_rows(
    context: _CodeUnitContext,
) -> tuple[list[dict[str, object]], list[_InstructionInfo], dict[int, str]]:
    instructions = list(
        dis.get_instructions(
            context.code,
            show_caches=context.base.options.show_caches,
            adaptive=context.base.options.adaptive,
        )
    )
    label_map = _label_map(instructions)
    code_bytes = context.code.co_code
    rows: list[dict[str, object]] = []
    infos: list[_InstructionInfo] = []
    for idx, instr in enumerate(instructions):
        instr_index = _instruction_index(instr, idx)
        positions_payload, span_start, span_end = _positions_payload(
            instr.positions,
            context.base.source_index,
        )
        instr_id = _instruction_id(context, instr, instr_index)
        row, info = _instruction_row(
            context=context,
            instr=instr,
            instr_index=instr_index,
            instr_id=instr_id,
            positions_payload=positions_payload,
            span_start=span_start,
            span_end=span_end,
            label_map=label_map,
            code_bytes=code_bytes,
        )
        rows.append(row)
        infos.append(info)
    return rows, infos, label_map


def _label_map(instructions: Sequence[dis.Instruction]) -> dict[int, str]:
    return {instr.offset: _label_for_offset(instr.offset) or "" for instr in instructions}


def _instruction_row(
    *,
    context: _CodeUnitContext,
    instr: dis.Instruction,
    instr_index: int,
    instr_id: str,
    positions_payload: dict[str, object] | None,
    span_start: int | None,
    span_end: int | None,
    label_map: dict[int, str],
    code_bytes: bytes,
) -> tuple[dict[str, object], _InstructionInfo]:
    argval_info = _argval_fields(instr.argval)
    offsets = _offset_info(instr)
    jump_info = _jump_info(instr, label_map)
    line_info = _line_info(instr)
    byte_info = _byte_info(instr, code_bytes)
    row: dict[str, object] = {
        "repo": context.base.repo,
        "commit": context.base.commit,
        "rel_path": context.base.rel_path,
        "code_unit_id": context.code_unit_id,
        "instr_id": instr_id,
        "instr_physical_id": _instruction_physical_id(context.code_unit_id, instr),
        "instr_index": instr_index,
        "start_offset": instr.start_offset,
        "offset": instr.offset,
        "cache_offset": instr.cache_offset,
        "end_offset": instr.end_offset,
        "ext_arg_len": offsets.ext_arg_len,
        "op_len": offsets.op_len,
        "cache_len": offsets.cache_len,
        "opcode": instr.opcode,
        "opname": instr.opname,
        "baseopcode": instr.baseopcode,
        "baseopname": instr.baseopname,
        "arg": instr.arg,
        "argrepr": instr.argrepr,
        "argval_kind": argval_info.kind,
        "argval_str": argval_info.text,
        "argval_int": argval_info.int_value,
        "argval_repr": argval_info.repr_text,
        "is_jump_target": instr.is_jump_target,
        "jump_target_offset": jump_info.target_offset,
        "jump_target_label": jump_info.target_label,
        "label": jump_info.label,
        "starts_line": line_info.starts_line,
        "line_number": line_info.line_number,
        "pos": positions_payload,
        "span_start_byte": span_start,
        "span_end_byte": span_end,
        "cache_info": _cache_info_payload(instr.cache_info),
        "cache_bytes": byte_info.cache_bytes,
        "op_bytes": byte_info.op_bytes,
    }
    info = _InstructionInfo(
        instr=instr,
        instr_id=instr_id,
        instr_index=instr_index,
        span_start_byte=span_start,
        span_end_byte=span_end,
    )
    return row, info


def _exception_rows(
    context: _CodeUnitContext,
    *,
    label_map: dict[int, str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    entries = _exception_entries(context.code)
    for index, entry in enumerate(entries):
        start_label = label_map.get(entry.start) or _label_for_offset(entry.start)
        end_label = label_map.get(entry.end) or _label_for_offset(entry.end)
        target_label = label_map.get(entry.target) or _label_for_offset(entry.target)
        rows.append(
            {
                "repo": context.base.repo,
                "commit": context.base.commit,
                "rel_path": context.base.rel_path,
                "code_unit_id": context.code_unit_id,
                "exc_entry_index": index,
                "start_offset": entry.start,
                "end_offset": entry.end,
                "target_offset": entry.target,
                "depth": entry.depth,
                "lasti": entry.lasti,
                "start_label": start_label,
                "end_label": end_label,
                "target_label": target_label,
            }
        )
    return rows


def _exception_entries(code: CodeType) -> list[object]:
    bytecode = dis.Bytecode(code)
    entries = getattr(bytecode, "exception_entries", None)
    if entries is None:
        return []
    return list(entries)


def _block_start_offsets(
    instructions: Sequence[_InstructionInfo],
    exception_entries: Sequence[object],
) -> list[int]:
    if not instructions:
        return []
    boundaries = _instruction_boundaries(instructions)
    boundaries.update(_exception_boundaries(exception_entries))
    return sorted(boundaries)


def _instruction_boundaries(instructions: Sequence[_InstructionInfo]) -> set[int]:
    boundaries: set[int] = {instructions[0].instr.offset}
    for idx, info in enumerate(instructions):
        boundaries.update(_boundaries_for_instruction(instructions, idx, info.instr))
    return boundaries


def _boundaries_for_instruction(
    instructions: Sequence[_InstructionInfo],
    idx: int,
    instr: dis.Instruction,
) -> set[int]:
    boundaries: set[int] = set()
    if instr.is_jump_target:
        boundaries.add(instr.offset)
    jump_target = instr.jump_target if isinstance(instr.jump_target, int) else None
    if jump_target is not None:
        boundaries.add(jump_target)
    next_offset = instructions[idx + 1].instr.offset if idx + 1 < len(instructions) else None
    boundaries.update(_next_offset_boundary(instr.opname, jump_target, next_offset))
    return boundaries


def _next_offset_boundary(
    opname: str | None,
    jump_target: int | None,
    next_offset: int | None,
) -> set[int]:
    if next_offset is None:
        return set()
    if _is_terminator(opname):
        return {next_offset}
    if jump_target is not None:
        return {next_offset}
    return set()


def _exception_boundaries(exception_entries: Sequence[object]) -> set[int]:
    boundaries: set[int] = set()
    for entry in exception_entries:
        start = getattr(entry, "start", None)
        target = getattr(entry, "target", None)
        if isinstance(start, int):
            boundaries.add(start)
        if isinstance(target, int):
            boundaries.add(target)
    return boundaries


def _build_blocks(
    context: _CodeUnitContext,
    *,
    instructions: Sequence[_InstructionInfo],
    label_map: dict[int, str],
    exception_entries: Sequence[object],
) -> tuple[list[dict[str, object]], list[_BlockInfo], dict[int, str]]:
    rows: list[dict[str, object]] = []
    blocks: list[_BlockInfo] = []
    offset_to_block_id: dict[int, str] = {}
    if not instructions:
        return rows, blocks, offset_to_block_id
    state = _BlockBuildState(
        rows=rows,
        blocks=blocks,
        offset_to_block_id=offset_to_block_id,
        label_map=label_map,
        context=context,
    )
    boundaries = set(_block_start_offsets(instructions, exception_entries))
    current: list[_InstructionInfo] = []
    for info in instructions:
        if info.instr.offset in boundaries and current:
            _finalize_block(current, state=state)
            current = []
        current.append(info)
    if current:
        _finalize_block(current, state=state)
    return rows, blocks, offset_to_block_id


def _finalize_block(
    infos: Sequence[_InstructionInfo],
    *,
    state: _BlockBuildState,
) -> None:
    first = infos[0]
    last = infos[-1]
    start_offset = first.instr.offset
    end_offset = last.instr.end_offset
    block_id = _stable_id("py_bc_block", state.context.code_unit_id, start_offset, end_offset)
    block_kind = "entry" if not state.blocks else "body"
    start_label = state.label_map.get(start_offset)
    state.rows.append(
        {
            "repo": state.context.base.repo,
            "commit": state.context.base.commit,
            "rel_path": state.context.base.rel_path,
            "block_id": block_id,
            "code_unit_id": state.context.code_unit_id,
            "start_offset": start_offset,
            "end_offset": end_offset,
            "start_label": start_label,
            "kind": block_kind,
            "anchor_span_start_byte": first.span_start_byte,
            "anchor_span_end_byte": last.span_end_byte,
            "first_instr_index": first.instr_index,
            "last_instr_index": last.instr_index,
        }
    )
    state.blocks.append(
        _BlockInfo(
            block_id=block_id,
            start_offset=start_offset,
            end_offset=end_offset,
            first_instr_index=first.instr_index,
            last_instr_index=last.instr_index,
            last_instr_id=last.instr_id,
            last_instr=last.instr,
            anchor_span_start_byte=first.span_start_byte,
            anchor_span_end_byte=last.span_end_byte,
        )
    )
    state.offset_to_block_id[start_offset] = block_id


def _cfg_edges(
    context: _CodeUnitContext,
    *,
    blocks: Sequence[_BlockInfo],
    offset_to_block_id: dict[int, str],
    exception_entries: Sequence[object],
) -> list[dict[str, object]]:
    edges: list[dict[str, object]] = []
    edge_keys: set[str] = set()
    block_list = list(blocks)
    _append_jump_edges(
        edges,
        edge_keys,
        context=context,
        block_list=block_list,
        offset_to_block_id=offset_to_block_id,
    )
    _append_exception_edges(
        edges,
        edge_keys,
        context=context,
        block_list=block_list,
        offset_to_block_id=offset_to_block_id,
        exception_entries=exception_entries,
    )
    return edges


def _append_edge(
    edges: list[dict[str, object]],
    edge_keys: set[str],
    *,
    context: _CodeUnitContext,
    src_block_id: str,
    dst_block_id: str,
    kind: str,
    cond_instr_id: str | None,
    exc_entry_index: int | None,
) -> None:
    edge_id = _stable_id(
        "py_bc_cfg",
        context.code_unit_id,
        src_block_id,
        dst_block_id,
        kind,
        cond_instr_id,
        exc_entry_index,
    )
    if edge_id in edge_keys:
        return
    edge_keys.add(edge_id)
    edges.append(
        {
            "repo": context.base.repo,
            "commit": context.base.commit,
            "rel_path": context.base.rel_path,
            "edge_id": edge_id,
            "code_unit_id": context.code_unit_id,
            "src_block_id": src_block_id,
            "dst_block_id": dst_block_id,
            "kind": kind,
            "cond_instr_id": cond_instr_id,
            "exc_entry_index": exc_entry_index,
        }
    )


def _append_jump_edges(
    edges: list[dict[str, object]],
    edge_keys: set[str],
    *,
    context: _CodeUnitContext,
    block_list: Sequence[_BlockInfo],
    offset_to_block_id: dict[int, str],
) -> None:
    for index, block in enumerate(block_list):
        last_instr = block.last_instr
        jump_target = last_instr.jump_target if isinstance(last_instr.jump_target, int) else None
        next_block = block_list[index + 1] if index + 1 < len(block_list) else None
        if jump_target is not None:
            _append_jump_target_edges(
                edges,
                edge_keys,
                context=context,
                block=block,
                next_block=next_block,
                jump_target=jump_target,
                offset_to_block_id=offset_to_block_id,
            )
            continue
        if _is_terminator(last_instr.opname) or next_block is None:
            continue
        _append_edge(
            edges,
            edge_keys,
            context=context,
            src_block_id=block.block_id,
            dst_block_id=next_block.block_id,
            kind="FALLTHROUGH",
            cond_instr_id=None,
            exc_entry_index=None,
        )


def _append_jump_target_edges(
    edges: list[dict[str, object]],
    edge_keys: set[str],
    *,
    context: _CodeUnitContext,
    block: _BlockInfo,
    next_block: _BlockInfo | None,
    jump_target: int,
    offset_to_block_id: dict[int, str],
) -> None:
    dst_block_id = offset_to_block_id.get(jump_target)
    if dst_block_id is not None:
        kind = "JUMP" if _is_unconditional_jump(block.last_instr.opname) else "BRANCH"
        cond_instr_id = None if kind == "JUMP" else block.last_instr_id
        _append_edge(
            edges,
            edge_keys,
            context=context,
            src_block_id=block.block_id,
            dst_block_id=dst_block_id,
            kind=kind,
            cond_instr_id=cond_instr_id,
            exc_entry_index=None,
        )
    if not _is_unconditional_jump(block.last_instr.opname) and next_block is not None:
        _append_edge(
            edges,
            edge_keys,
            context=context,
            src_block_id=block.block_id,
            dst_block_id=next_block.block_id,
            kind="FALLTHROUGH",
            cond_instr_id=block.last_instr_id,
            exc_entry_index=None,
        )


def _append_exception_edges(
    edges: list[dict[str, object]],
    edge_keys: set[str],
    *,
    context: _CodeUnitContext,
    block_list: Sequence[_BlockInfo],
    offset_to_block_id: dict[int, str],
    exception_entries: Sequence[object],
) -> None:
    for entry_index, entry in enumerate(exception_entries):
        target_block_id = _exception_target_block(entry, offset_to_block_id)
        if target_block_id is None:
            continue
        start, end = _exception_span(entry)
        if start is None or end is None:
            continue
        for block in block_list:
            if block.start_offset >= end or block.end_offset <= start:
                continue
            _append_edge(
                edges,
                edge_keys,
                context=context,
                src_block_id=block.block_id,
                dst_block_id=target_block_id,
                kind="EXCEPTION",
                cond_instr_id=None,
                exc_entry_index=entry_index,
            )


def _exception_target_block(
    entry: object,
    offset_to_block_id: dict[int, str],
) -> str | None:
    target_offset = getattr(entry, "target", None)
    if not isinstance(target_offset, int):
        return None
    return offset_to_block_id.get(target_offset)


def _exception_span(entry: object) -> tuple[int | None, int | None]:
    start = getattr(entry, "start", None)
    end = getattr(entry, "end", None)
    if not isinstance(start, int) or not isinstance(end, int):
        return None, None
    return start, end


def _defuse_event_from_op(
    opname: str | None,
    argval: object,
) -> tuple[str, str | None, str | None, float | None] | None:
    if opname is None:
        return None
    if opname.startswith("LOAD_"):
        space = _space_from_opname(opname)
        name = argval if isinstance(argval, str) else None
        return "USE", space, name, 1.0
    if opname.startswith("STORE_"):
        space = _space_from_opname(opname)
        name = argval if isinstance(argval, str) else None
        return "DEF", space, name, 1.0
    if opname.startswith("DELETE_"):
        space = _space_from_opname(opname)
        name = argval if isinstance(argval, str) else None
        return "KILL", space, name, 0.9
    return None


def _space_from_opname(opname: str) -> str | None:
    suffix_map = {
        "FAST": "local",
        "GLOBAL": "global",
        "NAME": "name",
        "DEREF": "free",
        "ATTR": "attribute",
        "SUBSCR": "subscript",
        "METHOD": "attribute",
    }
    for suffix, space in suffix_map.items():
        if opname.endswith(suffix):
            return space
    return None


def _build_defuse_events(
    context: _CodeUnitContext,
    *,
    instructions: Sequence[_InstructionInfo],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for info in instructions:
        opname = info.instr.baseopname or info.instr.opname
        event = _defuse_event_from_op(opname, info.instr.argval)
        if event is None:
            continue
        event_kind, space, name, confidence = event
        event_id = _stable_id(
            "py_bc_defuse",
            context.code_unit_id,
            info.instr_id,
            event_kind,
            space,
            name,
        )
        rows.append(
            {
                "repo": context.base.repo,
                "commit": context.base.commit,
                "rel_path": context.base.rel_path,
                "event_id": event_id,
                "code_unit_id": context.code_unit_id,
                "instr_id": info.instr_id,
                "instr_index": info.instr_index,
                "event_kind": event_kind,
                "space": space,
                "name": name,
                "confidence": confidence,
            }
        )
    return rows


def _iter_code_units(
    code: CodeType,
    *,
    context: _BytecodeContext,
) -> list[_CodeUnitInfo]:
    units: list[_CodeUnitInfo] = []

    def _walk(
        current: CodeType,
        parent_code_unit_id: str | None,
        parent_qualpath: str | None,
        counters: dict[tuple[str, int], int],
    ) -> None:
        qualpath = _code_unit_qualpath(
            context,
            code=current,
            parent_qualpath=parent_qualpath,
        )
        key = (current.co_name, current.co_firstlineno)
        ordinal = counters.get(key, 0) + 1
        counters[key] = ordinal
        code_unit_id = _code_unit_id(
            context,
            qualpath=qualpath,
            firstlineno=_normalize_line(current.co_firstlineno),
            ordinal=ordinal,
        )
        units.append(
            _CodeUnitInfo(
                code=current,
                code_unit_id=code_unit_id,
                parent_code_unit_id=parent_code_unit_id,
                qualpath=qualpath,
                ordinal=ordinal,
            )
        )
        child_counts: dict[tuple[str, int], int] = {}
        for const in current.co_consts:
            if isinstance(const, CodeType):
                _walk(const, code_unit_id, qualpath, child_counts)

    _walk(code, None, None, {})
    return units


def _append_code_unit_row(
    buffers: _DisBuffers,
    *,
    context: _BytecodeContext,
    unit: _CodeUnitInfo,
    span_start: int | None,
    span_end: int | None,
    kind: str,
) -> None:
    co_qualname = getattr(unit.code, "co_qualname", None)
    buffers.code_units.append(
        {
            "repo": context.repo,
            "commit": context.commit,
            "rel_path": context.rel_path,
            "code_unit_id": unit.code_unit_id,
            "parent_code_unit_id": unit.parent_code_unit_id,
            "qualpath": unit.qualpath,
            "co_name": unit.code.co_name,
            "co_qualname": co_qualname if isinstance(co_qualname, str) else None,
            "kind": kind,
            "co_firstlineno": _normalize_line(unit.code.co_firstlineno),
            "span_start_byte": span_start,
            "span_end_byte": span_end,
            "flags": unit.code.co_flags,
            "argcount": unit.code.co_argcount,
            "posonlyargcount": unit.code.co_posonlyargcount,
            "kwonlyargcount": unit.code.co_kwonlyargcount,
            "nlocals": unit.code.co_nlocals,
            "stacksize": unit.code.co_stacksize,
            "varnames": list(unit.code.co_varnames) or None,
            "names": list(unit.code.co_names) or None,
            "freevars": list(unit.code.co_freevars) or None,
            "cellvars": list(unit.code.co_cellvars) or None,
            "bytecode_len": len(unit.code.co_code),
            "exceptiontable_len": len(getattr(unit.code, "co_exceptiontable", b"")),
            "python_version": sys.version.split()[0],
            "bytecode_magic": importlib.util.MAGIC_NUMBER,
        }
    )


def _process_code_unit(
    context: _BytecodeContext,
    *,
    unit: _CodeUnitInfo,
    buffers: _DisBuffers,
) -> None:
    unit_context = _CodeUnitContext(
        base=context,
        code=unit.code,
        code_unit_id=unit.code_unit_id,
        qualpath=unit.qualpath,
    )
    instruction_rows, instruction_infos, label_map = _build_instruction_rows(unit_context)
    for row in instruction_rows:
        buffers.instructions.append(row)
    span_start, span_end = _code_unit_span_from_positions(
        instruction_infos,
        context.source_index,
        unit.code,
    )
    kind = _code_unit_kind(unit.code, context.source_index)
    _append_code_unit_row(
        buffers,
        context=context,
        unit=unit,
        span_start=span_start,
        span_end=span_end,
        kind=kind,
    )
    exception_entries = dis.Bytecode(unit.code).exception_entries
    if context.options.include_exception_table:
        for row in _exception_rows(unit_context, label_map=label_map):
            buffers.exceptions.append(row)
    if context.options.include_cfg:
        block_rows, block_infos, offset_map = _build_blocks(
            unit_context,
            instructions=instruction_infos,
            label_map=label_map,
            exception_entries=exception_entries,
        )
        for row in block_rows:
            buffers.blocks.append(row)
        cfg_rows = _cfg_edges(
            unit_context,
            blocks=block_infos,
            offset_to_block_id=offset_map,
            exception_entries=exception_entries,
        )
        for row in cfg_rows:
            buffers.cfg_edges.append(row)
    if context.options.include_defuse:
        for row in _build_defuse_events(unit_context, instructions=instruction_infos):
            buffers.defuse_events.append(row)


def _process_module(
    context: _BytecodeContext,
    *,
    module: ModuleRecord,
    source_text: str,
    buffers: _DisBuffers,
    warnings: list[str],
) -> None:
    try:
        code = compile(
            source_text,
            str(module.file_path),
            "exec",
            dont_inherit=True,
            optimize=context.options.optimize,
        )
    except (SyntaxError, ValueError, TypeError) as exc:
        message = f"Bytecode compile failed for {module.rel_path}: {exc}"
        warnings.append(message)
        LOG.warning("%s", message)
        return
    for unit in _iter_code_units(code, context=context):
        _process_code_unit(context, unit=unit, buffers=buffers)


class DisExtractStep(BaseExtractStep):
    """Bytecode extraction step with port injection."""

    def __init__(
        self,
        discovery: ModuleDiscoveryPort,
        *,
        options: BytecodeExtractOptions | None = None,
    ) -> None:
        super().__init__(discovery=discovery)
        self._options = options or BytecodeExtractOptions()

    def execute(
        self,
        modules: Sequence[ModuleRecord],
        *,
        repo: str,
        commit: str,
    ) -> DisExtractResult:
        """Execute bytecode extraction for the provided modules.

        Returns
        -------
        DisExtractResult
            Result bundle with row payloads and execution status.
        """
        try:
            buffers = _build_dis_buffers()
        except (KeyError, RuntimeError) as exc:
            return DisExtractResult(result=ExecutionResult.failed(str(exc)))

        warnings: list[str] = []
        for module, source_text, source_index in self._iter_python_source_bundles(modules):
            context = _BytecodeContext(
                repo=repo,
                commit=commit,
                rel_path=module.rel_path,
                module_name=module.module_name,
                source_index=source_index,
                options=self._options,
            )
            _process_module(
                context,
                module=module,
                source_text=source_text,
                buffers=buffers,
                warnings=warnings,
            )

        LOG.info(
            "Bytecode extraction: repo=%s commit=%s code_units=%d instr=%d",
            repo,
            commit,
            buffers.code_units.row_count,
            buffers.instructions.row_count,
        )
        return DisExtractResult(
            result=ExecutionResult.ok(warnings=tuple(warnings)),
            code_unit_rows=buffers.code_units.data,
            instruction_rows=buffers.instructions.data,
            exception_rows=buffers.exceptions.data,
            block_rows=buffers.blocks.data,
            cfg_edge_rows=buffers.cfg_edges.data,
            defuse_event_rows=buffers.defuse_events.data,
            code_unit_row_count=buffers.code_units.row_count,
            instruction_row_count=buffers.instructions.row_count,
            exception_row_count=buffers.exceptions.row_count,
            block_row_count=buffers.blocks.row_count,
            cfg_edge_row_count=buffers.cfg_edges.row_count,
            defuse_event_row_count=buffers.defuse_events.row_count,
        )

    def _iter_python_source_bundles(
        self,
        modules: Sequence[ModuleRecord],
    ) -> Iterable[tuple[ModuleRecord, str, LineIndexedSource]]:
        for module in modules:
            if not module.rel_path.endswith(".py"):
                continue
            source_bytes = self._discovery.read_module_bytes(module)
            if source_bytes is None:
                source_text = self._discovery.read_module_source(module)
                if source_text is None:
                    continue
                source_bytes = source_text.encode("utf-8", errors="replace")
            source_text, source_index = _build_source_index(source_bytes)
            yield module, source_text, source_index


__all__ = ["DisExtractResult", "DisExtractStep"]
