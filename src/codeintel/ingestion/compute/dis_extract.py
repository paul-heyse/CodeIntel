"""Bytecode extraction step with port injection."""

from __future__ import annotations

import dis
import hashlib
import importlib.util
import inspect
import io
import logging
import marshal
import sys
import tempfile
import time
import tokenize
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from types import CodeType
from typing import TYPE_CHECKING

from codeintel.build.hamilton.execution_result import ExecutionResult
from codeintel.build.hamilton.native.options.ingestion import BytecodeExtractOptions
from codeintel.core.columnar.rows import (
    ColumnarBatchCollector,
    ColumnarRows,
    columnar_batch_collector_for_table_key,
    empty_table_for_table,
)
from codeintel.ingestion.compute.base import (
    BaseExtractStep,
    finalize_arrow_readers,
    persist_arrow_tables,
)
from codeintel.ingestion.context import IngestionContext, resolve_repo_commit
from codeintel.ingestion.infrastructure.cst_utils import LineIndexedSource

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import pyarrow as pa

    from codeintel.ingestion.infrastructure.py_frontend import PyFrontend
    from codeintel.ingestion.ports.discovery import ModuleDiscoveryPort, ModuleRecord
    from codeintel.ingestion.ports.storage import IngestStoragePort

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
    code_unit_rows_reader: pa.Table = field(
        default_factory=lambda: empty_table_for_table(PY_BC_CODE_UNITS_TABLE_KEY)
    )
    instruction_rows_reader: pa.Table = field(
        default_factory=lambda: empty_table_for_table(PY_BC_INSTRUCTIONS_TABLE_KEY)
    )
    exception_rows_reader: pa.Table = field(
        default_factory=lambda: empty_table_for_table(PY_BC_EXCEPTION_TABLE_KEY)
    )
    block_rows_reader: pa.Table = field(
        default_factory=lambda: empty_table_for_table(PY_BC_BLOCKS_TABLE_KEY)
    )
    cfg_edge_rows_reader: pa.Table = field(
        default_factory=lambda: empty_table_for_table(PY_BC_CFG_EDGES_TABLE_KEY)
    )
    defuse_event_rows_reader: pa.Table = field(
        default_factory=lambda: empty_table_for_table(PY_BC_DEFUSE_EVENTS_TABLE_KEY)
    )
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
class _ExceptionEntry:
    start: int
    end: int
    target: int
    depth: int | None
    lasti: bool | None


@dataclass(frozen=True, slots=True)
class _BytecodeContext:
    repo: str
    commit: str
    rel_path: str
    module_name: str
    source_index: LineIndexedSource
    options: BytecodeExtractOptions
    frontend: PyFrontend | None


@dataclass(frozen=True, slots=True)
class _CodeUnitContext:
    base: _BytecodeContext
    code: CodeType
    code_unit_id: str
    qualpath: str


@dataclass(frozen=True, slots=True)
class _CodeUnitRowSpec:
    parent_code_unit_id: str | None
    span_start: int | None
    span_end: int | None
    kind: str


@dataclass(frozen=True, slots=True)
class _DisCollectors:
    code_units: ColumnarBatchCollector
    instructions: ColumnarBatchCollector
    exceptions: ColumnarBatchCollector
    blocks: ColumnarBatchCollector
    cfg_edges: ColumnarBatchCollector
    defuse_events: ColumnarBatchCollector


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


@dataclass(frozen=True, slots=True)
class _InstructionRowInputs:
    context: _CodeUnitContext
    instr: dis.Instruction
    instr_index: int
    instr_id: str
    positions_payload: dict[str, object] | None
    span_start: int | None
    span_end: int | None
    label_map: dict[int, str]
    code_bytes: bytes


@dataclass(frozen=True, slots=True)
class _CodeUnitRowInputs:
    unit: _CodeUnitInfo
    span_start: int | None
    span_end: int | None
    kind: str


@dataclass(frozen=True, slots=True)
class _ModuleDisResult:
    warnings: list[str]
    code_unit_batches: list[pa.RecordBatch]
    instruction_batches: list[pa.RecordBatch]
    exception_batches: list[pa.RecordBatch]
    block_batches: list[pa.RecordBatch]
    cfg_edge_batches: list[pa.RecordBatch]
    defuse_event_batches: list[pa.RecordBatch]
    code_unit_row_count: int
    instruction_row_count: int
    exception_row_count: int
    block_row_count: int
    cfg_edge_row_count: int
    defuse_event_row_count: int


@dataclass(frozen=True, slots=True)
class _DisTables:
    code_units: pa.Table
    instructions: pa.Table
    exceptions: pa.Table
    blocks: pa.Table
    cfg_edges: pa.Table
    defuse_events: pa.Table


@dataclass(frozen=True, slots=True)
class _DisModuleJob:
    module: ModuleRecord
    source_text: str
    source_index: LineIndexedSource
    repo: str
    commit: str
    options: BytecodeExtractOptions
    frontend: PyFrontend | None


def _stable_id(*parts: object) -> str:
    payload = "|".join("" if part is None else str(part) for part in parts)
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=16)
    return digest.hexdigest()


def _cache_key(
    context: _BytecodeContext,
    *,
    rel_path: str,
) -> str:
    python_version = sys.version.split()[0]
    payload = "|".join(
        [
            context.repo,
            context.commit,
            rel_path,
            python_version,
            str(context.options.compile_flags),
            str(context.options.optimize),
            str(context.options.dont_inherit),
        ]
    )
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=16)
    return digest.hexdigest()


def _cache_path(cache_dir: Path, cache_key: str) -> Path:
    return cache_dir / f"{cache_key}.marshal"


def _cache_enabled(options: BytecodeExtractOptions) -> bool:
    return options.enable_cache and options.cache_dir is not None


def _load_cached_code(
    context: _BytecodeContext,
    *,
    rel_path: str,
    warnings: list[str],
) -> CodeType | None:
    if not _cache_enabled(context.options):
        return None
    cache_dir = context.options.cache_dir
    if cache_dir is None:
        return None
    payload = _read_cache_payload(
        context=context,
        cache_dir=cache_dir,
        rel_path=rel_path,
        warnings=warnings,
    )
    if payload is None:
        return None
    return _decode_cached_payload(payload=payload, rel_path=rel_path, warnings=warnings)


def _read_cache_payload(
    *,
    context: _BytecodeContext,
    cache_dir: Path,
    rel_path: str,
    warnings: list[str],
) -> bytes | None:
    cache_key = _cache_key(context, rel_path=rel_path)
    cache_file = _cache_path(cache_dir, cache_key)
    try:
        return cache_file.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as exc:
        warnings.append(f"Bytecode cache read failed for {rel_path}: {exc}")
        return None


def _decode_cached_payload(
    *,
    payload: bytes,
    rel_path: str,
    warnings: list[str],
) -> CodeType | None:
    try:
        cached = marshal.loads(payload)
    except (ValueError, TypeError) as exc:
        warnings.append(f"Bytecode cache decode failed for {rel_path}: {exc}")
        return None
    if not isinstance(cached, CodeType):
        warnings.append(f"Bytecode cache type mismatch for {rel_path}")
        return None
    return cached


def _store_cached_code(
    context: _BytecodeContext,
    *,
    rel_path: str,
    code: CodeType,
    warnings: list[str],
) -> None:
    if not _cache_enabled(context.options):
        return
    cache_dir = context.options.cache_dir
    if cache_dir is None:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = _cache_key(context, rel_path=rel_path)
    cache_file = _cache_path(cache_dir, cache_key)
    try:
        with tempfile.NamedTemporaryFile(dir=cache_dir, delete=False) as handle:
            handle.write(marshal.dumps(code))
            temp_path = Path(handle.name)
        temp_path.replace(cache_file)
    except OSError as exc:
        warnings.append(f"Bytecode cache write failed for {rel_path}: {exc}")


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


def _build_dis_collectors(options: BytecodeExtractOptions) -> _DisCollectors:
    return _DisCollectors(
        code_units=columnar_batch_collector_for_table_key(
            PY_BC_CODE_UNITS_TABLE_KEY,
            batch_size=options.batch_size,
        ),
        instructions=columnar_batch_collector_for_table_key(
            PY_BC_INSTRUCTIONS_TABLE_KEY,
            batch_size=options.batch_size,
        ),
        exceptions=columnar_batch_collector_for_table_key(
            PY_BC_EXCEPTION_TABLE_KEY,
            batch_size=options.batch_size,
        ),
        blocks=columnar_batch_collector_for_table_key(
            PY_BC_BLOCKS_TABLE_KEY,
            batch_size=options.batch_size,
        ),
        cfg_edges=columnar_batch_collector_for_table_key(
            PY_BC_CFG_EDGES_TABLE_KEY,
            batch_size=options.batch_size,
        ),
        defuse_events=columnar_batch_collector_for_table_key(
            PY_BC_DEFUSE_EVENTS_TABLE_KEY,
            batch_size=options.batch_size,
        ),
    )


def _flush_dis_collectors(collectors: _DisCollectors) -> None:
    collectors.code_units.flush()
    collectors.instructions.flush()
    collectors.exceptions.flush()
    collectors.blocks.flush()
    collectors.cfg_edges.flush()
    collectors.defuse_events.flush()


def _merge_dis_batches(
    collector: ColumnarBatchCollector,
    *,
    batches: list[pa.RecordBatch],
    row_count: int,
) -> None:
    if not batches and row_count == 0:
        return
    collector.flush()
    collector.batches.extend(batches)
    collector.row_count += row_count


def _merge_dis_result(collectors: _DisCollectors, result: _ModuleDisResult) -> None:
    _merge_dis_batches(
        collectors.code_units,
        batches=result.code_unit_batches,
        row_count=result.code_unit_row_count,
    )
    _merge_dis_batches(
        collectors.instructions,
        batches=result.instruction_batches,
        row_count=result.instruction_row_count,
    )
    _merge_dis_batches(
        collectors.exceptions,
        batches=result.exception_batches,
        row_count=result.exception_row_count,
    )
    _merge_dis_batches(
        collectors.blocks,
        batches=result.block_batches,
        row_count=result.block_row_count,
    )
    _merge_dis_batches(
        collectors.cfg_edges,
        batches=result.cfg_edge_batches,
        row_count=result.cfg_edge_row_count,
    )
    _merge_dis_batches(
        collectors.defuse_events,
        batches=result.defuse_event_batches,
        row_count=result.defuse_event_row_count,
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


_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1


def _safe_int64(value: int) -> int | None:
    if _INT64_MIN <= value <= _INT64_MAX:
        return value
    return None


def _argval_fields(argval: object) -> _ArgvalInfo:
    if argval is None:
        return _ArgvalInfo(None, None, None, None)
    if isinstance(argval, str):
        return _ArgvalInfo("str", argval, None, None)
    if isinstance(argval, bool):
        return _ArgvalInfo("bool", None, None, _truncate_repr(argval))
    if isinstance(argval, int):
        safe_value = _safe_int64(argval)
        repr_text = None if safe_value is not None else _truncate_repr(argval)
        return _ArgvalInfo("int", None, safe_value, repr_text)
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
            _InstructionRowInputs(
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
        )
        rows.append(row)
        infos.append(info)
    return rows, infos, label_map


def _label_map(instructions: Sequence[dis.Instruction]) -> dict[int, str]:
    return {instr.offset: _label_for_offset(instr.offset) or "" for instr in instructions}


def _instruction_row(inputs: _InstructionRowInputs) -> tuple[dict[str, object], _InstructionInfo]:
    argval_info = _argval_fields(inputs.instr.argval)
    offsets = _offset_info(inputs.instr)
    jump_info = _jump_info(inputs.instr, inputs.label_map)
    line_info = _line_info(inputs.instr)
    byte_info = _byte_info(inputs.instr, inputs.code_bytes)
    row: dict[str, object] = {
        "repo": inputs.context.base.repo,
        "commit": inputs.context.base.commit,
        "rel_path": inputs.context.base.rel_path,
        "code_unit_id": inputs.context.code_unit_id,
        "instr_id": inputs.instr_id,
        "instr_physical_id": _instruction_physical_id(inputs.context.code_unit_id, inputs.instr),
        "instr_index": inputs.instr_index,
        "start_offset": inputs.instr.start_offset,
        "offset": inputs.instr.offset,
        "cache_offset": inputs.instr.cache_offset,
        "end_offset": inputs.instr.end_offset,
        "ext_arg_len": offsets.ext_arg_len,
        "op_len": offsets.op_len,
        "cache_len": offsets.cache_len,
        "opcode": inputs.instr.opcode,
        "opname": inputs.instr.opname,
        "baseopcode": inputs.instr.baseopcode,
        "baseopname": inputs.instr.baseopname,
        "arg": inputs.instr.arg,
        "argrepr": inputs.instr.argrepr,
        "argval_kind": argval_info.kind,
        "argval_str": argval_info.text,
        "argval_int": argval_info.int_value,
        "argval_repr": argval_info.repr_text,
        "is_jump_target": inputs.instr.is_jump_target,
        "jump_target_offset": jump_info.target_offset,
        "jump_target_label": jump_info.target_label,
        "label": jump_info.label,
        "starts_line": line_info.starts_line,
        "line_number": line_info.line_number,
        "pos": inputs.positions_payload,
        "span_start_byte": inputs.span_start,
        "span_end_byte": inputs.span_end,
        "cache_info": _cache_info_payload(inputs.instr.cache_info),
        "cache_bytes": byte_info.cache_bytes,
        "op_bytes": byte_info.op_bytes,
    }
    info = _InstructionInfo(
        instr=inputs.instr,
        instr_id=inputs.instr_id,
        instr_index=inputs.instr_index,
        span_start_byte=inputs.span_start,
        span_end_byte=inputs.span_end,
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


def _exception_entry(entry: object) -> _ExceptionEntry | None:
    start = getattr(entry, "start", None)
    end = getattr(entry, "end", None)
    target = getattr(entry, "target", None)
    if not isinstance(start, int) or not isinstance(end, int) or not isinstance(target, int):
        return None
    depth = getattr(entry, "depth", None)
    lasti = getattr(entry, "lasti", None)
    return _ExceptionEntry(
        start=start,
        end=end,
        target=target,
        depth=depth if isinstance(depth, int) else None,
        lasti=lasti if isinstance(lasti, bool) else None,
    )


def _exception_entries(code: CodeType) -> list[_ExceptionEntry]:
    bytecode = dis.Bytecode(code)
    entries = getattr(bytecode, "exception_entries", None)
    if not entries:
        return []
    normalized: list[_ExceptionEntry] = []
    for entry in entries:
        normalized_entry = _exception_entry(entry)
        if normalized_entry is not None:
            normalized.append(normalized_entry)
    return normalized


def _block_start_offsets(
    instructions: Sequence[_InstructionInfo],
    exception_entries: Sequence[_ExceptionEntry],
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


def _exception_boundaries(exception_entries: Sequence[_ExceptionEntry]) -> set[int]:
    boundaries: set[int] = set()
    for entry in exception_entries:
        boundaries.add(entry.start)
        boundaries.add(entry.target)
    return boundaries


def _build_blocks(
    context: _CodeUnitContext,
    *,
    instructions: Sequence[_InstructionInfo],
    label_map: dict[int, str],
    exception_entries: Sequence[_ExceptionEntry],
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
    exception_entries: Sequence[_ExceptionEntry],
) -> list[dict[str, object]]:
    state = _EdgeBuildState(edges=[], edge_keys=set(), context=context)
    block_list = list(blocks)
    _append_jump_edges(
        state,
        block_list=block_list,
        offset_to_block_id=offset_to_block_id,
    )
    _append_exception_edges(
        state,
        block_list=block_list,
        offset_to_block_id=offset_to_block_id,
        exception_entries=exception_entries,
    )
    return state.edges


def _append_edge(state: _EdgeBuildState, *, spec: _EdgeSpec) -> None:
    edge_id = _stable_id(
        "py_bc_cfg",
        state.context.code_unit_id,
        spec.src_block_id,
        spec.dst_block_id,
        spec.kind,
        spec.cond_instr_id,
        spec.exc_entry_index,
    )
    if edge_id in state.edge_keys:
        return
    state.edge_keys.add(edge_id)
    state.edges.append(
        {
            "repo": state.context.base.repo,
            "commit": state.context.base.commit,
            "rel_path": state.context.base.rel_path,
            "edge_id": edge_id,
            "code_unit_id": state.context.code_unit_id,
            "src_block_id": spec.src_block_id,
            "dst_block_id": spec.dst_block_id,
            "kind": spec.kind,
            "cond_instr_id": spec.cond_instr_id,
            "exc_entry_index": spec.exc_entry_index,
        }
    )


def _append_jump_edges(
    state: _EdgeBuildState,
    *,
    block_list: Sequence[_BlockInfo],
    offset_to_block_id: dict[int, str],
) -> None:
    for index, block in enumerate(block_list):
        last_instr = block.last_instr
        jump_target = last_instr.jump_target if isinstance(last_instr.jump_target, int) else None
        next_block = block_list[index + 1] if index + 1 < len(block_list) else None
        if jump_target is not None:
            _append_jump_target_edges(
                state,
                block=block,
                next_block=next_block,
                jump_target=jump_target,
                offset_to_block_id=offset_to_block_id,
            )
            continue
        if _is_terminator(last_instr.opname) or next_block is None:
            continue
        _append_edge(
            state,
            spec=_EdgeSpec(
                src_block_id=block.block_id,
                dst_block_id=next_block.block_id,
                kind="FALLTHROUGH",
                cond_instr_id=None,
                exc_entry_index=None,
            ),
        )


def _append_jump_target_edges(
    state: _EdgeBuildState,
    *,
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
            state,
            spec=_EdgeSpec(
                src_block_id=block.block_id,
                dst_block_id=dst_block_id,
                kind=kind,
                cond_instr_id=cond_instr_id,
                exc_entry_index=None,
            ),
        )
    if not _is_unconditional_jump(block.last_instr.opname) and next_block is not None:
        _append_edge(
            state,
            spec=_EdgeSpec(
                src_block_id=block.block_id,
                dst_block_id=next_block.block_id,
                kind="FALLTHROUGH",
                cond_instr_id=block.last_instr_id,
                exc_entry_index=None,
            ),
        )


def _append_exception_edges(
    state: _EdgeBuildState,
    *,
    block_list: Sequence[_BlockInfo],
    offset_to_block_id: dict[int, str],
    exception_entries: Sequence[_ExceptionEntry],
) -> None:
    for entry_index, entry in enumerate(exception_entries):
        target_block_id = _exception_target_block(entry, offset_to_block_id)
        if target_block_id is None:
            continue
        start, end = _exception_span(entry)
        for block in block_list:
            if block.start_offset >= end or block.end_offset <= start:
                continue
            _append_edge(
                state,
                spec=_EdgeSpec(
                    src_block_id=block.block_id,
                    dst_block_id=target_block_id,
                    kind="EXCEPTION",
                    cond_instr_id=None,
                    exc_entry_index=entry_index,
                ),
            )


def _exception_target_block(
    entry: _ExceptionEntry,
    offset_to_block_id: dict[int, str],
) -> str | None:
    return offset_to_block_id.get(entry.target)


def _exception_span(entry: _ExceptionEntry) -> tuple[int, int]:
    return entry.start, entry.end


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
    collectors: _DisCollectors,
    *,
    context: _BytecodeContext,
    inputs: _CodeUnitRowInputs,
) -> None:
    unit = inputs.unit
    co_qualname = getattr(unit.code, "co_qualname", None)
    collectors.code_units.append(
        {
            "repo": context.repo,
            "commit": context.commit,
            "rel_path": context.rel_path,
            "code_unit_id": unit.code_unit_id,
            "parent_code_unit_id": unit.parent_code_unit_id,
            "qualpath": unit.qualpath,
            "co_name": unit.code.co_name,
            "co_qualname": co_qualname if isinstance(co_qualname, str) else None,
            "kind": inputs.kind,
            "co_firstlineno": _normalize_line(unit.code.co_firstlineno),
            "span_start_byte": inputs.span_start,
            "span_end_byte": inputs.span_end,
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
            "optimize": context.options.optimize,
            "dont_inherit": context.options.dont_inherit,
        }
    )


def _process_code_unit(
    context: _BytecodeContext,
    *,
    unit: _CodeUnitInfo,
    collectors: _DisCollectors,
) -> None:
    unit_context = _CodeUnitContext(
        base=context,
        code=unit.code,
        code_unit_id=unit.code_unit_id,
        qualpath=unit.qualpath,
    )
    instruction_rows, instruction_infos, label_map = _build_instruction_rows(unit_context)
    for row in instruction_rows:
        collectors.instructions.append(row)
    span_start, span_end = _code_unit_span_from_positions(
        instruction_infos,
        context.source_index,
        unit.code,
    )
    kind = _code_unit_kind(unit.code, context.source_index)
    _append_code_unit_row(
        collectors,
        context=context,
        inputs=_CodeUnitRowInputs(
            unit=unit,
            span_start=span_start,
            span_end=span_end,
            kind=kind,
        ),
    )
    exception_entries = _exception_entries(unit.code)
    if context.options.include_exception_table:
        for row in _exception_rows(unit_context, label_map=label_map):
            collectors.exceptions.append(row)
    if context.options.include_cfg:
        block_rows, block_infos, offset_map = _build_blocks(
            unit_context,
            instructions=instruction_infos,
            label_map=label_map,
            exception_entries=exception_entries,
        )
        for row in block_rows:
            collectors.blocks.append(row)
        cfg_rows = _cfg_edges(
            unit_context,
            blocks=block_infos,
            offset_to_block_id=offset_map,
            exception_entries=exception_entries,
        )
        for row in cfg_rows:
            collectors.cfg_edges.append(row)
    if context.options.include_defuse:
        for row in _build_defuse_events(unit_context, instructions=instruction_infos):
            collectors.defuse_events.append(row)


def _process_module(
    context: _BytecodeContext,
    *,
    module: ModuleRecord,
    source_text: str,
    collectors: _DisCollectors,
    warnings: list[str],
) -> None:
    code = _load_cached_code(context, rel_path=module.rel_path, warnings=warnings)
    if code is None and context.frontend is not None:
        code = context.frontend.get_code(
            module,
            dont_inherit=context.options.dont_inherit,
            optimize=context.options.optimize,
            flags=context.options.compile_flags,
        )
    if code is None:
        try:
            code = compile(
                source_text,
                str(module.file_path),
                "exec",
                dont_inherit=context.options.dont_inherit,
                optimize=context.options.optimize,
                flags=context.options.compile_flags,
            )
        except (SyntaxError, ValueError, TypeError) as exc:
            message = f"Bytecode compile failed for {module.rel_path}: {exc}"
            warnings.append(message)
            LOG.warning("%s", message)
            return
        _store_cached_code(context, rel_path=module.rel_path, code=code, warnings=warnings)
    for unit in _iter_code_units(code, context=context):
        _process_code_unit(context, unit=unit, collectors=collectors)


def _module_size_bytes(module: ModuleRecord) -> int | None:
    try:
        return module.file_path.stat().st_size
    except OSError:
        return None


def _extract_module_rows(
    job: _DisModuleJob,
) -> _ModuleDisResult:
    collectors = _build_dis_collectors(job.options)
    warnings: list[str] = []
    context = _BytecodeContext(
        repo=job.repo,
        commit=job.commit,
        rel_path=job.module.rel_path,
        module_name=job.module.module_name,
        source_index=job.source_index,
        options=job.options,
        frontend=job.frontend,
    )
    start_time = time.monotonic()
    _process_module(
        context,
        module=job.module,
        source_text=job.source_text,
        collectors=collectors,
        warnings=warnings,
    )
    elapsed = time.monotonic() - start_time
    max_seconds = job.options.max_module_seconds
    if max_seconds is not None and max_seconds > 0 and elapsed > max_seconds:
        warnings.append(
            f"Bytecode module budget exceeded for {job.module.rel_path}: {elapsed:.2f}s"
        )
    _flush_dis_collectors(collectors)
    return _ModuleDisResult(
        warnings=warnings,
        code_unit_batches=list(collectors.code_units.batches),
        instruction_batches=list(collectors.instructions.batches),
        exception_batches=list(collectors.exceptions.batches),
        block_batches=list(collectors.blocks.batches),
        cfg_edge_batches=list(collectors.cfg_edges.batches),
        defuse_event_batches=list(collectors.defuse_events.batches),
        code_unit_row_count=collectors.code_units.row_count,
        instruction_row_count=collectors.instructions.row_count,
        exception_row_count=collectors.exceptions.row_count,
        block_row_count=collectors.blocks.row_count,
        cfg_edge_row_count=collectors.cfg_edges.row_count,
        defuse_event_row_count=collectors.defuse_events.row_count,
    )


class DisExtractStep(BaseExtractStep):
    """Bytecode extraction step with port injection."""

    def __init__(
        self,
        discovery: ModuleDiscoveryPort,
        *,
        options: BytecodeExtractOptions | None = None,
        frontend: PyFrontend | None = None,
    ) -> None:
        super().__init__(discovery=discovery, frontend=frontend)
        self._options = options or BytecodeExtractOptions()

    def execute(
        self,
        modules: Sequence[ModuleRecord],
        *,
        repo: str | None = None,
        commit: str | None = None,
        context: IngestionContext | None = None,
        storage: IngestStoragePort | None = None,
    ) -> DisExtractResult:
        """Execute bytecode extraction for the provided modules.

        Returns
        -------
        DisExtractResult
            Result bundle with row payloads and execution status.
        """
        resolved_repo, resolved_commit = resolve_repo_commit(
            context=context,
            repo=repo,
            commit=commit,
        )
        if not self._options.enable:
            return DisExtractResult(
                result=ExecutionResult.skip("Bytecode extraction disabled by options")
            )
        options = self._options
        try:
            collectors = _build_dis_collectors(options)
        except (KeyError, RuntimeError) as exc:
            return DisExtractResult(result=ExecutionResult.failed(str(exc)))

        warnings: list[str] = []
        module_bundles_iter = self._iter_python_source_bundles(
            modules,
            warnings=warnings,
            max_module_bytes=options.max_module_bytes,
        )
        worker_count = max(options.max_workers, 1)
        module_bundles = list(module_bundles_iter) if worker_count > 1 else module_bundles_iter
        if worker_count > 1 and isinstance(module_bundles, list) and len(module_bundles) > 1:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_map = {
                    executor.submit(
                        _extract_module_rows,
                        _DisModuleJob(
                            module=module,
                            source_text=source_text,
                            source_index=source_index,
                            repo=resolved_repo,
                            commit=resolved_commit,
                            options=options,
                            frontend=self._frontend,
                        ),
                    ): module
                    for module, source_text, source_index in module_bundles
                }
                for future in future_map:
                    result = future.result()
                    if result.warnings:
                        warnings.extend(result.warnings)
                    _merge_dis_result(collectors, result)
        else:
            for module, source_text, source_index in module_bundles:
                result = _extract_module_rows(
                    _DisModuleJob(
                        module=module,
                        source_text=source_text,
                        source_index=source_index,
                        repo=resolved_repo,
                        commit=resolved_commit,
                        options=options,
                        frontend=self._frontend,
                    )
                )
                warnings.extend(result.warnings)
                _merge_dis_result(collectors, result)

        LOG.info(
            "Bytecode extraction: repo=%s commit=%s code_units=%d instr=%d",
            resolved_repo,
            resolved_commit,
            collectors.code_units.row_count,
            collectors.instructions.row_count,
        )
        finalized_tables, finalize_warnings = finalize_arrow_readers(
            {
                PY_BC_CODE_UNITS_TABLE_KEY: collectors.code_units.to_reader(),
                PY_BC_INSTRUCTIONS_TABLE_KEY: collectors.instructions.to_reader(),
                PY_BC_EXCEPTION_TABLE_KEY: collectors.exceptions.to_reader(),
                PY_BC_BLOCKS_TABLE_KEY: collectors.blocks.to_reader(),
                PY_BC_CFG_EDGES_TABLE_KEY: collectors.cfg_edges.to_reader(),
                PY_BC_DEFUSE_EVENTS_TABLE_KEY: collectors.defuse_events.to_reader(),
            }
        )
        warnings.extend(finalize_warnings)
        tables = _DisTables(
            code_units=finalized_tables[PY_BC_CODE_UNITS_TABLE_KEY],
            instructions=finalized_tables[PY_BC_INSTRUCTIONS_TABLE_KEY],
            exceptions=finalized_tables[PY_BC_EXCEPTION_TABLE_KEY],
            blocks=finalized_tables[PY_BC_BLOCKS_TABLE_KEY],
            cfg_edges=finalized_tables[PY_BC_CFG_EDGES_TABLE_KEY],
            defuse_events=finalized_tables[PY_BC_DEFUSE_EVENTS_TABLE_KEY],
        )
        scope = f"{resolved_repo}@{resolved_commit}"
        persist_arrow_tables(
            storage,
            {
                PY_BC_CODE_UNITS_TABLE_KEY: tables.code_units,
                PY_BC_INSTRUCTIONS_TABLE_KEY: tables.instructions,
                PY_BC_EXCEPTION_TABLE_KEY: tables.exceptions,
                PY_BC_BLOCKS_TABLE_KEY: tables.blocks,
                PY_BC_CFG_EDGES_TABLE_KEY: tables.cfg_edges,
                PY_BC_DEFUSE_EVENTS_TABLE_KEY: tables.defuse_events,
            },
            scope=scope,
        )
        return DisExtractResult(
            result=ExecutionResult.ok(warnings=tuple(warnings)),
            code_unit_rows={},
            instruction_rows={},
            exception_rows={},
            block_rows={},
            cfg_edge_rows={},
            defuse_event_rows={},
            code_unit_rows_reader=tables.code_units,
            instruction_rows_reader=tables.instructions,
            exception_rows_reader=tables.exceptions,
            block_rows_reader=tables.blocks,
            cfg_edge_rows_reader=tables.cfg_edges,
            defuse_event_rows_reader=tables.defuse_events,
            code_unit_row_count=tables.code_units.num_rows,
            instruction_row_count=tables.instructions.num_rows,
            exception_row_count=tables.exceptions.num_rows,
            block_row_count=tables.blocks.num_rows,
            cfg_edge_row_count=tables.cfg_edges.num_rows,
            defuse_event_row_count=tables.defuse_events.num_rows,
        )

    def _iter_python_source_bundles(
        self,
        modules: Sequence[ModuleRecord],
        *,
        warnings: list[str],
        max_module_bytes: int | None,
    ) -> Iterable[tuple[ModuleRecord, str, LineIndexedSource]]:
        for module in modules:
            if not module.rel_path.endswith(".py"):
                continue
            if max_module_bytes is not None:
                size_bytes = _module_size_bytes(module)
                if size_bytes is not None and size_bytes > max_module_bytes:
                    warnings.append(
                        f"Skipping bytecode for {module.rel_path} (size {size_bytes} bytes)"
                    )
                    continue
            if self._frontend is not None:
                bundle = self._frontend.get_source_bundle(module)
                if bundle is None:
                    continue
                source_bytes = bundle.source_bytes
                source_text = bundle.source_text
                source_index = bundle.source_index
            else:
                source_bytes = self._discovery.read_module_bytes(module)
                if source_bytes is None:
                    source_text = self._discovery.read_module_source(module)
                    if source_text is None:
                        continue
                    source_bytes = source_text.encode("utf-8", errors="replace")
                source_text, source_index = _build_source_index(source_bytes)
            if max_module_bytes is not None and len(source_bytes) > max_module_bytes:
                warnings.append(
                    f"Skipping bytecode for {module.rel_path} (size {len(source_bytes)} bytes)"
                )
                continue
            yield module, source_text, source_index


__all__ = ["DisExtractResult", "DisExtractStep"]
