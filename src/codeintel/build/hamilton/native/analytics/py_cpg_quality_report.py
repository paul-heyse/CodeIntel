"""Python CPG quality report metrics."""

from __future__ import annotations

import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

import pyarrow as pa

from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.patterns import (
    TableTargetContext,
    attach_table_target_template,
    build_single_table_target_spec,
)
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.transforms.table_contract import TableContractSpec
from codeintel.build.tabular.arrow_ops import iter_rows
from codeintel.build.tabular.conversion import tabular_to_arrow_table
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.columnar.rows import empty_table_for_table, table_for_rows
from codeintel.core.data_models.ids import normalize_decimal_id
from codeintel.core.execution.ids import RUN_PREFIX_ANALYTICS, new_run_id
from codeintel.core.serialization.payload import decode_payload

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, InferableTabularInput)

PY_CPG_QUALITY_REPORT_TARGET_NAME = "py_cpg_quality_report"
PY_CPG_QUALITY_REPORT_TABLE_KEY = "analytics.py_cpg_quality_report"

PY_CPG_QUALITY_REPORT_CONTRACT = TableContractSpec(
    table_key=PY_CPG_QUALITY_REPORT_TABLE_KEY,
    domain="analytics",
    target=PY_CPG_QUALITY_REPORT_TARGET_NAME,
    ops_module=None,
    columns_to_pass=(),
    required_cols=(),
    clip_column=None,
    input_name="py_cpg_quality_report__base",
)


@dataclass(frozen=True, slots=True)
class _AnchorRate:
    total: int
    anchored: int

    @property
    def rate(self) -> float | None:
        if self.total == 0:
            return None
        return self.anchored / self.total


@dataclass(frozen=True, slots=True)
class _CfgReachability:
    total_blocks: int
    reachable_blocks: int

    @property
    def rate(self) -> float | None:
        if self.total_blocks == 0:
            return None
        return self.reachable_blocks / self.total_blocks


@dataclass(frozen=True, slots=True)
class _DefuseCoverage:
    event_count: int
    edge_count: int

    @property
    def rate(self) -> float | None:
        if self.event_count == 0:
            return None
        return self.edge_count / self.event_count


@dataclass(frozen=True, slots=True)
class _PyCpgQualityInputsA:
    instructions: InferableTabularInput
    scopes: InferableTabularInput
    blocks: InferableTabularInput
    inspect_objects: InferableTabularInput


@dataclass(frozen=True, slots=True)
class _PyCpgQualityInputsB:
    cfg_edges: InferableTabularInput
    defuse_events: InferableTabularInput
    cpg_edges: InferableTabularInput


def _reader_rows(table: pa.Table) -> Iterable[dict[str, object]]:
    for batch in table.to_batches():
        if batch.num_rows > 0:
            yield from iter_rows(batch)


def _reader_has_columns(table: pa.Table, columns: Iterable[str]) -> bool:
    return set(columns).issubset(table.schema.names)


def _count_rows(table: pa.Table) -> int:
    return table.num_rows


def _anchor_rate(
    table: pa.Table,
    *,
    anchor_column: str,
) -> _AnchorRate:
    total = 0
    anchored = 0
    has_column = anchor_column in table.schema.names
    for batch in table.to_batches():
        total += batch.num_rows
        if not has_column or batch.num_rows == 0:
            continue
        anchored += batch.num_rows - batch.column(anchor_column).null_count
    return _AnchorRate(total=total, anchored=anchored)


def _cfg_reachability(
    blocks_reader: pa.Table,
    edges_reader: pa.Table,
) -> _CfgReachability:
    blocks_by_unit = _blocks_by_unit(blocks_reader)
    edges_by_unit = _edges_by_unit(edges_reader)
    total_blocks = 0
    reachable_blocks = 0
    for code_unit_id, blocks in blocks_by_unit.items():
        total_blocks += len(blocks)
        entry_block = _entry_block_id(blocks)
        if entry_block is None:
            continue
        adjacency = _adjacency_for_unit(edges_by_unit.get(code_unit_id, []))
        reachable_blocks += _reachable_count(entry_block, adjacency)
    return _CfgReachability(total_blocks=total_blocks, reachable_blocks=reachable_blocks)


def _blocks_by_unit(
    blocks_reader: pa.Table,
) -> dict[str, list[tuple[str, int | None, int | None]]]:
    if not _reader_has_columns(
        blocks_reader,
        ("code_unit_id", "block_id", "start_offset", "first_instr_index"),
    ):
        return {}
    blocks_by_unit: dict[str, list[tuple[str, int | None, int | None]]] = {}
    for row in _reader_rows(blocks_reader):
        code_unit_id = row.get("code_unit_id")
        block_id = row.get("block_id")
        if not isinstance(code_unit_id, str) or not isinstance(block_id, str):
            continue
        start_offset = row.get("start_offset")
        first_instr_index = row.get("first_instr_index")
        first_index = first_instr_index if isinstance(first_instr_index, int) else None
        blocks_by_unit.setdefault(code_unit_id, []).append(
            (
                block_id,
                start_offset if isinstance(start_offset, int) else None,
                first_index,
            )
        )
    return blocks_by_unit


def _edges_by_unit(edges_reader: pa.Table) -> dict[str, list[tuple[str, str]]]:
    if not _reader_has_columns(
        edges_reader,
        ("code_unit_id", "src_block_id", "dst_block_id"),
    ):
        return {}
    edges_by_unit: dict[str, list[tuple[str, str]]] = {}
    for row in _reader_rows(edges_reader):
        code_unit_id = row.get("code_unit_id")
        src_block = row.get("src_block_id")
        dst_block = row.get("dst_block_id")
        if not isinstance(code_unit_id, str) or not isinstance(src_block, str):
            continue
        if not isinstance(dst_block, str):
            continue
        edges_by_unit.setdefault(code_unit_id, []).append((src_block, dst_block))
    return edges_by_unit


def _entry_block_id(blocks: list[tuple[str, int | None, int | None]]) -> str | None:
    if not blocks:
        return None
    entry_block, *_ = min(
        blocks,
        key=lambda item: (
            item[1] if item[1] is not None else 1_000_000_000,
            item[2] if isinstance(item[2], int) else 1_000_000_000,
        ),
    )
    return entry_block


def _adjacency_for_unit(edges: list[tuple[str, str]]) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = {}
    for src, dst in edges:
        adjacency.setdefault(src, set()).add(dst)
    return adjacency


def _reachable_count(entry_block: str, adjacency: dict[str, set[str]]) -> int:
    seen: set[str] = set()
    stack = [entry_block]
    while stack:
        block_id = stack.pop()
        if block_id in seen:
            continue
        seen.add(block_id)
        stack.extend(sorted(adjacency.get(block_id, set())))
    return len(seen)


def _defuse_event_count(reader: pa.Table) -> int:
    if not _reader_has_columns(reader, ("event_kind", "space")):
        return 0
    count = 0
    for row in _reader_rows(reader):
        if row.get("event_kind") in {"DEF", "USE"} and row.get("space") in {
            "local",
            "free",
            "global",
        }:
            count += 1
    return count


def _scan_cpg_edges(
    reader: pa.Table,
) -> tuple[int, set[int]]:
    has_edge_kind = "edge_kind" in reader.schema.names
    has_extras = "extras_json" in reader.schema.names
    has_src = "src_cpg_node_id" in reader.schema.names
    if not has_edge_kind:
        return 0, set()
    defuse_edge_count = 0
    inspect_anchor_ids: set[int] = set()
    for row in _reader_rows(reader):
        edge_kind = row.get("edge_kind")
        if edge_kind in {"DEFINES_BINDING", "USES_BINDING"} and has_extras:
            extras = decode_payload(row.get("extras_json"))
            if isinstance(extras, dict) and extras.get("space") in {"local", "free", "global"}:
                defuse_edge_count += 1
        if edge_kind == "INSPECT_ANCHORS_AST" and has_src:
            src_id = normalize_decimal_id(row.get("src_cpg_node_id"))
            if src_id is not None:
                inspect_anchor_ids.add(src_id)
    return defuse_edge_count, inspect_anchor_ids


def _resolve_run_id(env: BuildEnv) -> str:
    run_context = env.run_context
    if run_context is None:
        return new_run_id(RUN_PREFIX_ANALYTICS)
    return run_context.run_id


def py_cpg_quality_report__inputs_a(
    q__core__py_bc_instructions: InferableTabularInput,
    q__core__py_sym_scopes: InferableTabularInput,
    q__core__py_bc_blocks: InferableTabularInput,
    q__core__py_inspect_objects: InferableTabularInput,
) -> _PyCpgQualityInputsA:
    return _PyCpgQualityInputsA(
        instructions=q__core__py_bc_instructions,
        scopes=q__core__py_sym_scopes,
        blocks=q__core__py_bc_blocks,
        inspect_objects=q__core__py_inspect_objects,
    )


def py_cpg_quality_report__inputs_b(
    q__core__py_bc_cfg_edges: InferableTabularInput,
    q__core__py_bc_defuse_events: InferableTabularInput,
    q__graph__cpg_edges: InferableTabularInput,
) -> _PyCpgQualityInputsB:
    return _PyCpgQualityInputsB(
        cfg_edges=q__core__py_bc_cfg_edges,
        defuse_events=q__core__py_bc_defuse_events,
        cpg_edges=q__graph__cpg_edges,
    )


def py_cpg_quality_report__base(
    env: BuildEnv,
    py_cpg_quality_report__inputs_a: _PyCpgQualityInputsA,
    py_cpg_quality_report__inputs_b: _PyCpgQualityInputsB,
) -> pa.Table:
    """Build run-level Python CPG quality metrics.

    Returns
    -------
    pyarrow.Table
        Table containing run-level quality metrics.
    """
    instruction_rate = _anchor_rate(
        tabular_to_arrow_table(py_cpg_quality_report__inputs_a.instructions),
        anchor_column="span_start_byte",
    )
    symtable_rate = _anchor_rate(
        tabular_to_arrow_table(py_cpg_quality_report__inputs_a.scopes),
        anchor_column="anchor_ast_node_id",
    )
    cfg_rate = _cfg_reachability(
        tabular_to_arrow_table(py_cpg_quality_report__inputs_a.blocks),
        tabular_to_arrow_table(py_cpg_quality_report__inputs_b.cfg_edges),
    )
    defuse_event_count = _defuse_event_count(
        tabular_to_arrow_table(py_cpg_quality_report__inputs_b.defuse_events),
    )
    defuse_edge_count, inspect_anchor_ids = _scan_cpg_edges(
        tabular_to_arrow_table(py_cpg_quality_report__inputs_b.cpg_edges),
    )
    defuse_rate = _DefuseCoverage(
        event_count=defuse_event_count,
        edge_count=defuse_edge_count,
    )
    inspect_total = _count_rows(
        tabular_to_arrow_table(py_cpg_quality_report__inputs_a.inspect_objects)
    )
    inspect_rate = _AnchorRate(total=inspect_total, anchored=len(inspect_anchor_ids))

    rows = [
        {
            "repo": env.repo,
            "commit": env.commit,
            "run_id": _resolve_run_id(env),
            "instruction_count": instruction_rate.total,
            "instruction_anchored_count": instruction_rate.anchored,
            "instruction_anchor_rate": instruction_rate.rate,
            "sym_scope_count": symtable_rate.total,
            "sym_scope_anchored_count": symtable_rate.anchored,
            "sym_scope_anchor_rate": symtable_rate.rate,
            "cfg_block_count": cfg_rate.total_blocks,
            "cfg_reachable_block_count": cfg_rate.reachable_blocks,
            "cfg_reachability_rate": cfg_rate.rate,
            "defuse_event_count": defuse_rate.event_count,
            "defuse_edge_count": defuse_rate.edge_count,
            "defuse_resolution_rate": defuse_rate.rate,
            "inspect_object_count": inspect_rate.total,
            "inspect_anchored_count": inspect_rate.anchored,
            "inspect_anchor_rate": inspect_rate.rate,
            "created_at": datetime.now(UTC),
        }
    ]
    if not rows:
        return empty_table_for_table(PY_CPG_QUALITY_REPORT_TABLE_KEY)
    table, _ = table_for_rows(PY_CPG_QUALITY_REPORT_TABLE_KEY, rows)
    return table


_MODULE = sys.modules[__name__]
_PY_CPG_QUALITY_REPORT_TABLE_TARGET_SPEC = build_single_table_target_spec(
    context=TableTargetContext.from_contract(
        contract=PY_CPG_QUALITY_REPORT_CONTRACT,
        input_type=pa.Table,
    )
)
attach_table_target_template(_MODULE, spec=_PY_CPG_QUALITY_REPORT_TABLE_TARGET_SPEC)
py_cpg_quality_report__table = _MODULE.py_cpg_quality_report__table
py_cpg_quality_report__table_materializations = (
    _MODULE.py_cpg_quality_report__table_materializations
)
t__py_cpg_quality_report = _MODULE.t__py_cpg_quality_report


__all__ = [
    "py_cpg_quality_report__base",
    "py_cpg_quality_report__table",
    "t__py_cpg_quality_report",
]
