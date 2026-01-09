"""Classify how functions use extracted data models.

For new code, use ``build_data_model_usage_rows`` with Hamilton materializers.

The Hamilton native module is at:
``codeintel.build.hamilton.native.analytics.data_models``
"""

from __future__ import annotations

import ast
import json
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pyarrow as pa

from codeintel.build.analytics.compute.evidence.collection import EvidenceCollector
from codeintel.build.analytics.utilities.ast import call_name, snippet_from_lines
from codeintel.build.tabular.arrow_ops import iter_rows
from codeintel.build.tabular.compute_masks import FilterExprContext
from codeintel.core.columnar.rows import ColumnarRowBuffer, columnar_buffer_for_table_key
from codeintel.core.data_models.ids import normalize_decimal_id
from codeintel.core.paths import normalize_path
from codeintel.core.query_results import coerce_str
from codeintel.core.schemas.row_models import columns_for_table_key

if TYPE_CHECKING:
    from collections.abc import Sequence

    from codeintel.build.analytics.parsing.ast_cache import FunctionAst
    from codeintel.config.primitives import SnapshotRef

DATA_MODEL_USAGE_TABLE_KEY = "analytics.data_model_usage"


def _columns_for_table(table_key: str) -> list[str]:
    columns = columns_for_table_key(table_key)
    if not columns:
        msg = f"No schema columns registered for {table_key}"
        raise ValueError(msg)
    return list(columns)


DATA_MODEL_USAGE_COLS = _columns_for_table(DATA_MODEL_USAGE_TABLE_KEY)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelInfo:
    """Minimal model identity used during usage classification."""

    model_id: str
    name: str
    module: str


@dataclass
class ModelUsageResult:
    """Usage kinds and evidence collected for a single function."""

    usage_kinds: dict[str, set[str]] = field(default_factory=dict)
    evidence: dict[tuple[str, str], EvidenceCollector] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelUsageArtifacts:
    """Reusable lookup tables for model usage classification."""

    ast_by_goid: dict[int, FunctionAst]
    module_map: dict[str, str]
    param_types: dict[int, dict[str, str]]
    model_index: ModelIndex
    subsystem_map: dict[str, tuple[str, str]]


@dataclass(frozen=True)
class DataModelUsageInputs:
    """Inputs required to compute data model usage rows."""

    module_map: dict[str, str]
    ast_by_goid: dict[int, FunctionAst]
    models_frame: pa.Table | None = None
    subsystem_modules_frame: pa.Table | None = None
    subsystems_frame: pa.Table | None = None
    function_types_frame: pa.Table | None = None
    missing_goids: set[int] | None = None


@dataclass(frozen=True)
class ModelIndex:
    """Grouping for model lookup tables."""

    by_name: dict[str, list[ModelInfo]]
    by_name_lower: dict[str, list[ModelInfo]]
    by_qualified: dict[str, ModelInfo]
    by_qualified_lower: dict[str, ModelInfo]


ORM_CREATE_METHODS = {"create", "add", "merge", "bulk_create"}
ORM_READ_METHODS = {
    "get",
    "filter",
    "filter_by",
    "all",
    "first",
    "one",
    "one_or_none",
    "scalar_one",
    "scalar_one_or_none",
    "query",
    "select",
    "join",
    "where",
}
ORM_UPDATE_METHODS = {"update", "save", "commit", "refresh", "refresh_from_db"}
ORM_DELETE_METHODS = {"delete", "remove"}
ORM_SERIALIZE_METHODS = {
    "dict",
    "json",
    "model_dump",
    "model_dump_json",
    "asdict",
    "astuple",
    "values",
    "values_list",
    "to_dict",
}
ORM_VALIDATE_METHODS = {"validate", "model_validate", "full_clean"}


def _usage_for_attr(attr_name: str) -> str | None:
    buckets = [
        ("serialize", ORM_SERIALIZE_METHODS),
        ("update", ORM_UPDATE_METHODS),
        ("validate", ORM_VALIDATE_METHODS),
        ("delete", ORM_DELETE_METHODS),
        ("read", ORM_READ_METHODS),
        ("create", ORM_CREATE_METHODS),
    ]
    for kind, group in buckets:
        if attr_name in group:
            return kind
    return None


def _parse_param_types(raw: str | dict[str, object] | None) -> dict[str, str]:
    if raw is None:
        return {}
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
    else:
        parsed = raw
    if not isinstance(parsed, dict):
        return {}
    result: dict[str, str] = {}
    for key, value in parsed.items():
        if value is None:
            continue
        result[str(key)] = str(value)
    return result


def _param_types_from_frame(
    frame: pa.Table | None,
    *,
    repo: str,
    commit: str,
) -> dict[int, dict[str, str]]:
    if frame is None or frame.num_rows == 0:
        return {}
    param_types: dict[int, dict[str, str]] = {}
    for row in _rows_for_snapshot(frame, repo=repo, commit=commit):
        goid_int = normalize_decimal_id(row.get("function_goid_h128"))
        if goid_int is None:
            continue
        extras = row.get("extras")
        if isinstance(extras, Mapping):
            raw_param_types = extras.get("param_types")
        else:
            raw_param_types = row.get("param_types")
        parsed_input = raw_param_types if isinstance(raw_param_types, (str, dict)) else None
        param_types[goid_int] = _parse_param_types(parsed_input)
    return param_types


def _build_model_index(
    models: list[ModelInfo],
) -> ModelIndex:
    simple: dict[str, list[ModelInfo]] = {}
    simple_lower: dict[str, list[ModelInfo]] = {}
    qualified: dict[str, ModelInfo] = {}
    qualified_lower: dict[str, ModelInfo] = {}
    for model in models:
        simple.setdefault(model.name, []).append(model)
        simple_lower.setdefault(model.name.lower(), []).append(model)
        qualified_name = f"{model.module}.{model.name}"
        qualified[qualified_name] = model
        qualified_lower[qualified_name.lower()] = model
    return ModelIndex(
        by_name=simple,
        by_name_lower=simple_lower,
        by_qualified=qualified,
        by_qualified_lower=qualified_lower,
    )


def _resolve_model(name: str, index: ModelIndex) -> ModelInfo | None:
    normalized = name
    if normalized in index.by_qualified:
        return index.by_qualified[normalized]
    if normalized.lower() in index.by_qualified_lower:
        return index.by_qualified_lower[normalized.lower()]
    candidates = index.by_name.get(normalized) or index.by_name.get(normalized.split(".")[-1], [])
    if not candidates:
        candidates = index.by_name_lower.get(normalized.lower()) or index.by_name_lower.get(
            normalized.split(".")[-1].lower()
        )
    return candidates[0] if candidates else None


def _resolve_model_from_hint(type_hint: str | None, index: ModelIndex) -> ModelInfo | None:
    if not type_hint:
        return None
    normalized = re.sub(r"[^A-Za-z0-9_\.]", " ", type_hint)
    tokens = normalized.split()
    for token in tokens:
        resolved = _resolve_model(token, index)
        if resolved is not None:
            return resolved
    return None


class ModelUsageVisitor(ast.NodeVisitor):
    """Collect per-model usage kinds and evidence from a function AST."""

    def __init__(
        self,
        *,
        index: ModelIndex,
        rel_path: str,
        lines: list[str],
        max_examples: int,
    ) -> None:
        self._index = index
        self._rel_path = rel_path
        self._lines = lines
        self._max_examples = max_examples
        self._model_vars: dict[str, ModelInfo] = {}
        self.result = ModelUsageResult()

    def _resolve_model_name(self, name: str) -> ModelInfo | None:
        return _resolve_model(name, self._index)

    def _record_model_usage(self, model: ModelInfo, kind: str, lineno: int) -> None:
        self._record_usage(model.model_id, kind, lineno)

    def _handle_constructor_call(self, node: ast.Call) -> None:
        target_name = call_name(node.func)
        resolved_class = self._resolve_model_name(target_name or "")
        if resolved_class is not None:
            self._record_model_usage(resolved_class, "create", node.lineno)

    def _handle_attribute_call(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Attribute):
            return
        base = node.func.value
        attr_name = node.func.attr
        usage = _usage_for_attr(attr_name)
        if usage is not None:
            self._record_usage_for_base(base, usage, node.lineno)
            self._handle_manager_operations(base, attr_name, node, usage)
        self._handle_collection_mutations(base, attr_name, node)
        self._handle_query_targets(base, attr_name, node)

    def _record_usage_for_base(self, base: ast.AST, usage: str, lineno: int) -> None:
        if isinstance(base, ast.Name):
            model = self._model_vars.get(base.id)
            if model is not None:
                self._record_model_usage(model, usage, lineno)

    def _handle_manager_operations(
        self, base: ast.AST, attr_name: str, node: ast.Call, usage: str
    ) -> None:
        if not (isinstance(base, ast.Attribute) and base.attr == "objects"):
            return
        owner_name = call_name(base.value) or ""
        owner_model = self._resolve_model_name(owner_name)
        manager_usage = usage or _usage_for_attr(attr_name)
        if owner_model is not None and manager_usage is not None:
            self._record_model_usage(owner_model, manager_usage, node.lineno)

    def _handle_collection_mutations(self, base: ast.AST, attr_name: str, node: ast.Call) -> None:
        if not (isinstance(base, ast.Name) and attr_name in {"add", "merge", "delete"}):
            return
        for arg in node.args:
            if isinstance(arg, ast.Call):
                call_model = self._resolve_model_name(call_name(arg.func) or "")
                if call_model is not None:
                    self._record_model_usage(call_model, "create", node.lineno)
            if isinstance(arg, ast.Name):
                model = self._model_vars.get(arg.id)
                if model is not None:
                    self._record_model_usage(model, "delete", arg.lineno)

    def _handle_query_targets(self, base: ast.AST, attr_name: str, node: ast.Call) -> None:
        if not (isinstance(base, ast.Name) and attr_name in {"query", "select", "join"}):
            return
        for arg in node.args:
            query_model = self._resolve_model_name(call_name(arg) or "")
            if query_model is not None:
                self._record_model_usage(query_model, "read", node.lineno)

    def _handle_function_call(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Name)
            and node.func.id == "dict"
            and node.args
            and isinstance(node.args[0], ast.Name)
        ):
            model = self._model_vars.get(node.args[0].id)
            if model is not None:
                self._record_model_usage(model, "serialize", node.lineno)

    def _record_arg_models(self, node: ast.Call) -> None:
        for arg in node.args:
            if isinstance(arg, ast.Name) and arg.id in self._model_vars:
                self._record_usage(self._model_vars[arg.id].model_id, "serialize", arg.lineno)
            if isinstance(arg, ast.Call):
                call_model = self._resolve_model_name(call_name(arg.func) or "")
                if call_model is not None:
                    self._record_model_usage(call_model, "create", arg.lineno)

    def seed_parameters(self, param_types: dict[str, str]) -> None:
        """Prime the visitor with models inferred from function parameters."""
        for name, type_hint in param_types.items():
            resolved = _resolve_model_from_hint(type_hint, self._index)
            if resolved is not None:
                self._model_vars[name] = resolved

    def visit_Assign(self, node: ast.Assign) -> None:
        """Capture assignments that construct or bind models."""
        self._maybe_capture_assignment(node.targets, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Handle annotated assignments for potential model creation."""
        targets = [node.target] if node.target is not None else []
        self._maybe_capture_assignment(targets, node.value)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Track creation, serialization, and validation calls on models."""
        self._handle_constructor_call(node)
        self._handle_attribute_call(node)
        self._handle_function_call(node)
        self._record_arg_models(node)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Identify attribute reads and writes on tracked models."""
        if isinstance(node.value, ast.Name):
            model = self._model_vars.get(node.value.id)
            if model is not None:
                if isinstance(node.ctx, ast.Store):
                    self._record_usage(model.model_id, "update", node.lineno)
                else:
                    self._record_usage(model.model_id, "read", node.lineno)
        self.generic_visit(node)

    def visit_Delete(self, node: ast.Delete) -> None:
        """Capture model deletions."""
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in self._model_vars:
                self._record_usage(self._model_vars[target.id].model_id, "delete", target.lineno)
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                base = target.value.id
                if base in self._model_vars:
                    self._record_usage(self._model_vars[base].model_id, "delete", target.lineno)
        self.generic_visit(node)

    def _maybe_capture_assignment(self, targets: Sequence[ast.expr], value: ast.AST | None) -> None:
        if not isinstance(value, ast.Call):
            if isinstance(value, ast.AST):
                self.generic_visit(value)
            return
        resolved, usage_kind = self._resolve_assignment_call(value)
        if resolved is not None:
            self._bind_targets_to_model(targets, resolved)
            self._record_model_usage(resolved, usage_kind, getattr(value, "lineno", 0))
        self._capture_query_results(targets, value)
        self.generic_visit(value)

    def _resolve_assignment_call(self, value: ast.Call) -> tuple[ModelInfo | None, str]:
        call_target = call_name(value.func) or ""
        resolved = self._resolve_model_name(call_target)
        usage_kind = "create"
        if isinstance(value.func, ast.Attribute):
            usage_hint = _usage_for_attr(value.func.attr)
            if usage_hint is not None:
                usage_kind = usage_hint
            if isinstance(value.func.value, ast.Attribute) and value.func.value.attr == "objects":
                owner_name = call_name(value.func.value.value) or ""
                resolved = self._resolve_model_name(owner_name)
        return resolved, usage_kind

    def _bind_targets_to_model(self, targets: Sequence[ast.expr], model: ModelInfo) -> None:
        for target in targets:
            if isinstance(target, ast.Name):
                self._model_vars[target.id] = model

    def _capture_query_results(self, targets: Sequence[ast.expr], value: ast.Call) -> None:
        if not isinstance(value.func, ast.Attribute):
            return
        base_call = value.func.value
        if not isinstance(base_call, ast.Call):
            return
        if not self._is_query_result_accessor(value.func.attr, base_call.func):
            return
        for arg in base_call.args:
            query_model = self._resolve_model_name(call_name(arg) or "")
            if query_model is None:
                continue
            self._bind_targets_to_model(targets, query_model)
            self._record_model_usage(query_model, "read", getattr(value, "lineno", 0))

    @staticmethod
    def _is_query_result_accessor(attr_name: str, func: ast.AST) -> bool:
        result_attrs = {"first", "one", "scalar_one", "first_or_none"}
        is_result_accessor = attr_name in result_attrs
        is_query_call = isinstance(func, ast.Attribute) and func.attr in {"query", "select"}
        return is_result_accessor or is_query_call

    def _record_usage(self, model_id: str, kind: str, lineno: int | None) -> None:
        if lineno is None:
            lineno = 0
        usage = self.result.usage_kinds.setdefault(model_id, set())
        usage.add(kind)
        evidence_key = (model_id, kind)
        collector = self.result.evidence.setdefault(
            evidence_key, EvidenceCollector(max_samples=self._max_examples)
        )
        snippet = snippet_from_lines(self._lines, lineno, lineno)
        collector.add_sample(
            path=self._rel_path,
            line_span=(lineno, lineno),
            snippet=snippet,
            details={"model_id": model_id, "usage": kind},
        )


def _load_models_from_frame(
    frame: pa.Table | None,
    *,
    repo: str,
    commit: str,
) -> list[ModelInfo]:
    if frame is None or frame.num_rows == 0:
        return []
    models: list[ModelInfo] = []
    for row in _rows_for_snapshot(frame, repo=repo, commit=commit):
        model_id = row.get("model_id")
        model_name = row.get("model_name")
        module = row.get("module")
        if model_id is None or model_name is None or module is None:
            continue
        models.append(ModelInfo(model_id=str(model_id), name=str(model_name), module=str(module)))
    return models


def _subsystem_by_module_from_frames(
    subsystem_modules_frame: pa.Table | None,
    subsystems_frame: pa.Table | None,
    *,
    repo: str,
    commit: str,
) -> dict[str, tuple[str, str]]:
    mapping: dict[str, tuple[str, str]] = {}
    if subsystem_modules_frame is None or subsystem_modules_frame.num_rows == 0:
        return mapping
    modules_filtered = _rows_for_snapshot(subsystem_modules_frame, repo=repo, commit=commit)
    subsystems_filtered = (
        _rows_for_snapshot(subsystems_frame, repo=repo, commit=commit)
        if subsystems_frame is not None and subsystems_frame.num_rows > 0
        else None
    )
    name_by_id: dict[str, str] = {}
    if subsystems_filtered is not None:
        for row in subsystems_filtered:
            subsystem_id = row.get("subsystem_id")
            name = row.get("name")
            if subsystem_id is None or name is None:
                continue
            name_by_id[str(subsystem_id)] = coerce_str(name, ctx="subsystems.name")
    for row in modules_filtered:
        module = row.get("module")
        subsystem_id = row.get("subsystem_id")
        if module is None or subsystem_id is None:
            continue
        module_name = coerce_str(module, ctx="subsystem_modules.module")
        subsystem_id_text = coerce_str(subsystem_id, ctx="subsystem_modules.subsystem_id")
        subsystem_name = name_by_id.get(subsystem_id_text, "")
        mapping[module_name] = (subsystem_id_text, subsystem_name)
    return mapping


def _context_for_module(
    module: str | None, subsystem_map: dict[str, tuple[str, str]]
) -> dict[str, object]:
    context: dict[str, object] = {}
    if module:
        context["module"] = module
        subsystem = subsystem_map.get(module)
        if subsystem is not None:
            subsystem_id, subsystem_name = subsystem
            context["subsystem_id"] = subsystem_id
            if subsystem_name:
                context["subsystem_name"] = subsystem_name
    return context


def build_data_model_usage_rows(
    snapshot: SnapshotRef,
    inputs: DataModelUsageInputs,
) -> list[dict[str, object]]:
    """Build data_model_usage rows without writing to database.

    Analyze per-function data model read/write usage patterns and return
    a columnar LazyFrame suitable for materialization via Hamilton.

    Parameters
    ----------
    snapshot
        Repository and commit identifiers.
    inputs
        Bundled inputs including module map, ASTs, and optional frames.

    Returns
    -------
    list[dict[str, object]]
        Row mappings with extras (usage_kinds, evidence, context) and created_at.

    Notes
    -----
    This is the pure compute version that returns a LazyFrame without writing.
    Use with the Hamilton materializers to persist.

    Examples
    --------
    >>> frame = build_data_model_usage_rows(snapshot, ...)
    >>> frame.collect().height > 0
    True
    """
    max_examples_per_usage = 3
    models = _load_models_from_frame(
        inputs.models_frame,
        repo=snapshot.repo,
        commit=snapshot.commit,
    )
    if not models:
        log.info(
            "No data models found for %s@%s; skipping usage analysis",
            snapshot.repo,
            snapshot.commit,
        )
        return []

    model_index = _build_model_index(models)
    subsystem_map = _subsystem_by_module_from_frames(
        inputs.subsystem_modules_frame,
        inputs.subsystems_frame,
        repo=snapshot.repo,
        commit=snapshot.commit,
    )

    missing = inputs.missing_goids or set()
    if missing:
        log.debug(
            "Skipping %d functions without AST spans during model usage analysis",
            len(missing),
        )

    param_types = _param_types_from_frame(
        inputs.function_types_frame,
        repo=snapshot.repo,
        commit=snapshot.commit,
    )
    artifacts = ModelUsageArtifacts(
        ast_by_goid=inputs.ast_by_goid,
        module_map=inputs.module_map,
        param_types=param_types,
        model_index=model_index,
        subsystem_map=subsystem_map,
    )
    buffer = columnar_buffer_for_table_key(DATA_MODEL_USAGE_TABLE_KEY)
    _build_usage_rows(
        artifacts=artifacts,
        repo=snapshot.repo,
        commit=snapshot.commit,
        max_examples_per_usage=max_examples_per_usage,
        buffer=buffer,
    )
    rows: list[dict[str, object]] = []
    for idx in range(buffer.row_count):
        row = {name: buffer.data[name][idx] for name in buffer.columns}
        rows.append(row)
    return rows


def _rows_for_snapshot(
    frame: pa.Table,
    *,
    repo: str,
    commit: str,
) -> list[dict[str, object]]:
    context = FilterExprContext(repo=repo, commit=commit)
    filtered = context.apply(frame)
    return list(iter_rows(filtered))


def _build_usage_rows(
    *,
    artifacts: ModelUsageArtifacts,
    repo: str,
    commit: str,
    max_examples_per_usage: int,
    buffer: ColumnarRowBuffer,
) -> None:
    now = datetime.now(tz=UTC)
    for goid, func_ast in artifacts.ast_by_goid.items():
        rel_path = normalize_path(func_ast.rel_path)
        module = artifacts.module_map.get(rel_path)
        visitor = ModelUsageVisitor(
            index=artifacts.model_index,
            rel_path=rel_path,
            lines=func_ast.lines,
            max_examples=max_examples_per_usage,
        )
        visitor.seed_parameters(artifacts.param_types.get(goid, {}))
        visitor.visit(func_ast.node)
        for model_id, kinds in visitor.result.usage_kinds.items():
            evidence_map: dict[str, list[dict[str, object]]] = {}
            for usage in kinds:
                collector = visitor.result.evidence.get((model_id, usage))
                evidence_map[usage] = collector.to_dicts() if collector is not None else []
            context = _context_for_module(module, artifacts.subsystem_map)
            buffer.append(
                {
                    "repo": repo,
                    "commit": commit,
                    "model_id": model_id,
                    "function_goid_h128": goid,
                    "extras": {
                        "usage_kinds": sorted(kinds),
                        "evidence": dict(evidence_map),
                        "context": dict(context) if context else None,
                    },
                    "created_at": now,
                }
            )
