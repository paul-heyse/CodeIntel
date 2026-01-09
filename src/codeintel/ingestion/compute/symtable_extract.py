"""Symtable extraction step with port injection."""

from __future__ import annotations

import ast
import hashlib
import io
import logging
import symtable
import tokenize
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from codeintel.build.hamilton.execution_result import ExecutionResult
from codeintel.build.hamilton.native.options.ingestion import SymtableExtractOptions
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
from codeintel.ingestion.infrastructure.ast_facts import (
    AstSpan,
    ast_node_id,
    ast_span_for_node,
)
from codeintel.ingestion.infrastructure.cst_utils import LineIndexedSource

if TYPE_CHECKING:
    from collections.abc import Sequence
    from symtable import Symbol, SymbolTable

    import pyarrow as pa

    from codeintel.ingestion.infrastructure.py_frontend import PyFrontend
    from codeintel.ingestion.ports.discovery import ModuleDiscoveryPort, ModuleRecord
    from codeintel.ingestion.ports.storage import IngestStoragePort

LOG = logging.getLogger(__name__)
_INT32_MIN = -(2**31)
_INT32_MAX = 2**31 - 1


def _safe_int32(value: object) -> int | None:
    if not isinstance(value, int) or isinstance(value, bool):
        return None
    if value < _INT32_MIN or value > _INT32_MAX:
        return None
    return value


PY_SYM_SCOPES_TABLE_KEY = "core.py_sym_scopes"
PY_SYM_SYMBOLS_TABLE_KEY = "core.py_sym_symbols"
PY_SYM_SCOPE_EDGES_TABLE_KEY = "core.py_sym_scope_edges"
PY_SYM_NAMESPACE_EDGES_TABLE_KEY = "core.py_sym_namespace_edges"
PY_SYM_FUNCTION_PARTITIONS_TABLE_KEY = "core.py_sym_function_partitions"
PY_SYM_BINDINGS_TABLE_KEY = "core.py_sym_bindings"
PY_SYM_RESOLUTION_EDGES_TABLE_KEY = "core.py_sym_resolution_edges"


@dataclass(frozen=True)
class SymtableExtractResult:
    """Result bundle for symtable extraction."""

    result: ExecutionResult
    scope_rows: ColumnarRows = field(default_factory=dict)
    symbol_rows: ColumnarRows = field(default_factory=dict)
    scope_edge_rows: ColumnarRows = field(default_factory=dict)
    namespace_edge_rows: ColumnarRows = field(default_factory=dict)
    function_partition_rows: ColumnarRows = field(default_factory=dict)
    binding_rows: ColumnarRows = field(default_factory=dict)
    resolution_edge_rows: ColumnarRows = field(default_factory=dict)
    scope_rows_reader: pa.Table = field(
        default_factory=lambda: empty_table_for_table(PY_SYM_SCOPES_TABLE_KEY)
    )
    symbol_rows_reader: pa.Table = field(
        default_factory=lambda: empty_table_for_table(PY_SYM_SYMBOLS_TABLE_KEY)
    )
    scope_edge_rows_reader: pa.Table = field(
        default_factory=lambda: empty_table_for_table(PY_SYM_SCOPE_EDGES_TABLE_KEY)
    )
    namespace_edge_rows_reader: pa.Table = field(
        default_factory=lambda: empty_table_for_table(PY_SYM_NAMESPACE_EDGES_TABLE_KEY)
    )
    function_partition_rows_reader: pa.Table = field(
        default_factory=lambda: empty_table_for_table(PY_SYM_FUNCTION_PARTITIONS_TABLE_KEY)
    )
    binding_rows_reader: pa.Table = field(
        default_factory=lambda: empty_table_for_table(PY_SYM_BINDINGS_TABLE_KEY)
    )
    resolution_edge_rows_reader: pa.Table = field(
        default_factory=lambda: empty_table_for_table(PY_SYM_RESOLUTION_EDGES_TABLE_KEY)
    )
    scope_row_count: int = 0
    symbol_row_count: int = 0
    scope_edge_row_count: int = 0
    namespace_edge_row_count: int = 0
    function_partition_row_count: int = 0
    binding_row_count: int = 0
    resolution_edge_row_count: int = 0


@dataclass(frozen=True, slots=True)
class _SymtableTables:
    scope_rows_table: pa.Table
    symbol_rows_table: pa.Table
    scope_edge_rows_table: pa.Table
    namespace_edge_rows_table: pa.Table
    function_partition_rows_table: pa.Table
    binding_rows_table: pa.Table
    resolution_edge_rows_table: pa.Table

    def as_mapping(self) -> dict[str, pa.Table]:
        return {
            PY_SYM_SCOPES_TABLE_KEY: self.scope_rows_table,
            PY_SYM_SYMBOLS_TABLE_KEY: self.symbol_rows_table,
            PY_SYM_SCOPE_EDGES_TABLE_KEY: self.scope_edge_rows_table,
            PY_SYM_NAMESPACE_EDGES_TABLE_KEY: self.namespace_edge_rows_table,
            PY_SYM_FUNCTION_PARTITIONS_TABLE_KEY: self.function_partition_rows_table,
            PY_SYM_BINDINGS_TABLE_KEY: self.binding_rows_table,
            PY_SYM_RESOLUTION_EDGES_TABLE_KEY: self.resolution_edge_rows_table,
        }


@dataclass(frozen=True, slots=True)
class _ScopeInfo:
    table: SymbolTable
    scope_id: str
    parent_scope_id: str | None
    scope_type: str
    scope_name: str
    qualpath: str
    lineno: int | None
    local_id: int | None
    is_nested: bool | None
    is_optimized: bool | None
    has_children: bool | None


@dataclass(frozen=True, slots=True)
class _ScopeIdContext:
    repo: str
    commit: str
    rel_path: str


@dataclass(frozen=True, slots=True)
class _AstAnchor:
    node_id: str
    span: AstSpan


@dataclass(frozen=True, slots=True)
class _ModuleContext:
    repo: str
    commit: str
    module: ModuleRecord
    source_text: str
    source_index: LineIndexedSource
    ast_tree: ast.AST | None


@dataclass(frozen=True, slots=True)
class _SymtableCollectors:
    scopes: ColumnarBatchCollector
    symbols: ColumnarBatchCollector
    scope_edges: ColumnarBatchCollector
    namespace_edges: ColumnarBatchCollector
    function_partitions: ColumnarBatchCollector
    bindings: ColumnarBatchCollector
    resolution_edges: ColumnarBatchCollector


@dataclass(frozen=True, slots=True)
class _ScopeAnchorBundle:
    anchors: dict[str, _AstAnchor | None]
    confidence: dict[str, float | None]
    reason: dict[str, str | None]


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


def _build_symtable_collectors(options: SymtableExtractOptions) -> _SymtableCollectors:
    return _SymtableCollectors(
        scopes=columnar_batch_collector_for_table_key(
            PY_SYM_SCOPES_TABLE_KEY,
            batch_size=options.batch_size,
        ),
        symbols=columnar_batch_collector_for_table_key(
            PY_SYM_SYMBOLS_TABLE_KEY,
            batch_size=options.batch_size,
        ),
        scope_edges=columnar_batch_collector_for_table_key(
            PY_SYM_SCOPE_EDGES_TABLE_KEY,
            batch_size=options.batch_size,
        ),
        namespace_edges=columnar_batch_collector_for_table_key(
            PY_SYM_NAMESPACE_EDGES_TABLE_KEY,
            batch_size=options.batch_size,
        ),
        function_partitions=columnar_batch_collector_for_table_key(
            PY_SYM_FUNCTION_PARTITIONS_TABLE_KEY,
            batch_size=options.batch_size,
        ),
        bindings=columnar_batch_collector_for_table_key(
            PY_SYM_BINDINGS_TABLE_KEY,
            batch_size=options.batch_size,
        ),
        resolution_edges=columnar_batch_collector_for_table_key(
            PY_SYM_RESOLUTION_EDGES_TABLE_KEY,
            batch_size=options.batch_size,
        ),
    )


def _flush_symtable_collectors(collectors: _SymtableCollectors) -> None:
    collectors.scopes.flush()
    collectors.symbols.flush()
    collectors.scope_edges.flush()
    collectors.namespace_edges.flush()
    collectors.function_partitions.flush()
    collectors.bindings.flush()
    collectors.resolution_edges.flush()


def _parse_symtable(
    context: _ModuleContext,
    *,
    warnings: list[str],
) -> SymbolTable | None:
    try:
        return symtable.symtable(context.source_text, context.module.rel_path, "exec")
    except (SyntaxError, ValueError, TypeError) as exc:
        message = f"Symtable parse failed for {context.module.rel_path}: {exc}"
        warnings.append(message)
        LOG.warning("%s", message)
        return None


def _parse_ast(
    context: _ModuleContext,
    *,
    warnings: list[str],
) -> ast.AST | None:
    if context.ast_tree is not None:
        return context.ast_tree
    try:
        return ast.parse(
            context.source_text,
            filename=str(context.module.file_path),
            type_comments=True,
        )
    except (SyntaxError, ValueError, TypeError) as exc:
        message = f"AST parse failed for {context.module.rel_path}: {exc}"
        warnings.append(message)
        LOG.warning("%s", message)
        return None


def _resolve_scope_anchors(
    scope_index: dict[int, _ScopeInfo],
    *,
    module_anchor: _AstAnchor | None,
    anchors: dict[tuple[str, str, int], _AstAnchor],
) -> _ScopeAnchorBundle:
    scope_anchor: dict[str, _AstAnchor | None] = {}
    anchor_confidence: dict[str, float | None] = {}
    anchor_reason: dict[str, str | None] = {}
    for info in scope_index.values():
        anchor, confidence, reason = _scope_anchor(
            info,
            module_anchor=module_anchor,
            anchors=anchors,
        )
        scope_anchor[info.scope_id] = anchor
        anchor_confidence[info.scope_id] = confidence
        anchor_reason[info.scope_id] = reason
    for info in scope_index.values():
        if scope_anchor[info.scope_id] is not None:
            continue
        parent_id = info.parent_scope_id
        if parent_id is None:
            continue
        parent_anchor = scope_anchor.get(parent_id)
        if parent_anchor is None:
            continue
        scope_anchor[info.scope_id] = parent_anchor
        anchor_confidence[info.scope_id] = 0.3
        anchor_reason[info.scope_id] = "parent_scope_fallback"
    return _ScopeAnchorBundle(
        anchors=scope_anchor,
        confidence=anchor_confidence,
        reason=anchor_reason,
    )


def _scope_key(scope_name: str, ordinal: int) -> str:
    if ordinal <= 1 and not scope_name.startswith("<"):
        return scope_name
    return f"{scope_name}#{ordinal}"


def _scope_id(
    context: _ScopeIdContext,
    *,
    scope_type: str,
    qualpath: str,
    lineno: int | None,
    ordinal: int,
) -> str:
    return _stable_id(
        "sym_scope",
        context.repo,
        context.commit,
        context.rel_path,
        scope_type,
        qualpath,
        lineno,
        ordinal,
    )


def _symbol_row_id(scope_id: str, name: str) -> str:
    return f"{scope_id}:SYM:{name}"


def _binding_id(scope_id: str, name: str) -> str:
    return f"{scope_id}:BIND:{name}"


def _edge_id(kind: str, src: str, dst: str) -> str:
    return _stable_id("sym_edge", kind, src, dst)


def _build_scope_index(
    table: SymbolTable,
    *,
    scope_id_context: _ScopeIdContext,
    module_name: str,
) -> tuple[dict[int, _ScopeInfo], list[tuple[str, str]]]:
    scopes: dict[int, _ScopeInfo] = {}
    edges: list[tuple[str, str]] = []

    def _walk(
        current: SymbolTable,
        parent_scope_id: str | None,
        parent_qualpath: str,
        name_counts: dict[str, int],
    ) -> None:
        scope_type = current.get_type().name
        raw_name = current.get_name()
        scope_name = module_name if scope_type == "MODULE" else raw_name
        scope_name = scope_name or "<anonymous>"
        lineno = current.get_lineno()
        local_id = _safe_int32(current.get_id())
        is_nested = current.is_nested()
        is_optimized = current.is_optimized()
        has_children = current.has_children()

        ordinal = name_counts.get(scope_name, 0) + 1
        name_counts[scope_name] = ordinal

        qualpath = module_name if scope_type == "MODULE" else parent_qualpath
        if scope_type != "MODULE":
            qualpath = f"{parent_qualpath}::{_scope_key(scope_name, ordinal)}"
        scope_id = _scope_id(
            scope_id_context,
            scope_type=scope_type,
            qualpath=qualpath,
            lineno=lineno,
            ordinal=ordinal,
        )
        scopes[id(current)] = _ScopeInfo(
            table=current,
            scope_id=scope_id,
            parent_scope_id=parent_scope_id,
            scope_type=scope_type,
            scope_name=scope_name,
            qualpath=qualpath,
            lineno=lineno,
            local_id=local_id,
            is_nested=is_nested,
            is_optimized=is_optimized,
            has_children=has_children,
        )
        if parent_scope_id is not None:
            edges.append((parent_scope_id, scope_id))
        child_counts: dict[str, int] = {}
        for child in current.get_children():
            _walk(child, scope_id, qualpath, child_counts)

    _walk(table, None, module_name, {})
    return scopes, edges


def _build_ast_anchor_index(
    tree: ast.AST,
    source_index: LineIndexedSource,
    *,
    rel_path: str,
) -> tuple[_AstAnchor | None, dict[tuple[str, str, int], _AstAnchor]]:
    module_anchor: _AstAnchor | None = None
    anchors: dict[tuple[str, str, int], _AstAnchor] = {}
    for node in ast.walk(tree):
        module_candidate = _module_anchor_from_node(node, source_index, rel_path)
        if module_candidate is not None:
            module_anchor = module_candidate
            continue
        anchor_item = _named_anchor_from_node(node, source_index, rel_path)
        if anchor_item is not None:
            key, anchor = anchor_item
            anchors[key] = anchor
    return module_anchor, anchors


def _module_anchor_from_node(
    node: ast.AST,
    source_index: LineIndexedSource,
    rel_path: str,
) -> _AstAnchor | None:
    if not isinstance(node, ast.Module):
        return None
    span = ast_span_for_node(node, source_index)
    if span is None:
        return None
    node_id = ast_node_id(rel_path, "Module", span)
    return _AstAnchor(node_id=node_id, span=span)


def _named_anchor_from_node(
    node: ast.AST,
    source_index: LineIndexedSource,
    rel_path: str,
) -> tuple[tuple[str, str, int], _AstAnchor] | None:
    anchor = _anchor_for_definition(node, source_index, rel_path)
    if anchor is not None:
        return anchor
    return _anchor_for_typing_node(node, source_index, rel_path)


def _anchor_for_definition(
    node: ast.AST,
    source_index: LineIndexedSource,
    rel_path: str,
) -> tuple[tuple[str, str, int], _AstAnchor] | None:
    if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    span = ast_span_for_node(node, source_index)
    if span is None:
        return None
    node_kind = type(node).__name__
    node_id = ast_node_id(rel_path, node_kind, span)
    anchor = _AstAnchor(node_id=node_id, span=span)
    return (node_kind, node.name, span.start_line), anchor


def _anchor_for_typing_node(
    node: ast.AST,
    source_index: LineIndexedSource,
    rel_path: str,
) -> tuple[tuple[str, str, int], _AstAnchor] | None:
    alias_anchor = _anchor_for_type_alias(node, source_index, rel_path)
    if alias_anchor is not None:
        return alias_anchor
    return _anchor_for_type_param(node, source_index, rel_path)


def _anchor_for_type_alias(
    node: ast.AST,
    source_index: LineIndexedSource,
    rel_path: str,
) -> tuple[tuple[str, str, int], _AstAnchor] | None:
    if not isinstance(node, ast.TypeAlias):
        return None
    name_node = node.name
    name = name_node.id if isinstance(name_node, ast.Name) else None
    if name is None:
        return None
    span = ast_span_for_node(node, source_index)
    if span is None:
        return None
    node_id = ast_node_id(rel_path, "TypeAlias", span)
    return ("TypeAlias", name, span.start_line), _AstAnchor(node_id=node_id, span=span)


def _anchor_for_type_param(
    node: ast.AST,
    source_index: LineIndexedSource,
    rel_path: str,
) -> tuple[tuple[str, str, int], _AstAnchor] | None:
    if not isinstance(node, (ast.TypeVar, ast.TypeVarTuple, ast.ParamSpec)):
        return None
    name = node.name if isinstance(node.name, str) else None
    if name is None:
        return None
    span = ast_span_for_node(node, source_index)
    if span is None:
        return None
    node_kind = type(node).__name__
    node_id = ast_node_id(rel_path, node_kind, span)
    return (node_kind, name, span.start_line), _AstAnchor(node_id=node_id, span=span)


def _scope_anchor(
    scope: _ScopeInfo,
    *,
    module_anchor: _AstAnchor | None,
    anchors: dict[tuple[str, str, int], _AstAnchor],
) -> tuple[_AstAnchor | None, float | None, str | None]:
    if scope.scope_type == "MODULE":
        return _anchor_for_module_scope(module_anchor)
    lineno = scope.lineno
    if lineno is None:
        return None, None, None
    normalized_line = max(lineno - 1, 0)
    handler = _SCOPE_ANCHOR_HANDLERS.get(scope.scope_type)
    if handler is None:
        return None, None, None
    return handler(scope, anchors, normalized_line)


def _anchor_for_module_scope(
    module_anchor: _AstAnchor | None,
) -> tuple[_AstAnchor | None, float | None, str | None]:
    if module_anchor is None:
        return None, None, None
    return module_anchor, 1.0, "module"


def _anchor_for_function_scope(
    scope: _ScopeInfo,
    anchors: dict[tuple[str, str, int], _AstAnchor],
    normalized_line: int,
) -> tuple[_AstAnchor | None, float | None, str | None]:
    anchor = anchors.get(("FunctionDef", scope.scope_name, normalized_line))
    if anchor is None:
        anchor = anchors.get(("AsyncFunctionDef", scope.scope_name, normalized_line))
    if anchor is None:
        return None, None, None
    return anchor, 1.0, "function"


def _anchor_for_class_scope(
    scope: _ScopeInfo,
    anchors: dict[tuple[str, str, int], _AstAnchor],
    normalized_line: int,
) -> tuple[_AstAnchor | None, float | None, str | None]:
    anchor = anchors.get(("ClassDef", scope.scope_name, normalized_line))
    if anchor is None:
        return None, None, None
    return anchor, 1.0, "class"


def _anchor_for_type_alias_scope(
    scope: _ScopeInfo,
    anchors: dict[tuple[str, str, int], _AstAnchor],
    normalized_line: int,
) -> tuple[_AstAnchor | None, float | None, str | None]:
    anchor = anchors.get(("TypeAlias", scope.scope_name, normalized_line))
    if anchor is None:
        return None, None, None
    return anchor, 0.9, "type_alias"


def _anchor_for_type_variable_scope(
    scope: _ScopeInfo,
    anchors: dict[tuple[str, str, int], _AstAnchor],
    normalized_line: int,
) -> tuple[_AstAnchor | None, float | None, str | None]:
    for kind in ("TypeVar", "TypeVarTuple", "ParamSpec"):
        anchor = anchors.get((kind, scope.scope_name, normalized_line))
        if anchor is not None:
            return anchor, 0.9, "type_variable"
    return None, None, None


def _anchor_for_type_parameters_scope(
    scope: _ScopeInfo,
    anchors: dict[tuple[str, str, int], _AstAnchor],
    normalized_line: int,
) -> tuple[_AstAnchor | None, float | None, str | None]:
    anchor = anchors.get(("FunctionDef", scope.scope_name, normalized_line))
    if anchor is None:
        anchor = anchors.get(("AsyncFunctionDef", scope.scope_name, normalized_line))
    if anchor is None:
        anchor = anchors.get(("ClassDef", scope.scope_name, normalized_line))
    if anchor is None:
        anchor = anchors.get(("TypeAlias", scope.scope_name, normalized_line))
    if anchor is None:
        return None, None, None
    return anchor, 0.7, "type_parameters_owner"


def _anchor_for_annotation_scope(
    scope: _ScopeInfo,
    anchors: dict[tuple[str, str, int], _AstAnchor],
    normalized_line: int,
) -> tuple[_AstAnchor | None, float | None, str | None]:
    anchor = anchors.get(("FunctionDef", scope.scope_name, normalized_line))
    if anchor is None:
        anchor = anchors.get(("AsyncFunctionDef", scope.scope_name, normalized_line))
    if anchor is None:
        anchor = anchors.get(("ClassDef", scope.scope_name, normalized_line))
    if anchor is None:
        anchor = anchors.get(("TypeAlias", scope.scope_name, normalized_line))
    if anchor is None:
        return None, None, None
    return anchor, 0.6, "annotation_owner"


_ScopeAnchorHandler = Callable[
    [_ScopeInfo, dict[tuple[str, str, int], _AstAnchor], int],
    tuple[_AstAnchor | None, float | None, str | None],
]


_SCOPE_ANCHOR_HANDLERS: dict[str, _ScopeAnchorHandler] = {
    "ANNOTATION": _anchor_for_annotation_scope,
    "FUNCTION": _anchor_for_function_scope,
    "CLASS": _anchor_for_class_scope,
    "TYPE_ALIAS": _anchor_for_type_alias_scope,
    "TYPE_VARIABLE": _anchor_for_type_variable_scope,
    "TYPE_PARAMETERS": _anchor_for_type_parameters_scope,
}


def _binding_kind(symbol: Symbol) -> tuple[str, str, bool]:
    if symbol.is_nonlocal():
        binding_kind = "nonlocal_ref"
        scoping_class = "NONLOCAL"
        declared_here = False
    elif symbol.is_declared_global() or symbol.is_global():
        binding_kind = "global_ref"
        scoping_class = "GLOBAL"
        declared_here = False
    elif symbol.is_free():
        binding_kind = "free_ref"
        scoping_class = "FREE"
        declared_here = False
    elif symbol.is_parameter():
        binding_kind = "param"
        scoping_class = "LOCAL"
        declared_here = True
    elif symbol.is_imported():
        binding_kind = "import"
        scoping_class = "LOCAL"
        declared_here = True
    elif symbol.is_namespace():
        binding_kind = "namespace"
        scoping_class = "LOCAL"
        declared_here = True
    elif symbol.is_annotated() and not symbol.is_assigned():
        binding_kind = "annot_only"
        scoping_class = "LOCAL"
        declared_here = True
    elif symbol.is_local() or symbol.is_assigned():
        binding_kind = "local"
        scoping_class = "LOCAL"
        declared_here = True
    else:
        binding_kind = "unknown"
        scoping_class = "UNKNOWN"
        declared_here = False
    return binding_kind, scoping_class, declared_here


def _symtable_names(table: SymbolTable, attr_name: str) -> list[str]:
    getter = getattr(table, attr_name, None)
    if callable(getter):
        values = getter()
        if not isinstance(values, Iterable):
            return []
        if isinstance(values, list):
            return list(values)
        return list(values)
    return []


def _scope_line(scope: _ScopeInfo) -> int | None:
    if scope.lineno is None:
        return None
    return max(scope.lineno - 1, 0)


def _ancestor_scopes(
    scope: _ScopeInfo,
    scope_by_id: dict[str, _ScopeInfo],
) -> list[_ScopeInfo]:
    ancestors: list[_ScopeInfo] = []
    current_id = scope.parent_scope_id
    while current_id is not None:
        ancestor = scope_by_id.get(current_id)
        if ancestor is None:
            break
        ancestors.append(ancestor)
        current_id = ancestor.parent_scope_id
    return ancestors


def _resolve_global_binding(
    ancestors: list[_ScopeInfo],
    *,
    name: str,
) -> tuple[str | None, str, float, str]:
    module_scope = next(
        (ancestor for ancestor in ancestors if ancestor.scope_type == "MODULE"), None
    )
    if module_scope is None:
        return None, "GLOBAL", 0.0, "module_missing"
    target = _binding_id(module_scope.scope_id, name)
    return target, "GLOBAL", 1.0, "resolved_global"


def _resolve_nonlocal_binding(
    ancestors: list[_ScopeInfo],
    *,
    name: str,
    binding_by_scope: dict[str, dict[str, dict[str, object]]],
) -> tuple[str | None, str, float, str]:
    for ancestor in ancestors:
        if ancestor.scope_type != "FUNCTION":
            continue
        binding = binding_by_scope.get(ancestor.scope_id, {}).get(name)
        if binding and binding.get("declared_here"):
            return _binding_id(ancestor.scope_id, name), "NONLOCAL", 1.0, "resolved_nonlocal"
    return None, "NONLOCAL", 0.0, "unresolved_nonlocal"


def _resolve_free_binding(
    ancestors: list[_ScopeInfo],
    *,
    name: str,
    binding_by_scope: dict[str, dict[str, dict[str, object]]],
) -> tuple[str | None, str, float, str]:
    for ancestor in ancestors:
        binding = binding_by_scope.get(ancestor.scope_id, {}).get(name)
        if binding and binding.get("declared_here"):
            confidence = 1.0 if ancestor.scope_type != "MODULE" else 0.7
            reason = "resolved_free" if ancestor.scope_type != "MODULE" else "module_fallback"
            return _binding_id(ancestor.scope_id, name), "FREE", confidence, reason
    return None, "FREE", 0.0, "unresolved_free"


def _resolution_target(
    *,
    scope: _ScopeInfo,
    name: str,
    binding_by_scope: dict[str, dict[str, dict[str, object]]],
    scope_by_id: dict[str, _ScopeInfo],
) -> tuple[str | None, str, float, str]:
    if scope.parent_scope_id is None:
        return None, "UNKNOWN", 0.0, "no_parent"
    ancestors = _ancestor_scopes(scope, scope_by_id)
    binding_kind = binding_by_scope.get(scope.scope_id, {}).get(name, {}).get("binding_kind")
    if binding_kind == "global_ref":
        return _resolve_global_binding(ancestors, name=name)
    if binding_kind == "nonlocal_ref":
        return _resolve_nonlocal_binding(
            ancestors,
            name=name,
            binding_by_scope=binding_by_scope,
        )
    if binding_kind == "free_ref":
        return _resolve_free_binding(
            ancestors,
            name=name,
            binding_by_scope=binding_by_scope,
        )
    return None, "UNKNOWN", 0.0, "not_ref_binding"


def _append_scope_rows(
    collectors: _SymtableCollectors,
    *,
    context: _ModuleContext,
    scope_index: dict[int, _ScopeInfo],
    anchors: _ScopeAnchorBundle,
) -> None:
    for info in scope_index.values():
        anchor = anchors.anchors.get(info.scope_id)
        collectors.scopes.append(
            {
                "repo": context.repo,
                "commit": context.commit,
                "rel_path": context.module.rel_path,
                "scope_id": info.scope_id,
                "scope_local_id": info.local_id,
                "parent_scope_id": info.parent_scope_id,
                "scope_type": info.scope_type,
                "scope_name": info.scope_name,
                "qualpath": info.qualpath,
                "lineno": _scope_line(info),
                "is_nested": info.is_nested,
                "is_optimized": info.is_optimized,
                "has_children": info.has_children,
                "anchor_ast_node_id": anchor.node_id if anchor else None,
                "span_start_byte": anchor.span.start_byte if anchor else None,
                "span_end_byte": anchor.span.end_byte if anchor else None,
                "anchor_confidence": anchors.confidence.get(info.scope_id),
                "anchor_reason": anchors.reason.get(info.scope_id),
            }
        )


def _append_scope_edges(
    collectors: _SymtableCollectors,
    *,
    context: _ModuleContext,
    scope_edges: list[tuple[str, str]],
) -> None:
    for parent_id, child_id in scope_edges:
        collectors.scope_edges.append(
            {
                "repo": context.repo,
                "commit": context.commit,
                "rel_path": context.module.rel_path,
                "parent_scope_id": parent_id,
                "child_scope_id": child_id,
                "edge_kind": "PARENT_SCOPE",
            }
        )


def _append_symbols_and_bindings(
    collectors: _SymtableCollectors,
    *,
    context: _ModuleContext,
    scope_index: dict[int, _ScopeInfo],
) -> dict[str, dict[str, dict[str, object]]]:
    binding_by_scope: dict[str, dict[str, dict[str, object]]] = {}
    for info in scope_index.values():
        table = info.table
        scope_id = info.scope_id
        for symbol in table.get_symbols():
            name = symbol.get_name()
            symbol_id = _symbol_row_id(scope_id, name)
            collectors.symbols.append(
                {
                    "repo": context.repo,
                    "commit": context.commit,
                    "rel_path": context.module.rel_path,
                    "scope_id": scope_id,
                    "symbol_row_id": symbol_id,
                    "name": name,
                    "is_referenced": symbol.is_referenced(),
                    "is_assigned": symbol.is_assigned(),
                    "is_imported": symbol.is_imported(),
                    "is_annotated": symbol.is_annotated(),
                    "is_parameter": symbol.is_parameter(),
                    "is_local": symbol.is_local(),
                    "is_global": symbol.is_global(),
                    "is_declared_global": symbol.is_declared_global(),
                    "is_nonlocal": symbol.is_nonlocal(),
                    "is_free": symbol.is_free(),
                    "is_namespace": symbol.is_namespace(),
                    "namespace_count": len(symbol.get_namespaces())
                    if symbol.is_namespace()
                    else None,
                }
            )
            binding_kind, scoping_class, declared_here = _binding_kind(symbol)
            binding_row: dict[str, object] = {
                "repo": context.repo,
                "commit": context.commit,
                "rel_path": context.module.rel_path,
                "binding_id": _binding_id(scope_id, name),
                "scope_id": scope_id,
                "name": name,
                "binding_kind": binding_kind,
                "declared_here": declared_here,
                "referenced_here": symbol.is_referenced(),
                "assigned_here": symbol.is_assigned(),
                "annotated_here": symbol.is_annotated(),
                "scoping_class": scoping_class,
            }
            collectors.bindings.append(binding_row)
            binding_by_scope.setdefault(scope_id, {})[name] = binding_row

            if symbol.is_namespace():
                namespaces = symbol.get_namespaces()
                for namespace in namespaces:
                    child_info = scope_index.get(id(namespace))
                    if child_info is None:
                        continue
                    collectors.namespace_edges.append(
                        {
                            "repo": context.repo,
                            "commit": context.commit,
                            "rel_path": context.module.rel_path,
                            "scope_id": scope_id,
                            "symbol_row_id": symbol_id,
                            "name": name,
                            "child_scope_id": child_info.scope_id,
                            "edge_kind": "BINDS_NAMESPACE",
                            "is_ambiguous": len(namespaces) != 1,
                        }
                    )

        if info.scope_type == "FUNCTION":
            collectors.function_partitions.append(
                {
                    "repo": context.repo,
                    "commit": context.commit,
                    "rel_path": context.module.rel_path,
                    "scope_id": scope_id,
                    "parameters": _symtable_names(table, "get_parameters"),
                    "locals": _symtable_names(table, "get_locals"),
                    "globals": _symtable_names(table, "get_globals"),
                    "nonlocals": _symtable_names(table, "get_nonlocals"),
                    "frees": _symtable_names(table, "get_frees"),
                }
            )
    return binding_by_scope


def _append_resolution_edges(
    collectors: _SymtableCollectors,
    *,
    context: _ModuleContext,
    binding_by_scope: dict[str, dict[str, dict[str, object]]],
    scope_by_id: dict[str, _ScopeInfo],
) -> None:
    for scope_id, bindings in binding_by_scope.items():
        info = scope_by_id.get(scope_id)
        if info is None:
            continue
        for name, binding in bindings.items():
            if binding.get("binding_kind") not in {"global_ref", "nonlocal_ref", "free_ref"}:
                continue
            target, kind, confidence, reason = _resolution_target(
                scope=info,
                name=name,
                binding_by_scope=binding_by_scope,
                scope_by_id=scope_by_id,
            )
            binding_id = str(binding["binding_id"])
            dst_binding_id = f"{scope_id}:unknown" if target is None else target
            edge_id = _edge_id(kind, binding_id, dst_binding_id)
            collectors.resolution_edges.append(
                {
                    "repo": context.repo,
                    "commit": context.commit,
                    "rel_path": context.module.rel_path,
                    "edge_id": edge_id,
                    "src_binding_id": binding_id,
                    "dst_binding_id": dst_binding_id,
                    "kind": kind,
                    "confidence": confidence,
                    "reason": reason,
                }
            )


def _process_module(
    context: _ModuleContext,
    collectors: _SymtableCollectors,
    *,
    warnings: list[str],
) -> None:
    table = _parse_symtable(context, warnings=warnings)
    if table is None:
        return
    tree = _parse_ast(context, warnings=warnings)
    module_anchor: _AstAnchor | None = None
    anchors: dict[tuple[str, str, int], _AstAnchor] = {}
    if tree is not None:
        module_anchor, anchors = _build_ast_anchor_index(
            tree,
            context.source_index,
            rel_path=context.module.rel_path,
        )
    scope_id_context = _ScopeIdContext(
        repo=context.repo,
        commit=context.commit,
        rel_path=context.module.rel_path,
    )
    scope_index, scope_edges = _build_scope_index(
        table,
        scope_id_context=scope_id_context,
        module_name=context.module.module_name,
    )
    anchor_bundle = _resolve_scope_anchors(
        scope_index,
        module_anchor=module_anchor,
        anchors=anchors,
    )
    _append_scope_rows(
        collectors,
        context=context,
        scope_index=scope_index,
        anchors=anchor_bundle,
    )
    _append_scope_edges(
        collectors,
        context=context,
        scope_edges=scope_edges,
    )
    binding_by_scope = _append_symbols_and_bindings(
        collectors,
        context=context,
        scope_index=scope_index,
    )
    scope_by_id = {info.scope_id: info for info in scope_index.values()}
    _append_resolution_edges(
        collectors,
        context=context,
        binding_by_scope=binding_by_scope,
        scope_by_id=scope_by_id,
    )


class SymtableExtractStep(BaseExtractStep):
    """Symtable extraction step with port injection."""

    def __init__(
        self,
        discovery: ModuleDiscoveryPort,
        *,
        options: SymtableExtractOptions | None = None,
        frontend: PyFrontend | None = None,
    ) -> None:
        """Initialize the symtable extraction step.

        Parameters
        ----------
        discovery
            Discovery port for reading module source.
        options
            Symtable extraction options.
        frontend
            Optional shared frontend cache for source and AST reuse.
        """
        super().__init__(discovery, frontend=frontend)
        self._options = options or SymtableExtractOptions()

    def execute(
        self,
        modules: Sequence[ModuleRecord],
        *,
        repo: str | None = None,
        commit: str | None = None,
        context: IngestionContext | None = None,
        storage: IngestStoragePort | None = None,
    ) -> SymtableExtractResult:
        """Execute symtable extraction for the provided modules.

        Returns
        -------
        SymtableExtractResult
            Result bundle with row payloads and execution status.
        """
        resolved_repo, resolved_commit = resolve_repo_commit(
            context=context,
            repo=repo,
            commit=commit,
        )
        options = self._options
        if not options.enable:
            return SymtableExtractResult(
                result=ExecutionResult.skip("Symtable extraction disabled by options")
            )
        try:
            collectors = _build_symtable_collectors(options)
        except (KeyError, RuntimeError) as exc:
            return SymtableExtractResult(result=ExecutionResult.failed(str(exc)))

        warnings: list[str] = []
        for module, source_text, source_index, tree in self._iter_python_source_bundles(modules):
            module_context = _ModuleContext(
                repo=resolved_repo,
                commit=resolved_commit,
                module=module,
                source_text=source_text,
                source_index=source_index,
                ast_tree=tree,
            )
            _process_module(module_context, collectors, warnings=warnings)
            _flush_symtable_collectors(collectors)

        tables, finalize_warnings = _finalize_symtable_tables(collectors)
        warnings.extend(finalize_warnings)
        scope = f"{resolved_repo}@{resolved_commit}"
        persist_arrow_tables(storage, tables.as_mapping(), scope=scope)
        return _symtable_result(tables, warnings)

    def _iter_python_source_bundles(
        self,
        modules: Sequence[ModuleRecord],
    ) -> Iterable[tuple[ModuleRecord, str, LineIndexedSource, ast.AST | None]]:
        for module in modules:
            if not module.rel_path.endswith(".py"):
                continue
            if self._frontend is not None:
                bundle = self._frontend.get_source_bundle(module)
                if bundle is None:
                    continue
                tree = self._frontend.get_ast(module)
                yield module, bundle.source_text, bundle.source_index, tree
                continue
            source_bytes = self._discovery.read_module_bytes(module)
            if source_bytes is None:
                source_text = self._discovery.read_module_source(module)
                if source_text is None:
                    continue
                source_bytes = source_text.encode("utf-8", errors="replace")
            source_text, source_index = _build_source_index(source_bytes)
            yield module, source_text, source_index, None


def _finalize_symtable_tables(
    collectors: _SymtableCollectors,
) -> tuple[_SymtableTables, list[str]]:
    finalized_tables, finalize_warnings = finalize_arrow_readers(
        {
            PY_SYM_SCOPES_TABLE_KEY: collectors.scopes.to_reader(),
            PY_SYM_SYMBOLS_TABLE_KEY: collectors.symbols.to_reader(),
            PY_SYM_SCOPE_EDGES_TABLE_KEY: collectors.scope_edges.to_reader(),
            PY_SYM_NAMESPACE_EDGES_TABLE_KEY: collectors.namespace_edges.to_reader(),
            PY_SYM_FUNCTION_PARTITIONS_TABLE_KEY: collectors.function_partitions.to_reader(),
            PY_SYM_BINDINGS_TABLE_KEY: collectors.bindings.to_reader(),
            PY_SYM_RESOLUTION_EDGES_TABLE_KEY: collectors.resolution_edges.to_reader(),
        }
    )
    return (
        _SymtableTables(
            scope_rows_table=finalized_tables[PY_SYM_SCOPES_TABLE_KEY],
            symbol_rows_table=finalized_tables[PY_SYM_SYMBOLS_TABLE_KEY],
            scope_edge_rows_table=finalized_tables[PY_SYM_SCOPE_EDGES_TABLE_KEY],
            namespace_edge_rows_table=finalized_tables[PY_SYM_NAMESPACE_EDGES_TABLE_KEY],
            function_partition_rows_table=finalized_tables[PY_SYM_FUNCTION_PARTITIONS_TABLE_KEY],
            binding_rows_table=finalized_tables[PY_SYM_BINDINGS_TABLE_KEY],
            resolution_edge_rows_table=finalized_tables[PY_SYM_RESOLUTION_EDGES_TABLE_KEY],
        ),
        finalize_warnings,
    )


def _symtable_result(
    tables: _SymtableTables,
    warnings: Iterable[str],
) -> SymtableExtractResult:
    return SymtableExtractResult(
        result=ExecutionResult.ok(warnings=tuple(warnings)),
        scope_rows={},
        symbol_rows={},
        scope_edge_rows={},
        namespace_edge_rows={},
        function_partition_rows={},
        binding_rows={},
        resolution_edge_rows={},
        scope_rows_reader=tables.scope_rows_table,
        symbol_rows_reader=tables.symbol_rows_table,
        scope_edge_rows_reader=tables.scope_edge_rows_table,
        namespace_edge_rows_reader=tables.namespace_edge_rows_table,
        function_partition_rows_reader=tables.function_partition_rows_table,
        binding_rows_reader=tables.binding_rows_table,
        resolution_edge_rows_reader=tables.resolution_edge_rows_table,
        scope_row_count=tables.scope_rows_table.num_rows,
        symbol_row_count=tables.symbol_rows_table.num_rows,
        scope_edge_row_count=tables.scope_edge_rows_table.num_rows,
        namespace_edge_row_count=tables.namespace_edge_rows_table.num_rows,
        function_partition_row_count=tables.function_partition_rows_table.num_rows,
        binding_row_count=tables.binding_rows_table.num_rows,
        resolution_edge_row_count=tables.resolution_edge_rows_table.num_rows,
    )


__all__ = ["SymtableExtractResult", "SymtableExtractStep"]
