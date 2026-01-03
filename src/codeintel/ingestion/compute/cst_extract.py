"""CST extraction step with port injection.

This module provides a pure domain logic implementation for extracting
LibCST concrete syntax trees, using ports for all I/O operations.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import tokenize
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import libcst as cst
from intervaltree import IntervalTree
from libcst import metadata
from libcst.helpers import get_full_name_for_node

from codeintel.build.hamilton.execution_result import ExecutionResult
from codeintel.core.columnar.rows import (
    ColumnarRowBuffer,
    ColumnarRows,
    columnar_buffer_for_table_key,
)
from codeintel.ingestion.compute.base import BaseExtractStep
from codeintel.ingestion.infrastructure.ast_facts import (
    AstCollectContext,
    AstNodeRecord,
    AstSpan,
    collect_ast_nodes,
)
from codeintel.ingestion.infrastructure.cst_utils import (
    CstCaptureConfig,
    CstCaptureVisitor,
    LineIndexedSource,
    NormalizedSpan,
    SourceBundle,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from libcst.metadata.scope_provider import Access, Scope

    from codeintel.ingestion.ports.discovery import ModuleDiscoveryPort, ModuleRecord

log = logging.getLogger(__name__)

CST_NODES_TABLE_KEY = "core.cst_nodes"
PARSE_MANIFEST_TABLE_KEY = "core.parse_manifest"
SYNTAX_SPANS_TABLE_KEY = "core.syntax_spans"
SYNTAX_NODES_TABLE_KEY = "core.syntax_nodes"
SYNTAX_EDGES_TABLE_KEY = "core.syntax_edges"
SYNTAX_SCOPES_TABLE_KEY = "core.syntax_scopes"
SYNTAX_DEFS_TABLE_KEY = "core.syntax_defs"
SYNTAX_REFS_TABLE_KEY = "core.syntax_refs"
SYNTAX_CALLS_TABLE_KEY = "core.syntax_calls"
SYNTAX_CALL_ARGS_TABLE_KEY = "core.syntax_call_args"
SYNTAX_FUNC_PARAMS_TABLE_KEY = "core.syntax_func_params"
SYNTAX_IMPORTS_TABLE_KEY = "core.syntax_imports"

SYNTAX_PRODUCER = "libcst"
SYNTAX_LANGUAGE = "python"
SYNTAX_EDGE_KIND = "AST_CHILD"

ASYNC_FUNC_DEF = getattr(cst, "AsyncFunctionDef", cst.FunctionDef)

CstRow = tuple[str, str, str, dict[str, list[int]], str, tuple[str, ...], tuple[str, ...]]

CST_CAPTURE_CONFIG = CstCaptureConfig(
    kinds=(
        cst.Module,
        cst.FunctionDef,
        ASYNC_FUNC_DEF,
        cst.ClassDef,
        cst.Assign,
        cst.AnnAssign,
        cst.AugAssign,
        cst.Import,
        cst.ImportFrom,
        cst.Call,
        cst.Return,
        cst.Raise,
        cst.Yield,
        cst.If,
        cst.Else,
        cst.For,
        cst.While,
        cst.With,
        cst.Try,
        cst.ExceptHandler,
        cst.Match,
    ),
    snippet_limit=200,
)


@dataclass(frozen=True)
class CstExtractResult:
    """Result bundle for CST extraction."""

    result: ExecutionResult
    rows: ColumnarRows = field(default_factory=dict)
    parse_manifest_rows: ColumnarRows = field(default_factory=dict)
    syntax_spans_rows: ColumnarRows = field(default_factory=dict)
    syntax_nodes_rows: ColumnarRows = field(default_factory=dict)
    syntax_edges_rows: ColumnarRows = field(default_factory=dict)
    syntax_scopes_rows: ColumnarRows = field(default_factory=dict)
    syntax_defs_rows: ColumnarRows = field(default_factory=dict)
    syntax_refs_rows: ColumnarRows = field(default_factory=dict)
    syntax_calls_rows: ColumnarRows = field(default_factory=dict)
    syntax_call_args_rows: ColumnarRows = field(default_factory=dict)
    syntax_func_params_rows: ColumnarRows = field(default_factory=dict)
    syntax_imports_rows: ColumnarRows = field(default_factory=dict)
    row_count: int = 0
    parse_manifest_row_count: int = 0
    syntax_spans_row_count: int = 0
    syntax_nodes_row_count: int = 0
    syntax_edges_row_count: int = 0
    syntax_scopes_row_count: int = 0
    syntax_defs_row_count: int = 0
    syntax_refs_row_count: int = 0
    syntax_calls_row_count: int = 0
    syntax_call_args_row_count: int = 0
    syntax_func_params_row_count: int = 0
    syntax_imports_row_count: int = 0


@dataclass(frozen=True)
class ScopeFrame:
    """Tracking state for nested scopes."""

    scope_id: str
    scope_kind: str


@dataclass(frozen=True, slots=True)
class _ParseManifestContext:
    repo: str
    commit: str
    rel_path: str
    producer: str
    source_index: LineIndexedSource
    encoding: str | None
    default_indent: str | None
    default_newline: str | None
    has_trailing_newline: bool | None
    future_imports: list[str] | None
    parser_backend: str | None
    libcst_version: str | None


@dataclass(frozen=True)
class _ParsedCstModule:
    parsed_module: cst.Module
    source_text: str
    source_index: LineIndexedSource
    encoding: str
    manifest_context: _ParseManifestContext
    wrapper: metadata.MetadataWrapper


@dataclass(slots=True)
class _CstBuffers:
    cst: ColumnarRowBuffer
    parse_manifest: ColumnarRowBuffer
    spans: ColumnarRowBuffer
    syntax_nodes: ColumnarRowBuffer
    syntax_edges: ColumnarRowBuffer
    scopes: ColumnarRowBuffer
    defs: ColumnarRowBuffer
    refs: ColumnarRowBuffer
    calls: ColumnarRowBuffer
    call_args: ColumnarRowBuffer
    func_params: ColumnarRowBuffer
    imports: ColumnarRowBuffer


class CstVisitor(CstCaptureVisitor):
    """Collect CST rows using shared capture helpers."""

    def __init__(
        self,
        rel_path: str,
        module_name: str,
        source: SourceBundle,
    ) -> None:
        """Initialize visitor.

        Parameters
        ----------
        rel_path
            Relative path to the file.
        module_name
            Python module name.
        source
            Bundle of text, raw bytes, and encoding for the module.
        """
        super().__init__(
            rel_path,
            module_name,
            source,
            config=CST_CAPTURE_CONFIG,
        )


@dataclass(slots=True)
class _SyntaxGraphFrame:
    node_id: str | None
    child_ordinal: int = 0


@dataclass(frozen=True, slots=True)
class _SyntaxContext:
    repo: str
    commit: str
    rel_path: str
    producer: str
    language: str


@dataclass(frozen=True, slots=True)
class _DefExtrasInput:
    container_def_id: str | None
    is_async: bool | None = None
    bases: Sequence[cst.Arg] | None = None
    decorators: Sequence[cst.Decorator] | None = None
    params: cst.Parameters | None = None
    returns_node: cst.CSTNode | None = None
    docstring: str | None = None
    qualified_node: cst.CSTNode | None = None


@dataclass(frozen=True, slots=True)
class _SyntaxNodeCandidate:
    node_id: str
    start_byte: int
    end_byte: int
    node_kind: str
    order: int


@dataclass(frozen=True, slots=True)
class _SyntaxNodeIndex:
    tree: IntervalTree
    exact: dict[tuple[int, int], list[_SyntaxNodeCandidate]]


class SyntaxGraphVisitor(cst.CSTVisitor):
    """Collect canonical syntax nodes and edges for CPG stitching."""

    METADATA_DEPENDENCIES = (
        metadata.PositionProvider,
        metadata.ByteSpanPositionProvider,
    )

    def __init__(
        self,
        *,
        context: _SyntaxContext,
        source_index: LineIndexedSource,
        snippet_limit: int,
    ) -> None:
        self.repo = context.repo
        self.commit = context.commit
        self.rel_path = context.rel_path
        self.producer = context.producer
        self.language = context.language
        self.source_index = source_index
        self.snippet_limit = snippet_limit

        self.node_rows: list[dict[str, object]] = []
        self.edge_rows: list[dict[str, object]] = []
        self._seen_node_ids: set[str] = set()
        self._stack: list[_SyntaxGraphFrame] = []

    def on_visit(self, node: cst.CSTNode) -> bool:
        span = self._span_for_node(node)
        node_id = None
        if span is not None:
            node_id = _stable_id(
                "syntax_node",
                self.rel_path,
                self.producer,
                type(node).__name__,
                span.start_line,
                span.start_col,
                span.end_line,
                span.end_col,
                span.start_byte,
                span.end_byte,
            )
            if node_id not in self._seen_node_ids:
                self._seen_node_ids.add(node_id)
                preview = self.source_index.slice(
                    span.start_line,
                    span.start_col,
                    span.end_line,
                    span.end_col,
                )
                self.node_rows.append(
                    {
                        "repo": self.repo,
                        "commit": self.commit,
                        "rel_path": self.rel_path,
                        "producer": self.producer,
                        "language": self.language,
                        "node_id": node_id,
                        "node_kind": type(node).__name__,
                        "raw_kind": type(node).__name__,
                        "start_line": span.start_line,
                        "start_col": span.start_col,
                        "end_line": span.end_line,
                        "end_col": span.end_col,
                        "start_byte": span.start_byte,
                        "end_byte": span.end_byte,
                        "text_preview": preview[: self.snippet_limit],
                        "extras_json": None,
                    }
                )

        parent_frame = self._last_parent_frame()
        if node_id is not None and parent_frame is not None:
            ordinal = parent_frame.child_ordinal
            parent_frame.child_ordinal += 1
            self.edge_rows.append(
                {
                    "repo": self.repo,
                    "commit": self.commit,
                    "rel_path": self.rel_path,
                    "producer": self.producer,
                    "parent_node_id": parent_frame.node_id,
                    "child_node_id": node_id,
                    "edge_kind": SYNTAX_EDGE_KIND,
                    "field_name": None,
                    "child_ordinal": ordinal,
                }
            )

        self._stack.append(_SyntaxGraphFrame(node_id=node_id))
        return True

    def on_leave(self, original_node: cst.CSTNode) -> None:
        _ = original_node
        if self._stack:
            self._stack.pop()

    def _last_parent_frame(self) -> _SyntaxGraphFrame | None:
        for frame in reversed(self._stack):
            if frame.node_id is not None:
                return frame
        return None

    def _span_for_node(self, node: cst.CSTNode) -> NormalizedSpan | None:
        try:
            pos = self.get_metadata(metadata.PositionProvider, node)
        except KeyError:
            return None
        if not isinstance(pos, metadata.CodeRange):
            return None
        byte_span = self._byte_span_for_node(node)
        return self.source_index.span_from_range(pos, byte_span)

    def _byte_span_for_node(self, node: cst.CSTNode) -> metadata.CodeSpan | None:
        try:
            byte_span = self.get_metadata(metadata.ByteSpanPositionProvider, node)
        except KeyError:
            return None
        if isinstance(byte_span, metadata.CodeSpan):
            return byte_span
        return None


class SyntaxFactsVisitor(cst.CSTVisitor):
    """Collect syntax fact tables from a LibCST module."""

    METADATA_DEPENDENCIES = (
        metadata.PositionProvider,
        metadata.ByteSpanPositionProvider,
        metadata.ParentNodeProvider,
        metadata.ExpressionContextProvider,
        metadata.QualifiedNameProvider,
        metadata.ScopeProvider,
    )

    def __init__(
        self,
        *,
        context: _SyntaxContext,
        source_index: LineIndexedSource,
        access_map: dict[int, Access] | None = None,
    ) -> None:
        self.repo = context.repo
        self.commit = context.commit
        self.rel_path = context.rel_path
        self.producer = context.producer
        self.source_index = source_index

        self.span_rows: list[dict[str, object]] = []
        self.scopes: list[dict[str, object]] = []
        self.defs: list[dict[str, object]] = []
        self.refs: list[dict[str, object]] = []
        self.calls: list[dict[str, object]] = []
        self.call_args: list[dict[str, object]] = []
        self.func_params: list[dict[str, object]] = []
        self.imports: list[dict[str, object]] = []
        self._span_ids: set[str] = set()
        self._scope_stack: list[ScopeFrame] = []
        self._access_map = access_map or {}
        self._def_node_ids: dict[int, str] = {}

    def visit_Module(self, node: cst.Module) -> bool:
        self._enter_scope(node, "module")
        return True

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        container_def_id = self._container_def_id(node)
        extras = self._def_extras(
            _DefExtrasInput(
                container_def_id=container_def_id,
                bases=node.bases,
                decorators=node.decorators,
                docstring=self._docstring(node),
                qualified_node=node,
            )
        )
        def_id = self._record_named_def(node.name, node.name.value, "class", extras=extras)
        if def_id is not None:
            self._def_node_ids[id(node)] = def_id
        self._enter_scope(node, "class")
        return True

    def visit_FunctionDef(self, node: cst.CSTNode) -> bool:
        return self._visit_function_like(node)

    def visit_Lambda(self, node: cst.Lambda) -> bool:
        self._enter_scope(node, "lambda")
        self._record_params(node.params)
        return True

    def visit_AsyncFunctionDef(self, node: cst.CSTNode) -> bool:
        return self._visit_function_like(node)

    def on_leave(self, original_node: cst.CSTNode) -> None:
        if not isinstance(
            original_node,
            (cst.Module, cst.ClassDef, cst.FunctionDef, ASYNC_FUNC_DEF, cst.Lambda),
        ):
            return
        if self._span_for_node(original_node) is None:
            return
        self._exit_scope()

    def _visit_function_like(self, node: cst.CSTNode) -> bool:
        kind = "function"
        if ASYNC_FUNC_DEF is not cst.FunctionDef and isinstance(node, ASYNC_FUNC_DEF):
            kind = "async_function"
        name = getattr(node, "name", None)
        if isinstance(name, cst.Name):
            container_def_id = self._container_def_id(node)
            extras = self._def_extras(
                _DefExtrasInput(
                    container_def_id=container_def_id,
                    is_async=kind == "async_function",
                    decorators=getattr(node, "decorators", None),
                    params=getattr(node, "params", None),
                    returns_node=getattr(node, "returns", None),
                    docstring=self._docstring(node),
                    qualified_node=node,
                )
            )
            def_id = self._record_named_def(name, name.value, kind, extras=extras)
            if def_id is not None:
                self._def_node_ids[id(node)] = def_id
        self._enter_scope(node, kind)
        params = getattr(node, "params", None)
        if isinstance(params, cst.Parameters):
            self._record_params(params)
        return True

    def visit_Assign(self, node: cst.Assign) -> bool:
        for target in node.targets:
            for name in self._iter_name_targets(target.target):
                self._record_named_def(name, name.value, "local")
        return True

    def visit_AnnAssign(self, node: cst.AnnAssign) -> bool:
        for name in self._iter_name_targets(node.target):
            self._record_named_def(name, name.value, "local")
        return True

    def visit_AugAssign(self, node: cst.AugAssign) -> bool:
        for name in self._iter_name_targets(node.target):
            self._record_named_def(name, name.value, "local")
        return True

    def visit_For(self, node: cst.For) -> bool:
        for name in self._iter_name_targets(node.target):
            self._record_named_def(name, name.value, "local")
        return True

    def visit_CompFor(self, node: cst.CompFor) -> bool:
        for name in self._iter_name_targets(node.target):
            self._record_named_def(name, name.value, "local")
        return True

    def visit_Call(self, node: cst.Call) -> bool:
        self._record_call(node)
        return True

    def visit_Import(self, node: cst.Import) -> bool:
        for alias in node.names:
            self._record_import_alias(alias, import_kind="import", module=None, level=None)
            self._record_import_binding(alias)
        return True

    def visit_ImportFrom(self, node: cst.ImportFrom) -> bool:
        module = self._node_text(node.module) if node.module is not None else None
        level = self._relative_level(node.relative)
        if isinstance(node.names, cst.ImportStar):
            self._record_import_star(node.names, module=module, level=level)
            return True

        for alias in node.names:
            self._record_import_alias(
                alias,
                import_kind="from_import",
                module=module,
                level=level,
            )
            self._record_import_binding(alias)
        return True

    def visit_Name(self, node: cst.Name) -> bool:
        if self._is_definition_name(node):
            return True
        extras = self._ref_extras(node)
        self._record_ref(
            node,
            node.value,
            ref_kind="identifier",
            span_kind="identifier",
            extras=extras,
        )
        return True

    def visit_Attribute(self, node: cst.Attribute) -> bool:
        if self._is_assignment_target(node):
            return True
        name = self._node_text(node)
        if name is None:
            return True
        extras = self._ref_extras(node)
        self._record_ref(node, name, ref_kind="attribute", span_kind="attribute", extras=extras)
        return True

    def _current_scope_id(self) -> str | None:
        if not self._scope_stack:
            return None
        return self._scope_stack[-1].scope_id

    def _enter_scope(self, node: cst.CSTNode, scope_kind: str) -> None:
        span = self._span_for_node(node)
        if span is None:
            return
        scope_id = _stable_id(
            "scope",
            self.repo,
            self.commit,
            self.rel_path,
            self.producer,
            scope_kind,
            span.start_line,
            span.start_col,
            span.end_line,
            span.end_col,
        )
        parent_scope_id = self._current_scope_id()
        self.scopes.append(
            {
                "repo": self.repo,
                "commit": self.commit,
                "rel_path": self.rel_path,
                "producer": self.producer,
                "scope_id": scope_id,
                "scope_kind": scope_kind,
                "start_line": span.start_line,
                "start_col": span.start_col,
                "end_line": span.end_line,
                "end_col": span.end_col,
                "parent_scope_id": parent_scope_id,
            }
        )
        self._scope_stack.append(ScopeFrame(scope_id=scope_id, scope_kind=scope_kind))

    def _exit_scope(self) -> None:
        if self._scope_stack:
            self._scope_stack.pop()

    def _record_params(self, params: cst.Parameters) -> None:
        for ordinal, (param, param_kind) in enumerate(self._iter_params(params)):
            extras = self._param_def_extras(param, param_kind)
            def_id = self._record_named_def(param.name, param.name.value, "param", extras=extras)
            span = self._span_for_node(param.name)
            if def_id is None or span is None:
                continue
            func_def_id = self._container_def_id(param)
            if func_def_id is None:
                continue
            span_id = self._ensure_span(span, "identifier")
            node_id = _stable_id(
                "syntax_node",
                self.rel_path,
                self.producer,
                type(param.name).__name__,
                span.start_line,
                span.start_col,
                span.end_line,
                span.end_col,
                span.start_byte,
                span.end_byte,
            )
            self.func_params.append(
                {
                    "repo": self.repo,
                    "commit": self.commit,
                    "rel_path": self.rel_path,
                    "producer": self.producer,
                    "func_def_id": func_def_id,
                    "param_def_id": def_id,
                    "param_ordinal": ordinal,
                    "param_kind": param_kind,
                    "param_name": param.name.value,
                    "param_start_line": span.start_line,
                    "param_start_col": span.start_col,
                    "param_end_line": span.end_line,
                    "param_end_col": span.end_col,
                    "param_start_byte": span.start_byte,
                    "param_end_byte": span.end_byte,
                    "param_span_id": span_id,
                    "param_node_id": node_id,
                    "extras_json": extras,
                }
            )

    @staticmethod
    def _iter_params(params: cst.Parameters) -> Iterable[tuple[cst.Param, str]]:
        for param in params.posonly_params:
            yield param, "posonly"
        for param in params.params:
            yield param, "positional"
        for param in params.kwonly_params:
            yield param, "kwonly"
        if isinstance(params.star_arg, cst.Param):
            yield params.star_arg, "varargs"
        if isinstance(params.star_kwarg, cst.Param):
            yield params.star_kwarg, "varkw"

    def _record_named_def(
        self,
        node: cst.CSTNode,
        name: str,
        def_kind: str,
        *,
        span_kind: str = "identifier",
        extras: dict[str, object] | None = None,
    ) -> str | None:
        scope_id = self._current_scope_id()
        if scope_id is None:
            return None
        span = self._span_for_node(node)
        if span is None:
            return None
        span_id = self._ensure_span(span, span_kind)
        def_id = _stable_id(
            "def",
            self.repo,
            self.commit,
            self.rel_path,
            self.producer,
            def_kind,
            name,
            span_id,
            scope_id,
        )
        self.defs.append(
            {
                "repo": self.repo,
                "commit": self.commit,
                "rel_path": self.rel_path,
                "producer": self.producer,
                "def_id": def_id,
                "scope_id": scope_id,
                "span_id": span_id,
                "def_kind": def_kind,
                "name": name,
                "start_line": span.start_line,
                "start_col": span.start_col,
                "end_line": span.end_line,
                "end_col": span.end_col,
                "start_byte": span.start_byte,
                "end_byte": span.end_byte,
                "extras_json": extras,
            }
        )
        return def_id

    def _record_ref(
        self,
        node: cst.CSTNode,
        name: str,
        *,
        ref_kind: str,
        span_kind: str,
        extras: dict[str, object] | None = None,
    ) -> None:
        scope_id = self._current_scope_id()
        if scope_id is None:
            return
        span = self._span_for_node(node)
        if span is None:
            return
        span_id = self._ensure_span(span, span_kind)
        ref_id = _stable_id(
            "ref",
            self.repo,
            self.commit,
            self.rel_path,
            self.producer,
            ref_kind,
            name,
            span_id,
            scope_id,
        )
        self.refs.append(
            {
                "repo": self.repo,
                "commit": self.commit,
                "rel_path": self.rel_path,
                "producer": self.producer,
                "ref_id": ref_id,
                "scope_id": scope_id,
                "span_id": span_id,
                "ref_kind": ref_kind,
                "name": name,
                "start_line": span.start_line,
                "start_col": span.start_col,
                "end_line": span.end_line,
                "end_col": span.end_col,
                "start_byte": span.start_byte,
                "end_byte": span.end_byte,
                "extras_json": extras,
            }
        )

    def _record_call(self, node: cst.Call) -> None:
        scope_id = self._current_scope_id()
        if scope_id is None:
            return
        span = self._span_for_node(node)
        if span is None:
            return
        span_id = self._ensure_span(span, "call_expr")
        call_node_id = _stable_id(
            "syntax_node",
            self.rel_path,
            self.producer,
            type(node).__name__,
            span.start_line,
            span.start_col,
            span.end_line,
            span.end_col,
            span.start_byte,
            span.end_byte,
        )
        callee_span_id = None
        callee_text = self._node_text(node.func)
        callee_span = self._span_for_node(node.func)
        callee_start_byte = None
        callee_end_byte = None
        if callee_span is not None:
            callee_span_id = self._ensure_span(callee_span, "callee_expr")
            callee_start_byte = callee_span.start_byte
            callee_end_byte = callee_span.end_byte
        call_id = _stable_id(
            "call",
            self.repo,
            self.commit,
            self.rel_path,
            self.producer,
            span_id,
            scope_id,
            callee_span_id,
            callee_text,
        )
        extras = self._call_extras(node)
        self.calls.append(
            {
                "repo": self.repo,
                "commit": self.commit,
                "rel_path": self.rel_path,
                "producer": self.producer,
                "call_id": call_id,
                "call_node_id": call_node_id,
                "scope_id": scope_id,
                "span_id": span_id,
                "callee_span_id": callee_span_id,
                "callee_text": callee_text,
                "arg_count": len(node.args),
                "start_line": span.start_line,
                "start_col": span.start_col,
                "end_line": span.end_line,
                "end_col": span.end_col,
                "start_byte": span.start_byte,
                "end_byte": span.end_byte,
                "callee_start_byte": callee_start_byte,
                "callee_end_byte": callee_end_byte,
                "extras_json": extras,
            }
        )
        self._record_call_args(call_id, node)

    def _record_call_args(self, call_id: str, node: cst.Call) -> None:
        for ordinal, arg in enumerate(node.args):
            span = self._span_for_node(arg.value)
            if span is None:
                continue
            span_id = self._ensure_span(span, "arg_expr")
            node_id = _stable_id(
                "syntax_node",
                self.rel_path,
                self.producer,
                type(arg.value).__name__,
                span.start_line,
                span.start_col,
                span.end_line,
                span.end_col,
                span.start_byte,
                span.end_byte,
            )
            arg_kind, arg_name = self._arg_kind_and_name(arg)
            self.call_args.append(
                {
                    "repo": self.repo,
                    "commit": self.commit,
                    "rel_path": self.rel_path,
                    "producer": self.producer,
                    "call_id": call_id,
                    "arg_ordinal": ordinal,
                    "arg_kind": arg_kind,
                    "arg_name": arg_name,
                    "arg_start_line": span.start_line,
                    "arg_start_col": span.start_col,
                    "arg_end_line": span.end_line,
                    "arg_end_col": span.end_col,
                    "arg_start_byte": span.start_byte,
                    "arg_end_byte": span.end_byte,
                    "arg_span_id": span_id,
                    "arg_expr_node_id": node_id,
                    "extras_json": None,
                }
            )

    @staticmethod
    def _arg_kind_and_name(arg: cst.Arg) -> tuple[str, str | None]:
        if arg.star == "**":
            return "kwargs", None
        if arg.star == "*":
            return "starargs", None
        if isinstance(arg.keyword, cst.Name):
            return "keyword", arg.keyword.value
        return "positional", None

    def _record_import_alias(
        self,
        alias: cst.ImportAlias,
        *,
        import_kind: str,
        module: str | None,
        level: int | None,
    ) -> None:
        scope_id = self._current_scope_id()
        if scope_id is None:
            return
        name_text = self._node_text(alias.name)
        alias_text = None
        if alias.asname is not None:
            alias_text = self._binding_name_from_node(alias.asname.name)
        span_node = alias.asname.name if alias.asname is not None else alias.name
        span = self._span_for_node(span_node) or self._span_for_node(alias)
        if span is None:
            return
        span_id = self._ensure_span(span, "import_name")
        if import_kind == "from_import":
            module_value = module
            name_value = name_text
        else:
            module_value = name_text
            name_value = None
        import_id = _stable_id(
            "import",
            self.repo,
            self.commit,
            self.rel_path,
            self.producer,
            import_kind,
            module_value,
            name_value,
            alias_text,
            level,
            span_id,
            scope_id,
        )
        extras = self._import_extras(
            alias,
            import_kind=import_kind,
            module=module_value,
            level=level,
            is_star=False,
        )
        self.imports.append(
            {
                "repo": self.repo,
                "commit": self.commit,
                "rel_path": self.rel_path,
                "producer": self.producer,
                "import_id": import_id,
                "scope_id": scope_id,
                "span_id": span_id,
                "import_kind": import_kind,
                "module": module_value,
                "name": name_value,
                "alias": alias_text,
                "level": level,
                "start_line": span.start_line,
                "start_col": span.start_col,
                "end_line": span.end_line,
                "end_col": span.end_col,
                "start_byte": span.start_byte,
                "end_byte": span.end_byte,
                "extras_json": extras,
            }
        )

    def _record_import_star(
        self,
        alias: cst.ImportStar,
        *,
        module: str | None,
        level: int | None,
    ) -> None:
        scope_id = self._current_scope_id()
        if scope_id is None:
            return
        span = self._span_for_node(alias)
        if span is None:
            return
        span_id = self._ensure_span(span, "import_name")
        import_id = _stable_id(
            "import",
            self.repo,
            self.commit,
            self.rel_path,
            self.producer,
            "from_import",
            module,
            "*",
            None,
            level,
            span_id,
            scope_id,
        )
        extras = self._import_extras(
            alias,
            import_kind="from_import",
            module=module,
            level=level,
            is_star=True,
        )
        self.imports.append(
            {
                "repo": self.repo,
                "commit": self.commit,
                "rel_path": self.rel_path,
                "producer": self.producer,
                "import_id": import_id,
                "scope_id": scope_id,
                "span_id": span_id,
                "import_kind": "from_import",
                "module": module,
                "name": "*",
                "alias": None,
                "level": level,
                "start_line": span.start_line,
                "start_col": span.start_col,
                "end_line": span.end_line,
                "end_col": span.end_col,
                "start_byte": span.start_byte,
                "end_byte": span.end_byte,
                "extras_json": extras,
            }
        )

    def _record_import_binding(self, alias: cst.ImportAlias) -> None:
        binding_node, binding_name = self._binding_target(alias)
        if binding_node is None or binding_name is None:
            return
        self._record_named_def(
            binding_node,
            binding_name,
            "import_alias",
            span_kind="import_name",
        )

    def _binding_target(self, alias: cst.ImportAlias) -> tuple[cst.CSTNode | None, str | None]:
        if alias.asname is not None:
            return alias.asname.name, self._binding_name_from_node(alias.asname.name)
        return alias.name, self._binding_name_from_node(alias.name)

    @staticmethod
    def _binding_name_from_node(node: cst.CSTNode) -> str | None:
        full_name = get_full_name_for_node(node)
        if isinstance(full_name, str) and full_name:
            return full_name
        return None

    def _node_text(self, node: cst.CSTNode | None) -> str | None:
        if node is None:
            return None
        span = self._span_for_node(node)
        if span is None:
            return None
        return self.source_index.slice(
            span.start_line,
            span.start_col,
            span.end_line,
            span.end_col,
        )

    def _span_for_node(self, node: cst.CSTNode) -> NormalizedSpan | None:
        try:
            pos = self.get_metadata(metadata.PositionProvider, node)
        except KeyError:
            return None
        if not isinstance(pos, metadata.CodeRange):
            return None
        byte_span = self._byte_span_for_node(node)
        return self.source_index.span_from_range(pos, byte_span)

    def _byte_span_for_node(self, node: cst.CSTNode) -> metadata.CodeSpan | None:
        try:
            byte_span = self.get_metadata(metadata.ByteSpanPositionProvider, node)
        except KeyError:
            return None
        if isinstance(byte_span, metadata.CodeSpan):
            return byte_span
        return None

    def _ensure_span(self, span: NormalizedSpan, span_kind: str) -> str:
        span_id = _stable_id(
            "span",
            self.repo,
            self.commit,
            self.rel_path,
            self.producer,
            span_kind,
            span.start_line,
            span.start_col,
            span.end_line,
            span.end_col,
        )
        if span_id in self._span_ids:
            return span_id
        self._span_ids.add(span_id)
        self.span_rows.append(
            {
                "repo": self.repo,
                "commit": self.commit,
                "rel_path": self.rel_path,
                "producer": self.producer,
                "span_id": span_id,
                "span_kind": span_kind,
                "start_line": span.start_line,
                "start_col": span.start_col,
                "end_line": span.end_line,
                "end_col": span.end_col,
                "start_byte": span.start_byte,
                "end_byte": span.end_byte,
            }
        )
        return span_id

    def _parent(self, node: cst.CSTNode) -> cst.CSTNode | None:
        try:
            parent = self.get_metadata(metadata.ParentNodeProvider, node)
        except KeyError:
            return None
        return parent if isinstance(parent, cst.CSTNode) else None

    def _is_definition_name(self, node: cst.Name) -> bool:
        parent = self._parent(node)
        if parent is None:
            return False
        if isinstance(parent, (cst.ClassDef, cst.FunctionDef, ASYNC_FUNC_DEF, cst.Param)):
            return getattr(parent, "name", None) is node
        if isinstance(
            parent,
            (cst.AssignTarget, cst.AnnAssign, cst.AugAssign, cst.For, cst.CompFor),
        ):
            return parent.target is node
        if isinstance(parent, (cst.ImportAlias, cst.AsName)):
            return parent.name is node
        return False

    def _is_assignment_target(self, node: cst.CSTNode) -> bool:
        parent = self._parent(node)
        if parent is None:
            return False
        if isinstance(
            parent,
            (cst.AssignTarget, cst.AnnAssign, cst.AugAssign, cst.For, cst.CompFor),
        ):
            return parent.target is node
        return False

    def _container_def_id(self, node: cst.CSTNode) -> str | None:
        parent = self._parent(node)
        while parent is not None:
            def_id = self._def_node_ids.get(id(parent))
            if def_id is not None:
                return def_id
            parent = self._parent(parent)
        return None

    def _def_extras(self, payload: _DefExtrasInput) -> dict[str, object] | None:
        extras: dict[str, object] = {}
        extras.update(self._optional_entry("container_def_id", payload.container_def_id))
        extras.update(self._optional_entry("is_async", payload.is_async))
        extras.update(
            self._optional_entry(
                "decorators",
                self._decorator_payload(payload.decorators) or None,
            )
        )
        extras.update(self._optional_entry("bases", self._bases_payload(payload.bases) or None))
        extras.update(self._optional_entry("params", self._params_payload(payload.params) or None))
        extras.update(
            self._optional_entry(
                "returns_code",
                self._returns_payload(payload.returns_node),
            )
        )
        extras.update(self._optional_entry("docstring", payload.docstring))
        extras.update(
            self._optional_entry(
                "qnames",
                self._qualified_name_payload(payload.qualified_node) or None,
            )
        )
        return extras or None

    @staticmethod
    def _optional_entry(key: str, value: object | None) -> dict[str, object]:
        if value is None:
            return {}
        return {key: value}

    def _bases_payload(self, bases: Sequence[cst.Arg] | None) -> list[str]:
        if not bases:
            return []
        payload: list[str] = []
        for base in bases:
            base_text = self._node_text(base.value)
            if base_text:
                payload.append(base_text)
        return payload

    def _param_def_extras(self, param: cst.Param, param_kind: str) -> dict[str, object]:
        extras: dict[str, object] = {
            "param_kind": param_kind,
            "has_annotation": param.annotation is not None,
            "has_default": param.default is not None,
        }
        if param.annotation is not None:
            annotation_code = self._node_text(param.annotation)
            if annotation_code:
                extras["annotation_code"] = annotation_code
        if param.default is not None:
            default_code = self._node_text(param.default)
            if default_code:
                extras["default_code"] = default_code
        return extras

    def _params_payload(self, params: cst.Parameters | None) -> list[dict[str, object]]:
        if params is None:
            return []
        payload: list[dict[str, object]] = []
        for param, param_kind in self._iter_params(params):
            entry: dict[str, object] = {
                "name": param.name.value,
                "kind": param_kind,
                "has_annotation": param.annotation is not None,
                "has_default": param.default is not None,
            }
            if param.annotation is not None:
                annotation_code = self._node_text(param.annotation)
                if annotation_code:
                    entry["annotation_code"] = annotation_code
            if param.default is not None:
                default_code = self._node_text(param.default)
                if default_code:
                    entry["default_code"] = default_code
            payload.append(entry)
        return payload

    def _returns_payload(self, node: cst.CSTNode | None) -> str | None:
        if node is None:
            return None
        return self._node_text(node) or None

    def _ref_extras(self, node: cst.CSTNode) -> dict[str, object] | None:
        extras: dict[str, object] = {}
        role = self._expression_context_name(node)
        if role is not None:
            extras["role"] = role
        access = self._access_map.get(id(node))
        if access is not None:
            extras["scope_kind"] = type(access.scope).__name__
            referents = self._referents_payload(access)
            if referents:
                extras["referents"] = referents
            extras["is_annotation"] = access.is_annotation
            extras["is_type_hint"] = access.is_type_hint
        else:
            scope_kind = self._current_scope_kind()
            if scope_kind is not None:
                extras["scope_kind"] = scope_kind
        qnames = self._qualified_name_payload(node)
        if qnames:
            extras["qnames"] = qnames
        return extras or None

    def _call_extras(self, node: cst.Call) -> dict[str, object] | None:
        kw_count = sum(1 for arg in node.args if arg.keyword is not None)
        star_count = sum(1 for arg in node.args if arg.star == "*")
        starstar_count = sum(1 for arg in node.args if arg.star == "**")
        extras: dict[str, object] = {
            "kw_arg_count": kw_count,
            "star_arg_count": star_count,
            "starstar_arg_count": starstar_count,
        }
        caller_def_id = self._container_def_id(node)
        if caller_def_id is not None:
            extras["caller_def_id"] = caller_def_id
        qnames = self._qualified_name_payload(node)
        if qnames:
            extras["callee_qnames"] = qnames
        return extras or None

    @staticmethod
    def _import_extras(
        node: cst.CSTNode,
        *,
        import_kind: str,
        module: str | None,
        level: int | None,
        is_star: bool,
    ) -> dict[str, object] | None:
        extras: dict[str, object] = {
            "stmt_kind": import_kind,
            "is_star": is_star,
        }
        if isinstance(node, cst.ImportAlias):
            extras["imported"] = node.evaluated_name
            extras["asname"] = node.evaluated_alias
        if module is not None:
            extras["module"] = module
        if level is not None:
            extras["relative_level"] = level
        return extras or None

    def _referents_payload(self, access: Access) -> list[dict[str, object]]:
        payload: list[dict[str, object]] = []
        for referent in access.referents:
            entry: dict[str, object] = {
                "assignment_name": referent.name,
                "assignment_kind": type(referent).__name__,
            }
            node = getattr(referent, "node", None)
            if isinstance(node, cst.CSTNode):
                span = self._span_for_node(node)
                if span is not None:
                    entry["span_id"] = self._ensure_span(span, "referent")
                qnames = self._qualified_name_payload(node)
                if qnames:
                    entry["qnames"] = qnames
            payload.append(entry)
        return payload

    def _qualified_name_payload(self, node: cst.CSTNode | None) -> list[dict[str, str]]:
        if node is None:
            return []
        try:
            qnames = self.get_metadata(metadata.QualifiedNameProvider, node)
        except KeyError:
            return []
        if not isinstance(qnames, Iterable):
            return []
        payload: list[dict[str, str]] = []
        for qname in qnames:
            name = getattr(qname, "name", None)
            if not isinstance(name, str):
                continue
            entry: dict[str, str] = {"name": name}
            source = getattr(qname, "source", None)
            if source is not None:
                entry["source"] = str(source)
            payload.append(entry)
        return payload

    def _expression_context_name(self, node: cst.CSTNode) -> str | None:
        try:
            context = self.get_metadata(metadata.ExpressionContextProvider, node)
        except KeyError:
            return None
        if context is None:
            return None
        name = getattr(context, "name", None)
        if isinstance(name, str):
            return name.lower()
        return str(context)

    def _decorator_payload(self, decorators: Sequence[cst.Decorator] | None) -> list[str]:
        if not decorators:
            return []
        payload: list[str] = []
        for decorator in decorators:
            text = self._node_text(decorator.decorator)
            if text:
                payload.append(text)
        return payload

    @staticmethod
    def _docstring(node: cst.CSTNode) -> str | None:
        getter = getattr(node, "get_docstring", None)
        if callable(getter):
            docstring = getter(clean=True)
            if isinstance(docstring, str) and docstring:
                return docstring
        return None

    def _current_scope_kind(self) -> str | None:
        if not self._scope_stack:
            return None
        return self._scope_stack[-1].scope_kind

    @staticmethod
    def _iter_name_targets(target: cst.CSTNode) -> Iterable[cst.Name]:
        if isinstance(target, cst.Name):
            yield target
        elif isinstance(target, (cst.Tuple, cst.List)):
            for element in target.elements:
                yield from SyntaxFactsVisitor._iter_name_targets(element.value)
        elif isinstance(target, cst.StarredElement):
            yield from SyntaxFactsVisitor._iter_name_targets(target.value)

    @staticmethod
    def _relative_level(relative: Sequence[cst.CSTNode] | None) -> int | None:
        if relative is None:
            return None
        level = len(relative)
        if level == 0:
            return None
        return level


def _stable_id(*parts: object) -> str:
    payload = json.dumps(parts, separators=(",", ":"), ensure_ascii=False)
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()


def _decode_source_bytes(source_bytes: bytes) -> tuple[str, str]:
    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(source_bytes).readline)
    except (SyntaxError, LookupError):
        encoding = "utf-8"
    try:
        return source_bytes.decode(encoding), encoding
    except UnicodeDecodeError:
        return source_bytes.decode(encoding, errors="replace"), encoding


def _future_imports_payload(module: cst.Module) -> list[str] | None:
    future_imports = getattr(module, "future_imports", None)
    if not future_imports:
        return None
    return [str(entry) for entry in future_imports]


def _libcst_version() -> str | None:
    version = getattr(cst, "__version__", None)
    if isinstance(version, str):
        return version
    return None


def _build_access_map(scope_map: Mapping[cst.CSTNode, Scope | None]) -> dict[int, Access]:
    access_map: dict[int, Access] = {}
    for scope in {scope for scope in scope_map.values() if scope is not None}:
        for access in scope.accesses:
            node = access.node
            if isinstance(node, cst.CSTNode):
                access_map.setdefault(id(node), access)
    return access_map


def _build_syntax_node_index(
    node_rows: Sequence[Mapping[str, object]],
) -> _SyntaxNodeIndex:
    tree = IntervalTree()
    exact: dict[tuple[int, int], list[_SyntaxNodeCandidate]] = {}
    for order, row in enumerate(node_rows):
        start_byte = row.get("start_byte")
        end_byte = row.get("end_byte")
        node_id = row.get("node_id")
        node_kind = row.get("node_kind")
        if not isinstance(start_byte, int) or not isinstance(end_byte, int):
            continue
        if not isinstance(node_id, str) or not isinstance(node_kind, str):
            continue
        candidate = _SyntaxNodeCandidate(
            node_id=node_id,
            start_byte=start_byte,
            end_byte=end_byte,
            node_kind=node_kind,
            order=order,
        )
        exact.setdefault((start_byte, end_byte), []).append(candidate)
        if end_byte <= start_byte:
            continue
        tree.addi(start_byte, end_byte, candidate)
    return _SyntaxNodeIndex(tree=tree, exact=exact)


def _interval_candidates(intervals: Iterable[object]) -> list[_SyntaxNodeCandidate]:
    candidates: list[_SyntaxNodeCandidate] = []
    for interval in intervals:
        data = getattr(interval, "data", None)
        if isinstance(data, _SyntaxNodeCandidate):
            candidates.append(data)
    return candidates


def _pick_smallest_candidate(
    candidates: Iterable[_SyntaxNodeCandidate],
) -> _SyntaxNodeCandidate | None:
    candidate_list = list(candidates)
    if not candidate_list:
        return None
    return min(candidate_list, key=lambda item: (item.end_byte - item.start_byte, item.order))


def _match_exact(
    index: _SyntaxNodeIndex,
    span: AstSpan,
) -> tuple[str, str] | None:
    exact_matches = index.exact.get((span.start_byte, span.end_byte))
    candidate = _pick_smallest_candidate(exact_matches or [])
    if candidate is None:
        return None
    return candidate.node_id, "EXACT"


def _match_point(
    index: _SyntaxNodeIndex,
    span: AstSpan,
) -> tuple[str, str] | None:
    candidates = _interval_candidates(index.tree.at(span.start_byte))
    candidate = _pick_smallest_candidate(candidates)
    if candidate is not None:
        return candidate.node_id, "POINT"
    if span.start_byte > 0:
        candidates = _interval_candidates(index.tree.at(span.start_byte - 1))
        candidate = _pick_smallest_candidate(candidates)
        if candidate is not None:
            return candidate.node_id, "POINT_ADJACENT"
    return None


def _match_overlap(
    index: _SyntaxNodeIndex,
    span: AstSpan,
) -> tuple[str, str] | None:
    candidates = _interval_candidates(index.tree.overlap(span.start_byte, span.end_byte))
    containing = [
        candidate
        for candidate in candidates
        if candidate.start_byte <= span.start_byte and candidate.end_byte >= span.end_byte
    ]
    candidate = _pick_smallest_candidate(containing)
    if candidate is not None:
        return candidate.node_id, "CONTAINS"
    candidate = _pick_smallest_candidate(candidates)
    if candidate is not None:
        return candidate.node_id, "OVERLAP"
    return None


def _match_syntax_node(
    index: _SyntaxNodeIndex,
    span: AstSpan,
) -> tuple[str, str] | None:
    match = _match_exact(index, span)
    if match is None and span.start_byte == span.end_byte:
        match = _match_point(index, span)
    if match is None and span.start_byte != span.end_byte:
        match = _match_overlap(index, span)
    return match


def _ast_payload(record: AstNodeRecord, match_kind: str) -> dict[str, object]:
    payload: dict[str, object] = {
        "ast_node_id": record.node_id,
        "ast_kind": record.kind,
        "ast_start_line": record.span.start_line,
        "ast_start_col_utf8": record.span.start_col_utf8,
        "ast_end_line": record.span.end_line,
        "ast_end_col_utf8": record.span.end_col_utf8,
        "ast_start_byte": record.span.start_byte,
        "ast_end_byte": record.span.end_byte,
        "match_kind": match_kind,
    }
    if record.extras:
        payload.update(record.extras)
    return payload


def _merge_ast_into_syntax_nodes(
    syntax_nodes: list[dict[str, object]],
    ast_nodes: list[AstNodeRecord],
) -> None:
    if not syntax_nodes or not ast_nodes:
        return
    index = _build_syntax_node_index(syntax_nodes)
    by_node: dict[str, list[dict[str, object]]] = {}
    for record in ast_nodes:
        match = _match_syntax_node(index, record.span)
        if match is None:
            continue
        node_id, match_kind = match
        by_node.setdefault(node_id, []).append(_ast_payload(record, match_kind))
    if not by_node:
        return
    for payloads in by_node.values():
        payloads.sort(
            key=lambda item: (
                item["ast_start_byte"],
                item["ast_end_byte"],
                item["ast_kind"],
            )
        )
    for row in syntax_nodes:
        node_id = row.get("node_id")
        if not isinstance(node_id, str):
            continue
        payloads = by_node.get(node_id)
        if not payloads:
            continue
        extras = row.get("extras_json")
        merged = dict(extras) if isinstance(extras, dict) else {}
        merged["ast_nodes"] = payloads
        row["extras_json"] = merged


def _parse_manifest_row(
    context: _ParseManifestContext,
    *,
    parse_ok: bool,
    error: Exception | None,
) -> dict[str, object]:
    error_kind = None
    error_message = None
    error_line = None
    error_col = None
    error_snippet = None

    if error is not None:
        error_kind = type(error).__name__
        error_message = str(error)
        if isinstance(error, cst.ParserSyntaxError):
            raw_line = getattr(error, "raw_line", None)
            raw_column = getattr(error, "raw_column", None)
            if isinstance(raw_line, int) and raw_line > 0:
                error_line = raw_line - 1
            if isinstance(raw_column, int) and raw_column >= 0:
                error_col = raw_column
            if error_line is not None:
                error_snippet = context.source_index.line_snippet(error_line)

    return {
        "repo": context.repo,
        "commit": context.commit,
        "rel_path": context.rel_path,
        "producer": context.producer,
        "parse_ok": parse_ok,
        "encoding": context.encoding,
        "default_indent": context.default_indent,
        "default_newline": context.default_newline,
        "has_trailing_newline": context.has_trailing_newline,
        "future_imports": context.future_imports,
        "parser_backend": context.parser_backend,
        "libcst_version": context.libcst_version,
        "error_kind": error_kind,
        "error_message": error_message,
        "error_line": error_line,
        "error_col": error_col,
        "error_snippet": error_snippet,
    }


def _build_cst_buffers() -> _CstBuffers:
    return _CstBuffers(
        cst=columnar_buffer_for_table_key(CST_NODES_TABLE_KEY),
        parse_manifest=columnar_buffer_for_table_key(PARSE_MANIFEST_TABLE_KEY),
        spans=columnar_buffer_for_table_key(SYNTAX_SPANS_TABLE_KEY),
        syntax_nodes=columnar_buffer_for_table_key(SYNTAX_NODES_TABLE_KEY),
        syntax_edges=columnar_buffer_for_table_key(SYNTAX_EDGES_TABLE_KEY),
        scopes=columnar_buffer_for_table_key(SYNTAX_SCOPES_TABLE_KEY),
        defs=columnar_buffer_for_table_key(SYNTAX_DEFS_TABLE_KEY),
        refs=columnar_buffer_for_table_key(SYNTAX_REFS_TABLE_KEY),
        calls=columnar_buffer_for_table_key(SYNTAX_CALLS_TABLE_KEY),
        call_args=columnar_buffer_for_table_key(SYNTAX_CALL_ARGS_TABLE_KEY),
        func_params=columnar_buffer_for_table_key(SYNTAX_FUNC_PARAMS_TABLE_KEY),
        imports=columnar_buffer_for_table_key(SYNTAX_IMPORTS_TABLE_KEY),
    )


def _flush_cst_rows(buffers: _CstBuffers, rows: Iterable[CstRow]) -> None:
    for rel_path, node_id, kind, span, snippet, parents, qnames in rows:
        buffers.cst.append(
            {
                "path": rel_path,
                "node_id": node_id,
                "kind": kind,
                "span": span,
                "text_preview": snippet,
                "parents": list(parents),
                "qnames": list(qnames),
            }
        )


def _build_manifest_context(
    context: _SyntaxContext,
    *,
    source_index: LineIndexedSource,
    encoding: str,
    libcst_version: str | None,
) -> _ParseManifestContext:
    return _ParseManifestContext(
        repo=context.repo,
        commit=context.commit,
        rel_path=context.rel_path,
        producer=context.producer,
        source_index=source_index,
        encoding=encoding,
        default_indent=None,
        default_newline=None,
        has_trailing_newline=None,
        future_imports=None,
        parser_backend=context.producer,
        libcst_version=libcst_version,
    )


def _build_source_index(source_bytes: bytes) -> tuple[LineIndexedSource, str]:
    source_text, encoding = _decode_source_bytes(source_bytes)
    source_index = LineIndexedSource(source_text, source_bytes, encoding=encoding)
    return source_index, encoding


def _parse_module_context(
    *,
    context: _SyntaxContext,
    source_bytes: bytes,
    buffers: _CstBuffers,
    warnings: list[str],
) -> _ParsedCstModule | None:
    source_index, encoding = _build_source_index(source_bytes)
    libcst_version = _libcst_version()
    manifest_context = _build_manifest_context(
        context,
        source_index=source_index,
        encoding=encoding,
        libcst_version=libcst_version,
    )
    try:
        parsed_module = cst.parse_module(source_bytes)
    except (cst.ParserSyntaxError, ValueError, TypeError, RuntimeError) as exc:
        buffers.parse_manifest.append(
            _parse_manifest_row(
                manifest_context,
                parse_ok=False,
                error=exc,
            )
        )
        message = f"Failed to parse {context.rel_path}: {exc}"
        warnings.append(message)
        log.warning("%s", message)
        return None
    source_text = parsed_module.code
    encoding = parsed_module.encoding if isinstance(parsed_module.encoding, str) else encoding
    source_index = LineIndexedSource(source_text, source_bytes, encoding=encoding)
    manifest_context = _ParseManifestContext(
        repo=context.repo,
        commit=context.commit,
        rel_path=context.rel_path,
        producer=context.producer,
        source_index=source_index,
        encoding=encoding,
        default_indent=parsed_module.default_indent,
        default_newline=parsed_module.default_newline,
        has_trailing_newline=parsed_module.has_trailing_newline,
        future_imports=_future_imports_payload(parsed_module),
        parser_backend=context.producer,
        libcst_version=libcst_version,
    )
    wrapper = metadata.MetadataWrapper(parsed_module, unsafe_skip_copy=True)
    buffers.parse_manifest.append(
        _parse_manifest_row(
            manifest_context,
            parse_ok=True,
            error=None,
        )
    )
    return _ParsedCstModule(
        parsed_module=parsed_module,
        source_text=source_text,
        source_index=source_index,
        encoding=encoding,
        manifest_context=manifest_context,
        wrapper=wrapper,
    )


def _resolve_scope_map(
    wrapper: metadata.MetadataWrapper,
    *,
    rel_path: str,
    warnings: list[str],
) -> Mapping[cst.CSTNode, Scope | None]:
    try:
        return wrapper.resolve(metadata.ScopeProvider)
    except (ValueError, TypeError, RuntimeError) as exc:
        message = f"Scope metadata failed for {rel_path}: {exc}"
        warnings.append(message)
        log.warning("%s", message)
        return {}


def _extract_module_syntax(
    *,
    module: ModuleRecord,
    context: _SyntaxContext,
    source_bytes: bytes,
    buffers: _CstBuffers,
    emit_ast_nodes: bool,
) -> list[str]:
    warnings: list[str] = []
    parsed_context = _parse_module_context(
        context=context,
        source_bytes=source_bytes,
        buffers=buffers,
        warnings=warnings,
    )
    if parsed_context is None:
        return warnings

    cst_visitor = CstVisitor(
        rel_path=context.rel_path,
        module_name=module.module_name,
        source=SourceBundle(
            text=parsed_context.source_text,
            source_bytes=source_bytes,
            encoding=parsed_context.encoding,
        ),
    )
    syntax_graph_visitor = SyntaxGraphVisitor(
        context=context,
        source_index=parsed_context.source_index,
        snippet_limit=CST_CAPTURE_CONFIG.snippet_limit,
    )
    scope_map = _resolve_scope_map(
        parsed_context.wrapper,
        rel_path=context.rel_path,
        warnings=warnings,
    )
    syntax_visitor = SyntaxFactsVisitor(
        context=context,
        source_index=parsed_context.source_index,
        access_map=_build_access_map(scope_map) if scope_map else {},
    )

    def _ast_node_id(node: object, span: AstSpan) -> str:
        return _stable_id(
            "ast_node",
            context.repo,
            context.commit,
            context.rel_path,
            type(node).__name__,
            span.start_line,
            span.start_col_utf8,
            span.end_line,
            span.end_col_utf8,
            span.start_byte,
            span.end_byte,
        )

    ast_nodes = (
        collect_ast_nodes(
            parsed_context.source_text,
            parsed_context.source_index,
            node_id_factory=_ast_node_id,
            context=AstCollectContext(
                warnings=warnings,
                source_label=context.rel_path,
            ),
        )
        if emit_ast_nodes
        else []
    )
    try:
        parsed_context.wrapper.visit(cst_visitor)
        parsed_context.wrapper.visit(syntax_graph_visitor)
        parsed_context.wrapper.visit(syntax_visitor)
    except (ValueError, TypeError, RuntimeError) as exc:
        message = f"Failed to extract syntax for {context.rel_path}: {exc}"
        warnings.append(message)
        log.warning("%s", message)
        return warnings

    if emit_ast_nodes and ast_nodes:
        _merge_ast_into_syntax_nodes(syntax_graph_visitor.node_rows, ast_nodes)

    _flush_cst_rows(buffers, cst_visitor.rows)

    buffers.spans.extend(syntax_visitor.span_rows)
    buffers.syntax_nodes.extend(syntax_graph_visitor.node_rows)
    buffers.syntax_edges.extend(syntax_graph_visitor.edge_rows)
    buffers.scopes.extend(syntax_visitor.scopes)
    buffers.defs.extend(syntax_visitor.defs)
    buffers.refs.extend(syntax_visitor.refs)
    buffers.calls.extend(syntax_visitor.calls)
    buffers.call_args.extend(syntax_visitor.call_args)
    buffers.func_params.extend(syntax_visitor.func_params)
    buffers.imports.extend(syntax_visitor.imports)
    return warnings


class CstExtractStep(BaseExtractStep):
    """CST extraction step with port injection.

    This step extracts LibCST concrete syntax trees from modules,
    using ports for all I/O operations.

    Parameters
    ----------
    discovery
        Discovery port for reading module source.
    emit_ast_nodes
        Whether to merge CPython AST facts into syntax nodes.
    """

    def __init__(
        self,
        discovery: ModuleDiscoveryPort,
        *,
        emit_ast_nodes: bool = True,
    ) -> None:
        super().__init__(discovery)
        self._emit_ast_nodes = emit_ast_nodes

    def execute(
        self,
        modules: Sequence[ModuleRecord],
        *,
        repo: str,
        commit: str,
    ) -> CstExtractResult:
        """Execute CST extraction on the provided modules.

        Parameters
        ----------
        modules
            Modules to process.
        repo
            Repository identifier.
        commit
            Commit identifier.

        Returns
        -------
        CstExtractResult
            Result bundle with row tuples and execution status.
        """
        try:
            buffers = _build_cst_buffers()
        except (KeyError, RuntimeError) as exc:
            return CstExtractResult(result=ExecutionResult.failed(str(exc)))

        warnings: list[str] = []

        for module, source_bytes in self._iter_python_source_bytes(modules):
            context = _SyntaxContext(
                repo=repo,
                commit=commit,
                rel_path=module.rel_path,
                producer=SYNTAX_PRODUCER,
                language=SYNTAX_LANGUAGE,
            )
            warnings.extend(
                _extract_module_syntax(
                    module=module,
                    context=context,
                    source_bytes=source_bytes,
                    buffers=buffers,
                    emit_ast_nodes=self._emit_ast_nodes,
                )
            )

        log.info(
            "CST extraction: repo=%s commit=%s rows=%d",
            repo,
            commit,
            buffers.cst.row_count,
        )

        return CstExtractResult(
            result=ExecutionResult.ok(warnings=tuple(warnings)),
            rows=buffers.cst.data,
            parse_manifest_rows=buffers.parse_manifest.data,
            syntax_spans_rows=buffers.spans.data,
            syntax_nodes_rows=buffers.syntax_nodes.data,
            syntax_edges_rows=buffers.syntax_edges.data,
            syntax_scopes_rows=buffers.scopes.data,
            syntax_defs_rows=buffers.defs.data,
            syntax_refs_rows=buffers.refs.data,
            syntax_calls_rows=buffers.calls.data,
            syntax_call_args_rows=buffers.call_args.data,
            syntax_func_params_rows=buffers.func_params.data,
            syntax_imports_rows=buffers.imports.data,
            row_count=buffers.cst.row_count,
            parse_manifest_row_count=buffers.parse_manifest.row_count,
            syntax_spans_row_count=buffers.spans.row_count,
            syntax_nodes_row_count=buffers.syntax_nodes.row_count,
            syntax_edges_row_count=buffers.syntax_edges.row_count,
            syntax_scopes_row_count=buffers.scopes.row_count,
            syntax_defs_row_count=buffers.defs.row_count,
            syntax_refs_row_count=buffers.refs.row_count,
            syntax_calls_row_count=buffers.calls.row_count,
            syntax_call_args_row_count=buffers.call_args.row_count,
            syntax_func_params_row_count=buffers.func_params.row_count,
            syntax_imports_row_count=buffers.imports.row_count,
        )

    def _iter_python_source_bytes(
        self,
        modules: Sequence[ModuleRecord],
    ) -> Iterable[tuple[ModuleRecord, bytes]]:
        for module in modules:
            if not module.rel_path.endswith(".py"):
                continue
            source_bytes = self._discovery.read_module_bytes(module)
            if source_bytes is not None:
                yield module, source_bytes


__all__ = [
    "CST_CAPTURE_CONFIG",
    "CstExtractResult",
    "CstExtractStep",
    "CstVisitor",
]
