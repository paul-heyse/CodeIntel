"""CST extraction step with port injection.

This module provides a pure domain logic implementation for extracting
LibCST concrete syntax trees, using ports for all I/O operations.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import libcst as cst
from libcst import metadata

from codeintel.build.hamilton.execution_result import ExecutionResult
from codeintel.core.columnar.rows import ColumnarRows, columnar_buffer_for_table_key
from codeintel.ingestion.compute.base import BaseExtractStep
from codeintel.ingestion.infrastructure.cst_utils import (
    CstCaptureConfig,
    CstCaptureVisitor,
    LineIndexedSource,
    NormalizedSpan,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from codeintel.ingestion.ports.discovery import ModuleRecord

log = logging.getLogger(__name__)

CST_NODES_TABLE_KEY = "core.cst_nodes"
PARSE_MANIFEST_TABLE_KEY = "core.parse_manifest"
SYNTAX_SPANS_TABLE_KEY = "core.syntax_spans"
SYNTAX_SCOPES_TABLE_KEY = "core.syntax_scopes"
SYNTAX_DEFS_TABLE_KEY = "core.syntax_defs"
SYNTAX_REFS_TABLE_KEY = "core.syntax_refs"
SYNTAX_CALLS_TABLE_KEY = "core.syntax_calls"
SYNTAX_IMPORTS_TABLE_KEY = "core.syntax_imports"

SYNTAX_PRODUCER = "libcst"

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
    syntax_scopes_rows: ColumnarRows = field(default_factory=dict)
    syntax_defs_rows: ColumnarRows = field(default_factory=dict)
    syntax_refs_rows: ColumnarRows = field(default_factory=dict)
    syntax_calls_rows: ColumnarRows = field(default_factory=dict)
    syntax_imports_rows: ColumnarRows = field(default_factory=dict)
    row_count: int = 0
    parse_manifest_row_count: int = 0
    syntax_spans_row_count: int = 0
    syntax_scopes_row_count: int = 0
    syntax_defs_row_count: int = 0
    syntax_refs_row_count: int = 0
    syntax_calls_row_count: int = 0
    syntax_imports_row_count: int = 0


@dataclass(frozen=True)
class ScopeFrame:
    """Tracking state for nested scopes."""

    scope_id: str
    scope_kind: str


class CstVisitor(CstCaptureVisitor):
    """Collect CST rows using shared capture helpers."""

    def __init__(self, rel_path: str, module_name: str, source: str, source_bytes: bytes) -> None:
        """Initialize visitor.

        Parameters
        ----------
        rel_path
            Relative path to the file.
        module_name
            Python module name.
        source
            Source code text.
        source_bytes
            UTF-8 encoded source bytes.
        """
        super().__init__(
            rel_path,
            module_name,
            source,
            config=CST_CAPTURE_CONFIG,
            source_bytes=source_bytes,
        )


class SyntaxFactsVisitor(cst.CSTVisitor):
    """Collect syntax fact tables from a LibCST module."""

    METADATA_DEPENDENCIES = (
        metadata.PositionProvider,
        metadata.ParentNodeProvider,
    )

    def __init__(
        self,
        *,
        repo: str,
        commit: str,
        rel_path: str,
        producer: str,
        source_index: LineIndexedSource,
    ) -> None:
        self.repo = repo
        self.commit = commit
        self.rel_path = rel_path
        self.producer = producer
        self.source_index = source_index

        self.span_rows: list[dict[str, object]] = []
        self.scopes: list[dict[str, object]] = []
        self.defs: list[dict[str, object]] = []
        self.refs: list[dict[str, object]] = []
        self.calls: list[dict[str, object]] = []
        self.imports: list[dict[str, object]] = []
        self._span_ids: set[str] = set()
        self._scope_stack: list[ScopeFrame] = []

    def visit_Module(self, node: cst.Module) -> bool:
        self._enter_scope(node, "module")
        return True

    def leave_Module(self, original_node: cst.Module) -> None:
        self._exit_scope()

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        self._record_named_def(node.name, node.name.value, "class")
        self._enter_scope(node, "class")
        return True

    def leave_ClassDef(self, original_node: cst.ClassDef) -> None:
        self._exit_scope()

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        self._record_named_def(node.name, node.name.value, "function")
        self._enter_scope(node, "function")
        self._record_params(node.params)
        return True

    def leave_FunctionDef(self, original_node: cst.FunctionDef) -> None:
        self._exit_scope()

    def visit_AsyncFunctionDef(self, node: cst.AsyncFunctionDef) -> bool:
        self._record_named_def(node.name, node.name.value, "async_function")
        self._enter_scope(node, "async_function")
        self._record_params(node.params)
        return True

    def leave_AsyncFunctionDef(self, original_node: cst.AsyncFunctionDef) -> None:
        self._exit_scope()

    def visit_Lambda(self, node: cst.Lambda) -> bool:
        self._enter_scope(node, "lambda")
        self._record_params(node.params)
        return True

    def leave_Lambda(self, original_node: cst.Lambda) -> None:
        self._exit_scope()

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
        self._record_ref(node, node.value, ref_kind="identifier", span_kind="identifier")
        return True

    def visit_Attribute(self, node: cst.Attribute) -> bool:
        if self._is_assignment_target(node):
            return True
        name = self._node_text(node)
        if name is None:
            return True
        self._record_ref(node, name, ref_kind="attribute", span_kind="attribute")
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
        for param in self._iter_params(params):
            self._record_named_def(param.name, param.name.value, "param")

    @staticmethod
    def _iter_params(params: cst.Parameters) -> Iterable[cst.Param]:
        yield from params.posonly_params
        yield from params.params
        yield from params.kwonly_params
        if isinstance(params.star_arg, cst.Param):
            yield params.star_arg
        if isinstance(params.star_kwarg, cst.Param):
            yield params.star_kwarg

    def _record_named_def(
        self,
        node: cst.CSTNode,
        name: str,
        def_kind: str,
        *,
        span_kind: str = "identifier",
    ) -> None:
        scope_id = self._current_scope_id()
        if scope_id is None:
            return
        span = self._span_for_node(node)
        if span is None:
            return
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
            }
        )

    def _record_ref(
        self,
        node: cst.CSTNode,
        name: str,
        *,
        ref_kind: str,
        span_kind: str,
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
        callee_span_id = None
        callee_text = self._node_text(node.func)
        callee_span = self._span_for_node(node.func)
        if callee_span is not None:
            callee_span_id = self._ensure_span(callee_span, "callee_expr")
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
        self.calls.append(
            {
                "repo": self.repo,
                "commit": self.commit,
                "rel_path": self.rel_path,
                "producer": self.producer,
                "call_id": call_id,
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
            }
        )

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
        alias_text = alias.asname.name.value if alias.asname is not None else None
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
            return alias.asname.name, alias.asname.name.value
        name = self._binding_name_from_node(alias.name)
        return alias.name, name

    @staticmethod
    def _binding_name_from_node(node: cst.CSTNode) -> str | None:
        if isinstance(node, cst.Name):
            return node.value
        if isinstance(node, cst.Attribute):
            value = node
            while isinstance(value, cst.Attribute):
                value = value.value
            if isinstance(value, cst.Name):
                return value.value
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
        return self.source_index.span_from_range(pos)

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
            return self.get_metadata(metadata.ParentNodeProvider, node)
        except KeyError:
            return None

    def _is_definition_name(self, node: cst.Name) -> bool:
        parent = self._parent(node)
        if parent is None:
            return False
        if isinstance(parent, (cst.FunctionDef, cst.ClassDef, cst.Param)):
            return parent.name is node
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


def _parse_manifest_row(
    *,
    repo: str,
    commit: str,
    rel_path: str,
    producer: str,
    parse_ok: bool,
    error: Exception | None,
    source_index: LineIndexedSource,
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
                error_snippet = source_index.line_snippet(error_line)

    return {
        "repo": repo,
        "commit": commit,
        "rel_path": rel_path,
        "producer": producer,
        "parse_ok": parse_ok,
        "error_kind": error_kind,
        "error_message": error_message,
        "error_line": error_line,
        "error_col": error_col,
        "error_snippet": error_snippet,
    }


class CstExtractStep(BaseExtractStep):
    """CST extraction step with port injection.

    This step extracts LibCST concrete syntax trees from modules,
    using ports for all I/O operations.

    Parameters
    ----------
    discovery
        Discovery port for reading module source.
    """

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
            cst_buffer = columnar_buffer_for_table_key(CST_NODES_TABLE_KEY)
            parse_manifest_buffer = columnar_buffer_for_table_key(PARSE_MANIFEST_TABLE_KEY)
            spans_buffer = columnar_buffer_for_table_key(SYNTAX_SPANS_TABLE_KEY)
            scopes_buffer = columnar_buffer_for_table_key(SYNTAX_SCOPES_TABLE_KEY)
            defs_buffer = columnar_buffer_for_table_key(SYNTAX_DEFS_TABLE_KEY)
            refs_buffer = columnar_buffer_for_table_key(SYNTAX_REFS_TABLE_KEY)
            calls_buffer = columnar_buffer_for_table_key(SYNTAX_CALLS_TABLE_KEY)
            imports_buffer = columnar_buffer_for_table_key(SYNTAX_IMPORTS_TABLE_KEY)
        except (KeyError, RuntimeError) as exc:
            return CstExtractResult(result=ExecutionResult.failed(str(exc)))

        warnings: list[str] = []

        for module, source in self._iter_python_sources(modules):
            source_bytes = source.encode("utf-8")
            source_index = LineIndexedSource(source, source_bytes)
            try:
                wrapper = metadata.MetadataWrapper(
                    cst.parse_module(source),
                    unsafe_skip_copy=True,
                )
            except (cst.ParserSyntaxError, ValueError, TypeError, RuntimeError) as exc:
                parse_manifest_buffer.append(
                    _parse_manifest_row(
                        repo=repo,
                        commit=commit,
                        rel_path=module.rel_path,
                        producer=SYNTAX_PRODUCER,
                        parse_ok=False,
                        error=exc,
                        source_index=source_index,
                    )
                )
                message = f"Failed to parse {module.rel_path}: {exc}"
                warnings.append(message)
                log.warning("%s", message)
                continue

            parse_manifest_buffer.append(
                _parse_manifest_row(
                    repo=repo,
                    commit=commit,
                    rel_path=module.rel_path,
                    producer=SYNTAX_PRODUCER,
                    parse_ok=True,
                    error=None,
                    source_index=source_index,
                )
            )

            cst_visitor = CstVisitor(
                rel_path=module.rel_path,
                module_name=module.module_name,
                source=source,
                source_bytes=source_bytes,
            )
            syntax_visitor = SyntaxFactsVisitor(
                repo=repo,
                commit=commit,
                rel_path=module.rel_path,
                producer=SYNTAX_PRODUCER,
                source_index=source_index,
            )
            try:
                wrapper.visit(cst_visitor)
                wrapper.visit(syntax_visitor)
            except (ValueError, TypeError, RuntimeError) as exc:
                message = f"Failed to extract syntax for {module.rel_path}: {exc}"
                warnings.append(message)
                log.warning("%s", message)
                continue

            for row in cst_visitor.rows:
                rel_path, node_id, kind, span, snippet, parents, qnames = row
                cst_buffer.append(
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

            spans_buffer.extend(syntax_visitor.span_rows)
            scopes_buffer.extend(syntax_visitor.scopes)
            defs_buffer.extend(syntax_visitor.defs)
            refs_buffer.extend(syntax_visitor.refs)
            calls_buffer.extend(syntax_visitor.calls)
            imports_buffer.extend(syntax_visitor.imports)

        log.info(
            "CST extraction: repo=%s commit=%s rows=%d",
            repo,
            commit,
            cst_buffer.row_count,
        )

        return CstExtractResult(
            result=ExecutionResult.ok(warnings=tuple(warnings)),
            rows=cst_buffer.data,
            parse_manifest_rows=parse_manifest_buffer.data,
            syntax_spans_rows=spans_buffer.data,
            syntax_scopes_rows=scopes_buffer.data,
            syntax_defs_rows=defs_buffer.data,
            syntax_refs_rows=refs_buffer.data,
            syntax_calls_rows=calls_buffer.data,
            syntax_imports_rows=imports_buffer.data,
            row_count=cst_buffer.row_count,
            parse_manifest_row_count=parse_manifest_buffer.row_count,
            syntax_spans_row_count=spans_buffer.row_count,
            syntax_scopes_row_count=scopes_buffer.row_count,
            syntax_defs_row_count=defs_buffer.row_count,
            syntax_refs_row_count=refs_buffer.row_count,
            syntax_calls_row_count=calls_buffer.row_count,
            syntax_imports_row_count=imports_buffer.row_count,
        )


__all__ = [
    "CST_CAPTURE_CONFIG",
    "CstExtractResult",
    "CstExtractStep",
    "CstVisitor",
]
