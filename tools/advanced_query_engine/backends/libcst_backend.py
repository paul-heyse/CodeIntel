"""LibCST-backed definition indexing utilities."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import PurePosixPath

import libcst as cst
from intervaltree import Interval, IntervalTree
from libcst.metadata import ByteSpanPositionProvider, MetadataWrapper

from tools.advanced_query_engine.contracts import Span


@dataclass(frozen=True)
class DefRecord:
    """Definition record extracted from LibCST."""

    path: str
    kind: str
    name: str
    qname: str
    span: Span
    container_qname: str | None
    signature: str | None
    docstring: str | None


@dataclass
class LibCSTIndex:
    """Index of definitions for a Python source file."""

    module_qname: str
    path: str
    defs: list[DefRecord]
    name_map: dict[str, list[DefRecord]]
    _span_tree: IntervalTree | None = None

    def span_tree(self) -> IntervalTree:
        """Return an interval tree for definition spans.

        Returns
        -------
        IntervalTree
            Interval tree for definition spans.
        """
        if self._span_tree is None:
            intervals = [
                Interval(
                    record.span.start_byte,
                    record.span.end_byte,
                    record,
                )
                for record in self.defs
            ]
            self._span_tree = IntervalTree(intervals)
        return self._span_tree

    def enclosing_def(self, byte_offset: int) -> DefRecord | None:
        """Return the smallest enclosing definition for a byte offset.

        Returns
        -------
        DefRecord | None
            The smallest definition that contains the byte offset.
        """
        hits = self.span_tree().at(byte_offset)
        if not hits:
            return None
        best = min(hits, key=lambda iv: iv.end - iv.begin)
        if isinstance(best.data, DefRecord):
            return best.data
        return None

    def by_name(self, name: str) -> list[DefRecord]:
        """Return definitions matching a name.

        Returns
        -------
        list[DefRecord]
            Definitions that share the provided name.
        """
        return self.name_map.get(name, [])


def path_to_module_qname(rel_path: str) -> str:
    """Convert a repo-relative Python file path to a module qualified name.

    Returns
    -------
    str
        Module qualified name derived from the path.
    """
    path = PurePosixPath(rel_path)
    if path.suffix == ".py":
        path = path.with_suffix("")
    parts = list(path.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def build_def_index(rel_path: str, source_bytes: bytes) -> LibCSTIndex:
    """Build a LibCST definition index for a Python file.

    Returns
    -------
    LibCSTIndex
        Definition index for the file.

    Raises
    ------
    ValueError
        If the LibCST parser fails on the file.
    """
    module_qname = path_to_module_qname(rel_path)
    try:
        module = cst.parse_module(source_bytes)
    except cst.ParserSyntaxError as exc:
        msg = f"LibCST parse failed for {rel_path}"
        raise ValueError(msg) from exc
    wrapper = MetadataWrapper(module)
    collector = _DefCollector(module=module, module_qname=module_qname, path=rel_path)
    wrapper.visit(collector)

    name_map: dict[str, list[DefRecord]] = {}
    for record in collector.defs:
        name_map.setdefault(record.name, []).append(record)

    for records in name_map.values():
        records.sort(key=lambda item: (item.span.start_byte, item.span.end_byte, item.qname))
    collector.defs.sort(key=lambda item: (item.span.start_byte, item.span.end_byte, item.qname))

    return LibCSTIndex(
        module_qname=module_qname,
        path=rel_path,
        defs=collector.defs,
        name_map=name_map,
    )


class _DefCollector(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (ByteSpanPositionProvider,)

    def __init__(self, *, module: cst.Module, module_qname: str, path: str) -> None:
        self._module = module
        self._module_qname = module_qname
        self._path = path
        self._stack: list[str] = []
        self.defs: list[DefRecord] = []

    def visit_ClassDef(self, node: cst.ClassDef) -> bool:
        self._add_class(node)
        self._stack.append(node.name.value)
        return True

    def leave_ClassDef(self, _node: cst.ClassDef) -> None:
        if self._stack:
            self._stack.pop()

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        self._add_function(node)
        self._stack.append(node.name.value)
        return True

    def leave_FunctionDef(self, _node: cst.FunctionDef) -> None:
        if self._stack:
            self._stack.pop()

    def _current_container_qname(self) -> str | None:
        parts = [self._module_qname, *self._stack]
        normalized = [part for part in parts if part]
        return ".".join(normalized) if normalized else None

    def _make_qname(self, name: str) -> str:
        parts = [self._module_qname, *self._stack, name]
        return ".".join([part for part in parts if part])

    def _span_for(self, node: cst.CSTNode) -> Span:
        span = self.get_metadata(ByteSpanPositionProvider, node)
        return Span(
            path=self._path,
            start_byte=int(span.start),
            end_byte=int(span.start + span.length),
        )

    def _add_class(self, node: cst.ClassDef) -> None:
        span = self._span_for(node)
        qname = self._make_qname(node.name.value)
        container = self._current_container_qname()
        signature = _class_signature(self._module, node)
        docstring = _extract_docstring(node)
        self.defs.append(
            DefRecord(
                path=self._path,
                kind="class",
                name=node.name.value,
                qname=qname,
                span=span,
                container_qname=container,
                signature=signature,
                docstring=docstring,
            )
        )

    def _add_function(self, node: cst.FunctionDef) -> None:
        span = self._span_for(node)
        qname = self._make_qname(node.name.value)
        container = self._current_container_qname()
        kind = "method" if self._stack else "function"
        signature = _function_signature(self._module, node)
        docstring = _extract_docstring(node)
        self.defs.append(
            DefRecord(
                path=self._path,
                kind=kind,
                name=node.name.value,
                qname=qname,
                span=span,
                container_qname=container,
                signature=signature,
                docstring=docstring,
            )
        )


def _extract_docstring(node: cst.CSTNode) -> str | None:
    value: str | None = None
    body = getattr(node, "body", None)
    statements = getattr(body, "body", None) if body is not None else None
    if isinstance(statements, list) and statements:
        first = statements[0]
        if isinstance(first, cst.SimpleStatementLine) and len(first.body) == 1:
            expr = first.body[0]
            if isinstance(expr, cst.Expr) and isinstance(expr.value, cst.SimpleString):
                try:
                    parsed = ast.literal_eval(expr.value.value)
                except (ValueError, SyntaxError):
                    parsed = None
                if isinstance(parsed, str):
                    value = parsed
    return value


def _function_signature(module: cst.Module, node: cst.FunctionDef) -> str:
    params_text = module.code_for_node(node.params)
    signature = f"def {node.name.value}{params_text}"
    if node.returns is not None:
        return_text = module.code_for_node(node.returns.annotation)
        signature = f"{signature} -> {return_text}"
    return signature


def _class_signature(module: cst.Module, node: cst.ClassDef) -> str:
    bases = [module.code_for_node(arg.value) for arg in node.bases]
    if bases:
        return f"class {node.name.value}({', '.join(bases)})"
    return f"class {node.name.value}"


__all__ = ["DefRecord", "LibCSTIndex", "build_def_index", "path_to_module_qname"]
