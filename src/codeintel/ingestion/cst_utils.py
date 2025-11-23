"""Shared CST utilities for ingestion visitors."""

from __future__ import annotations

from dataclasses import dataclass

import libcst as cst
from libcst import metadata


@dataclass(frozen=True)
class CstCaptureConfig:
    """Configuration for CST capture."""

    kinds: tuple[type[cst.CSTNode], ...]
    snippet_limit: int = 200


class LineIndexedSource:
    """Precompute line offsets for efficient span slicing."""

    def __init__(self, source: str) -> None:
        self.source = source
        self.line_offsets: list[int] = []
        offset = 0
        for line in source.splitlines(keepends=True):
            self.line_offsets.append(offset)
            offset += len(line)

    def slice(self, start_line: int, start_col: int, end_line: int, end_col: int) -> str:
        """
        Return substring for a span; empty string on bounds errors.

        Returns
        -------
        str
            Extracted substring or empty string when indices are invalid.
        """
        try:
            start_idx = self.line_offsets[start_line - 1] + start_col
            end_idx = self.line_offsets[end_line - 1] + end_col
            return self.source[start_idx:end_idx]
        except (IndexError, ValueError):
            return ""


class CstCaptureVisitor(cst.CSTVisitor):
    """Reusable visitor that records CST node rows."""

    METADATA_DEPENDENCIES = (metadata.PositionProvider,)

    def __init__(
        self,
        rel_path: str,
        module_name: str,
        source: str,
        config: CstCaptureConfig,
    ) -> None:
        self.rel_path = rel_path
        self.module_name = module_name
        self.source_index = LineIndexedSource(source)
        self.config = config

        self.rows: list[
            tuple[str, str, str, dict[str, list[int]], str, tuple[str, ...], tuple[str, ...]]
        ] = []
        self._seen_ids: set[str] = set()
        self._scope_stack: list[str] = []
        self._parent_kinds: list[str] = []

    def on_visit(self, node: cst.CSTNode) -> bool:
        """
        Handle pre-visit bookkeeping and record CST row.

        Returns
        -------
        bool
            Always True to continue traversal.
        """
        kind = type(node).__name__
        self._parent_kinds.append(kind)
        if isinstance(
            node,
            (cst.ClassDef, getattr(cst, "AsyncFunctionDef", cst.FunctionDef), cst.FunctionDef),
        ):
            name_node = getattr(node, "name", None)
            if name_node is not None and hasattr(name_node, "value"):
                self._scope_stack.append(name_node.value)
        self._record(node, kind)
        return True

    def on_leave(self, original_node: cst.CSTNode) -> None:
        """Pop scope tracking on exit."""
        if (
            isinstance(
                original_node,
                (cst.ClassDef, getattr(cst, "AsyncFunctionDef", cst.FunctionDef), cst.FunctionDef),
            )
            and self._scope_stack
        ):
            self._scope_stack.pop()
        self._parent_kinds.pop()

    def _record(self, node: cst.CSTNode, kind: str) -> None:
        if not isinstance(node, self.config.kinds):
            return
        try:
            pos = self.get_metadata(metadata.PositionProvider, node)
        except KeyError:
            return
        if not isinstance(pos, metadata.CodeRange):
            return

        start = pos.start
        end = pos.end
        span = {"start": [start.line, start.column], "end": [end.line, end.column]}
        snippet = self.source_index.slice(start.line, start.column, end.line, end.column)

        parents = tuple(self._parent_kinds[:-1])
        qnames = (self._current_qualname(),)
        node_id = f"{self.rel_path}:{kind}:{start.line}:{start.column}:{end.line}:{end.column}"

        if node_id in self._seen_ids:
            return
        self._seen_ids.add(node_id)

        self.rows.append(
            (
                self.rel_path,
                node_id,
                kind,
                span,
                snippet[: self.config.snippet_limit],
                parents,
                qnames,
            )
        )

    def _current_qualname(self) -> str:
        if not self._scope_stack:
            return self.module_name
        return f"{self.module_name}." + ".".join(self._scope_stack)
