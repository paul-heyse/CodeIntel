"""Shared CST utilities for ingestion visitors."""

from __future__ import annotations

from dataclasses import dataclass

import libcst as cst
from libcst import metadata

from codeintel.core.spans import normalize_byte_span


@dataclass(frozen=True)
class CstCaptureConfig:
    """Configuration for CST capture."""

    kinds: tuple[type[cst.CSTNode], ...]
    snippet_limit: int = 200


@dataclass(frozen=True)
class NormalizedSpan:
    """Normalized span with 0-based line/column and optional byte offsets."""

    start_line: int
    start_col: int
    end_line: int
    end_col: int
    start_byte: int | None
    end_byte: int | None


@dataclass(frozen=True, slots=True)
class SourceBundle:
    """Container for decoded source text, raw bytes, and encoding."""

    text: str
    source_bytes: bytes
    encoding: str


class LineIndexedSource:
    """Precompute line/byte offsets for efficient span slicing."""

    def __init__(
        self,
        source: str,
        source_bytes: bytes | None = None,
        *,
        encoding: str = "utf-8",
    ) -> None:
        self.source = source
        self.encoding = encoding
        self.source_bytes = (
            source_bytes
            if source_bytes is not None
            else source.encode(
                encoding,
                errors="replace",
            )
        )
        self.line_offsets: list[int] = []
        self.byte_offsets: list[int] = []
        self.lines = source.splitlines(keepends=True)
        offset = 0
        byte_offset = 0
        for line in self.lines:
            self.line_offsets.append(offset)
            self.byte_offsets.append(byte_offset)
            offset += len(line)
            byte_offset += len(self._encode_text(line))

    def slice(self, start_line: int, start_col: int, end_line: int, end_col: int) -> str:
        """
        Return substring for a 0-based span; empty string on bounds errors.

        Returns
        -------
        str
            Extracted substring or empty string when indices are invalid.
        """
        try:
            start_idx = self.line_offsets[start_line] + start_col
            end_idx = self.line_offsets[end_line] + end_col
            return self.source[start_idx:end_idx]
        except (IndexError, ValueError):
            return ""

    def line_snippet(self, line: int) -> str | None:
        """Return a single 0-based line without trailing newline characters.

        Returns
        -------
        str | None
            Line text or None when out of bounds.
        """
        if line < 0:
            return None
        try:
            return self.lines[line].rstrip("\r\n")
        except IndexError:
            return None

    def span_from_range(
        self,
        pos: metadata.CodeRange,
        byte_span: metadata.CodeSpan | None = None,
    ) -> NormalizedSpan:
        """Return normalized span data for a metadata range.

        Returns
        -------
        NormalizedSpan
            Normalized 0-based span with byte offsets.
        """
        start_line = max(pos.start.line - 1, 0)
        end_line = max(pos.end.line - 1, 0)
        start_col = pos.start.column
        end_col = pos.end.column
        if byte_span is None:
            start_byte, end_byte = self.byte_span(start_line, start_col, end_line, end_col)
        else:
            start_byte, end_byte = self._byte_span_from_code_span(byte_span)
        return NormalizedSpan(
            start_line=start_line,
            start_col=start_col,
            end_line=end_line,
            end_col=end_col,
            start_byte=start_byte,
            end_byte=end_byte,
        )

    def byte_span(
        self,
        start_line: int,
        start_col: int,
        end_line: int,
        end_col: int,
    ) -> tuple[int | None, int | None]:
        """Return byte offsets for the provided 0-based span.

        Returns
        -------
        tuple[int | None, int | None]
            Start and end byte offsets (end exclusive), or None for invalid positions.
        """
        start_byte = self.byte_offset(start_line, start_col)
        end_byte = self.byte_offset(end_line, end_col)
        normalized = normalize_byte_span(start_byte, end_byte)
        if normalized is None:
            return None, None
        return normalized

    def byte_offset(self, line: int, col: int) -> int | None:
        """Return encoded byte offset for a 0-based line/column.

        Returns
        -------
        int | None
            Byte offset or None when out of bounds.
        """
        if line < 0 or col < 0:
            return None
        try:
            line_text = self.lines[line]
            base = self.byte_offsets[line]
        except IndexError:
            return None
        if col > len(line_text):
            return None
        prefix = line_text[:col]
        return base + len(self._encode_text(prefix))

    def byte_offset_from_utf8(self, line: int, utf8_col: int) -> int | None:
        """Return byte offset for a UTF-8 byte column (AST location semantics).

        Parameters
        ----------
        line
            0-based line index.
        utf8_col
            UTF-8 byte offset within the line (AST col_offset semantics).

        Returns
        -------
        int | None
            Byte offset in the file encoding, or None when indices are invalid.
        """
        char_col = self._char_offset_from_utf8(line, utf8_col)
        if char_col is None:
            return None
        return self.byte_offset(line, char_col)

    def _char_offset_from_utf8(self, line: int, utf8_col: int) -> int | None:
        if line < 0 or utf8_col < 0:
            return None
        try:
            line_text = self.lines[line]
        except IndexError:
            return None
        utf8_bytes = line_text.encode("utf-8", errors="replace")
        if utf8_col > len(utf8_bytes):
            return None
        try:
            prefix_text = utf8_bytes[:utf8_col].decode("utf-8")
        except UnicodeDecodeError:
            return None
        return len(prefix_text)

    @staticmethod
    def _byte_span_from_code_span(
        byte_span: metadata.CodeSpan,
    ) -> tuple[int | None, int | None]:
        normalized = normalize_byte_span(byte_span.start, byte_span.start + byte_span.length)
        if normalized is None:
            return None, None
        return normalized

    def _encode_text(self, text: str) -> bytes:
        try:
            return text.encode(self.encoding, errors="replace")
        except LookupError:
            return text.encode("utf-8", errors="replace")


class CstCaptureVisitor(cst.CSTVisitor):
    """Reusable visitor that records CST node rows."""

    METADATA_DEPENDENCIES = (metadata.PositionProvider,)

    def __init__(
        self,
        rel_path: str,
        module_name: str,
        source: SourceBundle,
        config: CstCaptureConfig,
    ) -> None:
        self.rel_path = rel_path
        self.module_name = module_name
        self.source_index = LineIndexedSource(
            source.text,
            source.source_bytes,
            encoding=source.encoding,
        )
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
        start_line = max(start.line - 1, 0)
        end_line = max(end.line - 1, 0)
        span = {"start": [start_line, start.column], "end": [end_line, end.column]}
        snippet = self.source_index.slice(start_line, start.column, end_line, end.column)

        parents = tuple(self._parent_kinds[:-1])
        qnames = (self._current_qualname(),)
        node_id = f"{self.rel_path}:{kind}:{start_line}:{start.column}:{end_line}:{end.column}"

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
