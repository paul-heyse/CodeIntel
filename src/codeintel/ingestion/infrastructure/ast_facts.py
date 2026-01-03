"""Shared AST extraction helpers for spans and node payloads."""

from __future__ import annotations

import ast
import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass

from codeintel.ingestion.infrastructure.cst_utils import LineIndexedSource

LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AstSpan:
    """Normalized AST span with byte offsets."""

    start_line: int
    start_col_utf8: int
    end_line: int
    end_col_utf8: int
    start_byte: int
    end_byte: int


@dataclass(frozen=True, slots=True)
class AstNodeRecord:
    """Collected AST node record with span and extras."""

    node_id: str
    kind: str
    span: AstSpan
    extras: dict[str, object] | None
    node: ast.AST


def _ast_module_span(source_index: LineIndexedSource) -> AstSpan:
    end_byte = len(source_index.source_bytes)
    if not source_index.lines:
        return AstSpan(
            start_line=0,
            start_col_utf8=0,
            end_line=0,
            end_col_utf8=0,
            start_byte=0,
            end_byte=end_byte,
        )
    end_line = len(source_index.lines) - 1
    end_col_utf8 = len(source_index.lines[end_line].encode("utf-8", errors="replace"))
    return AstSpan(
        start_line=0,
        start_col_utf8=0,
        end_line=end_line,
        end_col_utf8=end_col_utf8,
        start_byte=0,
        end_byte=end_byte,
    )


def ast_span_for_node(
    node: ast.AST,
    source_index: LineIndexedSource,
) -> AstSpan | None:
    """Return an AST node span with byte offsets when available."""
    if isinstance(node, ast.Module):
        return _ast_module_span(source_index)
    lineno = getattr(node, "lineno", None)
    col_offset = getattr(node, "col_offset", None)
    if not isinstance(lineno, int) or not isinstance(col_offset, int):
        return None
    end_lineno = getattr(node, "end_lineno", None)
    end_col_offset = getattr(node, "end_col_offset", None)
    if not isinstance(end_lineno, int):
        end_lineno = lineno
    if not isinstance(end_col_offset, int):
        end_col_offset = col_offset
    if end_lineno < lineno:
        end_lineno = lineno
        end_col_offset = col_offset
    start_line = max(lineno - 1, 0)
    end_line = max(end_lineno - 1, 0)
    start_byte = source_index.byte_offset_from_utf8(start_line, col_offset)
    end_byte = source_index.byte_offset_from_utf8(end_line, end_col_offset)
    if start_byte is None or end_byte is None:
        return None
    if end_byte < start_byte:
        return None
    return AstSpan(
        start_line=start_line,
        start_col_utf8=col_offset,
        end_line=end_line,
        end_col_utf8=end_col_offset,
        start_byte=start_byte,
        end_byte=end_byte,
    )


def _ast_type_ignores_payload(module: ast.Module) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    for ignore in getattr(module, "type_ignores", []):
        lineno = getattr(ignore, "lineno", None)
        if not isinstance(lineno, int):
            continue
        tag = getattr(ignore, "tag", None)
        entry: dict[str, object] = {"line": max(lineno - 1, 0)}
        if isinstance(tag, str) and tag:
            entry["tag"] = tag
        payload.append(entry)
    return payload


def _ast_ctx_payload(node: ast.AST) -> dict[str, object]:
    ctx = getattr(node, "ctx", None)
    if ctx is None:
        return {}
    return {"ctx": type(ctx).__name__.lower()}


def _ast_type_comment_payload(node: ast.AST) -> dict[str, object]:
    type_comment = getattr(node, "type_comment", None)
    if isinstance(type_comment, str) and type_comment:
        return {"type_comment": type_comment}
    return {}


def _ast_module_payload(node: ast.AST) -> dict[str, object]:
    if isinstance(node, ast.Module):
        type_ignores = _ast_type_ignores_payload(node)
        if type_ignores:
            return {"type_ignores": type_ignores}
    return {}


def _ast_name_payload(node: ast.AST) -> dict[str, object]:
    if isinstance(node, ast.Name):
        return {"identifier": node.id}
    if isinstance(node, ast.Attribute):
        return {"attribute": node.attr}
    if isinstance(node, ast.arg):
        return {"name": node.arg}
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {"name": node.name}
    return {}


def _ast_import_payload(node: ast.AST) -> dict[str, object]:
    if isinstance(node, ast.alias):
        payload: dict[str, object] = {"imported": node.name}
        if node.asname is not None:
            payload["asname"] = node.asname
        return payload
    if isinstance(node, ast.ImportFrom):
        payload: dict[str, object] = {}
        if node.module is not None:
            payload["module"] = node.module
        if node.level:
            payload["level"] = node.level
        return payload
    return {}


def _ast_constant_payload(node: ast.AST) -> dict[str, object]:
    if isinstance(node, ast.Constant):
        return {"constant_kind": type(node.value).__name__}
    return {}


def ast_node_extras(node: ast.AST) -> dict[str, object] | None:
    """Return structured extras for an AST node."""
    extras: dict[str, object] = {}
    for payload in (
        _ast_ctx_payload(node),
        _ast_type_comment_payload(node),
        _ast_module_payload(node),
        _ast_name_payload(node),
        _ast_import_payload(node),
        _ast_constant_payload(node),
    ):
        if payload:
            extras.update(payload)
    return extras or None


def ast_node_id(rel_path: str, node_kind: str, span: AstSpan) -> str:
    """Return a stable-ish AST node identifier for a file span."""
    payload = (
        f"{rel_path}:{node_kind}:{span.start_byte}:{span.end_byte}:"
        f"{span.start_line}:{span.end_line}"
    )
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()


def collect_ast_nodes(
    source_text: str,
    source_index: LineIndexedSource,
    *,
    node_id_factory: Callable[[ast.AST, AstSpan], str],
    warnings: list[str] | None = None,
    source_label: str | None = None,
    parsed: ast.AST | None = None,
) -> list[AstNodeRecord]:
    """Collect AST node records with spans and extras."""
    tree = parsed
    label = source_label or "<source>"
    if tree is None:
        try:
            tree = ast.parse(source_text, type_comments=True)
        except (SyntaxError, ValueError, TypeError) as exc:
            message = f"AST parse failed for {label}: {exc}"
            if warnings is not None:
                warnings.append(message)
            LOG.warning("%s", message)
            return []
    records: list[AstNodeRecord] = []
    for node in ast.walk(tree):
        span = ast_span_for_node(node, source_index)
        if span is None:
            continue
        node_id = node_id_factory(node, span)
        records.append(
            AstNodeRecord(
                node_id=node_id,
                kind=type(node).__name__,
                span=span,
                extras=ast_node_extras(node),
                node=node,
            )
        )
    records.sort(key=lambda record: (record.span.start_byte, record.span.end_byte, record.kind))
    return records


__all__ = [
    "AstNodeRecord",
    "AstSpan",
    "ast_node_extras",
    "ast_node_id",
    "ast_span_for_node",
    "collect_ast_nodes",
]
