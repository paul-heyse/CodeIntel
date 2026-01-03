"""Typed payloads for tree-sitter extras metadata."""

from __future__ import annotations

from typing import NotRequired, TypedDict


class TreeSitterCaptureExtras(TypedDict):
    """Extras payload for tree-sitter captures."""

    query_hash: str
    pattern_index: int | None
    capture_index: int | None
    pattern_count: int
    capture_count: int
    field_name: str | None
    field_id: int | None


class TreeSitterTokenExtras(TypedDict):
    """Extras payload for tree-sitter tokens and trivia."""

    query_hash: str
    pattern_index: int | None
    capture_index: int | None
    capture_name: str
    pattern_count: int
    capture_count: int
    field_name: str | None
    field_id: int | None
    literal_kind: NotRequired[str]


class TreeSitterNodeExtras(TypedDict, total=False):
    """Extras payload for tree-sitter nodes."""


class TreeSitterParseErrorExtras(TypedDict):
    """Extras payload for tree-sitter parse errors."""

    node_type: str | None
    has_error: bool | None
    parse_state: int | None


__all__ = [
    "TreeSitterCaptureExtras",
    "TreeSitterNodeExtras",
    "TreeSitterParseErrorExtras",
    "TreeSitterTokenExtras",
]
