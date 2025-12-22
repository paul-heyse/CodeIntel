"""Helpers for extracting payloads from MCP tool results."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Protocol, TypeGuard, runtime_checkable


@runtime_checkable
class _HasContent(Protocol):
    content: Sequence[object]


@runtime_checkable
class _HasText(Protocol):
    text: str


def _has_text(value: object) -> TypeGuard[_HasText]:
    text = getattr(value, "text", None)
    return isinstance(text, str)


def _has_content(value: object) -> TypeGuard[_HasContent]:
    return hasattr(value, "content")


def extract_payload(tool_result: object) -> dict[str, object]:
    """Extract the payload dictionary from an MCP tool result.

    Returns
    -------
    dict[str, object]
        Parsed payload dictionary.

    Raises
    ------
    TypeError
        If the tool result is not a supported shape.
    """
    if _has_content(tool_result):
        content_list = tool_result.content
        if isinstance(content_list, Sequence) and content_list:
            first_content = content_list[0]
            if _has_text(first_content):
                return json.loads(first_content.text)

    if isinstance(tool_result, list) and tool_result:
        first_content = tool_result[0]
        if _has_text(first_content):
            return json.loads(first_content.text)

    if isinstance(tool_result, dict):
        return tool_result

    msg = f"Unexpected tool result type: {type(tool_result)}"
    raise TypeError(msg)


__all__ = ["extract_payload"]
