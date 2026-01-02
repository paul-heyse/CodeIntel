"""Tests for advanced query engine contracts."""

from __future__ import annotations

import msgspec
import pytest
from tools.advanced_query_engine.contracts import QueryRequest


def _base_payload() -> dict[str, object]:
    return {
        "type": "pattern.scan",
        "text": "print('hello')",
        "repo_root": "/repo",
    }


def test_query_request_rejects_unknown_fields() -> None:
    """Reject unknown fields in QueryRequest payloads."""
    payload = _base_payload()
    payload["unknown"] = 123
    with pytest.raises(msgspec.ValidationError):
        msgspec.convert(payload, type=QueryRequest)


def test_query_request_rejects_invalid_type() -> None:
    """Reject invalid query type literals."""
    payload = _base_payload()
    payload["type"] = "invalid.type"
    with pytest.raises(msgspec.ValidationError):
        msgspec.convert(payload, type=QueryRequest)
