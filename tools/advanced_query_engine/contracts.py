"""Contracts and data models for the advanced query engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import msgspec

type JSONScalar = str | int | float | bool | None
type JSONValue = JSONScalar | list[JSONValue] | dict[str, JSONValue]

type SymbolKind = Literal[
    "module",
    "class",
    "function",
    "method",
    "variable",
    "import",
    "callsite",
    "route",
    "config_key",
    "test",
]

type QueryType = Literal[
    "symbol.resolve",
    "refs.find",
    "callgraph.slice",
    "pattern.scan",
    "contract.lookup",
    "wiring.map",
    "precedent.search",
    "impact.slice",
]


@dataclass(frozen=True)
class Span:
    """Describe a byte span with optional line/column coordinates.

    Parameters
    ----------
    path:
        Repo-relative path (POSIX-style preferred).
    start_byte:
        Inclusive byte offset.
    end_byte:
        Exclusive byte offset.
    start_line:
        1-indexed line number for display, if available.
    start_col:
        0-indexed column for display, if available.
    end_line:
        1-indexed line number for display, if available.
    end_col:
        0-indexed column for display, if available.
    """

    path: str
    start_byte: int
    end_byte: int
    start_line: int | None = None
    start_col: int | None = None
    end_line: int | None = None
    end_col: int | None = None

    def to_dict(self) -> dict[str, JSONValue]:
        """Return a JSON-serializable dict representation.

        Returns
        -------
        dict[str, JSONValue]
            Serialized span payload.
        """
        payload: dict[str, JSONValue] = {
            "path": self.path,
            "start_byte": self.start_byte,
            "end_byte": self.end_byte,
        }
        if self.start_line is not None:
            payload["start_line"] = self.start_line
        if self.start_col is not None:
            payload["start_col"] = self.start_col
        if self.end_line is not None:
            payload["end_line"] = self.end_line
        if self.end_col is not None:
            payload["end_col"] = self.end_col
        return payload


@dataclass(frozen=True)
class SymbolId:
    """Stable symbol identifier for search results."""

    kind: SymbolKind
    stable: str
    qname: str | None = None

    def to_dict(self) -> dict[str, JSONValue]:
        """Return a JSON-serializable dict representation.

        Returns
        -------
        dict[str, JSONValue]
            Serialized symbol id payload.
        """
        payload: dict[str, JSONValue] = {
            "kind": self.kind,
            "stable": self.stable,
        }
        if self.qname is not None:
            payload["qname"] = self.qname
        return payload


@dataclass(frozen=True)
class EvidenceSnippet:
    """Snippet with surrounding context for grounding."""

    span: Span
    text: str
    context_before: list[str]
    context_after: list[str]

    def to_dict(self) -> dict[str, JSONValue]:
        """Return a JSON-serializable dict representation.

        Returns
        -------
        dict[str, JSONValue]
            Serialized snippet payload.
        """
        return {
            "span": self.span.to_dict(),
            "text": self.text,
            "context_before": list(self.context_before),
            "context_after": list(self.context_after),
        }


@dataclass(frozen=True)
class SymbolRecord:
    """Symbol definition record used by search responses."""

    symbol_id: SymbolId
    name: str
    kind: SymbolKind
    def_span: Span
    signature: str | None = None
    docstring: str | None = None

    def to_dict(self) -> dict[str, JSONValue]:
        """Return a JSON-serializable dict representation.

        Returns
        -------
        dict[str, JSONValue]
            Serialized symbol record payload.
        """
        payload: dict[str, JSONValue] = {
            "symbol_id": self.symbol_id.to_dict(),
            "name": self.name,
            "kind": self.kind,
            "def_span": self.def_span.to_dict(),
        }
        if self.signature is not None:
            payload["signature"] = self.signature
        if self.docstring is not None:
            payload["docstring"] = self.docstring
        return payload


@dataclass(frozen=True)
class MatchRecord:
    """Normalized match record across backends."""

    engine: str
    path: str
    span: Span
    rule_id: str | None = None
    pattern_id: str | None = None
    captures: dict[str, list[EvidenceSnippet]] | None = None
    snippet: EvidenceSnippet | None = None

    def to_dict(self) -> dict[str, JSONValue]:
        """Return a JSON-serializable dict representation.

        Returns
        -------
        dict[str, JSONValue]
            Serialized match record payload.
        """
        payload: dict[str, JSONValue] = {
            "engine": self.engine,
            "path": self.path,
            "span": self.span.to_dict(),
        }
        if self.rule_id is not None:
            payload["rule_id"] = self.rule_id
        if self.pattern_id is not None:
            payload["pattern_id"] = self.pattern_id
        if self.snippet is not None:
            payload["snippet"] = self.snippet.to_dict()
        if self.captures:
            payload["captures"] = {
                key: [snip.to_dict() for snip in values] for key, values in self.captures.items()
            }
        return payload


class QueryBudget(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Execution budget for search queries."""

    max_files: int = 300
    max_matches: int = 2000
    max_depth: int = 0
    max_seconds: float | None = None
    context_lines: int = 1


class QueryRequest(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Search request for the advanced query engine."""

    type: QueryType
    text: str
    repo_root: str
    scope_paths: list[str] | None = None
    budget: QueryBudget | None = None
    options: dict[str, JSONValue] | None = None


class QueryResponse(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Search response payload."""

    summary: str
    primary: list[dict[str, JSONValue]]
    related: dict[str, list[dict[str, JSONValue]]]
    debug: dict[str, JSONValue]

    def to_dict(self) -> dict[str, JSONValue]:
        """Return a JSON-serializable dict representation.

        Returns
        -------
        dict[str, JSONValue]
            Serialized response payload.
        """
        return {
            "summary": self.summary,
            "primary": list(self.primary),
            "related": {key: list(values) for key, values in self.related.items()},
            "debug": dict(self.debug),
        }


def query_request_schema() -> dict[str, JSONValue]:
    """Return the JSON schema for QueryRequest.

    Returns
    -------
    dict[str, JSONValue]
        JSON schema payload.
    """
    return msgspec.json.schema(QueryRequest)


def query_response_schema() -> dict[str, JSONValue]:
    """Return the JSON schema for QueryResponse.

    Returns
    -------
    dict[str, JSONValue]
        JSON schema payload.
    """
    return msgspec.json.schema(QueryResponse)


__all__ = [
    "EvidenceSnippet",
    "JSONValue",
    "MatchRecord",
    "QueryBudget",
    "QueryRequest",
    "QueryResponse",
    "QueryType",
    "Span",
    "SymbolId",
    "SymbolKind",
    "SymbolRecord",
    "query_request_schema",
    "query_response_schema",
]
