"""Semantic registry compiler tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from codeintel.core.hamilton import tags as ht
from codeintel.serving.semantic.registry_compiler import (
    SemanticTagValidationError,
    compile_semantic_registry,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from codeintel.core.schemas.primitives import TableSchema


@dataclass(frozen=True)
class FakeVariable:
    """Stub Hamilton variable for registry compilation tests."""

    name: str
    tags: dict[str, object]


class FakeTagQuery:
    """Minimal tag query stub for registry tests."""

    def __init__(self, variables: list[FakeVariable]) -> None:
        """Initialize with stub variables."""
        self._variables = variables

    def query(self, tag_filter: Mapping[str, object]) -> tuple[FakeVariable, ...]:
        """Return stub variables regardless of the tag filter.

        Returns
        -------
        tuple[FakeVariable, ...]
            Stub variables in declaration order.
        """
        _ = tag_filter
        return tuple(self._variables)


class FakeSchemaProvider:
    """Schema provider stub for registry tests."""

    def __init__(self) -> None:
        """Initialize an empty schema registry."""
        self._schemas: dict[str, TableSchema] = {}

    def get_table_schema(self, table_key: str) -> TableSchema | None:
        """Return schema for table_key when present.

        Returns
        -------
        TableSchema | None
            Schema for table_key when available.
        """
        return self._schemas.get(table_key)

    def require_table_schema(self, table_key: str) -> TableSchema:
        """Return schema for table_key or raise when missing.

        Returns
        -------
        TableSchema
            Schema for the table key.

        Raises
        ------
        KeyError
            When table_key is not present in the stub registry.
        """
        schema = self.get_table_schema(table_key)
        if schema is None:
            msg = f"Unknown table schema: {table_key}"
            raise KeyError(msg)
        return schema

    def iter_table_schemas(self) -> Iterable[TableSchema]:
        """Iterate over known table schemas.

        Returns
        -------
        Iterable[TableSchema]
            Iterable of known schemas.
        """
        return self._schemas.values()


def test_registry_compiler_emits_view() -> None:
    """Compile semantic registry views from valid tags."""
    variable = FakeVariable(
        name="semantic.view",
        tags={
            ht.TAG_LAYER: "semantic",
            ht.TAG_OUTPUT_KIND: ht.OUTPUT_KIND_SEMANTIC_VIEW,
            ht.TAG_SEMANTIC_ID: "function.architecture",
            ht.TAG_KIND: "view",
            ht.TAG_ENTITY: "function",
            ht.TAG_GRAIN: "per_function",
            ht.TAG_VERSION: "1",
            ht.TAG_TABLE_KEY: "docs.v_function_architecture",
            ht.TAG_MCP_VISIBLE: "1",
        },
    )
    registry = compile_semantic_registry(
        tag_query=FakeTagQuery([variable]),
        schema_provider=FakeSchemaProvider(),
        version="v1",
    )
    assert registry.version == "v1"
    assert registry.views[0]["id"] == "function.architecture"


def test_registry_compiler_rejects_missing_tags() -> None:
    """Reject semantic views missing required tags."""
    variable = FakeVariable(
        name="semantic.view",
        tags={
            ht.TAG_LAYER: "semantic",
            ht.TAG_OUTPUT_KIND: ht.OUTPUT_KIND_SEMANTIC_VIEW,
            ht.TAG_SEMANTIC_ID: "function.architecture",
            ht.TAG_KIND: "view",
            ht.TAG_ENTITY: "function",
            ht.TAG_GRAIN: "per_function",
            ht.TAG_TABLE_KEY: "docs.v_function_architecture",
        },
    )
    with pytest.raises(SemanticTagValidationError):
        compile_semantic_registry(
            tag_query=FakeTagQuery([variable]),
            schema_provider=FakeSchemaProvider(),
            version="v1",
        )
