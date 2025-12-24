"""Function types seed pack for typing analytics.

This module provides FunctionTypesPack which seeds analytics.function_types
with type annotation data for testing typing-related analytics.

The pack depends on CORE_PACK and uses its GOID definitions to create
realistic type annotation data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from tests._helpers.fixtures.rows import FunctionTypesRow, dataclass_row, insert_rows
from tests._helpers.seeds.core import (
    CORE_PACK,
    GOID_FUNC_A,
    GOID_FUNC_B,
    GOID_FUNC_C,
    GOID_HELPER,
    MOD_A_PATH,
    MOD_B_PATH,
    MOD_C_PATH,
    MOD_UTIL_PATH,
)

if TYPE_CHECKING:
    from tests._helpers.context import SeedPack, TestContext


# =============================================================================
# Function Types Pack Implementation
# =============================================================================


@dataclass
class FunctionTypesPack:
    """Seed pack for function type annotations.

    Seeds analytics.function_types with type annotation data for functions
    from CORE_PACK. Provides realistic typing scenarios including fully typed,
    partially typed, and untyped functions.

    Attributes
    ----------
    name : str
        Unique pack identifier.
    include_fully_typed : bool
        Whether to include fully typed function examples.
    include_partial_typed : bool
        Whether to include partially typed function examples.
    include_untyped : bool
        Whether to include untyped function examples.
    """

    name: str = "function_types"
    include_fully_typed: bool = True
    include_partial_typed: bool = True
    include_untyped: bool = True

    @property
    def dependencies(self) -> tuple[SeedPack, ...]:
        """Return seed packs that must be applied before this one.

        Returns
        -------
        tuple[SeedPack, ...]
            CorePack is required for GOID data.
        """
        return (CORE_PACK,)

    def apply(self, ctx: TestContext) -> None:
        """Apply function types seeds to the test context.

        Seeds analytics.function_types with type annotation data.

        Parameters
        ----------
        ctx
            Test context to seed.
        """
        now = datetime.now(UTC)
        self._seed_function_types(ctx, now)

    def _seed_function_types(self, ctx: TestContext, now: datetime) -> None:
        """Seed function types table.

        Parameters
        ----------
        ctx
            Test context with gateway.
        now
            Timestamp for created_at fields.
        """
        rows: list[FunctionTypesRow] = []

        if self.include_fully_typed:
            # func_a: fully typed with 2 parameters
            rows.append(
                dataclass_row(
                    FunctionTypesRow,
                    function_goid_h128=GOID_FUNC_A,
                    urn=f"urn:codeintel:{ctx.repo}:{ctx.commit}:{MOD_A_PATH}#func_a",
                    repo=ctx.repo,
                    commit=ctx.commit,
                    rel_path=MOD_A_PATH,
                    language="python",
                    kind="function",
                    qualname="func_a",
                    start_line=1,
                    end_line=10,
                    total_params=2,
                    annotated_params=2,
                    unannotated_params=0,
                    param_typed_ratio=1.0,
                    has_return_annotation=True,
                    return_type="int",
                    return_type_source="annotation",
                    type_comment=None,
                    param_types_json=json.dumps({"x": "int", "y": "int"}),
                    fully_typed=True,
                    partial_typed=False,
                    untyped=False,
                    typedness_bucket="fully_typed",
                    typedness_source="annotation",
                    created_at=now,
                )
            )

            # helper: fully typed with 1 parameter
            rows.append(
                dataclass_row(
                    FunctionTypesRow,
                    function_goid_h128=GOID_HELPER,
                    urn=f"urn:codeintel:{ctx.repo}:{ctx.commit}:{MOD_UTIL_PATH}#helper",
                    repo=ctx.repo,
                    commit=ctx.commit,
                    rel_path=MOD_UTIL_PATH,
                    language="python",
                    kind="function",
                    qualname="helper",
                    start_line=1,
                    end_line=5,
                    total_params=1,
                    annotated_params=1,
                    unannotated_params=0,
                    param_typed_ratio=1.0,
                    has_return_annotation=True,
                    return_type="int",
                    return_type_source="annotation",
                    type_comment=None,
                    param_types_json=json.dumps({"value": "int"}),
                    fully_typed=True,
                    partial_typed=False,
                    untyped=False,
                    typedness_bucket="fully_typed",
                    typedness_source="annotation",
                    created_at=now,
                )
            )

        if self.include_partial_typed:
            # func_b: partially typed - has return but missing param annotation
            rows.append(
                dataclass_row(
                    FunctionTypesRow,
                    function_goid_h128=GOID_FUNC_B,
                    urn=f"urn:codeintel:{ctx.repo}:{ctx.commit}:{MOD_B_PATH}#func_b",
                    repo=ctx.repo,
                    commit=ctx.commit,
                    rel_path=MOD_B_PATH,
                    language="python",
                    kind="function",
                    qualname="func_b",
                    start_line=1,
                    end_line=15,
                    total_params=1,
                    annotated_params=0,
                    unannotated_params=1,
                    param_typed_ratio=0.0,
                    has_return_annotation=True,
                    return_type="int",
                    return_type_source="annotation",
                    type_comment=None,
                    param_types_json=json.dumps({}),
                    fully_typed=False,
                    partial_typed=True,
                    untyped=False,
                    typedness_bucket="partial_typed",
                    typedness_source="annotation",
                    created_at=now,
                )
            )

        if self.include_untyped:
            # func_c: untyped - generator with no annotations
            rows.append(
                dataclass_row(
                    FunctionTypesRow,
                    function_goid_h128=GOID_FUNC_C,
                    urn=f"urn:codeintel:{ctx.repo}:{ctx.commit}:{MOD_C_PATH}#func_c",
                    repo=ctx.repo,
                    commit=ctx.commit,
                    rel_path=MOD_C_PATH,
                    language="python",
                    kind="function",
                    qualname="func_c",
                    start_line=1,
                    end_line=8,
                    total_params=0,
                    annotated_params=0,
                    unannotated_params=0,
                    param_typed_ratio=0.0,
                    has_return_annotation=False,
                    return_type="",
                    return_type_source="none",
                    type_comment=None,
                    param_types_json=json.dumps({}),
                    fully_typed=False,
                    partial_typed=False,
                    untyped=True,
                    typedness_bucket="untyped",
                    typedness_source="none",
                    created_at=now,
                )
            )

        if rows:
            insert_rows(ctx.gateway, rows)


# Default instance for common usage
FUNCTION_TYPES_PACK = FunctionTypesPack()


__all__ = [
    "FUNCTION_TYPES_PACK",
    "FunctionTypesPack",
]
