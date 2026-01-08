"""Tests for Arrow ops helpers."""

from __future__ import annotations

import pyarrow as pa
import pytest

from codeintel.build.tabular.arrow_ops import (
    ArrowJoinSpec,
    arrow_join_tables,
    dedupe_table_for_table,
)
from tests._helpers.assertions.expectation_assertions import expect_equal


def test_dedupe_table_for_table_removes_duplicate_keys() -> None:
    """dedupe_table_for_table should drop duplicate primary-key rows."""
    table = pa.table(
        {
            "module": ["alpha", "alpha", "beta"],
            "path": ["alpha.py", "alpha.py", "beta.py"],
            "repo": ["repo", "repo", "repo"],
        }
    )

    deduped = dedupe_table_for_table("core.modules", table)

    expect_equal(deduped.num_rows, 2)
    keys = set(
        zip(
            deduped.column("module").to_pylist(),
            deduped.column("path").to_pylist(),
            strict=True,
        )
    )
    expect_equal(keys, {("alpha", "alpha.py"), ("beta", "beta.py")})


def test_arrow_join_tables_rejects_null_keys() -> None:
    """arrow_join_tables should fail validation when join keys contain NULLs."""
    left = pa.table({"id": [1, None], "value": ["a", "b"]})
    right = pa.table({"id": [1, 2], "value": ["x", "y"]})
    spec = ArrowJoinSpec(on=["id"], how="left", validate="1:m")

    with pytest.raises(ValueError, match="NULL keys detected"):
        _ = arrow_join_tables(left, right, spec=spec)
