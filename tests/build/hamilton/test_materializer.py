"""Tests for Hamilton native materializer utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pandas as pd
import pytest

from codeintel.build.hamilton.native import materializer
from codeintel.build.hamilton.native.materializer import MaterializationContext, materialize_table
from codeintel.config.primitives import SnapshotRef

if TYPE_CHECKING:
    import ibis.expr.types as ir

    from codeintel.storage.gateway.protocol import StorageGateway


class FakeScalar:
    """Fake Ibis scalar wrapper for testing count()."""

    def __init__(self, value: int) -> None:
        """Initialize with a scalar value."""
        self.value = value
        self.execute_calls = 0

    def execute(self) -> int:
        """Return the stored value and track calls.

        Returns
        -------
        int
            Stored scalar value.
        """
        self.execute_calls += 1
        return self.value


class FakeTable:
    """Minimal stand-in for an Ibis Table expression."""

    def __init__(self, df: pd.DataFrame, count_value: int | None = None) -> None:
        """Store DataFrame and configure count value."""
        self.df = df
        self.count_value = count_value if count_value is not None else len(df)
        self.count_calls = 0
        self.execute_calls = 0

    def count(self) -> FakeScalar:
        """Return a fake scalar representing row count.

        Returns
        -------
        FakeScalar
            Scalar returning the configured count value.
        """
        self.count_calls += 1
        return FakeScalar(self.count_value)

    def execute(self) -> pd.DataFrame:
        """Return the backing DataFrame and record execution count.

        Returns
        -------
        pandas.DataFrame
            Stored DataFrame for this fake table.
        """
        self.execute_calls += 1
        return self.df


class FakePolicy:
    """Tracks delete_for_snapshot calls."""

    def __init__(self) -> None:
        """Initialize call recorder."""
        self.calls: list[tuple[str, str, str]] = []

    def delete_for_snapshot(self, table_key: str, *, repo: str, commit: str) -> None:
        """Record delete operations."""
        self.calls.append((table_key, repo, commit))


@dataclass
class FakeWriteResult:
    """Captured write results."""

    table_key: str
    data: pd.DataFrame | FakeTable


class FakeIbisGateway:
    """Records writes without touching DuckDB."""

    def __init__(self) -> None:
        """Initialize write recorder."""
        self.writes: list[FakeWriteResult] = []

    def write(
        self,
        table_key: str,
        data: pd.DataFrame | FakeTable,
        columns: object | None = None,
        on_conflict: object | None = None,
    ) -> FakeWriteResult:
        """Record write calls.

        Returns
        -------
        FakeWriteResult
            Captured write data for assertions.
        """
        _ = (columns, on_conflict)
        result = FakeWriteResult(table_key=table_key, data=data)
        self.writes.append(result)
        return result


class FakeGateway:
    """Minimal gateway exposing policy + ibis for materializer."""

    def __init__(self) -> None:
        """Initialize fake policy and ibis gateways."""
        self.policy = FakePolicy()
        self.ibis = FakeIbisGateway()


def expect(*, condition: bool, message: str) -> None:
    """Fail the test with the provided message when condition is False."""
    if not condition:
        pytest.fail(message)


def test_materialize_table_uses_policy_and_insert_select() -> None:
    """materialize_table should delete snapshot rows then insert via IbisGateway."""
    df = pd.DataFrame({"repo": ["r"], "commit": ["c"], "value": [1]})
    table = FakeTable(df, count_value=5)
    gateway = FakeGateway()
    snapshot = SnapshotRef(repo="r", commit="c", repo_root=Path("repo"))
    expected_row_count = 5

    ref = materialize_table(
        MaterializationContext(
            gateway=cast("StorageGateway", gateway), snapshot=snapshot, validate=False
        ),
        "analytics.example",
        cast("ir.Table", table),
    )

    expect(
        condition=gateway.policy.calls == [("analytics.example", "r", "c")],
        message=f"Unexpected delete calls: {gateway.policy.calls}",
    )
    expect(
        condition=bool(gateway.ibis.writes) and gateway.ibis.writes[0].data is table,
        message="Expected materializer to write the Ibis expression",
    )
    expect(
        condition=table.count_calls == 1,
        message=f"Count should run once, got {table.count_calls}",
    )
    expect(
        condition=table.execute_calls == 0,
        message="Execute should not run when validation is disabled",
    )
    expect(
        condition=ref.row_count == expected_row_count,
        message=f"Row count mismatch: {ref.row_count}",
    )


def test_materialize_table_validates_when_schema_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """materialize_table should validate DataFrame when schema is present."""
    df = pd.DataFrame(
        {"repo": ["r", "r"], "commit": ["c", "c"], "value": [1, 2]},
    )
    table = FakeTable(df)
    gateway = FakeGateway()
    snapshot = SnapshotRef(repo="r", commit="c", repo_root=Path("repo"))

    class StubSchema:
        def __init__(self) -> None:
            """Capture validate call metadata."""
            self.calls: list[dict[str, object]] = []

        def validate(self, frame: pd.DataFrame, *, lazy: bool = False) -> pd.DataFrame:
            """Record validation calls and return the frame.

            Returns
            -------
            pandas.DataFrame
                Frame passed to validation (unchanged).
            """
            self.calls.append({"frame": frame.copy(), "lazy": lazy})
            return frame

    schema = StubSchema()
    monkeypatch.setattr(materializer, "get_pandera_schema", lambda _: schema)

    ref = materialize_table(
        MaterializationContext(
            gateway=cast("StorageGateway", gateway), snapshot=snapshot, validate=True
        ),
        "analytics.example",
        cast("ir.Table", table),
    )

    expect(condition=bool(schema.calls), message="Schema.validate should be invoked")
    expect(
        condition=schema.calls[0]["lazy"] is False,
        message=f"Validation should be eager, got {schema.calls[0]['lazy']}",
    )
    expect(condition=bool(gateway.ibis.writes), message="Expected a write to be recorded")
    written_df = gateway.ibis.writes[0].data
    if not isinstance(written_df, pd.DataFrame):
        pytest.fail("Write should receive a DataFrame")
    expect(
        condition=len(written_df) == len(df),
        message=f"Unexpected row count: {len(written_df)}",
    )
    expect(
        condition=table.count_calls == 0,
        message="Count should not run during validation",
    )
    expect(
        condition=table.execute_calls == 1,
        message="Execute should run once during validation",
    )
    expect(
        condition=ref.row_count == len(df),
        message=f"Row count mismatch: {ref.row_count}",
    )
