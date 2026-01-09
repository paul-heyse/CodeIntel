"""External plan runner integration tests."""

from __future__ import annotations

import pyarrow as pa

from codeintel.core.columnar.conversion import reader_to_table, table_to_reader
from codeintel.core.columnar.iter import iter_rows
from codeintel.core.columnar.plan_ops import (
    ExternalPlanRequest,
    ExternalPlanSpec,
    register_external_plan_runner,
    run_external_plan,
)


def _request(engine: str) -> ExternalPlanRequest:
    return ExternalPlanRequest(
        spec=ExternalPlanSpec(engine=engine, payload={"plan": "demo"}),
        dataset=None,
        filter_expr=None,
        columns=None,
        scan_options=None,
        use_threads=False,
    )


def test_run_external_plan_reader() -> None:
    """External runners returning readers should execute through the registry."""
    engine = "test_reader_engine"

    def runner(*, request: ExternalPlanRequest) -> pa.RecordBatchReader:
        assert request.spec.engine == engine
        table = pa.table({"id": [1, 2]})
        return table_to_reader(table)

    register_external_plan_runner(engine, runner)
    reader = run_external_plan(_request(engine))
    table = reader_to_table(reader)
    rows = list(iter_rows(table, columns=("id",)))
    assert rows == [{"id": 1}, {"id": 2}]


def test_run_external_plan_table() -> None:
    """External runners returning tables should be coerced to readers."""
    engine = "test_table_engine"

    def runner(*, request: ExternalPlanRequest) -> pa.Table:
        assert request.spec.engine == engine
        return pa.table({"value": [3]})

    register_external_plan_runner(engine, runner)
    reader = run_external_plan(_request(engine))
    table = reader_to_table(reader)
    rows = list(iter_rows(table, columns=("value",)))
    assert rows == [{"value": 3}]
