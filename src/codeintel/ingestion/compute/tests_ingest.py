"""Test catalog ingestion step with port injection.

This module provides a pure domain logic implementation for ingesting
pytest test metadata into analytics.test_catalog, using ports for all
I/O operations.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from codeintel.build.hamilton.execution_result import ExecutionResult
from codeintel.core.columnar.rows import (
    ColumnarRowBuffer,
    ColumnarRows,
    columnar_buffer_for_table_key,
    empty_table_for_table,
    table_for_columnar_rows,
)
from codeintel.ingestion.compute.base import persist_arrow_tables
from codeintel.ingestion.context import IngestionContext, resolve_repo_commit
from codeintel.ingestion.engine.results import parse_test_duration, parse_test_markers

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    import pyarrow as pa

    from codeintel.ingestion.ports.discovery import ModuleRecord
    from codeintel.ingestion.ports.storage import IngestStoragePort

log = logging.getLogger(__name__)
TEST_CATALOG_TABLE_KEY = "analytics.test_catalog"


def _build_test_catalog_rows(
    buffer: ColumnarRowBuffer,
    tests: Sequence[Mapping[str, object]],
    *,
    repo: str,
    commit: str,
    created_at: datetime,
) -> None:
    for test in tests:
        nodeid = str(test.get("nodeid", ""))
        if not nodeid:
            continue
        outcome = str(test.get("outcome", test.get("status", "unknown")))
        duration_s = parse_test_duration(test)
        if duration_s == 0.0:
            raw_duration = test.get("duration")
            if isinstance(raw_duration, (int, float)):
                duration_s = float(raw_duration)
        duration_ms = duration_s * 1000.0
        markers = parse_test_markers(test)
        rel_path = nodeid.split("::", maxsplit=1)[0] if "::" in nodeid else nodeid
        qualname = nodeid.split("::", maxsplit=1)[1] if "::" in nodeid else None
        parametrized = "[" in nodeid or "parametrize" in markers
        flaky = "flaky" in markers

        buffer.append(
            {
                "test_id": nodeid,
                "test_goid_h128": None,
                "urn": None,
                "repo": repo,
                "commit": commit,
                "rel_path": rel_path,
                "qualname": qualname,
                "kind": "test",
                "status": outcome,
                "duration_ms": duration_ms,
                "extras": {"markers": list(markers)},
                "parametrized": parametrized,
                "flaky": flaky,
                "created_at": created_at,
            }
        )


class TestsIngestStep:
    """Test catalog ingestion step with port injection.

    This step ingests pytest JSON reports into analytics.test_catalog, using
    ports for all I/O operations.
    """

    @staticmethod
    def execute(
        _modules: Sequence[ModuleRecord],
        *,
        repo: str | None = None,
        commit: str | None = None,
        json_report_path: Path,
        context: IngestionContext | None = None,
        storage: IngestStoragePort | None = None,
    ) -> TestsIngestResult:
        """Execute test catalog ingestion.

        Parameters
        ----------
        _modules
            Modules for reference (not directly used).
        repo
            Repository identifier.
        commit
            Commit identifier.
        json_report_path
            Path to the pytest JSON report.
        context
            Optional ingestion context supplying repo/commit defaults.
        storage
            Optional storage port for persisting Arrow outputs.

        Returns
        -------
        TestsIngestResult
            Result bundle with row tuples and execution status.
        """
        resolved_repo, resolved_commit = resolve_repo_commit(
            context=context,
            repo=repo,
            commit=commit,
        )
        created_at = datetime.now(UTC)

        if not json_report_path.exists():
            log.warning("Test report not found: %s", json_report_path)
            return TestsIngestResult(
                result=ExecutionResult.skip("Test report not found"),
            )

        try:
            data = json.loads(json_report_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("Failed to read test report: %s", exc)
            return TestsIngestResult(
                result=ExecutionResult.failed(f"Failed to read test report: {exc}")
            )

        tests = data.get("tests", [])
        buffer = columnar_buffer_for_table_key(TEST_CATALOG_TABLE_KEY)
        _build_test_catalog_rows(
            buffer,
            tests,
            repo=resolved_repo,
            commit=resolved_commit,
            created_at=created_at,
        )

        log.info(
            "Test catalog ingest: repo=%s commit=%s tests=%d",
            resolved_repo,
            resolved_commit,
            len(tests),
        )

        rows_reader, row_count = table_for_columnar_rows(
            TEST_CATALOG_TABLE_KEY,
            buffer.data,
        )
        scope = f"{resolved_repo}@{resolved_commit}"
        persist_arrow_tables(
            storage,
            {TEST_CATALOG_TABLE_KEY: rows_reader},
            scope=scope,
        )
        return TestsIngestResult(
            result=ExecutionResult.ok(),
            rows=buffer.data,
            rows_reader=rows_reader,
            row_count=row_count,
        )


@dataclass(frozen=True)
class TestsIngestResult:
    """Result bundle for tests ingestion."""

    result: ExecutionResult
    rows: ColumnarRows = field(default_factory=dict)
    rows_reader: pa.Table = field(
        default_factory=lambda: empty_table_for_table(TEST_CATALOG_TABLE_KEY)
    )
    row_count: int = 0


__all__ = ["TestsIngestResult", "TestsIngestStep"]
