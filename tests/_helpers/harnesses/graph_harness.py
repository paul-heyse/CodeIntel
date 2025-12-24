"""Graph target harness helpers for Hamilton execution tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

from codeintel.build.hamilton.native.graphs.call_graph import CALL_GRAPH_TABLE_KEYS
from codeintel.build.hamilton.native.graphs.import_graph import IMPORT_GRAPH_TABLE_KEYS
from tests._helpers.assertions.table_assertions import assert_table_has_rows
from tests._helpers.assertions.target_record_assertions import assert_record_has_datasets
from tests._helpers.harnesses.hamilton_build import (
    HamiltonBuildHarness,
    HarnessConfig,
    HarnessOpenOptions,
)
from tests._helpers.fixtures.repos import write_callgraph_alias_repo

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from codeintel.build.hamilton.run_records import TargetRunRecord


DEFAULT_GRAPH_TARGETS: tuple[str, ...] = ("call_graph", "import_graph")


@dataclass
class GraphTargetHarness:
    """Harness wrapper for graph target execution."""

    harness: HamiltonBuildHarness

    @classmethod
    def open(
        cls,
        tmp_path: Path,
        *,
        harness_config: HarnessConfig | None = None,
        options: HarnessOpenOptions | None = None,
    ) -> GraphTargetHarness:
        """Create a graph harness with a default callgraph repo.

        Returns
        -------
        GraphTargetHarness
            Harness wrapper bound to a HamiltonBuildHarness.
        """
        resolved = options or HarnessOpenOptions(
            repo_strategy="writer",
            repo_writer=write_callgraph_alias_repo,
        )
        base = HamiltonBuildHarness.open(tmp_path, harness=harness_config, options=resolved)
        return cls(base)

    def run_targets(self, targets: Iterable[str] | None = None) -> dict[str, TargetRunRecord]:
        """Run graph targets and return records by target name.

        Returns
        -------
        dict[str, TargetRunRecord]
            Mapping of target name to TargetRunRecord.
        """
        requested = tuple(targets or DEFAULT_GRAPH_TARGETS)
        result = self.harness.run_targets(requested)
        return {target: self.harness.record(target, result=result) for target in requested}

    def assert_graph_tables(self, *, min_rows: int = 1) -> None:
        """Assert core graph tables exist with at least min_rows rows."""
        gateway = self.harness.ctx.gateway
        for table_key in (*CALL_GRAPH_TABLE_KEYS, *IMPORT_GRAPH_TABLE_KEYS):
            assert_table_has_rows(gateway, table_key, min_rows=min_rows)

    @staticmethod
    def assert_graph_datasets(record: TargetRunRecord) -> None:
        """Assert graph datasets are present in a TargetRunRecord."""
        expected = set(CALL_GRAPH_TABLE_KEYS) | set(IMPORT_GRAPH_TABLE_KEYS)
        assert_record_has_datasets(record, sorted(expected))

    @staticmethod
    def assert_call_graph_datasets(record: TargetRunRecord) -> None:
        """Assert call graph datasets are present in a TargetRunRecord."""
        assert_record_has_datasets(record, CALL_GRAPH_TABLE_KEYS)

    @staticmethod
    def assert_import_graph_datasets(record: TargetRunRecord) -> None:
        """Assert import graph datasets are present in a TargetRunRecord."""
        assert_record_has_datasets(record, IMPORT_GRAPH_TABLE_KEYS)

    def close(self) -> None:
        """Close the underlying HamiltonBuildHarness."""
        self.harness.close()

    def __enter__(self) -> Self:
        """Return the harness for context manager usage.

        Returns
        -------
        Self
            This harness instance.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        self.close()


__all__ = [
    "DEFAULT_GRAPH_TARGETS",
    "GraphTargetHarness",
]
