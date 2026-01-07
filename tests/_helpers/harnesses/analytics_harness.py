"""Analytics target harness helpers for Hamilton execution tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

from tests._helpers.assertions.table_assertions import assert_table_has_rows
from tests._helpers.fixtures.repos import write_graph_metrics_repo
from tests._helpers.harnesses.hamilton_build import (
    HamiltonBuildHarness,
    HarnessConfig,
    HarnessOpenOptions,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from codeintel.build.hamilton.run_records import TargetRunRecord


DEFAULT_ANALYTICS_TARGETS: tuple[str, ...] = ("function_types",)
FUNCTION_TYPES_TABLE_KEY = "analytics.function_types"


@dataclass
class AnalyticsTargetHarness:
    """Harness wrapper for analytics target execution."""

    harness: HamiltonBuildHarness

    @classmethod
    def open(
        cls,
        tmp_path: Path,
        *,
        harness_config: HarnessConfig | None = None,
        options: HarnessOpenOptions | None = None,
    ) -> AnalyticsTargetHarness:
        """Create an analytics harness with a default repo fixture.

        Returns
        -------
        AnalyticsTargetHarness
            Harness wrapper bound to a HamiltonBuildHarness.
        """
        resolved = options or HarnessOpenOptions(
            repo_strategy="writer",
            repo_writer=write_graph_metrics_repo,
        )
        base = HamiltonBuildHarness.open(tmp_path, harness=harness_config, options=resolved)
        return cls(base)

    def run_targets(self, targets: Iterable[str] | None = None) -> dict[str, TargetRunRecord]:
        """Run analytics targets and return records by target name.

        Returns
        -------
        dict[str, TargetRunRecord]
            Mapping of target name to TargetRunRecord.
        """
        requested = tuple(targets or DEFAULT_ANALYTICS_TARGETS)
        result = self.harness.run_targets(requested)
        return {target: self.harness.record(target, result=result) for target in requested}

    def assert_function_types(self, *, min_rows: int = 1) -> None:
        """Assert function types table exists with at least min_rows rows."""
        gateway = self.harness.ctx.gateway
        assert_table_has_rows(gateway, FUNCTION_TYPES_TABLE_KEY, min_rows=min_rows)

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
    "DEFAULT_ANALYTICS_TARGETS",
    "AnalyticsTargetHarness",
]
