"""Pure domain logic steps for ingestion.

This package contains step implementations that use port injection for
all I/O operations. Steps are pure domain logic with no direct
dependencies on storage or tool implementations.

Each step follows the pattern:
1. Accept ports via constructor injection
2. Execute pure logic that uses ports for I/O
3. Return a StepResult with row counts and status

Example
-------
```python
from codeintel.ingestion.adapters import DuckDBStorageAdapter, FilesystemDiscoveryAdapter
from codeintel.ingestion.steps import AstExtractStep

# Create adapters
storage = DuckDBStorageAdapter(gateway)
discovery = FilesystemDiscoveryAdapter(repo_root)

# Create and execute step
step = AstExtractStep(storage=storage, discovery=discovery)
result = step.execute(modules, repo="my-repo", commit="abc123")
print(f"Wrote {result.rows_written} rows")
```
"""

from __future__ import annotations

from dataclasses import dataclass, field

from codeintel.ingestion.steps.ast_extract import AstExtractStep
from codeintel.ingestion.steps.config_ingest import ConfigIngestStep
from codeintel.ingestion.steps.coverage_ingest import CoverageIngestStep
from codeintel.ingestion.steps.cst_extract import CstExtractStep
from codeintel.ingestion.steps.docstrings_extract import DocstringsExtractStep
from codeintel.ingestion.steps.repo_scan import RepoScanStep
from codeintel.ingestion.steps.scip_ingest import ScipIngestResult, ScipIngestStep
from codeintel.ingestion.steps.tests_ingest import TestsIngestStep
from codeintel.ingestion.steps.typing_ingest import TypingIngestStep


@dataclass
class StepResult:
    """Result from executing an ingestion step.

    Attributes
    ----------
    rows_written
        Total number of rows written across all tables.
    table_counts
        Mapping of table names to row counts.
    errors
        List of error messages encountered.
    skipped
        Whether the step was skipped.
    skip_reason
        Reason for skipping if applicable.
    """

    rows_written: int = 0
    table_counts: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str | None = None

    @property
    def success(self) -> bool:
        """Return True if no errors occurred.

        Returns
        -------
        bool
            True if no errors.
        """
        return not self.errors and not self.skipped

    @staticmethod
    def ok(
        rows_written: int = 0,
        table_counts: dict[str, int] | None = None,
    ) -> StepResult:
        """Create a successful result.

        Parameters
        ----------
        rows_written
            Total rows written.
        table_counts
            Optional mapping of table names to counts.

        Returns
        -------
        StepResult
            Success result.
        """
        return StepResult(
            rows_written=rows_written,
            table_counts=table_counts or {},
        )

    @staticmethod
    def fail(error: str) -> StepResult:
        """Create a failed result.

        Parameters
        ----------
        error
            Error message.

        Returns
        -------
        StepResult
            Failure result.
        """
        return StepResult(errors=[error])

    @staticmethod
    def skip(reason: str) -> StepResult:
        """Create a skipped result.

        Parameters
        ----------
        reason
            Reason for skipping.

        Returns
        -------
        StepResult
            Skipped result.
        """
        return StepResult(skipped=True, skip_reason=reason)


__all__ = [
    "AstExtractStep",
    "ConfigIngestStep",
    "CoverageIngestStep",
    "CstExtractStep",
    "DocstringsExtractStep",
    "RepoScanStep",
    "ScipIngestResult",
    "ScipIngestStep",
    "StepResult",
    "TestsIngestStep",
    "TypingIngestStep",
]
