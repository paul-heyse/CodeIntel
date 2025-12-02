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

from codeintel.ingestion.steps.ast_extract import AstExtractStep
from codeintel.ingestion.steps.base import StepResult
from codeintel.ingestion.steps.config_ingest import ConfigIngestStep
from codeintel.ingestion.steps.coverage_ingest import CoverageIngestStep
from codeintel.ingestion.steps.cst_extract import CstExtractStep
from codeintel.ingestion.steps.docstrings_extract import DocstringsExtractStep
from codeintel.ingestion.steps.repo_scan import RepoScanStep
from codeintel.ingestion.steps.scip_ingest import ScipIngestResult, ScipIngestStep
from codeintel.ingestion.steps.tests_ingest import TestsIngestStep
from codeintel.ingestion.steps.typing_ingest import TypingIngestStep

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
