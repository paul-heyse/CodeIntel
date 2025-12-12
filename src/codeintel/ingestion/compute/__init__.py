"""Pure domain logic computation layer for ingestion.

This package contains pure computation implementations that use port injection
for all I/O operations. Each computation follows the pattern:

1. Accept ports via constructor injection
2. Execute pure logic that uses ports for I/O
3. Return a result with row counts and status

This is analogous to graphs/compute/ - stateless computation with no direct
database or filesystem dependencies.

Modules
-------
- ast_extract: Python AST extraction and metrics
- cst_extract: LibCST concrete syntax tree extraction
- docstrings_extract: Docstring parsing and extraction
- typing_ingest: Type annotation analysis
- coverage_ingest: Coverage data processing
- tests_ingest: Test results processing
- scip_ingest: SCIP symbol indexing
- config_ingest: Configuration file flattening
- repo_scan: Repository scanning and module discovery

Example
-------
```python
from codeintel.ingestion.adapters import DuckDBStorageAdapter, FilesystemDiscoveryAdapter
from codeintel.ingestion.compute import AstExtractStep


storage = DuckDBStorageAdapter(gateway)
discovery = FilesystemDiscoveryAdapter(repo_root)


step = AstExtractStep(storage=storage, discovery=discovery)
result = step.execute(modules, repo="my-repo", commit="abc123")
print(f"Wrote {result.rows_written} rows")
```
"""

from __future__ import annotations

from codeintel.ingestion.compute.ast_extract import AstExtractStep
from codeintel.ingestion.compute.base import StepResult
from codeintel.ingestion.compute.config_ingest import ConfigIngestStep
from codeintel.ingestion.compute.coverage_ingest import CoverageIngestStep
from codeintel.ingestion.compute.cst_extract import CstExtractStep
from codeintel.ingestion.compute.docstrings_extract import DocstringsExtractStep
from codeintel.ingestion.compute.repo_scan import RepoScanStep
from codeintel.ingestion.compute.scip_ingest import ScipIngestResult, ScipIngestStep
from codeintel.ingestion.compute.tests_ingest import TestsIngestStep
from codeintel.ingestion.compute.typing_ingest import TypingIngestStep

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
