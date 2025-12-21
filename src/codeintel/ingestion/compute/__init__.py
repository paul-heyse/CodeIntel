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
- config_ingest: Configuration file flattening
- repo_scan: Repository scanning and module discovery

Note
----
This package is intended for internal use by Hamilton-native ingestion targets.
Import modules directly (e.g., ``codeintel.ingestion.compute.ast_extract``) rather
than relying on package-level re-exports.
"""

from __future__ import annotations

__all__: list[str] = []
