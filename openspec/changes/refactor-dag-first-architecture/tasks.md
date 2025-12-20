## 1. Implementation
- [ ] 1.1 Define artifact layout and versioning for SchemaManifest, DatasetCatalog, and SemanticRegistry.
- [ ] 1.2 Implement artifact loader/provider API in core (manifest-backed SchemaProvider + caches).
- [ ] 1.3 Emit SchemaManifest from the global DAG (deterministic ordering + provenance fields).
- [ ] 1.4 Emit DatasetCatalog from DAG + contracts (dependencies, exports, validation profile).
- [ ] 1.5 Emit SemanticRegistry from TagIndex with schema-backed columns.
- [ ] 1.6 Replace runtime schema/contract providers with artifact-backed providers.
- [ ] 1.7 Centralize JSON Schema generation via core SchemaService and remove duplicates.
- [ ] 1.8 Rebuild metadata.datasets from DatasetCatalog at bootstrap; add dataset_stats if needed.
- [ ] 1.9 Standardize validation on ValidationRunner and implement profile-based check sets.
- [ ] 1.10 Replace serving/build error envelopes with ProblemDetail and unify the error catalog.
- [ ] 1.11 Consolidate export format registry and rename jsonl -> ndjson across build/serving/CLI.
- [ ] 1.12 Route all writes through a single writer (Warehouse or new facade); update DataSavers/IO.
- [ ] 1.13 Remove deprecated providers/shims and update references to new artifacts.

## 2. Tests
- [ ] 2.1 Add import-time safety tests for artifact-backed providers.
- [ ] 2.2 Add determinism tests for SchemaManifest and DatasetCatalog.
- [ ] 2.3 Add export-format tests for ndjson canonicalization.
- [ ] 2.4 Add error-envelope tests ensuring ProblemDetail for HTTP/MCP.
- [ ] 2.5 Add writer-path tests for consistent validation and schema hashing.

## 3. Documentation
- [ ] 3.1 Update DAG-first refinement docs to reflect artifact-first boundary.
- [ ] 3.2 Document DatasetCatalog schema and artifact locations.
- [ ] 3.3 Document ndjson naming and deprecate jsonl references.

## 4. Validation
- [ ] 4.1 Run `openspec validate refactor-dag-first-architecture --strict`.
- [ ] 4.2 Run `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`.
- [ ] 4.3 Run `uv run pytest -q`.
