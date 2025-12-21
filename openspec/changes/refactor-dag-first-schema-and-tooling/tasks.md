## 1. Schema Authority
- [ ] 1.1 Make SchemaIndex/UnifiedSchemaProvider canonical for DAG outputs
- [ ] 1.2 Refactor SCHEMA_REGISTRY to project from the unified provider + constraints overlay
- [ ] 1.3 Restrict declared schemas to source-only inputs and explicit overrides
- [ ] 1.4 Generate Pandera schemas and row bindings from TableSchema + constraints
- [ ] 1.5 Add schema provenance and inference status to schema manifests
- [ ] 1.6 Add validation to reject missing overrides for non-inferable outputs
- [ ] 1.7 Enforce hard-failure policy for inference without non-DAG alternatives
- [ ] 1.8 Implement constraint enforcement order (Hamilton, Pandera, Pydantic)

## 2. Tool Execution Unification
- [ ] 2.1 Add ToolService/ToolRunner to BuildEnv Providers as canonical dependencies
- [ ] 2.2 Replace Real* tool helpers and SubprocessToolRunner with ToolService-backed adapters
- [ ] 2.3 Update Hamilton targets and analytics tooling to use ToolService via BuildEnv
- [ ] 2.4 Remove legacy tool execution entrypoints from public APIs

## 3. Hamilton Ingestion Consolidation
- [ ] 3.1 Create Hamilton-native SCIP ingestion target that writes core.scip_* tables
- [ ] 3.2 Remove or internalize non-DAG ingestion entrypoints (ScipIngestStep, etc.)
- [ ] 3.3 Align ingestion results on unified ExecutionResult semantics

## 4. Change Detection and Hashing Alignment
- [ ] 4.1 Align ingestion change detection with build input hashes/fingerprints
- [ ] 4.2 Surface file-state hashes or deltas into target options for hashing
- [ ] 4.3 Ensure skip/rebuild decisions use a single hash/fingerprint authority
- [ ] 4.4 Persist change-detection deltas alongside build manifests

## 5. Resource and DI Alignment
- [ ] 5.1 Expose analytics ResourceRegistry via BuildEnv Providers or unify registry access
- [ ] 5.2 Update analytics resource construction to use injected providers
- [ ] 5.3 Remove migration-only compatibility shims after alignment

## 6. Registry and Row Serialization Convergence
- [ ] 6.1 Converge registries on Hamilton tags/TargetSpec metadata for discovery
- [ ] 6.2 Centralize row serialization on schema registry row models

## 7. Tests and Validation
- [ ] 7.1 Add schema provenance and inference tests
- [ ] 7.2 Add tool execution integration tests for ToolService usage
- [ ] 7.3 Add ingestion target tests for SCIP table writes
- [ ] 7.4 Add change detection/hash alignment tests
- [ ] 7.5 Add resource registry alignment tests
- [ ] 7.6 Add row serialization consistency tests
- [ ] 7.7 Run `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
- [ ] 7.8 Run `uv run pytest -q`
