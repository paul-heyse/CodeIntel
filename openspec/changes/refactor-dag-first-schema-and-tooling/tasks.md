## 1. Schema Authority
- [x] 1.1 Make SchemaIndex/UnifiedSchemaProvider canonical for DAG outputs
- [x] 1.2 Refactor SCHEMA_REGISTRY to project from the unified provider + constraints overlay
- [x] 1.3 Restrict declared schemas to source-only inputs and explicit overrides
- [x] 1.4 Generate Pandera schemas and row bindings from TableSchema + constraints
- [x] 1.5 Add schema provenance and inference status to schema manifests
- [x] 1.6 Add validation to reject missing overrides for non-inferable outputs
- [x] 1.7 Enforce hard-failure policy for inference without non-DAG alternatives
- [x] 1.8 Implement constraint enforcement order (Hamilton, Pandera, Pydantic)
- [x] 1.9 Introduce contract resolution modes and DAG-free registry for CLI enumeration

## 2. Tool Execution Unification
- [x] 2.1 Add ToolService/ToolRunner to BuildEnv Providers as canonical dependencies
- [x] 2.2 Replace Real* tool helpers and SubprocessToolRunner with ToolService-backed adapters
- [x] 2.3 Update Hamilton targets and analytics tooling to use ToolService via BuildEnv
- [x] 2.4 Remove legacy tool execution entrypoints from public APIs

## 3. Hamilton Ingestion Consolidation
- [x] 3.1 Create Hamilton-native SCIP ingestion target that writes core.scip_* tables
- [x] 3.2 Remove or internalize non-DAG ingestion entrypoints (ScipIngestStep, etc.)
- [x] 3.3 Align ingestion results on unified ExecutionResult semantics

## 4. Change Detection and Hashing Alignment
- [x] 4.1 Align ingestion change detection with build input hashes/fingerprints
- [x] 4.2 Surface file-state hashes or deltas into target options for hashing
- [x] 4.3 Ensure skip/rebuild decisions use a single hash/fingerprint authority
- [x] 4.4 Persist change-detection deltas alongside build manifests

## 5. Resource and DI Alignment
- [x] 5.1 Expose analytics ResourceRegistry via BuildEnv Providers or unify registry access
- [x] 5.2 Update analytics resource construction to use injected providers
- [x] 5.3 Remove migration-only compatibility shims after alignment

## 6. Registry and Row Serialization Convergence
- [x] 6.1 Converge registries on Hamilton tags/TargetSpec metadata for discovery
- [x] 6.2 Centralize row serialization on schema registry row models

## 7. Tests and Validation
- [x] 7.1 Add schema provenance and inference tests
- [x] 7.2 Add tool execution integration tests for ToolService usage
- [x] 7.3 Add ingestion target tests for SCIP table writes
- [x] 7.4 Add change detection/hash alignment tests
- [x] 7.5 Add resource registry alignment tests
- [x] 7.6 Add row serialization consistency tests
- [x] 7.7 Run `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
- [ ] 7.8 Run `uv run pytest -q`
