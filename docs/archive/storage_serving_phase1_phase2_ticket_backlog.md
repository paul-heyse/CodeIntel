# Storage and Serving Phase 1-2 Ticket Backlog

This document converts Phase 1 and Phase 2 of
`docs/storage_serving_architecture_alignment_plan.md` into concrete tickets
with file-level task checklists.

## Phase 1: Canonical SQLGlot AST pipeline

### P1-1: Canonical AST normalization and fingerprinting

Goal
- Standardize AST canonicalization, hashing, and serialization so every
  serving query uses the same deterministic pipeline.

Primary files
- `src/codeintel/storage/sqlglot_tools.py`
- `src/codeintel/serving/semantic/query_ast.py`
- `src/codeintel/serving/semantic/fingerprints.py`

Checklist
- [ ] Add a single canonicalization entrypoint for serving queries that
      normalizes identifiers and qualifies expressions.
- [ ] Ensure fingerprints are derived from the canonical AST rendering,
      not ad hoc SQL strings.
- [ ] Add tests that show stable hashes for equivalent ASTs.
- [ ] Add tests that show hash changes for semantic changes.

Acceptance criteria
- Every serving query produces a canonical SQL representation and stable
  fingerprint in `SemanticQueryResponse`.
- Fingerprints are deterministic across identical inputs.

### P1-2: AST capability registry and envelope

Goal
- Centralize allowed operations and supported function sets for the
  query AST (no duplicated per-engine rules).

Primary files
- `src/codeintel/serving/semantic/routing.py`
- `src/codeintel/serving/semantic/sqlglot_query_builder.py`
- `src/codeintel/serving/semantic/filter_compiler.py`
- `src/codeintel/serving/semantic/duckdb_relation_builder.py`

Checklist
- [ ] Create a shared AST capability map (operators, functions, nodes).
- [ ] Replace duplicated operator checks with the shared map.
- [ ] Ensure query builders reject unsupported constructs early.

Acceptance criteria
- Polars and DuckDB compilers share the same allowed operator set.
- Unsupported AST nodes are rejected deterministically.

### P1-3: AST lineage and diff instrumentation

Goal
- Expose lineage and semantic diffs from the canonical AST for
  observability and debugging.

Primary files
- `src/codeintel/storage/sqlglot_tools.py`
- `src/codeintel/serving/semantic/kernel.py`

Checklist
- [ ] Add standardized AST lineage extraction in the kernel for explain
      responses.
- [ ] Attach semantic diffs when a query is re-issued with the same view
      ID but different AST.
- [ ] Emit lineage metadata into IPC metadata where feasible.

Acceptance criteria
- Explain responses include lineage references where available.
- Diff payloads are stable and derived from canonical ASTs.

## Phase 2: DuckDB relation plan as execution backbone

### P2-1: Expand AST -> DuckDB Expression compiler

Goal
- Cover the full supported AST envelope in the DuckDB relation compiler
  without raw SQL strings.

Primary files
- `src/codeintel/serving/semantic/duckdb_relation_builder.py`
- `src/codeintel/storage/queries/expressions.py`

Checklist
- [ ] Add support for remaining scalar functions and casts used in
      semantic views.
- [ ] Extend join predicate coverage to include AND trees and aliases.
- [ ] Centralize function mapping to DuckDB Expression API.

Acceptance criteria
- Supported ASTs compile to DuckDB relations without SQL strings.
- Compiler rejects unsupported nodes with actionable errors.

### P2-2: DuckDB relation -> Polars adapter

Goal
- Replace the peer Polars engine with a DuckDB-backed adapter using
  `relation.pl(lazy=True)`.

Primary files
- `src/codeintel/serving/semantic/engines/polars_engine.py`
- `src/codeintel/serving/semantic/engines/duckdb_engine.py`
- `src/codeintel/serving/semantic/engines/protocol.py`

Checklist
- [ ] Remove direct dataset scans from the Polars engine path.
- [ ] Accept a DuckDB relation and convert to a Polars LazyFrame.
- [ ] Ensure alignment with contract schemas stays in the DuckDB path.

Acceptance criteria
- Polars execution is always downstream of DuckDB relations.
- Polars engine does not access datasets directly.

### P2-3: DuckDB-first routing defaults

Goal
- Simplify engine routing to prefer DuckDB for auto mode, with Polars
  available only as a secondary adapter.

Primary files
- `src/codeintel/serving/semantic/routing.py`
- `src/codeintel/serving/semantic/engines/registry.py`
- `src/codeintel/serving/settings.py`

Checklist
- [ ] Enforce DuckDB-first ordering in auto preference.
- [ ] Add guardrails to prevent Polars selection when DuckDB is present
      and compatible.
- [ ] Update settings defaults to DuckDB-first behavior.

Acceptance criteria
- Auto routing selects DuckDB unless explicitly overridden.
- Polars can still be requested intentionally when needed.

### P2-4: DuckDB authoritative contracts (PR 1)

Goal
- Resolve contract schemas from DuckDB metadata and relations, not from
  dataset manifests.

Primary files
- `src/codeintel/serving/semantic/schema_contracts.py`
- `src/codeintel/serving/semantic/kernel.py`
- `src/codeintel/serving/semantic/engines/duckdb_engine.py`
- `src/codeintel/serving/semantic/engines/polars_engine.py`

Checklist
- [x] Replace manifest-backed contract resolution with DuckDB-backed
      resolution.
- [x] Merge registry metadata into DuckDB-derived schemas.
- [x] Update all call sites to pass DuckDB connections.

Acceptance criteria
- Serving contract resolution never consults dataset manifests.
- DuckDB relation schema is the authoritative contract.
