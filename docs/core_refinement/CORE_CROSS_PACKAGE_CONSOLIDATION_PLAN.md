# CORE Cross-Package Consolidation Plan

Status: draft

## Goal

Consolidate duplicated/shared functionality centered in `src/codeintel/core` and the primary consumer
packages (`build/`, `storage/`, `serving/`, `cli/`, `analytics/`, `graphs/`, `ingestion/`) to:

- Remove duplication and semantic drift
- Improve layering hardness (clear dependency direction)
- Increase maintainability/extensibility without reducing functionality
- Keep quality gates green (Ruff, Pyright strict, Pyrefly, pytest)

## Scope (items covered)

This plan covers the following consolidation opportunities:

1. RFC 9457 Problem Details duplication (core/cli/serving)
2. `ValidationResult` type-name collisions (core/options, core/plugins, cli, analytics)
3. Path normalization duplication and unsafe semantics (`normalize_path`)
4. Parsed-code model duplication (`ParsedFunction`/`ParsedModule`) across core vs graphs
5. Table-key parsing + dataset contract default logic duplicated across build vs storage (+ scattered splits)
6. ColumnType parsing/allowed-set duplication (core serde vs cli vs serving)
7. ColumnType → dtype mapping + Pandera → JSON Schema conversion duplication (core vs build; plus ibis dtype map)
8. Type normalization drift between hashing vs inference (DECIMAL/BIGINT semantics)
9. Hashing/fingerprinting duplication (stable JSON canonicalization; SQL fingerprinting)
10. Layering violations where `core` directly depends on `storage` (catalog implementation; duckdb exception bundle)

## Non-goals

- Changing external CLI/HTTP/MCP semantics (response payload shape, exit codes, HTTP codes) unless explicitly
  justified and tested.
- Large-scale renames/reorganization of unrelated modules.
- Introducing new frameworks or new runtime dependencies without a clear consolidation payoff.

## Quality gates / acceptance criteria

For every phase (and at the end):

- `uv run python -m tools.quality_report --output build/quality-results/quality_report.json` passes with
  **zero** Ruff/Pyright/Pyrefly errors.
- `uv run pytest -q` passes.
- No new `type: ignore`, no lint suppressions, no commented-out code.
- Public APIs removed only after call-sites are migrated (or after an explicit deprecation window).

## High-level sequencing strategy

Prefer “introduce canonical API → migrate call-sites → delete duplicates” with small PR-sized slices.

Suggested order:

1. Low-risk pure-logic consolidation inside `core` (`normalize_path`)
2. Narrow “leaf” utilities that many packages use (ColumnType parsing, table-key parsing)
3. Larger cross-cutting types (`ProblemDetail`, validation outcomes)
4. Schema conversions (dtype maps, pandera→json schema) and hashing/type normalization
5. Hardening layering boundaries (move storage-backed catalog impl out of `core`)

## Phase 0 — Baseline + inventory (1 PR)

1. Capture current usage to reduce migration risk:
   - Run `rg` for the following symbols and record counts + primary call sites:
     - `ProblemDetail`
     - `ValidationResult`
     - `normalize_path`
     - `schema_hash`, `canonical_type`, `normalize_duckdb_type`
     - `split_table_key`, `table_key.split(".", ...)`
     - `pandera_to_json_schema`, `_dtype_for_column_type`, `dtype_map`
     - `sqlglot_canonical_sha256`, `fingerprint_sql_duckdb`
2. Establish a “golden” contract for behaviors that must not change:
   - Problem Details JSON fields (CLI and HTTP)
   - Path normalization outputs for representative inputs
   - Schema hashing stability rules (explicitly documented)
   - Dataset export default filename rules

Deliverable: a short checklist section appended to this doc with “verified baseline behaviors” and links
to tests (existing or newly added) that lock them in.

## Phase 1 — Path normalization: single semantics everywhere (1 PR)

### Current state

- Canonical normalization exists in `src/codeintel/core/paths/normalize.py` (`normalize_path(path: str | Path)`).
- A second implementation exists in `src/codeintel/core/catalog/span_index.py` that uses
  `path.replace("\\\\", "/").lstrip("./")` and can incorrectly rewrite `../x` → `x`.

### Plan

1. Remove the local `normalize_path` function in `src/codeintel/core/catalog/span_index.py`.
2. Import and use `codeintel.core.paths.normalize_path` instead.
3. Add tests for `normalize_path` covering:
   - Windows separators
   - `.` and `./` behavior
   - `../` behavior (ensure it is not silently stripped)
   - idempotency: `normalize_path(normalize_path(x)) == normalize_path(x)`
4. Run quality gates.

### Acceptance

- All call-sites use exactly one `normalize_path` implementation.
- Span lookups continue to work for typical repo-relative paths and become strictly safer for edge cases.

## Phase 2 — Problem Details: one canonical data object + per-transport adapters (1–2 PRs)

### Current state

Three `ProblemDetail` classes exist:

- Core dataclass: `src/codeintel/core/errors/problem_details.py`
- CLI dataclass: `src/codeintel/cli/errors/_cli_errors.py`
- Serving HTTP pydantic model: `src/codeintel/serving/http/errors.py`

### Design decision

- **Canonical representation**: `codeintel.core.errors.problem_details.ProblemDetail` (dataclass).
- **Transport models**:
  - CLI: use canonical dataclass directly (serialize via `to_dict()` / `to_json()`).
  - HTTP: keep a pydantic model, but rename it to avoid a second `ProblemDetail` type in the codebase
    (e.g., `HttpProblemDetail`), and provide a `from_core_problem_detail(...)` adapter.

### Plan

1. CLI:
   - Replace uses of `codeintel.cli.errors._cli_errors.ProblemDetail` with imports from
     `codeintel.core.errors.problem_details`.
   - If CLI-specific fields are required, store them in `extensions` with a documented namespacing
     convention (e.g., `cli_*` keys), rather than forking the dataclass.
   - Delete `ProblemDetail` dataclass from `src/codeintel/cli/errors/_cli_errors.py`.
2. Serving HTTP:
   - Rename pydantic `ProblemDetail` to `HttpProblemDetail`.
   - Add a single adapter function:
     - `http_problem_from_error_response(...)` and/or `http_problem_from_core_problem(...)`
       that converts the canonical core dataclass (plus correlation id) into the HTTP model.
   - Ensure HTTP responses still emit `application/problem+json` and include current extension fields.
3. Tests:
   - Add snapshot-style tests for CLI JSON problem output and HTTP problem response serialization
     (exclude unstable correlation/instance ids by injecting fixed values).
4. Docs:
   - Update any docs that describe Problem Details to point at the canonical core class.

### Acceptance

- Only one “ProblemDetail” *data* type exists; transport-specific models are explicitly named.
- CLI and HTTP outputs are byte-for-byte compatible (aside from explicitly allowed fields like correlation id).

## Phase 3 — Validation outcomes: eliminate name collisions and share patterns (1–3 PRs)

### Current state

Multiple unrelated “ValidationResult” types exist:

- Options validation: `src/codeintel/core/options/protocol.py`
- Plugin validation: `src/codeintel/core/plugins/types/protocol.py`
- CLI input validation: `src/codeintel/cli/introspection/validation.py`
- Analytics parsing: `src/codeintel/analytics/parsing/compute.py` (row container)

### Design options (pick one up-front)

Option A (recommended): Introduce one canonical type for “boolean outcome + messages”

- Add `codeintel.core.validation.outcome.ValidationOutcome`:
  - fields: `ok: bool`, `errors: tuple[str, ...]`, `warnings: tuple[str, ...]`
  - helpers: `.success()`, `.failure(*errors)`, `.with_warnings(*warnings)`, `.merge(...)`
- Migrate:
  - `core.options.ValidationResult` → alias or replace with `ValidationOutcome`
  - `core.plugins.types.ValidationResult` → rename to `PluginValidationResult` (or reuse `ValidationOutcome`)
  - Keep CLI’s generic validation result as-is but rename to `ParseResult`/`FieldValidationResult`
    if it frequently leaks into other layers.
  - Rename analytics’ row container to `ValidationRows` (it is not an “ok/errors” outcome).

Option B: Keep multiple types but rename all except one (lower effort, less consistency).

### Plan (Option A)

1. Create `src/codeintel/core/validation/outcome.py` (new canonical type).
2. Update `src/codeintel/core/options/protocol.py`:
   - Replace its `ValidationResult` with the canonical type (or re-export it).
3. Update `src/codeintel/core/plugins/types/protocol.py`:
   - Either reuse `ValidationOutcome` or rename local class to `PluginValidationResult`.
4. Update all cross-package imports and docstrings referencing old names.
5. Rename analytics row container:
   - `src/codeintel/analytics/parsing/compute.py`: `ValidationResult` → `ValidationRows`.
6. Tests:
   - Add small unit tests for merge behavior and serialization (if any).
7. Run quality gates.

### Acceptance

- No ambiguous “ValidationResult” types crossing package boundaries.
- Shared behavior is implemented once and reused.

## Phase 4 — Parsing models: unify core + graphs representations (2–4 PRs)

### Current state

- Core “canonical” parsed types: `src/codeintel/core/parsing/models.py`
- Graphs has parallel types: `src/codeintel/graphs/ports/parsing.py`

Core already contains “graph compatibility fields” (`is_async`, `decorator_names`, `parameters`), which
is a strong signal these should converge.

### Plan

1. Define a single “common parsed function” contract in core:
   - Either:
     - Make `codeintel.core.parsing.models.ParsedFunction` the single canonical type for both, or
     - Define Protocols in `codeintel.core.parsing.protocols` (for “lightweight view” compatibility).
2. Add explicit conversion helpers:
   - `core → graphs` and/or `graphs → core` conversion functions during migration (temporary).
3. Migrate graphs:
   - Update graphs callgraph code to import from core where possible.
   - Remove `src/codeintel/graphs/ports/parsing.py` dataclasses once all call sites are migrated.
4. Update analytics and other imports to converge on core.
5. Tests:
   - Add a small “round trip” test for conversion helpers if they exist during transition.

### Acceptance

- One canonical parsed-code model.
- No duplicated dataclasses for the same concept.

## Phase 5 — Table keys + contract defaults: core-owned utilities (2–5 PRs)

### Current state

- Table key utility exists in storage only: `src/codeintel/storage/helpers/table_key.py`
- Build and storage duplicate contract-default logic:
  - `src/codeintel/build/schemas/contract_provider.py`
  - `src/codeintel/storage/contracts/provider.py`
- Many call sites manually do `table_key.split(".", 1)` across the repo.

### Design

Create a core-owned module that is dependency-safe for both build and storage:

- `codeintel.core.schemas.table_keys`
  - `split_table_key(table_key: str) -> tuple[str, str]` (strict)
  - `try_split_table_key(...) -> tuple[str, str] | None` (optional helper)
  - `schema_prefix(table_key: str) -> str | None`
  - `table_name(table_key: str) -> str`
  - `is_docs_view_key(table_key: str) -> bool` (canonical)
- `codeintel.core.schemas.contract_defaults`
  - `_exportable_by_default(...)`
  - `default_json_schema_id(...)`
  - `default_jsonl_filename(...)`
  - `default_parquet_filename(...)`
  - `owner_package_from_schema_prefix(...)`

### Plan

1. Add the new core modules and unit tests for:
   - table key parsing behavior
   - exportable-by-default matrix (core/analytics/graph/build/docs views)
   - default filename/id generation
2. Migrate storage contract provider to use the core module (delete its duplicated helpers).
3. Migrate build contract provider to use the core module (delete its duplicated helpers).
4. Replace scattered `table_key.split(".", 1)` usage where it is semantically table-key parsing:
   - Start with build/storage providers + any table-key parsing in serving/analytics utilities.
5. Run quality gates.

### Acceptance

- Only one implementation of contract-default rules exists.
- Table-key parsing is consistent and tested.

## Phase 6 — ColumnType parsing + allowed-set: single source of truth (1–3 PRs)

### Current state

- Core column parsing in `src/codeintel/core/schemas/serde.py`
- CLI build_schema has its own `_ALLOWED_COLUMN_TYPES` and `_parse_column_type`
- Serving semantic inventory hard-codes allowed set and parser

### Design

Expose a public parsing API in core, and reuse it everywhere:

- `codeintel.core.schemas.types` (or similar):
  - `ALLOWED_COLUMN_TYPES: frozenset[str]`
  - `parse_column_type(value: object, *, ctx: str) -> ColumnType` (or `field=` naming)

### Plan

1. Create the canonical parser and set in core (can wrap existing `serde._parse_column_type`).
2. Update:
   - `src/codeintel/cli/handlers/build_schema.py`
   - `src/codeintel/serving/semantic/inventory.py`
   - `src/codeintel/core/schemas/serde.py` (internals call the shared helper)
3. Delete local copies/hard-coded sets.
4. Add tests:
   - parsing accepts valid literals
   - rejects invalid types with informative error messages
5. Run quality gates.

### Acceptance

- Only one ColumnType validation surface exists.
- Serving/CLI stay in lock-step with core schema primitives.

## Phase 7 — Schema conversion unification (dtype maps + pandera→json schema) (2–5 PRs)

### Current state

- ColumnType → Pandera dtype mapping duplicated:
  - `src/codeintel/core/schemas/pandera_gen.py`
  - `src/codeintel/build/hamilton/contracts/schemas/pandera_schemas.py`
- Pandera → JSON Schema duplicated:
  - `src/codeintel/core/schemas/json_schema_gen.py`
  - `src/codeintel/build/hamilton/contracts/schemas/pandera_schemas.py`
- ColumnType → ibis dtype map lives in storage:
  - `src/codeintel/storage/schema_roundtrip.py`

### Design

Create a single mapping layer in core:

- `codeintel.core.schemas.type_mappings`
  - `pandera_dtype_for_column_type(col_type: ColumnType) -> PanderaDtype`
  - `ibis_dtype_for_column_type(col_type: ColumnType) -> ibis.DataType`
  - `json_schema_type_for_column_type(col_type: ColumnType) -> dict[str, object]`

Then:

- Core owns “primitive conversions” (TableSchema → Pandera schema, TableSchema → JSON schema, Pandera → JSON schema)
- Build owns “constraints + enrichment” (checks, schema descriptions) and composes on top, rather than copying.

### Plan

1. Add `type_mappings` module in core and migrate:
   - `core/schemas/pandera_gen.py` to call it
   - `core/schemas/json_schema_gen.py` to call it (for TableSchema conversions)
2. Build:
   - Replace `_dtype_for_column_type` and `pandera_to_json_schema` duplicates with imports from core.
   - Keep build-only augmentation hooks:
     - e.g., apply `_SCHEMA_DESCRIPTIONS` after calling core conversion.
3. Storage:
   - Migrate `storage/schema_roundtrip.py` to use `ibis_dtype_for_column_type` mapping from core.
4. Tests:
   - Add cross-module tests asserting consistency (same ColumnType maps to same family across pandera/ibis/jsonschema).
5. Run quality gates.

### Acceptance

- One authoritative mapping exists for ColumnType conversions.
- Build and storage stop drifting on dtype semantics.

## Phase 8 — Type normalization + schema hashing alignment (1–3 PRs)

### Current state

- `codeintel.core.schemas.hashing.canonical_type()` collapses all DECIMAL* and BIGINT to BIGINT
  (hash intentionally ignores scale/precision).
- `codeintel.build.schemas.infer_duckdb.normalize_duckdb_type()` distinguishes DECIMAL and DECIMAL(38,0).

### Decision to make explicit

Choose and document the contract for schema hashing:

- If schema hashing is intended to detect changes that affect query semantics, collapsing DECIMAL may be too lossy.
- If schema hashing is intended to be stable across minor/representation differences, collapsing may be correct,
  but then inference/mapping should align with that rule.

### Plan

1. Create `codeintel.core.schemas.duckdb_types`:
   - Move/duplicate `normalize_duckdb_type` into core (pure string/regex; no sqlglot dependency needed).
2. Decide whether `canonical_type` should:
   - preserve `DECIMAL(38,0)` vs `DECIMAL`, and/or
   - preserve `BIGINT` vs `DECIMAL(38,0)` where relevant.
3. If changing hashing semantics:
   - Introduce a versioned hash function (e.g., `schema_hash_v1`, `schema_hash_v2`) and migrate call sites safely.
   - Update stored metadata migration paths (storage bootstrap expectations, caches).
4. Add tests that lock the intended behavior and prevent future drift.
5. Run quality gates.

### Acceptance

- Type normalization contract is explicit and shared.
- Hash stability is preserved or deliberately versioned with a migration.

## Phase 9 — Hashing/fingerprinting deduplication (1–3 PRs)

### Current state

- Serving implements stable JSON canonicalization + short hash (`src/codeintel/serving/semantic/fingerprints.py`).
- Core has hashing utilities (`src/codeintel/core/hashing/fingerprint.py`, `src/codeintel/core/hashing/content.py`).
- SQL fingerprinting duplicated:
  - Storage: `src/codeintel/storage/sqlglot_tools.py`
  - Serving MCP: `src/codeintel/serving/mcp/sql_fingerprint.py`

### Plan

1. Extract “stable JSON dumps” and “short sha256 hex” into `codeintel.core.hashing`:
   - Implement as small, dependency-free helpers.
2. Update serving semantic fingerprints to import and reuse core helpers.
3. SQL fingerprinting:
   - Replace `serving/mcp/sql_fingerprint.py` with calls to `storage/sqlglot_tools.fingerprint_sql_duckdb`
     (or move a minimal shared sql fingerprint helper to core if layering requires).
4. Delete duplicated serving implementations once migrated.
5. Add tests for:
   - stable JSON output determinism
   - SQL fingerprint stability for representative queries (including parse-failure fallback semantics)
6. Run quality gates.

### Acceptance

- Only one implementation exists for each hashing/fingerprinting primitive.
- Fingerprints remain stable and deterministic across packages.

## Phase 10 — Layering hardness: remove `core → storage` dependencies (2–6 PRs)

### Current state (examples)

- `core` catalog implementation imports storage gateway (`src/codeintel/core/catalog/service.py:25`).
- `core` plugin execution errors import DuckDB error bundles from storage (`src/codeintel/core/errors/execution.py:11`).

These blur the “core is a shared, dependency-light substrate” boundary.

### Target architecture

- `core`:
  - protocols, data models, pure logic
  - may depend on third-party libs when justified, but must not import other CodeIntel top-level packages
    (`build`, `storage`, `serving`, `cli`, `analytics`, `graphs`, `ingestion`) at runtime
  - type-only imports allowed under `TYPE_CHECKING`
- Concrete implementations live in the owning layer:
  - storage-backed catalog: `codeintel.storage.catalog.*`
  - duckdb exception bundles: `codeintel.storage.*` (and injected upward)

### Plan

1. Catalog:
   - Extract storage-backed pieces from `codeintel.core.catalog.service` into `codeintel.storage.catalog.service`.
   - Leave behind in core:
     - `SpanIndex`, `FunctionSpan` primitives, and catalog protocols/interfaces.
   - Update call sites (graphs/analytics/etc.) to import the storage-backed implementation from `storage`.
   - If backwards compatibility is needed:
     - keep a thin facade in `core` that *only* re-exports types (not implementations), or
     - provide an explicit migration path and delete the old module after a deprecation window.
2. Plugin execution catchable errors:
   - Move `DUCKDB_ERRORS` usage out of core:
     - Option A: define `PLUGIN_CATCHABLE_ERRORS` without duckdb specifics in core and let storage layer extend it.
     - Option B: inject the extra exception tuple from storage into the execution environment/context.
3. Add a CI-style enforcement step (local-only to start):
   - A simple import graph check ensuring `src/codeintel/core/**` does not import `codeintel.storage.*` at runtime.
4. Run quality gates + full pytest.

### Acceptance

- `core` no longer imports storage (except under `TYPE_CHECKING`).
- Concrete integrations move to the correct layer without breaking functionality.

## Rollout checklist (applies to every phase)

For each PR:

- [ ] Update or add tests to lock in intended behavior
- [ ] Migrate call sites in all affected top-level packages
- [ ] Delete the duplicate implementation(s)
- [ ] Run `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
- [ ] Run `uv run pytest -q`
- [ ] Update docs and module docstrings to point to the canonical API

## Appendix — Quick reference: primary duplication sites

- ProblemDetail:
  - `src/codeintel/core/errors/problem_details.py`
  - `src/codeintel/cli/errors/_cli_errors.py`
  - `src/codeintel/serving/http/errors.py`
- ValidationResult:
  - `src/codeintel/core/options/protocol.py`
  - `src/codeintel/core/plugins/types/protocol.py`
  - `src/codeintel/cli/introspection/validation.py`
  - `src/codeintel/analytics/parsing/compute.py`
- normalize_path:
  - `src/codeintel/core/paths/normalize.py`
  - `src/codeintel/core/catalog/span_index.py`
- ParsedFunction/ParsedModule:
  - `src/codeintel/core/parsing/models.py`
  - `src/codeintel/graphs/ports/parsing.py`
- Contract defaults + table-key parsing:
  - `src/codeintel/build/schemas/contract_provider.py`
  - `src/codeintel/storage/contracts/provider.py`
  - `src/codeintel/storage/helpers/table_key.py`
  - scattered `table_key.split(".", 1)` call sites
- Schema conversions:
  - `src/codeintel/core/schemas/pandera_gen.py`
  - `src/codeintel/build/hamilton/contracts/schemas/pandera_schemas.py`
  - `src/codeintel/core/schemas/json_schema_gen.py`
  - `src/codeintel/storage/schema_roundtrip.py`
- SQL fingerprinting:
  - `src/codeintel/storage/sqlglot_tools.py`
  - `src/codeintel/serving/mcp/sql_fingerprint.py`

