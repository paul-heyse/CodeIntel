# Storage Typing Hardening + "Type Ignore" Elimination — Implementation Plan

**Status**: Implementation plan (storage-first)  
**Last updated**: 2025-12-18  
**Primary scope**: `src/codeintel/storage/**`  
**Secondary scope (optional, later)**: `src/codeintel/serving/**` consumers of repository outputs

## 0) Why This Plan Exists

Static typing in the storage layer currently depends on a small number of “escape hatches”:

- Ibis operator overloading does not type-check cleanly in many cases, leading to repeated `cast("Any", ...)`
  in filters, `.isin(...)`, `.isnull()` checks, and boolean composition.
- Query results (`.execute()`, `.fetchone()`) are dynamically typed (DuckDB/Pandas scalars), causing ad-hoc casts.
- DataFrame → dict conversion erases types and pushes `Any` across repository boundaries.

Even if there are few or no literal `# type: ignore` suppressions in `src/codeintel/storage/**`, the same root
causes show up as `Any` propagation and repeated cast patterns. This plan removes the *need* for any suppression
by making types first-class in the design.

## 1) Goals and Non‑Goals

### Goals (must)

1) **Single typing boundary for Ibis expression semantics**
   - Remove scattered `cast("Any", ...)` and centralize all “Ibis typing shims” in one module.
2) **Runtime-checked scalar execution**
   - Replace “trust me” casts of `.execute()` results with typed adapters that enforce contracts.
3) **Tighter repository return surfaces**
   - Avoid leaking `dict[str, Any]` as the default storage API shape.
4) **Best‑in‑class maintainability**
   - Make the intended typing pattern obvious, uniform, and test-covered.

### Non‑goals (for this tranche)

- Upstream changes to Ibis typing stubs.
- Broad refactors in build/serving unless required by storage contract changes.
- Introducing permanent lint/type suppressions (no `# type: ignore`, no unused `# noqa`).

## 2) Baseline Inventory (What We’re Fixing)

### Current suppression pressure (storage-only)

- Literal `# type: ignore`: **none** in `src/codeintel/storage/**`.
- “Implicit suppression” via repeated casting:
  - `cast("Any", ...)`: ~30 occurrences concentrated in:
    - `src/codeintel/storage/ibis_types.py`
    - `src/codeintel/storage/queries/safe.py`
    - `src/codeintel/storage/repositories/*`
    - `src/codeintel/storage/views/ibis_views.py`

### Root causes

- Ibis expressions are typed as Python operators (`==`, `&`, `.isin`) rather than Ibis expression nodes.
- Scalar results returned from DuckDB/Ibis are dynamically typed.
- Repository conversion through Pandas `to_dict` erases types.

## 3) Acceptance Gates (must stay green)

Run after each workstream:

```bash
uv run ruff check --fix src/codeintel/storage tests/storage
uv run pyright --warnings --pythonversion=3.13 src/codeintel/storage tests/storage
uv run pyrefly check src/codeintel/storage tests/storage
uv run pytest -q tests/storage
```

## 4) Workstreams and Sequencing

### W1 — Centralize Ibis Predicate Typing (Primary typing seam)

**Objective**: eliminate `cast("Any", ...)` at call sites by providing a complete set of typed predicate
and helper APIs.

**Current seam**: `src/codeintel/storage/ibis_types.py`

**Design**

- Treat `codeintel.storage.ibis_types` as the *only* location where we:
  - cast from “operator overload returns object/bool per stubs” → `it.BooleanValue` / `it.Value`
  - normalize predicate composition
  - handle the two `.isin(...)` shapes (values vs column/subquery)

**Implementation tasks**

1) Introduce explicit typed predicate helpers to cover current call-site patterns:
   - `eq(column: it.Value, value: object) -> it.BooleanValue`
   - `ne(column: it.Value, value: object) -> it.BooleanValue`
   - `is_null(column: it.Value) -> it.BooleanValue`
   - `not_null(column: it.Value) -> it.BooleanValue`
   - `isin_values(column: it.Value, values: Sequence[object]) -> it.BooleanValue`
   - `isin_column(column: it.Value, values: it.Value) -> it.BooleanValue`
2) Replace “raw boolean ops” composition with one canonical combinator API:
   - `and_predicates(...)`, `or_predicates(...)` should accept only typed predicates (or normalize inputs once).
3) Refactor call sites to use the seam:
   - `src/codeintel/storage/queries/safe.py` (all `.filter(cast("Any", ...))` patterns)
   - `src/codeintel/storage/repositories/data_models.py` (`isin` + filters)
   - `src/codeintel/storage/repositories/subsystems.py` (subquery `.isin`)
   - `src/codeintel/storage/views/ibis_views.py` (existing `cast("Any", ...)` filters)

**Acceptance criteria**

- No `cast("Any", ...)` remains outside `src/codeintel/storage/ibis_types.py`.
- Call sites express intent with a small, consistent vocabulary (`filter_by`, `and_predicates`, `isin_values`, etc.).

**Tests to add**

- Unit tests validating each new helper’s return type and that generated SQL includes expected predicates.

---

### W2 — Runtime‑Checked Scalar Execution Adapters (Hardening + typing)

**Objective**: stop relying on unchecked casts from dynamic query results to Python scalars.

**Current patterns**

- `cast("int", expr.count().execute())`
- `float(cast("Any", result))`

**Design**

Introduce a single module for scalar normalization:

- `src/codeintel/storage/query_results.py` (new)
  - `coerce_int(value: object, *, ctx: str) -> int`
  - `coerce_float(value: object, *, ctx: str) -> float`
  - `coerce_optional_float(value: object, *, ctx: str) -> float | None`
  - `execute_int(expr: it.Value, *, ctx: str) -> int` (for ibis scalar exprs)
  - `execute_optional_int(...)`, etc. as needed

Conversions must be strict and raise domain errors (`QueryError` or `ValueError`) with actionable context.

**Implementation tasks**

1) Implement `query_results.py` with explicit supported scalar types (DuckDB scalar, numpy scalar, python int/float).
2) Update `src/codeintel/storage/queries/safe.py` to use adapters.
3) Update any remaining “count/aggregate execute” sites in storage.

**Acceptance criteria**

- No `cast("int", expr.execute())` remains in storage hot paths; scalar results are coerced via one module.
- Errors become “typed contract violations” rather than silent coercions.

**Tests to add**

- Coercion unit tests for representative scalar types (python, numpy, Decimal if present).
- A regression test for `safe_*` helpers to ensure behavior remains stable.

---

### W3 — Repository Return Surface Tightening (Reduce `Any` leakage)

**Objective**: stop making `dict[str, Any]` the de facto storage API contract.

**Current baseline**

- `RowDict = dict[str, Any]` in `src/codeintel/storage/repositories/base.py`.
- Several repositories return `list[RowDict]` / `RowDict | None`.

**Design options**

- **Option A (recommended)**: introduce JSON-safe values and return `JsonObject`:
  - `JsonValue = str | int | float | bool | None | list[JsonValue] | dict[str, JsonValue]`
  - `JsonObject = dict[str, JsonValue]`
  - Replace `RowDict` with `JsonObject`
- **Option B (stronger)**: define per-method `TypedDict` or dataclass row models for stable shapes, and only
  serialize to JSON at the HTTP boundary.

**Implementation tasks**

1) Implement Option A quickly to eliminate `Any` propagation.
2) For the most important query surfaces, incrementally adopt Option B:
   - `SubsystemSummaryRow`, `FunctionSummaryRow`, etc. as `TypedDict` with precise fields.
3) Ensure conversions explicitly handle nulls and numeric coercions.

**Acceptance criteria**

- Repository public surfaces no longer return `dict[str, Any]` by default.
- Type checkers can validate downstream consumers without `Any` bleed-through.

**Tests to add**

- A “row serialization contract” test ensuring `JsonObject` values are JSON serializable without lossy coercion.

---

### W4 — Typed Query IR (`CompiledSelect`) for SQLGlot Consumers (Extensibility)

**Objective**: standardize “compiled query artifacts” (SQL + AST + dependencies + fingerprint) to reduce repeated
parsing/canonicalization and keep query metadata consistent.

**Current baseline**

- SQLGlot toolkit exists (`src/codeintel/storage/sqlglot_tools.py`), but call sites still tend to pass raw strings.

**Design**

Add a small immutable value object:

- `src/codeintel/storage/sql/compiled.py` (new module path is flexible)
  - `CompiledSelect(sql: str, ast: exp.Expression, tables: frozenset[str], fingerprint: str)`
  - Construction uses `parse_one_duckdb`, `extract_table_refs`, `fingerprint_sql_duckdb`.

**Implementation tasks**

1) Create `CompiledSelect` + constructor function(s).
2) Migrate internal SQL consumers that need “SQL + metadata”:
   - view diffs
   - dependency extraction
   - perimeter validation (where appropriate)
3) Ensure stable canonicalization and fingerprint semantics remain centralized.

**Acceptance criteria**

- “SQL + metadata” is represented in one object, not recomputed ad hoc.
- Adding query caching/telemetry later becomes mechanical.

**Tests to add**

- Fingerprint stability test and dependency extraction correctness using the `CompiledSelect` path.

---

### W5 — Protocol‑First Boundaries for Testability (Hardening)

**Objective**: reduce friction caused by concrete third‑party types in signatures, enabling simpler fakes and
more precise typing.

**Examples**

- DuckDB connection usage can often be expressed as:
  - `execute(query: str, parameters: object | None = None) -> object`
  - `table(name: str) -> DuckDBRelation`

**Implementation tasks**

1) Identify “execution-only” and “table-only” call sites and accept minimal Protocols instead of
   `duckdb.DuckDBPyConnection` directly.
2) Keep the concrete types in `gateway.protocol` for external integration points, but let internal helpers accept
   protocols when they don’t need the full surface.

**Acceptance criteria**

- Tests can use simple recorders/fakes without type checker complaints.
- Storage logic is easier to unit-test without spinning DuckDB in every case.

---

### W6 — Decommission and Cleanup (No compat shims left behind)

**Objective**: fully remove legacy patterns once the new seams exist.

**Implementation tasks**

1) Remove any remaining call-site casts that should now be handled by W1/W2.
2) Ensure no new suppressions have appeared (`# type: ignore`, `# noqa`) during refactors.
3) Normalize documentation:
   - Add a short “How to write typed Ibis filters” section to a storage README/doc if present.

**Acceptance criteria**

- `src/codeintel/storage/**` contains no `# type: ignore` and no scattered `cast("Any", ...)` usage.
- Typed helper modules are the single source-of-truth for predicate typing and scalar coercion.

## 5) Rollout Guidance (Minimize Rework)

Recommended order is W1 → W2 → W3 (Option A) → W4 → W5 → W6.

- W1 first removes the largest volume of repeated patterns and reduces future refactor churn.
- W2 hardens scalar correctness and improves downstream reliability (telemetry, metrics, caches).
- W3 reduces `Any` leakage and makes future serving refactors safer.

## 6) Deliverables Summary

- New / expanded modules:
  - `src/codeintel/storage/ibis_types.py` (expanded, becomes the single typing seam)
  - `src/codeintel/storage/query_results.py` (new)
  - `src/codeintel/storage/sql/compiled.py` (new, path/name flexible)
- Refactors:
  - `src/codeintel/storage/queries/safe.py` (no raw casts; uses W1/W2)
  - `src/codeintel/storage/repositories/*` (no raw casts; tighter return types)
  - `src/codeintel/storage/views/ibis_views.py` (no raw casts)
- Tests:
  - New `tests/storage/test_*` suites covering predicate typing, scalar coercion, and query IR.

