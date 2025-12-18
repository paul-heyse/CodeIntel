# Core Ibis Typing Facade + Build (Hamilton) Boundary Hardening — Implementation Plan

**Status**: Implementation plan (design-aligned)  
**Last updated**: 2025-12-18  
**Primary scope**: `src/codeintel/core/**`, `src/codeintel/build/**`  
**Secondary scope**: `src/codeintel/storage/**` (consumers; canonical storage primitives remain storage-owned)  

## 0) Why This Plan Exists

We are becoming increasingly **Hamilton DAG-centric** for build execution. This creates two pressures:

1) **Typing semantics must be runtime-safe**: Hamilton evaluates annotations at runtime and is strict about type
   matching across nodes.
2) **Ibis typing friction must be centralized**: Ibis operator overloading is hard for static type checkers to
   model, which leads to repeated `cast("Any", ...)` at call sites unless we provide a single “escape hatch”.

The goal is to:

- Keep **Hamilton/boundary-specific concerns** owned by the build layer (`src/codeintel/build/hamilton/**`), where
  the DAG integration lives.
- Provide a **shared Ibis typing facade** in a neutral layer (`src/codeintel/core/**`) so both build and storage
  can use it without inverting dependencies (storage must never import build).

## 1) Goals and Non‑Goals

### Goals (must)

1) **One shared Ibis typing facade**
   - A single module in `src/codeintel/core` that contains the minimal, centralized casts and typed wrappers.
2) **Zero `cast("Any", ...)` in build target modules**
   - Target modules remain clean and declarative; any unavoidable casts happen in the shared facade only.
3) **Build owns Hamilton boundary contracts**
   - Concrete, dispatch-friendly types at saver boundaries and executor boundaries.
4) **Immediate decommissioning of deprecated surfaces**
   - After migration, remove old import surfaces (or keep only a short-lived re-export shim if unavoidable).
5) **Guardrails + tests enforce the new rules**
   - Prevent regressions by failing fast in CI and via local checks.

### Non-goals (for this tranche)

- Upstream changes to Ibis typing stubs.
- Introducing a “lenient Hamilton adapter” to bypass strict type matching.
- Broad refactors in serving unless a build → serving interface requires it.

## 2) Target Architecture (Ownership & Layering)

### 2.1 Ownership boundaries (ideal)

**Build-owned (Hamilton/DAG execution layer)**

- `src/codeintel/build/hamilton/runtime_typing.py`
  - Runtime-available typing re-exports used in any module that Hamilton loads.
- `src/codeintel/build/hamilton/execution_result.py`
  - Canonical executor boundary (`ExecutionResult`) and any conversion helpers.
- `src/codeintel/build/hamilton/save_to.py`
  - Saver decorator variants controlling DAG-exposed metadata node typing and saver dispatch ergonomics.

**Core-owned (shared utilities, safe for both build + storage)**

- `src/codeintel/core/ibis_typing.py` (new)
  - The only module allowed to contain the minimal `cast("Any", ...)` required to type Ibis semantics.

**Storage-owned (data layer primitives and semantics)**

- Gateway/warehouse/repositories/views remain storage-owned.
- Storage may import `codeintel.core.ibis_typing` but must never import build.

### 2.2 “Ibis typing” public API design

The shared facade should provide a small vocabulary that covers 95% of call sites:

- Predicates:
  - `ibis_bool(expr: object) -> ir.BooleanValue`
  - `and_predicates(*predicates: object) -> ir.BooleanValue`
  - `or_predicates(*predicates: object) -> ir.BooleanValue`
  - `isin_values(column: ir.Value, values: Iterable[object]) -> ir.BooleanValue`
  - `is_null(column: ir.Value) -> ir.BooleanValue`
  - `not_null(column: ir.Value) -> ir.BooleanValue`
  - Comparators: `eq/ne/ge/gt/le/lt`
- Table utilities:
  - `filter_by(table: TableT, *predicates: object) -> TableT`
  - `table_has_column(table: ir.Table, column: str) -> bool`
  - `select_columns(table: ir.Table, *columns: str) -> ir.Table`
- Aggregations + arithmetic:
  - `col_sum/col_nunique/...`
  - `add/sub/mul/truediv`, `cast_dtype`, `fillna`

Design constraints:

- Keep the surface **small** and **stable**.
- Prefer **composability** (predicates, `filter_by`) over many bespoke helpers.
- Do not leak `Any` outward: return `ir.BooleanValue` / `ir.Value` / `ir.Table`.

## 3) Workstreams and Sequencing

### W0 — Baseline inventory + freeze new surfaces

**Objective**: stop churn during migration.

Tasks:

1) Inventory current call sites:
   - `rg 'cast\\(\"Any\"|cast\\(Any' src/codeintel/build src/codeintel/storage`
   - `rg 'from codeintel\\.storage\\.ibis_types|from codeintel\\.build\\.ibis_typing' src`
2) Decide whether we need a transitional re-export (timeboxed) for:
   - `codeintel.storage.ibis_types` → `codeintel.core.ibis_typing`
   - `codeintel.build.ibis_typing` → `codeintel.core.ibis_typing`
3) Freeze rule: “no new `cast("Any", ...)` outside the facade”.

Acceptance gate:

- A baseline report is recorded in this doc (counts + hotspots).

---

### W1 — Introduce `src/codeintel/core/ibis_typing.py` (canonical shared facade)

**Objective**: create the shared module and copy in the minimum necessary logic.

Tasks:

1) Create `src/codeintel/core/ibis_typing.py` with:
   - predicate wrappers
   - aggregations and arithmetic helpers used in build DAG nodes
   - generic `filter_by` using Python 3.13 type parameters
2) Ensure:
   - all public functions have NumPy docstrings
   - line length ≤ 100
   - no `TYPE_CHECKING` imports needed for runtime use in Hamilton-loaded modules (safe either way)

Acceptance gates:

```bash
uv run ruff check --fix src/codeintel/core/ibis_typing.py
uv run pyright --warnings --pythonversion=3.13 src/codeintel/core/ibis_typing.py
uv run pyrefly check src/codeintel/core/ibis_typing.py
```

---

### W2 — Migrate build code to the shared facade (build owns DAG ergonomics)

**Objective**: build target modules become free of `cast("Any", ...)` and use the shared facade.

Tasks:

1) Update imports across `src/codeintel/build/**`:
   - Replace `from codeintel.build.ibis_typing import ...` with `from codeintel.core.ibis_typing import ...`
   - Replace any remaining `cast("Any", ...)` call sites with facade helpers.
2) Keep Hamilton node annotations runtime-safe:
   - Any typing symbol used in node annotations must be runtime available.
3) If we keep `src/codeintel/build/ibis_typing.py` temporarily:
   - convert it to a pure re-export shim (no casts, no logic)
   - mark as deprecated and schedule deletion in W4.

Acceptance gates:

```bash
uv run ruff check --fix src/codeintel/build
uv run pyright --warnings --pythonversion=3.13 src/codeintel/build
uv run pyrefly check src/codeintel/build
uv run pytest -q -n0 tests/build/hamilton --no-cov
```

---

### W3 — Migrate storage code to the shared facade (storage remains canonical for storage primitives)

**Objective**: storage uses the same facade but retains ownership of data semantics (warehouse/gateway).

Tasks:

1) Replace imports:
   - `from codeintel.storage.ibis_types import ...` → `from codeintel.core.ibis_typing import ...`
2) Enforce that storage call sites do not reintroduce local `cast("Any", ...)` patterns.
3) Decide what to do with `src/codeintel/storage/ibis_types.py`:
   - Preferred: delete after migration (fast decommissioning).
   - If unavoidable: keep as a short-lived re-export shim and immediately open a follow-up task to delete.

Acceptance gates (storage-scoped):

```bash
uv run ruff check --fix src/codeintel/storage tests/storage
uv run pyright --warnings --pythonversion=3.13 src/codeintel/storage tests/storage
uv run pyrefly check src/codeintel/storage tests/storage
uv run pytest -q tests/storage
```

---

### W4 — Build boundary typing hardening (Hamilton-specific)

**Objective**: ensure saver dispatch and DAG boundary types remain dispatch-friendly and Hamilton-stable.

Tasks:

1) Saver dispatch boundaries:
   - `DuckDBRowsSaver`: ensure saved-node outputs are `tuple[...] | None` or `list[...] | None` at the boundary.
   - `DuckDBIbisTableSaver`: ensure `ir.Table | None`.
   - `FileArtifactSaver`: ensure `bytes | str | Path | ArtifactWritePlan | None`.
2) Materialization metadata:
   - Use concrete `dict[str, object]` at Hamilton node boundaries.
   - Allow a named alias only in non-node/helper code if it does not affect Hamilton type equality.
3) Executor boundary:
   - Ensure executor-style materialization always receives `ExecutionResult` (use conversion nodes as needed).

Acceptance gates:

- Hamilton driver builds without strict type mismatches.
- `uv run pytest -q -n0 tests/build/hamilton --no-cov` passes.

---

### W5 — Enforce the new rules (guardrails + tests)

**Objective**: prevent regressions.

Tasks:

1) Update `tools/guardrails.py`:
   - Ban `cast("Any", ...)` in:
     - `src/codeintel/build/**`
     - `src/codeintel/storage/**`
   - Allow only in:
     - `src/codeintel/core/ibis_typing.py`
   - Ban `ComputeResult = Any` in build.
2) Add small unit tests:
   - Build-native scan test for forbidden patterns under `src/codeintel/build/hamilton/native/**`.
   - Optional: storage scan test under `src/codeintel/storage/**` (keep fast).

Acceptance gates:

```bash
uv run python -m tools.guardrails
uv run pytest -q tests/build/hamilton -k guardrails --no-cov
```

---

### W6 — Full verification pass

**Objective**: everything is green under the repo’s standard gates.

Commands:

```bash
uv run python -m tools.quality_report --output build/quality-results/quality_report.json
uv run pytest -q
```

If output-hash based snapshots exist and change, update snapshots in the canonical repo manner (documented in the
relevant test suite).

## 4) Decommissioning Checklist (must)

After migrations:

- Delete `src/codeintel/build/ibis_typing.py` if it is only a compatibility shim.
- Delete `src/codeintel/storage/ibis_types.py` if it is only a compatibility shim.
- Remove any now-unused imports and helper wrappers.
- Ensure guardrails prevent reintroduction.

## 5) Risk Management / Gotchas

1) **Hamilton type equality**
   - Avoid alias types at Hamilton node boundaries when Hamilton compares runtime types strictly.
2) **Runtime evaluation of annotations**
   - Any type used in a Hamilton-loaded module annotation must exist at runtime (no TYPE_CHECKING-only symbols).
3) **Incremental migration strategy**
   - Prefer a single sweep migration (update imports everywhere) followed immediately by deletion of deprecated
     modules, to prevent drift and “two sources of truth”.

## 6) Definition of Done

- `codeintel.core.ibis_typing` is the only location containing Ibis-related `cast("Any", ...)`.
- Build target modules (`src/codeintel/build/hamilton/native/**`) contain no `cast("Any", ...)`.
- Storage call sites contain no local Ibis typing casts; they use the core facade.
- Build boundary types remain concrete and saver dispatch remains deterministic.
- Guardrails + tests enforce the rules.
- `tools.quality_report` and `pytest -q` are green.

