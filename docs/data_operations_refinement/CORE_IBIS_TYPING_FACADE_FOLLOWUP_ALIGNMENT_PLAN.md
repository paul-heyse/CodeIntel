# Core Ibis Typing Facade — Followup Alignment Plan (Storage + Build)

**Status**: Implementation plan (followups)  
**Last updated**: 2025-12-18  
**Primary scope**: `src/codeintel/storage/**`, `src/codeintel/build/**`, `src/codeintel/core/ibis_typing.py`  
**Secondary scope**: `src/codeintel/analytics/**`, `src/codeintel/graphs/**`, `src/codeintel/ingestion/**`, `tools/guardrails.py`  

## 0) Why This Plan Exists

We’ve implemented the **shared** Ibis typing facade (`codeintel.core.ibis_typing`) and removed the legacy
module-local escape hatches (`codeintel.storage.ibis_types`, `codeintel.build.ibis_typing`). This followup work
finishes the migration by:

1) Providing a single typed entry point for `gateway.ibis.table(...)` to prevent repeated “table typing” patterns.
2) Enforcing the new architecture via guardrails (fail fast on regressions).
3) Eliminating remaining `cast("Any", ...)` hotspots outside build/storage (primarily Ibis-related).
4) Updating docs/plans to reflect the canonical new modules and patterns.

The primary outcome is **reduced cognitive load** and **less typing drift** as the codebase becomes more DAG and
query centric.

## 1) Goals and Non‑Goals

### Goals (must)

1) **Typed table acquisition**
   - Introduce a storage-owned, typed wrapper for `gateway.ibis.table(...)` used across storage/build/consumers.
2) **Hard enforcement of the new architecture**
   - Guardrails ban reintroducing legacy import surfaces and constrain `cast("Any", ...)` to one place.
3) **Remove remaining “Any cast” pressure**
   - Replace Ibis-related `cast("Any", ...)` in analytics/graphs/ingestion with facade helpers or typed adapters.
4) **Documentation alignment**
   - Plans and docs stop referencing deleted modules and instead point to the canonical, shared facade.

### Non-goals (for this tranche)

- Rewriting storage/build layering boundaries beyond what is needed for typed table acquisition.
- Upstream contributions to Ibis typing stubs.
- Introducing new suppressions (no `# type: ignore`, no guardrail disable flags).

## 2) Canonical Architecture (Post-migration)

### 2.1 Canonical ownership

- **Core**: `src/codeintel/core/ibis_typing.py`
  - The only “escape hatch” for Ibis operator typing friction.
  - The only module allowed to contain the minimal `cast("Any", ...)`.
- **Storage**: `src/codeintel/storage/**`
  - Owns database contracts and gateway semantics.
  - Provides a typed adapter for `gateway.ibis.table(...)` (to remove repeated table typing patterns).
- **Build**: `src/codeintel/build/hamilton/**`
  - Owns Hamilton-specific boundary contracts (execution and saver metadata types).

### 2.2 Canonical usage patterns

**Predicate composition / Ibis operations**

- Use `from codeintel.core.ibis_typing import ...` helpers (e.g., `filter_by`, `and_predicates`, `isin_values`).

**Gateway table acquisition**

- Use `codeintel.storage.gateway.ibis_facade` (introduced in this plan).

## 3) Acceptance Gates (must stay green)

Run after each workstream:

```bash
uv run python -m tools.quality_report --output build/quality-results/quality_report.json
uv run pytest -q
```

Additionally, while iterating on one domain, it’s acceptable to run targeted gates:

```bash
uv run ruff check --fix src/codeintel/storage tests/storage
uv run pyright --warnings --pythonversion=3.13 src/codeintel/storage tests/storage
uv run pyrefly check src/codeintel/storage tests/storage
uv run pytest -q tests/storage
```

## 4) Workstreams and Sequencing

### W0 — Baseline inventory + hotspots list

**Objective**: make the remaining work explicit and measurable.

Tasks:

1) Inventory table acquisition hotspots:
   - `rg -n "gateway\\.ibis\\.table\\(" src/codeintel tests`
2) Inventory all remaining `cast("Any", ...)` occurrences outside `codeintel.core.ibis_typing`:
   - `rg -n 'cast\\(\\s*(?:\"Any\"|\\x27Any\\x27|Any)\\s*,' src/codeintel`
3) Classify each occurrence:
   - Ibis predicate/operator friction (should move to `codeintel.core.ibis_typing`)
   - “typed table” issues (should move to storage `ibis_facade`)
   - non-Ibis dynamic typing (should become a narrower cast, a Protocol, or a runtime check helper)
4) Inventory docs/code comments still referencing removed modules:
   - `rg -n "codeintel\\.storage\\.ibis_types|codeintel\\.build\\.ibis_typing" docs src tests`

Deliverable:

- A short list (in this doc or a scratch note) of:
  - top 10 remaining `cast("Any", ...)` sites by frequency/impact
  - top 10 `gateway.ibis.table(...)` call sites
  - all docs referencing removed modules

### W1 — Introduce a storage-owned typed Ibis gateway facade

**Objective**: prevent repeated “table typing” patterns and standardize access.

New module:

- `src/codeintel/storage/gateway/ibis_facade.py`

API sketch:

- `def table(gateway: StorageGateway, table_key: str) -> ir.Table`
  - Returns `gateway.ibis.table(table_key)` with the correct return annotation.
  - Keeps runtime behavior identical; only typing changes.
- Optionally (only if needed by call sites):
  - `def table_as[TableT: ir.Table](gateway: StorageGateway, table_key: str) -> TableT`
    - For call sites that use backend-specific `ir.Table` subclasses and want to preserve that type.

Design constraints:

- Must remain **storage-owned** (storage owns gateway semantics).
- Must not reintroduce `cast("Any", ...)` at call sites.
- Must not depend on build.

Acceptance criteria:

- At least 1–3 representative call sites are migrated to prove the ergonomics.
- No behavioral changes.

### W2 — Migrate storage/build/consumers to the typed facade (table acquisition)

**Objective**: eliminate ad-hoc table typing in day-to-day code.

Tasks:

1) Update storage call sites (highest priority):
   - Replace:
     - `tbl: ir.Table = gateway.ibis.table("...")`
     - `tbl = cast("Any", gateway.ibis.table("..."))`
   - With:
     - `tbl = ibis_facade.table(gateway, "...")`
2) Update build call sites where it improves clarity:
   - Build already depends on storage; this is allowed and reduces duplication.
3) Update analytics/graphs/ingestion call sites:
   - Prefer `ibis_facade.table(...)` when a `StorageGateway` is present.
4) Ensure imports are absolute and correctly grouped.

Acceptance criteria:

- Most (or all) `gateway.ibis.table(...)` call sites move behind the facade.
- Any remaining direct uses are justified (and ideally documented with a short comment).

### W3 — Guardrails: ban removed import surfaces and tighten `cast("Any", ...)`

**Objective**: prevent regressions and keep the architecture stable.

Tasks:

1) Extend `tools/guardrails.py`:
   - Add a rule banning import strings:
     - `codeintel.storage.ibis_types`
     - `codeintel.build.ibis_typing`
   - Message should point to:
     - `codeintel.core.ibis_typing`
2) Tighten the existing `cast("Any", ...)` rule:
   - Target state: allow `cast("Any", ...)` only in `src/codeintel/core/ibis_typing.py`.
   - Rollout strategy:
     - Phase A: keep scope `src/codeintel/**` but allowlist a small set of known transitional files (timeboxed).
     - Phase B: remove transitional allowlist once W4 completes.
3) Optional but recommended: add a small test (fast scan) under `tests/`:
   - Fails if forbidden import strings or `cast("Any", ...)` occur outside the allowlist.
   - Keep it simple and deterministic (no patching fixtures; comply with `tests/test_testing_contract.py`).

Acceptance criteria:

- Guardrails fail fast on reintroduction of deleted modules.
- Guardrails clearly point contributors to the canonical modules.

### W4 — Sweep remaining Ibis-related `cast("Any", ...)` outside build/storage

**Objective**: remove the remaining pressure that will block tightening guardrails globally.

Tasks:

1) For each remaining `cast("Any", ...)` in non-build/storage modules:
   - If it’s an Ibis predicate/operator issue, replace with `codeintel.core.ibis_typing` helpers.
   - If the helper does not exist, add it to `src/codeintel/core/ibis_typing.py` (small, composable).
2) Common patterns to normalize:
   - `tbl.filter(cast("Any", ...))` → `filter_by(tbl, ...)` or `ibis_bool(...)`
   - `col.isin(cast("Any", [...]))` → `isin_values(col, [...])`
   - boolean composition: `cast("Any", (a == b) & (c == d))` → `and_predicates(a == b, c == d)`
3) Ensure no new “mini facades” are introduced elsewhere.

Acceptance criteria:

- After this workstream, `rg 'cast\\(\\s*(?:\"Any\"|\\x27Any\\x27|Any)\\s*,' src/codeintel` returns matches only
  in `src/codeintel/core/ibis_typing.py`.

### W5 — Documentation + plan alignment

**Objective**: prevent confusion and stop docs from pointing to deleted modules.

Tasks:

1) Update the following docs/plans where they reference deleted modules:
   - `docs/data_operations_refinement/STORAGE_TYPING_HARDENING_AND_TYPE_IGNORE_ELIMINATION_PLAN.md`
   - Any other plan mentioning `codeintel.storage.ibis_types` or `codeintel.build.ibis_typing`
2) Update examples to:
   - `from codeintel.core.ibis_typing import ...`
   - and (for table acquisition) `from codeintel.storage.gateway import ibis_facade`

Acceptance criteria:

- No docs reference deleted modules.
- Docs clearly state: “`codeintel.core.ibis_typing` is the only escape hatch.”

### W6 — Final verification pass

Run:

```bash
uv run python -m tools.quality_report --output build/quality-results/quality_report.json
uv run pytest -q
```

If tightening guardrails is phased, ensure Phase B is complete by the end of W6.

## 5) Decommissioning Checklist (must)

- No `codeintel.storage.ibis_types` references remain (already deleted; enforce via guardrails).
- No `codeintel.build.ibis_typing` references remain (already deleted; enforce via guardrails).
- `cast("Any", ...)` exists only in `src/codeintel/core/ibis_typing.py`.
- Table acquisition uses the storage `ibis_facade` unless a justified exception exists.

