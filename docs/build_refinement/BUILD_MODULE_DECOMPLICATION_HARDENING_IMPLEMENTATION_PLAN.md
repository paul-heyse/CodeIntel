# Build Module — Decomplication & Hardening Implementation Plan (Hamilton-First)

This plan focuses on **reducing conceptual surface area** in `src/codeintel/build` by removing
parallel execution pathways, standardizing contracts, and enforcing invariants early.

It intentionally prioritizes changes that make the system **easier to run, extend, and maintain**
over adding new capabilities.

---

## Goals

1. **One execution path**: Hamilton is the canonical build execution system; remove parallel stacks.
2. **One contract surface**: standardize result/metadata shapes so downstream tooling is generic.
3. **Hard invariants**: push correctness checks to build-time guardrails (fast, deterministic).
4. **Low-friction extension**: adding a new target is “boring” (clear steps, no hidden coupling).

## Non-goals (explicit)

- Do not add net-new runtime services (Hamilton UI/trackers/stores) as part of this plan.
- Do not introduce cross-run caching systems unless explicitly requested later.
- Do not redesign storage/serving APIs; keep scope centered on `src/codeintel/build` and
  build-associated tests.

## Scope boundary

- **In-scope**: `src/codeintel/build/**`, `src/codeintel/hamilton/**`, and tests that validate build.
- **Conditionally in-scope**: other packages only to **remove imports** on deprecated build modules
  (strictly necessary for decommissioning).

## Acceptance gates (run after each slice)

```bash
uv run python -m tools.quality_report --output build/quality-results/quality_report.json
uv run pytest -q
```

For large refactors, prefer running a targeted subset repeatedly, then full suite at the end:

```bash
uv run pytest -q tests/build/hamilton
```

## Current baseline (already present)

- Graph invariants validator exists: `codeintel.build.hamilton.validate.validate_graph()`.
- Guardrail test exists: `tests/build/hamilton/test_tag_guardrails.py`.
- Canonical tag keys/constants exist: `src/codeintel/hamilton/tags.py`.
- Canonical compute result type exists and is partially adopted:
  `src/codeintel/build/hamilton/execution_result.py` (`ExecutionResult`).
- Typed saver-metadata boundary helper exists:
  `src/codeintel/build/hamilton/save_to.py` (`SaveToObjectMetadataDecorator`).

---

# Workstreams

Each workstream is broken into slices designed to be shippable independently. Deprecation and
deletion happen immediately once callsites migrate (no dead-code retention).

## Workstream D1 — Decommission the legacy plugin execution stack

### Why this decomplicates

Multiple “execution stacks” create ambiguity and drift. Removing the legacy stack ensures there is
exactly one mental model: **Hamilton build execution**.

### Target deletions (end-state)

- `src/codeintel/build/context.py`
- `src/codeintel/build/context_base.py`
- `src/codeintel/build/result.py`
- `src/codeintel/build/protocols.py`

### Slice D1.1 — Callsite inventory + dependency map (repo-wide)

- [ ] Search for imports of the legacy modules across `src/` and `tests/`.
- [ ] Categorize callsites:
  - build runtime orchestration
  - tests/fakes
  - CLI entrypoints (if any)
  - external-facing imports (if any)
- [ ] Produce a small “migration map”:
  - old symbol → new symbol (or removal)
  - required behavioral parity notes

### Slice D1.2 — Migrate internal callsites to Hamilton equivalents

- [ ] Replace usages of legacy context/result types with Hamilton equivalents:
  - `BuildEnv` (`src/codeintel/build/hamilton/env.py`)
  - `HamiltonBuildExecutor` (`src/codeintel/build/hamilton/executor.py`)
  - `HamiltonRuntime`/`build_driver` (`src/codeintel/build/hamilton/driver_factory.py`)
- [ ] Ensure callers no longer require legacy “plugin protocol” surfaces.

### Slice D1.3 — Delete the legacy modules

- [ ] Delete the files listed above.
- [ ] Update any docs/tests that referenced them.
- [ ] Confirm no remaining imports.

### Definition of done

- [ ] No production code imports any legacy module.
- [ ] No tests import any legacy module.
- [ ] All build execution happens through Hamilton (executor + driver factory).

---

## Workstream D2 — Make graph validation a mandatory guardrail (not test-only)

### Why this decomplicates

If invariants are enforced early and deterministically, debugging becomes “fix the validator output”
instead of hunting runtime errors.

### Slice D2.1 — Decide the canonical enforcement point

Choose one (prefer a single source of truth):

- Option A (preferred): integrate into `tools.guardrails` so it runs in the quality suite
  consistently without requiring `pytest`.
- Option B: run as the first step of build execution (executor preflight), so users get immediate
  feedback when running builds locally.

### Slice D2.2 — Implement the integration

- [ ] Add a guardrails entry that:
  - calls `validate_graph()`,
  - prints `validation_result_to_json(...)` on failure,
  - exits non-zero on errors.
- [ ] Ensure it is fast and deterministic (no DB I/O, no network, no filesystem writes except logs).

### Slice D2.3 — Keep a minimal test, but avoid double enforcement

- [ ] Keep the test as a backstop, but avoid doing two divergent checks.
- [ ] If integrated into guardrails, update the test to simply assert the guardrail entrypoint
  remains functional (or remove the test if redundant and guardrails is authoritative).

### Definition of done

- [ ] Graph validation runs in the standard quality suite (or build preflight).
- [ ] Failure output is actionable and stable.

---

## Workstream D3 — Canonical tagging helpers (prevent “tag drift” at the source)

### Why this decomplicates

When authors write tags by hand, inconsistency accumulates. Helpers make the “right thing” the
default, and make refactors safer.

### Slice D3.1 — Introduce tagging helper module

- [ ] Create a small dependency-light module, e.g. `codeintel.build.hamilton.tagging`, that exposes
  helpers returning Hamilton’s `@tag` decorator with canonical keys and values:
  - `tag_compute(domain, target, *, extra_tags=...)`
  - `tag_dataset(domain, target, table_key, *, output_kind=None, extra_tags=...)`
  - `tag_materialize(domain, target, *, extra_tags=...)`
  - `tag_artifact(domain, target, artifact, *, extra_tags=...)`
- [ ] Helpers must:
  - use constants from `src/codeintel/hamilton/tags.py`,
  - avoid repeating string literals like `"node_type"`,
  - support extension via `extra_tags`.

### Slice D3.2 — Migrate representative native modules

Migrate a small representative set first (one per domain), then expand:

- [ ] ingestion native targets
- [ ] analytics native targets
- [ ] export native targets

### Slice D3.3 — Remove ad hoc tag dict construction patterns

- [ ] Replace repeated `@tag(domain="...", target="...", node_type="...")` usage with helpers.
- [ ] Keep tags explicit at call sites (helpers should not hide important semantic identity).

### Definition of done

- [ ] New targets use helpers by default.
- [ ] Existing targets in core domains migrated.
- [ ] Validator still enforces invariants, but violations become rare.

---

## Workstream D4 — Standardize compute result shapes (ExecutionResult everywhere)

### Why this decomplicates

Generic tooling (record writers, templates, tests) can rely on a single result contract, and
individual modules stop defining bespoke dataclasses.

### Slice D4.1 — Define the canonical contract surface

- [ ] Confirm the canonical type is `ExecutionResult` and finalize semantics:
  - what constitutes “success”
  - how errors are represented (`error: str | None`)
  - standard shape for `table_counts`
- [ ] Add any minimal convenience helpers needed for adoption (no overdesign).

### Slice D4.2 — Migrate native targets

- [ ] Replace bespoke “compute result” dataclasses in native targets with `ExecutionResult`.
- [ ] Update tests accordingly.

### Slice D4.3 — Migrate templates and record builders

- [ ] Ensure template materialization functions accept `ExecutionResult` without `Any`.
- [ ] Remove conversion glue where it exists.

### Slice D4.4 — Delete deprecated dataclasses

- [ ] Delete old result dataclasses once callsites migrate.
- [ ] Remove exports and update import paths.

### Definition of done

- [ ] No bespoke `{success, error, table_counts}` dataclasses remain in build.
- [ ] Tooling is generic across targets because shapes are standardized.

---

## Workstream D5 — Canonical materialization metadata schema (eliminate “dict soup”)

### Why this decomplicates

Materialization metadata is currently easy to drift because it is passed around as `dict[str, ...]`
with implicit keys. Typed schemas turn that into an explicit contract.

### Slice D5.1 — Inventory all materialization metadata producers/consumers

- [ ] Identify every producer of saver/materializer metadata dicts:
  - Hamilton savers (including project-local `SaveToObjectMetadataDecorator`)
  - IO adapters
  - materializers/record writers
- [ ] Identify every consumer:
  - record builders
  - tracking writers
  - tests

### Slice D5.2 — Define typed metadata dataclasses

- [ ] Define a minimal set of typed metadata types (examples; adjust to reality):
  - `TableMaterializationMetadata`
  - `FileArtifactMaterializationMetadata`
  - `MultiMaterializationMetadata`
- [ ] Provide strict parsing (`from_mapping(...)`) with explicit errors on missing keys/wrong types.

### Slice D5.3 — Enforce schema at boundaries

- [ ] At the Hamilton boundary:
  - allow Hamilton to pass `dict[str, object]`,
  - immediately parse into typed metadata for internal use.
- [ ] Ensure record writers accept typed metadata (and only convert to dict for serialization if
  necessary).

### Slice D5.4 — Remove legacy metadata parsing/shaping

- [ ] Delete old parsing utilities and ad hoc metadata keys.
- [ ] Update tests to assert typed metadata behavior (not raw dict keys).

### Definition of done

- [ ] Internal code uses typed metadata objects.
- [ ] Dicts are only used at the Hamilton boundary and serialization boundaries.

---

## Workstream D6 — Scope Hamilton caching to reduce correctness/operational complexity

### Why this decomplicates

Caching is a high-footgun surface if not treated as a deliberate product feature. If you do not
need cross-run caching yet, defaulting it “off” reduces confusion and removes the need to reason
about hashability and invalidation.

### Slice D6.1 — Audit current usage

- [ ] Enumerate all uses of:
  - Hamilton `@cache(...)` (from `hamilton.function_modifiers`)
  - Python `functools.cache`/`lru_cache` (ensure they are not conflated)
- [ ] Identify which uses are redundant because the executor already computes the full closure once.

### Slice D6.2 — Decide the default

Choose one:

- Option A (preferred): **disable Hamilton caching adapter by default** in the build driver factory,
  and keep an explicit opt-in flag for targeted workflows.
- Option B: keep caching adapter but remove inessential `@cache(format="memory")` usage.

### Slice D6.3 — Implement and simplify

- [ ] Update `build_driver`/executor defaults to match the decision.
- [ ] Remove redundant `@cache(format="memory")` where it provides no real value.
- [ ] Ensure schema inference and other workflows that pass unhashable objects remain unaffected.

### Definition of done

- [ ] Developers are not surprised by caching behavior.
- [ ] Cache settings are explicit and minimal.

---

## Workstream D7 — Centralize run options into one typed object

### Why this decomplicates

Today, behavior is controlled by a mix of `BuildEnv` fields and executor flags. Consolidating
execution options makes behavior discoverable and reduces implicit coupling.

### Slice D7.1 — Define `BuildExecutionOptions`

- [ ] Create a dataclass that contains execution behavior flags only (not resources):
  - profile
  - strict_contracts / validation toggles
  - parallel backend + max_workers
  - enable telemetry/progress/timing
  - enable Hamilton cache + cache_dir
- [ ] Ensure it can be constructed from CLI/env/config cleanly.

### Slice D7.2 — Refactor executor/driver wiring

- [ ] `HamiltonBuildExecutor` consumes `BuildExecutionOptions` and maps it to:
  - `HookOptions`
  - parallel adapter selection
  - driver factory config/caching toggles
- [ ] Reduce duplication between `executor.py` and `driver_factory.py`.

### Definition of done

- [ ] One place to understand and modify run behavior.
- [ ] `BuildEnv` remains a pure “resources + identity” bundle.

---

## Workstream D8 — Make extension points explicit (adding targets is boring)

### Why this decomplicates

A system is “easy to extend” when contributors can follow a deterministic checklist with guardrails
catching mistakes early.

### Slice D8.1 — Write “Add a target” playbook

- [ ] Add a short internal playbook section (in this doc or a dedicated doc) covering:
  - where to register target spec
  - how to implement native override nodes
  - how to tag using helpers (D3)
  - how to ensure result/metadata contracts (D4/D5)
  - how to validate locally (D2 gates)

### Slice D8.2 — Add a minimal “new target scaffold” (optional)

Only if it reduces work (avoid overengineering):

- [ ] A template module snippet / example native target module that demonstrates the canonical
  pattern.

### Slice D8.3 — Add a guardrail for new target compliance

- [ ] Add a validator rule that flags new targets missing required tags or using deprecated
  patterns (e.g., bespoke result dataclasses), with actionable messages.

### Definition of done

- [ ] Adding a new target requires minimal context.
- [ ] Mistakes are caught before runtime.

---

# Recommended execution order

This ordering maximizes early decomplication and reduces merge conflicts:

1. **D2** (mandatory validation guardrail) — makes refactors safer immediately.
2. **D3** (tag helpers) — reduces tag drift while touching many modules.
3. **D4** (ExecutionResult everywhere) — simplifies result handling and tests.
4. **D5** (typed materialization metadata) — stabilizes IO/record contracts.
5. **D7** (centralized options) — reduces “where do I configure this?” confusion.
6. **D6** (scope caching) — remove footguns once behavior is explicit.
7. **D1** (delete legacy stack) — do after migration safety rails are strong.
8. **D8** (extension playbook + compliance) — codify the new steady state.

# Deliverables checklist

- [ ] Legacy plugin execution stack fully removed.
- [ ] Graph validation runs as a mandatory guardrail (not only tests).
- [ ] Canonical tagging helpers exist and are adopted.
- [ ] `ExecutionResult` is the standard compute-result type across build.
- [ ] Materialization metadata is typed internally; dicts only at boundaries.
- [ ] Hamilton caching is simplified and explicit (no surprises).
- [ ] Run behavior is configured via one typed options object.
- [ ] “Add a target” is documented and enforced by guardrails.

