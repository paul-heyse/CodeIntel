# Config Refinement — Best-in-Class Runtime Consolidation Plan

**Status**: Implementation plan  
**Last updated**: 2025-12-18  
**Primary scope**: `src/codeintel/config/**`  
**Secondary scope** (required call-site migrations): `src/codeintel/cli/**`, `src/codeintel/ingestion/**`,
`src/codeintel/serving/**`, `src/codeintel/storage/**`, `src/codeintel/graphs/**`, `src/codeintel/analytics/**`,
and `tests/**`.

## 0) Goals (Best-in-Class Target Shape)

By the end of this plan:

1) **One canonical runtime config flow** (CLI + serving + build/ingestion all produce the same primitives).
2) **One canonical serving settings surface** (single env schema, single validation path).
3) **One canonical tool identity + tool config surface** (no duplicated tool-name enums, no duplicated
   “resolve path / build env” logic, no config↔ingestion typing coupling).
4) **No config↔storage import cycles** (eliminate `config.datasets.*` importing storage; remove importlib hacks
   whose sole purpose is cycle avoidance).
5) **Smaller, sharper `codeintel.config`**:
   - CLI boundary models stay explicit and contained.
   - “Domain enums” that aren’t configuration move to domain owners (analytics/core).
   - `codeintel.config.__all__` exports only stable public APIs.

## 1) Current State (Why We’re Changing)

### A) Serving config is split-brain

- `src/codeintel/config/serving_models.py` defines a Pydantic `ServingConfig` and `verify_db_identity`.
- `src/codeintel/serving/settings.py` defines a dataclass `ServingSettings` and is used broadly by serving.
- Result: two env schemas, two defaults sets, duplicated parsing/validation, and drift risk.

### B) Tool config is duplicated and cross-layer coupled

- `src/codeintel/config/models.py` defines `ToolsConfig` (Pydantic) with:
  - tool executable fields
  - `resolve_path(...)`
  - `build_env(...)`
  - optional report paths
- `src/codeintel/config/primitives.py` defines `ToolBinaries` (dataclass) with a second `resolve_path(...)`.
- `ToolsConfig` typing currently references ingestion tool identity (`ToolName`) via TYPE_CHECKING, which is the
  wrong dependency direction: config shouldn’t depend on ingestion’s internal enum.

### C) Graph feature flags are parsed in multiple places

- Graph feature flags (`GraphFeatureFlags`) exist in `src/codeintel/config/primitives.py`, but env parsing is
  implemented separately in:
  - `src/codeintel/config/serving_models.py`
  - `src/codeintel/cli/config/service.py`
- Result: inconsistent semantics across entrypoints and duplicated parsing logic.

### D) Config↔storage cycle exists (and is being “papered over”)

- `src/codeintel/config/datasets/dataflow.py` imports storage contracts provider.
- `src/codeintel/storage/metadata/bootstrap.py` uses `importlib.import_module(...)` to avoid an import cycle.
- This prevents clean layering enforcement and complicates refactors.

### E) “Domain types” live in config even when they aren’t configuration

- `src/codeintel/config/parser_types.py` defines `FunctionParserKind`, which is analytics parsing domain logic.

## 2) Target Architecture (Concrete End State)

### 2.1 Canonical primitives

Introduce a single internal primitive bundle (name is illustrative):

- `codeintel.core.runtime.primitives.RuntimePrimitives`
  - `snapshot: SnapshotRef`
  - `paths: BuildPaths`
  - `tools: ToolBinaries` (or a renamed internal tool config)
  - `graph_backend: GraphBackendConfig`
  - `graph_features: GraphFeatureFlags`
  - `profiles: ScanProfiles | None` (optional)
  - (optional) `serving: ServingRuntimeConfig` when needed by CLI

All entrypoints should construct and pass `RuntimePrimitives` (or a sibling “ResolvedRuntime” object that embeds it)
so there is **one** source of truth for runtime defaults/validation.

### 2.2 Serving config unification

Canonical serving config lives under `codeintel.serving` (not `codeintel.config`) and is composed:

- `codeintel.serving.config.ServingIdentity` (repo_root, repo, commit, db_path, read_only, mode/api_base_url)
- `codeintel.serving.settings.ServingSettings` (operational toggles)
- `codeintel.serving.config.ServingRuntimeConfig` = identity + settings

Then:

- Serving runtime uses only `ServingSettings` + (when needed) `ServingIdentity`.
- CLI resolution returns `ServingRuntimeConfig` (or returns `ServingSettings` + identity separately).
- `codeintel.config.serving_models` is deleted after migration.

### 2.3 Tooling unification

Canonical tool identity and execution environment builder live in core:

- `codeintel.core.tools.ToolName` (or `codeintel.core.tools.names.ToolName`)
- `codeintel.core.tools.ToolBinaries` (internal dataclass; stable mapping)
- `codeintel.core.tools.ToolEnv` helper (build env mapping consistently)

Then:

- Ingestion tool runner imports `ToolName` from core (no “duplicate enum”).
- Config CLI boundary `ToolsConfig` becomes a pure boundary model with a single conversion:
  - `ToolsConfig.to_binaries() -> ToolBinaries`
  - (and/or) `ToolsConfig.to_env(tool, base_env=...)` delegates to a shared helper
- Remove duplicated `resolve_path` implementations.

### 2.4 Dataset dataflow builder moves out of config

Move dataflow types/builders to a storage-owned module, fed by explicit contract/provider inputs:

- `codeintel.storage.contracts.dataflow` owns:
  - `DataflowNode`, `DataflowEdge`
  - `build_contract_dataflow_graph(...)`
  - docs aliasing logic (if it is truly part of metadata graph)

Then:

- `src/codeintel/storage/metadata/bootstrap.py` imports it directly (no importlib shim).
- `src/codeintel/config/datasets/dataflow.py` is deleted.

### 2.5 “Domain enum” relocation

- Move `FunctionParserKind` from `codeintel.config.parser_types` to analytics parsing (or core parsing).
  - Recommended: `codeintel.analytics.parsing.types.FunctionParserKind`
  - If you want parsers to be a cross-domain plugin point, place it in `codeintel.core.parsing`.

## 3) Workstreams & Phases (Sequenced to Minimize Rework)

### Phase 0 — Inventory + invariants (no behavior change)

**Deliverables**

- A short doc (inline in PR or as notes) listing:
  - all env vars used by `ServingConfig` and `ServingSettings`
  - all env vars used to build graph feature flags across the repo
  - all current call sites of `ToolsConfig`, `ToolBinaries`, and `ToolName`
  - all current call sites of dataset dataflow graph builders/types

**Acceptance gates**

```bash
uv run python -m tools.quality_report --output build/quality-results/quality_report.json
uv run pytest -q
```

---

### Phase 1 — Core env parsing helpers (foundation)

**Goal**: eliminate ad-hoc env parsing utilities and standardize semantics.

**Steps**

1) Create `src/codeintel/core/env.py` (or `src/codeintel/core/env/__init__.py`) with:
   - `get_bool(name, default=...)`
   - `get_int(name, default=..., *, min_value=None, max_value=None)`
   - `get_float(...)`
   - `get_path(name, default=None, *, must_exist=False)`
   - `is_set(name)` semantics (distinguish unset vs explicitly empty)
2) Migrate:
   - `src/codeintel/config/serving_models.py:_parse_env_flag`
   - `src/codeintel/cli/config/service.py:build_graph_feature_flags_from_env`
   - any similar env parsers in serving/settings, storage, build, etc.

**Compatibility strategy**

- Keep old helper functions temporarily as wrappers that call `codeintel.core.env` (no behavior change).
- Remove wrappers at the end of Phase 4+ once all call sites migrate.

**Acceptance gates**

Run full quality gates and targeted config/serving tests.

---

### Phase 2 — Graph feature flags: single loader

**Goal**: exactly one implementation that maps `CODEINTEL_GRAPH_*` env vars to `GraphFeatureFlags`.

**Steps**

1) Add `@classmethod GraphFeatureFlags.from_env(...)` in `src/codeintel/config/primitives.py`
   (or relocate `GraphFeatureFlags` to `codeintel.core.runtime` if you want all runtime primitives there).
2) Replace env parsing in:
   - `src/codeintel/config/serving_models.py`
   - `src/codeintel/cli/config/service.py`
   with calls to the canonical loader.

**Acceptance gates**

- Full repo checks.
- Explicit smoke: create flags via env in CLI path and serving path and assert equality.

---

### Phase 3 — Tool identity + tool config unification

**Goal**: one enum, one mapping, one “build env” behavior; config no longer depends on ingestion types.

**Steps**

1) Move tool identity enum to core:
   - Introduce `src/codeintel/core/tools/names.py` with `ToolName` (StrEnum).
2) Update ingestion tool runner to import `ToolName` from core and remove the local enum.
3) Decide canonical “internal tools config” representation:
   - Either keep `ToolBinaries` as dataclass (relocate to `codeintel.core.tools.config`) and delete the duplicate.
4) Update `src/codeintel/config/models.py:ToolsConfig`:
   - Remove dependency on ingestion’s ToolName for typing.
   - Add explicit conversion `ToolsConfig.to_binaries()` (or `to_tool_binaries()`).
   - Make `resolve_path(...)` / `build_env(...)` delegate to shared core helpers.
5) Update `src/codeintel/config/builder.py` to use the canonical internal tool config type.

**Compatibility strategy**

- If a public API depends on `ToolsConfig.resolve_path(...)`, keep it, but implement it via core helpers so there’s
  no second mapping.

**Acceptance gates**

- Run the ingestion tool runner tests (if present) and full suite.
- Explicitly validate that tool resolution behavior is unchanged (same resolved command vectors).

---

### Phase 4 — Serving config unification (delete `codeintel.config.serving_models`)

**Goal**: one canonical serving config, owned by serving.

**Steps**

1) Introduce `codeintel.serving.config`:
   - `ServingIdentity` (repo_root/repo/commit/db_path/read_only/mode/api_base_url)
   - `verify_db_identity(...)` moved here (or into `codeintel.serving.runtime` if preferred)
2) Decide canonical “settings” vs “identity” boundary:
   - `ServingSettings` remains operational toggles + env schema for serving runtime.
   - `ServingIdentity` is repo/db identity + validation.
3) Migrate CLI resolution to return serving identity/settings from serving-owned modules.
4) Migrate any other `ServingConfig` call sites (currently mostly CLI resolution).
5) Delete `src/codeintel/config/serving_models.py` after all call sites migrate.

**Acceptance gates**

- Serving contract check: `uv run python -m codeintel.serving.contracts.check_operation_contracts`
- Run serving tests (if any) plus full suite.

---

### Phase 5 — Runtime primitives unification (`RuntimePrimitives`)

**Goal**: one internal “runtime bundle” produced by CLI resolution and by config builders.

**Steps**

1) Add `RuntimePrimitives` (recommended location: `src/codeintel/core/runtime/primitives.py`).
2) Update:
   - `src/codeintel/config/builder.py` to produce `RuntimePrimitives` (or embed it).
   - `src/codeintel/cli/resolution/runtime.py` to produce `RuntimePrimitives` consistently.
3) Ensure “defaults” live in exactly one place (avoid CLI-only defaults vs builder-only defaults).

**Acceptance gates**

- Ensure all entrypoints produce equivalent primitives for the same input parameters.
- Full suite.

---

### Phase 6 — Remove config↔storage cycles: relocate dataset dataflow to storage

**Goal**: config no longer imports storage; remove importlib hacks.

**Steps**

1) Create `src/codeintel/storage/contracts/dataflow.py` (or `.../graph.py`) and move:
   - `DataflowNode`, `DataflowEdge`
   - `build_contract_dataflow_graph`
   - docs aliasing logic (if metadata-owned)
2) Update:
   - `src/codeintel/storage/metadata/bootstrap.py` to import directly (delete importlib shim).
3) Delete `src/codeintel/config/datasets/dataflow.py`.
4) Update any remaining imports (e.g., type-checking imports) accordingly.

**Acceptance gates**

- Full suite.
- Verify metadata bootstrap produces the same nodes/edges and persists them identically.

---

### Phase 7 — Relocate non-config domain enums (`FunctionParserKind`)

**Goal**: keep `codeintel.config` focused on runtime config, not analytics domain enums.

**Steps**

1) Move `FunctionParserKind` from `src/codeintel/config/parser_types.py` into:
   - `src/codeintel/analytics/parsing/types.py` (recommended)
2) Update analytics parser registry imports.
3) Delete `src/codeintel/config/parser_types.py`.

**Acceptance gates**

- Analytics parsing tests (if present) plus full suite.

---

### Phase 8 — Public API tightening + doc hygiene

**Goal**: make the “public surface” obvious and stable; reduce drift.

**Steps**

1) Audit and tighten `src/codeintel/config/__init__.py` exports:
   - Export only stable primitives and CLI boundary models.
   - Avoid exporting modules that are primarily internal migration scaffolding.
2) Fix docstrings that reference non-existent APIs (e.g., `ConfigBuilder.graph_metrics()` reference in
   `src/codeintel/config/models.py`).
3) Add a short “Config layering guide” section in `docs/Config_refinement/` describing:
   - what belongs in config vs core vs domain packages
   - how to introduce new env vars (single loader, single validation)

**Acceptance gates**

- Full `tools.quality_report` + `pytest -q`.

## 4) Decommissioning Policy (No Legacy Left Behind)

This plan expects a two-step migration pattern:

1) **Introduce new canonical API**, migrate call sites.
2) **Delete old API** once zero call sites remain.

Compatibility wrappers are allowed only when they:

- are short-lived (explicitly scheduled for removal in the next phase),
- are thin delegations (no duplicated logic),
- and are fully removed by the end of Phase 8.

## 5) Testing & Acceptance Gates (Per Phase)

After each phase:

```bash
uv run python -m tools.quality_report --output build/quality-results/quality_report.json
uv run pytest -q
```

When iterating quickly for config-only phases:

```bash
uv run ruff check --fix src/codeintel/config src/codeintel/cli src/codeintel/serving src/codeintel/core
uv run pyright --warnings --pythonversion=3.13 src/codeintel/config src/codeintel/cli src/codeintel/serving src/codeintel/core
uv run pyrefly check src/codeintel/config src/codeintel/cli src/codeintel/serving src/codeintel/core
uv run pytest -q tests/config
```

## 6) Risks & Mitigations

- **Env var semantics drift** (unset vs “0” vs empty):
  - Mitigation: Phase 1 centralizes parsing; add explicit tests for env parsing behavior.
- **Serving config migration breaks deployments**:
  - Mitigation: keep `ServingSettings` env var names stable; if names change, support aliases for one release
    cycle and then delete.
- **Tool resolution behavior changes** (PATH vs explicit paths):
  - Mitigation: add small focused tests asserting resolved executable behavior for absolute/relative names.
- **Layering regressions**:
  - Mitigation: after Phase 6, layering can be enforced more strictly because cycles are removed; optionally add a
    lightweight layering check in `tools/` (not in `codeintel.config` package).

