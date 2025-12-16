# Hamilton Phase 3 + Phase 4 Completion Plan (Remaining Work)

Last updated: 2025-12-13

This document is a **comprehensive, implementation-oriented plan** to finish the **remaining**
Hamilton **Phase 3** and **Phase 4** scope for this repo, based on the current codebase state.
It is intended to be used as a “tracking board” for a sequence of small, reviewable PRs.

References:
- `docs/Hamilton_integration/Hamilton_apache_phase3.md`
- `docs/Hamilton_integration/Hamilton_apache_phase4.md`
- `docs/Hamilton_integration/Wave1_Final_Summary.md`
- `docs/Hamilton_integration/Wave2_Final_Complete_Summary.md`

---

## 0) Executive Summary

### What’s already in place (high-level)

**Phase 3 foundations exist:**
- Dynamic node generation, split module generation, native target registry, and an `auto` driver
  composition path (`src/codeintel/build/hamilton/nodes/node_factory.py`,
  `src/codeintel/build/hamilton/native/registry.py`, `src/codeintel/build/hamilton/driver_factory.py`).
- Native targets exist for analytics/exports/graphs, and tool-style ingestion prototypes exist
  (`src/codeintel/build/hamilton/native/...`).

**Phase 4 “asset catalog v1” exists (in meaningful form):**
- Schemas for `build.asset_versions`, `build.run_asset_versions`, `build.asset_lineage`,
  `build.asset_aliases`, `build.asset_diffs` (`src/codeintel/config/datasets/schemas.py`).
- Persistence APIs and a basic emitter from build runs (`src/codeintel/build/assets/emitter.py`,
  `src/codeintel/storage/tracking/asset_tracking.py`).
- CLI commands for `build.assets`, `build.lineage`, `build.promote`, `build.resolve`, `build.diff`
  (`src/codeintel/cli/commands/build.py`, `src/codeintel/cli/handlers/build.py`).
- A Phase 4 smoke test already exists (`tests/build/hamilton/test_pr28_phase4_asset_catalog.py`).

### What’s still missing (the real remaining scope)

**Phase 3 remaining gaps (must-do for correctness):**
1. **Contracts are not fully authoritative** yet:
   - Several targets still have **empty `OutputContract.tables`/`artifacts`**, relying on the
     legacy default `OutputTarget.table_keys` behavior.
   - `call_graph_views` is currently **materializing outputs that are not declared in its contract**
     (contract is `tables=()` but code writes `graph.v_*` tables).
2. **CLI cannot run the Hamilton `auto` mode** (native+generated composition), even though it exists.
3. **`--strict-contracts` is plumbed but not enforced** at runtime (no active enforcement context).
4. **`--wrapper-allowlist` is warning-only**, not a hard gate.
5. **Graph export lacks node tags / node-kind metadata** (Phase 3 PR-26 intent).
6. **Tool-style native ingestion pipelines are incomplete**:
   - `typing` “succeeds” without materializing its contract tables.
   - Tool nodes execute even when the target is up-to-date (skip gating not pushed upstream).

**Phase 4 remaining gaps (feature work beyond the current v1):**
1. **Stable fingerprint/version policy for cross-commit reuse** is not implemented as designed
   (current hashes include `repo+commit` via `input_hash`).
2. **Impact analysis**, **cross-commit reuse**, **partition-level incremental**, **backend abstraction
   and parallelism**, **remote cache**, **contracts-as-code scanning**, **asset/version graph export**,
   **CI reporting**, **backfill orchestration**, **run environment capture**, and **AssetRef unification**
   remain to be implemented (Phase 4 PR-34 → PR-45 themes).

---

## 1) Phase 3 Remaining Work (Completion Roadmap)

### Guiding acceptance criteria (Phase 3 “done”)

Phase 3 is “complete” when:
1. **Every target has a correct, explicit `OutputContract`** (tables and/or artifacts) and **no**
   target relies on “implicit default table keys” for correctness.
2. CLI can run mixed native+wrapper builds via **`--hamilton-mode auto`**.
3. **Strict contract enforcement is real**:
   - When `--strict-contracts` is enabled, *any* write outside the current target’s contract
     fails the target.
4. **Wrapper allowlist gating is real** (warn → hard error in plan/run, depending on policy).
5. Graph export output includes **node tags** and **node_kind** for Hamilton nodes.
6. Native tool targets (`typing`, `scip`) have correct skip gating and materialize declared outputs.

---

### P3-PR-01 — Contract parity completion + contract linter hardening (high priority)

**Why:** Phase 3’s “contract-first DAG” only works if contracts are authoritative and complete.

#### Current gaps to close

1) Targets with empty contracts (examples; keep this list updated via audit script):
- `config_ingest`
- `cfg_dfg_metrics`
- `graph_validation`
- `function_effects`
- `function_contracts`
- `history_timeseries`
- `data_models`
- `data_model_usage`
- `config_data_flow`
- `semantic_roles`
- `subsystem_graph_metrics`
- `subsystem_agreement`
- `test_profile`
- `test_graph_metrics`
- `symbol_graph_metrics`
- `behavioral_coverage`
- `entrypoints`
- `external_deps`
- `profiles`
- `function_ast_features`

2) Contract mismatch:
- `call_graph_views` currently has `contract.tables=()` but writes `graph.v_function_call_counts`
  and `graph.v_call_depth_stats`.

#### Implementation tasks

1) Add explicit contracts for every target in `src/codeintel/build/registry.py`:
   - For each target, define `OutputContract(tables=(...))` by referencing
     `_DATASET_TABLE_SCHEMAS["schema.table"]` for *every* table the target produces.
   - For artifact-producing targets, ensure `OutputContract(artifacts=(...))` is correct and that
     templates are renderable with `BuildPaths`.

2) Fix `validate_contracts(...)` to validate the **contract** directly:
   - Validate `target.contract.table_keys` (not `target.table_keys` which can be auto-derived).
   - Add an explicit check: **targets with a plugin must declare outputs in the contract**
     (`contract.tables` and/or `contract.artifacts`), unless the target is intentionally “no-op”.
   - Validate that *every* `contract.table_key` exists in `get_table_schemas()`.

3) Fix PR-16 test semantics:
   - Update `tests/build/hamilton/test_pr16_contract_parity.py` so it checks contract tables
     (`target.contract.table_keys`) rather than derived `target.table_keys`.
   - Add targeted assertions for multi-table targets and known “special cases”.

#### Tests to add/update

- Update: `tests/build/hamilton/test_pr16_contract_parity.py`
- Add: `tests/build/hamilton/test_pr16_contract_linter_strict.py`
  - “No plugin target has empty contract outputs”
  - “All contract table keys exist in schema registry”
  - “Artifact templates render with `BuildPaths`”

#### CLI snapshot updates (optional)

- Add a new help snapshot that ensures “contracts are mentioned” only if CLI help text changes.

#### Acceptance gates

```bash
uv run python -m tools.quality_report --output build/quality-results/quality_report.json
uv run pytest -q tests/build/hamilton/test_pr16_contract_parity.py
```

---

### P3-PR-02 — Add missing table schemas + fix `call_graph_views` contract (high priority)

**Why:** `call_graph_views` currently writes tables that are not registered in dataset schemas and
not declared in contracts, which breaks “contracts are truth”.

#### Decision to make (pick one and stick to it)

1) **Treat “views” as tables** (materialized tables with `graph.v_*` keys).
   - Pros: simplest; aligns with current `materialize_tables(...)` behavior.
   - Cons: naming says “view” but physically materialized.

2) Treat them as logical views (`asset_kind=view`) and materialize as actual DB views.
   - Pros: semantics match naming.
   - Cons: requires storage-level support for views and richer schema handling.

**Recommendation:** Option 1 now (materialized tables), rename later if needed.

#### Implementation tasks

1) Add schemas for:
   - `graph.v_function_call_counts`
   - `graph.v_call_depth_stats`

   in `src/codeintel/config/datasets/schemas.py` (or a dedicated module if the repo has a pattern
   for “derived table schemas”).

2) Update `CALL_GRAPH_VIEWS_TARGET` in `src/codeintel/build/registry.py`:
   - Declare `OutputContract(tables=(...))` for those table keys.

3) Align native implementation with contract:
   - Ensure `src/codeintel/build/hamilton/native/graphs/call_graph_views.py` uses the *same* keys
     as the contract (no drift).

#### Tests

- Add: `tests/build/hamilton/test_pr22_call_graph_views_contract.py`
  - Verify contract includes the two tables.
  - Verify schema registry includes the two keys.

#### Acceptance gates

```bash
uv run python -m tools.quality_report --output build/quality-results/quality_report.json
uv run pytest -q tests/build/hamilton/test_pr22_call_graph_views_contract.py
```

---

### P3-PR-03 — Expose Hamilton `auto` mode in CLI (high priority)

**Why:** The repo already has an `auto` driver (native + generated module composition) but CLI
restricts `--hamilton-mode` to `phase0|generated`. This blocks real mixed execution.

#### Implementation tasks

1) Update CLI argument validation to allow `auto`:
   - `src/codeintel/cli/commands/build.py` (help text + choice list)
   - `src/codeintel/cli/handlers/build.py` validation currently allows only
     `("phase0", "generated")`.

2) Update node-mode mapping to support `auto`:
   - Ensure the build execution args can map `hamilton_mode="auto"` →
     Hamilton runtime mode `"auto"` (see `src/codeintel/build/hamilton/driver_factory.py`).

3) Add a CLI snapshot for help output (so it’s stable):
   - Update `tests/build/hamilton/snapshots/manifest.yaml` with a new case that snapshots
     `codeintel build run --help` or `codeintel build run ... --help` and ensures `auto` appears
     in the help text.

#### Tests

- Add: `tests/build/hamilton/test_pr18_cli_allows_auto_mode.py`
  - Validate `build.run` parameter parsing accepts `--hamilton-mode auto`.

#### Acceptance gates

```bash
uv run python -m tools.quality_report --output build/quality-results/quality_report.json
uv run pytest -q tests/build/hamilton/test_pr18_cli_allows_auto_mode.py
pytest -m cli_snapshot --cli-snapshot-tags pr18 --update-cli-snapshots
```

---

### P3-PR-04 — Make `--strict-contracts` actually enforceable (high priority)

**Why:** The flag exists (`BuildEnv.strict_contracts`) and the enforcement helper exists
(`ContractEnforcer`), but it is never activated for a target execution.

#### Target behavior

When `strict_contracts=True`:
- Any `gateway.ibis.write(table_key, ...)` outside the active target’s contract fails the target.
- Any file artifact write outside the active target’s contract fails the target.

#### Implementation approach (recommended)

1) Ensure the enforcement context is set per target execution:
   - Wrap each target execution in `with ContractEnforcer.for_target(target, strict=env.strict_contracts): ...`

2) Enforce at the **actual write boundaries**:
   - Storage layer is the reliable choke-point: `StorageGateway.ibis.write(...)` knows the table key.
   - Native materializers already call `ContractEnforcer.validate_table_write(...)`, but should rely
     on the same enforcement context as the wrapper/plugin path.

3) Ensure failure semantics are consistent:
   - Contract violation → `ContractViolationError` → `TargetRunRecord(status="failed", error=...)`
   - Downstream targets should be skipped with `upstream_failed` gating.

#### Files likely involved

- `src/codeintel/build/hamilton/contracts/enforcement.py` (may remain mostly unchanged)
- `src/codeintel/build/hamilton/nodes/targets_phase0.py` (wrapper/plugin execution path)
- `src/codeintel/storage/gateway/...` (where `ibis.write` is implemented)
- `src/codeintel/build/hamilton/native/materializer.py` and
  `src/codeintel/build/hamilton/native/artifact_materializer.py` (ensure consistent usage)

#### Tests to add

- `tests/build/hamilton/test_pr27_strict_contracts_violation_fails.py`
  - Define a tiny fake plugin that writes to a table key not in its contract and verify failure.
  - Define a tiny native node that tries to materialize an undeclared table and verify failure.

#### Acceptance gates

```bash
uv run python -m tools.quality_report --output build/quality-results/quality_report.json
uv run pytest -q tests/build/hamilton/test_pr27_strict_contracts_violation_fails.py
```

---

### P3-PR-05 — Enforce wrapper allowlist as a real gate (medium/high priority)

**Why:** Phase 3 intended a “wrapper shrink” policy: wrappers should be temporary shims.
Currently the allowlist only emits a `DeprecationWarning`.

#### Policy decision

Pick one:
1) **Fail planning** when a wrapper target is not allowlisted (recommended).
2) Allow planning but **fail execution** before running any targets.

**Recommendation:** Fail planning. It provides immediate feedback and prevents partial runs.

#### Implementation tasks

- Convert warning in `src/codeintel/build/hamilton/planner.py` to a hard error (e.g., `ValueError`
  or a project-specific exception type).
- Ensure CLI surfaces the failure cleanly in `build.plan` and `build.run --dry-run`.

#### Tests

- `tests/build/hamilton/test_pr27_wrapper_allowlist_enforced.py`
  - Construct a plan with `wrapper_allowlist` excluding a known wrapper target and assert it fails.

---

### P3-PR-06 — Finish native tool targets: correctness + skip gating (high priority)

**Why:** Tool-style native ingestion targets exist, but they currently violate Phase 3 invariants:
- `typing` reports success without writing its contract tables.
- Tool nodes run even when the target should be skipped.

#### 6A) `typing` must materialize its contract tables

Contract tables:
- `analytics.typedness`
- `analytics.static_diagnostics`

Implementation strategy:
1) Parse tool outputs (`pyright`, `pyrefly`, `ruff`) into a normalized in-memory representation.
2) Convert to Ibis expressions or DataFrames.
3) Materialize using `materialize_table(s)` to DuckDB with snapshot isolation.
4) Return `TargetRunRecord` with real row counts and dataset refs.

#### 6B) Move skip decision “upstream” of tool nodes (avoid wasted tool runs)

Hamilton will execute dependencies, so the “skip check” must be a dependency of tool nodes.

Recommended pattern:
- `typing__input_hash(env, graph) -> str`
- `typing__should_run(env, graph, typing__input_hash) -> bool`
- Tool nodes take `typing__should_run` and short-circuit (return “reused” references) when False.

For artifact paths, rely on deterministic contract templates (e.g., `{build_dir}/typing/...`) so
“reused” can reference existing files without rerun.

#### 6C) Apply the same pattern to `scip` (tool node gating)

If `scip` is purely artifact-based:
- Tool gating still matters to avoid re-indexing when up-to-date.

#### Tests to add

- `tests/build/hamilton/test_pr24_typing_materializes_tables.py`
  - Use an in-memory gateway + mocked tool executor returning deterministic JSON outputs.
  - Assert the expected tables exist after running the node(s) and row counts are non-null.

- `tests/build/hamilton/test_pr24_tool_skip_gates_tool_execution.py`
  - Use a spy/mock to ensure tools are not invoked when skip says “up to date”.

---

### P3-PR-07 — Graph export enrichment (node tags + node_kind) (medium priority)

**Why:** Phase 3 PR-26 intent was to make graph export reflect Hamilton node metadata, not just
TargetGraph metadata. This is required to understand mixed native/wrapper DAGs.

#### Desired export JSON shape (minimum)

For each node:
- `node_name` (Hamilton node function name)
- `target` (build target name if applicable)
- `impl_kind` (`native|wrapper`)
- `node_kind` (`compute|materialize|tool|parse|target|loader|dataset|artifact`)
- `tags` (Hamilton tags dict)
- `outputs` (contract outputs, if this is a target node)

#### Implementation tasks

1) Update `src/codeintel/build/hamilton/observability.py` to collect node tags from the composed
   Hamilton runtime (not only from the target registry).
2) Add a stable export contract to support snapshots/tests.
3) Add `tests/build/hamilton/test_pr26_graph_export_includes_tags.py`.

---

### P3-PR-08 — Phase 3 closure: invariants + docs sync (medium priority)

At the end of Phase 3 completion, add a single “closure PR” that:
- Updates docs that are currently known to be stale relative to implementation.
- Adds a checklist-based validation suite (or extends the existing one).
- Ensures CLI snapshot manifest covers the new CLI surface area.

---

## 2) Phase 4 Remaining Work (Completion Roadmap)

### Current Phase 4 baseline (already implemented)

These are present today:
- Schemas: `build.asset_versions`, `build.run_asset_versions`, `build.asset_lineage`,
  `build.asset_aliases`, `build.asset_diffs` (`src/codeintel/config/datasets/schemas.py`)
- Persistence: `src/codeintel/storage/tracking/asset_tracking.py`
- Emitter: `src/codeintel/build/assets/emitter.py` invoked after Hamilton runs
  (`src/codeintel/build/hamilton/executor.py`)
- CLI: `build.assets`, `build.lineage`, `build.promote`, `build.resolve`, `build.diff`
- Tests: `tests/build/hamilton/test_pr28_phase4_asset_catalog.py`

### Key remaining Phase 4 acceptance criteria

Phase 4 is “complete” when:
1. `version_hash` is stable and meaningfully content-addressed across commits (at least v1).
2. Users can answer:
   - “What changed?” (`build diff`)
   - “What depends on this?” (`build impact` / `build lineage`)
   - “Can I reuse prior outputs?” (cross-commit reuse)
3. Execution can scale:
   - local parallelism where safe
   - incremental/partition-level updates for partitionable tables
4. Reproducibility and governance exist:
   - run environment capture
   - quality gate policies/invariants
5. The ref model is simplified (DatasetRef/ArtifactRef converge toward AssetRef).

---

### P4-PR-01 — Fingerprinting v1: make `version_hash` cross-commit stable (high priority)

**Why:** Current version computation uses `TargetRunRecord.input_hash`, which includes
`snapshot.repo + snapshot.commit` (`src/codeintel/build/hashing.py`). That prevents meaningful
cross-commit reuse and promotion semantics.

#### Requirements

- `version_hash` must not depend on `commit` directly.
- `meta.fingerprint` must record the policy version (e.g., `fast_v1`, `stable_v1`).
- The policy must be deterministic, cheap enough for default usage, and safe to evolve.

#### Proposed v1 stable policy (tables)

Start with an incremental “good enough” policy:
- `schema_hash` (from schema registry)
- `row_count` (snapshot-filtered)
- `target.options_hash`
- **upstream asset version hashes** (from `run_asset_versions` or manifests)

Then add optional stronger modes:
- `stable_v2`: include a content hash of sampled rows or key hash.

#### Proposed v1 stable policy (artifacts)

- Artifact type + bytes + content hash (if feasible)
- Options hash
- Upstream version hashes

#### Implementation tasks

- Add a `FingerprintPolicy` abstraction in `src/codeintel/build/assets/fingerprinting.py`.
- Update `src/codeintel/build/assets/emitter.py` to compute version hashes using the policy.
- Add compatibility: keep old “fast” fingerprints working, but mark them as legacy if needed.

#### Tests

- Add: `tests/build/hamilton/test_pr31_version_hash_is_commit_independent.py`
  - Construct two fake snapshots with different commits and verify the stable policy yields
    the same hash when all other components are equal.

---

### P4-PR-02 — Impact analysis v1: `build impact` (high priority)

**Goal:** Given an asset key or target, compute downstream impacted assets/targets using
`build.asset_lineage`.

#### Implementation tasks

- Add CLI command `build.impact` and handler.
- Implement BFS/DFS over `asset_lineage` edges starting from:
  - a specific `(asset_kind, asset_key, version_hash)` OR
  - the latest version of an asset in a snapshot (or alias).
- Output:
  - impacted assets
  - impacted targets (optional mapping via registry ownership)

#### Tests + snapshots

- Add unit tests against an in-memory gateway with a synthetic lineage graph.
- Add CLI snapshot for `build impact --help`.

---

### P4-PR-03 — Cross-commit reuse v1 (“inherit”) (high priority)

**Goal:** Allow a build on commit B to reuse asset versions from commit A when inputs are
compatible, recording “inherit/reused” semantics in `build.run_asset_versions`.

#### Implementation tasks

- Decide CLI UX:
  - `codeintel build run ... --reuse-from-run <run_id>`
  - or `--reuse-from-commit <sha>` / `--base-commit <sha>`
- Update planning:
  - introduce plan status `inherit` vs `skip` vs `compute`.
- Update execution:
  - materializers can short-circuit and register reused versions (and ensure artifacts exist).

#### Tests

- Add a tiny integration-ish test that:
  1) runs target(s) for commit A
  2) runs target(s) for commit B with reuse enabled
  3) asserts the run B mappings point to A’s versions and `resolution_kind="reused"`

---

### P4-PR-04 — Partition-level incremental (medium priority)

**Goal:** For partitionable tables, update only changed partitions.

#### Design steps

- Identify partitionable tables and encode partitioning metadata in contracts (or execution config).
- Add schema(s) for partition metadata, e.g.:
  - `build.asset_partitions`
  - `build.run_asset_partitions`

#### Implementation tasks

- Extend materializer to support:
  - delete/overwrite of partition subsets
  - per-partition row counts
- Emit partition lineage edges if meaningful.

---

### P4-PR-05 — Execution backend abstraction + local parallelism (medium priority)

**Goal:** Support safe local parallelism for independent targets/nodes.

#### Implementation tasks

- Introduce an execution backend interface (e.g., `SerialBackend`, `ThreadPoolBackend`).
- Ensure storage gateway and contract enforcement are safe under concurrency.
- Add per-target concurrency policy gates (start conservative).

#### Tests

- Determinism tests (same inputs → same outputs).
- Concurrency safety tests (no partial writes; contract enforcement still works).

---

### P4-PR-06 — Remote cache / CAS (artifacts first) (medium priority)

**Goal:** Support a content-addressable artifact store for reuse across machines/CI.

#### Implementation tasks

- Define a `CasStore` interface (local FS implementation first).
- On artifact materialization, upload content keyed by `version_hash`.
- On reuse/inherit, fetch artifacts if missing locally.

---

### P4-PR-07 — Contracts-as-code: derive contracts from Hamilton tags (medium priority)

**Goal:** Reduce duplicated source-of-truth between registry contracts and Hamilton node modules.

#### Implementation tasks

- Define tagging conventions for outputs, e.g.:
  - `@tag(outputs=("analytics.table_a", "analytics.table_b"))`
  - `@tag(artifacts=("export_jsonl",))`
- Implement a scanner that imports native modules and derives contract candidates.
- Add a “contract diff” report (JSON + text) and optionally a CI gate.

---

### P4-PR-08 — Graph exports 2.0: asset graph + version graph (medium priority)

**Goal:** Export:
- Asset graph (logical assets)
- Version graph (asset_versions nodes with asset_lineage edges)

Formats:
- JSON
- Mermaid
- DOT

---

### P4-PR-09 — PR/CI report generation (medium priority)

**Goal:** Generate a single report artifact that includes:
- quality report summary (ruff/pyright/pyrefly)
- build plan summary
- run_nodes profile summary (if enabled)
- asset changes summary (new versions, diffs, promotions)

---

### P4-PR-10 — Backfill orchestration + time series tables (medium priority)

**Goal:** Provide a first-class way to backfill targets over a commit range and record results in
time series tables (e.g., history metrics).

---

### P4-PR-11 — Run environment capture (medium priority)

**Goal:** Persist a reproducibility record per run:
- python version
- OS
- tool versions
- build config hash
- git dirty state (optional)

---

### P4-PR-12 — Quality gates 2.0: invariants + blocking policies (medium priority)

**Goal:** Move from “best effort” validation to configurable, enforceable policies:
- schema validation required for certain targets
- strict contracts required in CI
- prohibit wrapper targets unless allowlisted

---

### P4-PR-13 — Consolidation: unify `DatasetRef`/`ArtifactRef` into `AssetRef` (high effort)

**Goal:** Reduce conceptual surface area and align with Phase 4 asset catalog as the canonical
identity model.

#### Implementation tasks

- Introduce `AssetRef` with:
  - `asset_kind`, `asset_key`, `repo`, `commit`, optional `version_hash`, and metadata.
- Update Hamilton IO nodes and run record structures to carry AssetRefs.
- Provide a compatibility layer for existing refs to avoid a big-bang change.

---

## 3) Suggested Sequencing (Dependency-Aware)

### Phase 3 ordering (recommended)

1) P3-PR-01 Contract parity completion + linter hardening
2) P3-PR-02 call_graph_views schemas + contract alignment
3) P3-PR-03 CLI `auto` mode
4) P3-PR-06 Tool target correctness + skip gating (typing/scip)
5) P3-PR-04 Strict contracts enforcement
6) P3-PR-05 Wrapper allowlist hard gate
7) P3-PR-07 Graph export enrichment
8) P3-PR-08 Closure PR (docs/tests/snapshots)

### Phase 4 ordering (recommended)

1) P4-PR-01 Stable version hashing (enables reuse/impact)
2) P4-PR-02 Impact analysis
3) P4-PR-03 Cross-commit reuse
4) P4-PR-08 Graph exports 2.0 (asset/version)
5) P4-PR-05 Backend abstraction + local parallelism
6) P4-PR-06 Remote cache/CAS
7) P4-PR-07 Contracts-as-code derivation
8) P4-PR-09 CI report generation
9) P4-PR-11 Run environment capture
10) P4-PR-12 Quality gates 2.0
11) P4-PR-10 Backfill orchestration
12) P4-PR-13 AssetRef unification

---

## 4) Working Conventions (per repo AOP)

For each PR:
1) Run:
   ```bash
   uv run python -m tools.quality_report --output build/quality-results/quality_report.json
   ```
2) Fix all Ruff/Pyright/Pyrefly findings (no suppressions).
3) Run:
   ```bash
   uv run pytest -q
   ```
4) If CLI snapshots were touched:
   ```bash
   pytest -m cli_snapshot --cli-snapshot-tags <tag> --update-cli-snapshots
   ```

