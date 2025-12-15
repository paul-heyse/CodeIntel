Below is a **post‑PR‑73 PR‑by‑PR tracking board** in the same format you’ve been using (tasks ✅ / tests ✅ / CLI snapshot ✅), but updated to match your current repo structure in **CodeIntel_PR_73_complete.zip** (notably: `src/codeintel/build/hamilton/*`, `src/codeintel/build/schemas/*`, existing snapshot runner + `tests/build/hamilton/snapshots/manifest.yaml`, and your existing `GraphSource` option already wired into build CLI).

The overall intent is to make the system **truly DAG‑first** and **Hamilton‑centric**:

* **Hamilton graph is the source of truth**
* **BuildSpec** is the compiled “contract” of that graph (targets, deps, outputs, schemas, artifacts, semantic layer pointers)
* **Validator gates** make correctness + drift detection automatic
* **Native targets can depend on native outputs** (no special-casing) by ensuring the generated helper nodes exist for native targets too

---

## Phase 5 tracking board (post‑PR‑73): BuildSpec + validator gates + DAG‑first consolidation

### PR‑74 — Auto mode: generate dataset/loader/artifact nodes for *native* targets too

**Why:** This is the single most important “enabler” PR. Right now your auto mode excludes *the entire target* from the generated module (`exclude_targets=native_names`), which also suppresses the `d__…` and `q__…` helper nodes for native outputs. That blocks native→native composition once you migrate more targets.

**Goal:** In `mode="auto"`, only exclude the **target/materialize nodes** for native targets (to avoid name collisions), but still generate:

* `d__schema__table` dataset nodes
* `q__schema__table` loader nodes
* `a__artifact` artifact nodes

…so downstream nodes can read native outputs via the same `q__…` convention.

#### Code changes

* **Modify**

  * `src/codeintel/build/hamilton/nodes/node_factory.py`

    * Add a new option to `GenerationOptions`, e.g.:

      * `exclude_target_nodes_for_targets: frozenset[str] = frozenset()`
      * (keep existing `exclude_targets` semantics intact for “exclude everything”, if you still want it)
    * Change target generation loop:

      * If target in `exclude_target_nodes_for_targets`: **skip `t__{target}` creation** only
      * Still emit dataset/loader/artifact nodes for that target
  * `src/codeintel/build/hamilton/driver_factory.py`

    * In `mode="auto"`, pass `exclude_target_nodes_for_targets=native_target_names` (instead of excluding the whole target)
* **Optional refactor (nice)**

  * In `node_factory.py`, split `_generate_nodes_for_target(...)` into:

    * `_emit_target_node(...)`
    * `_emit_dataset_nodes(...)`
    * `_emit_loader_nodes(...)`
    * `_emit_artifact_nodes(...)`
  * This makes exclusion logic trivial.

#### Tests to add (under `tests/build/hamilton/`)

* `tests/build/hamilton/test_pr74_auto_mode_native_outputs_have_helpers.py`

  * Assert `list_available_nodes_compat(mode="auto")` contains helper nodes for known native outputs, e.g.:

    * `d__analytics__goid_risk_factors` and `q__analytics__goid_risk_factors` (from native `risk_factors`)
    * `d__graph__v_function_call_counts` and `q__graph__v_function_call_counts` (from native `call_graph_views`)
* `tests/build/hamilton/test_pr74_auto_mode_no_duplicate_target_nodes.py`

  * Ensure there is still only one `t__risk_factors` node in the graph (i.e., generated module did *not* generate the target node for native targets)

#### Snapshots

* None required.

---

### PR‑75 — BuildSpec primitives + deterministic JSON + hashing

**Why:** BuildSpec becomes the single, stable compiled artifact that your CI gates can assert against (and your serving layer can advertise).

#### Code changes

* **Add**

  * `src/codeintel/build/spec/__init__.py`
  * `src/codeintel/build/spec/primitives.py`
  * `src/codeintel/build/spec/serdes.py` (or keep in primitives)
* Suggested minimal datamodel:

  * `BuildSpec`
  * `TargetSpec`
  * `DatasetSpec` (table_key + schema_hash + optionally columns)
  * `ArtifactSpec` (artifact name + format + optional default path template)
  * `SemanticSpec` (optional pointer: semantic_registry hash/version)
* Add **canonical JSON** generation:

  * Stable ordering for lists (`sorted(...)`)
  * `json.dumps(..., sort_keys=True, separators=(",", ":"))`
  * `buildspec_hash = sha256(canonical_json.encode()).hexdigest()`

#### Example skeleton

```python
# src/codeintel/build/spec/primitives.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class DatasetSpec:
    table_key: str
    schema_hash: str

@dataclass(frozen=True)
class ArtifactOutSpec:
    name: str
    # optional: "jsonl", "parquet", "scip", etc
    kind: str | None = None

@dataclass(frozen=True)
class TargetSpec:
    name: str
    domain: str
    impl_kind: Literal["native", "wrapper"]
    deps: tuple[str, ...]
    outputs: tuple[str, ...]        # table_keys
    artifacts: tuple[ArtifactOutSpec, ...]

@dataclass(frozen=True)
class BuildSpec:
    spec_version: int
    targets: tuple[TargetSpec, ...]
    datasets: tuple[DatasetSpec, ...]
    buildspec_hash: str
```

#### Tests to add

* `tests/build/hamilton/test_pr75_buildspec_serdes_is_deterministic.py`

  * Construct a small `BuildSpec` object, serialize twice, assert identical
  * Roundtrip `to_json()` → `from_json()`

#### Snapshots

* None required.

---

### PR‑76 — BuildSpec compiler: compile from Hamilton DAG (auto mode)

**Why:** This is the “DAG‑first” move: compile targets/outputs/deps *from the actual Hamilton graph*, not a parallel registry.

#### Code changes

* **Add**

  * `src/codeintel/build/spec/compile.py`
* **Modify**

  * `src/codeintel/build/hamilton/introspect.py`

    * Add `derive_target_outputs(...)` (datasets + artifacts) derived from node dependencies

#### Compiler strategy (deterministic, DAG‑first)

1. Build runtime: `build_driver(mode="auto")`
2. Identify all **materialize** nodes: `tags[TAG_NODE_TYPE] == NODE_TYPE_MATERIALIZE`
3. For each materialize node:

   * `target_name = tags[TAG_TARGET]`
   * deps = existing `derive_target_dependencies(runtime)`
4. Outputs:

   * scan `dataset` nodes whose dependency includes this materialize node
   * output table_key from `dataset_node.tags[TAG_TABLE_KEY]`
5. Artifacts:

   * scan `artifact` nodes dependent on the materialize node
   * artifact name from `artifact_node.tags[TAG_ARTIFACT]`
6. Dataset schemas:

   * use your canonical schema provider (post‑PR‑73) to fetch `TableSchema`
   * hash with `codeintel.core.schemas.hashing.stable_schema_hash(...)`
7. `impl_kind`:

   * for now: `codeintel.build.hamilton.native.registry.is_native_target(name)`
     (later you can tag materialize nodes with `impl_kind` directly)

#### Tests to add

* `tests/build/hamilton/test_pr76_buildspec_compiler_outputs_match_dag.py`

  * Compile spec and assert:

    * `risk_factors` target exists
    * `analytics.goid_risk_factors` appears in its `outputs`
    * `export_jsonl` has artifact `jsonl_export`
* `tests/build/hamilton/test_pr76_buildspec_compiler_is_stable.py`

  * Compile twice, compare `buildspec_hash` identical

#### Snapshots

* None yet.

---

### PR‑77 — BuildSpec CLI + CLI snapshots + optional “write to file”

**Why:** You’ll want a stable CLI for CI, and for humans to inspect.

#### Code changes

* **Add**

  * `src/codeintel/cli/commands/build_spec.py`
  * `src/codeintel/cli/handlers/build_spec.py`
* **Modify**

  * `src/codeintel/cli/commands/build.py`

    * `build_app.command(build_spec_app, name="spec")`

#### CLI shape

* `codeintel build spec compile [--format json] [--output PATH] [--include-columns/--no-include-columns]`

  * Default prints to stdout
* Optional:

  * `codeintel build spec show` (summary counts)
  * `codeintel build spec diff <old> <new>` (later PR)

#### Tests to add

* `tests/build/hamilton/test_pr77_build_spec_cli_outputs_valid_json.py`

  * run via cyclopts test runner pattern you already use (similar to schema tests)

#### Snapshot changes (under `tests/build/hamilton/snapshots/`)

Update `tests/build/hamilton/snapshots/manifest.yaml` with new cases:

* `pr77_build_spec_help.txt`

  * args: `["build", "spec", "--help"]`
* `pr77_build_spec_compile_help.txt`

  * args: `["build", "spec", "compile", "--help"]`
* `pr77_build_spec_compile_auto.json`

  * args: `["build", "spec", "compile", "--format", "json"]`
  * snapshot: `tests/build/hamilton/snapshots/pr77_build_spec_compile_auto.json`
  * (Make sure the output does **not** include timestamps/absolute paths.)

---

### PR‑78 — Hamilton graph validator gate + CLI `build validate`

**Why:** This becomes the choke point for correctness and drift. Once this is in place, subsequent PRs become “clean” because validators enforce invariants.

#### Code changes

* **Add**

  * `src/codeintel/build/hamilton/validate.py` (or `validator.py`)
  * `src/codeintel/cli/handlers/build_validate.py` **or** extend `cli/handlers/build.py`
  * `src/codeintel/cli/commands/build_validate.py` **or** add `BuildValidateCommand` in `cli/commands/build.py`

#### Validator invariants (suggested)

At minimum:

* Every materialize node has:

  * `domain`, `target`, `node_type="materialize"`
* Every dataset node has:

  * `table_key`, `domain`, `node_type="dataset"`
* Every artifact node has:

  * `artifact`, `domain`, `node_type="artifact"`
* Every produced `table_key` resolves through the canonical schema provider
* No `table_key` is “produced” by more than one target (uniqueness)
* Derived target dependency graph from Hamilton has no cycles
* (Optional) Derived deps match `TargetGraph` deps **only while TargetGraph still exists** (warn-only)

#### Tests to add

* `tests/build/hamilton/test_pr78_graph_validator_clean_auto.py`

  * `validate_graph(mode="auto")` returns no errors
* `tests/build/hamilton/test_pr78_graph_validator_finds_duplicate_producers.py`

  * create a tiny fake runtime graph (or monkeypatch) with two dataset nodes pointing to same table_key → ensure error

#### Snapshot changes

Add to manifest:

* `pr78_build_validate_help.txt`

  * args: `["build", "validate", "--help"]`
* Optional: `pr78_build_validate_auto.json`

  * args: `["build", "validate", "--format", "json"]`

---

### PR‑79 — Flip defaults to DAG‑first: `graph_source="hamilton"` by default (aggressive but correct)

**Why:** You already have `GraphSource` plumbed everywhere and `target_graph_from_hamilton(runtime)` implemented. With PR‑78 validator gates, you can now confidently default to Hamilton-derived dependencies and treat `targetgraph` as “legacy mode”.

#### Code changes

* **Modify**

  * `src/codeintel/cli/commands/build.py`

    * change defaults for `graph_source` fields:

      * `"targetgraph"` → `"hamilton"`
    * update help strings accordingly (this will cause snapshot updates)
  * `src/codeintel/cli/handlers/build.py`

    * remove the “default targetgraph” fallback in `parse_graph_source(...)`

#### Tests to add

* `tests/build/hamilton/test_pr79_graph_source_default_is_hamilton.py`

  * ensure command handler defaults are now hamilton
* `tests/build/hamilton/test_pr79_targetgraph_and_hamilton_sources_equivalent_for_wrapper_targets.py`

  * pick a wrapper-only subset and assert closure sets match (this is a safety net)

#### Snapshot updates

This will change help text defaults. Expect to **update existing snapshot contents** (filenames can stay the same), particularly:

* `pr09_plan_help.txt`
* `pr14_graph_help.txt`
* `pr15_explain_help.txt`
* and any other help snapshots that mention “targetgraph (default)”

No new snapshot filenames required unless you want to add a “default changed” proof snapshot.

---

### PR‑80 — Batch schema inference for schema compilation (speed + determinism)

**Why:** Your schema compilation path (`build schema compile --infer-native`) currently infers table schemas one-by-one. Batch inference lets you seed once and infer many outputs in one ephemeral DuckDB session, which is faster and more reliable in CI.

#### Code changes

* **Modify**

  * `src/codeintel/build/schemas/provider_hamilton.py`

    * Add something like `infer_table_schemas(table_keys: Iterable[str]) -> dict[str, TableSchema]`
  * `src/codeintel/build/schemas/compile.py`

    * When `infer_native=True` and provider is Hamilton-enabled, call batch inference for all inferable keys up front and populate provider cache
  * `src/codeintel/build/schemas/seed_harness.py`

    * Ensure seeding can happen once per batch run

#### Tests to add

* `tests/build/hamilton/test_pr80_schema_compile_uses_batch_inference.py`

  * Use a small spy/mocking seam (e.g., injectable “infer function”) and assert it’s called once for a batch, not N times
* `tests/build/hamilton/test_pr80_schema_manifest_identical_batch_vs_individual.py`

  * Compile manifest in “batch” mode and “legacy per-table” mode for a small set → identical JSON (minus ordering, which should be stable)

#### Snapshots

* None required unless you want a new `pr80_schema_compile_help.txt` documenting a `--batch-infer` flag (optional).

---

## Optional follow‑on PRs (recommended), once PR‑74..PR‑80 land

These aren’t strictly required for “BuildSpec + validator gates”, but they’re the next obvious consolidation moves once you have those gates.

### PR‑81 — Use BuildSpec as the single metadata source for serving `/meta` + MCP inventory

* **Goal:** serving no longer re-discovers datasets/views ad hoc; it loads the current `BuildSpec` artifact (or buildspec hash) and returns:

  * semantic layer version/hash
  * dataset inventory + schema hashes
  * target inventory + dependencies
* **Likely files**

  * `src/codeintel/serving/meta.py` (or wherever your meta endpoint is)
  * `src/codeintel/serving/semantic_registry.py` integration

### PR‑82 — Start deleting legacy contract shims (`src/codeintel/config/datasets/contracts.py`)

* **Goal:** if you’re comfortable breaking backward compatibility (you are), eliminate:

  * `_JSON_SCHEMA_BY_DATASET_NAME`
  * `_DEFAULT_JSONL_FILENAMES`
  * `_DEFAULT_PARQUET_FILENAMES`
* Replace with:

  * deterministic defaults derived from `table_key`
  * optional override tags in Hamilton (later)

This PR will probably require updating anything that still calls `codeintel.config.datasets.contracts.*` shims.

### PR‑83 — Add a native-target scaffolder for rapid migration (template generator)

* **Goal:** make future “migrate target X to native Hamilton” PRs extremely fast & consistent.
* Example: `codeintel build native scaffold --target call_graph`

  * generates `src/codeintel/build/hamilton/native/graphs/call_graph.py` skeleton with:

    * `t__call_graph__compute`
    * `t__call_graph` materializer
    * correct tags
    * placeholder tests

---

## How to execute this board in the “mergeable & safe” style you like

A very clean gating order is:

1. **PR‑74** (native outputs have helper nodes)
2. **PR‑75–77** (BuildSpec compile + CLI + snapshot)
3. **PR‑78** (validator gate)
4. **PR‑79** (flip defaults)
5. **PR‑80** (batch inference speedups)

After that, almost every subsequent consolidation PR becomes easy because:

* BuildSpec gives you a stable compiled view of the world
* Validator catches drift immediately
* DAG-first planning is now the default behavior

---

If you want, I can also append a **single “DoD gate checklist”** for PR‑74..PR‑80 (like you did in Phase 2) that explicitly verifies:

* native outputs have `d__/q__/a__` helpers in auto mode
* BuildSpec compile is stable (hash unchanged across two compiles)
* validator returns 0 issues
* CLI snapshots include spec/validate commands
* graph_source default is hamilton and old snapshots updated correctly

# DoD gate checklist #

Here’s a single **PR‑74…PR‑80 “Definition of Done” gate checklist** you can use exactly like Phase 2: it’s a compact, enforceable set of checks that prove the migration is complete and stable.

---

# Phase 5 DoD Gate (PR‑74 … PR‑80)

## Gate 0 — Basic hygiene (must be true before anything else)

* [ ] `pytest` passes locally and in CI (no xfails added to “get green”).
* [ ] No new “legacy graph_source” defaults remain in CLI help or docs.
* [ ] No hardcoded schema sources were reintroduced (schema provider remains canonical).

---

## Gate 1 — Native outputs have `d__/q__/a__` helpers in `auto` mode (PR‑74)

### Required behavior

* [ ] For **native targets**, the `auto` driver includes **dataset nodes** (`d__…`) and **loader nodes** (`q__…`, and `df__…` if you generate them), and **artifact nodes** (`a__…`) for the native outputs.
* [ ] The `auto` driver **does not** generate a duplicate `t__<target>` node for native targets.

### Concrete checks

* [ ] Pick 2–3 native targets with clear outputs (e.g., `risk_factors`, one semantic view target, one export/artifact target) and verify:

  * dataset node exists for each produced table: `d__<schema>__<table>`
  * loader node exists: `q__<schema>__<table>`
  * artifact node exists when applicable: `a__<artifact>`
* [ ] Confirm there is exactly one `t__risk_factors` node in the graph.

### Tests

* [ ] `tests/build/hamilton/test_pr74_auto_mode_native_outputs_have_helpers.py` passes
* [ ] `tests/build/hamilton/test_pr74_auto_mode_no_duplicate_target_nodes.py` passes

---

## Gate 2 — BuildSpec compilation is deterministic & stable (PR‑75…PR‑77)

### Required behavior

* [ ] `codeintel build spec compile --format json` produces:

  * stable ordering
  * stable schema hashes
  * a stable `buildspec_hash` (no timestamps, absolute paths, or run IDs embedded)
* [ ] Running the command twice on the same codebase yields **identical output**.

### Concrete checks

* [ ] Run twice and compare hashes:

  * `buildspec_hash` is identical
  * serialized JSON output matches exactly

### Tests

* [ ] `tests/build/hamilton/test_pr75_buildspec_serdes_is_deterministic.py` passes
* [ ] `tests/build/hamilton/test_pr76_buildspec_compiler_is_stable.py` passes
* [ ] `tests/build/hamilton/test_pr76_buildspec_compiler_outputs_match_dag.py` passes
* [ ] `tests/build/hamilton/test_pr77_build_spec_cli_outputs_valid_json.py` passes

### CLI snapshots

* [ ] Snapshot cases added & passing:

  * `build spec --help` → `pr77_build_spec_help.txt`
  * `build spec compile --help` → `pr77_build_spec_compile_help.txt`
  * `build spec compile --format json` → `pr77_build_spec_compile_auto.json`
* [ ] Snapshot output excludes dynamic fields (or normalization rules handle them).

---

## Gate 3 — Graph validator returns 0 issues in `auto` mode (PR‑78)

### Required behavior

`codeintel build validate` (or equivalent) must report **zero** issues for the current repo.

Minimum invariants it should check:

* [ ] Every materialize node has required tags (`domain`, `target`, `node_type=materialize`)
* [ ] Every dataset node has `table_key` and `node_type=dataset`
* [ ] Every produced `table_key` resolves via SchemaProvider
* [ ] No duplicate producers for the same `table_key`
* [ ] Derived target dependency graph is acyclic
* [ ] (Optional during transition) TargetGraph parity warnings don’t fail the run

### Tests

* [ ] `tests/build/hamilton/test_pr78_graph_validator_clean_auto.py` passes
* [ ] `tests/build/hamilton/test_pr78_graph_validator_finds_duplicate_producers.py` passes

### CLI snapshots

* [ ] Snapshot cases added & passing:

  * `build validate --help` → `pr78_build_validate_help.txt`
  * (Optional) `build validate --format json` → `pr78_build_validate_auto.json`

---

## Gate 4 — `graph_source` default is Hamilton, and snapshots are updated (PR‑79)

### Required behavior

* [ ] CLI defaults now prefer `graph_source=hamilton` everywhere relevant (plan/graph/explain).
* [ ] Help text, docs, and CLI outputs reflect the new default.
* [ ] Old snapshots that referenced targetgraph as default are updated.

### Concrete checks

* [ ] Run `codeintel build plan --help` and confirm the default `graph_source` is Hamilton.
* [ ] Run an un-flagged plan/graph/explain command and confirm it uses Hamilton-derived dependencies by default.

### Tests

* [ ] `tests/build/hamilton/test_pr79_graph_source_default_is_hamilton.py` passes
* [ ] `tests/build/hamilton/test_pr79_targetgraph_and_hamilton_sources_equivalent_for_wrapper_targets.py` passes (or the equivalent “safety net” test)

### Snapshot updates

* [ ] All impacted help snapshots updated (no lingering “targetgraph (default)” anywhere):

  * plan help snapshots
  * graph help snapshots
  * explain help snapshots
* [ ] Snapshot suite passes cleanly without “update mode”.

---

## Gate 5 — Batch schema inference is the default fast path (PR‑80)

### Required behavior

* [ ] Schema compilation uses one ephemeral DB session per compile run (or equivalent), not “one DB per table.”
* [ ] Batch-inferred schemas match legacy per-table inference output (for a representative subset).
* [ ] No nondeterminism introduced (ordering, types, nullability stable).

### Concrete checks

* [ ] Compile schema manifest twice → identical.
* [ ] Compare “batch mode” vs “legacy mode” output for a subset (or run a dedicated test).

### Tests

* [ ] `tests/build/hamilton/test_pr80_schema_compile_uses_batch_inference.py` passes
* [ ] `tests/build/hamilton/test_pr80_schema_manifest_identical_batch_vs_individual.py` passes

### Snapshots

* [ ] None required unless you added CLI flags/help output changes (then add/refresh help snapshots).

---

# Single-command CI gate (recommended)

Add a CI job that runs:

1. Unit + integration:

* [ ] `pytest -q`

2. CLI snapshot suite (tag-filtered):

* [ ] `pytest -m cli_snapshot -q`
  (or `--cli-snapshot-tags pr77,pr78` if you use tag subsets)

3. Optional: explicit “no-issue validator” smoke:

* [ ] `codeintel build validate --format json` returns `"issues": []`

---

# Minimal “Go/No-Go” summary (for a release checklist)

✅ Ship PR‑74..PR‑80 only if all are true:

* [ ] Auto mode exposes helpers for native outputs (`d__/q__/a__`) and no duplicate `t__` nodes
* [ ] `build spec compile` output is deterministic and hash-stable
* [ ] `build validate` returns zero issues
* [ ] Spec/validate CLI snapshots exist and pass
* [ ] `graph_source` default is Hamilton and all help snapshots are updated
* [ ] Batch inference is active and produces identical manifests to the old path

---

If you want, I can also turn this into a single file you can drop into the repo (e.g., `docs/build/phase5_dod_gate.md`) and a tiny “meta test” (`tests/build/hamilton/test_phase5_dod_gate.py`) that asserts these invariants programmatically (including running the CLI snapshot manifest subset).

# meta test to implement #