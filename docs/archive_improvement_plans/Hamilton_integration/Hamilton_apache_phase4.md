Below is (1) a **Phase 4 asset catalog schema** (DuckDB) that supports **versions + lineage + diffs + promotion from day one**, and (2) a **Phase 4 PR-by-PR tracking board (PR‑28 … PR‑45)** in the same format you’ve been using (tasks, tests, CLI snapshots).

This Phase 4 plan assumes you’re carrying forward the Phase 2/3 baseline: Hamilton is already the source of truth with planning (`compute_plan()`), manifest prefetch, `DatasetRef`/`ArtifactRef`, loader nodes (`q__*`, `df__*`), run target persistence, graph exports, staleness explanation, and the YAML-driven CLI snapshot framework.  
It also assumes you preserve Phase 1 correctness invariants (closure execution, upstream failure gating, run tracking, dataset lineage, observability). 
And it builds directly on the Phase 3 “pure compute + explicit materializers” direction (materialization boundaries, caching becomes meaningful, target-by-target migration). 

---

# Phase 4 Asset Catalog Schema (DuckDB)

## Design goals

**Day-one support** for:

* **Asset identity**: “what exists?”
* **Asset versions**: “what version did we produce?”
* **Lineage**: “what did it depend on?”
* **Diffs**: “what changed between versions?”
* **Promotion/Aliases**: “what should users treat as ‘blessed’?”

…and it should integrate cleanly with the existing per-run/per-target observability you already have (`build.run_targets`, `build history --run-id`, etc.). 

## Day-one tables

You can ship Phase 4 with these tables **immediately** and expand later.

### 1) `build.assets` (dimension)

One row per logical asset.

**Key columns**

* `asset_kind` (`table|view|artifact`)
* `asset_key` (e.g., `analytics.function_metrics`, `scip_index`)
* `owner_target` (which target “owns” producing it)
* `contract` (JSON: schema expectations, primary keys, etc.)
* `created_at`, `updated_at`

### 2) `build.asset_versions` (fact)

One row per produced (or reused) asset version.

**Key columns**

* `asset_kind`, `asset_key`
* `version_hash` (content-addressed identity)
* `repo`, `commit` (snapshot identity; matches your DatasetRef v2 style) 
* `run_id`, `target`, `impl_kind`
* `input_hash`, `options_hash`
* `schema_hash`, `row_count`, `bytes`
* `location` (table name/view name/file path/URI)
* `status` (`materialized|reused|failed`)
* `meta` (JSON)

### 3) `build.asset_lineage` (edges)

Edges between *asset versions* (not just logical assets). This is crucial for diffs and replay.

**Key columns**

* downstream: (`asset_kind`, `asset_key`, `version_hash`)
* upstream: (`asset_kind`, `asset_key`, `version_hash`)
* `edge_kind`: `reads_from|depends_on|produces|reuses`
* `meta` (JSON: dependency reason, join keys, etc.)

### 4) `build.asset_aliases` (promotion/labels)

This is “promotion from day one” without committing you to a complicated release system.

Examples:

* alias `latest` per asset
* alias `main` per asset
* alias `release-2025.01` per asset
* alias `prod` later (even if not in prod today)

**Key columns**

* `alias` (string)
* `asset_kind`, `asset_key`
* `version_hash`
* `set_by_run_id`, `set_at`
* `note`

### 5) `build.asset_diffs` (cached diffs)

Store computed diffs so repeated comparisons are cheap.

**Key columns**

* `asset_kind`, `asset_key`
* `from_version_hash`, `to_version_hash`
* `diff_kind`: `schema|rowcount|keys|sample|profile`
* `summary` JSON
* `computed_at`, `computed_by_run_id`

---

## Suggested implementation in your existing `TableSchema` style

You already define build schemas in a `TableSchema(...)`/`Column(...)` style (e.g., Phase 2 run targets persistence) and then ensure schema creation during runtime. 

Here’s a concrete schema proposal in that style:

```python
# src/codeintel/build/assets/schemas.py
from __future__ import annotations

from codeintel.storage.schemas import Column, TableSchema  # adjust import path to your project

ASSET_SCHEMAS: tuple[TableSchema, ...] = (
    TableSchema(
        schema="build",
        name="assets",
        columns=[
            Column("asset_kind", "VARCHAR", nullable=False),
            Column("asset_key", "VARCHAR", nullable=False),
            Column("owner_target", "VARCHAR"),
            Column("contract", "JSON"),
            Column("created_at", "TIMESTAMP", nullable=False),
            Column("updated_at", "TIMESTAMP", nullable=False),
        ],
        primary_key=("asset_kind", "asset_key"),
    ),
    TableSchema(
        schema="build",
        name="asset_versions",
        columns=[
            Column("asset_kind", "VARCHAR", nullable=False),
            Column("asset_key", "VARCHAR", nullable=False),
            Column("version_hash", "VARCHAR", nullable=False),

            Column("repo", "VARCHAR", nullable=False),
            Column("commit", "VARCHAR", nullable=False),

            Column("run_id", "VARCHAR"),
            Column("target", "VARCHAR"),
            Column("impl_kind", "VARCHAR"),  # wrapper|native

            Column("status", "VARCHAR", nullable=False),  # materialized|reused|failed
            Column("location", "VARCHAR"),

            Column("input_hash", "VARCHAR"),
            Column("options_hash", "VARCHAR"),

            Column("schema_hash", "VARCHAR"),
            Column("row_count", "BIGINT"),
            Column("bytes", "BIGINT"),

            Column("created_at", "TIMESTAMP", nullable=False),
            Column("meta", "JSON"),
        ],
        primary_key=("asset_kind", "asset_key", "version_hash"),
    ),
    TableSchema(
        schema="build",
        name="asset_lineage",
        columns=[
            Column("downstream_kind", "VARCHAR", nullable=False),
            Column("downstream_key", "VARCHAR", nullable=False),
            Column("downstream_version", "VARCHAR", nullable=False),

            Column("upstream_kind", "VARCHAR", nullable=False),
            Column("upstream_key", "VARCHAR", nullable=False),
            Column("upstream_version", "VARCHAR", nullable=False),

            Column("edge_kind", "VARCHAR", nullable=False),  # reads_from|depends_on|...
            Column("created_at", "TIMESTAMP", nullable=False),
            Column("meta", "JSON"),
        ],
        primary_key=(
            "downstream_kind", "downstream_key", "downstream_version",
            "upstream_kind", "upstream_key", "upstream_version",
            "edge_kind",
        ),
    ),
    TableSchema(
        schema="build",
        name="asset_aliases",
        columns=[
            Column("alias", "VARCHAR", nullable=False),
            Column("asset_kind", "VARCHAR", nullable=False),
            Column("asset_key", "VARCHAR", nullable=False),
            Column("version_hash", "VARCHAR", nullable=False),

            Column("set_by_run_id", "VARCHAR"),
            Column("set_at", "TIMESTAMP", nullable=False),
            Column("note", "VARCHAR"),
        ],
        primary_key=("alias", "asset_kind", "asset_key"),
    ),
    TableSchema(
        schema="build",
        name="asset_diffs",
        columns=[
            Column("asset_kind", "VARCHAR", nullable=False),
            Column("asset_key", "VARCHAR", nullable=False),

            Column("from_version_hash", "VARCHAR", nullable=False),
            Column("to_version_hash", "VARCHAR", nullable=False),

            Column("diff_kind", "VARCHAR", nullable=False),
            Column("summary", "JSON"),

            Column("computed_at", "TIMESTAMP", nullable=False),
            Column("computed_by_run_id", "VARCHAR"),
        ],
        primary_key=("asset_kind", "asset_key", "from_version_hash", "to_version_hash", "diff_kind"),
    ),
)
```

### “Day one” invariants

* `build.asset_versions.status='reused'` must still have a valid `version_hash` (you are pointing to an existing version).
* `build.asset_lineage` edges should be inserted for both:

  * **materialized** outputs (full lineage)
  * **reused** outputs (edge_kind=`reuses` pointing at the upstream version)

---

# Phase 4 Snapshot Tag Taxonomy Extensions

You already have a well-structured snapshot taxonomy (PR tags + command tags + format tags + scope tags + mode tags). 
Extend it with:

* **PR tags**: `pr28` … `pr45`
* **Command tags**: add `assets`, `lineage`, `diff`, `impact`, `promote`, `backfill`, `contracts`, `report`
* **Format tags**: reuse `json|text|dot|mermaid`
* **Scope tags**: reuse `tiny|integration`
* **Mode tags**: reuse `generated|phase0` and add `auto` if Phase 3 introduced it

---

# Phase 4 PR-by-PR Tracking Board (PR‑28 … PR‑45)

Each PR includes:

* **Tasks checklist**
* **Tests checklist** under `tests/build/hamilton/`
* **CLI snapshots** (add to `tests/build/hamilton/snapshots/manifest.yaml` + new golden files)

You’ll use the same snapshot runner you already have (manifest-driven, dynamic field stripping, tag filters, etc.).  

---

## PR‑28 — Asset catalog schema + persistence API skeleton

### Tasks

* [ ] Add `src/codeintel/build/assets/schemas.py` (tables above)
* [ ] Add `gateway.build.ensure_asset_tables()` (or reuse existing schema initializer)
* [ ] Add persistence API (`AssetCatalog`):

  * `upsert_asset(...)`
  * `insert_asset_version(...)`
  * `insert_lineage_edges(...)`
  * `set_alias(...)`
  * `get_asset_versions(...)`
* [ ] Ensure schema creation happens in your “build init” path (same place you ensured `run_targets`). 

### Tests

* [ ] `tests/build/hamilton/test_pr28_asset_schema_creation.py`
* [ ] `tests/build/hamilton/test_pr28_asset_catalog_insert_select.py`

### CLI snapshots

(Help-only first, to lock CLI surfaces before semantics)

* **Command**

  ```bash
  codeintel build assets --help
  ```

  **Snapshot**: `pr28_assets_help.txt` (text)

* **Command**

  ```bash
  codeintel build lineage --help
  ```

  **Snapshot**: `pr28_lineage_help.txt` (text)

---

## PR‑29 — Emit asset versions + lineage from materializers (wrapper + native)

### Tasks

* [ ] Hook into **table materialization** and **artifact materialization**:

  * compute `schema_hash`, `row_count`
  * compute `version_hash` (fast fingerprint is OK initially)
* [ ] Insert into `build.asset_versions`
* [ ] Build lineage edges automatically:

  * downstream = produced assets
  * upstream = all input assets referenced via loader nodes (`q__*`, `df__*`, `a__*`)
* [ ] Ensure this works when a target is **skipped** (reused) and still emits `reused` version records.

### Tests

* [ ] `test_pr29_asset_versions_written_on_run.py` (integration-ish tiny run)
* [ ] `test_pr29_lineage_edges_written.py`

### CLI snapshots

* **Command**

  ```bash
  codeintel build assets --format json
  ```

  **Snapshot**: `pr29_assets_list.json`

*(Even if the list is empty in tiny mode, it should be stable.)*

---

## PR‑30 — CLI: `build assets` + `build lineage` (first useful UX)

### Tasks

* [ ] Implement `codeintel build assets`:

  * list assets
  * `--asset <key>` filter
  * `--versions` show versions
* [ ] Implement `codeintel build lineage --asset <key> --direction up|down --depth N`
* [ ] JSON output is canonical; optional pretty text.

### Tests

* [ ] `test_pr30_assets_cli_returns_stable_json.py`
* [ ] `test_pr30_lineage_cli_depth.py`

### CLI snapshots

* **Command**

  ```bash
  codeintel build assets --format json --versions --asset analytics.function_metrics
  ```

  **Snapshot**: `pr30_assets_versions_function_metrics.json`

* **Command**

  ```bash
  codeintel build lineage --format json --asset analytics.goid_risk_factors --depth 2
  ```

  **Snapshot**: `pr30_lineage_risk_factors_depth2.json`

---

## PR‑31 — Asset fingerprinting v1: stable `version_hash` policy

### Tasks

* [ ] Implement `version_hash` policy:

  * `fast`: hash of (`asset_kind`, `asset_key`, `schema_hash`, `row_count`, `input_hash`, `options_hash`)
  * allow future `strong` mode (table sample hash / full scan)
* [ ] Add `--fingerprint strong|fast` config flag (default `fast`)
* [ ] Store fingerprint mode in `asset_versions.meta`

### Tests

* [ ] `test_pr31_version_hash_stable_for_same_inputs.py`
* [ ] `test_pr31_version_hash_changes_when_schema_changes.py`

### CLI snapshots

* **Command**

  ```bash
  codeintel build assets --format json --asset analytics.goid_risk_factors --versions
  ```

  **Snapshot**: `pr31_assets_versions_include_version_hash.json`

---

## PR‑32 — Promotions/Aliases: `build promote` + `build resolve`

### Tasks

* [ ] Implement `codeintel build promote`:

  * `--asset <key>`
  * `--alias main|latest|release-...`
  * `--version-hash ...` or `--from-run-id ...`
* [ ] Implement `codeintel build resolve --asset <key> --alias main`
* [ ] Ensure aliases are purely metadata (no copy)

### Tests

* [ ] `test_pr32_promote_sets_alias.py`
* [ ] `test_pr32_resolve_returns_version.py`

### CLI snapshots

* **Command**

  ```bash
  codeintel build promote --help
  ```

  **Snapshot**: `pr32_promote_help.txt`

* **Command**

  ```bash
  codeintel build resolve --format json --asset analytics.goid_risk_factors --alias main
  ```

  **Snapshot**: `pr32_resolve_risk_factors_main.json`

---

## PR‑33 — Asset diffs: `build diff` + cached `build.asset_diffs`

### Tasks

* [ ] Implement `codeintel build diff`:

  * `--asset <key>`
  * `--from <version-hash|commit|alias>`
  * `--to <version-hash|commit|alias>`
* [ ] Support diff kinds:

  * schema diff (columns added/removed/type changes)
  * row_count diff
  * optional “profile diff” (null % by column, etc.)
* [ ] Write diff results into `build.asset_diffs`

### Tests

* [ ] `test_pr33_schema_diff_detects_column_add.py`
* [ ] `test_pr33_diff_is_cached.py`

### CLI snapshots

* **Command**

  ```bash
  codeintel build diff --format json --asset analytics.goid_risk_factors --from main --to latest
  ```

  **Snapshot**: `pr33_diff_risk_factors_main_vs_latest.json`

---

## PR‑34 — Impact analysis v1: `build impact`

### Tasks

* [ ] Implement `codeintel build impact --base <commit> --head <commit>`
* [ ] Start with rule-based mapping:

  * if any `*.py` changes → impact ingestion + downstream (configurable)
  * if only docs changed → impact nothing
* [ ] Output:

  * changed files
  * impacted targets
  * impacted assets
  * “because” explanations

### Tests

* [ ] `test_pr34_impact_docs_only_impacts_none.py`
* [ ] `test_pr34_impact_python_change_impacts_ingestion.py`

### CLI snapshots

* **Command**

  ```bash
  codeintel build impact --format json --base COMMIT_A --head COMMIT_B
  ```

  **Snapshot**: `pr34_impact_example.json`
  *(Normalize commit IDs via snapshot replace rules.)*

---

## PR‑35 — Cross-commit reuse v1: base builds + “inherit” plan status

### Tasks

* [ ] Add `--base <commit>` to `build plan` and `build run`
* [ ] Extend planner to output a new status: `inherit`
* [ ] Execution semantics:

  * inherited targets do not compute
  * their DatasetRefs/AssetVersions point to base commit versions (no copy)
* [ ] Record `reused` asset versions + lineage edge_kind=`reuses`

### Tests

* [ ] `test_pr35_plan_includes_inherit_status.py`
* [ ] `test_pr35_run_reuses_base_assets.py`

### CLI snapshots

* **Command**

  ```bash
  codeintel build plan risk_factors --format json --base COMMIT_A
  ```

  **Snapshot**: `pr35_plan_inherit_risk_factors.json`

---

## PR‑36 — Partition-level incremental (for file-partitionable tables)

### Tasks

* [ ] Add optional table partition metadata (file_path/module_id/etc.)
* [ ] Add table: `build.asset_partitions` (new schema)
* [ ] Materializer supports:

  * delete+replace partitions only for changed inputs
* [ ] Integrate with impact results (changed files → changed partitions)

### Tests

* [ ] `test_pr36_partition_replace_only_changes_affected_rows.py`
* [ ] `test_pr36_partition_lineage_written.py`

### CLI snapshots

* **Command**

  ```bash
  codeintel build plan modules --format json --base COMMIT_A
  ```

  **Snapshot**: `pr36_plan_partition_incremental_hint.json`
  *(Even if this is just metadata in the plan entry.)*

---

## PR‑37 — Execution backend abstraction + local parallelism

### Tasks

* [ ] Add `ExecutionBackend` interface:

  * local threads backend
  * local processes backend (optional)
* [ ] Add `--backend threads|processes|sync` to `build run`
* [ ] Ensure materializers obey safe write rules (single-writer where needed)

### Tests

* [ ] `test_pr37_backend_flag_parses.py`
* [ ] `test_pr37_threads_backend_smoke.py`

### CLI snapshots

* **Command**

  ```bash
  codeintel build run --help
  ```

  **Snapshot**: `pr37_run_help_backend_option.txt`

---

## PR‑38 — Remote cache / content-addressable store (artifacts first)

### Tasks

* [ ] Implement local filesystem CAS:

  * `cas/<version_hash>/...`
* [ ] Store artifact outputs in CAS; record `location` as CAS path/URI
* [ ] Add `--cache-dir` / config option
* [ ] Optional: table export/import to Parquet in CAS (can be Phase 4.2)

### Tests

* [ ] `test_pr38_artifact_put_get_roundtrip.py`
* [ ] `test_pr38_reuse_from_cache_marks_status_reused.py`

### CLI snapshots

* **Command**

  ```bash
  codeintel build assets --format json --asset scip_index --versions
  ```

  **Snapshot**: `pr38_assets_artifact_versions_cached.json`

---

## PR‑39 — Contracts-as-code v1: scan Hamilton tags to build contracts

### Tasks

* [ ] Add `@output_contract(...)` decorator on native materializers
* [ ] Implement scanner that builds a `TargetGraph` (or “contract graph”) from module tags
* [ ] Add `codeintel build contracts dump --format json`
* [ ] Add `codeintel build contracts validate` (strict mode)

### Tests

* [ ] `test_pr39_contract_scan_finds_targets.py`
* [ ] `test_pr39_contract_validate_reports_no_errors.py`

### CLI snapshots

* **Command**

  ```bash
  codeintel build contracts dump --format json
  ```

  **Snapshot**: `pr39_contracts_dump.json`

---

## PR‑40 — Graph exports 2.0: asset graph + version graph

### Tasks

* [ ] Extend `build graph` to support:

  * `--kind targets|assets|versions`
* [ ] Asset graph nodes = assets, edges = lineage
* [ ] Version graph nodes = asset_versions, edges = asset_lineage
* [ ] Include tags for readability (build on existing graph export functionality). 

### Tests

* [ ] `test_pr40_asset_graph_contains_expected_edges.py`
* [ ] `test_pr40_version_graph_export_mermaid.py`

### CLI snapshots

* **Command**

  ```bash
  codeintel build graph --kind assets --format mermaid --asset analytics.goid_risk_factors
  ```

  **Snapshot**: `pr40_graph_assets_risk_factors.mmd`

---

## PR‑41 — PR/CI report generation (markdown + JSON)

### Tasks

* [ ] Add `codeintel build report --base <commit> --head <commit>`
* [ ] Output includes:

  * plan summary
  * impacted assets
  * key diffs (optional list)
  * links to graph exports (if written to files)
* [ ] Write both JSON and Markdown formats

### Tests

* [ ] `test_pr41_report_markdown_is_deterministic.py`
* [ ] `test_pr41_report_json_schema.py`

### CLI snapshots

* **Command**

  ```bash
  codeintel build report --format text --base COMMIT_A --head COMMIT_B
  ```

  **Snapshot**: `pr41_report_markdown.txt`

---

## PR‑42 — Backfill orchestration + time series tables

### Tasks

* [ ] Add `codeintel build backfill --from <commit> --to <commit> --targets ...`
* [ ] Add `build.backfills` and `build.backfill_runs` tables
* [ ] Ensure backfills persist:

  * run IDs
  * success/fail
  * produced asset versions per commit

### Tests

* [ ] `test_pr42_backfill_creates_records.py`
* [ ] `test_pr42_backfill_respects_force_and_base.py`

### CLI snapshots

* **Command**

  ```bash
  codeintel build backfill --help
  ```

  **Snapshot**: `pr42_backfill_help.txt`

---

## PR‑43 — Reproducibility: run environment capture

### Tasks

* [ ] Add `build.run_env` table:

  * python version, package lock hash, tool versions, platform, config hash
* [ ] Persist env at run start and completion
* [ ] Extend `build history --run-id` to include env (optional new CLI `build run-info`)

### Tests

* [ ] `test_pr43_run_env_written.py`
* [ ] `test_pr43_run_info_includes_env.py`

### CLI snapshots

* **Command**

  ```bash
  codeintel build history --run-id hamilton-test-0001 --format json
  ```

  **Snapshot**: `pr43_history_includes_run_env.json`

---

## PR‑44 — Quality gates 2.0: invariants + blocking policies

You already have optional Pandera validation (`--validate-outputs`). 
Phase 4 turns “quality” into a first-class system.

### Tasks

* [ ] Add invariants DSL (uniqueness, non-null, referential, distribution)
* [ ] Persist results in `build.asset_quality`
* [ ] Planner can apply policies:

  * warn vs fail
  * block downstream if invariant fails (preserves upstream gating behavior) 

### Tests

* [ ] `test_pr44_invariant_failure_marks_target_failed.py`
* [ ] `test_pr44_downstream_blocked_on_quality_fail.py`

### CLI snapshots

* **Command**

  ```bash
  codeintel build validate --format json --asset analytics.goid_risk_factors --alias latest
  ```

  **Snapshot**: `pr44_validate_risk_factors_latest.json`

---

## PR‑45 — Consolidation: unify DatasetRef/ArtifactRef under AssetRef + docs

### Tasks

* [ ] Introduce `AssetRef` as the canonical output reference
* [ ] Keep compatibility:

  * DatasetRef/ArtifactRef can be derived views of AssetRef
* [ ] Update planner/explain outputs to include:

  * asset keys
  * version hashes (when available)
* [ ] Documentation refresh + migration notes

### Tests

* [ ] `test_pr45_assetref_backcompat_datasetref.py`
* [ ] `test_pr45_plan_includes_asset_metadata.py`

### CLI snapshots

* **Command**

  ```bash
  codeintel build plan risk_factors --format json
  ```

  **Snapshot**: `pr45_plan_includes_asset_versions.json`

---

# Optional: Phase 4 “golden” snapshot manifest additions pattern

You can follow the same snapshot framework structure already in your repo (`tests/build/hamilton/snapshots/…`, YAML manifest, dynamic key normalization, tag/pattern filters).  

Example case entry pattern:

```yaml
- name: "pr33_diff_risk_factors_main_vs_latest"
  tags: ["pr33", "diff", "json", "tiny"]
  args: ["build", "diff", "--format", "json", "--asset", "analytics.goid_risk_factors", "--from", "main", "--to", "latest"]
  snapshot: "pr33_diff_risk_factors_main_vs_latest.json"
```

---

If you want the next step, I can also provide a **Phase 4 “Definition of Done” checklist** (like Phase 2) that explicitly verifies: asset versions written on every run, lineage edges complete, `impact`/`inherit` semantics correct, diff outputs stable, and promotions resolvable—so you can gate Phase 4 progress cleanly in CI.


# Phase 4 definition of done checklist #

Below is a **Phase 4 “Definition of Done” (DoD) checklist** you can use as a **hard CI gate**. It’s written to layer cleanly on top of what you already have from Phase 2/3: Hamilton as the source of truth, per-target run tracking, DatasetRef v2 snapshot identity, and the manifest-driven CLI golden snapshot framework. 

---

## Phase 4 Definition of Done

### A. Asset catalog foundation is real (tables, invariants, and APIs)

**Goal:** the asset catalog exists as a *first-class* system of record, not an “optional logging addon”.

* [ ] **DuckDB schemas/tables for the asset catalog exist and are auto-created** in the same way as existing build schemas (e.g., how `build.run_targets` was introduced). 
* [ ] **Canonical identifiers are stable and deterministic**

  * [ ] Dataset assets use a stable `asset_key` derived from `DatasetRef.table_key`
  * [ ] Artifact assets use a stable `asset_key` derived from `ArtifactRef` identity
  * [ ] Ordering is deterministic everywhere (sort by `asset_key` / `version_id` / etc.) to keep diffs and CLI output stable.
* [ ] **Asset catalog APIs exist behind the same gateway abstraction** as the rest of `build.*` persistence (so tests can use real components, like Phase 2 did). 
* [ ] **Indexing exists for “hot” queries** (examples):

  * `asset_versions(asset_key, repo, commit)`
  * `lineage_edges(consumer_version_id)`, `lineage_edges(producer_version_id)`
  * `promotions(env, asset_key)`

**CI gates (unit/integration)**

* [ ] A schema test that asserts the catalog tables exist and have the expected columns (mirrors the `build.run_targets` schema-style tests). 

---

### B. Asset versions are written on **every** run (including “all skipped” runs)

**Goal:** after Phase 4, *every run* produces an auditable “asset state” you can diff, promote, and trace.

> You already have run-level and per-target run tracking (`build.run_targets`). Phase 4 extends that into a full asset state log. 

**Hard requirements**

* [ ] **Every computed target that materializes outputs writes at least one asset version row**

  * [ ] One row per produced dataset table (table-level versions), derived from `DatasetRef` identity
  * [ ] One row per produced artifact, derived from `ArtifactRef` identity
* [ ] **Every skipped target still produces a resolvable asset version association**

  * Acceptable implementation patterns:

    * (Preferred) Write a `run_asset_state` / `run_asset_versions` table mapping `run_id -> (asset_key, version_id, resolution_kind=reused)`
    * Or record “reused version id” rows in `asset_versions` with a `reused_from_version_id` pointer
* [ ] **Every run can be “reconstructed”**

  * Given `run_id`, you can list the *exact* asset versions used/produced in that run (even if zero targets computed due to cache/skip).
* [ ] **Snapshot identity is preserved**

  * Asset versions must retain `(repo, commit)` (directly or indirectly), consistent with DatasetRef v2’s purpose of full snapshot identity. 

**CI gates**

* [ ] Integration test: run a small closure twice

  * Run 1: assets produced → versions written
  * Run 2: everything skips → still emits a resolvable “asset state” for the new run_id
* [ ] A “nothing computed” test case must still produce a valid asset-state record.

---

### C. Lineage edges are **complete** and queryable (dataset + artifact lineage)

**Goal:** “best in class” lineage means you can answer: *what produced this? what did it depend on? what else breaks if it changes?*

**Hard requirements**

* [ ] **Lineage is recorded at the version level**

  * For each produced **asset version**, record edges to the **specific upstream asset versions** it used.
* [ ] **Lineage includes datasets and artifacts**

  * Dataset dependencies come from loader nodes / DatasetRefs (which already carry repo/commit for lineage correctness). 
  * Artifact dependencies come from ArtifactRefs (also snapshot-identified). 
* [ ] **Completeness rule**

  * For every materialized asset version `V_out`, the set of upstream version edges equals the set of dependencies visible in the Hamilton plan/closure for that target (modulo “pure config inputs” if you model those separately).
* [ ] **No orphan outputs**

  * Every asset version must be attributable to:

    * a `run_id`
    * a `target` (producer)
    * and has lineage edges (unless it is a declared “source asset”)
* [ ] **Lineage traversal is deterministic**

  * Query results must be stable ordering for CLI/diff stability.

**CI gates**

* [ ] Unit test: creating lineage edges from a fabricated `TargetRunRecord` produces the expected edge set.
* [ ] Integration test: run a known tiny closure and assert:

  * number of produced asset versions
  * number of lineage edges
  * exact adjacency for a chosen downstream asset

---

### D. Impact + inherit semantics are correct (and provably so)

This is the section that makes Phase 4 “feel” like a real system: **change management**.

#### D1. Impact semantics

**Definition (recommended):**

* *Impact(asset_version X)* returns the set of **downstream asset keys** (or targets) that are transitively dependent on X through lineage edges.
* Optional refinements:

  * include/exclude “same-target variants”
  * include “promotion baseline” comparison (impact relative to env state)

**DoD checks**

* [ ] Impact results are **sound** (no missing downstream assets)
* [ ] Impact results are **deterministic** (stable ordering)
* [ ] Impact results are **explainable**

  * either return at least one witness path per impacted asset, or provide a `--why`/`--path` option

**CI gates**

* [ ] Unit test: small synthetic lineage graph with multiple branches; assert impacted set and at least one path.
* [ ] Snapshot tests: `codeintel build impact ...` output is stable and golden-tested (using your existing CLI snapshot framework). 

#### D2. Inherit semantics (promotion inheritance / resolution)

**Definition (recommended):**

* Promotions define an **environment view** (e.g., `dev`, `staging`, `prod`) mapping `asset_key -> pinned_version_id`.
* **Inherit** means: when promoting a set of assets, upstream dependencies resolve in a deterministic way:

  * Either (Preferred) you promote a *closure bundle* (target + all upstream pinned versions), or
  * You promote only the requested assets, and all upstream assets are inherited from the destination env’s existing pins.

**DoD checks**

* [ ] Promoting a target asset does not silently change unrelated upstream pins unless explicitly requested
* [ ] If “closure promotion” is enabled, the pinned set is complete and resolvable
* [ ] Promotion semantics are identical whether assets are datasets or artifacts

**CI gates**

* [ ] Integration test:

  1. create env `staging` with baseline pins
  2. promote a downstream asset version with inheritance semantics
  3. resolve the downstream asset in `staging`
  4. verify upstream versions are either unchanged (inherit-from-env) or updated (closure mode), exactly per policy

---

### E. Diff outputs are stable, meaningful, and version-aware

**Goal:** diffs are what turn asset catalog + versioning into developer velocity.

**DoD checks**

* [ ] You can diff:

  * [ ] two asset versions (`asset_key` + `version_a` vs `version_b`)
  * [ ] two environments (`env_a` vs `env_b`) as an asset set diff
  * [ ] two runs (`run_a` vs `run_b`) as an asset set diff
* [ ] Diff output is **stable**:

  * deterministic ordering
  * stable formatting
  * dynamic fields (timestamps, run ids) normalized or excluded in CLI goldens
* [ ] Diff output is **actionable**:

  * schema diff (columns added/removed/changed) for dataset assets
  * row-count / key-count summary deltas (at minimum)
  * optional: sample of changed keys if you have stable primary keys

**CI gates**

* [ ] Unit tests for diff computation logic on synthetic metadata
* [ ] CLI golden snapshot tests for at least:

  * `diff asset`
  * `diff env`
  * `diff run`

This should directly reuse your manifest-driven snapshot infra and normalization rules. 

---

### F. Promotions are resolvable end-to-end (and usable by the planner/executor)

**Goal:** promotions are not just “writing a row”; they change how the system can be executed and reproduced.

**DoD checks**

* [ ] `promote` writes promotion state for an env
* [ ] `resolve` can answer: **what version is active for (env, asset_key)?**
* [ ] You can produce an **environment lock** (“resolved bundle”) deterministically

  * recommended: a CLI output that emits a fully-resolved list of pins (asset_key → version_id)
* [ ] The **planner can optionally plan against an env baseline**

  * i.e., “diff against what prod currently serves”
  * This becomes extremely powerful when combined with `explain`/`plan` (which you already have). 

**CI gates**

* [ ] Integration test: create two envs, promote different versions, resolve both, and ensure results differ as expected.
* [ ] CLI golden snapshots:

  * `env list`
  * `env show <env>`
  * `resolve <asset_key> --env <env>`
  * `promote ...` (at least verify it prints deterministic summary)

---

### G. CLI + UX completeness (best-in-class polish)

Since Phase 2 already established the CLI snapshot approach and tags, Phase 4 DoD should require **stable, snapshot-tested UX** for all new commands. 

**DoD checks**

* [ ] All Phase 4 CLI commands have:

  * `--help` stable and snapshot-tested
  * JSON output mode for machine consumption (when relevant)
  * human-readable mode with deterministic ordering
* [ ] All outputs are compatible with snapshot normalization (no uncontrolled randomness / timestamps dumped without being normalizable)

**CI gates**

* [ ] Snapshot suite includes Phase 4 tags (e.g., `phase4`, `pr28`…`pr45`, plus `assets`, `lineage`, `diff`, `promote`)
* [ ] CI runs snapshot tests with tag filtering (mirrors Phase 2 patterns)

---

### H. “Phase 4 CI Gate” is explicit and enforceable

This is the meta-DoD: you should be able to **turn Phase 4 requirements into one CI job**.

**DoD checks**

* [ ] There exists a single CI target (script or make task) that runs:

  * unit tests for asset/version/lineage/diff/promotion
  * at least one integration “golden run” covering:

    * compute
    * skip
    * promote
    * resolve
    * diff
  * CLI golden snapshot tests for Phase 4 tags (manifest-driven)

This builds directly on the Phase 2 snapshot framework you already have. 

---

## Recommended “minimum sufficient” Phase 4 gate suite

If you want the most leverage with the least test weight, make Phase 4 DoD *require* these:

1. **One integration test**: “run → run again (skip) → promote → resolve → diff”
2. **One lineage correctness test**: asserts exact edges for a tiny known closure
3. **One impact correctness test**: asserts impacted set + at least one witness path
4. **One snapshot manifest group** for Phase 4 CLI:

   * `assets list/show`
   * `lineage show/export`
   * `impact`
   * `diff`
   * `env/promote/resolve`

---

## Why this DoD matches your current foundation

* You already have **full snapshot identity for lineage correctness** via DatasetRef v2 `(repo, commit)`, so Phase 4 can safely model versions and cross-run tracking without hacks. 
* You already persist **per-target execution records** (`build.run_targets`), which is the natural bridge from “target-centric builds” to “asset-centric catalogs”. 
* You already have the **manifest-driven CLI golden snapshot framework**, so “diff outputs stable” and “promotion UX stable” can be hard-gated in CI (instead of being subjective). 

---

If you want, I can also translate this DoD into a **single `tests/build/hamilton/test_phase4_dod_gate.py`** skeleton that:

* provisions an ephemeral DuckDB,
* executes the minimal closure,
* asserts the exact catalog/version/lineage invariants,
* then runs a small set of CLI snapshot cases tagged `phase4`.

(You wouldn’t have to decide the final schema first; the gate can be written to assert invariants you *must* satisfy, while allowing schema evolution.)


# phase 4 dod gate skeleton #

Below is a **single Phase 4 DoD gate test skeleton** you can drop in as:

`tests/build/hamilton/test_phase4_dod_gate.py`

It is intentionally designed to:

1. provision an **ephemeral DuckDB**,
2. execute a **minimal Hamilton closure** using the same Phase‑1/2 execution semantics (closure execution + upstream gating + run tracking) ,
3. assert **asset catalog/version/lineage invariants** (without hardcoding the final schema beyond a small “required tables” set), and
4. run a subset of your **CLI golden snapshot cases** tagged `phase4` using your manifest-driven snapshot runner pattern .

It also aligns with the Phase 3/4 direction of *explicit materialization boundaries* (compute vs materialize) so the invariants you enforce here remain valid as targets become more Hamilton-native. 

---

## `tests/build/hamilton/test_phase4_dod_gate.py` (skeleton)

```python
from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import pytest

# DuckDB is expected to be available in your repo (it already underpins the build system).
import duckdb

# Reuse the manifest-driven CLI snapshot framework you already implemented in Phase 2.
# (paths based on Phase 2 report structure)
from tests.build.hamilton.snapshots._manifest import load_snapshot_manifest
from tests.build.hamilton.snapshots._runner import execute_and_assert_snapshot


# -----------------------------------------------------------------------------
# Configuration knobs (safe defaults; override in CI/env as needed)
# -----------------------------------------------------------------------------

DEFAULT_GATE_GOALS = tuple(
    x.strip()
    for x in os.getenv("CODEINTEL_PHASE4_GATE_GOALS", "risk_factors").split(",")
    if x.strip()
)
DEFAULT_PHASE4_SNAPSHOT_TAGS = tuple(
    x.strip()
    for x in os.getenv("CODEINTEL_PHASE4_SNAPSHOT_TAGS", "phase4").split(",")
    if x.strip()
)

# These env vars are placeholders — set to whatever your runtime uses to locate the DuckDB.
# If your CLI/runtime already accepts a config file / profile argument, you can set that instead.
ENV_DB_PATH_KEYS = (
    "CODEINTEL_DUCKDB_PATH",
    "CODEINTEL_DB_PATH",
    "CODEINTEL_STORAGE_PATH",
)

# Optional: make this test stricter as Phase 4 progresses
STRICT = os.getenv("CODEINTEL_PHASE4_GATE_STRICT", "1") not in ("0", "false", "False")


# -----------------------------------------------------------------------------
# Small DB helpers (schema-flexible)
# -----------------------------------------------------------------------------

def _sql_ident(schema: str, table: str) -> str:
    # DuckDB uses "schema"."table" quoting; keep it simple.
    return f'"{schema}"."{table}"'


def assert_table_exists(con: duckdb.DuckDBPyConnection, schema: str, table: str) -> None:
    row = con.execute(
        """
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = ? AND table_name = ?
        """,
        [schema, table],
    ).fetchone()
    assert row is not None, f"Missing required table: {schema}.{table}"


def assert_column_exists(
    con: duckdb.DuckDBPyConnection,
    schema: str,
    table: str,
    column: str,
) -> None:
    row = con.execute(
        """
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = ? AND table_name = ? AND column_name = ?
        """,
        [schema, table, column],
    ).fetchone()
    assert row is not None, f"Missing required column: {schema}.{table}.{column}"


def fetch_count(con: duckdb.DuckDBPyConnection, schema: str, table: str, where_sql: str = "", args: list[Any] | None = None) -> int:
    args = args or []
    q = f"SELECT COUNT(*) FROM {_sql_ident(schema, table)}"
    if where_sql:
        q += f" WHERE {where_sql}"
    return int(con.execute(q, args).fetchone()[0])


# -----------------------------------------------------------------------------
# Phase 4 invariants (schema-flexible, but *real*)
# -----------------------------------------------------------------------------

def assert_phase4_asset_catalog_minimum(
    con: duckdb.DuckDBPyConnection,
    *,
    run_id: str,
) -> None:
    """
    Asserts the minimum Phase 4 catalog invariants without hardcoding the full schema.

    Tighten these over time as your schema stabilizes.
    """
    # Required tables (Day-one asset catalog core)
    for t in ("assets", "asset_versions", "asset_lineage", "asset_aliases"):
        assert_table_exists(con, "build", t)

    # Required columns (minimal usable surface)
    assert_column_exists(con, "build", "asset_versions", "asset_key")
    assert_column_exists(con, "build", "asset_versions", "asset_kind")
    assert_column_exists(con, "build", "asset_versions", "version_hash")
    assert_column_exists(con, "build", "asset_versions", "status")
    assert_column_exists(con, "build", "asset_versions", "created_at")

    # Run linkage is strongly recommended (so a run can be reconstructed)
    # If you decide to model run->asset-state in a different table, adjust here.
    assert_column_exists(con, "build", "asset_versions", "run_id")

    # There must be at least one asset version written for this run.
    n_versions = fetch_count(con, "build", "asset_versions", where_sql="run_id = ?", args=[run_id])
    assert n_versions > 0, f"Expected asset_versions rows for run_id={run_id}, found 0"

    # At least some lineage should exist once you’re producing derived assets.
    # If your chosen gate goals are purely “source assets”, this may be 0 initially;
    # but in strict mode we want to ensure derived lineage works.
    n_edges = fetch_count(con, "build", "asset_lineage")
    if STRICT:
        assert n_edges > 0, "Expected at least one asset_lineage edge in STRICT mode"

    # Promotions must be structurally possible (aliases table exists).
    # In strict mode you can also require at least one alias row after you add a promote step.
    if STRICT:
        # Allow 0 for now (until promote/resolve commands are wired in the gate run).
        _ = fetch_count(con, "build", "asset_aliases")


def assert_phase4_referential_sanity(con: duckdb.DuckDBPyConnection) -> None:
    """
    Optional referential sanity check:
    Every lineage edge should point to existing asset_versions on both ends.
    """
    # If your lineage table references logical keys only (not version hashes), adapt this query.
    # This is written for the "version-level edges" schema from the Phase 4 proposal.
    # If columns differ, keep the invariant but update names.
    required_cols = [
        ("asset_lineage", "downstream_kind"),
        ("asset_lineage", "downstream_key"),
        ("asset_lineage", "downstream_version"),
        ("asset_lineage", "upstream_kind"),
        ("asset_lineage", "upstream_key"),
        ("asset_lineage", "upstream_version"),
    ]
    for table, col in required_cols:
        assert_column_exists(con, "build", table, col)

    # If there are no edges yet, don't fail here (Phase 4.0 may allow this temporarily).
    n_edges = fetch_count(con, "build", "asset_lineage")
    if n_edges == 0:
        return

    q = f"""
    SELECT COUNT(*)
    FROM {_sql_ident("build","asset_lineage")} e
    LEFT JOIN {_sql_ident("build","asset_versions")} dv
      ON dv.asset_kind = e.downstream_kind
     AND dv.asset_key = e.downstream_key
     AND dv.version_hash = e.downstream_version
    LEFT JOIN {_sql_ident("build","asset_versions")} uv
      ON uv.asset_kind = e.upstream_kind
     AND uv.asset_key = e.upstream_key
     AND uv.version_hash = e.upstream_version
    WHERE dv.version_hash IS NULL OR uv.version_hash IS NULL
    """
    broken = int(con.execute(q).fetchone()[0])
    assert broken == 0, f"Found {broken} lineage edges pointing to missing asset_versions"


# -----------------------------------------------------------------------------
# Build execution harness (intentionally “adapter friendly”)
# -----------------------------------------------------------------------------

def _make_env_overrides_for_db(db_path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    # Populate multiple keys so this is drop-in regardless of which one your runtime uses.
    for k in ENV_DB_PATH_KEYS:
        env[k] = str(db_path)
    # Keep test output quieter and deterministic
    env["CODEINTEL_LOG_LEVEL"] = "WARNING"
    env["CODEINTEL_TEST_MODE"] = "1"
    return env


def run_minimal_hamilton_closure(
    *,
    db_path: Path,
    goals: tuple[str, ...],
    extra_env: dict[str, str],
) -> str:
    """
    Executes a minimal closure and returns run_id.

    This is intentionally implemented as a small shim:
    - If you already have test fixtures for gateway/runtime/env, use them here.
    - Otherwise, wire this to your HamiltonBuildExecutor directly.

    IMPORTANT: This function should actually run the build so that catalog tables are populated.
    """
    # Option A (preferred): run the build using your internal executor directly.
    # This keeps the test hermetic and bypasses CLI formatting variance.
    #
    # NOTE: Replace the TODO imports/constructors with the real ones from your codebase.
    # The Phase 1/2 reports confirm these objects exist and are used in production paths. :contentReference[oaicite:3]{index=3} :contentReference[oaicite:4]{index=4}

    from codeintel.build.hamilton.executor import HamiltonBuildExecutor
    from codeintel.build.hamilton.env import BuildEnv
    from codeintel.storage.gateway import StorageGateway
    from codeintel.storage.snapshot import SnapshotRef
    from codeintel.build.paths import BuildPaths
    from codeintel.build.providers import Providers
    from codeintel.build.config import BuildConfig

    # Ensure directories exist
    workdir = db_path.parent
    repo_root = workdir / "repo_root"
    repo_root.mkdir(parents=True, exist_ok=True)

    # Create gateway bound to the ephemeral DuckDB.
    # TODO: adjust constructor to match your StorageGateway API.
    gateway = StorageGateway.open_duckdb(db_path=str(db_path))

    snapshot = SnapshotRef(repo="test/repo", commit="phase4-gate", repo_root=str(repo_root))

    paths = BuildPaths(
        build_dir=str(workdir / "build"),
        scip_dir=str(workdir / "scip"),
        document_output_dir=str(workdir / "out"),
    )

    providers = Providers.default()
    config = BuildConfig.default()

    # Prefetch manifest index (Phase 2 pattern).
    manifests = gateway.build.list_manifests(repo=snapshot.repo, commit=snapshot.commit)
    manifest_index = {m.target: m for m in manifests}

    env = BuildEnv(
        gateway=gateway,
        snapshot=snapshot,
        paths=paths,
        providers=providers,
        config=config,
        profile="default",
        force_targets=frozenset(),
        manifest_index=manifest_index,
        validate_outputs=False,
    )

    executor = HamiltonBuildExecutor(profile="default", mode="generated")
    result = executor.run(env=env, targets=list(goals))

    assert result.run_id, "Expected non-empty run_id from Hamilton build result"
    assert result.success is True, f"Build failed unexpectedly: {result.error}"
    return result.run_id


# -----------------------------------------------------------------------------
# CLI snapshot execution helper
# -----------------------------------------------------------------------------

def run_phase4_cli_snapshots(
    *,
    request: pytest.FixtureRequest,
    extra_env: dict[str, str],
) -> None:
    """
    Runs a subset of CLI snapshot cases tagged `phase4`.

    This piggybacks on your existing manifest-driven snapshot runner. :contentReference[oaicite:5]{index=5}
    """
    update = bool(request.config.getoption("--update-cli-snapshots"))

    snapshots_dir = Path(__file__).parent / "snapshots"
    manifest_yaml = snapshots_dir / "manifest.yaml"
    manifest_json = snapshots_dir / "manifest.json"
    manifest_path = manifest_yaml if manifest_yaml.exists() else manifest_json

    manifest = load_snapshot_manifest(manifest_path)
    required_tags = set(DEFAULT_PHASE4_SNAPSHOT_TAGS)

    selected = [c for c in manifest.cases if required_tags.intersection(set(c.tags))]
    assert selected, (
        f"No CLI snapshot cases found with tags {sorted(required_tags)}. "
        f"Add at least one Phase 4 case to {manifest_path.name} (tag it 'phase4')."
    )

    for case in selected:
        merged_env = dict(extra_env)
        if case.env:
            merged_env.update(case.env)

        # SnapshotCase is frozen; create a modified case with merged env.
        case2 = replace(case, env=merged_env)

        execute_and_assert_snapshot(
            manifest=manifest,
            snapshots_dir=snapshots_dir,
            case=case2,
            update=update,
        )


# -----------------------------------------------------------------------------
# The actual Phase 4 DoD gate test
# -----------------------------------------------------------------------------

@pytest.mark.phase4_dod_gate
def test_phase4_dod_gate(tmp_path: Path, request: pytest.FixtureRequest) -> None:
    """
    Phase 4 DoD gate:
      1) Run minimal closure on ephemeral DuckDB
      2) Assert asset catalog/version/lineage invariants
      3) Run Phase 4 CLI goldens (tagged 'phase4')
    """
    db_path = tmp_path / "phase4_gate.duckdb"
    env_overrides = _make_env_overrides_for_db(db_path)

    # Step 1: execute minimal closure
    run_id = run_minimal_hamilton_closure(
        db_path=db_path,
        goals=DEFAULT_GATE_GOALS,
        extra_env=env_overrides,
    )

    # Step 2: assert catalog/version/lineage invariants (schema-flexible)
    con = duckdb.connect(str(db_path))
    try:
        assert_phase4_asset_catalog_minimum(con, run_id=run_id)
        assert_phase4_referential_sanity(con)
    finally:
        con.close()

    # Step 3: run CLI snapshot cases tagged 'phase4'
    run_phase4_cli_snapshots(request=request, extra_env=env_overrides)
```

---

## How to wire this into CI (minimal steps)

1. Add the file above.
2. Add a marker entry (optional but nice) in your `pytest.ini`:

   ```ini
   markers =
     phase4_dod_gate: Phase 4 end-to-end DoD gate
   ```
3. Add at least **one** CLI snapshot case tagged `phase4` in `tests/build/hamilton/snapshots/manifest.yaml` (you already have this YAML-based system). 
   Example:

   ```yaml
   - name: "phase4_assets_help"
     tags: ["phase4", "assets", "text", "tiny"]
     kind: "text"
     args: ["build", "assets", "--help"]
     snapshot: "phase4_assets_help.txt"
   ```

Then run locally:

```bash
pytest -m phase4_dod_gate
```

And in CI, you can pin it as its own job:

```bash
pytest -m phase4_dod_gate -q
```

---

## Notes on the few TODOs in the skeleton

The only “repo-specific” part is the `StorageGateway.open_duckdb(...)`, `SnapshotRef(...)`, `BuildPaths(...)`, `Providers.default()`, and `BuildConfig.default()` constructors inside `run_minimal_hamilton_closure()`. The skeleton uses names consistent with how Phase 1/2 describe the runtime wiring and env structure.  

If your constructors are named slightly differently, you’ll only need to adjust that one section.

---

If you want, I can also provide a **Phase 4 “minimal seed harness”** (a helper that seeds just enough upstream tables into the ephemeral DuckDB so `risk_factors` can run quickly and deterministically), which makes this DoD gate both fast and very reliable.


# Phase 4 minimal seed harness #

Below is a **drop‑in “Phase 4 minimal seed harness”** that does two things:

1. **Seeds just enough upstream datasets** in an ephemeral DuckDB so the `risk_factors` plugin can run deterministically (no external tools, no big ingests).

2. **Pre-seeds output manifests for the entire dependency closure** of `risk_factors` so Hamilton will **skip** every expensive upstream target and only compute `risk_factors`.

This is intentionally designed to be **stable under schema evolution** (it introspects DuckDB to fill NOT NULL columns with safe defaults), while still letting you explicitly control the values that determine `risk_score`/`risk_level`.

---

## Drop-in helper module

### `tests/build/hamilton/_phase4_seed_harness.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Mapping

from codeintel.build.config import BuildConfig
from codeintel.build.hashing import compute_input_hash, compute_options_hash
from codeintel.build.manifest import OutputManifest
from codeintel.build.registry import get_target_graph
from codeintel.config.primitives import SnapshotRef
from codeintel.storage.gateway.protocol import StorageGateway


# -----------------------------
# Seed model
# -----------------------------

@dataclass(frozen=True)
class RiskFactorsSeed:
    """Minimal, deterministic inputs for risk_factors.

    The defaults intentionally hit:
      complexity_bucket='high' + static_diagnostics.has_errors=True -> risk_score=0.9 -> risk_level='high'
    """
    repo: str = "demo/repo"
    commit: str = "deadbeef"

    # identity
    rel_path: str = "src/app.py"
    function_goid_h128: int = 123456789
    urn: str = "goid:demo/repo#python:function:src.app.my_func"
    language: str = "python"
    kind: str = "function"
    qualname: str = "app.my_func"
    start_line: int = 10
    end_line: int = 40

    # function_metrics knobs
    loc: int = 30
    logical_loc: int = 20
    cyclomatic_complexity: int = 17
    complexity_bucket: str = "high"  # drives risk branch #1

    # coverage_functions knobs
    executable_lines: int = 50
    covered_lines: int = 10
    coverage_ratio: float = 0.2
    tested: bool = True

    # hotspots knobs (used in later branches; kept deterministic)
    hotspot_score: float = 0.90

    # typedness knobs
    type_error_count: int = 3
    annotation_ratio_json: str = '{"params": 0.5, "returns": 0.0}'
    untyped_defs: int = 1
    overlay_needed: bool = False

    # static_diagnostics knobs (drives risk branch #1)
    pyrefly_errors: int = 0
    pyright_errors: int = 2
    ruff_errors: int = 1

    # test_coverage_edges knobs
    test_id: str = "tests/test_app.py::test_my_func"
    test_goid_h128: int = 999999
    last_test_status: str = "passed"


# -----------------------------
# Small DuckDB seeding helpers
# -----------------------------

def _default_for_duckdb_type(type_str: str, *, now: datetime) -> object:
    """Return a safe default for NOT NULL columns we didn't explicitly seed."""
    t = type_str.upper()

    if "VARCHAR" in t or "CHAR" in t or "TEXT" in t:
        return ""
    if t.startswith(("DECIMAL", "NUMERIC")):
        return 0
    if t in {
        "INTEGER",
        "BIGINT",
        "SMALLINT",
        "TINYINT",
        "UBIGINT",
        "UINTEGER",
        "USMALLINT",
    }:
        return 0
    if t in {"DOUBLE", "FLOAT", "REAL"}:
        return 0.0
    if t == "BOOLEAN":
        return False
    if t == "JSON":
        return "{}"
    if "TIMESTAMP" in t or t == "DATE":
        return now

    raise ValueError(f"No default mapping for NOT NULL DuckDB type: {type_str}")


def _pragma_table_info(con, table_key: str) -> list[tuple]:
    # DuckDB: PRAGMA table_info('schema.table') returns:
    # (cid, name, type, notnull, dflt_value, pk)
    return con.execute(f"PRAGMA table_info('{table_key}')").fetchall()


def insert_row_partial(
    gateway: StorageGateway,
    table_key: str,
    row: Mapping[str, object],
    *,
    now: datetime,
) -> None:
    """Insert a partially specified row, auto-filling NOT NULL columns.

    This makes the seed resilient to schema changes: if a new NOT NULL column is added,
    the seed continues to work without immediately requiring edits (it will get a safe default).
    """
    con = gateway.con
    info = _pragma_table_info(con, table_key)

    # Build column ordering and constraints from DuckDB
    col_order: list[str] = []
    col_type: dict[str, str] = {}
    col_notnull: dict[str, bool] = {}

    for (_cid, name, typ, notnull, _dflt, _pk) in info:
        col_order.append(name)
        col_type[name] = typ
        col_notnull[name] = bool(notnull)

    full = dict(row)

    # Fill missing NOT NULL columns
    for name in col_order:
        if name in full:
            continue
        if col_notnull.get(name, False):
            full[name] = _default_for_duckdb_type(col_type[name], now=now)

    # Emit INSERT with deterministic column order
    insert_cols = [c for c in col_order if c in full]
    placeholders = ", ".join(["?"] * len(insert_cols))
    cols_sql = ", ".join(insert_cols)
    values = [full[c] for c in insert_cols]

    con.execute(
        f"INSERT INTO {table_key} ({cols_sql}) VALUES ({placeholders})",
        values,
    )


# -----------------------------
# Phase 4 minimal seed
# -----------------------------

def seed_risk_factors_minimal(
    gateway: StorageGateway,
    seed: RiskFactorsSeed,
    *,
    now: datetime | None = None,
    reset: bool = True,
) -> None:
    """Seed the minimal upstream tables that RiskFactorsPlugin reads."""
    now = now or datetime(2020, 1, 1, tzinfo=UTC)  # deterministic inputs

    con = gateway.con

    if reset:
        # Clear repo/commit-scoped tables
        con.execute("DELETE FROM analytics.function_metrics WHERE repo = ? AND commit = ?", [seed.repo, seed.commit])
        con.execute("DELETE FROM analytics.function_types WHERE repo = ? AND commit = ?", [seed.repo, seed.commit])
        con.execute("DELETE FROM analytics.coverage_functions WHERE repo = ? AND commit = ?", [seed.repo, seed.commit])
        con.execute("DELETE FROM analytics.typedness WHERE repo = ? AND commit = ?", [seed.repo, seed.commit])
        con.execute("DELETE FROM analytics.static_diagnostics WHERE repo = ? AND commit = ?", [seed.repo, seed.commit])
        con.execute("DELETE FROM analytics.test_coverage_edges WHERE repo = ? AND commit = ?", [seed.repo, seed.commit])
        con.execute("DELETE FROM analytics.goid_risk_factors WHERE repo = ? AND commit = ?", [seed.repo, seed.commit])

        # hotspots has no repo/commit; scope by path
        con.execute("DELETE FROM analytics.hotspots WHERE rel_path = ?", [seed.rel_path])

        # also clear manifests to avoid accidental hash collisions in reuse scenarios
        con.execute("DELETE FROM build.output_manifests WHERE repo = ? AND commit = ?", [seed.repo, seed.commit])

    # --- function_metrics (drives complexity_bucket/cyclomatic/etc) ---
    insert_row_partial(
        gateway,
        "analytics.function_metrics",
        {
            "function_goid_h128": seed.function_goid_h128,
            "urn": seed.urn,
            "repo": seed.repo,
            "commit": seed.commit,
            "rel_path": seed.rel_path,
            "language": seed.language,
            "kind": seed.kind,
            "qualname": seed.qualname,
            "start_line": seed.start_line,
            "end_line": seed.end_line,
            "loc": seed.loc,
            "logical_loc": seed.logical_loc,
            "cyclomatic_complexity": seed.cyclomatic_complexity,
            "complexity_bucket": seed.complexity_bucket,
            "created_at": now,
        },
        now=now,
    )

    # --- function_types (optional but nice: fills typedness_bucket/source columns in output) ---
    insert_row_partial(
        gateway,
        "analytics.function_types",
        {
            "function_goid_h128": seed.function_goid_h128,
            "urn": seed.urn,
            "repo": seed.repo,
            "commit": seed.commit,
            "rel_path": seed.rel_path,
            "language": seed.language,
            "kind": seed.kind,
            "qualname": seed.qualname,
            "typedness_bucket": "partial",
            "typedness_source": "seed",
            "created_at": now,
        },
        now=now,
    )

    # --- coverage_functions (drives coverage_ratio/tested output fields) ---
    insert_row_partial(
        gateway,
        "analytics.coverage_functions",
        {
            "function_goid_h128": seed.function_goid_h128,
            "urn": seed.urn,
            "repo": seed.repo,
            "commit": seed.commit,
            "rel_path": seed.rel_path,
            "language": seed.language,
            "kind": seed.kind,
            "qualname": seed.qualname,
            "executable_lines": seed.executable_lines,
            "covered_lines": seed.covered_lines,
            "coverage_ratio": seed.coverage_ratio,
            "tested": seed.tested,
            "created_at": now,
        },
        now=now,
    )

    # --- hotspots (drives hotspot_score output field, and risk branch #2 if you choose) ---
    insert_row_partial(
        gateway,
        "analytics.hotspots",
        {
            "rel_path": seed.rel_path,
            "commit_count": 10,
            "author_count": 2,
            "lines_added": 120,
            "lines_deleted": 40,
            "complexity": float(seed.cyclomatic_complexity),
            "score": seed.hotspot_score,
        },
        now=now,
    )

    # --- typedness (drives file_typed_ratio output field) ---
    # typedness schema has NOT NULL columns, so supply all deterministically
    insert_row_partial(
        gateway,
        "analytics.typedness",
        {
            "repo": seed.repo,
            "commit": seed.commit,
            "path": seed.rel_path,
            "type_error_count": seed.type_error_count,
            "annotation_ratio": seed.annotation_ratio_json,
            "untyped_defs": seed.untyped_defs,
            "overlay_needed": seed.overlay_needed,
        },
        now=now,
    )

    # --- static_diagnostics (drives has_errors/total_errors; risk branch #1) ---
    total_errors = seed.pyrefly_errors + seed.pyright_errors + seed.ruff_errors
    insert_row_partial(
        gateway,
        "analytics.static_diagnostics",
        {
            "repo": seed.repo,
            "commit": seed.commit,
            "rel_path": seed.rel_path,
            "pyrefly_errors": seed.pyrefly_errors,
            "pyright_errors": seed.pyright_errors,
            "ruff_errors": seed.ruff_errors,
            "total_errors": total_errors,
            "has_errors": total_errors > 0,
        },
        now=now,
    )

    # --- test_coverage_edges (drives test_count/failing_test_count/last_test_status) ---
    insert_row_partial(
        gateway,
        "analytics.test_coverage_edges",
        {
            "test_id": seed.test_id,
            "test_goid_h128": seed.test_goid_h128,
            "function_goid_h128": seed.function_goid_h128,
            "urn": seed.urn,
            "repo": seed.repo,
            "commit": seed.commit,
            "rel_path": seed.rel_path,
            "qualname": seed.qualname,
            "covered_lines": seed.covered_lines,
            "executable_lines": seed.executable_lines,
            "coverage_ratio": seed.coverage_ratio,
            "last_status": seed.last_test_status,
            "created_at": now,
        },
        now=now,
    )


def seed_skip_manifests_for_goal(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    *,
    goal: str = "risk_factors",
    config: BuildConfig | None = None,
    include_goal: bool = False,
    computed_at: datetime | None = None,
) -> dict[str, OutputManifest]:
    """Pre-seed build.output_manifests so Hamilton skips upstream expensive targets.

    This computes input_hashes using the *real* TargetGraph closure and writes manifests
    in dependency order, so `should_skip(...)` returns True during the run.

    Returns a manifest_index mapping you can pass into BuildEnv(manifest_index=...).
    """
    cfg = config or BuildConfig.empty()
    ts = computed_at or datetime(2020, 1, 1, tzinfo=UTC)

    graph = get_target_graph()
    closure = graph.topological_order([goal])

    manifest_index: dict[str, OutputManifest] = {}

    for name in closure:
        if (name == goal) and (not include_goal):
            continue

        target = graph.get(name)

        raw_params = cfg.parameters_for(name)
        options_hash = compute_options_hash(raw_params)

        input_hash = compute_input_hash(
            target=target,
            snapshot=snapshot,
            gateway=gateway,
            options_hash=options_hash,
            manifests=manifest_index,
        )

        manifest = OutputManifest(
            target=name,
            repo=snapshot.repo,
            commit=snapshot.commit,
            plugin=target.plugin,
            computed_at=ts,
            duration_ms=0.0,
            input_hash=input_hash,
            output_hash=None,
            row_count=None,
            options_hash=options_hash,
        )

        gateway.build.save_manifest(manifest)
        manifest_index[name] = manifest

    return manifest_index
```

---

## How to use it in your Phase 4 DoD gate test

Here’s the intended pattern inside `tests/build/hamilton/test_phase4_dod_gate.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import cast

from codeintel.build.config import BuildConfig
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.executor import HamiltonBuildExecutor
from codeintel.config.primitives import SnapshotRef, BuildPaths
from codeintel.storage.gateway.factory import open_memory_gateway

from tests._helpers.fakes.fake_providers import FakeProviders
from tests.build.hamilton._phase4_seed_harness import (
    RiskFactorsSeed,
    seed_risk_factors_minimal,
    seed_skip_manifests_for_goal,
)

def test_phase4_dod_gate_minimal_seed_allows_risk_factors(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    snapshot = SnapshotRef(repo="demo/repo", commit="deadbeef", repo_root=repo_root)
    paths = BuildPaths.from_repo_root(repo_root, build_dir=tmp_path / "build")
    config = BuildConfig.empty()

    gateway = open_memory_gateway(repo=snapshot.repo, commit=snapshot.commit)

    # 1) seed minimal upstream data
    seed = RiskFactorsSeed(repo=snapshot.repo, commit=snapshot.commit)
    seed_risk_factors_minimal(gateway, seed)

    # 2) seed manifests so all upstream targets skip
    manifest_index = seed_skip_manifests_for_goal(
        gateway,
        snapshot,
        goal="risk_factors",
        config=config,
    )

    # 3) run only risk_factors through Hamilton
    providers = cast(object, FakeProviders.defaults())  # satisfies provider attributes used by ctx
    env = BuildEnv(
        gateway=gateway,
        snapshot=snapshot,
        paths=paths,
        providers=providers,
        config=config,
        manifest_index=manifest_index,
    )

    result = HamiltonBuildExecutor().run(env=env, targets=("risk_factors",))
    assert result.success

    # 4) assert deterministic core outputs (ignore created_at)
    rows = gateway.con.execute(
        \"\"\"
        SELECT risk_score, risk_level, test_count, failing_test_count, last_test_status
        FROM analytics.goid_risk_factors
        WHERE repo = ? AND commit = ?
        \"\"\",
        [snapshot.repo, snapshot.commit],
    ).fetchall()

    assert rows == [(0.9, "high", 1, 0, "passed")]
```

---

## Why this harness is “best-in-class” for Phase 4 gating

* **Fast & tool-free:** upstream targets are skipped via real manifest hashing, so no SCIP/typecheck/coverage runners are needed.
* **Deterministic:** you control the branch of the risk scoring logic by setting a few seed fields.
* **Schema-resilient:** `insert_row_partial()` introspects DuckDB and fills missing NOT NULL columns automatically, so small schema evolutions don’t break your DoD gate immediately.
* **Graph-resilient:** `seed_skip_manifests_for_goal()` computes the dependency closure from the *actual* `TargetGraph`, so if Phase 3/4 refactors dependencies, the harness adapts without you rewriting the skip list.

---

## If you want one extra tightening for Phase 4

Once you start writing the asset catalog/version/lineage tables, you can extend the harness with a **single optional hook**:

* `seed_asset_catalog_prereqs(gateway, snapshot, manifest_index)` that inserts “prior versions” for each skipped target/table so lineage edges can reference them even when you skip upstream work.

I didn’t include it above because you explicitly wanted the harness to be usable **before** you lock the final Phase 4 schema—but it’s very easy to add once your catalog tables exist.  
