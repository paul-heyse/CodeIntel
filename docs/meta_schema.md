
You already have ~70% of the “meta schema registry” foundation in place; the missing piece is to **promote it to the sole authoritative plane** and then **delete the in-code duplication**.

Below is a comprehensive implementation plan that (a) unifies all schema “formats” behind a single canonical dataclass, (b) replaces most static schema declarations with **schema derivation + persistence**, and (c) makes internal ops schema-agnostic by moving contract enforcement to **I/O boundaries only**.

I’m basing this plan on your Phase6 architecture description and the current build/hamilton+storage wiring described there. 

---

## 0) Current “schema surface area” (what we will collapse)

### Canonical-ish dataclasses already present

* `codeintel.core.schemas.primitives.{TableSchema, Column, Index, TableWritePolicy}` = minimal schema language.
* `codeintel.core.schemas.contract_primitives.DatasetContract` = dataset metadata + optional `TableSchema` + export knobs.
* `codeintel.core.manifests.SchemaManifest` + `TableProvenance` = already a durable schema+provenance bundle.
* `codeintel.build.schemas.schema_index.SchemaIndex` + `SchemaDerivation` = DAG-first derivation + (optional) Ibis inference + override fallback.

### “Duplication / complication” sources

* Static table schema registry: `codeintel.core.schemas.table_registry.TABLE_SCHEMAS` (+ `output_registry.OUTPUT_TABLE_SCHEMAS`).
* Target compile step hard-resolves schemas early: `build/hamilton/target_spec_compiler.py` imports `core.schemas.table_registry.get_table_schema` and fails strict when missing.
* Pandera schema registry + overlays: `build/hamilton/contracts/schemas/*` (in practice: structure derives from `TableSchema`; checks are overlays).
* Metadata already exists but is under-leveraged:

  * `metadata.canonical_catalogs` already stores a canonical “dataset_contracts” catalog (bootstrap path exists).
  * `metadata.table_schema_registry` now stores `(table_key, schema_digest, schema_hash)` + provenance,
    replacing the old `dataset_schema_registry`.

### Key leverage you already have

* `build/schemas/compile.compile_schema_manifest(...)` produces a deterministic `SchemaManifest` and already records derivation kind/status/errors.
* `build/hamilton/native/export/serving_artifacts.py` already compiles `schema_manifest.json` as a build artifact (i.e., a canonical “meta product” exists—just not made authoritative).

---

## 1) End-state invariants (the “standard”)

### 1.1 One canonical schema dataclass, everything else is a renderer

* **Canonical schema IR for tables = `TableSchema`** (keep it).
* “Different schema type” becomes *a rendering setting*:

  * `TableSchema -> DuckDB DDL` (already via policy backend)
  * `TableSchema -> Pandera DataFrameSchema` (already via `pandera_gen`)
  * `TableSchema -> JSON schema` (already via json schema tooling)
  * (optional) `TableSchema -> pyarrow.Schema` and `-> ibis.Schema` (add thin adapters + optional caching)

**Net effect:** you no longer “store schemas as Pandera objects” or “store schemas as Ibis objects”; you store `TableSchema` (+ provenance), and render on demand.

### 1.2 Boundary-only contracts; internal ops do not declare schemas

* **Only**:

  * ingress boundaries (declared sources) and
  * egress boundaries (materialized tables / exported artifacts / served payloads)
    carry explicit contracts.
* Everything internal is:

  * inferable (Ibis/SQL/plan introspection), and/or
  * validated by lightweight invariants (Pandera checks) *at boundary save*.

### 1.3 Meta DB is the authoritative registry (not python files)

DuckDB `metadata.*` becomes the single authoritative store for:

* **dataset contract catalog** (already via `metadata.canonical_catalogs`)
* **schema manifest catalog(s)** (add)
* **relationalized “current schema + provenance”** (add/extend)
* **schema version store** (add; optional but highly recommended)

Python code becomes:

* minimal boundary declarations (sources + non-inferable outputs),
* inference logic, and
* renderers/validators.

---

## 2) Meta schema (DuckDB) proposal

You already have `metadata.canonical_catalogs` and some supporting tables. Keep those. Extend `metadata` with a *versioned* schema store + a *current pointer* table.

### 2.1 New/extended tables (all under `metadata.*`)

#### A) `metadata.schema_versions` (content-addressed schema store)

Purpose: de-duplicate identical schemas; enable stable diffs and caching.

Columns:

* `schema_digest` **PK** (VARCHAR) — `fingerprint(TableSchema.to_json_obj())`
* `schema_json` (JSON) — full `TableSchema` JSON object
* `schema_hash` (VARCHAR) — legacy/compat hash (`core.schemas.hashing.schema_hash(schema)`)
* `renderer_cache` (JSON, NULL) — optional cache (arrow/ibis/etc payloads)
* `created_at` (TIMESTAMPTZ)

Indexes:

* `idx_schema_versions_schema_hash` on `(schema_hash)`

#### B) `metadata.table_schema_registry` (replaces `metadata.dataset_schema_registry`)

Columns:

* `table_key` **PK** (VARCHAR)
* `schema_digest` (VARCHAR) — FK-like pointer to `schema_versions.schema_digest`
* `schema_hash` (VARCHAR)
* `derivation_kind` (VARCHAR) — `explicit_override|inferred_relation|declared_source|view_inferred`
* `derivation_source` (VARCHAR) — usually target name or `"declared"|"duckdb"`
* `inference_status` (VARCHAR, NULL) — `inferred|override|disabled|error|pending`
* `inference_error` (VARCHAR, NULL)
* `catalog_hash` (VARCHAR, NULL) — hash of the schema manifest entry that produced this row
* `updated_at` (TIMESTAMPTZ)

Indexes:

* `(derivation_kind)`
* `(inference_status)`
* `(catalog_hash)`

#### C) `metadata.schema_manifest_runs`

Purpose: attach a schema-manifest catalog hash to a run (so “what schema set did this run use?” is queryable).

Columns:

* `run_id` **PK** (VARCHAR) — same run id you already use in build tracking
* `repo` (VARCHAR)
* `commit` (VARCHAR)
* `manifest_kind` (VARCHAR) — e.g. `"schema_manifest_v2"`
* `catalog_hash` (VARCHAR)
* `created_at` (TIMESTAMPTZ)

Index:

* `(repo, commit, created_at)`

### 2.2 Canonical catalogs (`metadata.canonical_catalogs`)

Keep as the blob store for:

* `dataset_contracts` (already)
* **add** `schema_manifest_v2`
* (optional but aligned with your serving artifacts) `semantic_registry_v1`, `buildspec_v*`, `environment_v1`

The catalog hash is your stable dedupe key; the relational tables provide query ergonomics and “current pointers”.

---

## 3) Meta operations (code to add)

### 3.1 New storage accessor: `SchemaCatalogTracking`

Create: `src/codeintel/storage/tracking/schema_catalog.py`

Responsibilities:

* `ensure_metadata_schema_tables(con)` (or rely on existing `apply_metadata_ddl`)
* `upsert_schema_manifest(run_id, repo, commit, manifest: SchemaManifest) -> str`

  * compute `catalog_hash=fingerprint(manifest.to_json_obj())`
  * `upsert_canonical_catalog(... catalog_kind="schema_manifest_v2" ...)`
  * write `metadata.schema_manifest_runs`
  * for each table/view:

    * compute `schema_digest`
    * upsert `metadata.schema_versions`
    * upsert `metadata.table_schema_registry` (provenance from `manifest.table_provenance/view_provenance`)
* `load_table_schema(table_key) -> TableSchema | None` (join registry -> versions)
* `prefill_schema_index(schema_index, *, table_keys=None)`

  * loads inferred schemas from `table_schema_registry`+`schema_versions` and calls `SchemaIndex.prefill_cache(...)`
  * (optional) only prefill where derivation_kind == `inferred_relation` and inference_status in {`inferred`,`override`}

### 3.2 Wire into gateway

Modify:

* `src/codeintel/storage/gateway/protocol.py` (add `schemas: SchemaCatalogTracking`)
* `src/codeintel/storage/gateway/accessors.py` (instantiate `self.schemas = SchemaCatalogTracking(self)`)

### 3.3 “Meta sync” command path

Add a CLI entry that forces regeneration/persistence:

* `codeintel meta.sync`:

  * compiles `SchemaManifestRequest(all_targets=True, include_views=True, include_artifacts=True)`
  * persists via `gateway.schemas.upsert_schema_manifest(...)`
  * (optional) persists `dataset_contracts` catalog too (see §4)

This gives you an explicit, deterministic meta refresh primitive independent of running a full build closure.

---

## 4) Boundary contracts plan (what is declared vs inferred)

### 4.1 Contract authoring becomes: “only what cannot be inferred”

Define a clear policy:

**Must remain explicitly declared (P0)**

* Source tables (ingress): anything not produced by DAG.
* Non-inferable outputs: any target output table_key that cannot be inferred from an Ibis compute node (e.g. row/tuple-based ingestion materializations).

**Should become inferred (P1)**

* Any table written from an Ibis expression (`DuckDBIbisTableSaver`) where the compute graph is `q__`-driven and passes your inference constraints.

### 4.2 Where these declarations live

* Keep a minimal declared-source provider (already: `core.schemas.declared/source_declared_schema_provider`)
* Keep a minimal overrides module for non-inferable outputs (you can keep it in `core.schemas.output_registry`, but the end-state should be: *only* non-inferable + metadata tables live there)

### 4.3 Persist dataset contracts from build, stop bootstrapping them in storage

Right now storage can “bootstrap” `dataset_contracts` by importing build-time contract service. That’s convenient but couples layers.

End-state:

* build (or explicit CLI) produces and persists `dataset_contracts` into `metadata.canonical_catalogs`.
* storage gateway **only loads** it; if missing in RW mode, it errors with a message telling you to run `codeintel meta.sync` (or run the “serving_artifacts/meta_catalogs” target).

Implementation:

* Move the “contract catalog build” logic into a build-owned step:

  * create `src/codeintel/build/meta/catalog_compile.py`:

    * `compile_contract_catalog() -> dict[table_key, DatasetContract]` using `get_enriched_contract_service().iter_contracts()`
    * serialize via `contract_to_json_obj`
    * persist to canonical catalogs (`catalog_kind="dataset_contracts"`)
* Update `storage/contracts/bootstrap.py` to become a dev/test-only escape hatch (or delete once migration is complete).

---

## 5) Core refactor to remove schema declarations from internal ops

This is the big “collapse complexity” move: **stop resolving table schemas at target compile time**.

### 5.1 Change `OutputContract` to carry output identity, not resolved `TableSchema`

Today, `OutputContract.tables` is `tuple[TableSchema, ...]` and target compilation resolves via `core.schemas.table_registry.get_table_schema(...)`.

Refactor:

* Introduce `TableOutputDescriptor` in `src/codeintel/build/contracts.py`:

  * `table_key: str`
  * `schema_digest: str | None = None`  (optional pointer; usually filled only at boundary/persistence time)
  * `write_policy: TableWritePolicy | None = None` (optional)
* Change `OutputContract`:

  * `tables: tuple[TableOutputDescriptor, ...]` (instead of `TableSchema`)
  * Keep convenience `table_keys` property.
  * Keep `validate()` logic (duplicates, empty keys, etc.)
* Any code that needs an actual `TableSchema` gets it from `SchemaService` (which may be prefilled from metadata caches).

### 5.2 Update target compilation to stop schema lookup

Modify: `src/codeintel/build/hamilton/target_spec_compiler.py`

* Replace `_resolve_table_schemas(...)` with `_resolve_table_outputs(...)` that just validates table keys and builds `TableOutputDescriptor(table_key=...)`.
* Remove dependency on `core.schemas.table_registry.get_table_schema` entirely from compile stage.
* This eliminates the early “schema must exist in python registry” constraint that forces duplication.

### 5.3 Update schema derivation index construction

Modify: `src/codeintel/build/schemas/schema_index.py::build_schema_index`

* Stop relying on `target.contract.get_table(table_key)` for overrides.
* Instead, resolve overrides from the *declared provider* (or a dedicated overrides provider) for non-inferable outputs.

  * i.e. `override_schema = declared_provider.get_table_schema(table_key)` for non-inferable keys.
* Inferability still computed from `SchemaInferenceService.inferable_table_keys(...)`.

Result: outputs become inferable without any `TableSchema` living in `table_registry`.

### 5.4 Prefill inference cache from metadata before computing manifests/plans

Modify: `src/codeintel/build/target_metadata.py::get_target_metadata_service()`

* After `schema_index = build_schema_index(...)`, add:

  * `env.gateway.schemas.prefill_schema_index(schema_index)` **when an env/gateway exists**
  * If you can’t access gateway here (you currently don’t), do it in:

    * `BuildRunContext.build_env(...load_schema_service...)` or
    * `HamiltonBuildExecutor._build_runtime(...)` right after gateway is available.
* This ensures inference is “db-cached”: schema inference becomes a fallback, not the steady-state.

---

## 6) Integrate into pipelines: where meta updates happen

You have two good integration points; I recommend doing both:

### 6.1 Always persist meta catalogs as part of `serving_artifacts`

Modify: `src/codeintel/build/hamilton/native/export/serving_artifacts.py`

* After `_schema_manifest_json(env)` compiles the manifest, additionally call:

  * `gateway.schemas.upsert_schema_manifest(run_id, repo, commit, manifest)`
* Also persist:

  * contract catalog (`dataset_contracts`) (optional here; but it’s a natural pairing)
  * semantic registry + buildspec as canonical catalogs (optional)

This makes “publishing artifacts” also produce “authoritative meta state”.

### 6.2 Also run meta sync at build start (optional but makes it “standard”)

Modify: `src/codeintel/build/hamilton/executor.py::HamiltonBuildExecutor.run(...)` (or CLI handler)

* Before executing closure, compute *latest* manifest hash and compare to what DB already has:

  * if unchanged: skip
  * else: persist new `schema_manifest_v2` and update `metadata.table_schema_registry`
    This ensures *every build* keeps meta updated even when `serving_artifacts` isn’t selected.

---

## 7) Tests / validation gates (non-negotiable to keep this safe)

Add/extend tests in these areas:

1. **Schema manifest round-trip**

* compile manifest → persist to metadata → load back → equality on `TableSchema.to_json_obj()` for all entries.

2. **SchemaIndex prefill correctness**

* seed DB with inferred schemas, ensure `SchemaIndex.get_table_schema(table_key)` returns without invoking inference (spy inferer).

3. **Target compilation no longer depends on `table_registry`**

* remove a schema entry from `core.schemas.table_registry` for an inferable output and assert `compile_output_targets_from_driver(..., strict=True)` still passes.

4. **Storage gateway no longer bootstraps contracts from build (end-state)**

* open gateway read-only with existing catalogs: success
* open gateway read-only without catalogs: fails with clear error
* open gateway RW without catalogs: either fails (preferred) or requires explicit `meta.sync`.

---

## 8) PR-by-PR implementation plan (concrete file ops)

### PR-01: Add schema catalog persistence primitives

**Create**

* `src/codeintel/storage/tracking/schema_catalog.py` (SchemaCatalogTracking)
* `src/codeintel/storage/metadata/schema_catalog.py` (optional helper layer if you want metadata separation)

**Modify**

* `src/codeintel/storage/metadata/schema.py` (add new table schemas)
* `src/codeintel/storage/metadata/ddl.py` (auto-picks up new tables)
* `src/codeintel/storage/gateway/protocol.py` (add `schemas`)
* `src/codeintel/storage/gateway/accessors.py` (instantiate `self.schemas`)

**No deletions yet.**

---

### PR-02: Persist `SchemaManifest` into DuckDB metadata

**Modify**

* `src/codeintel/build/hamilton/native/export/serving_artifacts.py`

  * after compiling manifest, call `env.gateway.schemas.upsert_schema_manifest(...)`

**Modify (optional)**

* `src/codeintel/storage/metadata/sync.py`

  * add `sync_table_schema_registry_from_latest_manifest(con)` convenience

---

### PR-03: Stop resolving `TableSchema` during target compilation

**Modify**

* `src/codeintel/build/contracts.py`

  * add `TableOutputDescriptor`
  * change `OutputContract.tables` to use descriptors
  * preserve `table_keys` API

**Modify**

* `src/codeintel/build/targets.py` (any helper expecting `TableSchema` objects)
* `src/codeintel/build/hamilton/target_spec_compiler.py`

  * remove `core.schemas.table_registry.get_table_schema` usage
  * build table outputs from saver-derived keys only

**Expected deletions**

* none (but you should remove any now-dead `_resolve_table_schemas()` helper).

---

### PR-04: Rebuild SchemaIndex override semantics

**Modify**

* `src/codeintel/build/schemas/schema_index.py`

  * override lookup comes from declared/override provider, not `OutputContract`

**Modify**

* `src/codeintel/build/schemas/provider_unified.py` (if needed to expose declared vs override providers cleanly)

---

### PR-05: Prefill schema inference cache from metadata

**Modify**

* `src/codeintel/build/hamilton/executor.py` (best place: you have `env.gateway` there)

  * before execution, call `env.gateway.schemas.prefill_schema_index(...)` (or equivalent)
* `src/codeintel/build/schemas/service.py` (if you want schema service construction to optionally prefill)

---

### PR-06: Contract catalog becomes build-owned; storage only loads

**Create**

* `src/codeintel/build/meta/contract_catalog.py` (compile + persist dataset_contracts catalog)

**Modify**

* `src/codeintel/storage/gateway/factory.py`

  * remove automatic “bootstrap from build” path in `_ensure_contract_catalog`
  * replace with “load required; if missing → clear error w/ remediation command”
* `src/codeintel/storage/contracts/bootstrap.py`

  * demote to dev/test-only or delete after migration

---

### PR-07: Delete/trim static schema registries (the payoff)

**Modify**

* `src/codeintel/core/schemas/table_registry.py`

  * remove DAG-output schemas that are inferable and now persisted
  * keep only:

    * metadata tables
    * declared source schemas (if any)
    * non-inferable output schemas (row-based ingestion, etc.)

**Modify**

* `src/codeintel/core/schemas/output_registry.py`

  * same trimming; or collapse into a single “non-inferable overrides” module

**Delete (likely)**

* Any schema generator modules that only existed to keep `table_registry` updated for inferable outputs.

---

## 9) What this buys you immediately (and why it stays simple)

* “Schema format” stops being a proliferation of object types; it’s a renderer from `TableSchema`.
* Schema inference stops being an expensive/fragile runtime activity; it becomes a **one-time compilation** persisted into DuckDB and reused.
* Internal nodes stop requiring schema declarations; only saver/materializers enforce contracts.
* Storage layer stops importing build to synthesize contracts; it just loads canonical catalogs.

If you want, I can also write the exact `TableSchema(...)` declarations for the new `metadata.schema_versions`, `metadata.table_schema_registry`, and `metadata.schema_manifest_runs` tables in the style you use in `storage/metadata/schema.py`, plus the precise upsert SQL for each operation (matching your existing `BuildTracking`/`AssetTracking` patterns).

Below are (A) the **exact `TableSchema(...)` declarations** (compatible with your existing `ColumnType` set — no BLOB), and (B) the **exact upsert/insert SQL** + the matching **`DuckDBPolicyBackend.upsert(...)` calls** in the same style as `AssetTracking`/`BuildTracking`.

---

## A) `src/codeintel/storage/metadata/schema.py` — add these `TableSchema(...)` declarations

Drop these right after `CANONICAL_CATALOGS_TABLE = ...` and before `METADATA_TABLES = ...`, then include them in `METADATA_TABLES`.

```python
# --- NEW: schema registry tables ---------------------------------------------

SCHEMA_VERSIONS_TABLE = TableSchema(
    schema="metadata",
    name="schema_versions",
    columns=[
        # content-addressed key for the full TableSchema JSON (stable fingerprint)
        Column("schema_digest", "VARCHAR", nullable=False),

        # legacy/stable “shape hash” (ordered name:type pairs), used broadly today
        Column("schema_hash", "VARCHAR", nullable=False),

        # canonical schema IR payload (TableSchema.to_json_obj()) as JSON
        Column("schema_json", "JSON", nullable=False),

        # optional future cache bucket for renderer products (arrow/ibis/etc)
        # NOTE: kept JSON because ColumnType does not currently support BLOB.
        Column("renderer_cache", "JSON"),

        Column("created_at", "TIMESTAMPTZ", nullable=False),
    ],
    primary_key=("schema_digest",),
    indexes=(
        Index("idx_schema_versions_schema_hash", ("schema_hash",)),
    ),
)


TABLE_SCHEMA_REGISTRY_TABLE = TableSchema(
    schema="metadata",
    name="table_schema_registry",
    columns=[
        Column("table_key", "VARCHAR", nullable=False),

        # points into metadata.schema_versions
        Column("schema_digest", "VARCHAR", nullable=False),
        Column("schema_hash", "VARCHAR", nullable=False),

        # provenance / derivation metadata (lightweight, query-friendly)
        Column("derivation_kind", "VARCHAR", nullable=False),
        Column("derivation_source", "VARCHAR"),
        Column("inference_status", "VARCHAR"),
        Column("inference_error", "VARCHAR"),

        # ties back to the canonical catalog hash for schema_manifest_v2 (optional but high leverage)
        Column("catalog_hash", "VARCHAR"),

        Column("updated_at", "TIMESTAMPTZ", nullable=False),
    ],
    primary_key=("table_key",),
    indexes=(
        Index("idx_table_schema_registry_schema_digest", ("schema_digest",)),
        Index("idx_table_schema_registry_schema_hash", ("schema_hash",)),
        Index("idx_table_schema_registry_derivation_kind", ("derivation_kind",)),
        Index("idx_table_schema_registry_inference_status", ("inference_status",)),
        Index("idx_table_schema_registry_catalog_hash", ("catalog_hash",)),
    ),
)


SCHEMA_MANIFEST_RUNS_TABLE = TableSchema(
    schema="metadata",
    name="schema_manifest_runs",
    columns=[
        Column("run_id", "VARCHAR", nullable=False),
        Column("repo", "VARCHAR", nullable=False),
        Column("commit", "VARCHAR", nullable=False),

        # e.g. "schema_manifest_v2"
        Column("manifest_kind", "VARCHAR", nullable=False),

        # points into metadata.canonical_catalogs (catalog_kind + catalog_hash)
        Column("catalog_hash", "VARCHAR", nullable=False),

        Column("created_at", "TIMESTAMPTZ", nullable=False),
    ],
    primary_key=("run_id",),
    indexes=(
        Index("idx_schema_manifest_runs_repo_commit", ("repo", "commit", "created_at")),
        Index("idx_schema_manifest_runs_manifest_kind", ("manifest_kind", "created_at")),
        Index("idx_schema_manifest_runs_catalog_hash", ("catalog_hash",)),
    ),
)
```

Now include them in `METADATA_TABLES` (near the top is fine, right after `CANONICAL_CATALOGS_TABLE`):

```python
METADATA_TABLES: tuple[TableSchema, ...] = (
    EXPORT_AUDIT_TABLE,
    CANONICAL_CATALOGS_TABLE,

    # NEW
    SCHEMA_VERSIONS_TABLE,
    TABLE_SCHEMA_REGISTRY_TABLE,
    SCHEMA_MANIFEST_RUNS_TABLE,

    # existing
    ...
)
```

Optional (but consistent): add the new constants to `__all__` if you want direct imports.

---

## B) Upsert operations — exact SQL + “tracking-style” calls

### 1) Insert schema versions (content-addressed; conflict = do nothing)

**Exact SQL (DuckDB):**

```sql
INSERT INTO metadata.schema_versions (
    schema_digest,
    schema_hash,
    schema_json,
    renderer_cache,
    created_at
)
VALUES (?, ?, ?, ?, ?)
ON CONFLICT (schema_digest) DO NOTHING
```

**Matching `DuckDBPolicyBackend.upsert(...)` call (AssetTracking style):**

```python
from codeintel.core.time import utc_now
from codeintel.storage.helpers.json import encode_json_compact
from codeintel.storage.upsert import UpsertSpec

rows = [
    (
        schema_digest,
        schema_hash,
        encode_json_compact(schema_json_obj),
        encode_json_compact(renderer_cache) if renderer_cache is not None else None,
        created_at or utc_now(),
    ),
    # ...
]

gateway.policy.upsert(
    "metadata.schema_versions",
    rows,
    columns=(
        "schema_digest",
        "schema_hash",
        "schema_json",
        "renderer_cache",
        "created_at",
    ),
    upsert=UpsertSpec(
        conflict_columns=("schema_digest",),
        # update_columns=None => DO NOTHING (see DuckDBPolicyBackend._build_upsert)
    ),
)
```

---

### 2) Upsert the “current pointer” registry by `table_key`

**Exact SQL (DuckDB):**

```sql
INSERT INTO metadata.table_schema_registry (
    table_key,
    schema_digest,
    schema_hash,
    derivation_kind,
    derivation_source,
    inference_status,
    inference_error,
    catalog_hash,
    updated_at
)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (table_key) DO UPDATE SET
    schema_digest = excluded.schema_digest,
    schema_hash = excluded.schema_hash,
    derivation_kind = excluded.derivation_kind,
    derivation_source = excluded.derivation_source,
    inference_status = excluded.inference_status,
    inference_error = excluded.inference_error,
    catalog_hash = excluded.catalog_hash,
    updated_at = excluded.updated_at
```

**Matching `DuckDBPolicyBackend.upsert(...)` call:**

```python
from codeintel.core.time import utc_now
from codeintel.storage.upsert import UpsertSpec

rows = [
    (
        table_key,
        schema_digest,
        schema_hash,
        derivation_kind,
        derivation_source,
        inference_status,
        inference_error,
        catalog_hash,
        updated_at or utc_now(),
    ),
    # ...
]

gateway.policy.upsert(
    "metadata.table_schema_registry",
    rows,
    columns=(
        "table_key",
        "schema_digest",
        "schema_hash",
        "derivation_kind",
        "derivation_source",
        "inference_status",
        "inference_error",
        "catalog_hash",
        "updated_at",
    ),
    upsert=UpsertSpec(
        conflict_columns=("table_key",),
        update_columns=(
            "schema_digest",
            "schema_hash",
            "derivation_kind",
            "derivation_source",
            "inference_status",
            "inference_error",
            "catalog_hash",
            "updated_at",
        ),
    ),
)
```

---

### 3) Upsert run → manifest linkage (1 row per `run_id`)

**Exact SQL (DuckDB):**

```sql
INSERT INTO metadata.schema_manifest_runs (
    run_id,
    repo,
    commit,
    manifest_kind,
    catalog_hash,
    created_at
)
VALUES (?, ?, ?, ?, ?, ?)
ON CONFLICT (run_id) DO UPDATE SET
    repo = excluded.repo,
    commit = excluded.commit,
    manifest_kind = excluded.manifest_kind,
    catalog_hash = excluded.catalog_hash,
    created_at = excluded.created_at
```

**Matching `DuckDBPolicyBackend.upsert(...)` call:**

```python
from codeintel.core.time import utc_now
from codeintel.storage.upsert import UpsertSpec

rows = [
    (run_id, repo, commit, manifest_kind, catalog_hash, created_at or utc_now()),
    # ...
]

gateway.policy.upsert(
    "metadata.schema_manifest_runs",
    rows,
    columns=("run_id", "repo", "commit", "manifest_kind", "catalog_hash", "created_at"),
    upsert=UpsertSpec(
        conflict_columns=("run_id",),
        update_columns=("repo", "commit", "manifest_kind", "catalog_hash", "created_at"),
    ),
)
```

---

### Note on “Arrow/Ibis persistence”

Right now your `ColumnType` set does **not** include `BLOB`, so the above uses `renderer_cache: JSON` as the “escape hatch”. If you later decide you *really* want Arrow IPC bytes, the clean move is:

* add `"BLOB"` to `ColumnType`,
* update any DDL / coercion logic if needed (your DDL path is SQLGlot-driven and should be fine),
* switch `renderer_cache` → `arrow_ipc BLOB` (or keep both).

If you want next, I can write the **exact new tracking accessor module** (dataclasses + `SchemaCatalogTracking` class) with these three batch methods, matching your project’s naming/import conventions and returning row counts in the same style as `AssetTracking`.


Below is an **exact new tracking accessor module** in the same pattern as `AssetTracking` / `BuildTracking`: frozen record dataclasses + a `SchemaCatalogTracking` accessor with the three **batch upsert methods** and row-count returns (DuckDBPolicyBackend returns `len(rows)`).

Create:

`src/codeintel/storage/tracking/schema_catalog_tracking.py`

```python
"""Schema catalog tracking persistence for DuckDB.

This module provides persistence for the schema registry plane:

- metadata.schema_versions: content-addressed store of canonical TableSchema JSON
- metadata.table_schema_registry: current-pointer mapping from table_key -> schema_digest (+ provenance)
- metadata.schema_manifest_runs: linkage from run_id -> canonical catalog hash for a schema manifest

All DuckDB access is encapsulated here, following the storage accessor pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from codeintel.core.time import utc_now
from codeintel.storage.helpers.json import encode_json_compact
from codeintel.storage.upsert import UpsertSpec

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from codeintel.storage.gateway.protocol import StorageGateway


@dataclass(frozen=True)
class SchemaVersionRecord:
    """Record of a content-addressed schema version.

    Attributes
    ----------
    schema_digest
        Stable fingerprint of the canonical schema JSON payload.
    schema_hash
        Legacy/compat schema hash (e.g., ordered name:type signature hash).
    schema_json
        Canonical schema payload (TableSchema.to_json_obj()).
    renderer_cache
        Optional JSON cache for renderer products (e.g., ibis/arrow renderings).
    created_at
        Timestamp when this schema version was first recorded.
    """

    schema_digest: str
    schema_hash: str
    schema_json: dict[str, Any]
    renderer_cache: dict[str, Any] | None = None
    created_at: datetime | None = None


@dataclass(frozen=True)
class TableSchemaRegistryRecord:
    """Record linking a table_key to its current schema version pointer.

    Attributes
    ----------
    table_key
        Schema-qualified dataset key (e.g., "core.modules").
    schema_digest
        Pointer to metadata.schema_versions.schema_digest.
    schema_hash
        Legacy/compat hash for quick comparisons.
    derivation_kind
        Provenance category (e.g., explicit_override, inferred_relation, declared_source).
    derivation_source
        Optional provenance details (e.g., target name, module).
    inference_status
        Optional status (e.g., inferred, override, error, pending).
    inference_error
        Optional error summary for failed inference.
    catalog_hash
        Optional pointer to schema manifest canonical catalog hash (schema_manifest_v2).
    updated_at
        Timestamp when this pointer was last updated.
    """

    table_key: str
    schema_digest: str
    schema_hash: str
    derivation_kind: str
    derivation_source: str | None = None
    inference_status: str | None = None
    inference_error: str | None = None
    catalog_hash: str | None = None
    updated_at: datetime | None = None


@dataclass(frozen=True)
class SchemaManifestRunRecord:
    """Record linking a build/run_id to a persisted schema manifest catalog hash.

    Attributes
    ----------
    run_id
        Build/pipeline run identifier.
    repo
        Repository slug.
    commit
        Commit SHA.
    manifest_kind
        Canonical kind label (e.g., "schema_manifest_v2").
    catalog_hash
        Catalog hash in metadata.canonical_catalogs for this manifest.
    created_at
        Timestamp when this mapping was recorded.
    """

    run_id: str
    repo: str
    commit: str
    manifest_kind: str
    catalog_hash: str
    created_at: datetime | None = None


class SchemaCatalogTracking:
    """Accessor for schema registry persistence under metadata.*.

    Provides batch persistence methods aligned with other tracking accessors.
    """

    def __init__(self, gateway: StorageGateway) -> None:
        """Initialize schema catalog tracking accessor.

        Parameters
        ----------
        gateway
            Storage gateway providing database access.
        """
        self._gateway = gateway
        self._con = gateway.con
        self._backend = gateway.policy

    def record_schema_versions_batch(self, records: Sequence[SchemaVersionRecord]) -> int:
        """Insert schema version rows (content-addressed; conflict => do nothing).

        Returns
        -------
        int
            Number of rows written to the schema_versions table (len(records)).
        """
        if not records:
            return 0

        now = utc_now()
        rows = [
            (
                r.schema_digest,
                r.schema_hash,
                encode_json_compact(r.schema_json),
                encode_json_compact(r.renderer_cache) if r.renderer_cache is not None else None,
                r.created_at or now,
            )
            for r in records
        ]

        return self._backend.upsert(
            "metadata.schema_versions",
            rows,
            columns=(
                "schema_digest",
                "schema_hash",
                "schema_json",
                "renderer_cache",
                "created_at",
            ),
            upsert=UpsertSpec(
                conflict_columns=("schema_digest",),
                update_columns=None,  # ON CONFLICT DO NOTHING
            ),
        )

    def record_table_schema_registry_batch(self, records: Sequence[TableSchemaRegistryRecord]) -> int:
        """Upsert current schema pointers for table keys.

        Returns
        -------
        int
            Number of rows written to the table_schema_registry table (len(records)).
        """
        if not records:
            return 0

        now = utc_now()
        rows = [
            (
                r.table_key,
                r.schema_digest,
                r.schema_hash,
                r.derivation_kind,
                r.derivation_source,
                r.inference_status,
                r.inference_error,
                r.catalog_hash,
                r.updated_at or now,
            )
            for r in records
        ]

        return self._backend.upsert(
            "metadata.table_schema_registry",
            rows,
            columns=(
                "table_key",
                "schema_digest",
                "schema_hash",
                "derivation_kind",
                "derivation_source",
                "inference_status",
                "inference_error",
                "catalog_hash",
                "updated_at",
            ),
            upsert=UpsertSpec(
                conflict_columns=("table_key",),
                update_columns=(
                    "schema_digest",
                    "schema_hash",
                    "derivation_kind",
                    "derivation_source",
                    "inference_status",
                    "inference_error",
                    "catalog_hash",
                    "updated_at",
                ),
            ),
        )

    def record_schema_manifest_runs_batch(self, records: Sequence[SchemaManifestRunRecord]) -> int:
        """Upsert run -> schema manifest catalog linkages.

        Returns
        -------
        int
            Number of rows written to the schema_manifest_runs table (len(records)).
        """
        if not records:
            return 0

        now = utc_now()
        rows = [
            (
                r.run_id,
                r.repo,
                r.commit,
                r.manifest_kind,
                r.catalog_hash,
                r.created_at or now,
            )
            for r in records
        ]

        return self._backend.upsert(
            "metadata.schema_manifest_runs",
            rows,
            columns=("run_id", "repo", "commit", "manifest_kind", "catalog_hash", "created_at"),
            upsert=UpsertSpec(
                conflict_columns=("run_id",),
                update_columns=("repo", "commit", "manifest_kind", "catalog_hash", "created_at"),
            ),
        )


__all__ = [
    "SchemaCatalogTracking",
    "SchemaManifestRunRecord",
    "SchemaVersionRecord",
    "TableSchemaRegistryRecord",
]
```

### Minimal wiring (so this is actually usable)

When you’re ready to integrate (separate PR is fine), the standard pattern is:

* `src/codeintel/storage/gateway/protocol.py`: add `schemas: SchemaCatalogTracking`
* `src/codeintel/storage/gateway/accessors.py`:

  * import `SchemaCatalogTracking`
  * add `schemas: SchemaCatalogTracking = field(init=False)`
  * in `__post_init__`: `self.schemas = SchemaCatalogTracking(self)`

If you want, next I can also provide the **exact companion helper** that takes a compiled `SchemaManifest` and produces the three record batches (schema_versions + table_schema_registry + schema_manifest_runs) deterministically, so callers don’t have to hand-assemble these records.


Absolutely — here’s the **companion helper** that turns a compiled `SchemaManifest` into the three deterministic record batches (plus the computed `catalog_hash` + payload so you can also upsert into `metadata.canonical_catalogs` without re-deriving anything).

Create:

`src/codeintel/storage/tracking/schema_catalog_compile.py`

```python
"""Schema catalog batch compiler.

This module converts a compiled SchemaManifest into the persistence record batches
required by SchemaCatalogTracking:

- SchemaVersionRecord batch for metadata.schema_versions
- TableSchemaRegistryRecord batch for metadata.table_schema_registry
- SchemaManifestRunRecord batch for metadata.schema_manifest_runs

It also computes the canonical catalog payload + hash for storing the manifest
into metadata.canonical_catalogs (catalog_kind="schema_manifest_v2").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from codeintel.core.hashing.fingerprint import fingerprint
from codeintel.core.schemas.hashing import schema_hash as compute_schema_hash
from codeintel.core.time import utc_now
from codeintel.storage.tracking.schema_catalog_tracking import (
    SchemaManifestRunRecord,
    SchemaVersionRecord,
    TableSchemaRegistryRecord,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime

    from codeintel.core.manifests import SchemaManifest
    from codeintel.core.schemas.primitives import TableSchema
    from codeintel.core.manifests import TableProvenance


_SCHEMA_MANIFEST_CATALOG_KIND = "schema_manifest_v2"


@dataclass(frozen=True)
class SchemaCatalogBatches:
    """Compiled persistence batches + canonical catalog payload for a SchemaManifest."""

    catalog_kind: str
    catalog_hash: str
    catalog_payload: dict[str, Any]

    schema_versions: tuple[SchemaVersionRecord, ...]
    table_schema_registry: tuple[TableSchemaRegistryRecord, ...]
    schema_manifest_runs: tuple[SchemaManifestRunRecord, ...]


def compile_schema_catalog_batches(
    manifest: SchemaManifest,
    *,
    run_id: str,
    repo: str,
    commit: str,
    now: datetime | None = None,
    catalog_kind: str = _SCHEMA_MANIFEST_CATALOG_KIND,
    manifest_kind: str = _SCHEMA_MANIFEST_CATALOG_KIND,
    include_views: bool = True,
    strict_provenance: bool = True,
    strict_hash_match: bool = True,
    catalog_inputs: Mapping[str, Any] | None = None,
) -> SchemaCatalogBatches:
    """Compile SchemaManifest -> schema registry persistence batches.

    Parameters
    ----------
    manifest
        Compiled schema manifest. For best results, compile with include_provenance=True.
    run_id
        Run identifier for schema_manifest_runs linkage.
    repo
        Repository identifier.
    commit
        Commit SHA.
    now
        Timestamp used consistently across all generated records (defaults to utc_now()).
    catalog_kind
        Catalog kind for metadata.canonical_catalogs.
    manifest_kind
        Kind label stored in metadata.schema_manifest_runs.
    include_views
        When True, include view schemas in schema_versions and table_schema_registry batches.
    strict_provenance
        When True, raise if required provenance is missing for any included table/view.
    strict_hash_match
        When True, raise if provenance.schema_hash disagrees with compute_schema_hash(table_schema).
    catalog_inputs
        Optional inputs metadata to include in the canonical catalog payload under "inputs".

        Note: this helper returns catalog_payload and catalog_hash; actual upsert into
        metadata.canonical_catalogs is performed elsewhere via upsert_canonical_catalog().

    Returns
    -------
    SchemaCatalogBatches
        Catalog payload/hash + record batches for persistence.

    Raises
    ------
    ValueError
        If manifest is not v2, or strict_provenance is True and provenance is missing,
        or strict_hash_match is True and a provenance hash mismatch is detected.
    """
    if not getattr(manifest, "is_v2", False):
        msg = f"Expected SchemaManifest v2; got version={getattr(manifest, 'version', None)!r}"
        raise ValueError(msg)

    ts = now or utc_now()

    # Canonical catalog payload (storeable in metadata.canonical_catalogs)
    payload: dict[str, Any] = dict(manifest.to_json_obj())  # already deterministic ordering inside
    if catalog_inputs is not None:
        # Preserve a consistent, explicit place for inputs without mutating the manifest schema.
        payload["inputs"] = dict(catalog_inputs)
    catalog_hash = fingerprint(payload)

    # ---- helpers -------------------------------------------------------------

    def _sorted_by_table_key(schemas: tuple[TableSchema, ...]) -> tuple[TableSchema, ...]:
        return tuple(sorted(schemas, key=lambda s: s.table_key))

    def _require_provenance(
        *,
        table_key: str,
        prov: TableProvenance | None,
        kind: str,
    ) -> TableProvenance | None:
        if prov is not None:
            return prov
        if strict_provenance:
            msg = (
                f"Missing {kind} provenance for {table_key}. "
                "Compile the manifest with include_provenance=True (recommended)."
            )
            raise ValueError(msg)
        return None

    def _schema_digest(schema: TableSchema) -> str:
        return fingerprint(schema.to_json_obj())

    # ---- schema_versions (content-addressed) ---------------------------------

    schema_versions_by_digest: dict[str, SchemaVersionRecord] = {}

    all_tables: tuple[TableSchema, ...] = _sorted_by_table_key(manifest.tables)
    all_views: tuple[TableSchema, ...] = _sorted_by_table_key(manifest.views) if include_views else ()

    for schema in all_tables + all_views:
        schema_json = schema.to_json_obj()
        digest = fingerprint(schema_json)
        if digest in schema_versions_by_digest:
            continue
        schema_versions_by_digest[digest] = SchemaVersionRecord(
            schema_digest=digest,
            schema_hash=compute_schema_hash(schema),
            schema_json=schema_json,
            renderer_cache=None,
            created_at=ts,
        )

    schema_versions = tuple(
        schema_versions_by_digest[digest] for digest in sorted(schema_versions_by_digest)
    )

    # ---- table_schema_registry (current pointers) ----------------------------

    registry_records: list[TableSchemaRegistryRecord] = []

    for table in all_tables:
        table_key = table.table_key
        prov = _require_provenance(
            table_key=table_key,
            prov=manifest.table_provenance.get(table_key),
            kind="table",
        )
        computed_hash = compute_schema_hash(table)
        prov_hash = prov.schema_hash if prov is not None else computed_hash
        if strict_hash_match and prov is not None and prov_hash != computed_hash:
            msg = (
                f"Schema hash mismatch for {table_key}: "
                f"provenance={prov_hash} computed={computed_hash}"
            )
            raise ValueError(msg)

        registry_records.append(
            TableSchemaRegistryRecord(
                table_key=table_key,
                schema_digest=_schema_digest(table),
                schema_hash=prov_hash,
                derivation_kind=(prov.derivation_kind if prov is not None else "explicit_override"),
                derivation_source=(prov.derivation_source if prov is not None else "manifest"),
                inference_status=(prov.inference_status if prov is not None else None),
                inference_error=(prov.inference_error if prov is not None else None),
                catalog_hash=catalog_hash,
                updated_at=ts,
            )
        )

    for view in all_views:
        view_key = view.table_key
        prov = _require_provenance(
            table_key=view_key,
            prov=manifest.view_provenance.get(view_key),
            kind="view",
        )
        computed_hash = compute_schema_hash(view)
        prov_hash = prov.schema_hash if prov is not None else computed_hash
        if strict_hash_match and prov is not None and prov_hash != computed_hash:
            msg = (
                f"Schema hash mismatch for {view_key}: "
                f"provenance={prov_hash} computed={computed_hash}"
            )
            raise ValueError(msg)

        registry_records.append(
            TableSchemaRegistryRecord(
                table_key=view_key,
                schema_digest=_schema_digest(view),
                schema_hash=prov_hash,
                derivation_kind=(prov.derivation_kind if prov is not None else "view_inferred"),
                derivation_source=(prov.derivation_source if prov is not None else "duckdb"),
                inference_status=(prov.inference_status if prov is not None else None),
                inference_error=(prov.inference_error if prov is not None else None),
                catalog_hash=catalog_hash,
                updated_at=ts,
            )
        )

    table_schema_registry = tuple(sorted(registry_records, key=lambda r: r.table_key))

    # ---- schema_manifest_runs (run linkage) ----------------------------------

    schema_manifest_runs = (
        SchemaManifestRunRecord(
            run_id=run_id,
            repo=repo,
            commit=commit,
            manifest_kind=manifest_kind,
            catalog_hash=catalog_hash,
            created_at=ts,
        ),
    )

    return SchemaCatalogBatches(
        catalog_kind=catalog_kind,
        catalog_hash=catalog_hash,
        catalog_payload=payload,
        schema_versions=schema_versions,
        table_schema_registry=table_schema_registry,
        schema_manifest_runs=schema_manifest_runs,
    )


__all__ = [
    "SchemaCatalogBatches",
    "compile_schema_catalog_batches",
]
```

### Intended usage (caller side)

```python
from codeintel.storage.metadata.catalogs import CanonicalCatalogEntry, upsert_canonical_catalog
from codeintel.storage.tracking.schema_catalog_compile import compile_schema_catalog_batches

batches = compile_schema_catalog_batches(
    manifest,
    run_id=run_id,
    repo=env.repo,
    commit=env.commit,
    strict_provenance=True,  # strongly recommended
    catalog_inputs={"source": "serving_artifacts", "read_only": bool(env.gateway.config.read_only)},
)

# 1) upsert canonical catalog payload
entry = CanonicalCatalogEntry(
    catalog_kind=batches.catalog_kind,
    catalog_hash=batches.catalog_hash,
    payload=batches.catalog_payload,
    inputs=None,
    created_at=utc_now(),
)
upsert_canonical_catalog(env.gateway, entry)

# 2) persist the registry tables
env.gateway.schemas.record_schema_versions_batch(batches.schema_versions)
env.gateway.schemas.record_table_schema_registry_batch(batches.table_schema_registry)
env.gateway.schemas.record_schema_manifest_runs_batch(batches.schema_manifest_runs)
```

---

## One important fix (so your `schema_versions` insert is actually “DO NOTHING”)

In your earlier `SchemaCatalogTracking.record_schema_versions_batch`, you want **ON CONFLICT DO NOTHING**. Your DuckDB backend treats `update_columns=None` as “update all non-conflict columns”, so the correct setting is **`update_columns=()`**.

Patch (inside `record_schema_versions_batch`):

```python
upsert=UpsertSpec(
    conflict_columns=("schema_digest",),
    update_columns=(),  # forces DO NOTHING in DuckDBPolicyBackend._build_upsert
),
```

If you want, next I can also provide the “single public method” `persist_schema_manifest(...)` on `SchemaCatalogTracking` that internally calls this compiler + canonical catalog upsert + the 3 batch writes inside one transaction.

Below is the **exact “single public method”** `persist_schema_manifest(...)` added onto `SchemaCatalogTracking`, implemented as an **atomic transaction** that:

1. compiles a `SchemaManifest` → record batches (via your companion helper)
2. upserts the canonical catalog entry (`metadata.canonical_catalogs`)
3. writes `schema_versions`, `table_schema_registry`, `schema_manifest_runs`

I’m also including the **critical fix** to ensure `schema_versions` is truly **ON CONFLICT DO NOTHING** under your `DuckDBPolicyBackend` semantics.

---

## Patch: `src/codeintel/storage/tracking/schema_catalog_tracking.py`

Add this dataclass near the other record dataclasses:

```python
@dataclass(frozen=True, slots=True)
class PersistSchemaManifestResult:
    """Summary of a schema manifest persistence transaction."""

    catalog_kind: str
    catalog_hash: str
    tables: int
    views: int
    schema_versions_rows: int
    table_schema_registry_rows: int
    schema_manifest_runs_rows: int
```

### 1) Fix: `record_schema_versions_batch` must use `update_columns=()`

Your backend treats `update_columns=None` as “update all non-conflict columns”. To get DO NOTHING, you must pass an **empty sequence**.

```python
return self._backend.upsert(
    "metadata.schema_versions",
    rows,
    columns=(
        "schema_digest",
        "schema_hash",
        "schema_json",
        "renderer_cache",
        "created_at",
    ),
    upsert=UpsertSpec(
        conflict_columns=("schema_digest",),
        update_columns=(),  # ON CONFLICT DO NOTHING
    ),
)
```

### 2) Add: `persist_schema_manifest(...)` (single public method)

Drop this into the `SchemaCatalogTracking` class:

```python
from collections.abc import Mapping

from codeintel.storage.metadata import build_catalog_entry, upsert_canonical_catalog

# ...

class SchemaCatalogTracking:
    # existing __init__ + record_*_batch methods ...

    def persist_schema_manifest(
        self,
        manifest: "SchemaManifest",
        *,
        run_id: str,
        repo: str,
        commit: str,
        catalog_inputs: Mapping[str, object] | None = None,
        include_views: bool = True,
        strict_provenance: bool = True,
        strict_hash_match: bool = True,
        now: "datetime | None" = None,
        catalog_kind: str = "schema_manifest_v2",
        manifest_kind: str = "schema_manifest_v2",
    ) -> PersistSchemaManifestResult:
        """Persist SchemaManifest into canonical catalogs + schema registry tables atomically.

        This performs (in a single DuckDB transaction):
        1) Compile manifest -> record batches + catalog payload/hash
        2) Upsert metadata.canonical_catalogs for schema_manifest_v2
        3) Insert schema_versions (content-addressed; DO NOTHING on conflict)
        4) Upsert table_schema_registry (current pointer per table_key)
        5) Upsert schema_manifest_runs (run linkage)

        Returns
        -------
        PersistSchemaManifestResult
            Counts/hashes summarizing the persistence operation.

        Raises
        ------
        RuntimeError
            If gateway is read-only.
        ValueError
            If strict checks fail in batch compilation (e.g., missing provenance).
        """
        if getattr(self._gateway, "config", None) is not None and self._gateway.config.read_only:
            msg = "Cannot persist schema manifest into a read-only storage gateway"
            raise RuntimeError(msg)

        # Local import avoids circular import:
        # schema_catalog_compile imports record dataclasses from this module.
        from codeintel.storage.tracking.schema_catalog_compile import (
            compile_schema_catalog_batches,
        )

        batches = compile_schema_catalog_batches(
            manifest,
            run_id=run_id,
            repo=repo,
            commit=commit,
            now=now,
            catalog_kind=catalog_kind,
            manifest_kind=manifest_kind,
            include_views=include_views,
            strict_provenance=strict_provenance,
            strict_hash_match=strict_hash_match,
            catalog_inputs=dict(catalog_inputs) if catalog_inputs is not None else None,
        )

        entry = build_catalog_entry(
            catalog_kind=batches.catalog_kind,
            catalog_hash=batches.catalog_hash,
            payload=batches.catalog_payload,
            inputs=dict(catalog_inputs) if catalog_inputs is not None else None,
        )

        with self._backend.transaction():
            # 1) canonical catalog (metadata.canonical_catalogs)
            upsert_canonical_catalog(self._gateway, entry)

            # 2) schema registry tables
            n_schema_versions = self.record_schema_versions_batch(batches.schema_versions)
            n_registry = self.record_table_schema_registry_batch(batches.table_schema_registry)
            n_runs = self.record_schema_manifest_runs_batch(batches.schema_manifest_runs)

        return PersistSchemaManifestResult(
            catalog_kind=batches.catalog_kind,
            catalog_hash=batches.catalog_hash,
            tables=len(manifest.tables),
            views=len(manifest.views) if include_views else 0,
            schema_versions_rows=n_schema_versions,
            table_schema_registry_rows=n_registry,
            schema_manifest_runs_rows=n_runs,
        )
```

---

## Intended call-site usage

```python
result = gateway.schemas.persist_schema_manifest(
    manifest,
    run_id=run_id,
    repo=repo,
    commit=commit,
    catalog_inputs={"source": "serving_artifacts"},
    include_views=True,
    strict_provenance=True,
)

# result.catalog_hash is now the canonical pointer you can carry in run metadata/logs
```
