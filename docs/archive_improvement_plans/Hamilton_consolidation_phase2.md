
# Hamilton consolidation phase 2 - Hamilton, DAG first approach with focus on schemas #

Here’s how I’d push your codebase the rest of the way toward a truly **Hamilton‑first, DAG‑first** system—especially around **schemas as a first‑class product of the DAG** (so you stop hand‑reconciling “schema config” vs “what the DAG actually produces”).

I’m going to focus on two things:

1. **Further consolidation opportunities** (removing remaining “legacy” gravity wells outside `src/codeintel/build`)

2. A **DAG‑first schema authority** design where **Hamilton (and the DAG) becomes the single source of truth** for:

* table schemas
* schema hashing / versioning
* DDL creation / drift detection
* row bindings + JSON schema exports
* “what tables exist” catalog metadata

…and the rest of the codebase consumes that—never the other way around.

---

## 1) What’s still “not DAG-first” in the current state (and why it matters)

Even with PR‑55 nearly done, there are a few remaining places where the **schema source of truth is still effectively outside the Hamilton DAG**, and that creates the exact reconciliation burden you’re describing.

### A. Schema hashes + versioning still depend on `config.datasets.schemas.TABLE_SCHEMAS`

You currently compute schema hash in Phase‑4 asset fingerprinting via:

* `src/codeintel/build/assets/fingerprinting.py` → `compute_table_schema_hash()` → reads `codeintel.config.datasets.schemas.TABLE_SCHEMAS`

That means:

* schema hash isn’t “DAG output” (it’s “static config output”)
* asset version hashes are implicitly coupled to the legacy schema registry

### B. DuckDB DDL / ensure_table still imports dataset contracts from `codeintel.config.datasets`

Your DDL + table creation path is still wired to:

* `src/codeintel/storage/duckdb_policy_backend.py` → `_dataset_contracts_by_table_key()` via importlib into `codeintel.config.datasets.contracts`

That means:

* table creation is *not* derived from the DAG closure / compiled plan
* any future “schema = DAG” shift is blocked until `DuckDBPolicyBackend` stops importing `config.datasets.*`

### C. Hamilton schema registry exists (good), but is still “fed by config”

You’ve got:

* `src/codeintel/build/hamilton/contracts/schemas/` with `SCHEMA_REGISTRY`

…but `builder.py` still builds schemas by pulling from config contracts / pandera schema definitions.

So: the “build-owned registry” exists (excellent), but it’s not yet the *authority*.

---

## 2) The end state you’re aiming for: “Schema Authority = DAG”

What you’re describing (“only specify raw datasets + calculations; schemas flow automatically”) is not only practical—it’s **exactly the right direction** if you’re willing to be aggressive pre‑production.

The cleanest mental model:

### ✅ Goal: The DAG produces **two parallel products**

1. **Data product DAG**: tables, views, artifacts
2. **Schema product DAG**: schema manifests, hashes, JSON schema exports, row models

…and **the schema product DAG is derived from the data product DAG**, not a separate config universe.

This yields the property you want:

> If you change an upstream raw schema or a compute expression, the downstream schemas update deterministically, and every downstream consumer sees it automatically.

---

## 3) The “DAG-first schema authority” architecture I recommend

### 3.1 Introduce a single interface: `SchemaProvider`

You want one place in the system that answers:

* “What is the schema for `analytics.goid_risk_factors`?”
* “What is its schema hash?”
* “Give me Pandera schema / JSON schema / row binding”
* “Give me upstream/downstream schema lineage”

Everything else calls this.

**Create a small protocol in `core`** (so storage + build can depend on it without cycles):

```python
# src/codeintel/core/schemas/provider.py
from __future__ import annotations
from typing import Protocol, Iterable

from codeintel.core.schemas.primitives import TableSchema  # (move primitives here)

class SchemaProvider(Protocol):
    def get_table_schema(self, table_key: str) -> TableSchema | None: ...
    def require_table_schema(self, table_key: str) -> TableSchema: ...
    def iter_table_schemas(self) -> Iterable[TableSchema]: ...
```

Then implement it in build:

```python
# src/codeintel/build/schemas/provider.py
from __future__ import annotations
from functools import lru_cache

from codeintel.core.schemas.provider import SchemaProvider
from codeintel.core.schemas.primitives import TableSchema
from codeintel.build.registry import get_target_graph, derive_schemas_from_targets

class TargetGraphSchemaProvider(SchemaProvider):
    @lru_cache
    def _schemas(self) -> dict[str, TableSchema]:
        graph = get_target_graph()
        return derive_schemas_from_targets(graph.all_targets)

    def get_table_schema(self, table_key: str) -> TableSchema | None:
        return self._schemas().get(table_key)

    def require_table_schema(self, table_key: str) -> TableSchema:
        schema = self.get_table_schema(table_key)
        if schema is None:
            raise KeyError(f"Unknown table schema: {table_key}")
        return schema

    def iter_table_schemas(self):
        return self._schemas().values()
```

This is the “bridge” provider. It’s not yet schema-inferred-from-Ibis, but it **centralizes the consumption path** so you can switch providers later without rewriting the world.

---

### 3.2 Make schema hashing a pure function of `TableSchema`

Right now schema hashing logic is duplicated (and tied to `TABLE_SCHEMAS`).

Create one canonical hasher:

```python
# src/codeintel/core/schemas/hashing.py
from __future__ import annotations
import hashlib
from codeintel.core.schemas.primitives import TableSchema

def canonical_type(type_str: str) -> str:
    upper = type_str.upper()
    if upper in {"TIMESTAMPTZ", "TIMESTAMP WITH TIME ZONE"}:
        return "TIMESTAMPTZ"
    if upper.startswith("DECIMAL") or upper == "BIGINT":
        return "BIGINT"
    return upper

def schema_hash(table: TableSchema) -> str:
    parts = [f"{c.name}:{canonical_type(c.type)}" for c in table.columns]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
```

Then update fingerprinting:

```python
# src/codeintel/build/assets/fingerprinting.py
from codeintel.core.schemas.hashing import schema_hash
from codeintel.build.schemas.provider import TargetGraphSchemaProvider

_SCHEMA_PROVIDER = TargetGraphSchemaProvider()

def compute_table_schema_hash(table_key: str) -> str | None:
    table = _SCHEMA_PROVIDER.get_table_schema(table_key)
    if table is None:
        return None
    return schema_hash(table)
```

✅ This is a *huge* consolidation win immediately:

* asset versioning is now driven by the build graph’s schema authority
* `config.datasets.schemas.TABLE_SCHEMAS` stops being “special”

---

### 3.3 Decouple `DuckDBPolicyBackend` from `config.datasets.*`

This is one of the highest ROI refactors you can do now.

Instead of `DuckDBPolicyBackend` importing dataset contracts internally, make it accept a `SchemaProvider`.

```python
# src/codeintel/storage/duckdb_policy_backend.py
from codeintel.core.schemas.provider import SchemaProvider

class DuckDBPolicyBackend:
    def __init__(self, gateway: MinimalGateway, *, schema_provider: SchemaProvider | None = None):
        self._gateway = gateway
        self._schema_provider = schema_provider
```

Then in `ensure_table()`:

```python
def ensure_table(self, table_key: str) -> None:
    if table_key in _TABLE_CREATION_DENYLIST:
        return
    table_schema = self._schema_provider.require_table_schema(table_key)
    ddl = _build_create_table(table_schema, if_not_exists=True)
    self._gateway.execute(ddl.sql(dialect=DUCKDB_DIALECT))
```

And wire it from build env creation / bootstrap:

```python
from codeintel.build.schemas.provider import TargetGraphSchemaProvider
backend = DuckDBPolicyBackend(env.gateway, schema_provider=TargetGraphSchemaProvider())
```

✅ Once this is done:

* storage is no longer coupled to `codeintel.config.datasets.contracts`
* you can later swap the schema provider to “Hamilton inferred schemas” without touching storage again

---

## 4) Now the aggressive part: schemas inferred from the DAG (not declared twice)

What you asked for—“intermediate and input schemas are functions of raw inputs + calculations”—is most achievable if you enforce a rule:

> **All table-producing compute nodes return Ibis expressions (or Arrow tables with a schema).**

You already have strong examples (e.g., `risk_factors`, `hotspots`) returning `ir.Table`. That’s the correct direction.

### 4.1 Add a second provider: `HamiltonInferredSchemaProvider`

This provider does not read static `TableSchema` for *derived* tables. It *computes* it.

There are two viable inference strategies:

#### Strategy A (fast, pure): use Ibis expression `.schema()`

* Pros: no DB needed, purely compile-time
* Cons: type inference can be “unknown” in some edge cases

#### Strategy B (best-in-class, deterministic): compile into ephemeral DuckDB and `DESCRIBE`

* Pros: extremely robust, matches DuckDB’s real typing
* Cons: a bit more machinery

Given your “best-in-class” constraint, I recommend **Strategy B**.

---

### 4.2 The “Schema Compile” pipeline (like dbt compile, but for Hamilton/Ibis)

Add a build-time command:

* `codeintel build schema compile --targets ...`
* It:

  1. builds the Hamilton runtime closure
  2. for every target table produced by a native compute node returning Ibis:

     * compiles the expression
     * computes output column names + duckdb types deterministically
  3. writes a `schema_manifest.json` (or `.yaml`) artifact
  4. optionally upserts schema hashes into your catalog tables

#### Core API

```python
# src/codeintel/build/schemas/compile.py
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class CompiledTableSchema:
    table_key: str
    columns: list[tuple[str, str]]  # (name, duckdb_type)

def compile_schema_for_expr(con, expr_sql: str) -> list[tuple[str, str]]:
    # DuckDB can describe a query without executing it materially
    rows = con.execute(f"DESCRIBE {expr_sql}").fetchall()
    # rows like: (column_name, column_type, null, key, default, extra)
    return [(str(r[0]), str(r[1])) for r in rows]
```

Then:

```python
def compile_target_table_schemas(env, runtime, target: str) -> list[CompiledTableSchema]:
    # 1) execute compute node(s) only to get ibis expr(s), not materializers
    # 2) ibis.to_sql(expr)
    # 3) DESCRIBE sql
    # 4) return compiled schema objects
    ...
```

This gives you the “DAG produces schema” property.

---

### 4.3 How to map “target → output tables → compute expr(s)” reliably

You already have a strong convention:

* compute node: `t__{target}__compute`
* materialize node: `t__{target}`

I would formalize this convention into a tiny helper:

```python
# src/codeintel/build/hamilton/naming.py
def compute_node(target: str) -> str:
    return f"t__{target}__compute"
```

Then the schema compiler can do:

* call `runtime.dr.execute([compute_node(target)], inputs=...)`
* inspect the returned object:

  * if `ir.Table`: one table
  * if `dict[str, ir.Table]`: multi-table output
  * else: fallback to declared schema

That allows “schema is derived” for the Ibis-native surface, without requiring new tags everywhere.

---

## 5) The key “DAG-first” shift you asked about: schema drives everything downstream

Once you have either:

* a `SchemaProvider` that returns inferred schemas for derived tables, or
* a compiled manifest that becomes the canonical record

…you can flip all the “schema consumers” to use it.

### Consumers to flip (high value)

#### A) Table creation / DDL

* `DuckDBPolicyBackend.ensure_table()`
* `ensure_all_schemas()`
* any “bootstrap” routines

All should route through the provider/manifest.

#### B) Asset versioning schema hash

* `build/assets/fingerprinting.compute_table_schema_hash`
* emitter logic (`build/assets/emitter.py`)

No more `config.datasets.schemas.TABLE_SCHEMAS`.

#### C) Contract validation (Pandera)

Your `materialize_table(... validate=True)` path currently pulls Pandera schemas from the registry.

In the best DAG-first world:

* Pandera schemas are **generated from the inferred TableSchema**, not hand-authored.
* Optional *additional constraints* can still be applied (uniques, ranges), but those become DAG metadata too.

A good split is:

* **shape + types** come from inferred schema
* **constraints** come from explicit “quality nodes” / declared checks

---

## 6) Is “schema inferred from calculations” fragile? The real risks + how to make it safe

You asked the right question. The risks are real, but manageable—and the mitigations are exactly what a best-in-class system should do.

### Risk 1: accidental schema changes from small compute edits

Mitigation:

* treat schema as a build artifact with an explicit diff + review gate
* store schema manifest in git or in the catalog with “promotion”

You already have the Phase‑4 promotion primitives—this fits perfectly.

### Risk 2: type inference differences across DuckDB/Ibis versions

Mitigation:

* pin versions (already likely)
* canonicalize types (`TIMESTAMP WITH TIME ZONE` → `TIMESTAMPTZ`, etc.)
* snapshot test the schema manifest in CI

### Risk 3: dynamic columns (JSON keys, pivoted columns)

Mitigation:

* explicitly constrain these to `JSON` / `MAP` / `STRUCT` columns
* or forbid “dynamic output columns” for persisted assets

Given your goals, I’d enforce:

> persisted tables must have statically determinable schemas

---

## 7) Additional consolidation opportunities (beyond schemas)

If your priority is “best-in-class Hamilton system” and removing bespoke logic, here are the biggest consolidation levers I still see:

### A) Unify node metadata tags (`node_type` vs `node_kind`)

Right now native modules mix tag keys (`node_type` and `node_kind`). Pick one canonical set, e.g.:

* `domain`
* `target`
* `node_type` in `{loader, compute, materialize, artifact}`
* `output_kind` in `{table, view, artifact, schema}`

Then make your graph exports and schema compilation read those keys consistently.

### B) Stop maintaining two DAGs where possible (TargetGraph vs Hamilton graph)

You still use `TargetGraph` as the planning/metadata DAG, and Hamilton as the execution DAG.

Long-term best-in-class: derive the “target DAG” from Hamilton introspection:

* targets are materialize nodes
* dependencies are actual Hamilton dependencies
* outputs come from materializer metadata (datasets produced)

That eliminates drift risk.

### C) Consolidate old “metadata.dataset_schema_registry” with Phase‑4 asset catalog

You already store schema hashes in `build.asset_versions`.
If `metadata.dataset_schema_registry` is still used only for drift detection, you can:

* replace it with “expected schema hash” in the asset catalog tables, or
* compute drift by comparing inferred schema hash vs actual table schema

This removes another “legacy schema registry” surface.

---

## 8) A concrete “next sequence” I would implement (aggressive, but clean)

If you want a crisp, execution-oriented path from *today* to *fully DAG-first schemas*, I’d do it in this order:

1. **Introduce `SchemaProvider` + move schema hashing into `core`**
   (No behavior change, just consolidation.)

2. **Make `DuckDBPolicyBackend` accept `SchemaProvider`**
   (Storage decoupled from `config.datasets.*`.)

3. **Update fingerprinting + asset emitter to use `SchemaProvider`**
   (Schema hashes now come from build-owned authority.)

4. **Implement “schema compile” for Ibis-native targets**

   * execute compute nodes only
   * compile SQL
   * `DESCRIBE` in ephemeral DuckDB
   * produce `TableSchema`

5. **Switch provider for derived tables to use inferred schemas**

   * static schemas remain only for raw/source tables (or non-Ibis targets)

6. **Generate Pandera schemas + row bindings from inferred `TableSchema`**

   * delete hand-maintained row bindings gradually
   * keep only *constraints* as explicit declarations

7. **Add a schema manifest snapshot test** under `tests/build/hamilton/snapshots/`

   * fail CI on unreviewed schema drift

---

## 9) If you want, I can turn this into PR‑56…PR‑65 in your usual tracking format

I know you didn’t explicitly ask for the PR board in this message—but since that’s been your preferred way to execute, I can translate the above into:

* PR‑56: SchemaProvider protocol + core schema hashing
* PR‑57: DuckDBPolicyBackend schema_provider injection
* PR‑58: Fingerprinting/emitter use provider
* PR‑59: Schema compile command + manifest writer
* PR‑60: Inferred schemas for Ibis-native targets
* PR‑61: Pandera-from-TableSchema generator
* PR‑62: RowBinding-from-schema as default, delete legacy row bindings
* PR‑63: Schema manifest CLI snapshot + CI gate
* PR‑64: Unify node tags (`node_type`)
* PR‑65: Optional: derive TargetGraph dependencies from Hamilton graph

…with per‑PR tasks, tests, and snapshot filenames.

If you want me to do that, I’ll base it strictly on your current repo structure (the same way as before) and keep the PRs “mergeable and safe” while still aggressively moving you to the final state.


# Detailed tracker for Hamilton integration phase 2 #

Below is the **PR‑56 … PR‑65 tracking board** to get you to a truly **DAG‑first, Hamilton‑first schema authority** using **Strategy B** (compile into ephemeral DuckDB + `DESCRIBE`) — and then to push the schema authority all the way through: DDL, fingerprinting/versioning, validation, row bindings, and CI drift gates.

I’m assuming your current system already has:

* generated node mode + loader nodes (`q__*`, `df__*`) 
* query loader nodes that load a `DatasetRef` into an Ibis expression via the Ibis adapter 
* a CLI snapshot framework under `tests/build/hamilton/snapshots/` with a YAML manifest 
* a Pandera “contract hook” (`get_pandera_schema`, `validate_dataframe`, etc.) already wired in 

---

# PR‑56 — Core schema authority primitives (SchemaProvider + canonical hashing)

## Goal

Create a single, stable schema “language” in `codeintel.core` and a **SchemaProvider** interface so everything (DDL, fingerprinting, contracts, diffs) can depend on **one** schema representation.

## Tasks checklist

* [ ] Add new core package:

  * [ ] `src/codeintel/core/schemas/__init__.py`
  * [ ] `src/codeintel/core/schemas/primitives.py`
  * [ ] `src/codeintel/core/schemas/provider.py`
  * [ ] `src/codeintel/core/schemas/hashing.py`
  * [ ] `src/codeintel/core/schemas/serde.py` (JSON read/write; optional but recommended)
* [ ] Define core primitives (minimal but complete):

  * `Column(name: str, type: str, nullable: bool = True, …)`
  * `TableSchema(schema: str, name: str, columns: list[Column], primary_key: tuple[str, ...] = (), …)`
  * `table_key` property: `f"{schema}.{name}"`
* [ ] Define `SchemaProvider` protocol:

  * `get_table_schema(table_key) -> TableSchema | None`
  * `require_table_schema(table_key) -> TableSchema`
  * `iter_table_schemas() -> Iterable[TableSchema]`
* [ ] Add canonical hashing:

  * `canonical_type(type_str) -> str`
  * `schema_hash(table_schema) -> str` (stable ordering + normalized types)
* [ ] Convert any existing “TableSchema/Column” imports in build/storage to point to core **without changing behavior** (pure consolidation).

### Code snippet (drop‑in)

```python
# src/codeintel/core/schemas/provider.py
from __future__ import annotations
from typing import Iterable, Protocol

from codeintel.core.schemas.primitives import TableSchema

class SchemaProvider(Protocol):
    def get_table_schema(self, table_key: str) -> TableSchema | None: ...
    def require_table_schema(self, table_key: str) -> TableSchema: ...
    def iter_table_schemas(self) -> Iterable[TableSchema]: ...
```

```python
# src/codeintel/core/schemas/hashing.py
from __future__ import annotations
import hashlib
from codeintel.core.schemas.primitives import TableSchema

_CANON = {
    "TIMESTAMP WITH TIME ZONE": "TIMESTAMPTZ",
    "TIMESTAMP_TZ": "TIMESTAMPTZ",
}

def canonical_type(type_str: str) -> str:
    t = type_str.strip().upper()
    t = _CANON.get(t, t)
    t = " ".join(t.split())
    return t

def schema_hash(schema: TableSchema) -> str:
    parts = [f"{c.name}:{canonical_type(c.type)}:{'N' if c.nullable else 'NN'}" for c in schema.columns]
    payload = f"{schema.schema}.{schema.name}|" + "|".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
```

## Tests checklist (`tests/build/hamilton/`)

* [ ] `test_pr56_schema_hashing.py`

  * [ ] hash stable across identical schemas
  * [ ] type canonicalization works (`timestamp with time zone` → `TIMESTAMPTZ`)
  * [ ] column ordering affects hash (unless you intentionally sort — decide and enforce)

## CLI snapshots

* [ ] None.

---

# PR‑57 — Inject SchemaProvider into DuckDBPolicyBackend (DDL uses schema authority)

## Goal

Remove any remaining storage‑side coupling to “dataset config/contracts.” Storage should only need:

* an execution connection
* and a **SchemaProvider** for DDL/schema enforcement

## Tasks checklist

* [ ] Update `src/codeintel/storage/duckdb_policy_backend.py`:

  * [ ] Add optional `schema_provider: SchemaProvider | None`
  * [ ] Make `ensure_table(table_key)` use `schema_provider.require_table_schema(table_key)`
  * [ ] Remove any importlib-based dataset contract lookup from inside storage
* [ ] Add a helper to convert `TableSchema` → DuckDB CREATE TABLE SQL:

  * Prefer to reuse your existing SQLGlot/DDL utilities if you already have them.
  * Otherwise implement a minimal DDL builder in storage (but reuse core primitives).
* [ ] Wire schema provider from build runtime/bootstrap:

  * Wherever `DuckDBPolicyBackend` is constructed, pass a provider (even a “declared schemas provider” initially).

### Code snippet (constructor + ensure)

```python
# src/codeintel/storage/duckdb_policy_backend.py
from __future__ import annotations
from codeintel.core.schemas.provider import SchemaProvider

class DuckDBPolicyBackend:
    def __init__(self, con, *, schema_provider: SchemaProvider | None = None) -> None:
        self._con = con
        self._schema_provider = schema_provider

    def ensure_table(self, table_key: str) -> None:
        if self._schema_provider is None:
            raise RuntimeError("DuckDBPolicyBackend requires a schema_provider for ensure_table()")

        schema = self._schema_provider.require_table_schema(table_key)
        self._ensure_schema_exists(schema.schema)

        ddl_sql = build_create_table_sql(schema, if_not_exists=True)  # your helper
        self._con.execute(ddl_sql)
```

## Tests checklist (`tests/build/hamilton/`)

* [ ] `test_pr57_duckdb_policy_schema_provider.py`

  * [ ] provision in‑memory DuckDB
  * [ ] stub provider with one `TableSchema`
  * [ ] `ensure_table()` creates schema + table
  * [ ] assert `PRAGMA table_info('schema.table')` matches expected columns

## CLI snapshots

* [ ] None.

---

# PR‑58 — Fingerprinting + asset emitter use SchemaProvider (schema hash becomes DAG‑owned)

## Goal

Stop computing schema hashes from legacy registries. **Fingerprinting should depend only on the schema authority**.

## Tasks checklist

* [ ] Update `src/codeintel/build/assets/fingerprinting.py`

  * [ ] replace any `TABLE_SCHEMAS` / config registry dependency with `schema_provider.require_table_schema(table_key)`
  * [ ] compute schema hash via `codeintel.core.schemas.hashing.schema_hash`
* [ ] Update `src/codeintel/build/assets/emitter.py` (or wherever asset versions get written)

  * [ ] use the new schema hash
  * [ ] ensure schema hash stored on every run for table assets
* [ ] Add a single build-owned provider (for now):

  * [ ] `src/codeintel/build/schemas/provider_declared.py` → reads “declared raw/known table schemas” (your current DatasetSchema standard)
  * This provider will later be replaced by inferred provider (PR‑60)

## Tests checklist (`tests/build/hamilton/`)

* [ ] `test_pr58_fingerprinting_schema_hash.py`

  * [ ] stub schema provider
  * [ ] ensure `compute_table_schema_hash(table_key)` returns `schema_hash(table_schema)`
  * [ ] ensure “missing schema” behavior is explicit (None vs error — pick one and enforce)

## CLI snapshots

* [ ] None.

---

# PR‑59 — `codeintel build schema compile` command + manifest writer (declared-only first)

## Goal

Introduce the schema “product” as a first‑class artifact:

* `codeintel build schema compile …`
* writes a stable `schema_manifest.json` (or prints it)

Start with **declared schemas only** so the CLI/UX and file formats stabilize before inference.

## Tasks checklist

* [ ] Add schema CLI group under build:

  * [ ] `src/codeintel/cli/commands/build_schema.py` (recommended) or extend build commands file
  * [ ] subcommands:

    * [ ] `compile`
    * [ ] (optional scaffolding) `show`, `diff` stubs returning “not implemented”
* [ ] Add build-side schema manifest types:

  * [ ] `src/codeintel/build/schemas/manifest.py`

    * `SchemaManifest(version, generated_at?, tables: list[TableSchema], …)`
  * [ ] `src/codeintel/build/schemas/compile.py`

    * `compile_schema_manifest(targets=…, provider=…) -> SchemaManifest`
* [ ] Implement output options:

  * [ ] `--format json` (default)
  * [ ] `--output <path>` (write file) and/or default stdout
  * [ ] `--targets` / `--module` / `--all` parity with build graph selection
* [ ] Ensure stable output:

  * [ ] sort tables by `table_key`
  * [ ] sort columns by “declared order” (or name; pick one and enforce)

### Code snippet (manifest)

```python
# src/codeintel/build/schemas/manifest.py
from __future__ import annotations
from dataclasses import dataclass
from codeintel.core.schemas.primitives import TableSchema

@dataclass(frozen=True)
class SchemaManifest:
    version: str
    tables: tuple[TableSchema, ...]

    def to_json_obj(self) -> dict[str, object]:
        return {
            "version": self.version,
            "tables": [t.to_json_obj() for t in self.tables],
        }
```

## Tests checklist (`tests/build/hamilton/`)

* [ ] `test_pr59_schema_compile_manifest.py`

  * [ ] compile manifest from stub provider
  * [ ] verify deterministic ordering
  * [ ] verify JSON roundtrip (optional)

## CLI snapshots (`tests/build/hamilton/snapshots/`)

Add cases to `manifest.yaml` 

* [ ] Command: `codeintel build schema --help`

  * Snapshot file: `pr59_schema_help.txt`
* [ ] Command: `codeintel build schema compile --help`

  * Snapshot file: `pr59_schema_compile_help.txt`

**Manifest entries (example):**

```yaml
- name: "pr59_schema_help"
  tags: ["pr59", "schema", "help", "tiny", "text"]
  args: ["build", "schema", "--help"]
  kind: "text"
  snapshot: "pr59_schema_help.txt"

- name: "pr59_schema_compile_help"
  tags: ["pr59", "schema", "compile", "help", "tiny", "text"]
  args: ["build", "schema", "compile", "--help"]
  kind: "text"
  snapshot: "pr59_schema_compile_help.txt"
```

---

# PR‑60 — Inferred schemas for Ibis‑native targets (Strategy B: ephemeral DuckDB + DESCRIBE)

## Goal

Make schemas a **function of raw inputs + calculations**:

* For Ibis-native targets, infer output schema by:

  1. building empty upstream tables in ephemeral DuckDB
  2. compiling Ibis expression to SQL
  3. `DESCRIBE <sql>` to get deterministic DuckDB types
  4. converting to `TableSchema`

This is the “DAG-first schema authority” core.

## Tasks checklist

* [ ] Add inference engine:

  * [ ] `src/codeintel/build/schemas/infer_duckdb.py`

    * `infer_table_schema_from_ibis(expr, duckdb_con) -> TableSchema`
    * type canonicalization mapping
* [ ] Add “ephemeral schema compile env”:

  * [ ] `src/codeintel/build/schemas/ephemeral_duckdb.py`

    * create in-memory duckdb connection
    * ensure schemas exist
    * create empty upstream tables from declared schemas
* [ ] Integrate with Hamilton/native compute:

  * [ ] Determine “Ibis-native targets”:

    * by registry flag (`impl_kind == native`), or
    * by presence of a compute node convention `t__{target}__compute` returning `ibis.expr.types.Table`
  * [ ] Execute compute nodes to obtain Ibis expressions
  * [ ] Compile in topological order so upstream inferred schemas are available when needed
* [ ] Introduce a composite provider:

  * [ ] `src/codeintel/build/schemas/provider_hamilton.py`

    * returns declared schemas for raw/source tables
    * returns inferred schemas for native derived tables
    * fallback to declared output schema for non-native/plugin-wrapper tables (until migrated)

### Key implementation trick (leveraging existing loader nodes)

Your loader nodes already map `q__*` to Ibis expressions loaded from `DatasetRef` via the Ibis adapter , so schema compilation can:

* fabricate `DatasetRef(table_key=…)` for upstreams
* create empty tables for those table_keys in ephemeral DuckDB
* let loaders produce Ibis tables against the ephemeral DB

That keeps the “DAG wiring” consistent with real execution, which is exactly what you want.

### Code snippet (DESCRIBE)

```python
# src/codeintel/build/schemas/infer_duckdb.py
from __future__ import annotations
import re
from codeintel.core.schemas.primitives import Column, TableSchema
from codeintel.core.schemas.hashing import canonical_type

def _strip_trailing_semicolon(sql: str) -> str:
    return re.sub(r";\s*$", "", sql.strip())

def infer_schema_from_sql(*, con, sql: str, table_schema: str, table_name: str) -> TableSchema:
    sql = _strip_trailing_semicolon(sql)
    rows = con.execute(f"DESCRIBE {sql}").fetchall()
    cols = [Column(name=str(r[0]), type=canonical_type(str(r[1])), nullable=True) for r in rows]
    return TableSchema(schema=table_schema, name=table_name, columns=cols)
```

## Tests checklist (`tests/build/hamilton/`)

* [ ] `test_pr60_infer_schema_describe.py`

  * [ ] create ephemeral duckdb
  * [ ] create empty upstream table
  * [ ] build a simple ibis expression (select + mutate)
  * [ ] infer schema and assert column names/types
* [ ] `test_pr60_provider_hamilton_fallback.py`

  * [ ] provider returns inferred schema for a “native” target
  * [ ] provider returns declared schema for a “wrapper” target

## CLI snapshots

* [ ] None required yet (this PR is compute-heavy).
* (You *can* add a `--infer-native` flag to `schema compile` here, but the drift gate belongs in PR‑63.)

---

# PR‑61 — Pandera-from-TableSchema generator (contracts become schema-authority derived)

## Goal

Stop hand-authoring type schemas twice. Instead:

* `TableSchema` is canonical
* Pandera is derived for validation boundaries (post-write / optional)

You already have Pandera hooks wired (`get_pandera_schema`, `validate_dataframe`)  — this PR makes them **schema-authority driven**.

## Tasks checklist

* [ ] Add generator:

  * [ ] `src/codeintel/core/schemas/pandera_gen.py`

    * duckdb type → pandera dtype mapping
    * nullable handling
* [ ] Update `codeintel.build.hamilton.contracts.pandera_hook`:

  * [ ] if explicit schema exists, use it
  * [ ] else generate from `SchemaProvider.require_table_schema(table_key)`
* [ ] Ensure `--validate-outputs` now validates against generated schema by default

  * (explicit overrides remain allowed for constraints/semantic checks)

## Tests checklist (`tests/build/hamilton/`)

* [ ] `test_pr61_pandera_generated_from_table_schema.py`

  * [ ] generate Pandera schema from TableSchema
  * [ ] validate a DataFrame with matching types
* [ ] `test_pr61_validate_outputs_uses_provider_schema.py`

  * [ ] stub provider + call `get_pandera_schema(table_key)` and ensure non-None

## CLI snapshots

* [ ] None.

---

# PR‑62 — RowBinding-from-schema defaults + remove legacy row bindings

## Goal

Make row bindings a **derived convenience**, not a second schema registry:

* default: generate row model / binding from `TableSchema`
* delete any legacy hand-maintained row-binding catalogs

## Tasks checklist

* [ ] Add row model generator:

  * [ ] `src/codeintel/core/schemas/row_models.py`

    * `row_model_for(table_schema) -> type[dataclass]` (or pydantic model)
    * caching by table_key + schema_hash
* [ ] Update any places that require row bindings:

  * replace “registry lookup” with generator + caching
* [ ] Delete legacy row-binding modules (outside build) after migration

## Tests checklist (`tests/build/hamilton/`)

* [ ] `test_pr62_row_model_generation.py`

  * [ ] generated model has expected fields
  * [ ] stable class name (e.g., `Analytics__risk_factors__Row`)
* [ ] `test_pr62_row_model_cache_keys_on_schema_hash.py`

## CLI snapshots

* [ ] None.

---

# PR‑63 — Schema manifest snapshots + CI drift gate (the “no reconciliation” guarantee)

## Goal

Turn schema inference into a **hard CI gate**:

* Schema manifest is deterministic
* Drift is reviewed like any other code change

This is the step that makes your “no manual reconciliation” ideal real.

## Tasks checklist

* [ ] Extend `build schema compile`:

  * [ ] `--infer-native/--infer` (default on once stable)
  * [ ] `--only-native` (useful for early gating)
  * [ ] `--output -` (stdout) default
  * [ ] `--stable` (forces deterministic ordering + canonical types)
* [ ] Add a “schema diff” mode (optional, but best-in-class):

  * [ ] `codeintel build schema diff --expected path/to/schema_manifest.json`
  * exit code 1 on diff
* [ ] Add CLI snapshot case that prints a small, deterministic manifest:

  * recommend: **native-only** manifest first (smaller surface, less churn)

## Tests checklist (`tests/build/hamilton/`)

* [ ] `test_pr63_schema_manifest_is_stable.py`

  * [ ] compile twice, assert identical JSON
* [ ] (Optional but strong) `test_pr63_schema_compile_exit_codes.py`

  * `schema diff` returns 1 when drift exists, 0 otherwise

## CLI snapshots (`tests/build/hamilton/snapshots/`)

* [ ] Command:

  * `codeintel build schema compile --only-native --format json --stable`
* [ ] Snapshot file:

  * `pr63_schema_manifest_native.json`

**Manifest entry (example):**

```yaml
- name: "pr63_schema_manifest_native"
  tags: ["pr63", "schema", "manifest", "json", "integration"]
  args: ["build", "schema", "compile", "--only-native", "--format", "json", "--stable"]
  kind: "json"
  snapshot: "pr63_schema_manifest_native.json"
```

---

# PR‑64 — Unify node tags (canonical `node_type`) + make graph exports/schema compile consume tags

## Goal

Make the DAG readable and machine-actionable at scale:

* one canonical tag key: `node_type`
* canonical values: `loader.query`, `loader.dataframe`, `compute`, `materialize`, `artifact`, etc.
* schema compilation uses tags to discover “inferable outputs”
* graph exports label nodes using these tags

Your loader nodes already emit `node_type="query"` and `node_type="dataframe"` today   — this PR just makes it consistent everywhere.

## Tasks checklist

* [ ] Create tag constants:

  * [ ] `src/codeintel/build/hamilton/tags.py`
* [ ] Update generated node factory:

  * [ ] ensure all nodes get:

    * `domain`
    * `target` (for t__ nodes)
    * `table_key` (for loader/dataset nodes)
    * `node_type` (canonical string)
* [ ] Update native nodes to match:

  * compute nodes: `node_type="compute"`
  * materializers: `node_type="materialize"`
* [ ] Update observability exporters to include/use tags in output

## Tests checklist (`tests/build/hamilton/`)

* [ ] `test_pr64_all_nodes_have_node_type_tag.py`

  * [ ] build driver, export DAG info, assert node_type present
* [ ] `test_pr64_loader_tags_are_canonical.py`

  * [ ] `q__*` nodes are `loader.query`
  * [ ] `df__*` nodes are `loader.dataframe`

## CLI snapshots

* [ ] Possible snapshot updates if `build graph` output includes new tag fields.

  * If so, add/refresh:

    * `pr64_build_graph_json_small.json` (or update an existing graph snapshot)

---

# PR‑65 — Optional: derive TargetGraph dependencies from Hamilton graph (eliminate dual DAG drift)

## Goal

Remove the last major source of “drift potential”:

* instead of maintaining a TargetGraph separately,
* derive dependency edges and closure ordering from the Hamilton graph (plus tags/contracts)

This makes Hamilton truly “the DAG source of truth.”

## Tasks checklist

* [ ] Add extractor:

  * [ ] `src/codeintel/build/hamilton/introspect.py`

    * list nodes
    * read dependencies
    * filter materializers / targets via tags
* [ ] Add builder:

  * [ ] `TargetGraph.from_hamilton(runtime)` (or a thin adapter)
* [ ] Update planner/graph export to optionally use Hamilton-derived graph:

  * [ ] feature flag: `--graph-source targetgraph|hamilton` (default stays current until proven)
* [ ] Once validated, consider flipping default later.

## Tests checklist (`tests/build/hamilton/`)

* [ ] `test_pr65_hamilton_graph_matches_targetgraph_for_small_subset.py`

  * choose a tiny closure and assert same closure ordering + edges
* [ ] `test_pr65_hamilton_graph_filters_to_targets_only.py`

  * ensure dataset/loader nodes aren’t treated as “targets”

## CLI snapshots

* [ ] None required (optional PR).

---

## Recommended sequencing (keeps every PR mergeable + safe)

1. PR‑56 → PR‑57 → PR‑58 (core authority + storage + fingerprinting)
2. PR‑59 (CLI + manifest format)
3. PR‑60 (inference engine)
4. PR‑61/62 (generated contracts/bindings)
5. PR‑63 (CI drift gate)
6. PR‑64 (tags consistency)
7. PR‑65 (optional: eliminate dual DAG drift)

---

## Two “best-in-class” guardrails to add early (I’d put them in PR‑63)

These make “schemas inferred from DAG” *robust* rather than fragile:

1. **Schema output is sorted + canonicalized**

   * stable column ordering
   * canonical type normalization
   * stable JSON formatting

2. **Fail closed on unknown types (with explicit escape hatch)**

   * if `DESCRIBE` returns a type you can’t map/canonicalize, fail with a clear message
   * allow `--allow-unknown-types` for debugging only

---

If you want, I can also add a “mini seed harness” plan for PR‑60/63 specifically: a deterministic way to seed only the minimal upstream empty tables required for a native target’s compute node, so schema compilation never depends on real data and stays fast in CI.


# mini seed harness #

Below is a concrete “mini seed harness” plan that fits exactly what you want for PR‑60 / PR‑63: **schema compilation that is deterministic, fast, and never depends on real data**, by seeding only the **minimal set of empty upstream tables** needed to type-check/compile a native Ibis compute node and then using DuckDB `DESCRIBE` on the compiled SQL.

This is explicitly aligned with the Phase‑3 native target pattern you already have (e.g. `t__risk_factors__compute(...) -> ir.Table` + `t__risk_factors(...) -> TargetRunRecord`) and the Strategy B approach you chose (compile → `DESCRIBE`). 

---

## What the mini seed harness must guarantee

### Determinism

* Works on **fresh ephemeral DuckDB** (in-memory or temp file).
* Creates **empty tables only** (no row inserts).
* Uses **stable schema sources** (SchemaProvider / schema manifest / declared registry) so DDL and inferred output are identical across runs.

### Minimality

* For a given compute closure, seed **only** the upstream `q__...` dependencies actually required.
* Do not bootstrap the entire dataset registry/schema pack by default.

### Speed

* O(#upstream inputs) DDL statements.
* No I/O, no repo scanning, no external tools.

### Correctness

* Ensures Ibis can “see” input tables with the right column types so compiled SQL is valid.
* Ensures DuckDB can `DESCRIBE (SELECT …)` to return the output types.

---

## Key idea

**Seed empty upstream tables → create Ibis `con.table()` / `gateway.ibis.table()` expressions → execute compute node(s) to get output Ibis expression → compile to SQL → DuckDB `DESCRIBE` to get output schema.**

No real data is needed, because `DESCRIBE` uses **type inference**, not execution results.

---

## Mini seed harness design (production code)

### New module

Create a small, focused helper you can reuse from:

* PR‑60 schema inference engine
* PR‑63 schema manifest CLI + CI gate test

Proposed file:

`src/codeintel/build/schemas/seed_harness.py`

### Core types

```python
# src/codeintel/build/schemas/seed_harness.py
from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from types import ModuleType
from typing import Any, Callable

import ibis.expr.types as ir

from codeintel.config.datasets.primitives import TableSchema
from codeintel.storage.gateway import StorageGateway


class SchemaProvider:
    """Phase 3/4+ provider interface (PR-56) — here just the shape we need."""
    def require_table_schema(self, table_key: str) -> TableSchema:
        raise NotImplementedError


def qparam_to_table_key(qparam: str) -> str:
    """
    Convert q__schema__table_name -> schema.table_name
    Assumes table_key has exactly one schema prefix.
    """
    if not qparam.startswith("q__"):
        msg = f"Expected q__ param, got: {qparam}"
        raise ValueError(msg)
    payload = qparam.removeprefix("q__")
    schema, rest = payload.split("__", 1)
    return f"{schema}.{rest}"


def extract_qparams_from_callable(fn: Callable[..., Any]) -> set[str]:
    out: set[str] = set()
    for name in inspect.signature(fn).parameters.keys():
        if name.startswith("q__"):
            out.add(name)
    return out


def extract_qparams_for_target_module(target: str, mod: ModuleType) -> set[str]:
    """
    Best “no-Hamilton-internals” approach:
    union q__ params across *all* functions in the module that belong to the target.

    This protects you if you later split compute into multiple Hamilton nodes
    (t__<target>__fan_in, t__<target>__join, etc.).
    """
    prefix = f"t__{target}__"
    qparams: set[str] = set()

    for name, obj in vars(mod).items():
        if not callable(obj):
            continue
        if name.startswith(prefix):
            qparams |= extract_qparams_from_callable(obj)

    return qparams


@dataclass
class MiniSeedHarness:
    gateway: StorageGateway
    schema_provider: SchemaProvider

    # Cache: avoid reissuing CREATE TABLE for the same key.
    _seeded: set[str] = field(default_factory=set)

    def ensure_seeded_table(self, table_key: str) -> None:
        """
        Create an empty table with the provider’s schema if it doesn't exist.
        """
        if table_key in self._seeded:
            return

        schema = self.schema_provider.require_table_schema(table_key)

        # Delegate to your canonical DDL builder in PR-57+ (preferred),
        # or implement a tiny DDL generator with sqlglot.
        #
        # Option A (preferred): gateway.policy.ensure_table_from_schema(table_key, schema)
        # Option B: ddl = create_table_sql(schema) and con.execute(ddl)

        self._ensure_table_via_policy(table_key, schema)

        self._seeded.add(table_key)

    def _ensure_table_via_policy(self, table_key: str, schema: TableSchema) -> None:
        """
        Placeholder for the best-in-class implementation:
        PR-57 injects schema_provider into DuckDBPolicyBackend so policy can
        build DDL correctly with quoting, order, constraints, etc.
        """
        # Recommended final shape after PR-57:
        # self.gateway.policy.ensure_table(table_key)
        #
        # If you keep policy.ensure_table(table_key) as the API, then
        # policy should consult self.schema_provider internally.
        self.gateway.policy.ensure_table(table_key)

    def ibis_input(self, qparam: str) -> ir.Table:
        """
        Return an Ibis table expression for a q__... param, seeding the table first.
        """
        table_key = qparam_to_table_key(qparam)
        self.ensure_seeded_table(table_key)
        return self.gateway.ibis.table(table_key)

    def build_inputs(self, qparams: set[str]) -> dict[str, Any]:
        """
        Build driver.execute inputs for all q__ params in the closure.
        """
        return {q: self.ibis_input(q) for q in sorted(qparams)}
```

### Why this structure is “best-in-class”

* The harness is **purely infrastructural**: it doesn’t “know” anything about targets beyond naming conventions.
* It is robust to you refactoring a target module into multiple compute nodes later (because it unions `q__...` across module functions).
* It centralizes all “create empty upstream tables” logic in one place.

---

## Integration point in PR‑60

In PR‑60, you’ll have something like:

* `infer_schema_for_native_target(...) -> TableSchema`
* Under the hood:

  1. Build ephemeral gateway
  2. Build overlay schema provider (declared + already inferred)
  3. Determine qparams needed
  4. Seed minimal upstream empty tables
  5. Execute compute to get Ibis expr
  6. Compile SQL
  7. DuckDB `DESCRIBE` → output schema
  8. Register inferred schema in overlay and create the *output* empty table (optional but recommended)

### PR‑60: recommended inference skeleton

```python
# src/codeintel/build/schemas/infer_ibis_native.py
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

import duckdb
import ibis.expr.types as ir

from codeintel.build.hamilton.driver_factory import build_driver
from codeintel.build.hamilton.native.registry import is_native_target
from codeintel.storage.gateway import open_memory_gateway
from codeintel.storage.gateway_cache import close_gateways

from .seed_harness import MiniSeedHarness, extract_qparams_for_target_module


@dataclass
class InferenceResult:
    table_key: str
    schema: Any  # TableSchema
    sql: str


def infer_schema_for_target(
    *,
    target: str,
    table_key: str,
    schema_provider: Any,   # SchemaProvider (overlay)
) -> InferenceResult:
    # 1) Minimal ephemeral gateway (no full schema bootstrap)
    gateway = open_memory_gateway(apply_schema=False, validate_schema=False, ensure_views=False)
    try:
        # 2) Load native module (from unified registry in your real code)
        # Here: assume native module path is known. In your PR-60 code,
        # fetch it from the unified registry registration for this target.
        native_mod = import_module(f"codeintel.build.hamilton.native.analytics.{target}")

        # 3) Determine qparams needed for this target module
        qparams = extract_qparams_for_target_module(target, native_mod)

        # 4) Seed minimal upstream empty tables and create Ibis inputs
        harness = MiniSeedHarness(gateway=gateway, schema_provider=schema_provider)
        q_inputs = harness.build_inputs(qparams)

        # 5) Execute compute node to get Ibis expression
        runtime = build_driver(mode="auto")
        compute_node = f"t__{target}__compute"

        # IMPORTANT: you pass q__ inputs so Hamilton doesn't try to resolve loaders/datasets.
        out = runtime.dr.execute([compute_node], inputs=q_inputs)
        expr = out[compute_node]
        if not isinstance(expr, ir.Table):
            raise TypeError(f"{compute_node} returned {type(expr)}; expected ibis Table")

        # 6) Compile to SQL
        sql = gateway.ibis.compile(expr)  # adapt to your ibis adapter surface

        # 7) DESCRIBE to infer output schema
        # inferred = describe_sql_to_table_schema(gateway.con, sql)
        inferred = ...  # PR-60 implement

        return InferenceResult(table_key=table_key, schema=inferred, sql=sql)
    finally:
        gateway.close()
        close_gateways()
```

**Note:** you’ll adjust the exact Ibis compile API (`gateway.ibis.compile(expr)` vs `expr.compile()` vs `backend.compile(expr)`), but the harness logic remains identical.

---

## Mini seed harness plan for PR‑63 (CI + CLI gate)

PR‑63’s goal is: **schema manifest generation and drift gating**, using the same deterministic seeding.

### What PR‑63 should do with the harness

* Run schema inference for a curated subset of Ibis-native targets (or all, if fast enough).
* Write a manifest artifact (YAML or JSON) containing:

  * table_key
  * schema hash
  * columns/types/nullability
  * optional: compiled SQL hash (not the SQL itself, unless you want it)
* CI gate asserts:

  * manifest output matches committed snapshot
  * no “unknown / NULL” types
  * no missing upstream schemas

This is where the “mini seed harness” matters most: **CI can generate schemas without real data**.

---

## PR‑60 / PR‑63 deliverables: tasks, tests, snapshots

### PR‑60: Mini seed harness + deterministic inference tests

#### Tasks checklist

* [ ] Add `src/codeintel/build/schemas/seed_harness.py` implementing:

  * `qparam_to_table_key`
  * `extract_qparams_for_target_module`
  * `MiniSeedHarness.ensure_seeded_table()` (cache + DDL call)
  * `MiniSeedHarness.build_inputs()`
* [ ] Ensure the DDL path is canonical:

  * preferred: `gateway.policy.ensure_table(table_key)` uses injected `schema_provider` (post PR‑57)
* [ ] Add “strict mode” invariants:

  * if a required upstream schema is missing → raise with a message naming:

    * target
    * qparam
    * table_key
* [ ] Add a small debug surface:

  * `MiniSeedHarness.seeded_table_keys()` returns sorted list (for logging/tests)

#### Tests checklist (under `tests/build/hamilton/`)

Add:

`tests/build/hamilton/test_pr60_seed_harness.py`

Suggested tests:

1. **Seeds only minimal tables**

* import native module for `risk_factors`
* compute `qparams = extract_qparams_for_target_module("risk_factors", mod)`
* run harness.build_inputs(qparams) (but don’t execute compute)
* assert that only the expected upstream tables exist in DuckDB:

  * `analytics.function_metrics`
  * `graph.call_graph_edges`

2. **Compute compiles on empty tables**

* build driver `mode="auto"`
* call execute on `t__risk_factors__compute` with q__ inputs from harness
* compile SQL
* run `DESCRIBE` and assert expected column names appear (types may be asserted once your describe→TableSchema mapping lands)

You can also do a second target like `subsystems` to ensure the pattern generalizes.

---

### PR‑63: CI gate + CLI snapshot (schema manifest)

#### Tasks checklist

* [ ] In your schema compile CLI implementation, ensure it uses:

  * ephemeral DB
  * schema provider overlay
  * `MiniSeedHarness` for inputs
* [ ] Write schema manifest file (YAML preferred)
* [ ] Add a drift gate test that:

  * runs schema compile for a small deterministic set (e.g. `risk_factors`, `subsystems`, `hotspots`)
  * compares manifest output to a committed golden

#### Tests checklist

Add:

`tests/build/hamilton/test_pr63_schema_manifest_gate.py`

* Creates ephemeral execution
* Calls the schema compile function directly (avoid CLI if you want speed)
* Asserts:

  * manifest has expected targets
  * schema hashes are stable
  * no missing required schemas
  * no NULL/UNKNOWN types

#### Snapshot additions (under `tests/build/hamilton/snapshots/`)

Even if you keep it help-only at first, add at minimum:

* `pr63_schema_help.txt`
* manifest case in `tests/build/hamilton/snapshots/manifest.yaml`:

```yaml
- name: "pr63_schema_help"
  tags: ["pr63", "schema", "text", "tiny"]
  args: ["build", "schema", "--help"]
  exit_code: 0
  snapshot: "pr63_schema_help.txt"
  kind: "text"
```

If you want an **actual manifest-output snapshot** (recommended once the command exists and is deterministic):

* `pr63_schema_manifest_risk_factors.yaml` (or `.json`)
* case (example):

```yaml
- name: "pr63_schema_manifest_risk_factors"
  tags: ["pr63", "schema", "manifest", "yaml", "integration"]
  args: ["build", "schema", "compile", "--targets", "risk_factors", "--format", "yaml"]
  exit_code: 0
  snapshot: "pr63_schema_manifest_risk_factors.yaml"
  kind: "text"
  replace:
    - pattern: "/tmp/[^\\s]+"
      repl: "<TMP>"
```

The whole point of the seed harness is that this case becomes reliable.

---

## Two practical “gotchas” (and how to harden against them)

### 1) Compute nodes that don’t use `q__` inputs

Some of your native compute functions currently take only `env` and then query via `env.gateway` (or do non-Ibis work). Those cannot be inferred via this mechanism.

Best-in-class rule for PR‑60/63 scope:

* Only infer schemas for targets whose compute outputs are `ir.Table` **and** whose compute closure declares its upstream tables via `q__...` inputs.

Add a clear error message:

* “Target X is not schema-inferable: compute closure has no q__ inputs or does not return ir.Table.”

### 2) Ambiguous/NULL output types

DuckDB can sometimes report `NULL`-ish/unknown types if expressions are untyped (e.g., `ibis.literal(None)` without a cast).

Hardening:

* In inference: after `DESCRIBE`, reject any column type that maps to “unknown”.
* In code style: require explicit casts for:

  * empty string literals vs NULL
  * numeric division where integer-vs-float matters
  * CASE expressions where branches differ in type

---

## Summary: what you’ll get if you implement this

* PR‑60: you can infer output table schemas **purely from the DAG** (Ibis compute graph + DuckDB type system) without any stored data.
* PR‑63: you can lock that into CI with a schema manifest drift gate, and it will be **fast and deterministic** because the harness seeds only the minimal empty upstream tables.

