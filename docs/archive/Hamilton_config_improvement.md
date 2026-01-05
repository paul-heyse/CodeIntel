Below is what I’d change **in light of the current `CodeIntel_PostStorageDecoupling` codebase** to (a) get you to a first successful run faster, (b) make schema inference *actually robust* in the presence of loaders/seed suites, and (c) simplify the build layer while staying aligned with your “Hamilton DAG + PyArrow/Polars-first” target architecture.

I’m going to focus on **deltas**—what I’d adjust from the plan/patches you already implemented—based on what’s *actually in your repo right now*.

---

## 0) The most likely root cause of your schema inference failures

### Problem A: inference uses a dummy BuildEnv (repo/commit/path) that breaks any loader-based dependency

In `codeintel/build/schemas/inference_service.py`, schema inference constructs a synthetic env via `_inference_env()`:

* `repo="demo/repo"`
* `commit="deadbeef"`
* `repo_root=Path.cwd()`

That becomes the `env` used when inference executes compute nodes.

This is fatal for any inferable output whose compute graph touches **dataset loaders** (e.g., `load_snapshot_lazyframe` / `load_relation` / `patterns.load_table`) because those loaders resolve snapshot directories using:

* `env.paths.dataset_root_dir`
* `env.commit` or `DatasetRef.commit`

So inference ends up looking for datasets under **snapshot “deadbeef”**, which of course does not exist. Even if you have a **seed suite manifest**, inference overwrites DatasetRef nodes (see below), so you still lose the correct snapshot id.

### Problem B: inference overrides DatasetRef nodes in a way that can defeat seed suite manifests

In `inference_service._infer_job_schema()`:

```py
if job.dataset_refs:
    overrides.update(_dataset_ref_overrides(job=job, env=env))
```

and `_dataset_ref_overrides` uses `env.commit` (currently the dummy “deadbeef” env) as the DatasetRef commit.

Even if your DAG config is correctly seeding datasets from `--seed-suite-manifest ...` (which your executor does in `build/hamilton/executor.py`), inference overrides can stomp those values.

### Problem C: inference currently seeds only `q__*` inputs, not loader nodes

Your `DatasetSeedHarness` is great—but today it only seeds `q__...` params. Inference doesn’t (yet) treat loader/query nodes as seedable “schema-only” nodes, so any compute path that includes a loader will still try to hit disk.

---

## 0.5) Corrections / clarifications from review

These are small but important adjustments to the above plan so the fixes are robust in practice.

* **Loader overrides must be reusable tabular inputs.** `DatasetSeedHarness.seed_table(...)` returns a
  `RecordBatchReader`, which is a one-shot iterator. When you override loader nodes, **convert it once**
  to a `pl.LazyFrame` (or `pa.Table`) and store that in overrides. Do not return the raw reader.
* **Make sure `ci.data_node` survives tag parsing.** If you add a custom tag on saver nodes, add it to
  the tag whitelist/tag-spec parsing so it is not dropped during DAG compilation/tag queries.
* **Loader detection should be tag-driven, not name-driven.** Rely strictly on
  `TAG_NODE_TYPE=loader.query` + `TAG_TABLE_KEY`, and ensure both `load_relation` and
  `patterns.load_table` emit identical loader tags so inference sees them uniformly.
* **Keep dataset outputs non-optional.** If any dataset node currently returns `Optional[LazyFrame]`,
  normalize it at the dataset boundary to an empty typed frame to avoid inference failures.

### Representative patterns (exact calls)

```py
# Inference loader override: cache a reusable LazyFrame
from codeintel.build.tabular.conversion import tabular_to_lazyframe

def _loader_overrides(job: _ComputeInferenceJob, harness: DatasetSeedHarness) -> dict[str, object]:
    overrides: dict[str, object] = {}
    for node_name, table_key in job.loader_nodes:
        reader = harness.seed_table(table_key)
        overrides[node_name] = tabular_to_lazyframe(reader)
    return overrides
```

```py
# Tag allowlist: add a custom key once (tag_spec.py)
TagKey = Literal[
    # existing keys...
    "ci.data_node",
]
```

```py
# Loader detection: use tag keys, not name heuristics
from codeintel.core.hamilton import tags as ht

if isinstance(tags, dict) and tags.get(ht.TAG_NODE_TYPE) == ht.NODE_TYPE_LOADER_QUERY:
    table_key = tags.get(ht.TAG_TABLE_KEY)
```

```py
# Dataset outputs must be empty typed frames, not None
import pyarrow as pa
import polars as pl

def empty_like(schema: pa.Schema) -> pl.LazyFrame:
    table = pa.Table.from_batches([], schema=schema)
    return pl.from_arrow(table).lazy()
```

### Checklist (corrections)

- [x] Loader overrides convert `RecordBatchReader` to a reusable `pl.LazyFrame`.
- [x] `ci.data_node` is added to tag allowlists and is preserved by tag parsing.
- [x] Loader detection uses `TAG_NODE_TYPE` + `TAG_TABLE_KEY`, not names.
- [x] Dataset outputs never return `None`; empty typed frames are the default.

---

## 1) P0 fix: make schema inference “env-correct” (and stop using “deadbeef”)

### Change: pass the *real* BuildEnv into schema inference

**Best-in-class**: schema inference should run with the **same snapshot context** as the build run (or the seed-suite snapshot), not a dummy.

Concretely:

* Extend `SchemaInferenceService.infer_table_schema()` and `infer_table_schemas()` to accept `env: BuildEnv | None`.
* If `env` is provided, **don’t call `_inference_env()`**.
* Keep `_inference_env()` only as a fallback for rare “offline inference” workflows.

Representative change:

```py
# inference_service.py
class SchemaInferenceService:
    def infer_table_schema(self, table_key: str, *, env: BuildEnv | None = None) -> TableSchema | None:
        schema_inputs = self._build_inputs(env=env)
        ...

def infer_table_schemas(table_keys, *, schema_inputs: SchemaInferenceInputs, env: BuildEnv | None = None) -> dict[str, TableSchema]:
    ...
    resolved_env = env or _inference_env(gateway=gateway, force_targets=...)
```

Then update `SchemaIndex._resolve_inferred_schema()` (or the provider wrapper) to pass the runtime env when it triggers inference. The cleanest is to have `SchemaIndex` carry an `env_provider: Callable[[], BuildEnv]` or a direct `env: BuildEnv`.

**Why this matters**
It eliminates a whole class of “can’t find snapshot” errors, and it also makes your inference results reflect the *actual profile/variants/settings* used in execution.

---

## 1.5) P0 enhancement: derive snapshot from Git via Dulwich (fallback env)

When a real `BuildEnv` is not available (offline inference or tooling), use Dulwich to
discover repo root + HEAD commit and build a correct `SnapshotRef`. This replaces
the “deadbeef” fallback with a real commit identifier.

### Representative patterns (exact calls)

```py
from pathlib import Path
from dulwich.repo import Repo

def dulwich_snapshot() -> SnapshotRef | None:
    repo = Repo.discover(Path.cwd())
    repo_root = Path(repo.path).resolve()
    commit = repo.head().decode("ascii", errors="ignore")
    return SnapshotRef.from_args(
        repo=repo_root.name or "repo",
        commit=commit,
        repo_root=repo_root,
    )
```

```py
from dulwich import porcelain

def dulwich_is_dirty(repo_root: Path) -> bool:
    status = porcelain.status(repo_root)
    return bool(status.staged or status.unstaged or status.untracked)
```

### Checklist (dulwich environment discovery)

- [x] `Repo.discover(Path.cwd())` used for fallback repo discovery.
- [x] `repo.head()` decoded to a 40‑char commit string for snapshot id.
- [x] `SnapshotRef` built from repo root + commit instead of a dummy fallback.
- [x] Optional: `porcelain.status` used to mark `git_dirty` in run metadata.

Notes:
- Schema inference now fails fast when dulwich cannot resolve a snapshot.

---

## 2) P0 fix: seed *loader nodes* during inference (not just q__ inputs)

This is the single highest-leverage robustness improvement for an inference-driven DAG that also uses P0→P1 seeding.

### Target design goal

> “Schema inference should never require reading datasets from disk; it should run purely from contracts/observations/hints and return empty-but-typed tabular values.”

### What to change

During inference dependency traversal, detect loader nodes (they are already tagged with `node_type = loader.query`) and override them with an **empty typed LazyFrame** (or RecordBatchReader) produced by the seed harness.

You already have all ingredients:

* Loader nodes are tagged via `tag_loader_query` (node_type `loader.query`, plus table_key).
* `DatasetSeedHarness.seed_table(table_key)` can produce an empty `RecordBatchReader` (schema-only) using observed schema or declared schema.
* You have conversion helpers to get a LazyFrame if you want (`tabular_to_lazyframe`).

#### Step 1: extend inference requirements to include loader nodes

Add a field:

```py
@dataclass
class _InferenceRequirements:
    qparams: set[str] = field(default_factory=set)
    dataset_refs: set[str] = field(default_factory=set)
    loader_nodes: set[tuple[str, str]] = field(default_factory=set)  # (node_name, table_key)
    requires_env: bool = False
    requires_catalog: bool = False
```

#### Step 2: detect loader nodes in `_inspect_inference_dependency`

Representative snippet:

```py
from codeintel.core.hamilton import tags as ht

def _inspect_inference_dependency(dep: object) -> _InferenceRequirementsUpdate:
    tags = getattr(dep, "tags", None)
    if isinstance(tags, dict) and tags.get(ht.TAG_NODE_TYPE) == ht.NODE_TYPE_LOADER_QUERY:
        table_key = tags.get(ht.TAG_TABLE_KEY)
        if isinstance(table_key, str) and table_key:
            req = _InferenceRequirements(loader_nodes={(dep.name, table_key)})
            # IMPORTANT: skip children so you don't pull in DatasetRef/target record deps
            return _InferenceRequirementsUpdate(requirements=req, skip_children=True)

    # existing logic for env/catalog/q__/DatasetRef...
```

This is key: **skip children** so inference doesn’t even traverse into dataset_ref/target_record paths for loader nodes.

#### Step 3: override loader nodes at execute time

In `_infer_job_schema()`:

```py
from codeintel.build.tabular.conversion import tabular_to_lazyframe

def _loader_overrides(job: _ComputeInferenceJob, harness: DatasetSeedHarness) -> dict[str, object]:
    overrides: dict[str, object] = {}
    for node_name, table_key in job.loader_nodes:
        reader = harness.seed_table(table_key)
        frame = tabular_to_lazyframe(reader)
        overrides[node_name] = frame  # empty, typed LazyFrame (reusable)
    return overrides
```

Then:

```py
overrides = dict(base_overrides)
overrides.update(_loader_overrides(job, harness))
```

### Why this is best-in-class

* Inference becomes **purely contract-driven** and never touches the filesystem.
* You can infer schemas for downstream/enrichment DAGs **before** any P0 materialization exists on disk.
* Seed suite manifests stop being a brittle runtime-only feature; they integrate into inference cleanly.

---

## 3) P0 fix: stop overriding DatasetRef nodes in ways that break seeding

Once loader nodes are overridden, you can usually remove or heavily narrow DatasetRef overrides.

### Recommendation

* Only override DatasetRef nodes if:

  * they would otherwise depend on a target record (`TargetRunRecord`) or a `t__*` node, **and**
  * you are not already overriding the loader node that consumes them.

But with the loader override pattern above, the cleanest is:

* **Do not traverse into** loader dependencies (skip children).
* Therefore, DatasetRef nodes are no longer in the requirements set for that job.
* Therefore `_dataset_ref_overrides()` is not needed for loader-related cases.

You can keep dataset_ref override logic only for the rare case a compute node directly consumes a DatasetRef without going through a loader (which you can also discourage architecturally).

---

## 4) P0 fix: make `_output_data_node()` unambiguous by tagging the data node name

You currently infer the “output data node” by inspecting dependencies of the saver node (`_output_data_node()` uses a heuristic over dependency tags).

This is a common source of subtle inference breakage when a saver node depends on multiple tabular nodes.

### Change

In `SaveToObjectMetadataDecorator._build_saver_tags()` (in `build/hamilton/save_to.py`), add a tag that explicitly records the data node being materialized:

```py
tags["ci.data_node"] = node_.name
```

Then simplify `_output_data_node()`:

```py
data_node = output.tags.get("ci.data_node")
if isinstance(data_node, str):
    return data_node
```

**Result:** schema inference becomes deterministic and removes a whole class of “picked the wrong dep” failures.

**Note:** add `ci.data_node` to any tag allowlists/TagSpec parsing to ensure it survives DAG compilation.

---

## 5) P0 fix: enforce “never return None for dataset outputs” in inference mode

Inference currently raises if the executed compute returns `None`:

```py
if expr_obj is None:
    raise TypeError(...)
```

That’s good—but you’ll keep tripping over it if some dataset nodes return `Optional[LazyFrame]` for “skips”.

### Best practice

Dataset output nodes should always return an **empty typed frame** rather than `None`.

Where you truly want “skip”, do it at the materialization layer (e.g., saver writes nothing but still returns a result).

Representative helper:

```py
def empty_like(table_key: str) -> pl.LazyFrame:
    schema = get_schema_provider().require_table_schema(table_key)  # or injected schema
    arrow = arrow_schema_from_table_schema(schema)
    return pl.from_arrow(pa.Table.from_batches([], schema=arrow)).lazy()
```

(For build-layer purity, don’t call global schema provider here; prefer passing schemas/hints in or using the harness approach during inference.)

---

## 6) Streamlining: where build is “too big” and what I’d consolidate

Here are the refactors that reduce surface area without sacrificing architecture.

### A) Collapse “dataset_ref + load_relation + patterns.load_table”

You effectively have **two loader systems**:

* “support nodes” (`dataset_ref` + `load_relation`)
* “patterns loaders” (`patterns.load_table`)

Pick one.

**Best-in-class**: keep *one* loader primitive:

* `load_snapshot_lazyframe(env, table_key, snapshot_id)` as the core
* one node factory to generate `l__...` nodes
* remove or deprecate the alternative path

This simplifies:

* inference seeding logic (one place to override)
* runtime debugging (one place to log missing snapshot dirs)
* schema inference tagging (one node_type)

**Also:** ensure the surviving loader path emits `TAG_NODE_TYPE=loader.query` + `TAG_TABLE_KEY`
consistently so inference overrides are uniform.

### B) Remove dead APIs / branches that add optionality

Example: `load_query()` in `patterns/loaders.py` always raises. That’s a maintenance hazard. Delete it and delete its config paths.

### C) Collapse schema “hint/tag” plumbing into one contract surface

Right now schema can come from:

* OUTPUT_TABLE_SCHEMAS
* override registry (DuckDB)
* observation store
* schema output tags

That’s good, but a bit sprawling.

**Streamline** by forcing a single rule:

1. Arrow schema metadata is the “contract artifact”
2. TableSchema is the in-memory representation
3. Schema output tags are a *debugging aid*, not a primary source

So: treat schema output tags as hints only, and keep the canonical contract in Arrow schema metadata / TableSchema providers.

### D) Convert JSON manifest models to msgspec (big code shrink + speed)

Your `core/manifests.py` is dataclass-heavy with hand parsing.

Switch to msgspec:

```py
import msgspec

class DatasetSuiteManifest(msgspec.Struct, frozen=True):
    suite_manifest_version: int
    suite_kind: str
    repo: str
    commit: str
    created_at: str
    dataset_manifest_paths: dict[str, str]
    tool_versions: dict[str, str] | None = None

def load_suite_manifest(path: Path) -> DatasetSuiteManifest:
    return msgspec.json.decode(path.read_bytes(), type=DatasetSuiteManifest)

def dump_suite_manifest(m: DatasetSuiteManifest, path: Path) -> None:
    path.write_bytes(msgspec.json.encode(m))
```

This reduces parsing boilerplate dramatically and prevents “stringly typed” drift.

---

## 7) Intensify Hamilton usage to reduce code and increase stability

### A) Use `@parameterize` for repetitive “extract rows” nodes

Your ingestion modules (SCIP/libcst/treesitter) likely have repeated patterns:

* take a step/result object
* pluck a dataset
* enforce columns
* return LazyFrame

Parameterize those from a table spec list so engineers don’t hand-maintain 7–15 nearly identical functions.

### B) Enforce “LazyFrame-only” in compute nodes, and isolate all `.collect()` to materializers

You already have a nice pattern where ArrowDatasetSaver handles writing and can stream.

Make it a rule:

* dataset nodes return **pl.LazyFrame**
* never `.collect()` in the DAG except in special debug nodes or export/artifact nodes
* this makes schema inference far more reliable (schema-only planning works)

### C) Use Hamilton lifecycle hooks for inference diagnostics

Add a build-mode “graph lint” hook:

* validate every dataset node has:

  * TAG_TABLE_KEY
  * TAG_NODE_TYPE=dataset
  * schema output tag present OR inferred schema available
* validate inferable nodes do not depend on `t__*` unless permitted

This becomes the “fast fail” mechanism before any expensive execution.

### D) Use advanced Hamilton modifiers where they reduce surface area

* **`@with_columns` (polars)** for schema-stable column sub-DAGs; keeps transforms pure and reusable.
* **`pipe_input` / `pipe_output` + `step(...).when(...)`** for config-gated preprocessing without
  wrapper nodes.
* **`@inject`** to wire feature registries into compute nodes without duplicating logic.
* **`parameterize_frame`** only when you accept its experimental API surface.

#### Representative patterns (exact calls)

```py
# with_columns (polars)
import polars as pl
from hamilton.plugins.h_polars import with_columns

def a_plus_b(a: pl.Series, b: pl.Series) -> pl.Series:
    return a + b

@with_columns(
    a_plus_b,
    columns_to_pass=["a", "b"],
    select=["a_plus_b"],
    namespace="features",
)
def enriched(df: pl.DataFrame) -> pl.DataFrame:
    return df
```

```py
# pipe_input + step.when(...) (config-gated preprocess)
from hamilton.function_modifiers import pipe_input, step, value

def drop_nulls(df: pl.LazyFrame) -> pl.LazyFrame:
    return df.drop_nulls()

@pipe_input(step(drop_nulls).when(config_key="ci.enable_drop_nulls"), on_input="frame")
def normalize(frame: pl.LazyFrame) -> pl.LazyFrame:
    return frame
```

```py
# inject (registry-driven wiring)
from hamilton.function_modifiers import inject, source

@inject(features={"loc": source("loc"), "cyclo": source("cyclomatic_complexity")})
def risk_score(features: dict[str, float]) -> float:
    return 0.2 * features["loc"] + 0.8 * features["cyclo"]
```

```py
# parameterize_frame (experimental)
from hamilton.experimental.decorators.parameterize_frame import parameterize_frame

@parameterize_frame(spec_df)
def compute_risks(loc: pl.Series, cyclo: pl.Series, w: float) -> pl.DataFrame:
    return pl.DataFrame({"risk": loc * w + cyclo * (1 - w)})
```

#### Checklist (Hamilton modifiers)

- [x] `with_columns` used for column sub-DAGs that must be schema-stable.
- [x] `pipe_input`/`pipe_output` used for config-gated pre/post transforms.
- [x] `@inject` only used for registry-driven wiring (no implicit hidden deps).
- [x] `parameterize_frame` only used where experimental API risk is acceptable.

---

## 8) Intensify PyArrow + Polars usage for stability and inference

### A) Prefer Arrow schema propagation for “schema-only execution”

When you seed loader nodes with empty typed frames/readers, Polars can compute output schema without scanning data.

### B) Make schema inference resilient to Polars operations that require data

Some Polars expressions can become “needs data to resolve dtype” in edge cases (e.g., user-defined functions, some casts).

Best-in-class mitigation:

* add a small internal helper library of “schema-stable transforms”
* for anything that risks dtype ambiguity, wrap with explicit `.cast(...)` or use schema hints/tagging

### C) Use Arrow dictionary encoding policies consistently

You already have extras policies and schema metadata generation; take the next step:

* encode “dictionary encode recommended” in schema metadata for high-cardinality vs low-cardinality
* let the writer pick dictionary encoding for relevant columns automatically (reduces storage and speeds scans)

### D) Use Polars plan introspection and shared execution

* `LazyFrame.explain()` / `profile()` to debug inference-time schema behavior.
* `polars.collect_all(...)` to share common subplans across related outputs.

### E) Tune Parquet write knobs for downstream scan performance

* Set `row_group_size` and page size on `sink_parquet` to reduce scan overhead.
* Prefer injecting `row_index_name` at scan time instead of post-hoc `with_row_index()`.

#### Representative patterns (exact calls)

```py
# Polars plan introspection + shared execution
plan = lazyframe.explain()
result, profile = lazyframe.profile()
frames = pl.collect_all([lf_a, lf_b, lf_c])
```

```py
# Parquet sink tuning (row groups + page size)
lazyframe.sink_parquet(
    path,
    compression="zstd",
    row_group_size=200_000,
    data_page_size=1_048_576,
    statistics=True,
)
```

```py
# Scan-time row index injection
lf = pl.scan_parquet(path, row_index_name="row_id", row_index_offset=0)
```

#### Checklist (Polars + Arrow)

- [x] `explain()` and `profile()` are used for inference-time diagnostics.
- [x] `collect_all(...)` used where shared subplans reduce work.
- [x] `sink_parquet` uses explicit `row_group_size` and `data_page_size`.
- [x] `scan_parquet(..., row_index_name=...)` used for stable row IDs.

---

## 9) Leverage intervaltree/sortedcontainers where it actually moves the needle

Your “code metadata DB” will live or die on fast span joins:

* SCIP occurrences ↔ tree-sitter nodes ↔ libcst nodes ↔ file text ranges

**Best-in-class** is to normalize everything into a common span key and then use interval joins.

A practical approach:

* Build an `IntervalTree` per `(repo, commit, doc_id)` for “scope spans”
* Query it for each occurrence span to attach scope/ancestor information

Representative snippet:

```py
from intervaltree import IntervalTree

def build_scope_tree(scopes: list[tuple[int, int, dict]]) -> IntervalTree:
    t = IntervalTree()
    for start, end, payload in scopes:
        t.addi(start, end, payload)
    return t

def annotate_occurrences(tree: IntervalTree, occs: list[tuple[int, int]]) -> list[dict]:
    out = []
    for s, e in occs:
        matches = sorted(tree.overlap(s, e), key=lambda iv: (iv.begin, iv.end))
        out.append({"start": s, "end": e, "scopes": [m.data for m in matches]})
    return out
```

This is much faster and cleaner than repeated joins on “closest enclosing node” using purely tabular methods.

**Span policy note:** standardize on half-open `[start, end)` byte spans to match IntervalTree semantics.

### Representative patterns (exact calls)

```py
# Normalize inclusive spans -> half-open spans
def to_half_open(start: int, end_inclusive: int) -> tuple[int, int]:
    return start, end_inclusive + 1
```

```py
from intervaltree import IntervalTree

tree = IntervalTree()
tree.addi(start, end, payload)  # end is exclusive
matches = sorted(tree.overlap(start, end))
```

### Checklist (intervaltree)

- [x] All spans stored as half-open `[start, end)` byte ranges.
- [x] Overlap queries use `tree.overlap(start, end)` with exclusive end.
- [x] Span normalization is centralized (no ad hoc +1 logic in callers).

---

## 10) Pandera: use it to validate **contracts**, not just dataframes

Right now you’re mostly doing custom validators (e.g., `TableSchemaColumnsValidator`).

**Best-in-class** use of Pandera here is:

* compile a Pandera schema from your `TableSchema` contract
* validate on:

  * materialization boundary (before writing)
  * loader boundary (after reading, optional)
* include PK constraints + nullability + dtype checks

Representative pattern:

```py
import pandera as pa
import pandera.polars as pap

def pandera_schema_from_contract(ts: TableSchema) -> pap.DataFrameSchema:
    cols = {}
    for col in ts.columns:
        cols[col.name] = pap.Column(
            dtype=_polars_dtype_from_table_schema(col.type),
            nullable=col.nullable,
            required=True,
        )
    return pap.DataFrameSchema(cols, coerce=False, strict=False)
```

Then in your ArrowDatasetSaver path (or a Hamilton DataValidator), validate **only if enabled** and emit structured failure diagnostics.

**Operational knobs:** use `lazy=True` for aggregated error reports and `strict="filter"` to drop
unexpected columns at boundaries.

### Representative patterns (exact calls)

```py
import pandera.polars as pap

schema = pap.DataFrameSchema(
    cols,
    strict="filter",
    coerce=False,
)
validated = schema.validate(lazyframe, lazy=True)
```

```py
import pandera as pa

@pa.check_types
def run_ingest(frame: pa.typing.polars.LazyFrame[MySchema]) -> pa.typing.polars.LazyFrame[MySchema]:
    return frame
```

### Checklist (pandera)

- [x] Boundary validation uses `lazy=True` with aggregated errors.
- [x] `strict="filter"` used where extra columns should be dropped.
- [x] `check_types` reserved for CLI/test paths (not hot DAG paths).

---

## 11) msgspec + orjson: use them for config/manifests and event payloads

### Where to use msgspec

* suite manifest
* dataset manifest
* schema inference error rows (if you serialize them)
* decision trace / run record artifacts
* JSON Schema export (`msgspec.json.schema(...)`) for contract publishing

### Where to use orjson

* large JSON payloads you don’t want to type strictly (e.g., “extras” blobs)
* logging event serialization (if you emit structured logs)
* deterministic hashing (`OPT_SORT_KEYS`) and JSONL output (`OPT_APPEND_NEWLINE`)

### Representative patterns (exact calls)

```py
import msgspec

class DatasetManifest(msgspec.Struct, frozen=True):
    repo: str
    commit: str
    datasets: dict[str, str]

encoded = msgspec.json.encode(DatasetManifest(repo="r", commit="c", datasets={}))
decoded = msgspec.json.decode(encoded, type=DatasetManifest)
schema = msgspec.json.schema(DatasetManifest)
```

```py
import orjson

payload = {"event": "schema.infer.fail", "table_key": "analytics.graph_metrics"}
blob = orjson.dumps(payload, option=orjson.OPT_SORT_KEYS | orjson.OPT_APPEND_NEWLINE)
```

### Checklist (msgspec + orjson)

- [x] Manifest/config types modeled with `msgspec.Struct`.
- [x] JSON Schema exported with `msgspec.json.schema(...)`.
- [x] Deterministic hashing uses `orjson.OPT_SORT_KEYS`.
- [x] JSONL output uses `orjson.OPT_APPEND_NEWLINE`.

---

## 12) structlog: make debugging inference failures tractable

Inference failures are often “one node in a huge DAG”.

Add structlog and standardize event fields:

* `event="schema.infer.fail"`
* `table_key`
* `data_node`
* `target`
* `dependency_path` (list of node names)
* `mode` (`schema_inference` vs `run`)
* `repo`, `commit`, `run_id`

Then in `SchemaIndex.record_inference_error(...)`, log via structlog as well as storing rows.

This makes “why did inference fail?” diagnosable in seconds.

**Recommended config:** `BytesLoggerFactory` + `JSONRenderer(serializer=orjson.dumps)` with
`contextvars.merge_contextvars` early in the processor chain.

### Representative patterns (exact calls)

```py
import logging
import orjson
import structlog

structlog.configure(
    cache_logger_on_first_use=True,
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.JSONRenderer(serializer=orjson.dumps),
    ],
    logger_factory=structlog.BytesLoggerFactory(),
)

log = structlog.get_logger()
log.info("schema.infer.fail", table_key="analytics.graph_metrics", target="graph_metrics")
```

### Checklist (structlog)

- [x] `contextvars.merge_contextvars` first in processor chain.
- [x] `JSONRenderer(serializer=orjson.dumps)` for fast structured output.
- [x] `BytesLoggerFactory` used for atomic JSON emission.
- [x] Error events include `table_key`, `data_node`, `target`, `repo`, `commit`, `run_id`.

---

## Consolidated checklist

- [x] Inference runs with real `BuildEnv` (no dummy repo/commit).
- [x] Dulwich fallback builds `SnapshotRef` from repo root + HEAD commit when needed.
- [x] Loader overrides convert `RecordBatchReader` to a reusable `pl.LazyFrame`.
- [x] Loader nodes are detected by `TAG_NODE_TYPE=loader.query` + `TAG_TABLE_KEY`.
- [x] Loader overrides skip children to avoid `DatasetRef`/target record traversal.
- [x] DatasetRef overrides are only used for direct consumers (rare).
- [x] Dataset outputs return empty typed frames, never `None`.
- [x] `ci.data_node` tag added to saver nodes and tag allowlists.
- [x] `_output_data_node()` uses the `ci.data_node` tag directly.
- [x] `with_columns` used for schema-stable column sub-DAGs.
- [x] `pipe_input`/`pipe_output` used for config-gated transforms.
- [x] `@inject` used only for registry-driven wiring.
- [x] `parameterize_frame` used only when experimental API risk is acceptable.
- [x] `LazyFrame.explain()`/`profile()` used for inference diagnostics.
- [x] `collect_all(...)` used for shared subplans across outputs.
- [x] `sink_parquet` uses explicit `row_group_size` and `data_page_size`.
- [x] `scan_parquet(..., row_index_name=...)` used for stable row IDs.
- [x] Span policy standardized to half-open `[start, end)` byte ranges.
- [x] Interval lookups use `tree.overlap(start, end)` with exclusive end.
- [x] Pandera validation uses `lazy=True` with aggregated error reports.
- [x] Pandera uses `strict="filter"` where extra columns should be dropped.
- [x] `check_types` reserved for CLI/test paths only.
- [x] Manifests/config modeled with `msgspec.Struct`.
- [x] JSON Schema exported via `msgspec.json.schema(...)`.
- [x] Deterministic hashing uses `orjson.OPT_SORT_KEYS`.
- [x] JSONL output uses `orjson.OPT_APPEND_NEWLINE`.
- [x] structlog uses `contextvars.merge_contextvars` first in processor chain.
- [x] structlog uses `JSONRenderer(serializer=orjson.dumps)` + `BytesLoggerFactory`.
- [x] Error events include `table_key`, `data_node`, `target`, `repo`, `commit`, `run_id`.

---

# What I would do first (ordered, practical)

1. **Pass real BuildEnv into inference** (stop “deadbeef”).
2. **Override loader nodes during inference** using `DatasetSeedHarness.seed_table(...)` and `tabular_to_lazyframe(...)`.
3. **Ensure loader overrides are reusable** (do not pass raw `RecordBatchReader` into the DAG).
4. Add `ci.data_node` tag on saver nodes and simplify `_output_data_node()` selection.
5. Tighten the rule: dataset nodes should never return `None`; use empty typed frames.
6. Add a CLI “inferability inventory” report that prints *exactly why* tables are non-inferable and which nodes caused it.
7. Only then: start codebase slimming (merge loader systems, msgspec manifests, remove dead APIs).

If you want, I can also outline a **very specific “schema inference debug checklist”** for your engineers (what to log, which nodes to inspect, and how to reproduce a failing table_key inference in isolation with a tiny script) based on the above changes.
