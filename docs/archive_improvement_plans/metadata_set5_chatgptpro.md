You can bolt these three datasets cleanly into your existing pipeline with the same patterns you’re already using for `function_metrics`, `coverage_functions`, `config_values`, etc.

I’ll break it into:

1. `analytics.data_models` – extract structured data models (dataclasses, Pydantic, TypedDicts, ORMs).
2. `analytics.data_model_usage` – where those models are created/read/updated/etc.
3. `analytics.config_data_flow` – function‑level config usage + call‑chain context.
4. Pipeline wiring & docs views.

---

## 1. `analytics.data_models` – extracting the models

### 1.1 Table schema

Add a new table definition to `config/schemas/tables.py` under the `analytics` schema.

```python
"analytics.data_models": TableSchema(
    schema="analytics",
    name="data_models",
    columns=[
        Column("repo", "VARCHAR", nullable=False),
        Column("commit", "VARCHAR", nullable=False),

        Column("model_id", "VARCHAR", nullable=False),         # stable hash
        Column("goid_h128", "DECIMAL(38,0)"),                  # class GOID when applicable

        Column("model_name", "VARCHAR", nullable=False),       # e.g. "User"
        Column("module", "VARCHAR", nullable=False),           # e.g. "app.models.user"
        Column("rel_path", "VARCHAR", nullable=False),         # repo-relative path

        Column("model_kind", "VARCHAR", nullable=False),       # dataclass, pydantic_model, typeddict, protocol, django_model, sqlalchemy_model, attrs_class, etc.
        Column("base_classes_json", "JSON"),                   # list of {"name": ..., "qualname": ...}

        Column("fields_json", "JSON", nullable=False),         # list[FieldSpec]
        Column("relationships_json", "JSON"),                  # list[RelationshipSpec]

        Column("doc_short", "VARCHAR"),
        Column("doc_long", "VARCHAR"),

        Column("created_at", "TIMESTAMP", nullable=False),
    ],
    primary_key=("repo", "commit", "model_id"),
    description="Extracted data models (dataclasses, Pydantic, TypedDicts, ORMs) and their fields/relationships.",
)
```

Field JSON shapes (document in comments / README):

```json
// fields_json
[
  {
    "name": "id",
    "type": "int",
    "required": true,
    "has_default": false,
    "default_expr": null,
    "constraints": {
      "gt": 0,
      "max_length": null,
      "regex": null
    },
    "source": "annotation | dataclass_field | pydantic_field | typed_dict"
  }
]

// relationships_json
[
  {
    "field": "user",
    "target_model_id": "abcd1234ef567890",
    "target_model_name": "User",
    "multiplicity": "one"     // "one" | "many"
    "kind": "reference",      // "reference" | "foreign_key" | "backref"
    "via": "annotation"       // "annotation" | "orm_foreign_key" | "orm_relationship"
  }
]
```

### 1.2 Config model

In `config/models.py` add:

```python
@dataclass(frozen=True)
class DataModelsConfig:
    repo: str
    commit: str

    @classmethod
    def from_paths(cls, *, repo: str, commit: str) -> "DataModelsConfig":
        return cls(repo=repo, commit=commit)
```

### 1.3 Implementation: `analytics/data_models.py`

Create `analytics/data_models.py`. This will:

* Walk class definitions via AST/CST.
* Use `core.goids` to attach GOIDs.
* Look for dataclass/Pydantic/TypedDict/ORM patterns.
* Extract fields & relationships.

Imports (roughly):

```python
from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime

import duckdb

from codeintel.config.models import DataModelsConfig
from codeintel.config.schemas.sql_builder import ensure_schema

log = logging.getLogger(__name__)
```

#### 1.3.1 Find candidate classes

Use `core.ast_nodes` + `core.modules`:

```sql
SELECT
  a.path AS rel_path,
  a.qualname,
  a.name      AS class_name,
  a.lineno    AS start_line,
  a.end_lineno AS end_line,
  m.module,
  m.repo,
  m.commit,
  a.decorators,
  a.docstring
FROM core.ast_nodes a
JOIN core.modules m
  ON m.path = a.path
WHERE a.node_type = 'ClassDef';
```

You already store decorators & docstrings in `ast_nodes`.

For each row:

* Determine **model_kind** by:

  * Dataclass:

    * `@dataclass` or `@dataclasses.dataclass` in `decorators` list.

  * Pydantic model:

    * Base classes include `BaseModel`, `pydantic.BaseModel`, etc.
    * You can get bases from the CST (if needed) or add them to `ast_nodes` later; otherwise you can inspect `stmts_json` for class definition; but you already have class AST; easiest is to re‑parse the file AST in Python.

  * TypedDict:

    * Base class `TypedDict` / `typing.TypedDict` / `typing_extensions.TypedDict`.

  * Protocol:

    * Base class `Protocol` / `typing.Protocol`.

  * ORM model (non‑framework‑specific initial pass):

    * Base name `Model` and module tagged `infra`/`db`, or inherits from `Base` imported from `sqlalchemy.orm`.

Focus first on **dataclass**, **pydantic_model**, **typeddict**, **protocol**; treat others as `generic_class` until you refine.

Retrieve `goid_h128` for each class via `core.goids` (matching `rel_path`, `kind='class'`, `qualname`, `start_line`):

```sql
SELECT goid_h128
FROM core.goids
WHERE rel_path = ?
  AND kind = 'class'
  AND qualname = ?
```

#### 1.3.2 Compute `model_id`

`model_id` should be stable and not rely on the DB primary key. Simple pattern:

```python
def compute_model_id(repo: str, commit: str, module: str, qualname: str, model_kind: str) -> str:
    raw = f"{repo}:{commit}:{module}:{qualname}:{model_kind}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
```

#### 1.3.3 Extract fields

You can lean on:

* `core.ast_nodes` for class body statements; or
* Re‑parsing the module with `ast` / LibCST and using existing helpers in `ingestion/ast_utils.py`.

Heuristics:

* For **dataclasses**:

  * Look at class‑level assignments with annotations:

    ```python
    class User:
        id: int
        name: str = "anon"
        age: int = field(default=0)
    ```

  * For each annotated assignment:

    * `name` from target id.
    * `type` from annotation string (you already store param types as strings in `function_types`; reuse same AST→string helper).
    * `default` if the value isn’t `MISSING`:

      * Constant values (`"foo"`, `42`, `True`) → serialize directly.
      * `field(...)` calls → `has_default=True`, `constraints` from keyword args.

* For **Pydantic models**:

  * Class attributes with annotations:

    ```python
    class User(BaseModel):
        id: int
        name: str = Field(..., max_length=50)
    ```

  * For `Field(...)`:

    * `required = arg == Ellipsis` or no default.
    * `constraints` from known kwargs: `gt`, `lt`, `min_length`, `max_length`, `regex`, `default_factory`, etc.

* For **TypedDict**:

  * Class body entries: `x: int`, `y: str`.
  * Required vs optional:

    * For `class User(TypedDict, total=False)`: all fields optional.
    * For `total` kwarg at class base call.

* For **Protocol**:

  * Focus on method signatures (fields_json could be empty or include attributes; up to you).

Emit fields as described above.

#### 1.3.4 Extract relationships

Two simple passes:

1. **Annotation‑based**:

   * Gather all `model_name`s of data models in this repo into a set, keyed by `(module, class_name)` and by `class_name` alone.
   * For each field type string:

     * If `type == "OtherModel"` or looks like `list[OtherModel]` / `Sequence[OtherModel]` / `Optional[OtherModel]` where `OtherModel` is a known model:

       * `multiplicity = "one"` or `"many"` depending on container.
       * `kind = "reference"`.
       * `target_model_id` from lookup.

2. **ORM‑style relationships** (optional first pass):

   * In class body, detect:

     * `ForeignKey("other_table")` calls in ORMs.
     * `relationship("OtherModel")` from SQLAlchemy.

   * Map `"OtherModel"` → target model if present; mark `kind="foreign_key"` or `"relationship"`.

Append these to `relationships_json`.

#### 1.3.5 Docs & base classes

* `doc_short` / `doc_long`:

  * Use `core.docstrings` (class docstrings) where available, matching `(rel_path, qualname)` to get structured doc text.

* `base_classes_json`:

  * Extract base class names/qualnames from AST for context (helpful for ORMs or frameworks).

#### 1.3.6 Insert into DuckDB

Standard pattern:

```python
def compute_data_models(con: duckdb.DuckDBPyConnection, cfg: DataModelsConfig) -> None:
    ensure_schema(con, "analytics.data_models")
    con.execute(
        "DELETE FROM analytics.data_models WHERE repo = ? AND commit = ?",
        [cfg.repo, cfg.commit],
    )

    # ... build list of rows ...
    rows = [
        (
            repo, commit,
            model_id, goid_h128,
            model_name, module, rel_path,
            model_kind, json.dumps(base_classes),
            json.dumps(fields), json.dumps(rels),
            doc_short, doc_long,
            datetime.now(tz=UTC),
        )
        for ...
    ]

    con.executemany(
        """
        INSERT INTO analytics.data_models (
            repo, commit,
            model_id, goid_h128,
            model_name, module, rel_path,
            model_kind, base_classes_json,
            fields_json, relationships_json,
            doc_short, doc_long,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
```

---

## 2. `analytics.data_model_usage` – how models flow through functions

### 2.1 Table schema

Add another table to `tables.py`:

```python
"analytics.data_model_usage": TableSchema(
    schema="analytics",
    name="data_model_usage",
    columns=[
        Column("repo", "VARCHAR", nullable=False),
        Column("commit", "VARCHAR", nullable=False),

        Column("model_id", "VARCHAR", nullable=False),
        Column("function_goid_h128", "DECIMAL(38,0)", nullable=False),

        Column("usage_kinds_json", "JSON", nullable=False),  # e.g. ["create","read","serialize"]
        Column("evidence_json", "JSON"),                     # examples of code locations
        Column("context_json", "JSON"),                      # entrypoint/subsystem info

        Column("created_at", "TIMESTAMP", nullable=False),
    ],
    primary_key=("repo", "commit", "model_id", "function_goid_h128"),
    description="Per function/model usage summary and context (CRUD/validate/serialize).",
)
```

We aggregate **all** usage kinds per (model,function) into one row, not one row per kind.

### 2.2 Config model

```python
@dataclass(frozen=True)
class DataModelUsageConfig:
    repo: str
    commit: str
    max_examples_per_usage: int = 5

    @classmethod
    def from_paths(cls, *, repo: str, commit: str) -> "DataModelUsageConfig":
        return cls(repo=repo, commit=commit)
```

### 2.3 Implementation: `analytics/data_model_usage.py`

This module will combine:

* `analytics.data_models` (model_id, goid_h128, model_name, module).
* `core.goids` + `analytics.function_metrics` to map functions to files/modules.
* AST/DFG/CFG for usage patterns:

  * `graph.dfg_edges` / `graph.cfg_blocks.stmts_json`.
* Call graph (`graph.call_graph_edges`) to understand transitive flows (optional).
* Subsystems / entrypoints when available:

  * `analytics.subsystems` + `analytics.subsystem_modules`.

Signature:

```python
def compute_data_model_usage(
    con: duckdb.DuckDBPyConnection,
    cfg: DataModelUsageConfig,
) -> None:
    ...
```

#### 2.3.1 Direct usage heuristics per function

For each function GOID:

1. **Creation**:

   * If AST shows calls to the model’s class:

     * via call graph: `graph.call_graph_edges` where `caller_goid_h128 = function_goid` and `callee_goid_h128` corresponds to the model’s class constructor GOID.
     * Or via unresolved calls where callee text equals the model name, using `evidence_json` from `call_graph_edges`.

   * Mark `"create"`.

2. **Reads**:

   * For parameters annotated as `ModelType` (via `analytics.function_types.param_types`), any attribute access `param.field` is a read.
   * DFG evidence: `graph.dfg_edges` where `src_symbol` maps to model instance variable and is used at multiple points.

   Detect `Attribute` nodes or subscript access on variables that we know are model instances.

3. **Updates**:

   * Attribute assignments `instance.field = ...`.
   * Calls to mutating methods `.update(...)`, `.append(...)` on containers of models (less critical).

4. **Deletes**:

   * Calls to `session.delete(model)` (ORM semantics).
   * `del instance`.

5. **Serialize**:

   * Pydantic: `.dict()`, `.json()`.
   * Dataclasses: `asdict(model)`, `astuple(model)`.
   * Passing model instance into `json.dumps` or logging frameworks.

6. **Validate**:

   * Pydantic: `Model.parse_obj`, `Model.validate`, or model functions where role is `validator` (if you later add semantic roles).
   * Dataclasses: custom validators (harder; you can skip for now).

For each `(model_id, function_goid)` accumulate a `set` of usage kinds.

**Where do we get “model instance variables”?**

* **Parameters**: param types from `function_types` that match a `model_name`.
* **Local variables**: results of `Model(...)` calls we detect in the AST.

You already have AST span and statement serialization; reuse helpers for `stmts_json` parsing from CFG builder to avoid re‑parsing files from scratch.

#### 2.3.2 Evidence JSON

Collect a small sample (cap at `max_examples_per_usage`):

```json
{
  "create": [
    {"rel_path": "src/app/service.py", "line": 42, "snippet": "User(id=user_id)"},
    ...
  ],
  "read": [
    {"rel_path": "...", "line": 88, "snippet": "user.name"},
    ...
  ],
  "update": [...]
}
```

This is extremely helpful for an LLM to justify why it thinks the usage is “update” vs “read”.

#### 2.3.3 Context JSON

Context can encode:

* **Subsystem**:

  * Map function → module via `function_metrics.rel_path` → `core.modules.module`.
  * Join module → subsystem via `analytics.subsystem_modules`.
  * Pull `subsystem_id`, `subsystem_name`.

* **Entrypoint role** (optional; you might already have this if you build semantic roles later):

  * Use `tags` on modules (`api`, `cli`, `infra`) from `core.modules.tags`.
  * Or use heuristics: if module tag includes `"api"` → `"api_entrypoint"`, `"cli"` → `"cli_entrypoint"`, tests → `"test"`.

Shape:

```json
{
  "module": "app.api.users",
  "subsystem_id": "abc123",
  "subsystem_name": "app.api",
  "entrypoint_role": "api_handler"   // if available
}
```

#### 2.3.4 Insert rows

Same pattern:

```python
ensure_schema(con, "analytics.data_model_usage")
con.execute(
    "DELETE FROM analytics.data_model_usage WHERE repo = ? AND commit = ?",
    [cfg.repo, cfg.commit],
)

rows = [
    (
        repo, commit,
        model_id, function_goid_h128,
        json.dumps(sorted(list(usage_kinds))),
        json.dumps(evidence),
        json.dumps(context),
        datetime.now(tz=UTC),
    )
    for ...
]

con.executemany(
    """
    INSERT INTO analytics.data_model_usage (
        repo, commit,
        model_id, function_goid_h128,
        usage_kinds_json, evidence_json, context_json,
        created_at
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """,
    rows,
)
```

---

## 3. `analytics.config_data_flow` – function‑level config usage & call chains

You already have `analytics.config_values` that map config keys to **files/modules**.

We’ll refine that to **functions** + optional call chains.

### 3.1 Table schema

Add to `tables.py`:

```python
"analytics.config_data_flow": TableSchema(
    schema="analytics",
    name="config_data_flow",
    columns=[
        Column("repo", "VARCHAR", nullable=False),
        Column("commit", "VARCHAR", nullable=False),

        Column("config_key", "VARCHAR", nullable=False),
        Column("config_path", "VARCHAR", nullable=False),

        Column("function_goid_h128", "DECIMAL(38,0)", nullable=False),
        Column("usage_kind", "VARCHAR", nullable=False),       # "read", "conditional_branch", "logging", "write", ...
        Column("evidence_json", "JSON"),

        Column("call_chain_id", "VARCHAR"),                    # may be null if not computed
        Column("call_chain_json", "JSON"),                     # list of GOIDs/URNs to this function

        Column("created_at", "TIMESTAMP", nullable=False),
    ],
    primary_key=("repo", "commit", "config_key", "config_path", "function_goid_h128", "usage_kind", "call_chain_id"),
    description="Function-level config key usage and call-chain context from entrypoints.",
)
```

### 3.2 Config model

```python
@dataclass(frozen=True)
class ConfigDataFlowConfig:
    repo: str
    commit: str
    max_paths_per_usage: int = 3
    max_path_length: int = 10

    @classmethod
    def from_paths(cls, *, repo: str, commit: str) -> "ConfigDataFlowConfig":
        return cls(repo=repo, commit=commit)
```

### 3.3 Implementation: `analytics/config_data_flow.py`

Signature:

```python
def compute_config_data_flow(
    con: duckdb.DuckDBPyConnection,
    cfg: ConfigDataFlowConfig,
) -> None:
    ...
```

This module will:

1. Map **config keys → files** via `analytics.config_values`.
2. Within each file, identify **functions** and **expressions** that use each key.
3. Classify `usage_kind` per `(key,function)`.
4. Optionally compute **call chains** from entrypoints to those functions via the call graph.

#### 3.3.1 Map keys to candidate functions

Start with:

```sql
SELECT
  cv.config_path,
  cv.key           AS config_key,
  path             AS rel_path,
  reference_modules
FROM analytics.config_values cv
UNNEST(cv.reference_paths) AS t(path)
WHERE cv.repo = ? AND cv.commit = ?;
```

This tells you which files reference each key.

Then join to functions in those files:

```sql
SELECT
  cv.config_path,
  cv.config_key,
  fm.function_goid_h128,
  fm.rel_path,
  fm.start_line,
  fm.end_line
FROM (
  SELECT config_path, key AS config_key, path AS rel_path
  FROM analytics.config_values cv
  UNNEST(cv.reference_paths) AS t(path)
  WHERE cv.repo = ? AND cv.commit = ?
) cv
JOIN analytics.function_metrics fm
  ON fm.repo = ? AND fm.commit = ?
 AND fm.rel_path = cv.rel_path;
```

This gives you all functions in every file that references the config key.

#### 3.3.2 AST‑level usage classification

For each `(config_key, rel_path, function)` triple:

* Parse the function body AST using `graph.cfg_blocks.stmts_json` or using the existing AST utilities (`AstSpanIndex`).

* Look for string literals equal to the key, and canonical usage patterns:

  1. **Simple read** (`usage_kind = "read"`):

     * `settings["feature_flag_X"]`
     * `settings.get("feature_flag_X", ...)`
     * `os.getenv("FEATURE_FLAG_X")`
     * `config("feature_flag_X")`

  2. **Conditional branch** (`"conditional_branch"`):

     * Config read appears in `if`/`elif`/`while` conditions:

       ```python
       if feature_flag_X:
           ...
       if config.get("feature_flag_X"):
           ...
       ```

  3. **Logging** (`"logging"`):

     * Config value is passed into a logging call:

       ```python
       logger.info("flag %s", settings["feature_flag_X"])
       ```

  4. **Write / override** (`"write"`):

     * Assignments into config structures, environment, etc.:

       ```python
       settings["feature_flag_X"] = False
       os.environ["FEATURE_FLAG_X"] = "0"
       ```

If you can’t robustly classify the expression, you can at least mark `"read"`.

Collect a few example locations into `evidence_json` per usage_kind:

```json
{
  "read": [
    {"line": 42, "snippet": "os.getenv('FEATURE_FLAG_X')"}
  ],
  "conditional_branch": [
    {"line": 75, "snippet": "if feature_flags['FEATURE_FLAG_X']:"}
  ]
}
```

Emit one row per usage_kind.

#### 3.3.3 Call chains from entrypoints

Optional but powerful.

1. **Build call graph**:

   * Use the existing NetworkX call graph overlay you already use for graph metrics: a `DiGraph` over `graph.call_graph_edges` keyed by `function_goid_h128`.

2. **Identify entrypoints**:

   Heuristics:

   * Functions in modules tagged `"cli"` or `"api"` via `core.modules.tags`.
   * Top‑level `main` functions in `cli` modules.
   * Any other role classification you add later (e.g., from `semantic_roles_functions`).

   Collect them into a set `entry_goids`.

3. **For each config usage function `f`**, find up to `cfg.max_paths_per_usage` simple paths from any entrypoint to `f`:

   * Use reversed call graph (edges `callee -> caller`) and BFS/DFS bounded by `cfg.max_path_length`.
   * For each path `[entry, ..., f]`:

     * `call_chain_id = sha1(f"{repo}:{commit}:{config_key}:{path_goids}")[:16]`
     * `call_chain_json = [list of URNs or GOIDs]`.

   If no path found, leave `call_chain_id`/`call_chain_json` null; the function is still a “config reader” but not reachable from a known entrypoint.

Emit one row per `(config_key, config_path, function_goid, usage_kind, call_chain_id)`.

---

## 4. Wiring into your pipeline

### 4.1 Orchestration steps

In `orchestration/steps.py`, add three new steps.

```python
from codeintel.analytics.data_models import compute_data_models
from codeintel.analytics.data_model_usage import compute_data_model_usage
from codeintel.analytics.config_data_flow import compute_config_data_flow
from codeintel.config.models import DataModelsConfig, DataModelUsageConfig, ConfigDataFlowConfig
```

Then:

```python
@dataclass
class DataModelsStep:
    name: str = "data_models"
    deps: Sequence[str] = ("ast", "goids", "docstrings")  # class defs + docstrings + GOIDs

    def run(self, ctx: PipelineContext, con: duckdb.DuckDBPyConnection) -> None:
        _log_step(self.name)
        cfg = DataModelsConfig.from_paths(repo=ctx.repo, commit=ctx.commit)
        compute_data_models(con, cfg)


@dataclass
class DataModelUsageStep:
    name: str = "data_model_usage"
    deps: Sequence[str] = ("data_models", "callgraph", "cfg", "dfg", "function_analytics")

    def run(self, ctx: PipelineContext, con: duckdb.DuckDBPyConnection) -> None:
        _log_step(self.name)
        cfg = DataModelUsageConfig.from_paths(repo=ctx.repo, commit=ctx.commit)
        compute_data_model_usage(con, cfg)


@dataclass
class ConfigDataFlowStep:
    name: str = "config_data_flow"
    deps: Sequence[str] = ("config_ingest", "callgraph", "function_analytics")

    def run(self, ctx: PipelineContext, con: duckdb.DuckDBPyConnection) -> None:
        _log_step(self.name)
        cfg = ConfigDataFlowConfig.from_paths(repo=ctx.repo, commit=ctx.commit)
        compute_config_data_flow(con, cfg)
```

Add these steps into your step graph after the prerequisites.

### 4.2 Prefect tasks

In `orchestration/prefect_flow.py` define tasks:

```python
@task(name="data_models", retries=1, retry_delay_seconds=2)
def t_data_models(repo: str, commit: str, db_path: Path) -> None:
    con = _connect(db_path)
    cfg = DataModelsConfig.from_paths(repo=repo, commit=commit)
    compute_data_models(con, cfg)
    con.close()


@task(name="data_model_usage", retries=1, retry_delay_seconds=2)
def t_data_model_usage(repo: str, commit: str, db_path: Path) -> None:
    con = _connect(db_path)
    cfg = DataModelUsageConfig.from_paths(repo=repo, commit=commit)
    compute_data_model_usage(con, cfg)
    con.close()


@task(name="config_data_flow", retries=1, retry_delay_seconds=2)
def t_config_data_flow(repo: str, commit: str, db_path: Path) -> None:
    con = _connect(db_path)
    cfg = ConfigDataFlowConfig.from_paths(repo=repo, commit=commit)
    compute_config_data_flow(con, cfg)
    con.close()
```

Wire them into the flow after their dependencies (callgraph, cfg/dfg, config, etc.).

### 4.3 Export mappings

In `docs_export/export_jsonl.py` and `export_parquet.py`, register the new datasets:

```python
JSONL_DATASETS.update({
    "analytics.data_models": "data_models.jsonl",
    "analytics.data_model_usage": "data_model_usage.jsonl",
    "analytics.config_data_flow": "config_data_flow.jsonl",
})

PARQUET_DATASETS.update({
    "analytics.data_models": "data_models.parquet",
    "analytics.data_model_usage": "data_model_usage.parquet",
    "analytics.config_data_flow": "config_data_flow.parquet",
})
```

### 4.4 Docs views for agents

In `storage/views.py`, add convenience views:

**Data model catalog:**

```sql
CREATE OR REPLACE VIEW docs.v_data_models AS
SELECT
  dm.repo,
  dm.commit,
  dm.model_id,
  dm.goid_h128,
  dm.model_name,
  dm.module,
  dm.rel_path,
  dm.model_kind,
  dm.fields_json,
  dm.relationships_json,
  dm.doc_short,
  dm.doc_long,
  dm.created_at
FROM analytics.data_models dm;
```

**Data model usage:**

```sql
CREATE OR REPLACE VIEW docs.v_data_model_usage AS
SELECT
  u.repo,
  u.commit,
  u.model_id,
  dm.model_name,
  dm.model_kind,
  u.function_goid_h128,
  fp.qualname        AS function_qualname,
  fp.rel_path        AS function_rel_path,
  fp.risk_score,
  fp.coverage_ratio,
  u.usage_kinds_json,
  u.context_json,
  u.evidence_json,
  u.created_at
FROM analytics.data_model_usage u
LEFT JOIN analytics.data_models dm
  ON dm.repo = u.repo AND dm.commit = u.commit AND dm.model_id = u.model_id
LEFT JOIN analytics.function_profile fp
  ON fp.repo = u.repo AND fp.commit = u.commit AND fp.function_goid_h128 = u.function_goid_h128;
```

**Config data flow:**

```sql
CREATE OR REPLACE VIEW docs.v_config_data_flow AS
SELECT
  c.repo,
  c.commit,
  c.config_key,
  c.config_path,
  c.function_goid_h128,
  fp.qualname        AS function_qualname,
  fp.rel_path        AS function_rel_path,
  fp.risk_score,
  fp.coverage_ratio,
  c.usage_kind,
  c.evidence_json,
  c.call_chain_id,
  c.call_chain_json,
  c.created_at
FROM analytics.config_data_flow c
LEFT JOIN analytics.function_profile fp
  ON fp.repo = c.repo AND fp.commit = c.commit AND fp.function_goid_h128 = c.function_goid_h128;
```

This gives your agent a very direct “entity‑centric” handle:

* “Show me all Pydantic models and where they’re created/updated/serialized.”
* “Show me all code paths controlled by `feature_flag_X`.”
* “Where in subsystem Y do we read `service.database.url`?”

---

If you want, next I can draft the **DuckDB SQL** for computing basic `data_models` directly (similar to what we did for `function_profile`), or help design a few canonical MCP tools around these new datasets (e.g., `get_data_model_profile`, `find_config_flows_for_key`).
