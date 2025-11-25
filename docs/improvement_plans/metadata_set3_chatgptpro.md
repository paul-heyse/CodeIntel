Here’s a concrete, CodeIntel‑native implementation plan for the **semantic roles / side‑effects / contracts** scope, wired into your existing DuckDB schemas, NetworkX analytics spine, and orchestration flow. I’ll reuse your existing patterns (schemas in `config/schemas/tables.py`, analytics modules under `analytics/`, NetworkX overlays for CFG/DFG/call graph, docs views in `storage/views.py`, etc.).

High‑level: we’ll add three new analytics families:

1. `analytics.function_effects` – per‑function side‑effect/purity classification.
2. `analytics.function_contracts` – per‑function inferred pre/postconditions, raises, and nullability.
3. `analytics.semantic_roles_*` – semantic roles for functions and modules (api_handler, cli_command, repo, validator, etc.).

…and then thread them into `function_profile.*` + `docs.v_function_architecture` so your agent gets all this in a single row.

---

## 0. Where this plugs into the current pipeline

Quick reminder of how things flow today:

* Ingestion builds `core.*` & `graph.*` (AST/CST, GOIDs, call graph, CFG/DFG, symbol uses, coverage, tests, config).
* Analytics build `analytics.*` (function_metrics, function_types, coverage_functions, hotspots, typedness, static_diagnostics, goid_risk_factors, graph_metrics_*, subsystems, CFG/DFG metrics, test graph metrics, etc.).
* Docs views (`docs.*`) denormalize for agents (function/module/subsystem architecture views, CFG/DFG block views). 

We’ll:

* Add **three new analytics tables** and small helpers.
* Add **three new analytics modules** under `analytics/`.
* Add **three orchestration steps**.
* Extend **`function_profile.*`** and `docs.v_function_architecture` to surface the new columns.

---

## 1. function_effects.* – side‑effect / purity classification

### 1.1 Table design

Add this to `config/schemas/tables.py` as `analytics.function_effects` (mirrors style of `function_metrics`, `goid_risk_factors`). 

```python
"analytics.function_effects": TableSchema(
    schema="analytics",
    name="function_effects",
    columns=[
        Column("repo", "VARCHAR", nullable=False),
        Column("commit", "VARCHAR", nullable=False),
        Column("function_goid_h128", "DECIMAL(38,0)", nullable=False),

        # Core effect flags
        Column("is_pure", "BOOLEAN", nullable=False),              # best-effort
        Column("uses_io", "BOOLEAN", nullable=False),              # files, network, stdout/stderr, logging
        Column("touches_db", "BOOLEAN", nullable=False),
        Column("uses_time", "BOOLEAN", nullable=False),
        Column("uses_randomness", "BOOLEAN", nullable=False),
        Column("modifies_globals", "BOOLEAN", nullable=False),
        Column("modifies_closure", "BOOLEAN", nullable=False),
        Column("spawns_threads_or_tasks", "BOOLEAN", nullable=False),

        # Optional extra meta
        Column("has_transitive_effects", "BOOLEAN", nullable=False),
        Column("purity_confidence", "DOUBLE"),
        Column("effects_json", "JSON"),  # structured evidence: calls, writes, etc.

        Column("created_at", "TIMESTAMP", nullable=False),
    ],
    primary_key=("repo", "commit", "function_goid_h128"),
    description="Side-effect and purity classification per function GOID.",
)
```

Notes:

* `has_transitive_effects` means “no direct effects, but calls other effectful functions”.
* `is_pure` = no direct effect flags **and** `has_transitive_effects = FALSE`.
  When anything is unknown/ambiguous, we can set `purity_confidence < 1` and still set `is_pure` conservatively to `FALSE`.

### 1.2 Config model

In `config/models.py` define:

```python
@dataclass(frozen=True)
class FunctionEffectsConfig:
    repo: str
    commit: str

    # Tunables for analysis depth & patterns
    max_call_depth: int = 3          # breadth-first depth for transitive propagation
    require_all_callees_pure: bool = True
    io_apis: dict[str, list[str]] = field(default_factory=dict)
    db_apis: dict[str, list[str]] = field(default_factory=dict)
    time_apis: dict[str, list[str]] = field(default_factory=dict)
    random_apis: dict[str, list[str]] = field(default_factory=dict)
    threading_apis: dict[str, list[str]] = field(default_factory=dict)

    @classmethod
    def from_paths(cls, *, repo: str, commit: str) -> "FunctionEffectsConfig":
        return cls(repo=repo, commit=commit)
```

Populate defaults in `config/defaults.py` or similar, e.g.:

* `io_apis = {"builtins": ["open", "print"], "pathlib": ["Path.open"], "logging": ["info","warning",...], "requests": ["get","post",...]}`
* `db_apis = {"sqlalchemy": ["Session", "engine"], "psycopg": ["connect"], "asyncpg": ["connect"]}`, etc.

Config can be YAML‑/toml‑tuned later; for now keep reasonable built‑ins.

### 1.3 Implementation module: `analytics/function_effects.py`

Create a new module that leans heavily on:

* **CFG/DFG**: `graph.cfg_blocks`, `graph.cfg_edges`, `graph.dfg_edges`. 
* **Call graph**: `graph.call_graph_edges`. 
* **GOIDs & modules**: `core.goids`, `core.modules`, `core.ast_nodes`/`cfg_blocks.stmts_json`.
* **NetworkX overlays** already used for call graph/CFG/DFG analytics. 

Signature:

```python
def compute_function_effects(
    con: duckdb.DuckDBPyConnection,
    cfg: FunctionEffectsConfig,
) -> None:
    ...
```

#### 1.3.1 Direct effects detection

For each function GOID:

1. **Gather structural context**:

   * From `analytics.function_metrics` get `rel_path`, `qualname`, `language`. 
   * From `graph.cfg_blocks` pull all `stmts_json` rows for that `function_goid_h128`. Each `stmts_json` contains serialized AST for statements in that block.

2. **Parse statements**:

   Use the same helper(s) you use in CFG/DFG analytics to deserialize `stmts_json` into simple Python structures, or directly into LibCST/ast nodes if you already do that. (You already have AST utilities under `ingestion/ast_utils.py` used by function metrics and CFG builder.)

   During this walk, classify:

   * **Calls** (to detect IO, DB, time, randomness, threading):

     For each call expression:

     * Extract:

       * `func_name` (simple name: `print`, `open`, `sleep`)
       * `qualifier` / module attribute path (`requests.get`, `time.sleep`, `random.random`)

     * First try to leverage **call graph edges**:

       * Join on `graph.call_graph_edges` where `caller_goid_h128 = function_goid_h128`.
       * For edges where `callee_goid_h128 IS NULL`, inspect `evidence_json` (your call graph edges already store AST snippet / callee text) to recover the callee name/module string.

     * Compare resolved `(module, func_name)` against config patterns:

       * IO: `io_apis`
       * DB: `db_apis`
       * Time: `time_apis`
       * Random: `random_apis`
       * Threading: `threading_apis`

     Set flags:

     * `uses_io |= is_io_call`
     * `touches_db |= is_db_call`
     * `uses_time |= is_time_call`
     * `uses_randomness |= is_random_call`
     * `spawns_threads_or_tasks |= is_threading_call`

   * **Global / closure modification**:

     * Look for `global` and `nonlocal` statements in the AST of this function:

       * `modifies_globals |= bool(global_stmts)`
       * `modifies_closure |= bool(nonlocal_stmts)`

     * Optionally, use DFG: **writes** (`use_kind` = `write`/`update`) to names known to live in module scope (via `core.ast_nodes` classification) also set `modifies_globals = True`.

   * **Other effect hints**:

     * `os.environ[...] = ...` → treat as global side effect.
     * Writes to files (`open(..., 'w')`, `Path.write_text`) → `uses_io = True`.

3. **Direct flag summary**:

   For each function, you now have direct booleans for:

   ```text
   uses_io
   touches_db
   uses_time
   uses_randomness
   modifies_globals
   modifies_closure
   spawns_threads_or_tasks
   ```

   Set `direct_effectful = any(...)`.

#### 1.3.2 Transitive effects via call graph

Use your existing NetworkX call graph overlay (you already construct a `DiGraph` over `graph.call_graph_edges` to compute centralities, layers, SCCs).

1. Build `G_call = nx.DiGraph()` over all functions:

   * Nodes: `function_goid_h128`.
   * Edges: `caller → callee` (only with `callee_goid_h128` resolved).

2. Mark each node with `node["direct_effectful"] = direct_effectful`.

3. For each function, perform a bounded BFS to depth `cfg.max_call_depth`:

   * If any reachable callee has `direct_effectful = True`, set `has_transitive_effects = True`.

4. Compute `is_pure`:

   ```python
   is_pure = (
       not direct_effectful
       and not has_transitive_effects
       # Optionally also require no randomness/time
   )
   ```

5. Compute `purity_confidence`:

   Simple heuristic:

   * Start at 1.0.
   * Subtract for unresolved callees (edges with `callee_goid_h128 IS NULL`), e.g. `-0.1` per unresolved call up to a floor.
   * Subtract if `cfg.max_call_depth` cut off more call graph than we wanted.

#### 1.3.3 Evidence JSON

Build an `effects_json` object per function, e.g.:

```json
{
  "io_calls": [
    {"callee": "open", "module": "builtins", "rel_path": "...", "line": 42},
    {"callee": "get", "module": "requests", "rel_path": "...", "line": 88}
  ],
  "db_calls": [
    {"callee": "execute", "module": "sqlalchemy.engine.Connection", "line": 120}
  ],
  "time_calls": [...],
  "random_calls": [...],
  "threading_calls": [...],
  "globals": ["config_cache"],
  "nonlocals": ["state"],
  "transitive_effects_via": [<callee_goid_h128>, ...]
}
```

You don’t need to go overboard; the main value is a couple of concrete examples an agent can cite.

#### 1.3.4 Insertion

Use your existing pattern:

* `ensure_schema(con, "analytics.function_effects")` (from `config/schemas/sql_builder.py`). 
* `DELETE FROM analytics.function_effects WHERE repo = ? AND commit = ?`.
* Bulk insert using the tuple helpers strategy from `analytics/coverage_analytics.py`. 

Optionally add a `FunctionEffectsRow` TypedDict + helper in `models/rows.py` for consistency.

### 1.4 Orchestration and export

**Orchestration:**

In `orchestration/steps.py` add: 

```python
@dataclass
class FunctionEffectsStep:
    name: str = "function_effects"
    # Needs GOIDs, callgraph, CFG/DFG, and basic function analytics
    deps: Sequence[str] = (
        "goids",
        "callgraph",
        "cfg",
        "dfg",
        "function_analytics",  # function_metrics + function_types
    )

    def run(self, ctx: PipelineContext, con: duckdb.DuckDBPyConnection) -> None:
        _log_step(self.name)
        cfg = FunctionEffectsConfig.from_paths(repo=ctx.repo, commit=ctx.commit)
        compute_function_effects(con, cfg)
```

Wire it into the Prefect flow (`orchestration/prefect_flow.py`) after function analytics and graph metrics:

```python
@task(name="function_effects", retries=1, retry_delay_seconds=2)
def t_function_effects(repo: str, commit: str, db_path: Path) -> None:
    con = _connect(db_path)
    cfg = FunctionEffectsConfig.from_paths(repo=repo, commit=commit)
    compute_function_effects(con, cfg)
    con.close()
```

**Export:**

In `docs_export/export_jsonl.py` and `export_parquet.py`, extend dataset maps:

```python
JSONL_DATASETS["analytics.function_effects"] = "function_effects.jsonl"
PARQUET_DATASETS["analytics.function_effects"] = "function_effects.parquet"
```

**Profiles & docs views:**

* In the function profile builder (`analytics/profiles.py` or equivalent), left‑join `analytics.function_effects` by `(repo, commit, function_goid_h128)` and add `is_pure`, `uses_io`, `touches_db`, etc. as columns on `function_profile.*`.
* `docs.v_function_architecture` (which already joins function_profile, graph metrics, CFG/DFG metrics, etc.) will automatically expose these if it selects `function_profile.*`. If not, explicitly add them in the view SQL in `storage/views.py`.

---

## 2. function_contracts.* – pre/postconditions & invariants

### 2.1 Table design

Add `analytics.function_contracts` to `config/schemas/tables.py`:

```python
"analytics.function_contracts": TableSchema(
    schema="analytics",
    name="function_contracts",
    columns=[
        Column("repo", "VARCHAR", nullable=False),
        Column("commit", "VARCHAR", nullable=False),
        Column("function_goid_h128", "DECIMAL(38,0)", nullable=False),

        Column("preconditions_json", "JSON"),   # list[Precondition]
        Column("postconditions_json", "JSON"),  # list[Postcondition]
        Column("raises_json", "JSON"),          # list[RaiseSpec]
        Column("param_nullability_json", "JSON"),  # {param: "non_null"/"nullable"/"unknown"}
        Column("return_nullability", "VARCHAR"),   # same enum

        Column("contract_confidence", "DOUBLE"),
        Column("created_at", "TIMESTAMP", nullable=False),
    ],
    primary_key=("repo", "commit", "function_goid_h128"),
    description="Inferred pre/postconditions and nullability contracts per function.",
)
```

JSON shapes (document in comments / docs):

```json
// preconditions_json / postconditions_json
[
  {
    "kind": "non_null",      // "non_null" | "len_gt" | "ge" | "in_set" | "instance_of" | ...
    "param": "user_id",      // or "return" for postconditions
    "value": null,           // threshold, set, or type, depending on kind
    "source": "assert",      // "assert" | "if_guard" | "docstring" | "type_hint"
    "rel_path": "src/foo.py",
    "line": 42
  }
]

// raises_json
[
  {
    "exception": "ValueError",
    "source": "raise",       // or "docstring"
    "message_snippet": "user_id must be positive",
    "rel_path": "src/foo.py",
    "line": 43
  }
]
```

### 2.2 Config model

In `config/models.py`:

```python
@dataclass(frozen=True)
class FunctionContractsConfig:
    repo: str
    commit: str

    # basic knobs
    max_conditions_per_func: int = 64

    @classmethod
    def from_paths(cls, *, repo: str, commit: str) -> "FunctionContractsConfig":
        return cls(repo=repo, commit=commit)
```

### 2.3 Implementation module: `analytics/function_contracts.py`

This module will read:

* Types: `analytics.function_types` (param types, return_type, typedness_bucket). 
* Docstrings: `core.docstrings` (params/returns/raises parsed by `docstring_parser`).
* CFG/DFG / AST: `graph.cfg_blocks.stmts_json` to find asserts, if‑guards, raise statements.
* Coverage/tests only if you want to refine (optional, not necessary).

Signature:

```python
def compute_function_contracts(
    con: duckdb.DuckDBPyConnection,
    cfg: FunctionContractsConfig,
) -> None:
    ...
```

#### 2.3.1 Nullability from type hints & docstrings

1. **Type‑based nullability**:

   From `analytics.function_types` per function: 

   * For each parameter type string, e.g. `"Optional[int]"`, `"int | None"`, `"Union[str, None]"`, mark:

     * if contains `"Optional["` or `| None` or `"None"` anywhere → `"nullable"`
     * else → `"non_null"` (unless there is literally no annotation in which case `"unknown"`)

   * For `return_type`:

     * same logic → `return_nullability`.

2. **Docstring‑based refinement**:

   From `core.docstrings`:

   * For each param section: if description contains `"optional"`, `"may be None"`, etc. → upgrade param to `"nullable"`.

   * If description contains `"required"`, `"must not be None"`, `"non-empty"` → treat as `"non_null"` and possibly attach a `precondition` entry of kind `non_null` / `len_gt`.

   * For the return section: if description says `"returns None if not found"` or `"may return None"` → set `return_nullability = "nullable"`.

#### 2.3.2 Preconditions from asserts & guards

Use CFG block statements:

* For each function, from `graph.cfg_blocks` gather `stmts_json` and parse to AST nodes.

Look for:

1. **`assert` statements**:

   * `assert x is not None`, `assert x`, `assert len(xs) > 0`, `assert isinstance(x, Foo)`.

   Pattern‑match a small subset:

   * `assert <Name>` → precondition `kind="truthy"`.
   * `assert <Name> is not None` → `kind="non_null"`.
   * `assert len(<Name>) > 0` or `>= 1` → `kind="len_gt", value=0`.
   * `assert isinstance(<Name>, <Type>)` → `kind="instance_of"`.

2. **Guarded raises**:

   `if` + `raise` shapes:

   ```python
   if x is None:
       raise ValueError("x is required")
   if not items:
       raise ValueError("no items")
   if x < 0:
       raise ValueError("must be non-negative")
   ```

   For each `if` whose body contains a `raise`:

   * Negate the guard to form the precondition:

     * `if x is None: raise` → precondition `x is not None` (`non_null`).
     * `if not items: raise` → precondition `len(items) > 0` (`len_gt`, value=0).
     * `if x < 0: raise` → precondition `x >= 0` (`ge`, value=0).

   * Also populate `raises_json` with the exception.

3. **Docstring params**:

   If docstring param description has pattern “must be one of {…}”, you can emit an `in_set` precondition.

All preconditions are appended to `preconditions_json` up to `cfg.max_conditions_per_func`.

#### 2.3.3 Postconditions

We can get useful, cheap postconditions from:

1. **Return type & docstring**:

   * If `return_type` indicates `bool` (annotation contains `"bool"`) and name matches `is_*/has_*`, we can emit a high‑level postcondition:

     ```json
     {"kind": "returns_bool_predicate", "param": "return", "source": "type_hint"}
     ```

   * If `return_nullability = "non_null"` and there are no `return None` branches detected in AST, we can emit:

     ```json
     {"kind": "non_null", "param": "return", "source": "type_hint"}
     ```

2. **Asserts on local `result` before return**:

   Pattern:

   ```python
   result = compute(...)
   assert result is not None
   return result
   ```

   * If we detect a final `return result` and prior `assert result is not None` or `len(result) > 0`, we can emit:

     * `postcondition: non_null return`
     * or `len_gt 0` for the return.

3. **Docstring “Raises…”**:

   If docstring says “Raises ValueError if X invalid; otherwise returns normalized value”. You can optionally encode that as a postcondition but the main value is in `raises_json`.

#### 2.3.4 Raises

From AST:

* Walk all `Raise` nodes in the function body; extract exception type name from:

  * `raise ValueError("...")` → `"ValueError"`
  * `raise CustomError(...)` → `"CustomError"`

From `core.docstrings.raises` array: 

* For each `Raises` entry, add to `raises_json` if not already present.

Optional: link each raise to a precondition (e.g., from guard inference) by including a `precondition_index` field.

#### 2.3.5 contract_confidence

Simple scoring:

* Start at 0.0.
* +0.3 if we have type hints.
* +0.3 if we have docstrings.
* +0.3 if we have at least one assert/guard‑derived precondition.
* Cap at 1.0.

This is a coarse “how strong is the contract” signal.

#### 2.3.6 Insertion

Exactly as with `function_effects`:

* `ensure_schema(con, "analytics.function_contracts")`.
* Delete by repo/commit.
* Bulk insert rows.

### 2.4 Orchestration & export

**Step:**

```python
@dataclass
class FunctionContractsStep:
    name: str = "function_contracts"
    deps: Sequence[str] = (
        "function_analytics",   # function_metrics + function_types
        "docstrings",
        "cfg",                  # stmts_json available
    )

    def run(self, ctx: PipelineContext, con: duckdb.DuckDBPyConnection) -> None:
        _log_step(self.name)
        cfg = FunctionContractsConfig.from_paths(repo=ctx.repo, commit=ctx.commit)
        compute_function_contracts(con, cfg)
```

Add a Prefect task `t_function_contracts` to the flow after typing+docstrings+CFG. 

**Export:**

Add to JSONL/Parquet exports:

```python
JSONL_DATASETS["analytics.function_contracts"] = "function_contracts.jsonl"
PARQUET_DATASETS["analytics.function_contracts"] = "function_contracts.parquet"
```

**Profiles & docs views:**

* Extend `function_profile.*` by left‑joining contracts table:

  * Optionally just bring in:

    * `param_nullability_json`
    * `return_nullability`
    * small derived booleans like `returns_nullable`, `has_preconditions`, `has_raises`.

* In `docs.v_function_architecture`, include these fields so agents see them side‑by‑side with risk, coverage, CFG/DFG metrics, etc.

---

## 3. semantic_roles.* – “what kind of thing is this?”

We’ll classify both functions **and modules**.

### 3.1 Table designs

#### 3.1.1 `analytics.semantic_roles_functions`

```python
"analytics.semantic_roles_functions": TableSchema(
    schema="analytics",
    name="semantic_roles_functions",
    columns=[
        Column("repo", "VARCHAR", nullable=False),
        Column("commit", "VARCHAR", nullable=False),
        Column("function_goid_h128", "DECIMAL(38,0)", nullable=False),

        Column("role", "VARCHAR", nullable=False),
        Column("framework", "VARCHAR"),
        Column("role_confidence", "DOUBLE", nullable=False),
        Column("role_sources_json", "JSON"),

        Column("created_at", "TIMESTAMP", nullable=False),
    ],
    primary_key=("repo", "commit", "function_goid_h128"),
    description="Semantic roles for functions/methods (api_handler, cli_command, validator, etc.).",
)
```

Target `role` enum (documented, not enforced):

`api_handler | cli_command | service | repository | domain_model | validator | helper | test | test_helper | config_loader | worker | other`.

Framework examples: `fastapi_endpoint`, `click_command`, `pytest_fixture`, `pydantic_model`, etc.

#### 3.1.2 `analytics.semantic_roles_modules`

```python
"analytics.semantic_roles_modules": TableSchema(
    schema="analytics",
    name="semantic_roles_modules",
    columns=[
        Column("repo", "VARCHAR", nullable=False),
        Column("commit", "VARCHAR", nullable=False),
        Column("module", "VARCHAR", nullable=False),

        Column("role", "VARCHAR", nullable=False),         # api, cli, service, repository, domain, test, config, infra, ...
        Column("role_confidence", "DOUBLE", nullable=False),
        Column("role_sources_json", "JSON"),

        Column("created_at", "TIMESTAMP", nullable=False),
    ],
    primary_key=("repo", "commit", "module"),
    description="Semantic roles for modules (api, cli, repository, domain, config, test, ...).",
)
```

### 3.2 Config model

```python
@dataclass(frozen=True)
class SemanticRolesConfig:
    repo: str
    commit: str

    enable_llm_refinement: bool = False   # future hook
    max_roles_per_module: int = 3

    @classmethod
    def from_paths(cls, *, repo: str, commit: str) -> "SemanticRolesConfig":
        return cls(repo=repo, commit=commit)
```

### 3.3 Implementation module: `analytics/semantic_roles.py`

This will use:

* Names & locations: `analytics.function_metrics`, `core.modules`, `core.ast_nodes`.
* Tags & owners: `analytics.tags_index` + `tags_index.yaml`.
* Docstrings: `core.docstrings`. 
* Framework clues: decorators in AST / docstrings.
* Call patterns: call graph metrics (`analytics.graph_metrics_functions`), function effects (`analytics.function_effects`).
* Tests: `analytics.test_catalog`, `test_coverage_edges`, module tags on `tests/`.

Signature:

```python
def compute_semantic_roles(
    con: duckdb.DuckDBPyConnection,
    cfg: SemanticRolesConfig,
) -> None:
    ...
```

#### 3.3.1 Feature extraction for functions

For each function GOID:

* **Basic metadata** (from `function_metrics`):

  * `name`, `qualname`, `rel_path`, `language`, `kind`, `start_line`, `end_line`.

* **Module + tags**:

  * Join with `core.modules` on `rel_path` to get `module`, `tags`, `owners`.

* **Docstrings**:

  * Join with `core.docstrings` via `(rel_path, qualname)` mapping or via `goid_crosswalk`. Extract:

    * `short_desc`, `long_desc`.
    * `params`, `returns`, `raises`.

* **Decorators**:

  * From `core.ast_nodes` or `cfg_blocks.stmts_json`, find decorators on the function:

    * `@app.get`, `@router.post` → FastAPI endpoint.
    * `@click.command`, `@click.group` → CLI command.
    * `@pytest.fixture` → test helper.
    * `@dataclass`, `@pydantic.dataclasses.dataclass`, `@BaseModel` subclass → domain model methods.

* **Effects & role‑ish behavior**:

  * Join `analytics.function_effects` for `uses_io`, `touches_db`, `uses_time`, etc.
  * Join test graph metrics to see if the function is a test (`analytics.test_catalog` via GOID) or heavily used in tests.

* **Call context**:

  * From `analytics.graph_metrics_functions`, fetch fan‑in/fan‑out, centralities, leaf vs hub. A function with high fan‑in/fan‑out in a “service” module is more likely a `service` role.

#### 3.3.2 Role scoring heuristics (functions)

For each function, compute a **score per candidate role**. Pseudocode style:

* Initialize `scores = defaultdict(float)`.

* **api_handler**:

  * If decorator matches `FastAPI` pattern (`@router.get`, `@app.post`, etc.) → `scores["api_handler"] += 1.0`, `framework="fastapi_endpoint"`.
  * If module tag contains `"api"` or module path under `app/routes` / `api/` → `+0.5`.

* **cli_command**:

  * Decorator `@click.command` / `@click.group` / `@app.command` (Typer) → `+1.0`, `framework="click_command"` or `typer_command`.
  * Function name `main` in `cli` or `scripts` module → `+0.7`.

* **test / test_helper**:

  * If `rel_path` or module path under `tests/` or tag `"test"`:

    * Name starts with `test_` → `role="test"` with high score.
    * Decorator `@pytest.fixture` → `role="test_helper"`, `framework="pytest_fixture"`.

* **repository**:

  * Module tags include `"infra"`, `"db"`, `"repository"` → `+0.5`.
  * Function name starts with `get_`, `fetch_`, `save_`, `update_`, `delete_`.
  * `touches_db` from `function_effects` → `+0.8`.

* **service**:

  * Module tags include `"service"` or path includes `service/`, `use_cases/`.
  * High fan‑in / fan‑out from call graph metrics.
  * Calls repository functions (callee roles can be reused in a refinement pass).

* **validator**:

  * Name starts with `validate_`, `check_`, `ensure_`.
  * Has preconditions and raises `ValueError` frequently; `function_contracts` has `preconditions_json` and `raises_json`. → major boost.

* **config_loader**:

  * Module path under `config/`, `settings/`, `env/`.
  * Reads from `analytics.config_values` keys or `os.environ`, `dotenv`. → `uses_io` and references config keys. 

* **helper**:

  * Small, low‑risk functions (`loc` small, complexity low, risk_score low), called mostly by one or two higher‑level functions (`fan_in` modest), no IO/DB. Good fallback when no stronger role is present.

Then:

* Normalize scores:

  * `role = argmax(scores)` if `max_score >= threshold`, else `role="other"`.
  * `role_confidence = min(1.0, max_score)` or scaled.

* Build `role_sources_json` with the specific features used, e.g.:

  ```json
  {
    "decorators": ["router.get"],
    "module_tags": ["api"],
    "name_hint": "get_user",
    "effects": ["touches_db"],
    "graph": {"call_fan_in": 10, "call_fan_out": 2}
  }
  ```

#### 3.3.3 Module roles

For modules, aggregate function roles + tags:

* For each `core.modules.module`:

  * Summarize:

    * Most common function role.
    * Tags (`api`, `cli`, `infra`, `domain`, `test`).
    * Import graph position (import fan‑in/out, subsystem membership if you’ve already built `subsystems.*`).

* Heuristics:

  * If many functions with `role="api_handler"` or tag `"api"` → module `role="api"`.

  * If in `cli` package or many `cli_command` functions → `role="cli"`.

  * If many repository functions + DB IO → `role="repository"`.

  * If mostly tests → `role="test"`.

  * Else infer from tags & location: `config`, `domain`, `infra`.

Compute `role_confidence` proportional to fraction of functions in that role and presence of strong tags.

#### 3.3.4 Optional LLM refinement

Keep this **optional**, controlled via `SemanticRolesConfig.enable_llm_refinement`. If enabled:

* Collect a small sample of function candidates and their heuristic role + evidence, e.g.:

  ```json
  {
    "name": "get_user",
    "module": "app.api.users",
    "doc_short": "Fetch a user by id",
    "effects": {"touches_db": true},
    "decorators": ["router.get"],
    "tags": ["api"],
    "heuristic_role": "api_handler"
  }
  ```

* Call an external LLM endpoint (or tool) to adjust `role` and `role_confidence`.

Design this as a separate helper so your core pipeline still works without external calls; you can implement it later.

### 3.4 Orchestration & export

**Step:**

```python
@dataclass
class SemanticRolesStep:
    name: str = "semantic_roles"
    deps: Sequence[str] = (
        "function_analytics",    # function_metrics + function_types
        "function_effects",
        "function_contracts",
        "tags_index",
        # optionally "subsystems" if you want subsystem-aware roles
    )

    def run(self, ctx: PipelineContext, con: duckdb.DuckDBPyConnection) -> None:
        _log_step(self.name)
        cfg = SemanticRolesConfig.from_paths(repo=ctx.repo, commit=ctx.commit)
        compute_semantic_roles(con, cfg)
```

Hook it into Prefect after the other analytics steps but before docs export. 

**Export:**

```python
JSONL_DATASETS["analytics.semantic_roles_functions"] = "semantic_roles_functions.jsonl"
JSONL_DATASETS["analytics.semantic_roles_modules"] = "semantic_roles_modules.jsonl"
PARQUET_DATASETS["analytics.semantic_roles_functions"] = "semantic_roles_functions.parquet"
PARQUET_DATASETS["analytics.semantic_roles_modules"] = "semantic_roles_modules.parquet"
```

**Profiles & docs views:**

* Extend `function_profile.*` with:

  * `role`, `framework`, `role_confidence` (simple join).
* Extend `module_profile.*` with module roles.
* Update:

  * `docs.v_function_architecture` to include function role & framework.
  * `docs.v_module_architecture` to include module role.
  * `docs.v_subsystem_summary` could include distribution of roles within each subsystem (e.g. “subsystem X: 10 api_handlers, 4 repositories, 20 helpers”).

These become prime filters for agents (“show me high‑risk api_handlers in subsystem Y that touch DB and lack preconditions”).

---

## 4. Integration checkpoints & testing

To make this robust:

1. **Schema migration**: run the DuckDB schema creation for the three new tables and ensure `ensure_schema` passes.

2. **Dry‑run on CodeIntel itself** (your repo) ([GitHub][1])

   * Expect to see:

     * CLI functions in `cli/` classified as `cli_command`.
     * API handlers in whatever `app/routes` modules you have as `api_handler`.
     * Config readers (using `config_values` / `pyproject.toml` / env) as `config_loader`.
     * Test functions/fixtures as `test` / `test_helper`.

3. **Spot‑check contracts**:

   * Pick a few functions with obvious asserts / docstrings; verify `preconditions_json` & `raises_json` look sane.

4. **Performance**:

   * Reuse existing NetworkX graphs and AST/CFG deserialization helpers; avoid reparsing the entire repo from scratch.
   * Limit guard/contract detection to simple patterns; you can always expand later.

---

If you’d like, I can next sketch concrete SQL/duckdb queries for joining `function_effects` and `function_contracts` into your existing `function_profile.*` builder, or propose a couple of canonical queries your MCP tools could expose (e.g., “high‑risk api_handlers with no preconditions and low coverage”).

[1]: https://github.com/paul-heyse/CodeIntel "GitHub - paul-heyse/CodeIntel"

# duckdb queries #

You can treat this as “just more joins” in the same style as your existing `RiskFactorsStep` SQL: everything is keyed on `(repo, commit, function_goid_h128)` and you denormalize into `analytics.function_profile.*`, which `docs.v_function_architecture` then sits on top of.

Below are concrete DuckDB SQL snippets you can drop into your `ProfilesStep` (or whatever is currently building `function_profile.*`) to pull in `function_effects` and `function_contracts`.

---

## 1. Simple “query‑time” join (baseline)

First, the trivial pattern you’ll use everywhere once the tables exist:

```sql
SELECT
  fp.*,
  fe.is_pure,
  fe.uses_io,
  fe.touches_db,
  fe.uses_time,
  fe.uses_randomness,
  fe.modifies_globals,
  fe.modifies_closure,
  fe.spawns_threads_or_tasks,
  fe.has_transitive_effects,
  fe.purity_confidence,
  fc.param_nullability_json,
  fc.return_nullability,
  fc.preconditions_json,
  fc.postconditions_json,
  fc.raises_json,
  fc.contract_confidence
FROM analytics.function_profile AS fp
LEFT JOIN analytics.function_effects AS fe
  ON fe.repo = fp.repo
 AND fe.commit = fp.commit
 AND fe.function_goid_h128 = fp.function_goid_h128
LEFT JOIN analytics.function_contracts AS fc
  ON fc.repo = fp.repo
 AND fc.commit = fp.commit
 AND fc.function_goid_h128 = fp.function_goid_h128
WHERE fp.repo = ? AND fp.commit = ?;
```

That’s the “dumb join”: **no schema changes**, just combine all three views ad‑hoc. The rest of this answer shows how to *bake* the relevant bits into `analytics.function_profile.*` so agents see them in the main profile row.

---

## 2. Extending `analytics.function_profile` in your ProfilesStep

### 2.1. New columns on `analytics.function_profile`

In `config/schemas/tables.py`, extend the `analytics.function_profile` schema with the derived flags from both tables (you don’t need the full JSON blobs there unless you want them):

```python
# in TableSchema("analytics.function_profile", ...)
Column("is_pure", "BOOLEAN"),
Column("uses_io", "BOOLEAN"),
Column("touches_db", "BOOLEAN"),
Column("uses_time", "BOOLEAN"),
Column("uses_randomness", "BOOLEAN"),
Column("modifies_globals", "BOOLEAN"),
Column("modifies_closure", "BOOLEAN"),
Column("spawns_threads_or_tasks", "BOOLEAN"),
Column("has_transitive_effects", "BOOLEAN"),
Column("purity_confidence", "DOUBLE"),

Column("param_nullability_json", "JSON"),
Column("return_nullability", "VARCHAR"),  # 'non_null' | 'nullable' | 'unknown'
Column("has_preconditions", "BOOLEAN"),
Column("has_postconditions", "BOOLEAN"),
Column("has_raises", "BOOLEAN"),
Column("contract_confidence", "DOUBLE"),
```

(If you prefer, you can also surface the raw `preconditions_json` / `postconditions_json` / `raises_json` in the profile; I’ll show how to derive booleans either way.)

---

### 2.2. CTE‑based builder: plugging into your existing profile query

Assuming your current `ProfilesStep` does something like:

```python
con.execute("DELETE FROM analytics.function_profile WHERE repo = ? AND commit = ?", [ctx.repo, ctx.commit])

con.execute(
    """
    INSERT INTO analytics.function_profile
    SELECT
      -- lots of columns from fm, ft, rf, cf, tests, calls, docstrings...
    FROM ...
    WHERE fm.repo = ? AND fm.commit = ?
    """,
    [ctx.repo, ctx.commit],
)
```

You can refactor that INSERT into a `WITH` query and bring in `function_effects` / `function_contracts` as CTEs.

Here’s a **concrete template** that matches your existing join style (same JSON operators, same repo/commit filters). 

```sql
DELETE FROM analytics.function_profile
WHERE repo = ? AND commit = ?;

INSERT INTO analytics.function_profile (
    -- identity
    function_goid_h128,
    urn,
    repo,
    commit,
    rel_path,
    module,
    language,
    kind,
    qualname,
    start_line,
    end_line,

    -- structure & types
    loc,
    logical_loc,
    cyclomatic_complexity,
    param_count,
    keyword_params,
    vararg,
    kwarg,
    total_params,
    return_type,
    typedness_bucket,
    file_typed_ratio,
    static_error_count,

    -- coverage & tests
    coverage_ratio,
    tested,
    tests_touching,
    failing_tests,
    slow_tests,

    -- callgraph
    call_fan_in,
    call_fan_out,
    call_is_leaf,

    -- risk
    risk_score,
    risk_level,
    risk_component_coverage,
    risk_component_complexity,
    risk_component_static,
    risk_component_hotspot,

    -- NEW: effects
    is_pure,
    uses_io,
    touches_db,
    uses_time,
    uses_randomness,
    modifies_globals,
    modifies_closure,
    spawns_threads_or_tasks,
    has_transitive_effects,
    purity_confidence,

    -- NEW: contracts
    param_nullability_json,
    return_nullability,
    has_preconditions,
    has_postconditions,
    has_raises,
    contract_confidence,

    -- docs + tags
    doc_short,
    doc_long,
    tags,
    owners,
    created_at
)
WITH
-- Base function/risk row, similar to goid_risk_factors but joined to modules/docstrings/tests/calls
base AS (
    SELECT
        fm.function_goid_h128,
        fm.urn,
        fm.repo,
        fm.commit,
        fm.rel_path,
        m.module,
        fm.language,
        fm.kind,
        fm.qualname,
        fm.start_line,
        fm.end_line,

        fm.loc,
        fm.logical_loc,
        fm.cyclomatic_complexity,
        fm.param_count,
        fm.keyword_only_params AS keyword_params,
        fm.has_varargs          AS vararg,
        fm.has_varkw            AS kwarg,
        ft.total_params,
        ft.return_type,
        rf.typedness_bucket,
        rf.file_typed_ratio,
        rf.static_error_count,

        cf.coverage_ratio,
        cf.tested,

        tm.test_count         AS tests_touching,
        tm.failing_test_count AS failing_tests,
        tm.slow_test_count    AS slow_tests,

        cg.call_fan_in,
        cg.call_fan_out,
        cg.call_is_leaf,

        rf.risk_score,
        rf.risk_level,
        rf.risk_component_coverage,
        rf.risk_component_complexity,
        rf.risk_component_static,
        rf.risk_component_hotspot,

        d.short_desc AS doc_short,
        d.long_desc  AS doc_long,
        m.tags,
        m.owners
    FROM analytics.function_metrics AS fm
    LEFT JOIN analytics.function_types AS ft
      ON ft.function_goid_h128 = fm.function_goid_h128
     AND ft.repo = fm.repo
     AND ft.commit = fm.commit
    LEFT JOIN analytics.goid_risk_factors AS rf
      ON rf.function_goid_h128 = fm.function_goid_h128
     AND rf.repo = fm.repo
     AND rf.commit = fm.commit
    LEFT JOIN analytics.coverage_functions AS cf
      ON cf.function_goid_h128 = fm.function_goid_h128
     AND cf.repo = fm.repo
     AND cf.commit = fm.commit
    LEFT JOIN (
        SELECT
            e.function_goid_h128,
            COUNT(DISTINCT e.test_id)                                  AS test_count,
            COUNT(DISTINCT CASE WHEN t.status IN ('failed','error')
                                THEN e.test_id END)                    AS failing_test_count,
            COUNT(DISTINCT CASE WHEN t.status = 'passed'
                                THEN e.test_id END FILTER (WHERE t.duration_ms > 500.0)) AS slow_test_count
        FROM analytics.test_coverage_edges AS e
        LEFT JOIN analytics.test_catalog AS t
          ON t.test_id = e.test_id
        GROUP BY e.function_goid_h128
    ) AS tm
      ON tm.function_goid_h128 = fm.function_goid_h128
    LEFT JOIN (
        SELECT
            caller_goid_h128 AS function_goid_h128,
            COUNT(DISTINCT caller_goid_h128) FILTER (WHERE callee_goid_h128 IS NOT NULL) AS call_fan_out,
            COUNT(DISTINCT callee_goid_h128) FILTER (WHERE callee_goid_h128 IS NOT NULL) AS call_fan_in,
            CASE
              WHEN COUNT(DISTINCT callee_goid_h128) FILTER (WHERE callee_goid_h128 IS NOT NULL) = 0
              THEN TRUE ELSE FALSE
            END AS call_is_leaf
        FROM graph.call_graph_edges
        GROUP BY caller_goid_h128
    ) AS cg
      ON cg.function_goid_h128 = fm.function_goid_h128
    LEFT JOIN core.modules AS m
      ON m.path  = fm.rel_path
     AND m.repo  = fm.repo
     AND m.commit = fm.commit
    LEFT JOIN core.docstrings AS d
      ON d.repo     = fm.repo
     AND d.commit   = fm.commit
     AND d.rel_path = fm.rel_path
     AND d.qualname = fm.qualname
    WHERE fm.repo = ? AND fm.commit = ?
),

-- NEW: effects CTE
effects AS (
    SELECT
        repo,
        commit,
        function_goid_h128,
        is_pure,
        uses_io,
        touches_db,
        uses_time,
        uses_randomness,
        modifies_globals,
        modifies_closure,
        spawns_threads_or_tasks,
        has_transitive_effects,
        purity_confidence
    FROM analytics.function_effects
    WHERE repo = ? AND commit = ?
),

-- NEW: contracts CTE with useful derived flags
contracts AS (
    SELECT
        repo,
        commit,
        function_goid_h128,
        param_nullability_json,
        return_nullability,
        contract_confidence,
        COALESCE(json_array_length(preconditions_json), 0)  > 0 AS has_preconditions,
        COALESCE(json_array_length(postconditions_json), 0) > 0 AS has_postconditions,
        COALESCE(json_array_length(raises_json), 0)         > 0 AS has_raises
    FROM analytics.function_contracts
    WHERE repo = ? AND commit = ?
)

SELECT
    b.function_goid_h128,
    -- identity & location
    (SELECT urn FROM analytics.goid_risk_factors rf
      WHERE rf.function_goid_h128 = b.function_goid_h128
        AND rf.repo = b.repo AND rf.commit = b.commit
      LIMIT 1) AS urn,
    b.repo,
    b.commit,
    b.rel_path,
    b.module,
    b.language,
    b.kind,
    b.qualname,
    b.start_line,
    b.end_line,

    -- structure & types
    b.loc,
    b.logical_loc,
    b.cyclomatic_complexity,
    b.param_count,
    b.keyword_params,
    b.vararg,
    b.kwarg,
    b.total_params,
    b.return_type,
    b.typedness_bucket,
    b.file_typed_ratio,
    b.static_error_count,

    -- coverage & tests
    b.coverage_ratio,
    b.tested,
    COALESCE(b.tests_touching, 0),
    COALESCE(b.failing_tests, 0),
    COALESCE(b.slow_tests, 0),

    -- callgraph
    COALESCE(b.call_fan_in,  0),
    COALESCE(b.call_fan_out, 0),
    COALESCE(b.call_is_leaf, FALSE),

    -- risk
    b.risk_score,
    b.risk_level,
    b.risk_component_coverage,
    b.risk_component_complexity,
    b.risk_component_static,
    b.risk_component_hotspot,

    -- effects
    COALESCE(fe.is_pure, FALSE)                 AS is_pure,
    COALESCE(fe.uses_io, FALSE)                 AS uses_io,
    COALESCE(fe.touches_db, FALSE)              AS touches_db,
    COALESCE(fe.uses_time, FALSE)               AS uses_time,
    COALESCE(fe.uses_randomness, FALSE)         AS uses_randomness,
    COALESCE(fe.modifies_globals, FALSE)        AS modifies_globals,
    COALESCE(fe.modifies_closure, FALSE)        AS modifies_closure,
    COALESCE(fe.spawns_threads_or_tasks, FALSE) AS spawns_threads_or_tasks,
    COALESCE(fe.has_transitive_effects, FALSE)  AS has_transitive_effects,
    fe.purity_confidence,

    -- contracts
    fc.param_nullability_json,
    fc.return_nullability,
    COALESCE(fc.has_preconditions, FALSE) AS has_preconditions,
    COALESCE(fc.has_postconditions, FALSE) AS has_postconditions,
    COALESCE(fc.has_raises, FALSE) AS has_raises,
    fc.contract_confidence,

    -- docs & tags
    b.doc_short,
    b.doc_long,
    b.tags,
    b.owners,
    CURRENT_TIMESTAMP AS created_at
FROM base AS b
LEFT JOIN effects   AS fe
  ON fe.repo = b.repo
 AND fe.commit = b.commit
 AND fe.function_goid_h128 = b.function_goid_h128
LEFT JOIN contracts AS fc
  ON fc.repo = b.repo
 AND fc.commit = b.commit
 AND fc.function_goid_h128 = b.function_goid_h128
;
```

Parameters (in Python):

```python
con.execute(
    PROFILE_SQL,
    [
        ctx.repo, ctx.commit,  # base CTE
        ctx.repo, ctx.commit,  # effects
        ctx.repo, ctx.commit,  # contracts
    ],
)
```

Key points:

* **Join keys**: we always join on `(repo, commit, function_goid_h128)` to avoid accidental cross‑repo bleed, mirroring how `RiskFactorsStep` joins `function_metrics`, `function_types`, `hotspots`, etc. 
* **Derived flags in CTEs**: `contracts` computes `has_preconditions`/`has_postconditions`/`has_raises` once from the JSON; you just surface booleans in `function_profile`.
* **NULL‑safe defaults** with `COALESCE` so that missing `function_effects`/`function_contracts` rows don’t break the profile (e.g., `is_pure` defaults to `FALSE`, `has_preconditions` defaults to `FALSE`).

---

## 3. Updating docs views

If your `docs.v_function_architecture` view currently does:

```sql
SELECT
  fp.*,
  gm.*,          -- graph_metrics_functions
  tgm.*,         -- test_graph_metrics_functions
  cfgm.*,
  dfgm.*,
  sm.subsystem_id,
  ...
FROM analytics.function_profile AS fp
LEFT JOIN analytics.graph_metrics_functions AS gm USING (repo, commit, function_goid_h128)
...
```

…then **you don’t need to change it**: the new columns will automatically appear wherever `SELECT fp.*` is used.

If instead you enumerate columns explicitly, just add the new ones there as well.

---

If you paste your current `ProfilesStep` SQL at some point, I can give you a byte‑for‑byte patch that adds the new joins and columns without changing anything else, but this should be enough to wire `function_effects` and `function_contracts` cleanly into the main profile layer.
