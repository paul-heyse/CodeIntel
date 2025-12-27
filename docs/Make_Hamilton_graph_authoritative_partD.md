Below is a **repo-concrete, breaking-change–optimized** implementation plan for:

> **(7) Use `resolve_from_config` + `pipe_input`/`mutate`/`with_columns` to collapse “variant logic” and cross‑cutting transforms**

Premise: all prior phases are assumed landed (single composition root, `DagCatalog`, saver-derived output inventory, caching unification, planning DAG products). You explicitly requested that **all legacy paths are immediately deprecated and deleted**; this plan therefore **restructures modules aggressively** and replaces any ad hoc `if/else` backend branching and duplicated “normalize/clean/enrich” logic with **Hamilton-native compile-time variability + graph-local pipelines**.

---

# 0) End-state contract

## 0.1 Variants are compile-time DAG edits, not runtime branching

All “backend selection” and “feature-set selection” is expressed as **compile-time graph edits** via `resolve_from_config`, i.e. `resolve_from_config(decorate_with=callable(config)->decorator)`. No node bodies should contain branching of the form `if cfg.backend == ...:`; instead, variant selection happens by returning a decorator such as:

* `with_columns(...)` (plugin-specific)
* `pipe_input(step(...).when(...), ...)`
* `inject(...)`
* `mutate(apply_to(...), ...)` (within-module cross-cutting rewrites)

This makes variants **auditable via DAG structure** (list variables + tags + lineage), and makes config diffs deterministic.

## 0.2 Cross-cutting transforms are explicit DAG structure

Pre/post transforms are expressed by:

* `pipe_input` / `step(...).when(...)` (input normalizations + step gating + stable naming)
* `mutate` (post-process multiple nodes without touching them, per-module)
* `with_columns` (tabular column subDAGs executed inside dataframe engine)

Transform logic is not duplicated across dozens of table nodes; it is centralized and applied declaratively.

## 0.3 Backend-specific tabular operations are isolated

Polars vs pandas vs polars-lazy differences are isolated to:

* a single selector function returning the correct `with_columns` decorator
* a small set of backend-specific step/mutator functions (dtype casting, null normalization, etc.)

Downstream table nodes do not need backend-awareness beyond applying the canonical “table decorator”.

---

# 1) Enable power-user mode once, at the composition root

`resolve_from_config` requires `hamilton.enable_power_user_mode=True`. This must be set only by the runtime composition root (phase 6), never inside DAG nodes.

### Files MODIFIED

* `src/codeintel/runtime/compose.py`

  * Set:

    ```python
    import hamilton
    hamilton.enable_power_user_mode = True
    ```
  * Ensure this is executed before driver instantiation.

> Hard rule: **no other module** mutates Hamilton global settings.

---

# 2) Define a canonical “variant configuration surface” (typed, validated, hash-stable)

This must become the *single* place that encodes:

* dataframe backend selection
* feature-set selection
* cleaning strictness selection
* transform toggles
* namespace conventions

### Files CREATED

1. `src/codeintel/runtime/variants.py`

Define:

* `DataFrameBackend = Literal["pandas", "polars", "polars_lazy"]`
* `CleanMode = Literal["off", "lenient", "strict"]`
* `FeatureSetName = str`

`@dataclass(frozen=True, slots=True)`

* `VariantConfig`

  * `df_backend: DataFrameBackend`
  * `clean_mode: CleanMode`
  * `feature_sets: dict[str, tuple[str, ...]]`

    * keyed by logical table identifier (e.g. `table_key`)
    * values are lists of column-op node names to include (for `with_columns(select=...)`)
  * `enable_common_columns: bool`
  * `enable_schema_enforcement: bool`
  * `enable_canonicalization: bool`
  * `enable_value_clipping: bool`
  * `max_loc_clip: int` (example param; generalize as needed)
  * `null_policy: Literal["preserve", "drop_bad_rows"]`

`VariantConfig.validate()`:

* verify all feature-set entries reference allowed column ops for that table namespace (see §4)
* enforce deterministic ordering of feature lists
* compute `variant_fingerprint` used in cache keys and manifests (stable serialization)

### Files MODIFIED

* `src/codeintel/build/config.py` (or `src/codeintel/runtime/config.py` depending on where your runtime config lives post-phase6)

  * Replace all scattered booleans/flags controlling “pandas/polars/strictness” with a single `VariantConfig`.
  * Delete legacy config keys immediately (no compatibility shim).

* `src/codeintel/runtime/runtime_bundle.py`

  * Add `variants: VariantConfig` field.
  * Ensure `fingerprint` includes `variants.variant_fingerprint`.

---

# 3) Introduce a canonical tabular transform toolkit

This toolkit is the consolidation point: it provides *decorator factories* and *backend-specific primitive steps*.

## 3.1 Backend-specific primitive steps (kwarg-friendly, pipe-compatible)

### Files CREATED

2. `src/codeintel/build/hamilton/transforms/tabular_steps.py`

This module defines **pure**, kwarg-friendly functions to be used as `step(...)` stages and `mutate(...)` mutators. Constraints:

* no positional-only args
* no `*args/**kwargs`
* deterministic output

Examples (implement backend-specific overloads where necessary):

* `_drop_bad_rows(df, required_cols: tuple[str,...])`
* `_clip_numeric(df, col: str, max_value: float)`
* `_cast_schema(df, schema: dict[str, str])`
* `_normalize_nulls(df, policy: str)`
* `_sort_columns(df, column_order: tuple[str,...])`

Polars vs pandas:

* if you support multiple backends, implement parallel functions:

  * `_drop_bad_rows_pandas`, `_drop_bad_rows_polars`, `_drop_bad_rows_polars_lazy`
  * similarly for casting/clipping/null normalization

Keep them in one module to avoid scattering.

## 3.2 Backend selection helper for `with_columns`

### Files CREATED

3. `src/codeintel/build/hamilton/transforms/with_columns_backend.py`

Provide a single function returning the correct `with_columns` decorator given config:

```python
from __future__ import annotations
from typing import Callable, Sequence

from hamilton.plugins.h_pandas import with_columns as with_columns_pd
from hamilton.plugins.h_polars import with_columns as with_columns_pl
from hamilton.plugins.h_polars_lazyframe import with_columns as with_columns_pl_lazy

def select_with_columns(df_backend: str):
    if df_backend == "pandas":
        return with_columns_pd
    if df_backend == "polars":
        return with_columns_pl
    if df_backend == "polars_lazy":
        return with_columns_pl_lazy
    raise ValueError(f"Unsupported df_backend={df_backend!r}")
```

This prevents backend branching from leaking into table modules.

## 3.3 Canonical decorator factories using `resolve_from_config`

### Files CREATED

4. `src/codeintel/build/hamilton/transforms/decorators.py`

This is the core consolidation that deletes 80% of bespoke logic.

Provide the following decorator factories:

### A) `decorate_table_inputs(...)` via `pipe_input`

This returns a `pipe_input(...)` decorator with step-level gating via config and stable naming.

```python
from hamilton.function_modifiers import pipe_input, step, value, resolve_from_config

from .tabular_steps import _drop_bad_rows_pandas, _drop_bad_rows_polars, _clip_numeric, _normalize_nulls

def _pipe_cleaning(df_backend: str, clean_mode: str, null_policy: str, max_loc_clip: int):
    if clean_mode == "off":
        return lambda fn: fn  # identity decorator
    drop = _drop_bad_rows_polars if df_backend.startswith("polars") else _drop_bad_rows_pandas
    return pipe_input(
        step(drop, required_cols=value(("loc", "cyclo"))).when(clean_mode="strict"),
        step(_normalize_nulls, policy=value(null_policy)).named("nulls", namespace="prep"),
        step(_clip_numeric, col=value("loc"), max_value=value(max_loc_clip)).named("loc_clip", namespace="prep"),
        on_input="df",     # explicit to avoid accidental first-arg piping
        namespace="prep",
    )

def pipe_clean_df():
    return resolve_from_config(decorate_with=_pipe_cleaning)
```

Usage becomes:

```python
@pipe_clean_df()
def some_table(df: Frame, ...) -> Frame: ...
```

### B) `decorate_table_features(...)` via `with_columns` + `resolve_from_config`

Select column ops per table_key and backend at compile-time:

```python
from hamilton.function_modifiers import resolve_from_config
from .with_columns_backend import select_with_columns

def _decorate_features(df_backend: str, feature_sets: dict, table_key: str, columns_to_pass: tuple[str,...], ops_module):
    selected = feature_sets.get(table_key, ())
    wc = select_with_columns(df_backend)
    return wc(
        ops_module,
        columns_to_pass=list(columns_to_pass),
        select=list(selected),
        namespace=f"feat__{table_key}",
    )

def with_features(*, table_key: str, columns_to_pass: tuple[str,...], ops_module):
    return resolve_from_config(
        decorate_with=lambda df_backend, feature_sets: _decorate_features(
            df_backend=df_backend,
            feature_sets=feature_sets,
            table_key=table_key,
            columns_to_pass=columns_to_pass,
            ops_module=ops_module,
        )
    )
```

### C) `decorate_table_outputs(...)` via `mutate` (module-local cross cutting)

Because `mutate` currently targets functions in the same module, we use it to apply post-processing to many tables **within consolidated table modules** (see §5).

Pattern:

* keep all table-producing functions for a domain in *one module*
* define mutators in the same module and apply with `mutate(apply_to(...), ...)`

Optionally config-gate the mutate application using `resolve_from_config` returning identity vs mutate decorator.

---

# 4) Establish “column subDAG” modules (feature ops) per table family

These modules contain only **pure, column-level** functions; they do not build full tables. They are consumed by `with_columns`.

### Files CREATED

5. `src/codeintel/build/hamilton/column_ops/__init__.py`
6. `src/codeintel/build/hamilton/column_ops/function_features.py`
7. `src/codeintel/build/hamilton/column_ops/module_features.py`
8. `src/codeintel/build/hamilton/column_ops/risk_features.py`
   (Exact partitioning by table family; keep them small and stable.)

Guidelines:

* Each function is a single column op:

  * pandas backend: inputs are `pd.Series`
  * polars eager backend: inputs are `pl.Series`
  * polars lazy backend: inputs are `pl.Expr`
* All ops must be namespace-safe (names are node names; collisions are managed by `namespace=` in `with_columns`).

The `VariantConfig.feature_sets` must only reference ops defined in these modules (validated in `VariantConfig.validate()`).

---

# 5) Consolidate analytics tables into a small number of “table modules” to exploit `mutate`

To use `mutate` effectively (module-local targeting), we intentionally **merge** the currently fragmented analytics modules into consolidated table modules. This is the “delete legacy immediately” part.

## 5.1 New consolidated table modules

### Files CREATED

9. `src/codeintel/build/hamilton/native/analytics/tables_functions.py`
10. `src/codeintel/build/hamilton/native/analytics/tables_modules.py`
11. `src/codeintel/build/hamilton/native/analytics/tables_risk.py`
12. `src/codeintel/build/hamilton/native/analytics/tables_dependencies.py`
13. `src/codeintel/build/hamilton/native/analytics/tables_coverage.py`

Each module contains:

* **raw table constructors** (minimal logic, no cleanup/no backend branching)
* **decorators** applied uniformly:

  * `@pipe_clean_df()` (pipe_input)
  * `@with_features(table_key=..., ...)` (with_columns)
* module-local `@mutate(...)` to apply canonicalization/schema enforcement across multiple table nodes

### Example pattern (module-local)

```python
# tables_functions.py (illustrative)
from __future__ import annotations
from hamilton.function_modifiers import mutate, apply_to, value, resolve_from_config
from codeintel.build.hamilton.transforms.decorators import pipe_clean_df, with_features
from codeintel.build.hamilton.column_ops import function_features
from codeintel.build.hamilton.transforms.tabular_steps import _sort_columns, _cast_schema

@pipe_clean_df()
@with_features(table_key="functions", columns_to_pass=("symbol","loc","cyclo"), ops_module=function_features)
def functions_table(df: "Frame", env: "BuildEnv") -> "Frame":
    # minimal: assemble base columns; no branching
    return df

def _maybe_mutate(enable_canonicalization: bool, enable_schema_enforcement: bool):
    decs = []
    if enable_canonicalization:
        decs.append(mutate(
            apply_to(functions_table, column_order=value(("repo","commit","symbol","loc","cyclo"))),
        ))
    if enable_schema_enforcement:
        decs.append(mutate(
            apply_to(functions_table, schema=value({"symbol":"string","loc":"int64","cyclo":"int64"})),
        ))
    # chain decorators in order; identity if none
    def chain(fn):
        out = fn
        for d in decs:
            out = d(out)
        return out
    return chain

@resolve_from_config(decorate_with=_maybe_mutate)
def _postprocess(df: "Frame", column_order=None, schema=None) -> "Frame":
    if column_order is not None:
        df = _sort_columns(df, column_order)
    if schema is not None:
        df = _cast_schema(df, schema)
    return df
```

This gives you:

* no repeated canonicalization code inside table bodies
* config-driven enable/disable
* table-specific parameters via `apply_to(..., column_order=value(...))`

## 5.2 Legacy analytics modules deleted (immediate deprecation)

### Files DELETED

Delete the fragmented modules replaced by the consolidated files above:

* `src/codeintel/build/hamilton/native/analytics/classification_targets.py`
* `src/codeintel/build/hamilton/native/analytics/config_graph_targets.py`
* `src/codeintel/build/hamilton/native/analytics/coverage_targets.py`
* `src/codeintel/build/hamilton/native/analytics/dependency_targets.py`
* `src/codeintel/build/hamilton/native/analytics/function_detail_targets.py`
* `src/codeintel/build/hamilton/native/analytics/function_metrics.py`
* `src/codeintel/build/hamilton/native/analytics/hotspots.py`
* `src/codeintel/build/hamilton/native/analytics/metadata_targets.py`
* `src/codeintel/build/hamilton/native/analytics/metrics_targets.py`
* `src/codeintel/build/hamilton/native/analytics/risk_factors.py`
* `src/codeintel/build/hamilton/native/analytics/subsystem_cache_targets.py`
* `src/codeintel/build/hamilton/native/analytics/subsystem_targets.py`

(Any additional `native/analytics/*.py` files containing table logic should be deleted or folded into the new `tables_*.py` modules.)

### Files MODIFIED

* `src/codeintel/build/hamilton/native/analytics/__init__.py`

  * Export only the new consolidated modules.
  * Remove all legacy exports.

---

# 6) Apply the same consolidation strategy to other “variant-heavy” subsystems

## 6.1 Ingestion: collapse strict/lenient preprocessing with `pipe_input`

### Files CREATED

14. `src/codeintel/build/hamilton/native/ingestion/pipelines.py`

* Defines ingestion-specific steps:

  * path normalization, decoding, record filtering
* Exposes `pipe_*` decorator factories (pipe_input wrappers) keyed by `clean_mode` / `null_policy`

### Files MODIFIED

* `src/codeintel/build/hamilton/native/ingestion/scip.py`
* `src/codeintel/build/hamilton/native/ingestion/scip_proto.py`
* `src/codeintel/build/hamilton/native/ingestion/ingest_targets.py`
* `src/codeintel/build/hamilton/native/ingestion/extraction_targets.py`

Refactor: remove all inline cleanup branches; instead:

* apply `@pipe_input(...)` to normalize/validate ingestion records before downstream transforms
* use step gating `.when(clean_mode="strict")` for strict-only validations

## 6.2 Graph construction: collapse backend selection with `resolve_from_config` returning `inject`

Where graph construction has “variant algorithm selection” (e.g., NetworkX vs custom), do not branch in node bodies; instead, use `resolve_from_config` to **rewire dependencies**.

### Files CREATED

15. `src/codeintel/build/hamilton/native/graphs/variants.py`

* Defines `resolve_from_config` selectors returning `inject(...)` mapping to the chosen upstream implementation nodes.
* Pattern:

```python
from hamilton.function_modifiers import resolve_from_config, inject, source

def _pick_graph_impl(graph_backend: str):
    if graph_backend == "networkx":
        return inject(g=source("call_graph_nx"))
    if graph_backend == "compact":
        return inject(g=source("call_graph_compact"))
    raise ValueError(graph_backend)

@resolve_from_config(decorate_with=_pick_graph_impl)
def call_graph(g) -> "CallGraph":
    return g
```

### Files MODIFIED

* `src/codeintel/build/hamilton/native/graphs/call_graph.py`
* `src/codeintel/build/hamilton/native/graphs/import_graph.py`
* `src/codeintel/build/hamilton/native/graphs/cfg_dfg.py`
* `src/codeintel/build/hamilton/native/graphs/graph_targets.py`

Refactor: split implementations into explicitly named nodes (`*_nx`, `*_compact`), then use injected alias nodes as the canonical surface.

---

# 7) Introduce a single “table node decorator” to eliminate per-node boilerplate

Most code shrink comes from preventing every new table node from manually stacking 6 decorators.

### Files CREATED

16. `src/codeintel/build/hamilton/transforms/table_contract.py`

Define one decorator factory to apply:

* `pipe_clean_df()` (pipe_input)
* `with_features(...)` (with_columns)
* tagging (domain/target/table_key)
* optional output canonicalization hooks (module-local mutate remains)

Example:

```python
from __future__ import annotations
from codeintel.build.hamilton.transforms.decorators import pipe_clean_df, with_features
from codeintel.build.hamilton.tagging import tag_table  # or your canonical tag helpers

def table_contract(*, table_key: str, domain: str, target: str, ops_module, columns_to_pass: tuple[str,...]):
    def deco(fn):
        fn = tag_table(table_key=table_key, domain=domain, target=target)(fn)
        fn = pipe_clean_df()(fn)
        fn = with_features(table_key=table_key, columns_to_pass=columns_to_pass, ops_module=ops_module)(fn)
        return fn
    return deco
```

Then every table definition becomes:

```python
@table_contract(table_key="functions", domain="analytics", target="functions", ops_module=function_features, columns_to_pass=("symbol","loc","cyclo"))
def functions_table(df: Frame, env: BuildEnv) -> Frame: ...
```

This is the “anti-footgun” design: **new tables inherit policy automatically**.

---

# 8) Remove legacy transform utilities and inline cleaning code

This phase assumes immediate deletion; therefore you must delete:

* any bespoke `normalize_df`, `clean_*`, `ensure_*` helper modules that are now redundant
* any duplicated “clip/cast/drop rows” logic embedded in node bodies

### Files DELETED (canonical set)

* Any module under `src/codeintel/build/**` or `src/codeintel/build/hamilton/native/**` whose sole purpose is dataframe cleanup or backend branching; specifically delete/replace:

  * `src/codeintel/build/hamilton/native/analytics/*` legacy list (see §5.2)
  * any `*_pandas.py`/`*_polars.py` duplication if it becomes unused after `with_columns_backend.py` + `tabular_steps.py`

### Files MODIFIED (sweep)

* All remaining `src/codeintel/build/hamilton/native/**` table-producing modules:

  * remove inline backend branching
  * remove inline cleanup blocks
  * replace with `@table_contract(...)` or the underlying `pipe_clean_df/with_features` stack

---

# 9) Driver module list update (new modules must be included)

### Files MODIFIED

* `src/codeintel/runtime/module_resolver.py` (phase 6 artifact)

  * Ensure new consolidated modules are discovered/included:

    * `native/analytics/tables_*.py`
    * `native/ingestion/pipelines.py`
    * `native/graphs/variants.py`
    * `column_ops/*` modules (if they’re imported by table modules, explicit inclusion not strictly required, but safe to include as part of module set)

* `src/codeintel/runtime/compose.py`

  * Ensure config contains `VariantConfig` keys referenced by `resolve_from_config` callables:

    * `df_backend`
    * `clean_mode`
    * `feature_sets`
    * `enable_*` toggles

> Important: config keys must match callable parameter names in `resolve_from_config(decorate_with=...)`.

---

# 10) Tests: enforce “no variant branching in bodies” + validate DAG shape changes with config

### Files CREATED

17. `tests/variants/test_no_variant_branching_in_nodes.py`

* AST-based test that rejects `if <config>` branching inside DAG node bodies under:

  * `src/codeintel/build/hamilton/native/**`
* Heuristic:

  * disallow reading `cfg.*` or `env.config.*` inside node bodies
  * allow only in `resolve_from_config` callables in dedicated transform modules

18. `tests/variants/test_resolve_from_config_changes_dag_shape.py`

* Build driver twice with two variant configs:

  * `df_backend="pandas"` vs `"polars"`
  * `feature_sets={"functions": ("base_risk","loc_bucket")}` vs a different list
* Assert:

  * different sets of nodes exist under namespaces `feat__functions.*`
  * stable canonical node names exist (table nodes unchanged)
  * the set of materialized outputs remains invariant unless explicitly configured

19. `tests/variants/test_pipe_input_step_gating.py`

* For `clean_mode="strict"`, assert `prep.nulls`/`prep.loc_clip` intermediate nodes exist (or that pipeline steps are present in graph / executed).
* For `clean_mode="off"`, assert they are absent (identity decorator).

---

# 11) File index summary (additions / deletions / modifications)

## Files CREATED

* `src/codeintel/runtime/variants.py`
* `src/codeintel/build/hamilton/transforms/tabular_steps.py`
* `src/codeintel/build/hamilton/transforms/with_columns_backend.py`
* `src/codeintel/build/hamilton/transforms/decorators.py`
* `src/codeintel/build/hamilton/transforms/table_contract.py`
* `src/codeintel/build/hamilton/column_ops/__init__.py`
* `src/codeintel/build/hamilton/column_ops/function_features.py`
* `src/codeintel/build/hamilton/column_ops/module_features.py`
* `src/codeintel/build/hamilton/column_ops/risk_features.py`
* `src/codeintel/build/hamilton/native/analytics/tables_functions.py`
* `src/codeintel/build/hamilton/native/analytics/tables_modules.py`
* `src/codeintel/build/hamilton/native/analytics/tables_risk.py`
* `src/codeintel/build/hamilton/native/analytics/tables_dependencies.py`
* `src/codeintel/build/hamilton/native/analytics/tables_coverage.py`
* `src/codeintel/build/hamilton/native/ingestion/pipelines.py`
* `src/codeintel/build/hamilton/native/graphs/variants.py`
* `tests/variants/test_no_variant_branching_in_nodes.py`
* `tests/variants/test_resolve_from_config_changes_dag_shape.py`
* `tests/variants/test_pipe_input_step_gating.py`

## Files MODIFIED

* `src/codeintel/runtime/compose.py` (power-user mode + pass VariantConfig keys)
* `src/codeintel/build/config.py` (or runtime config module) to embed `VariantConfig` and delete legacy flags
* `src/codeintel/runtime/runtime_bundle.py` (add variants + fingerprinting)
* `src/codeintel/runtime/module_resolver.py` (include consolidated modules)
* `src/codeintel/build/hamilton/native/ingestion/{scip.py,scip_proto.py,ingest_targets.py,extraction_targets.py}` (remove inline variants; apply pipelines)
* `src/codeintel/build/hamilton/native/graphs/{call_graph.py,import_graph.py,cfg_dfg.py,graph_targets.py}` (split impls + inject alias)
* Any remaining `native/**` modules producing tables: replace inline transforms with `@table_contract(...)`

## Files DELETED (immediate legacy purge)

* `src/codeintel/build/hamilton/native/analytics/classification_targets.py`
* `src/codeintel/build/hamilton/native/analytics/config_graph_targets.py`
* `src/codeintel/build/hamilton/native/analytics/coverage_targets.py`
* `src/codeintel/build/hamilton/native/analytics/dependency_targets.py`
* `src/codeintel/build/hamilton/native/analytics/function_detail_targets.py`
* `src/codeintel/build/hamilton/native/analytics/function_metrics.py`
* `src/codeintel/build/hamilton/native/analytics/hotspots.py`
* `src/codeintel/build/hamilton/native/analytics/metadata_targets.py`
* `src/codeintel/build/hamilton/native/analytics/metrics_targets.py`
* `src/codeintel/build/hamilton/native/analytics/risk_factors.py`
* `src/codeintel/build/hamilton/native/analytics/subsystem_cache_targets.py`
* `src/codeintel/build/hamilton/native/analytics/subsystem_targets.py`
* Any bespoke dataframe-cleaning helper modules that become unused after adopting `tabular_steps.py` + `decorators.py` (delete aggressively; enforce via import graph).

---

# 12) Definition of Done (hard gates)

1. **No runtime branching for backend selection** exists in DAG node bodies (AST guard passes).
2. All supported variants are expressed as:

   * `resolve_from_config → with_columns/pipe_input/inject/mutate` (compile-time graph edits)
   * step-level gating via `.when(...)` inside pipe pipelines
3. “Cross-cutting transforms” (cleanup, canonicalization, schema enforcement, column enrichment) are centralized in:

   * `transforms/*` + `column_ops/*`
   * applied via `table_contract(...)` and module-local `mutate(...)`
4. Analytics subsystem collapses from many ad hoc modules into a small number of table modules, enabling `mutate` fan-out.
5. Changing `VariantConfig.feature_sets` changes only the namespaced feature subDAG (`feat__{table_key}` nodes), not the canonical table node name or output inventory.
6. For each `table_key`, the DAG explicitly shows:

   * preprocessing pipeline nodes (namespaced `prep.*`), if enabled
   * feature subDAG nodes (namespaced `feat__table_key.*`), if enabled
   * postprocessing mutators, if enabled

---

## Why this is the “maximal shrink” design

* `resolve_from_config` moves configuration variability to **compile-time decorator synthesis**, so the runtime is branch-free and introspectable.
* `pipe_input` makes preprocessing DAG-visible and config-gated without duplicating logic.
* `with_columns` converts repeated column engineering blocks into **small reusable pure functions** and keeps a column-level lineage DAG.
* `mutate` enables “policy application” (canonicalization/schema enforcement) across many tables without touching each function body, but only if you **consolidate tables per module**—hence the deliberate analytics module merge.

If you want the next follow-on after (7), the natural progression is: **(8) “Replace custom tag indexing with Hamilton tag filtering as the base primitive”** + **(9) “DuckDB Relation + Arrow/Polars end-to-end”**, because once transforms/variants are canonicalized, you can lock the internal table representation and eliminate conversions wholesale.


Below is a **repo-concrete, breaking-change–friendly**, **high lexical density** implementation plan for:

> **(8) Replace custom tag indexing with Hamilton tag filtering as the base primitive**

Assumptions (per prior phases): single composition root (`compose_runtime()`), `RuntimeBundle` exists, `DagCatalog` is the canonical derived metadata view, outputs are saver-derived, caching is Hamilton-native, and any surviving “build drivers inside random subsystems” patterns are already being eliminated.

This phase’s design invariant is narrowly scoped but extremely high leverage:

> Any time code wants “nodes that satisfy tag predicates”, it must use **Hamilton’s native tag query surface**:
> `Driver.list_available_variables(tag_filter=...)` (AND, exact match, multi-value, tag-exists).

No bespoke `TagIndex(tags_by_node=...)`, no pre-normalization/stringification, no parallel query semantics.

---

# 0) End-state contract

## 0.1 Canonical tag query primitive

* A single primitive is used everywhere:

  * `dr.list_available_variables(tag_filter={...})`
* `tag_filter` semantics used explicitly:

  * **exact match**: `{"node_type": "dataset"}`
  * **multi-values**: `{"node_type": ["dataset", "artifact"]}`
  * **tag exists**: `{"semantic_id": None}` (key must exist, any value)
  * **AND** across keys: `{"output_kind": "semantic_view", "mcp_visible": "1", "table_key": None}`

## 0.2 No custom tag normalization layer

* Tag values must be emitted in canonical types at decoration time:

  * booleans remain booleans (e.g. `hamilton.data_saver=True`)
  * enumerations/taxonomy remain strings (`node_type`, `domain`, `target`, `table_key`, `artifact`)
  * “list-like tags” remain `list[str]` (if you use them) rather than CSV-joined strings
* Any “truthiness” normalization (e.g. `"1"/"true"/"yes"`) is treated as legacy and deleted. Query semantics become type-stable.

## 0.3 Tag queries do not leak into “graph compiler” logic

* `DagCatalog` compilation may still iterate the full graph when it needs edges.
* But **selection** of subsets of nodes by tag is always done via tag filtering, not via `for node in dr.graph.nodes.values(): if ...`.

---

# 1) Delete the custom tag index (hard removal)

### Files DELETED

* `src/codeintel/build/hamilton/tag_index.py`

This removes:

* `TagIndex.from_runtime`, `TagIndex.from_modules`
* `tags_by_node` as a persistent mapping
* normalization/stringification/truthy logic that diverges from Hamilton’s semantics

### Files MODIFIED

* `src/codeintel/build/hamilton/__init__.py` (or wherever TagIndex is exported)

  * remove any export of `TagIndex`
* Any import sites updated per §3–§6

---

# 2) Introduce a canonical TagQuery service (driver-backed, cacheable, semantics-preserving)

While the base primitive is Hamilton’s `tag_filter`, you still want:

* consistent filter construction (no scattered tag dict literals),
* optional query memoization for repeated calls (serving endpoints),
* typed return helpers (turn “variable” objects into stable dict payloads).

### Files CREATED

## 2.1 `src/codeintel/runtime/tag_query.py`

Core implementation:

* `TagFilter = Mapping[str, object]`

* `HamiltonVar` protocol / structural type for `list_available_variables()` return elements:

  * `.name: str`
  * `.tags: dict[str, object]`
  * `.type: object | None`
  * `.is_external_input: bool`

* `TagQuery` (frozen/slots)

  * `driver`
  * `query(tag_filter: TagFilter) -> tuple[HamiltonVar, ...]`
  * `one(tag_filter: TagFilter) -> HamiltonVar | None` (optional)
  * `names(tag_filter: TagFilter) -> tuple[str, ...]`
  * internal memoization keyed by normalized `tag_filter` (optional but recommended)

**Memoization must not reimplement filter semantics**—it must only cache *results of `driver.list_available_variables(...)`*.

Minimal caching key normalizer (important because dict/list are not hashable):

```python
def _freeze_filter(tag_filter: Mapping[str, object]) -> tuple[tuple[str, object], ...]:
    def freeze(v: object) -> object:
        if isinstance(v, list):
            return tuple(v)
        if isinstance(v, dict):
            return tuple(sorted((str(k), freeze(val)) for k, val in v.items()))
        return v
    return tuple(sorted((str(k), freeze(v)) for k, v in tag_filter.items()))
```

`TagQuery.query()` then:

```python
class TagQuery:
    def __init__(self, dr):
        self._dr = dr
        self._cache: dict[tuple[tuple[str, object], ...], tuple[object, ...]] = {}

    def query(self, tag_filter: Mapping[str, object]) -> tuple[object, ...]:
        key = _freeze_filter(tag_filter)
        if key not in self._cache:
            self._cache[key] = tuple(self._dr.list_available_variables(tag_filter=dict(tag_filter)))
        return self._cache[key]
```

---

# 3) Add canonical tag-filter builders (single source for filter dicts)

This prevents ad hoc filter dicts from diverging over time.

### Files CREATED

## 3.1 `src/codeintel/core/hamilton/tag_filters.py`

Define reusable filter constructors for your taxonomy:

```python
from __future__ import annotations
from typing import Any

from codeintel.core.hamilton import tags as ht

def tf_datasets(*, table_key: str | None = None) -> dict[str, Any]:
    f: dict[str, Any] = {ht.TAG_NODE_TYPE: ht.NODE_TYPE_DATASET}
    if table_key is None:
        f[ht.TAG_TABLE_KEY] = None  # existence
    else:
        f[ht.TAG_TABLE_KEY] = table_key
    return f

def tf_artifacts(*, artifact: str | None = None) -> dict[str, Any]:
    f: dict[str, Any] = {ht.TAG_NODE_TYPE: ht.NODE_TYPE_ARTIFACT}
    f[ht.TAG_ARTIFACT] = None if artifact is None else artifact
    return f

def tf_semantic_views() -> dict[str, Any]:
    return {
        ht.TAG_OUTPUT_KIND: ht.OUTPUT_KIND_SEMANTIC_VIEW,
        ht.TAG_TABLE_KEY: None,
        ht.TAG_SEMANTIC_ID: None,
        ht.TAG_MCP_VISIBLE: "1",
    }

def tf_savers(*, role: str | None = None, sink: str | None = None) -> dict[str, Any]:
    f: dict[str, Any] = {"hamilton.data_saver": True}
    if role is not None:
        f["output_role"] = role
    if sink is not None:
        f["hamilton.data_saver.sink"] = sink
    return f
```

> You can keep this module dependency-light and import-only (no Driver import). It becomes the taxonomy ABI.

---

# 4) Wire TagQuery into the runtime bundle (so all subsystems consume the same query object)

### Files MODIFIED

* `src/codeintel/runtime/runtime_bundle.py`

  * add:

    * `tag_query: TagQuery`
* `src/codeintel/runtime/compose.py`

  * after driver build:

    * `tag_query = TagQuery(driver)`
  * store into bundle

This ensures:

* serving endpoints and build exports never construct ad hoc tag scanning helpers
* query caching is localized to the runtime’s lifetime

---

# 5) Replace TagIndex use sites with TagQuery + tag_filter (semantic registry, serving artifacts, CLI, etc.)

## 5.1 Semantic registry compilation

### Files MODIFIED

* `src/codeintel/build/serving/semantic_compile.py`

**Delete** `compile_semantic_registry_from_tag_index(...)`.

**Replace** with driver-backed compilation:

```python
from codeintel.core.hamilton.tag_filters import tf_semantic_views

def compile_semantic_registry_from_driver(*, schema_provider, dr, version="v1") -> CompiledSemanticRegistry:
    vars_ = dr.list_available_variables(tag_filter=tf_semantic_views())
    # translate vars_ -> view_tags mapping or compile directly
```

Prefer compiling directly from returned variable objects instead of building intermediate `view_tags` dict.

### Files MODIFIED (exports)

* Update `__all__` accordingly; remove tag_index entrypoints.

---

## 5.2 Serving artifacts export target

### Files MODIFIED

* `src/codeintel/build/hamilton/native/export/serving_artifacts.py`

**Remove**:

* `from codeintel.build.hamilton.tag_index import TagIndex`
* `TagIndex.from_modules(modules=(_ibis_views,))`
* `compile_semantic_registry_from_tag_index(...)`

**Replace** semantic registry construction with runtime driver query:

* Under single composition root (phase 6), this module should already receive `runtime`/`driver`/`catalog` inputs; choose the “official” injection (typically `runtime_bundle` fields or `env + catalog + tag_query`).

For example:

```python
def _semantic_registry_json(dr) -> str:
    schema_provider = get_schema_provider()
    compiled = compile_semantic_registry_from_driver(schema_provider=schema_provider, dr=dr, version="v1")
    return compiled.to_json() + "\n"
```

If you’ve already moved to `RuntimeBundle` injection:

* accept `tag_query: TagQuery` and call `tag_query.query(tf_semantic_views())`
* pass `dr` only if you need types/graph

> This eliminates both TagIndex and any secondary driver instantiation.

---

## 5.3 Storage view discovery already uses tag_filter, but still builds its own driver

In `src/codeintel/storage/views/discovery.py` you currently do `dr = Driver(config or {}, *modules)` and then `dr.list_available_variables(tag_filter=...)`. Under SCR this must become a pure consumer of an existing driver/tag_query.

### Files MODIFIED

* `src/codeintel/storage/views/discovery.py`

**Change API**:

* from: `discover_view_builders(modules, config=None)`
* to: `discover_view_builders(*, dr, modules=None)` or (preferred) `discover_view_builders(*, runtime: RuntimeBundle)`

**Implementation**:

* use `dr.list_available_variables(tag_filter={ht.TAG_OUTPUT_KIND: ...})` (already correct)
* locate builder callables:

  * preferred: `callable = dr.graph.nodes[node_name].callable` (if available in Hamilton Node)
  * fallback: keep `_find_callable(modules, node_name)` but ensure modules are passed from composition root, not re-discovered here

This makes view discovery a “tag query + callable resolution” problem, not a “build a driver to query tags” problem.

---

## 5.4 CLI / operational helpers that previously consulted TagIndex.tags_by_node

Any usage of `tags_by_node` for domain/target resolution must be replaced with:

* `catalog.targets[target].domain` (preferred; avoids tag query entirely)
* or `dr.graph.nodes[anchor_node].tags[domain]` if needed

### Files MODIFIED

* `src/codeintel/cli/handlers/build.py`

  * `_resolve_domain_for_goals(...)` should become:

    * `domain = runtime.catalog.targets[target_name].domain` (single source)
  * delete any TagIndex usage

---

# 6) Replace “manual tag scanning loops” with tag_filter queries (enforced by lint)

Even after deleting `TagIndex`, engineers/agents will reintroduce manual loops like:

```python
for name, node in dr.graph.nodes.items():
    if node.tags.get("node_type") == "...":
        ...
```

That recreates a second query semantic surface (and diverges subtly from Hamilton’s tag semantics). You want a policy gate.

### Files CREATED

## 6.1 `tools/lint_no_manual_tag_scans.py`

AST-based linter:

* forbid:

  * loops over `.graph.nodes` paired with `.tags.get(...)` in non-whitelisted modules
* allow:

  * `dag_catalog_compiler.py` (needs edges)
  * `validate.py` (if it must inspect tags deeply)
* enforce:

  * any “selection by tag predicate” must call:

    * `list_available_variables(tag_filter=...)` or `runtime.tag_query.query(...)`

Wire into CI (pytest invocation or pre-commit).

### Files MODIFIED

* CI config / `pyproject.toml` / `tox.ini` / `pytest.ini` (wherever you wire project linters)

---

# 7) Tighten tag-type discipline (so tag_filter is stable)

Once you stop stringifying tags, tag types matter. Enforce this at tag emission time and in validation.

### Files MODIFIED

* `src/codeintel/build/hamilton/validate.py`

  * add validator passes:

    * `mcp_visible` must be `"0"|"1"` (string)
    * `node_type` must be one of taxonomy strings
    * `hamilton.data_saver` must be boolean if present
    * `output_role` must be `"contract"|"internal"` if present
    * any list-valued tags must be `list[str]` (not CSV string) **or** ban list tags entirely unless you explicitly use them

This ensures `tag_filter={"hamilton.data_saver": True}` works globally.

---

# 8) Update remaining tag-based derivations to use tag_filter or catalog (no custom indexes)

This phase is not about reintroducing new indices. Any consumer that needs:

* datasets by table_key,
* semantic view nodes,
* saver nodes by sink,

must use either:

* catalog-derived indexes (preferred when you need target semantics), or
* tag_filter (preferred when you need node discovery semantics).

### Guidance rule

* If the question is **target/output** oriented → use `DagCatalog` (`targets`, `table_outputs`, `artifact_outputs`, `io`).
* If the question is **node discovery** oriented → use `TagQuery` (`list_available_variables(tag_filter=...)`).

---

# 9) Tests: prove equivalence and prevent regression

### Files CREATED

1. `tests/tags/test_tag_filter_discovery_semantics.py`

* Build a minimal driver fixture with nodes tagged:

  * dataset
  * artifact
  * semantic_view
  * data_saver
* Assert `TagQuery.query(tf_*)` returns the correct nodes
* Assert “tag exists” (`None`) matches presence semantics (semantic_id/table_key).

2. `tests/serving/test_semantic_registry_compiles_from_driver_tags.py`

* Compile semantic registry from a driver + schema_provider
* Assert deterministic ordering and content

3. `tests/lint/test_no_manual_tag_scans.py`

* Runs `tools/lint_no_manual_tag_scans.py` over repo tree

### Files DELETED

* Any tests specifically asserting `TagIndex.tags_by_node` behavior (remove entirely)

---

# 10) File index summary

## Created

* `src/codeintel/runtime/tag_query.py`
* `src/codeintel/core/hamilton/tag_filters.py`
* `tools/lint_no_manual_tag_scans.py`
* `tests/tags/test_tag_filter_discovery_semantics.py`
* `tests/serving/test_semantic_registry_compiles_from_driver_tags.py`
* `tests/lint/test_no_manual_tag_scans.py`

## Modified

* `src/codeintel/runtime/runtime_bundle.py`
* `src/codeintel/runtime/compose.py`
* `src/codeintel/build/serving/semantic_compile.py`
* `src/codeintel/build/hamilton/native/export/serving_artifacts.py`
* `src/codeintel/storage/views/discovery.py`
* `src/codeintel/cli/handlers/build.py`
* `src/codeintel/build/hamilton/validate.py`
* CI wiring / config for lint execution

## Deleted

* `src/codeintel/build/hamilton/tag_index.py`

---

# 11) Definition of Done (hard gates)

1. No `TagIndex` symbol exists; no `tags_by_node` mapping exists anywhere in runtime.
2. All tag-based discovery uses one of:

   * `runtime.tag_query.query(...)` (preferred)
   * `runtime.driver.list_available_variables(tag_filter=...)` (direct)
3. Manual tag scanning loops are prohibited outside allowlisted compiler/validator modules (linter enforced).
4. Semantic registry compilation does not require TagIndex; it compiles directly from driver tag-filter results.
5. View discovery does not build its own driver for tag discovery; it consumes the already composed driver/runtime bundle.
6. Tag type invariants are validated (boolean tags stay boolean; canonical string tags stay strings).

---

## Architectural payoff

Once this lands, tags become a **first-class query ABI** with one implementation: Hamilton’s tag filter engine. The codebase stops growing ad hoc “mini registries” (`TagIndex`, view registries, schema registries based on scanning dicts), and all semantic surfaces become either:

* **catalog-derived** (targets/outputs/IO), or
* **driver-derived** (node discovery via tags).

That is exactly the coherence property you want for extensibility: new node types and new serving surfaces are added by **tagging**, not by wiring yet another index.

Below is a **comprehensive, file-explicit, breaking-change-friendly implementation plan** for:

**(9) Major performance + code shrink: adopt DuckDB Relation API + Arrow/Polars end-to-end**

This plan is written to be executed by an AI LLM programming agent with strong Python systems competence, and assumes **all legacy (pandas/Ibis-first) codepaths are immediately deprecated and deleted**.

---

## 9.0 Design target and invariants

### 9.0.1 Canonical tabular “compute” type

Make **DuckDB’s `DuckDBPyRelation`** the *dominant internal representation* for all tabular computations:

* **Inside the DAG**: nodes consume/produce `duckdb.DuckDBPyRelation`.
* **At IO boundaries only**: convert to/from `pyarrow.Table` / `pyarrow.RecordBatchReader` / `polars.{DataFrame,LazyFrame}` when necessary.
* **Never** use pandas as an interchange format (only tolerated for thin compatibility if you can’t delete it immediately; but this plan assumes deletion).

This aligns with DuckDB’s lazy Relation semantics: method chaining builds an optimized plan; execution happens only when materializing outputs. DuckDB’s relational API supports `filter/project/join/aggregate/order` chains and remains lazy until you request results. 

### 9.0.2 Canonical zero-copy interchange

Standardize Arrow as the physical interchange format:

* DuckDB scans Arrow objects extremely efficiently (Arrow C++ interface; no copy) and can treat Arrow tables as queryable relations. 
* DuckDB relations can be materialized as Arrow tables or **streamed** as Arrow readers (RecordBatchReader) rather than forcing full-table materialization. 

### 9.0.3 Immediate architectural motivation grounded in Phase 6 architecture

Your Phase 6 architecture explicitly shows:

* “read boundary” loader nodes currently calling **Ibis adapter wrappers** (`load_dataset_df`, `load_dataset_ibis`) 
* “write boundary” saver includes **DuckDB/Ibis** saver `DuckDBIbisTableSaver` 

This is the seam: we replace the entire Ibis/pandas substrate with **DuckDB relation + Arrow/Polars**.

---

## 9.1 Repo-wide refactor strategy (high-level)

### 9.1.1 Delete the Ibis axis entirely

* Remove `ibis` from the build graph and gateway protocol.
* Replace “query loader returns ibis table” with “query loader returns relation”.
* Replace “dataframe loader returns pandas df” with “dataset loader returns relation”.

### 9.1.2 Replace *row-based* and *table-based* materialization with “relation materialization”

Make savers accept a `DuckDBPyRelation` (or coercible Tabular inputs), and implement materialization using:

* `relation.create(...)` / `create_view(...)` / `insert_into(...)` as appropriate
* `relation.to_parquet(...)` for file artifacts
* `relation.fetch_arrow_reader()` for streaming Arrow IPC artifacts 

### 9.1.3 Standardize “tabular coercion” centrally

A single, hardened coercion layer:

* `coerce_to_relation(conn, obj) -> DuckDBPyRelation`
* `coerce_to_arrow_reader(conn, rel) -> pa.RecordBatchReader`
* `coerce_to_polars(rel, lazy=...) -> pl.{DataFrame,LazyFrame}`

DuckDB can register Python objects (Arrow/Polars) via `conn.register(...)`; registration is pointer-based and the engine only pulls data when referenced in queries. 

---

## 9.2 File plan (additions / modifications / deletions)

### 9.2.1 Files to create

#### A) Core tabular types + coercion

1. `src/codeintel/build/tabular/types.py`

* Defines canonical union/protocol types and internal invariants.

2. `src/codeintel/build/tabular/duckdb_relation.py`

* Implements coercion and hardened utilities:

  * `coerce_to_relation(conn, obj, *, name_hint=None) -> DuckDBPyRelation`
  * `register_ephemeral(conn, obj, *, prefix="tmp") -> str` (returns unique view name)
  * `relation_schema(rel) -> pa.Schema` (via Arrow conversion only when required)

#### B) DuckDB context / connection policy

3. `src/codeintel/storage/duckdb/context.py`

* A single “session” object that unifies:

  * connection creation
  * engine configuration (threads, memory_limit, temp_directory)
  * extension loading
  * deterministic naming policy for schemas / temp views

This is also where you encode connection tuning hooks (threads/memory/temp dir) referenced in DuckDB advanced docs. 

#### C) Hamilton IO adapters (relation-native)

4. `src/codeintel/build/hamilton/io/duckdb_relation_adapter.py`

* Replaces the Ibis adapter wrapper module; supplies:

  * `load_dataset_relation(gateway, ref) -> DuckDBPyRelation`
  * `save_relation(gateway, *, rel, table_key, mode, ...) -> MaterializationMetadata`
  * `save_relation_rows(...)` only if you truly need row-level semantics; prefer relation-native CTAS/INSERT.

#### D) Relation-native materializers

5. `src/codeintel/build/hamilton/materializers/duckdb_relation_saver.py`

* Replaces `DuckDBIbisTableSaver` with relation-native saver.
* Implements **CTAS/INSERT/UPSERT** depending on contract semantics.

6. `src/codeintel/build/hamilton/materializers/arrow_ipc_artifact_saver.py`

* Writes Arrow IPC streams/files with explicit IPC options (compression, unify dictionaries, threads). PyArrow’s IPC options are the “real advanced surface”; use them deliberately. 

(If you prefer fewer files: fold IPC writing into existing `artifact_saver.py`; but creating a dedicated saver reduces accidental format drift and encourages reuse.)

---

### 9.2.2 Files to modify

#### A) Storage gateway protocol: remove Ibis, expose DuckDB surface

1. `src/codeintel/storage/gateway/protocol.py`

* Remove `ibis`-oriented attributes/methods.
* Add:

  * `duckdb: DuckDBContext` (or `duckdb_conn: duckdb.DuckDBPyConnection`)
  * `register(name: str, obj: Any) -> None`
  * `unregister(name: str) -> None`
  * `relation_from_table_key(table_key) -> DuckDBPyRelation`
  * `execute(sql: str, params: ...)` (parameterized, never f-string SQL)

DuckDB parameter substitution is a first-class primitive (positional `?`, named `$name`) and should be used systematically to avoid bespoke string sanitation and to enable prepared statement reuse. 

#### B) Support node generation: replace df__/q__ loaders with relation loaders

2. `src/codeintel/build/hamilton/nodes/support_factory.py`
   Currently:

* `df__*` nodes load pandas DataFrames via `load_dataset_df(...)` 
* `q__*` nodes load Ibis tables via `load_dataset_ibis(...)` 

Change to:

* Remove `_create_dataframe_node_function(...)` entirely.
* Replace with `_create_relation_node_function(...)` which returns `DuckDBPyRelation`.

Mechanically:

* `d__*` still returns `DatasetRef` (fine), but downstream loader nodes become relation loaders:

  * `r__<dataset_name>(env: BuildEnv, **kwargs) -> DuckDBPyRelation`
  * optionally keep `q__*` naming but return relation; consistency beats semantics.

Use `gateway.register(...)` to bind Arrow/Polars objects when needed (pointer-based, lazy pull). 

#### C) Materializers: replace Ibis saver, harden row saver

3. `src/codeintel/build/hamilton/materializers/duckdb_saver.py`

* DELETE (see deletions below); replaced by relation saver.

4. `src/codeintel/build/hamilton/materializers/duckdb_rows_saver.py`

* If retained, it must:

  * accept relation inputs (and coerce others)
  * implement inserts using DuckDB `executemany()` only when truly row-native
  * otherwise pivot to relation CTAS/INSERT plan

Note Phase 6 strict write validation hooks currently occur in `DuckDBRowsSaver.save_data(...)` via `ContractEnforcer.validate_table_write(...)`. 
That logic should be lifted to validate:

* schema contract (via DuckDB table introspection / Arrow schema)
* row count invariants
* mode invariants (append/replace/upsert)

#### D) Artifact saver: add streaming Arrow + Parquet direct paths

5. `src/codeintel/build/hamilton/materializers/artifact_saver.py`
   Extend to support:

* relation → parquet without materializing to Python:

  * `rel.to_parquet(path, ...)` (fast path)
* relation → Arrow IPC stream/file using `fetch_arrow_reader()` and PyArrow IPC writer with explicit `IpcWriteOptions`. 

#### E) Architecture-level documentation: keep updated

6. `architecture_ph6.md`

* Add a Phase 9 delta section describing:

  * removal of Ibis adapter plane
  * relation as internal tabular representation
  * Arrow IPC streaming as artifact primitive
  * supported tabular coercions

---

### 9.2.3 Files to delete

1. `src/codeintel/build/hamilton/io/ibis_adapter.py`

* Entire wrapper layer becomes obsolete once relation-native IO exists. Architecture explicitly shows loaders calling this module today. 

2. `src/codeintel/build/hamilton/materializers/duckdb_saver.py`

* `DuckDBIbisTableSaver` is explicitly Ibis-centric. 

3. Any `src/codeintel/storage/io/*` modules that exist only to support Ibis ingestion/export (repo-specific—delete aggressively once unused).

4. Any “DataFrame loader” codepaths, type aliases, and polymorphic adapters that exist solely to juggle `{pandas, ibis}` duality.

---

## 9.3 Concrete implementation steps (ordered, mechanically executable)

### 9.3.1 Introduce the “tabular substrate” once, then force usage everywhere

#### Step A — Create canonical tabular types and coercion

Create `src/codeintel/build/tabular/types.py`:

* Define:

  * `TabularInput = DuckDBPyRelation | pa.Table | pa.RecordBatchReader | pl.DataFrame | pl.LazyFrame`
  * `TabularRelation = duckdb.DuckDBPyRelation`

Create `src/codeintel/build/tabular/duckdb_relation.py` with **one** coercion function that everything uses:

```python
def coerce_to_relation(conn: duckdb.DuckDBPyConnection, obj: Any, *, name_hint: str | None = None) -> duckdb.DuckDBPyRelation:
    if isinstance(obj, duckdb.DuckDBPyRelation):
        return obj
    view = register_ephemeral(conn, obj, prefix=name_hint or "tmp")
    return conn.view(view)  # or conn.table(view), depending on how you register
```

Key points:

* Prefer explicit `conn.register(name, obj)` over relying on Python-variable lookup in `conn.sql(...)` for determinism and debuggability. DuckDB registration is pointer-based and lazy-pull. 
* Support Polars LazyFrame registration: DuckDB will execute the LazyFrame only when needed (materialized via Arrow). 

#### Step B — Encode “Arrow schema metadata as contract carrier”

When you must attach contract identifiers or schema versions to Arrow schemas, use Arrow’s schema metadata operations (schema is immutable; you create a new schema with metadata). The guide explicitly highlights `Schema.with_metadata(...)` / `remove_metadata(...)`. 

Use this to carry:

* `contract_id`
* `schema_version`
* `grain`
* `primary_key`
* etc.

(Where to store it depends on your contract enforcement layer; but Arrow schema metadata is a useful standardized carriage.)

Also: attach the uploaded PyArrow advanced guide here as reference for the agent. 

---

### 9.3.2 Refactor StorageGateway into a DuckDB-first gateway

#### Step A — Modify `src/codeintel/storage/gateway/protocol.py`

* Remove `ibis` axis.
* Add `duckdb_conn` or `duckdb_ctx`.

Require that all loaders/materializers obtain connection via gateway, not global imports, so the connection policy is centralized (threads/memory/temp dir).

DuckDB connection tuning should be set either at connect-time (`duckdb.connect(config={...})`) or via `SET ...` statements; advanced docs call out threads and memory_limit explicitly. 

#### Step B — Enforce parameterized SQL as a hard invariant

All SQL that includes runtime parameters must be parameterized (`?` or `$name`) and never interpolated. DuckDB supports both positional and named parameters, and prepares/binds safely. 

---

### 9.3.3 Replace support-module loaders: `df__*` and Ibis query loaders → relation loaders

#### Step A — Modify `src/codeintel/build/hamilton/nodes/support_factory.py`

From architecture:

* dataframe_fn calls `load_dataset_df(...)` in `ibis_adapter.py` 
* query loader calls `load_dataset_ibis(...)` 

Rewrite to:

* `load_dataset_relation(...)` in `duckdb_relation_adapter.py`
* loader nodes return `DuckDBPyRelation`

Implementation detail:

* For “produced tables” (outputs of previous targets), use `conn.table("<schema>.<table>")` to get a relation.
* For “external datasets” (parquet/csv artifacts), prefer DuckDB-native scanners (e.g., `from_parquet`) to avoid roundtrips; DuckDB relations can be created from file globs. 

#### Step B — Delete DataFrame loader node class entirely

* Remove any tag taxonomy, node naming, or factory branches that exist only for `df__*`.

This yields immediate code shrink and eliminates “format combinatorics.”

---

### 9.3.4 Replace materializers: Ibis saver → Relation saver, Arrow/Parquet direct emit

#### Step A — Create `duckdb_relation_saver.py`

Implement a saver that accepts `TabularInput`, coerces to relation, then writes using relation-native operations.

Minimum semantics:

* **replace**: `CREATE OR REPLACE TABLE ... AS SELECT * FROM rel`
* **append**: `INSERT INTO ... SELECT * FROM rel`
* **upsert**: use DuckDB MERGE if available, else emulate via staging table + delete/insert.

Your current architecture has a table saver `DuckDBIbisTableSaver` and row saver `DuckDBRowsSaver`; the relation saver supersedes the former. 

#### Step B — Harden `DuckDBRowsSaver` or delete it

If you retain row saver:

* Use it strictly for small, structurally row-native artifacts (e.g., run records), not large tables.
* For upserts, avoid Python loops: stage into relation then MERGE/INSERT.

#### Step C — Artifact saver: stream Arrow IPC and avoid full materialization

Modify `artifact_saver.py` (or add a dedicated saver) so that if `data` is a relation:

* prefer `rel.to_parquet(path)` for parquet artifacts (engine direct path)
* prefer streaming Arrow IPC for IPC artifacts:

Use **explicit IPC write options**:

* compression (`zstd` typically)
* threading enabled
* dictionary unification when writing file format / columnar dictionaries are present

PyArrow’s IPC advanced surface is exactly these knobs. 

Also: implement safe defaults (compat mode vs perf mode) exactly once in a shared helper so you don’t replicate options in 8 places.

---

### 9.3.5 Rewrite native target modules to operate on Relations (and only drop into Polars/Arrow surgically)

This is the large but straightforward mechanical rewrite:

#### Pattern A — “Relation algebra first”

Any node that used to do:

* pandas filtering/groupby/join
* ibis expression building
  should become one of:
* relation chaining: `rel.filter(...).project(...).join(...).aggregate(...)` 
* `conn.sql(<sql>, params)` returning a relation

#### Pattern B — “Polars as an escape hatch for non-SQL transforms”

When you truly need Polars:

* convert relation → `pl_df = rel.pl()` or `rel.pl(lazy=True)` depending on whether you want deferred Polars. 
* do Polars transforms
* register back to DuckDB and return a relation again

DuckDB can query Polars DataFrames and LazyFrames by name; it converts to Arrow and reads lazily. 

---

### 9.3.6 Performance “deep knobs” you should wire in now (because it shrinks later work)

#### A) Connection policy: threads/memory/temp dir centralized

Set defaults in `DuckDBContext`:

* threads
* memory_limit
* temp_directory

These knobs are explicitly supported and should not be scattered as ad-hoc `SET ...` statements across code. 

#### B) Use Arrow-vectorized DuckDB UDFs where you previously used Python loops

For compute that is “vectorizable but not expressible in SQL,” create Arrow UDFs (`type='arrow'`) so DuckDB passes Arrow arrays in chunks. This massively improves throughput compared to row-by-row Python UDFs and reduces bespoke “batching” code. 

---

## 9.4 Acceptance criteria (what “done” means)

### 9.4.1 Code deletion targets

* No imports of `ibis` in build/hamilton/storage subsystems.
* No `load_dataset_df` / pandas DataFrame loader nodes in support module generation.
* No `DuckDBIbisTableSaver` references; replaced by relation saver.

### 9.4.2 Runtime invariants

* All table-producing DAG nodes return `DuckDBPyRelation`.
* All IO saver nodes accept `TabularInput` but internally coerce to relation and materialize from relation.
* Artifact writing can operate in **streaming mode** (Arrow reader → IPC writer) without ever materializing full tables in Python memory. 

### 9.4.3 Observability invariants

* “Plan/explain” and “execution” can show relation plans (SQL + explain analyze) deterministically because SQL interpolation is parameterized and centralized. 

---

## 9.5 Minimal “agent execution script” (how the agent should proceed)

1. Implement `DuckDBContext` + `coerce_to_relation` first; add unit tests for coercion from:

   * relation
   * Arrow Table
   * Polars DF
   * Polars LazyFrame
     Validate that registration/unregistration is deterministic. 

2. Modify StorageGateway protocol and concrete gateway implementations to expose DuckDB connection/context.

3. Replace support-factory loader nodes:

   * delete `df__` loaders
   * rewrite `q__` loaders to return relation
   * migrate callsites away from `ibis_adapter` usage. 

4. Replace materializers:

   * delete `DuckDBIbisTableSaver`
   * implement relation saver
   * update row saver (or delete if redundant) 

5. Migrate native modules incrementally:

   * “leaf” transforms first (smallest nodes)
   * then higher-level joins/aggregations
   * ensure all tabular nodes return relation

6. Add streaming artifact writing (Arrow IPC options) once, then route all artifact writes through it. 

---

If you want, I can extend this plan with a **mechanical conversion rubric** for native target modules (how to rewrite a pandas/Ibis node into (a) relation chaining, (b) conn.sql + params, or (c) polars escape hatch + re-register), including “before/after” micro-patterns that reduce the probability of the agent generating non-deterministic SQL or accidentally materializing gigabyte tables in Python.
