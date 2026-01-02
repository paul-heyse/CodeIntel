# Hamilton Build Best-in-Class Implementation Plan

This plan integrates the agreed direction:
- Remove global module overrides by default.
- Treat Hamilton caching as a correctness feature plus audit/diagnostics.
- Treat analytics tables as part of the semantic surface.

It covers all identified improvements, including the lower criticality items, and
includes a representative go-forward code pattern for each change.

## Scope and sequencing

1. Composition hardening: module overrides, graph validation, cache correctness
2. Tagging and semantic surface consolidation
3. Boilerplate reduction via spec-driven target generation
4. Observability and dynamic execution hooks
5. Low-criticality cleanup and lineage improvements

Each phase should be incremental and keep the DAG functional at every step.

## 1) Remove global module overrides as the default

### Why
Global `allow_module_overrides()` masks duplicate node issues and makes module
ordering brittle. We will remove it by default and only enable overrides in
explicit, scoped scenarios.

### Implementation steps
- Remove global `allow_module_overrides()` calls in the standard builder paths.
- Add a config-driven override flag, default false.
- Add a duplicate-node validation step when overrides are disabled, so failures
  are explicit and actionable.
- Introduce a targeted override path for known use cases (for example, a
  specific plugin or test-only module set).

### Representative code pattern

```python
# codeintel/runtime/compose.py

def _build_driver_with_adapters(..., config: Mapping[str, Any], ...):
    builder = h_driver.Builder().with_config(dict(config)).with_modules(*modules)
    if config.get("ci.allow_module_overrides", False):
        builder = builder.allow_module_overrides()
    ...
    return builder.with_adapters(*adapters).build()
```

```python
# codeintel/runtime/compose.py

def _validate_no_duplicates(modules: Sequence[ModuleType]) -> None:
    # New helper: build FunctionGraph without overrides and fail on duplicates.
    h_driver.Builder().with_modules(*modules).build()
```

## 2) Add graph validation as a build gate

### Why
Graph validation exists but is not invoked, so missing tags and structural
issues are not caught early.

### Implementation steps
- Run `validate_graph` after the runtime bundle is built.
- Introduce a config toggle with three modes: "strict" (error), "warn" (log),
  "off" (skip).
- Include JSON-formatted diagnostics in error messages for fast triage.

### Representative code pattern

```python
# codeintel/runtime/compose.py

def compose_runtime(...):
    ...
    runtime_bundle = _build_runtime_bundle(...)
    mode = str(identity.config.get("ci.graph_validation", "strict"))
    if mode != "off":
        result = validate_graph(runtime=runtime_bundle, validate_schema=True)
        if result.has_errors:
            payload = validation_result_to_json(result, node_provenance=runtime_bundle.module_provenance)
            if mode == "strict":
                raise RuntimeError(payload)
            log.warning("graph.validation.warn %s", payload)
```

## 3) Make caching a correctness and audit feature

### Why
Caching is currently broad and implicit. For correctness, caching should be
explicit where deterministic, and opt-out where non-deterministic. For audit,
cache logs should be consistently written and ingested.

### Implementation steps
- Add a cache policy config block that declares default behavior and per-node
  overrides (default, recompute, disable, ignore).
- Use `@cache` explicitly for deterministic, expensive nodes.
- Use `@cache(behavior="recompute")` for non-deterministic or external I/O nodes.
- Use `@cache(behavior="ignore")` for env or credential nodes so they do not
  influence cache keys.
- Prefer structured cache logs by default (JSONL) and ingest them into
  observability.
- Use format-aware caching for large tables (parquet) and small dicts (json).

### Representative code pattern

```python
# codeintel/runtime/compose.py

def _cache_options_from_profile(...):
    return CacheAdapterOptions(
        cache_store=cache_store,
        default_behavior="disable",  # correctness-by-explicitness
        default_loader_behavior="disable",
        default_saver_behavior="disable",
        log_to_file=True,
    )
```

```python
# codeintel/build/hamilton/native/analytics/tables_risk.py
from hamilton.function_modifiers import cache

@cache(format="parquet")
def risk_factors__base(...):
    ...
```

```python
# non-deterministic example
from hamilton.function_modifiers import cache

@cache(behavior="recompute")
def wall_clock_timestamp() -> datetime:
    return datetime.now(tz=UTC)
```

## 4) Remove redundant dataset tagging

### Why
`@tag_dataset` is currently applied multiple times through `save_dataset`,
`table_contract`, and explicit decorators. This causes redundant tagging and
increases the chance of conflicts.

### Implementation steps
- Remove the tagging call inside `table_contract` so it only expresses
  cleaning, feature, and canonicalization policy.
- Standardize: if a table is materialized, `@save_dataset` is the canonical
  place to apply dataset tags. If it is not materialized, use `@tag_dataset`.
- Update call sites to remove redundant `@tag_dataset` when `@save_dataset` is
  present.

### Representative code pattern

```python
# codeintel/build/hamilton/transforms/table_contract.py

def table_contract(spec: TableContractSpec):
    def _decorator(fn):
        fn = pipe_clean_df(...)(fn)
        if spec.ops_module is not None:
            fn = with_features(...)(fn)
        return pipe_canonical_output(...)(fn)
    return _decorator
```

```python
# usage
@save_dataset(context=CTX, spec=DatasetSaveSpec(table_key=TABLE_KEY))
@table_contract(CONTRACT)
def table_node(base: pl.LazyFrame) -> pl.LazyFrame:
    return base
```

## 5) Introduce a spec-driven table target generator

### Why
Many analytics targets repeat the base -> table -> materializations -> target
pattern. A spec-driven generator reduces boilerplate and makes target creation
consistent.

### Implementation steps
- Add a `TableTargetSpec` and `attach_table_target_template` helper.
- Use `tagged_attach_node` to enforce consistent tags and names.
- Use a materialization collector builder for saver nodes.
- Migrate a small subset of analytics targets first, then scale.

### Representative code pattern

```python
# codeintel/build/hamilton/native/patterns/table_target.py
@dataclass(frozen=True, slots=True)
class TableTargetSpec:
    domain: str
    target: str
    table_key: str
    base_fn: Callable[..., pl.LazyFrame]
    contract: TableContractSpec
    save_spec: DatasetSaveSpec
    semantic_tags: Mapping[TagKey, TagValue]


def attach_table_target_template(module: ModuleType, spec: TableTargetSpec) -> None:
    def table_node(base: pl.LazyFrame) -> pl.LazyFrame:
        return base

    tagged_attach_node(
        module,
        node_name=f"{spec.table_key.replace('.', '__')}__table",
        fn=save_dataset(context=SaverContext(...), spec=spec.save_spec)(
            table_contract(spec.contract)(table_node)
        ),
        tag_spec=TagSpec.for_dataset(domain=spec.domain, target=spec.target, table_key=spec.table_key),
        extra_tags=spec.semantic_tags,
    )
```

## 6) Add task-level hooks for dynamic execution

### Why
Dynamic execution is already supported, but task-level observability is missing.
Adding Task hooks improves diagnosis and scheduling insights.

### Implementation steps
- Create a `TaskTelemetryHook` implementing TaskSubmission/Execution/Return
  hooks when dynamic execution is enabled.
- Wire it into `build_hooks` based on config or when dynamic execution is on.
- Record task-level timings and failure context to run records.

### Representative code pattern

```python
# codeintel/build/hamilton/hooks/task_lifecycle.py
class TaskTelemetryHook(TaskSubmissionHook, TaskExecutionHook, TaskReturnHook):
    def pre_task_submission(self, *, run_id: str, task_id: str, nodes: list[str], **_) -> None:
        ...
```

```python
# codeintel/build/hamilton/hooks/__init__.py
if options.enable_task_telemetry:
    hooks.append(TaskTelemetryHook(run_id, writer))
```

## 7) Make analytics tables semantic by default

### Why
Analytics tables are part of the semantic surface. They must be tagged with
semantic metadata so they show up in registries and validations.

### Implementation steps
- Introduce a helper to produce semantic tags from table metadata and schema.
- Extend `DatasetSaveSpec` or `TableContractSpec` to accept semantic tags.
- Update analytics tables to pass semantic tags via `extra_tags`.
- Ensure graph validation enforces semantic tag presence and consistency.

### Representative code pattern

```python
# codeintel/build/hamilton/semantic_tags.py

def semantic_table_tags(*, semantic_id: str, entity: str, grain: str, table_key: str) -> dict[TagKey, TagValue]:
    return {
        "layer": "semantic",
        "semantic_id": semantic_id,
        "kind": "table",
        "version": "1",
        "entity": entity,
        "grain": grain,
        "schema_ref": table_key,
        "entity_keys": "repo,commit,function_goid_h128",
        "join_keys": "repo,commit,function_goid_h128",
    }
```

```python
# usage
SEMANTIC_TAGS = semantic_table_tags(
    semantic_id="analytics.risk_factors.v1",
    entity="function",
    grain="per_function",
    table_key=RISK_FACTORS_TABLE_KEY,
)

@save_dataset(
    context=CTX,
    spec=DatasetSaveSpec(table_key=RISK_FACTORS_TABLE_KEY),
)
@table_contract(CONTRACT)
def risk_factors__table(risk_factors__base: pl.LazyFrame) -> pl.LazyFrame:
    return risk_factors__base
```

## 8) Improve tracker dag_name versioning

### Why
The UI uses dag_name as a version key. A more explicit name improves
comparisons and provenance.

### Implementation steps
- Add a template setting (for example, `hamilton_tracker.dag_name_template`).
- Allow placeholders for repo, commit, modules fingerprint, semantic version.
- Default to a stable template when none is provided.

### Representative code pattern

```python
# codeintel/runtime/compose.py

template = tracker_settings.dag_name_template or "codeintel::{repo}::{commit}::{modules}"
kwargs["dag_name"] = template.format(
    repo=env.snapshot.repo,
    commit=env.snapshot.commit,
    modules=modules_fingerprint[:8],
)
```

## 9) Replace pass-through nodes with @does identity

### Why
Many table nodes simply return their input after decorators. `@does` can
remove boilerplate and signal intent.

### Implementation steps
- Add a shared identity helper that accepts kwargs.
- Use `@does(identity)` for nodes that do not otherwise transform data.

### Representative code pattern

```python
# codeintel/build/hamilton/transforms/identity.py

def identity(**kwargs: object) -> object:
    return next(iter(kwargs.values()))
```

```python
from hamilton.function_modifiers import does

@does(identity)
def risk_factors__table(risk_factors__base: pl.LazyFrame) -> pl.LazyFrame:
    ...
```

## 10) Add extract_columns + tag_outputs for column lineage

### Why
Column-level lineage improves observability and allows fine-grained reuse.

### Implementation steps
- Identify wide tables with key semantic columns used by downstream nodes.
- Add `@extract_columns` to split columns into separate nodes.
- Add `@tag_outputs` to apply semantic tags per extracted column.
- Use these outputs in downstream calculations or exports.

### Representative code pattern

```python
from hamilton.function_modifiers import extract_columns, tag_outputs

@tag_outputs(
    risk_score={"semantic_id": "analytics.risk_score.v1", "dtype": "int"},
    risk_level={"semantic_id": "analytics.risk_level.v1", "dtype": "str"},
)
@extract_columns("risk_score", "risk_level")
def risk_factor_columns(risk_factors__table: pl.LazyFrame) -> pl.LazyFrame:
    return risk_factors__table
```

## Acceptance criteria

- Duplicate node names fail fast unless explicit overrides are enabled.
- Graph validation runs during composition and blocks errors in strict mode.
- Cache behavior is explicit and auditable, with log ingestion enabled.
- Analytics tables are tagged as semantic outputs and appear in registries.
- Boilerplate table target definitions can be generated from a spec helper.
- Dynamic execution includes task-level telemetry hooks.
- Column-level lineage is available for key tables.

