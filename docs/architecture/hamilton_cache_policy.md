# Hamilton Cache Policy (Build/Planning)

## Goals

- Avoid caching environment-bound or non-serializable objects.
- Keep plan/diagnostic outputs audit-only and non-controlling.
- Preserve deterministic caching for Arrow/Parquet datasets.

## Non-Cacheable Inputs (Always Ignore)

These inputs are external runtime objects and must not be cached:

- `env`
- `catalog`
- `tag_query`
- `cache_index`
- `cache_key_resolver`
- `schema_index`
- `semantic_registry`
- `runtime_fingerprint`
- `plan_request`

Policy: any Hamilton node with one of the names above is forced to
`@cache(behavior="ignore")` via the cache adapter override.

## Planning DAG Nodes (Ignore Cache)

The planning DAG is fast, uses runtime-bound objects, and should not be cached:

- `plan_context`
- `plan_target_closure`
- `plan_target_subgraph_nodes`
- `plan_node_versions`
- `plan_cache_probe`
- `plan_graph_inputs`
- `plan`
- `preflight_issues`
- `preflight_block_map`

Policy: explicitly decorate the above with `@cache(behavior="ignore")`.

## Dataclasses With init=False Fields

These classes contain non-init fields that should never be normalized into cache payloads:

- `BuildMetadataBundleWriter` (`src/codeintel/build/meta/bundle.py`)
  - `_lock`, `_jsonl_writers`, `_files`
- `UnifiedProviderSchema` (`src/codeintel/build/schemas/provider_unified.py`)
  - `schema_authority`
- `EvidenceCollector` (`src/codeintel/build/analytics/compute/evidence/collection.py`)
  - `cache`
- `BundleSchemaObservationProvider` (`src/codeintel/build/schemas/bundle_observations.py`)
  - `_cache`, `_loaded`
- `ConditionalHook` (`src/codeintel/build/hamilton/hooks/lifecycle.py`)
  - `_enabled`

Policy: cache normalization must skip `init=False` fields and recurse only through init fields.

## Telemetry and Decision Trace

Diagnostics and decision trace outputs are audit artifacts; they must never
influence control flow or caching decisions.

Policy:

- Emit diagnostics from runtime helpers (not DAG inputs).
- If future telemetry nodes are added to the DAG, mark them with
  `@cache(behavior="ignore")`.

## Validation

Targeted validation to confirm cache stability for the planning DAG:

```bash
./scripts/validate_cache_adapter.sh
```

Or run directly:

```bash
uv run codeintel build run --targets=ci_plan --verbose=1
```

## Cacheable Node Classes (Opt-In)

Caching is opt-in. The cache adapter default behavior is `disable`, and the
following node families are explicitly cache-enabled:

- Dataset output nodes created via `save_dataset`/`save_relation_table`:
  cached with `@cache(behavior="default", format="parquet")`.
- Analytics result aggregations (e.g., `graph_metrics_result`,
  `entrypoints_result`, `data_models_result`) are cached with
  `@cache(behavior="default")` to avoid recomputation of heavy analysis.

## Non-Cacheable Node Families

These node families should remain uncached:

- Planning/preflight nodes (plan/ci_plan pipeline)
- Runtime/environment injectors (env, catalog, cache index/resolver, schema registry)
- Telemetry/diagnostics emitters
