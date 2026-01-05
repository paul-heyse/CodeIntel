# Hamilton Native Telemetry + Decision Trace Replacement Plan

## Goals
- Replace the storage-backed decision trace target with a Hamilton-native, post-run artifact.
- Build a best-in-class, Hamilton-native telemetry pipeline for build runs.
- Remove OpenTelemetry usage from the Hamilton build path while preserving diagnostics depth.
- Ensure diagnostics are deterministic, storage-independent, and emitted under `build/`.

## Non-Goals
- Replacing non-Hamilton observability in other subsystems (serving, runtime plugins).
- Changing dataset semantics, target selection, or DAG composition behavior.
- Introducing new external services beyond optional Hamilton UI/Tracker.

## Current State (Summary)
- Decision trace is a DAG target that reads storage-backed cache manifests, which makes it
  a control input and requires `env.gateway`.
- Cache events are already available via `HamiltonCacheAdapter.logs(...)` and `log_to_file=True`,
  but not used for decision trace output.
- Node execution telemetry exists via `NodeTelemetryHook`, but only persisted through build
  metadata bundle workflows.
- OpenTelemetry is used in `src/codeintel/build/hamilton/executor.py` for trace IDs.

## Target State (Definition)
- Decision trace is generated after Hamilton execution using cache adapter logs + node telemetry.
- Diagnostics are emitted as deterministic artifacts in `build/diagnostics/` (or equivalent).
- OpenTelemetry is not used in the Hamilton build executor.
- HamiltonTracker becomes the preferred optional UI/telemetry path for deep diagnostics.

## Design Decisions
- **Decision trace source of truth:** `HamiltonCacheAdapter.logs(...)` + cache key resolver APIs.
- **Execution timing source of truth:** `NodeTelemetryHook` records.
- **Telemetry output format:** JSON/JSONL files under `build/diagnostics/` for deterministic auditing.
- **OpenTelemetry removal:** drop trace IDs from Hamilton build and rely on run_id + tracker tags.
- **Hamilton-native UI:** use `HamiltonTracker` where configured, with strong tag semantics.

## Proposed Artifact Outputs
- `build/decision_trace.json`: ordered decision trace records (cache hit/miss/store).
- `build/diagnostics/cache_events.jsonl`: raw cache event stream (per run).
- `build/diagnostics/node_telemetry.jsonl`: node execution timings/status.
- `build/diagnostics/run_summary.json`: aggregate summary (counts, durations, run_id).
- `build/diagnostics/dag.dot`: DAG graphviz export for lineage/debugging.
- `build/diagnostics/dag.json`: DAG JSON export for programmatic inspection.
- `build/diagnostics/dag.mermaid`: DAG Mermaid export for documentation/debugging.
- `build/diagnostics/cache_keys.jsonl`: cache key + data version snapshots by node.
- `build/diagnostics/cache_run_visualization.svg`: optional cache view rendering.

## Implementation Plan

### Phase 1: Decision Trace Replacement (Post-Run, Hamilton-Native)
- [ ] Add a decision trace builder that consumes `HamiltonCacheAdapter.logs(...)`.
- [ ] Resolve cache keys and data versions with `HamiltonCacheAdapter.get_cache_key(...)`
  and `HamiltonCacheAdapter.get_data_version(...)`, with task_id support.
- [ ] Join cache events with node telemetry records when present to populate `duration_ms`.
- [ ] Emit `build/decision_trace.json` from executor completion.
- [ ] Ensure decision trace is produced even when `env.gateway` is None.
- [ ] Remove `decision_trace` target auto-inclusion in build CLI execution.
- [ ] Delete or deprecate `src/codeintel/build/hamilton/native/export/decision_trace.py`
  once post-run generation is in place.

### Phase 2: Hamilton-Native Telemetry Pipeline
- [ ] Standardize telemetry output directory under `env.paths.build_dir` (e.g. `build/diagnostics`).
- [ ] Persist cache adapter structured logs to JSONL in the telemetry directory.
- [ ] Extend `NodeTelemetryHook` (or add a companion hook) to flush node telemetry
  to `build/diagnostics/node_telemetry.jsonl` in addition to metadata bundle writes.
- [ ] Add a run-level summary artifact with:
  - run_id, repo, commit, profile, domain
  - computed/skipped/failed target counts
  - cache hit/miss summary
  - total duration
- [ ] Export DAG artifacts (`dag.dot`, `dag.json`, `dag.mermaid`) into the same
  diagnostics directory for deterministic inspection.

### Phase 3: HamiltonTracker (UI/SDK) Integration Hardening
- [ ] Keep `HamiltonTracker` as optional adapter; ensure tags include:
  - `repo`, `commit`, `run_id`, `domain`, `profile`
  - `build.cache_dir`, `build.diagnostics_dir` (optional)
- [ ] Use Hamilton tracker constants to clamp data capture (disable stats in prod by default).
- [ ] Ensure tracker tags do not depend on OpenTelemetry trace IDs.
- [ ] Document tracker capture governance (CAPTURE_DATA_STATISTICS,
  MAX_LIST_LENGTH_CAPTURE, MAX_DICT_LENGTH_CAPTURE) as first-class config.

### Phase 4: Remove OpenTelemetry From Hamilton Build
- [ ] Remove `opentelemetry` import and `_current_trace_id` usage in
  `src/codeintel/build/hamilton/executor.py`.
- [ ] Remove trace-id tagging from Hamilton tracker tags.
- [ ] Confirm build execution has no OTel dependency and remains deterministic.

### Phase 5: Update DAG/CLI Wiring
- [ ] Remove `decision_trace` target from `build run` default target expansion.
- [ ] Ensure the decision trace is emitted in the CLI path after execution, using the
  runtime cache adapter + telemetry hooks.
- [ ] Update CLI artifact resolution to read from the new path, not DAG output records.
- [ ] Emit cache key + data version snapshots to `build/diagnostics/cache_keys.jsonl`
  using `HamiltonCacheAdapter.get_cache_key(...)` + `get_data_version(...)`.
- [ ] Add an optional cache visualization export (`dr.cache.view_run`) to
  `build/diagnostics/cache_run_visualization.svg`, gated to non-tasked runs.

### Phase 6: Validation & Quality Gates
- [ ] Run `uv run codeintel build run --all --verbose=1` and confirm:
  - no storage gateway required for decision trace
  - decision trace artifacts are written under `build/`
- [ ] Run `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`.
- [ ] Validate output artifacts are deterministic across two runs with no code changes.

## Hamilton-Native Reference Snippets (for Implementation)

These snippets map directly to Hamilton documented capabilities. Use them as
the preferred building blocks instead of non-Hamilton workarounds.

### Tracker Adapter (UI/Telemetry)
```python
from hamilton import driver
from hamilton_sdk import adapters

tracker = adapters.HamiltonTracker(
    project_id=PROJECT_ID,
    username="you@company.com",
    dag_name=f"codeintel::build::{repo}::{commit}",
    tags={
        "repo": repo,
        "commit": commit,
        "run_id": run_id,
        "domain": domain or "build",
        "profile": profile or "default",
        "build.diagnostics_dir": str(diagnostics_dir),
        "build.cache_dir": str(cache_dir),
    },
    hamilton_api_url=api_url,
    hamilton_ui_url=ui_url,
)

dr = (
    driver.Builder()
    .with_modules(*modules)
    .with_config(config)
    .with_adapters(tracker)
    .build()
)
```

### Cache Adapter with Structured Logs (JSONL)
```python
from hamilton import driver

dr = (
    driver.Builder()
    .with_modules(*modules)
    .with_config(config)
    .with_cache(
        log_to_file=True,
        default_behavior="default",
        default_loader_behavior="disable",
        default_saver_behavior="disable",
    )
    .build()
)
```

### Cache Event Inspection (In-Memory Logs)
```python
cache_adapter = dr.cache
run_id = cache_adapter.last_run_id
logs_by_node = cache_adapter.logs(run_id=run_id, level="info")
```

### Cache Key / Data Version Introspection
```python
cache_key = cache_adapter.get_cache_key(run_id=run_id, node_name=node_name)
data_version = cache_adapter.get_data_version(run_id=run_id, node_name=node_name)
```

### Cache Run Visualization (Optional)
```python
cache_adapter.view_run(
    run_id=run_id,
    output_file_path=str(diagnostics_dir / "cache_run_visualization.svg"),
)
```

### DAG Exports (Dot/JSON/Mermaid)
```python
export_dag_dot(dr, output_path=diagnostics_dir / "dag.dot")
export_dag_json(dr, output_path=diagnostics_dir / "dag.json")
export_dag_mermaid(dr, output_path=diagnostics_dir / "dag.mermaid")
```

### Lifecycle Hook for Node Telemetry
```python
from hamilton.lifecycle import base as lifecycle_base

class NodeTelemetryHook(
    lifecycle_base.BasePreNodeExecute,
    lifecycle_base.BasePostNodeExecute,
):
    ...
```

### Decision Trace from Cache Logs (Post-Run)
```python
logs_by_node = cache_adapter.logs(run_id=run_id, level="info")
entries = build_cache_manifest_entries(logs_by_node, cache_adapter)
write_decision_trace(diagnostics_dir / "decision_trace.json", entries)
```

## Risks & Mitigations
- **Cache logs missing on failure:** write partial traces early; include failure status in summary.
- **Parallelizable tasks:** include `task_id` in cache and telemetry joins to avoid collisions.
- **Telemetry volume:** keep JSONL files per-run, and rotate by `run_id` if needed.

## Acceptance Criteria
- Full build completes without `env.gateway` for decision trace and export targets.
- Decision trace is produced as a post-run artifact with cache status + timing.
- Node telemetry and cache event logs are available under `build/diagnostics/`.
- OpenTelemetry is no longer referenced by Hamilton build code.
- Optional HamiltonTracker integration works with stable tag metadata.
