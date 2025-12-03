# Runtime

The runtime module (`codeintel.runtime`) provides runtime context and identity
management for pipeline executions.

## Responsibility

- Generate and manage run identifiers
- Provide runtime context for all operations
- Coordinate cross-engine execution identity

## Architecture

```
┌─────────────────────────────────────────┐
│         Orchestrator                     │
│       (orchestrator.py)                  │
├─────────────────────────────────────────┤
│         RunContext                       │
│        (context.py)                      │
├─────────────────────────────────────────┤
│         ID Generation                    │
│          (ids.py)                        │
└─────────────────────────────────────────┘
```

## Key Entrypoints

- [`codeintel.runtime.context`][codeintel.runtime.context] - RunContext definition
- [`codeintel.runtime.orchestrator`][codeintel.runtime.orchestrator] - Context creation
- [`codeintel.runtime.ids`][codeintel.runtime.ids] - ID generation utilities

## RunContext

The central identity for a pipeline run:

```python
from codeintel.runtime import RunContext

# RunContext contains:
# - run_id: Unique identifier for this run
# - kind: Type of run (ingest, graphs, analytics, full)
# - trigger: What initiated the run (cli, api, scheduler)
# - snapshot: Repository snapshot being processed
```

## Creating Contexts

```python
from codeintel.runtime.orchestrator import new_run_context
from codeintel.config import SnapshotRef

snapshot = SnapshotRef(repo="org/repo", commit="abc123", repo_root=path)
ctx = new_run_context(snapshot=snapshot, kind="full", trigger="cli")
```

## Run Kinds

| Kind | Description |
|------|-------------|
| `ingest` | Ingestion-only run |
| `graphs` | Graph building run |
| `analytics` | Analytics computation run |
| `full` | Complete pipeline run |

## Dependencies

### Reads From

- Snapshot configuration
- Pipeline specifications

### Writes To

- Nothing directly (context is passed to other modules)

### Called By

- [`codeintel.pipeline`][codeintel.pipeline] executor
- Individual engine orchestrators

## ID Generation

Run IDs are generated using UUID7 for time-ordered identifiers:

```python
from codeintel.runtime.ids import generate_run_id

run_id = generate_run_id()  # Returns UUID7 string
```

## Usage Pattern

```python
from codeintel.runtime.orchestrator import new_run_context

# 1. Create context at pipeline start
ctx = new_run_context(snapshot=snapshot, kind="full", trigger="cli")

# 2. Pass to all engines
execute_recipe(recipe, run_context=ctx, ...)
run_graph_plugins(plan, context, run_context=ctx)
run_analytics_plugins(plan, context, run_context=ctx)

# 3. All steps are correlated by run_id
```

