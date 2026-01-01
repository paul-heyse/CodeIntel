# Pipeline

The pipeline module (`codeintel.pipeline`) provides unified orchestration for
running ingestion, graphs, and analytics as a coordinated pipeline.

## Responsibility

- Define declarative pipeline specifications
- Orchestrate multi-stage execution
- Track run progress and status
- Handle failure modes and recovery

## Architecture

```
┌─────────────────────────────────────────┐
│           CLI Layer                      │
│          (cli/*.py)                      │
├─────────────────────────────────────────┤
│         Executor Layer                   │
│        (executor.py)                     │
├─────────────────────────────────────────┤
│         Planner Layer                    │
│        (planner.py)                      │
├─────────────────────────────────────────┤
│          Spec Layer                      │
│         (spec.py)                        │
├─────────────────────────────────────────┤
│      Engine Delegation                   │
│ (ingestion, graphs, analytics engines)   │
└─────────────────────────────────────────┘
```

## Key Entrypoints

- [`codeintel.pipeline.spec`][codeintel.pipeline.spec] - Pipeline specifications
- [`codeintel.pipeline.planner`][codeintel.pipeline.planner] - Execution planning
- [`codeintel.pipeline.executor`][codeintel.pipeline.executor] - Pipeline execution
- [`codeintel.pipeline.cli`][codeintel.pipeline.cli] - CLI interface

## Pipeline Specs

Declarative specifications for what to run:

```python
from codeintel.pipeline.spec import PipelineSpec, PipelineStage

FULL_PIPELINE = PipelineSpec(
    id="full",
    description="Ingest + graphs + analytics",
    stages=(
        PipelineStage(module="ingestion", name="builtin.default"),
        PipelineStage(module="graphs", name="builtin.full"),
        PipelineStage(module="analytics", name="builtin.full"),
    ),
)
```

### Built-in Specs

| Spec | Stages |
|------|--------|
| `FULL_PIPELINE` | Ingestion → Graphs → Analytics |
| `INGEST_ONLY` | Ingestion only |
| `GRAPHS_ONLY` | Graphs only |
| `ANALYTICS_ONLY` | Analytics only |

## Run Tracking

All pipeline runs are tracked in the database:

```python
# Runs are tracked via gateway.runs
runs = gateway.runs
run = runs.fetch_run(run_id)
steps = runs.fetch_steps(run_id)
```

## Dependencies

### Reads From

- Pipeline specifications
- Engine configurations

### Writes To

- Run tracking tables
- Delegates to engine outputs

### Calls

- [`codeintel.ingestion`][codeintel.ingestion] recipes
- [`codeintel.build.graphs`][codeintel.build.graphs] runtime
- [`codeintel.build.analytics`][codeintel.build.analytics] pipeline bridge

## Extension Points

### Custom Pipeline Specs

```python
CUSTOM_SPEC = PipelineSpec(
    id="custom",
    stages=(
        PipelineStage(module="ingestion", name="builtin.incremental"),
        PipelineStage(module="analytics", name="builtin.full", required=False),
    ),
)
```

## CLI Usage

```bash
# Run full pipeline
codeintel pipeline run --mode full

# Run specific stages
codeintel pipeline run --mode ingest
codeintel pipeline run --mode graphs
codeintel pipeline run --mode analytics
```

