# Layering and Boundaries

This page defines the explicit layering rules for CodeIntel modules. All code
changes must respect these boundaries to prevent circular dependencies and
maintain architectural integrity.

## Layer Hierarchy

```
┌─────────────────────────────────────────────┐
│            Pipeline / Orchestration          │
│         (codeintel.pipeline)                 │
├─────────────────────────────────────────────┤
│               Serving Layer                  │
│         (codeintel.serving)                  │
├─────────────────────────────────────────────┤
│              Engine Layer                    │
│  (analytics, graphs, ingestion, storage)     │
├─────────────────────────────────────────────┤
│          Core / Config Layer                 │
│    (codeintel.core, codeintel.config)        │
└─────────────────────────────────────────────┘
```

## Import Rules

### Core / Config Layer

- **Modules**: `codeintel.core`, `codeintel.config`
- **Can import**: Standard library, third-party packages only
- **Must NOT import**: Any other `codeintel.*` modules

### Engine Layer

- **Modules**: `codeintel.build.analytics`, `codeintel.build.graphs`, `codeintel.ingestion`, `codeintel.storage`
- **Can import**: `codeintel.core`, `codeintel.config`, and each other where needed
- **Must NOT import**: `codeintel.serving`, `codeintel.pipeline`

### Serving Layer

- **Modules**: `codeintel.serving`
- **Can import**: All engine modules, core, config
- **Must NOT import**: `codeintel.pipeline`

### Pipeline / Orchestration Layer

- **Modules**: `codeintel.pipeline`
- **Can import**: All other modules (orchestrates everything)
- **Must NOT be imported by**: Any other layer

## Enforcement

Layering rules are enforced through:

1. **Architecture tests** in `tests/architecture/`
2. **Import linting** via Ruff and custom checks
3. **Code review** guidelines

## Common Patterns

### Cross-Engine Communication

Engines communicate through `codeintel.storage`:

```python
# analytics needs graph data
from codeintel.storage.gateway import StorageGateway

def compute(gateway: StorageGateway) -> None:
    call_graph = gateway.graph.call_graph()
    # process...
```

### Configuration Injection

All configuration flows from `codeintel.config`:

```python
from codeintel.config import ConfigBuilder, SnapshotRef

builder = ConfigBuilder.from_snapshot(snapshot)
cfg = builder.graph_metrics()
```

## Violations

If you encounter an import that violates these rules, consider:

1. **Moving shared code** to `codeintel.core` or `codeintel.config`
2. **Using protocols** instead of concrete types
3. **Passing data** through `StorageGateway` instead of direct imports

