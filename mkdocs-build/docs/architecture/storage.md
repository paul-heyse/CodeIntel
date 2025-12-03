# Storage

The storage module (`codeintel.storage`) provides the persistence layer using
DuckDB and Parquet with typed dataset contracts.

## Responsibility

- Manage DuckDB database connections
- Define and enforce dataset schemas
- Provide typed accessors for all tables and views
- Handle schema migrations and validation
- Track pipeline runs and steps

## Architecture

```
┌─────────────────────────────────────────┐
│         StorageGateway                   │
│          (gateway.py)                    │
├─────────────────────────────────────────┤
│      Namespace Accessors                 │
│   (CoreTables, GraphTables, etc.)        │
├─────────────────────────────────────────┤
│       Dataset Registry                   │
│     (registry_helpers.py)                │
├─────────────────────────────────────────┤
│        Schema Layer                      │
│   (schemas.py, views/*.py)               │
├─────────────────────────────────────────┤
│         DuckDB                           │
└─────────────────────────────────────────┘
```

## Key Entrypoints

- [`codeintel.storage.gateway`][codeintel.storage.gateway] - Main gateway and connection management
- [`codeintel.storage.schemas`][codeintel.storage.schemas] - Schema definitions
- [`codeintel.storage.run_tracking`][codeintel.storage.run_tracking] - Pipeline run tracking

## StorageGateway

The gateway is the primary interface for all storage operations:

```python
from codeintel.storage.gateway import open_gateway, StorageConfig

config = StorageConfig(db_path=Path("build/db/codeintel.duckdb"))
gateway = open_gateway(config)

# Typed namespace access
functions = gateway.analytics.functions()
call_graph = gateway.graph.call_graph()
modules = gateway.core.modules()
```

## Namespaces

| Namespace | Tables |
|-----------|--------|
| `core` | goids, modules, files |
| `graph` | call_graph, import_graph, cfg, dfg |
| `analytics` | functions, metrics, profiles, risk |
| `docs` | Views for documentation export |

## Dependencies

### Reads From

- DuckDB database file
- Parquet files for large datasets

### Writes To

- DuckDB tables
- Run tracking metadata

### Called By

- All other modules use storage for persistence

## Extension Points

### Adding a New Table

1. Define schema in `codeintel.config.datasets`
2. Add accessor method to appropriate namespace class
3. Create migration if needed
4. Register in dataset registry

### Custom Views

```python
# In storage/views/my_views.py
def create_my_view(con: DuckDBConnection) -> None:
    con.execute("""
        CREATE OR REPLACE VIEW docs.my_view AS
        SELECT ...
    """)
```

