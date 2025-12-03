# Ingestion

The ingestion module (`codeintel.ingestion`) extracts code metadata from
repositories and persists it to the storage layer.

## Responsibility

- Parse source files (AST, CST via tree-sitter and libcst)
- Extract SCIP index for semantic analysis
- Process coverage data
- Scan configuration files
- Build module and function indices

## Architecture

```
┌─────────────────────────────────────────┐
│           Recipe Layer                   │
│    (recipes/dsl.py, recipes/executor.py) │
├─────────────────────────────────────────┤
│           Plugin Layer                   │
│         (plugins/*.py)                   │
├─────────────────────────────────────────┤
│         Resource Providers               │
│         (resources/*.py)                 │
├─────────────────────────────────────────┤
│           Steps Layer                    │
│          (steps/*.py)                    │
├─────────────────────────────────────────┤
│         Port-Adapter Layer               │
│    (ports/*.py, adapters/*.py)           │
└─────────────────────────────────────────┘
```

## Key Entrypoints

- [`codeintel.ingestion.recipes.executor`][codeintel.ingestion.recipes.executor] - Recipe execution
- [`codeintel.ingestion.recipes.builtin`][codeintel.ingestion.recipes] - Built-in recipes
- [`codeintel.ingestion.ingest_service`][codeintel.ingestion.ingest_service] - Service facade

## Dependencies

### Reads From

- Repository source files
- VCS metadata (git)
- Coverage reports (JSON, XML)
- Configuration files (pyproject.toml, etc.)

### Writes To

- `codeintel.storage` tables (via gateway)
- Parquet files for large datasets

### Called By

- [`codeintel.pipeline`][codeintel.pipeline] orchestration
- CLI commands
- Tests

## Extension Points

### Adding a New Ingest Plugin

1. Create plugin in `ingestion/plugins/`
2. Implement `IngestPluginProtocol`
3. Register in plugin registry
4. Add to appropriate recipe

### Custom Recipes

```python
from codeintel.ingestion.recipes.dsl import IngestRecipe, IngestStage

CUSTOM_RECIPE = IngestRecipe(
    name="custom",
    stages=(
        IngestStage(plugin="repo_scan", required=True),
        IngestStage(plugin="my_custom_plugin", required=False),
    ),
)
```

## See Also

- [Detailed Architecture](../../docs/ANALYTICS_ARCHITECTURE.md#part-ii-ingestion-module)

