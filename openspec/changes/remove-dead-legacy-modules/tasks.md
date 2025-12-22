## 1. Decommission CLI and interface shims
- [ ] 1.1 Remove legacy CLI modules for completions, jobs runner, skip args, and MCP module
  entrypoint; update CLI docs/help if referenced.
- [ ] 1.2 Remove manifest re-export shims and update imports to codeintel.core.manifests.
- [ ] 1.3 Remove the _architecture module and update any tests/docs that reference it.

## 2. Decommission build and ingestion legacy helpers
- [ ] 2.1 Remove BuildToolAdapter and ChangeTracker and ensure ingestion runs only via
  Hamilton-native targets.
- [ ] 2.2 Remove build.hamilton.native.tools helpers and ensure tool execution uses injected
  ToolService/ToolRunner only.
- [ ] 2.3 Remove legacy result builders and ExecutionResult.fail alias; update any imports
  to the canonical result types.
- [ ] 2.4 Remove semantic_compile_hamilton and ensure semantic registry compilation uses
  build.serving.semantic_compile only.

## 3. Decommission schema and serialization legacy utilities
- [ ] 3.1 Remove row_serialization re-exports and static analytics column list modules; ensure
  schema registry ordering is the single source of truth.
- [ ] 3.2 Remove schema export/lineage/schema_docs/migration utilities and update any docs or
  tests that reference them.

## 4. Decommission storage compatibility utilities
- [ ] 4.1 Remove the ephemeral storage gateway helper and update any schema compilation
  workflows to use standard gateway configuration.
- [ ] 4.2 Remove the dataset catalog generator and align docs with versioned asset catalog
  tables.

## 5. Validation
- [ ] 5.1 Run quality report (ruff, pyright, pyrefly) and fix any issues.
- [ ] 5.2 Run pytest -q.
