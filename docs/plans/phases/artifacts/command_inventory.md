# Command Inventory

> **Generated:** December 10, 2024  
> **Purpose:** Track command migration status for Phase 5

## Summary

| Metric | Count |
|--------|-------|
| Total command files | 18 |
| Infrastructure files | 4 |
| Command definition files | 14 |
| Commands with `__call__` | ~40+ |
| Commands using `command_context` | Most |

## Infrastructure Files (Not Commands)

| File | Purpose | Migration Impact |
|------|---------|------------------|
| `commands/__init__.py` | Package init | None |
| `commands/_common.py` | Shared utilities, `make_root_app()` | Simplify after decorator |
| `commands/_help.py` | Help text utilities | None |
| `commands/app.py` | Root Cyclopts app | None |
| `commands/context.py` | `command_context` manager | **Delete in Phase 6** |

## Command Definition Files

### commands/jobs.py

| Command Class | Has `__call__` | Uses `command_context` | Handler | Runtime Req |
|---------------|----------------|------------------------|---------|-------------|
| `JobsListCommand` | Yes | Yes | `jobs_list_handler` | No |
| `JobsStatusCommand` | Yes | Yes | `jobs_status_handler` | No |
| `JobsOutputCommand` | Yes | Yes | `jobs_output_handler` | No |
| `JobsCancelCommand` | Yes | Yes | `jobs_cancel_handler` | No |
| `JobsCleanupCommand` | Yes | Yes | `jobs_cleanup_handler` | No |

---

### commands/health.py

| Command Class | Has `__call__` | Uses `command_context` | Handler | Runtime Req |
|---------------|----------------|------------------------|---------|-------------|
| `HealthCheckCommand` | Yes | Yes | `health_check_handler` | No |

---

### commands/ops.py

| Command Class | Has `__call__` | Uses `command_context` | Handler | Runtime Req |
|---------------|----------------|------------------------|---------|-------------|
| `OpListCommand` | Yes | Yes | `op_list_handler` | No |
| `OpCallCommand` | Yes | Yes | `op_call_handler` | Yes |

---

### commands/serve.py

| Command Class | Has `__call__` | Uses `command_context` | Handler | Runtime Req |
|---------------|----------------|------------------------|---------|-------------|
| `ServeHttpCommand` | Yes | Yes | `serve_http_handler` | Yes |
| `ServeMcpCommand` | Yes | Yes | `serve_mcp_handler` | Yes |

---

### commands/build.py

| Command Class | Has `__call__` | Uses `command_context` | Handler | Runtime Req |
|---------------|----------------|------------------------|---------|-------------|
| `BuildStatusCommand` | Yes | Yes | `build_status_handler` | Yes |
| `BuildRunCommand` | Yes | Yes | `build_run_handler` | Yes |
| `BuildHistoryCommand` | Yes | Yes | `build_history_handler` | Yes |

---

### commands/docs.py

| Command Class | Has `__call__` | Uses `command_context` | Handler | Runtime Req |
|---------------|----------------|------------------------|---------|-------------|
| `DocsExportCommand` | Yes | Yes | `docs_export_handler` | Yes |
| `DocsValidateCommand` | Yes | Yes | `docs_validate_handler` | Yes |

---

### commands/storage.py

| Command Class | Has `__call__` | Uses `command_context` | Handler | Runtime Req |
|---------------|----------------|------------------------|---------|-------------|
| `ValidateMacrosCommand` | Yes | Yes | `validate_macros_handler` | Yes |
| `GenerateMacrosCommand` | Yes | Yes | `generate_macros_handler` | No |
| `ProfileStorageCommand` | Yes | Yes | `profile_storage_handler` | Yes |

---

### commands/graphs.py

| Command Class | Has `__call__` | Uses `command_context` | Handler | Runtime Req |
|---------------|----------------|------------------------|---------|-------------|
| `GraphPluginsListCommand` | Yes | Yes | `graph_plugins_list_handler` | No |
| `GraphPluginsPlanCommand` | Yes | Yes | `graph_plugins_plan_handler` | No |

---

### commands/history.py

| Command Class | Has `__call__` | Uses `command_context` | Handler | Runtime Req |
|---------------|----------------|------------------------|---------|-------------|
| `HistoryTimeseriesCommand` | Yes | Yes | `history_timeseries_handler` | Yes |

---

### commands/datasets.py

| Command Class | Has `__call__` | Uses `command_context` | Handler | Runtime Req |
|---------------|----------------|------------------------|---------|-------------|
| `DatasetsListCommand` | Yes | Yes | `datasets_list_handler` | Yes |
| `DatasetsLintCommand` | Yes | Yes | `datasets_lint_handler` | Yes |
| `DatasetsSnapshotCommand` | Yes | Yes | `datasets_snapshot_handler` | No |
| `DatasetsDiffCommand` | Yes | Yes | `datasets_diff_handler` | No |

---

### commands/dataset_ops.py

| Command Class | Has `__call__` | Uses `command_context` | Handler | Runtime Req |
|---------------|----------------|------------------------|---------|-------------|
| `DatasetListCommand` | Yes | Yes | `dataset_list_handler` | Yes |
| `DatasetDescribeCommand` | Yes | Yes | `dataset_describe_handler` | Yes |
| `DatasetVerifyCommand` | Yes | Yes | `dataset_verify_handler` | Yes |

---

### commands/ide.py

| Command Class | Has `__call__` | Uses `command_context` | Handler | Runtime Req |
|---------------|----------------|------------------------|---------|-------------|
| `IdeHintsCommand` | Yes | Yes | `ide_hints_handler` | Yes |

---

### commands/plugins.py

| Command Class | Has `__call__` | Uses `command_context` | Handler | Runtime Req |
|---------------|----------------|------------------------|---------|-------------|
| `PluginsListCommand` | Yes | Yes | `plugins_list_handler` | No |
| `PluginsDiscoverCommand` | Yes | Yes | `plugins_discover_handler` | No |
| `PluginsInfoCommand` | Yes | Yes | `plugins_info_handler` | No |
| `PluginsPathsCommand` | Yes | Yes | `plugins_paths_handler` | No |
| `PluginsNewCommand` | Yes | Yes | `plugins_new_handler` | No |
| `PluginsTestCommand` | Yes | Yes | `plugins_test_handler` | No |
| `PluginsValidateCommand` | Yes | Yes | `plugins_validate_handler` | No |

---

### commands/subsystem.py

| Command Class | Has `__call__` | Uses `command_context` | Handler | Runtime Req |
|---------------|----------------|------------------------|---------|-------------|
| `SubsystemListCommand` | Yes | Yes | `subsystem_list_handler` | Yes |
| `SubsystemShowCommand` | Yes | Yes | `subsystem_show_handler` | Yes |
| `SubsystemProfilesCommand` | Yes | Yes | `subsystem_profiles_handler` | Yes |
| `SubsystemCoverageCommand` | Yes | Yes | `subsystem_coverage_handler` | Yes |
| `SubsystemMembershipCommand` | Yes | Yes | `subsystem_module_memberships_handler` | Yes |

---

### commands/config.py

| Command Class | Has `__call__` | Uses `command_context` | Handler | Runtime Req |
|---------------|----------------|------------------------|---------|-------------|
| `ConfigShowCommand` | Yes | No (direct impl) | N/A | No |
| `ConfigPathCommand` | Yes | No (direct impl) | N/A | No |

---

### commands/completions.py

| Command Class | Has `__call__` | Uses `command_context` | Handler | Runtime Req |
|---------------|----------------|------------------------|---------|-------------|
| Shell completion commands | Yes | No (direct impl) | N/A | No |

---

### commands/help_commands.py

| Command Class | Has `__call__` | Uses `command_context` | Handler | Runtime Req |
|---------------|----------------|------------------------|---------|-------------|
| Help-related commands | Yes | No (direct impl) | N/A | No |

---

## Common `__call__` Pattern

Most command classes follow this pattern:

```python
def __call__(self) -> None:
    runtime_cli = RuntimeCLI()
    output_cli = OutputFormatCLI(output_format=self.output_format)
    params: dict[str, object] = {
        "field1": self.field1,
        "field2": self.field2,
        # ... extract all dataclass fields
    }
    
    with command_context(
        "operation.id",
        runtime_cli,
        output_cli,
        params=params,
        require_runtime=True,  # varies
    ) as (ctx, renderer):
        result = handler(ctx)
        exit_code = renderer.render_result(result)
        if exit_code != 0:
            sys.exit(exit_code)
```

**Migration goal:** Replace with `@cli_command` decorator.

---

## Standard Fields

Commands typically include these fields:

| Field | Type | Purpose | Prevalence |
|-------|------|---------|------------|
| `output_format` | `OutputFormat` | Output format selection | Universal |
| `verbose` | `int` | Verbosity level (count flag) | Most |
| `project_root` | `Path \| None` | Project root override | Many |
| `repo` | `str \| None` | Repository slug | Some |
| `commit` | `str \| None` | Commit SHA | Some |
| `db_path` | `Path \| None` | Database path override | Some |

---

## Migration Priority

| Command File | Commands | Complexity | Priority |
|--------------|----------|------------|----------|
| `jobs.py` | 5 | Low | Phase 5 Early |
| `health.py` | 1 | Low | Phase 5 Early |
| `plugins.py` | 7 | Low | Phase 5 Early |
| `graphs.py` | 2 | Low | Phase 5 Early |
| `config.py` | 2 | Low | Phase 5 (may keep direct impl) |
| `completions.py` | - | Low | Phase 5 (may keep direct impl) |
| `datasets.py` | 4 | Medium | Phase 5 |
| `dataset_ops.py` | 3 | Medium | Phase 5 |
| `storage.py` | 3 | Medium | Phase 5 |
| `docs.py` | 2 | Medium | Phase 5 |
| `history.py` | 1 | Medium | Phase 5 |
| `build.py` | 3 | Medium | Phase 5 |
| `serve.py` | 2 | Medium | Phase 5 |
| `ops.py` | 2 | High | Phase 5 |
| `ide.py` | 1 | High | Phase 5 |
| `subsystem.py` | 5 | High | Phase 5 |

---

## Files to Delete in Phase 6

| File | Reason |
|------|--------|
| `commands/context.py` | Superseded by decorator internals |
