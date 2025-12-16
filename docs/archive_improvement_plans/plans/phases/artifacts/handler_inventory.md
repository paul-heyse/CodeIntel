# Handler Inventory

> **Generated:** December 10, 2024  
> **Purpose:** Track handler migration status for Phase 3

## Summary

| Metric | Count |
|--------|-------|
| Total handler files | 13 |
| Total handler functions | 40+ |
| Files with local param helpers | 11 |
| Files requiring runtime | 9 |
| Files requiring gateway | 10 |
| Files requiring graph_runtime | 3 |

## Context Types In Use

| Context Type | Location | Files Using |
|--------------|----------|-------------|
| `EnhancedHandlerContext` | `handlers/protocol.py` | All handler files |
| `HandlerContext` | `handlers/base.py` | Some utilities |

## Handler Details

### handlers/jobs.py

| Function | Context Type | Param Helpers | Runtime | Gateway | Graph Runtime |
|----------|--------------|---------------|---------|---------|---------------|
| `jobs_list_handler` | `EnhancedHandlerContext` | `_get_str_param`, `_get_int_param` | No | No | No |
| `jobs_status_handler` | `EnhancedHandlerContext` | `_require_str_param` | No | No | No |
| `jobs_output_handler` | `EnhancedHandlerContext` | `_require_str_param` | No | No | No |
| `jobs_cancel_handler` | `EnhancedHandlerContext` | `_require_str_param` | No | No | No |
| `jobs_cleanup_handler` | `EnhancedHandlerContext` | `_get_int_param` | No | No | No |

**Local param helpers:** `_get_str_param`, `_require_str_param`, `_get_int_param`

---

### handlers/health.py

| Function | Context Type | Param Helpers | Runtime | Gateway | Graph Runtime |
|----------|--------------|---------------|---------|---------|---------------|
| `health_check_handler` | `EnhancedHandlerContext` | None | No | No | No |

**Local param helpers:** None (uses module-level health checks)

---

### handlers/ops.py

| Function | Context Type | Param Helpers | Runtime | Gateway | Graph Runtime |
|----------|--------------|---------------|---------|---------|---------------|
| `op_list_handler` | `EnhancedHandlerContext` | `_get_str_param` | No | No | No |
| `op_call_handler` | `EnhancedHandlerContext` | `_require_str_param`, `_get_bool_param` | Yes | Yes | No |
| `dataset_list_handler` | `EnhancedHandlerContext` | None | Yes | Yes | No |
| `dataset_describe_handler` | `EnhancedHandlerContext` | `_require_str_param` | No | No | No |
| `dataset_verify_handler` | `EnhancedHandlerContext` | `_get_str_param` | Yes | Yes | No |
| `serve_http_handler` | `EnhancedHandlerContext` | `_get_str_param`, `_get_int_param`, `_get_bool_param` | Yes | Yes | No |
| `serve_mcp_handler` | `EnhancedHandlerContext` | `_get_bool_param` | Yes | Yes | No |

**Local param helpers:** `_get_str_param`, `_require_str_param`, `_get_int_param`, `_get_bool_param`, `_parse_param_value`

---

### handlers/build.py

| Function | Context Type | Param Helpers | Runtime | Gateway | Graph Runtime |
|----------|--------------|---------------|---------|---------|---------------|
| `build_status_handler` | `EnhancedHandlerContext` | `_get_str_param` | Yes | Yes | No |
| `build_run_handler` | `EnhancedHandlerContext` | `_get_str_param`, `_get_bool_param` | Yes | Yes | No |
| `build_history_handler` | `EnhancedHandlerContext` | `_get_str_param`, `_get_int_param` | Yes | Yes | No |

**Local param helpers:** `_get_str_param`, `_get_int_param`, `_get_bool_param`

---

### handlers/docs.py

| Function | Context Type | Param Helpers | Runtime | Gateway | Graph Runtime |
|----------|--------------|---------------|---------|---------|---------------|
| `docs_export_handler` | `EnhancedHandlerContext` | `_get_str_param`, `_get_bool_param`, `_extract_list_param` | Yes | Yes | No |
| `docs_validate_handler` | `EnhancedHandlerContext` | None | Yes | Yes | No |

**Local param helpers:** `_get_str_param`, `_get_bool_param`, `_extract_list_param`

---

### handlers/storage.py

| Function | Context Type | Param Helpers | Runtime | Gateway | Graph Runtime |
|----------|--------------|---------------|---------|---------|---------------|
| `validate_macros_handler` | `EnhancedHandlerContext` | `_get_str_param`, `_get_bool_param` | Yes | Yes | No |
| `generate_macros_handler` | `EnhancedHandlerContext` | None | No | No | No |
| `profile_storage_handler` | `EnhancedHandlerContext` | `_get_str_param`, `_get_bool_param` | Yes | Yes | No |

**Local param helpers:** `_get_str_param`, `_get_bool_param`

---

### handlers/graphs.py

| Function | Context Type | Param Helpers | Runtime | Gateway | Graph Runtime |
|----------|--------------|---------------|---------|---------|---------------|
| `graph_plugins_list_handler` | `EnhancedHandlerContext` | `_get_str_param`, `_get_bool_param`, `_extract_tuple_param` | No | No | No |
| `graph_plugins_plan_handler` | `EnhancedHandlerContext` | `_get_str_param`, `_extract_tuple_param` | No | No | No |

**Local param helpers:** `_get_str_param`, `_get_bool_param`, `_extract_tuple_param`

---

### handlers/history.py

| Function | Context Type | Param Helpers | Runtime | Gateway | Graph Runtime |
|----------|--------------|---------------|---------|---------|---------------|
| `history_timeseries_handler` | `EnhancedHandlerContext` | `_get_str_param`, `_require_str_param`, `_get_int_param`, `_get_path_param`, `_get_enum_str_param` | Yes | Yes | No |

**Local param helpers:** `_get_str_param`, `_require_str_param`, `_get_int_param`, `_get_path_param`, `_get_enum_str_param`

---

### handlers/datasets.py

| Function | Context Type | Param Helpers | Runtime | Gateway | Graph Runtime |
|----------|--------------|---------------|---------|---------|---------------|
| `datasets_list_handler` | `EnhancedHandlerContext` | `_get_str_param`, `_get_bool_param` | Yes | Yes | No |
| `datasets_lint_handler` | `EnhancedHandlerContext` | None | Yes | Yes | No |
| `datasets_snapshot_handler` | `EnhancedHandlerContext` | `_get_str_param` | No | No | No |
| `datasets_diff_handler` | `EnhancedHandlerContext` | `_get_str_param` | No | No | No |

**Local param helpers:** `_get_str_param`, `_get_bool_param`

---

### handlers/ide.py

| Function | Context Type | Param Helpers | Runtime | Gateway | Graph Runtime |
|----------|--------------|---------------|---------|---------|---------------|
| `ide_hints_handler` | `EnhancedHandlerContext` | `_get_rel_path` | Yes | Yes | Yes |

**Local param helpers:** `_get_rel_path`

---

### handlers/plugins.py

| Function | Context Type | Param Helpers | Runtime | Gateway | Graph Runtime |
|----------|--------------|---------------|---------|---------|---------------|
| `plugins_list_handler` | `EnhancedHandlerContext` | None | No | No | No |
| `plugins_discover_handler` | `EnhancedHandlerContext` | None | No | No | No |
| `plugins_info_handler` | `EnhancedHandlerContext` | `_require_str_param` | No | No | No |
| `plugins_paths_handler` | `EnhancedHandlerContext` | None | No | No | No |
| `plugins_new_handler` | `EnhancedHandlerContext` | `_require_str_param`, `_get_str_param` | No | No | No |
| `plugins_test_handler` | `EnhancedHandlerContext` | `_require_str_param` | No | No | No |
| `plugins_validate_handler` | `EnhancedHandlerContext` | `_require_str_param` | No | No | No |

**Local param helpers:** `_get_str_param`, `_require_str_param`

---

### handlers/subsystem.py

| Function | Context Type | Param Helpers | Runtime | Gateway | Graph Runtime |
|----------|--------------|---------------|---------|---------|---------------|
| `subsystem_list_handler` | `EnhancedHandlerContext` | `_get_int_param`, `_get_str_param` | Yes | Yes | Yes |
| `subsystem_show_handler` | `EnhancedHandlerContext` | `_require_str_param` | Yes | Yes | Yes |
| `subsystem_profiles_handler` | `EnhancedHandlerContext` | `_get_int_param` | Yes | Yes | Yes |
| `subsystem_coverage_handler` | `EnhancedHandlerContext` | `_get_int_param` | Yes | Yes | Yes |
| `subsystem_module_memberships_handler` | `EnhancedHandlerContext` | `_require_str_param` | Yes | Yes | Yes |

**Local param helpers:** `_get_int_param`, `_get_str_param`, `_require_str_param`

---

### handlers/protocol.py

**Purpose:** Defines `EnhancedHandlerContext` and `HandlerProtocol`.

| Export | Type | Description |
|--------|------|-------------|
| `EnhancedHandlerContext` | Dataclass | Primary context type for all handlers |
| `HandlerProtocol` | Protocol | Contract for handler functions |
| `handler_context` | Context manager | Creates handler context with cleanup |

---

### handlers/base.py

**Purpose:** Provides logging setup and `HandlerContext` (alternate context).

| Export | Type | Description |
|--------|------|-------------|
| `setup_logging` | Function | Configure logging based on verbosity |
| `HandlerContext` | Dataclass | Alternate context type (less used) |
| `build_handler_context` | Function | Factory for HandlerContext |
| `open_handler_gateway` | Function | Open gateway for handler use |

---

## Param Helper Function Patterns

The following local helper functions appear across multiple handlers:

| Helper Function | Files Using | Purpose |
|-----------------|-------------|---------|
| `_get_str_param` | 10 | Extract optional string param |
| `_require_str_param` | 6 | Extract required string param |
| `_get_int_param` | 7 | Extract optional int param |
| `_get_bool_param` | 7 | Extract optional bool param |
| `_get_path_param` | 1 | Extract optional path param |
| `_get_enum_str_param` | 1 | Extract enum as string |
| `_extract_list_param` | 1 | Extract list param |
| `_extract_tuple_param` | 1 | Extract tuple param |

**Migration note:** These will be replaced by `HandlerContext.param_*` methods in Phase 3.

---

## Migration Priority

| Handler File | Complexity | Priority | Notes |
|--------------|------------|----------|-------|
| `jobs.py` | Low | Phase 3 | No runtime/gateway needed |
| `health.py` | Low | Phase 3 | No param helpers |
| `plugins.py` | Low | Phase 3 | No runtime/gateway needed |
| `graphs.py` | Low | Phase 3 | No runtime/gateway needed |
| `datasets.py` | Medium | Phase 3 | Mixed requirements |
| `storage.py` | Medium | Phase 3 | Runtime/gateway needed |
| `docs.py` | Medium | Phase 3 | Runtime/gateway needed |
| `build.py` | Medium | Phase 3 | Runtime/gateway needed |
| `history.py` | Medium | Phase 3 | Many param helpers |
| `ops.py` | High | Phase 3 | Complex, many handlers |
| `ide.py` | High | Phase 3 | Requires graph_runtime |
| `subsystem.py` | High | Phase 3 | Requires graph_runtime |
