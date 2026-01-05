# CLI Legacy Decommission Plan

## Goals
- Remove dead or legacy CLI features to reduce surface area and maintenance cost.
- Align `src/codeintel/cli` with the current Cyclopts-first architecture.
- Delete deprecated/unused modules rather than leaving compatibility shims.

## Scope (to delete or fully decommission)
- Pipeline/batch execution utilities.
- CLI output envelope utilities (stdin/stdout helpers).
- CLI execution middleware/progress/types subsystem.
- Rendering table specifications and table helpers.
- Legacy streaming emitters tied to pipelines.
- Legacy YAML/JSON CLI config loader (TOML-only design).
- Legacy schema registry usage in dataset ops handlers.

## Non-goals
- Changing CLI command surfaces outside of removed features.
- Re-adding compatibility shims once removed.

## Phase 0: Inventory + safety checks
1. Run a reference sweep to confirm no runtime code depends on the deletion targets.
   - `rg -n "cli.project.pipelines|OutputEnvelope|render_table|TableSpec|ExecutionPipeline|ProgressTracker" src tests`
2. Record any new call sites and add them to this plan before deletion.

## Phase 1: Remove pipeline/batch subsystem
1. Delete `src/codeintel/cli/project/pipelines.py`.
2. Update `src/codeintel/cli/project/__init__.py` to remove pipeline exports:
   - `BatchOperation`, `BatchItemResult`, `BatchResult`, `PipelineConfig`,
     `execute_batch`, `load_batch`, `read_stdin_operations`, `stream_results`.
3. Remove streaming helpers in `src/codeintel/cli/rendering/service.py`:
   - `emit_stream_result`, `emit_stream_progress`, `emit_stream_summary`.
4. Update `src/codeintel/cli/rendering/service.py` docstring to remove
   references to “StreamingRenderer”.
5. Remove any docs referencing batch/pipeline execution if present.

## Phase 2: Remove CLI output envelope utilities
1. Delete `src/codeintel/cli/core/output.py`.
2. Update `src/codeintel/cli/core/__init__.py` to remove:
   - `OutputEnvelope`, `read_stdin_records`, `iter_stdin_records`,
     `merge_stdin_with_args`.
3. Update module-level docstring in `src/codeintel/cli/core/__init__.py`
   to remove “OutputEnvelope” references.

## Phase 3: Remove execution middleware/progress/types subsystem
1. Delete:
   - `src/codeintel/cli/execution/middleware.py`
   - `src/codeintel/cli/execution/progress.py`
   - `src/codeintel/cli/execution/types.py`
2. Update `src/codeintel/cli/execution/__init__.py` to export only the
   registry surface (`OperationSpec`, `register_operation`, `execute_operation`, etc.).
3. Remove any docstrings/examples in `src/codeintel/cli/execution/__init__.py`
   referencing middleware or progress.

## Phase 4: Remove rendering table specs + table helpers
1. Delete:
   - `src/codeintel/cli/rendering/specs.py`
   - `src/codeintel/cli/rendering/table.py`
2. Update `src/codeintel/cli/rendering/__init__.py` to remove exports:
   - `TableSpec`, `ColumnSpec`, `*_TABLE` constants.
3. Update `src/codeintel/cli/rendering/service.py`:
   - Remove `TableSpec` import and `render_table` API.
   - Remove table-specific branches in `render_cli_result`.
   - Update module docstring to remove references to table rendering.
4. Update `tests/cli/rendering/test_service.py`:
   - Remove TableSpec-based tests.
   - Keep only tests that validate supported render paths (text/json/error).

## Phase 5: Remove legacy YAML/JSON CLI config loader
1. Replace `src/codeintel/cli/config/loader.py` with TOML-only loader logic
   embedded in `src/codeintel/cli/config/service.py` or a new TOML-only module.
2. Remove YAML/JSON paths and parsing logic:
   - Delete `DEFAULT_CONFIG_PATHS` and YAML/JSON parsing functions.
3. Update `src/codeintel/cli/commands/config.py`:
   - `config init` writes TOML (not YAML).
   - `config path` uses TOML resolution logic only.
4. Update `tests/cli/config/*` to align with TOML-only config inputs.
5. Remove YAML dependency if no other modules require it.

## Phase 6: Remove legacy schema registry usage in dataset ops
1. Update `src/codeintel/cli/handlers/ops.py` to use canonical
   `codeintel.build.schemas` APIs (schema provider + contract iteration).
2. Remove any imports from `codeintel.build.hamilton.contracts.schemas` in CLI code.
3. Update dataset ops tests to match the canonical schema provider results.

## Phase 7: Final cleanup sweep
1. Run `rg` for deleted symbols and ensure no references remain.
2. Update any CLI docs or usage examples that mention removed features.

## Acceptance gates
- No references to deleted modules or symbols in `src/` or `tests/`.
- `uv run python -m tools.quality_report --output build/quality-results/quality_report.json` passes.
- Focused CLI tests pass:
  - `tests/cli/config`
  - `tests/cli/commands`
  - `tests/cli/handlers`
  - `tests/cli/rendering`

## Deletion checklist (by file)
- `src/codeintel/cli/project/pipelines.py`
- `src/codeintel/cli/core/output.py`
- `src/codeintel/cli/execution/middleware.py`
- `src/codeintel/cli/execution/progress.py`
- `src/codeintel/cli/execution/types.py`
- `src/codeintel/cli/rendering/specs.py`
- `src/codeintel/cli/rendering/table.py`
- `tests/cli/rendering/test_service.py` (replace with minimal coverage)

