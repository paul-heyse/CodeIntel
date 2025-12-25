# DAG End-to-End Pilot Plan (Straightforward Outputs)

## Purpose
Establish a full, end-to-end Apache Hamilton DAG path for the simplest outputs first.
The intent is to create a repeatable blueprint that validates the DAG-first basis across
build, storage, CLI, and serving, then reuse the pattern for more complex outputs.

This plan complements and draws on the shared library patterns in
`docs/full_dag_basis_implementation_plan.md`, but is scoped to a single, low-risk pilot
that proves the system end to end.

## Guiding principles
- DAG outputs are the source of truth for target materializations.
- Targets are defined by `t__*` anchors and DataSaver tags, not ad hoc logic.
- Contracts and schemas are first-class and drive support node generation.
- Build produces typed artifacts and records; serving consumes those artifacts only.
- CLI behavior and flags should derive from DAG metadata, not bespoke wiring.

## Scope
In scope:
- One straightforward output (and its minimal upstream support nodes).
- End-to-end path: Hamilton DAG -> materialization -> storage -> serving -> CLI.
- Contract alignment, schema materialization, and validation behavior.
- Observability and minimal diagnostics needed to validate the path.

Out of scope:
- Migration of all outputs or all target families.
- Performance tuning beyond what is required to validate the path.
- Refactoring of unrelated CLI or serving behavior.

## Selection criteria for the pilot output
Pick the simplest output that still exercises the full pipeline:
- Table-only or artifact-only materialization (no mixed or multi-output).
- No external services or network dependencies.
- Stable contract with clear schema and low cardinality.
- Minimal downstream fan-out.
- Used by at least one existing CLI or serving path (to validate end to end).

## Output inventory artifact template (YAML/JSON)
Use a lightweight artifact to keep output discovery in sync across build, storage,
CLI, and serving.

YAML (preferred for humans):
```yaml
version: 1
generated_at: "YYYY-MM-DD"
outputs:
  - target: "function_metrics"
    domain: "analytics"
    anchor: "t__function_metrics"
    materialization: "table"
    table_keys:
      - "analytics.function_metrics"
      - "analytics.function_types"
      - "analytics.function_validation"
    contracts:
      - "analytics.function_metrics"
      - "analytics.function_types"
      - "analytics.function_validation"
    downstream_consumers:
      - "storage.views"
      - "serving.search_index"
      - "cli.datasets"
    upstream_targets:
      - "modules"
      - "goids"
    notes: "Core analytics metric set used across serving and CLI."
```

JSON (machine-first alternative):
```json
{
  "version": 1,
  "generated_at": "YYYY-MM-DD",
  "outputs": [
    {
      "target": "function_metrics",
      "domain": "analytics",
      "anchor": "t__function_metrics",
      "materialization": "table",
      "table_keys": [
        "analytics.function_metrics",
        "analytics.function_types",
        "analytics.function_validation"
      ],
      "contracts": [
        "analytics.function_metrics",
        "analytics.function_types",
        "analytics.function_validation"
      ],
      "downstream_consumers": [
        "storage.views",
        "serving.search_index",
        "cli.datasets"
      ],
      "upstream_targets": [
        "modules",
        "goids"
      ],
      "notes": "Core analytics metric set used across serving and CLI."
    }
  ]
}
```
Canonical inventory artifact: `src/codeintel/core/registry/dag_output_inventory.yaml`.

Notes on schema behavior:
- `materialization` is validated to be one of: table, artifact, mixed.
- table outputs must declare `table_keys`; artifact outputs must not.
- `contracts` defaults to `table_keys` when omitted for table outputs.
- `pilot` is an optional boolean flag for highlighting the current focus output.

## Implementation insights (from initial inventory and CLI wiring)
- The shared-flags mechanism for Command[T] commands must be explicitly flattened so Cyclopts
  can parse `--output-format` and `--json` for new command groups.
- CLI output format resolution is driven by shared flags; new commands should keep the `flags`
  field and avoid ad hoc output-format options unless the command has a distinct format.
- Inventory loading should be centralized in `RegistryService` so build, CLI, and serving share
  a single source of truth for target metadata.
- Inventory validation must fail fast on duplicate targets and mismatched materialization rules
  to avoid downstream schema mismatches.

## Phase 0: Inventory and candidate selection
Deliverables:
- Output inventory artifact (YAML or JSON) listing:
  - target name
  - DAG anchor node name
  - materialization type (table/artifact/mixed)
  - contract/schema references
  - downstream consumers (CLI, serving, docs)
  - current materialization path (if any)
- Ranked candidate list with rationale tied to the criteria above:
  - `analytics.function_metrics` (`t__function_metrics`)
    - Table-only output (3 tables), no external services, stable contracts.
    - Heavily used by storage views, serving search index, and CLI datasets.
    - Already uses shared save patterns, so it is a clean DAG-first baseline.
  - `analytics.function_ast_features` (`t__function_ast_features`)
    - Single table, simple compute path, no external services.
    - Lighter downstream use, useful as a secondary validation target.
  - `analytics.coverage_functions` (`t__coverage_functions`)
    - Table-only with Ibis materialization, depends on coverage_lines ingest.
    - Representative of Ibis table saver path, but higher upstream coupling.
  - `analytics.data_models` (`t__data_models`)
    - Multi-table output (models/fields/relationships), more complex but internal.
    - Good follow-on once pilot patterns are validated.
- Selected pilot output:
  - `analytics.function_metrics` (`t__function_metrics`)
    - Most representative without external dependencies.
    - End-to-end coverage: build -> storage -> serving -> CLI paths already exist.

Acceptance criteria:
- Inventory is checked in and reviewed.
- Pilot output selection is documented and agreed.
Status:
- Completed: inventory artifact created at `src/codeintel/core/registry/dag_output_inventory.yaml`
  with analytics, graphs, ingestion, and export targets.
- Completed: pilot output remains `analytics.function_metrics` (`t__function_metrics`).

## Phase 1: DAG-first build path for the pilot
Tasks:
- Ensure the pilot output is produced via a `t__*` anchor.
- Use shared pattern helpers (`native/patterns/*`) for materialization records:
  - shared hash node
  - `record_from_materializations(...)`
  - target run record structure
- Ensure support nodes use contract-driven inputs when possible.
- Remove any direct env/gateway access from compute nodes for this output.
- Add or align dataset contract/schema for the pilot output.

Deliverables:
- DAG definition for the pilot output is canonical and uses shared patterns.
- Dataset contract and schema are aligned with the materialized output.
- Target run record includes all required metadata for downstream use.

Acceptance criteria:
- `t__*` anchor exists and uses shared record builder.
- Contract validation passes in lenient mode; strict mode available in CI.

## Phase 2: Storage and serving integration
Tasks:
- Ensure storage registration for the pilot output uses shared services.
- Confirm the serving layer reads only from published artifacts/tables.
- Add or update storage repository methods (if needed) for the pilot output.
- Update serving snapshot or materialization pipeline to reference the pilot output.

Deliverables:
- Storage uses shared boundary services (no direct DuckDB use outside storage).
- Serving snapshot references the pilot output via storage contracts.

Acceptance criteria:
- Serving path works with only the materialized output present.
- No direct access to build internals from serving for the pilot output.

## Phase 3: CLI and UX alignment
Tasks:
- Ensure CLI commands referencing the pilot output are DAG-driven.
- Derive CLI help and flags from shared metadata (where feasible).
- Add or update CLI validation and messaging aligned with new contracts.

Deliverables:
- CLI can read or export the pilot output with stable help/flags.
- CLI uses validation mode controls consistently.

Acceptance criteria:
- CLI snapshot tests for the pilot output are stable.
- CLI exit codes and errors reflect validation mode behavior.

Recent updates:
- Added `registry` CLI group with `registry outputs` and `registry validate` to expose the
  inventory and filtering via shared output formatting.
- Registered the command group in the root CLI app so it is available in the standard entrypoint.
- Added options for registry filters (domain, materialization, targets, table_key, pilot_only).

## Phase 4: End-to-end tests and artifacts
Tasks:
- Add a focused integration test for the pilot output path.
- Capture golden outputs or fixtures for serving validation.
- Update targeted tests where contracts or schema changes occur.

Deliverables:
- Pilot end-to-end test with deterministic artifacts.
- Updated goldens/fixtures for the pilot output.

Acceptance criteria:
- Pilot test passes in isolation and in CI subsets.
- No regressions in related analytics/build/serving tests.

## Phase 5: Replication checklist for next outputs
Create a short checklist that can be applied to the next output:
- Anchor `t__*` node exists and uses shared materialization patterns.
- Contract/schema aligned and referenced by support nodes.
- Storage/serving boundary uses shared services only.
- CLI entry points resolve via DAG metadata.
- Tests and goldens updated for the output.

Deliverable:
- A reusable migration checklist with concrete "done" criteria.

## Work completed so far (implementation status)
- Added canonical inventory artifact: `src/codeintel/core/registry/dag_output_inventory.yaml`.
- Added inventory loader and validation types: `src/codeintel/core/registry/service.py`.
- Added registry CLI commands: `src/codeintel/cli/commands/registry.py`.
- Added registry CLI option specs: `src/codeintel/cli/options/registry.py`.
- Registered registry app in the CLI entrypoint: `src/codeintel/cli/__init__.py`.
- Added tests for loader and CLI wiring:
  `tests/core/test_dag_output_inventory.py`, `tests/cli/test_registry_cli.py`.
- Added shared-flags flattening for Command[T] commands so `--output-format` works
  consistently across new command groups.

## Open follow-ups and validation
- Resolve the OpenTelemetry import dependency used during CLI/test import
  (`opentelemetry.propagators.baggage`) so pytest can run without import errors.
- Re-run the focused tests:
  `uv run pytest -q tests/core/test_dag_output_inventory.py tests/cli/test_registry_cli.py`.

## Risks and mitigations
- Contract/schema drift: enforce contract-first updates and tighten CI checks.
- DAG output discovery gaps: keep inventory in sync via tooling or CI guard.
- CLI snapshot churn: derive help text from stable metadata.
- Serving coupling: ensure serving only consumes materialized outputs.

## Next steps after the pilot
- Use the checklist to migrate the next-most-complex output.
- Refresh the broader DAG basis plan based on pilot learnings.
- Expand inventory to include all targets and prioritize by impact/risk.
