# Build Module — Add a Target Playbook (Hamilton-First)

This playbook documents the canonical, low-friction path for adding a new build target in the
Hamilton-first architecture.

## 1) Define the contract (target spec)

- Pick a unique `target_name` (e.g., `"my_target"`).
- Define contract outputs:
  - Tables: fully-qualified `schema.table` keys
  - Artifacts: `ArtifactSpec(name=..., path_template=..., description=...)`
- Register a `TARGET_SPECS` entry via `make_output_target(...)` in the relevant native module under:
  - `src/codeintel/build/hamilton/native/ingestion/`
  - `src/codeintel/build/hamilton/native/graphs/`
  - `src/codeintel/build/hamilton/native/analytics/`
  - `src/codeintel/build/hamilton/native/export/`

## 2) Implement Hamilton-native nodes

Use the small, consistent set of node kinds:

- **Compute/tool nodes**:
  - Pure compute: return an Ibis expression, row tuples, or a result container
  - Tool boundary: run external tools via `env.providers` (and return a result container)
- **Materialize nodes**:
  - Convert saver metadata into a `TargetRunRecord`

### Tagging (required)

Use canonical tagging helpers from `codeintel.build.hamilton.tagging`:

- `tag_compute(...)`
- `tag_materialize(...)`
- `tag_tool(...)`
- `tag_helper(...)`

## 3) Standardize result shapes (when appropriate)

If your compute node’s output is effectively `{success, table_counts, error}`, return
`ExecutionResult` (`src/codeintel/build/hamilton/execution_result.py`) instead of defining a new
dataclass.

If your compute stage must carry extra payload used by downstream nodes (e.g., CFG extraction
carrying intermediate results), keep a domain-specific dataclass. Ensure it still exposes:

- `success: bool`
- `table_counts: dict[str, int]`
- `error: str | None`

so `executor_materialize(...)` and other shared tooling can treat it uniformly.

## 4) Use typed saver metadata (`dict[str, object]`)

At the Hamilton boundary, saver metadata must be typed as `dict[str, object]`.

- Use `SaveToObjectMetadataDecorator` (not Hamilton’s `SaveToDecorator`).
- Parse metadata into typed structures (e.g., `DuckDBMaterializationMetadata.from_mapping(...)`)
  immediately when you need structured access.

## 5) Validate locally (fast gates)

Run the standard quality suite:

```bash
uv run python -m tools.quality_report --output build/quality-results/quality_report.json
uv run pytest -q
```

Notes:
- `tools/guardrails.py` includes graph validation, so Hamilton DAG breakages surface without a full
  suite run.

