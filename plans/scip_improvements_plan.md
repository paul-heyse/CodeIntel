# SCIP Implementation Improvements Plan

This plan captures the next set of improvements for SCIP indexing, parsing, and
downstream integration. Each scope item includes goal, rationale, implementation
steps, representative code patterns, and target files.

## 1. Pyright config path wiring for scip-python

Status: Completed

Goal:
- Ensure scip-python resolves imports and src-layout paths deterministically.

Rationale:
- scip-python relies on Pyright resolution. Explicitly wiring a config path
  avoids silent drift across environments.

Implementation:
1. Add `pyright_config_path` to `ScipIngestOptions`.
2. Stage `pyrightconfig.json` into the scip-python target base for the run.
3. Record the config path in options hashing for cache correctness.

Representative code pattern:
```python
with stage_pyright_config(
    target_base=target_base,
    pyright_config_path=pyright_config_path,
):
    run_scip_python(...)
```

Target files:
- `src/codeintel/build/hamilton/native/options/ingestion.py`
- `src/codeintel/ingestion/scip/cli.py`
- `src/codeintel/build/hamilton/native/ingestion/scip.py`

## 2. Env JSON fallback when pip is unavailable

Status: Completed

Goal:
- Make dependency discovery deterministic even when pip is absent.

Rationale:
- scip-python defaults to pip introspection; uv environments can omit pip.

Implementation:
1. Resolve env JSON: use pip when available, else generate `env.json` under the SCIP dir.
2. Hash the resolved env JSON path and content into options hash.
3. Persist the env source (pip vs json) to telemetry.

Representative code pattern:
```python
resolution = resolve_environment_json(
    environment_json=environment_json,
    scip_dir=scip_dir,
)
environment_json = resolution.environment_json
env_source = resolution.source
```

Target files:
- `scripts/gen_scip_env.py`
- `src/codeintel/ingestion/scip/incremental.py`
- `src/codeintel/ingestion/scip/telemetry.py`
- `src/codeintel/build/hamilton/native/ingestion/scip.py`

## 3. Node heap sizing for large repos

Status: Completed

Goal:
- Avoid Node OOM failures during indexing runs.

Rationale:
- scip-python is Node-based and can exceed default heap in large repos.

Implementation:
1. Add `scip_node_max_old_space_mb` to `ScipIngestOptions`.
2. Inject `NODE_OPTIONS` into tool execution environment when set.

Representative code pattern:
```python
env = {}
if scip_node_max_old_space_mb is not None and scip_node_max_old_space_mb > 0:
    env["NODE_OPTIONS"] = f"--max-old-space-size={scip_node_max_old_space_mb}"
run_options = ToolRunOptions(..., env=env or None)
```

Target files:
- `src/codeintel/build/hamilton/native/options/ingestion.py`
- `src/codeintel/ingestion/engine/scip.py`

## 4. Persist index metadata in a dedicated table

Status: Completed

Goal:
- Make index provenance queryable and auditable.

Rationale:
- index metadata (project_root, text_document_encoding, tool versions) is needed
  to debug mismatches and validate ingestion.

Implementation:
1. Add `core.scip_index_metadata` schema and row model.
2. Extend `ScipParsedIndex` to include metadata.
3. Emit a single metadata row per run during ingestion.

Representative code pattern:
```python
metadata_row = {
    "repo": env.snapshot.repo,
    "commit": env.snapshot.commit,
    "project_root": parsed.metadata.project_root,
    "text_document_encoding": parsed.metadata.text_document_encoding,
    "tool_name": parsed.metadata.tool_name,
    "tool_version": parsed.metadata.tool_version,
    "tool_arguments": parsed.metadata.tool_arguments,
    "created_at": created_at,
}
```

Target files:
- `src/codeintel/ingestion/scip/protobuf_parser.py`
- `src/codeintel/build/hamilton/native/ingestion/scip.py`
- `src/codeintel/core/schemas/output_registry.py`
- `src/codeintel/core/data_models/rows.py`

## 5. Generate scip_pb2.pyi alongside scip_pb2.py

Status: Completed

Goal:
- Improve typing fidelity for protobuf parsing and consumers.

Rationale:
- The protobuf docs recommend `--pyi_out` for typed stubs.

Implementation:
1. Add `--pyi_out` to the scip proto codegen command.
2. Save the stub artifact under `build/scip/proto/scip_pb2.pyi`.

Representative code pattern:
```python
args = [
    "-m",
    "grpc_tools.protoc",
    "-I",
    str(proto_dir),
    "--python_out",
    str(out_dir),
    "--pyi_out",
    str(out_dir),
    str(proto_path),
]
```

Target files:
- `src/codeintel/build/hamilton/native/ingestion/scip_proto.py`
- `scripts/scip_proto_codegen.sh`

## 6. Expand proto typing protocols for new fields

Status: Completed

Goal:
- Align typed protocols with newer SCIP occurrence fields.

Rationale:
- Protocols should reflect syntax_kind, enclosing ranges, and overrides.

Implementation:
1. Add optional fields to `OccurrenceProto` in `proto_types.py`.
2. Use those fields in parsing when present.

Representative code pattern:
```python
class OccurrenceProto(Protocol):
    symbol: str
    range: IntListProto
    symbol_roles: int
    syntax_kind: int
    enclosing_range: IntListProto
    override_documentation: StringListProto
    diagnostics: DiagnosticListProto
```

Target files:
- `src/codeintel/ingestion/scip/proto_types.py`
- `src/codeintel/ingestion/scip/protobuf_parser.py`

## 7. Merge symbol relationships across shards

Status: Completed

Goal:
- Preserve relationship edges even when docstrings are missing in a shard.

Rationale:
- The current "best symbol" scoring can drop relationships when a shard has
  weaker metadata.

Implementation:
1. Merge relationships by related symbol instead of replacing the symbol info.
2. Prefer richer documentation while unioning relationships.

Representative code pattern:
```python
merged = _copy_symbol_info(base_sym)
relationships = _merge_relationships(base_sym, shard_sym)
_clear_field(merged, "relationships")
merged.relationships.extend(relationships)
```

Target files:
- `src/codeintel/ingestion/scip/index_store.py`

## 8. Handle empty scope gracefully

Status: Completed

Goal:
- Avoid hard failures when the target scope has no Python files.

Rationale:
- Empty scopes are a valid configuration for constrained runs.

Implementation:
1. Convert "no symbols/occurrences" into a warning + empty tables.
2. Emit a specific telemetry decision so it is visible in diagnostics.

Representative code pattern:
```python
if payload.symbol_row_count == 0 and payload.occurrence_row_count == 0:
    return IngestStep(result=ExecutionResult.ok(message="SCIP scope empty"), payload=empty_tables)
```

Target files:
- `src/codeintel/build/hamilton/native/ingestion/scip.py`
- `src/codeintel/ingestion/scip/telemetry.py`

## 9. Track environment provenance in telemetry and module state

Status: Completed

Goal:
- Make environment changes visible and cache-safe.

Rationale:
- External dependency resolution affects symbol identity and external symbols.

Implementation:
1. Add `environment_source` (pip or json) to telemetry and module state.
2. Include it in options hashing to trigger rebuilds when it changes.

Representative code pattern:
```python
payload["environment_source"] = env_source
telemetry.environment_source = env_source
```

Target files:
- `src/codeintel/ingestion/scip/telemetry.py`
- `src/codeintel/build/hamilton/native/ingestion/scip.py`
- `src/codeintel/core/schemas/output_registry.py`

## 10. Surface syntax_kind and documentation in syntax enrichment

Status: Completed

Goal:
- Improve symbol overlays and downstream search relevance.

Rationale:
- syntax_kind and documentation add semantic context to syntax-enriched facts.

Implementation:
1. Join `syntax_kind` and `documentation` from scip span xref.
2. Propagate these columns into resolved syntax tables.

Representative code pattern:
```python
span = tabular_to_scoped_table(
    q__core__scip_occurrence_span_xref,
    columns=[..., "syntax_kind", "documentation"],
    scope=None,
    require_scope_columns=False,
)
```

Target files:
- `src/codeintel/build/hamilton/native/ingestion/syntax_enrich.py`
- `src/codeintel/core/schemas/output_registry.py`

## 11. External symbol usage analytics

Status: Completed

Goal:
- Turn `core.scip_external_symbols` into dependency usage insights.

Rationale:
- External symbol data can power supply-chain and ecosystem metrics.

Implementation:
1. Add a dataset that aggregates package usage counts by repo/commit.
2. Track top packages and versions used by call sites.

Representative code pattern:
```python
usage = external_symbols.group_by(
    ["repo", "commit", "package_name", "package_version"]
).aggregate([("symbol", "count")])
```

Target files:
- `src/codeintel/build/hamilton/native/analytics/*`
- `src/codeintel/core/schemas/output_registry.py`

## 12. Diagnostics analytics rollup

Status: Completed

Goal:
- Quantify indexing health and failure reasons.

Rationale:
- scip diagnostics are currently stored but not summarized for quality checks.

Implementation:
1. Add an analytics table grouped by severity and source.
2. Track per-file diagnostics counts and top errors.

Representative code pattern:
```python
rollup = diagnostics.group_by(
    ["repo", "commit", "rel_path", "severity"]
).aggregate([("message", "count")])
```

Target files:
- `src/codeintel/build/hamilton/native/analytics/*`
- `src/codeintel/core/schemas/output_registry.py`

## 13. Persist scip-python stdout/stderr artifacts

Status: Completed

Goal:
- Make tool outputs available for later inspection.

Rationale:
- Debugging index failures often requires tool stdout/stderr context.

Implementation:
1. Save stdout/stderr under `build/scip/runs/` per run_id.
2. Link artifact paths in run telemetry or a run artifact table.

Representative code pattern:
```python
run_dir = scip_dir / "runs" / run_id
run_dir.mkdir(parents=True, exist_ok=True)
(run_dir / "stdout.log").write_text(result.stdout, encoding="utf-8")
(run_dir / "stderr.log").write_text(result.stderr, encoding="utf-8")
```

Target files:
- `src/codeintel/build/hamilton/native/ingestion/scip.py`
- `src/codeintel/ingestion/engine/scip.py`
- `src/codeintel/ingestion/scip/telemetry.py`
