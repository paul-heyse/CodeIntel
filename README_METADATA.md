# CodeIntel Metadata Outputs

This document describes the consolidated artifacts emitted by `generate_documents.sh` under `Document Output/`. Each dataset is produced by the CodeIntel enrichment pipeline and captures different facets of the repository graph. The intent is to give downstream AI agents enough semantic context to reason about the codebase without re-running the heavy analysis steps.

## 1. GOID Registry (`goids.parquet` / `goids.jsonl`)

**Purpose**: Stable identifier canonicalization for all Python entities (modules, functions, classes, and CFG blocks). Downstream graph tables reference GOIDs exclusively.

**Origin**: `CodeIntel.enrich.goid_builder.GOIDBuilder` walks the AST index (`build/enrich/ast/ast_nodes.*`). For each module and code element it constructs an `EntityDescriptor` and hashes it via `CodeIntel.ids.goid.compute_goid`, which embeds repo+commit, language, relative path, kind, and normalized qualname.

**Columns**

| Column        | Type        | Description |
|---------------|-------------|-------------|
| `goid_h128`   | decimal(38) | 128-bit integer hash key for the entity. This is the canonical foreign key in all other tables. |
| `urn`         | string      | Human-readable GOID URN: `goid:<repo>/<path>#<language>:<kind>:<qualname>?s=<start>&e=<end>`. |
| `repo`        | string      | Repository slug passed to the builder. |
| `commit`      | string      | Commit SHA at analysis time. |
| `rel_path`    | string      | Repo-relative file path. |
| `language`    | string      | Lowercase language tag (`python`). |
| `kind`        | string      | Entity kind (`module`, `function`, `class`, `method`, `block`). Blocks correspond to CFG basic blocks. |
| `qualname`    | string      | Dotted qualified name synthesized from AST scope. Modules use `pkg.module`. |
| `start_line`  | int         | First line (1-based) spanned by the entity in `rel_path`. |
| `end_line`    | int/null    | Last line if bounded; null for modules or unknown. |
| `created_at`  | timestamp   | Generation timestamp. |

## 2. GOID Crosswalk (`goid_crosswalk.parquet` / `goid_crosswalk.jsonl`)

**Purpose**: Anchor GOIDs to the multiple structural sources used during enrichment: AST nodes, SCIP symbols, chunk IDs, and future CST/CFG references. Allows reversible mapping from any identifier to a GOID.

**Origin**: Emitted alongside the registry from `GOIDBuilder.write_artifacts`. Each GOID may have multiple crosswalk entries (e.g., one per AST occurrence).

**Columns**

| Column          | Type        | Description |
|-----------------|-------------|-------------|
| `goid`          | string      | GOID URN (text form). |
| `lang`          | string      | Language tag. |
| `module_path`   | string/null | Dotted module path derived from `rel_path` sans `.py`. |
| `file_path`     | string/null | Repository-relative file path. |
| `start_line`    | int/null    | Start line of the associated AST/CST/CFG element. |
| `end_line`      | int/null    | End line. |
| `scip_symbol`   | string/null | SCIP symbol if this GOID was matched to a SCIP occurrence. Populated later when the pipeline cross-references SCIP JSON against GOIDs. |
| `ast_qualname`  | string/null | Qualified name reported by LibCST/AST. |
| `cst_node_id`   | string/null | Future hook for CST IDs (currently null unless CST pipeline emits anchors). |
| `chunk_id`      | int/string  | Chunk identifier used by embedding pipelines, typically `<path>:<start>:<end>`. |
| `symbol_id`     | string/null | Reserved for alternate symbol registries. |
| `updated_at`    | timestamp   | Last write time. |

## 3. Call Graph (`call_graph_nodes.*`, `call_graph_edges.*`)

**Purpose**: Static call graph capturing callables (nodes) and callsites (edges) across the repo. Used for impact analysis and topology-aware tooling.

**Origin**: `CodeIntel.enrich.callgraph.CallGraphBuilder`. Inputs:
- AST + scope info per file (`collect_python_files`).
- Imports resolved via `_ImportResolver`.
- Optional SCIP signal for higher-confidence matches.

**Nodes Columns**

| Column         | Type         | Description |
|----------------|--------------|-------------|
| `goid_h128`    | decimal(38)  | GOID hash for the callable. |
| `language`     | string       | Language (python). |
| `kind`         | string       | Callable kind: `function`, `method`, `class` (for `__call__`), etc. |
| `arity`        | int          | Number of positional parameters detected. |
| `is_public`    | bool         | Heuristic based on naming (leading underscore => false). |
| `rel_path`     | string       | Repo-relative file path containing the callable. |

**Edges Columns**

| Column              | Type         | Description |
|---------------------|--------------|-------------|
| `caller_goid_h128`  | decimal(38)  | GOID hash of the caller. |
| `callee_goid_h128`  | decimal(38)  | GOID hash of the callee; null if unresolved/dynamic. |
| `callsite_path`     | string/null  | File path holding the call expression. |
| `callsite_line`     | int/null     | 1-based line number for the call. |
| `callsite_col`      | int/null     | Column offset. |
| `language`          | string       | Language at the callsite. |
| `kind`              | string       | Edge kind (`direct`, `builtin`, etc.). |
| `resolved_via`      | string       | Provenance of the match: `scip`, `scope`, `heuristic`. Higher confidence flows are prioritized when deduplicating. |
| `confidence`        | float        | Builder-assigned confidence in [0,1]. |
| `evidence_json`     | json         | Additional context (AST text snippets, import alias). |

Edges are deduplicated per `(caller, callee, line, column)` and sorted for deterministic output. Any unresolved call still carries the caller metadata and callsite span.

## 4. Control-Flow Graph (CFG) (`cfg_blocks.*`, `cfg_edges.*`)

**Purpose**: Per-function control-flow scaffolding capturing basic block structure and intra-procedural control edges.

**Origin**: `CodeIntel.enrich.cfg.CFGBuilder`. Inputs:
- AST per function (via `collect_function_info`).
- Block splitting at control constructs (if/else, loops, break/continue, try/except/finally).
- Entry and exit blocks are synthesized for each function.
- GOID entries for both functions and blocks are created at build time.

**Blocks Columns**

| Column               | Type        | Description |
|----------------------|-------------|-------------|
| `function_goid_h128` | decimal(38) | GOID hash of the function owning the blocks. |
| `block_idx`          | int         | Stable block index generated during CFG construction. |
| `block_id`           | string      | Canonical identifier (`<function-goid>:block<idx>`). |
| `label`              | string      | `<kind>:<idx>` label summarizing entry/exit/conditional. |
| `file_path`          | string/null | Path to the file containing the block. |
| `start_line`         | int/null    | First line covered by the block. |
| `end_line`           | int/null    | Last line covered by the block. |
| `kind`               | string      | Block kind (`entry`, `body`, `exit`, `handler`). |
| `stmts_json`         | json        | Serialized AST metadata of statements in the block. |
| `in_degree`          | int         | Number of predecessor edges (computed when building). |
| `out_degree`         | int         | Number of successor edges. |

**Edges Columns**

| Column               | Type        | Description |
|----------------------|-------------|-------------|
| `function_goid_h128` | decimal(38) | Owner function. |
| `src_block_idx`      | int         | Source block index. |
| `dst_block_idx`      | int         | Destination block index. |
| `edge_type`          | string      | `fallthrough`, `true`, `false`, `loop`, `exception`. |
| `cond_json`          | json/null   | Serialized AST of the guard if applicable. |
| `src`/`dst`          | string      | Canonical node IDs mirroring `block_id`. |

## 5. Data-Flow Graph (DFG) (`dfg_edges.*`)

**Purpose**: Intra-procedural data-flow edges capturing definition/use relationships of symbols per block.

**Origin**: Same CFG builder; after block construction it performs a def-use walk tracking variable bindings, phi-like merges, and uses per block.

**Columns**

| Column               | Type        | Description |
|----------------------|-------------|-------------|
| `function_goid_h128` | decimal(38) | Function owning the edge. |
| `src_block_idx`      | int         | Block containing the definition. |
| `dst_block_idx`      | int         | Block containing the use. |
| `src_symbol`         | string      | Variable or temporary defined at the source. |
| `dst_symbol`         | string      | Symbol referenced at the destination. |
| `via_phi`            | bool        | True when this edge models a merge (phi) node from multiple predecessors. |
| `use_kind`           | string      | `read`, `write`, `update`, etc. |

DFG edges can be joined to CFG blocks (via `src_block_idx`/`dst_block_idx`) and to their own synthetic nodes via `dfg_nodes.*` (not exported separately). Each edge may produce corresponding nodes in `dfg_nodes.parquet` when multi-hop traversals are needed.

## 6. JSONL vs Parquet

Each dataset is written twice:
- **Parquet**: Columnar format aligned with the DuckDB catalog; best for analytics and SQL.
- **JSONL**: (Generated via DuckDB `COPY` from Parquet) for LLM ingestion, allowing streaming of each row as a JSON object.

The JSON files reside directly under `Document Output/` with names matching the Parquet base (`goids.jsonl`, `call_graph_edges.jsonl`, etc.). They contain the exact column/value pairs described above.

## 7. Generation Workflow Summary

1. `scip-python` indexes the repository and emits `CodeIntel/index.scip` + JSON view.
2. `CodeIntel.cli.enrich_pipeline all` runs LibCST, AST, analytics, and stores outputs under `CodeIntel/io/ENRICHED`.
3. Dedicated graph commands (`CodeIntel.cli.enrich goids|callgraph|cfg|dfg`) consume the repo and enrichment output, emitting the graph datasets.
4. `generate_documents.sh` copies all artifacts into `Document Output/` and runs the Parquet→JSONL conversion for the graph tables.

Downstream consumers can therefore:
- Join any dataset on `goid_h128`/`goid` to relate nodes, edges, and crosswalk entries.
- Use JSONL files as streaming corpora for LLM context windows.
- Re-run `generate_documents.sh` after code changes to refresh the datasets with new analyses.

## 8. AST Metrics (`ast_metrics.jsonl`)

**Purpose**: Per-file aggregate metrics computed during AST extraction (e.g., node counts, average nesting depth). Useful for heuristic scoring and selecting files that warrant deeper analysis.

**Origin**: Produced by `CodeIntel.enrich.ast_indexer.AstIndexer` while writing `ast_nodes.*`. Metrics are emitted via `CodeIntel.enrich.analytics.ast_metrics`.

**Fields**

| Field            | Type    | Description |
|------------------|---------|-------------|
| `rel_path`       | string  | Repo-relative file path. |
| `node_count`     | int     | Total AST nodes parsed. |
| `function_count` | int     | Number of function/method defs. |
| `class_count`    | int     | Number of class defs. |
| `avg_depth`      | float   | Average AST depth. |
| `max_depth`      | int     | Maximum depth observed. |
| `complexity`     | float   | Heuristic complexity score (sum of decision points). |
| `generated_at`   | string  | ISO timestamp of extraction. |

## 9. AST Nodes (`ast_nodes.jsonl`)

**Purpose**: Raw AST node rows capturing file path, node type, qualname, spans, and parent relationships. Serves as the canonical source for GOID generation, analytics, and CFG/CallGraph builders.

**Origin**: `CodeIntel.enrich.ast_indexer.AstIndexer` (LibCST-based AST extraction). Written under `enriched/ast/ast_nodes.*`, duplicated to the doc root for convenience.

**Fields**

| Field          | Type    | Description |
|----------------|---------|-------------|
| `path`         | string  | Repo-relative file path. |
| `node_type`    | string  | LibCST node type (`FunctionDef`, `ClassDef`, etc.). |
| `name`         | string  | Symbol name (if applicable). |
| `qualname`     | string  | Fully qualified name. |
| `lineno`       | int     | Start line. |
| `end_lineno`   | int     | End line. |
| `col_offset`   | int     | Start column. |
| `end_col_offset` | int   | End column. |
| `parent_qualname` | string | Qualified name of the parent scope. |
| `decorators`   | list    | Decorator names (if any). |
| `docstring`    | string/null | Extracted docstring. |
| `hash`         | string  | Content hash (used for dedupe). |

## 10. AST Metrics Analytics (`hotspots.jsonl`, `typedness.jsonl`)

### 10a. Hotspots (`hotspots.jsonl`)

**Purpose**: Summaries of files considered “hotspots” based on change history, churn, and complexity metrics.

**Origin**: `CodeIntel.services.enrich.analytics.hotspots`. Combines git history (commits, authors) with AST metrics.

**Fields**

| Field             | Type   | Description |
|-------------------|--------|-------------|
| `rel_path`        | string | File path. |
| `commit_count`    | int    | Commits touching the file within the configured window. |
| `author_count`    | int    | Distinct authors in the same window. |
| `lines_added`     | int    | Net lines added. |
| `lines_deleted`   | int    | Net lines removed. |
| `complexity`      | float  | Derived from AST metrics. |
| `score`           | float  | Composite hotspot score used for ranking. |

### 10b. Typedness (`typedness.jsonl`)

**Purpose**: Tracks the extent of type annotation coverage per file. Provides signal for type migration work.

**Origin**: `CodeIntel.services.enrich.exports.write_typedness_output` during the pipeline’s analytics stage.

**Fields**

| Field              | Type   | Description |
|--------------------|--------|-------------|
| `path`             | string | File path. |
| `type_error_count` | int    | Max of Pyrefly/Pyright error counts for the file. |
| `annotation_ratio` | object | Ratios for parameter and return annotations (`{"params": float, "returns": float}`). |
| `untyped_defs`     | int    | Count of top-level functions lacking full annotations. |
| `overlay_needed`   | bool   | Whether a stub overlay is recommended. |

### 10c. Function Metrics (`function_metrics.*`)

**Purpose**: Per-function structural and complexity metrics keyed by GOID.

**Origin**: `CodeIntel.services.enrich.function_metrics` via `function-metrics` CLI command.

**Fields**

| Field                   | Type        | Description |
|-------------------------|-------------|-------------|
| `function_goid_h128`    | decimal     | GOID hash for the function. |
| `urn`                   | string      | GOID URN for the function. |
| `repo`                  | string      | Repository identifier used for GOID generation. |
| `commit`                | string      | Commit hash used for GOID generation. |
| `rel_path`              | string      | File path relative to repo root. |
| `language`              | string      | Language identifier (`python`). |
| `kind`                  | string      | `function` or `method` based on containing class. |
| `qualname`              | string      | Qualified name (class/function nesting). |
| `start_line`            | int         | Starting line number. |
| `end_line`              | int         | Ending line number. |
| `loc`                   | int         | Physical lines of code (`end_line - start_line + 1`). |
| `logical_loc`           | int         | Non-blank/non-comment lines within the span. |
| `param_count`           | int         | Total parameters including varargs/varkw. |
| `positional_params`     | int         | Positional-only + positional params. |
| `keyword_only_params`   | int         | Keyword-only parameter count. |
| `has_varargs`           | bool        | Whether `*args` is present. |
| `has_varkw`             | bool        | Whether `**kwargs` is present. |
| `is_async`              | bool        | True when defined with `async def`. |
| `is_generator`          | bool        | True when the body contains `yield` or `yield from`. |
| `return_count`          | int         | Number of `return` statements (excluding nested defs). |
| `yield_count`           | int         | Number of yield statements. |
| `raise_count`           | int         | Number of `raise` statements. |
| `cyclomatic_complexity` | int         | `1 + decision_points` found in control flow. |
| `max_nesting_depth`     | int         | Maximum control-structure nesting depth. |
| `stmt_count`            | int         | Count of top-level statements in the function body. |
| `decorator_count`       | int         | Decorator count. |
| `has_docstring`         | bool        | True when a docstring is present. |
| `complexity_bucket`     | string      | `low` (<=5), `medium` (<=10), or `high` (otherwise). |
| `created_at`            | string      | ISO-8601 timestamp for the analytics run. |

Join on `function_goid_h128` or `urn` with `goids.*` / `goid_crosswalk.*` /
graph tables (`cfg_blocks.*`, `dfg_edges.*`).

### 10d. Function Types (`function_types.*`)

**Purpose**: Per-function typedness and signature details keyed by GOID.

**Origin**: `CodeIntel.services.enrich.function_types` via `function-types` CLI command.

**Fields**

| Field                 | Type             | Description |
|-----------------------|------------------|-------------|
| `function_goid_h128`  | decimal          | GOID hash for the function. |
| `urn`                 | string           | GOID URN for the function. |
| `repo`                | string           | Repository identifier used for GOID generation. |
| `commit`              | string           | Commit hash used for GOID generation. |
| `rel_path`            | string           | File path relative to repo root. |
| `language`            | string           | Language identifier (`python`). |
| `kind`                | string           | `function` or `method`. |
| `qualname`            | string           | Qualified name (class/function nesting). |
| `start_line`          | int              | Starting line number. |
| `end_line`            | int              | Ending line number. |
| `total_params`        | int              | Parameters counted (excluding `self`/`cls`). |
| `annotated_params`    | int              | Parameters with annotations (excluding `self`/`cls`). |
| `unannotated_params`  | int              | Parameters lacking annotations. |
| `param_typed_ratio`   | float            | `annotated_params / total_params` (defaults to 1.0 when zero params). |
| `has_return_annotation` | bool           | True when a return annotation is present. |
| `return_type`         | string/null      | Return annotation text. |
| `return_type_source`  | string           | Source of return type (`annotation`/`unknown`). |
| `type_comment`        | string/null      | Type comment if available (currently null). |
| `param_types`         | object           | Map of parameter name → annotation string/null. |
| `fully_typed`         | bool             | True when all counted params and return are annotated. |
| `partial_typed`       | bool             | True when some (but not all) annotations are present. |
| `untyped`             | bool             | True when no annotations are present. |
| `typedness_bucket`    | string           | `typed` / `partial` / `untyped`. |
| `typedness_source`    | string           | `annotations`, `mixed`, or `unknown`. |
| `created_at`          | string           | ISO-8601 timestamp for the analytics run. |

Aggregates by `rel_path` align with `typedness.*` (per-file ratios); joins use
`function_goid_h128` or `urn` to connect with GOID/crosswalk and graph tables.

## 11. Tags Index (`tags_index.yaml`)

**Purpose**: Canonical index of tagging rules applied to the codebase—mapping files/modules to semantic tags (e.g., “infra”, “ml”, “api”).

**Origin**: `CodeIntel.enrich.tags` when the pipeline runs with tagging enabled. The YAML file summarizes rule definitions and their matched targets.

**Structure**

```yaml
- tag: "api"
  description: "Public HTTP entrypoints"
  includes:
    - "CodeIntel/app/routes/*"
  excludes:
    - "CodeIntel/app/routes/tests/*"
  matches:
    - "CodeIntel/app/routes/catalog_read.py"
    - ...
```

Each entry lists the resolved matches (files) for downstream consumption. Tools can use this to seed filtered analyses or to understand ownership boundaries.

## 12. SCIP Index (`index.scip.json`)

**Purpose**: Language-agnostic symbol graph produced by `scip-python`, capturing symbol definitions, references, and documentation. Serves as the canonical interface for cross-language tooling (search, jump-to-def, etc.).

**Origin**: Generated via `scip-python index ../src` and exported to JSON using `scip print --json`. Stored at `Document Output/index.scip.json`.

**Structure**: JSON array of documents, each containing:
- `relative_path`: File path.
- `occurrences`: Symbol occurrences with ranges, roles (definition/reference), and symbol IDs.
- `symbols`: Symbol metadata (signature, documentation, kind) keyed by SCIP symbol strings.

Consumers can join SCIP symbols with GOIDs via the crosswalk’s `scip_symbol` column.

## 13. Modules Registry (`modules.jsonl`)

**Purpose**: Mapping of module metadata produced during enrichment. Each row summarizes module path, repo metadata, and aggregated stats used by tagging and analytics.

**Origin**: `CodeIntel.enrich.pipeline_helpers.build_module_row`, emitted under `enriched/modules/modules.jsonl`.

**Fields**

| Field        | Type   | Description |
|--------------|--------|-------------|
| `module`     | string | Dotted module name. |
| `path`       | string | Repo-relative path. |
| `repo`       | string | Repo identifier. |
| `commit`     | string | Commit SHA. |
| `language`   | string | Language tag. |
| `tags`       | list   | Tags applied to the module. |
| `owners`     | list   | Owners inferred from tagging rules (if enabled). |

## 14. Repo Map (`repo_map.json`)

**Purpose**: High-level repository metadata containing module-to-path mappings, checksum info, and overlay configuration used by downstream tooling.

**Origin**: Emitted by `CodeIntel.services.enrich.scan` after scanning files. Lives at `enriched/repo_map.json`, duplicated to the doc root.

**Structure**

```json
{
  "repo": "CodeIntel",
  "commit": "deadbeef",
  "modules": {
    "CodeIntel.app.routes.catalog_read": "CodeIntel/app/routes/catalog_read.py",
    ...
  },
  "overlays": {...},
  "generated_at": "2024-01-01T00:00:00Z"
}
```

## 15. CST Nodes (`cst_nodes.jsonl`)

**Purpose**: Raw CST (Concrete Syntax Tree) nodes captured by the CST build pipeline. Complementary to AST nodes, preserving full syntax (including whitespace/comments) for tools that require exact source spans.

**Origin**: `CodeIntel.cst_build.cst_cli`. After emitting `cst_nodes.jsonl.gz`, the document generation script normalizes it to `cst_nodes.jsonl`.

**Fields**

| Field          | Type   | Description |
|----------------|--------|-------------|
| `path`         | string | Repo-relative file path. |
| `node_id`      | string | Stable identifier for the CST node. |
| `kind`         | string | Concrete syntax kind (Token, Module, FunctionDef, etc.). |
| `span`         | object | `{ "start": [line, col], "end": [line, col] }` capturing exact offsets. |
| `text_preview` | string | Snippet of the underlying source. |
| `parents`      | list   | Stack of parent node kinds/IDs. |
| `qnames`       | list   | Qualified names provided when applicable. |

The CST nodes can be joined with GOID crosswalk entries via future `cst_node_id` fields, enabling precise mapping between GOIDs and concrete syntax.

## 16. Import Graph Edges (`import_graph_edges.*`)

**Purpose**: Normalized module import graph capturing directed edges between modules along with degree metadata and cycle grouping.

**Origin**: Built from LibCST import metadata during `write_graph_outputs` via `CodeIntel.enrich.graph.io.write_import_edges`. Stored under `enriched/graphs/import_graph_edges.parquet` and mirrored to JSONL during document generation.

**Columns**

| Column        | Type   | Description |
|---------------|--------|-------------|
| `src_module`  | string | Source module name (derived from module path). |
| `dst_module`  | string | Target module imported by the source. |
| `src_fan_out` | int    | Out-degree of the source module. |
| `dst_fan_in`  | int    | In-degree of the destination module. |
| `cycle_group` | int    | Strongly connected component identifier from Tarjan SCC. |

Edges can be joined to modules (by `src_module`/`dst_module`) and to per-module analytics such as hotspots and typedness.

## 17. Symbol Use Edges (`symbol_use_edges.*`)

**Purpose**: SCIP-based def→use relationships linking where symbols are defined and where they are referenced across the codebase.

**Origin**: Derived from the SCIP index by `CodeIntel.uses_builder.build_use_graph` and exported via `write_uses_output`. Lives at `enriched/graphs/symbol_use_edges.parquet`.

**Columns**

| Column        | Type   | Description |
|---------------|--------|-------------|
| `symbol`      | string | SCIP symbol identifier. |
| `def_path`    | string | Repo-relative path where the symbol is defined. |
| `use_path`    | string | Repo-relative path where the symbol is referenced. |
| `same_file`   | bool   | True when definition and use share the same file. |
| `same_module` | bool   | True when both paths map to the same module. |

Join `symbol` to `index.scip.json` or GOID crosswalk SCIP symbols to enrich with semantic metadata.

## 18. Config Values (`config_values.*`)

**Purpose**: Per-key view of discovered configuration files, showing where each config key is referenced in code.

**Origin**: Flattened from the config indexer output (`index_config_files`) and enrichment references (`prepare_config_state`), exported via `write_config_output` to `enriched/analytics/config_values.parquet`.

**Columns**

| Column              | Type   | Description |
|---------------------|--------|-------------|
| `config_path`       | string | Repo-relative path to the config file. |
| `format`            | string | Detected format (`yaml`, `toml`, `json`, `ini`, `env`, `other`). |
| `key`               | string | Normalized config key path (e.g., `service.database.host`). |
| `reference_paths`   | list   | Sorted list of files referencing this config file/key. |
| `reference_modules` | list   | Corresponding module names for `reference_paths`. |
| `reference_count`   | int    | Number of distinct referencing files. |

Use the reference lists to trace blast radius for configuration changes and join to modules for richer metadata.

## 19. Static Diagnostics (`static_diagnostics.*`)

**Purpose**: Per-file counts of static type-checker errors aggregated from Pyrefly and Pyright runs.

**Origin**: Produced from `FileTypeSignals` captured during pipeline preparation and emitted via `write_static_diagnostics_output` to `enriched/analytics/static_diagnostics.parquet`.

**Columns**

| Column           | Type | Description |
|------------------|------|-------------|
| `rel_path`       | string | Repo-relative file path. |
| `pyrefly_errors` | int  | Error count reported by Pyrefly. |
| `pyright_errors` | int  | Error count reported by Pyright. |
| `total_errors`   | int  | Sum of Pyrefly and Pyright errors. |
| `has_errors`     | bool | True when `total_errors` > 0. |

These rows complement `typedness.jsonl` and hotspots to prioritize files with static analysis issues.

## 20. Line Coverage (`coverage_lines.*`)

**Purpose**: Fine-grained line-level coverage captured from coverage.py runs. Use to locate uncovered regions and to join line spans to GOIDs via `goid_xwalk`.

**Origin**: `CodeIntel.cli.enrich_analytics coverage-detailed` reads the `.coverage` database (with dynamic contexts enabled) and writes `enriched/analytics/coverage/coverage_lines.{parquet,jsonl}`.

**Columns**

| Column          | Type   | Description |
|-----------------|--------|-------------|
| `repo`          | string | Repository identifier from `repo_map.json`. |
| `commit`        | string | Commit SHA from `repo_map.json`. |
| `rel_path`      | string | Repo-relative path of the measured file. |
| `line`          | int    | 1-based line number. |
| `is_executable` | bool   | True when coverage marked the line as a statement. |
| `is_covered`    | bool   | True when the line executed at least once. |
| `hits`          | int    | Hit count best-effort (1 when covered, 0 otherwise). |
| `context_count` | int    | Number of distinct coverage contexts that touched the line. |
| `created_at`    | string | ISO-8601 timestamp of extraction. |

## 21. Function Coverage (`coverage_functions.*`)

**Purpose**: Per-function coverage derived by grouping `coverage_lines` over GOID spans. Primary join point for “is this function tested?” queries.

**Origin**: Aggregated by `coverage-detailed`, uses GOID spans from `goids.parquet` and lines from `coverage_lines.jsonl`.

**Columns**

| Column                | Type   | Description |
|-----------------------|--------|-------------|
| `function_goid_h128`  | string | GOID hash for the function/method. |
| `urn`                 | string | GOID URN. |
| `repo`                | string | Repository identifier. |
| `commit`              | string | Commit SHA. |
| `rel_path`            | string | Repo-relative path of the function. |
| `language`            | string | Language tag (`python`). |
| `kind`                | string | `function` or `method`. |
| `qualname`            | string | Qualified name within the file. |
| `start_line`          | int    | Function start line. |
| `end_line`            | int    | Function end line. |
| `executable_lines`    | int    | Executable statement count within the span. |
| `covered_lines`       | int    | Executed statement count within the span. |
| `coverage_ratio`      | float? | `covered_lines / executable_lines`, null when no executable lines. |
| `tested`              | bool   | True when `covered_lines > 0`. |
| `untested_reason`     | string | `no_executable_code`, `no_tests`, or empty string. |
| `created_at`          | string | ISO-8601 timestamp of extraction. |

## 22. Test Catalog (`test_catalog.*`)

**Purpose**: Canonical list of pytest tests (functions/methods/parametrized cases) with metadata needed for impact analysis.

**Origin**: `CodeIntel.cli.enrich_analytics test-analytics` reads a pytest JSON report (from `pytest-json-report`) and attaches GOID matches when available.

**Columns**

| Column           | Type        | Description |
|------------------|-------------|-------------|
| `test_id`        | string      | Pytest nodeid (e.g., `tests/test_app.py::TestFoo::test_bar[param]`). |
| `test_goid_h128` | string?     | GOID for the test callable when matched. |
| `urn`            | string?     | GOID URN when matched. |
| `repo`           | string      | Repository identifier. |
| `commit`         | string      | Commit SHA. |
| `rel_path`       | string      | Repo-relative path of the test file. |
| `qualname`       | string?     | Qualified name within the file. |
| `kind`           | string      | `parametrized_case` or `function`. |
| `status`         | string      | Test outcome (`passed`, `failed`, `error`, `skipped`, etc.). |
| `duration_ms`    | float       | Duration in milliseconds. |
| `markers`        | array       | Pytest markers/keywords. |
| `parametrized`   | bool        | True when nodeid includes parameters. |
| `flaky`          | bool        | True when `flaky` marker is present. |
| `created_at`     | string      | ISO-8601 timestamp of extraction. |

## 23. Test Coverage Edges (`test_coverage_edges.*`)

**Purpose**: Bipartite edges linking tests to the functions they executed, derived from coverage contexts.

**Origin**: `test-analytics` reads the `.coverage` database (with `dynamic_context = test_function`) and `test_catalog.jsonl`, producing `enriched/analytics/tests/test_coverage_edges.{parquet,jsonl}`.

**Columns**

| Column              | Type   | Description |
|---------------------|--------|-------------|
| `test_id`           | string | Pytest nodeid. |
| `test_goid_h128`    | string?| GOID for the test when matched. |
| `function_goid_h128`| string | Target function GOID exercised by the test. |
| `urn`               | string | Target GOID URN. |
| `repo`              | string | Repository identifier. |
| `commit`            | string | Commit SHA. |
| `rel_path`          | string | Target file path. |
| `qualname`          | string | Target function qualname. |
| `covered_lines`     | int    | Lines in the target executed by this test. |
| `executable_lines`  | int    | Total executable lines in the target span. |
| `coverage_ratio`    | float? | Per-edge coverage ratio. |
| `last_status`       | string | Status of the test in the referenced run. |
| `created_at`        | string | ISO-8601 timestamp of extraction. |

## 24. GOID Risk Factors (`goid_risk_factors.*`)

**Purpose**: Aggregated risk signals per function GOID combining coverage, tests, complexity, typedness, hotspots, and static diagnostics.

**Origin**: `CodeIntel.cli.enrich_analytics risk-factors` joins analytics tables under `enriched/analytics/` (function metrics/types, coverage, tests, hotspots, typedness, static_diagnostics).

**Columns (selected)**

| Column               | Type    | Description |
|----------------------|---------|-------------|
| `function_goid_h128` | string  | GOID hash. |
| `urn`                | string  | GOID URN. |
| `repo`               | string  | Repository identifier. |
| `commit`             | string  | Commit SHA. |
| `rel_path`           | string  | File path. |
| `language`           | string  | Language tag. |
| `kind`               | string  | Entity kind (`function`/`method`). |
| `qualname`           | string  | Qualified name. |
| `loc`                | int?    | Physical lines of code (from function_metrics). |
| `logical_loc`        | int?    | Logical LOC. |
| `cyclomatic_complexity` | int? | Cyclomatic complexity. |
| `complexity_bucket`  | string? | Derived complexity bucket. |
| `typedness_bucket`   | string? | Annotation bucket (`typed`/`partial`/`untyped`). |
| `typedness_source`   | string? | Annotation source metadata. |
| `hotspot_score`      | float?  | File-level hotspot score. |
| `file_typed_ratio`   | float?  | File-level annotation ratio. |
| `static_error_count` | int?    | File-level static error count. |
| `has_static_errors`  | bool?   | True when static errors are present. |
| `executable_lines`   | int?    | Executable line count (coverage). |
| `covered_lines`      | int?    | Covered line count (coverage). |
| `coverage_ratio`     | float?  | Coverage ratio. |
| `tested`             | bool?   | True when covered lines > 0. |
| `test_count`         | int     | Distinct tests touching the function. |
| `failing_test_count` | int     | Distinct failing tests. |
| `last_test_status`   | string  | Last test status seen (`all_passing`/`some_failing`/`untested`/`unknown`). |
| `risk_score`         | float   | Heuristic 0–1 risk score. |
| `risk_level`         | string  | Risk bucket (`low`/`medium`/`high`). |
| `tags`               | list    | Tags from module metadata. |
| `owners`             | list    | Owners when available. |
| `created_at`         | string  | ISO-8601 timestamp of aggregation. |
