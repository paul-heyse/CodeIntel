# Python AST + Bytecode + Symtable + Inspect CPG Implementation Plan (Detailed)

## Context
We want to maximize Python native AST functionality to produce richer, more accurate CPGs.
This plan expands the current AST/CST ingestion and adds symtable, dis bytecode, and
optional inspect overlays with deterministic contracts and validation gates.

Reference anchors:
- docs/python_library_reference/python_ast_libraries_and_cpg_construction.md
- docs/CPG_construction.md
- src/codeintel/ingestion/compute/cst_extract.py
- src/codeintel/ingestion/compute/ast_extract.py
- src/codeintel/build/hamilton/native/ingestion/extraction_targets.py
- src/codeintel/build/hamilton/native/options/ingestion.py

## Goals
- Make CPython AST the canonical semantic layer for definitions, references, and spans.
- Add compiler-grade scope/binding resolution via symtable.
- Add bytecode-derived CFG/DFG with exception-aware control flow from dis.
- Provide optional inspect-based runtime overlays for signatures and wrappers.
- Keep all joins byte-span anchored and deterministic across runs.
- Add validation gates that detect binder/CFG/DFG drift early.

## Non-goals
- No runtime instrumentation or tracing beyond optional inspect (no exec/eval for analysis).
- No replacement of SCIP; SCIP remains the symbol identity layer.
- No single-pass "everything at once" rollout; changes are staged and gated.

## Target Architecture (Layered CPG)
- Syntax layer: CST + AST nodes and edges (byte spans are canonical).
- Symbol layer: SCIP symbols + occurrences; symtable scopes/bindings as lexical truth.
- Bytecode layer: dis code units, instructions, exception table, blocks, CFG edges.
- Dataflow layer: def/use + reaching defs from bytecode and symtable bindings.
- Optional runtime overlay: inspect facts for signatures, wrappers, annotations.
- Unified CPG: nodes and edges welded by span and symbol/binding anchors.

## Data Contracts (Summary)
### AST/CST (existing, expanded)
- core.ast_nodes: add byte spans, ctx/type_comment/name/import/constant extras.
- core.ast_metrics: keep, but ensure AST spans and node ids are stable.
- core.syntax_nodes: keep, but merge AST payloads into extras_json.
- core.syntax_defs/refs/calls/params/imports: keep; align to AST spans where possible.

### Symtable (new)
- py_sym_scopes: scope inventory with stable-ish scope_id + parent links.
- py_sym_symbols: per-scope symbol flags (local/global/free/nonlocal/imported/etc).
- py_sym_scope_edges: parent-child scope edges.
- py_sym_namespace_edges: name binds to namespace scopes.
- py_sym_function_partitions: parameters/locals/globals/nonlocals/frees.
- py_sym_bindings (derived): binding slots per scope.
- py_sym_resolution_edges (derived): FREE/NONLOCAL/GLOBAL resolution edges.

### Bytecode (new)
- py_bc_code_units: code object inventory and metadata.
- py_bc_instructions: instruction rows with baseopname, offsets, cache info, positions.
- py_bc_exception_table: parsed exception entries.
- py_bc_blocks (derived): basic blocks.
- py_bc_cfg_edges (derived): CFG edges (normal + exception).
- py_bc_defuse_events (derived): DEF/USE/KILL events keyed by baseopname.

### Inspect (optional)
- py_inspect_objects: runtime object inventory.
- py_inspect_members_static: descriptor-preserving member surface.
- py_inspect_unwrap_hops: wrapper chains and signature stop points.
- py_inspect_signatures + py_inspect_signature_params: signature facts.
- py_inspect_annotations_kv: safe annotations (eval_str=False).
- py_inspect_source: source anchoring for objects.
- py_inspect_object_anchors (derived): object -> AST def.
- py_inspect_object_symbol_map (derived): object -> SCIP symbol.

## Implementation Phases (with checklists)

### Phase 0: Baseline and guardrails
- [x] Confirm byte-span line index is stored for every ingested file.
- [x] Add a run meta table for compiler and environment facts:
  - [x] Define schema (python_version, magic_number, optimize, dont_inherit, flags).
  - [x] Emit run meta rows during ingestion (repo, commit, run_id).
  - [x] Register table in schema registry + row models.
- [x] Add ingestion options toggles for new extractors (symtable/dis/inspect).
- [x] Decide stable-ish id strategy for scopes and code units (span anchored).

### Phase 1: AST enrichment (native AST first-class)
- [x] Expand AST extraction payload in src/codeintel/ingestion/compute/ast_extract.py:
  - byte spans for all nodes
  - ctx/type_comment/name/import/constant extras
  - include FunctionDef/ClassDef names and arg names
- [x] Extend AST -> syntax merge in src/codeintel/ingestion/compute/cst_extract.py:
  - attach expanded AST payload to core.syntax_nodes extras_json
- [x] Add AST def/use facts:
  - Name(Store/Load), Attribute, Subscript, Import, Param defs
  - align spans and ids to syntax nodes where possible
- [x] Add tests for AST span correctness and payload completeness.

### Phase 2: Symtable extraction (scopes + bindings as truth)
- [x] Add src/codeintel/ingestion/compute/symtable_extract.py:
  - extract py_sym_scopes, py_sym_symbols, py_sym_scope_edges
  - emit py_sym_namespace_edges and py_sym_function_partitions
- [x] Add anchoring to AST spans:
  - MODULE -> file span
  - FUNCTION/CLASS -> AST def node by name + lineno
  - TYPE_* scopes -> owning def or annotation span (low confidence if needed)
- [x] Derive py_sym_bindings and py_sym_resolution_edges deterministically.
- [x] Add new ingestion target for symtable tables.
- [x] Add schema definitions for symtable tables.
- [x] Add validation checks:
  - [x] Compare symtable freevars vs code object co_freevars.
  - [x] Validate declared_global/nonlocal resolution edges exist.
  - [x] Surface failures in guardrails/graph validation.

### Phase 3: Bytecode extraction (dis -> CFG/DFG substrate)
- [x] Add src/codeintel/ingestion/compute/dis_extract.py:
  - compile with deterministic flags
  - enumerate all nested code objects
  - emit py_bc_code_units and py_bc_instructions
  - parse exception table into py_bc_exception_table
- [x] Add derived tables:
  - py_bc_blocks (block boundaries)
  - py_bc_cfg_edges (normal + exception edges)
  - py_bc_defuse_events (baseopname classifier)
- [x] Anchor instructions to byte spans via Instruction.positions + line index.
- [x] Add tests for:
  - CFG edges (if/loop/try/with)
  - exception table parsing
  - label mapping and block boundaries

### Phase 4: DFG wiring (bytecode + symtable bindings)
- [x] Map code_unit_id -> scope_id using AST anchors.
- [x] Resolve DEF/USE events to bindings via symtable resolve_binding.
- [x] Run reaching-defs over py_bc_cfg_edges and emit REACHES edges.
- [x] Emit DEFINES_BINDING/USES_BINDING edges for bytecode events.
- [x] Emit bytecode instruction -> AST/syntax anchors for stable joins.
- [x] Add stack-effect DFG modeling for transient values (stack/register-like).
- [x] Emit memory access edges for attr/subscript/global operations.
- [x] Project bytecode CALL opcodes into explicit callsite nodes.
- [x] Add DFG sanity checks:
  - [x] Validate LOAD_FAST -> local/param binding edges.
  - [x] Validate LOAD_DEREF -> FREE/NONLOCAL binding edges.
  - [x] Validate LOAD_GLOBAL -> module/global binding edges.

### Phase 5: Inspect overlay (optional, isolated)
- [x] Add inspect extraction worker (in-process, static-safe allowlist):
  - getmembers_static, get_annotations(eval_str=False)
  - signature(raw/wrapped), unwrap hops, source anchors
- [x] Isolate inspect extraction in a subprocess with timeouts and budgets.
- [x] Emit inspect tables (objects, members, unwrap, signatures, annotations, source).
- [x] Emit derived anchors to AST/SCIP.
- [x] Add optional callsite param wiring using inspect signatures.
- [x] Project unwrap hops into WRAPS/DECORATES edges in the CPG.
- [x] Add class/descriptor topology extraction and graph projection.
- [x] Add frame/traceback extraction with bytecode instruction anchoring.
- [x] Add generator/coroutine/asyncgen state + locals extraction.
- [x] Add BoundArguments-based call binding edges for runtime callsites.
- [x] Add budgets and toggles in ingestion options to disable by default.

### Phase 6: CPG projection and schema wiring
- [x] Update schema registry to include py_sym_* and py_bc_* tables.
- [x] Add CPG nodes/edges:
  - SCOPE, BINDING, BC_CODE_UNIT, BC_INSTR, CFG_BLOCK
  - edges: OWNS_SCOPE, PARENT_SCOPE, DECLARES, RESOLVES_TO, CFG_*, REACHES
- [x] Bridge BINDING <-> SCIP SYMBOL via AST def/use anchors.
- [x] Project py_sym_namespace_edges into namespace binding edges.
- [x] Ensure node and edge ids are stable and byte-span anchored.

### Phase 7: Validation, fixtures, and drift gates
- [x] Add tests for symtable/dis/inspect extraction outputs.
- [x] Add micro-fixture suite:
  - [x] global/nonlocal/free symtable coverage
  - [x] nested functions + comprehensions
  - [x] try/except/finally/with CFG edges
  - [x] decorators and wrappers (inspect)
  - [x] match/case + async control flow coverage
- [x] Add a run-level quality report:
  - [x] instruction span anchoring rate
  - [x] symtable anchor rate
  - [x] CFG reachability sanity
  - [x] DFG def/use resolution coverage
- [x] Gate regressions with targeted pytest subsets and segmented runs.
  - [x] Add `tools/pytest_gate.py` helper for targeted + segmented pytest runs.
  - [x] Default targeted subset covers symtable/dis/ast/inspect ingestion tests.

### Phase 8: Performance and incrementalization
- [x] Add incremental caching for code unit compilation where possible.
- [x] Bound memory by streaming row buffers and lazy materialization.
  - [x] Replace ColumnarRowBuffer with ColumnarBatchCollector for CST/symtable/dis/inspect extractors.
  - [x] Flush collectors per module to emit per-file RecordBatches (bounded peak memory).
  - [x] Expose batch_size options for ingestion extractors (default to DEFAULT_ARROW_BATCH_SIZE).
  - [x] Align streamed readers to contracts via align_reader_to_contract; unify schemas when needed.
  - [x] Avoid list(reader)/table materialization; prefer LazyFrameStream.to_reader and streaming readers.
  - [x] Use RecordBatchReader.from_stream / __arrow_c_stream__ for iterable batch sources.
  - [x] Enforce single-consume semantics for RecordBatchReader-backed datasets (no double scans).
  - [x] Update ingestion tests to validate RecordBatchReader outputs (not ColumnarRows).
  - [x] Refactor analytics counters to use streamed batches (avoid tabular_to_frame collect()).
- [x] Add per-file size thresholds and timeouts for dis/inspect. (Per-module time budgets are warn-only.)
- [x] Add configurable parallelism for dis extraction by file.

### Phase 9: Rollout and monitoring
- [x] Ship symtable tables first (low risk, high value).
- [x] Ship bytecode CFG next with validation gates.
- [x] Enable DFG reachability in staged rollout.
- [x] Enable inspect overlay only for curated allowlists.

## File-by-File Change List (Initial Pass)

### New ingestion steps
- [x] src/codeintel/ingestion/compute/symtable_extract.py
  - symtable extraction and derived bindings/resolution edges
- [x] src/codeintel/ingestion/compute/dis_extract.py
  - dis code units, instructions, exception table, derived CFG/DFG tables
- [x] src/codeintel/ingestion/compute/inspect_extract.py
  - optional inspect extraction (static-safe allowlist, in-process)

### AST/CST enhancements
- [x] src/codeintel/ingestion/compute/ast_extract.py
  - expand payloads, byte spans, AST def/use facts
- [x] src/codeintel/ingestion/compute/cst_extract.py
  - merge expanded AST payloads into syntax nodes
  - ensure AST span join logic uses byte offsets consistently

### Ingestion targets and options
- [x] src/codeintel/build/hamilton/native/ingestion/extraction_targets.py
  - add new targets for symtable, bytecode, inspect
- [x] src/codeintel/build/hamilton/native/options/ingestion.py
  - add options for new targets and safety budgets
- [x] src/codeintel/core/registry/dag_output_inventory.yaml
  - register new ingestion outputs for symtable/bytecode/inspect

### Schema and storage wiring
- [x] src/codeintel/core/schemas/output_registry.py
  - add py_sym_*, py_bc_*, py_inspect_* table definitions
- [x] src/codeintel/core/schemas/generated_rows/core.py
  - add row models for new datasets
- [x] src/codeintel/storage/gateway/registry_generated.py
  - regenerate registry mappings for new tables

### CPG projection
- [x] src/codeintel/build/hamilton/native/graphs/cpg.py
  - add CPG nodes/edges for scopes, bindings, bytecode, inspect overlays
- [x] src/codeintel/build/hamilton/native/analytics/graph_validation.py
  - add validation gates for symtable/dis/dfg invariants

### Tests and fixtures
- [x] tests/ingestion/test_symtable_extract.py
- [x] tests/ingestion/test_dis_extract_cfg.py
- [x] tests/ingestion/test_dis_extract_defuse.py
- [x] tests/ingestion/test_ast_span_joins.py
- [x] tests/ingestion/test_inspect_overlay.py (optional, sandboxed)

## Remaining Work Checklist (Detailed)

### Baseline and guardrails
- [x] Add run meta table for compiler and environment facts
  - [x] Define schema (python_version, magic_number, optimize, dont_inherit, flags)
  - [x] Emit run meta rows during ingestion (repo, commit, run_id)
  - [x] Register table in schema registry + row models

### AST/CST enrichment
- [x] Expand AST extraction payload in `src/codeintel/ingestion/compute/ast_extract.py`
  - byte spans for all nodes
  - ctx/type_comment/name/import/constant extras
  - include FunctionDef/ClassDef names and arg names
- [x] Merge expanded AST payload into syntax nodes in `src/codeintel/ingestion/compute/cst_extract.py`
  - attach AST payload to core.syntax_nodes extras_json
  - ensure byte-span joins stay deterministic
- [x] Emit AST def/use facts
  - Name(Store/Load), Attribute, Subscript, Import, Param defs
  - align spans and ids to syntax nodes where possible
- [x] Add AST span correctness and payload completeness tests

### Schema + registry + storage wiring
- [x] Register py_sym_*, py_bc_*, py_inspect_* tables in schema service
- [x] Define Arrow schemas and metadata for new datasets
- [x] Add row model definitions and serialization coverage
- [x] Update output registry and warehouse mapping for new datasets

### DFG wiring (bytecode + symtable bindings)
- [x] Map code_unit_id -> scope_id using AST anchors
- [x] Resolve DEF/USE events to bindings via symtable resolution
- [x] Run reaching-defs over py_bc_cfg_edges and emit REACHES edges
- [x] Emit DEFINES_BINDING/USES_BINDING edges for bytecode events
- [x] Emit bytecode instruction -> AST/syntax anchors
  - Graph mapping: add BYTECODE_ANCHOR (BC_INSTR -> AST_NODE) and/or BYTECODE_COVERS edges
- [x] Add stack-effect DFG modeling for transient values
  - Graph mapping: emit STACK_DEF/STACK_USE edges from BC_INSTR nodes or create VALUE nodes
- [x] Emit memory access edges for attr/subscript/global operations
  - Graph mapping: emit READS_ATTR/WRITES_ATTR, READS_SUBSCR/WRITES_SUBSCR, READS_GLOBAL edges
- [x] Project bytecode CALL opcodes into explicit callsites
  - Graph mapping: create CALLSITE nodes, add CALLS edges to callee symbols, link to syntax call nodes
- [x] Add DFG sanity checks
  - [x] Validate LOAD_FAST -> local/param binding edges
  - [x] Validate LOAD_DEREF -> FREE/NONLOCAL binding edges
  - [x] Validate LOAD_GLOBAL -> module/global binding edges

### Symtable advanced scope mapping
- [x] Anchor ANNOTATION/TYPE_ALIAS/TYPE_PARAMETERS/TYPE_VARIABLE scopes to AST spans
  - [x] ANNOTATION scope anchor
  - [x] TYPE_ALIAS scope anchor
  - [x] TYPE_VARIABLE scope anchor
  - [x] TYPE_PARAMETERS scope anchor
  - Graph mapping: ensure OWNS_SCOPE edges exist for type/meta scopes with confidence metadata
- [x] Project py_sym_namespace_edges into CPG edges
  - Graph mapping: emit BINDS_NAMESPACE/DECLARES_NAMESPACE edges from binding -> scope

### Inspect overlay hardening
- [x] Isolate inspect extraction in a subprocess (avoid in-process import side effects)
  - [x] Subprocess runner + IPC payload format
  - [x] Timeout handling + cancellation
  - [x] Error propagation + diagnostics
- [x] Enforce timeouts and memory budgets per module and per run (per-module budgets warn-only)
  - [x] Per-module wall-clock budget (warn-only)
  - [x] Per-run memory ceiling
  - [x] Budget enforcement reporting
- [x] Emit derived anchors to AST/SCIP and confidence metadata
- [x] Add optional callsite param wiring using inspect signatures
- [x] Add allowlist validation and error reporting for blocked imports
  - [x] Warn on missing allowlist modules + summarize skipped modules

### Inspect runtime topology and call binding
- [x] Project unwrap hops into graph edges
  - Graph mapping: emit WRAPS/DECORATES edges between INSPECT_OBJECT nodes
- [x] Add class/descriptor topology extraction (MRO/classify_class_attrs/getattr_static)
  - Graph mapping: emit INHERITS, DECLARES_ATTR, OVERRIDES, DESCRIPTOR edges
- [x] Add frame/traceback extraction with instruction anchoring
  - Graph mapping: emit FRAME_AT_INSTR/TRACEBACK_AT_INSTR edges to BC_INSTR nodes
- [x] Add generator/coroutine/asyncgen state + locals extraction
  - Graph mapping: emit HAS_STATE edges from runtime object to state nodes or edges with props
- [x] Add BoundArguments-based call binding edges
  - Graph mapping: emit BINDS_ARG edges (callsite arg -> signature param) with confidence

### CPG projection and validation wiring
- [x] Add nodes/edges for scopes, bindings, bytecode code units, CFG blocks
- [x] Bridge bindings to SCIP symbols via AST def/use anchors
- [x] Add validation gates for symtable/dis/dfg invariants
- [x] Ensure node/edge ids remain span anchored and stable

### Tests, fixtures, and drift gates
- [x] Add micro-fixture suite
  - [x] global/nonlocal/free coverage
  - [x] nested functions + comprehensions
  - [x] decorators/wrappers coverage
  - [x] match/case + async control flow coverage
- [x] Add CFG fixtures for try/except/finally/with
- [x] Add tests for symtable/dis/inspect extraction outputs
- [x] Add quality report metrics (anchoring rates, DFG coverage)
- [x] Segment pytest runs by affected directories
  - [x] Default segment list: ingestion, build, graphs, storage, serving, runtime, analytics

### Performance + rollout
- [x] Add incremental caching for code unit compilation
  - [x] Cache key = (repo, commit, rel_path, python_version, flags)
  - [x] Reuse compiled code objects across runs
- [x] Bound memory by streaming row buffers and lazy materialization
  - [x] Stream ColumnarBatchCollector flushes per file for CST/symtable/dis/inspect
  - [x] Emit RecordBatchReader streams directly from extractors (no intermediate tables)
  - [x] Align streamed readers to contracts (align_reader_to_contract + optional unify_schemas)
  - [x] Replace arrow_table_from_lazyframe with LazyFrameStream.to_reader for CPG nodes/edges
  - [x] Avoid list(reader) and pa.Table.from_batches(list(reader)) in ingestion/graph paths
  - [x] Document and enforce single-consume semantics for RecordBatchReader datasets
  - [x] Update ingestion tests to consume RecordBatchReader outputs
  - [x] Refactor py_cpg_quality_report metrics to stream counts (no eager collect)
- [x] Add per-file size thresholds and timeouts for dis/inspect (per-module timeouts warn-only)
  - [x] Size cutoff (bytes/lines) per module
  - [x] Per-file timeout guard (warn-only)
- [x] Add configurable parallelism for dis extraction
  - [x] Per-run worker limit with backpressure
  - [x] Deterministic scheduling and ordering
- [x] Stage rollout: symtable -> bytecode CFG -> DFG -> inspect overlay
  - [x] Default-on symtable tables (ingestion.symtable.enable = true)
  - [x] Bytecode CFG behind flag + validation gate (ingestion.bytecode.include_cfg)
  - [x] DFG reachability behind flag (graph.cpg.enable_reaches or equivalent)
  - [x] Inspect overlay allowlist-only (ingestion.inspect.enable + allowlist)
  - [x] Wire gating into build config (codeintel.build.toml / profile yaml)
  - [x] Add rollout checklist and notes for profile sequencing (full vs fast)

## Acceptance Criteria
- AST nodes and syntax nodes have consistent byte spans and stable ids.
- Symtable extraction produces scope/binding nodes with resolution edges.
- Bytecode extraction yields CFG edges that match golden fixtures.
- DFG def/use edges resolve to symtable bindings and pass invariants.
- Optional inspect overlay anchors runtime objects to AST/SCIP with clear confidence.

## Validation Gates (Must Pass)
- Symtable freevars match code object co_freevars for functions.
- Every nonlocal/global reference has a resolution edge.
- Every CFG block has deterministic boundaries and valid successors.
- Bytecode instructions anchor to syntax spans at high coverage for non-synthetic ops.
- No unresolved bindings without explicit "unknown" markers.

## Risks and Mitigations
- Bytecode offsets drift across Python versions.
  - Mitigation: stable ids use spans + baseopname; keep physical ids separately.
- Inspect can execute code during import.
  - Mitigation: isolate in subprocess; static-safe APIs; allowlist only.
- Symtable line-based anchoring is ambiguous.
  - Mitigation: prefer AST name + lineno match; store confidence + reason.

## Rollout Notes
- Start with symtable tables and scope/binding edges (low-risk, high value).
- Add bytecode CFG with exception edges next; verify with fixtures.
- Add DFG reachability once CFG is stable and binding resolution is trusted.
- Add inspect overlay last and keep disabled by default.

## Rollout Checklist (Profiles)
- [x] Build config toggles are present in `codeintel.build.toml.example`.
- [x] Full profile enables symtable + bytecode CFG/DFG reachability; inspect remains allowlist-only.
- [x] Fast profile disables bytecode/DFG/inspect by default, keeps symtable on.
- [x] Documented gate runner: `uv run python -m tools.pytest_gate`.
