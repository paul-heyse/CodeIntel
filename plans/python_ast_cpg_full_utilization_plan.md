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
- [ ] Confirm byte-span line index is stored for every ingested file.
- [ ] Add a run meta table for compiler and environment facts:
  - python version, magic number, optimize flags, dont_inherit.
- [x] Add ingestion options toggles for new extractors (symtable/dis/inspect).
- [ ] Decide stable-ish id strategy for scopes and code units (span anchored).

### Phase 1: AST enrichment (native AST first-class)
- [ ] Expand AST extraction payload in src/codeintel/ingestion/compute/ast_extract.py:
  - byte spans for all nodes
  - ctx/type_comment/name/import/constant extras
  - include FunctionDef/ClassDef names and arg names
- [ ] Extend AST -> syntax merge in src/codeintel/ingestion/compute/cst_extract.py:
  - attach expanded AST payload to core.syntax_nodes extras_json
- [ ] Add AST def/use facts:
  - Name(Store/Load), Attribute, Subscript, Import, Param defs
  - align spans and ids to syntax nodes where possible
- [ ] Add tests for AST span correctness and payload completeness.

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
- [ ] Add schema definitions for symtable tables.
- [ ] Add validation checks:
  - co_freevars vs symtable frees
  - declared_global/nonlocal resolution edges exist

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
- [ ] Add tests for:
  - CFG edges (if/loop/try/with)
  - exception table parsing
  - label mapping and block boundaries

### Phase 4: DFG wiring (bytecode + symtable bindings)
- [ ] Map code_unit_id -> scope_id using AST anchors.
- [ ] Resolve DEF/USE events to bindings via symtable resolve_binding.
- [ ] Run reaching-defs over py_bc_cfg_edges and emit REACHES edges.
- [ ] Emit DEFINES_BINDING/USES_BINDING edges for bytecode events.
- [ ] Add DFG sanity checks:
  - LOAD_FAST -> local/param binding
  - LOAD_DEREF -> resolved FREE/NONLOCAL binding
  - LOAD_GLOBAL -> module binding

### Phase 5: Inspect overlay (optional, isolated)
- [x] Add inspect extraction worker (in-process, static-safe allowlist):
  - getmembers_static, get_annotations(eval_str=False)
  - signature(raw/wrapped), unwrap hops, source anchors
- [ ] Isolate inspect extraction in a subprocess with timeouts and budgets.
- [x] Emit inspect tables (objects, members, unwrap, signatures, annotations, source).
- [ ] Emit derived anchors to AST/SCIP.
- [ ] Add optional callsite param wiring using inspect signatures.
- [x] Add budgets and toggles in ingestion options to disable by default.

### Phase 6: CPG projection and schema wiring
- [ ] Update schema registry to include py_sym_* and py_bc_* tables.
- [ ] Add CPG nodes/edges:
  - SCOPE, BINDING, BC_CODE_UNIT, BC_INSTR, CFG_BLOCK
  - edges: OWNS_SCOPE, PARENT_SCOPE, DECLARES, RESOLVES_TO, CFG_*, REACHES
- [ ] Bridge BINDING <-> SCIP SYMBOL via AST def/use anchors.
- [ ] Ensure node and edge ids are stable and byte-span anchored.

### Phase 7: Validation, fixtures, and drift gates
- [ ] Add micro-fixture suite:
  - global/nonlocal/free, nested functions, comprehensions
  - try/except/finally/with CFG edges
  - decorators and wrappers (inspect)
- [ ] Add a run-level quality report:
  - instruction span anchoring rate
  - symtable anchor rate
  - CFG reachability sanity
  - DFG def/use resolution coverage
- [ ] Gate regressions with targeted pytest subsets and segmented runs.

### Phase 8: Performance and incrementalization
- [ ] Add incremental caching for code unit compilation where possible.
- [ ] Bound memory by streaming row buffers and lazy materialization.
- [ ] Add per-file size thresholds and timeouts for dis/inspect.
- [ ] Add configurable parallelism for dis extraction by file.

### Phase 9: Rollout and monitoring
- [ ] Ship symtable tables first (low risk, high value).
- [ ] Ship bytecode CFG next with validation gates.
- [ ] Enable DFG reachability in staged rollout.
- [ ] Enable inspect overlay only for curated allowlists.

## File-by-File Change List (Initial Pass)

### New ingestion steps
- [x] src/codeintel/ingestion/compute/symtable_extract.py
  - symtable extraction and derived bindings/resolution edges
- [x] src/codeintel/ingestion/compute/dis_extract.py
  - dis code units, instructions, exception table, derived CFG/DFG tables
- [x] src/codeintel/ingestion/compute/inspect_extract.py
  - optional inspect extraction (static-safe allowlist, in-process)

### AST/CST enhancements
- [ ] src/codeintel/ingestion/compute/ast_extract.py
  - expand payloads, byte spans, AST def/use facts
- [ ] src/codeintel/ingestion/compute/cst_extract.py
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
- [ ] src/codeintel/build/schemas/service.py
  - add py_sym_* and py_bc_* tables
- [ ] src/codeintel/storage/schema/arrow_schema.py
  - table definitions for new datasets
- [ ] src/codeintel/core/schemas/arrow_metadata.py
  - metadata to describe new datasets

### CPG projection
- [ ] src/codeintel/build/graphs/*
  - add CPG nodes/edges for scopes, bindings, CFG/DFG
- [ ] src/codeintel/build/graphs/validation/*
  - add validation gates for symtable/dis/dfg invariants

### Tests and fixtures
- [ ] tests/ingestion/test_symtable_extract.py
- [ ] tests/ingestion/test_dis_extract_cfg.py
- [ ] tests/ingestion/test_dis_extract_defuse.py
- [ ] tests/ingestion/test_ast_span_joins.py
- [ ] tests/ingestion/test_inspect_overlay.py (optional, sandboxed)

## Remaining Work Checklist (Detailed)

### AST/CST enrichment
- [ ] Expand AST extraction payload in `src/codeintel/ingestion/compute/ast_extract.py`
  - byte spans for all nodes
  - ctx/type_comment/name/import/constant extras
  - include FunctionDef/ClassDef names and arg names
- [ ] Merge expanded AST payload into syntax nodes in `src/codeintel/ingestion/compute/cst_extract.py`
  - attach AST payload to core.syntax_nodes extras_json
  - ensure byte-span joins stay deterministic
- [ ] Emit AST def/use facts
  - Name(Store/Load), Attribute, Subscript, Import, Param defs
  - align spans and ids to syntax nodes where possible
- [ ] Add AST span correctness and payload completeness tests

### Schema + registry + storage wiring
- [ ] Register py_sym_*, py_bc_*, py_inspect_* tables in schema service
- [ ] Define Arrow schemas and metadata for new datasets
- [ ] Add row model definitions and serialization coverage
- [ ] Update output registry and warehouse mapping for new datasets

### DFG wiring (bytecode + symtable bindings)
- [ ] Map code_unit_id -> scope_id using AST anchors
- [ ] Resolve DEF/USE events to bindings via symtable resolution
- [ ] Run reaching-defs over py_bc_cfg_edges and emit REACHES edges
- [ ] Emit DEFINES_BINDING/USES_BINDING edges for bytecode events
- [ ] Add DFG sanity checks (LOAD_FAST/LOAD_DEREF/LOAD_GLOBAL mapping)

### Inspect overlay hardening
- [ ] Isolate inspect extraction in a subprocess (avoid in-process import side effects)
- [ ] Enforce timeouts and memory budgets per module and per run
- [ ] Emit derived anchors to AST/SCIP and confidence metadata
- [ ] Add optional callsite param wiring using inspect signatures
- [ ] Add allowlist validation and error reporting for blocked imports

### CPG projection and validation wiring
- [ ] Add nodes/edges for scopes, bindings, bytecode code units, CFG blocks
- [ ] Bridge bindings to SCIP symbols via AST def/use anchors
- [ ] Add validation gates for symtable/dis/dfg invariants
- [ ] Ensure node/edge ids remain span anchored and stable

### Tests, fixtures, and drift gates
- [ ] Add micro-fixture suite (global/nonlocal/free, nested, comprehensions)
- [ ] Add CFG fixtures for try/except/finally/with
- [ ] Add tests for symtable/dis/inspect extraction outputs
- [ ] Add quality report metrics (anchoring rates, DFG coverage)
- [ ] Segment pytest runs by affected directories

### Performance + rollout
- [ ] Add incremental caching for code unit compilation
- [ ] Bound memory by streaming row buffers and lazy materialization
- [ ] Add per-file size thresholds and timeouts for dis/inspect
- [ ] Add configurable parallelism for dis extraction
- [ ] Stage rollout: symtable -> bytecode CFG -> DFG -> inspect overlay

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
