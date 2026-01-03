# CPG Stage-F Call Wiring: Binding/Role Columns + Descriptor Semantics Plan

## Context
We want to extend Stage-F call wiring to cover the additional mappings described in:
- `docs/CPG_additional_algorithms.md`
- `docs/CPG_construction.md`

The current call wiring in `src/codeintel/build/hamilton/native/graphs/call_wiring.py` resolves
explicit call sites by welding callee spans to SCIP occurrences. It does not yet model:
- constructor + `__init__` dual targets
- bound vs unbound method calls
- `@classmethod` / `@staticmethod` / `@property` binding semantics
- implicit descriptor get/set calls
- augmented assignment lowering for descriptors

This plan adds those semantics and **elevates binding/role data to first-class columns**
instead of `extras_json`.

## Goals
- Add deterministic, schema-first call wiring for method binding and descriptor protocol.
- Promote binding/role fields to top-level columns in call wiring tables.
- Preserve existing SCIP-based resolution for explicit calls.
- Keep callsite identity stable and deterministic for implicit calls (desugaring).
- Provide golden fixtures for descriptor reads/writes and augmented assignment.

## Functional Scope Items (Stage-F Call Wiring)
- classmethod/staticmethod binding
- bound vs unbound instance methods
- property read -> implicit getter call
- explicit descriptor .__get__ call
- implicit descriptor get for instance + class access (obj=None) and arg binding

## Non-goals
- No runtime instrumentation or dynamic execution for analysis.
- No full dynamic dispatch resolution beyond deterministic syntax + SCIP + decorators.
- No changes to CFG/DFG algorithms beyond call wiring outputs in Stage-F.

## Data Contract Changes (First-Class Columns)

### 1) `graph.cpg_call_targets`
Add columns (names fixed for determinism and testability):
- `target_role` (`VARCHAR`, non-null): `"primary" | "init"`
- `binding_kind` (`VARCHAR`, non-null):  
  `"constructor" | "init" | "bound_method" | "unbound_method" | "classmethod" |
  "staticmethod" | "property_get" | "property_set" | "descriptor_get" |
  "descriptor_set" | "descriptor_set_augassign" | "operator_dunder" | "unknown"`
- `origin` (`VARCHAR`, non-null): `"syntax_call" | "descriptor_desugar" | "augassign_desugar"`
- `call_kind` (`VARCHAR`, nullable):  
  `"explicit" | "implicit_descriptor_get" | "implicit_descriptor_set" |
  "implicit_descriptor_set_augassign" | "implicit_augassign_iadd"`
- `augop` (`VARCHAR`, nullable): for augmented assignment (e.g., `"+="`)

Keep `extras_json` for forward compatibility, but **do not** store binding/role fields there.

### 2) `graph.cpg_edges_arg_to_param`
Add columns:
- `arg_slot` (`VARCHAR`, nullable):  
  `"positional:0"`, `"positional:1"`, `"keyword:<name>"`,
  `"implicit:receiver"`, `"implicit:obj"`, `"implicit:objtype"`, `"implicit:none"`
- `arg_role` (`VARCHAR`, nullable):  
  `"positional" | "keyword" | "receiver" | "descriptor_obj" | "descriptor_objtype" |
  "descriptor_none" | "augassign_value"`
- `arg_is_implicit` (`BOOLEAN`, nullable)
- `call_kind` (`VARCHAR`, nullable): mirrors the callsite kind (explicit vs implicit)
- `augop` (`VARCHAR`, nullable): set for augmented assignment synthesized edges

### 3) `graph.cpg_edges_ret_to_call`
Add columns:
- `target_role` (`VARCHAR`, nullable): keep return edges tied to `"primary"` targets
- `call_kind` (`VARCHAR`, nullable)
- `origin` (`VARCHAR`, nullable)

### Schema tasks
Update:
- `src/codeintel/core/schemas/output_registry.py`
- `src/codeintel/core/schemas/generated_rows/graph.py`
- any schema catalog or generated contract artifacts that mirror these tables

## Implementation Phases (with Checklists)

### Phase 0: Schema and Contract Alignment
- [ ] Add new columns to call wiring schemas (`graph.cpg_call_targets`,
      `graph.cpg_edges_arg_to_param`, `graph.cpg_edges_ret_to_call`).
- [ ] Update typed row models in `src/codeintel/core/schemas/generated_rows/graph.py`.
- [ ] Update any schema export or registry utilities that assume column lists.
- [ ] Ensure Arrow metadata and Pandera schema generation reflect the new columns.

### Phase 1: Definition Metadata Catalog (Decorators + Container Context)
Build a deterministic, cached metadata frame used by call wiring:
- [ ] From `core.syntax_defs` and/or `core.syntax_defs_resolved`:
  - def_kind (function/class)
  - container_def_id (class owner)
  - decorators list (from extras)
  - def_name
- [ ] Normalize decorator signals:
  - `@classmethod` -> `decorator_kind="classmethod"`
  - `@staticmethod` -> `decorator_kind="staticmethod"`
  - `@property` -> `decorator_kind="property_get"`
  - `@<name>.setter` -> `decorator_kind="property_set"` + link setter name to property
- [ ] Emit a derived `method_kind` per def:
  - `"instance_method" | "classmethod" | "staticmethod" | "property_get" | "property_set"`
- [ ] Keep this as a helper frame in `call_wiring.py` (not a new table unless needed).

### Phase 2: Explicit Call Target Enrichment
Extend `cpg_call_targets` to infer binding roles for explicit `Call` nodes:
- [ ] Use def metadata + call shape to set `binding_kind`:
  - Classmethod/staticmethod/property derived from decorators.
  - Bound vs unbound method via:
    - callee text shape (`X.y`), and
    - whether base resolves to a class def (use `core.syntax_defs`).
- [ ] Add constructor dual targets:
  - target_role="primary", binding_kind="constructor" for class call
  - target_role="init", binding_kind="init" for `__init__` if present
- [ ] Set `origin="syntax_call"`, `call_kind="explicit"` for these rows.
- [ ] Update `cpg_edges_ret_to_call` to use **only primary targets**.

### Phase 3: Implicit Descriptor Reads (property + __get__)
Synthesize call targets for attribute reads:
- [ ] Detect attribute loads from AST payload in `core.syntax_nodes`:
  - AST `Attribute` with `ctx=Load` (or AST data attached to syntax nodes).
- [ ] Resolve attribute target:
  - property getter (via decorator metadata)
  - descriptor `__get__` (via descriptor catalog)
- [ ] Emit implicit call targets:
  - `binding_kind="property_get"` or `"descriptor_get"`
  - `origin="descriptor_desugar"`
  - `call_kind="implicit_descriptor_get"`
- [ ] Build ARG→PARAM edges:
  - implicit receiver (`self`)
  - implicit obj/objtype for descriptor gets
  - class-access case: `obj=None` (use `arg_slot="implicit:none"`)

### Phase 4: Descriptor Writes (property setter + __set__)
Lower assignment to implicit calls:
- [ ] Detect `Assign`/`AnnAssign`/`AugAssign` on attribute targets:
  - AST `Attribute` with `ctx=Store`
- [ ] Resolve attribute target:
  - property setter (`@p.setter`)
  - descriptor `__set__`
- [ ] Emit implicit call targets:
  - `binding_kind="property_set"` or `"descriptor_set"`
  - `origin="descriptor_desugar"`
  - `call_kind="implicit_descriptor_set"`
- [ ] Build ARG→PARAM edges:
  - `obj` arg from receiver
  - `value` arg from RHS expression
  - `self` arg implicit only if needed for descriptor instance

### Phase 5: Augmented Assignment Descriptor Lowering
Implement the Stage-F augmented assignment lowering for descriptor targets:
- [ ] Detect `AugAssign` on attribute target.
- [ ] Synthesize **two** implicit calls:
  - `descriptor_get` with `call_kind="implicit_descriptor_get"`
  - `descriptor_set` with `call_kind="implicit_descriptor_set_augassign"`
- [ ] Optionally synthesize operator dunder call (`__iadd__`, `__add__`):
  - `binding_kind="operator_dunder"`, `call_kind="implicit_augassign_iadd"`
- [ ] Emit `arg_to_param` for `value` as a structured `AugAssignValue`
  reference with `base_read_call_id` and RHS span.

### Phase 6: Arg-to-Param Wiring (Explicit + Implicit)
Extend `cpg_edges_arg_to_param` to handle:
- [ ] implicit receiver mapping for bound method / classmethod / property get
- [ ] descriptor protocol implicit args (`obj`, `objtype`, `obj=None`)
- [ ] constructor mapping: bind user args to `__init__` params only
- [ ] emit `arg_slot`, `arg_role`, `arg_is_implicit`, `call_kind`, `augop`

### Phase 7: Ret-to-Call Semantics
Ensure return edges reflect semantic return flows:
- [ ] Constructors: RET edges for class call only (not `__init__`)
- [ ] Descriptor sets: no RET edges
- [ ] Descriptor gets / property gets: RET edges allowed
- [ ] Attach `target_role` + `call_kind` columns

### Phase 8: Unified CPG Projection
Keep `graph.cpg_edges`/`graph.cpg_nodes` in sync:
- [ ] Update `src/codeintel/build/hamilton/native/graphs/cpg.py` to
  include binding/role fields in `extras_json` for call wiring edges
  (optional but recommended for single-graph debugging).
- [ ] Maintain current PK/ordinal logic; do not change ID hashes.

### Phase 9: Fixtures + Golden Tests
Add the micro-fixtures and golden tests specified in `docs/CPG_additional_algorithms.md`:
- [ ] `tests/fixtures/call_wiring/f3_dynamicish_calls.py`
- [ ] `tests/fixtures/cpg_descriptor_writes/write_basic.py`
- [ ] `tests/fixtures/cpg_stage_f/augassign_descriptor.py`
- [ ] Golden assertions over:
  - `graph.cpg_call_targets`
  - `graph.cpg_edges_arg_to_param`
  - `graph.cpg_edges_ret_to_call`
- [ ] Use span-projection joins (as shown in the doc) to avoid hash brittleness.

## File-by-File Change List (Initial Pass)

### Schema / Contracts
- `src/codeintel/core/schemas/output_registry.py`
  - Add new columns for call wiring tables.
- `src/codeintel/core/schemas/generated_rows/graph.py`
  - Extend typed rows with new columns.

### Call Wiring Implementation
- `src/codeintel/build/hamilton/native/graphs/call_wiring.py`
  - Build def metadata (decorators, container class, method kind).
  - Create descriptor/property catalogs.
  - Synthesize implicit calls and arg edges.
  - Populate new columns for binding/role fields.

### CPG Projection
- `src/codeintel/build/hamilton/native/graphs/cpg.py`
  - Include binding/role fields in edge extras for unified graph view.

### Ingestion / Syntax Sources (if needed)
- `src/codeintel/ingestion/compute/cst_extract.py`
  - Ensure decorator payloads are normalized and stable.
- `src/codeintel/ingestion/compute/ast_extract.py`
  - Confirm attribute load/store context is attached to syntax nodes.

### Tests
- `tests/fixtures/call_wiring/f3_dynamicish_calls.py`
- `tests/fixtures/cpg_descriptor_writes/write_basic.py`
- `tests/fixtures/cpg_stage_f/augassign_descriptor.py`
- New golden tests under `tests/golden/` for call wiring tables.

## Acceptance Criteria
- Call wiring tables include binding/role columns with no reliance on `extras_json`.
- All Stage-F fixtures pass with deterministic spans and call IDs.
- Constructor calls produce primary+init targets and return edges target only primary.
- Descriptor reads/writes and augassign lower to implicit call targets with correct arg mapping.

## Open Questions
- Should `arg_slot` and `arg_role` be normalized across all edges or only implicit edges?
- Should operator dunder calls (`__iadd__`/`__add__`) be required in Stage-F or staged later?
- Do we want a dedicated `graph.cpg_call_targets_v2` table for migration, or migrate in place?
