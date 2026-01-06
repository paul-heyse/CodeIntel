Below is a **conservative, best‑in‑class re-organization PR plan** that keeps your current **outputs** stable (`graph.cpg_nodes`, `graph.cpg_edges`) and keeps the **build/storage separation** intact, while making the Hamilton computations **more modular, faster to iterate on, easier to test**, and much clearer.

The core idea is:

* **Keep** `src/codeintel/build/hamilton/native/graphs/cpg/` as the *public surface* (so `t__cpg` and `graph_targets.py` don’t change).
* Introduce a new **internal plane decomposition** under `src/codeintel/build/hamilton/native/graphs/cpg2/` that:

  * exposes **many small “frame nodes”** (syntax plane, symbol plane, flow plane, overlays…),
  * and lets `cpg.cpg_nodes` / `cpg.cpg_edges` become thin, deterministic **assembly nodes** (concat → dedupe → align).

This is the same “plane-by-plane” concept we’ve discussed, but applied **without schema changes** and in a way you can migrate **mechanically** first (move verbatim), then optimize (replace dict indexes with joins).

---

## 0) How Hamilton will discover the new modules

Your `native/discovery.py` automatically imports every `.py` module and every **package** (dir with `__init__.py`) under `native/graphs/`.

So if you create:

```
src/codeintel/build/hamilton/native/graphs/cpg2/__init__.py
```

…it will be automatically discovered and included.

**Important rule:** do **not** define `cpg_nodes` or `cpg_edges` inside `cpg2`, or you’ll risk name collisions.
Instead, in `cpg2` define only **prefixed** node names like:

* `cpg2_nodes__...`
* `cpg2_edges__...`

Then the existing `cpg.cpg_nodes` and `cpg.cpg_edges` depend on those.

No storage dependency changes are introduced; everything stays within `src/codeintel/build`.

---

## 1) New module skeleton: `graphs/cpg2/` (plane-by-plane)

### Proposed directory tree

```
src/codeintel/build/hamilton/native/graphs/
  cpg/                       # stays public, stable
    __init__.py
    nodes.py
    edges.py
    ids.py
    bytecode.py
    inspect_overlay.py
    _legacy.py               # shrinks over time, then deleted

  cpg2/                      # new internal plane decomposition
    __init__.py
    types.py                 # shared dataclasses/types moved out of _legacy
    shared_ids.py            # stable ids + ordinals + pk helpers
    shared_payloads.py       # extras_json encoding helpers
    shared_tables.py         # rows->table helpers + select columns + common glue

    plane_nodes_syntax.py    # syntax/cst/ast/scip/goid/tokens/trivia/module/cfg-block nodes -> cpg node frames
    plane_nodes_python.py    # symtable + bytecode + inspect nodes -> cpg node frames
    plane_nodes_assemble.py  # cpg2_nodes__frames aggregator

    plane_edges_syntax.py    # core.syntax_edges -> cpg edges frame
    plane_edges_symbol.py    # scip xrefs/relationships/goid + import_graph -> edges frames
    plane_edges_flow.py      # cfg/dfg/cdg + call_graph + call_wiring -> edges frames
    plane_edges_overlay_symtable.py   # py_sym + ast-binding overlays -> edge frames
    plane_edges_overlay_bytecode.py   # py_bc overlays -> edge frames
    plane_edges_overlay_inspect.py    # inspect overlays -> edge frames
    plane_edges_assemble.py  # cpg2_edges__frames aggregator
```

**Why this shape works well in Hamilton:**

* `cpg2_*` plane modules expose *lots* of small Hamilton nodes (`cpg2_nodes__...`, `cpg2_edges__...`).
* `cpg.cpg_nodes` and `cpg.cpg_edges` become pure “assembly”:

  * concat frames
  * dedupe
  * align to contract

That gives you:

* easy profiling (per-plane compute is visible),
* easy gating (turn overlays off cleanly),
* easy extension (add one new frame node, add it to the aggregator).

---

## 2) The anchor map keys (PK columns) you should standardize

A huge contributor to robustness is centralizing “what identifies a node” into one place.

Create a dict in `cpg2/shared_ids.py` (or `cpg2/types.py`) like:

### Node anchor PK columns (used to compute `cpg_node_id`)

| Upstream table key                 | Node kind                 | Anchor PK columns                                |
| ---------------------------------- | ------------------------- | ------------------------------------------------ |
| `core.syntax_nodes`                | `SYNTAX_NODE`             | `repo, commit, rel_path, producer, node_id`      |
| `core.ast_nodes`                   | `AST_NODE`                | `hash`                                           |
| `core.scip_symbol_information`     | `SCIP_SYMBOL`             | `repo, commit, symbol`                           |
| `core.goids`                       | `GOID`                    | `goid_h128`                                      |
| `graph.cfg_blocks`                 | `CFG_BLOCK`               | `function_goid_h128, block_idx`                  |
| `graph.import_modules`             | `MODULE`                  | `repo, commit, module`                           |
| `core.ts_tokens`                   | `TS_TOKEN`                | `repo, commit, rel_path, language, token_id`     |
| `core.ts_trivia`                   | `TS_TRIVIA`               | `repo, commit, rel_path, language, trivia_id`    |
| `core.py_sym_scopes`               | `SCOPE`                   | `repo, commit, rel_path, scope_id`               |
| `core.py_sym_bindings`             | `BINDING`                 | `repo, commit, rel_path, binding_id`             |
| `core.py_bc_code_units`            | `BC_CODE_UNIT`            | `repo, commit, rel_path, code_unit_id`           |
| `core.py_bc_instructions`          | `BC_INSTR`                | `repo, commit, rel_path, code_unit_id, instr_id` |
| `core.py_bc_blocks`                | `BC_BLOCK`                | `repo, commit, rel_path, block_id`               |
| `core.py_inspect_objects`          | `INSPECT_OBJECT`          | `repo, commit, object_id`                        |
| `core.py_inspect_signatures`       | `INSPECT_SIGNATURE`       | `repo, commit, signature_id`                     |
| `core.py_inspect_signature_params` | `INSPECT_SIGNATURE_PARAM` | `repo, commit, signature_id, param_index`        |

In the conservative pass, you can keep your existing `_pk_from_row(table_key, pk_dict)` logic, but route all PK creation through these definitions so you don’t have “silent drift” across edge builders.

---

## 3) Join-built vs algorithm-built: what moves where

This is the core “best-in-class organization” criterion: **make joins joins**, and keep **algorithms isolated**.

### Join-built edges (should become Arrow joins over time)

These can (and should) migrate away from dict-index lookups:

* `core.syntax_edges` → AST parent/child edges (pure structural join)
* SCIP symbol layer:

  * `core.scip_occurrence_syntax_xref` + `core.scip_occurrence_span_xref`
  * `core.scip_symbol_relationships`
  * `core.scip_symbol_goid_xref`
* Linking graphs:

  * `graph.call_graph_edges` (GOID→GOID)
  * `graph.import_graph_edges` (MODULE→MODULE)
* Flow conversions (your CPG layer over existing graphs):

  * `graph.cfg_edges` + `graph.cfg_blocks` (+ `core.goids` for repo/commit/rel_path)
  * `graph.dfg_edges` + `graph.cfg_blocks` (+ `core.goids`)
  * `graph.cdg_edges` + `graph.cfg_blocks` (+ `core.goids`)
* Call wiring conversions:

  * `graph.cpg_edges_calls` + `core.syntax_nodes` + `graph.cfg_blocks`
  * `graph.cpg_edges_arg_to_param` + `core.syntax_nodes`
  * `graph.cpg_edges_ret_to_call` + `core.syntax_nodes`

### Algorithm-built edges (keep isolated)

These should remain in “overlay algorithm modules”, even if they use joins internally:

* bytecode simulation / stack effects (`py_bc_stack_edges_to_cpg`)
* bytecode reaches / def-use modeling (`enable_reaches`)
* inspect runtime inference (unwrap chains, class MRO)
* AST anchoring heuristics (bytecode→ast matching, inspect→ast matching)

This separation is the key to clarity and speed: your base planes remain predictable; overlays are the only “fuzzy inference” parts.

---

## 4) Exact new Hamilton node signatures to introduce

These are the *concrete nodes* you add in `cpg2/` so that `cpg.cpg_nodes` and `cpg.cpg_edges` become tiny.

### 4.1 Nodes: core “tables to frames” nodes

Add in `cpg2/plane_nodes_syntax.py` / `cpg2/plane_nodes_python.py`:

#### Convert the big input bundle once

```py
def cpg2_nodes__core_tables(
    cpg_nodes__core_inputs: _CpgNodeCoreInputs,
) -> _CpgNodeCoreLazyFrames: ...
```

```py
def cpg2_nodes__graph_tables(
    cpg_nodes__graph_inputs: _CpgNodeGraphInputs,
) -> _CpgNodeGraphLazyFrames: ...
```

#### One node-frame per upstream table

Each returns a `pa.Table` already shaped like CPG nodes columns.

```py
def cpg2_nodes__syntax_nodes_frame(
    cpg2_nodes__core_tables: _CpgNodeCoreLazyFrames,
) -> pa.Table: ...
```

```py
def cpg2_nodes__ast_nodes_frame(env: BuildEnv, cpg2_nodes__core_tables: _CpgNodeCoreLazyFrames) -> pa.Table: ...
def cpg2_nodes__scip_symbols_frame(cpg2_nodes__core_tables: _CpgNodeCoreLazyFrames) -> pa.Table: ...
def cpg2_nodes__goids_frame(cpg2_nodes__core_tables: _CpgNodeCoreLazyFrames) -> pa.Table: ...
```

```py
def cpg2_nodes__py_sym_scopes_frame(cpg2_nodes__core_tables: _CpgNodeCoreLazyFrames) -> pa.Table: ...
def cpg2_nodes__py_sym_bindings_frame(cpg2_nodes__core_tables: _CpgNodeCoreLazyFrames) -> pa.Table: ...
```

```py
def cpg2_nodes__py_bc_code_units_frame(cpg2_nodes__core_tables: _CpgNodeCoreLazyFrames) -> pa.Table: ...
def cpg2_nodes__py_bc_instructions_frame(cpg2_nodes__core_tables: _CpgNodeCoreLazyFrames) -> pa.Table: ...
def cpg2_nodes__py_bc_blocks_frame(cpg2_nodes__core_tables: _CpgNodeCoreLazyFrames) -> pa.Table: ...
```

```py
def cpg2_nodes__py_inspect_objects_frame(cpg2_nodes__core_tables: _CpgNodeCoreLazyFrames) -> pa.Table: ...
def cpg2_nodes__py_inspect_signatures_frame(cpg2_nodes__core_tables: _CpgNodeCoreLazyFrames) -> pa.Table: ...
def cpg2_nodes__py_inspect_signature_params_frame(cpg2_nodes__core_tables: _CpgNodeCoreLazyFrames) -> pa.Table: ...
```

```py
def cpg2_nodes__ts_tokens_frame(cpg2_nodes__core_tables: _CpgNodeCoreLazyFrames) -> pa.Table: ...
def cpg2_nodes__ts_trivia_frame(cpg2_nodes__core_tables: _CpgNodeCoreLazyFrames) -> pa.Table: ...
```

```py
def cpg2_nodes__cfg_blocks_frame(
    cpg2_nodes__graph_tables: _CpgNodeGraphLazyFrames,
    cpg2_nodes__core_tables: _CpgNodeCoreLazyFrames,
) -> pa.Table: ...
```

```py
def cpg2_nodes__import_modules_frame(cpg2_nodes__graph_tables: _CpgNodeGraphLazyFrames) -> pa.Table: ...
```

#### The frame aggregator

In `cpg2/plane_nodes_assemble.py`:

```py
def cpg2_nodes__frames(
    cpg2_nodes__syntax_nodes_frame: pa.Table,
    cpg2_nodes__ast_nodes_frame: pa.Table,
    cpg2_nodes__scip_symbols_frame: pa.Table,
    cpg2_nodes__goids_frame: pa.Table,
    cpg2_nodes__py_sym_scopes_frame: pa.Table,
    cpg2_nodes__py_sym_bindings_frame: pa.Table,
    cpg2_nodes__py_bc_code_units_frame: pa.Table,
    cpg2_nodes__py_bc_instructions_frame: pa.Table,
    cpg2_nodes__py_bc_blocks_frame: pa.Table,
    cpg2_nodes__py_inspect_objects_frame: pa.Table,
    cpg2_nodes__py_inspect_signatures_frame: pa.Table,
    cpg2_nodes__py_inspect_signature_params_frame: pa.Table,
    cpg2_nodes__ts_tokens_frame: pa.Table,
    cpg2_nodes__ts_trivia_frame: pa.Table,
    cpg2_nodes__cfg_blocks_frame: pa.Table,
    cpg2_nodes__import_modules_frame: pa.Table,
) -> list[pa.Table]:
    return [...]
```

Then `cpg.cpg_nodes` becomes:

```py
def cpg_nodes(cpg2_nodes__frames: list[pa.Table]) -> InferableTabularInput:
    # concat → dedupe → align
```

---

### 4.2 Edges: one “frame node” per edge source, plus overlay frame group

Add in `cpg2/plane_edges_*.py`:

#### Structural + symbol

```py
def cpg2_edges__syntax_edges_frame(cpg_edge_core_inputs: _CpgEdgeCoreInputs) -> pa.Table: ...
def cpg2_edges__scip_occurrence_edges_frame(cpg_edge_core_inputs: _CpgEdgeCoreInputs) -> pa.Table: ...
def cpg2_edges__scip_symbol_relationships_frame(cpg_edge_core_inputs: _CpgEdgeCoreInputs) -> pa.Table: ...
def cpg2_edges__scip_symbol_goid_edges_frame(cpg_edge_core_inputs: _CpgEdgeCoreInputs) -> pa.Table: ...
def cpg2_edges__import_graph_edges_frame(cpg_edge_core_inputs: _CpgEdgeCoreInputs) -> pa.Table: ...
```

#### Flow + linking

```py
def cpg2_edges__call_graph_edges_frame(cpg_edge_core_inputs: _CpgEdgeCoreInputs) -> pa.Table: ...
def cpg2_edges__cfg_edges_frame(cpg_edge_core_inputs: _CpgEdgeCoreInputs) -> pa.Table: ...
def cpg2_edges__dfg_edges_frame(cpg_edge_core_inputs: _CpgEdgeCoreInputs) -> pa.Table: ...
def cpg2_edges__cdg_edges_frame(cpg_edge_core_inputs: _CpgEdgeCoreInputs) -> pa.Table: ...
```

#### Call wiring

```py
def cpg2_edges__call_wiring_calls_frame(cpg_edge_core_inputs: _CpgEdgeCoreInputs) -> pa.Table: ...
def cpg2_edges__call_wiring_arg_to_param_frame(cpg_edge_core_inputs: _CpgEdgeCoreInputs) -> pa.Table: ...
def cpg2_edges__call_wiring_ret_to_call_frame(cpg_edge_core_inputs: _CpgEdgeCoreInputs) -> pa.Table: ...
```

#### Overlays (gated by options)

Split overlays into explicit frame nodes so each is independently cacheable and testable:

```py
def cpg2_edges__overlay_symtable_scope_frame(
    cpg_edge_overlay_scope_inputs: _CpgOverlayScopeInputs,
    cpg__overlay_options: CpgOverlayOptions,
) -> pa.Table: ...
```

```py
def cpg2_edges__overlay_symtable_binding_frame(
    cpg_edge_overlay_symbol_inputs: _CpgOverlaySymbolInputs,
    cpg__overlay_options: CpgOverlayOptions,
) -> pa.Table: ...
```

```py
def cpg2_edges__overlay_symtable_resolution_frame(...)-> pa.Table: ...
def cpg2_edges__overlay_symtable_binding_symbol_frame(...)-> pa.Table: ...
def cpg2_edges__overlay_ast_binding_frame(...)-> pa.Table: ...
```

Bytecode overlays:

```py
def cpg2_edges__overlay_bc_instruction_ast_frame(
    cpg_edge_overlay_bytecode_inputs: _CpgOverlayBytecodeInputs,
    cpg_edge_overlay_syntax_call_inputs: _CpgOverlaySyntaxCallInputs,
    cpg__overlay_options: CpgOverlayOptions,
) -> pa.Table: ...
```

```py
def cpg2_edges__overlay_bc_callsite_frame(...)-> pa.Table: ...
def cpg2_edges__overlay_bc_callsite_symbol_frame(...)-> pa.Table: ...
def cpg2_edges__overlay_bc_stack_frame(...)-> pa.Table: ...
def cpg2_edges__overlay_bc_cfg_frame(...)-> pa.Table: ...
def cpg2_edges__overlay_bc_defuse_binding_frame(...)-> pa.Table: ...
def cpg2_edges__overlay_bc_reaches_frame(
    ...,
    cpg__options: CpgOptions,
)-> pa.Table: ...
```

Inspect overlays:

```py
def cpg2_edges__overlay_inspect_signature_frame(
    cpg_edge_overlay_inspect_core_inputs: _CpgOverlayInspectCoreInputs,
    cpg__overlay_options: CpgOverlayOptions,
) -> pa.Table: ...
```

```py
def cpg2_edges__overlay_inspect_to_ast_frame(...)-> pa.Table: ...
def cpg2_edges__overlay_inspect_to_scip_frame(...)-> pa.Table: ...
def cpg2_edges__overlay_inspect_class_mro_frame(...)-> pa.Table: ...
def cpg2_edges__overlay_inspect_class_attr_frame(...)-> pa.Table: ...
def cpg2_edges__overlay_inspect_runtime_state_frame(...)-> pa.Table: ...
def cpg2_edges__overlay_inspect_unwrap_frame(...)-> pa.Table: ...
```

#### Edge frame aggregator

In `cpg2/plane_edges_assemble.py`:

```py
def cpg2_edges__frames(
    cpg2_edges__syntax_edges_frame: pa.Table,
    cpg2_edges__scip_occurrence_edges_frame: pa.Table,
    cpg2_edges__scip_symbol_relationships_frame: pa.Table,
    cpg2_edges__scip_symbol_goid_edges_frame: pa.Table,
    cpg2_edges__call_graph_edges_frame: pa.Table,
    cpg2_edges__import_graph_edges_frame: pa.Table,
    cpg2_edges__cfg_edges_frame: pa.Table,
    cpg2_edges__dfg_edges_frame: pa.Table,
    cpg2_edges__cdg_edges_frame: pa.Table,
    cpg2_edges__call_wiring_calls_frame: pa.Table,
    cpg2_edges__call_wiring_arg_to_param_frame: pa.Table,
    cpg2_edges__call_wiring_ret_to_call_frame: pa.Table,
    # overlay frames (many)
    cpg2_edges__overlay_symtable_scope_frame: pa.Table,
    ...
) -> list[pa.Table]:
    return [...]
```

Then `cpg.cpg_edges` becomes:

```py
def cpg_edges(cpg2_edges__frames: list[pa.Table]) -> InferableTabularInput:
    # concat → dedupe → align
```

---

## 5) “Which _legacy.py functions move to which new plane file”

This is the concrete mapping you asked for.

### 5.1 `cpg2/types.py` (shared dataclasses / public types)

Move these **classes** out of `_legacy.py` unchanged:

* `CpgOverlayOptions`
* `_CpgSymbolInputs`, `_CpgFlowInputs`
* `_OccurrenceRolePayload`
* `_CpgNodeCoreInputs`, `_CpgNodeSyntaxInputs`, `_CpgNodePyInputs`, `_CpgNodeInspectInputs`, `_CpgNodeGraphInputs`, `_CpgNodeInputs`
* `_CpgNodeCoreLazyFrames`, `_CpgNodeGraphLazyFrames`
* `_CpgLinkInputs`, `_CpgCallWiringInputs`, `_CpgSyntaxNodeInputs`
* `_CpgOverlayEdgeInputs`, `_CpgOverlayScopeInputs`, `_CpgOverlaySymbolInputs`, `_CpgOverlayBytecodeInputs`, `_CpgOverlaySyntaxCallInputs`
* `_CpgOverlayInspectCoreInputs`, `_CpgOverlayInspectRuntimeInputs`, `_CpgOverlayInspectInputs`
* `_CpgEdgeCoreInputs`
* `_CpgOverlayRegistryEntry`

(Plane-specific helper dataclasses stay with their plane modules.)

---

### 5.2 `cpg2/shared_ids.py`

Move these functions unchanged:

* `_stable_int_hash`
* `_stable_cpg_id`
* `_stable_ordinal`
* `_stable_cpg_id_from_row`
* `_stable_ordinal_from_row`
* `_pk_from_row`
* `_ordinal_from_row`
* `_pk_json_from_row`
* `_instruction_cpg_id`
* `stable_cpg_id` (public)
* `instruction_cpg_id` (public)

This makes “identity” logic centralized and removes it from plane code.

---

### 5.3 `cpg2/shared_payloads.py`

Move unchanged:

* `_row_to_payload`
* `_encode_optional_payload`

---

### 5.4 `cpg2/shared_tables.py`

Move unchanged:

* `_rows_to_table`
* `_node_rows_to_table`
* `_edge_rows_to_table`
* `_filter_valid_values`
* `_select_node_columns`
* `_select_edge_columns`
* `_frame_to_reader`
* `_arrow_join_frames`

---

### 5.5 `cpg2/plane_nodes_syntax.py`

Move unchanged conversion functions + index helpers used only by node conversion:

* `_syntax_nodes_to_cpg`
* `_ast_nodes_to_cpg`
* `_scip_symbols_to_cpg`
* `_goids_to_cpg`
* `_cfg_blocks_to_cpg`
* `_import_modules_to_cpg`
* `_ts_tokens_to_cpg`
* `_ts_trivia_to_cpg`

Add the new Hamilton frame nodes listed in §4.1.

---

### 5.6 `cpg2/plane_nodes_python.py`

Move unchanged:

* `_py_sym_scopes_to_cpg`
* `_py_sym_bindings_to_cpg`
* `_py_bc_code_units_to_cpg`
* `_py_bc_instructions_to_cpg`
* `_py_bc_blocks_to_cpg`
* `_py_inspect_objects_to_cpg`
* `_py_inspect_signatures_to_cpg`
* `_py_inspect_signature_params_to_cpg`

Add the matching `cpg2_nodes__*_frame` nodes.

---

### 5.7 `cpg2/plane_edges_syntax.py`

Move unchanged:

* `_syntax_edges_to_cpg`

---

### 5.8 `cpg2/plane_edges_symbol.py`

Move unchanged:

* `_scip_occurrence_edges_to_cpg` + all its helpers (`_occurrence_roles`, role resolvers, span indexes, fallback rows, etc.)
* `_scip_symbol_relationships_to_cpg`
* `_scip_symbol_goid_edges_to_cpg`
* `_import_graph_edges_to_cpg`

---

### 5.9 `cpg2/plane_edges_flow.py`

Move unchanged:

* `_cfg_edges_to_cpg` + `_cfg_block_index`
* `_dfg_edges_to_cpg` + `_dfg_edge_row`
* `_cdg_edges_to_cpg` + `_cdg_edge_row`
* `_call_graph_edges_to_cpg`
* `_call_wiring_calls_to_cpg` + `_block_id_index` + `_syntax_node_index`
* `_call_wiring_arg_to_param_to_cpg`
* `_call_wiring_ret_to_call_to_cpg`

---

### 5.10 `cpg2/plane_edges_overlay_symtable.py`

Move unchanged:

* `_ast_binding_edges_to_cpg` (and its helper structs/functions for AST events & binding resolution)
* `_py_sym_scope_edges_to_cpg`
* `_py_sym_namespace_edges_to_cpg`
* `_py_sym_binding_edges_to_cpg`
* `_py_sym_resolution_edges_to_cpg`
* `_py_sym_binding_symbol_edges_to_cpg`

---

### 5.11 `cpg2/plane_edges_overlay_bytecode.py`

Move unchanged:

* `_py_bc_instruction_ast_edges_to_cpg`
* `_py_bc_callsite_edges_to_cpg`
* `_py_bc_callsite_symbol_edges_to_cpg`
* `_py_bc_memory_edges_to_cpg`
* `_py_bc_stack_edges_to_cpg`
* `_py_bc_cfg_edges_to_cpg`
* `_py_bc_defuse_binding_edges_to_cpg`
* `_py_bc_reaches_edges_to_cpg`
* plus all bytecode helper dataclasses/functions they rely on

Also keep the public wrappers (for backward imports) by reexporting from `cpg/bytecode.py`:

* `py_bc_callsite_edges_to_cpg`
* `py_bc_callsite_symbol_edges_to_cpg`
* `py_bc_stack_edges_to_cpg`

---

### 5.12 `cpg2/plane_edges_overlay_inspect.py`

Move unchanged:

* `_inspect_arg_to_param_edges_to_cpg`
* `_py_inspect_signature_edges_to_cpg`
* `_inspect_to_ast_edges_to_cpg`
* `_inspect_to_scip_edges_to_cpg`
* `_py_inspect_class_mro_edges_to_cpg`
* `_py_inspect_class_attr_edges_to_cpg`
* `_py_inspect_runtime_state_edges_to_cpg`
* `_py_inspect_unwrap_edges_to_cpg`
* plus inspect helper dataclasses/functions

And keep public wrappers via `cpg/inspect_overlay.py`:

* `inspect_to_ast_edges_to_cpg`
* `py_inspect_unwrap_edges_to_cpg`

---

## 6) Changes to existing `cpg/` files (public surface stays stable)

### 6.1 `src/codeintel/build/hamilton/native/graphs/cpg/_legacy.py`

PR plan:

1. **Phase 1:** stop adding anything new here.
2. **Phase 2:** replace contents with imports/shims pointing at `cpg2/`.
3. **Phase 3:** delete once all references moved.

### 6.2 `cpg/nodes.py`, `cpg/edges.py`, `cpg/ids.py`, `cpg/bytecode.py`, `cpg/inspect_overlay.py`

Change them from reexporting `_legacy` to reexporting `cpg2.*`.

This preserves all external import paths you might have elsewhere in the repo.

### 6.3 `cpg/__init__.py`

This file can remain mostly as-is, but should import the *public* symbols from their new home (still under `cpg/` wrappers).

### 6.4 `cpg.cpg_nodes` and `cpg.cpg_edges` become thin assembly nodes

They should do only:

* receive `cpg2_nodes__frames` / `cpg2_edges__frames`
* concat via `concat_tables_unified`
* dedupe via `dedupe_table_for_table`
* align via `align_table_to_contract`
* return `InferableTabularInput` via `_frame_to_reader`

That’s it.

---

## 7) The mechanical “do this in order” checklist

This is the “move verbatim first, then optimize” path you explicitly want.

### Phase A — Pure mechanical extraction (no behavior changes)

1. Create `graphs/cpg2/` package + empty modules.
2. Move **types** first:

   * move the shared dataclasses into `cpg2/types.py`
   * update imports in `_legacy.py` to import from `cpg2.types`
3. Move **shared helpers** next:

   * ids → `cpg2/shared_ids.py`
   * payloads → `cpg2/shared_payloads.py`
   * table helpers → `cpg2/shared_tables.py`
4. Move **node conversion functions** into:

   * `cpg2/plane_nodes_syntax.py`
   * `cpg2/plane_nodes_python.py`
5. Move **edge conversion functions** into:

   * `cpg2/plane_edges_syntax.py`
   * `cpg2/plane_edges_symbol.py`
   * `cpg2/plane_edges_flow.py`
   * overlays into their three overlay modules
6. In each plane module, add the new **Hamilton frame nodes** (`cpg2_nodes__...`, `cpg2_edges__...`) that just call the moved `_..._to_cpg()` functions.
7. Add `cpg2_nodes__frames` and `cpg2_edges__frames` aggregator nodes.

At this point, nothing is wired into public outputs yet; you can run DAGs and inspect the plane nodes.

### Phase B — Wire the public nodes to the plane outputs

8. Rewrite `cpg.cpg_nodes` to depend on `cpg2_nodes__frames` and assemble.
9. Rewrite `cpg.cpg_edges` to depend on `cpg2_edges__frames` and assemble.
10. Confirm output row counts + schema are stable.

### Phase C — Now do the “best” improvements (replace dict-index with joins)

11. Replace these dict-index patterns with Arrow joins:

* `_cfg_block_index` usage in `_cfg_edges_to_cpg`, `_dfg_edges_to_cpg`, `_cdg_edges_to_cpg`
  **Join plan:** `cfg_edges` → join with `cfg_blocks` (src) → join with `cfg_blocks` (dst) → compute block_idx pairs → compute node ids.

* `_syntax_node_index` usage in call wiring and occurrence edges
  **Join plan:** join `graph.cpg_edges_*` tables to `core.syntax_nodes` on `(repo, commit, node_id)` (and bring `rel_path`, `producer`).

* `_block_id_index` usage for call wiring calls
  **Join plan:** join `graph.cpg_edges_calls` on `callee_entry_block_id` to `cfg_blocks` to get `(function_goid_h128, block_idx)`.

12. Only after join refactors are stable, delete `_legacy.py`.

---

## 8) What you get immediately after this refactor

Even before any join optimizations, this gives you:

* **Robustness:** each plane node is isolated; failures are localized.
* **Execution clarity:** Hamilton graph shows per-plane work instead of one monolith.
* **Extensibility:** adding a new output is “add a new frame node + add it to aggregator”.
* **Performance headroom:** the join refactors become straightforward because every edge builder is now its own file and node.

---

If you want, I can take the next step and produce a **second document** that is even more “PR-ready”: for each file above, a literal **checklist of edits** (imports to change, where to cut/paste from `_legacy.py`, and the exact `__all__` exports to keep your package surface stable).



According to a document from **January 5, 2026**, your Hamilton DAG already enumerates the “planes” that feed CPG (AST, LibCST-derived canonical `core.syntax_*`, SCIP tables + xrefs, GOIDs, CFG/DFG/CDG/PDG, symtable/bytecode/inspect overlays, and tree-sitter tokens/trivia).  

Below is the **“second document”** you asked for: a **PR-ready, file-by-file checklist** that turns the current monolithic `src/codeintel/build/hamilton/native/graphs/cpg/_legacy.py` into a **plane-by-plane internal structure** while keeping your **public package surface stable** (`codeintel.build.hamilton.native.graphs.cpg.*` exports don’t change) and keeping it **build-only** (no `src/codeintel/storage` coupling; Arrow/Parquet outputs remain the seam). 

---

# CPG Refactor PR-Ready Checklist (plane-by-plane, conservative API)

## Non-negotiables (guardrails)

### A) Public API must remain stable

**Keep these files and their `__all__` lists stable:**

* `src/codeintel/build/hamilton/native/graphs/cpg/__init__.py`
* `src/codeintel/build/hamilton/native/graphs/cpg/nodes.py`
* `src/codeintel/build/hamilton/native/graphs/cpg/edges.py`
* `src/codeintel/build/hamilton/native/graphs/cpg/ids.py`
* `src/codeintel/build/hamilton/native/graphs/cpg/bytecode.py`
* `src/codeintel/build/hamilton/native/graphs/cpg/inspect_overlay.py`

**Rule:** consumers that import `from codeintel.build.hamilton.native.graphs.cpg import ...` should not notice the refactor.

### B) No storage coupling

* [ ] Add **no imports** from `codeintel.storage.*`
* [ ] Outputs remain `pyarrow.Table` / `RecordBatchReader` (and the target materialization writes Parquet datasets elsewhere, as you already do) 

### C) Conservative behavior first, improvements second

* Phase 1: **Move code verbatim** into plane modules (no semantic changes).
* Phase 2: Replace hotspots (Python loops → Arrow joins / vectorized compute) only after the refactor is stable.

---

# 0) Mechanical move order (prevents import cycles)

Do the PR in this order (you can commit each step independently):

1. Add `cpg/cpg2/` scaffolding + shared foundation.
2. Move **node** builders + `cpg_nodes*` into `cpg2/nodes.py`.
3. Move **edge** builders + `cpg_edges*` into `cpg2/edges.py`.
4. Move overlay-public wrappers (`stable_cpg_id`, `instruction_cpg_id`, `py_bc_*`, `inspect_*`) into `cpg2/public.py`.
5. Update the thin wrapper modules (`cpg/nodes.py`, `cpg/edges.py`, `cpg/ids.py`, …) to point to `cpg2/*`.
6. Convert `cpg/_legacy.py` into a **shim** (re-export only), or delete it once nothing imports it.

---

# 1) New files to add (internal plane-by-plane implementation)

> **Key idea:** `cpg/cpg2/*` is internal. `cpg/*` stays stable and simply re-exports.

## 1.1 `src/codeintel/build/hamilton/native/graphs/cpg/cpg2/__init__.py` (NEW)

* [ ] Create the package directory: `cpg/cpg2/`
* [ ] Create `__init__.py` with a short docstring (“Internal CPG plane-by-plane implementation”).
* [ ] Do **not** import heavy modules here (avoid import-time cost / cycles).
* [ ] Minimal `__all__` is fine (or omit entirely).

**No cut/paste required**.

---

## 1.2 `.../cpg/cpg2/constants.py` (NEW)

**Goal:** centralize all table keys + column contracts + ordinal mod.

**Cut/paste from `_legacy.py`:**

* [ ] Copy the constant block:

  * `CPG_TARGET_NAME`
  * `CPG_NODES_TABLE_KEY`, `CPG_EDGES_TABLE_KEY`
  * all the upstream table keys (`SYNTAX_NODES_TABLE_KEY`, `SCIP_SYMBOLS_TABLE_KEY`, `GOIDS_TABLE_KEY`, …)
  * `ORDINAL_MOD`
* [ ] Copy the contract column tuples:

  * `_CPG_NODE_COLUMNS = columns_for_table_key(...) or (...)`
  * `_CPG_EDGE_COLUMNS = columns_for_table_key(...) or (...)`

**Imports to add in `constants.py`:**

* [ ] `from __future__ import annotations`
* [ ] `from codeintel.core.schemas.row_models import columns_for_table_key`

**Exports (`__all__`) in `constants.py`:**

```python
__all__ = [
    "CPG_TARGET_NAME",
    "CPG_NODES_TABLE_KEY",
    "CPG_EDGES_TABLE_KEY",
    "ORDINAL_MOD",
    "SYNTAX_NODES_TABLE_KEY",
    "SYNTAX_CALLS_TABLE_KEY",
    "SYNTAX_CALL_ARGS_TABLE_KEY",
    "SCIP_SYMBOLS_TABLE_KEY",
    "GOIDS_TABLE_KEY",
    "CFG_BLOCKS_TABLE_KEY",
    "IMPORT_MODULES_TABLE_KEY",
    "TS_TOKENS_TABLE_KEY",
    "TS_TRIVIA_TABLE_KEY",
    "AST_NODES_TABLE_KEY",
    "PY_SYM_SCOPES_TABLE_KEY",
    "PY_SYM_BINDINGS_TABLE_KEY",
    "PY_SYM_SCOPE_EDGES_TABLE_KEY",
    "PY_SYM_NAMESPACE_EDGES_TABLE_KEY",
    "PY_SYM_RESOLUTION_EDGES_TABLE_KEY",
    "PY_BC_CODE_UNITS_TABLE_KEY",
    "PY_BC_INSTRUCTIONS_TABLE_KEY",
    "PY_BC_BLOCKS_TABLE_KEY",
    "PY_BC_CFG_EDGES_TABLE_KEY",
    "PY_BC_DEFUSE_EVENTS_TABLE_KEY",
    "PY_INSPECT_OBJECTS_TABLE_KEY",
    "PY_INSPECT_CLASS_MRO_TABLE_KEY",
    "PY_INSPECT_CLASS_ATTRS_TABLE_KEY",
    "PY_INSPECT_UNWRAP_TABLE_KEY",
    "PY_INSPECT_SIGNATURES_TABLE_KEY",
    "PY_INSPECT_SIGNATURE_PARAMS_TABLE_KEY",
    "PY_INSPECT_SOURCE_TABLE_KEY",
    "PY_INSPECT_RUNTIME_STATE_TABLE_KEY",
    "_CPG_NODE_COLUMNS",
    "_CPG_EDGE_COLUMNS",
]
```

---

## 1.3 `.../cpg/cpg2/options.py` (NEW)

**Goal:** isolate option-loading (`@cache`) and overlay gating.

**Cut/paste from `_legacy.py`:**

* [ ] `cpg__options(env: BuildEnv) -> CpgOptions`
* [ ] `@dataclass CpgOverlayOptions`
* [ ] `cpg__overlay_options(env: BuildEnv) -> CpgOverlayOptions`

**Imports to add in `options.py`:**

* [ ] `from dataclasses import dataclass`
* [ ] `from hamilton.function_modifiers import cache`
* [ ] `from codeintel.build.hamilton.env import BuildEnv`
* [ ] `from codeintel.build.hamilton.native.options.graphs import CpgOptions`
* [ ] `from codeintel.build.hamilton.native.options.ingestion import (SymtableExtractOptions, BytecodeExtractOptions, InspectExtractOptions)`
* [ ] `from codeintel.build.hamilton.options_loading import load_target_options`

**Exports (`__all__`) in `options.py`:**

```python
__all__ = [
    "CpgOverlayOptions",
    "cpg__options",
    "cpg__overlay_options",
]
```

---

## 1.4 `.../cpg/cpg2/foundation.py` (NEW)

**Goal:** all shared helpers used by nodes + edges.

**Cut/paste from `_legacy.py`:**

* [ ] stable-id & ordinal helpers:

  * `_stable_int_hash`, `_stable_cpg_id`, `_stable_ordinal`,
  * `_stable_cpg_id_from_row`, `_stable_ordinal_from_row`
* [ ] payload helpers:

  * `_row_to_payload`, `_encode_optional_payload`,
  * `_pk_json_from_row`, `_payload_json_from_row`
* [ ] row→table helpers:

  * `_rows_to_table`, `_node_rows_to_table`, `_edge_rows_to_table`
* [ ] selection/validation helpers:

  * `_filter_valid_values`,
  * `_select_node_columns`, `_select_edge_columns`
* [ ] pk/ordinal wrappers:

  * `_pk_from_row`, `_ordinal_from_row`
* [ ] join wrapper:

  * `JoinSpec` dataclass
  * `_arrow_join_frames`
* [ ] alignment wrapper:

  * `_frame_to_reader`
* [ ] Arrow type constants:

  * `_CPG_DECIMAL_TYPE`, `_CPG_NODE_DECIMAL_COLUMNS`, `_CPG_EDGE_DECIMAL_COLUMNS`

**Imports to add in `foundation.py`:**

* [ ] `from __future__ import annotations`
* [ ] `from dataclasses import dataclass`
* [ ] `from collections.abc import Mapping, Sequence`
* [ ] `import pyarrow as pa`
* [ ] `import pyarrow.compute as pc`
* [ ] from your existing utilities (as in `_legacy.py`):

  * `stable_decimal_id`, `stable_int_hash`, `ensure_table_columns`, `select_table_columns` as needed
  * `ArrowJoinSpec`, `arrow_join_tables`, `align_table_to_contract`, `empty_table` helpers, etc.
* [ ] `from codeintel.core.serialization.payload import encode_payload`

**Important import change:** `foundation.py` should import `_CPG_NODE_COLUMNS/_CPG_EDGE_COLUMNS` from `cpg2.constants`, not compute them itself:

* [ ] `from .constants import _CPG_NODE_COLUMNS, _CPG_EDGE_COLUMNS, ORDINAL_MOD`

**Exports (`__all__`) in `foundation.py`:**
Keep it explicit so plane modules can import cleanly:

```python
__all__ = [
    "JoinSpec",
    "_arrow_join_frames",
    "_edge_rows_to_table",
    "_encode_optional_payload",
    "_filter_valid_values",
    "_frame_to_reader",
    "_node_rows_to_table",
    "_ordinal_from_row",
    "_payload_json_from_row",
    "_pk_from_row",
    "_pk_json_from_row",
    "_row_to_payload",
    "_select_edge_columns",
    "_select_node_columns",
    "_stable_cpg_id",
    "_stable_cpg_id_from_row",
    "_stable_int_hash",
    "_stable_ordinal",
    "_stable_ordinal_from_row",
]
```

---

## 1.5 `.../cpg/cpg2/nodes.py` (NEW)

**Goal:** `cpg_nodes` pipeline + node plane builders.

**Cut/paste from `_legacy.py` into `cpg2/nodes.py`:**

### A) Input dataclasses + bundlers

* [ ] dataclasses:

  * `_CpgNodeCoreInputs`, `_CpgNodeSyntaxInputs`, `_CpgNodePyInputs`, `_CpgNodeInspectInputs`,
  * `_CpgNodeGraphInputs`, `_CpgNodeInputs`,
  * `_CpgNodeCoreLazyFrames`, `_CpgNodeGraphLazyFrames`
* [ ] bundler nodes:

  * `cpg_nodes__syntax_inputs`
  * `cpg_nodes__py_inputs`
  * `cpg_nodes__inspect_inputs`
  * `cpg_nodes__core_inputs`
  * `cpg_nodes__graph_inputs`
  * `cpg_nodes__inputs`

### B) Lazyframe coercion helpers

* [ ] `_core_lazyframes`, `_graph_lazyframes`

### C) Node plane converters

* [ ] `_syntax_node_keys`, `_syntax_nodes_to_cpg`
* [ ] `_ast_nodes_to_cpg`
* [ ] `_scip_symbols_to_cpg`
* [ ] `_goids_to_cpg`
* [ ] `_cfg_block_index`, `_block_id_index`, `_syntax_node_index`
* [ ] `_cfg_blocks_to_cpg`
* [ ] `_import_modules_to_cpg`
* [ ] `_ts_tokens_to_cpg`, `_ts_trivia_to_cpg`
* [ ] `_py_sym_scopes_to_cpg`, `_py_sym_bindings_to_cpg`
* [ ] `_py_bc_code_units_to_cpg`, `_py_bc_instructions_to_cpg`, `_py_bc_blocks_to_cpg`
* [ ] `_py_inspect_objects_to_cpg`, `_py_inspect_signatures_to_cpg`, `_py_inspect_signature_params_to_cpg`

### D) Main Hamilton node

* [ ] `cpg_nodes(env: BuildEnv, cpg_nodes__inputs: _CpgNodeInputs) -> InferableTabularInput`

**Imports to change/add in `cpg2/nodes.py`:**

* [ ] Replace imports that pointed to `_legacy` with:

  * `from .constants import CPG_NODES_TABLE_KEY, ...`
  * `from .foundation import _frame_to_reader, _select_node_columns, _node_rows_to_table, ...`
* [ ] Keep existing Arrow utilities imports the same as legacy:

  * `tabular_to_table`, `concat_tables_unified`, `dedupe_table_for_table`, `align_table_to_contract`, `empty_table_for_table`, etc.
* [ ] Keep `SpanResolver` import if still used by `_ast_nodes_to_cpg`.

**Exports (`__all__`) in `cpg2/nodes.py`:**
Match the names your wrappers need:

```python
__all__ = [
    "CPG_NODES_TABLE_KEY",
    "cpg_nodes",
    "cpg_nodes__syntax_inputs",
    "cpg_nodes__py_inputs",
    "cpg_nodes__inspect_inputs",
    "cpg_nodes__core_inputs",
    "cpg_nodes__graph_inputs",
    "cpg_nodes__inputs",
]
```

---

## 1.6 `.../cpg/cpg2/edges_overlays/` package (NEW)

Create a small overlay package to keep `cpg2/edges.py` readable.

### 1.6.1 `.../cpg/cpg2/edges_overlays/__init__.py` (NEW)

* [ ] Create file with minimal docstring.
* [ ] `__all__ = ["symtable", "bytecode", "inspect"]` (optional).

---

### 1.6.2 `.../cpg/cpg2/edges_overlays/symtable.py` (NEW)

**Cut/paste from `_legacy.py`:**

* [ ] `_py_sym_scope_edges_to_cpg`
* [ ] `_py_sym_namespace_edges_to_cpg` + `_namespace_edge_row`
* [ ] `_py_sym_binding_edges_to_cpg`
* [ ] `_py_sym_resolution_edges_to_cpg` + `_py_sym_resolution_edge_row`
* [ ] `_py_sym_binding_symbol_edges_to_cpg`
* [ ] `_scope_qualname_index`, `_symbol_display_index`, `_binding_symbol_edge_rows`
* [ ] any small helper dataclasses these functions rely on (if present above them)

**Imports to add:**

* [ ] `pyarrow as pa`
* [ ] `from ..foundation import _edge_rows_to_table, _select_edge_columns, _pk_from_row, _payload_json_from_row, ...`
* [ ] `from ..constants import ... TABLE_KEY constants ...`

**Exports:**

```python
__all__ = [
    "_py_sym_scope_edges_to_cpg",
    "_py_sym_namespace_edges_to_cpg",
    "_py_sym_binding_edges_to_cpg",
    "_py_sym_resolution_edges_to_cpg",
    "_py_sym_binding_symbol_edges_to_cpg",
]
```

---

### 1.6.3 `.../cpg/cpg2/edges_overlays/bytecode.py` (NEW)

**Cut/paste from `_legacy.py`:**

* [ ] Reaching-defs / def-use core:

  * `_assign_events_to_blocks`, `_compute_reaching_defs`, `_emit_reaches_edges`, etc.
  * `_instruction_cpg_id`, `_binding_cpg_id`
  * `_py_bc_defuse_binding_edges_to_cpg`
  * `_py_bc_reaches_edges_to_cpg`
* [ ] Bytecode CFG edge mapping:

  * `_py_bc_cfg_edges_to_cpg`
* [ ] Instruction ↔ AST anchor:

  * `_bytecode_ast_anchor_edge_row`, `_py_bc_instruction_ast_edges_to_cpg`
* [ ] Callsite edges:

  * `_py_bc_callsite_edges_to_cpg`, `_py_bc_callsite_symbol_edges_to_cpg` (+ their helpers)
* [ ] Memory edges:

  * `_py_bc_memory_edges_to_cpg` (+ helpers)
* [ ] Stack edges:

  * `_py_bc_stack_edges_to_cpg` (+ the opcode stack-effect helpers)

**Imports:**

* [ ] `import opcode` (still needed)
* [ ] `pyarrow as pa`, `pyarrow.compute as pc` if used
* [ ] `from ..foundation import ...`
* [ ] `from ..constants import ...`
* [ ] Any helper imports currently in `_legacy.py` (keep them identical initially)

**Exports:**

```python
__all__ = [
    "_instruction_cpg_id",
    "_py_bc_cfg_edges_to_cpg",
    "_py_bc_defuse_binding_edges_to_cpg",
    "_py_bc_reaches_edges_to_cpg",
    "_py_bc_instruction_ast_edges_to_cpg",
    "_py_bc_callsite_edges_to_cpg",
    "_py_bc_callsite_symbol_edges_to_cpg",
    "_py_bc_memory_edges_to_cpg",
    "_py_bc_stack_edges_to_cpg",
]
```

---

### 1.6.4 `.../cpg/cpg2/edges_overlays/inspect.py` (NEW)

**Cut/paste from `_legacy.py`:**

* [ ] Arg-to-param mapping machinery:

  * `_assign_args_to_params` and its helpers (`_map_positional_arg`, `_map_keyword_arg`, etc.)
  * `_inspect_arg_to_param_edges_to_cpg`
  * `_py_inspect_signature_edges_to_cpg`
* [ ] Inspect → AST anchoring:

  * `_inspect_to_ast_edges_to_cpg` (+ helpers)
* [ ] Inspect → SCIP anchoring:

  * `_inspect_to_scip_edges_to_cpg`
* [ ] Class overlays:

  * `_py_inspect_class_mro_edges_to_cpg`
  * `_py_inspect_class_attr_edges_to_cpg`
* [ ] Runtime state overlays:

  * `_py_inspect_runtime_state_edges_to_cpg` (+ helpers)
* [ ] Unwrap:

  * `_py_inspect_unwrap_edges_to_cpg`

**Imports:**

* [ ] `import inspect` if still used
* [ ] `pyarrow as pa`
* [ ] `from ..foundation import ...`
* [ ] `from ..constants import ...`

**Exports:**

```python
__all__ = [
    "_inspect_arg_to_param_edges_to_cpg",
    "_py_inspect_signature_edges_to_cpg",
    "_inspect_to_ast_edges_to_cpg",
    "_inspect_to_scip_edges_to_cpg",
    "_py_inspect_class_mro_edges_to_cpg",
    "_py_inspect_class_attr_edges_to_cpg",
    "_py_inspect_runtime_state_edges_to_cpg",
    "_py_inspect_unwrap_edges_to_cpg",
]
```

---

## 1.7 `.../cpg/cpg2/edges.py` (NEW)

**Goal:** core edge planes + edge input bundlers + `cpg_edges`.

**Cut/paste from `_legacy.py` into `cpg2/edges.py`:**

### A) Input dataclasses and bundlers

* [ ] `_CpgSymbolInputs`, `_CpgFlowInputs`, `_CpgLinkInputs`, `_CpgCallWiringInputs`, `_CpgSyntaxNodeInputs`
* [ ] overlay input dataclasses:

  * `_CpgOverlayEdgeInputs`, `_CpgOverlayScopeInputs`, `_CpgOverlaySymbolInputs`, `_CpgOverlayBytecodeInputs`, `_CpgOverlaySyntaxCallInputs`,
  * `_CpgOverlayInspectCoreInputs`, `_CpgOverlayInspectRuntimeInputs`, `_CpgOverlayInspectInputs`
* [ ] `_CpgEdgeCoreInputs`, `_CpgOverlayRegistryEntry`
* [ ] bundler functions:

  * `cpg_edge_symbol_inputs`
  * `cpg_edge_flow_inputs`
  * `cpg_edge_link_inputs`
  * `cpg_edge_call_wiring_inputs`
  * `cpg_edge_syntax_node_inputs`
  * all the `cpg_edge_overlay_*_inputs` functions
  * `cpg_edge_overlay_inputs`
  * `cpg_edge_core_inputs`

### B) Core (non-overlay) edge planes

* [ ] `_syntax_edges_to_cpg`
* [ ] occurrence-role and SCIP wiring:

  * `_occurrence_*` helpers
  * `_scip_occurrence_edges_to_cpg`
  * `_scip_symbol_relationships_to_cpg`
  * `_scip_symbol_goid_edges_to_cpg`
* [ ] link plane:

  * `_call_graph_edges_to_cpg`
  * `_import_graph_edges_to_cpg`
* [ ] flow plane:

  * `_cfg_edges_to_cpg`
  * `_dfg_edges_to_cpg` (+ `_dfg_edge_row`)
  * `_cdg_edges_to_cpg` (+ `_cdg_edge_row`)
  * **Optional:** add `_pdg_edges_to_cpg` (see “Optional additions” below)
* [ ] call wiring plane:

  * `_call_wiring_calls_to_cpg`
  * `_call_wiring_arg_to_param_to_cpg`
  * `_call_wiring_ret_to_call_to_cpg`

### C) Overlay registry/orchestration

* [ ] `_overlay_frames(...)` (but change implementation to call overlay modules)

  * Instead of having all overlay builders in this file, import:

    * `from .edges_overlays.symtable import ...`
    * `from .edges_overlays.bytecode import ...`
    * `from .edges_overlays.inspect import ...`

### D) Main Hamilton node

* [ ] `cpg_edges(...) -> InferableTabularInput`

**Imports to change/add in `cpg2/edges.py`:**

* [ ] Replace `_legacy` references with:

  * `from .constants import CPG_EDGES_TABLE_KEY, ...`
  * `from .foundation import _frame_to_reader, _select_edge_columns, _edge_rows_to_table, ...`
* [ ] Import overlay options:

  * `from .options import cpg__overlay_options, cpg__options`
* [ ] Import overlay plane functions from `cpg2/edges_overlays/*`.

**Exports (`__all__`) in `cpg2/edges.py`:**
Match your current wrapper `cpg/edges.py`:

```python
__all__ = [
    "CPG_EDGES_TABLE_KEY",
    "cpg_edge_symbol_inputs",
    "cpg_edge_flow_inputs",
    "cpg_edge_link_inputs",
    "cpg_edge_call_wiring_inputs",
    "cpg_edge_syntax_node_inputs",
    "cpg_edge_overlay_scope_inputs",
    "cpg_edge_overlay_symbol_inputs",
    "cpg_edge_overlay_bytecode_inputs",
    "cpg_edge_overlay_syntax_call_inputs",
    "cpg_edge_overlay_inspect_core_inputs",
    "cpg_edge_overlay_inspect_runtime_inputs",
    "cpg_edge_overlay_inspect_inputs",
    "cpg_edge_overlay_inputs",
    "cpg_edge_core_inputs",
    "cpg_edges",
]
```

---

## 1.8 `.../cpg/cpg2/public.py` (NEW)

**Goal:** hold the “public wrapper” functions that other modules re-export.

**Cut/paste from `_legacy.py`:**

* [ ] `instruction_cpg_id(...) -> int` (calls `_instruction_cpg_id` from overlay bytecode module)
* [ ] `stable_cpg_id(table_key, pk) -> int` (calls foundation `_stable_cpg_id`)
* [ ] `py_bc_callsite_symbol_edges_to_cpg(...) -> pa.Table`
* [ ] `py_bc_callsite_edges_to_cpg(...) -> pa.Table`
* [ ] `py_bc_stack_edges_to_cpg(...) -> pa.Table`
* [ ] `py_inspect_unwrap_edges_to_cpg(...) -> pa.Table`
* [ ] `inspect_to_ast_edges_to_cpg(...) -> pa.Table`

**Imports:**

* [ ] `import pyarrow as pa`
* [ ] `from collections.abc import Mapping`
* [ ] `from .foundation import _stable_cpg_id`
* [ ] `from .edges_overlays.bytecode import _instruction_cpg_id, _py_bc_callsite_edges_to_cpg, _py_bc_callsite_symbol_edges_to_cpg, _py_bc_stack_edges_to_cpg`
* [ ] `from .edges_overlays.inspect import _py_inspect_unwrap_edges_to_cpg, _inspect_to_ast_edges_to_cpg`

**Exports (`__all__`) in `public.py`:**

```python
__all__ = [
    "instruction_cpg_id",
    "stable_cpg_id",
    "py_bc_callsite_symbol_edges_to_cpg",
    "py_bc_callsite_edges_to_cpg",
    "py_bc_stack_edges_to_cpg",
    "py_inspect_unwrap_edges_to_cpg",
    "inspect_to_ast_edges_to_cpg",
]
```

---

# 2) Existing files to edit (thin wrappers + stable surface)

## 2.1 `src/codeintel/build/hamilton/native/graphs/cpg/nodes.py` (EDIT)

**Current state:** aliases everything from `_legacy`.

**Edits:**

* [ ] Change import:

  * FROM: `from codeintel.build.hamilton.native.graphs.cpg import _legacy`
  * TO: `from codeintel.build.hamilton.native.graphs.cpg.cpg2 import nodes as _nodes`
* [ ] Update assignments:

  * `CPG_NODES_TABLE_KEY = _nodes.CPG_NODES_TABLE_KEY`
  * `cpg_nodes__syntax_inputs = _nodes.cpg_nodes__syntax_inputs`
  * …and so on

**Keep `__all__` EXACTLY as-is** (no changes).

---

## 2.2 `.../cpg/edges.py` (EDIT)

**Edits:**

* [ ] Change import:

  * TO: `from codeintel.build.hamilton.native.graphs.cpg.cpg2 import edges as _edges`
* [ ] Update assignments accordingly:

  * `CPG_EDGES_TABLE_KEY = _edges.CPG_EDGES_TABLE_KEY`
  * `cpg_edge_symbol_inputs = _edges.cpg_edge_symbol_inputs`
  * … etc.

**Keep `__all__` EXACTLY as-is**.

---

## 2.3 `.../cpg/ids.py` (EDIT)

**Edits:**

* [ ] Change import:

  * FROM: `_legacy`
  * TO: `from codeintel.build.hamilton.native.graphs.cpg.cpg2.public import stable_cpg_id`

**Keep `__all__` EXACTLY as-is**.

---

## 2.4 `.../cpg/bytecode.py` (EDIT)

**Edits:**

* [ ] Change import:

  * TO: `from codeintel.build.hamilton.native.graphs.cpg.cpg2.public import (instruction_cpg_id, py_bc_callsite_edges_to_cpg, py_bc_callsite_symbol_edges_to_cpg, py_bc_stack_edges_to_cpg)`

**Keep `__all__` EXACTLY as-is**.

---

## 2.5 `.../cpg/inspect_overlay.py` (EDIT)

**Edits:**

* [ ] Change import:

  * TO: `from codeintel.build.hamilton.native.graphs.cpg.cpg2.public import (inspect_to_ast_edges_to_cpg, py_inspect_unwrap_edges_to_cpg)`

**Keep `__all__` EXACTLY as-is**.

---

## 2.6 `.../cpg/__init__.py` (EDIT)

**Goal:** stop importing `_legacy` at import-time; import from `cpg2` instead.

**Edits:**

* [ ] Remove: `from codeintel.build.hamilton.native.graphs.cpg import _legacy`
* [ ] Add:

  * `from codeintel.build.hamilton.native.graphs.cpg.cpg2.constants import (... all table key constants ...)`
  * `from codeintel.build.hamilton.native.graphs.cpg.cpg2.options import cpg__options, cpg__overlay_options`

**Keep existing imports of wrapper modules:**

* keep `from codeintel.build.hamilton.native.graphs.cpg.nodes import ...`
* keep `from codeintel.build.hamilton.native.graphs.cpg.edges import ...`
* keep `from codeintel.build.hamilton.native.graphs.cpg.ids import stable_cpg_id`
* keep `from codeintel.build.hamilton.native.graphs.cpg.bytecode import ...`
* keep `from codeintel.build.hamilton.native.graphs.cpg.inspect_overlay import ...`

**Keep `__all__` EXACTLY as-is.**
(Only the *source* of constants changes.)

---

## 2.7 `.../cpg/_legacy.py` (EDIT → convert to shim)

Once wrappers point to `cpg2`, `_legacy.py` should no longer be imported by public modules.

**Two options:**

### Option A (recommended): turn `_legacy.py` into a compatibility shim

* [ ] Replace file contents with re-exports so any accidental imports still work:

  * `from .cpg2.constants import *`
  * `from .cpg2.options import *`
  * `from .cpg2.nodes import *`
  * `from .cpg2.edges import *`
  * `from .cpg2.public import *`
* [ ] Keep the existing `_legacy.py` module docstring.
* [ ] Keep the existing `__all__` list from old `_legacy.py` (or regenerate it to include the same public names).

### Option B: delete `_legacy.py`

* [ ] Only do this if `grep -R "_legacy" src/codeintel/build/hamilton/native/graphs/cpg` is empty.

---

# 3) Optional “easy adds” you can do *during* the refactor (still conservative schema)

These are safe because they **don’t change** the `graph.cpg_nodes` / `graph.cpg_edges` column contracts—only add more edge rows.

## 3.1 Include PDG edges in CPG (optional)

Your DAG already has `graph.pdg_edges` as a combination of DFG and CDG. 
If you want explicit PDG layer edges in CPG:

* [ ] In `cpg2/edges.py`, add a new input to `_CpgFlowInputs` for `pdg_edges` and wire it from `q__graph__pdg_edges`.
* [ ] Add `_pdg_edges_to_cpg(pdg_edges: pa.Table) -> pa.Table`:

  * Map to CPG edge schema with `edge_layer="FLOW"` and `edge_kind` from PDG `edge_kind` column (or hardcode `"PDG"` and store original kind in `extras_json`).
* [ ] Gate it behind `CpgOptions` if you want toggles.

---

# 4) PR-ready verification checklist

## 4.1 Zero storage coupling

* [ ] `grep -R "codeintel.storage" src/codeintel/build/hamilton/native/graphs/cpg` returns nothing.

## 4.2 Public API stable

* [ ] `python -c "from codeintel.build.hamilton.native.graphs.cpg import *"` succeeds.
* [ ] The `__all__` lists in the wrapper modules are unchanged.

## 4.3 Contracts stable

* [ ] `graph.cpg_nodes` and `graph.cpg_edges` still go through:

  * `dedupe_table_for_table(...)`
  * `align_table_to_contract(...)` / `empty_table_for_table(...)`

## 4.4 Hamilton discovery stable

* [ ] You did **not** add a new top-level graphs domain module/package named `cpg2` (keep it nested under `cpg/`).
* [ ] The Hamilton target spec in `graph_targets.py` still references `base_node="cpg_nodes"` / `"cpg_edges"` (unchanged).

---


