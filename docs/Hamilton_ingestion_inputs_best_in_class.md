
Below is a **best‑in‑class scope delta implementation plan** (narrative + representative code) that builds directly on your “improved alignment” codebase and pushes it toward the design goal you stated:

**Design goal:** *produce the richest possible, queryable code‑metadata database for AI programming models*, with:

* **Hamilton DAG** as the orchestration + inference engine
* **PyArrow** as the standardized columnar object model + storage format
* **DuckDB** as a *consumer/serving layer only* (load + query + publish), not the primary compute substrate

This plan is shaped by (a) your attached “best‑in‑class” scip/libcst/tree‑sitter design notes and (b) what’s already present in the improved alignment repo (notably: schema registry, ArrowDatasetStore, dataset manifests, DatasetRef + dataset-backed loaders, and a serving snapshot service).

---

## 1) What changes vs today (delta summary)

### A. Introduce a canonical “P0 Index Suite” contract

A single “P0 index suite” run produces a **reusable snapshot** of foundational datasets (Arrow) plus a suite manifest. After that, every other DAG run becomes:

> **Load P0 datasets → enrich/infer → export → (optionally) publish DuckDB snapshot**

This is the same conceptual direction you already identified, but we’ll implement it in a way that *plugs into the codebase’s existing DatasetRef + loader patterns* (so it’s not bolted-on). The suite manifest becomes the bridge between “expensive upfront indexing” and “fast downstream inference.”

### B. Replace “generic CST node dumping” as the *primary* interface with “syntax fact tables”

You can keep `core.cst_nodes` as a forensic/debug table, but the best-in-class plan hinges on **normalized syntax facts** (spans, scopes, defs, refs, imports, calls, etc.) extracted in one LibCST pass and emitted as **multiple tables**. Your best‑in‑class doc calls out exactly this pattern and the key metadata providers to use. 

### C. Make “deterministic joins” first-class by standardizing coordinates + span identity

To stitch LibCST ↔ Tree-sitter ↔ SCIP, you need a canonical coordinate system. Your notes explicitly call out normalizing LibCST (1‑based) positions to match a 0‑based standard for joins, and the plan must also reconcile GOIDs (currently 1‑based via AST) with SCIP (0‑based). 

### D. Add a “SCIP resolution layer” that produces explicit xref tables

You already ingest SCIP tables (`core.scip_*`). What’s missing for best-in-class is a dedicated layer that produces deterministic mappings (symbol ↔ occurrence ↔ span ↔ goid) so downstream analysis doesn’t “guess.” This should be implemented as new derived tables or an extension of `core.goid_crosswalk`, not a replacement for existing SCIP tables.

### E. Add a Tree‑sitter query pack runner as the cross-language ingestion engine

Tree‑sitter should not be “store the whole parse tree.” Best-in-class is: parse bytes → run **query packs** → emit normalized capture tables (and optionally translate them into your canonical `core.syntax_*` facts).
Tree‑sitter’s docs emphasize using **byte offsets** and using locals/injection query vocabularies to extract structured signals cleanly. 

### F. Push PyArrow harder: declared schemas + streaming + dataset scanning

Your repo already treats Arrow as storage. The delta is: treat Arrow as the *standardized data object* even more aggressively:

* declared schemas everywhere (not inferred “by accident”)
* record batch streaming for large tables
* dataset scanners for projection/filter and efficient downstream loads
  These are core PyArrow strengths (dataset/scanner/record batch reader patterns). 

### G. Align plan naming and contracts with current ingestion targets

The current ingestion targets are `modules`, `ast`, `cst`, `docstrings`, `scip`,
and `scip_proto` (Hamilton target names; implemented as `t__*` functions). This plan
should either evolve those targets or add new ones that follow the same naming
scheme, instead of introducing mismatched target names.

---

## 2) Target end-state architecture (best-in-class)

### The data plane

1. **Inputs** (repo files, SCIP index, etc.)
2. **P0 Index Suite DAG**

   * LibCST syntax facts (Python)
   * Tree-sitter captures (cross-language + embedded DSL)
   * SCIP ingestion (protobuf)
   * deterministic crosswalk/xref tables
   * stable manifests + suite manifest
3. **P1/P2 Enrichment DAGs**

   * resolve refs/calls/imports using SCIP + crosswalks
   * build graphs (call/import/symbol-use)
   * compute model-oriented tables: “symbol cards,” “evidence spans,” “semantic chunks,” etc.
4. **Serving**

   * DuckDB loads Arrow snapshots via manifests
   * publish serving snapshot manifest

### The orchestration plane

* **Hamilton**: defines targets + dependencies
* **Materializers**: persist Arrow datasets (+ manifests) and optionally publish DuckDB snapshots

---

## 3) Phase-by-phase implementation plan (with representative code)

### Phase 0 — Define “P0 Index Suite” and a suite manifest artifact

#### 0.1 Add a `DatasetSuiteManifest` model

You already have dataset manifests and serving snapshot manifests. Add a **lighter-weight**
manifest (alongside `ArrowDatasetManifest` in `codeintel/core/manifests.py`) that just
declares “this suite contains these datasets (table_key → dataset_manifest_path) for
repo/commit.” Use the existing `write_manifest_json` helper so formatting and metadata
are consistent with other manifests.

Representative model:

```python
# codeintel/core/manifests.py (near ServingSnapshotManifest)

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping

@dataclass(frozen=True, slots=True)
class DatasetSuiteManifest:
    suite_manifest_version: int
    suite_kind: str  # "p0_index_suite"
    repo: str
    commit: str
    created_at: str  # isoformat
    dataset_manifest_paths: Mapping[str, str]  # table_key -> dataset_manifest_path
    tool_versions: Mapping[str, str] | None = None
```

Rationale:

* dataset manifests already exist; this is just a bundle that downstream DAGs can consume.
* it’s “build-graph neutral” and doesn’t imply DuckDB serving (unlike serving snapshot).

#### 0.2 Implement `codeintel build bootstrap-index-suite`

This command runs the “P0 targets list” and then writes `build/bootstrap/index_suite.json`.

Your best-in-class notes already treat this as the natural operational primitive. 

**P0 target set (suggested initial):**

* `modules` (repo scan tables)
* `ast` (existing; used by GOIDs today)
* `cst` (existing; evolve this into `syntax_index` or add a new `syntax_index` target)
* `scip_proto` (required to generate `scip_pb2` bindings)
* `scip` (existing ingestion target)
* `goids` (existing)
* `scip_resolution` (new derived target)
* `tree_sitter_index` (new, can be staged later but belongs in P0)
* `parse_manifest` tables for each parser (new, required)

Representative CLI logic:

```python
# codeintel/cli/commands/build_bootstrap.py

def bootstrap_index_suite(
    repo: str,
    commit: str,
    *,
    out_path: Path = Path("build/bootstrap/index_suite.json"),
) -> None:
    # 1) run build with a fixed target list
    run_targets(
        [
            "modules",
            "ast",
            "cst",  # or "syntax_index" once introduced
            "scip_proto",
            "scip",
            "goids",
            "scip_resolution",
            "tree_sitter_index",
        ],
        persist_dataset_manifests=True,
    )

    # 2) collect per-table manifest paths
    dataset_manifest_paths = {}
    for table_key in P0_TABLE_KEYS:
        manifest_path = dataset_manifest_path(env.paths.dataset_root_dir, table_key, snapshot_id=commit)
        if manifest_path.exists():
            dataset_manifest_paths[table_key] = str(manifest_path)

    suite = DatasetSuiteManifest(
        suite_manifest_version=1,
        suite_kind="p0_index_suite",
        repo=repo,
        commit=commit,
        created_at=datetime.utcnow().isoformat(),
        dataset_manifest_paths=dataset_manifest_paths,
        tool_versions=_collect_tool_versions(),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(suite), indent=2))
```

---

### Phase 1 — Add “seed suite manifest” support the *right* way (using DatasetRef + loaders)

Your repo already has:

* `DatasetRef`
* dataset-backed loaders that load snapshots by `table_key + snapshot_id`
* support nodes that generate loader nodes (`q__…`) for downstream queries

So the best-in-class approach is:

> **Make `--seed-suite-manifest` generate DatasetRef nodes for each table_key**
> so all downstream targets can simply depend on `q__core__…` inputs without re-running P0.

#### 1.1 Add seeded DatasetRef support nodes

Your existing support node generator creates dataset nodes (`d__…`) by reading from a producer target record (which requires running the producer). For seeding, we need an alternate path: dataset nodes that are *constants* built from the suite manifest.

Representative code:

```python
# codeintel/build/hamilton/nodes/seed_suite_nodes.py

from collections.abc import Mapping, Sequence
from hamilton.function_modifiers import parameterize, resolve_from_config, value
from codeintel.build.hamilton.io.dataset_ref import DatasetRef
from codeintel.build.hamilton.naming import dataset_node

def _decorate_seeded_dataset_nodes(
    ci_seeded_datasets: Sequence[Mapping[str, str]] | None = None,
):
    mapping = {}
    for spec in (ci_seeded_datasets or []):
        table_key = spec["table_key"]
        mapping[dataset_node(table_key)] = {
            "table_key": value(table_key),
            "repo": value(spec["repo"]),
            "commit": value(spec["commit"]),
        }
    return parameterize(**mapping)

@resolve_from_config(decorate_with=_decorate_seeded_dataset_nodes)
def seeded_dataset_ref(table_key: str, repo: str, commit: str) -> DatasetRef:
    return DatasetRef(table_key=table_key, repo=repo, commit=commit, source_target="seed_suite")
```

Now your existing loader nodes (the `q__…` nodes) can load snapshots from Arrow dataset directories with `snapshot_id = commit`.

#### 1.2 Wire CLI flag `--seed-suite-manifest`

When set, the build runner:

* reads suite manifest
* populates `ci_seeded_datasets` config
* **disables** record-dependent dataset nodes (or ensures the seeded ones win)

Representative parsing:

```python
# build command handler (conceptual)

suite = DatasetSuiteManifest.parse(seed_suite_manifest_path)
config["ci_seeded_datasets"] = [
  {"table_key": k, "repo": suite.repo, "commit": suite.commit}
  for k in suite.dataset_manifest_paths.keys()
]
config["ci_support_include_dataset_nodes"] = False
config["ci_support_include_loader_nodes"] = True
```

Result:

* Downstream DAGs can run **only enrichment/export targets**
* They will still have `q__core__…` inputs, but loaded from the P0 snapshot

---

### Phase 2 — Canonical span/coordinate standardization (unlock deterministic joins)

This is the single most important correctness step for “best-in-class stitching.”

#### 2.1 Adopt a canonical coordinate standard

* **byte offsets**: `[start_byte, end_byte)` (tree-sitter and robust slicing)
* **0-based line/col**: (SCIP + tree-sitter-compatible)
* normalized everywhere

Tree-sitter guidance strongly emphasizes byte offsets as the correctness anchor.
Your best-in-class notes explicitly call out normalizing LibCST to 0-based, and
the plan should explicitly correct GOIDs (AST-derived, 1-based) into the same
0-based coordinate standard used for SCIP occurrences.

#### 2.2 Update LibCST extraction to parse bytes-first + normalize positions

LibCST supports parsing bytes (preserving encoding fidelity) and metadata wrappers
require using the wrapper’s module identity. Switch ingestion to bytes-first parsing,
capture `ByteSpanPositionProvider`, and record parse manifest fields (encoding,
indent, newline, future imports) alongside parse status.

Representative helper:

```python
def normalize_pos(pos) -> tuple[int, int]:
    # LibCST PositionProvider is 1-based line, 0-based column in many cases;
    # your best-in-class doc wants 0-based standard across everything.
    return (pos.line - 1, pos.column)

def normalize_span(span) -> dict[str, int]:
    return {
        "start_line": span.start.line - 1,
        "start_col": span.start.column,
        "end_line": span.end.line - 1,
        "end_col": span.end.column,
    }
```

#### 2.3 Add a small “line index” table (optional but best-in-class)

To convert SCIP line/col to byte offsets reliably (especially with non-ASCII), create:

* `core.file_line_index(rel_path, line, start_byte, end_byte, encoding)`

This enables deterministic:

* `scip_occurrence (line,col)` → `(start_byte,end_byte)`
* joins between tools even when encoding differs

#### 2.4 Normalize GOIDs and crosswalk coordinates

GOIDs are currently derived from AST line numbers (1-based). Add an explicit
normalization step so GOID rows and `core.goid_crosswalk` store 0-based
line/column data (or clearly encode the base in metadata if a staged migration
is required).

---

### Phase 3 — Implement `syntax_index` (LibCST syntax facts, not just nodes)

Implementation note: this should either evolve the existing `cst` target or add a
new target named `syntax_index` that follows the same Hamilton naming pattern
(function `t__syntax_index` and related boundary nodes).

Your best-in-class notes define the extraction strategy very clearly:

* parse bytes-first
* use `PositionProvider`, `ParentNodeProvider`, `QualifiedNameProvider`, `ScopeProvider`
* compute stable scope IDs
* emit normalized fact tables (spans/scopes/defs/refs/calls/imports) 

#### 3.1 Add new core tables (schemas declared up front)

At minimum (Python via LibCST), implement:

* `core.parse_manifest` (parser run results)
* `core.syntax_spans`
* `core.syntax_scopes`
* `core.syntax_defs`
* `core.syntax_refs`
* `core.syntax_imports`
* `core.syntax_calls`
* `core.call_arguments` (optional but high value)

Representative schema snippet (your schema system can express LIST/STRUCT, so avoid JSON where possible):

```python
# codeintel/core/schemas/output_registry.py (representative)

TableSchema(
  table_key="core.syntax_calls",
  columns=(
    Column("repo", ColumnType.VARCHAR),
    Column("commit", ColumnType.VARCHAR),
    Column("rel_path", ColumnType.VARCHAR),
    Column("producer", ColumnType.VARCHAR),
    Column("language", ColumnType.VARCHAR),
    Column("start_byte", ColumnType.BIGINT),
    Column("end_byte", ColumnType.BIGINT),
    Column("start_line", ColumnType.INTEGER),
    Column("start_col", ColumnType.INTEGER),
    Column("end_line", ColumnType.INTEGER),
    Column("end_col", ColumnType.INTEGER),

    Column("callee_text", ColumnType.VARCHAR),     # raw callee expression text
    Column("scope_id", ColumnType.VARCHAR),
    Column("qualified_context", ColumnType.VARCHAR, nullable=True),

    # resolution columns filled later (P1)
    Column("scip_symbol", ColumnType.VARCHAR, nullable=True),
    Column("resolved_goid_h128", ColumnType.VARCHAR, nullable=True),

    Column("extras", ColumnType.JSON, nullable=True),
  ),
  primary_key=("repo", "commit", "rel_path", "producer", "start_byte", "end_byte"),
)
```

#### 3.2 Extract all tables in one LibCST pass

Do *one* parse per file and one metadata resolution per file.

Representative extraction skeleton:

```python
import libcst as cst
from libcst.metadata import (
    MetadataWrapper,
    PositionProvider,
    ParentNodeProvider,
    QualifiedNameProvider,
    ScopeProvider,
    ByteSpanPositionProvider,
)

class SyntaxFactsVisitor(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (
        PositionProvider,
        ParentNodeProvider,
        QualifiedNameProvider,
        ScopeProvider,
        ByteSpanPositionProvider,
    )

    def __init__(self, path: str, raw: bytes):
        self.path = path
        self.raw = raw
        self.calls = []
        self.refs = []
        self.imports = []
        # etc.

    def visit_Call(self, node: cst.Call) -> None:
        pos = self.get_metadata(PositionProvider, node)
        bspan = self.get_metadata(ByteSpanPositionProvider, node)
        scope = self.get_metadata(ScopeProvider, node)

        self.calls.append({
          "repo": self.repo,
          "commit": self.commit,
          "rel_path": self.path,
          "producer": "libcst",
          "language": "python",
          "start_byte": bspan.start,
          "end_byte": bspan.end,
          "start_line": pos.start.line - 1,
          "start_col": pos.start.column,
          "end_line": pos.end.line - 1,
          "end_col": pos.end.column,
          "callee_text": cst.Module([]).code_for_node(node.func),
          "scope_id": stable_scope_id(scope),
        })
```

Key best-in-class details:

* parse **bytes-first** (encoding fidelity) and record encoding/newline/indent in parse manifest
* wrapper identity matters for metadata: use `wrapper.visit(visitor)` not some copied module object
* use `ByteSpanPositionProvider` plus `ScopeProvider`, `QualifiedNameProvider`,
  `FullyQualifiedNameProvider`, and `ExpressionContextProvider` for stable facts
* normalize to 0-based line indexing for joins
* use `FullRepoManager` for repo-wide qualified name resolution when available

#### 3.3 Emit to Arrow datasets with declared schemas (no inference-by-default)

Use your existing ArrowDatasetSaver + schema registry, but make sure these tables
are in the override registry so schema is fully declared. Keep `core.cst_nodes` as
debug/forensics, but pivot downstream joins to `core.syntax_*` tables.

---

### Phase 4 — Implement `scip_resolution` xref tables (deterministic stitching)

Your best-in-class notes explicitly call out a dedicated “SCIP resolution” step to produce deterministic join keys and xrefs.

#### 4.0 Normalize SCIP encoding + spans

Capture `Document.position_encoding` and `Metadata.text_document_encoding` during
SCIP ingestion, and store `position_encoding` alongside occurrences/diagnostics.
Add a byte-span normalization step (using `core.file_line_index`) so each
occurrence can be joined by `(rel_path, start_byte, end_byte)` when available.

#### 4.1 Add xref tables

Recommended minimal set:

* `core.scip_symbol_goid_xref`
  maps `{scip_symbol → goid_h128}` (definitions only)
* `core.scip_occurrence_span_xref`
  maps `{rel_path + (start_byte,end_byte or line/col) → scip_symbol + roles + enclosing_symbol}`

These are **derived tables**; do not replace `core.scip_*`. They can also feed
`core.goid_crosswalk` so GOID joins remain centralized.

Also: store role bitmask + derived booleans, since roles are bit flags (Definition=1, Import=2, Write=4, Read=8, etc.). 

#### 4.2 How to compute it (practical, high-precision)

1. Filter SCIP occurrences to **definitions** using roles bitmask. 
2. Join definition occurrences to your canonical def spans (from AST defs or LibCST def spans)
3. Use that to map symbol → goid

Representative Polars join sketch:

```python
defs = scip_occurrences.filter((pl.col("symbol_roles") & 1) != 0)  # Definition bit
xref = defs.join(
    goids,
    on=["rel_path", "start_line", "start_col"],  # prefer byte span if you have it
    how="left",
).select(["symbol", "goid_h128", "rel_path", "start_line", "start_col"])
```

#### 4.3 Use `enclosing_range` for robustness

SCIP occurrence `range` is often the identifier span, while the *def node span* might be larger; joining via `enclosing_range` is often the right primary key when present.

---

### Phase 5 — Implement `syntax_enrich` (resolved refs/calls/imports)

Your best-in-class notes describe a P1 step that enriches syntax refs/calls/imports by joining to the SCIP xrefs.

Deliverables:

* `core.syntax_refs_resolved`
* `core.syntax_calls_resolved`
* `core.syntax_imports_resolved`

These tables should add:

* `scip_symbol`
* `resolved_goid_h128`
* `resolution_confidence` (numeric, allows downstream heuristics)
* `resolution_source` (e.g., “scip_exact_span”, “scip_enclosing”, “heuristic”)

Representative enrichment logic:

```python
resolved_calls = syntax_calls.join(
    scip_occurrence_span_xref,
    on=["rel_path", "start_byte", "end_byte"],  # best if you normalized to bytes
    how="left"
).join(
    scip_symbol_goid_xref,
    on=["scip_symbol"],
    how="left"
).with_columns([
    pl.when(pl.col("goid_h128").is_not_null())
      .then(pl.lit("scip_exact_span"))
      .otherwise(pl.lit(None))
      .alias("resolution_source")
])
```

---

### Phase 6 — Add Tree-sitter ingestion as query packs (cross-language + embedded DSL)

Your best-in-class notes treat tree-sitter as a **query-pack emitter** that produces `core.ts_captures` and `core.ts_parse_errors`.

Tree-sitter best practices to apply:

* use **byte offsets** as primary truth
* use **locals query capture vocab** to extract defs/refs/scopes across grammars
* use **injection queries** to detect embedded languages like SQL-in-strings and parse them separately
* protect runtime with QueryCursor match limits / range limiting, and cancel via progress callbacks
* use the 0.25+ API (`Language(capsule)`, `Parser(Language)`) and check ABI compatibility

#### 6.1 New tables

* `core.parse_manifest` (reuse with `parser="tree_sitter"`)
* `core.ts_captures(repo, commit, rel_path, language, query_pack, capture_name, start_byte, end_byte, start_row, start_col, end_row, end_col, node_type, text_preview, extras)`
* `core.ts_parse_errors(repo, commit, rel_path, language, error_type, message, start_byte, end_byte, …)`

#### 6.2 Query pack organization

Repository layout recommendation:

```
codeintel/ingestion/tree_sitter/
  registry.py              # language registry, grammar loading
  runner.py                # parse + query exec
  packs/
    python/
      locals.scm
      imports.scm
      calls.scm
      injections.scm
    ts/
      locals.scm
      imports.scm
    go/
      locals.scm
      imports.scm
```

#### 6.3 Representative tree-sitter runner

```python
from tree_sitter import LANGUAGE_VERSION, MIN_COMPATIBLE_LANGUAGE_VERSION
from tree_sitter import Language, Parser, Query, QueryCursor
import tree_sitter_python as tspython

def _assert_lang_abi(lang) -> None:
    if not (MIN_COMPATIBLE_LANGUAGE_VERSION <= lang.abi_version <= LANGUAGE_VERSION):
        raise RuntimeError("Tree-sitter language ABI not supported")


LANG = Language(tspython.language())
_assert_lang_abi(LANG)
PARSER = Parser(LANG)


def run_query_pack(tree, source_bytes, pack_name, query_text):
    q = Query(LANG, query_text)
    cursor = QueryCursor(q, match_limit=10_000)
    for capture_name, nodes in cursor.captures(tree.root_node).items():
        for node in nodes:
            yield {
              "capture_name": capture_name,
              "node_type": node.type,
              "start_byte": node.start_byte,
              "end_byte": node.end_byte,
              "start_row": node.start_point.row,
              "start_col": node.start_point.column,
              "end_row": node.end_point.row,
              "end_col": node.end_point.column,
              "text_preview": source_bytes[node.start_byte:node.end_byte][:200].decode("utf-8", "replace"),
            }
```

#### 6.4 Translating tree-sitter captures into your canonical syntax tables

For best-in-class, treat tree-sitter as the cross-language *producer* of the same canonical tables:

* `core.syntax_scopes` (from locals queries)
* `core.syntax_defs` (defs)
* `core.syntax_refs` (refs)
* `core.syntax_imports` (imports)
* `core.syntax_calls` (calls)

This gives you one unified interface to downstream inference, regardless of language.

---

### Phase 7 — PyArrow “full advantage” upgrades (schema, streaming, scanning)

Your best-in-class goal explicitly calls out exploiting Arrow more deeply. The PyArrow dataset + scanner + record-batch patterns are the key primitives. 

#### 7.1 Use `RecordBatchReader` for big outputs (streaming)

Where tables could be huge (tree-sitter captures on large monorepos, occurrence tables, edges), avoid building gigantic Python lists.

Pattern:

* generator yields RecordBatches
* ArrowDatasetStore writes dataset from reader
* schema is declared, validated, and reused

#### 7.2 Standardize dataset scanning for downstream loads

For downstream DAGs, prefer:

* pyarrow.dataset scanner (projection + filter)
* polars lazy scan from parquet directory when appropriate
* avoid loading entire tables unless needed

This is already consistent with your current loader infrastructure; the delta is: make it the **default** for enrichment.

---

## 4) “Best-in-class” table/target map (what to implement)

This aligns with your best‑in‑class notes:

### P0

* `cst` (evolve to `syntax_index`) → `core.syntax_spans/scopes/defs/refs/calls/imports`
  (plus `core.parse_manifest`)
* `scip_proto` → protobuf bindings (`scip_pb2`)
* `scip` → `core.scip_*` ingestion tables
* `goids` → `core.goids` + `core.goid_crosswalk`
* `scip_resolution` → derived xref tables (and/or crosswalk extensions)
* `tree_sitter_index` → `core.ts_captures/ts_parse_errors` (+ `core.parse_manifest`)
* `bootstrap-index-suite` → suite manifest artifact

### P1

* `syntax_enrich` → resolved refs/calls/imports

### P2 (optional but “best-in-class”)

* embedded DSL parsing (SQL, regex, bash) from injection captures
* “symbol cards” table: definition span + docstring + signature + references summary + graph neighborhood
* LLM evidence chunking tables: snippet windows + provenance

---

## 5) Practical sequencing (how to ship this without destabilizing)

### Milestone 1 (fastest path to “meaningful best-in-class”)

1. `DatasetSuiteManifest` + `bootstrap-index-suite`
2. `--seed-suite-manifest` support via seeded DatasetRef nodes
3. coordinate normalization + add bytes where missing
4. `syntax_index` (LibCST) producing **calls/imports/refs/defs/scopes** (not just cst_nodes)

### Milestone 2

5. `scip_resolution` xrefs
6. `syntax_enrich` resolved tables
7. start updating graphs to prefer resolved tables

### Milestone 3

8. `tree_sitter_index` + query packs (start with 1–2 languages + injections)
9. translate tree-sitter captures → canonical `core.syntax_*` for non-Python

### Milestone 4

10. embedded DSL (SQL-in-strings etc.) + LLM-facing “symbol cards” and “evidence spans”

---

## 6) Implementation checklists (file-by-file)

### Phase 0 — P0 suite manifest + seeded loaders

- [x] `src/codeintel/core/manifests.py`: add `DatasetSuiteManifest` (version/kind, dataset_manifest_paths) and JSON helpers.
- [x] `src/codeintel/cli/commands/build.py`: add a `build bootstrap-index-suite` command entry.
- [x] `src/codeintel/cli/handlers/build.py`: implement handler to run the P0 target list and write the suite manifest.
- [x] `src/codeintel/build/config.py`: add config fields for `seed_suite_manifest_path` and/or `ci_seeded_datasets`.
- [x] `src/codeintel/build/hamilton/nodes/support_nodes.py`: add seeded `DatasetRef` node generation (prefer seeded refs).
- [x] `src/codeintel/build/hamilton/nodes/support_spec.py`: wire `ci_seeded_datasets` + node include flags into config.
- [x] `src/codeintel/build/hamilton/native/patterns/loaders.py`: accept seeded `DatasetRef` inputs.

### Phase 1 — Schema registry + parse manifest + syntax facts (LibCST)

- [x] `src/codeintel/core/schemas/output_registry.py`: add `core.parse_manifest` and `core.syntax_*` table schemas.
- [x] `src/codeintel/core/schemas/generated_rows/*`: regenerate row models for new tables.
- [x] `src/codeintel/ingestion/compute/cst_extract.py`: switch to bytes-first parsing and emit parse manifest rows.
- [x] `src/codeintel/ingestion/infrastructure/cst_utils.py`: add byte spans and normalize to 0-based line/col.
- [x] `src/codeintel/build/hamilton/native/ingestion/extraction_targets.py`: evolve `cst` target or add `syntax_index` target to materialize `core.syntax_*` + `core.parse_manifest`.
- [x] `src/codeintel/build/hamilton/native/ingestion/__init__.py`: export the new target(s).

### Phase 2 — Coordinate normalization + GOID alignment

- [ ] `src/codeintel/ingestion/compute/ast_extract.py`: normalize AST line base to 0-based for GOID joins.
- [ ] `src/codeintel/build/hamilton/native/graphs/goids.py`: adjust GOID and crosswalk rows to the 0-based contract.
- [ ] `src/codeintel/core/schemas/output_registry.py`: add `core.file_line_index` schema (if adopted).
- [ ] `src/codeintel/build/hamilton/native/ingestion/file_line_index.py`: add a target to materialize `core.file_line_index`.

### Phase 3 — SCIP ingestion upgrades (encoding + streaming)

- [x] `src/codeintel/ingestion/scip/protobuf_parser.py`: add streaming decode (metadata first) and capture encoding fields.
- [x] `src/codeintel/ingestion/ports/tools.py`: extend `ScipDocument` / `ScipOccurrence` with encoding metadata.
- [x] `src/codeintel/ingestion/scip/models.py`: add metadata fields needed for schema alignment.
- [x] `src/codeintel/ingestion/scip/rows.py`: emit `position_encoding`, `text_document_encoding`, and nullable byte spans.
- [x] `src/codeintel/ingestion/engine/scip.py`: thread new metadata into `ScipDocument`/`ScipOccurrence`.
- [x] `src/codeintel/build/hamilton/native/ingestion/scip.py`: plumb new columns into materialized tables.

### Phase 4 — `scip_resolution` xref targets

- [ ] `src/codeintel/core/schemas/output_registry.py`: add `core.scip_symbol_goid_xref` + `core.scip_occurrence_span_xref`.
- [ ] `src/codeintel/core/schemas/generated_rows/*`: regenerate row models.
- [ ] `src/codeintel/build/hamilton/native/ingestion/scip_resolution.py`: new target building xref tables (Polars joins).
- [ ] `src/codeintel/build/hamilton/native/ingestion/__init__.py`: export the new target.

### Phase 5 — Tree-sitter ingestion targets

- [ ] `src/codeintel/ingestion/tree_sitter/registry.py`: language registry + ABI checks.
- [ ] `src/codeintel/ingestion/tree_sitter/runner.py`: parse bytes, run query packs, capture errors.
- [ ] `src/codeintel/ingestion/tree_sitter/packs/*`: query pack definitions by language.
- [ ] `src/codeintel/core/schemas/output_registry.py`: add `core.ts_captures` + `core.ts_parse_errors`.
- [ ] `src/codeintel/build/hamilton/native/ingestion/tree_sitter.py`: new target to materialize tree-sitter tables.
- [ ] `src/codeintel/build/hamilton/native/ingestion/__init__.py`: export the new target.

### Phase 6 — Arrow streaming + dataset scanning defaults

- [ ] `src/codeintel/build/hamilton/materializers/arrow_dataset_saver.py`: accept `RecordBatchReader` and stream writes.
- [ ] `src/codeintel/storage/datasets/arrow_store.py`: prefer dataset scanner + projection pushdown.
- [ ] `src/codeintel/build/hamilton/native/patterns/loaders.py`: default to scanner-based loads for enrichment.

---

## 7) Acceptance criteria (so engineers know “done”)

### Index suite correctness

* Running `bootstrap-index-suite` produces:

  * Arrow datasets for each P0 table
  * dataset manifests
  * one suite manifest listing them

### Seeded enrichment correctness

* Running `build run --seed-suite-manifest build/bootstrap/index_suite.json --targets syntax_enrich`:

  * does **not** re-run P0 compute
  * loads P0 datasets via snapshot loaders
  * produces enriched resolved tables

### Stitching correctness

* For Python codebases with SCIP:

  * ≥95% of call sites / name uses that have SCIP occurrences get `scip_symbol`
  * ≥X% of those resolve to internal `goid_h128` via xref
  * deltas are measurable via parse_manifest + counts

---

## 7) Why this is “still inference-driven” (but practical)

You’re not abandoning inference; you’re making inference *composable*:

* **P0** gives you a grounded, schema-declared, multi-tool factual substrate (syntax + symbols + occurrences)
* **P1+** is where Hamilton shines: resolution, graph inference, heuristics, enrichment, exports
* The suite manifest turns “high upfront indexing” into a reusable cached foundation

That’s exactly the intent of your best‑in‑class notes: build stable, deterministic fact tables and then derive everything else from them. 

---

If you want the next deliverable after this plan (still narrative-first), the most useful would be a **concrete “P0 Index Suite contract” document** listing:

* exact target names
* exact table keys
* required columns + primary keys (especially for span/join identity)
* which columns must be 0-based vs byte-based
* and a small example suite manifest JSON

That document tends to eliminate 80% of implementation ambiguity for teams working in parallel.

According to a document from your attached best‑in‑class design notes (timestamp not available in file metadata), here is a **concrete P0 Index Suite contract** that teams can implement against in parallel with minimal ambiguity.

This contract is intentionally “hard‑edged”: **exact target names**, **exact table keys**, **required columns + primary keys**, **coordinate base rules**, and a **small example suite manifest JSON**.

---

# P0 Index Suite Contract v1

## Design goal

**Goal:** produce a deterministic, schema-declared “P0 snapshot” of *code intelligence facts* (SCIP + syntax facts from LibCST / tree-sitter) as **Arrow datasets**, then make all downstream Hamilton DAGs operate as:

> **load P0 datasets → enrich → export**

This keeps heavy parsing/indexing costs out of iterative enrichment runs and makes the data contract stable for LLM-facing metadata services.

---

## 1) Exact target names in the P0 suite

### 1.1 Orchestrator target

* **`bootstrap/index_suite`**
  Produces the suite manifest artifact and guarantees that every P0 table listed below exists (even if empty for unsupported languages).

This is the “one command” that downstream DAGs will reference via `--seed-suite-manifest …`.

### 1.2 Required P0 producer targets

These are the targets the orchestrator runs (directly or indirectly):

1. **`modules`**
2. **`ast`** (existing; used by GOIDs)
3. **`cst`** (existing; evolve into `syntax_index` or add a new `syntax_index` target)
4. **`scip_proto`** (required to generate `scip_pb2`)
5. **`scip`**
6. **`goids`**
7. **`scip_resolution`**
8. **`tree_sitter_index`**
9. **`parse_manifest`** (parser status tables)

The design doc also calls out the **Hamilton boundary nodes** used to enforce the “tools compute → Arrow emit” separation:

* For `syntax_index`: boundary nodes like `t__syntax_index__extract` and `t__syntax_index__emit`
* For `scip_resolution`: boundary node like `t__scip_resolution__xref`
* For `scip_proto`: `t__scip_proto__run` (codegen)

> Contract implication: **downstream DAGs must not re-run** these tool-boundary steps if a suite manifest is provided.

---

## 2) Exact table keys produced by the P0 suite

### 2.1 P0 table → owner target mapping

The best-in-class plan defines the **canonical P0 table keys** and their owning targets as:

#### Produced by `syntax_index` (P0 syntax facts)

* `core.syntax_spans`
* `core.syntax_scopes`
* `core.syntax_defs`
* `core.syntax_refs`
* `core.syntax_calls`
* `core.syntax_imports`

#### Produced by `scip_resolution` (P0 SCIP crosswalk)

* `core.scip_symbol_goid_xref`
* `core.scip_occurrence_span_xref`

These are derived tables that complement (not replace) `core.scip_*`. They may also
feed or extend `core.goid_crosswalk` to keep joins centralized.

### 2.2 Additional “must-have” P0 table

Separately, the design specifies a P0 parser status table:

* `core.parse_manifest`

**Contract decision:** `core.parse_manifest` is **required** and owned by `syntax_index` (it is how you measure parser coverage and safely gate downstream enrichments).

---

## 3) Coordinate system and join identity rules

This is the #1 place teams diverge accidentally. These rules are “contract law”.

### 3.1 Range semantics: always half-open `[start, end)`

* **All ranges are half-open**: start inclusive, end exclusive.
* SCIP occurrences explicitly use half-open ranges and encode them as **3 or 4 ints**:

  * 4: `[startLine, startCharacter, endLine, endCharacter]`
  * 3: `[startLine, startCharacter, endCharacter]` (endLine inferred = startLine)

### 3.2 0-based lines/columns (normalized)

* In P0 tables, all `*_line`, `*_col`, `*_character` fields are **0-based**.
* The `core.syntax_scopes` schema is explicitly defined as “Normalized 0-based half-open line/col…”

**LibCST gotcha:** `ParserSyntaxError.raw_line` is **1-indexed**, while `raw_column` is **0-indexed**.
Contract requirement: normalize LibCST-derived line numbers to 0-based before writing to Arrow.
GOIDs are currently derived from AST line numbers (1-based); contract requires
normalizing GOID tables (and `core.goid_crosswalk`) to 0-based as part of this work.

### 3.3 Byte-based offsets

Some producers naturally produce byte offsets:

* tree-sitter `Range` includes `start_byte` / `end_byte` and `start_point` / `end_point` (where points are 0-based logical coordinates)

**Contract rule:** if a table includes `*_byte` fields, they are:

* **0-based**
* **half-open**
* measured in **UTF‑8 byte offsets from file start**

### 3.4 SCIP “character” units are encoding-dependent

SCIP “character” offsets depend on `Document.position_encoding`:

* UTF16 for JVM/.NET/JS/TS indexers
* UTF32 for Python indexers
* UTF8 byte offsets for Go/Rust/C++ indexers

**Contract requirement:** `core.scip_occurrences` and the derived xref tables must
carry enough info (at minimum `position_encoding`) to interpret their character
offsets correctly, and capture `Metadata.text_document_encoding` so byte offsets
can be computed reliably for non-UTF-8 files.

### 3.5 Join identity rule (span/join keys)

For every table row that represents a “thing at a span”, we require:

* a deterministic ID (`*_id`) that is stable across reruns given the same repo+commit+rel_path and normalized range
* plus the normalized range fields themselves (so joins can be done without hashing, and for debugging)

Recommended ID recipe (representative snippet):

```python
import hashlib, json

def stable_id(*parts: object) -> str:
    b = json.dumps(parts, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.blake2b(b, digest_size=16).hexdigest()

# Example: a span identity
span_id = stable_id("span", repo, commit, rel_path, producer, start_line, start_col, end_line, end_col)
```

---

## 4) Table contracts (required columns + primary keys)

Below are the **canonical minimal schemas**. Teams may add columns, but must not remove/rename required columns without bumping schema versions.

### 4.1 `core.parse_manifest` (required)

**Purpose:** for each file + producer, record parse success/failure and error location.

**Primary key:** `(repo, commit, rel_path, producer)`

**Required columns:**

* `repo` (string)
* `commit` (string)
* `rel_path` (string)
* `producer` (string; allowed values at minimum: `"libcst"`, `"tree_sitter"`)
* `parse_ok` (bool)
* `error_kind` (string, nullable)
* `error_message` (string, nullable)
* `error_line` (int32, nullable, **0-based**)
* `error_col` (int32, nullable, **0-based**)
* `error_snippet` (string, nullable)

**0-based vs byte-based:**

* `error_line`, `error_col` are 0-based
* no byte offsets required here

---

### 4.2 `core.syntax_scopes` (required)

**Purpose:** hierarchical scope tree (module/class/function/lambda/etc.) per file per producer.

**Primary key:** `(repo, commit, rel_path, producer, scope_id)`

**Required columns:**

* `repo` (string)
* `commit` (string)
* `rel_path` (string)
* `producer` (string; `"libcst"` or `"tree_sitter"`)
* `scope_id` (string; deterministic)
* `scope_kind` (string)
* `start_line` (int32, 0-based)
* `start_col` (int32, 0-based)
* `end_line` (int32, 0-based)
* `end_col` (int32, 0-based)
* `parent_scope_id` (string, nullable)

**0-based vs byte-based:**

* `start_*` / `end_*` are 0-based line/col, half-open

---

### 4.3 `core.syntax_spans` (required)

**Purpose:** a shared, deduplicated “span inventory” so defs/refs/calls/imports can reference spans by ID (and downstream join-by-span becomes cheap).

**Primary key:** `(repo, commit, rel_path, producer, span_id)`

**Required columns:**

* `repo` (string)
* `commit` (string)
* `rel_path` (string)
* `producer` (string; `"libcst"` or `"tree_sitter"`)
* `span_id` (string; deterministic)
* `span_kind` (string; e.g. `"identifier"|"attribute"|"call_expr"|"import_stmt"|...`)
* `start_line` (int32, 0-based)
* `start_col` (int32, 0-based)
* `end_line` (int32, 0-based)
* `end_col` (int32, 0-based)
* `start_byte` (int64, 0-based, nullable allowed)
* `end_byte` (int64, 0-based, nullable allowed)

**0-based vs byte-based:**

* `*_line`, `*_col`: 0-based
* `*_byte`: UTF‑8 byte offsets (0-based), half-open
  (tree-sitter natively provides `start_byte`/`end_byte`)

---

### 4.4 `core.syntax_defs` (required)

**Purpose:** definition/binding sites discovered by syntax tools (names introduced).

**Primary key:** `(repo, commit, rel_path, producer, def_id)`

**Required columns:**

* `repo`, `commit`, `rel_path`, `producer`
* `def_id` (string; deterministic)
* `scope_id` (string; must exist in `core.syntax_scopes`)
* `span_id` (string; must exist in `core.syntax_spans`)
* `def_kind` (string; e.g. `"function"|"class"|"param"|"local"|"import_alias"|...`)
* `name` (string; identifier text)
* `start_line`, `start_col`, `end_line`, `end_col` (int32; redundant but required for easy joins)
* `start_byte`, `end_byte` (int64; nullable allowed)

**0-based vs byte-based:** same as `core.syntax_spans`.

---

### 4.5 `core.syntax_refs` (required)

**Purpose:** name-use/reference sites (names used, not introduced).

**Primary key:** `(repo, commit, rel_path, producer, ref_id)`

**Required columns:**

* `repo`, `commit`, `rel_path`, `producer`
* `ref_id` (string; deterministic)
* `scope_id` (string; must exist in `core.syntax_scopes`)
* `span_id` (string; must exist in `core.syntax_spans`)
* `ref_kind` (string; `"identifier"|"attribute"|...`)
* `name` (string; used identifier text)
* `start_line`, `start_col`, `end_line`, `end_col` (int32)
* `start_byte`, `end_byte` (int64; nullable allowed)

---

### 4.6 `core.syntax_calls` (required)

**Purpose:** call sites (function/method invocation facts).

**Primary key:** `(repo, commit, rel_path, producer, call_id)`

**Required columns:**

* `repo`, `commit`, `rel_path`, `producer`
* `call_id` (string; deterministic)
* `scope_id` (string)
* `span_id` (string; call-expression span)
* `callee_span_id` (string, nullable; span of the callee expression if extractable)
* `callee_text` (string, nullable; best-effort textual form)
* `arg_count` (int32, nullable)
* `start_line`, `start_col`, `end_line`, `end_col` (int32)
* `start_byte`, `end_byte` (int64; nullable)

---

### 4.7 `core.syntax_imports` (required)

**Purpose:** import facts (one row per imported name).

**Primary key:** `(repo, commit, rel_path, producer, import_id)`

**Required columns:**

* `repo`, `commit`, `rel_path`, `producer`
* `import_id` (string; deterministic)
* `scope_id` (string)
* `span_id` (string; span of the import clause or imported name)
* `import_kind` (string; `"import"|"from_import"`)
* `module` (string, nullable; e.g. `"pkg.subpkg"`)
* `name` (string, nullable; imported name)
* `alias` (string, nullable)
* `level` (int32, nullable; relative import level)
* `start_line`, `start_col`, `end_line`, `end_col` (int32)
* `start_byte`, `end_byte` (int64; nullable)

---

### 4.8 `core.scip_symbol_goid_xref` (required)

**Purpose:** stable mapping from SCIP symbols to your internal definition identity (GOID) + definition location.

**Primary key:** `(repo, commit, scip_symbol)`

**Required columns:**

* `repo` (string)
* `commit` (string)
* `scip_symbol` (string)
* `goid_h128` (string or int64; **your canonical global ID**)
* `def_rel_path` (string, nullable)
* `def_start_line` (int32, 0-based, nullable)
* `def_start_character` (int32, 0-based, nullable)
* `def_end_line` (int32, 0-based, nullable)
* `def_end_character` (int32, 0-based, nullable)
* `position_encoding` (string or int32 enum; required if any `*_character` fields present)
* `text_document_encoding` (string, nullable)

**Important base note:** the design explicitly highlights current base mismatches with GOIDs (“goids start_line is currently 1-based”), requiring conversion until unified. A representative join in the plan uses `def_start_line_0 + 1 = goids.start_line`.

---

### 4.9 `core.scip_occurrence_span_xref` (required)

**Purpose:** stable occurrence-level join table (SCIP occurrence → symbol → goid), used to connect syntax spans to semantic entities.

**Primary key:** `(repo, commit, occurrence_id)`

**Required columns:**

* `repo` (string)
* `commit` (string)
* `rel_path` (string)
* `occurrence_id` (string; deterministic)
* `scip_symbol` (string, nullable)
* `symbol_roles` (int32, nullable; bitset)
* `start_line` (int32, 0-based)
* `start_character` (int32, 0-based)
* `end_line` (int32, 0-based)
* `end_character` (int32, 0-based)
* `position_encoding` (string or int32 enum; required)
* `text_document_encoding` (string, nullable)
* `start_byte` (int64, nullable; when byte spans are computed)
* `end_byte` (int64, nullable; when byte spans are computed)
* `goid_h128` (string or int64, nullable; resolved if possible)

**Range normalization requirement:** support both 3-int and 4-int SCIP range encodings; normalize by inferring `end_line = start_line` when needed.

Representative normalization snippet:

```python
from typing import NamedTuple

class Range4(NamedTuple):
    start_line: int
    start_char: int
    end_line: int
    end_char: int

def normalize_range(r: list[int]) -> Range4:
    if len(r) == 3:
        sl, sc, ec = r
        return Range4(sl, sc, sl, ec)
    if len(r) == 4:
        sl, sc, el, ec = r
        return Range4(sl, sc, el, ec)
    raise ValueError(r)
```

(This is consistent with the ingestion guidance in the SCIP parsing notes.)

---

## 5) Example suite manifest JSON (small)

This is the artifact written by `bootstrap/index_suite`. It is intentionally tiny: downstream only needs the dataset manifest paths and minimal provenance.

```json
{
  "suite_manifest_version": 1,
  "suite_kind": "p0_index_suite",
  "created_at": "2026-01-01T00:00:00Z",
  "repo": "my-org/my-repo",
  "commit": "8c0c8f1c9b2d3a4e5f...",
  "targets_ran": [
    "modules",
    "ast",
    "cst",
    "scip_proto",
    "scip",
    "goids",
    "scip_resolution",
    "tree_sitter_index"
  ],
  "dataset_manifest_paths": {
    "core.parse_manifest": "build/datasets/core.parse_manifest/manifest.json",
    "core.syntax_spans": "build/datasets/core.syntax_spans/manifest.json",
    "core.syntax_scopes": "build/datasets/core.syntax_scopes/manifest.json",
    "core.syntax_defs": "build/datasets/core.syntax_defs/manifest.json",
    "core.syntax_refs": "build/datasets/core.syntax_refs/manifest.json",
    "core.syntax_calls": "build/datasets/core.syntax_calls/manifest.json",
    "core.syntax_imports": "build/datasets/core.syntax_imports/manifest.json",
    "core.goids": "build/datasets/core.goids/manifest.json",
    "core.goid_crosswalk": "build/datasets/core.goid_crosswalk/manifest.json",
    "core.scip_symbols": "build/datasets/core.scip_symbols/manifest.json",
    "core.scip_occurrences": "build/datasets/core.scip_occurrences/manifest.json",
    "core.scip_symbol_information": "build/datasets/core.scip_symbol_information/manifest.json",
    "core.scip_symbol_relationships": "build/datasets/core.scip_symbol_relationships/manifest.json",
    "core.scip_diagnostics": "build/datasets/core.scip_diagnostics/manifest.json",
    "core.scip_external_symbols": "build/datasets/core.scip_external_symbols/manifest.json",
    "core.scip_module_state": "build/datasets/core.scip_module_state/manifest.json",
    "core.scip_symbol_goid_xref": "build/datasets/core.scip_symbol_goid_xref/manifest.json",
    "core.scip_occurrence_span_xref": "build/datasets/core.scip_occurrence_span_xref/manifest.json",
    "core.ts_captures": "build/datasets/core.ts_captures/manifest.json",
    "core.ts_parse_errors": "build/datasets/core.ts_parse_errors/manifest.json"
  }
}
```

**Contract requirement:** downstream loaders must treat this manifest as authoritative—if a table key exists in `dataset_manifest_paths`, it must be loadable as an Arrow dataset and queryable in DuckDB.

---

## 6) Practical notes that prevent “parallel implementation drift”

1. **Do not invent new coordinate bases per table.**
   Everything in P0 is 0-based and half-open; encoding-dependent SCIP character offsets must carry `position_encoding`.

2. **`producer` is not optional for syntax tables.**
   The design expects `producer='libcst'|'tree_sitter'` for syntax facts, enabling multi-producer comparisons and fallbacks.

3. **If GOIDs remain 1-based temporarily, the conversion is explicit and localized.**
   The plan calls out the mismatch and shows the join conversion pattern.

---

If you want the next drill-down after *this* contract (still narrative-first), the highest ROI is usually a **“P0 test corpus + golden expectations”**: 5–10 tiny files (Python + 1 non-Python language) with expected rows for scopes/defs/refs/imports/calls + a tiny `.scip` fixture, so teams can run contract conformance tests while building independently.

Below is what I’d treat as a **best‑in‑class SCIP ingestion implementation** for your “Hamilton orchestrates, Arrow/Polars compute” architecture—**protobuf‑first**, **Arrow‑materialized**, and **Polars‑centric** for all downstream transformations.

The key idea is to make SCIP ingestion:

1. **contract-driven** (declared Arrow schemas + metadata),
2. **protobuf-native** (no JSON roundtrips), and
3. **streaming/bounded-memory** (don’t load the full `Index` into RAM unless you have to),
   while still feeding your Hamilton DAG with stable, composable datasets.

---

## Design goal for SCIP ingest in your architecture

**Design goal:** turn `index.scip` (protobuf) into a **small set of canonical Arrow datasets** that are:

* **schema-declared** and versioned (Arrow schema + schema metadata as the contract surface),
* **stable joinable** (deterministic doc/span/symbol keys),
* **streaming-producible** (bounded memory),
* and **Polars/DuckDB friendly** (Parquet/Arrow datasets; DuckDB remains a consumer/serving layer).

This aligns with SCIP’s own schema guidance: consumers should be able to process the index in a streaming fashion, and indexers should emit metadata first to enable that style of consumption. 

---

## Why protobuf-first (and not JSON) for SCIP

Your earlier “best in class” direction already called this out: **do not use `scip print --json` as a production ingest path**—it’s useful for debugging, but it throws away performance and creates an unnecessary format hop. 

Instead: generate Python bindings from `scip.proto` and parse the binary directly. 

---

## Step 1 — Protobuf bindings + version discipline

### 1A) Generate and vendor `scip_pb2`

SCIP’s `index.scip` is a serialized protobuf message of type `scip.Index`. To read it in Python, you generate classes from `scip.proto` via `protoc`. 

Representative snippet:

```bash
# one-time (or pinned in build tooling)
protoc --python_out=. scip.proto
```

Then:

```python
from scip_pb2 import Index
idx = Index()
idx.ParseFromString(open("index.scip", "rb").read())
```

That said, best-in-class for large repos is **not** to parse the entire file into one in-memory `Index` object—see Step 2.

**CodeIntel alignment:** use the existing `scip_proto` target (`t__scip_proto__run`)
to generate `scip_pb2` via `grpc_tools.protoc`, and keep the helper script
`scripts/scip_proto_codegen.sh` as the local mirror. Avoid ad-hoc codegen in
individual targets.

### 1B) Treat protobuf identity correctly

For contract stability, treat **(protobuf `full_name` + field numbers)** as the stable schema identity (not Python module layout), and avoid protobuf ABI footguns (import side effects; no subclassing generated messages). 

This matters because you’ll want to evolve your Arrow schemas safely while SCIP evolves, and you’ll want CI to detect toolchain drift.

### 1C) Keep `index.scip` as an immutable raw artifact

Even if you materialize Arrow tables, keep the original `index.scip` as a first-class artifact (for provenance + reprocessing). This also gives you forward-compat safety even if you choose to “strip unknown fields” at your Arrow boundary later (see protobuf unknown-field policy harness). 

---

## Step 2 — Best-in-class: streaming protobuf reader for `index.scip`

### Why

SCIP’s schema explicitly supports streaming consumption and recommends metadata first. 

### What to do

Implement a **wire-format streaming reader** for the top-level `Index` message that:

* reads the file sequentially,
* yields **metadata** first,
* then yields **Document** messages one at a time,
* optionally yields **external_symbols** one at a time,
* and skips unknown fields (or preserves raw bytes if you want).

This avoids the “load entire Index into memory” path.

### How (representative code)

This uses standard protobuf wire rules (tags are varints; messages are length-delimited). You can reuse ideas from your protobuf harness patterns (varint encoding/unknown policy patterns are documented there). 

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import BinaryIO, Iterator, Optional, Tuple

from scip_pb2 import Index, Metadata, Document, SymbolInformation


def _read_varint(f: BinaryIO) -> int:
    shift = 0
    out = 0
    while True:
        b = f.read(1)
        if not b:
            raise EOFError
        byte = b[0]
        out |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return out
        shift += 7


def _skip_field(f: BinaryIO, wire_type: int) -> None:
    if wire_type == 0:      # varint
        _ = _read_varint(f)
    elif wire_type == 1:    # 64-bit
        f.read(8)
    elif wire_type == 2:    # length-delimited
        n = _read_varint(f)
        f.read(n)
    elif wire_type == 5:    # 32-bit
        f.read(4)
    else:
        raise ValueError(f"Unsupported wire_type={wire_type}")


@dataclass(frozen=True)
class ScipIndexStream:
    metadata: Optional[Metadata]
    documents: Iterator[Document]
    external_symbols: Iterator[SymbolInformation]


def stream_index_scip(path: str) -> ScipIndexStream:
    # Use descriptors to avoid hardcoding field numbers.
    meta_no = Index.DESCRIPTOR.fields_by_name["metadata"].number
    docs_no = Index.DESCRIPTOR.fields_by_name["documents"].number
    ext_no = Index.DESCRIPTOR.fields_by_name["external_symbols"].number

    f = open(path, "rb")

    metadata: Optional[Metadata] = None

    def iter_documents() -> Iterator[Document]:
        nonlocal metadata
        while True:
            try:
                tag = _read_varint(f)
            except EOFError:
                return
            field_no = tag >> 3
            wire_type = tag & 0x7

            if wire_type != 2:
                _skip_field(f, wire_type)
                continue

            n = _read_varint(f)
            payload = f.read(n)

            if field_no == meta_no:
                m = Metadata()
                m.ParseFromString(payload)
                metadata = m
                continue

            if field_no == docs_no:
                d = Document()
                d.ParseFromString(payload)
                yield d
                continue

            # for now: skip everything else here
            # (external symbols handled by separate iterator if desired)
            continue

    # If you want external_symbols streaming too, you can do a second pass (or multiplex above).
    # Best-in-class: multiplex and write out external_symbols as you encounter them.
    def iter_external_symbols() -> Iterator[SymbolInformation]:
        return iter(())  # placeholder

    return ScipIndexStream(metadata=metadata, documents=iter_documents(), external_symbols=iter_external_symbols())
```

**Note:** This is intentionally “representative”. In production, you’ll likely multiplex docs + external symbols in one pass and have the downstream writer consume both streams. The core point is: **bounded memory**.

---

## Step 3 — What to extract from SCIP into Arrow datasets

SCIP’s core structures you care about (and how they map):

### 3A) Index metadata (1 row per run)

The metadata includes project root (URI-encoded absolute path), and the encoding of source files on disk (`text_document_encoding`). 

You should store:

* `run_id`
* `project_root_uri`
* `text_document_encoding`
* tool info + project name/version (if present)
* ingestion timestamp + codeintel version

### 3B) Documents (1 row per file)

Each `Document` includes `relative_path`, `language`, `occurrences`, `symbols`, and optionally `text`; plus `position_encoding` describing how occurrence character offsets are encoded. 

Best practice: **do not rely on `Document.text`**—it’s usually omitted; read file content from `project_root` + `relative_path` instead. 

Store:

* `run_id`
* `relative_path`
* `language`
* `position_encoding`
* `file_size_bytes`, `sha1`/`xxh3_64` (optional but extremely useful)
* `line_count` (optional)

### 3C) Occurrences (many rows per file)

Occurrences are where you get defs/refs and span locations.

* `Occurrence.symbol_roles` is a **bitmask**; you derive booleans like `is_definition` by testing bits. 
* `Occurrence.range` is an int list describing start/end positions; handle the “3-int vs 4-int” encoding shape. 

Store:

* `run_id`, `relative_path`
* `occ_idx` (stable per document for PK safety)
* `symbol`
* `symbol_roles` + derived booleans (definition/import/read/write/etc.)
* `syntax_kind` (keep raw + stringified)
* normalized span columns:

  * `start_line`, `start_col`, `end_line`, `end_col` (SCIP units)
  * **canonical** `start_byte`, `end_byte` (for joining with tree-sitter; see Step 4)

### 3D) SymbolInformation (definitions + docs + relationships)

`Document.symbols[]` contains the symbols “defined” in the doc (including some indirect defs via relationships), and `SymbolInformation.relationships` encode links like `is_reference`, `is_implementation`, `is_type_definition`, `is_definition`. 

Store:

* `symbol`
* `kind` + `display_name`
* documentation / signature docs (if present)
* relationships flattened into a separate table for graph traversal

### 3E) Diagnostics

Diagnostics have a defined structure: severity + code + message + source + tags. 

Even if you don’t use them immediately, capturing diagnostics is high value for:

* index quality monitoring
* LLM “why is this unresolved?” explanations

---

## Step 4 — Best-in-class join identity: compute canonical byte spans

To unify SCIP with tree-sitter and libcst, you need a **canonical span coordinate system**. Tree-sitter is byte-based; SCIP ranges are line/“character offset in some encoding”.

SCIP explicitly tells you:

* `Metadata.text_document_encoding` is how files are encoded on disk, and
* `Document.position_encoding` controls how occurrence offsets should be interpreted. 

### Recommendation

During ingest (or as a tightly-coupled P0 enrichment right after ingest), compute:

* `start_byte`, `end_byte` in the **raw file bytes** (as read from disk).

This yields a **cross-tool stable span key**:

```
span_pk = (run_id, relative_path, start_byte, end_byte)
```

…and makes SCIP ↔ tree-sitter joins deterministic.

### Representative span mapper (per document)

```python
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class SpanMapper:
    # line_start_bytes[i] = byte offset where line i starts
    line_start_bytes: List[int]
    # for each line, a mapping from codepoint index -> byte offset within that line
    # (in best-in-class implementations you compress this / compute on demand)
    line_cp_to_byte: List[List[int]]

def build_span_mapper(text: str, file_encoding: str) -> SpanMapper:
    # text is decoded using file_encoding
    lines = text.splitlines(keepends=True)
    line_start_bytes: List[int] = []
    line_cp_to_byte: List[List[int]] = []
    cursor = 0

    for ln in lines:
        line_start_bytes.append(cursor)
        # map each codepoint boundary to byte offset
        offsets = [0]
        b = 0
        for ch in ln:
            b += len(ch.encode(file_encoding, errors="replace"))
            offsets.append(b)
        line_cp_to_byte.append(offsets)
        cursor += len(ln.encode(file_encoding, errors="replace"))

    return SpanMapper(line_start_bytes=line_start_bytes, line_cp_to_byte=line_cp_to_byte)

def linecol_utf32_to_bytes(m: SpanMapper, line: int, col_cp: int) -> int:
    return m.line_start_bytes[line] + m.line_cp_to_byte[line][col_cp]
```

Then you handle `position_encoding`:

* If offsets are UTF‑32/codepoints: use directly.
* If UTF‑16: convert code-unit offsets to codepoint offsets (surrogate-aware).
* If UTF‑8: offsets may already be byte-like (but only safe if your file encoding is UTF‑8).

Even if you initially only support the encodings you see in your environment, wiring the design around `position_encoding` and `text_document_encoding` prevents subtle correctness bugs later. 

---

## Step 5 — Arrow-first materialization with schema metadata as the contract

### 5A) Declare Arrow schemas (don’t infer at runtime)

Use Arrow schema metadata as the **single contract artifact**, including PK and version tags.

Arrow supports attaching schema-level metadata via `Schema.with_metadata(...)`. 

Representative pattern:

```python
import pyarrow as pa

SCIP_OCC_SCHEMA = pa.schema([
    pa.field("run_id", pa.string()),
    pa.field("relative_path", pa.string()),
    pa.field("occ_idx", pa.int32()),
    pa.field("symbol", pa.string()),
    pa.field("symbol_roles", pa.int32()),
    pa.field("start_line", pa.int32()),
    pa.field("start_col", pa.int32()),
    pa.field("end_line", pa.int32()),
    pa.field("end_col", pa.int32()),
    pa.field("start_byte", pa.int64()),
    pa.field("end_byte", pa.int64()),
]).with_metadata({
    b"codeintel.contract": b"core.scip_occurrences@v1",
    b"codeintel.pk": b"run_id,relative_path,occ_idx",
})
```

This aligns perfectly with the “schema is the contract surface” discipline you’ve been building elsewhere.

### 5B) Write as Arrow datasets (Parquet) for interoperability

Use Arrow datasets for your lakehouse-style table-of-files. The Arrow dataset API supports partitioned Parquet writing. 

Representative write:

```python
import pyarrow.dataset as ds

ds.write_dataset(
    table,
    base_dir=out_dir,
    format="parquet",
    partitioning=ds.partitioning(pa.schema([pa.field("run_id", pa.string())]), flavor="hive"),
    file_options=ds.ParquetFileWriteOptions(compression="zstd"),
)
```

### 5C) Keep ingestion streaming-friendly

Avoid `scanner.to_table()` patterns for large scans (it materializes). Prefer `Scanner.to_batches()` / record batch streaming in downstream steps (also important when DuckDB consumes Arrow). 

---

## Step 6 — Polars-centric enrichment (after ingest)

**Keep protobuf decode → Arrow materialization as “ingestion”.**
Do your heavy “inference” (symbol parsing, role classification, relationship graph expansions, joins) in Polars, scanning the Parquet datasets.

Polars’ intended model:

* `scan_*` creates an optimizable lazy plan (predicate/projection pushdown),
* `sink_*` can execute and write in streaming/out-of-core mode. 

Representative:

```python
import polars as pl

occ = pl.scan_parquet(".../core.scip_occurrences/")
# derive booleans from roles bitmask
occ_enriched = occ.with_columns([
    (pl.col("symbol_roles") & pl.lit(1) != 0).alias("is_definition"),  # replace 1 with actual enum bit
])

# write without materializing full frame (when feasible)
occ_enriched.sink_parquet(".../derived/scip_occurrences_enriched/")
```

---

## Step 7 — Best-in-class symbol parsing (dedupe-first)

SCIP symbol strings have a grammar; naive splitting can be wrong (spaces may be escaped by doubling). The doc explicitly calls out that `" ".split(...)` is not safe. 

**Best-in-class approach:**

* Extract unique symbols from `SymbolInformation` + `Occurrence.symbol`
* Parse symbols once into a `core.scip_symbol_parts` table keyed by `symbol`
* Join symbol parts back when needed (Polars join)

This avoids parsing the same symbol millions of times.

---

## Step 8 — How this feeds Hamilton cleanly

Hamilton nodes should traffic in:

* paths to datasets,
* manifests,
* Arrow schemas (contract objects),
* Polars `LazyFrame`s (optional), but **not protobuf objects**.

SCIP ingest becomes a single “calculator” step that produces:

* `core.scip_*` datasets (Parquet dirs)
* `dataset_manifest.json` (paths, schemas, row counts, checksums)

Then Hamilton downstream DAG does:

* `pl.scan_parquet` / `pyarrow.dataset.dataset(...).scanner(...)` (depending on step)
* enrich
* export
* DuckDB “serving load”

This is directly aligned with the “P0 suite manifest → downstream loader” pattern you’ve already adopted.

---

## Practical “best-in-class” checklist for your SCIP ingest

1. ✅ **protobuf-first** parsing (no JSON) 
2. ✅ **streaming Index reader** (metadata first) 
3. ✅ **don’t rely on Document.text**; read files from disk using `project_root` + `relative_path` 
4. ✅ **normalize occurrence ranges** (3-int vs 4-int forms) 
5. ✅ **bitmask roles → derived booleans** 
6. ✅ **compute canonical byte spans** (for joining with tree-sitter) using `text_document_encoding` + `position_encoding` 
7. ✅ **Arrow schemas declared + metadata PK** (Schema.with_metadata) 
8. ✅ **Polars lazy scan + sink** for enrichment/out-of-core writes 
9. ✅ **protobuf contract discipline** (full_name + field numbers; no subclassing; import side effects) 
10. ✅ **symbol parsing is dedupe-first**; avoid naive tokenization pitfalls 

---

## If you want one concrete “next step” artifact (still narrative-first)

The highest leverage follow-on would be a **SCIP-specific Arrow contract page** (like your P0 suite contract) that lists:

* exact dataset keys for `core.scip_*`,
* required columns + PK (and “span key” rules),
* required encoding semantics (0-based, bytes vs chars),
* and which derived columns are permitted in P0 vs P1.

That document is what prevents teams from independently inventing incompatible “almost the same” SCIP tables.

If you want, I can produce that SCIP contract page next, consistent with the P0 suite contract style you already adopted.

According to a document from **January 1, 2026**, here is a **SCIP-specific Arrow contract page** (P0-level) for your **`core.scip_*`** datasets, written in the same “contract-first” style as the broader P0 index suite contract.

This is intended to be the **single source of truth** that prevents parallel teams from building “almost-compatible” SCIP tables.

---

# SCIP Arrow Contract v1

**Scope:** Canonical ingestion of a single `index.scip` protobuf (SCIP) into a **stable set of Arrow datasets** (`core.scip_*`) that downstream Hamilton DAGs can load deterministically.

## Design goal

1. **Max fidelity, minimal interpretation:** P0 must preserve the full semantic payload of SCIP (documents, occurrences, symbol info, relationships, diagnostics) with only the *minimum normalization required for interop*.
2. **Stable join keys:** Provide unambiguous keys for:

   * document identity
   * occurrence identity
   * symbol identity
   * span identity (with explicit coordinate semantics)
3. **Arrow schema = contract:** the Arrow schema (including `schema.metadata`) is the contract artifact; downstream code should not reinvent PK logic or coordinate assumptions.

SCIP’s structure and key semantics (metadata, documents, occurrences, symbol roles, relationships, diagnostics, encoding rules) are summarized in the attached SCIP deep dive. In particular: `Metadata.project_root` and `Metadata.text_document_encoding` must be respected, `Document.position_encoding` governs how to interpret occurrence offsets, and occurrence ranges can be 3-int or 4-int and must be normalized. 

---

## Contract versioning

Every dataset schema MUST include:

* `schema.metadata[b"codeintel.contract"] = b"scip_arrow@v1"`
* `schema.metadata[b"codeintel.dataset_key"] = b"<dataset_key>"`
* `schema.metadata[b"codeintel.pk"] = b"<comma-separated PK columns>"`
* `schema.metadata[b"codeintel.span.coordinate_system"]` (when spans exist; see below)

> This makes schema drift detectable via schema-only checks, and enables your “PK extraction helper” to work universally.

---

## Dataset keys (exact)

P0 SCIP ingestion produces these **dataset keys** (aligned to the current codebase):

1. `core.scip_symbols`
2. `core.scip_occurrences`
3. `core.scip_symbol_information`
4. `core.scip_symbol_relationships`
5. `core.scip_diagnostics`
6. `core.scip_external_symbols`
7. `core.scip_module_state`

Optional extensions (future, if added):

* `core.scip_metadata`
* `core.scip_documents`

> Note: `Index.external_symbols[]` is optional in SCIP but must be supported; those symbol infos flow into `core.scip_symbols` with `is_external=true`. 

---

## Standard “snapshot identity” columns (required everywhere)

All `core.scip_*` datasets MUST include these identity columns. In the current
codebase, these are `repo` + `commit` + `created_at` (timestamp). The contract
names below can be treated as aliases until a schema migration consolidates
them under a single naming scheme:

* `repo_id: string` (alias: `repo`)
  Canonical repo identity (your system-wide ID)
* `snapshot_id: string` (alias: `commit`)
  Immutable revision identity (commit SHA / content-addressed snapshot)
* `ingest_id: string` (alias: `created_at` or a run id)
  Unique run id (uuid/ulid). Allows multiple ingests per snapshot without collisions.

These are part of every dataset PK (directly or indirectly) so multi-snapshot storage is always safe.

---

# Encoding + span semantics (non-negotiable)

## 1) File encoding vs protobuf string encoding

SCIP `Metadata.text_document_encoding` specifies the encoding of **source files on disk** referenced by `Document.relative_path`, and is explicitly not the same as protobuf string encoding. 

Contract rule:

* If you ever read file contents to compute derived offsets (P1), you MUST decode using `text_document_encoding` (see P1 section).

## 2) Position encoding for offsets

`Document.position_encoding` governs the meaning of “character offsets” in occurrence ranges and depends on indexer language (Python indexers commonly use UTF32 offsets). 

Contract rule (P0):

* Store offsets exactly as SCIP provides them, and store `position_encoding` on the document row.
* Do **not** silently reinterpret offsets into some other unit in P0.

The attached guide enumerates a practical mapping:

* `1 = UTF8 byte offsets`
* `2 = UTF16 code unit offsets`
* `3 = UTF32 code unit offsets` 

## 3) Range normalization (3-int vs 4-int)

SCIP occurrence ranges are either:

* 3 ints: `[start_line, start_char, end_char]` (single-line)
* 4 ints: `[start_line, start_char, end_line, end_char]` 

Contract rule (P0):

* Persist both:

  * the raw `range` length (`range_len`)
  * the normalized 4-tuple columns (see `core.scip_occurrences`)
* Normalization must follow:

  * if 3-int: `(sl, sc, sl, ec)`
  * if 4-int: `(sl, sc, el, ec)` 

## 4) Span coordinate systems (two-tier contract)

Because SCIP offsets may be UTF8/UTF16/UTF32 code units, while tree-sitter often uses byte offsets, **we standardize spans as two distinct coordinate systems**:

### A) P0 span coordinate system (required): `scip_code_unit`

* Columns: `start_line`, `start_col`, `end_line`, `end_col`
* Meaning: **line is 0-based**; col is an offset in the document’s `position_encoding` unit.
* Schema metadata:

  * `b"codeintel.span.coordinate_system" = b"scip_code_unit"`
  * `b"codeintel.span.requires_position_encoding" = b"true"`

### B) P1 optional canonical interop system: `utf8_byte`

* Columns (optional in P1): `start_byte`, `end_byte` (0-based UTF-8 byte offsets in the file)
* Schema metadata:

  * `b"codeintel.span.coordinate_system" = b"utf8_byte"`
* Computation requires decoding file bytes and converting code-unit offsets (see the provided conversion helper patterns). 

---

# Primary keys + identity rules

## Document identity

* **Document PK** is `(repo_id, snapshot_id, doc_path)`
* `doc_path` is exactly `Document.relative_path` (no URI prefix), stored as a normalized POSIX-like relative path.

Why: SCIP defines `Document.relative_path` under `Metadata.project_root` and this is the stable per-file key inside an index. 

### Recommended acceleration ID (optional, not PK)

* `doc_id: fixed_size_binary(16)` = stable hash of `(repo_id, snapshot_id, doc_path)`
* Used for fast joins without string-heavy keys.

## Symbol identity

* **Symbol PK** is `(repo_id, snapshot_id, symbol)`
* `symbol` is the SCIP symbol string as emitted.

### Recommended acceleration ID (optional, not PK)

* `symbol_id: fixed_size_binary(16)` = stable hash of `(repo_id, snapshot_id, symbol)`

## Occurrence identity

An occurrence is inherently positioned within a document’s occurrence list.

* **Occurrence PK** is `(repo_id, snapshot_id, doc_path, occurrence_idx)`
* `occurrence_idx` is the 0-based index in `Document.occurrences[]`

This is guaranteed unique without guessing whether two occurrences can share a span.

## Span identity

For span-based joins, we define two IDs (parallel to the two coordinate systems):

### P0 span key (required)

* `span_cu_id: fixed_size_binary(16)`
* computed from: `(repo_id, snapshot_id, doc_path, position_encoding, start_line, start_col, end_line, end_col)`

### P1 interop span key (optional)

* `span_utf8_id: fixed_size_binary(16)`
* computed from: `(repo_id, snapshot_id, doc_path, start_byte, end_byte)`

---

# Table contracts

Below: **required columns**, Arrow types (suggested), **PK**, and key semantic notes.

## 1) `core.scip_metadata`

Optional extension (not implemented in current ingestion). Use this table if you
want a dedicated place for run-level metadata beyond `core.scip_module_state`.

**Grain:** 1 row per `(repo_id, snapshot_id, ingest_id)`

**PK:** `repo_id, snapshot_id, ingest_id`

Required columns:

* `repo_id: string`
* `snapshot_id: string`
* `ingest_id: string`
* `project_root_uri: string`
  SCIP `Metadata.project_root` (URI-encoded absolute path). 
* `text_document_encoding: int16`
  SCIP `Metadata.text_document_encoding`. 
* `tool_name: string`
* `tool_version: string`
* `protocol_version: string` (or int fields if your proto exposes them separately)

Optional columns (nullable):

* `project_name: string`
* `project_version: string`
* `project_namespace: string`
* `scip_index_sha256: fixed_size_binary(32)` (content hash of raw `index.scip`)

**Schema metadata:**

* `codeintel.pk = b"repo_id,snapshot_id,ingest_id"`
* `codeintel.contract = b"scip_arrow@v1"`
* `codeintel.dataset_key = b"core.scip_metadata"`

---

## 2) `core.scip_documents`

Optional extension (not implemented in current ingestion). Use this table if you
want a dedicated document registry separate from `core.scip_occurrences`.

**Grain:** 1 row per document

**PK:** `repo_id, snapshot_id, doc_path`

Required columns:

* `repo_id: string`
* `snapshot_id: string`
* `ingest_id: string`
* `doc_path: string` (`Document.relative_path`) 
* `language: string` 
* `position_encoding: int16` (`Document.position_encoding`) 
* `occurrence_count: int32`
* `symbol_count: int32`

Optional (nullable):

* `document_text: large_string` (SCIP `Document.text`, often omitted) 

Recommended (optional):

* `doc_id: fixed_size_binary(16)`

**Schema metadata:**

* `codeintel.pk = b"repo_id,snapshot_id,doc_path"`

---

## 3) `core.scip_symbol_information`

**Purpose:** best-available symbol metadata from `Document.symbols[]` (deduped by
symbol, not by document).

**Grain:** 1 row per symbol info entry (unique symbol)

**PK:** `repo_id, snapshot_id, symbol`

Required columns:

* `repo_id: string`
* `snapshot_id: string`
* `ingest_id: string`
* `symbol: string`

Optional (nullable, but recommended when present in proto):

* `documentation: list<large_string>`
* `kind: int16`
* `display_name: string`
* `signature_documentation: list<large_string>`
* `enclosing_symbol: string`

Recommended:

* `symbol_id: fixed_size_binary(16)`

**Schema metadata:**

* `codeintel.pk = b"repo_id,snapshot_id,doc_path,doc_symbol_idx"`

---

## 4) `core.scip_occurrences`

**Grain:** 1 row per occurrence entry

**PK:** `repo_id, snapshot_id, doc_path, occurrence_idx`

Required columns:

* `repo_id: string`
* `snapshot_id: string`
* `ingest_id: string`
* `doc_path: string`
* `occurrence_idx: int32` (0-based index in `Document.occurrences[]`)
* `symbol: string` (nullable; empty string should be normalized to null)
* `symbol_roles: int32` (bitmask)
* `syntax_kind: int32` (nullable / optional; treat as open-ended) 
* `position_encoding: int16` (required for interpreting `*_col` offsets)
* `text_document_encoding: string` (nullable; copy of metadata for convenience)

Range (required normalized form):

* `range_len: int8` (3 or 4)
* `start_line: int32`
* `start_col: int32`
* `end_line: int32`
* `end_col: int32`

Enclosing range (optional but strongly recommended to persist if present):

* `encl_range_len: int8` (nullable)
* `encl_start_line: int32` (nullable)
* `encl_start_col: int32` (nullable)
* `encl_end_line: int32` (nullable)
* `encl_end_col: int32` (nullable)

Override documentation (optional):

* `override_documentation: large_string` (nullable)

Diagnostics linkage:

* `diagnostic_count: int16`

Span identity (required in P0):

* `span_cu_id: fixed_size_binary(16)`

Recommended:

* `doc_id: fixed_size_binary(16)`
* `symbol_id: fixed_size_binary(16)`
* `start_byte: int64` (nullable; P1 when byte spans are computed)
* `end_byte: int64` (nullable; P1 when byte spans are computed)

**Role bit semantics:** `symbol_roles` is a bitmask; at minimum Definition=1, Import=2, Write=4, Read=8 are expected and should be derivable without heuristics. 

**Schema metadata:**

* `codeintel.pk = b"repo_id,snapshot_id,doc_path,occurrence_idx"`
* `codeintel.span.coordinate_system = b"scip_code_unit"`
* `codeintel.span.requires_position_encoding = b"true"`

---

## 5) `core.scip_symbols`

**Grain:** 1 row per symbol definition entry (deduped by path + symbol)

**PK:** `repo_id, snapshot_id, doc_path, symbol`

Required columns:

* `repo_id: string`
* `snapshot_id: string`
* `ingest_id: string`
* `doc_path: string`
* `symbol: string`
* `documentation: large_string` (nullable)

Recommended:

* `symbol_id: fixed_size_binary(16)`

**Why we carry this table:** it provides a fast symbol lookup keyed by file path; richer
metadata lives in `core.scip_symbol_information` and external refs in
`core.scip_external_symbols`.

**Schema metadata:**

* `codeintel.pk = b"repo_id,snapshot_id,doc_path,symbol"`

---

## 6) `core.scip_symbol_relationships`

**Grain:** 1 row per relationship entry on a symbol

**PK:** `repo_id, snapshot_id, symbol, related_symbol, relationship_kind`

Required columns:

* `repo_id: string`
* `snapshot_id: string`
* `ingest_id: string`
* `symbol: string` (source symbol)
* `related_symbol: string`
* `relationship_kind: string` (`"reference"|"implementation"|"type_definition"|"definition"`)

Relationship semantics are represented as a single `relationship_kind` string.

Recommended:

* `symbol_id: fixed_size_binary(16)`
* `related_symbol_id: fixed_size_binary(16)`

**Schema metadata:**

* `codeintel.pk = b"repo_id,snapshot_id,symbol,related_symbol,relationship_kind"`

---

## 7) `core.scip_diagnostics`

**Grain:** 1 row per diagnostic attached to an occurrence

**PK:** `repo_id, snapshot_id, doc_path, occurrence_idx, diagnostic_idx`

Required columns:

* `repo_id: string`
* `snapshot_id: string`
* `ingest_id: string`
* `doc_path: string`
* `occurrence_idx: int32`
* `diagnostic_idx: int16`
* `severity: int16`
* `message: large_string`

Optional:

* `code: string` (nullable)
* `source: string` (nullable)
* `tags: list<int16>` (nullable)

Diagnostic field expectations and optionality are described in the SCIP guide; `message` is the must-have field. 

**Schema metadata:**

* `codeintel.pk = b"repo_id,snapshot_id,doc_path,occurrence_idx,diagnostic_idx"`

---

## 8) `core.scip_external_symbols`

**Grain:** 1 row per external symbol reference

**PK:** `repo_id, snapshot_id, symbol`

Required columns:

* `repo_id: string`
* `snapshot_id: string`
* `ingest_id: string`
* `symbol: string`
* `package_manager: string` (nullable)
* `package_name: string` (nullable)
* `package_version: string` (nullable)

---

## 9) `core.scip_module_state`

**Grain:** 1 row per module state entry (incremental indexing)

**PK:** `repo_id, snapshot_id, doc_path`

Required columns:

* `repo_id: string`
* `snapshot_id: string`
* `doc_path: string`
* `content_hash: string`
* `options_hash: string` (nullable)
* `tool_version: string` (nullable)
* `shard_path: string`
* `updated_at: timestamp`

---

# Derived columns policy: P0 vs P1

## P0 allowed derived columns (canonicalization-only)

These are allowed in P0 because they reduce ambiguity and do not require loading file bodies:

* `range_len` + normalized `start_*`/`end_*` columns (from SCIP range normalization) 
* `span_cu_id` (hash-based identity for code-unit span)
* `doc_id`, `symbol_id` (hash acceleration IDs)
* `diagnostic_count` (count only; full records in `core.scip_diagnostics`)
* **Optional** role booleans derived strictly from `symbol_roles` bit tests:

  * `is_definition`, `is_import`, `is_read`, `is_write` (no heuristics) 

## P1 derived columns (requires file text or heavier interpretation)

These should be **P1 enrichment**, because they either require reading source files or add interpretive parsing:

* `start_byte`, `end_byte`, `span_utf8_id` (nullable in schema; requires decoding file bytes)
* `token_text` (slice from file contents)
* parsed symbol-string components (package/descriptor breakdown)
* “container resolution” (mapping enclosing_range → CST/AST scope objects)
* relationship-augmented reference sets (“synthetic references”)

---

# Representative Arrow schema snippet (one table)

Here’s an example for `core.scip_occurrences` showing **how the schema itself encodes the contract**:

```python
import pyarrow as pa

SCIP_OCCURRENCES_SCHEMA = pa.schema(
    [
        ("repo_id", pa.string()),
        ("snapshot_id", pa.string()),
        ("ingest_id", pa.string()),
        ("doc_path", pa.string()),
        ("occurrence_idx", pa.int32()),

        ("symbol", pa.string()),
        ("symbol_roles", pa.int32()),
        ("syntax_kind", pa.int32()),
        ("position_encoding", pa.int16()),
        ("text_document_encoding", pa.string()),

        ("range_len", pa.int8()),
        ("start_line", pa.int32()),
        ("start_col", pa.int32()),
        ("end_line", pa.int32()),
        ("end_col", pa.int32()),

        ("start_byte", pa.int64()),
        ("end_byte", pa.int64()),

        ("encl_range_len", pa.int8()),
        ("encl_start_line", pa.int32()),
        ("encl_start_col", pa.int32()),
        ("encl_end_line", pa.int32()),
        ("encl_end_col", pa.int32()),

        ("override_documentation", pa.large_string()),
        ("diagnostic_count", pa.int16()),

        ("span_cu_id", pa.fixed_size_binary(16)),
        ("doc_id", pa.fixed_size_binary(16)),
        ("symbol_id", pa.fixed_size_binary(16)),
    ],
    metadata={
        b"codeintel.contract": b"scip_arrow@v1",
        b"codeintel.dataset_key": b"core.scip_occurrences",
        b"codeintel.pk": b"repo_id,snapshot_id,doc_path,occurrence_idx",
        b"codeintel.span.coordinate_system": b"scip_code_unit",
        b"codeintel.span.requires_position_encoding": b"true",
    },
)
```

---

# Minimal example (rows + a “SCIP slice” of suite manifest)

## Example `core.scip_occurrences` row (illustrative)

```json
{
  "repo_id": "github.com/acme/widgets",
  "snapshot_id": "9f2c...e1a",
  "ingest_id": "01J...ULID",
  "doc_path": "src/widgets/foo.py",
  "occurrence_idx": 17,
  "symbol": "scip-python python widgets 9f2c... module/foo.function:do_thing()",
  "symbol_roles": 1,
  "syntax_kind": 0,
  "position_encoding": 3,
  "text_document_encoding": "utf-8",
  "range_len": 3,
  "start_line": 41,
  "start_col": 4,
  "end_line": 41,
  "end_col": 12,
  "start_byte": 885,
  "end_byte": 893,
  "span_cu_id": "<16 bytes>",
  "diagnostic_count": 0
}
```

## “SCIP portion” of suite manifest JSON (illustrative)

```json
{
  "tables": {
    "core.scip_symbol_information": {"dataset_manifest_path": "build/arrow/core/scip_symbol_information/_manifest.json"},
    "core.scip_occurrences": {"dataset_manifest_path": "build/arrow/core/scip_occurrences/_manifest.json"},
    "core.scip_symbols": {"dataset_manifest_path": "build/arrow/core/scip_symbols/_manifest.json"},
    "core.scip_symbol_relationships": {"dataset_manifest_path": "build/arrow/core/scip_symbol_relationships/_manifest.json"},
    "core.scip_diagnostics": {"dataset_manifest_path": "build/arrow/core/scip_diagnostics/_manifest.json"},
    "core.scip_external_symbols": {"dataset_manifest_path": "build/arrow/core/scip_external_symbols/_manifest.json"},
    "core.scip_module_state": {"dataset_manifest_path": "build/arrow/core/scip_module_state/_manifest.json"}
  }
}
```

---

# Practical “don’t let teams diverge” validation checklist

A contract conformance test for SCIP ingestion should verify:

1. All required dataset keys exist.
2. Each table schema metadata contains:

   * `codeintel.contract`, `codeintel.dataset_key`, `codeintel.pk`
3. PK columns exist and are non-null for every row.
4. `range_len in {3,4}` and normalized columns obey the normalization rule. 
5. Relationship rows preserve boolean flags as-is (no collapsing into enums). 
6. `symbol_roles` is preserved; derived booleans (if present) must match bit tests. 
7. `syntax_kind` treated as open-ended integer (do not fail on unknown values). 

---

If you want, the next most useful follow-on (still narrative-first) would be a **SCIP-specific “golden expectations” fixture** tied to this contract: a tiny `index.scip` (or a textproto equivalent) plus expected rows for these 7 tables, so teams can implement ingestion independently and prove they match the contract byte-for-byte.

Below is a **SCIP-specific “golden expectations” fixture** that is:

* **small** (2 documents, a handful of symbols/occurrences),
* **contract-surface oriented** (it exercises *exactly* the 7 `core.scip_*` tables you already expose),
* **deterministic** (fixed `repo/commit/created_at/updated_at`),
* **high coverage** for ingestion logic: **range normalization (3-int and 4-int)**, **SymbolRole bitmask**, **relationships fan-out**, **occurrence diagnostics**, and **external symbols package triple parsing**.

It uses a **textproto** representation of `index.scip` (so it’s human-reviewable and easy to edit). That aligns with protobuf’s text format workflow (parse/merge semantics) and keeps the fixture stable.

---

## 1) Fixture layout

Recommended folder layout in your repo:

```
tests/fixtures/scip_golden_v1/
  README.md
  index.textproto
  shards/
    manifest.json
  expected/
    core.scip_symbols.jsonl
    core.scip_occurrences.jsonl
    core.scip_symbol_information.jsonl
    core.scip_symbol_relationships.jsonl
    core.scip_diagnostics.jsonl
    core.scip_external_symbols.jsonl
    core.scip_module_state.jsonl
```

The **golden contract** is:

* `index.textproto` (SCIP Index payload)
* `shards/manifest.json` (for `core.scip_module_state`)
* the 7 `expected/*.jsonl` files (canonical expected rows)

---

## 2) The tiny “source corpus” this index represents (for human clarity)

You do **not** need these files for ingestion tests (the ingestion reads `index.scip/textproto`), but they help engineers sanity-check the ranges and symbols:

### `a.py`

```py
class Base:
    """Doc for Base"""
    def greet(self, name: str) -> str:
        """Doc for greet"""
        return name

def foo(x: int) -> int:
    """Doc for foo"""
    return x + 1
```

### `b.py`

```py
from a import Base, foo

class Derived(Base):
    """Doc for Derived"""
    def greet(self, name: str) -> str:
        """Doc for Derived.greet"""
        return foo(1)
```

---

## 3) `index.textproto` (the SCIP fixture)

Key SCIP semantics that this fixture intentionally exercises:

* `Index.metadata` exists and is first (streaming-friendly).
* `Metadata.project_root` is a URI-encoded absolute root; text encoding + per-document position encoding are explicit.
* `Occurrence.range` uses **3-int** and **4-int** forms; roles are a **bitmask** (Definition=1, Import=2, ReadAccess=8, etc.).
* `SymbolInformation.relationships` fan out into multiple edges (reference/implementation/type_definition/definition).
* Diagnostics are attached to an occurrence range.

Create this file:

### `tests/fixtures/scip_golden_v1/index.textproto`

```textproto
metadata {
  tool_info {
    name: "fixture-indexer"
    version: "0.0.1"
    arguments: "--fixture"
  }
  project_root: "file:///workspace/scip_golden_v1"
  text_document_encoding: UTF8
}

documents {
  relative_path: "a.py"
  language: "python"
  position_encoding: UTF32CodeUnitOffsetFromLineStart

  # class Base:
  occurrences {
    range: 0
    range: 6
    range: 10
    symbol: "scip-python python fixture 1.0 a/Base#"
    symbol_roles: 1  # Definition
  }

  # def greet(...)
  occurrences {
    range: 2
    range: 8
    range: 13
    symbol: "scip-python python fixture 1.0 a/Base#greet()."
    symbol_roles: 1  # Definition
  }

  # def foo(...)
  occurrences {
    range: 6
    range: 4
    range: 7
    symbol: "scip-python python fixture 1.0 a/foo()."
    symbol_roles: 1  # Definition
  }

  symbols {
    symbol: "scip-python python fixture 1.0 a/Base#"
    documentation: "Doc for Base"
    kind: Class
    display_name: "Base"
  }

  symbols {
    symbol: "scip-python python fixture 1.0 a/Base#greet()."
    documentation: "Doc for greet"
    kind: Method
    display_name: "greet"
    signature_documentation {
      language: "python"
      text: "def greet(self, name: str) -> str"
    }
  }

  symbols {
    symbol: "scip-python python fixture 1.0 a/foo()."
    documentation: "Doc for foo"
    kind: Function
    display_name: "foo"
    signature_documentation {
      language: "python"
      text: "def foo(x: int) -> int"
    }
  }
}

documents {
  relative_path: "b.py"
  language: "python"
  position_encoding: UTF32CodeUnitOffsetFromLineStart

  # from a import Base, foo
  occurrences {
    range: 0
    range: 14
    range: 18
    symbol: "scip-python python fixture 1.0 a/Base#"
    symbol_roles: 2  # Import
  }
  occurrences {
    range: 0
    range: 20
    range: 23
    symbol: "scip-python python fixture 1.0 a/foo()."
    symbol_roles: 2  # Import
  }

  # class Derived(Base):
  occurrences {
    # 4-int form (startLine,startCol,endLine,endCol)
    range: 2
    range: 6
    range: 2
    range: 13
    symbol: "scip-python python fixture 1.0 b/Derived#"
    symbol_roles: 1  # Definition
  }

  # Base in class Derived(Base)
  occurrences {
    range: 2
    range: 14
    range: 18
    symbol: "scip-python python fixture 1.0 a/Base#"
    symbol_roles: 8  # ReadAccess
  }

  # def greet(...) in Derived
  occurrences {
    range: 4
    range: 8
    range: 13
    symbol: "scip-python python fixture 1.0 b/Derived#greet()."
    symbol_roles: 1  # Definition
  }

  # foo(1) call
  occurrences {
    range: 6
    range: 15
    range: 18
    symbol: "scip-python python fixture 1.0 a/foo()."
    symbol_roles: 8  # ReadAccess
    diagnostics {
      severity: Error
      code: "E0001"
      message: "Example diagnostic: bad call"
      source: "fixture"
    }
  }

  symbols {
    symbol: "scip-python python fixture 1.0 b/Derived#"
    documentation: "Doc for Derived"
    kind: Class
    display_name: "Derived"
    relationships {
      symbol: "scip-python python fixture 1.0 a/Base#"
      is_implementation: true
    }
  }

  symbols {
    symbol: "scip-python python fixture 1.0 b/Derived#greet()."
    documentation: "Doc for Derived.greet"
    kind: Method
    display_name: "greet"
    relationships {
      symbol: "scip-python python fixture 1.0 a/Base#greet()."
      is_reference: true
      is_implementation: true
      is_type_definition: true
      is_definition: true
    }
  }
}

external_symbols {
  symbol: "scip-python pip requests 2.31.0 requests/get()."
  documentation: "External symbol doc (optional)"
  kind: Function
  display_name: "get"
}
```

Notes:

* This fixture uses **0-based** lines/cols and **half-open** ranges (SCIP standard), and explicitly sets a per-document position encoding (Python → UTF32 recommended).
* It includes both `range` encodings (3-int and 4-int), because your parser normalizes both.
* It uses **SymbolRole bitmask** values consistent with SCIP (Definition=1, Import=2, ReadAccess=8…).
* It ensures **relationships fan out** into multiple edges per `Relationship` message.

---

## 4) `shards/manifest.json` fixture (for `core.scip_module_state`)

This is the *only* extra artifact needed for the 7th table. It’s deterministic and tiny.

### `tests/fixtures/scip_golden_v1/shards/manifest.json`

```json
{
  "version": 1,
  "generated_at": "2026-01-01T00:00:00+00:00",
  "records": {
    "a.py": {
      "rel_path": "a.py",
      "content_hash": "aaaaaaaa",
      "options_hash": "opts1",
      "tool_version": "fixture-indexer@0.0.1",
      "shard_path": "shards/aa/aaaaaaaa__a.py.scip",
      "updated_at": "2026-01-01T00:00:00+00:00"
    },
    "b.py": {
      "rel_path": "b.py",
      "content_hash": "bbbbbbbb",
      "options_hash": "opts1",
      "tool_version": "fixture-indexer@0.0.1",
      "shard_path": "shards/bb/bbbbbbbb__b.py.scip",
      "updated_at": "2026-01-01T00:00:01+00:00"
    }
  }
}
```

---

## 5) Expected rows for the 7 tables

All expected rows below assume:

* `repo = "fixture_repo"`
* `commit = "deadbeef"`
* `created_at = "2026-01-01T00:00:00+00:00"`

### 5.1 `core.scip_symbols`

### `tests/fixtures/scip_golden_v1/expected/core.scip_symbols.jsonl`

```jsonl
{"repo":"fixture_repo","commit":"deadbeef","rel_path":"a.py","symbol":"scip-python python fixture 1.0 a/Base#","documentation":"Doc for Base","created_at":"2026-01-01T00:00:00+00:00"}
{"repo":"fixture_repo","commit":"deadbeef","rel_path":"a.py","symbol":"scip-python python fixture 1.0 a/Base#greet().","documentation":"Doc for greet","created_at":"2026-01-01T00:00:00+00:00"}
{"repo":"fixture_repo","commit":"deadbeef","rel_path":"a.py","symbol":"scip-python python fixture 1.0 a/foo().","documentation":"Doc for foo","created_at":"2026-01-01T00:00:00+00:00"}
{"repo":"fixture_repo","commit":"deadbeef","rel_path":"b.py","symbol":"scip-python python fixture 1.0 b/Derived#","documentation":"Doc for Derived","created_at":"2026-01-01T00:00:00+00:00"}
{"repo":"fixture_repo","commit":"deadbeef","rel_path":"b.py","symbol":"scip-python python fixture 1.0 b/Derived#greet().","documentation":"Doc for Derived.greet","created_at":"2026-01-01T00:00:00+00:00"}
```

---

### 5.2 `core.scip_occurrences`

### `tests/fixtures/scip_golden_v1/expected/core.scip_occurrences.jsonl`

```jsonl
{"repo":"fixture_repo","commit":"deadbeef","rel_path":"a.py","symbol":"scip-python python fixture 1.0 a/Base#","start_line":0,"start_col":6,"end_line":0,"end_col":10,"roles":1,"created_at":"2026-01-01T00:00:00+00:00"}
{"repo":"fixture_repo","commit":"deadbeef","rel_path":"a.py","symbol":"scip-python python fixture 1.0 a/Base#greet().","start_line":2,"start_col":8,"end_line":2,"end_col":13,"roles":1,"created_at":"2026-01-01T00:00:00+00:00"}
{"repo":"fixture_repo","commit":"deadbeef","rel_path":"a.py","symbol":"scip-python python fixture 1.0 a/foo().","start_line":6,"start_col":4,"end_line":6,"end_col":7,"roles":1,"created_at":"2026-01-01T00:00:00+00:00"}

{"repo":"fixture_repo","commit":"deadbeef","rel_path":"b.py","symbol":"scip-python python fixture 1.0 a/Base#","start_line":0,"start_col":14,"end_line":0,"end_col":18,"roles":2,"created_at":"2026-01-01T00:00:00+00:00"}
{"repo":"fixture_repo","commit":"deadbeef","rel_path":"b.py","symbol":"scip-python python fixture 1.0 a/foo().","start_line":0,"start_col":20,"end_line":0,"end_col":23,"roles":2,"created_at":"2026-01-01T00:00:00+00:00"}

{"repo":"fixture_repo","commit":"deadbeef","rel_path":"b.py","symbol":"scip-python python fixture 1.0 b/Derived#","start_line":2,"start_col":6,"end_line":2,"end_col":13,"roles":1,"created_at":"2026-01-01T00:00:00+00:00"}
{"repo":"fixture_repo","commit":"deadbeef","rel_path":"b.py","symbol":"scip-python python fixture 1.0 a/Base#","start_line":2,"start_col":14,"end_line":2,"end_col":18,"roles":8,"created_at":"2026-01-01T00:00:00+00:00"}

{"repo":"fixture_repo","commit":"deadbeef","rel_path":"b.py","symbol":"scip-python python fixture 1.0 b/Derived#greet().","start_line":4,"start_col":8,"end_line":4,"end_col":13,"roles":1,"created_at":"2026-01-01T00:00:00+00:00"}
{"repo":"fixture_repo","commit":"deadbeef","rel_path":"b.py","symbol":"scip-python python fixture 1.0 a/foo().","start_line":6,"start_col":15,"end_line":6,"end_col":18,"roles":8,"created_at":"2026-01-01T00:00:00+00:00"}
```

---

### 5.3 `core.scip_symbol_information`

### `tests/fixtures/scip_golden_v1/expected/core.scip_symbol_information.jsonl`

```jsonl
{"repo":"fixture_repo","commit":"deadbeef","symbol":"scip-python python fixture 1.0 a/Base#","documentation":"Doc for Base","kind":7,"display_name":"Base","signature":null,"enclosing_symbol":null,"created_at":"2026-01-01T00:00:00+00:00"}
{"repo":"fixture_repo","commit":"deadbeef","symbol":"scip-python python fixture 1.0 a/Base#greet().","documentation":"Doc for greet","kind":26,"display_name":"greet","signature":"def greet(self, name: str) -> str","enclosing_symbol":null,"created_at":"2026-01-01T00:00:00+00:00"}
{"repo":"fixture_repo","commit":"deadbeef","symbol":"scip-python python fixture 1.0 a/foo().","documentation":"Doc for foo","kind":17,"display_name":"foo","signature":"def foo(x: int) -> int","enclosing_symbol":null,"created_at":"2026-01-01T00:00:00+00:00"}
{"repo":"fixture_repo","commit":"deadbeef","symbol":"scip-python python fixture 1.0 b/Derived#","documentation":"Doc for Derived","kind":7,"display_name":"Derived","signature":null,"enclosing_symbol":null,"created_at":"2026-01-01T00:00:00+00:00"}
{"repo":"fixture_repo","commit":"deadbeef","symbol":"scip-python python fixture 1.0 b/Derived#greet().","documentation":"Doc for Derived.greet","kind":26,"display_name":"greet","signature":null,"enclosing_symbol":null,"created_at":"2026-01-01T00:00:00+00:00"}
```

---

### 5.4 `core.scip_symbol_relationships`

### `tests/fixtures/scip_golden_v1/expected/core.scip_symbol_relationships.jsonl`

```jsonl
{"repo":"fixture_repo","commit":"deadbeef","symbol":"scip-python python fixture 1.0 b/Derived#","related_symbol":"scip-python python fixture 1.0 a/Base#","relationship_kind":"implementation","created_at":"2026-01-01T00:00:00+00:00"}

{"repo":"fixture_repo","commit":"deadbeef","symbol":"scip-python python fixture 1.0 b/Derived#greet().","related_symbol":"scip-python python fixture 1.0 a/Base#greet().","relationship_kind":"reference","created_at":"2026-01-01T00:00:00+00:00"}
{"repo":"fixture_repo","commit":"deadbeef","symbol":"scip-python python fixture 1.0 b/Derived#greet().","related_symbol":"scip-python python fixture 1.0 a/Base#greet().","relationship_kind":"implementation","created_at":"2026-01-01T00:00:00+00:00"}
{"repo":"fixture_repo","commit":"deadbeef","symbol":"scip-python python fixture 1.0 b/Derived#greet().","related_symbol":"scip-python python fixture 1.0 a/Base#greet().","relationship_kind":"type_definition","created_at":"2026-01-01T00:00:00+00:00"}
{"repo":"fixture_repo","commit":"deadbeef","symbol":"scip-python python fixture 1.0 b/Derived#greet().","related_symbol":"scip-python python fixture 1.0 a/Base#greet().","relationship_kind":"definition","created_at":"2026-01-01T00:00:00+00:00"}
```

---

### 5.5 `core.scip_diagnostics`

### `tests/fixtures/scip_golden_v1/expected/core.scip_diagnostics.jsonl`

```jsonl
{"repo":"fixture_repo","commit":"deadbeef","rel_path":"b.py","start_line":6,"start_col":15,"end_line":6,"end_col":18,"severity":"Error","code":"E0001","message":"Example diagnostic: bad call","source":"fixture","created_at":"2026-01-01T00:00:00+00:00"}
```

---

### 5.6 `core.scip_external_symbols`

### `tests/fixtures/scip_golden_v1/expected/core.scip_external_symbols.jsonl`

```jsonl
{"repo":"fixture_repo","commit":"deadbeef","symbol":"scip-python pip requests 2.31.0 requests/get().","package_manager":"pip","package_name":"requests","package_version":"2.31.0","created_at":"2026-01-01T00:00:00+00:00"}
```

---

### 5.7 `core.scip_module_state`

### `tests/fixtures/scip_golden_v1/expected/core.scip_module_state.jsonl`

```jsonl
{"repo":"fixture_repo","commit":"deadbeef","rel_path":"a.py","content_hash":"aaaaaaaa","options_hash":"opts1","tool_version":"fixture-indexer@0.0.1","shard_path":"shards/aa/aaaaaaaa__a.py.scip","updated_at":"2026-01-01T00:00:00+00:00"}
{"repo":"fixture_repo","commit":"deadbeef","rel_path":"b.py","content_hash":"bbbbbbbb","options_hash":"opts1","tool_version":"fixture-indexer@0.0.1","shard_path":"shards/bb/bbbbbbbb__b.py.scip","updated_at":"2026-01-01T00:00:01+00:00"}
```

---

## 6) How teams use this fixture to prove conformance

### A) Convert `index.textproto` → binary `index.scip` (optional)

You can keep the fixture as textproto only, or generate `index.scip` as a build artifact. The main point is: the textproto is the canonical source.

If you want to generate binary:

* parse textproto into an `Index` protobuf and `SerializeToString()`
* protobuf’s text format tooling supports this workflow cleanly

(Your repo already includes `scip.proto`; teams can generate language bindings as needed.)

### B) Run ingestion and compare to expected

In tests, set:

* `repo="fixture_repo"`, `commit="deadbeef"`
* `created_at = datetime(2026,1,1, tzinfo=UTC)` (must be pinned)

Then:

1. parse index
2. build the 6 SCIP-derived tables
3. read `shards/manifest.json` and build `core.scip_module_state`
4. compare produced Arrow tables to expected JSONL (or to expected Parquet if you decide to materialize Parquet)

---

## 7) Why this fixture eliminates ambiguity (and what it catches)

This single fixture tends to catch “silent divergence” across teams:

* **range normalization**: 3-int vs 4-int handling
* **role bitmask correctness**: Definition/Import/ReadAccess bits
* **relationship expansion**: one `Relationship` → multiple edges
* **diagnostic extraction** from `Occurrence.diagnostics`
* **external symbol package triple parsing** (scheme + 3 package tokens; consistent with SCIP symbol grammar)

---

If you’d like, I can also provide one small add-on variant fixture (**`scip_golden_v2_unicode`**) that includes a 🚀 character plus a second document using `UTF16CodeUnitOffsetFromLineStart`, specifically to force teams to confront position encoding semantics early (even if your P0 tables don’t yet persist `position_encoding`).


Below is a **“P0 test corpus + golden expectations” pack** you can drop into `tests/` so teams can build LibCST / tree-sitter / SCIP ingestion independently, while still converging on the same **P0 normalized tables** (scopes, defs, refs, imports, calls). It’s explicitly **byte-span-first** (tree-sitter and LibCST both support byte-accurate spans; and byte offsets are the correctness anchor for incremental parsing)  .

---

## 0) Design goal (why this test pack exists)

**Goal:** Make “contract conformance” mechanical: given the same tiny repo, every implementation must output the same P0 rows (or intentionally explain deltas), and **SCIP can be used as the high-certainty resolver** by span-joining syntax facts to SCIP occurrences .

Key choices:

* **Byte spans are the join key** (exact slicing, deterministic stable IDs) .
* **tree-sitter uses bytes + 0-based points**, so we standardize to those conventions  .
* **SCIP occurrence ranges are 0-based line/character, and Python indexers use UTF32 character offsets**—this matters if you slice text by range; our corpus is ASCII so offsets align cleanly .

---

## 1) Test corpus layout (7 files, Python + JS)

Put this under e.g. `tests/corpus/p0/`:

```
tests/corpus/p0/
  corpus_py/
    pkg/
      __init__.py
      a.py
      b.py
      sub/
        __init__.py
        c.py
    bad_syntax.py
  corpus_js/
    mod.js
  expected/
    scopes.csv
    defs.csv
    imports.csv
    refs.csv
    calls.csv
    scip_occurrence_assertions.jsonl
```

### Source files (exact contents)

**`tests/corpus/p0/corpus_py/pkg/__init__.py`**

```py
"""Test corpus package."""

from .a import Foo, add, X
```

**`tests/corpus/p0/corpus_py/pkg/a.py`**

```py
"""Module A: basic defs."""

X = 1


def add(x: int, y: int) -> int:
    return x + y


class Foo:
    def __init__(self, v: int):
        self.v = v

    def inc(self, n: int = 1) -> int:
        self.v += n
        return self.v


def top() -> int:
    f = Foo(X)
    return add(f.inc(), 3)
```

**`tests/corpus/p0/corpus_py/pkg/b.py`**

```py
"""Module B: imports + calls."""

from .a import Foo as AFoo, add
import math as m


def use() -> float:
    f = AFoo(2)
    z = add(f.inc(), 3)
    return m.sqrt(z)
```

**`tests/corpus/p0/corpus_py/pkg/sub/__init__.py`**

```py
```

**`tests/corpus/p0/corpus_py/pkg/sub/c.py`**

```py
"""Module C: relative import + async."""

from ..a import add


async def coro(x: int) -> int:
    return add(x, 1)


def call_coro():
    return coro(41)
```

**`tests/corpus/p0/corpus_py/bad_syntax.py`**
(intentionally invalid to ensure LibCST parse errors are materialized as structured artifacts, not logs) 

```py
def oops(:
    pass
```

**`tests/corpus/p0/corpus_js/mod.js`**

```js
// JS corpus for tree-sitter
export function mul(a, b) { return a * b; }
export const TWO = 2;
console.log(mul(TWO, 3));
```

---

## 2) Golden expectations (P0 normalized tables)

### Conventions these goldens assume

* `start_byte/end_byte` are UTF-8 byte offsets into the file contents. Byte correctness is critical for tree-sitter edit/changed_ranges workflows .
* `scope_id` is a stable string key; in production you may hash it, but **the components must be reproducible**.
* `symbol` / `callee_symbol` are “canonical symbols” (your internal namespace). SCIP symbols are verbose; you’ll typically store them too, but normalize to your canonical symbols via xref/lookup.

### `expected/scopes.csv`

```csv
scope_id,lang,file_path,kind,name,parent_scope_id,start_byte,end_byte
py:corpus_py/pkg/__init__.py::module,python,corpus_py/pkg/__init__.py,module,<module>,,0,55
py:corpus_py/pkg/a.py::module,python,corpus_py/pkg/a.py,module,<module>,,0,293
py:corpus_py/pkg/a.py::add,python,corpus_py/pkg/a.py,function,add,py:corpus_py/pkg/a.py::module,37,88
py:corpus_py/pkg/a.py::Foo,python,corpus_py/pkg/a.py,class,Foo,py:corpus_py/pkg/a.py::module,88,233
py:corpus_py/pkg/a.py::Foo.__init__,python,corpus_py/pkg/a.py,method,__init__,py:corpus_py/pkg/a.py::Foo,103,155
py:corpus_py/pkg/a.py::Foo.inc,python,corpus_py/pkg/a.py,method,inc,py:corpus_py/pkg/a.py::Foo,155,233
py:corpus_py/pkg/a.py::top,python,corpus_py/pkg/a.py,function,top,py:corpus_py/pkg/a.py::module,233,293
py:corpus_py/pkg/b.py::module,python,corpus_py/pkg/b.py,module,<module>,,0,166
py:corpus_py/pkg/b.py::use,python,corpus_py/pkg/b.py,function,use,py:corpus_py/pkg/b.py::module,85,166
py:corpus_py/pkg/sub/c.py::module,python,corpus_py/pkg/sub/c.py,module,<module>,,0,155
py:corpus_py/pkg/sub/c.py::coro,python,corpus_py/pkg/sub/c.py,function,coro,py:corpus_py/pkg/sub/c.py::module,64,118
py:corpus_py/pkg/sub/c.py::call_coro,python,corpus_py/pkg/sub/c.py,function,call_coro,py:corpus_py/pkg/sub/c.py::module,118,155
py:corpus_py/bad_syntax.py::module,python,corpus_py/bad_syntax.py,module,<module>,,0,20
js:corpus_js/mod.js::module,javascript,corpus_js/mod.js,module,<module>,,0,121
js:corpus_js/mod.js::mul,javascript,corpus_js/mod.js,function,mul,js:corpus_js/mod.js::module,29,73
```

### `expected/defs.csv`

(Imports are represented both in `imports.csv` and as `import_binding` defs to capture the local binding surface.)

```csv
def_id,lang,file_path,scope_id,symbol,kind,name,start_byte,end_byte
def:py:pkg:Foo,python,corpus_py/pkg/__init__.py,py:corpus_py/pkg/__init__.py::module,py:pkg.a:Foo,import_binding,Foo,43,46
def:py:pkg:add,python,corpus_py/pkg/__init__.py,py:corpus_py/pkg/__init__.py::module,py:pkg.a:add,import_binding,add,48,51
def:py:pkg:X,python,corpus_py/pkg/__init__.py,py:corpus_py/pkg/__init__.py::module,py:pkg.a:X,import_binding,X,53,54
def:py:pkg.a:X,python,corpus_py/pkg/a.py,py:corpus_py/pkg/a.py::module,py:pkg.a:X,const,X,29,30
def:py:pkg.a:add,python,corpus_py/pkg/a.py,py:corpus_py/pkg/a.py::add,py:pkg.a:add,function,add,41,44
def:py:pkg.a:Foo,python,corpus_py/pkg/a.py,py:corpus_py/pkg/a.py::Foo,py:pkg.a:Foo,class,Foo,94,97
def:py:pkg.a:Foo.__init__,python,corpus_py/pkg/a.py,py:corpus_py/pkg/a.py::Foo.__init__,py:pkg.a:Foo.__init__,method,__init__,107,115
def:py:pkg.a:Foo.inc,python,corpus_py/pkg/a.py,py:corpus_py/pkg/a.py::Foo.inc,py:pkg.a:Foo.inc,method,inc,159,162
def:py:pkg.a:top,python,corpus_py/pkg/a.py,py:corpus_py/pkg/a.py::top,py:pkg.a:top,function,top,237,240
def:py:pkg.b:AFoo,python,corpus_py/pkg/b.py,py:corpus_py/pkg/b.py::module,py:pkg.a:Foo,import_binding,AFoo,56,60
def:py:pkg.b:add,python,corpus_py/pkg/b.py,py:corpus_py/pkg/b.py::module,py:pkg.a:add,import_binding,add,62,65
def:py:pkg.b:m,python,corpus_py/pkg/b.py,py:corpus_py/pkg/b.py::module,py:stdlib:math,import_binding,m,81,82
def:py:pkg.b:use,python,corpus_py/pkg/b.py,py:corpus_py/pkg/b.py::use,py:pkg.b:use,function,use,89,92
def:py:pkg.sub.c:add,python,corpus_py/pkg/sub/c.py,py:corpus_py/pkg/sub/c.py::module,py:pkg.a:add,import_binding,add,58,61
def:py:pkg.sub.c:coro,python,corpus_py/pkg/sub/c.py,py:corpus_py/pkg/sub/c.py::coro,py:pkg.sub.c:coro,function,coro,74,78
def:py:pkg.sub.c:call_coro,python,corpus_py/pkg/sub/c.py,py:corpus_py/pkg/sub/c.py::call_coro,py:pkg.sub.c:call_coro,function,call_coro,122,131
def:js:mod:mul,javascript,corpus_js/mod.js,js:corpus_js/mod.js::mul,js:mod:mul,function,mul,45,48
def:js:mod:TWO,javascript,corpus_js/mod.js,js:corpus_js/mod.js::module,js:mod:TWO,const,TWO,86,89
```

### `expected/imports.csv`

```csv
lang,file_path,scope_id,import_kind,module,name,alias,start_byte,end_byte
python,corpus_py/pkg/__init__.py,py:corpus_py/pkg/__init__.py::module,import_from,.a,Foo,,43,46
python,corpus_py/pkg/__init__.py,py:corpus_py/pkg/__init__.py::module,import_from,.a,add,,48,51
python,corpus_py/pkg/__init__.py,py:corpus_py/pkg/__init__.py::module,import_from,.a,X,,53,54
python,corpus_py/pkg/b.py,py:corpus_py/pkg/b.py::module,import_from,.a,Foo,AFoo,49,60
python,corpus_py/pkg/b.py,py:corpus_py/pkg/b.py::module,import_from,.a,add,,62,65
python,corpus_py/pkg/b.py,py:corpus_py/pkg/b.py::module,import,math,,m,73,82
python,corpus_py/pkg/sub/c.py,py:corpus_py/pkg/sub/c.py::module,import_from,..a,add,,58,61
```

### `expected/calls.csv`

```csv
lang,file_path,scope_id,callee_symbol,callee_name,call_kind,start_byte,end_byte
python,corpus_py/pkg/a.py,py:corpus_py/pkg/a.py::top,py:pkg.a:Foo,Foo,instantiate,259,262
python,corpus_py/pkg/a.py,py:corpus_py/pkg/a.py::top,py:pkg.a:add,add,call,277,280
python,corpus_py/pkg/a.py,py:corpus_py/pkg/a.py::top,py:pkg.a:Foo.inc,inc,call,283,286
python,corpus_py/pkg/b.py,py:corpus_py/pkg/b.py::use,py:pkg.a:Foo,AFoo,instantiate,113,117
python,corpus_py/pkg/b.py,py:corpus_py/pkg/b.py::use,py:pkg.a:add,add,call,129,132
python,corpus_py/pkg/b.py,py:corpus_py/pkg/b.py::use,py:pkg.a:Foo.inc,inc,call,135,138
python,corpus_py/pkg/b.py,py:corpus_py/pkg/b.py::use,py:stdlib:math.sqrt,sqrt,call,158,162
python,corpus_py/pkg/sub/c.py,py:corpus_py/pkg/sub/c.py::coro,py:pkg.a:add,add,call,106,109
python,corpus_py/pkg/sub/c.py,py:corpus_py/pkg/sub/c.py::call_coro,py:pkg.sub.c:coro,coro,call,146,150
javascript,corpus_js/mod.js,js:corpus_js/mod.js::module,js:builtin:console.log,log,call,103,106
javascript,corpus_js/mod.js,js:corpus_js/mod.js::module,js:mod:mul,mul,call,107,110
```

### `expected/refs.csv`

(Calls also appear as refs with `role=call`; plus a few “read” refs.)

```csv
lang,file_path,scope_id,symbol,name,role,start_byte,end_byte
python,corpus_py/pkg/a.py,py:corpus_py/pkg/a.py::top,py:pkg.a:Foo,Foo,call,259,262
python,corpus_py/pkg/a.py,py:corpus_py/pkg/a.py::top,py:pkg.a:X,X,read,263,264
python,corpus_py/pkg/a.py,py:corpus_py/pkg/a.py::top,py:pkg.a:add,add,call,277,280
python,corpus_py/pkg/a.py,py:corpus_py/pkg/a.py::top,py:pkg.a:Foo.inc,inc,call,283,286
python,corpus_py/pkg/b.py,py:corpus_py/pkg/b.py::use,py:pkg.a:Foo,AFoo,call,113,117
python,corpus_py/pkg/b.py,py:corpus_py/pkg/b.py::use,py:pkg.a:add,add,call,129,132
python,corpus_py/pkg/b.py,py:corpus_py/pkg/b.py::use,py:pkg.a:Foo.inc,inc,call,135,138
python,corpus_py/pkg/b.py,py:corpus_py/pkg/b.py::use,py:stdlib:math,m,read,156,157
python,corpus_py/pkg/b.py,py:corpus_py/pkg/b.py::use,py:stdlib:math.sqrt,sqrt,call,158,162
python,corpus_py/pkg/sub/c.py,py:corpus_py/pkg/sub/c.py::coro,py:pkg.a:add,add,call,106,109
python,corpus_py/pkg/sub/c.py,py:corpus_py/pkg/sub/c.py::call_coro,py:pkg.sub.c:coro,coro,call,146,150
javascript,corpus_js/mod.js,js:corpus_js/mod.js::module,js:builtin:console,console,read,95,102
javascript,corpus_js/mod.js,js:corpus_js/mod.js::module,js:builtin:console.log,log,call,103,106
javascript,corpus_js/mod.js,js:corpus_js/mod.js::module,js:mod:mul,mul,call,107,110
javascript,corpus_js/mod.js,js:corpus_js/mod.js::module,js:mod:TWO,TWO,read,111,114
```

---

## 3) SCIP fixture: how to generate + what to assert (without guessing symbol strings)

### 3.1 Generate `index.scip` deterministically

Treat `scip-python` as an external tool invoked via subprocess; on success you get
an `index.scip` in the working directory. For repo consistency, standardize
`--project-name CodeIntel` (omit namespace/version) and pass `--environment` when
pip-based discovery is unavailable.

Representative fixture builder:

```py
# scripts/fixtures/build_scip_fixture.py
from __future__ import annotations
import subprocess
from pathlib import Path

def build_scip_fixture(repo_root: Path) -> Path:
    corpus_root = repo_root / "tests/corpus/p0"
    workdir = corpus_root / "scip_workdir"
    workdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "scip-python", "index", str(corpus_root / "corpus_py"),
        "--project-name", "CodeIntel",
        # "--environment", "env.json",  # optional for non-pip environments
    ]
    subprocess.run(cmd, cwd=workdir, check=True, capture_output=True, text=True)

    out = workdir / "index.scip"
    assert out.exists()
    return out
```

### 3.2 Parse `index.scip` in Python (protobuf-first)

To parse the binary, generate Python bindings from `scip.proto` and then `Index.ParseFromString(...)` .

```py
from scip_pb2 import Index

data = Path("index.scip").read_bytes()
index = Index()
index.ParseFromString(data)
```

(That exact pattern is shown in your SCIP deep dive doc) .

### 3.3 Golden expectations for SCIP, expressed as *range+text+role* assertions

SCIP occurrences expose `(symbol, symbol_roles, range)`  and ranges are 0-based; for Python indexers, character offsets are UTF32 code units . Because this corpus is ASCII-only, UTF32 “character offsets” match simple Python string indices, making assertions stable.

Create: **`expected/scip_occurrence_assertions.jsonl`** (these are the minimum “must exist” facts). Example entries:

```jsonl
{"relative_path":"corpus_py/pkg/a.py","range":[2,0,2,1],"expected_text":"X","must_have_role":"Definition"}
{"relative_path":"corpus_py/pkg/a.py","range":[5,4,5,7],"expected_text":"add","must_have_role":"Definition"}
{"relative_path":"corpus_py/pkg/a.py","range":[18,8,18,11],"expected_text":"Foo","must_have_role":null}
{"relative_path":"corpus_py/pkg/b.py","range":[8,15,8,19],"expected_text":"AFoo","must_have_role":null}
{"relative_path":"corpus_py/pkg/sub/c.py","range":[6,11,6,14],"expected_text":"add","must_have_role":null}
```

**Test logic** (sketch):

* Find an occurrence in that `Document.relative_path` whose `range` matches exactly (SCIP range encoding is 3 or 4 ints; normalize to 4) .
* Verify the slice of the original file at that range equals `expected_text` (respecting UTF32 offsets; ASCII makes this trivial) .
* If `must_have_role=="Definition"`, check `occ.symbol_roles & Definition != 0` (SCIP role bitmask conventions) .

### 3.4 Optional but very powerful: “golden snapshot” instead of protobuf diffs

For “human review” regression tests, use `scip snapshot` and commit the resulting annotated tree. Snapshot output is explicitly intended to be greppable/diffable; it’s a common regression-testing workflow . For Python, set comment syntax to `#` .

```bash
scip snapshot --from=index.scip --to tests/corpus/p0/expected/scip_snapshot --comment-syntax="#"
```

---

## 4) How teams use this pack (conformance workflow)

1. Each team builds its extractor (LibCST, tree-sitter, SCIP protobuf ingestion).
2. They must output P0 tables with the **same schemas and join keys** (byte spans + stable IDs); LibCST explicitly supports byte-span providers and recommends byte spans for exact slicing , and tree-sitter’s APIs are byte-first (edits/ranges depend on byte correctness) .
3. CI step compares produced tables to `tests/corpus/p0/expected/*.csv` (byte-perfect equality).
4. SCIP conformance is validated via `scip_occurrence_assertions.jsonl` + optional `scip snapshot` diff.

If you want, I can also add a compact “comparison harness” snippet (PyArrow read → schema check → sort-by PK → row equality) that’s resilient to column-order drift and uses schema metadata as the contract surface (Arrow schemas/metadata are meant for this kind of immutability/contract discipline) .

According to a document from **January 1, 2026**, here’s a compact **“comparison harness”** you can drop into your test suite to validate **P0 contract conformance** and **golden expectations** using **PyArrow**:

* **Read**: Parquet dataset → `pyarrow.Table`
* **Schema contract check**:

  * required columns present
  * (optionally) no extra columns
  * per-column type + nullability match
  * **schema metadata** keys match (treat metadata as the contract surface)
  * diagnostics use `schema.to_string(show_*_metadata=...)` for readable diffs
* **Canonical sort**: by PK using `pc.SortOptions` + `pc.sort_indices`, then `table.take(...)`
* **Row equality**: column-wise list equality after sorting (robust to row ordering drift; schema check protects against “1 vs 1.0” style silent coercions)

---

## 1) Contract + harness (single file)

```python
# tests/utils/arrow_contract_harness.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence, Optional

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds


@dataclass(frozen=True)
class ArrowTableContract:
    """
    Minimal contract surface:
      - schema: required columns + types (+ optional schema metadata keys)
      - pk: canonical primary key for deterministic sorting + equality
    """
    name: str
    schema: pa.Schema
    pk: tuple[str, ...]

    # Behavior knobs (keep strict for P0)
    allow_extra_columns: bool = False
    check_schema_metadata: bool = True  # schema-level metadata is the contract surface


def _schema_debug(s: pa.Schema) -> str:
    # Debug printing knobs are explicitly supported
    return s.to_string(
        show_field_metadata=True,
        show_schema_metadata=True,
        truncate_metadata=False,
    )


def read_parquet_dataset(path: str | Path, *, columns: Optional[Sequence[str]] = None) -> pa.Table:
    """
    Load a Parquet dataset from a file or directory into a single Table.
    (For golden tests, materializing to a Table is usually fine.)
    """
    dataset = ds.dataset(str(path), format="parquet")
    scanner = dataset.scanner(columns=list(columns) if columns is not None else None)
    return scanner.to_table()


def assert_schema_conforms(actual: pa.Schema, contract: ArrowTableContract) -> None:
    expected = contract.schema

    exp_names = list(expected.names)
    act_names = list(actual.names)

    missing = [c for c in exp_names if c not in act_names]
    if missing:
        raise AssertionError(
            f"[{contract.name}] Missing required columns: {missing}\n\n"
            f"EXPECTED SCHEMA:\n{_schema_debug(expected)}\n\n"
            f"ACTUAL SCHEMA:\n{_schema_debug(actual)}\n"
        )

    if not contract.allow_extra_columns:
        extras = [c for c in act_names if c not in exp_names]
        if extras:
            raise AssertionError(
                f"[{contract.name}] Unexpected extra columns: {extras}\n\n"
                f"EXPECTED SCHEMA:\n{_schema_debug(expected)}\n\n"
                f"ACTUAL SCHEMA:\n{_schema_debug(actual)}\n"
            )

    # Column-by-column type + nullability checks (order-insensitive).
    for name in exp_names:
        ef = expected.field(name)  # schema.field(i|name) is supported
        af = actual.field(name)

        if af.type != ef.type or af.nullable != ef.nullable:
            raise AssertionError(
                f"[{contract.name}] Column '{name}' mismatch:\n"
                f"  expected: type={ef.type}, nullable={ef.nullable}\n"
                f"  actual:   type={af.type}, nullable={af.nullable}\n\n"
                f"EXPECTED SCHEMA:\n{_schema_debug(expected)}\n\n"
                f"ACTUAL SCHEMA:\n{_schema_debug(actual)}\n"
            )

    # Schema metadata as a contract surface.
    # Attach metadata via Schema.with_metadata(...) in your contract definition:contentReference[oaicite:4]{index=4}.
    if contract.check_schema_metadata:
        exp_meta = expected.metadata or {}
        act_meta = actual.metadata or {}

        # Treat expected metadata as "required keys must match exactly";
        # allow actual to include extra metadata keys.
        for k, v in exp_meta.items():
            if act_meta.get(k) != v:
                raise AssertionError(
                    f"[{contract.name}] Schema metadata mismatch for key={k!r}:\n"
                    f"  expected: {v!r}\n"
                    f"  actual:   {act_meta.get(k)!r}\n\n"
                    f"EXPECTED SCHEMA:\n{_schema_debug(expected)}\n\n"
                    f"ACTUAL SCHEMA:\n{_schema_debug(actual)}\n"
                )


def sort_table_by_pk(table: pa.Table, pk: Sequence[str]) -> pa.Table:
    """
    Deterministic sort using Arrow compute.

    SortOptions + sort_indices + table.take are explicitly supported:contentReference[oaicite:5]{index=5}.
    """
    for c in pk:
        if c not in table.column_names:
            raise AssertionError(f"PK column '{c}' missing from table columns={table.column_names}")

    idx = pc.sort_indices(
        table,
        options=pc.SortOptions(
            sort_keys=[(c, "ascending") for c in pk],
            null_placement="at_end",
        ),
    )
    return table.take(idx)


def assert_table_matches_golden(
    *,
    actual_path: str | Path,
    expected_path: str | Path,
    contract: ArrowTableContract,
) -> None:
    actual = read_parquet_dataset(actual_path)
    expected = read_parquet_dataset(expected_path)

    # 1) Schema contract conformance
    assert_schema_conforms(actual.schema, contract)
    assert_schema_conforms(expected.schema, contract)  # sanity: your golden should conform too

    # 2) Canonical row order
    a_sorted = sort_table_by_pk(actual, contract.pk)
    e_sorted = sort_table_by_pk(expected, contract.pk)

    # 3) Basic shape checks
    if a_sorted.num_rows != e_sorted.num_rows:
        raise AssertionError(
            f"[{contract.name}] Row count mismatch:\n"
            f"  expected={e_sorted.num_rows}\n"
            f"  actual={a_sorted.num_rows}\n"
        )

    # 4) Value equality (column-wise), resilient to column-order drift.
    # We only compare contract columns (ignore extras if allow_extra_columns=True).
    a_dict = a_sorted.to_pydict()
    e_dict = e_sorted.to_pydict()

    for col in contract.schema.names:
        av = a_dict[col]
        ev = e_dict[col]
        if av != ev:
            # Find first diff to make failures actionable
            first = next((i for i, (x, y) in enumerate(zip(av, ev)) if x != y), None)
            raise AssertionError(
                f"[{contract.name}] Column '{col}' mismatch.\n"
                f"  first_diff_row={first}\n"
                f"  expected[{first}]={ev[first] if first is not None else None!r}\n"
                f"  actual[{first}]={av[first] if first is not None else None!r}\n"
            )
```

**Why this matches your “contract-as-Arrow-schema” goal**

* The harness explicitly leans on **schema equality semantics + metadata** as the contract surface (and uses `schema.to_string(show_*_metadata=...)` for clean diffs).
* Sorting is done using Arrow’s compute layer (`SortOptions` → `sort_indices` → `table.take`) so it’s deterministic and doesn’t depend on pandas semantics.
* To attach the contract identity/version (and PK, if you want) you should use `Schema.with_metadata(...)` (and you can strip metadata with `remove_metadata()` when needed).

---

## 2) Example contract definition (schema metadata + PK)

This is the key pattern: **schema metadata is the contract identity**.

```python
# codeintel/contracts/p0.py
import pyarrow as pa

SCOPES_V1 = pa.schema(
    [
        pa.field("repo", pa.string(), nullable=False),
        pa.field("commit", pa.string(), nullable=False),
        pa.field("rel_path", pa.string(), nullable=False),
        pa.field("start_byte", pa.int64(), nullable=False),
        pa.field("end_byte", pa.int64(), nullable=False),
        pa.field("scope_kind", pa.string(), nullable=False),
        pa.field("symbol", pa.string(), nullable=True),
        # ...
    ]
).with_metadata({  # Schema.with_metadata is the supported attach path:contentReference[oaicite:9]{index=9}
    b"codeintel.contract": b"p0.scopes@v1",
    b"codeintel.pk": b'repo,commit,rel_path,start_byte,end_byte,scope_kind',
})

SCOPES_CONTRACT = ArrowTableContract(
    name="p0.scopes@v1",
    schema=SCOPES_V1,
    pk=("repo", "commit", "rel_path", "start_byte", "end_byte", "scope_kind"),
)
```

---

## 3) Usage in a golden test

```python
# tests/test_p0_scopes_golden.py
from pathlib import Path
from codeintel.contracts.p0 import SCOPES_CONTRACT
from tests.utils.arrow_contract_harness import assert_table_matches_golden

def test_p0_scopes_golden():
    assert_table_matches_golden(
        actual_path=Path("build/bootstrap/p0/scopes/"),     # your pipeline output
        expected_path=Path("tests/golden/p0/scopes/"),      # committed fixture output
        contract=SCOPES_CONTRACT,
    )
```

---

## 4) One optional “best-in-class” refinement (if you shard / evolve schemas)

If your golden datasets are **sharded** or you’re merging outputs from multiple sources, Arrow’s recommended pattern is **unify → cast → concat**, and it has specific metadata preservation rules (schema metadata comes from the *first* schema; field metadata comes from the schema where the field is first defined). This matters if you treat metadata as part of the contract, which you do.

---

If you want one more micro-deliverable after this, I can also provide a tiny helper that **extracts the PK directly from `schema.metadata[b"codeintel.pk"]`** (so the schema becomes the *single* contract artifact and you don’t repeat PK in code).

Below is a **tiny, reusable helper** that makes the Arrow schema itself the *single* PK contract artifact by reading the primary key directly from `schema.metadata[b"codeintel.pk"]`.

Arrow schema metadata is the right place for this because it’s **immutable, versionable, and carried alongside the schema** (metadata keys/values are represented as `bytes` in Python). 

---

## Primary key extraction helper

### Contract expectation

* `schema.metadata[b"codeintel.pk"]` exists
* value is UTF‑8 encoded
* preferred encoding: **JSON list of strings** (allows composite PKs unambiguously)

  * example: `b'["repo","path","span_start","span_end"]'`
* tolerated encoding: **comma-separated list** (for convenience)

  * example: `b"repo,path,span_start,span_end"`

### Code

```python
from __future__ import annotations

import json
from typing import Iterable, Mapping, Sequence, Tuple, Union, overload

import pyarrow as pa


PK_METADATA_KEY: bytes = b"codeintel.pk"


def _schema_metadata(schema: pa.Schema) -> Mapping[bytes, bytes]:
    # Arrow represents schema metadata as bytes->bytes (may be None).
    return schema.metadata or {}


def pk_from_schema(
    schema: pa.Schema,
    *,
    key: bytes = PK_METADATA_KEY,
    require_present: bool = True,
    require_in_schema: bool = True,
) -> Tuple[str, ...]:
    """
    Returns the PK columns declared in Arrow schema metadata.

    Supports:
      - JSON list encoding: b'["col1","col2"]'
      - CSV encoding:       b"col1,col2"

    Raises a ValueError with a clear message if missing/invalid.
    """
    md = _schema_metadata(schema)
    raw = md.get(key)

    if raw is None:
        if require_present:
            raise ValueError(
                f"Missing PK contract: schema.metadata[{key!r}] is not set. "
                f"Set it to a UTF-8 JSON array like b'[\"col1\",\"col2\"]'."
            )
        return tuple()

    try:
        text = raw.decode("utf-8").strip()
    except Exception as e:
        raise ValueError(
            f"Invalid PK contract: schema.metadata[{key!r}] must be UTF-8 bytes. Got {raw!r}."
        ) from e

    # Preferred: JSON list
    cols: Sequence[str]
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid PK contract JSON in schema.metadata[{key!r}]: {text!r}"
            ) from e
        if not isinstance(parsed, list) or not all(isinstance(x, str) for x in parsed):
            raise ValueError(
                f"Invalid PK contract JSON in schema.metadata[{key!r}]: expected a JSON list of strings, got {parsed!r}"
            )
        cols = parsed
    else:
        # Fallback: comma-separated
        cols = [c.strip() for c in text.split(",") if c.strip()]

    if not cols:
        raise ValueError(
            f"Invalid PK contract: schema.metadata[{key!r}] decoded to empty PK: {text!r}"
        )

    # Enforce uniqueness + preserve order
    out: list[str] = []
    seen: set[str] = set()
    for c in cols:
        if c in seen:
            raise ValueError(
                f"Invalid PK contract: duplicate PK column {c!r} in schema.metadata[{key!r}]={text!r}"
            )
        seen.add(c)
        out.append(c)

    if require_in_schema:
        missing = [c for c in out if schema.get_field_index(c) == -1]
        if missing:
            raise ValueError(
                f"PK contract columns not in schema fields: {missing}. "
                f"PK={out}, fields={[f.name for f in schema]}."
            )

    return tuple(out)


def pk_from_table(
    table: pa.Table,
    *,
    key: bytes = PK_METADATA_KEY,
    require_present: bool = True,
    require_in_schema: bool = True,
) -> Tuple[str, ...]:
    """Convenience wrapper for pa.Table."""
    return pk_from_schema(
        table.schema,
        key=key,
        require_present=require_present,
        require_in_schema=require_in_schema,
    )
```

---

## How you set the PK in the schema (one-liner pattern)

You’ll typically define schemas in your “P0 contracts” module and attach metadata once, up front:

```python
import pyarrow as pa

schema = pa.schema(
    [
        pa.field("repo", pa.string()),
        pa.field("path", pa.string()),
        pa.field("span_start", pa.int32()),
        pa.field("span_end", pa.int32()),
        # ...
    ]
).with_metadata(
    {
        b"codeintel.pk": b'["repo","path","span_start","span_end"]',
        b"codeintel.contract_version": b"1",
        b"codeintel.table_key": b"cst.scopes",
    }
)
```

Because Arrow schema metadata is a first-class, immutable contract surface, this scales well as you add more invariants over time (PK, table key, contract version, coordinate conventions, etc.). 

---

## Tiny integration points

### 1) Golden comparison harness (no repeated PK constants)

Anywhere you currently do:

* “read expected”
* “read actual”
* “sort by PK”
* “row-compare”

…replace “PK list” with:

```python
pk = pk_from_table(actual_table)
# sort/compare using pk
```

### 2) Dataset load/registration logic (manifest-driven)

When your seed-suite loader reads a dataset from disk into Arrow, you can validate contract before loading into DuckDB:

```python
pk = pk_from_schema(dataset_schema)
# validate pk columns exist, etc.
```

---

If you want to push this one notch further (still very small): add a second helper that reads **all contract metadata** into a structured `Contract` object (pk, table_key, version, coordinate conventions), but the snippet above is the minimal “stop repeating PK in code” building block.
