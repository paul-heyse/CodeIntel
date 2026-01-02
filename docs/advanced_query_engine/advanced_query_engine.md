
Here’s a framework I’ve found works well for **LLM programming agents**: classify “searches” by **intent** (what decision the agent is trying to make), then define a **best-in-class result contract** (what the tool should return so the agent can act without re-searching).

This is also how mature code-intel systems think about retrieval: *go-to-definition / references / call hierarchy* (via index protocols like SCIP or LSP) for semantic navigation, and *text/AST search* for pattern scanning. ([GitHub][1])

---

## A. The “search intent” taxonomy (with ideal outputs)

For each category: **information sought → takeaways → ideal output**.

### 1) Locate the “thing” (Symbol Resolution)

* **Information sought:** “Where is X defined?” “What file owns this symbol?”
* **Takeaways:** canonical definition location; symbol identity (stable ID); signature/type.
* **Ideal output:**

  * `symbol_id` (stable), `def_span` (file + byte/line range), `kind` (func/class/var), `signature`/type summary, docstring preview.
  * This maps cleanly to *Go to definition* style capabilities in SCIP/LSP. ([GitHub][1])

### 2) Enumerate usages (Reference & Override Discovery)

* **Information sought:** “Who calls/uses/imports/instantiates X?” “What overrides/implements this?”
* **Takeaways:** blast radius candidates; “who will break”; extension points.
* **Ideal output:**

  * A deduped, ranked list of **reference sites** with `ref_span`, `enclosing_symbol`, and a short snippet; plus **grouping** (by module/package/test vs prod).
  * This is exactly what code-intel indices are built for (“Find references”, “Find implementations”). ([GitHub][1])

### 3) Trace the call path (Call Hierarchy / Entry→Sink)

* **Information sought:** “How does execution flow from entrypoint to X?” “What does X call next?”
* **Takeaways:** where to intercept; what layers are involved; what invariants must hold along the path.
* **Ideal output:**

  * A small **call-path slice**: `(caller → callsite → callee)` edges with spans and (optionally) arg/param names.
  * LSP formalizes call hierarchy requests (prepare + incoming/outgoing calls), which is the right semantic substrate for this search type. ([Microsoft GitHub][2])

### 4) Pattern / policy scan (Lexical or AST)

* **Information sought:** “Find all occurrences of pattern P” (APIs, anti-patterns, TODOs, unsafe calls).
* **Takeaways:** inventory + counts; hotspots; “how many edits will I need?”
* **Ideal output:**

  * **Machine-readable match records**: `path`, `start/end`, `matched_text`, `context_before/after`, `rule_id` (if AST-based), stable ordering.
  * If using ripgrep, emitting JSON is key because an agent can reliably parse + post-process results. ([Iepathos][3])

### 5) “What is the contract?” (Tests, Docs, Examples)

* **Information sought:** expected behavior, edge cases, invariants, error modes.
* **Takeaways:** the acceptance criteria for the change; what must not regress.
* **Ideal output:**

  * The **minimal test set** exercising the component; key assertions; docs/examples closest to the behavior; plus links/spans to the “source of truth.”

### 6) “How is it wired?” (Config / DI / Plugin Registration)

* **Information sought:** entrypoints, routing, registries, environment flags, dependency injection, build steps.
* **Takeaways:** what you must update besides code (config schemas, CLI flags, docs, manifests).
* **Ideal output:**

  * A wiring map: `entrypoint → framework hook → module/function`, plus config keys and defaults; ideally anchored to spans.

### 7) “What should I imitate?” (Precedent / Similarity Search)

* **Information sought:** “Show me how the repo does this elsewhere” (similar endpoint, similar data model, similar pattern).
* **Takeaways:** repo conventions; copy-adapt template; avoids style regressions.
* **Ideal output:**

  * Top-K exemplars with **why they match** (same interface, same call pattern, same data types), and the *diff-able* core snippet.

### 8) Change impact / scope confirmation (Transitive Dependency Slice)

* **Information sought:** transitive callers/callees; affected modules; likely affected tests; public API surface touched.
* **Takeaways:** “small vs large change”; risk score; required coordination.
* **Ideal output:**

  * A **bounded transitive slice** (e.g., depth=2 callers + depth=1 callees), plus “public boundary crossings” (imports across packages, exported symbols touched).


> Why this matters now: repo-level agent benchmarks (e.g., SWE-bench / SWE-bench Verified) increasingly reward agents that can **navigate the repository structure** and retrieve the right context efficiently, not just pattern-match locally. Graph-based repo navigation approaches like RepoGraph explicitly target this need. ([SWE-bench][4])

---

## B. A best-in-class “search response contract” (what every tool should return)

No matter the intent, you want outputs that are:

1. **Actionable:** enough to decide the next edit step *without re-querying*.
2. **Grounded:** every claim ties to a span/snippet.
3. **Composable:** results can be merged across tools (text search + SCIP + AST).
4. **Deterministic:** stable ordering + stable IDs so golden tests/CI diffs work.

A practical universal schema:

* **Query metadata:** `intent`, `scope_hint` (project/package/module), `budget` (max results / depth), `tool_used`.
* **Primary answer:** 1–3 sentence summary of what was found.
* **Artifacts (typed):**

  * `symbols[]`: `{symbol_id, kind, name, def_span, signature, doc}`
  * `snippets[]`: `{path, span, text, enclosure_symbol_id?, tags(test|prod|generated)}`
  * `edges[]`: `{edge_type(call|import|override|read|write), src_symbol_id, dst_symbol_id, callsite_span?}`
  * `aggregates`: counts by file/module, top hotspots
* **Next-step suggestions:** 2–5 follow-up queries (e.g., “show incoming calls for symbol_id X”, “find tests referencing Y”).

This matches what semantic code-index protocols and IDE protocols are designed to support (definition/references/call hierarchy), and what text search tools can output in parseable form (e.g., ripgrep JSON). ([GitHub][1])

---

## C. How agents typically chain these search types for “code change scope → design → edits”

A high-performing sequence (and what to return at each step):

1. **Locate** the primary symbols (defs + signatures).
2. **Enumerate usages** (refs grouped by prod vs tests).
3. **Call-path slice** (entrypoints → target).
4. **Contract scan** (tests/docs/examples closest to the behavior).
5. **Pattern scan** (related code patterns; policy constraints).
6. **Impact slice** (bounded transitive callers + public boundary crossings).
7. **History/ownership** (if behavior is surprising or risky).

If you want, I can turn this into a **concrete “search profile catalog”** (namespaced profiles like `symbol.resolve`, `refs.find`, `callgraph.slice`, `contract.tests_minset`, etc.) with exact output JSON shapes per profile—so an agent can request the right “search mode” and get the right kind of payload every time.

[1]: https://github.com/sourcegraph/scip?utm_source=chatgpt.com "sourcegraph/scip: SCIP Code Intelligence Protocol"
[2]: https://microsoft.github.io/language-server-protocol/specifications/specification-3-16/?utm_source=chatgpt.com "Language Server Protocol Specification - 3.16"
[3]: https://iepathos.github.io/ripgrep/?utm_source=chatgpt.com "ripgrep User Guide"
[4]: https://www.swebench.com/SWE-bench/?utm_source=chatgpt.com "Overview - SWE-bench"

Below is a **pragmatic “search serving” program structure** that uses:

* **rpygrep** for *fast candidate discovery + snippet extraction* (typed `rg --json` results, safe caps, context lines). 
* **tree-sitter** for *language-agnostic structural extraction* (incremental parse, windowed queries, match limits, query-pack linting). 
* **ast-grep-py** for *high-signal structural matching + captures* (pattern/rule system, metavariable captures, stable byte offsets). 
* **LibCST** for *Python-only semantic enrichment* (bytes-first parsing, spans, scope/qualified names, docstrings, container chains). 

The key is: **every query type is a plan** that (1) finds candidates cheaply, (2) resolves structure, (3) enriches semantics, and (4) emits a “no-followup bundle”.

---

## 1) Unifying contract: spans + symbols + evidence

Make everything converge on *byte-based spans* (because tree-sitter + ast-grep + LibCST can all give byte offsets; rpygrep gives offsets per match line and submatches).

```python
# contracts.py
from dataclasses import dataclass
from typing import Literal, Optional

@dataclass(frozen=True)
class Span:
    path: str                 # repo-relative posix
    start_byte: int
    end_byte: int             # exclusive
    start_line: Optional[int] = None  # 1-indexed (for display)
    start_col: Optional[int] = None   # 0-indexed
    end_line: Optional[int] = None
    end_col: Optional[int] = None

SymbolKind = Literal["module","class","function","method","variable","import","callsite","route","config_key","test"]

@dataclass(frozen=True)
class SymbolId:
    # stable: can be span-based, and optionally enriched w/ qualified names
    kind: SymbolKind
    stable: str               # e.g., sha256(f"{path}:{start}:{end}:{kind}")[:16]
    qname: Optional[str] = None   # best-effort semantic id (LibCST QN/FQN)

@dataclass(frozen=True)
class EvidenceSnippet:
    span: Span
    text: str                 # excerpt
    context_before: list[str]
    context_after: list[str]
```

**Best-in-class depth rule:** every result object should include:

* **primary span** + **container span** (enclosing class/function/module),
* **semantic identity** (qualified name candidates when possible),
* **evidence snippet** (context lines pre-sliced),
* and **at least one “related” slice** that would otherwise be a follow-up (top refs, top tests, wiring edges, etc.).

---

## 2) Package layout: backends + indexers + query handlers

```
searchkit/
  contracts.py
  service.py                # SearchService: dispatch plans, budgets, caching
  budgets.py                # max files, max matches, time budgets
  cache.py                  # file hash, parse caches, index caches

  backends/
    rpygrep_backend.py      # file discovery + lexical search + snippets
    treesitter_backend.py   # parse cache + query packs + windowed execution
    astgrep_backend.py      # structural pattern search + captures
    libcst_backend.py       # python semantic index + enrichment

  indexes/
    python_index.py         # defs/imports/calls/refs/docstrings using LibCST
    ts_index.py             # multi-lang defs/imports/calls via tree-sitter queries

  handlers/
    q1_resolve_symbol.py
    q2_find_usages.py
    q3_call_paths.py
    q4_pattern_scan.py
    q5_contract_lookup.py
    q6_wiring_map.py
    q7_precedent_search.py
    q8_impact_slice.py

  ranking.py                # scoring + grouping
  render.py                 # normalize to agent payload schema
```

---

## 3) Backends: what each one “owns”

### 3.1 rpygrep backend = *candidate discovery + snippets (fast, bounded)*

Use it for:

* “where might this live?” (file candidates)
* “show me evidence lines” (context windows)
* fast scans across docs/tests/config.

rpygrep’s value is its **typed JSON parsing**, `max_count` / `max_file_size` / `max_depth` safety rails, and easy context lines. 

### 3.2 tree-sitter backend = *cross-language structural index + incremental windows*

Use it for:

* non-Python files (or “good enough” structure without semantic resolution),
* “defs/calls/imports” extraction via query packs,
* incremental parsing (`old_tree` + `changed_ranges`) and windowed queries with match limits. 

### 3.3 ast-grep backend = *structural pattern scanning with captures*

Use it for:

* policy scans and structural patterns (especially “find *this shape*”),
* capturing metavariables and returning byte spans via `node.range().start.index/end.index`,
* fast, declarative rule packs (plus constraints/transforms when useful). 

### 3.4 LibCST backend = *Python semantic enrichment*

Use it for:

* byte-accurate parsing (`parse_module(bytes)`),
* spans via metadata providers (Position + ByteSpan),
* semantic identity and relationships: ScopeProvider, QualifiedNameProvider, ParentNodeProvider, docstrings, etc. 

---

## 4) The “plan” abstraction (how you avoid follow-ups)

Each query handler returns a **bundle** that’s intentionally over-complete.

```python
# service.py
from dataclasses import dataclass
from typing import Any, Protocol

@dataclass(frozen=True)
class QueryRequest:
    type: str
    text: str
    repo_root: str
    scope_paths: list[str] | None = None
    budget: dict[str, Any] | None = None

@dataclass(frozen=True)
class QueryResponse:
    summary: str
    primary: list[dict[str, Any]]
    related: dict[str, list[dict[str, Any]]]  # refs/tests/wiring/etc
    debug: dict[str, Any]                     # counts, caps hit, partial flags

class QueryHandler(Protocol):
    def run(self, req: QueryRequest) -> QueryResponse: ...
```

**Core idea:** all 8 query types reuse shared index slices:

* `defs`, `imports`, `calls`, `refs`, `docstrings`, `routes/config keys`, `tests/examples`.
  You can build these lazily per file and cache by file hash.

---

## 5) The 8 query types: concrete tool choreography + “no-followup bundle”

### Q1) Symbol resolution (Locate the thing)

**Goal:** “What is X? Where defined? What’s the signature/contract? Who exports it?”

**Plan**

1. **rpygrep**: find candidate defs (`class X`, `def X`, `X =`, plus `from ... import X`). Use strict caps. 
2. **LibCST** (for `.py` candidates): parse bytes-first; extract:

   * definition spans (Position + ByteSpan),
   * docstring, signature shape,
   * qualified name candidates,
   * container chain (enclosing class/function). 
3. **Bundle extras to avoid followup**:

   * top N **references** (Q2-lite) + top N **call sites**,
   * likely **tests** referencing it (Q5-lite),
   * “wiring hints” if it looks like a handler/entrypoint (decorators, registry patterns).

### Q2) Enumerate usages (References + overrides)

**Goal:** “Where is X used? Is it read/write/call? Any overrides/implementations?”

**Plan**

1. **rpygrep**: quick lexical occurrences (fast, broad). 
2. **LibCST** for Python files:

   * classify occurrences via ExpressionContext/Scope analysis (read/write/delete),
   * attach referents where possible (don’t collapse ambiguity),
   * for class methods: collect `ClassDef`/bases; approximate override sets by name + inheritance edges (syntax-level unless you add type-resolution later). 
3. **Bundle extras**:

   * group by `prod vs tests`, by package/module,
   * attach “top containers” (functions/classes that contain most hits),
   * include *nearby* occurrences (same file, same class) to avoid “search again for context”.

### Q3) Trace call paths / call hierarchy

**Goal:** “How does execution reach X? What does X call?”

**Plan**

1. Build/refresh a **call index**:

   * **LibCST**: collect `Call` nodes, attach caller container, attach callee *qualified name candidates* (ambiguity preserved). 
   * **tree-sitter** fallback for non-Python: query pack extracts `call_expression`/`function_definition`-style nodes and spans; less semantic, still useful. 
2. Path finding:

   * BFS on a bounded graph (depth + node budget),
   * prefer edges with stronger evidence (callee resolves to qname vs mere dotted syntax).
3. **Bundle extras**:

   * include entrypoint candidates (routes, `__main__`, CLI commands) discovered via Q6 primitives,
   * include “argument shape” summary at each callsite (kwargs/starargs counts),
   * include *both* incoming and outgoing slices (call hierarchy view).

### Q4) Pattern / policy scan (Lexical + AST)

**Goal:** “Find all instances of pattern P; return structured matches + context + quickfix hints.”

**Plan**

1. **rpygrep** for lexical patterns (fast, especially for strings, TODOs, config keys). 
2. **ast-grep-py** for structural patterns:

   * pattern metavariables (`$A`, `$$$ARGS`) + captures + `range()` offsets,
   * optional config rules (constraints/utils/transform) for precision,
   * stable match records (path + byte span + capture texts). 
3. Optionally **tree-sitter queries** for non-Python structural scans with windowing + match limits. 
4. **Bundle extras**:

   * “enclosing symbol” (function/class) for every match (via LibCST ParentNodeProvider for Python),
   * severity/classification tags (policy id, confidence),
   * **dedup/overlap policy** (drop-inner vs error) so results are deterministic.

### Q5) Contract lookup (tests/docs/examples)

**Goal:** “What proves behavior? What examples exist? What should not regress?”

**Plan**

1. **rpygrep** scan in conventional areas (`tests/`, `docs/`, `README*`, `examples/`) for:

   * symbol name, qualified-like strings, key error messages. 
2. **LibCST** (Python tests):

   * resolve whether the test *actually references* the symbol (import + call) vs string mention,
   * extract minimal test set: closest tests by semantic linkage + proximity. 
3. **Bundle extras**:

   * return “test entrypoints” (pytest function names, fixtures involved),
   * docstrings at definition + any examples nearby,
   * provide the *assertion lines* or “golden snapshot” references (by span).

### Q6) Wiring / config / DI / registry mapping

**Goal:** “How is this invoked/registered/configured?”

**Plan**

1. **rpygrep**: find framework signatures / registration calls / env var keys. 
2. **ast-grep-py**: structural patterns for decorators and registries:

   * e.g., `@router.get($PATH)` captures `$PATH`,
   * `registry.register($NAME, $OBJ)` captures identifiers,
   * `Settings(..., env_prefix=...)` captures config knobs. 
3. **LibCST** enrich:

   * attach handler function qname + span,
   * attach config key literals and where read (`os.environ[...]`, `getenv`, settings models),
   * container chains and imports so the agent knows “what file owns wiring.” 
4. **Bundle extras**:

   * produce a “wiring map” graph: `entrypoint → router/registry → handler`,
   * include inferred “public boundary crossings” (imports across packages).

### Q7) Precedent / similarity search (Show me how this repo does it)

**Goal:** return exemplars **with reasons**, not just matches.

**Plan**

1. Candidate pool:

   * use Q4-ish structural scans (ast-grep or tree-sitter) to collect “things of the same kind” (e.g., all FastAPI routes, all DatasetContract defs, all CLI commands).
2. Feature extraction:

   * **LibCST**: derive lightweight “semantic fingerprints” (decorators, called APIs, import set, param shape, docstring tokens). 
3. Ranking:

   * score by overlap on: decorator kind, call set, signature shape, shared imports, lexical similarity of names.
4. **Bundle extras**:

   * return top K exemplars with a “why this matches” field and the exact spans/snippets.

### Q8) Impact / scope confirmation (Transitive slice)

**Goal:** “If I change X, what breaks? What do I need to touch?”

**Plan**

1. Inputs:

   * from Q2: direct references,
   * from Q3: call edges,
   * from Q6: wiring edges,
   * optional import graph edges (LibCST imports; tree-sitter fallback).
2. Graph slice:

   * bounded BFS/DFS with budgets (depth, max nodes),
   * annotate boundary crossings (package transitions; test/prod transitions).
3. **Bundle extras**:

   * emit a “change checklist” automatically populated from discovered edges:

     * files likely needing updates,
     * tests likely to fail,
     * config docs needing edits,
     * “public surface touched” heuristic.

---

## 6) Two indices you should maintain (even if lazily)

### 6.1 PythonSemanticIndex (LibCST-backed)

Per `.py` file, cache:

* defs (class/function/method) + spans + docstrings,
* imports (normalized),
* calls (caller → callee qname candidates),
* refs (read/write + referents sets),
* container map (node span → enclosing def id).

LibCST gives you the primitives for bytes-first parsing and rich metadata; treat parse errors as structured artifacts, not failures. 

### 6.2 StructuralIndex (tree-sitter + query packs)

For non-Python (and for “fast shape” even in Python), cache:

* parse trees (incremental if you want),
* query packs for defs/imports/calls,
* windowed execution by changed ranges; enforce `match_limit` and detect partial results. 

---

## 7) Practical “no follow-up” payload shape (what your service returns)

Return a single envelope with consistent sections across all query types:

```json
{
  "summary": "...",
  "primary": [ /* the main answers */ ],
  "related": {
    "references": [ ... ],
    "calls_in": [ ... ],
    "calls_out": [ ... ],
    "tests": [ ... ],
    "wiring": [ ... ],
    "examples": [ ... ]
  },
  "debug": {
    "caps_hit": false,
    "partial": false,
    "scanned_files": 123,
    "parsed_files": 18
  }
}
```

Even for Q1, include *some* Q2/Q5/Q6 slices by default (small top-N), because that’s what prevents the agent from immediately issuing follow-ups.

---

## 8) Why this composition works (in one sentence)

* **rpygrep** finds *where to look* cheaply,
* **ast-grep / tree-sitter** find *what shape exists* structurally,
* **LibCST** tells you *what it actually means* in Python,
  and your handlers always return a **bundled answer**: primary + the next 1–2 steps precomputed.

If you want, I can sketch a **minimal concrete implementation** of `SearchService` + one handler (e.g., Q1 Symbol Resolution) that shows the actual call chain across the four backends and emits the full “no-followup bundle” JSON.

Got it – I’m on it and have updated the instructions accordingly.


According to a document from **December 24, 2025**, you can treat “code search for agents” as a pipeline problem: *compile declarative query packs, interrogate their IR/contracts up front, then run them with range limits + deterministic outputs*. Below is a **complete implementation plan** for a Python program that uses **rpygrep + tree-sitter + LibCST + ast-grep-py** to serve your 8 query types with “best‑in‑class” depth (extra metadata + expanded scope to avoid followups).

---

## 0. Non-negotiable design constraints

### 0.1 One span model for all engines

Make **byte spans** the canonical coordinate system across all outputs, and compute line/col lazily (or from cached line indexes). This avoids inconsistencies across tools (tree-sitter and ripgrep fundamentally operate on byte offsets; LibCST explicitly distinguishes byte spans and code ranges).

**Canonical span types**

* `ByteSpan = {path, start_byte, end_byte}` (end exclusive)
* `CodeRange = {start_line, start_col, end_line, end_col}`

LibCST position semantics are **start inclusive / end exclusive**, with **lines 1-indexed** and **cols 0-indexed**; byte spans are bytes-not-chars (UTF‑8 safe).

### 0.2 Determinism is a feature

Every handler must output:

* stable ordering: `(path, start_byte, end_byte, kind, sha1(excerpt))`
* stable IDs: derived from `(path, start_byte, byte_len, kind)` (hash optional)

LibCST guidance explicitly frames indexes as “facts” keyed by spans + stable IDs, not CST objects, and highlights ambiguity preservation and determinism as core invariants.

### 0.3 Two-phase execution everywhere

For all 8 query types:

1. **Candidate generation**: cheap lexical narrowing (rpygrep) and/or prebuilt indexes
2. **Semantic confirmation + enrichment**: parse only candidate files, attach metadata (scope, qnames, call edges), return “bundle” output

This is how you hit “avoid followup” without parsing the world for every question.

---

## 1. Program structure

### 1.1 Package layout

```
codeintel/
  api/                 # request/response models, JSON schema
  repo/                # file enumeration, content store, line index
  engines/
    rpygrep_engine.py
    treesitter_engine.py
    libcst_engine.py
    astgrep_engine.py
  index/
    schema.py          # dataclasses for symbol/ref/call/import/test/wiring rows
    builder.py         # batch index build + incremental update hooks
    store.py           # in-memory store + optional persistence
  queries/
    q1_symbol.py
    q2_refs.py
    q3_callpath.py
    q4_pattern.py
    q5_contract.py
    q6_wiring.py
    q7_precedent.py
    q8_impact.py
  ranking/
    heuristics.py
  util/
    spans.py
    hashing.py
    snippet.py
```

### 1.2 Core internal record schema

You want *rows*, not objects with pointers:

**SymbolRow**

* `symbol_id`: stable hash
* `path`, `def_bspan`, `def_range`
* `language`
* `kind ∈ {function,class,method,variable,constant,module}`
* `name`, `qname_candidates[]` (list, not scalar)
* `signature_summary` (string + structured form)
* `doc_preview` (first N lines)
* `container_symbol_id | None`
* `exported: bool`
* `confidence: float` and `provenance: {engine, rule_id|query_id}`

**RefRow**

* `ref_id`, `path`, `ref_bspan`, `ref_range`
* `role ∈ {call,import,instantiate,inherit,override,read,write,decorator,wiring}`
* `name_text` (lexical) + `qname_candidates[]` (semantic)
* `enclosing_symbol_id`
* `snippet`

**CallEdge**

* `caller_symbol_id`
* `callsite_bspan`
* `callee_label` (syntactic dotted label)
* `callee_symbol_candidates[]`
* `arg_map`: `{param_name -> arg_snippet}` when possible
* `confidence`

**WiringEdge**

* `entry_kind ∈ {route,cli,plugin,di,envflag,configkey}`
* `entry_key` (string; e.g. route path, env var)
* `hook_bspan`
* `target_symbol_candidates[]`

---

## 2. Engine integration plan

## 2.1 rpygrep engine for lexical narrowing and fast inventory

You rely on ripgrep JSON because it’s structured, evented, and byte-offset anchored. rpygrep exposes the event stream (`BeginEvent`, `MatchEvent`, `ContextEvent`, `EndEvent`, `SummaryEvent`) and provides options for context lines, file limits, and caps like `max_count` / `max_file_size`.

### Core snippet: streaming JSON events into `LexMatch` rows

```python
# engines/rpygrep_engine.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Iterator, Optional

import rpygrep
from rpygrep import Options, patterns_are_not_regex

@dataclass(frozen=True)
class LexSubmatch:
    start_byte_in_line: int
    end_byte_in_line: int
    match_text: str

@dataclass(frozen=True)
class LexMatch:
    path: str
    line_number: int
    line_text: str
    submatches: list[LexSubmatch]
    # optional context
    before: list[str]
    after: list[str]

def run_rg_json(
    pattern: str,
    paths: list[str],
    *,
    fixed_string: bool,
    glob: Optional[list[str]] = None,
    max_count: Optional[int] = None,
    before_context: int = 0,
    after_context: int = 0,
    max_file_size: Optional[str] = None,
) -> Iterator[LexMatch]:
    # rpygrep uses Options; docs highlight context + caps + JSON emission.:contentReference[oaicite:8]{index=8}:contentReference[oaicite:9]{index=9}
    opts = Options(as_json=True)
    opts.before_context = before_context
    opts.after_context = after_context
    if max_count is not None:
        opts.max_count = max_count
    if max_file_size is not None:
        opts.max_file_size = max_file_size
    if glob:
        opts.glob = glob

    if fixed_string:
        patterns_are_not_regex(opts)  # switch rg mode for literal patterns.:contentReference[oaicite:10]{index=10}

    # Maintain rolling context windows keyed by current file.
    before_buf: list[str] = []
    after_pending: int = 0
    last_match: Optional[LexMatch] = None

    for ev in rpygrep.run(pattern, paths, opts):  # yields typed events incl MatchEvent/SummaryEvent.:contentReference[oaicite:11]{index=11}
        t = type(ev).__name__
        if t == "ContextEvent":
            # attach to before/after depending on whether we just matched
            line = ev.data["lines"]["text"]
            if after_pending > 0 and last_match is not None:
                last_match.after.append(line)
                after_pending -= 1
            else:
                before_buf.append(line)
                if len(before_buf) > before_context:
                    before_buf.pop(0)

        elif t == "MatchEvent":
            d = ev.data
            path = d["path"]["text"]
            line = d["lines"]["text"]
            line_no = d["line_number"]
            subs = [
                LexSubmatch(sm["start"], sm["end"], sm["match"]["text"])
                for sm in d["submatches"]
            ]
            m = LexMatch(path, line_no, line, subs, before=before_buf[:], after=[])
            yield m
            last_match = m
            after_pending = after_context
            before_buf = []  # reset per rg semantics

        else:
            continue
```

**What this enables for “best-in-class depth”**

* you can always return: `path + exact line + submatch byte offsets + context`, and then *upgrade* those to global `ByteSpan` via file line-index tables (next section)

---

## 2.2 tree-sitter engine for cross-language structural extraction

### 2.2.1 Query compilation and IR introspection

You compile with `Query(language, source)` and should interrogate its IR: `pattern_count`, capture tables via `capture_name`, and (optionally) pattern source slices via `start_byte_for_pattern` / `end_byte_for_pattern`.

This is critical for your agent UX: every match should include **which pattern** produced it and what that pattern “means”.

### 2.2.2 Query execution with guardrails

At scale you must:

* limit the scanned region with `QueryCursor.set_byte_range` / `set_point_range`
* cap pathological queries with `set_match_limit`, and detect overflow via `did_exceed_match_limit`

### Core snippet: query pack compilation + safe run wrapper

```python
# engines/treesitter_engine.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterator

from tree_sitter import Query, QueryCursor, QueryError

@dataclass(frozen=True)
class TSMatch:
    pattern_index: int
    captures: dict[str, list[Any]]  # Node list per capture name

def compile_query(language, source: str) -> Query:
    try:
        return Query(language, source)  # canonical constructor; QueryError on invalid source.:contentReference[oaicite:14]{index=14}
    except QueryError as e:
        raise ValueError(f"Invalid tree-sitter query: {e}") from e

def build_capture_index(q: Query) -> dict[int, str]:
    # q.capture_name(i) is the canonical capture lookup.:contentReference[oaicite:15]{index=15}
    return {i: q.capture_name(i) for i in range(q.capture_count)}

def run_query_bounded(
    q: Query,
    root_node,
    *,
    byte_start: int | None = None,
    byte_end: int | None = None,
    match_limit: int = 50_000,
) -> Iterator[TSMatch]:
    cur = QueryCursor(q)
    cur.set_match_limit(match_limit)  # cap runaway patterns.:contentReference[oaicite:16]{index=16}
    if byte_start is not None and byte_end is not None:
        cur.set_byte_range(byte_start, byte_end)  # range-limited scanning.:contentReference[oaicite:17]{index=17}

    for m in cur.matches(root_node):
        # m.captures is a list[(Node, capture_id)] in py-tree-sitter;
        # normalize to dict[name -> [Node...]]
        out: dict[str, list[Any]] = {}
        for node, cap_id in m.captures:
            name = q.capture_name(cap_id)
            out.setdefault(name, []).append(node)
        yield TSMatch(m.pattern_index, out)

    if cur.did_exceed_match_limit():  # detect truncation; mark result partial.:contentReference[oaicite:18]{index=18}
        # upstream: attach {partial=true, reason="match_limit"} to response
        return
```

### 2.2.3 Incremental parsing to support query 8 and “confirm scope” fast

Tree-sitter incremental parsing is a two-step contract: `Tree.edit(...)` then `Parser.parse(new_bytes, old_tree=...)`, then compute invalidation with `old_tree.changed_ranges(new_tree)`. Also: `changed_ranges` is only valid if you edited the *old* tree correctly; and `Node.text` is invalid after an edit unless reacquired. 

You will use this to:

* update indexes for changed ranges only
* run structural queries only over invalidated spans

---

## 2.3 LibCST engine for Python-only semantic enrichment

LibCST is your “semantic upgrade path” for Python:

* You can consume metadata via `wrapper.resolve(Provider)` or via visitor `METADATA_DEPENDENCIES` + `get_metadata`.
* Do not persist CST nodes; wrapper deep-copies modules by default, so node identity is wrapper-specific.
* For repo-wide semantics (FullyQualifiedNameProvider / TypeInferenceProvider), use `FullRepoManager` caches; for bytes-first fidelity, parse bytes yourself and inject cache from `get_cache_for_path`.

### Core snippet: bytes-first wrapper creation with repo cache

```python
# engines/libcst_engine.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import libcst as cst
from libcst.metadata import (
    MetadataWrapper,
    FullRepoManager,
    FullyQualifiedNameProvider,
    ScopeProvider,
    QualifiedNameProvider,
    PositionProvider,
    ByteSpanPositionProvider,
)

@dataclass(frozen=True)
class PyFileContext:
    wrapper: MetadataWrapper

def build_repo_manager(repo_root: Path, rel_paths: set[str]) -> FullRepoManager:
    providers = {FullyQualifiedNameProvider}  # add TypeInferenceProvider optionally
    mgr = FullRepoManager(repo_root, rel_paths, providers)
    mgr.resolve_cache()  # explicit precompute is recommended for controlled cost/forking.:contentReference[oaicite:24]{index=24}
    return mgr

def load_python_file(repo_root: Path, rel_path: str, mgr: FullRepoManager | None) -> PyFileContext:
    raw = (repo_root / rel_path).read_bytes()
    module = cst.parse_module(raw)  # bytes parse supported & preserves encoding/newlines at module level.:contentReference[oaicite:25]{index=25}
    cache = mgr.get_cache_for_path(rel_path) if mgr is not None else None  # per-file cache shard.:contentReference[oaicite:26]{index=26}
    w = MetadataWrapper(module, cache=cache)
    return PyFileContext(wrapper=w)
```

**Provider set you should standardize for your 8 queries**

* always: `PositionProvider`, `ByteSpanPositionProvider` (spans are your keys)
* references/contracts: `ScopeProvider` (assignments/accesses)
* symbol identity: `QualifiedNameProvider` (module-relative), plus optionally `FullyQualifiedNameProvider` via manager cache
* container chain: `ParentNodeProvider` (for enclosing symbol)

---

## 2.4 ast-grep-py engine for AST pattern scan and rewrite planning

ast-grep-py is your “AST rule engine”:

* `find/find_all` accept **rule kwargs** or a **config object** supporting `constraints` and `utils`
* constraints run **after** core rule matching, and constrained metavariables usually don’t work inside `not`
* Python replacement is **not metavariable-templated**; you must construct replacement strings from captures (e.g. `node["A"].text()`)
* edits must be planned, overlap-resolved, then applied in descending-start order for safety

### Core snippet: deterministic match records from ast-grep nodes

```python
# engines/astgrep_engine.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ast_grep_py import SgRoot  # type: ignore

@dataclass(frozen=True)
class AGMatch:
    path: str
    start: int
    end: int
    kind: str
    excerpt: str
    captures: list[dict[str, Any]]
    rule_id: str

def run_astgrep_find_all(
    *,
    src: str,
    path: str,
    language: str,
    rule_id: str,
    config_or_kwargs: Mapping[str, Any],
) -> list[AGMatch]:
    root = SgRoot(src, language).root()
    # kwargs vs config supported; config enables constraints/utils.:contentReference[oaicite:37]{index=37}
    is_full_config = any(k in config_or_kwargs for k in ("rule", "utils", "constraints", "transform"))
    nodes = root.find_all(dict(config_or_kwargs)) if is_full_config else root.find_all(**dict(config_or_kwargs))

    # stable ordering by range start/end/kind/text hash is recommended for deterministic outputs.:contentReference[oaicite:38]{index=38}
    def key(n):
        r = n.range()
        return (int(r.start.index), int(r.end.index), n.kind(), n.text())
    nodes = sorted(nodes, key=key)

    out: list[AGMatch] = []
    for n in nodes:
        r = n.range()
        start, end = int(r.start.index), int(r.end.index)
        out.append(
            AGMatch(
                path=path,
                start=start,
                end=end,
                kind=n.kind(),
                excerpt=n.text(),
                captures=[],  # fill from node["A"].text() etc in capture-plan layer
                rule_id=rule_id,
            )
        )
    return out
```

---

## 3. Index build plan

You can answer all 8 queries from a unified index with ~6 row families:

1. `symbols` (defs)
2. `refs` (read/write/import/inherit/override/call sites)
3. `calls` (call edges, callee labels + candidates)
4. `imports` (module graph + imported names)
5. `wiring` (framework/registration edges)
6. `tests` (test functions + which symbols they touch + assertion summaries)

### 3.1 Python indexing recipe with LibCST

Use the LibCST “facts not CST objects” stance: emit spans + IDs + strings; preserve ambiguity as lists.

**Defs**

* `FunctionDef`, `ClassDef`: emit SymbolRow
* signature extraction: normalize params using LibCST `Parameters` fields 1:1 (posonly, slash sentinel, star args, kwonly, kwargs).

**Refs**

* Use `ScopeProvider` to iterate `scope.assignments` and `scope.accesses`, where `Access.referents` is a set of candidate assignments (store list).

**Calls**

* For each `Call` node:

  * `callee_label = helpers.get_full_name_for_node(call.func)` as cheap syntactic label
  * semantic candidates from `QualifiedNameProvider` / `FullyQualifiedNameProvider` where available (store list)

**Imports**

* `Import` / `ImportFrom`: store edges; optionally compute absolute module fallback using LibCST helpers for relative imports if you don’t enable full repo providers.

### 3.2 Non-Python indexing recipe with tree-sitter

For each language `L`, define query packs:

* `defs.scm` captures: `@def.name`, `@def.node`, optional `@def.sig`
* `calls.scm` captures: `@call.callee`, `@call.node`
* `imports.scm` captures: `@import.module`, `@import.name`

Use query IR introspection to:

* prebuild capture tables (`capture_name`)
* record pattern provenance (`start_byte_for_pattern`/`end_byte_for_pattern`) so every match can explain “why it matched”

### 3.3 Pattern packs validation

Before running query packs at scale, lint them against `node-types.json` to avoid silent-zero-match failures and to keep range-limiting viable (rooted/local patterns).

---

## 4. Unified query API

### 4.1 Common request envelope

All 8 query types share:

```json
{
  "query_type": "symbol.resolve | symbol.refs | call.path | pattern.scan | contract | wiring | precedent | impact",
  "language": "python | ts | go | ...",
  "scope": { "paths": ["..."], "exclude": ["..."], "globs": ["**/*.py"] },
  "budget": {
    "max_files": 400,
    "max_matches": 20000,
    "max_depth": 3,
    "timeout_ms": 2000
  },
  "expansion": {
    "include_top_refs": 20,
    "include_tests": 10,
    "include_wiring": true
  }
}
```

**Critical invariant:** every response returns `{partial: bool, truncated_reason?: ...}` when any underlying engine hit limits (rg `max_count`, tree-sitter match limit, etc.).

---

# 5. Query handlers

## Query 1: Symbol resolution

### Request

```json
{
  "query_type": "symbol.resolve",
  "symbol": "X",
  "context": { "path": "optional/current.py", "byte_offset": 1234 },
  "language": "python",
  "budget": { "max_defs": 50 }
}
```

### Execution

1. **Candidate generation**

* rpygrep fixed-string scan for likely def forms (language-specific templates)

  * Python: `"def X"`, `"class X"`, `"X ="`, `"import X"`, `"from ... import X"`
* also allow direct lookup via `symbols_by_name["X"]` if index already built.

2. **Semantic confirmation**

* If `language=="python"`:

  * parse candidate files with LibCST wrapper (bytes-first recommended)
  * emit candidate defs; attach:

    * `ByteSpanPositionProvider` for `def_span`
    * `QualifiedNameProvider` / `FullyQualifiedNameProvider` candidates
    * signature from normalized Parameters fields
    * doc preview from docstring node (store as excerpt text)

3. **Ranking**
   Score each candidate def:

* exact name match (required)
* prefer exported/public (top-level, not `_` prefixed)
* prefer closest to context path (same module/package)
* prefer “definition” nodes over assignments/import reexports

4. **Best-in-class expansions to avoid followups**
   Return alongside the winning def:

* top N refs (Query 2) grouped by prod/test
* immediate call neighbors (Query 3, depth=1 outgoing + 1 incoming if index exists)
* wiring edges mentioning it (Query 6)
* impact slice (Query 8, shallow)

### Response

```json
{
  "symbol_id": "...",
  "def_span": { "path":"...", "start_byte":..., "end_byte":... },
  "kind": "function|class|...",
  "signature": { "text":"...", "params":[...], "returns":"..." },
  "doc_preview": "...",
  "qname_candidates": ["..."],
  "related": { "top_refs":[...], "wiring":[...], "impact":[...] },
  "partial": false
}
```

---

## Query 2: Enumerate usages

### Request

```json
{
  "query_type": "symbol.refs",
  "symbol_id": "optional",
  "symbol": "X",
  "language": "python",
  "grouping": ["by_module", "prod_vs_test"],
  "budget": { "max_refs": 2000 }
}
```

### Execution

1. **Candidate generation**

* rpygrep fixed-string for word-boundary occurrences of `X` (or import forms)
* if index exists, fetch `refs_by_symbol_id[symbol_id]`

2. **Semantic classification**

* Python:

  * use LibCST to classify each occurrence:

    * import vs call vs attribute vs assignment
  * use `ScopeProvider` to distinguish reads/writes (access vs assignment) and preserve referent ambiguity sets
  * attach enclosing symbol via parent chain (ParentNodeProvider-based)

3. **Dedup + stable sort**

* dedup by `(path, start_byte, end_byte, role)`
* stable order as described in §0.2

4. **Ranking**

* direct calls > imports > comments/docs
* prod > tests
* closer modules first

### Response

```json
{
  "symbol_id": "...",
  "refs": [
    {
      "ref_span": {...},
      "role": "call|import|read|write|inherit|override",
      "enclosing_symbol_id": "...",
      "snippet": "...",
      "confidence": 0.0-1.0
    }
  ],
  "groups": { "by_module": {...}, "prod_vs_test": {...} },
  "partial": true|false
}
```

---

## Query 3: Trace call path

### Request

```json
{
  "query_type": "call.path",
  "target_symbol_id": "...",
  "entrypoints": [
    {"kind":"function", "symbol_id":"..."},
    {"kind":"wiring", "entry_key":"/v1/foo"}
  ],
  "direction": "entry_to_target|outgoing_from_target",
  "budget": { "max_depth": 6, "max_paths": 5 }
}
```

### Execution

Prereq: you need a call graph index (`calls` edges) from LibCST for Python and tree-sitter packs for other languages.

1. Resolve `entrypoints`

* if wiring entrypoints are provided, resolve to handler symbol IDs via wiring index (Query 6 index)

2. BFS with budgets

* BFS over `caller -> callee_candidates`
* maintain:

  * `visited_symbol_id` with best-known depth
  * `path` as list of edges
* prune:

  * if edge confidence < threshold and budget tight
  * if symbol is in excluded directories

3. Edge enrichment
   For each edge include:

* `caller_symbol_id`
* `callsite_span`
* `callee_symbol_id` (resolved) or `callee_label` (syntactic)
* argument mapping:

  * Python: map call args to def params using stored signature normalization (posonly/kwonly/star args)

### Response

Return up to `max_paths` paths, each a list of `(caller → callsite → callee)` edges, with spans + (optional) arg maps.

---

## Query 4: Pattern and policy scan

You support two submodes: **lexical** (rpygrep) and **AST** (ast-grep + optionally tree-sitter query packs).

### Request

```json
{
  "query_type": "pattern.scan",
  "mode": "lexical|astgrep|treesitter",
  "language": "python",
  "pattern": "TODO|unsafe_call(",
  "astgrep": { "rule_id":"no-unsafe", "config": { ... } },
  "budget": { "max_matches": 50000, "before_context": 2, "after_context": 2 }
}
```

### Execution

* `mode=="lexical"`: use rpygrep JSON events; return structured match records (path + line + submatches + context)
* `mode=="astgrep"`:

  * use config-mode rules when you need constraints/utils; note constraints are post-filtered and have `not` limitations
  * return match records with `start_index/end_index` and stable ordering (as in snippet)
* `mode=="treesitter"`:

  * compile query, introspect capture table, run bounded cursor with match limits

### Best-in-class output

Return:

* `match_text_sha1`
* `rule_id` / `pattern_index` provenance
* `enclosing_symbol_id` (computed via LibCST ParentNodeProvider for Python, or tree-sitter parent walk)
* stable ordering guarantees

---

## Query 5: Contract extraction

### Request

```json
{
  "query_type": "contract",
  "target_symbol_id": "...",
  "include": { "tests": true, "docs": true, "examples": true },
  "budget": { "max_tests": 25, "max_docs_hits": 50 }
}
```

### Execution

1. **Candidate discovery**

* tests: rpygrep for refs to symbol name within `/tests`, `/test`, `*_test.py`, etc.
* docs/examples: rpygrep in `README*`, `docs/**`, `examples/**`

2. **Confirm + extract**

* parse candidate test files with LibCST
* identify:

  * test function defs (name prefix `test_` or framework conventions)
  * assertions: `assert` statements + common patterns (`pytest.raises`, etc.)
* attach spans + minimal snippets

3. **Minimal test set selection**
   Greedy set cover:

* universe = “distinct behaviors” proxies:

  * unique call edges into the target
  * unique error-mode assertions (exception types, messages)
    Pick tests maximizing uncovered behaviors until cap.

### Response

* ranked tests (with key assertions)
* docs/examples references
* “source of truth” anchors: links to def spans + assertion spans

---

## Query 6: Wiring discovery

### Request

```json
{
  "query_type": "wiring",
  "target_symbol_id": "optional",
  "target_symbol": "optional",
  "wiring_kinds": ["route","cli","plugin","di","envflag","configkey"],
  "budget": { "max_edges": 2000 }
}
```

### Execution

You implement wiring as **declarative pattern packs** per framework style:

* routes: decorator-based, registration-call-based
* plugins: `register(X)` patterns
* config/env: `os.environ["KEY"]`, `getenv("KEY")`, config schema registries

Implementation strategy:

1. Candidate generation: rpygrep by framework keywords (`@router.`, `add_route`, `register`, env key strings)
2. Structural confirmation: ast-grep patterns
3. Semantic resolution: LibCST resolves handler symbol candidates and returns spans

Use ast-grep config mode when you need reusable `utils` matchers (e.g., “is string literal path”).

### Response

Return a wiring map:

* `entrypoint → framework hook span → target symbol candidate(s)`
* plus extracted keys/defaults where present

---

## Query 7: Precedent search

### Request

```json
{
  "query_type": "precedent",
  "prototype": { "symbol_id":"..." } | { "snippet_span": {...} },
  "kind": "function|class|...",
  "k": 5,
  "budget": { "candidate_pool": 2000 }
}
```

### Execution

1. Candidate pool

* all symbols of same kind
* optionally constrained by wiring kind (e.g., only route handlers)

2. Fingerprint extraction
   For each candidate:

* signature shape (arity, kwonly presence, return annotation presence)
* decorator set (syntactic dotted name)
* called API labels (callee_label bag)
* import modules used
* docstring keywords
  Store as:
* `feature_set_hashes` (e.g., 64-bit hashes)
* plus a small human-readable “why” map

3. Similarity
   Weighted Jaccard:

* decorators weight high
* called APIs medium
* signature shape medium
* doc keywords low

### Response

Top‑K exemplars:

* why matched (top overlapping features)
* diff-able core snippet (span + excerpt)
* links to their wiring (if any) to show “where to imitate”

---

## Query 8: Change impact and scope confirmation

### Request

```json
{
  "query_type": "impact",
  "target_symbol_id": "...",
  "slice": { "caller_depth": 2, "callee_depth": 1 },
  "include": { "imports": true, "tests": true, "public_boundary": true },
  "budget": { "max_nodes": 5000 }
}
```

### Execution

1. Build a bounded transitive slice

* Upward callers: follow `calls` edges reversed + `refs` that are calls/imports
* Downward callees: follow `calls` edges forward

2. Boundary crossings
   Define package boundary as first N path components (configurable).
   Mark edges where:

* caller package != callee package
* symbol exported/public is touched (python: top-level non-underscore; other langs configurable)

3. Likely affected tests
   Intersect slice nodes with `tests → symbols` relations built in Query 5 indexing.

4. Incremental speedups
   If repo state changes:

* update only invalidated spans using tree-sitter incremental parse changed ranges
* rerun call/def extraction queries only in those ranges using `QueryCursor.set_byte_range`

### Response

* slice graph (nodes + edges)
* boundary crossing list
* affected tests
* risk score + explanation:

  * `risk = w1*#callers + w2*#packages + w3*#public_crossings + w4*#tests`

---

## 6. “Best-in-class depth” defaults that prevent followups

### 6.1 Always return bundles, not single answers

Even Query 1 should return a small “bundle”:

* definition
* top references (10–20)
* top wiring edges
* top tests
* 1-hop call neighbors

Agents ask followups when you omit the *next* natural question; bundling suppresses that.

### 6.2 Always annotate confidence and provenance

For every row:

* `engine`: `rpygrep|libcst|treesitter|astgrep`
* `rule_id|query_id|pattern_index`
* `confidence`
* `partial/truncated` flags

Tree-sitter and rg both have explicit truncation/limit surfaces you can detect and must propagate.

### 6.3 Never store CST/AST node identities

LibCST wrappers deep-copy; storing nodes across wrappers is explicitly called out as the #1 failure mode; persist spans + IDs only.

---

## 7. What this gives you for each query type

* **Q1** returns canonical def + qnames + signature/doc + “next questions” bundle
* **Q2** returns classified refs, grouped and ranked, with enclosing symbols and snippets
* **Q3** returns call slices with spans and (when possible) arg/param maps
* **Q4** returns machine-readable match records with rule provenance and stable ordering
* **Q5** returns minimal tests + key assertions + docs/examples anchors
* **Q6** returns wiring edges (entrypoint→hook→handler) + config keys/defaults when present
* **Q7** returns top exemplars with “why” and diff-able snippet spans
* **Q8** returns bounded dependency slice + boundary crossings + affected tests + risk

If you want, the next useful artifact would be: **a concrete “query pack library”** (tree-sitter `.scm` packs + ast-grep config packs + python framework wiring packs) for your target languages/frameworks, because that’s the part that turns this architecture into a turnkey agent capability.

Got it – I’m on it and have updated the implementation to include rpygrep in the wiring packs.


[Download the Python query pack library (zip)](sandbox:/mnt/data/python_query_pack_library.zip)

This is a **Python-only** query pack library containing:

* **tree-sitter `.scm` packs** (Python): defs/imports/calls/tests + wiring fallbacks
* **ast-grep config packs** (Python, YAML): FastAPI/Flask/Click/Typer/argparse/env/plugin-entrypoints + a small policy pack
* **wiring pack specs** (JSON): **each wiring pack explicitly starts with an rpygrep candidate stage** (core), then confirms structurally via ast-grep, and optionally falls back to tree-sitter

The library structure mirrors the “candidate → confirm → anchor” pipeline design:

* **rpygrep** stage gives scalable candidate selection, bounded output knobs, and consistent evidence context (line + submatch spans + optional before/after context). The included runner script demonstrates building searches with safe defaults, exclusions, context, and regex behavior via the rpygrep surface shown in the deep-dive. 
* **ast-grep** stage provides structural confirmation using code-parsed patterns with metavariables (`$NAME`, `$$$ARGS`) and post-match constraints. 
* **tree-sitter** stage provides uniform span anchoring and “pack contract” tooling (query is an IR with pattern/capture tables), plus bounded execution knobs (`set_*_range`, `match_limit`) on `QueryCursor`. 

## What’s inside the zip

Top-level:

* `manifest.json` – enumerates all packs (tree-sitter, ast-grep, rpygrep presets/pattern groups, wiring packs)
* `README.md` – pack philosophy + output contract

### rpygrep assets

* `rpygrep/presets/default_interactive.json`
* `rpygrep/presets/audit_deterministic.json`
* `rpygrep/patterns/*.json` – pattern groups used by wiring packs (FastAPI routes, Flask routes, Click, Typer, argparse, env vars, entrypoints, Depends)

These are designed to map cleanly onto rpygrep’s search-builder behaviors (safe defaults, regex engine selection, output shaping via context/max_count/max_file_size). 

### tree-sitter packs (Python)

* `tree_sitter/python/defs_full.scm`
* `tree_sitter/python/imports_full.scm`
* `tree_sitter/python/calls_full.scm`
* `tree_sitter/python/tests_py.scm`
* `tree_sitter/python/wiring_fastapi.scm`
* `tree_sitter/python/wiring_flask.scm`

These are authored for `QueryCursor.matches()` (relational/coherent results), which means patterns are written to avoid duplicate capture names inside a single match (because `matches()` returns `dict[capture_name -> Node]` and can drop duplicates). 

### ast-grep packs (Python, YAML)

* `ast_grep/python/fastapi.yaml`
* `ast_grep/python/flask.yaml`
* `ast_grep/python/click.yaml`
* `ast_grep/python/typer.yaml`
* `ast_grep/python/argparse.yaml`
* `ast_grep/python/env.yaml`
* `ast_grep/python/entrypoints.yaml`
* `ast_grep/python/policy_security.yaml`

Rules use ast-grep’s metavariable pattern syntax (`$NAME`, `$$$ARGS`), and apply constraints as post-filters (so your runner can do “match broadly → constrain precisely”). 

### wiring packs (Python) — **with rpygrep embedded**

* `wiring_packs/python/fastapi_routes.json`
* `wiring_packs/python/fastapi_depends.json`
* `wiring_packs/python/flask_routes.json`
* `wiring_packs/python/click_cli.json`
* `wiring_packs/python/typer_cli.json`
* `wiring_packs/python/argparse_cli.json`
* `wiring_packs/python/env_vars.json`
* `wiring_packs/python/entrypoints.json`

Each wiring pack spec begins with:

1. **rpygrep candidate stage** (preset + pattern group)
2. **ast-grep structural stage** (rule IDs + rule file)
3. optional **tree-sitter fallback stage**

This directly encodes “rpygrep is core” into the wiring-pack contract.

### tools

* `tools/run_rpygrep_profile.py` – reference runner that loads a preset + pattern group and emits normalized JSON
* `tools/validate_pack_manifest.py`
* `tools/README.md`

The runner follows the rpygrep approach where you can treat literal patterns as escaped regex (or switch to fixed-strings mode) depending on your preset; this aligns with the deep-dive guidance showing `re.escape(...)` when you want “literal semantics” without fixed-string global flags. 

## Notes for your implementation (so this becomes turnkey)

* **rpygrep output contract**: if you use `run_direct(...)`, you can additionally collect a `SummaryEvent` (timing/counts) for telemetry; that’s useful for agent budgets and determinism checks. 
* **tree-sitter execution**: your “best-in-class” runner should treat `Query` as an IR (pattern indexes, capture tables) and enforce bounded execution via `QueryCursor` (`match_limit`, `set_byte_range/set_point_range`). 
* **tree-sitter pack variants**: if you later add “safe mode”/“fast path” variants, treat them as separate compiled queries (toggle disabling is destructive on a Query instance). 

If you want the next artifact after this zip, the obvious follow-on is a **“wiring pack executor”** that takes one of the `wiring_packs/python/*.json` specs and produces the final **wiring map** edges (with spans + enriched metadata), using the bounded-execution and output-shaping knobs described above.

[Download the Python wiring pack executor (zip)](sandbox:/mnt/data/python_wiring_pack_executor.zip)

This artifact is a **turnkey “wiring pack executor”** for **Python-only repos** that:

* **always starts with rpygrep** (candidate discovery is mandatory)
* confirms/extracts structure via **ast-grep-py**
* enriches spans + enclosing symbol **qname** via **LibCST**
* emits a **wiring edge JSON** that’s meant to be agent-consumable without follow-up queries

---

## What’s in the zip

### 1) `wiring_executor/` (the executor)

Core entrypoints:

* `execute_pack(repo_root, pack_file, pack_root=...) -> dict`
* CLI:

  * `python -m wiring_executor.cli run ...`
  * `python -m wiring_executor.cli run-all ...`

Key behaviors:

* **rpygrep stage** runs first and yields candidate files (bounded by `--max-candidate-files`)
* **ast-grep stage** runs only on candidate files
* **LibCST enrichment** computes:

  * enclosing def (function/method/class)
  * qualified name (module + nested scopes)
  * byte spans + line/col spans
* optional “best effort” handler resolution:

  * if a match captures `HANDLER=foo`, it tries to resolve `foo` in-file
  * optionally, it can do a bounded **cross-repo** rpygrep `def foo` lookup and confirm with LibCST

### 2) `packs/` (self-contained, runnable packs)

A complete set of **Python wiring packs** (all start with rpygrep):

* `packs/wiring_packs/python/fastapi_routes.json`
* `packs/wiring_packs/python/fastapi_depends.json`
* `packs/wiring_packs/python/flask_routes.json`
* `packs/wiring_packs/python/click_cli.json`
* `packs/wiring_packs/python/typer_cli.json`
* `packs/wiring_packs/python/argparse_cli.json`
* `packs/wiring_packs/python/env_vars.json`
* `packs/wiring_packs/python/entrypoints.json`

Plus:

* `packs/rpygrep/presets/*.json`
* `packs/rpygrep/patterns/*.json`
* `packs/ast_grep/python/*.yaml` (rules for each wiring domain)
* `packs/manifest.json` (what’s included)

---

## Install requirements (runtime)

```bash
pip install rpygrep ast-grep-py libcst pyyaml
# AND ripgrep binary `rg` must be available in PATH
```

---

## Run it

### Run one pack (example: FastAPI routes)

```bash
python -m wiring_executor.cli run \
  --repo /path/to/your/repo \
  --pack /path/to/unzipped_artifact/packs/wiring_packs/python/fastapi_routes.json \
  --pack-root /path/to/unzipped_artifact/packs \
  --out wiring_fastapi_routes.json
```

### Run all included packs

```bash
python -m wiring_executor.cli run-all \
  --repo /path/to/your/repo \
  --packs /path/to/unzipped_artifact/packs/wiring_packs/python/*.json \
  --pack-root /path/to/unzipped_artifact/packs \
  --out wiring_all.json
```

Notes:

* If you omit `--pack-root`, the executor attempts to auto-detect it by walking upward until it finds `rpygrep/presets/`.
* Use `--no-cross-file-resolve` if you want zero extra repo-wide resolution passes.

---

## Output shape (what you get)

Each result is:

```json
{
  "pack_id": "...",
  "entry_kind": "...",
  "framework": "...",
  "partial": false,
  "edges": [
    {
      "edge_id": "…stable hash…",
      "pack_id": "...",
      "framework": "...",
      "entry_kind": "route|cli|envflag|plugin|di",
      "entry_key": "…human+agent key…",
      "hook_span": { "path": "...", "start_byte": 123, "end_byte": 156, "start_line": 10, "start_col": 2, "end_line": 10, "end_col": 35 },
      "target": {
        "kind": "function|method|class",
        "name": "handler_name",
        "qname": "pkg.module.Class.handler_name",
        "def_span": { "path": "...", "start_byte": 1000, "end_byte": 1400, "start_line": 40, ... }
      },
      "match": {
        "rule_id": "py.fastapi.route.decorator",
        "captures": { "PATH": "\"/items\"", "METHOD": "get", ... },
        "capture_spans": { "PATH": { ... }, "MATCH_NODE": { ... } },
        "rg_pattern_ids": ["@\\s*..."],
        "handler_resolution": { "strategy": "...", "candidates": [...] }
      },
      "evidence": {
        "excerpt": "@router.get(\"/items\")",
        "context": { "first_line": 9, "last_line": 11, "lines": ["...", "...", "..."] }
      }
    }
  ],
  "debug": { "rg_files": [...], "rg_hit_count": 123, "rg_partial": false }
}
```

This is intentionally structured so an agent can:

* rank edges
* group by entry_kind/framework/module
* jump to precise spans
* reason about “what’s wired to what” without having to ask you to run more searches

---

## Extending the executor (the “wiring packs are the API”)

If you want to add new wiring domains (still Python-only), you generally only do three things:

1. Add a new `packs/rpygrep/patterns/<domain>.json` (broad candidate discovery)
2. Add a new `packs/ast_grep/python/<domain>.yaml` (precise extraction + captures)
3. Add a new `packs/wiring_packs/python/<domain>.json` that combines them

The executor already supports:

* per-rule `entry_key_by_rule`
* per-rule `target_symbol_hint_by_rule`
* pack-level postprocess ops including `python.join_argparse_subcommands` (multi-match join step)

If you want, the next artifact after this would be a **“wiring edge → graph builder”** that turns these edges into a NetworkX graph (entrypoints/config keys/tests ↔ symbols) and supports queries like “transitive blast radius from this route” directly on the wiring graph.
