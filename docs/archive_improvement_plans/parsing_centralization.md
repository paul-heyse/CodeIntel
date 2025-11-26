

Here’s a concrete implementation plan to turn **parsing + span resolution + validation** into a coherent, first-class subsystem, with example code you can hand straight to agents.

I’ll cover:

1. Goals and current ingredients
2. Target package layout
3. Parser registry & parsing service
4. Span resolution service
5. Validation subsystem (base + domain reporters)
6. Wiring analytics modules through the subsystem
7. Docs & views for validation surfaces
8. Suggested implementation order / agent guidance

---

## 1. Goals & current ingredients

You already have:

* **Conceptual doc** `docs/analytics_parsing_and_span_resolution.md` tying together:

  * `FunctionParserKind` and `FunctionParserRegistry` (parser registry)
  * `resolve_span` for span resolution
  * `ValidationReporter` and `analytics.function_validation` table as the place where parse/span gaps are logged
* Code pieces:

  * `src/codeintel/analytics/functions/parsing.py` (created in the previous refactor)
  * Older modules like `analytics/function_parsing.py` and `analytics/span_resolver.py` (pre-split)
  * `tests/analytics/test_span_resolver.py`
  * `function_ast_cache` and `AnalyticsContext.function_ast_map` which give you cached function ASTs

The **design goal** now:

> For any analytic that needs ASTs or spans, there’s **one obvious way** to get them (the parsing subsystem), and **one obvious way** to report problems (validation reporters writing into validation tables).

---

## 2. Target package layout

Create a **cross-cutting package** for parsing & validation:

```text
src/codeintel/analytics/
    parsing/
        __init__.py
        models.py          # dataclasses for parse results & spans
        registry.py        # FunctionParserKind / FunctionParserRegistry
        function_parsing.py# actual Python parser impls (moved here)
        span_resolver.py   # resolve_span & helpers (moved here)
        validation.py      # BaseValidationReporter + domain-specific reporters
```

And then treat `analytics/functions/parsing.py` as the **functions-analytics adapter** that imports from `analytics.parsing`.

Compatibility shims:

* Keep `src/codeintel/analytics/span_resolver.py` as a thin wrapper that re-exports from `analytics.parsing.span_resolver` (so existing imports still work until you update them).

---

## 3. Parser registry & parsing service

### 3.1 `parsing/models.py`: basic types

Define reusable types for parse/AST/spans:

```python
# src/codeintel/analytics/parsing/models.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class SourceSpan:
    """Source span in (path, [start_line, end_line], [start_col, end_col])."""

    path: Path
    start_line: int
    start_col: int
    end_line: int
    end_col: int


@dataclass(frozen=True)
class ParsedFunction:
    """
    Language-agnostic parsed function representation.

    This is what analytics consume.
    """

    path: Path
    qualname: str
    function_goid_h128: int | None
    span: SourceSpan
    ast: Any  # language-specific AST node (Python ast.AST, etc.)
    docstring: str | None
    param_annotations: Mapping[str, Any]
    return_annotation: Any | None
    param_any_flags: Mapping[str, bool]
    return_is_any: bool
```

This gives every analytic a single “shape” to depend on.

### 3.2 `parsing/registry.py`: parser registry

Formalize the parser registry your doc already describes:

```python
# src/codeintel/analytics/parsing/registry.py
from __future__ import annotations

from collections.abc import Callable
from enum import Enum, auto
from pathlib import Path
from typing import Dict, Iterable, List

from .models import ParsedFunction


class FunctionParserKind(Enum):
    PYTHON = auto()
    PYTHON_TEST = auto()
    # future: JS, TS, etc.


# Signature for a per-file parser
ParseModuleFn = Callable[[Path, bytes], Iterable[ParsedFunction]]


class FunctionParserRegistry:
    """
    Registry for function parsers keyed by FunctionParserKind.

    Central place to look up the correct parser for a given module.
    """

    def __init__(self) -> None:
        self._by_kind: Dict[FunctionParserKind, ParseModuleFn] = {}

    def register(self, kind: FunctionParserKind, fn: ParseModuleFn) -> None:
        if kind in self._by_kind:
            raise ValueError(f"Parser already registered for kind {kind}")
        self._by_kind[kind] = fn

    def get(self, kind: FunctionParserKind) -> ParseModuleFn:
        try:
            return self._by_kind[kind]
        except KeyError as exc:
            raise KeyError(f"No parser registered for kind {kind}") from exc


_registry = FunctionParserRegistry()


def register_parser(kind: FunctionParserKind, fn: ParseModuleFn) -> None:
    _registry.register(kind, fn)


def get_parser(kind: FunctionParserKind) -> ParseModuleFn:
    return _registry.get(kind)
```

You can wire this up in module-level initialization:

```python
# src/codeintel/analytics/parsing/function_parsing.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .models import ParsedFunction, SourceSpan
from .registry import FunctionParserKind, register_parser
# import python-specific parsing helpers from existing code
import ast

def parse_python_module(path: Path, content: bytes) -> Iterable[ParsedFunction]:
    # reuse your existing function_parsing logic here
    source = content.decode("utf-8")
    module_ast = ast.parse(source, filename=str(path))
    # walk AST, build ParsedFunction objects with SourceSpan...
    yield from _extract_functions_from_ast(path, module_ast, source)


# Registration at import time
register_parser(FunctionParserKind.PYTHON, parse_python_module)
```

This pulls your existing Python-specific logic into a single place and makes it straightforward to add more parser kinds later.

### 3.3 `analytics/functions/parsing.py` becomes an adapter

Instead of owning parsing logic itself, `src/codeintel/analytics/functions/parsing.py` should be a thin adapter:

```python
# src/codeintel/analytics/functions/parsing.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from codeintel.analytics.parsing.models import ParsedFunction
from codeintel.analytics.parsing.registry import FunctionParserKind, get_parser


def parse_functions_in_module(
    path: Path,
    content: bytes,
    *,
    kind: FunctionParserKind = FunctionParserKind.PYTHON,
) -> Iterable[ParsedFunction]:
    """
    Adapter for function-level analytics: parse one module into ParsedFunction objects.
    """
    parser = get_parser(kind)
    return parser(path, content)
```

Any function-level analytics that need parsed functions can now import from `analytics.functions.parsing` and remain ignorant of registry details.

---

## 4. Span resolution service

Span resolution is the glue between “function identity” (GOID) and source code slices.

### 4.1 Move `span_resolver` under `analytics/parsing/`

Create:

```python
# src/codeintel/analytics/parsing/span_resolver.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping

from .models import ParsedFunction, SourceSpan


@dataclass(frozen=True)
class SpanResolutionResult:
    function_goid_h128: int
    span: SourceSpan
    path: Path


class SpanResolutionError(Exception):
    """Raised when we cannot resolve a span for a function GOID."""


def build_span_index(
    parsed_functions: Iterable[ParsedFunction],
) -> Mapping[int, SourceSpan]:
    """
    Build a mapping from function_goid_h128 to SourceSpan.

    This can be stored in memory or persisted to DuckDB for later lookup.
    """
    index: Dict[int, SourceSpan] = {}
    for fn in parsed_functions:
        if fn.function_goid_h128 is None:
            continue
        index[fn.function_goid_h128] = fn.span
    return index


def resolve_span(
    *,
    function_goid_h128: int,
    span_index: Mapping[int, SourceSpan],
) -> SpanResolutionResult:
    """
    Resolve a function GOID to a SourceSpan, or raise SpanResolutionError.

    This is the canonical way to get spans in analytics code.
    """
    try:
        span = span_index[function_goid_h128]
    except KeyError as exc:
        raise SpanResolutionError(
            f"No span for function_goid_h128={function_goid_h128}"
        ) from exc

    return SpanResolutionResult(
        function_goid_h128=function_goid_h128,
        span=span,
        path=span.path,
    )
```

This is a simplified abstraction. In your code, you may combine:

* `function_ast_cache`
* SCIP indices
* AST / CST offsets

…to build a richer `span_index`. But the *contract* is: **analytics never dig around in ASTs to find spans — they call `resolve_span`**.

### 4.2 Compatibility shim

Keep a stub at the old location:

```python
# src/codeintel/analytics/span_resolver.py
from __future__ import annotations

from .parsing.span_resolver import (
    SpanResolutionError,
    SpanResolutionResult,
    build_span_index,
    resolve_span,
)

__all__ = [
    "SpanResolutionError",
    "SpanResolutionResult",
    "build_span_index",
    "resolve_span",
]
```

This lets existing imports keep working while you update them gradually.

---

## 5. Validation subsystem

Now, unify validation into a small, explicit subsystem so every domain can use the same patterns.

### 5.1 `parsing/validation.py`: base + function/graph reporters

```python
# src/codeintel/analytics/parsing/validation.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, List, Optional, TypeVar

from codeintel.storage.gateway import StorageGateway
from codeintel.models.rows import (
    FunctionValidationRow,
    function_validation_row_to_tuple,
    GraphValidationRow,
    graph_validation_row_to_tuple,
)

RowT = TypeVar("RowT")


@dataclass
class BaseValidationReporter(Generic[RowT]):
    """
    Generic validation reporter, collecting rows in memory and flushing to DuckDB.

    Subclasses are responsible for creating the concrete row type.
    """

    repo: str
    commit: str
    rows: List[RowT] = field(default_factory=list)

    def flush(self, gateway: StorageGateway) -> None:
        """Subclasses must override to write rows to the appropriate table."""
        raise NotImplementedError


@dataclass
class FunctionValidationReporter(BaseValidationReporter[FunctionValidationRow]):
    """
    Validation reporter for function-level parsing/span issues.
    """

    parse_failed: int = 0
    span_not_found: int = 0
    unknown_functions: int = 0

    def record(
        self,
        *,
        function_goid_h128: Optional[int],
        kind: str,
        message: str,
    ) -> None:
        if kind == "parse_failed":
            self.parse_failed += 1
        elif kind == "span_not_found":
            self.span_not_found += 1
        elif kind == "unknown_function":
            self.unknown_functions += 1

        row: FunctionValidationRow = {
            "repo": self.repo,
            "commit": self.commit,
            "function_goid_h128": function_goid_h128,
            "kind": kind,
            "message": message,
        }
        self.rows.append(row)

    def flush(self, gateway: StorageGateway) -> None:
        if not self.rows:
            return
        tuples = [function_validation_row_to_tuple(r) for r in self.rows]
        con = gateway.con
        con.executemany(
            """
            INSERT INTO analytics.function_validation (
                repo, commit, function_goid_h128, kind, message
            ) VALUES (?, ?, ?, ?, ?)
            """,
            tuples,
        )
        self.rows.clear()


@dataclass
class GraphValidationReporter(BaseValidationReporter[GraphValidationRow]):
    """
    Validation reporter for graph-level issues (e.g., missing nodes/edges).
    """

    def record(
        self,
        *,
        graph_name: str,
        entity_id: str,
        kind: str,
        message: str,
    ) -> None:
        row: GraphValidationRow = {
            "repo": self.repo,
            "commit": self.commit,
            "graph_name": graph_name,
            "entity_id": entity_id,
            "kind": kind,
            "message": message,
        }
        self.rows.append(row)

    def flush(self, gateway: StorageGateway) -> None:
        if not self.rows:
            return
        tuples = [graph_validation_row_to_tuple(r) for r in self.rows]
        con = gateway.con
        con.executemany(
            """
            INSERT INTO analytics.graph_validation (
                repo, commit, graph_name, entity_id, kind, message
            ) VALUES (?, ?, ?, ?, ?)
            """,
            tuples,
        )
        self.rows.clear()
```

You can add more reporters later (e.g., `DocstringValidationReporter`, `HistoryValidationReporter`) by subclassing `BaseValidationReporter`.

### 5.2 Use in function analytics

In `analytics/functions/metrics.py`, you now use the centralized reporter instead of an ad hoc one:

```python
from codeintel.analytics.parsing.validation import FunctionValidationReporter
from codeintel.analytics.parsing.span_resolver import SpanResolutionError, resolve_span

def compute_function_metrics_and_types(
    gateway: StorageGateway,
    cfg: FunctionAnalyticsConfig,
    *,
    context: AnalyticsContext | None = None,
) -> None:
    con = gateway.con
    reporter = FunctionValidationReporter(cfg.repo, cfg.commit)

    # Build span index once (from context.function_ast_map or parsed functions)
    span_index = _build_span_index_for_snapshot(con, cfg, context)

    metrics_rows: list[FunctionMetricsRow] = []
    type_rows: list[FunctionTypesRow] = []

    for func in _iter_functions(con, cfg.repo, cfg.commit):
        try:
            span_result = resolve_span(
                function_goid_h128=func.goid,
                span_index=span_index,
            )
        except SpanResolutionError as exc:
            reporter.record(
                function_goid_h128=func.goid,
                kind="span_not_found",
                message=str(exc),
            )
            continue

        # typedness, metrics, type rows as before...
        ...

    _write_metrics(gateway, metrics_rows)
    _write_types(gateway, type_rows)
    reporter.flush(gateway)
```

Any parse failures deeper in the parsing logic should also call `reporter.record(kind="parse_failed", ...)`.

### 5.3 Use in graph analytics

`analytics.graph_validation` already persists graph validation issues; now it can just use `GraphValidationReporter`:

```python
from codeintel.analytics.parsing.validation import GraphValidationReporter

def validate_graphs(
    gateway: StorageGateway,
    cfg: GraphValidationConfig,
    *,
    context: AnalyticsContext | None = None,
) -> None:
    con = gateway.con
    reporter = GraphValidationReporter(cfg.repo, cfg.commit)

    # Example: ensure all call_graph nodes have a known function GOID
    for node in context.call_graph.nodes:
        if not _is_valid_function_goid(node, con, cfg.repo, cfg.commit):
            reporter.record(
                graph_name="call_graph",
                entity_id=str(node),
                kind="unknown_function_node",
                message="Node has no matching function record",
            )

    reporter.flush(gateway)
```

You get a **uniform validation story** across domains.

---

## 6. Wiring analytics modules through the subsystem

The main changes to analytics modules will be mechanical:

1. **Imports:**

   Replace direct imports of:

   * `analytics.span_resolver.resolve_span`
   * local `ValidationReporter` classes

   …with:

   ```python
   from codeintel.analytics.parsing.span_resolver import resolve_span, SpanResolutionError
   from codeintel.analytics.parsing.validation import FunctionValidationReporter, GraphValidationReporter
   ```

2. **Function-level analytics that depend on spans:**

   * `analytics/functions/metrics.py`
   * `analytics/cfg_dfg/...` (for CFG/DFG metrics keyed to function spans)
   * `analytics/semantic_roles`
   * `analytics/function_contracts` / `function_effects`
   * `analytics/config_data_flow` (if it uses AST spans)

   All of these should:

   * Build or reuse a **span index** (from context or DB).
   * Call `resolve_span(…)`.
   * On error, report via `FunctionValidationReporter` instead of swallowing or ad-hoc logging.

3. **Graph-level analytics:**

   * `analytics.graph_validation` should use `GraphValidationReporter`.
   * Any module that discovers inconsistent edges or missing nodes should report through it.

4. **Docstring / history / doc ingestion (optional extension):**

   * If you later add docstring validation or history consistency checks, create dedicated reporters (or extend the base one) that still follow the same path: collect rows → flush via `StorageGateway`.

---

## 7. Docs & views for validation

You already export `analytics.graph_validation` and `analytics.function_validation` as JSONL via docs export. To make this more consumable for MCP/agents:

### 7.1 Create a `docs.v_validation_summary` view

In your SQL schema (or view builder), define:

```sql
CREATE VIEW docs.v_validation_summary AS
SELECT
    'function' AS domain,
    repo,
    commit,
    function_goid_h128 AS entity_id,
    kind,
    message
FROM analytics.function_validation
UNION ALL
SELECT
    'graph' AS domain,
    repo,
    commit,
    NULL AS entity_id,
    kind,
    message
FROM analytics.graph_validation;
```

If you add more domains (docstrings, history), you can extend this view.

### 7.2 Export it in docs export

In your docs export builder (where you currently list the datasets to write to JSONL), add `docs.v_validation_summary` as one of the tables to export, alongside the per-table JSONLs.

Then the MCP layer / external tools can:

* Pull `docs.v_validation_summary` in one shot.
* Quickly assess “how healthy is this snapshot?” based on counts / types of validation events.

---

## 8. Implementation order & agent instructions

### 8.1 Suggested order

1. **Create parsing subsystem skeleton**

   * Add `analytics/parsing/__init__.py`, `models.py`, `registry.py`, `span_resolver.py`, `validation.py`.
   * Move existing parser code into `function_parsing.py` and register it in the registry.
   * Add shim `analytics/span_resolver.py` that re-exports.

2. **Wire function analytics**

   * Update `analytics/functions/metrics.py` to use:

     * `analytics.functions.parsing.parse_functions_in_module` (adapter)
     * `FunctionValidationReporter`
     * `resolve_span` from `analytics.parsing.span_resolver`
   * Ensure all parse/span failures are reported, not silently ignored.

3. **Wire graph validation**

   * Update `analytics.graph_validation` to use `GraphValidationReporter`.
   * Ensure any graph consistency checks feed into it.

4. **Update other span-dependent analytics**

   * `cfg_dfg`, `semantic_roles`, `function_contracts`, `config_data_flow` to use `resolve_span` + `FunctionValidationReporter`.

5. **Add `docs.v_validation_summary` and export**

   * Create the view, wire it into docs export.

6. **Clean up old validation code**

   * Remove any duplicated `ValidationReporter` classes.
   * Remove ad-hoc logging of parse/span issues in favor of the reporters.

### 8.2 Agent-friendly instructions

In `AGENTS.md` or a new `docs/improvement_plans/parsing_span_validation.md`, spell out:

* **Never parse or traverse ASTs ad-hoc to get spans.**
  Always go through `analytics.parsing` (`parse_functions_in_module`, `resolve_span`).

* **Any time parsing or span resolution fails, report it.**
  Use `FunctionValidationReporter` / `GraphValidationReporter` or another appropriate `BaseValidationReporter` subclass.

* **Validation tables are the source of truth about data quality.**
  Don’t add new “validation-like” tables; extend the existing reporters and rows.

---

If you’d like, I can next turn this into that markdown improvement-plan file (`docs/improvement_plans/parsing_span_validation_subsystem.md`) with actual TODO checkboxes and per-module work items, so you can drop it into the repo as the “playbook” for agents.
