
Let’s turn Epic 4 into something you can actually ship: a **shared AST semantics library** that all the “AST zoo” modules call into instead of each re-inventing their own little walker.

I’ll structure this as:

1. New `analytics/ast_features` package (models, patterns, extractors).
2. Wire **IO + concurrency** into the feature library (and refactor `tests_profiles/behavioral_tags` to use it).
3. Hook **function features** into `AnalyticsContext` and **semantic_roles**.
4. Show how to start using features from `data_models`, `entrypoint_detectors`, and `dependencies`.
5. Golden “feature contracts” tests.

I’ll go deep on IO + function features and on semantic_roles / test behavioral tags as exemplars; the rest follow the same pattern.

---

## 1. New `analytics/ast_features` package

### 1.1 Package layout

Create:

```text
analytics/ast_features/
    __init__.py
    model.py
    patterns.py
    extract.py
```

#### 1.1.1 `analytics/ast_features/model.py`

This is where the core “feature vector” lives. We’ll also move `IoFlags` here so both functions and tests can share IO semantics.

```python
# analytics/ast_features/model.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, FrozenSet

from codeintel.analytics.function_ast_cache import FunctionAst


@dataclass(frozen=True)
class IoFlags:
    """Flags describing IO usage within a span or function."""

    uses_network: bool = False
    uses_db: bool = False
    uses_filesystem: bool = False
    uses_subprocess: bool = False

    @property
    def io_bound(self) -> bool:
        """Return True when any IO flag is set."""
        return self.uses_network or self.uses_db or self.uses_filesystem or self.uses_subprocess


@dataclass(frozen=True)
class FunctionAstFeatures:
    """
    Semantic feature vector derived from a single function AST.

    These are *inputs* to higher-level classifiers (roles, entrypoints, etc.),
    not classifications themselves.
    """

    # Identity / context
    goid: int
    rel_path: str
    qualname: str
    is_async: bool

    # Source-level info
    decorators: tuple[str, ...]               # normalized decorator strings
    imports: Mapping[str, str]                # alias -> module path (e.g. "np" -> "numpy")
    libraries_used: FrozenSet[str]            # root libraries used in calls ("requests", "sqlalchemy")

    # IO + concurrency
    io_flags: IoFlags                         # shared IO flags
    uses_concurrency_lib: bool
    uses_threading: bool
    uses_asyncio_lib: bool

    # Framework / domain hints
    http_client_libs: FrozenSet[str]
    http_server_libs: FrozenSet[str]
    db_libs: FrozenSet[str]
    message_libs: FrozenSet[str]

    # Config / feature flag usage
    config_read_count: int
    feature_flag_count: int

    # Escape hatch for future additions
    extra: Mapping[str, object] = field(default_factory=dict)


__all__ = [
    "IoFlags",
    "FunctionAstFeatures",
]
```

> **Important:** we’ll remove the old `IoFlags` definition from `analytics/tests_profiles/types.py` and import this one there (see §2.2).

---

#### 1.1.2 `analytics/ast_features/patterns.py`

This centralizes the pattern registries you currently have spread across modules (the IO spec + concurrency libs from `tests_profiles/behavioral_tags`, plus some generic HTTP/DB/message libs).

```python
# analytics/ast_features/patterns.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


# IO spec, moved from tests_profiles.behavioral_tags.DEFAULT_IO_SPEC
DEFAULT_IO_SPEC: dict[str, dict[str, list[str]]] = {
    "network": {
        "libs": ["requests", "httpx", "urllib3", "aiohttp", "socket", "boto3", "paramiko"],
        "funcs": ["get", "post", "put", "delete", "request", "send"],
    },
    "db": {
        "libs": ["sqlalchemy", "psycopg2", "asyncpg", "pymysql", "pymongo", "redis"],
        "funcs": ["execute", "session", "commit", "query"],
    },
    "filesystem": {
        "libs": ["pathlib", "os", "shutil"],
        "funcs": ["open", "remove", "unlink", "copy", "move", "rmdir", "mkdir"],
    },
    "subprocess": {
        "libs": ["subprocess"],
        "funcs": ["run", "Popen", "call", "check_call", "check_output"],
    },
}

CONCURRENCY_LIBS: set[str] = {
    "asyncio",
    "anyio",
    "trio",
    "threading",
    "concurrent",
    "multiprocessing",
}

HTTP_CLIENT_LIBS: set[str] = {"requests", "httpx", "aiohttp"}
HTTP_SERVER_LIBS: set[str] = {"fastapi", "flask", "django", "starlette"}
DB_LIBS: set[str] = {"sqlalchemy", "psycopg2", "asyncpg", "pymysql", "pymongo", "redis"}
MESSAGE_LIBS: set[str] = {"kafka", "pika", "celery", "kombu"}


@dataclass(frozen=True)
class AstFeaturePatterns:
    """
    Bundle of patterns used to derive FunctionAstFeatures.

    Allows easier testing and later customization (e.g., project-specific patterns).
    """

    io_spec: Mapping[str, dict[str, list[str]]] = DEFAULT_IO_SPEC
    concurrency_libs: set[str] = CONCURRENCY_LIBS
    http_client_libs: set[str] = HTTP_CLIENT_LIBS
    http_server_libs: set[str] = HTTP_SERVER_LIBS
    db_libs: set[str] = DB_LIBS
    message_libs: set[str] = MESSAGE_LIBS


DEFAULT_PATTERNS = AstFeaturePatterns()
```

> Later, `data_models`, `entrypoint_detectors`, `dependencies`, etc. can **extend** these patterns (e.g. add project-specific libs) without editing each module separately.

---

#### 1.1.3 `analytics/ast_features/extract.py`

This is the workhorse: build import maps, walk the function AST, and produce `FunctionAstFeatures`.

```python
# analytics/ast_features/extract.py

from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Any

from codeintel.analytics.ast_features.model import FunctionAstFeatures, IoFlags
from codeintel.analytics.ast_features.patterns import AstFeaturePatterns, DEFAULT_PATTERNS
from codeintel.analytics.ast_utils import resolve_call_target, safe_unparse
from codeintel.analytics.function_ast_cache import FunctionAst
from codeintel.ingestion.ast_utils import parse_python_module
```

Build an import map once per module (we’ll reuse the logic from `tests_profiles/_build_import_map`):

```python
def build_import_map(tree: ast.AST) -> dict[str, str]:
    """
    Build alias -> module mapping from import statements.

    This is a shared primitive for both function and test AST analyses.
    """
    mapping: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    mapping[alias.asname] = alias.name
                else:
                    root = alias.name.split(".", maxsplit=1)[0]
                    mapping[root] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            module = node.module
            for alias in node.names:
                if alias.asname:
                    mapping[alias.asname] = module
                else:
                    root = alias.name.split(".", maxsplit=1)[0]
                    mapping[root] = f"{module}.{alias.name}"
    return mapping
```

A helper for IO flags, derived from your existing `_update_io_flags` in `behavioral_tags`:

```python
def io_flags_from_call(
    node: ast.Call,
    import_map: Mapping[str, str],
    existing: IoFlags,
    *,
    patterns: AstFeaturePatterns,
) -> IoFlags:
    """Update IoFlags based on a single Call node."""

    func = node.func
    root_name: str | None = None
    attr: str | None = None
    if isinstance(func, ast.Name):
        root_name = func.id
        attr = func.id
    elif isinstance(func, ast.Attribute):
        attr = func.attr
        value = func.value
        while isinstance(value, ast.Attribute):
            value = value.value
        if isinstance(value, ast.Name):
            root_name = value.id

    if root_name is None:
        return existing

    module = import_map.get(root_name, root_name)
    module_root = module.split(".", maxsplit=1)[0]
    attr_lower = attr.lower() if attr is not None else None

    uses_network = existing.uses_network
    uses_db = existing.uses_db
    uses_filesystem = existing.uses_filesystem
    uses_subprocess = existing.uses_subprocess

    network_spec = patterns.io_spec["network"]
    db_spec = patterns.io_spec["db"]
    filesystem_spec = patterns.io_spec["filesystem"]
    subprocess_spec = patterns.io_spec["subprocess"]

    if module_root in network_spec["libs"] or (
        attr_lower is not None and attr_lower in network_spec["funcs"]
    ):
        uses_network = True
    if module_root in db_spec["libs"] or (
        attr_lower is not None and attr_lower in db_spec["funcs"]
    ):
        uses_db = True
    if module_root in filesystem_spec["libs"] or (
        attr_lower is not None and attr_lower in filesystem_spec["funcs"]
    ):
        uses_filesystem = True
    if module_root in subprocess_spec["libs"] or (
        attr_lower is not None and attr_lower in subprocess_spec["funcs"]
    ):
        uses_subprocess = True

    return IoFlags(
        uses_network=uses_network,
        uses_db=uses_db,
        uses_filesystem=uses_filesystem,
        uses_subprocess=uses_subprocess,
    )
```

A visitor that aggregates features:

```python
@dataclass
class _FunctionFeatureState:
    decorators: list[str]
    libraries_used: set[str]
    io_flags: IoFlags
    uses_concurrency_lib: bool
    uses_threading: bool
    uses_asyncio_lib: bool
    http_client_libs: set[str]
    http_server_libs: set[str]
    db_libs: set[str]
    message_libs: set[str]
    config_read_count: int
    feature_flag_count: int


class FunctionFeatureVisitor(ast.NodeVisitor):
    """
    Walk a function node and compute semantic features.

    Uses import_map + patterns; should be run inside a module-level context.
    """

    def __init__(
        self,
        import_map: Mapping[str, str],
        patterns: AstFeaturePatterns,
    ) -> None:
        self.import_map = import_map
        self.patterns = patterns
        self.state = _FunctionFeatureState(
            decorators=[],
            libraries_used=set(),
            io_flags=IoFlags(),
            uses_concurrency_lib=False,
            uses_threading=False,
            uses_asyncio_lib=False,
            http_client_libs=set(),
            http_server_libs=set(),
            db_libs=set(),
            message_libs=set(),
            config_read_count=0,
            feature_flag_count=0,
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # Record decorators once, normalized
        for decorator in node.decorator_list:
            dec_str = safe_unparse(decorator)
            if dec_str:
                self.state.decorators.append(dec_str)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        # Decs for async functions too
        for decorator in node.decorator_list:
            dec_str = safe_unparse(decorator)
            if dec_str:
                self.state.decorators.append(dec_str)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        target = resolve_call_target(node.func, self.import_map)
        if target.library:
            self.state.libraries_used.add(target.library)

        # IO flags
        self.state.io_flags = io_flags_from_call(
            node,
            self.import_map,
            self.state.io_flags,
            patterns=self.patterns,
        )

        # Concurrency libs
        lib_root = target.library.split(".", maxsplit=1)[0] if target.library else None
        if lib_root in self.patterns.concurrency_libs:
            self.state.uses_concurrency_lib = True
        if lib_root == "threading":
            self.state.uses_threading = True
        if lib_root == "asyncio":
            self.state.uses_asyncio_lib = True

        # HTTP / DB / messaging classification
        if lib_root in self.patterns.http_client_libs:
            self.state.http_client_libs.add(lib_root)
        if lib_root in self.patterns.http_server_libs:
            self.state.http_server_libs.add(lib_root)
        if lib_root in self.patterns.db_libs:
            self.state.db_libs.add(lib_root)
        if lib_root in self.patterns.message_libs:
            self.state.message_libs.add(lib_root)

        # Config / feature flags: cheap heuristic via dotted name segments
        dotted = target.dotted or ""
        if "feature_flag" in dotted or ".flag(" in dotted:
            self.state.feature_flag_count += 1
        if ".config" in dotted or ".settings" in dotted:
            self.state.config_read_count += 1

        self.generic_visit(node)
```

Finally, the main entrypoint:

```python
def compute_function_features(
    fn: FunctionAst,
    *,
    repo_root: Path | None = None,
    patterns: AstFeaturePatterns = DEFAULT_PATTERNS,
) -> FunctionAstFeatures:
    """
    Compute FunctionAstFeatures from a FunctionAst instance.

    Parameters
    ----------
    fn:
        FunctionAst produced by function_ast_cache.load_function_asts.
    repo_root:
        Optional repo root; when provided, we parse the full file from disk
        to build a module import map. Otherwise we parse from fn.lines.
    patterns:
        Pattern bundle controlling IO + framework classification.
    """
    if repo_root is not None:
        module_path = (repo_root / fn.rel_path).resolve()
        parsed = parse_python_module(module_path)
        if parsed is None:
            module_tree = ast.parse("".join(fn.lines), filename=str(module_path))
        else:
            module_tree = parsed.tree
    else:
        module_tree = ast.parse("".join(fn.lines), filename=fn.rel_path)

    import_map = build_import_map(module_tree)

    visitor = FunctionFeatureVisitor(import_map=import_map, patterns=patterns)
    visitor.visit(fn.node)

    state = visitor.state
    decorators = tuple(state.decorators)

    return FunctionAstFeatures(
        goid=fn.goid,
        rel_path=fn.rel_path,
        qualname=fn.qualname,
        is_async=isinstance(fn.node, ast.AsyncFunctionDef),
        decorators=decorators,
        imports=import_map,
        libraries_used=frozenset(state.libraries_used),
        io_flags=state.io_flags,
        uses_concurrency_lib=state.uses_concurrency_lib,
        uses_threading=state.uses_threading,
        uses_asyncio_lib=state.uses_asyncio_lib,
        http_client_libs=frozenset(state.http_client_libs),
        http_server_libs=frozenset(state.http_server_libs),
        db_libs=frozenset(state.db_libs),
        message_libs=frozenset(state.message_libs),
        config_read_count=state.config_read_count,
        feature_flag_count=state.feature_flag_count,
        extra={},
    )
```

`__init__.py` can just re-export:

```python
# analytics/ast_features/__init__.py

from .model import IoFlags, FunctionAstFeatures
from .patterns import DEFAULT_PATTERNS, AstFeaturePatterns
from .extract import compute_function_features, build_import_map

__all__ = [
    "IoFlags",
    "FunctionAstFeatures",
    "AstFeaturePatterns",
    "DEFAULT_PATTERNS",
    "compute_function_features",
    "build_import_map",
]
```

---

## 2. Refactor tests IO semantics to use ast_features

### 2.1 Move IoFlags into `ast_features.model` and import it

In `analytics/tests_profiles/types.py`, remove the existing IoFlags class definition and instead import it:

**Before:**

```python
from dataclasses import dataclass

...

@dataclass(frozen=True)
class IoFlags:
    """Flags describing IO usage within a test."""

    uses_network: bool = False
    uses_db: bool = False
    uses_filesystem: bool = False
    uses_subprocess: bool = False

    @property
    def io_bound(self) -> bool:
        ...
```

**After:**

```python
from dataclasses import dataclass
from codeintel.analytics.ast_features.model import IoFlags

...

# no IoFlags definition here; reuse the shared one
```

Keep `IoFlags` in the `__all__` list, since it’s still part of the test types API.

---

### 2.2 Move IO patterns & helpers out of `behavioral_tags`

In `analytics/tests_profiles/behavioral_tags.py`, remove:

* `DEFAULT_IO_SPEC`
* `CONCURRENCY_LIBS`
* `_update_io_flags`
* `_call_root_and_attr`
* `_uses_concurrency`

and import them from `ast_features`, as appropriate.

**Before:**

```python
from codeintel.ingestion.ast_utils import parse_python_module
...

DEFAULT_IO_SPEC: dict[str, dict[str, list[str]]] = {...}
CONCURRENCY_LIBS: set[str] = {...}

...

def _update_io_flags(node: ast.Call, config: SpanConfig, existing: IoFlags) -> IoFlags:
    ...

def _uses_concurrency(node: ast.Call, config: SpanConfig) -> bool:
    ...
```

**After:**

```python
from codeintel.analytics.ast_features.patterns import DEFAULT_IO_SPEC, CONCURRENCY_LIBS
from codeintel.analytics.ast_features.model import IoFlags
from codeintel.analytics.ast_features.extract import io_flags_from_call
...

# SpanConfig stays, but no DEFAULT_IO_SPEC / CONCURRENCY_LIBS here
```

Update `_update_span_state` to call the shared `io_flags_from_call`:

```python
def _update_span_state(
    node: ast.AST,
    config: SpanConfig,
    state: SpanState,
) -> None:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        state.uses_fixtures = state.uses_fixtures or _uses_fixtures(node)
    if isinstance(node, ast.Assert):
        state.assert_count += 1
        state.has_boundary_asserts = state.has_boundary_asserts or _is_boundary_assert(node)
    if isinstance(node, ast.Raise):
        state.raise_count += 1
    if isinstance(node, (ast.With, ast.AsyncWith)) and _with_uses_pytest_raises(node):
        state.uses_pytest_raises = True
    if isinstance(node, ast.Call):
        if _is_pytest_raises(node.func):
            state.uses_pytest_raises = True

        # IO flags via ast_features
        state.io_flags = io_flags_from_call(
            node,
            config.import_map,
            state.io_flags,
            patterns=DEFAULT_PATTERNS,  # or a test-specific patterns bundle
        )

        state.uses_concurrency = state.uses_concurrency or _uses_concurrency(node, config)
```

Now tests and general functions both share the same IO inference logic via `io_flags_from_call`, and the patterns live in `ast_features.patterns`.

---

## 3. Integrate function features into AnalyticsContext & semantic_roles

### 3.1 Attach features to AnalyticsContext (optional but very nice)

Extend `AnalyticsContext` in `analytics/context.py`:

**Before:**

```python
@dataclass(frozen=True)
class AnalyticsContext:
    repo: str
    commit: str
    repo_root: Path
    catalog: FunctionCatalogProvider
    module_map: dict[str, str]
    function_ast_map: dict[int, FunctionAst]
    missing_function_goids: set[int]
    call_graph: nx.DiGraph
    import_graph: nx.DiGraph | None
    symbol_module_graph: nx.Graph | None
    symbol_function_graph: nx.Graph | None
    created_at: datetime
    snapshot_id: str
```

**After:**

```python
from codeintel.analytics.ast_features.model import FunctionAstFeatures

@dataclass(frozen=True)
class AnalyticsContext:
    ...
    function_ast_map: dict[int, FunctionAst]
    missing_function_goids: set[int]
    function_features_map: dict[int, FunctionAstFeatures]  # NEW
    ...
```

Then, in `ensure_analytics_context` (or wherever you build it), compute features once:

```python
from codeintel.analytics.ast_features.extract import compute_function_features

def ensure_analytics_context(...):
    ...
    function_ast_map, missing = load_function_asts(gateway, request)
    features_map: dict[int, FunctionAstFeatures] = {}
    for goid, fn_ast in function_ast_map.items():
        features_map[goid] = compute_function_features(
            fn_ast,
            repo_root=cfg.repo_root,
        )

    return AnalyticsContext(
        repo=cfg.repo,
        commit=cfg.commit,
        repo_root=cfg.repo_root,
        catalog=provider,
        module_map=module_map,
        function_ast_map=function_ast_map,
        missing_function_goids=missing,
        function_features_map=features_map,
        call_graph=call_graph,
        import_graph=import_graph,
        symbol_module_graph=symbol_module_graph,
        symbol_function_graph=symbol_function_graph,
        created_at=now,
        snapshot_id=snapshot_id,
    )
```

Now **every analytics module** that gets an `AnalyticsContext` can see shared function features.

---

### 3.2 Use features from semantic_roles

Right now `semantic_roles.py` builds its own `FunctionContext` with decorators, rel_path, qualname, etc., and scoring functions like `_score_api_handlers` and `_score_cli_commands` inspect those.

We’ll:

* Add `features: FunctionAstFeatures | None` to `FunctionContext`.
* Populate it from `AnalyticsContext.function_features_map`.
* Rewrite a couple of scoring functions to use features as primary input.

**Extend FunctionContext:**

```python
# analytics/semantic_roles.py

from codeintel.analytics.ast_features.model import FunctionAstFeatures

@dataclass(frozen=True)
class FunctionContext:
    goid: int
    rel_path: str
    module: str | None
    name: str
    loc: int | None
    tags: list[str]
    decorators: list[str]
    rel_path_lower: str
    module_name: str | None
    roles: RoleAccumulator
    features: FunctionAstFeatures | None = None  # NEW
```

When building `FunctionContext` (where you currently construct it from AST / meta), look up the features:

```python
def _build_function_contexts(
    module_meta: dict[str, ModuleMeta],
    ast_map: dict[int, FunctionAst],
    features_map: dict[int, FunctionAstFeatures],
) -> dict[int, FunctionContext]:
    contexts: dict[int, FunctionContext] = {}
    for goid, fn_ast in ast_map.items():
        meta = module_meta.get(fn_ast.rel_path)
        features = features_map.get(goid)
        # existing tags/decorators logic
        decorators = [safe_unparse(dec) for dec in fn_ast.node.decorator_list]
        rel_path_lower = fn_ast.rel_path.lower()
        # ...
        contexts[goid] = FunctionContext(
            goid=goid,
            rel_path=fn_ast.rel_path,
            module=meta.module_name if meta is not None else None,
            name=fn_ast.qualname.split(".")[-1],
            loc=meta.loc if meta is not None else None,
            tags=meta.tags if meta is not None else [],
            decorators=decorators,
            rel_path_lower=rel_path_lower,
            module_name=meta.module_name if meta is not None else None,
            roles=RoleAccumulator(),
            features=features,
        )
    return contexts
```

Now we can rewrite scoring functions.

**Example: `_score_api_handlers` and `_score_cli_commands`**

**Before:**

```python
def _score_api_handlers(context: FunctionContext, accumulator: RoleAccumulator) -> None:
    for dec in context.decorators:
        dec_lower = dec.lower()
        if "fastapi" in dec_lower or "app.get" in dec_lower:
            accumulator.bump("api_handler", 0.7, f"decorator:{dec}", framework_hint="fastapi")
        if "flask" in dec_lower or "bp.route" in dec_lower:
            accumulator.bump("api_handler", 0.6, f"decorator:{dec}", framework_hint="flask")
```

**After (use features):**

```python
def _score_api_handlers(context: FunctionContext, accumulator: RoleAccumulator) -> None:
    features = context.features
    if features is None:
        # fallback: old decorator heuristics
        for dec in context.decorators:
            dec_lower = dec.lower()
            if "fastapi" in dec_lower or "app.get" in dec_lower:
                accumulator.bump("api_handler", 0.7, f"decorator:{dec}", framework_hint="fastapi")
            if "flask" in dec_lower or "bp.route" in dec_lower:
                accumulator.bump("api_handler", 0.6, f"decorator:{dec}", framework_hint="flask")
        return

    if features.http_server_libs:
        # any HTTP server library usage is a strong hint
        libs = ",".join(sorted(features.http_server_libs))
        accumulator.bump("api_handler", 0.7, f"http_server_libs:{libs}")

    # decorators are still useful for nuance
    for dec in features.decorators:
        dec_lower = dec.lower()
        if "fastapi" in dec_lower:
            accumulator.bump("api_handler", 0.2, f"decorator:{dec}", framework_hint="fastapi")
        if "flask" in dec_lower:
            accumulator.bump("api_handler", 0.2, f"decorator:{dec}", framework_hint="flask")
```

**CLI:**

```python
def _score_cli_commands(context: FunctionContext, accumulator: RoleAccumulator) -> None:
    features = context.features

    # Decorator-based hints
    for dec in context.decorators:
        dec_lower = dec.lower()
        if "click." in dec_lower:
            accumulator.bump("cli_command", 0.8, f"decorator:{dec}", framework_hint="click")
        if "typer." in dec_lower:
            accumulator.bump("cli_command", 0.8, f"decorator:{dec}", framework_hint="typer")

    if "cli" in context.rel_path_lower or "commands" in context.rel_path_lower:
        accumulator.bump("cli_command", 0.4, "path:cli")
    if context.name in {"main", "cli"}:
        accumulator.bump("cli_command", 0.3, "name:entrypoint")

    # Library-based hints via features
    if features is not None and "click" in features.libraries_used:
        accumulator.bump("cli_command", 0.5, "library:click")
    if features is not None and "typer" in features.libraries_used:
        accumulator.bump("cli_command", 0.5, "library:typer")
```

This shows the pattern: **features first, AST-decorators as nuance/fallback**.

---

## 4. Using features in data_models, entrypoint_detectors, dependencies

You don’t need to rewrite these modules wholesale right away; the key is to give them access to `FunctionAstFeatures` and start moving repeated logic into feature-based checks.

### 4.1 Expose features from `function_ast_cache` for convenience

Add a helper in `analytics/ast_features/extract.py`:

```python
from codeintel.storage.gateway import StorageGateway
from codeintel.analytics.context import AnalyticsContextConfig, ensure_analytics_context

def load_function_features_for_repo(
    gateway: StorageGateway,
    cfg: AnalyticsContextConfig,
) -> dict[int, FunctionAstFeatures]:
    """
    Convenience helper to build a feature map without manually calling ensure_analytics_context.

    Most callers (semantic_roles, dependencies, etc.) already call ensure_analytics_context,
    so this is mainly for modules that don’t yet.
    """
    ctx = ensure_analytics_context(
        gateway,
        cfg=cfg,
        context=None,
        runtime=None,
    )
    return ctx.function_features_map
```

Now, any module that already has an `AnalyticsContext` can just use `context.function_features_map`.

### 4.2 Example: entrypoint_detectors

Today `entrypoint_detectors.py` builds a lot of AST-handling of decorators & HTTP routes. For the *initial* Epic 4, I’d do a light touch:

* Use `FunctionAstFeatures` to supply import_map and HTTP-library presence.
* Keep existing AST-specific logic for route paths.

Pseudo-patch:

```python
# analytics/entrypoint_detectors.py

from codeintel.analytics.ast_features.model import FunctionAstFeatures

@dataclass(frozen=True)
class EntryPointCandidate:
    ...
    features: FunctionAstFeatures | None = None
```

When building candidates for a function, attach features from `AnalyticsContext.function_features_map[goid]` if present. Then in places where you check for HTTP libs or concurrency libs, you can:

```python
if features is not None and features.http_server_libs:
    # mark as potential HTTP entrypoint, weight up existing signals
```

You don’t have to rip out all existing AST code; just **enrich** with features.

### 4.3 Example: dependencies

Similarly, `dependencies.build_external_dependency_calls` currently:

* Loads patterns,
* Builds an alias map from imports,
* Walks each function AST with `DependencyVisitor`.

Use `FunctionAstFeatures` to:

* Pre-classify which functions are likely to have network/db calls (via `io_flags`, `libraries_used`).
* Skip analysis for obviously irrelevant functions.

For example:

```python
features = shared_context.function_features_map.get(goid)
if features is not None and not (features.io_flags.uses_network or features.db_libs):
    # skip this function as it’s unlikely to hit external deps
    continue
```

And you can later move common library classification patterns into `ast_features.patterns` so both `DependencyVisitor` and `FunctionFeatureVisitor` share them.

---

## 5. Feature-contract tests

### 5.1 Golden IO semantics

**New test file:** `tests/analytics/test_ast_features_io.py`

```python
from __future__ import annotations

import ast
from pathlib import Path

from codeintel.analytics.ast_features.extract import compute_function_features
from codeintel.analytics.function_ast_cache import FunctionAst


def _fn_from_source(src: str) -> FunctionAst:
    tree = ast.parse(src)
    fn_node = next(
        node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    return FunctionAst(
        goid=1,
        rel_path="mod.py",
        qualname="mod.fn",
        start_line=1,
        end_line=10,
        node=fn_node,
        lines=src.splitlines(keepends=True),
    )


def test_network_io_flag() -> None:
    src = "import requests\n\ndef fn():\n    requests.get('https://example.com')\n"
    fn = _fn_from_source(src)
    features = compute_function_features(fn)
    assert features.io_flags.uses_network
    assert not features.io_flags.uses_db
    assert "requests" in features.libraries_used
    assert "requests" in features.http_client_libs


def test_db_io_flag() -> None:
    src = "import sqlalchemy\n\ndef fn():\n    sqlalchemy.session().execute('SELECT 1')\n"
    fn = _fn_from_source(src)
    features = compute_function_features(fn)
    assert features.io_flags.uses_db
    assert "sqlalchemy" in features.db_libs
```

### 5.2 Integration with tests_profiles

**New test:** `tests/analytics/test_behavioral_tags_uses_ast_features.py`

* Assert that `_update_span_state` uses `io_flags_from_call` semantics by running `build_test_ast_index` on a snippet and checking `TestAstInfo.io_flags`.

### 5.3 Integration with semantic_roles

**New test:** `tests/analytics/test_semantic_roles_features.py`

* Create a tiny in-memory repo (or use a fixture) with two functions:

  * One FastAPI route.
  * One CLI function using click.

* Use `ensure_analytics_context` to build `AnalyticsContext` + features.

* Run `build_semantic_roles` and assert:

  * API handler gets higher `api_handler` score if `features.http_server_libs` contains `"fastapi"`.
  * CLI function gets `cli_command` score when `features.libraries_used` contains `"click"`.

---

## Phased rollout

To keep this manageable and low-risk, I’d roll it in phases:

1. **Phase 1 (IO foundation):**

   * Introduce `analytics/ast_features` (model, patterns, extract).
   * Move `IoFlags`, `DEFAULT_IO_SPEC`, `CONCURRENCY_LIBS` and IO-flag logic out of `tests_profiles/behavioral_tags` into `ast_features`.
   * Add IO unit tests.

2. **Phase 2 (function features + context):**

   * Compute `function_features_map` in `AnalyticsContext`.
   * Add features to `FunctionContext` in `semantic_roles`.
   * Refactor a couple of scoring functions to prefer features.

3. **Phase 3 (domain integration):**

   * Use `FunctionAstFeatures` as a filter / hint in `dependencies`, `entrypoint_detectors`, and `data_models`.
   * Gradually move repeated heuristics (e.g. IO, concurrency, HTTP/DB libs) out of those modules into `ast_features`.

4. **Phase 4 (optional persistence):**

   * If useful, introduce an `analytics.function_ast_features` dataset (row model + contract) so agents can query “why is this function classified that way?” from DuckDB directly.

If you want, next we can design that **`analytics.function_ast_features` dataset**: row model, dataset contract, and a plugin that persists features for all functions (i.e., an Epic 4.1 step to make AST features queryable for agents).

# analytics.function_ast_features dataset: row model, dataset contract, and a plugin that persists features for all functions #

Let’s make **`analytics.function_ast_features`** a real, first-class dataset: fully typed, contract-backed, and filled by a plugin that walks all functions and persists the features we defined in Epic 4.

I’ll break it into:

1. Table schema (config).
2. Row model + serializer (storage).
3. Dataset binding + contract (storage + analytics).
4. Mapping `FunctionAstFeatures → row` (analytics).
5. Plugin to compute & persist features.
6. Pipeline integration.
7. Tests & invariants.

I’ll assume the **Epic 4 core** is (or will be) in place:

* `analytics.ast_features.model.FunctionAstFeatures`
* `analytics.ast_features.extract.compute_function_features`
* `analytics.function_ast_cache.FunctionAst` / `load_function_asts`
* Generic plugin harness (`analytics.plugins`, `analytics.plugin_runtime`) per Epic 2.

If something isn’t in yet, you can treat the snippets as the target state.

---

## 1. Add `analytics.function_ast_features` to the schema

**File:** `config/schemas/tables.py`

Add a new `TableSchema` entry to the `TABLE_SCHEMAS` dict:

```python
"analytics.function_ast_features": TableSchema(
    schema="analytics",
    name="function_ast_features",
    columns=[
        Column("repo", "VARCHAR", nullable=False),
        Column("commit", "VARCHAR", nullable=False),
        Column("function_goid_h128", "DECIMAL(38,0)", nullable=False),

        Column("rel_path", "VARCHAR", nullable=False),
        Column("qualname", "VARCHAR", nullable=False),
        Column("is_async", "BOOLEAN", nullable=False),

        # IO + concurrency
        Column("uses_network", "BOOLEAN", nullable=False),
        Column("uses_db", "BOOLEAN", nullable=False),
        Column("uses_filesystem", "BOOLEAN", nullable=False),
        Column("uses_subprocess", "BOOLEAN", nullable=False),
        Column("uses_concurrency_lib", "BOOLEAN", nullable=False),
        Column("uses_threading", "BOOLEAN", nullable=False),
        Column("uses_asyncio_lib", "BOOLEAN", nullable=False),

        # Libraries / frameworks
        Column("http_client_libs", "JSON", nullable=False),
        Column("http_server_libs", "JSON", nullable=False),
        Column("db_libs", "JSON", nullable=False),
        Column("message_libs", "JSON", nullable=False),

        # Config / feature flags
        Column("config_read_count", "INTEGER", nullable=False),
        Column("feature_flag_count", "INTEGER", nullable=False),

        # “Explainability” payloads
        Column("decorators", "JSON", nullable=False),
        Column("libraries_used", "JSON", nullable=False),

        Column("created_at", "TIMESTAMP", nullable=False),
    ],
    primary_key=("repo", "commit", "function_goid_h128"),
    description="Per-function AST-derived semantic features for explainability and classification.",
),
```

You can add indexes later if you want (e.g. on `(repo, commit)`, `rel_path`, `qualname`), but the primary key is the important part for idempotent updates.

---

## 2. Row model & serializer in `storage/rows.py`

**File:** `storage/rows.py`

Add a `TypedDict` for the row shape and a column list + serializer.

At the top (with other exports):

```python
__all__ = [
    # ...
    "FunctionAstFeaturesRow",
    "function_ast_features_row_to_tuple",
    # ...
]
```

Then define the row model:

```python
from typing import TypedDict

# ...

class FunctionAstFeaturesRow(TypedDict):
    """
    Row model for analytics.function_ast_features.

    Mirrors TABLE_SCHEMAS['analytics.function_ast_features'].
    """

    repo: str
    commit: str
    function_goid_h128: int

    rel_path: str
    qualname: str
    is_async: bool

    uses_network: bool
    uses_db: bool
    uses_filesystem: bool
    uses_subprocess: bool
    uses_concurrency_lib: bool
    uses_threading: bool
    uses_asyncio_lib: bool

    http_client_libs: list[str]
    http_server_libs: list[str]
    db_libs: list[str]
    message_libs: list[str]

    config_read_count: int
    feature_flag_count: int

    decorators: list[str]
    libraries_used: list[str]

    created_at: datetime
```

Column list in order, consistent with the schema:

```python
_FUNCTION_AST_FEATURES_COLUMNS: list[str] = [
    "repo",
    "commit",
    "function_goid_h128",
    "rel_path",
    "qualname",
    "is_async",
    "uses_network",
    "uses_db",
    "uses_filesystem",
    "uses_subprocess",
    "uses_concurrency_lib",
    "uses_threading",
    "uses_asyncio_lib",
    "http_client_libs",
    "http_server_libs",
    "db_libs",
    "message_libs",
    "config_read_count",
    "feature_flag_count",
    "decorators",
    "libraries_used",
    "created_at",
]
```

Serializer using the existing `_serialize_row` helper:

```python
def function_ast_features_row_to_tuple(
    row: FunctionAstFeaturesRow,
) -> tuple[object, ...]:
    """
    Serialize FunctionAstFeaturesRow into INSERT column order.
    """
    return _serialize_row(row, _FUNCTION_AST_FEATURES_COLUMNS)
```

This makes the row model available to the dataset registry and any generic dataset machinery.

---

## 3. Bind the dataset in `storage/datasets.py`

**File:** `storage/datasets.py`

### 3.1 Row binding

Find `ROW_BINDINGS_BY_TABLE_KEY` and add:

```python
from codeintel.storage import rows as row_models

ROW_BINDINGS_BY_TABLE_KEY: dict[str, RowBinding] = {
    # ...
    "analytics.function_ast_features": _row_binding(
        row_type=row_models.FunctionAstFeaturesRow,
        to_tuple=row_models.function_ast_features_row_to_tuple,
    ),
    # ...
}
```

### 3.2 JSONL & Parquet filenames (if not already automatic)

In `JSONL_FILENAMES_BY_TABLE_KEY`:

```python
JSONL_FILENAMES_BY_TABLE_KEY = {
    # ...
    "analytics.function_ast_features": "function_ast_features.jsonl",
    # ...
}
```

In `PARQUET_FILENAMES_BY_TABLE_KEY`:

```python
PARQUET_FILENAMES_BY_TABLE_KEY = {
    # ...
    "analytics.function_ast_features": "function_ast_features.parquet",
    # ...
}
```

Now the dataset registry knows how to persist this table to JSONL/Parquet and how to serialize rows.

---

## 4. Analytics-level row facade & dataset contract

### 4.1 Optionally re-export row model in `analytics/rows`

**File:** `analytics/rows/function_ast_features.py` (new, if you like)

```python
# analytics/rows/function_ast_features.py

from __future__ import annotations

from codeintel.storage.rows import (
    FunctionAstFeaturesRow,
    function_ast_features_row_to_tuple,
)

__all__ = ["FunctionAstFeaturesRow", "function_ast_features_row_to_tuple"]
```

And in `analytics/rows/__init__.py`:

```python
from .function_ast_features import FunctionAstFeaturesRow

__all__ = [
    # ...
    "FunctionAstFeaturesRow",
]
```

This is optional but keeps a nice “analytics-facing” import path.

---

### 4.2 Dataset contract helper in `analytics/datasets.py`

**File:** `analytics/datasets.py`

If you’ve already introduced `AnalyticsDatasetContract` for Epic 3, reuse it. Otherwise, here’s a minimal version:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Callable, Mapping, Any

from duckdb import DuckDBPyConnection

from codeintel.config.schemas.tables import TABLE_SCHEMAS, TableSchema
from codeintel.storage.datasets import load_dataset_registry, Dataset
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.rows import (
    FunctionAstFeaturesRow,
    function_ast_features_row_to_tuple,
)


RowMapping = Mapping[str, object]
ToTuple = Callable[[RowMapping], tuple[object, ...]]


@dataclass(frozen=True)
class AnalyticsDatasetContract:
    name: str
    table_key: str
    schema: TableSchema | None
    row_type: type[RowMapping]
    to_tuple: ToTuple
    dataset_meta: Dataset | None = None
```

Helper to build the specific contract:

```python
def get_function_ast_features_contract(
    gateway: StorageGateway,
) -> AnalyticsDatasetContract:
    con: DuckDBPyConnection = gateway.con
    registry = load_dataset_registry(con)

    name = "analytics.function_ast_features"
    dataset = registry.by_name.get(name)
    table_key = dataset.table_key if dataset is not None else name
    schema = TABLE_SCHEMAS.get(table_key)

    return AnalyticsDatasetContract(
        name=name,
        table_key=table_key,
        schema=schema,
        row_type=FunctionAstFeaturesRow,  # type: ignore[arg-type]
        to_tuple=function_ast_features_row_to_tuple,
        dataset_meta=dataset,
    )
```

And reuse your generic insertion helper (from Epic 3) or define a minimal one:

```python
from codeintel.ingestion.common import run_batch

def insert_analytics_rows(
    gateway: StorageGateway,
    contract: AnalyticsDatasetContract,
    rows: list[RowMapping],
    *,
    delete_params: list[object] | None = None,
    scope: str | None = None,
) -> None:
    if not rows:
        return

    con = gateway.con
    if delete_params is not None:
        con.execute(
            f"DELETE FROM {contract.table_key} WHERE repo = ? AND commit = ?",
            delete_params,
        )

    tuple_rows = [contract.to_tuple(row) for row in rows]
    run_batch(
        gateway,
        contract.table_key,
        tuple_rows,
        delete_params=None,  # already deleted above
        scope=scope,
    )
```

---

## 5. Mapping `FunctionAstFeatures` → `FunctionAstFeaturesRow`

**File:** `analytics/ast_features/persist.py` (new)

This is the glue between **in-memory features** and the persisted row model.

```python
# analytics/ast_features/persist.py

from __future__ import annotations

from datetime import UTC, datetime

from codeintel.analytics.ast_features.model import FunctionAstFeatures
from codeintel.storage.rows import FunctionAstFeaturesRow


def features_to_row(
    *,
    repo: str,
    commit: str,
    features: FunctionAstFeatures,
    created_at: datetime | None = None,
) -> FunctionAstFeaturesRow:
    """
    Convert FunctionAstFeatures into a FunctionAstFeaturesRow.

    Parameters
    ----------
    repo, commit:
        Snapshot identifiers for the features.
    features:
        Feature vector for a single function.
    created_at:
        Optional timestamp; defaults to now in UTC.
    """
    ts = created_at or datetime.now(tz=UTC)

    return FunctionAstFeaturesRow(
        repo=repo,
        commit=commit,
        function_goid_h128=int(features.goid),
        rel_path=features.rel_path,
        qualname=features.qualname,
        is_async=features.is_async,
        uses_network=features.io_flags.uses_network,
        uses_db=features.io_flags.uses_db,
        uses_filesystem=features.io_flags.uses_filesystem,
        uses_subprocess=features.io_flags.uses_subprocess,
        uses_concurrency_lib=features.uses_concurrency_lib,
        uses_threading=features.uses_threading,
        uses_asyncio_lib=features.uses_asyncio_lib,
        http_client_libs=sorted(features.http_client_libs),
        http_server_libs=sorted(features.http_server_libs),
        db_libs=sorted(features.db_libs),
        message_libs=sorted(features.message_libs),
        config_read_count=features.config_read_count,
        feature_flag_count=features.feature_flag_count,
        decorators=list(features.decorators),
        libraries_used=sorted(features.libraries_used),
        created_at=ts,
    )
```

You can extend this over time (e.g., add counts of decorators, domains, etc.) without changing the core feature computation.

---

## 6. Plugin to compute & persist features

We’ll treat this as a **generic analytics plugin** with stage `"function"`.

### 6.1 Plugin implementation

**File:** `analytics/functions/plugins_ast_features.py` (new) or fold into your existing `analytics/functions/plugins.py` if you have one.

```python
# analytics/functions/plugins_ast_features.py

from __future__ import annotations

from datetime import UTC, datetime

from codeintel.analytics.ast_features.extract import compute_function_features
from codeintel.analytics.ast_features.persist import features_to_row
from codeintel.analytics.function_ast_cache import FunctionAstLoadRequest, load_function_asts
from codeintel.analytics.plugins import (
    AnalyticsExecutionContext,
    AnalyticsPlugin,
    ResourceHints,
    register_analytics_plugin,
)
from codeintel.analytics.datasets import (
    get_function_ast_features_contract,
    insert_analytics_rows,
)
from codeintel.config.steps_analytics import FunctionAnalyticsStepConfig
from codeintel.storage.gateway import StorageGateway


def _function_ast_features_run(ctx: AnalyticsExecutionContext) -> dict[str, int]:
    """
    Plugin body: compute function AST features and persist them.

    This plugin is stage="function" and expects ctx.function_cfg to be present.
    """
    if ctx.function_cfg is None:
        msg = "FunctionAnalyticsStepConfig is required in AnalyticsExecutionContext.function_cfg"
        raise ValueError(msg)
    if not isinstance(ctx.gateway, StorageGateway):
        msg = "AnalyticsExecutionContext.gateway must be a StorageGateway"
        raise TypeError(msg)

    cfg: FunctionAnalyticsStepConfig = ctx.function_cfg
    gateway = ctx.gateway

    request = FunctionAstLoadRequest(
        repo=cfg.repo,
        commit=cfg.commit,
        repo_root=cfg.repo_root,
        catalog_provider=None,
        allowed_goids=None,
        max_files=None,
    )
    ast_by_goid, missing_goids = load_function_asts(gateway, request)
    created_at = datetime.now(tz=UTC)

    rows = []
    for goid, fn_ast in ast_by_goid.items():
        features = compute_function_features(fn_ast, repo_root=cfg.repo_root)
        row = features_to_row(
            repo=cfg.repo,
            commit=cfg.commit,
            features=features,
            created_at=created_at,
        )
        rows.append(row)

    contract = get_function_ast_features_contract(gateway)
    insert_analytics_rows(
        gateway,
        contract,
        rows,
        delete_params=[cfg.repo, cfg.commit],
        scope=f"{cfg.repo}@{cfg.commit}",
    )

    return {
        "functions_seen": len(ast_by_goid),
        "functions_missing": len(missing_goids),
        "rows_written": len(rows),
    }


FUNCTION_AST_FEATURES_PLUGIN = AnalyticsPlugin(
    name="functions.ast_features",
    description="Compute and persist AST-derived semantic features for each function.",
    stage="function",
    enabled_by_default=True,
    run=_function_ast_features_run,
    severity="fatal",
    depends_on=("goids",),  # needs function catalog populated
    provides=("analytics.function_ast_features",),
    requires=("core.function_meta",),
    options_model=None,
    options_default=None,
    resource_hints=ResourceHints(
        max_runtime_ms=120_000,
        requires_gpu=False,
        priority=15,
    ),
    version_hash=None,
    row_count_tables=("analytics.function_ast_features",),
)

register_analytics_plugin(FUNCTION_AST_FEATURES_PLUGIN)
```

Notes:

* We reuse `FunctionAnalyticsStepConfig` because it already knows `snapshot.repo`, `commit`, `repo_root`.
* We use `FunctionAstLoadRequest` / `load_function_asts` to obtain the `FunctionAst` objects.
* We call `compute_function_features` for each function and map into row dicts.

In the future, if you wire features into `AnalyticsContext.function_features_map`, you can short-circuit the `load_function_asts`/`compute_function_features` and reuse the existing map instead.

---

### 6.2 Integrate into the harness (Epic 2)

Your generic harness (`analytics/plugin_runtime.py`) should already be able to run multiple function-stage plugins in one shot.

In the function analytics pipeline, set `plugin_names` to include the AST features plugin as well as metrics:

```python
plugin_names = (
    "functions.ast_features",
    "functions.metrics",
)
```

Then plan & run:

```python
plan = plan_analytics_plugin_run(
    plugin_names=plugin_names,
    policy=policy,
    repo=cfg.repo,
    commit=cfg.commit,
    scope=scope,
    prior_manifest=prior_manifest or {},
    cfg_options={},      # possibly add per-plugin config later
    runtime_options={},
    run_id=ctx.run_id,
)

analytics_report = run_analytics_plugins(
    plan=plan,
    gateway=ctx.gateway,
    analytics_context=_analytics_context(ctx),
    graph_runtime=None,
    cfgs={"function": cfg},
    extra={},
)
```

The metrics plugin remains the same; we just added a new plugin that writes into `analytics.function_ast_features`.

---

## 7. Pipeline integration

**File:** `pipeline/orchestration/steps_analytics.py`

Update `FunctionAnalyticsStep.run` (or the equivalent) to run both plugins.

**Before (conceptually):**

```python
plugin_names = ("functions.metrics",)
plan = plan_analytics_plugin_run(...)
report = run_analytics_plugins(...)
```

**After:**

```python
plugin_names = (
    "functions.ast_features",
    "functions.metrics",
)

plan = plan_analytics_plugin_run(
    plugin_names=plugin_names,
    policy=policy,
    repo=cfg.repo,
    commit=cfg.commit,
    scope=scope,
    prior_manifest=prior_manifest or {},
    cfg_options={},
    runtime_options={},
    run_id=ctx.run_id,
)

report = run_analytics_plugins(
    plan=plan,
    gateway=ctx.gateway,
    analytics_context=_analytics_context(ctx),
    graph_runtime=None,
    cfgs={"function": cfg},
    extra={},
)
```

You can keep your existing logging of metrics summary and optionally also log AST features summary:

```python
ast_summary = {}
for rec in report.records:
    if rec.name == "functions.ast_features" and isinstance(rec.meta.get("result"), dict):
        ast_summary = rec.meta["result"]
        break

log.info(
    "function_ast_features summary functions_seen=%d missing=%d rows_written=%d",
    ast_summary.get("functions_seen", 0),
    ast_summary.get("functions_missing", 0),
    ast_summary.get("rows_written", 0),
)
```

---

## 8. Tests & invariants

### 8.1 Row model vs schema parity

**New test:** `tests/analytics/test_function_ast_features_row_contract.py`

```python
from __future__ import annotations

from typing import get_type_hints

from codeintel.config.schemas.tables import TABLE_SCHEMAS
from codeintel.storage.rows import FunctionAstFeaturesRow


def test_function_ast_features_row_matches_schema() -> None:
    schema = TABLE_SCHEMAS["analytics.function_ast_features"]
    expected_cols = [col.name for col in schema.columns]
    hints = get_type_hints(FunctionAstFeaturesRow)
    actual_keys = list(hints.keys())
    assert actual_keys == expected_cols, f"{actual_keys} != {expected_cols}"
```

This will immediately flag any drift between your table schema and `TypedDict`.

---

### 8.2 Plugin end-to-end

**New test:** `tests/analytics/test_function_ast_features_plugin.py`

Assuming you have fixtures for a small repo snapshot:

```python
from __future__ import annotations

from pathlib import Path

from codeintel.analytics.functions.plugins_ast_features import FUNCTION_AST_FEATURES_PLUGIN  # noqa: F401
from codeintel.analytics.plugin_runtime import (
    plan_analytics_plugin_run,
    run_analytics_plugins,
)
from codeintel.config import ConfigBuilder
from codeintel.config.steps_graphs import GraphPluginPolicy, GraphRunScope
from tests._helpers.fixtures import provisioned_gateway


def test_function_ast_features_plugin_populates_table(provisioned_gateway) -> None:
    gateway = provisioned_gateway.gateway
    snapshot = provisioned_gateway.snapshot

    builder = ConfigBuilder.from_snapshot(snapshot)
    cfg = builder.function_analytics(
        fail_on_missing_spans=False,
        parser=None,
    )

    policy = GraphPluginPolicy()
    scope = GraphRunScope()
    run_id = "test-ast-features-1"

    plan = plan_analytics_plugin_run(
        plugin_names=("functions.ast_features",),
        policy=policy,
        repo=cfg.repo,
        commit=cfg.commit,
        scope=scope,
        prior_manifest={},
        cfg_options={},
        runtime_options={},
        run_id=run_id,
    )

    report = run_analytics_plugins(
        plan=plan,
        gateway=gateway,
        analytics_context=None,
        graph_runtime=None,
        cfgs={"function": cfg},
        extra={},
    )

    # Sanity checks on report
    assert len(report.records) == 1
    rec = report.records[0]
    assert rec.name == "functions.ast_features"
    assert rec.status == "succeeded"

    # Confirm rows exist in DuckDB
    con = gateway.con
    row_count = con.execute(
        """
        SELECT COUNT(*) 
        FROM analytics.function_ast_features
        WHERE repo = ? AND commit = ?
        """,
        [cfg.repo, cfg.commit],
    ).fetchone()[0]
    assert row_count > 0
```

You can add a second test for **idempotency**:

* Run plugin once.
* Read rows back as `FunctionAstFeaturesRow`.
* Call `insert_analytics_rows` with `delete_params=[repo, commit]`.
* Verify row counts unchanged.

---

## What this gives you

After this Epic 4.1 step, you have:

* A fully typed, contract-backed dataset `analytics.function_ast_features`.
* A plugin that keeps it up to date per snapshot.
* A table agents can query to answer:

  * “Show me functions that use network IO but no DB access.”
  * “Which functions use `asyncio` but not any HTTP server frameworks?”
  * “Why did we classify this as a controller / CLI / job?” (by pairing this dataset with semantic_roles output).

From here, you can:

* Add a small **view** (e.g. `docs.v_function_ast_features`) joining this table with function profile.
* Expose it via serving as a **searchable surface** for agents (e.g. “find all functions touching S3 plus database”).

If you want, next we can design:

* A **function-ast-features search endpoint** in `serving`, or
* A `docs.v_function_ast_features` DuckDB view for richer docs/analytics.

# function-ast-features deep dive #

Love this one — this is where the “features as data” really pays off.

Let’s design a **search endpoint in `serving`** that lets you (and agents) say things like:

* *“Show me functions that use network IO but not DB.”*
* *“Functions in `services/` that use `asyncio` and `sqlalchemy`.”*
* *“Controllers with lots of feature flag checks.”*

We’ll build it as:

1. A small Pydantic **request/response model** and query builder.
2. A **FastAPI router** under `serving`.
3. Use of the **dataset contract** we just defined (`analytics.function_ast_features`) + optional join to function profile.
4. Tests and some notes on how agents can consume it.

---

## 1. API surface & request/response models

### 1.1 New module: `serving/ast_features_search.py`

Create a new module to keep things focused.

```python
# serving/ast_features_search.py

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Sequence

from fastapi import APIRouter, Depends, HTTPException

from codeintel.storage.gateway import StorageGateway
from codeintel.serving.dependencies import get_gateway  # whatever you already use
```

Define request filters:

```python
from pydantic import BaseModel, Field


class SortBy(str, Enum):
    QUALNAME = "qualname"
    REL_PATH = "rel_path"
    CONFIG_READS = "config_read_count"
    FEATURE_FLAGS = "feature_flag_count"
    NETWORK = "uses_network"
    DB = "uses_db"
    FILESYSTEM = "uses_filesystem"
    CONCURRENCY = "uses_concurrency_lib"


class SortDir(str, Enum):
    ASC = "asc"
    DESC = "desc"


class FunctionAstFeatureFilters(BaseModel):
    repo: str | None = Field(None, description="Filter by repo slug")
    commit: str | None = Field(None, description="Filter by commit hash")

    rel_path_contains: str | None = None
    qualname_contains: str | None = None

    uses_network: bool | None = None
    uses_db: bool | None = None
    uses_filesystem: bool | None = None
    uses_subprocess: bool | None = None
    uses_concurrency_lib: bool | None = None
    uses_threading: bool | None = None
    uses_asyncio_lib: bool | None = None

    # Libraries / frameworks
    any_libs: list[str] = Field(
        default_factory=list,
        description="Match functions that use ANY of these libraries (libraries_used)",
    )
    all_libs: list[str] = Field(
        default_factory=list,
        description="Match functions that use ALL of these libraries (libraries_used)",
    )
    any_http_server_libs: list[str] = Field(
        default_factory=list,
        description="Match any HTTP server library (fastapi, flask, django, ...)",
    )
    any_db_libs: list[str] = Field(
        default_factory=list,
        description="Match any DB library (sqlalchemy, psycopg2, ...)",
    )

    config_reads_min: int | None = None
    feature_flags_min: int | None = None
```

Pagination & overall request:

```python
class FunctionAstFeaturesSearchRequest(BaseModel):
    filters: FunctionAstFeatureFilters = Field(default_factory=FunctionAstFeatureFilters)
    limit: int = Field(50, ge=1, le=500)
    offset: int = Field(0, ge=0)
    sort_by: SortBy = SortBy.FEATURE_FLAGS
    sort_dir: SortDir = SortDir.DESC
```

Response models:

```python
class FunctionAstFeatureResult(BaseModel):
    repo: str
    commit: str
    function_goid_h128: int

    rel_path: str
    qualname: str
    is_async: bool

    uses_network: bool
    uses_db: bool
    uses_filesystem: bool
    uses_subprocess: bool
    uses_concurrency_lib: bool
    uses_threading: bool
    uses_asyncio_lib: bool

    http_client_libs: list[str]
    http_server_libs: list[str]
    db_libs: list[str]
    message_libs: list[str]

    config_read_count: int
    feature_flag_count: int

    # Optional extras from function_profile:
    subsystem: str | None = None
    risk_score: float | None = None


class FunctionAstFeaturesSearchResponse(BaseModel):
    total: int
    limit: int
    offset: int
    results: list[FunctionAstFeatureResult]
```

Router:

```python
router = APIRouter(
    prefix="/ast-features",
    tags=["ast-features"],
)
```

---

## 2. Query builder

We’ll build a SQL string + params based on filters.

### 2.1 Column selection and optional join

We want to:

* Always hit `analytics.function_ast_features` as the main table.
* Optionally join `analytics.function_profile` for subsystem / risk fields.

```python
def _base_select(with_profile: bool) -> str:
    if not with_profile:
        return """
        SELECT
            f.repo,
            f.commit,
            f.function_goid_h128,
            f.rel_path,
            f.qualname,
            f.is_async,
            f.uses_network,
            f.uses_db,
            f.uses_filesystem,
            f.uses_subprocess,
            f.uses_concurrency_lib,
            f.uses_threading,
            f.uses_asyncio_lib,
            f.http_client_libs,
            f.http_server_libs,
            f.db_libs,
            f.message_libs,
            f.config_read_count,
            f.feature_flag_count,
            f.decorators,
            f.libraries_used,
            NULL AS subsystem,
            NULL AS risk_score
        FROM analytics.function_ast_features f
        """
    # With join to function_profile
    return """
    SELECT
        f.repo,
        f.commit,
        f.function_goid_h128,
        f.rel_path,
        f.qualname,
        f.is_async,
        f.uses_network,
        f.uses_db,
        f.uses_filesystem,
        f.uses_subprocess,
        f.uses_concurrency_lib,
        f.uses_threading,
        f.uses_asyncio_lib,
        f.http_client_libs,
        f.http_server_libs,
        f.db_libs,
        f.message_libs,
        f.config_read_count,
        f.feature_flag_count,
        f.decorators,
        f.libraries_used,
        p.subsystem,
        p.risk_score
    FROM analytics.function_ast_features f
    LEFT JOIN analytics.function_profile p
      ON p.repo = f.repo
     AND p.commit = f.commit
     AND p.function_goid_h128 = f.function_goid_h128
    """
```

### 2.2 Filter predicates

We’ll build a list of `WHERE` fragments and params; only add conditions for filters that are set.

```python
def _build_where_clauses(filters: FunctionAstFeatureFilters) -> tuple[list[str], list[Any]]:
    where: list[str] = []
    params: list[Any] = []

    if filters.repo is not None:
        where.append("f.repo = ?")
        params.append(filters.repo)
    if filters.commit is not None:
        where.append("f.commit = ?")
        params.append(filters.commit)

    if filters.rel_path_contains:
        where.append("LOWER(f.rel_path) LIKE ?")
        params.append(f"%{filters.rel_path_contains.lower()}%")
    if filters.qualname_contains:
        where.append("LOWER(f.qualname) LIKE ?")
        params.append(f"%{filters.qualname_contains.lower()}%")

    def flag_clause(field: str, value: bool | None) -> None:
        if value is None:
            return
        where.append(f"f.{field} = ?")
        params.append(value)

    flag_clause("uses_network", filters.uses_network)
    flag_clause("uses_db", filters.uses_db)
    flag_clause("uses_filesystem", filters.uses_filesystem)
    flag_clause("uses_subprocess", filters.uses_subprocess)
    flag_clause("uses_concurrency_lib", filters.uses_concurrency_lib)
    flag_clause("uses_threading", filters.uses_threading)
    flag_clause("uses_asyncio_lib", filters.uses_asyncio_lib)

    if filters.config_reads_min is not None:
        where.append("f.config_read_count >= ?")
        params.append(filters.config_reads_min)

    if filters.feature_flags_min is not None:
        where.append("f.feature_flag_count >= ?")
        params.append(filters.feature_flags_min)

    # Libraries-based filters
    # We'll treat libs columns as JSON arrays of strings.
    # DuckDB: you can use json_contains or json_extract + list_contains. Here we
    # assume a helper function `list_contains` is available, or adapt to your current style.
    def any_libs_clause(column: str, libs: Sequence[str]) -> None:
        if not libs:
            return
        clauses: list[str] = []
        for lib in libs:
            clauses.append(f"list_contains(json_transform(f.{column}), ?)")
            params.append(lib)
        where.append("(" + " OR ".join(clauses) + ")")

    def all_libs_clause(column: str, libs: Sequence[str]) -> None:
        if not libs:
            return
        clauses: list[str] = []
        for lib in libs:
            clauses.append(f"list_contains(json_transform(f.{column}), ?)")
            params.append(lib)
        where.append(" AND ".join(clauses))

    any_libs_clause("libraries_used", filters.any_libs)
    all_libs_clause("libraries_used", filters.all_libs)
    any_libs_clause("http_server_libs", filters.any_http_server_libs)
    any_libs_clause("db_libs", filters.any_db_libs)

    return where, params
```

> `json_transform` / `list_contains` here are placeholders; in your actual DuckDB setup, you might use something like `list_contains(json_extract(f.libraries_used, '$'), ?)` or a view that already flattens JSON to lists. The important part is the pattern.

### 2.3 Sorting & pagination

Map `SortBy` to actual columns:

```python
_SORT_COLUMN_MAP: dict[SortBy, str] = {
    SortBy.QUALNAME: "f.qualname",
    SortBy.REL_PATH: "f.rel_path",
    SortBy.CONFIG_READS: "f.config_read_count",
    SortBy.FEATURE_FLAGS: "f.feature_flag_count",
    SortBy.NETWORK: "f.uses_network",
    SortBy.DB: "f.uses_db",
    SortBy.FILESYSTEM: "f.uses_filesystem",
    SortBy.CONCURRENCY: "f.uses_concurrency_lib",
}
```

Full query builder:

```python
def build_ast_features_query(
    filters: FunctionAstFeatureFilters,
    *,
    limit: int,
    offset: int,
    sort_by: SortBy,
    sort_dir: SortDir,
    with_profile: bool = True,
) -> tuple[str, list[Any]]:
    where_clauses, params = _build_where_clauses(filters)
    base = _base_select(with_profile)

    sql = base
    if where_clauses:
        sql += " WHERE " + " AND ".join(where_clauses)

    sort_col = _SORT_COLUMN_MAP.get(sort_by, "f.feature_flag_count")
    order = "ASC" if sort_dir == SortDir.ASC else "DESC"
    sql += f" ORDER BY {sort_col} {order}, f.rel_path ASC, f.qualname ASC"

    sql += " LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    return sql, params
```

We may also want a `COUNT(*)` query for `total`; we can reuse the same `WHERE`:

```python
def build_ast_features_count_query(filters: FunctionAstFeatureFilters) -> tuple[str, list[Any]]:
    where_clauses, params = _build_where_clauses(filters)
    sql = "SELECT COUNT(*) FROM analytics.function_ast_features f"
    if where_clauses:
        sql += " WHERE " + " AND ".join(where_clauses)
    return sql, params
```

---

## 3. Endpoint implementation

In `serving/ast_features_search.py`:

```python
@router.post(
    "/search",
    response_model=FunctionAstFeaturesSearchResponse,
    summary="Search function AST features",
)
def search_function_ast_features(
    request: FunctionAstFeaturesSearchRequest,
    gateway: StorageGateway = Depends(get_gateway),
) -> FunctionAstFeaturesSearchResponse:
    filters = request.filters
    sort_by = request.sort_by
    sort_dir = request.sort_dir

    # Optional: enforce repo/commit if your serving layer requires it
    if filters.repo is None or filters.commit is None:
        raise HTTPException(
            status_code=400,
            detail="repo and commit filters are required for AST feature search",
        )

    # Count first
    count_sql, count_params = build_ast_features_count_query(filters)
    con = gateway.con
    total = con.execute(count_sql, count_params).fetchone()[0]

    # Fetch results
    sql, params = build_ast_features_query(
        filters,
        limit=request.limit,
        offset=request.offset,
        sort_by=sort_by,
        sort_dir=sort_dir,
        with_profile=True,
    )
    rows = con.execute(sql, params).fetchall()

    results: list[FunctionAstFeatureResult] = []
    for row in rows:
        (
            repo,
            commit,
            goid,
            rel_path,
            qualname,
            is_async,
            uses_network,
            uses_db,
            uses_filesystem,
            uses_subprocess,
            uses_concurrency_lib,
            uses_threading,
            uses_asyncio_lib,
            http_client_libs,
            http_server_libs,
            db_libs,
            message_libs,
            config_read_count,
            feature_flag_count,
            decorators,
            libraries_used,
            subsystem,
            risk_score,
        ) = row

        results.append(
            FunctionAstFeatureResult(
                repo=repo,
                commit=commit,
                function_goid_h128=int(goid),
                rel_path=rel_path,
                qualname=qualname,
                is_async=is_async,
                uses_network=uses_network,
                uses_db=uses_db,
                uses_filesystem=uses_filesystem,
                uses_subprocess=uses_subprocess,
                uses_concurrency_lib=uses_concurrency_lib,
                uses_threading=uses_threading,
                uses_asyncio_lib=uses_asyncio_lib,
                http_client_libs=http_client_libs or [],
                http_server_libs=http_server_libs or [],
                db_libs=db_libs or [],
                message_libs=message_libs or [],
                config_read_count=config_read_count,
                feature_flag_count=feature_flag_count,
                subsystem=subsystem,
                risk_score=risk_score,
            )
        )

    return FunctionAstFeaturesSearchResponse(
        total=int(total),
        limit=request.limit,
        offset=request.offset,
        results=results,
    )
```

This endpoint:

* Ensures `repo` and `commit` are present (you can relax this).
* Returns `total` for pagination.
* Includes optional `subsystem` and `risk_score` where available.

---

## 4. Wire router into your `serving` app

**File:** `serving/api.py` (or wherever the FastAPI app is built)

```python
from fastapi import FastAPI

from codeintel.serving.ast_features_search import router as ast_features_router

app = FastAPI(...)
app.include_router(ast_features_router)
```

If you already have a main router pattern, just include it alongside your existing ones (`/query`, `/graphs`, etc.).

---

## 5. Tests

### 5.1 Query builder unit tests

**File:** `tests/serving/test_ast_features_search_query.py`

```python
from __future__ import annotations

from codeintel.serving.ast_features_search import (
    FunctionAstFeatureFilters,
    build_ast_features_query,
    build_ast_features_count_query,
    SortBy,
    SortDir,
)


def test_build_query_basic_filters() -> None:
    filters = FunctionAstFeatureFilters(
        repo="demo/repo",
        commit="abc123",
        uses_network=True,
        feature_flags_min=1,
    )
    sql, params = build_ast_features_query(
        filters,
        limit=10,
        offset=0,
        sort_by=SortBy.FEATURE_FLAGS,
        sort_dir=SortDir.DESC,
    )

    assert "FROM analytics.function_ast_features f" in sql
    assert "f.repo = ?" in sql
    assert "f.commit = ?" in sql
    assert "f.uses_network = ?" in sql
    assert "f.feature_flag_count >= ?" in sql
    assert sql.endswith("LIMIT ? OFFSET ?")
    assert params[:4] == ["demo/repo", "abc123", True, 1]
```

### 5.2 Endpoint integration test

**File:** `tests/serving/test_ast_features_search_endpoint.py`

Assuming you have test fixtures `test_app` (FastAPI) and `provisioned_gateway`:

```python
from __future__ import annotations

from fastapi.testclient import TestClient

from codeintel.serving.api import app
from tests._helpers.fixtures import provisioned_gateway
from codeintel.analytics.ast_features.persist import features_to_row
from codeintel.analytics.ast_features.model import FunctionAstFeatures, IoFlags
from codeintel.storage.gateway import StorageGateway
from codeintel.analytics.datasets import (
    get_function_ast_features_contract,
    insert_analytics_rows,
)
from datetime import UTC, datetime


client = TestClient(app)


def _seed_function_ast_features(gateway: StorageGateway, repo: str, commit: str) -> None:
    # Minimal synthetic feature row
    features = FunctionAstFeatures(
        goid=1,
        rel_path="app/services/user.py",
        qualname="app.services.user.get_user",
        is_async=False,
        decorators=(),
        imports={},
        libraries_used=frozenset(["requests", "sqlalchemy"]),
        io_flags=IoFlags(uses_network=True, uses_db=True),
        uses_concurrency_lib=False,
        uses_threading=False,
        uses_asyncio_lib=False,
        http_client_libs=frozenset(["requests"]),
        http_server_libs=frozenset(),
        db_libs=frozenset(["sqlalchemy"]),
        message_libs=frozenset(),
        config_read_count=2,
        feature_flag_count=1,
        extra={},
    )
    row = features_to_row(
        repo=repo,
        commit=commit,
        features=features,
        created_at=datetime.now(tz=UTC),
    )
    contract = get_function_ast_features_contract(gateway)
    insert_analytics_rows(
        gateway,
        contract,
        [row],
        delete_params=[repo, commit],
        scope=f"{repo}@{commit}",
    )


def test_ast_features_search_endpoint(provisioned_gateway) -> None:
    gateway = provisioned_gateway.gateway
    snapshot = provisioned_gateway.snapshot

    _seed_function_ast_features(gateway, snapshot.repo, snapshot.commit)

    response = client.post(
        "/ast-features/search",
        json={
            "filters": {
                "repo": snapshot.repo,
                "commit": snapshot.commit,
                "uses_network": True,
                "any_libs": ["requests"],
            },
            "limit": 10,
            "offset": 0,
            "sort_by": "feature_flag_count",
            "sort_dir": "desc",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    result = payload["results"][0]
    assert result["rel_path"] == "app/services/user.py"
    assert result["uses_network"] is True
    assert "requests" in result["http_client_libs"]
```

---

## 6. How agents can use this

With this endpoint, your MCP / agent layer can now:

* Call `/ast-features/search` with filters like:

  ```json
  {
    "filters": {
      "repo": "my/repo",
      "commit": "abc123",
      "uses_network": true,
      "uses_db": false,
      "rel_path_contains": "services",
      "feature_flags_min": 1
    },
    "sort_by": "feature_flag_count",
    "sort_dir": "desc",
    "limit": 20
  }
  ```

* Get back a list of functions plus their IO/concurrency flags, libs, subsystem, and risk, and then:

  * Ask follow-up questions per function.
  * Propose refactors (e.g., “wrap all network calls in a retry helper”).
  * Explain *why* it labelled something a “controller” / “entrypoint” by pairing this with your semantic_roles outputs.

If you’d like, we can next add:

* A **secondary endpoint** that, given a `function_goid_h128`, returns a “full explanation” object combining:

  * `function_ast_features`
  * `function_profile` (LOC, complexity, typedness)
  * `semantic_roles` classification and supporting evidence.

# secondary endpoint for function_goid_h128 deep dive #

Nice, let’s give you a **“why is this function interesting?”** endpoint.

We’ll build a `/functions/explain` endpoint that, given `(repo, commit, function_goid_h128)`, returns:

* **AST features** (IO, libs, concurrency).
* **Function profile** (location, complexity, typedness, docs).
* **Semantic roles** (role, confidence, evidence).

I’ll structure it as:

1. API models & router module.
2. Query helpers against DuckDB.
3. Endpoint implementation.
4. Wiring into your `serving` app.
5. Tests + a simple “reasons” builder.

I’ll assume:

* `analytics.function_ast_features` exists (as in Epic 4.1).
* `analytics.function_profile` and `analytics.semantic_roles_functions` exist (they do in your schema).
* You’re using FastAPI in `serving`.

---

## 1. New module: `serving/function_explain.py`

Create a new module for this endpoint.

```python
# serving/function_explain.py

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from codeintel.storage.gateway import StorageGateway
from codeintel.serving.dependencies import get_gateway  # your existing DI hook
```

### 1.1 Request model

We *need* repo & commit to uniquely identify the row (PK is `(repo, commit, function_goid_h128)`).

```python
class FunctionExplainRequest(BaseModel):
    repo: str = Field(..., description="Repository slug")
    commit: str = Field(..., description="Commit hash")
    function_goid_h128: int = Field(..., description="Function GOID (H128)")
```

### 1.2 AST features summary

This mirrors the important parts of `analytics.function_ast_features`:

```python
class FunctionAstFeaturesSummary(BaseModel):
    rel_path: str
    qualname: str
    is_async: bool

    uses_network: bool
    uses_db: bool
    uses_filesystem: bool
    uses_subprocess: bool

    uses_concurrency_lib: bool
    uses_threading: bool
    uses_asyncio_lib: bool

    http_client_libs: list[str]
    http_server_libs: list[str]
    db_libs: list[str]
    message_libs: list[str]

    config_read_count: int
    feature_flag_count: int

    decorators: list[str]
    libraries_used: list[str]
```

### 1.3 Function profile summary

Pull out the fields that actually help explanation (location, complexity, typedness, docs). Adjust to match your exact columns.

```python
class FunctionProfileSummary(BaseModel):
    rel_path: str
    module: str | None
    language: str | None
    kind: str | None
    qualname: str | None
    start_line: int | None
    end_line: int | None
    loc: int | None
    logical_loc: int | None
    cyclomatic_complexity: int | None
    complexity_bucket: str | None

    param_count: int | None
    positional_params: int | None
    keyword_params: int | None
    vararg: bool | None
    kwarg: bool | None

    typedness_bucket: str | None
    typed_param_ratio: float | None

    subsystem: str | None = None
    risk_score: float | None = None

    doc_short: str | None = None
    doc_long: str | None = None
```

> `typedness_bucket` / `typed_param_ratio` field names should align with your actual schema; adjust accordingly.

### 1.4 Semantic roles explanation

From `analytics.semantic_roles_functions`:

```python
class RoleEvidence(BaseModel):
    source: str
    weight: float | None = None
    detail: str | None = None


class FunctionRoleSummary(BaseModel):
    role: str | None
    framework: str | None
    role_confidence: float | None
    evidence: list[RoleEvidence] = Field(default_factory=list)
```

We’ll decode `role_sources_json` into `RoleEvidence` objects.

### 1.5 Full explanation object

We can also include a derived “reasons” list for agents / UI.

```python
class FunctionExplainResponse(BaseModel):
    repo: str
    commit: str
    function_goid_h128: int

    ast_features: FunctionAstFeaturesSummary | None
    profile: FunctionProfileSummary | None
    semantic_role: FunctionRoleSummary | None

    reasons: list[str] = Field(
        default_factory=list,
        description="Human-readable bullets explaining why this function is interesting.",
    )
```

Router setup:

```python
router = APIRouter(
    prefix="/functions",
    tags=["functions"],
)
```

---

## 2. Query helpers

We’ll write small helpers that each do one thing:

* Fetch AST features row.
* Fetch function profile row (+ maybe subsystem/risk).
* Fetch semantic roles row.

We’ll centralize JSON normalization to handle DuckDB’s JSON columns.

### 2.1 JSON normalization helper

```python
import json
from typing import Any


def _normalize_json(value: Any) -> Any:
    """
    Normalize DuckDB JSON value into native Python types.

    - If value is None -> None
    - If it's a str -> json.loads
    - Else assume it's already a decoded structure.
    """
    if value is None:
        return None
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value
```

### 2.2 Fetch AST features

```python
from codeintel.analytics.ast_features.model import FunctionAstFeatures  # optional, for typing

def fetch_ast_features_row(
    gateway: StorageGateway,
    repo: str,
    commit: str,
    function_goid_h128: int,
) -> FunctionAstFeaturesSummary | None:
    con = gateway.con
    row = con.execute(
        """
        SELECT
            rel_path,
            qualname,
            is_async,
            uses_network,
            uses_db,
            uses_filesystem,
            uses_subprocess,
            uses_concurrency_lib,
            uses_threading,
            uses_asyncio_lib,
            http_client_libs,
            http_server_libs,
            db_libs,
            message_libs,
            config_read_count,
            feature_flag_count,
            decorators,
            libraries_used
        FROM analytics.function_ast_features
        WHERE repo = ? AND commit = ? AND function_goid_h128 = ?
        """,
        [repo, commit, function_goid_h128],
    ).fetchone()

    if row is None:
        return None

    (
        rel_path,
        qualname,
        is_async,
        uses_network,
        uses_db,
        uses_filesystem,
        uses_subprocess,
        uses_concurrency_lib,
        uses_threading,
        uses_asyncio_lib,
        http_client_libs,
        http_server_libs,
        db_libs,
        message_libs,
        config_read_count,
        feature_flag_count,
        decorators,
        libraries_used,
    ) = row

    http_client_libs = _normalize_json(http_client_libs) or []
    http_server_libs = _normalize_json(http_server_libs) or []
    db_libs = _normalize_json(db_libs) or []
    message_libs = _normalize_json(message_libs) or []
    decorators = _normalize_json(decorators) or []
    libraries_used = _normalize_json(libraries_used) or []

    return FunctionAstFeaturesSummary(
        rel_path=rel_path,
        qualname=qualname,
        is_async=is_async,
        uses_network=uses_network,
        uses_db=uses_db,
        uses_filesystem=uses_filesystem,
        uses_subprocess=uses_subprocess,
        uses_concurrency_lib=uses_concurrency_lib,
        uses_threading=uses_threading,
        uses_asyncio_lib=uses_asyncio_lib,
        http_client_libs=http_client_libs,
        http_server_libs=http_server_libs,
        db_libs=db_libs,
        message_libs=message_libs,
        config_read_count=config_read_count,
        feature_flag_count=feature_flag_count,
        decorators=decorators,
        libraries_used=libraries_used,
    )
```

### 2.3 Fetch function profile (+ subsystem/risk)

We’ll join `function_profile` and `function_profile` (or another table) as needed. For now, use `function_profile` only; if your subsystem/risk info lives elsewhere, adjust accordingly.

```python
def fetch_function_profile_summary(
    gateway: StorageGateway,
    repo: str,
    commit: str,
    function_goid_h128: int,
) -> FunctionProfileSummary | None:
    con = gateway.con
    row = con.execute(
        """
        SELECT
            rel_path,
            module,
            language,
            kind,
            qualname,
            start_line,
            end_line,
            loc,
            logical_loc,
            cyclomatic_complexity,
            complexity_bucket,
            param_count,
            positional_params,
            keyword_params,
            vararg,
            kwarg,
            typedness_bucket,
            typed_param_ratio,
            subsystem,
            risk_score,
            doc_short,
            doc_long
        FROM analytics.function_profile
        WHERE repo = ? AND commit = ? AND function_goid_h128 = ?
        """,
        [repo, commit, function_goid_h128],
    ).fetchone()

    if row is None:
        return None

    (
        rel_path,
        module,
        language,
        kind,
        qualname,
        start_line,
        end_line,
        loc,
        logical_loc,
        cyclomatic_complexity,
        complexity_bucket,
        param_count,
        positional_params,
        keyword_params,
        vararg,
        kwarg,
        typedness_bucket,
        typed_param_ratio,
        subsystem,
        risk_score,
        doc_short,
        doc_long,
    ) = row

    return FunctionProfileSummary(
        rel_path=rel_path,
        module=module,
        language=language,
        kind=kind,
        qualname=qualname,
        start_line=start_line,
        end_line=end_line,
        loc=loc,
        logical_loc=logical_loc,
        cyclomatic_complexity=cyclomatic_complexity,
        complexity_bucket=complexity_bucket,
        param_count=param_count,
        positional_params=positional_params,
        keyword_params=keyword_params,
        vararg=vararg,
        kwarg=kwarg,
        typedness_bucket=typedness_bucket,
        typed_param_ratio=typed_param_ratio,
        subsystem=subsystem,
        risk_score=risk_score,
        doc_short=doc_short,
        doc_long=doc_long,
    )
```

Adjust field names to exactly match your `function_profile` schema.

### 2.4 Fetch semantic role

```python
def fetch_semantic_role_summary(
    gateway: StorageGateway,
    repo: str,
    commit: str,
    function_goid_h128: int,
) -> FunctionRoleSummary | None:
    con = gateway.con
    row = con.execute(
        """
        SELECT role, framework, role_confidence, role_sources_json
        FROM analytics.semantic_roles_functions
        WHERE repo = ? AND commit = ? AND function_goid_h128 = ?
        """,
        [repo, commit, function_goid_h128],
    ).fetchone()

    if row is None:
        return None

    role, framework, role_confidence, role_sources_json = row

    evidence: list[RoleEvidence] = []
    raw_sources = _normalize_json(role_sources_json)
    if isinstance(raw_sources, list):
        for src in raw_sources:
            if not isinstance(src, dict):
                continue
            evidence.append(
                RoleEvidence(
                    source=str(src.get("source", "")),
                    weight=src.get("weight"),
                    detail=src.get("detail") or src.get("reason"),
                )
            )

    return FunctionRoleSummary(
        role=role,
        framework=framework,
        role_confidence=role_confidence,
        evidence=evidence,
    )
```

If you also want module-level roles from `analytics.semantic_roles_modules`, you can add a second query and extend the response later.

---

## 3. Building “reasons” (explanation bullets)

This is optional but very nice for agents and UI: turn the combined info into a list of human-readable bullets.

```python
def build_explanation_reasons(
    *,
    ast_features: FunctionAstFeaturesSummary | None,
    profile: FunctionProfileSummary | None,
    semantic_role: FunctionRoleSummary | None,
) -> list[str]:
    reasons: list[str] = []

    if profile is not None:
        if profile.complexity_bucket:
            reasons.append(
                f"Complexity bucket: {profile.complexity_bucket} "
                f"(cyclomatic={profile.cyclomatic_complexity})"
            )
        if profile.typedness_bucket:
            reasons.append(
                f"Typedness: {profile.typedness_bucket} "
                f"(typed_param_ratio={profile.typed_param_ratio})"
            )
        if profile.subsystem:
            reasons.append(f"Subsystem: {profile.subsystem}")
        if profile.risk_score is not None:
            reasons.append(f"Risk score: {profile.risk_score:.2f}")

    if ast_features is not None:
        io_bits = []
        if ast_features.uses_network:
            io_bits.append("network")
        if ast_features.uses_db:
            io_bits.append("database")
        if ast_features.uses_filesystem:
            io_bits.append("filesystem")
        if ast_features.uses_subprocess:
            io_bits.append("subprocess")
        if io_bits:
            reasons.append(f"Performs IO: {', '.join(io_bits)}")

        if ast_features.uses_concurrency_lib or ast_features.uses_asyncio_lib or ast_features.uses_threading:
            kinds = []
            if ast_features.uses_asyncio_lib:
                kinds.append("asyncio")
            if ast_features.uses_threading:
                kinds.append("threading")
            if ast_features.uses_concurrency_lib and not kinds:
                kinds.append("other concurrency libs")
            reasons.append(f"Uses concurrency libraries: {', '.join(kinds)}")

        if ast_features.http_server_libs:
            reasons.append(
                f"Uses HTTP server frameworks: {', '.join(ast_features.http_server_libs)}"
            )
        if ast_features.http_client_libs:
            reasons.append(
                f"Uses HTTP clients: {', '.join(ast_features.http_client_libs)}"
            )
        if ast_features.db_libs:
            reasons.append(f"Uses DB libraries: {', '.join(ast_features.db_libs)}")

    if semantic_role is not None and semantic_role.role:
        summary = f"Semantic role: {semantic_role.role}"
        if semantic_role.framework:
            summary += f" (framework={semantic_role.framework})"
        if semantic_role.role_confidence is not None:
            summary += f" [confidence={semantic_role.role_confidence:.2f}]"
        reasons.append(summary)

        if semantic_role.evidence:
            top_sources = ", ".join(ev.source for ev in semantic_role.evidence[:3])
            reasons.append(f"Role evidence sources: {top_sources}")

    return reasons
```

---

## 4. Endpoint implementation

Now wire everything into the router.

```python
@router.post(
    "/explain",
    response_model=FunctionExplainResponse,
    summary="Explain a function using AST features, profile, and semantic roles",
)
def explain_function(
    request: FunctionExplainRequest,
    gateway: StorageGateway = Depends(get_gateway),
) -> FunctionExplainResponse:
    repo = request.repo
    commit = request.commit
    goid = request.function_goid_h128

    ast_features = fetch_ast_features_row(gateway, repo, commit, goid)
    profile = fetch_function_profile_summary(gateway, repo, commit, goid)
    semantic_role = fetch_semantic_role_summary(gateway, repo, commit, goid)

    # If we don't even have a profile row, treat as "not found"
    if profile is None and ast_features is None and semantic_role is None:
        raise HTTPException(
            status_code=404,
            detail="Function not found in analytics datasets",
        )

    reasons = build_explanation_reasons(
        ast_features=ast_features,
        profile=profile,
        semantic_role=semantic_role,
    )

    return FunctionExplainResponse(
        repo=repo,
        commit=commit,
        function_goid_h128=goid,
        ast_features=ast_features,
        profile=profile,
        semantic_role=semantic_role,
        reasons=reasons,
    )
```

---

## 5. Wire router into the app

**File:** `serving/api.py` (or wherever your FastAPI app is constructed):

```python
from fastapi import FastAPI

from codeintel.serving.ast_features_search import router as ast_features_router
from codeintel.serving.function_explain import router as function_explain_router

app = FastAPI(...)
app.include_router(ast_features_router)
app.include_router(function_explain_router)
```

Now you get both:

* `/ast-features/search` — broad feature search.
* `/functions/explain` — deep explanation for a single function.

---

## 6. Tests

### 6.1 Unit test: reasons builder

**File:** `tests/serving/test_function_explain_reasons.py`

```python
from __future__ import annotations

from codeintel.serving.function_explain import (
    FunctionAstFeaturesSummary,
    FunctionProfileSummary,
    FunctionRoleSummary,
    RoleEvidence,
    build_explanation_reasons,
)


def test_build_explanation_reasons_combines_sources() -> None:
    ast_summary = FunctionAstFeaturesSummary(
        rel_path="app/services/user.py",
        qualname="app.services.user.get_user",
        is_async=False,
        uses_network=True,
        uses_db=True,
        uses_filesystem=False,
        uses_subprocess=False,
        uses_concurrency_lib=True,
        uses_threading=False,
        uses_asyncio_lib=True,
        http_client_libs=["requests"],
        http_server_libs=["fastapi"],
        db_libs=["sqlalchemy"],
        message_libs=[],
        config_read_count=2,
        feature_flag_count=1,
        decorators=["@app.get('/users/{id}')"],
        libraries_used=["requests", "sqlalchemy", "fastapi"],
    )
    profile = FunctionProfileSummary(
        rel_path="app/services/user.py",
        module="app.services.user",
        language="python",
        kind="function",
        qualname="app.services.user.get_user",
        start_line=10,
        end_line=40,
        loc=30,
        logical_loc=25,
        cyclomatic_complexity=8,
        complexity_bucket="medium",
        param_count=2,
        positional_params=2,
        keyword_params=0,
        vararg=False,
        kwarg=False,
        typedness_bucket="partially_typed",
        typed_param_ratio=0.5,
        subsystem="user_service",
        risk_score=0.78,
        doc_short="Get a user.",
        doc_long=None,
    )
    role = FunctionRoleSummary(
        role="controller",
        framework="fastapi",
        role_confidence=0.91,
        evidence=[RoleEvidence(source="decorator:@app.get", weight=0.7)],
    )

    reasons = build_explanation_reasons(
        ast_features=ast_summary,
        profile=profile,
        semantic_role=role,
    )

    assert any("Complexity bucket: medium" in r for r in reasons)
    assert any("Performs IO: network, database" in r for r in reasons)
    assert any("Uses concurrency libraries" in r for r in reasons)
    assert any("Semantic role: controller" in r for r in reasons)
```

### 6.2 Endpoint integration test

**File:** `tests/serving/test_function_explain_endpoint.py`

You’ll seed minimal rows into:

* `analytics.function_ast_features`
* `analytics.function_profile`
* `analytics.semantic_roles_functions`

and hit `/functions/explain`.

```python
from __future__ import annotations

from datetime import UTC, datetime

from fastapi.testclient import TestClient

from codeintel.serving.api import app
from codeintel.storage.gateway import StorageGateway
from tests._helpers.fixtures import provisioned_gateway  # or similar


client = TestClient(app)


def _seed_explanation_rows(gateway: StorageGateway, repo: str, commit: str, goid: int) -> None:
    con = gateway.con
    now = datetime.now(tz=UTC)

    con.execute(
        """
        INSERT INTO analytics.function_ast_features (
            repo, commit, function_goid_h128,
            rel_path, qualname, is_async,
            uses_network, uses_db, uses_filesystem, uses_subprocess,
            uses_concurrency_lib, uses_threading, uses_asyncio_lib,
            http_client_libs, http_server_libs, db_libs, message_libs,
            config_read_count, feature_flag_count,
            decorators, libraries_used,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            repo,
            commit,
            goid,
            "app/services/user.py",
            "app.services.user.get_user",
            False,
            True,
            True,
            False,
            False,
            True,
            False,
            True,
            '["requests"]',
            '["fastapi"]',
            '["sqlalchemy"]',
            "[]",
            2,
            1,
            '["@app.get(\'/users/{id}\')"]',
            '["requests", "sqlalchemy", "fastapi"]',
            now,
        ],
    )

    con.execute(
        """
        INSERT INTO analytics.function_profile (
            function_goid_h128, urn, repo, commit, rel_path, module, language, kind, qualname,
            start_line, end_line, loc, logical_loc, cyclomatic_complexity, complexity_bucket,
            param_count, positional_params, keyword_params, vararg, kwarg,
            typedness_bucket, typed_param_ratio,
            subsystem, risk_score,
            doc_short, doc_long,
            created_at
        )
        VALUES (?, NULL, ?, ?, ?, ?, 'python', 'function', ?, 10, 40, 30, 25, 8, 'medium',
                2, 2, 0, FALSE, FALSE,
                'partially_typed', 0.5,
                'user_service', 0.78,
                'Get a user.', NULL,
                ?)
        """,
        [goid, repo, commit, "app/services/user.py", "app.services.user", "app.services.user.get_user", now],
    )

    con.execute(
        """
        INSERT INTO analytics.semantic_roles_functions (
            repo, commit, function_goid_h128,
            role, framework, role_confidence, role_sources_json, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            repo,
            commit,
            goid,
            "controller",
            "fastapi",
            0.91,
            '[{"source": "decorator:@app.get", "weight": 0.7}]',
            now,
        ],
    )


def test_function_explain_endpoint(provisioned_gateway) -> None:
    gateway = provisioned_gateway.gateway
    snapshot = provisioned_gateway.snapshot
    goid = 1

    _seed_explanation_rows(gateway, snapshot.repo, snapshot.commit, goid)

    response = client.post(
        "/functions/explain",
        json={
            "repo": snapshot.repo,
            "commit": snapshot.commit,
            "function_goid_h128": goid,
        },
    )
    assert response.status_code == 200
    payload = response.json()

    assert payload["repo"] == snapshot.repo
    assert payload["commit"] == snapshot.commit
    assert payload["function_goid_h128"] == goid

    ast = payload["ast_features"]
    prof = payload["profile"]
    role = payload["semantic_role"]
    reasons = payload["reasons"]

    assert ast["uses_network"] is True
    assert "requests" in ast["http_client_libs"]
    assert prof["subsystem"] == "user_service"
    assert role["role"] == "controller"
    assert any("Semantic role: controller" in r for r in reasons)
```

---

With this in place, you now have a **single, high-level explanation endpoint** that pulls together:

* Raw features (`function_ast_features`).
* Structural context (`function_profile`).
* Semantic classification + evidence (`semantic_roles_functions`).

This is a perfect surface for:

* Agents to rank functions by “interestingness”.
* UIs to show a “why we flagged this” pane.
* Future work: adding more slices (e.g. test coverage, history deltas) into the same explanation without changing callers.
