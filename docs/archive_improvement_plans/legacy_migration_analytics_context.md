You can treat this as “finish the AnalyticsContext migration and clean out the last crumbs.” The good news is: there’s already only one real runtime architecture; we’re just removing compatibility names and deprecated API surfaces. 

I’ll break the plan into:

1. Kill `AnalyticsContext`-style compat in `analytics/resources/*`
2. Remove `DeleteOptions.params` and “legacy adapter” surface in `analytics/adapters/base.py`
3. Clean up analytics metrics compat naming (small but easy win)

---

## 1. Kill `AnalyticsContext` compat in `analytics/resources/*`

### 1.1. Confirm nothing still uses the compat properties

You already did a search for `function_features` and only saw the property itself, not external usage. To be thorough, rerun this across the repo:

```bash
rg "function_features" src
rg "function_features_map" src
rg "module_asts" src
rg "function_asts" src
```

You should only see definitions in `analytics/resources/features.py` and `asts.py`, plus maybe comments. If that’s true, we’re free to remove the compat properties entirely and just keep the provider interface.

---

### 1.2. `analytics/resources/features.py`: remove `function_features*` aliases

**Before** (conceptual shape):

```python
class FeaturesProvider(ResourceProvider[FunctionFeaturesMap]):
    """Provides function-level features.

    For backwards compatibility with the legacy AnalyticsContext API we
    expose function_features and function_features_map, which mirror
    context.function_features_map.
    """

    ...

    @property
    def function_features(self) -> FunctionFeaturesMap:
        """Compat alias for legacy AnalyticsContext.function_features_map."""
        return self.get()

    @property
    def function_features_map(self) -> FunctionFeaturesMap:
        """Compat alias for legacy AnalyticsContext.function_features_map."""
        return self.get()
```

**After** (clean provider-only API):

```python
class FeaturesProvider(ResourceProvider[FunctionFeaturesMap]):
    """Provides function-level features.

    This provider is the canonical way to access function features.
    Call ``get()`` to obtain the feature map keyed by GOID.
    """

    ...
    # No compat aliases – callers use .get() directly.
```

Actions:

1. Delete the `function_features` and `function_features_map` properties entirely.
2. Update the class docstring to remove any mention of `AnalyticsContext` or `context.function_features_map`.

   * Keep it short and present-tense: the provider is the one true way.

If you have any similar compat properties (e.g. `subsystem_features`, `module_features`) explicitly described as “for legacy AnalyticsContext compatibility,” remove them the same way.

---

### 1.3. `analytics/resources/asts.py`: remove AST compat properties

`asts.py` has very similar wording: properties that exist just so old `AnalyticsContext` code like `context.module_asts` / `context.function_asts` would still have equivalents.

**Before (conceptual):**

```python
class AstsProvider(ResourceProvider[AstIndex]):
    """Provides ASTs for functions and modules.

    Legacy AnalyticsContext API exposed `context.function_asts` and
    `context.module_asts`; these properties mirror that shape.
    """

    ...

    @property
    def function_asts(self) -> Mapping[Goid, cst.CSTNode]:
        """Compat alias for legacy AnalyticsContext.function_asts."""
        return self.get().function_asts

    @property
    def module_asts(self) -> Mapping[str, cst.Module]:
        """Compat alias for legacy AnalyticsContext.module_asts."""
        return self.get().module_asts
```

**After:**

```python
class AstsProvider(ResourceProvider[AstIndex]):
    """Provides ASTs for functions and modules.

    The provider returns an `AstIndex` via `get()`, which exposes
    function and module ASTs keyed by GOID and module path.
    """

    ...
    # No compat aliases – callers use get() and the AstIndex API.
```

Actions:

1. Remove `function_asts` and `module_asts` properties (or whatever the exact compat names are).
2. Update the docstring to describe only the `AstIndex` object and `.get()`.

Any code that really needs “just the function ASTs map” can call:

```python
ast_index = asts_provider.get()
function_asts = ast_index.function_asts
```

–– but within this repo there shouldn’t be any existing callers, based on previous searches.

---

### 1.4. `analytics/resources/__init__.py`: stop narrating AnalyticsContext

This module’s top docstring and comments reference the migration:

> “These resource providers replace the old AnalyticsContext abstraction…”

You’ve already removed the old class; leaving those comments makes it sound like both might still exist.

**Before (conceptual):**

```python
"""Resource providers for analytics.

This package contains resource-provider abstractions that replace the
legacy AnalyticsContext and its bloated execution context. New code
should use these providers instead of AnalyticsContext.
"""
```

**After:**

```python
"""Resource providers for analytics.

This package defines resource-provider abstractions used throughout
the analytics pipeline (features, ASTs, graphs, etc.). These are the
canonical way to access shared data.
"""
```

Actions:

1. Remove “legacy AnalyticsContext” language from the module docstring.
2. If there are line-level comments like “kept for AnalyticsContext compat,” delete or rephrase them to pure “provider” language.

Optional: if you want to keep a bit of history, add a short note like:

> Historically, analytics used an `AnalyticsContext` object; that abstraction has been fully removed in favor of these providers.

…but don’t imply any runtime duality.

---

## 2. Remove `DeleteOptions.params` & “legacy adapter” surface

This is the one piece in analytics that actually still has a small legacy API surface, even though architecture is unified.

### 2.1. Simplify `DeleteOptions` and remove `DeleteParams`

**Before** (based on the snippet we saw):

```python
@dataclass(frozen=True)
class DeleteParams:
    repo: str | None = None
    commit: str | None = None


@dataclass(frozen=True)
class DeleteOptions:
    """Options controlling delete behavior.

    The `params` attribute is deprecated; use `repo` and `commit`
    instead. Params will be removed in a future release once all
    call-sites are migrated.
    """
    params: DeleteParams | None = None
    repo: str | None = None
    commit: str | None = None


def resolve_delete_params(options: DeleteOptions) -> DeleteParams:
    if options.params is not None:
        return options.params
    return DeleteParams(repo=options.repo, commit=options.commit)
```

**Target:** no `DeleteParams`, no `resolve_delete_params`, no “legacy adapter base.”

**After:**

```python
@dataclass(frozen=True)
class DeleteOptions:
    """Options controlling delete behavior.

    Callers must pass explicit `repo` and/or `commit` identifiers.
    """
    repo: str | None = None
    commit: str | None = None
```

Actions:

1. **Delete the `DeleteParams` dataclass entirely.**
2. **Delete `resolve_delete_params`** and any other helper whose only purpose is to translate `params` → `repo/commit`.
3. Simplify `DeleteOptions` to just `repo` and `commit`.

---

### 2.2. Update adapter base classes to use explicit fields

Somewhere lower in `analytics/adapters/base.py` you’ll have an abstract base that used the params helper, e.g.:

```python
class AnalyticsAdapter(ABC):
    ...

    def delete(self, options: DeleteOptions) -> None:
        """Delete rows for the given repo/commit."""
        params = resolve_delete_params(options)
        self._delete_for_params(params)

    @abstractmethod
    def _delete_for_params(self, params: DeleteParams) -> None:
        ...
```

Or similar.

**After**: make `DeleteOptions` the only thing, and don’t hide it behind another datatype.

For example:

```python
class AnalyticsAdapter(ABC):
    ...

    def delete(self, options: DeleteOptions) -> None:
        """Delete rows for the given repo/commit.

        If `repo` and `commit` are both None, implementations may interpret
        this as "delete for all snapshots" or reject the call, depending
        on their semantics.
        """
        self._delete_for_repo_commit(repo=options.repo, commit=options.commit)

    @abstractmethod
    def _delete_for_repo_commit(
        self,
        *,
        repo: str | None,
        commit: str | None,
    ) -> None:
        ...
```

Then, for each concrete adapter:

**Before (example):**

```python
class FunctionMetricsAdapter(AnalyticsAdapter):
    ...

    def _delete_for_params(self, params: DeleteParams) -> None:
        repo = params.repo
        commit = params.commit
        ...
```

**After:**

```python
class FunctionMetricsAdapter(AnalyticsAdapter):
    ...

    def _delete_for_repo_commit(
        self,
        *,
        repo: str | None,
        commit: str | None,
    ) -> None:
        ...
```

Actions:

1. Change the abstract method signature in the base class from `_delete_for_params(self, params: DeleteParams)` to `_delete_for_repo_commit(self, *, repo: str | None, commit: str | None)`.
2. Update **all subclasses** in `analytics/adapters/*` to implement the new method instead of the old one.
3. If any subclass or caller referred directly to `DeleteParams` or `resolve_delete_params`, strip that out and just use `DeleteOptions.repo` / `.commit`.

---

### 2.3. Update all callers constructing `DeleteOptions`

Search for creation of `DeleteOptions` and any use of `DeleteParams`:

```bash
rg "DeleteOptions\(" src/analytics
rg "DeleteParams" src/analytics
```

**Before:**

```python
opts = DeleteOptions(
    params=DeleteParams(repo=snapshot.repo, commit=snapshot.commit),
)

adapter.delete(opts)
```

**After:**

```python
opts = DeleteOptions(
    repo=snapshot.repo,
    commit=snapshot.commit,
)

adapter.delete(opts)
```

or, where you don’t pass params currently:

```python
# before
opts = DeleteOptions()

# after – unchanged, explicit default repo/commit=None
opts = DeleteOptions()
```

Once these are updated, nothing should mention `DeleteParams` or `params` anymore.

---

## 3. Metrics compat cleanup (small but nice)

This part is small but it removes another “legacy” label that can confuse future readers.

### 3.1. `analytics/core/plugins/functions/metrics.py`: rename legacy counters

There’s a section that currently looks roughly like:

```python
# legacy counter names for metrics/typedness
metrics_rows = result.metrics_rows
types_rows = result.types_rows

row_counts = {
    "analytics.function_metrics": len(metrics_rows),
    "analytics.function_types": len(types_rows),
}
```

With a comment like:

> `# NOTE: legacy counter names; keep for compatibility with old metrics pipeline`

Because you’re aggressively dropping compat, just rename them to match the datasets they represent and drop the comment.

**After:**

```python
function_metrics_rows = result.metrics_rows
function_types_rows = result.types_rows

row_counts = {
    "analytics.function_metrics": len(function_metrics_rows),
    "analytics.function_types": len(function_types_rows),
}
```

If the `result` object’s attributes still need to be named `metrics_rows` / `types_rows` for other parts of the code, that’s fine – you’ve localized the “weird” names to the compute return type and made plugin code self-explanatory.

---

### 3.2. `analytics/graph_rows/graph_metrics.py`: remove compat aliases

You likely have something like:

```python
# Compat aliases for old metric names
NodeCentralityRow = GraphMetricsRow
EdgeCentralityRow = GraphMetricsRow
```

or a mapping described as “compat for old metric names.”

Steps:

1. Search for the alias names (e.g. `NodeCentralityRow`) across analytics:

   ```bash
   rg "NodeCentralityRow" src/analytics
   rg "EdgeCentralityRow" src/analytics
   ```

2. If they are **only defined** and not used anywhere else (common in these sorts of shims):

   * Delete the alias declarations.
   * Delete any comments describing them as compat.

3. If they are used in a few places:

   * Replace those usages with the canonical type (e.g. `GraphMetricsRow`).
   * Then delete the aliases.

Result: no “compat alias” comments, and all code uses the canonical row type(s).

---

## 4. Sanity checklist

You’re “done” with this cluster when:

* [ ] `rg "AnalyticsContext" src/analytics` returns **no code** references (comments are optional; ideally removed or clearly historical).
* [ ] `analytics/resources/features.py` and `analytics/resources/asts.py` expose only provider-style APIs (no `function_features`, `function_asts`, etc. compat properties).
* [ ] `analytics/resources/__init__.py` docstring describes the provider pattern without implying runtime coexistence with `AnalyticsContext`.
* [ ] `analytics/adapters/base.py` defines **only** `DeleteOptions` with `repo` and `commit` (no `DeleteParams`, no `params` field).
* [ ] All analytics adapters implement `_delete_for_repo_commit(...)` (or equivalent explicit API), and no code refers to `DeleteParams`.
* [ ] Any “legacy counter names” comments in `analytics/core/plugins/functions/metrics.py` have been removed; variable names align with dataset semantics.
* [ ] Any “compat alias” comments or alias types in `analytics/graph_rows/graph_metrics.py` are removed, and only canonical row types remain.

After this, analytics has:

* A single runtime model (ExecutionContext + providers + plugin runtime).
* No references to the old `AnalyticsContext` object or its attribute names.
* No deprecated adapter structures (`DeleteOptions.params`) or “legacy counter name” shims.

If you’d like, the next cleanup we can design at this level of detail is for any remaining storage/metadata deprecations (e.g., old columns in `metadata_bootstrap`) or the small “legacy fallback” doc wording in `ingestion/change_tracker.py`.
