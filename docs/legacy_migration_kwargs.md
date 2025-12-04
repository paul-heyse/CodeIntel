The goal here is: **no more `from_kwargs` anywhere in analytics or graphs; all callers construct options explicitly and only in the go‑forward shape.** 

I’ll give you a concrete plan you can follow:

* First, quickly inventory the `from_kwargs` helpers.
* Then, handle **analytics** and **graphs** separately.
* For each, I’ll show “before” → “after” code sketches and what tests to fix.

---

## 0. Quick inventory pass

You already saw there are a handful of `from_kwargs` helpers with comments like “Build options from legacy kwargs” in:

* `analytics/core/registry.py`
* `analytics/graphs/runtime.py` (call sites)
* `graphs/core/*` (e.g. `protocol.py` / `registry.py`)

The fastest way to scope exactly what you have is:

```bash
rg "from_kwargs" src/analytics src/graphs
```

You should see something like:

* `analytics/core/registry.py: GraphRuntimeOptions.from_kwargs`
* maybe `analytics/core/registry.py: PluginRegistrationOptions.from_kwargs`
* `analytics/graphs/runtime.py` calling those
* `graphs/core/registry.py` or `graphs/core/protocol.py` with a similar helper (e.g. `GraphEngineOptions.from_kwargs` or `GraphPluginOptions.from_kwargs`) and its call sites

Everything below assumes that general shape; adjust names to what you actually see in your tree.

---

## 1. Analytics: kill `from_kwargs` and switch to explicit options

### 1.1. `GraphRuntimeOptions.from_kwargs` (analytics)

**Today (conceptually)**

In `analytics/core/registry.py` you have something like a `@dataclass` plus a compat constructor:

```python
@dataclass(frozen=True)
class GraphRuntimeOptions:
    snapshot: SnapshotRef
    graph_backend: GraphBackendConfig
    feature_flags: GraphFeatureFlags

    @classmethod
    def from_kwargs(cls, *, snapshot: SnapshotRef, **kwargs) -> "GraphRuntimeOptions":
        """Build options from legacy kwargs."""
        # accept multiple legacy names for the backend
        backend = (
            kwargs.pop("graph_backend_config", None)
            or kwargs.pop("graph_backend", None)
            or kwargs.pop("backend", None)
            or GraphBackendConfig()
        )
        feature_flags = kwargs.pop("feature_flags", None) or GraphFeatureFlags()

        if kwargs:
            raise TypeError(f"Unexpected kwargs for GraphRuntimeOptions: {sorted(kwargs)}")

        return cls(
            snapshot=snapshot,
            graph_backend=backend,
            feature_flags=feature_flags,
        )
```

Call sites (e.g. in `analytics/graphs/runtime.py`) likely look like:

```python
options = GraphRuntimeOptions.from_kwargs(
    snapshot=snapshot,
    graph_backend_config=graph_backend_config,
    feature_flags=feature_flags,
)
```

**Target**

* **Only** the dataclass constructor; no `from_kwargs`.
* A single canonical name for each field (e.g. `graph_backend` and `feature_flags`).

**Steps**

1. **Choose canonical field names**
   Use whatever you have in the dataclass fields already; e.g.:

   * `graph_backend: GraphBackendConfig`
   * `feature_flags: GraphFeatureFlags`

2. **Update all call sites**

   Search:

   ```bash
   rg "GraphRuntimeOptions\.from_kwargs" src/analytics src/graphs
   ```

   For each call:

   ```python
   # BEFORE
   options = GraphRuntimeOptions.from_kwargs(
       snapshot=snapshot,
       graph_backend_config=graph_backend_config,
       feature_flags=feature_flags,
   )
   ```

   change to:

   ```python
   # AFTER
   options = GraphRuntimeOptions(
       snapshot=snapshot,
       graph_backend=graph_backend_config,   # adjust name
       feature_flags=feature_flags,
   )
   ```

   If any callers pass additional “legacy” kwargs (e.g. `backend=...` or `graph_backend=...`), normalize them to the canonical argument at the call site.

3. **Delete `from_kwargs`**

   In `analytics/core/registry.py`:

   * Remove the entire `@classmethod def from_kwargs(...)` block.
   * Remove any comments/docstrings that talk about “legacy kwargs” for `GraphRuntimeOptions`.

4. **Run type checker / tests for analytics**

   * mypy/pyright should now fail if any stray `.from_kwargs(` exists.
   * Fix those by switching to the dataclass constructor as above.

---

### 1.2. Any other `from_kwargs` in `analytics/core/registry.py`

You may also have something like `PluginRegistrationOptions.from_kwargs` or similar.

Pattern is the same:

**Example “before”**

```python
@dataclass(frozen=True)
class PluginRegistrationOptions:
    name: str
    enabled: bool
    tags: frozenset[str]

    @classmethod
    def from_kwargs(cls, **kwargs) -> "PluginRegistrationOptions":
        """Build options from legacy kwargs."""
        name = kwargs.pop("name")
        enabled = kwargs.pop("enabled", True)
        tags = frozenset(kwargs.pop("tags", []))

        # maybe tolerate legacy names like "plugin_name" or "is_enabled"
        if "plugin_name" in kwargs:
            name = kwargs.pop("plugin_name")
        if "is_enabled" in kwargs:
            enabled = kwargs.pop("is_enabled")

        if kwargs:
            raise TypeError(...)

        return cls(name=name, enabled=enabled, tags=tags)
```

**After**

1. Replace call sites:

   ```python
   # BEFORE
   opts = PluginRegistrationOptions.from_kwargs(
       plugin_name="coverage_functions",
       is_enabled=True,
       tags={"coverage", "functions"},
   )

   # AFTER
   opts = PluginRegistrationOptions(
       name="coverage_functions",
       enabled=True,
       tags=frozenset({"coverage", "functions"}),
   )
   ```

   Normalize all legacy field names (`plugin_name`, `is_enabled`, etc.) at the call site.

2. Delete the `from_kwargs` method and any “legacy kwargs” comments.

3. Fix failing tests/type errors.

---

## 2. Graphs: remove graph `from_kwargs` helpers

The graphs side has a parallel story: `from_kwargs` used to smooth migration from an older graph runtime API.

You saw a comment like “Build options from legacy kwargs” in `graphs/core/protocol.py` or `graphs/core/registry.py`. The pattern will look very similar to analytics.

### 2.1. Inventory graph `from_kwargs`

Run:

```bash
rg "from_kwargs" src/graphs
```

You’ll likely see something like:

* `graphs/core/registry.py: GraphEngineOptions.from_kwargs`
* maybe `graphs/core/registry.py: GraphRuntimeConfig.from_kwargs`
* (and the call sites)

Open each definition. They’ll generally:

* Take `**kwargs`
* Accept multiple possible names for the same thing (e.g. `backend`, `nx_backend`, `graph_backend`)
* Normalise them into one dataclass.

### 2.2. Example refactor: `GraphEngineOptions.from_kwargs`

**Before (conceptual)**

```python
@dataclass(frozen=True)
class GraphEngineOptions:
    snapshot: SnapshotRef
    backend: GraphBackendConfig
    feature_flags: GraphFeatureFlags

    @classmethod
    def from_kwargs(cls, *, snapshot: SnapshotRef, **kwargs) -> "GraphEngineOptions":
        """Build options from legacy kwargs (for old engine callers)."""
        backend = (
            kwargs.pop("backend", None)
            or kwargs.pop("graph_backend", None)
            or kwargs.pop("graph_backend_config", None)
            or GraphBackendConfig()
        )
        feature_flags = kwargs.pop("feature_flags", None) or GraphFeatureFlags()
        if kwargs:
            raise TypeError(...)
        return cls(snapshot=snapshot, backend=backend, feature_flags=feature_flags)
```

Call site (e.g. in `graphs/runtime/manifest.py` or `graphs/runtime/executor.py`):

```python
engine_options = GraphEngineOptions.from_kwargs(
    snapshot=snapshot,
    graph_backend_config=graph_backend_config,
    feature_flags=feature_flags,
)
```

**After**

1. Update call sites to use constructor:

   ```python
   engine_options = GraphEngineOptions(
       snapshot=snapshot,
       backend=graph_backend_config,      # or whatever your field name is
       feature_flags=feature_flags,
   )
   ```

2. Delete `from_kwargs` and its docstring.

3. If any caller was passing `backend=...` or `graph_backend=...`, pick one canonical name and adjust all callers to match the dataclass field name.

4. Fix any failing tests / typing.

### 2.3. Example refactor: plugin/meta options

You may also have something like a `GraphPluginCallOptions.from_kwargs(plugin, **kwargs)` used to wrap a plugin and some options.

Same pattern:

**Before**

```python
@dataclass(frozen=True)
class GraphPluginCallOptions:
    plugin: GraphPlugin
    snapshot: SnapshotRef
    backend: GraphBackendConfig
    feature_flags: GraphFeatureFlags

    @classmethod
    def from_kwargs(
        cls,
        plugin: GraphPlugin,
        *,
        snapshot: SnapshotRef,
        **kwargs,
    ) -> "GraphPluginCallOptions":
        """Build options from legacy kwargs."""
        backend = kwargs.pop("backend", None) or GraphBackendConfig()
        feature_flags = kwargs.pop("feature_flags", None) or GraphFeatureFlags()
        ...
        return cls(
            plugin=plugin,
            snapshot=snapshot,
            backend=backend,
            feature_flags=feature_flags,
        )
```

**After**

* Callers become:

  ```python
  opts = GraphPluginCallOptions(
      plugin=plugin,
      snapshot=snapshot,
      backend=graph_backend_config,
      feature_flags=feature_flags,
  )
  ```

* Remove the compat constructor.

---

## 3. Clean up references to “legacy kwargs” in docs/comments

Once the code is updated:

1. **Search for “legacy kwargs” comments**

   ```bash
   rg "legacy kwargs" src/analytics src/graphs
   ```

   Or more broadly:

   ```bash
   rg "legacy" src/analytics src/graphs
   ```

2. In `analytics/core/registry.py` and `graphs/core/*`, remove or update any comments that still talk about:

   * “Build options from legacy kwargs”
   * “for backwards compatibility with old APIs”

   Replace with something like:

   > Options object for configuring graph runtime. Call via the dataclass constructor.

Now the code and comments agree: there is **no dynamic legacy kwargs pathway**.

---

## 4. Tests and usage changes you’ll need to make

Anywhere tests or helpers were using `from_kwargs`, update them the same way as runtime calls:

1. **Search test tree** for `.from_kwargs(`

   ```bash
   rg "from_kwargs" tests
   ```

2. For each, switch to explicit construction:

   ```python
   # BEFORE
   opts = GraphRuntimeOptions.from_kwargs(
       snapshot=make_snapshot(),
       graph_backend_config=cfg,
       feature_flags=GraphFeatureFlags(...),
   )

   # AFTER
   opts = GraphRuntimeOptions(
       snapshot=make_snapshot(),
       graph_backend=cfg,
       feature_flags=GraphFeatureFlags(...),
   )
   ```

   Adjust names to the dataclass fields.

3. If tests explicitly assert that “legacy names are accepted” (e.g., a test that passes `backend=` instead of `graph_backend=`), those tests are no longer valid and should be deleted or rewritten to assert the **explicit API** only.

---

## 5. Final “no compat” sanity check

You’ll know you’ve fully migrated when:

* [ ] `rg "from_kwargs" src/analytics src/graphs tests` returns **no results**.
* [ ] `rg "legacy kwargs" src/analytics src/graphs` returns **no results**.
* [ ] All option/config classes in analytics and graphs are only constructed via their dataclass `__init__`, not via helpers.
* [ ] Type checker and tests pass.

At that point:

* The **only** way to construct analytics/graph runtime & plugin options is via the explicit, go‑forward dataclass APIs.
* There is **no leftover compatibility code** smoothing over old keyword shapes.

If you’d like, next we can do the same kind of aggressive “no compat left” plan for the **CLI argparse helpers** or any other legacy cluster you’re ready to remove.
