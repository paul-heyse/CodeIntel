They’re not just three nice-but-separate refactors — they’re three **orthogonal axes** of the same thing:

* **What** a plugin is and needs → `CorePluginMetadata` 
* **How** it is configured for this run → `ConfigSource` / `ProfiledConfigSource` + `PluginOptionsResolver` 
* **What actually happened** when it ran → `PluginExecutionRecord` + hash helpers + `ManifestStore` 

That’s basically already a “transition data manifold”; we just haven’t drawn the connecting line all the way through yet.

Below I’ll show:

1. How they fit together conceptually.
2. Concrete integration points that *intentionally* use all three together (so you’re not doing three parallel projects).
3. A small “unified plugin‑run helper” you can actually add now that makes the unified, policy‑driven future mostly a wiring problem.

---

## 1. The conceptual manifold: one “plugin run” = metadata + options + record

For a *single plugin run* (say `analytics.function_metrics` on repo X, commit Y, `fast` profile), you now have three rich views:

1. **Plugin contract** – `CorePluginMetadata`

   * name, domain, kind, stage
   * `provides` / `requires`, `produces_tables`, `options_model` 

2. **Plugin configuration** – via `ConfigSource` / `ProfiledConfigSource` + `PluginOptionsResolver`

   * given `meta.name` and `meta.options_model`, you get a typed options object, merged from:

     * base config,
     * profile (`fast` vs `full`),
     * CLI overrides 

3. **Plugin execution state/history** – `PluginExecutionRecord`

   * identity + `variant` (profile),
   * `options_hash`, `input_hash`,
   * `upstream_state` (capability → hash of provider’s inputs),
   * `row_counts`, status, timings 

If you squint, that’s already a unified, policy‑driven object:

> “For plugin P in profile `fast`, with upstream state U, here is exactly what we asked it to do (options + metadata) and what happened when we did.”

What’s missing is just a bit of **glue code** that treats this combination as a first‑class thing instead of three separate concerns.

---

## 2. Where they naturally integrate (and how to exploit it)

### 2.1 Use metadata `requires/provides` to drive upstream_state

Right now, when we build `upstream_state` for analytics, we hand‑picked providers (e.g. “goids comes from graphs.goid_builder”). 

You can instead:

* build a **capability index** once from all `CorePluginMetadata`:

  ```python
  # capability -> canonical plugin_name
  capability_providers = {
      cap: meta.name
      for meta in all_plugin_metadata
      for cap in meta.provides
  }
  ```
* then, for *any* plugin, you can derive upstream_state purely from metadata:

  ```python
  def resolve_upstream_state(
      meta: CorePluginMetadata,
      repo: str,
      commit: str,
      scope_id: str | None,
      variant: str | None,
      manifest_store: ManifestStore,
  ) -> dict[str, str]:
      state: dict[str, str] = {}

      for required_cap in meta.requires:
          provider_name = capability_providers.get(required_cap)
          if not provider_name:
              continue
          rec = manifest_store.load_last_record(
              plugin_name=provider_name,
              repo=repo,
              commit=commit,
              scope_id=scope_id,
              variant=variant,
          )
          if rec:
              state[required_cap] = rec.input_hash
      return state
  ```

Now `upstream_state` is:

* **derived from metadata**, not hardcoded, and
* identical for graphs and analytics.

That’s a big “data‑manifold” win: the same `requires`/`provides` fields that power planning and manifests also drive caching/skip.

---

### 2.2 Make profile/variant explicit and consistent

You already have:

* `ProfiledConfigSource` taking `profile` (“fast”, “full”, “ci”) and merging base/profile/CLI plugin options. 
* `PluginExecutionRecord.variant: str | None` designed to hold exactly that kind of info. 

Tie them together:

* In the build executor, when you construct `TargetExecutionContext`, also record the active profile as `ctx.variant` or pass it into the per‑plugin run helper.
* When building `PluginExecutionRecord`, always set `variant = active_profile` and include it in the `input_payload` for `compute_input_hash`.

That ensures:

* Changing profile from `fast` → `full` inherently changes `input_hash` even if options happen to be the same – i.e., profile is part of the **policy** that drives re‑run semantics.

---

### 2.3 Let options + metadata **fully define** the hash inputs

Your `compute_options_hash` & `compute_input_hash` already live in core and are designed for reuse. 

The natural next step is:

> “Never compute any skip hash outside this central helper; always derive it from (metadata + options + upstream_state + profile).”

Concretely:

```python
def build_input_signature(
    meta: CorePluginMetadata,
    repo: str,
    commit: str,
    scope_id: str | None,
    variant: str | None,
    options: object | None,
    upstream_state: Mapping[str, str],
) -> tuple[str | None, str]:
    options_hash = compute_options_hash(meta.name, options)
    payload = {
        "repo": repo,
        "commit": commit,
        "plugin_name": meta.name,
        "plugin_version": meta.version,
        "scope_id": scope_id,
        "variant": variant or "",
        "options_hash": options_hash or "",
        "upstream_state": dict(upstream_state),
    }
    input_hash = compute_input_hash(payload)
    return options_hash, input_hash
```

Every runtime (graphs, analytics, later ingestion) uses this same function. That means:

* A policy change → config change → options change → `options_hash` change → `input_hash` change → clear, uniform reason to re‑run.

---

## 3. One small “PluginRunContext” helper that ties it all together

Here’s the concrete thing you can add that turns these three independent changes into a **single integrated abstraction** — without yet building a full ExecutionEngine.

Add this to `core/plugins/execution/manifest.py` or a sibling module:

```python
from dataclasses import dataclass
from typing import Mapping

from codeintel.core.plugins.types.metadata import CorePluginMetadata
from codeintel.core.plugins.execution.options import PluginOptionsResolver
from codeintel.core.plugins.execution.manifest import (
    PluginExecutionRecord,
    PluginStatus,
    ManifestStore,
)
# plus build_input_signature + resolve_upstream_state helpers from above

@dataclass
class PluginRunContext:
    """All the data needed to run a single plugin invocation."""

    meta: CorePluginMetadata
    repo: str
    commit: str
    scope_id: str | None
    variant: str | None

    options: object | None
    options_hash: str | None
    upstream_state: dict[str, str]
    input_hash: str

def prepare_plugin_run(
    *,
    meta: CorePluginMetadata,
    repo: str,
    commit: str,
    scope_id: str | None,
    variant: str | None,
    config_source: ConfigSource,
    manifest_store: ManifestStore,
    options_resolver: PluginOptionsResolver | None = None,
) -> PluginRunContext:
    """Centralized preparation for a plugin run.

    - resolves options via metadata.options_model + ConfigSource
    - derives upstream_state via metadata.requires + ManifestStore
    - computes options_hash + input_hash in a standard way
    """
    resolver = options_resolver or PluginOptionsResolver(config_source=config_source)

    options = None
    if meta.options_model is not None:
        options = resolver.get_options(meta, meta.options_model)

    upstream_state = resolve_upstream_state(
        meta=meta,
        repo=repo,
        commit=commit,
        scope_id=scope_id,
        variant=variant,
        manifest_store=manifest_store,
    )

    options_hash, input_hash = build_input_signature(
        meta=meta,
        repo=repo,
        commit=commit,
        scope_id=scope_id,
        variant=variant,
        options=options,
        upstream_state=upstream_state,
    )

    return PluginRunContext(
        meta=meta,
        repo=repo,
        commit=commit,
        scope_id=scope_id,
        variant=variant,
        options=options,
        options_hash=options_hash,
        upstream_state=upstream_state,
        input_hash=input_hash,
    )
```

Then, both graphs and analytics runtimes can do:

```python
run_ctx = prepare_plugin_run(
    meta=plugin.metadata,
    repo=repo,
    commit=commit,
    scope_id=scope_id,
    variant=active_profile,
    config_source=ctx.config_source,
    manifest_store=manifest_store,
)

# (optional) skip check:
last = manifest_store.load_last_record(
    plugin_name=run_ctx.meta.name,
    repo=run_ctx.repo,
    commit=run_ctx.commit,
    scope_id=run_ctx.scope_id,
    variant=run_ctx.variant,
)
if last and last.input_hash == run_ctx.input_hash:
    # skip plugin; record a SKIPPED entry if you want
    ...

# otherwise, execute:
started_at = now()
result = await plugin.execute(ctx_with_options(run_ctx.options))
finished_at = now()

record = PluginExecutionRecord(
    plugin_name=run_ctx.meta.name,
    version=run_ctx.meta.version,
    repo=run_ctx.repo,
    commit=run_ctx.commit,
    scope_id=run_ctx.scope_id,
    variant=run_ctx.variant,
    status=PluginStatus.SUCCESS,
    input_hash=run_ctx.input_hash,
    options_hash=run_ctx.options_hash,
    row_counts=result.row_counts,
    upstream_state=run_ctx.upstream_state,
    started_at=started_at,
    finished_at=finished_at,
    extra={},
)

manifest_store.append_record(record)
```

That tiny helper is your **integration node**:

* It uses **metadata** (`requires`, `options_model`, version…). 
* It uses the **policy/config system** (`ConfigSource`, `ProfiledConfigSource`, variants). 
* It uses the **manifest & hash system** (`ManifestStore`, `compute_options_hash`, `compute_input_hash`). 

And you can adopt it incrementally:

* First in `callgraph` and `function_metrics`,
* Then in other graph/analytics plugins,
* Eventually across ingestion/build as well.

---

## 4. So: independent changes, or a unified data manifold?

Both:

* **Yes**, each change is independently useful:

  * metadata cleans up identity/contract,
  * options system cleans up config,
  * manifest/hashing cleans up observability + caching.

* **But** they’re clearly designed to **snap together**:

  * `CorePluginMetadata` → tells you **what** a plugin needs/produces and **how** it’s configured (`options_model`).
  * `ConfigSource` / profiles → decide **how** that plugin behaves under a given policy.
  * `PluginExecutionRecord` → records **exactly what happened** under that policy, and its hashes become the canonical “state signature”.

By adding just a little glue (capability‑based upstream resolution + shared input signature + the `prepare_plugin_run` helper), you get a **single, coherent “plugin run” object** that:

* is policy/profile aware,
* is metadata‑driven,
* has uniform skip/re‑run semantics,
* works for graphs and analytics now,
* and can extend to ingestion/build later.

That’s exactly the “transition data manifold” you’re describing — and you can get there *without* a big‑bang engine rewrite; it’s mostly about reusing the core helpers you already planned, in one central place.
