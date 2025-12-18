

# 4) UI / telemetry SDK mechanics

## 4.0 Mental model

**Apache Hamilton UI = a self-hostable “observability + catalog + versioning” service** that becomes useful only when you attach a **tracking adapter** to your Driver. The adapter streams:

* **DAG structure + code provenance** (so the UI can version/browse the dataflow),
* **run telemetry + node-level timing/error attribution** (so you can compare runs and debug),
* **artifact metadata + data summaries/statistics** (so the catalog is searchable and executions are inspectable). ([Hamilton][1])

The UI itself explicitly targets four “first-class” surfaces: **run telemetry**, a **feature/artifact catalog**, a **DAG visualizer for lineage**, and a **project explorer with dataflow versioning**. ([Hamilton][1])

---

## 4.1 Running the UI: local vs deployed (and what changes)

### 4.1.1 Local mode (SQLite, single-machine)

Local mode is the fast way to get value during dev: install the extras and run the CLI:

```bash
pip install "sf-hamilton[ui,sdk]"
hamilton ui
# python -m hamilton.cli.__main__ ui  # on windows
```

This launches a browser at `localhost:8241`. The docs explicitly note local mode can handle small workflows, but recommend Postgres (deployed mode) for “full scalability and a multi-read/write db.” ([Hamilton Apache Incubator][2])

### 4.1.2 Docker/deployed mode (frontend + backend + Postgres)

Deployed mode is a docker-compose stack that starts **UI frontend**, **backend server**, and **Postgres**. Setup is:

```bash
git clone https://github.com/apache/hamilton
cd hamilton/ui
./run.sh
# UI available at http://localhost:8242
```

Docs call out:

* “Invalid HTTP_HOST” → set `HAMILTON_ALLOWED_HOSTS="*"` (or comma-separated hosts) for the backend container. ([Hamilton Apache Incubator][2])
* Building images locally: increase Docker memory to **10GB+**, then `./dev.sh --build` (or `./dev.sh` to mount local code). ([Hamilton Apache Incubator][2])
* Self-hosting behind a subpath: set `REACT_APP_HAMILTON_SUB_PATH=/hamilton` (must begin with `/`). ([Hamilton Apache Incubator][2])
* There’s also explicit mention you can run the UI on **Snowflake Container Services** (docs link out to a guide/example). ([Hamilton Apache Incubator][2])

### 4.1.3 Ports and why they matter to the tracker

The tracker defaults to sending telemetry to `localhost:8241/8242` and you can override both via `hamilton_api_url` and `hamilton_ui_url`. The docs explicitly note “if using docker the UI is on 8242.” ([Hamilton][3])

---

## 4.2 How “projects” and “DAG versions” are formed

### 4.2.1 Open-source UI flow (HamiltonTracker)

The UI workflow is:

1. start UI,
2. create/select a project in the UI,
3. attach `HamiltonTracker(project_id=..., username=..., dag_name=..., tags=...)` to your Driver,
4. run your DAG; logs include links back to the UI. ([Hamilton Apache Incubator][2])

### 4.2.2 Hosted UI flow (DAGWorksTracker) — same mental model, different auth

DAGWorks’ hosted docs make the “versioning semantics” explicit:

* On **Driver instantiation**, a new project “version” is created if code changed or you provide a new `dag_name`, and the DAG structure is saved.
* When the run is complete, run metadata and summaries are logged and appear in the UI. ([DAGWorks Documentation][4])

That’s useful even for self-hosted/open-source planning because it tells you the *intended boundary*: **driver build == “version check”**, **driver execute == “run record.”** ([DAGWorks Documentation][4])

---

## 4.3 What the UI is designed to show (feature-by-feature, with the mechanics behind each)

### 4.3.1 Telemetry: run history + comparisons + run data

The UI advertises “run tracking + telemetry” including history, comparisons, and “data for specific runs.” ([Hamilton][5])
Hamilton’s UI announcement blog further frames this as execution observability including profiling/traces and errors pinpointed at the function level. ([blog.dagworks.io][6])
The async-focused blog post calls out a **waterfall chart of node performance** and result inspection in the UI once you add the right adapter. ([blog.dagworks.io][7])

**Mechanics (design boundary):**

* You don’t “instrument nodes” manually; you attach a lifecycle adapter via `Builder.with_adapters(tracker)` and execute normally. ([Hamilton][3])
* Node attribution depends on stable node naming; intermediate nodes (pipes/extracts) will also appear unless you control naming/namespacing (Topic #1).

### 4.3.2 Feature / artifact catalog

The UI overview explicitly includes an “assets/features catalog” to view “functions, nodes, and assets across a history of runs.” ([Hamilton][5])
The UI announcement blog describes this as a combined transform + artifact catalog that auto-populates from runs and includes artifact metadata captured through “approved methods” (i.e., the Hamilton I/O/materialization patterns). ([blog.dagworks.io][6])

**Mechanics (how you make catalog entries “real”):**

* Prefer explicit loaders/savers/materializers (so artifacts are first-class, not incidental side effects).
* Use tags consistently so the catalog can be filtered by environment/team/version (see §4.5).

### 4.3.3 DAG visualizer / lineage and the code browser

The UI overview calls out:

* a DAG visualizer “for exploring and looking at your code, and determining lineage,”
* and a “browser” view to inspect dataflow structure and code. ([Hamilton][1])

**Mechanics:**

* The tracker must upload DAG structure + code context; that’s why “versioning” is tied to driver instantiation (you want the UI to see the graph before execution, not just the run after). ([DAGWorks Documentation][4])

### 4.3.4 Dataflow versioning

The UI overview includes “dataflow versioning” (select/compare versions). ([Hamilton][5])
Practically, version boundaries come from **(code changes OR `dag_name` changes)**, which is explicitly how the hosted tracker behaves; you should treat `dag_name` as a deliberate “semantic version key,” not a label. ([DAGWorks Documentation][4])

---

## 4.4 SDK wiring: HamiltonTracker (self-hosted UI)

### 4.4.1 Minimal integration (one adapter)

```python
from hamilton import driver
from hamilton_sdk import adapters

tracker = adapters.HamiltonTracker(
    project_id=PROJECT_ID,
    username="YOU@COMPANY.COM",          # same identity you entered in the UI
    dag_name="codeintel_semantic_v1",    # treat as “version key”
    tags={"environment": "DEV", "team": "CodeIntel", "version": "v1"},
)

dr = (
    driver.Builder()
    .with_modules(*your_modules)
    .with_config(your_config)
    .with_adapters(tracker)
    .build()
)

dr.execute([...])
```

This is the canonical pattern in the UI docs. ([Hamilton][3])

### 4.4.2 Pointing the tracker at a remote/self-hosted instance

```python
tracker = adapters.HamiltonTracker(
    project_id=PROJECT_ID,
    username="YOU@COMPANY.COM",
    dag_name="codeintel_semantic_v1",
    tags={"environment": "PROD", "team": "CodeIntel"},
    hamilton_api_url="http://YOUR_DOMAIN:8241",
    hamilton_ui_url="http://YOUR_DOMAIN:8242",  # docker UI port
)
```

The docs state you can override `hamilton_api_url` / `hamilton_ui_url`, and that defaults are localhost `8241/8242`. ([Hamilton][3])

---

## 4.5 Capture control: performance + data governance knobs (what to tune and how)

Hamilton documents a small but important set of capture controls in `hamilton_sdk.tracking.constants`, with three configuration planes:

* module defaults,
* config file (default `~/.hamilton.conf`),
* environment variables (prefixed with `HAMILTON_`),
* direct constant assignment (highest precedence). ([Hamilton][3])

### 4.5.1 The documented constants

* `CAPTURE_DATA_STATISTICS` (default True): whether to capture data insights/statistics
* `MAX_LIST_LENGTH_CAPTURE` (default 50): truncation for list capture
* `MAX_DICT_LENGTH_CAPTURE` (default 100): truncation for dict capture
* `DEFAULT_CONFIG_URI` (default `~/.hamilton.conf`) ([Hamilton][3])

### 4.5.2 Recommended “safe-by-default” profile (my strong suggestion)

For CodeIntel-style workloads (where objects can contain code, paths, secrets, etc.), I’d treat “data capture” as an explicit opt-in:

* Set `CAPTURE_DATA_STATISTICS=False` in production by environment variable.
* Keep list/dict max sizes small globally; allow local overrides in dev.

```bash
export HAMILTON_CAPTURE_DATA_STATISTICS=0
export HAMILTON_MAX_LIST_LENGTH_CAPTURE=20
export HAMILTON_MAX_DICT_LENGTH_CAPTURE=50
```

You can also do the same via `~/.hamilton.conf` under `[SDK_CONSTANTS]`. ([Hamilton][3])

*(This recommendation goes beyond the docs: the docs tell you how to configure; the “safe-by-default” policy is an operational stance that tends to prevent accidental leakage.)*

---

## 4.6 Async services: AsyncDriver + AsyncHamiltonTracker (FastAPI-grade integration)

Hamilton’s AsyncDriver docs include an example of using an async tracker adapter:

```python
from hamilton import async_driver
from hamilton_sdk import adapters

tracker_async = adapters.AsyncHamiltonTracker(
    project_id=1,
    username="elijah",
    dag_name="async_tracker",
)

dr = (
    await async_driver.Builder()
    .with_modules(async_module)
    .with_adapters(tracker_async)
    .build()
)
```

This is the documented “async builder + async tracker” pattern. ([Hamilton][8])

**Operational note:** the async builder is described as “more limited” (e.g., doesn’t support some features “for now”), so keep your service integration slim: one driver per worker, stable config surface, and avoid mixing in unsupported execution modes. ([Hamilton][8])

---

## 4.7 Hosted DAGWorks tracker: mechanics that matter even if you self-host

If you use DAGWorks hosted:

* You attach `adapters.DAGWorksTracker(project_id, api_key, username, dag_name, tags)` via `Builder.with_adapters(...)`. ([DAGWorks Documentation][9])
* Their docs call out that run tracking can have marginal overhead for *very short* executions due to connecting to the server, but for longer executions the impact “disappears.” ([DAGWorks Documentation][4])

Even if you’re self-hosting the open-source UI, that’s a good heuristic: **don’t over-telemetry micro-runs**; batch or raise the granularity where possible.

---

## 4.8 Holistic “best in class” example: production service + self-hosted UI + deterministic versioning + governance

This shows: docker-deployed UI, tracker endpoint override, strong `dag_name` semantics, tags aligned to your semantic layer, and capture governance.

### 4.8.1 Infra (UI stack)

* Deploy UI via docker-compose; fix host validation with `HAMILTON_ALLOWED_HOSTS`; set subpath if you reverse proxy. ([Hamilton Apache Incubator][2])

### 4.8.2 App (Driver factory)

```python
from __future__ import annotations

import os
from hamilton import driver
from hamilton_sdk import adapters

def build_codeintel_driver(*, modules: list[object], cfg: dict) -> driver.Driver:
    tracker = adapters.HamiltonTracker(
        project_id=int(os.environ["HAMILTON_PROJECT_ID"]),
        username=os.environ["HAMILTON_USERNAME"],
        # Make DAG versioning intentional:
        # - semantic layer version
        # - repo+commit fingerprint (or release)
        dag_name=f"codeintel::semantic_v1::{cfg['repo']}::{cfg['commit']}",
        tags={
            "environment": os.environ.get("APP_ENV", "prod"),
            "team": "CodeIntel",
            "repo": cfg["repo"],
            "commit": cfg["commit"],
            "semantic_version": "v1",
        },
        hamilton_api_url=os.environ.get("HAMILTON_API_URL", "http://ui-backend:8241"),
        hamilton_ui_url=os.environ.get("HAMILTON_UI_URL", "http://ui-frontend:8242"),
    )

    return (
        driver.Builder()
        .with_modules(*modules)
        .with_config(cfg)          # build-time config selection
        .with_adapters(tracker)    # telemetry + UI integration
        .build()
    )
```

### 4.8.3 Governance (capture tuning)

Use env vars / config file to clamp capture volume and disable stats in prod. The SDK supports env-var overrides with `HAMILTON_` prefix and documents precedence. ([Hamilton][3])

---

If you want, I can now “close the loop” by writing the **UI-facing semantic conventions** as a concrete spec (exact tag keys + required values + naming rules + a registry compiler), so the Hamilton UI’s catalog/versioning becomes the authoritative UI for your semantic layer — not just a nice-to-have telemetry screen.

[1]: https://hamilton.apache.org/hamilton-ui/ui/?utm_source=chatgpt.com "UI Overview - Hamilton"
[2]: https://hamilton.incubator.apache.org/hamilton-ui/ui/ "UI Overview - Hamilton"
[3]: https://hamilton.apache.org/concepts/ui/ "UI Overview - Hamilton"
[4]: https://docs.dagworks.io/capabilities?utm_source=chatgpt.com "Hosted Hamilton UI on DAGWorks Overview - DAGWorks Documentation"
[5]: https://hamilton.apache.org/hamilton-ui/ui/ "UI Overview - Hamilton"
[6]: https://blog.dagworks.io/p/hamilton-ui-streamlining-metadata?utm_source=chatgpt.com "Hamilton UI: Streamlining Metadata, Tracking, Lineage/Tracing, and Observability for Your Dataflows"
[7]: https://blog.dagworks.io/p/async-dataflows-in-hamilton?utm_source=chatgpt.com "Async Dataflows in Hamilton"
[8]: https://hamilton.dagworks.io/en/latest/reference/drivers/AsyncDriver/?utm_source=chatgpt.com "AsyncDriver - Hamilton"
[9]: https://docs.dagworks.io/adapter?utm_source=chatgpt.com "DAGWorks Tracking Adapter - DAGWorks Documentation"



# UI-facing semantic conventions spec for Hamilton

Goal: make **Hamilton UI + tracker tags** the authoritative “catalog + versioned contract surface” for your semantic layer, and make it **machine-readable** (registry compiler → serving tools, CI snapshots, docs).

This spec defines:

* exact **tag keys + required values**
* **naming + versioning rules** (node names vs semantic IDs vs dag_name)
* a **registry compiler** (Driver → `semantic_registry.json`) with validation
* optional **data validation hooks** (CI) so the UI/catalog cannot drift silently

---

## 1) Core invariants

### 1.1 Identity model

* **`semantic_id` is the stable public identifier** (what agents/clients use).
* **Hamilton node name** (function name) is implementation detail and may change.
* UI “DAG versions” are driven by **code changes** + your chosen **`dag_name`** conventions. (Treat `dag_name` as a stable *product key*.)

### 1.2 Minimal discoverability contract

Every public semantic output must satisfy:

* `layer == "semantic"`
* `semantic_id` exists
* `kind` exists (table/scalar/artifact)
* `entity` + `grain` exist

This is what enables both:

* UI filtering (semantic-only views), and
* registry compilation via `Driver.list_available_variables(tag_filter=...)`.

---

## 2) Tag taxonomy

### 2.1 Required tags for **public semantic outputs** (`layer="semantic"`)

These must exist on every semantic output node.

| Tag key       | Type |              Required | Example                               | Meaning                                    |
| ------------- | ---: | --------------------: | ------------------------------------- | ------------------------------------------ |
| `layer`       |  str |                     ✅ | `"semantic"`                          | Public semantic surface                    |
| `semantic_id` |  str |                     ✅ | `"function.risk_score.v1"`            | Stable API identifier                      |
| `kind`        |  str |                     ✅ | `"table"` / `"scalar"` / `"artifact"` | Output class                               |
| `entity`      |  str |                     ✅ | `"function"` / `"module"`             | Entity type                                |
| `grain`       |  str |                     ✅ | `"per_function"` / `"per_module"`     | Row-level grain                            |
| `version`     |  str |                     ✅ | `"1"`                                 | Semantic schema version (not code version) |
| `schema_ref`  |  str |  ✅ for `kind="table"` | `"semantic.function_risk_v1"`         | Logical schema name                        |
| `entity_keys` |  str |  ✅ for `kind="table"` | `"repo,commit,goid_h128"`             | Keys uniquely identifying entity rows      |
| `join_keys`   |  str |  ✅ for `kind="table"` | `"repo,commit,goid_h128"`             | Keys to join across semantic tables        |
| `dtype`       |  str | ✅ for `kind!="table"` | `"float64"` / `"str"`                 | Scalar/series dtype                        |
| `stability`   |  str |                     ✅ | `"experimental"` / `"stable"`         | Client-facing stability                    |

### 2.2 Strongly recommended tags (all semantic outputs)

These are not strictly required, but they make the UI genuinely useful.

| Tag key             | Example                                      | Why                         |
| ------------------- | -------------------------------------------- | --------------------------- |
| `owner`             | `"codeintel"` / `"platform"`                 | Ownership routing + triage  |
| `description`       | `"Risk score (v1) derived from LOC + cyclo"` | UI catalog readability      |
| `unit`              | `"score"` / `"ms"` / `"count"`               | Prevents metric misuse      |
| `pii`               | `"none"` / `"possible"` / `"yes"`            | Governance / capture policy |
| `source_system`     | `"duckdb"` / `"ibis"` / `"pandas"`           | Interpretation + debugging  |
| `materialization`   | `"duckdb_view"` / `"parquet"`                | How users should consume it |
| `materialized_name` | `"semantic.v_function_risk_v1"`              | Stable external object name |

### 2.3 Optional layer tags (non-public nodes)

Use these to keep the UI navigable without polluting the semantic registry.

* `layer="raw"`: ingestion / source-of-truth nodes
* `layer="intermediate"`: internal transforms/features
* `layer="docs"`: doc/export views intended for humans, not agents

The registry compiler in §4 defaults to `layer="semantic"` only.

---

## 3) Naming & versioning rules

### 3.1 Node name rules

* Node names can change; **semantic_id cannot** (treat as API).
* Never embed semantic versioning into node names; keep it in `semantic_id` and `version`.

### 3.2 Stable public object naming (DuckDB / warehouse)

If you materialize:

* always produce a stable external name (e.g., `semantic.v_function_risk_v1`)
* store it in tags:

  * `materialization="duckdb_view"`
  * `materialized_name="semantic.v_function_risk_v1"`

### 3.3 `dag_name` rules (UI version stream)

Treat `dag_name` as a **product key**, not an execution instance.

Recommended:

* `dag_name = "codeintel::semantic_v1"` (stable)
* per-run metadata goes in tracker tags:

  * `repo`, `commit`, `run_kind` (`full|incremental|ci`), `dataset_snapshot_id`, etc.

Avoid including commit in `dag_name` unless you explicitly want “one UI version per commit.”

### 3.4 Intermediate nodes: namespace aggressively

Everything produced by pipes/extracts/subdags should be namespaced so UI browsing is clean:

* intermediate nodes → `namespace="prep"` / `"feat"` / `"int"` etc.
* semantic outputs should live at top-level (no namespace) unless you intentionally group by domain.

---

## 4) Registry compiler

### 4.1 Registry schema (what you write to `semantic_registry.json`)

One row per semantic output (keyed by `semantic_id`).

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

@dataclass(frozen=True)
class SemanticEntry:
    semantic_id: str
    node_name: str               # Hamilton execution key
    kind: str                    # table|scalar|artifact
    entity: str
    grain: str
    version: str
    schema_ref: str | None
    entity_keys: tuple[str, ...]
    join_keys: tuple[str, ...]
    dtype: str | None
    stability: str
    materialization: str | None
    materialized_name: str | None
    tags: Mapping[str, Any]
    python_type: str | None      # for debugging (stringified)
```

### 4.2 Compiler + validator (drop-in)

```python
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

REQUIRED_BASE = {"layer", "semantic_id", "kind", "entity", "grain", "version", "stability"}

def _split_keys(v: Any) -> tuple[str, ...]:
    if v is None:
        return tuple()
    if isinstance(v, (list, tuple)):
        return tuple(str(x).strip() for x in v if str(x).strip())
    s = str(v)
    return tuple(x.strip() for x in s.split(",") if x.strip())

def compile_semantic_registry(dr, *, layer: str = "semantic") -> dict[str, SemanticEntry]:
    """
    Pulls semantic outputs from Hamilton metadata (tags) and returns semantic_id -> SemanticEntry.

    Contract: every semantic output must have layer=semantic and semantic_id tag.
    """
    nodes = dr.list_available_variables(tag_filter={"layer": layer, "semantic_id": None})
    out: dict[str, SemanticEntry] = {}

    for n in nodes:
        tags = dict(n.tags)
        missing = REQUIRED_BASE - tags.keys()
        if missing:
            raise ValueError(f"Node {n.name} missing required tags: {sorted(missing)}")

        kind = str(tags["kind"])
        schema_ref = tags.get("schema_ref")
        entity_keys = _split_keys(tags.get("entity_keys"))
        join_keys = _split_keys(tags.get("join_keys"))
        dtype = tags.get("dtype")

        if kind == "table":
            for k in ("schema_ref", "entity_keys", "join_keys"):
                if not tags.get(k):
                    raise ValueError(f"Semantic table node {n.name} missing tag: {k}")
            if not entity_keys or not join_keys:
                raise ValueError(f"Semantic table node {n.name} must declare non-empty entity_keys/join_keys")
        else:
            if not dtype:
                raise ValueError(f"Non-table semantic node {n.name} must declare dtype")

        semantic_id = str(tags["semantic_id"])
        if semantic_id in out:
            raise ValueError(f"Duplicate semantic_id={semantic_id} on nodes {out[semantic_id].node_name} and {n.name}")

        out[semantic_id] = SemanticEntry(
            semantic_id=semantic_id,
            node_name=n.name,
            kind=kind,
            entity=str(tags["entity"]),
            grain=str(tags["grain"]),
            version=str(tags["version"]),
            schema_ref=str(schema_ref) if schema_ref else None,
            entity_keys=entity_keys,
            join_keys=join_keys,
            dtype=str(dtype) if dtype else None,
            stability=str(tags["stability"]),
            materialization=str(tags.get("materialization")) if tags.get("materialization") else None,
            materialized_name=str(tags.get("materialized_name")) if tags.get("materialized_name") else None,
            tags=tags,
            python_type=str(getattr(n, "type", None)) if getattr(n, "type", None) is not None else None,
        )

    return out

def write_semantic_registry(path: str, registry: dict[str, SemanticEntry]) -> None:
    payload = {k: asdict(v) for k, v in registry.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, sort_keys=True, indent=2, default=str)
```

### 4.3 CI snapshot contract

In CI, you want:

* `semantic_registry.json` diff reviewable
* failure if:

  * duplicate `semantic_id`
  * missing required tags
  * missing `entity_keys/join_keys` for tables
  * missing `dtype` for scalars/artifacts

Optionally, in CI only, also execute each semantic table on a small fixture dataset and verify:

* declared `entity_keys` columns exist
* declared `join_keys` columns exist
* if you tag a `schema_ref`, verify output columns match the schema definition you keep elsewhere (Pandera/Ibis schema, etc.)

---

## 5) Tagging helpers (to prevent typos)

### 5.1 Semantic output decorator factory

```python
from __future__ import annotations

from hamilton.function_modifiers import tag

def semantic_output(
    *,
    semantic_id: str,
    kind: str,
    entity: str,
    grain: str,
    version: str,
    stability: str = "experimental",
    schema_ref: str | None = None,
    entity_keys: str | None = None,
    join_keys: str | None = None,
    dtype: str | None = None,
    materialization: str | None = None,
    materialized_name: str | None = None,
    owner: str = "codeintel",
    description: str | None = None,
    pii: str = "none",
):
    tags = {
        "layer": "semantic",
        "semantic_id": semantic_id,
        "kind": kind,
        "entity": entity,
        "grain": grain,
        "version": version,
        "stability": stability,
        "owner": owner,
        "pii": pii,
    }
    if description:
        tags["description"] = description
    if schema_ref:
        tags["schema_ref"] = schema_ref
    if entity_keys:
        tags["entity_keys"] = entity_keys
    if join_keys:
        tags["join_keys"] = join_keys
    if dtype:
        tags["dtype"] = dtype
    if materialization:
        tags["materialization"] = materialization
    if materialized_name:
        tags["materialized_name"] = materialized_name
    return tag(**tags)
```

---

## 6) End-to-end example (what “closed loop” looks like)

### 6.1 Define a semantic table node

```python
from __future__ import annotations

import pandas as pd

@semantic_output(
    semantic_id="function.risk_score.v1",
    kind="table",
    entity="function",
    grain="per_function",
    version="1",
    stability="stable",
    schema_ref="semantic.function_risk_v1",
    entity_keys="repo,commit,goid_h128",
    join_keys="repo,commit,goid_h128",
    materialization="duckdb_view",
    materialized_name="semantic.v_function_risk_v1",
    description="Function risk score v1 (LOC + cyclo + callgraph signals)",
)
def semantic_function_risk_v1(function_facts: pd.DataFrame, call_graph_edges: pd.DataFrame) -> pd.DataFrame:
    ...
```

### 6.2 Build driver with tracker tags aligned to the same semantic vocabulary

```python
from hamilton import driver
from hamilton_sdk import adapters

tracker = adapters.HamiltonTracker(
    project_id=PROJECT_ID,
    username=USER,
    dag_name="codeintel::semantic_v1",
    tags={
        "environment": ENV,
        "team": "CodeIntel",
        "semantic_version": "v1",
        "repo": repo,
        "commit": commit,
        "run_kind": run_kind,
    },
    hamilton_api_url=API_URL,
    hamilton_ui_url=UI_URL,
)

dr = (
    driver.Builder()
    .with_modules(...)
    .with_config(...)
    .with_adapters(tracker)
    .build()
)
```

### 6.3 Compile registry from the DAG (single source of truth)

```python
reg = compile_semantic_registry(dr)
write_semantic_registry("semantic_registry.json", reg)
```

### 6.4 Make UI authoritative

Operationally, you now have:

* UI shows *exactly* the semantic outputs (filter `layer=semantic`)
* `semantic_registry.json` is derived from the same tags the UI uses
* serving layer can read `semantic_registry.json` and map `semantic_id -> node_name -> execute/materialize`

---

## 7) Recommended “definition of done” checks

1. **Registry snapshot** checked into repo (or generated in CI and diffed).
2. **No duplicate semantic IDs** (compiler enforces).
3. **Required tags** enforced (compiler enforces).
4. **Key columns exist** in CI fixture execution for table outputs.
5. **UI tags** (tracker tags + node tags) share common vocabulary (`environment`, `repo`, `commit`, `semantic_version`) so UI filters naturally match your system concepts.

---

If you share your current tag usage (or a handful of representative semantic nodes), I can adapt this spec into a repo-ready `docs/semantic_conventions.md` + a `src/codeintel/serving/semantic_registry_compiler.py` that matches your existing entity keys (`repo/commit/module/goid_h128/urn`) and your dataset contract naming.
