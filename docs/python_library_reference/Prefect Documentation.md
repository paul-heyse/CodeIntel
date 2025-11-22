Below is a **comprehensive, implementation‑first reference to the Prefect Python library (focused on Prefect 3.x)**, written so an AI agent (or a human) can design, ship, and operate **best‑in‑class Prefect workflows**. It covers the mental model, APIs, orchestration features, infrastructure patterns, CI/CD, reliability tactics, observability, and gotchas—*with exact knobs and code you can paste into real projects*.

> **Version note.** This guide targets **Prefect 3.x** (GA since Sep 2024) and draws from the current docs, API reference, and maintainer guidance as of **Oct 25, 2025**. Where behavior differs from Prefect 2.x, this guide reflects v3 semantics (transactions-based caching, events/automations, workers & work pools, etc.). ([Prefect][1])

---

## 0) Prefect in one paragraph (mental model)

**Prefect turns Python functions into observable, orchestrated workflows**:

* A **`@flow`**–decorated function is a workflow entrypoint. **`@task`**–decorated functions are concurrent, cacheable, retryable units inside flows. Flows call tasks (and other flows) and return results just like normal Python. ([Prefect][2])
* Each call creates a **run** that goes through **states** (Scheduled → Running → Completed/Failed/…); you get **futures** to wait for or fetch results. ([Prefect][3])
* To run remotely / on a schedule you create a **Deployment** (server‑side representation of your flow), attach it to a **Work Pool**, and execute via a **Worker** (process that provisions infrastructure). ([Prefect][4])
* **Transactions & caching** give durable, idempotent execution; **Automations & events** let you trigger, route, or notify on anything that happens. ([Prefect][5])

---

## 1) Quick start (modern, v3‑style)

```python
from typing import Iterable
from pydantic import BaseModel
from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash
from prefect.futures import wait

class Params(BaseModel):
    urls: list[str]
    dedupe: bool = True

@task(retries=3, retry_delay_seconds=[1, 5, 15], cache_key_fn=task_input_hash)
def fetch(url: str) -> bytes:
    import httpx
    return httpx.get(url, timeout=10).content

@task
def parse(blob: bytes) -> dict:
    # parse to dict...
    return {"size": len(blob)}

@flow
def crawl(params: Params) -> list[dict]:
    logger = get_run_logger()
    futures = [parse.submit(fetch.submit(u)) for u in params.urls]
    # ensure all terminal futures finish before we return
    wait(futures)                       # or [f.result() for f in futures]
    results = [f.result() for f in futures]
    logger.info("parsed %d items", len(results))
    return results
```

* **Why this is “right” in v3**

  * Flows validate parameters with Pydantic; tasks can **retry** with per‑retry delays, **cache** via `task_input_hash`, and run **concurrently** via `.submit()` and `wait()`. ([Prefect][2])
  * Returning futures or waiting on them is required to resolve dependencies correctly with task runners. ([Prefect][6])

Run locally as plain Python (`crawl(Params(urls=[...]))`), or **deploy** to run via the Prefect API (see §6). ([Prefect][7])

---

## 2) Core objects & execution semantics

### 2.1 Flows

* Define with `@flow(...)`; call like a normal function or via deployment.
  Key args include `retries`, `retry_delay_seconds`, `timeout_seconds`, `persist_result`, `result_storage`, `result_serializer`, `validate_parameters`. ([Prefect][2])
* Supported function types: sync, async, instance/class/static methods, generators. ([Prefect][2])

### 2.2 Tasks

* Define with `@task(...)`. Important args:
  `retries`, `retry_delay_seconds` (single value, list, or callable for custom backoff), `timeout_seconds`, `cache_key_fn`, `cache_expiration`, `tags`, `persist_result`, `log_prints`. ([Prefect][8])
* **Concurrency** is per flow via **task runners** (ThreadPool default) or Dask/Ray runners (see §3). Use `.submit()` for parallelism and **futures** to gather results. ([Prefect][9])

### 2.3 Futures

* `PrefectFuture`: `.wait(timeout=...)`, `.result(timeout=..., raise_on_failure=True)`. Also top‑level `prefect.futures.wait(futures)` and `as_completed`‑style helpers in v3. ([Prefect][10])

### 2.4 States

* Rich state model for flow/task runs: `SCHEDULED`, `PENDING`, `RUNNING`, `COMPLETED`, `FAILED`, `CRASHED`, `CANCELLING`, `CANCELLED`, `PAUSED/SUSPENDED`, `LATE`, `RETRYING/...`. Use UI/CLI/API to inspect and act. ([Prefect][3])

---

## 3) Concurrency, task runners, and scaling patterns

* **Default**: thread‑pool runner; explicit parallelism via `task().submit(...)`. ([Prefect][9])
* **Dask**: `@flow(task_runner=DaskTaskRunner(...))` for multi‑process / distributed task execution (local cluster or external scheduler). ([prefecthq.github.io][11])
* **Ray**: `@flow(task_runner=RayTaskRunner(...))` for Ray clusters and resource‑based scheduling (e.g., GPUs). ([prefecthq.github.io][12])

**Global throttles / back‑pressure.**
Create **concurrency limits** for **tags**; tag tasks accordingly:

```bash
prefect concurrency-limit create api 8
```

```python
@task(tags=["api"])
def call_api(...): ...
```

Prefect enforces these limits across runs to protect upstream systems and maintain SLOs. ([Prefect][13])

---

## 4) Caching, results, and **transactions** (v3 superpower)

### 4.1 Durable results

Turn on persistence globally or per flow/task:

```bash
prefect config set PREFECT_RESULTS_PERSIST_BY_DEFAULT=true
```

```python
@flow(result_storage="s3/my-results")        # or pass a block instance
def my_flow(): ...
```

Use `prefect.filesystems.RemoteFileSystem` blocks for S3/GCS/Azure via `fsspec`. Ensure your **workers** can access the same result store. ([Prefect][14])

### 4.2 Caching policies & isolation

* `cache_key_fn=task_input_hash` caches by parameters; configure **isolation** (`READ_COMMITTED` or `SERIALIZABLE`) when concurrency could clash. ([Prefect][8])
* Advanced: customize cache policy with dedicated key storage & lock managers. ([Prefect][15])

### 4.3 Transactions: atomic units with rollback & idempotency

* Group tasks under a single **transaction** context for all‑or‑nothing behavior, with `@task.on_rollback` / `@task.on_commit` hooks and **idempotent** keys.
* For strict concurrency, set isolation to `SERIALIZABLE` and add a `LockManager` (e.g., filesystem or Redis). ([Prefect][5])

> **Why it matters**: Transactions unify **caching**, **idempotency**, and **side‑effect safety**. Use them around “write” phases (e.g., loading a warehouse) to guarantee exactly‑once semantics even under retries or multi‑run races. ([Prefect][5])

---

## 5) Reliability controls (retries, timeouts, cancellation, pauses)

* **Retries** on tasks/flows with constant, per‑attempt, or programmatic delays. Example: `retries=5, retry_delay_seconds=[1, 2, 4, 8, 16]`. ([Prefect][8])
* **Timeouts**: `@flow(timeout_seconds=...)` / `@task(timeout_seconds=...)` to fail stuck work. ([Prefect][7])
* **Cancellation**: cancel runs from UI/CLI (`prefect flow-run cancel <RUN_ID>`). Workers propagate cancellation to underlying jobs. Nested (sub‑)flows are best cancelled by cancelling the parent or orchestrating via separate deployments. ([Prefect][16])
* **Pausing / suspending** for human‑in‑the‑loop or external approvals; `suspend_flow_run()`/`pause` from Python or UI (requires deployment & persisted results). ([Prefect][17])

---

## 6) Deployments, scheduling, and ad‑hoc runs

**What a deployment gives you**: remote triggers & scheduling, versioned metadata, placement into work pools, parameters, and automations/observability hooks. ([Prefect][4])

### 6.1 Create & serve

* **Python**: `flow.deploy(...)` to register; `flow.serve(...)` to register **and** run a long‑lived process that keeps polling (great for local Process worker setups). ([Prefect][18])
* **CLI**: `prefect deploy` from a `prefect.yaml` (supports build/push/pull steps, code storage & environment prep). ([reference.prefect.io][19])

### 6.2 Schedules & triggers

* Cron, **interval**, or **RRULE**; add one or many schedules on a deployment. ([GitHub][20])
* **Ad‑hoc runs** (on demand): trigger via UI/CLI/Python SDK or REST (`/api/flow_runs` with idempotency keys). ([Prefect][21])

---

## 7) Workers & Work Pools (infrastructure abstraction)

* **Work pool** = where & how to run (process, Docker, Kubernetes, and serverless types for ECS, Azure ACI, and Cloud Run). **Worker** = client process that polls a pool and launches runs on that infra. ([Prefect][22])
* Start a worker:

  ```bash
  prefect work-pool create "k8s-pool" --type kubernetes
  prefect worker start -p "k8s-pool"
  ```

  Pass queues, healthchecks, run‑once, etc. via flags. ([Prefect][23])
* Configure job variables (image, env, resources, volume mounts, etc.) at the pool or per‑deployment. ([Prefect][24])
* **Custom workers**: build your own when you need bespoke provisioning. ([Prefect][25])

> **Tip**: For distributed compute needs, combine **Kubernetes worker** + Dask/Ray **task runner** to mix elastic orchestration with parallel execution. ([Prefect][26])

---

## 8) Configuration, secrets, blocks, and variables

* **Settings & profiles** (`prefect.toml`/`pyproject.toml`, `.env`, env vars `PREFECT_*`) with clear precedence; manage via CLI (`prefect config set`, `prefect profile use/inspect`). ([Prefect][27])
* **Blocks**: typed, reusable config/credentials/infrastructure resources (e.g., `prefect.filesystems.RemoteFileSystem`, `prefect.blocks.system.Secret`, plus cloud collections like **prefect‑aws**, **prefect‑gcp**). Register and load by slug. ([reference.prefect.io][28])
* **Variables**: non‑secret configs you reference at runtime (set in UI or code). ([GitHub][29])

---

## 9) Observability & ops

* **Logs**: per run, structured; use `get_run_logger()`; redirect `print` with `log_prints=True`. Customize logging behavior via settings. ([Prefect][30])
* **Artifacts**: create rich outputs in the UI (`create_markdown_artifact`, `create_table_artifact`, `create_link_artifact`, etc.) for dashboards, metrics, and audit trails. ([Prefect][31])
* **Events**: every state change or external webhook becomes an **event**; use **Automations** to trigger actions (run a deployment, notify Slack/email, call webhooks) on event patterns and filters. Manage via UI/CLI/REST. ([Prefect][32])

---

## 10) Event‑driven orchestration (Automations & Webhooks)

* Define automations with **triggers** (event presence/absence, sequences, compound logic) and **actions** (run a deployment, send email/Slack, HTTP request).
  CLI example:

  ```bash
  prefect automation create -f automation.yaml
  ```

  API: `POST /api/automations` with trigger schema. ([Prefect][33])
* **Webhooks** (Cloud): receive external events; template payloads into Prefect events and pass data to flows via templates. ([Prefect][34])
* **Chain deployments with events** (downstream runs when upstream completes). ([Prefect][35])

---

## 11) Python & REST clients (programmatic control)

* Use **`get_client()`** (async or sync) to read/update runs, schedule work, and query the platform; or install the slim **`prefect-client`** package for minimal environments. ([Prefect][36])
* **REST**: `/api/flow_runs` to create runs (supports **idempotency keys** and placement into work pools/queues). Useful for gateways/bridges. ([Prefect][37])

---

## 12) Testing strategies

* Use `prefect.testing.utilities.prefect_test_harness()` to spin a temporary local API/DB and run flows/tasks in isolation for unit tests. Pytest fixtures exist for hosted API if you need a subprocess server. ([Prefect][38])
* Mock `prefect.runtime.*` and loggers; validate artifact creation, event emission, and retry paths deterministically. ([Prefect][39])

---

## 13) Ecosystem & integrations

* Install **collections** for cloud infra/services: `prefect-aws` (ECS, S3, SQS, etc.), `prefect-gcp` (GCS, GKE), `prefect-kubernetes`, `prefect-dbt`, `prefect-snowflake`, many others—each brings blocks, tasks, and deploy steps. ([GitHub][40])
* Task runners and storage blocks integrate cleanly with these collections for hybrid/cloud execution.

---

## 14) Cloud vs self‑hosted

* **Prefect Cloud**: managed control plane with SSO/RBAC, audit logs, IP allowlists, webhooks, and enterprise governance. **Self‑hosted Server**: OSS API/UI you run yourself. Workflows and workers look the same in both. ([Prefect][41])

---

## 15) Best‑practice patterns (with code)

### 15.1 Fan‑out/fan‑in with global back‑pressure & durable results

```python
from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash
from prefect.futures import wait

@task(retries=4, retry_delay_seconds=[1,5,10,30],
      cache_key_fn=task_input_hash, tags=["api"])
def fetch_row(row_id: int) -> dict: ...

@task
def write_batch(rows: list[dict]) -> None: ...

@flow(result_storage="s3/warehouse-results", persist_result=True)
def etl(row_ids: list[int], batch_size: int = 100):
    logger = get_run_logger()
    futs = [fetch_row.submit(i) for i in row_ids]
    wait(futs)
    rows = [f.result() for f in futs]
    for i in range(0, len(rows), batch_size):
        write_batch.submit(rows[i:i+batch_size])
```

Create a concurrency limit for `"api"` and you have a **safe, idempotent** extractor that will retry, cache, and throttle across the fleet. ([Prefect][42])

### 15.2 Exactly‑once “commit” with **transactions** (guard side effects)

```python
from prefect import flow, task
from prefect.transactions import transaction, IsolationLevel

@task
def stage_to_tmp(dataset: str) -> str: ...
@task
def commit(dataset: str) -> None: ...
@stage_to_tmp.on_rollback
def cleanup(_txn): ...  # drop temp table

@flow
def load_dataset(dataset: str):
    with transaction(key=f"load:{dataset}",
                     isolation_level=IsolationLevel.SERIALIZABLE):
        tmp = stage_to_tmp(dataset)
        commit(tmp)
```

This ensures only one concurrent “load:dataset” wins and any staged side effects roll back on failure. ([Prefect][5])

### 15.3 Event‑driven DAG of deployments

* Upstream completes → event trigger → run downstream with templated parameters.
  Maintain loose coupling and independent retries/SLOs per deployment. ([Prefect][35])

### 15.4 Human‑in‑the‑loop approvals

* Persist results, **suspend** inside the flow, resume from UI or API after review. ([Prefect][17])

---

## 16) CLI cheat‑sheet (most used)

```bash
# Profiles & settings
prefect profile inspect            # show active settings
prefect config set KEY=VALUE       # e.g., PREFECT_API_URL=http://127.0.0.1:4200/api

# Deployments
prefect init                       # create prefect.yaml
prefect deploy                     # build/register from prefect.yaml
prefect flow serve path/to.py:my_flow --name my-deployment

# Work pools & workers
prefect work-pool create "k8s" --type kubernetes
prefect worker start -p "k8s"

# Concurrency
prefect concurrency-limit create api 8
prefect concurrency-limit ls

# Automations
prefect automation create -f automation.yaml
prefect automation ls

# Flow runs
prefect flow-run ls
prefect flow-run cancel <RUN_ID>
```

([Prefect][43])

---

## 17) Integration with your architecture

The patterns above map cleanly onto multi‑service or platform pipelines. For example, **workers bound to Docker/Kubernetes pools**, **transactions around mutating tasks**, and **automations** for cross‑system triggers are directly applicable to the high‑level design in your shared architecture note (e.g., flow entrypoints as service boundaries, events for inter‑service choreography, and blocks/variables for environment‑specific wiring). 

---

## 18) Advanced knobs & reference

* **Flow & task arguments (complete list)**: see Python API refs for `@flow`/`@task` to set retries, delays (list/callable), timeouts, persistence, serializers, in‑memory cache behavior, descriptions, tags, versions. ([Prefect][44])
* **Task runners**: default ThreadPool; switch to Dask or Ray for heavy parallelism. Configure clusters, addresses, resource hints (Ray). ([prefecthq.github.io][11])
* **Schedules**: cron / interval / rrule on deployments; time zones supported. ([GitHub][20])
* **Settings reference** (all `PREFECT_*`): paths, logging, retries defaults, results defaults, etc. ([Prefect][45])
* **Work pool types** (process, docker, kubernetes, serverless Cloud Run/ECS/ACI), job variables schema, and health checks. ([Prefect][4])
* **Artifacts API**: `create_markdown_artifact`, `create_table_artifact`, `create_link_artifact`, etc. ([Prefect][46])
* **Pause/suspend** & **cancellation** APIs and CLI. ([Prefect][17])
* **REST API**: official endpoints (Cloud and server share the same REST surface). ([Prefect][47])
* **Minimal client installs**: `prefect-client` for lambdas or slim tooling. ([PyPI][48])

---

## 19) Common pitfalls & how to avoid them

* **Forgetting to resolve terminal futures** before a flow returns → silent drops of pending work. *Always* `wait()` or return the futures so Prefect resolves dependencies. ([Prefect][6])
* **Relying on in‑memory results only** in distributed runs → downstream tasks can’t see prior outputs. Configure a shared `result_storage` block. ([Prefect][14])
* **Race conditions on side effects** (e.g., S3 writes) under retries → wrap in `transaction(key=..., isolation=SERIALIZABLE)` with a lock manager. ([Prefect][5])
* **Over‑mapping** tasks (fine‑grained submits) without global throttles → API rate‑limits. Add **concurrency limits** on a tag. ([Prefect][13])
* **Cancelling subflows only** → cancellation propagates from parent; orchestrate sub‑workflows as separate deployments if you need independent cancels. ([Prefect][2])

---

## 20) Full, production‑ready example

**Goal**: nightly event‑driven ELT on Kubernetes, Dask parallelism, durable results, back‑pressure, rollback safety, with Slack notification on failure.

```python
from typing import Iterable
from pydantic import BaseModel
from prefect import flow, task, get_run_logger
from prefect.tasks import task_input_hash
from prefect.transactions import transaction, IsolationLevel
from prefect.futures import wait
from prefect.artifacts import create_markdown_artifact

class LoadParams(BaseModel):
    date: str                  # e.g., "2025-10-24"
    sources: list[str]
    warehouse_table: str

@task(retries=3, retry_delay_seconds=[10,30,60],
      cache_key_fn=task_input_hash, tags=["api"], timeout_seconds=120)
def extract(src: str, date: str) -> bytes: ...
@task
def transform(blob: bytes) -> dict: ...
@task
def stage(rows: list[dict]) -> str: ...   # returns staging table
@task
def commit(staging_table: str, dest: str) -> None: ...
@stage.on_rollback
def drop_staging(_txn): ...               # DDL to drop staging table

@flow(result_storage="s3/prefect-results", persist_result=True)
def nightly(params: LoadParams) -> int:
    logger = get_run_logger()
    futs = [transform.submit(extract.submit(s, params.date)) for s in params.sources]
    wait(futs)
    rows = [f.result() for f in futs]

    with transaction(key=f"elt:{params.warehouse_table}:{params.date}",
                     isolation_level=IsolationLevel.SERIALIZABLE):
        stg = stage(rows)
        commit(stg, params.warehouse_table)

    create_markdown_artifact(
        key=f"elt/{params.warehouse_table}/{params.date}",
        markdown=f"**Loaded** {len(rows)} rows into `{params.warehouse_table}`"
    )
    return len(rows)
```

* Deploy to a **Kubernetes work pool**; set `task_runner=DaskTaskRunner(...)` on the flow for parallel mapping if desired. Add a concurrency limit for `"api"` at 8 to protect upstreams. Then configure an **Automation** to Slack on `flow-run:Failed` for this deployment. ([Prefect][22])

---

## 21) Where to look things up fast (authoritative)

* **Concepts**: flows, tasks, task runners, states, deployments, work pools, workers. ([Prefect][2])
* **How‑to**: write & run workflows; run concurrently; retries; persist results; serve flows; run in processes/Docker/K8s. ([Prefect][7])
* **Advanced**: transactions; caching isolation; direct submission to dynamic infra; cancel workflows. ([Prefect][5])
* **API refs**: Python SDK (flows, tasks, futures, flow_runs), REST endpoints, CLI. ([Prefect][49])

---

### Appendix A — Scheduling & chaining deployments (YAML sketch)

```yaml
# automation.yaml
name: run-downstream-on-upstream-complete
description: Kick off downstream when upstream completes successfully
trigger:
  type: event
  match:
    prefect.resource.id: "prefect.flow-run.*"
    prefect.event.name: "prefect.flow-run.Completed"
  any: true
actions:
  - type: run-deployment
    deployment_id: "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    parameters:
      run_date: "{{ event.resource.flow_run.parameters.date }}"
```

Create with:

```bash
prefect automation create -f automation.yaml
```

([Prefect][35])

---

### Appendix B — Minimal client: trigger runs programmatically

```python
import asyncio
from prefect import get_client

async def trigger(name: str, params: dict):
    async with get_client() as client:
        # look up deployment by name / handle; then schedule a run
        dep = (await client.read_deployments(name_like=name))[0]
        fr = await client.create_flow_run_from_deployment(dep.id, parameters=params)
        return fr.id

asyncio.run(trigger("nightly", {"date":"2025-10-24","sources":["a","b"],"warehouse_table":"fact"}))
```

(Concept: async client with convenience methods to schedule runs.) ([Prefect][36])

---

## Final notes

* If you need to orchestrate **LLM/agent** workflows, Prefect 3.x powers ControlFlow—a higher‑level agentic framework—but all core orchestration in this guide still applies. ([Prefect][50])
* This reference intentionally emphasizes the **v3** featureset (transactions, events/automations, modern deployments & workers). If you encounter v2 tutorials, prefer the v3 docs linked throughout.

If you’d like, tell me your stack (cloud, scheduler, data stores), and I’ll produce a **tailored deployment & automation blueprint** with exact pool types, blocks, and CI steps for that environment.

[1]: https://docs.prefect.io/v3/get-started/whats-new-prefect-3?utm_source=chatgpt.com "What's new in Prefect 3.0"
[2]: https://docs.prefect.io/v3/concepts/flows?utm_source=chatgpt.com "Flows"
[3]: https://docs.prefect.io/v3/concepts/states?utm_source=chatgpt.com "States"
[4]: https://docs.prefect.io/v3/concepts/deployments?utm_source=chatgpt.com "Deployments"
[5]: https://docs.prefect.io/v3/advanced/transactions "How to write transactional workflows - Prefect"
[6]: https://docs.prefect.io/v3/concepts/task-runners?utm_source=chatgpt.com "Task runners"
[7]: https://docs.prefect.io/v3/how-to-guides/workflows/write-and-run?utm_source=chatgpt.com "How to write and run a workflow"
[8]: https://docs.prefect.io/v3/api-ref/python/prefect-tasks?utm_source=chatgpt.com "tasks"
[9]: https://docs.prefect.io/v3/how-to-guides/workflows/run-work-concurrently?utm_source=chatgpt.com "How to run work concurrently"
[10]: https://docs.prefect.io/v3/api-ref/python/prefect-futures?utm_source=chatgpt.com "futures"
[11]: https://prefecthq.github.io/prefect-dask/task_runners/ "Task Runners - prefect-dask"
[12]: https://prefecthq.github.io/prefect-ray/ "prefect-ray"
[13]: https://docs.prefect.io/v3/concepts/tag-based-concurrency-limits?utm_source=chatgpt.com "Tag-based concurrency limits"
[14]: https://docs.prefect.io/v3/advanced/results?utm_source=chatgpt.com "How to persist workflow results"
[15]: https://docs.prefect.io/v3/api-ref/python/prefect-cache_policies?utm_source=chatgpt.com "cache_policies"
[16]: https://docs.prefect.io/v3/api-ref/python/prefect-cli-flow_run?utm_source=chatgpt.com "flow_run"
[17]: https://docs.prefect.io/v3/api-ref/python/prefect-flow_runs?utm_source=chatgpt.com "prefect.flow_runs"
[18]: https://docs.prefect.io/v3/how-to-guides/deployments/deploy-via-python?utm_source=chatgpt.com "How to deploy flows with Python"
[19]: https://reference.prefect.io/prefect/cli/deploy/?utm_source=chatgpt.com "prefect.cli.deploy"
[20]: https://github.com/PrefectHQ/prefect/issues/9823?utm_source=chatgpt.com "Prompt for deployment schedule in `prefect ..."
[21]: https://docs.prefect.io/v3/how-to-guides/deployments/run-deployments?utm_source=chatgpt.com "Trigger ad-hoc deployment runs"
[22]: https://docs.prefect.io/v3/concepts/work-pools?utm_source=chatgpt.com "Work pools"
[23]: https://docs.prefect.io/v3/api-ref/cli/work-pool?utm_source=chatgpt.com "prefect work-pool create"
[24]: https://docs.prefect.io/v3/how-to-guides/deployments/customize-job-variables?utm_source=chatgpt.com "How to override job configuration for specific deployments"
[25]: https://docs.prefect.io/v3/advanced/developing-a-custom-worker?utm_source=chatgpt.com "How to develop a custom worker"
[26]: https://docs.prefect.io/integrations/prefect-kubernetes?utm_source=chatgpt.com "prefect-kubernetes"
[27]: https://docs.prefect.io/v3/concepts/settings-and-profiles?utm_source=chatgpt.com "Settings and profiles"
[28]: https://reference.prefect.io/prefect/filesystems/?utm_source=chatgpt.com "prefect.filesystems"
[29]: https://github.com/PrefectHQ/prefect/discussions/5142?utm_source=chatgpt.com "Structured Logging with Datadog #5142 - PrefectHQ prefect"
[30]: https://docs.prefect.io/v3/api-ref/python/prefect-logging-loggers?utm_source=chatgpt.com "loggers"
[31]: https://docs.prefect.io/v3/concepts/artifacts?utm_source=chatgpt.com "Artifacts"
[32]: https://docs.prefect.io/v3/concepts/events?utm_source=chatgpt.com "Events"
[33]: https://docs.prefect.io/v3/how-to-guides/automations/creating-automations?utm_source=chatgpt.com "How to create automations"
[34]: https://docs.prefect.io/v3/concepts/webhooks?utm_source=chatgpt.com "Webhooks"
[35]: https://docs.prefect.io/v3/how-to-guides/automations/chaining-deployments-with-events?utm_source=chatgpt.com "How to chain deployments with events"
[36]: https://docs.prefect.io/v3/advanced/api-client?utm_source=chatgpt.com "How to use and configure the API client"
[37]: https://docs.prefect.io/v3/api-ref/rest-api/server/flow-runs/create-flow-run?utm_source=chatgpt.com "Create Flow Run"
[38]: https://docs.prefect.io/v3/how-to-guides/workflows/test-workflows?utm_source=chatgpt.com "How to test workflows"
[39]: https://docs.prefect.io/v3/api-ref/python/prefect-runtime-flow_run?utm_source=chatgpt.com "flow_run"
[40]: https://github.com/PrefectHQ/prefect-collection-registry?utm_source=chatgpt.com "PrefectHQ/prefect-collection-registry: Source of truth for ..."
[41]: https://www.prefect.io/cloud?utm_source=chatgpt.com "Open Source Pipeline Creation with Prefect Cloud"
[42]: https://docs.prefect.io/v3/api-ref/cli/concurrency-limit?utm_source=chatgpt.com "prefect concurrency-limit"
[43]: https://docs.prefect.io/v3/how-to-guides/configuration/manage-settings?utm_source=chatgpt.com "How to manage settings"
[44]: https://docs.prefect.io/v3/api-ref/python/prefect-flows?utm_source=chatgpt.com "flows"
[45]: https://docs.prefect.io/v3/api-ref/settings-ref?utm_source=chatgpt.com "Settings reference"
[46]: https://docs.prefect.io/v3/how-to-guides/workflows/artifacts?utm_source=chatgpt.com "How to produce workflow artifacts"
[47]: https://docs.prefect.io/v3/api-ref/rest-api/index?utm_source=chatgpt.com "REST API overview"
[48]: https://pypi.org/project/prefect-client/?utm_source=chatgpt.com "prefect-client"
[49]: https://docs.prefect.io/v3/api-ref?utm_source=chatgpt.com "API & SDK References"
[50]: https://www.prefect.io/blog/introducing-prefect-3-0?utm_source=chatgpt.com "Introducing Prefect 3.0"
