
# structlog deep dives (CodeIntel-oriented): power knobs + failure modes + minimal snippets

Below, each section A–M from the catalog is expanded into a “deep dive page” designed to be *directly actionable for an implementation agent* (clear levers, sharp edges, minimal patterns). All references are to structlog’s current stable docs (25.5.0). ([structlog][1])

---

## A) Core runtime model: bound logger + context + processors + wrapped logger

### Mental model (the invariants you build around)

* A **bound logger** is: *(context dict)* + *(processor chain)* + *(wrapped logger / sink)*. It manages context and runs processors; **it does not do I/O**—the wrapped logger does. ([structlog][2])
* A log call:

  1. copies context → event dict,
  2. adds call kwargs,
  3. adds `"event"` from first positional arg,
  4. runs processors in order,
  5. calls the wrapped logger method with the final processor output. ([structlog][2])
* Final processor return value contract:

  * can return `str|bytes|bytearray` (passed as single positional arg), or
  * can return `(args, kwargs)` tuple for `wrapped_logger.method(*args, **kwargs)`. ([structlog][2])

### Power knobs

* **Context scope:** use `.bind()` for local scope context; use contextvars (Section G) for “request/run global” context.
* **Processor ordering:** early processors should be cheap and schema-enforcing; last processor is a renderer (JSON/console/logfmt). ([structlog][3])
* **Wrapped logger choice:** Print vs Write vs Bytes affects atomicity and encoding strategy (Section C). ([structlog][4])

### Failure modes / gotchas

* Don’t confuse “context dict” with “contextvars global context”: they are merged only if you run `merge_contextvars` early. ([structlog][5])
* A processor can return “anything” only because the *next processor* is yours; structlog “only looks at the last processor return” (so type breaks are fine inside chain, but terminal return must match sink expectations). ([structlog][3])

### Minimal snippet (canonical shape)

```python
import structlog

# called at app init (see Section B for full config)
log = structlog.get_logger()

def f(repo_id: str) -> None:
    # localize + bind once (perf + stable context)
    l = log.bind(repo_id=repo_id)
    l.info("ingest_started", phase="scip")
```

### CodeIntel pattern (strongly recommended)

Treat each log event as a **structured row** with a stable schema:

* keys: `event`, `level`, `timestamp`, `logger`, plus your run identifiers (`run_id`, `repo_id`, `pipeline_step`, `target`, …).
* enforce with a custom processor early (e.g., ensure `event` is a `str`, coerce enums to strings, block huge blobs).

---

## B) Global configuration plane: `configure()` / `wrap_logger()` / defaults

### Mental model

* You call `structlog.configure(...)` once at process init; it sets global defaults used by `get_logger()` / `wrap_logger()` when you don’t override per-logger. ([structlog][6])
* `get_logger()` called in module scope returns a **lazy proxy** because it runs before you configure; the proxy materializes a correctly configured bound logger on first `bind()`/`new()` call. ([structlog][6])
* Config precedence is “most-specific wins”: args to `wrap_logger()` override configured defaults. ([structlog][7])

### Power knobs

* `processors`: your “schema + enrichment + rendering” pipeline.
* `wrapper_class`: choose `make_filtering_bound_logger(...)`, `stdlib.BoundLogger`, or custom wrapper. ([structlog][2])
* `logger_factory`: choose stdlib `LoggerFactory`, `WriteLoggerFactory`, `BytesLoggerFactory`, etc. ([structlog][6])
* `cache_logger_on_first_use`: speed vs reconfig/test flexibility and multiprocessing pickling constraints. ([structlog][8])

### Failure modes / gotchas

* **Never call `bind()` or `new()` in module/class scope** (before configuration): you’ll freeze in default config instead of your app config; use `get_logger(..., initial_values=...)` for pre-populated context. ([structlog][6])
* If `cache_logger_on_first_use=True`, later `configure()` won’t affect already cached loggers; and cached bound loggers are **not pickleable** (multiprocessing). ([structlog][8])

### Minimal snippet (CodeIntel “one function” bootstrap)

```python
import logging, os, structlog

def configure_structlog(*, mode: str) -> None:
    # "mode" could be: "dev"|"prod"|"test"
    level = logging.INFO if mode != "test" else logging.WARNING
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=(mode == "prod"),
        processors=[  # fill in (Section E)
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.PrintLoggerFactory(),
    )
```

---

## C) Wrapped logger sinks: Print/Write/Bytes (+ factories)

### Mental model

These are the “last hop” that actually emits output. Pick based on **atomicity** and **bytes-vs-str**.

### Power knobs

* `PrintLogger` / `PrintLoggerFactory`: convenience; uses `print(..., flush=True)`.
* `WriteLogger` / `WriteLoggerFactory`: uses `file.write(line + "\n")` **atomically** (important when mixing with stdlib logging handlers). ([structlog][4])
* `BytesLogger` / `BytesLoggerFactory`: emit bytes; pair with `JSONRenderer(serializer=orjson.dumps)` and avoid encode/decode ping-pong. ([structlog][8])

### Failure modes / gotchas

* If you output to the same stream as `logging.StreamHandler`, `PrintLogger` can interleave lines because `print()` writes message and newline separately; use `WriteLogger` instead. ([structlog][4])
* `BytesLoggerFactory` requires final processor output be bytes/bytearray; otherwise you’ll get type friction at the sink.

### Minimal snippet (production-fast bytes JSON)

```python
import logging, orjson, structlog

structlog.configure(
    cache_logger_on_first_use=True,
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.JSONRenderer(serializer=orjson.dumps),
    ],
    logger_factory=structlog.BytesLoggerFactory(),
)
```

This is essentially the “as fast as it gets” pattern described in the performance guide. ([structlog][8])

---

## D) Wrapper classes + filtering: `BoundLogger`, `make_filtering_bound_logger`, stdlib wrapper

### Mental model

* Filtering in structlog is best done **in the bound logger method** (pre-event-dict) for speed: methods below threshold become `return None`. ([structlog][2])

### Power knobs

* `structlog.make_filtering_bound_logger(min_level)`:

  * `min_level` uses stdlib numeric constants (`logging.INFO`, etc.), but structlog doesn’t depend on logging internals. ([structlog][2])
* `structlog.stdlib.BoundLogger`:

  * mirrors `logging.Logger` methods and has correct type hints. ([structlog][4])
  * async variants (`ainfo`, `aerror`, …) offload processing to a threadpool executor to avoid blocking event loop. ([structlog][4])

### Failure modes / gotchas

* If you depend on “debug logs for diagnosing tests,” remember filtering wrapper literally no-ops the method; you won’t see anything. ([structlog][2])
* If you use async threadpool logging (`ainfo`) heavily, you are paying extra CPU per entry; decide explicitly. ([structlog][8])

### Minimal snippet

```python
import logging, structlog

structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
)
log = structlog.get_logger()
log.debug("won't print")
log.info("will_print")
```

---

## E) Processor library: composing the event pipeline (enrichment → policy → renderer)

### Mental model

* Processor signature: `(logger, method_name, event_dict) -> next_value`. Each processor gets the previous return value; only the final return is passed to the sink. ([structlog][3])
* Final processor can return `dict|bytes|bytearray` in modern versions. ([structlog][3])

### CodeIntel “default order” (strong baseline)

**Recommended chain skeleton (prod JSON):**

1. `merge_contextvars` (global correlation) ([structlog][5])
2. *your schema/policy processors* (coercions, redactions, size caps)
3. `add_log_level` (uniform `level`) ([structlog][1])
4. `CallsiteParameterAdder` (optional; expensive) ([structlog][1])
5. exception handling: `format_exc_info` (flat) **or** `dict_tracebacks` (structured) ([structlog][9])
6. `TimeStamper(fmt="iso", utc=True)` ([structlog][8])
7. `JSONRenderer(serializer=orjson.dumps)` (terminal)

### Power knobs (high leverage, processor-by-processor)

**Renderers**

* `JSONRenderer(...)`: switch serializer to `orjson/msgspec/RapidJSON` for throughput. ([structlog][8])
* `KeyValueRenderer(sort_keys=..., key_order=..., drop_missing=..., repr_native_str=...)`: useful for CLI/dev; key ordering and missing-key policy matter for determinism. ([structlog][1])
* `LogfmtRenderer(..., bool_as_flag=True)`: logfmt; enforces key validity and has bool rendering policy. ([structlog][1])

**Schema manipulation**

* `EventRenamer(to=..., replace_by=...)`: use if you want `message` vs `event` or cross-platform consistency; keep it *right before renderer*. ([structlog][1])

**Exceptions**

* `ExceptionRenderer(...)` is the generic transformer; `format_exc_info` and `dict_tracebacks` are the common presets. ([structlog][9])
* `dict_tracebacks` produces JSON-friendly structured exception data (see Section F). ([structlog][10])

**Callsite**

* `CallsiteParameterAdder(parameters={...}, additional_ignores=[...])`: choose minimal parameters; note special handling for “foreign” `logging.LogRecord` events and efficiency guidance for `ProcessorFormatter` setups. ([structlog][1])

### Failure modes / gotchas

* `EventRenamer`: docs explicitly recommend placing it right before renderer because other processors may rely on `event`. ([structlog][1])
* `LogfmtRenderer`: raises `ValueError` on keys with non-printable/whitespace characters. ([structlog][1])
* If you want structured tracebacks, `dict_tracebacks` must run **before** JSON rendering; this is the documented pattern. ([structlog][1])

### Minimal snippet (a “policy + renderer” chain)

```python
import structlog

def drop_large_payloads(_, __, event_dict):
    # Example policy: prevent accidental huge blobs
    if "payload" in event_dict and len(str(event_dict["payload"])) > 10_000:
        event_dict["payload"] = "<omitted>"
    return event_dict

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        drop_large_payloads,
        structlog.processors.add_log_level,
        structlog.processors.dict_tracebacks,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.JSONRenderer(),
    ]
)
```

---

## F) Structured tracebacks as data: `structlog.tracebacks`

### Mental model

* `dict_tracebacks` is a convenience preset that replaces `exc_info` with a structured `exception` field suitable for JSON. ([structlog][10])
* Under the hood it uses `ExceptionDictTransformer`, which is built on `tracebacks.extract(...)`. ([structlog][11])

### Power knobs (what you control)

Use a custom `ExceptionDictTransformer(...)` when you need:

* `show_locals` on/off,
* `locals_max_length`, `locals_max_string` truncation,
* hiding `__dunder__` / `_sunder` locals,
* suppression lists, rich-based formatting choices. ([structlog][12])

### Failure modes / gotchas

* **Secrets in locals:** structured tracebacks can include locals; for production logs, default to `show_locals=False` unless you have a robust redaction story. (Transformer supports controls; see knobs above.) ([structlog][12])
* **Payload size:** structured exceptions can be large; consider truncation and/or dropping locals.

### Minimal snippet (custom structured tracebacks with tight locals policy)

```python
import structlog
from structlog.tracebacks import ExceptionDictTransformer
from structlog.processors import ExceptionRenderer

exc_as_dict = ExceptionRenderer(ExceptionDictTransformer(
    show_locals=False,
    locals_max_length=5,
    locals_max_string=120,
))

structlog.configure(processors=[exc_as_dict, structlog.processors.JSONRenderer()])
```

---

## G) Context propagation: `contextvars` (preferred) + `threadlocal` (legacy)

### G1) `structlog.contextvars` (what CodeIntel should default to)

**Mental model**

* Provides “global, but context-local” key/value storage that works with threads / asyncio / greenlets depending on execution context. ([structlog][5])
* Typical flow: `merge_contextvars` first processor; `clear_contextvars` at request/run start; `bind_contextvars` / `unbind_contextvars` for global correlation fields. ([structlog][5])

**Power knobs**

* Token-based restoration: `bind_contextvars()` returns tokens; `reset_contextvars(**tokens)` restores prior values (nestable overrides). ([structlog][5])

**Failure modes / gotchas (FastAPI/Starlette is the big one)**

* **Hybrid sync/async apps:** contextvar storage is isolated per concurrency method; in Starlette/FastAPI, values set in sync context may not appear in async logs and vice versa. You need to bind/clear in the same “side” where logs occur (usually async middleware for FastAPI). ([structlog][5])

**Minimal snippet (run-scoped correlation)**

```python
import uuid
from structlog.contextvars import clear_contextvars, bind_contextvars
import structlog

log = structlog.get_logger()

def start_run(repo_id: str) -> str:
    clear_contextvars()
    run_id = str(uuid.uuid4())
    bind_contextvars(run_id=run_id, repo_id=repo_id)
    log.info("run_started")
    return run_id
```

### G2) `structlog.threadlocal` (deprecated)

* Deprecated in favor of contextvars; kept as workaround in some environments; API maps 1:1 to contextvars equivalents. ([structlog][13])
* If you still use it: `merge_threadlocal` should be first processor; clear/bind at request boundaries. ([structlog][13])

---

## H) Standard library logging integration: `structlog.stdlib`

### Mental model (pick an integration tier)

structlog supports multiple “where does rendering happen?” options; the docs lay out:

* don’t integrate (leave third-party logs alone),
* render in structlog then pass strings to logging,
* render in logging using `render_to_log_kwargs`,
* **unify structlog + logging** with `ProcessorFormatter`. ([structlog][4])

### Power knobs

**1) Quick-start compatibility**

* `structlog.stdlib.recreate_defaults()` recreates defaults on top of stdlib logging, optionally configuring logging via `log_level`. ([structlog][4])

**2) `ProcessorFormatter` (the “one pipeline for everything” option)**

* On structlog side: processor chain must end with `ProcessorFormatter.wrap_for_formatter()`; don’t use `render_to_log_kwargs` in that setup. ([structlog][4])
* On logging side: configure `ProcessorFormatter(processors=..., foreign_pre_chain=...)`; use `remove_processors_meta` to drop `_record`/`_from_structlog`. ([structlog][4])

**3) Processor utilities**

* `filter_by_level` early-drop for stdlib thresholds. ([structlog][4])
* `add_logger_name`, `add_log_level`, `add_log_level_number`, `ExtraAdder`, `PositionalArgumentsFormatter`. ([structlog][4])

### Failure modes / gotchas

* Mixed output streams: if you use the same stream for structlog and `logging.StreamHandler`, prefer `WriteLogger` to avoid interleaving. ([structlog][4])
* With `ProcessorFormatter`, you must use `wrap_for_formatter` and must not use `render_to_log_kwargs`/`render_to_log_args_and_kwargs` in the structlog processor chain. ([structlog][4])

### Minimal snippet (ProcessorFormatter unification)

```python
import logging, structlog

shared = [
    structlog.stdlib.add_log_level,
    structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
]

structlog.configure(
    processors=shared + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

handler = logging.StreamHandler()
handler.setFormatter(structlog.stdlib.ProcessorFormatter(
    foreign_pre_chain=shared,
    processors=[
        structlog.stdlib.ProcessorFormatter.remove_processors_meta,
        structlog.dev.ConsoleRenderer(),
    ],
))
root = logging.getLogger()
root.addHandler(handler)
root.setLevel(logging.INFO)
```

This is the documented pattern (including `foreign_pre_chain` and `remove_processors_meta`). ([structlog][4])

---

## I) Twisted integration: `structlog.twisted`

### Mental model

* Twisted-specific bound logger has explicit API (`msg()`, `err()`) and is slightly faster / less magical than generic wrapper. ([structlog][14])
* Includes adapters/renderers for Twisted log observers (`EventAdapter`, Twisted JSON renderer helpers). ([structlog][14])

### Power knobs

* Use `structlog.twisted.LoggerFactory` if Twisted is your sink factory. ([structlog][6])

### Failure modes

* Calling non-existent methods is more obvious with explicit API (that’s a feature). ([structlog][14])

### Minimal snippet

```python
import structlog
structlog.configure(wrapper_class=structlog.twisted.BoundLogger)
log = structlog.get_logger()
log.msg("twisted_event", x=1)
```

---

## J) Development affordances: `structlog.dev` (ConsoleRenderer + pretty tracebacks)

### Mental model

* `ConsoleRenderer` is the dev-friendly renderer; if Rich or better-exceptions is installed it can pretty-print exceptions; but `format_exc_info()` must be absent for pretty exceptions. ([structlog][15])
* Console output is configurable post-instantiation; you can get the active renderer and mutate columns. ([structlog][15])

### Power knobs

* Column system: `columns=[Column(...), Column("", default_formatter)]` controls ordering, formatting, and can drop keys (formatter returns empty string). ([structlog][16])
* Env vars: `FORCE_COLOR`, `NO_COLOR` for output coloring decisions. ([structlog][15])

### Failure modes / gotchas

* If you leave `format_exc_info` in the chain, you won’t get Rich/better-exceptions pretty output. ([structlog][15])
* ConsoleRenderer is intentionally less strict about immutability (ergonomics); don’t treat it as a production sink. ([structlog][15])

### Minimal snippet (dev/prod split)

```python
import sys, structlog

shared = [structlog.contextvars.merge_contextvars, structlog.processors.add_log_level]

if sys.stderr.isatty():
    processors = shared + [structlog.dev.ConsoleRenderer()]
else:
    processors = shared + [structlog.processors.dict_tracebacks, structlog.processors.JSONRenderer()]

structlog.configure(processors=processors)
```

This “pretty in TTY, JSON in containers” pattern is straight from best practices. ([structlog][17])

---

## K) Testing utilities: capture + assertions without brittle string matching

### Mental model

* `capture_logs()` context manager captures structured entries; inside it, **all configured processors are disabled** unless you pass an explicit processors list. ([structlog][18])
* `capture_logs()` changes configuration, so it doesn’t affect already-cached loggers; recommended: don’t enable `cache_logger_on_first_use` during tests. ([structlog][18])

### Power knobs

* Capture after selected processors: `capture_logs(processors=[contextvars.merge_contextvars])`. ([structlog][18])
* Lower-level options:

  * `CapturingLoggerFactory` to inspect calls,
  * `ReturnLogger` for processor unit tests. ([structlog][18])

### Failure modes / gotchas

* If you rely on prod config caching, your tests may “mysteriously” not capture anything due to cached loggers; disable caching or reset defaults per test module. ([structlog][18])

### Minimal snippet (pytest-friendly capture)

```python
from structlog.testing import capture_logs
import structlog

def test_event_emission():
    with capture_logs() as logs:
        structlog.get_logger().info("hello", x=1)
    assert logs == [{"event": "hello", "x": 1, "log_level": "info"}]
```

([structlog][18])

---

## L) Typing surface: `structlog.typing` + “how to stay strict”

### Mental model

* `structlog.get_logger()` returns `Any` because the bound logger type is runtime-configured; the docs recommend `structlog.stdlib.get_logger()` for correct hints *or* explicitly annotate/cast. ([structlog][19])

### Power knobs

* In strict repos: define a “logger type alias” (e.g., stdlib wrapper) and enforce usage via imports.
* Use `BindableLogger` protocol when you only need `.bind()/unbind()/new()`—but it’s too weak to call actual log methods without a cast. ([structlog][19])

### Failure modes

* If you type everything as `BindableLogger`, you’ll end up casting everywhere; better: pick a concrete wrapper (often `structlog.stdlib.BoundLogger`) for app code. ([structlog][19])

### Minimal snippet (strict typing)

```python
import structlog
from structlog.stdlib import BoundLogger

log: BoundLogger = structlog.get_logger()
log.info("typed_ok")
```

([structlog][19])

---

## M) Performance + operational knobs (what actually moves the needle)

### High-leverage knobs (ranked)

1. **Filtering wrapper**: `make_filtering_bound_logger()` is fastest (`return None` below threshold). ([structlog][8])
2. **Avoid proxy overhead in hot loops**: make a local logger via `bind()` and reuse. ([structlog][8])
3. **Enable caching in prod**: `cache_logger_on_first_use=True` to assemble bound loggers once. ([structlog][8])
4. **Avoid stdlib logging bottleneck for your own logs**: prefer `WriteLoggerFactory` / `BytesLoggerFactory`. ([structlog][8])
5. **Fast JSON serializer**: configure `JSONRenderer` with `orjson/msgspec` and match sink type (bytes vs str). ([structlog][8])

### Failure modes (perf + correctness)

* Caching prevents late reconfiguration and breaks `capture_logs()` semantics for already cached loggers; don’t enable caching in tests. ([structlog][18])
* Caching + multiprocessing: cached bound logger not pickleable; avoid passing logger objects between processes. ([structlog][8])

### Minimal “fast prod” configuration (the doc’s own benchmarked baseline)

```python
import logging, orjson, structlog

structlog.configure(
    cache_logger_on_first_use=True,
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.format_exc_info,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.JSONRenderer(serializer=orjson.dumps),
    ],
    logger_factory=structlog.BytesLoggerFactory(),
)
```

This is the performance guide’s “production-ready” example and rationale. ([structlog][8])

---

If you want the *next* increment after this: I can turn this into a **CodeIntel logging “contract pack”** (one canonical config module + strict event-schema processor + redaction policy + pytest fixtures + golden log snapshots), mapped to your CLI/FastAPI/Hamilton run boundaries.

[1]: https://www.structlog.org/en/stable/api.html "API Reference — structlog 25.5.0 documentation"
[2]: https://www.structlog.org/en/stable/bound-loggers.html "Bound Loggers — structlog 25.5.0 documentation"
[3]: https://www.structlog.org/en/stable/processors.html "Processors — structlog 25.5.0 documentation"
[4]: https://www.structlog.org/en/stable/standard-library.html "Standard Library Logging — structlog 25.5.0 documentation"
[5]: https://www.structlog.org/en/stable/contextvars.html "Context Variables — structlog 25.5.0 documentation"
[6]: https://www.structlog.org/en/stable/configuration.html "Configuration — structlog 25.5.0 documentation"
[7]: https://www.structlog.org/en/0.4/configuration.html "Configuration — structlog  documentation"
[8]: https://www.structlog.org/en/stable/performance.html "Performance — structlog 25.5.0 documentation"
[9]: https://www.structlog.org/en/stable/exceptions.html "Exceptions — structlog 25.5.0 documentation"
[10]: https://www.structlog.org/en/stable/api.html?utm_source=chatgpt.com "API Reference — structlog 25.5.0 documentation"
[11]: https://www.structlog.org/en/24.1.0/_modules/structlog/processors.html?utm_source=chatgpt.com "Source code for structlog.processors"
[12]: https://www.structlog.org/en/24.3.0.post2/_modules/structlog/tracebacks.html?utm_source=chatgpt.com "structlog.tracebacks - structlog 24.3.0.post2 documentation"
[13]: https://www.structlog.org/en/stable/thread-local.html "Legacy Thread-local Context — structlog 25.5.0 documentation"
[14]: https://www.structlog.org/en/stable/twisted.html "Twisted — structlog 25.5.0 documentation"
[15]: https://www.structlog.org/en/stable/console-output.html "Console Output — structlog 25.5.0 documentation"
[16]: https://www.structlog.org/en/25.4.0/console-output.html "Console Output - structlog 25.4.0 documentation"
[17]: https://www.structlog.org/en/stable/logging-best-practices.html "Logging Best Practices — structlog 25.5.0 documentation"
[18]: https://www.structlog.org/en/stable/testing.html "Testing — structlog 25.5.0 documentation"
[19]: https://www.structlog.org/en/stable/typing.html "Type Hints — structlog 25.5.0 documentation"


## CodeIntel logging “contract pack”: canonical module + strict schema + redaction + pytest + golden snapshots

This is a **repo-ready design pack** you can hand to an agent and implement with minimal ambiguity. It assumes:

* **structlog** is the primary API for CodeIntel-owned logs.
* Optional: unify **stdlib logging** (third-party libs) through the same renderer pipeline.
* **contextvars** carry run/request/node metadata across CLI/FastAPI/Hamilton.

---

# 1) The contract (what every CodeIntel log line must satisfy)

## 1.1 Canonical top-level keys (baseline)

**Always present on emitted lines (CodeIntel-owned events):**

* `schema_version: int` (start with `1`)
* `timestamp: str` (ISO-8601)
* `level: str` (`debug|info|warning|error|critical`)
* `event: str` (namespaced, stable)
* `logger: str` (logical logger name; usually module or component)
* `run_id: str` (**required** for CodeIntel-owned events)

**Strongly recommended correlation keys:**

* `repo_id: str` (repo identity)
* `pipeline: str` (logical pipeline name, e.g., `build`, `ingest`, `serve`)
* `step: str` (logical phase/step, e.g., `scip_ingest`, `call_graph`)
* `target: str` (Hamilton materialization target name if relevant)
* `node: str` (Hamilton node name if relevant)

**Outcome keys (standardize):**

* `status: str` (`ok|error|skipped`)
* `duration_ms: int` (for span/step/node completion events)
* `error: {type:str, message:str, …}` (structured error payload)

## 1.2 Event namespace + required fields per boundary

Use **dot-separated** names to avoid collisions and to enable prefix policies:

### Run boundary (CLI, Hamilton top-level execution)

* `codeintel.run.started`: required `run_id`, `pipeline`, and either `command` or `dag_name`
* `codeintel.run.finished`: required `run_id`, `status`, `duration_ms`

### HTTP boundary (FastAPI)

* `codeintel.http.request.started`: required `run_id`, `request_id`, `method`, `path`
* `codeintel.http.request.finished`: required `run_id`, `request_id`, `status_code`, `duration_ms`

### Hamilton boundary (node execution)

* `codeintel.hamilton.node.started`: required `run_id`, `dag_name`, `node`
* `codeintel.hamilton.node.finished`: required `run_id`, `dag_name`, `node`, `status`, `duration_ms`

> Everything else can exist (it’s structured logging), but these base requirements give you stable machine parsing + golden snapshots.

---

# 2) File layout (drop-in “pack”)

Proposed package root (adjust to your repo conventions):

```
src/codeintel/observability/logging/
  __init__.py
  config.py              # canonical configure_logging()
  contract.py            # event requirements + validator + error type
  processors.py          # normalize/redact/cap + contract processor
  context.py             # contextvars helpers (run/request/node spans)
  integrations/
    __init__.py
    fastapi.py           # middleware
    hamilton.py          # lifecycle hook adapter
tests/logging/
  conftest.py            # fixtures
  harness.py             # snapshot normalization + assert helpers
  test_contract.py       # schema + redaction unit tests
  test_boundaries.py     # CLI/FastAPI/Hamilton boundary tests
tests/goldens/logging/
  test_run.jsonl
  test_hamilton_nodes.jsonl
  test_http_requests.jsonl
```

---

# 3) Canonical config module (`config.py`): one place to wire everything

### Goals

* Single entrypoint: **`configure_logging()`**
* Mode-based defaults: `dev|prod|test`
* Structured output by default (JSON); pretty console in dev TTY
* Contract enforcement:

  * `test`: **raise** on contract violation
  * `prod`: **annotate** (never lose logs in prod)
* Optional stdlib unification via ProcessorFormatter

```python
# src/codeintel/observability/logging/config.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import IO, Any, Callable, Iterable, Optional

import structlog

from .processors import (
    make_contract_processor,
    make_normalize_processor,
    make_redact_processor,
    make_size_cap_processor,
)
from .contract import ContractMode, SchemaSpec


class LogMode(str, Enum):
    DEV = "dev"
    PROD = "prod"
    TEST = "test"


class LogFormat(str, Enum):
    JSON = "json"
    CONSOLE = "console"


@dataclass(frozen=True)
class LogConfig:
    mode: LogMode = LogMode.PROD
    level: str = "info"  # used by filtering wrapper
    fmt: LogFormat = LogFormat.JSON
    include_callsite: bool = False
    structured_exceptions: bool = True  # dict_tracebacks vs format_exc_info
    cache_logger: bool = True  # disable in tests

    # sink selection
    stream: Optional[IO[str]] = None  # default stdout/stderr based on your policy
    use_bytes_sink: bool = False       # if JSON serializer returns bytes

    # contract knobs
    schema: SchemaSpec = SchemaSpec.v1()
    contract_mode: ContractMode = ContractMode.ANNOTATE  # overridden by mode

    # redaction/size-capping knobs
    redact_keys: tuple[str, ...] = (
        "password", "passwd", "secret", "token", "api_key", "apikey",
        "authorization", "cookie", "set-cookie", "session", "jwt",
    )
    max_string: int = 4000
    max_container: int = 50
    max_depth: int = 6


def configure_logging(cfg: LogConfig) -> None:
    """
    Canonical CodeIntel structlog configuration. Call exactly once at process init
    (CLI entry, ASGI app startup, or Hamilton runner init).
    """
    # Mode-derived overrides:
    contract_mode = cfg.contract_mode
    cache_logger = cfg.cache_logger
    fmt = cfg.fmt
    if cfg.mode == LogMode.TEST:
        contract_mode = ContractMode.RAISE
        cache_logger = False
        fmt = LogFormat.JSON
    elif cfg.mode == LogMode.DEV and fmt == LogFormat.JSON:
        # dev often wants pretty if interactive; keep JSON if you prefer.
        pass

    wrapper_class = structlog.make_filtering_bound_logger(cfg.level)

    # Core processors (order matters: merge -> enrich -> normalize -> redact/cap -> contract -> render)
    processors: list[Callable[..., Any]] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,  # safe even if you don't unify stdlib
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
    ]

    if cfg.include_callsite:
        from structlog.processors import CallsiteParameterAdder, CallsiteParameter
        processors.append(
            CallsiteParameterAdder(parameters={
                CallsiteParameter.FILENAME,
                CallsiteParameter.LINENO,
                CallsiteParameter.FUNC_NAME,
            })
        )

    if cfg.structured_exceptions:
        processors.append(structlog.processors.dict_tracebacks)
    else:
        processors.append(structlog.processors.format_exc_info)

    processors.extend([
        make_normalize_processor(max_depth=cfg.max_depth),
        make_redact_processor(redact_keys=cfg.redact_keys),
        make_size_cap_processor(
            max_string=cfg.max_string,
            max_container=cfg.max_container,
            max_depth=cfg.max_depth,
        ),
        make_contract_processor(schema=cfg.schema, mode=contract_mode),
    ])

    # Renderer + sink
    if fmt == LogFormat.CONSOLE:
        processors.append(structlog.dev.ConsoleRenderer())
        logger_factory = structlog.PrintLoggerFactory(file=cfg.stream)
    else:
        # JSON output; choose bytes vs str depending on serializer/sink
        processors.append(structlog.processors.JSONRenderer())
        logger_factory = structlog.WriteLoggerFactory(file=cfg.stream)

    structlog.configure(
        processors=processors,
        wrapper_class=wrapper_class,
        logger_factory=logger_factory,
        cache_logger_on_first_use=cache_logger,
    )
```

> If you want the fastest prod path: swap JSONRenderer serializer to orjson/msgspec and use `BytesLoggerFactory` + `use_bytes_sink=True` (but keep contract processor *before* renderer so schema stays independent of serializer quirks).

---

# 4) Strict schema processor + redaction + size caps (`processors.py`)

This is where “contract pack” earns its keep: it prevents accidental blob logs, secret leakage, non-JSON types, and schema drift.

```python
# src/codeintel/observability/logging/processors.py
from __future__ import annotations

from dataclasses import is_dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .contract import ContractMode, SchemaSpec, validate_event


REDACTED = "<redacted>"
OMITTED = "<omitted>"
TRUNC = "…"


def make_normalize_processor(*, max_depth: int) -> Callable[..., Any]:
    def normalize(_, __, event_dict: dict[str, Any]) -> dict[str, Any]:
        return _normalize_mapping(event_dict, max_depth=max_depth)
    return normalize


def make_redact_processor(*, redact_keys: Sequence[str]) -> Callable[..., Any]:
    redact_set = {k.lower() for k in redact_keys}

    def redact(_, __, event_dict: dict[str, Any]) -> dict[str, Any]:
        return _redact_mapping(event_dict, redact_set=redact_set)
    return redact


def make_size_cap_processor(*, max_string: int, max_container: int, max_depth: int) -> Callable[..., Any]:
    def cap(_, __, event_dict: dict[str, Any]) -> dict[str, Any]:
        return _cap_value(event_dict, max_string=max_string, max_container=max_container, max_depth=max_depth)
    return cap


def make_contract_processor(*, schema: SchemaSpec, mode: ContractMode) -> Callable[..., Any]:
    def enforce(_, __, event_dict: dict[str, Any]) -> dict[str, Any]:
        ok, issues = validate_event(event_dict, schema=schema)
        if ok:
            return event_dict
        if mode == ContractMode.RAISE:
            raise ValueError(f"Log contract violation: {issues}")
        # ANNOTATE: never drop prod logs; mark and continue
        event_dict["contract_violation"] = issues
        return event_dict
    return enforce


# ----------------------------
# Normalization (JSON-friendly)
# ----------------------------

def _normalize_mapping(obj: Mapping[str, Any], *, max_depth: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in obj.items():
        out[str(k)] = _normalize_value(v, max_depth=max_depth)
    return out


def _normalize_value(v: Any, *, max_depth: int) -> Any:
    if max_depth <= 0:
        return OMITTED

    if v is None or isinstance(v, (bool, int, float, str)):
        return v

    if isinstance(v, (bytes, bytearray, memoryview)):
        # Never emit raw bytes; this is almost always accidental in CodeIntel
        return "<bytes>"

    if isinstance(v, Path):
        return str(v)

    if isinstance(v, Enum):
        return v.name

    # Pydantic v2 models
    dump = getattr(v, "model_dump", None)
    if callable(dump):
        return _normalize_value(dump(), max_depth=max_depth - 1)

    # dataclasses
    if is_dataclass(v):
        return _normalize_value(asdict(v), max_depth=max_depth - 1)

    # sequences/mappings
    if isinstance(v, Mapping):
        return _normalize_mapping(v, max_depth=max_depth - 1)

    if isinstance(v, (list, tuple, set, frozenset)):
        return [_normalize_value(x, max_depth=max_depth - 1) for x in list(v)]

    # Avoid importing heavy deps; summarize by module prefix
    mod = type(v).__module__
    if mod.startswith("pyarrow"):
        return f"<pyarrow:{type(v).__name__}>"
    if mod.startswith("polars"):
        return f"<polars:{type(v).__name__}>"
    if mod.startswith("pandas"):
        return f"<pandas:{type(v).__name__}>"

    # Fallback: stringize
    try:
        return str(v)
    except Exception:
        return f"<unprintable:{type(v).__name__}>"


# --------------
# Redaction
# --------------

def _redact_mapping(obj: Mapping[str, Any], *, redact_set: set[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in obj.items():
        key = str(k)
        if key.lower() in redact_set:
            out[key] = REDACTED
            continue
        out[key] = _redact_value(v, redact_set=redact_set)
    return out


def _redact_value(v: Any, *, redact_set: set[str]) -> Any:
    if isinstance(v, Mapping):
        return _redact_mapping(v, redact_set=redact_set)
    if isinstance(v, list):
        return [_redact_value(x, redact_set=redact_set) for x in v]
    if isinstance(v, str):
        # Lightweight heuristic redaction for common auth headers
        if v.startswith("Bearer ") or v.startswith("Basic "):
            return REDACTED
    return v


# --------------
# Size capping
# --------------

def _cap_value(v: Any, *, max_string: int, max_container: int, max_depth: int) -> Any:
    if max_depth <= 0:
        return OMITTED

    if isinstance(v, str):
        return v if len(v) <= max_string else (v[: max_string - 1] + TRUNC)

    if isinstance(v, Mapping):
        out: dict[str, Any] = {}
        for i, (k, val) in enumerate(v.items()):
            if i >= max_container:
                out["__truncated__"] = True
                break
            out[str(k)] = _cap_value(val, max_string=max_string, max_container=max_container, max_depth=max_depth - 1)
        return out

    if isinstance(v, list):
        out_list = []
        for i, x in enumerate(v):
            if i >= max_container:
                out_list.append({"__truncated__": True})
                break
            out_list.append(_cap_value(x, max_string=max_string, max_container=max_container, max_depth=max_depth - 1))
        return out_list

    return v
```

---

# 5) Contract spec + validator (`contract.py`)

Keep this tight: base required fields + event-specific required fields for run/request/node boundaries.

```python
# src/codeintel/observability/logging/contract.py
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class ContractMode(str, Enum):
    RAISE = "raise"
    ANNOTATE = "annotate"


@dataclass(frozen=True)
class SchemaSpec:
    schema_version: int
    require_run_id_prefix: str = "codeintel."  # enforce run_id for CodeIntel-owned events

    # event-specific requirements
    required_by_event: dict[str, tuple[str, ...]] = None  # type: ignore[assignment]

    @staticmethod
    def v1() -> "SchemaSpec":
        return SchemaSpec(
            schema_version=1,
            required_by_event={
                "codeintel.run.started": ("run_id", "pipeline"),
                "codeintel.run.finished": ("run_id", "status", "duration_ms"),
                "codeintel.http.request.started": ("run_id", "request_id", "method", "path"),
                "codeintel.http.request.finished": ("run_id", "request_id", "status_code", "duration_ms"),
                "codeintel.hamilton.node.started": ("run_id", "dag_name", "node"),
                "codeintel.hamilton.node.finished": ("run_id", "dag_name", "node", "status", "duration_ms"),
            },
        )


BASE_REQUIRED = ("schema_version", "timestamp", "level", "event", "logger")


def validate_event(event_dict: Mapping[str, Any], *, schema: SchemaSpec) -> tuple[bool, list[str]]:
    issues: list[str] = []

    # base keys
    for k in BASE_REQUIRED:
        if k not in event_dict:
            issues.append(f"missing:{k}")

    # schema version
    sv = event_dict.get("schema_version")
    if sv is None:
        pass
    elif sv != schema.schema_version:
        issues.append(f"schema_version:{sv}!=expected:{schema.schema_version}")

    # event type
    event = event_dict.get("event")
    if not isinstance(event, str) or not event:
        issues.append("event:not_str_or_empty")
        return False, issues

    # CodeIntel-owned events must have run_id
    if event.startswith(schema.require_run_id_prefix) and "run_id" not in event_dict:
        issues.append("missing:run_id")

    # per-event required
    req = (schema.required_by_event or {}).get(event)
    if req:
        for k in req:
            if k not in event_dict:
                issues.append(f"missing:{k}")

    return (len(issues) == 0), issues
```

---

# 6) Boundary mapping: CLI + FastAPI + Hamilton

## 6.1 Context helpers (`context.py`)

Use contextvars for correlation keys and **tokens for scoped reset**.

```python
# src/codeintel/observability/logging/context.py
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator

from structlog.contextvars import clear_contextvars, bind_contextvars, reset_contextvars


def reset_all() -> None:
    clear_contextvars()


def bind(**kw: Any) -> dict[str, Any]:
    # returns tokens; caller should reset_contextvars(**tokens)
    return bind_contextvars(**kw)


@contextmanager
def scoped(**kw: Any) -> Iterator[None]:
    tokens = bind_contextvars(**kw)
    try:
        yield
    finally:
        reset_contextvars(**tokens)
```

## 6.2 CLI boundary (run span)

At CLI entry, bind run metadata once, log `run.started`, and always log `run.finished`.

```python
import time, uuid, structlog
from codeintel.observability.logging.context import reset_all, scoped

log = structlog.get_logger("codeintel")

def run_cli(command: str, pipeline: str, repo_id: str | None = None) -> int:
    run_id = str(uuid.uuid4())
    reset_all()
    t0 = time.perf_counter()

    with scoped(run_id=run_id, pipeline=pipeline, repo_id=repo_id):
        log.info("codeintel.run.started", schema_version=1, command=command, logger="codeintel")
        try:
            # execute command
            return_code = 0
            status = "ok"
            return return_code
        except Exception:
            status = "error"
            raise
        finally:
            dt_ms = int((time.perf_counter() - t0) * 1000)
            log.info("codeintel.run.finished", schema_version=1, status=status, duration_ms=dt_ms, logger="codeintel")
```

## 6.3 FastAPI boundary (middleware)

Bind request context (request_id, route, method, client) and emit start/finish events.

```python
# src/codeintel/observability/logging/integrations/fastapi.py
from __future__ import annotations

import time
import uuid
import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from ..context import scoped

log = structlog.get_logger("codeintel.http")


class StructlogRequestMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        t0 = time.perf_counter()

        with scoped(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        ):
            log.info("codeintel.http.request.started", schema_version=1, logger="codeintel.http")
            try:
                response: Response = await call_next(request)
                return response
            finally:
                dt_ms = int((time.perf_counter() - t0) * 1000)
                status_code = getattr(locals().get("response"), "status_code", 500)
                log.info(
                    "codeintel.http.request.finished",
                    schema_version=1,
                    logger="codeintel.http",
                    status_code=status_code,
                    duration_ms=dt_ms,
                )
```

> This assumes `run_id` is already bound at app startup (e.g., per “server run” or per-request if you treat requests as runs). If you want **every request to be a run**, set `run_id=request_id` in the middleware.

## 6.4 Hamilton boundary (lifecycle hook adapter)

Use a Hamilton **NodeExecutionHook** to log node spans with duration and outcome.

```python
# src/codeintel/observability/logging/integrations/hamilton.py
from __future__ import annotations

import time
import structlog

from ..context import scoped

log = structlog.get_logger("codeintel.hamilton")


class StructlogNodeExecutionHook:
    """
    Hamilton lifecycle adapter that logs before/after each node execution.
    Attach via Builder.with_adapters(...).
    """

    def run_before_node_execution(self, *, node_name: str, node_tags: dict, node_kwargs: dict, **_):
        # Store start time in kwargs for after-hook; Hamilton passes it through
        node_kwargs["__t0"] = time.perf_counter()
        with scoped(node=node_name, dag_name=node_tags.get("dag_name")):
            log.info("codeintel.hamilton.node.started", schema_version=1, logger="codeintel.hamilton")

    def run_after_node_execution(self, *, node_name: str, node_tags: dict, node_kwargs: dict, success: bool, error: Exception | None = None, **_):
        t0 = node_kwargs.pop("__t0", None)
        dt_ms = int((time.perf_counter() - t0) * 1000) if t0 else None
        with scoped(node=node_name, dag_name=node_tags.get("dag_name")):
            if success:
                log.info("codeintel.hamilton.node.finished", schema_version=1, logger="codeintel.hamilton", status="ok", duration_ms=dt_ms)
            else:
                log.error(
                    "codeintel.hamilton.node.finished",
                    schema_version=1,
                    logger="codeintel.hamilton",
                    status="error",
                    duration_ms=dt_ms,
                    error={"type": type(error).__name__, "message": str(error)} if error else None,
                    exc_info=True,
                )
```

> Exact hook method signatures vary slightly by Hamilton version/config. The key idea is stable: **before/after node**, compute duration, bind `node/dag_name`, emit start/finish contract events. If your runner already centralizes execution, you can implement this as a wrapper around `driver.execute()` without touching Hamilton hooks.

---

# 7) Pytest fixtures + golden JSONL snapshots

## 7.1 Test-time logging fixture (captures JSON lines deterministically)

Use a `StringIO` sink and parse JSONL.

```python
# tests/logging/conftest.py
from __future__ import annotations

import io
import json
import pytest

from codeintel.observability.logging.config import configure_logging, LogConfig, LogMode, LogFormat


@pytest.fixture()
def log_buffer():
    buf = io.StringIO()
    configure_logging(LogConfig(mode=LogMode.TEST, fmt=LogFormat.JSON, stream=buf, cache_logger=False))
    yield buf


def read_jsonl(buf: io.StringIO) -> list[dict]:
    buf.seek(0)
    lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
    return [json.loads(ln) for ln in lines]
```

## 7.2 Snapshot harness (normalize dynamic fields)

You must normalize `timestamp`, `run_id`, and anything else nondeterministic before comparing.

```python
# tests/logging/harness.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DYNAMIC_KEYS = {"timestamp", "run_id", "request_id"}


def normalize(rec: dict[str, Any]) -> dict[str, Any]:
    out = dict(rec)
    for k in list(out.keys()):
        if k in DYNAMIC_KEYS:
            out[k] = f"<{k}>"
    # also drop callsite/thread/process keys if you ever enable them in tests
    out.pop("lineno", None)
    out.pop("filename", None)
    out.pop("func_name", None)
    return out


def assert_jsonl_snapshot(records: list[dict[str, Any]], golden_path: Path, *, update: bool = False) -> None:
    norm = [normalize(r) for r in records]
    if update:
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text("\n".join(json.dumps(r, sort_keys=True) for r in norm) + "\n", encoding="utf-8")
        return

    expected = [json.loads(ln) for ln in golden_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert norm == expected
```

## 7.3 Example tests that exercise boundaries

```python
# tests/logging/test_boundaries.py
from __future__ import annotations

from pathlib import Path
import structlog

from .conftest import read_jsonl
from .harness import assert_jsonl_snapshot


def test_run_boundary(log_buffer, tmp_path):
    log = structlog.get_logger("codeintel")

    log.info("codeintel.run.started", schema_version=1, pipeline="build", command="codeintel build", logger="codeintel", run_id="R1")
    log.info("codeintel.run.finished", schema_version=1, status="ok", duration_ms=10, logger="codeintel", run_id="R1")

    recs = read_jsonl(log_buffer)
    assert_jsonl_snapshot(recs, Path("tests/goldens/logging/test_run.jsonl"), update=False)


def test_redaction(log_buffer):
    log = structlog.get_logger("codeintel")
    log.info("codeintel.run.started", schema_version=1, pipeline="build", logger="codeintel", run_id="R1", token="SECRET")
    recs = read_jsonl(log_buffer)
    assert recs[0]["token"] == "<redacted>"
```

To update goldens, you can wire your existing snapshot mechanism or add a `pytest` flag/env var and pass `update=True`.

---

# 8) “Minimum integration checklist” (agent-executable)

1. **Add package** `src/codeintel/observability/logging/` with modules above.
2. **CLI entrypoint**: call `configure_logging(from_env_or_defaults())` before any `get_logger().bind()` usage; bind `run_id/pipeline/repo_id` at command start.
3. **FastAPI app**: call `configure_logging(...)` at startup; install `StructlogRequestMiddleware`; decide whether each request is a “run” (bind `run_id=request_id`) or whether `run_id` is server-run scoped.
4. **Hamilton runner**: attach `StructlogNodeExecutionHook` adapter (or wrap execute) so every node emits start/finish events with `duration_ms` and `status`.
5. **Tests**: add `log_buffer` fixture + snapshot harness; add 2–3 goldens (run, node, http).
6. Flip prod config to `ContractMode.ANNOTATE` so contract issues never suppress logs; keep tests strict.

---

If you want one more increment after this: I can tailor this pack to your *actual* CodeIntel event taxonomy (run/step/target tables, Hamilton node tags you already use, your existing golden snapshot runner conventions, and your current CLI/FastAPI entrypoints) so it drops in with near-zero naming churn.


Below is the **tailored “logging contract pack”** mapped onto what’s *actually in your current CodeIntel repo* (from the `CodeIntel_InferenceDriven_improved_alignment.zip` you’ve been working from): your **existing event taxonomy**, your **TelemetryContext + semconv keys**, your **Hamilton tag system**, your **CLI/FastAPI entrypoints**, and your **golden update conventions** (`UPDATE_GOLDEN` / `--update-golden`).

---

# 0) What “actual CodeIntel taxonomy” looks like today (so we don’t rename things)

## 0.1 Existing event strings you already emit (keep these)

### CLI (Cyclopts wrapper)

* `cli.invocation` (and `cli.parse_error` as the warning prefix today) emitted in:

  * `src/codeintel/observability/cli.py::_emit_cli_log()`

### Observability “TelemetryEvent” logs

* `scip.teardown`, `build.shutdown`, `shutdown.error`, etc emitted via:

  * `src/codeintel/observability/teardown.py` → `TelemetryEvent` → `emit_event_log()`

### Hamilton build lifecycle (already “event-ish” strings)

* `build.hamilton.executor.start`, `build.hamilton.executor.complete`, `build.hamilton.executor.closure_error`, …

  * `src/codeintel/build/hamilton/executor.py`
* `build.hamilton.writer.*` failures

  * `src/codeintel/build/hamilton/run_writer.py`
* `build.hamilton.run_record.*`

  * `src/codeintel/build/hamilton/run_records.py`

### Serving

* `query_executed` with `extra={...}` payload

  * `src/codeintel/serving/metrics.py::log_query_metrics()`

**Goal:** keep these **event names exactly** and make their payloads **actually structured** + contract-validated.

---

# 1) The tailored contract: base schema + event-specific required keys

## 1.1 Base keys (match your current structured JSON fields)

These are already the “shape” your CLI formatter uses today (`StructuredLogFormatter`):

* `timestamp` (ISO-ish)
* `level`
* `logger`
* `event` (the *event name*, not a JSON string inside message)
* `message` (optional; only for truly “unstructured” lines)

## 1.2 Always-merged correlation context (reuse what you already have)

Your repo already defines a strong cross-cutting context layer:

* `TelemetryContext` in `src/codeintel/observability/telemetry_context.py`
* semconv keys in `src/codeintel/observability/semconv_keys.py`:

  * `codeintel.correlation_id`
  * `codeintel.run_id`
  * `codeintel.domain`
  * `codeintel.repo`
  * `codeintel.commit`
  * `codeintel.actor`

**Contract rule:** whenever `TelemetryContext` has values, they must be present in the emitted log entry under those semconv keys.

## 1.3 Event-specific required keys (tailored to your existing payloads)

Use a `required_by_event` mapping (contract processor) like:

* `cli.invocation`
  Required (keep existing key names to avoid churn):

  * `invocation_id`, `command`, `exit_code`, `duration_ms`
    Optional: `parse_duration_ms`, `is_parse_error`, `error_type`

* `build.hamilton.executor.start`
  Required:

  * `run_id`, `targets`
    Recommended: `codeintel.repo`, `codeintel.commit` (merge from TelemetryContext if you bind it; see §3.3)

* `build.hamilton.executor.complete`
  Required:

  * `run_id`, `success`, `duration_ms`
    Recommended: `computed_targets`, `skipped_targets`, `failed_targets` (these are all computed in `_finalize_run()`)

* `scip.teardown` (emitted through `TelemetryEvent`)
  Required (your existing payload keys):

  * `status`
    Recommended: `duration_ms`, and semconv fields `scip.*` already exist in span attributes.

* `query_executed` (serving)
  Required (your existing extra keys):

  * `endpoint`, `row_count`, `truncated`, `duration_ms`, `correlation_id`
    Recommended: engine hashes etc when present

**Mode behavior (very CodeIntel-friendly):**

* **tests**: violation → raise (fail fast)
* **prod**: violation → annotate with `contract_violation=[...]` (never drop logs)

---

# 2) Tailored module layout: integrate into your existing `codeintel.observability` package

Instead of adding a new “observability/logging/…” tree, slot it into what you already have:

```
src/codeintel/observability/
  structlog_contract.py      # SchemaSpec + validate_event(...)
  structlog_processors.py    # merge_telemetry_context, normalize, redact, cap, contract
  structlog_pipeline.py      # configure_codeintel_logging(mode, output_format, stream, ...)
```

And then **replace** the old CLI-only formatter usage with this pipeline at the entrypoints you already own.

---

# 3) Wiring it to your real entrypoints (CLI / FastAPI / Hamilton)

## 3.1 CLI entrypoint: `bootstrap_cli()` is the control point you already use

File: `src/codeintel/cli/execution/bootstrap.py`

Today:

* `_configure_logging(... structured=True)` calls `configure_structured_logging(...)` (your JSON formatter).

Target:

* `_configure_logging(... structured=True)` calls `configure_codeintel_logging(...)` (structlog pipeline).

**Key mapping to preserve your CLI behavior:**

* “structured logging on” remains: `structured_logging or active_config.telemetry.enabled`
* use `OutputFormat` (already resolved in `cli.commands.app:main()`) to select:

  * **JSON output** → JSON renderer
  * **TEXT output** → ConsoleRenderer (TTY-friendly) or still JSON if you want determinism

### Minimal “drop-in” call shape

```python
# in src/codeintel/cli/execution/bootstrap.py
from codeintel.observability.structlog_pipeline import configure_codeintel_logging

def _configure_logging(..., structured: bool = False) -> None:
    level = _determine_log_level(...)
    if structured:
        configure_codeintel_logging(
            level=level,
            mode="cli",
            output_format=("json" if config.output_format_is_json else "console"),
        )
    else:
        logging.basicConfig(...)
```

## 3.2 Make your existing CLI telemetry log *actually structured* (no embedded JSON strings)

File: `src/codeintel/observability/cli.py`

Today `_emit_cli_log()` does:

* `message = json.dumps(payload)`
* `log.info("cli.invocation %s", message)` (stringly-typed payload)

Change to (near-zero naming churn):

* message becomes the event string
* payload becomes `extra={...}`

```python
# src/codeintel/observability/cli.py::_emit_cli_log
payload = {
  "invocation_id": state.invocation_id,
  "command": _command_label(state.command_chain),
  "exit_code": exit_code,
  "duration_ms": (time.perf_counter() - state.start_ts) * 1000,
  "parse_duration_ms": state.parse_duration_ms,
  "is_parse_error": state.is_parse_error,
  "error_type": state.error_type,
}
if state.is_parse_error:
    log.warning("cli.parse_error", extra=payload)
else:
    log.info("cli.invocation", extra=payload)
```

**Why this matters with your codebase:** you already use `extra={...}` for serving metrics (`query_executed`). This makes CLI consistent with serving.

## 3.3 Hamilton build boundary: convert your “event-ish strings” into structured events

File: `src/codeintel/build/hamilton/executor.py`

Today:

```python
log.info("build.hamilton.executor.start run_id=%s targets=%s", run_id, targets)
```

Change to:

```python
log.info("build.hamilton.executor.start", extra={"run_id": context.run_id, "targets": requested_targets})
```

And in `_finalize_run()`:

```python
log.info(
  "build.hamilton.executor.complete",
  extra={"run_id": context.run_id, "success": success, "duration_ms": duration_ms,
         "computed_targets": computed, "skipped_targets": skipped, "failed_targets": failed},
)
```

**Optional (high value, low churn): bind TelemetryContext for build runs**
You already have `telemetry_context(...)` and semconv keys; wrap the build run in it once:

* Use `codeintel.observability.telemetry_context.telemetry_context(run_id=run_id, domain=domain, repo_commit=...)`
* Then every log line automatically gets `codeintel.run_id`, `codeintel.repo`, `codeintel.commit`, etc (via the merge processor in §4).

Where to do it:

* In `HamiltonExecutor.run()` around `_run_with_state(...)` (single place).

## 3.4 FastAPI: you already set correlation_id; add lightweight request start/finish events

File: `src/codeintel/serving/http/middleware.py`

You already do:

* generate correlation id
* `with telemetry_context(correlation_id=correlation_id): ...`

Add:

* `LOG.info("http.request.started", extra={"method": request.method, "path": request.url.path})`
* `LOG.info("http.request.finished", extra={"status_code": response.status_code, "duration_ms": ...})`

This yields consistent “boundary signals” without fighting FastAPI instrumentation.

---

# 4) The structlog pipeline: tailor processors to your *existing* policy + telemetry context

This is the core: format **stdlib logs + extra fields** into stable JSON, while merging:

* TelemetryContext semconv keys (already in your repo)
* OTEL trace context (you already do this in `OTELTraceAdapter`)
* size caps + redactions consistent with `ObservabilityPolicy`

## 4.1 “must-have” processors for CodeIntel

* `merge_telemetry_context()` → merges `current_telemetry_context().span_attributes()` into event dict
* `merge_trace_context()` → merges `trace_id/span_id` (same semantics as your current JSON formatter)
* `structlog.stdlib.ExtraAdder()` → pulls `LogRecord` extras (critical because your repo uses `extra={...}` a lot)
* `normalize_json()` → coerce enums/paths/dataclasses/pydantic safely
* `cap_payload()` → prevent huge accidental logs (keep your budgets aligned to `ObservabilityPolicy.budget`)
* `redact()` → reuse your redaction policy patterns (path/command segment keeping, auth header patterns)
* `contract_enforcer()` → required keys per event, with mode-dependent behavior

## 4.2 Pipeline config strategy (matches your “one canonical bootstrap” ethos)

* Provide `configure_codeintel_logging(level, mode, output_format, stream=...)`
* Internally uses `structlog.stdlib.ProcessorFormatter` so **all existing `logging.getLogger(...)` call sites benefit immediately**, including:

  * build executor logs
  * run writer logs
  * serving metrics
  * telemetry teardown events
  * third-party logs (optionally filtered/limited)

---

# 5) Golden log snapshots that match your existing conventions

You already have:

* `tests/_helpers/goldens/artifact_goldens.py` with `UPDATE_GOLDEN` support
* `tests/_helpers/pytest_options.py` that maps `--update-golden` → `UPDATE_GOLDEN=1`

## 5.1 Add a log-golden helper that reuses `artifact_goldens`

New file:

* `tests/_helpers/goldens/log_goldens.py`

Behavior:

* capture JSON lines from a StringIO StreamHandler
* parse into `list[dict]`
* normalize dynamic keys:

  * `timestamp` → `"<timestamp>"`
  * `trace_id/span_id` → `"<trace_id>" / "<span_id>"` if present
  * random ids (`invocation_id`, etc) → stable placeholders
* write actual as a JSON artifact to a temp file
* call `assert_json_artifact_matches_golden(actual_path=..., golden_path=...)`

## 5.2 Add goldens where you already keep them

Recommended locations (to match your existing layout):

* CLI log goldens: `tests/cli/_golden/cli_invocation_log.json`
* Build log goldens: `tests/build/_golden/build_executor_log.json` (create dir)
* Serving log goldens: `tests/serving/_golden/query_metrics_log.json` (create dir)

## 5.3 Minimal test targets (high signal, low flakiness)

* CLI: run a trivial Cyclopts app through `run_cli_with_telemetry(...)` and snapshot emitted log entries.
* Serving: call `log_query_metrics(QueryMetrics(...))` and snapshot output.
* Build: unit-test `_finalize_run()` or the run start/complete emission by calling those functions with small fakes and capturing logs (avoid full DAG execution for log tests).

---

# 6) “Near-zero naming churn” delta checklist (the exact edits to make first)

### P0: enable the pipeline without touching most call sites

1. **Add** `structlog_contract.py`, `structlog_processors.py`, `structlog_pipeline.py` under `src/codeintel/observability/`.
2. **Modify** `src/codeintel/cli/execution/bootstrap.py` to call `configure_codeintel_logging()` when `structured=True`.

At this point: **serving metrics logs** (`extra={...}`) start becoming JSON-rich automatically if you use ProcessorFormatter+ExtraAdder.

### P1: fix the worst offenders (stringified JSON payloads)

3. **Modify** `src/codeintel/observability/cli.py::_emit_cli_log()` to stop embedding JSON in message and use `extra=payload`.
4. **Modify** `src/codeintel/observability/events.py::emit_event_log()` similarly:

   * stop `"%s %s"` + `json.dumps`
   * do `logger.log(level, event_name, extra=dict(payload))`

### P2: upgrade build boundary signals

5. **Modify** `src/codeintel/build/hamilton/executor.py` start/complete/error logs to use `extra={...}` fields.
6. Optionally bind `telemetry_context(run_id=..., domain=..., repo_commit=...)` once per build run.

---

If you want the next increment after this tailored pack: I can write the **exact contract mapping dictionary** for *every* currently-emitted “event-ish” string in the repo (build.* / scip.* / shutdown.* / cli.* / serving.*), and a mechanical refactor plan that converts the top ~30 highest-signal log lines to structured `extra={...}` form (file-by-file, function-by-function), while leaving low-value debug logs untouched.
