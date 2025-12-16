At a high level:

* A single `types.py` at repo root is **not** best practice (it shadows the stdlib `types` module and tends to become a junk drawer).
* Having **shared type modules** *is* best practice — but organized by **domain** (e.g. `typing/json_types.py`, `typing/protocols.py`, etc.), not one giant file.
* “Best-in-class handling of stubborn `Any`” is mostly about:

  1. Keeping `Any` at **boundaries only**,
  2. Using **aliases, Protocols, TypedDict, generics, stubs, and type guards** to narrow it, and
  3. Configuring your type checker so `Any` cannot silently spread.

Below is a fairly complete guide aimed at someone building AI/LLM-heavy Python systems.

---

## 1. Mental model: `Any`, `Unknown`, and `object`

### 1.1 `Any` (PEP 484)

PEP 484 defines `Any` as a special type that **disables static checking** for values of that type: any operation on `Any` is allowed and type checkers must accept it. ([Python Enhancement Proposals (PEPs)][1])

Consequences:

* `Any` **flows**: if a value is `Any`, operations using it generally become `Any` too.
* Type checkers cannot guarantee correctness where `Any` flows; it’s an **unsoundness leak**.

For LLM-programmed systems, `Any` is essentially “turn off the safety net here”.

### 1.2 `Unknown` (Pyright/BasedPyright)

Pyright distinguishes between `Any` and **“unknown”** types that come from untyped or partially typed code. In strict mode it reports these with diagnostics like `reportUnknownVariableType`, `reportUnknownMemberType`, etc. ([Microsoft][2])

BasedPyright adds `reportAny` to treat *all* `Any` as an error, including explicit ones. ([BasedPyright][3])

Key idea: treat both `Any` and “Unknown” as **defects to fix or contain**.

### 1.3 `object` vs `Any`

Community discussions and mypy issues explicitly recommend using `object` instead of `Any` when you just want a “top” type but still want the checker to enforce narrowing. ([GitHub][4])

* `Any`: you can call any attribute, index, etc. — no error.
* `object`: you can’t call anything without narrowing; you must use `isinstance`, `cast`, or type guards.

So:

> Use `object` when you truly don’t know the type but want to stay safe.
> Use `Any` only as an **escape hatch at boundaries** and immediately narrow it.

---

## 2. Where nasty `Any` comes from

Summarizing the main pipelines (all of which are documented in mypy & pyright docs): ([GDevOps][5])

1. **Untyped or partially typed code you own**

   * Functions without annotations default to `Any` return types and parameters in mypy’s default mode.
2. **Untyped third-party libraries**

   * Missing type info -> symbols fall back to `Any` or “Unknown”.
3. **Dynamic data structures**

   * `dict[str, Any]`, `list[Any]` from JSON, YAML, databases, `inspect` output, etc.
4. **Decorators & higher-order functions**

   * Decorators written without ParamSpec/Concatenate collapse types to `Callable[..., Any]`.
5. **Varargs sinks and “do everything” helpers**

   * `def call_all(*args: Any, **kwargs: Any) -> Any: ...`
6. **Reflection/dynamic dispatch**

   * `getattr`, dynamic `__getattr__`, monkeypatching, plugin loading, etc.

Best practice is to **identify each of these sources** and treat them as **design problems**, not random annoyances.

---

## 3. Project layout: why `types.py` is a trap

### 3.1 Avoid a top-level `types.py`

Python already ships a standard library module called `types` that defines things like `FunctionType`, `SimpleNamespace`, and so on. ([Python documentation][6])

If you add your own `types.py` at the root of your project, then:

```python
import types
```

will import **your file**, not the stdlib module, whenever your project is on `PYTHONPATH`.

That’s surprisingly easy to hit for LLM agents and humans alike, and it creates subtle bugs. It also tends to become a **junk drawer** for every alias in the codebase.

### 3.2 Better patterns

A more robust approach (especially for LLM agents):

* Use a **`typing` or `types_` package**, not a single file:

  ```text
  codeintel/
    typing/
      __init__.py          # re-exports the stable public surface
      json_types.py        # JSON, config, metadata structures
      graph_types.py       # node IDs, edge IDs, etc.
      protocols.py         # plugin interfaces, backends
      vendor_stubs.py      # thin wrappers around untyped libs
  ```

* Re-export only the “blessed” aliases from `typing/__init__.py`:

  ```python
  # codeintel/typing/__init__.py
  from .json_types import JSON, JSONScalar
  from .graph_types import NodeId, EdgeId
  from .protocols import EmbeddingBackend, StoreBackend

  __all__ = [
      "JSON",
      "JSONScalar",
      "NodeId",
      "EdgeId",
      "EmbeddingBackend",
      "StoreBackend",
  ]
  ```

* If you need “global” types: use something like `codeintel/type_defs.py` — **not** `types.py`.

This structure is much friendlier for an AI agent: there’s a **discoverable, well-named namespace** for each conceptual cluster of types.

---

## 4. System-level enforcement: make `Any` noisy

### 4.1 Mypy flags

Mypy has a family of `--disallow-any-*` flags that let you ban specific uses of `Any`: ([Ubuntu Manpages][7])

Useful ones:

* `--disallow-any-expr` – disallow any expression whose type is `Any`, unless immediately cast or assigned to an explicitly-typed variable.
* `--disallow-any-unimported` – error if a type becomes `Any` because an import can’t be resolved.
* `--disallow-any-generics` – prevent `list[Any]`/`dict[Any, Any]` from silently appearing.
* `--disallow-any-explicit` – disallow writing `Any` explicitly in annotations (great once you’ve cleaned up).
* `--disallow-untyped-defs` – forbid untyped function definitions.

Strategy that works well with LLM agents:

* Start **module-by-module** with `# mypy: disallow-any-expr, disallow-untyped-defs`.
* Promote to project-wide once the worst offenders are gone.

### 4.2 Pyright/BasedPyright

Pyright’s `strict` mode treats any “unknown” type as an error and surfaces them via `reportUnknownVariableType`, `reportUnknownParameterType`, etc. ([Microsoft][2])

BasedPyright adds a `reportAny` rule that lets you directly ban `Any` usages. ([BasedPyright][3])

Good pattern:

* For modules you care about most, enable strict checking via config or `# pyright: strict` comment.
* Turn on `reportUnknown*` (and `reportAny` in BasedPyright) to force you to model external/untyped APIs properly.

For LLM agents, your “reward function” can literally be: **no new `Any`/Unknown diagnostics allowed in changed files**.

---

## 5. Patterns and recipes for difficult `Any` cases

### 5.1 JSON-ish data and dynamic dictionaries

#### 5.1.1 Structured aliases instead of `dict[str, Any]`

Define JSON once and reuse it everywhere:

```python
# codeintel/typing/json_types.py
from __future__ import annotations

from typing import Any, Dict, List, Union

JSONScalar = Union[str, int, float, bool, None]
JSON = Union[JSONScalar, List["JSON"], Dict[str, "JSON"]]
```

This pattern (or variants of it) shows up in real-world codebases and articles about typing JSON. ([Stack Overflow][8])

Use `JSON` in signatures instead of `Any`. You don’t get perfect shape info, but:

* The checker knows you’re operating on a recursive JSON structure.
* You avoid unconstrained `Any` everywhere.

#### 5.1.2 `TypedDict` for fixed-schema dicts

When you *do* know the schema, use `TypedDict` instead of `dict[str, Any]`. Mypy’s docs explicitly recommend this for objects represented as dicts. ([mypy Documentation][9])

```python
# codeintel/typing/json_types.py
from typing import TypedDict

class EmbeddingMetadata(TypedDict):
    id: str
    model: str
    dims: int
    source: str
    version: int
```

TypedDict is ideal for:

* API payloads, config objects, small “record” objects.
* Data that will eventually be turned into dataclasses/ORM models but isn’t yet.

### 5.2 Wrapping untyped third-party libraries

#### 5.2.1 Use PEP 561 stub packages when available

PEP 561 defines how to distribute type info via stub packages (`types-foo`), and mypy docs explain how they’re discovered. ([Python Enhancement Proposals (PEPs)][10])

* If a `types-<package>` exists on PyPI (`types-requests`, `types-pyperclip`, etc.), install it.
* This often turns large patches of `Any` into proper types with no code changes.

#### 5.2.2 Generate stubs for the subset you use

When a library has no types:

* Use mypy’s `stubgen` and follow PEP 561 packaging guidance to create stubs for your project. ([mypy Documentation][11])
* Place them in a `stubs/` directory or separate `*-stubs` package.

Sample stub for `foo_client`:

```python
# stubs/foo_client/__init__.pyi
from typing import TypedDict

class User(TypedDict):
    id: str
    name: str

def get_user(user_id: str) -> User: ...
```

This immediately eliminates `Any` for call sites that use `foo_client.get_user`.

#### 5.2.3 Typed wrappers as a façade

Sometimes the library is chaotic or stubbing everything is overkill. Best practice then is to **write a narrow wrapper with good types** and confine `Any` to that module.

```python
# codeintel/vendor/foo_wrapper.py
from __future__ import annotations

from typing import Protocol

# Protocol describing *what you actually use* from the lib
class FooClient(Protocol):
    def get_user(self, user_id: str) -> "User": ...
    def list_users(self) -> list["User"]: ...

class User(TypedDict):
    id: str
    name: str

def make_client(...) -> FooClient:
    # Internally imports and uses the untyped library; may use `cast`/`Any`
    ...
```

Everywhere else in the code:

```python
from codeintel.vendor.foo_wrapper import FooClient, make_client

client: FooClient = make_client(...)
user = client.get_user("123")  # fully typed
```

The rest of the codebase never sees `Any` from this library.

### 5.3 Protocols for plugin systems and duck typing

PEP 544 introduces `typing.Protocol` for **structural “duck typing”**, and both mypy and pyright support it. ([Python Enhancement Proposals (PEPs)][12])

Perfect for plugin/agent interfaces without going through `Any`.

```python
# codeintel/typing/protocols.py
from typing import Protocol, Iterable

class EmbeddingBackend(Protocol):
    model_name: str

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        ...
```

Use it everywhere:

```python
def run_embedding(backend: EmbeddingBackend, docs: list[str]) -> list[list[float]]:
    return backend.embed(docs)
```

Any class implementing `.embed` with compatible signature satisfies the Protocol — no need to expose `Any` and no tight coupling.

For LLM coding agents, Protocols are gold: they define exact behavioral contracts while still allowing flexible implementations.

### 5.4 Generics and ParamSpec for decorators & higher-order functions

PEP 612 added `ParamSpec` and `Concatenate` so decorators can **preserve call signatures** instead of returning `Callable[..., Any]`. ([Python Enhancement Proposals (PEPs)][13])

Basic pattern:

```python
from typing import Callable, ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

def log_calls(fn: Callable[P, R]) -> Callable[P, R]:
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        print("calling", fn.__name__)
        return fn(*args, **kwargs)
    return wrapper
```

Without this, you end up with decorators typed as `Callable[..., Any]` and you lose type info at every call site.

This is one of the **biggest real-world sources** of “hard to displace” `Any` in mature code; using ParamSpec fixes it.

### 5.5 `cast`, runtime validation, and type narrowing

#### 5.5.1 `typing.cast` semantics

`typing.cast(T, value)` is a runtime no-op that just tells the type checker “trust me, this is a `T`”. ([Stack Overflow][14])

So “best in class” usage:

* Only call `cast` **after** concrete runtime checks (or when you have external guarantees).
* Keep `cast` as close as possible to IO / dynamic boundaries.

Example with JSON:

```python
from typing import Any, cast
from .json_types import EmbeddingMetadata

def parse_metadata(raw: Any) -> EmbeddingMetadata:
    # runtime validation (can be simple asserts or pydantic)
    assert isinstance(raw, dict)
    assert isinstance(raw.get("id"), str)
    assert isinstance(raw.get("model"), str)
    ...
    return cast(EmbeddingMetadata, raw)
```

Elsewhere you operate on `EmbeddingMetadata` with full safety. The unsafe, “I know more than the checker” logic is confined.

#### 5.5.2 Type narrowing & TypeGuard / TypeIs

Type checkers already narrow types on `isinstance`, `if x is not None`, etc. ([mypy Documentation][15])

PEP 647 adds `TypeGuard[T]` for **user-defined** type guards. They let you narrow from something like `dict[str, object] | list[object]` to a `TypedDict` or more specific container. ([Python Enhancement Proposals (PEPs)][16])

For example:

```python
from typing import Any, TypeGuard
from .json_types import EmbeddingMetadata

def is_embedding_metadata(val: Any) -> TypeGuard[EmbeddingMetadata]:
    return (
        isinstance(val, dict)
        and isinstance(val.get("id"), str)
        and isinstance(val.get("model"), str)
    )

def process(val: Any) -> None:
    if is_embedding_metadata(val):
        # Here val is EmbeddingMetadata
        print(val["model"])
```

PEP 742 introduces `TypeIs` (Python 3.13+ / `typing_extensions`) to fix some limitations of `TypeGuard` and allow narrowing in both `if` and `else` branches. ([Redowan's Reflections][17])

Takeaway:

* For advanced narrowing of stubborn `Any` at runtime boundaries, **introduce TypeGuard/TypeIs helpers** that LLM agents can reuse rather than sprinkling `cast` everywhere.

### 5.6 `*args/**kwargs` and “catch all” functions

For highly generic functions:

* If you truly want “anything goes”, use:

  ```python
  def debug_log(*args: object, **kwargs: object) -> None: ...
  ```

  Using `object` rather than `Any` keeps the rest of the code honest; you can’t accidentally call methods on `args` without checks.

* If the function **forwards** to another callable, use ParamSpec to preserve types instead of `Any` (see the decorator example).

---

## 6. Migration strategies for large, partially typed codebases

Mypy’s “common issues” docs and CLI help both emphasize that you often need to change coding patterns to benefit from static typing. ([mypy Documentation][18])

A good pragmatic roadmap:

1. **Instrument, don’t guess**

   * Turn on `--disallow-any-unimported` and `--warn-return-any` in CI to find the biggest unsound spots first. ([mypy Documentation][19])
   * With Pyright, start with `reportUnknown*` in strict mode on key modules. ([GitHub][20])

2. **Fix “easy” Any sources**

   * Add missing annotations to hot-path functions.
   * Replace obvious `dict[str, Any]` with `TypedDict` or domain aliases.

3. **Contain third-party chaos**

   * Add stub packages or wrapper modules for the top offenders.
   * Once those are in place, you can enable `--disallow-any-unimported` and pyright `reportUnknown*` without drowning in noise.

4. **Harden internal APIs**

   * Add Protocols, generic types, and ParamSpec decorators to remove `Any` from internal signatures.

5. **Escalate strictness**

   * Gradually enable `--disallow-any-expr` and `--disallow-any-explicit` in more modules.
   * In Pyright/BasedPyright, turn on `reportAny` as a final step. ([Ubuntu Manpages][7])

In an AI-driven workflow, you can encode these steps into your agents’ prompts and CI gates: code that increases the Any/Unknown count fails, and the agent must refactor until the budget is met.

---

## 7. LLM-specific guidance

For “expert AI LLM programmers”, a few extra principles matter:

1. **Make the type surface obvious and discoverable**

   * Centralize domain types in `typing/` modules with clear names.
   * Document invariants there (in docstrings or comments) so LLMs see them when browsing.

2. **Teach your agents to treat `Any` as a bug**

   * In prompts, explicitly say: “Do not introduce new `Any` types; if unavoidable, isolate them in boundary modules and immediately narrow via validation + `cast` or TypeGuard.”

3. **Prefer Protocols over concrete types in public APIs**

   * That gives agents the flexibility to create simulated/mock implementations while still being type-safe.

4. **Use types to constrain search**

   * For LLMs, strong types radically shrink the search space of viable code changes; your “best-in-class” types ultimately make AI *more* productive.

---

## 8. Concrete checklist

When you see a stubborn `Any` and want to be “best in class”, walk this checklist:

1. **Is this value at a boundary (IO, JSON, plugin, untyped library)?**

   * Yes → keep `Any` **there**, but introduce:

     * A **TypedDict/dataclass** for its shape, plus
     * A **validator or TypeGuard/TypeIs** + `cast` that converts to the structured type.

2. **Is this coming from an untyped library?**

   * Check for `types-<lib>` or stub packages (PEP 561). ([Python Enhancement Proposals (PEPs)][10])
   * If none, write stubs or a thin typed wrapper.

3. **Is this from a loose helper/decorator?**

   * Replace `Callable[..., Any]` with `Callable[P, R]` using `ParamSpec`/`TypeVar`. ([Python Enhancement Proposals (PEPs)][13])

4. **Is it a dict standing in for an object?**

   * Replace with `TypedDict` (or TypedDict + Protocol, if you want structural behavior). ([mypy Documentation][9])

5. **Do you just need “some type” but not operations?**

   * Use `object`, not `Any`.

6. **Have you configured the checker to care?**

   * Turn on `--disallow-any-*` (mypy) or `reportUnknown*` / `reportAny` (Pyright/BasedPyright). ([Ubuntu Manpages][7])

7. **Do you need advanced narrowing?**

   * Introduce TypeGuard/TypeIs for complex runtime checks. ([Python Enhancement Proposals (PEPs)][16])

If you’d like, I can next take a **real module** from your codebase that currently uses `Any` in annoying ways and rewrite it using this playbook — with concrete stubs, Protocols, type guards, and checker config tuned specifically for that module.

[1]: https://peps.python.org/pep-0484/?utm_source=chatgpt.com "PEP 484 – Type Hints"
[2]: https://microsoft.github.io/pyright/?utm_source=chatgpt.com "Pyright - Microsoft Open Source"
[3]: https://docs.basedpyright.com/v1.29.4/benefits-over-pyright/new-diagnostic-rules/?utm_source=chatgpt.com "new diagnostic rules"
[4]: https://github.com/python/mypy/issues/9153?utm_source=chatgpt.com "When using`--disallow-any-expr`, silence complaints about ..."
[5]: https://gdevops.frama.io/opsindev/tuto-project/software_quality/mypy/help/help.html?utm_source=chatgpt.com "mypy –help — Tuto project"
[6]: https://docs.python.org/3/library/typing.html?utm_source=chatgpt.com "typing — Support for type hints"
[7]: https://manpages.ubuntu.com/manpages/jammy/man1/mypy.1.html?utm_source=chatgpt.com "mypy - Optional static typing for Python"
[8]: https://stackoverflow.com/questions/38005633/pep-484-type-annotations-with-own-types?utm_source=chatgpt.com "PEP-484 Type Annotations with own types"
[9]: https://mypy.readthedocs.io/en/stable/typed_dict.html?utm_source=chatgpt.com "TypedDict - mypy 1.18.2 documentation"
[10]: https://peps.python.org/pep-0561/?utm_source=chatgpt.com "PEP 561 – Distributing and Packaging Type Information"
[11]: https://mypy.readthedocs.io/en/stable/installed_packages.html?utm_source=chatgpt.com "Using installed packages - mypy 1.18.2 documentation"
[12]: https://peps.python.org/pep-0544/?utm_source=chatgpt.com "PEP 544 – Protocols: Structural subtyping (static duck typing)"
[13]: https://peps.python.org/pep-0612/?utm_source=chatgpt.com "PEP 612 – Parameter Specification Variables"
[14]: https://stackoverflow.com/questions/51457563/what-does-typing-cast-do-in-python?utm_source=chatgpt.com "What does typing.cast do in Python?"
[15]: https://mypy.readthedocs.io/en/latest/type_narrowing.html?utm_source=chatgpt.com "Type narrowing - MyPy documentation - Read the Docs"
[16]: https://peps.python.org/pep-0647/?utm_source=chatgpt.com "PEP 647 – User-Defined Type Guards"
[17]: https://rednafi.com/python/typeguard-vs-typeis/?utm_source=chatgpt.com "TypeIs does what I thought TypeGuard would do in Python"
[18]: https://mypy.readthedocs.io/en/stable/common_issues.html?utm_source=chatgpt.com "Common issues and solutions - mypy 1.18.2 documentation"
[19]: https://mypy.readthedocs.io/en/stable/error_code_list2.html?utm_source=chatgpt.com "Error codes for optional checks - mypy 1.18.2 documentation"
[20]: https://github.com/microsoft/pyright/discussions/7323?utm_source=chatgpt.com "Settings in pyrightconfig.json seemingly ignored #7323"
