
Below is a **third “Stage‑F call wiring” micro‑fixture** focused specifically on the next “dynamic‑ish but still statically tractable” Python call behaviors that materially improve CPG usefulness for LLMs:

* **classmethod / staticmethod binding**
* **bound vs unbound method calls**
* **descriptor access** (both **explicit** `.__get__(...)` calls and **implicit** descriptor invocation via attribute read, including the “obj is None” class‑access case)

This fixture is designed so an engineer can implement Stage‑F independently and still converge on identical outputs.

I’m assuming the same span/key discipline you’ve been using: **0‑based line/col**, `end_col` **exclusive**, columns measured in **UTF‑8 byte offsets** (this file is ASCII-only so byte==char). SCIP occurrence ranges are typically 0-based as well. 

---

## Fixture 3 source file

**Path (recommended):**
`tests/fixtures/call_wiring/f3_dynamicish_calls.py`

```python
# fixture 3: dynamic-ish but statically tractable calls
class K:
    CONST = 100

    def __init__(self, n: int):
        self.n = n

    def inst(self, x: int) -> int:
        return self.n + x

    @classmethod
    def cm(cls, x: int) -> int:
        return cls.CONST + x

    @staticmethod
    def sm(x: int) -> int:
        return x * 2

    @property
    def prop(self) -> int:
        return self.n * 10


class PlusOne:
    def __init__(self):
        pass

    def __get__(self, obj: "K", objtype=None) -> int:
        if obj is None:
            return 0
        return obj.n + 1


class HasDesc:
    def __init__(self):
        pass

    d = PlusOne()


def run():
    k = K(1)
    a = k.inst(10)
    b = K.inst(k, 11)
    c = K.cm(12)
    d = k.cm(13)
    e = K.sm(14)
    f = k.sm(15)
    g = k.prop
    h = HasDesc.d.__get__(k, HasDesc)
    i = HasDesc().d
    j = HasDesc.d
    return a, b, c, d, e, f, g, h, i, j
```

### Why these lines matter

This one file hits the “next hard tier” in Python call semantics:

1. **Bound instance method**: `k.inst(10)`
2. **Unbound method**: `K.inst(k, 11)`
3. **classmethod invoked via class**: `K.cm(12)`
4. **classmethod invoked via instance**: `k.cm(13)` (still binds `cls`, not `self`)
5. **staticmethod via class** and **via instance**: `K.sm(14)` / `k.sm(15)` (no implicit arg)
6. **property read**: `k.prop` (implicit descriptor get; best-in-class to model as getter call)
7. **explicit descriptor call**: `HasDesc.d.__get__(k, HasDesc)` (normal bound method call)
8. **implicit descriptor call (instance access)**: `HasDesc().d` (desugars to `PlusOne.__get__`)
9. **implicit descriptor call (class access)**: `HasDesc.d` (desugars to `PlusOne.__get__(obj=None, objtype=HasDesc)`)

---

## Canonical span keys for call sites in this fixture

Use your existing `span_key`/`call_id` convention; here’s a concrete, deterministic one:

```text
call_id := "{rel_path}:{start_line}:{start_col}-{end_line}:{end_col}"
```

For this exact file (and the path above), the relevant spans are:

```text
PlusOne()                           tests/fixtures/call_wiring/f3_dynamicish_calls.py:37:8-37:17
K(1)                                tests/fixtures/call_wiring/f3_dynamicish_calls.py:41:8-41:12
k.inst(10)                          tests/fixtures/call_wiring/f3_dynamicish_calls.py:42:8-42:18
K.inst(k, 11)                       tests/fixtures/call_wiring/f3_dynamicish_calls.py:43:8-43:21
K.cm(12)                            tests/fixtures/call_wiring/f3_dynamicish_calls.py:44:8-44:16
k.cm(13)                            tests/fixtures/call_wiring/f3_dynamicish_calls.py:45:8-45:16
K.sm(14)                            tests/fixtures/call_wiring/f3_dynamicish_calls.py:46:8-46:16
k.sm(15)                            tests/fixtures/call_wiring/f3_dynamicish_calls.py:47:8-47:16
k.prop                              tests/fixtures/call_wiring/f3_dynamicish_calls.py:48:8-48:14
HasDesc.d.__get__(k, HasDesc)       tests/fixtures/call_wiring/f3_dynamicish_calls.py:49:8-49:37
HasDesc()                           tests/fixtures/call_wiring/f3_dynamicish_calls.py:50:8-50:17
HasDesc().d                         tests/fixtures/call_wiring/f3_dynamicish_calls.py:50:8-50:19
HasDesc.d                           tests/fixtures/call_wiring/f3_dynamicish_calls.py:51:8-51:17
```

---

## Expected Stage‑F outputs (golden rows)

### Table 1: `cpg.call_targets` (expected)

**Design intent**: allow **multiple targets per call** (constructor = class target + init target; descriptor access = synthetic desugaring target).

Minimal columns that keep this deterministic and high-signal:

* `call_id` (PK component)
* `callee_qname` (PK component; normalized string)
* `target_role` (PK component; `"primary" | "init"`)
* `binding_kind` (how the binder interpreted the call)
* `origin` (`"syntax_call"` vs `"descriptor_desugar"`)

```python
expected_call_targets = [
  # PlusOne() in class body: treat as constructor (primary target) + init target.
  dict(call_id="...:37:8-37:17", callee_qname="PlusOne",          target_role="primary", binding_kind="constructor", origin="syntax_call"),
  dict(call_id="...:37:8-37:17", callee_qname="PlusOne.__init__", target_role="init",    binding_kind="init",        origin="syntax_call"),

  # K(1): constructor + init
  dict(call_id="...:41:8-41:12", callee_qname="K",               target_role="primary", binding_kind="constructor", origin="syntax_call"),
  dict(call_id="...:41:8-41:12", callee_qname="K.__init__",      target_role="init",    binding_kind="init",        origin="syntax_call"),

  # bound vs unbound instance method calls
  dict(call_id="...:42:8-42:18", callee_qname="K.inst",          target_role="primary", binding_kind="bound_method",  origin="syntax_call"),
  dict(call_id="...:43:8-43:21", callee_qname="K.inst",          target_role="primary", binding_kind="unbound_method", origin="syntax_call"),

  # classmethod: both call sites resolve to same function symbol
  dict(call_id="...:44:8-44:16", callee_qname="K.cm",            target_role="primary", binding_kind="classmethod", origin="syntax_call"),
  dict(call_id="...:45:8-45:16", callee_qname="K.cm",            target_role="primary", binding_kind="classmethod", origin="syntax_call"),

  # staticmethod: both call sites resolve to same function symbol; no implicit receiver
  dict(call_id="...:46:8-46:16", callee_qname="K.sm",            target_role="primary", binding_kind="staticmethod", origin="syntax_call"),
  dict(call_id="...:47:8-47:16", callee_qname="K.sm",            target_role="primary", binding_kind="staticmethod", origin="syntax_call"),

  # property read: best-in-class treat as descriptor get => getter call
  dict(call_id="...:48:8-48:14", callee_qname="K.prop",          target_role="primary", binding_kind="property_get", origin="descriptor_desugar"),

  # explicit descriptor call (a normal call-site to __get__)
  dict(call_id="...:49:8-49:37", callee_qname="PlusOne.__get__", target_role="primary", binding_kind="bound_method", origin="syntax_call"),

  # HasDesc(): constructor + init
  dict(call_id="...:50:8-50:17", callee_qname="HasDesc",         target_role="primary", binding_kind="constructor", origin="syntax_call"),
  dict(call_id="...:50:8-50:17", callee_qname="HasDesc.__init__",target_role="init",    binding_kind="init",        origin="syntax_call"),

  # implicit descriptor get on instance/class access => desugared to __get__
  dict(call_id="...:50:8-50:19", callee_qname="PlusOne.__get__", target_role="primary", binding_kind="descriptor_get", origin="descriptor_desugar"),
  dict(call_id="...:51:8-51:17", callee_qname="PlusOne.__get__", target_role="primary", binding_kind="descriptor_get", origin="descriptor_desugar"),
]
```

**Notes**

* The `"callee_qname"` string is intentionally **normalized** (you can also store raw SCIP symbol strings separately). SCIP symbols are verbose by design and encode tool/language/project/version; your ingestion can always keep them too. 
* In a real run, you can populate `"callee_qname"` by parsing SCIP symbol strings or using your existing crosswalk machinery (SCIP → GOID → qname).

---

### Table 2: `cpg.arg_to_param` (expected)

**Goal**: capture argument binding including implicit receiver binding for method/classmethod/property/descriptor.

Minimal deterministic columns:

* `call_id` (PK component)
* `arg_slot` (PK component; stable label)
* `param_name` (PK component)

Conventions used here:

* `arg_slot="positional:0"`, `positional:1`, …
* `arg_slot="implicit:receiver"` for implicit `self`/`cls`/descriptor object
* `arg_slot="implicit:objtype"` for descriptor `objtype`
* `arg_slot="implicit:none"` for descriptor class-access case (`obj=None`)

```python
expected_arg_to_param = [
  # K(1) binds to __init__(self, n) => expose only user args (constructor binder behavior)
  dict(call_id="...:41:8-41:12", arg_slot="positional:0", param_name="n"),

  # k.inst(10): implicit receiver -> self, positional:0 -> x
  dict(call_id="...:42:8-42:18", arg_slot="implicit:receiver", param_name="self"),
  dict(call_id="...:42:8-42:18", arg_slot="positional:0",      param_name="x"),

  # K.inst(k, 11): positional:0 -> self, positional:1 -> x
  dict(call_id="...:43:8-43:21", arg_slot="positional:0",       param_name="self"),
  dict(call_id="...:43:8-43:21", arg_slot="positional:1",       param_name="x"),

  # classmethod calls: implicit receiver -> cls, positional:0 -> x
  dict(call_id="...:44:8-44:16", arg_slot="implicit:receiver",  param_name="cls"),
  dict(call_id="...:44:8-44:16", arg_slot="positional:0",       param_name="x"),
  dict(call_id="...:45:8-45:16", arg_slot="implicit:receiver",  param_name="cls"),
  dict(call_id="...:45:8-45:16", arg_slot="positional:0",       param_name="x"),

  # staticmethod calls: only positional
  dict(call_id="...:46:8-46:16", arg_slot="positional:0",       param_name="x"),
  dict(call_id="...:47:8-47:16", arg_slot="positional:0",       param_name="x"),

  # property get: implicit receiver -> self
  dict(call_id="...:48:8-48:14", arg_slot="implicit:receiver",  param_name="self"),

  # explicit __get__(self, obj, objtype=None) called as HasDesc.d.__get__(k, HasDesc)
  dict(call_id="...:49:8-49:37", arg_slot="implicit:receiver",  param_name="self"),
  dict(call_id="...:49:8-49:37", arg_slot="positional:0",       param_name="obj"),
  dict(call_id="...:49:8-49:37", arg_slot="positional:1",       param_name="objtype"),

  # implicit descriptor get: HasDesc().d  ==> PlusOne.__get__(descriptor, obj, objtype)
  dict(call_id="...:50:8-50:19", arg_slot="implicit:receiver",  param_name="self"),     # descriptor object
  dict(call_id="...:50:8-50:19", arg_slot="implicit:obj",       param_name="obj"),      # obj = HasDesc() instance
  dict(call_id="...:50:8-50:19", arg_slot="implicit:objtype",   param_name="objtype"),  # objtype = HasDesc

  # implicit descriptor get: HasDesc.d  ==> PlusOne.__get__(descriptor, None, HasDesc)
  dict(call_id="...:51:8-51:17", arg_slot="implicit:receiver",  param_name="self"),
  dict(call_id="...:51:8-51:17", arg_slot="implicit:none",      param_name="obj"),
  dict(call_id="...:51:8-51:17", arg_slot="implicit:objtype",   param_name="objtype"),
]
```

**Best-in-class detail**: that last row-set is *the* important upgrade. It encodes that **class-level descriptor access is still a descriptor invocation**, with `obj=None` and `objtype=<owner class>`.

---

### Table 3: `cpg.ret_to_call` (expected)

Keep this minimal and purely about “return flows to call site”.

Columns:

* `call_id` (PK component)
* `callee_qname` (PK component)

```python
expected_ret_to_call = [
  # constructors: RET edge should point from class-call (instance result), not __init__
  dict(call_id="...:37:8-37:17", callee_qname="PlusOne"),
  dict(call_id="...:41:8-41:12", callee_qname="K"),
  dict(call_id="...:50:8-50:17", callee_qname="HasDesc"),

  # normal calls
  dict(call_id="...:42:8-42:18", callee_qname="K.inst"),
  dict(call_id="...:43:8-43:21", callee_qname="K.inst"),
  dict(call_id="...:44:8-44:16", callee_qname="K.cm"),
  dict(call_id="...:45:8-45:16", callee_qname="K.cm"),
  dict(call_id="...:46:8-46:16", callee_qname="K.sm"),
  dict(call_id="...:47:8-47:16", callee_qname="K.sm"),

  # descriptor gets (property + __get__)
  dict(call_id="...:48:8-48:14", callee_qname="K.prop"),
  dict(call_id="...:49:8-49:37", callee_qname="PlusOne.__get__"),
  dict(call_id="...:50:8-50:19", callee_qname="PlusOne.__get__"),
  dict(call_id="...:51:8-51:17", callee_qname="PlusOne.__get__"),
]
```

---

## What this fixture forces Stage‑F to get “right”

### 1) Binding kind must come from **definition decorators** + **call shape**

* `@classmethod`, `@staticmethod`, `@property` should be extracted from your syntax datasets (LibCST) and attached to symbol/function metadata.
* Then Stage‑F uses that to decide whether there is an **implicit receiver argument** and what it binds to.

### 2) Descriptor get requires a deliberate “desugaring” step

A best-in-class CPG for Python should not treat `k.prop` and `HasDesc().d` as “just attribute reads”; for LLM usage you want:

* a call target you can traverse (“this read invokes getter/descriptor logic”)
* parameter binding that makes control/data flow through that call explicit

### 3) Constructor calls should be modeled as “class-call return value”

If you tie `RET_TO_CALL` to `__init__`, you’ll encode the wrong semantics (since `__init__` returns `None`). Instead:

* treat `C(...)` as a call whose return is `instance_of(C)`
* optionally also emit a secondary init target (which we did) so you still retain init parameter binding and call graph visibility.

---

## Practical “how teams use this” guidance

* One engineer can implement the **classmethod/staticmethod/bound/unbound** binder.
* Another can implement the **descriptor desugaring** binder.
* Both converge because the expected outputs clearly differentiate:

  * `origin="syntax_call"` vs `origin="descriptor_desugar"`
  * `binding_kind` values
  * explicit vs implicit `ARG_TO_PARAM` rows.

---

## Where SCIP fits (and why the fixture is stable)

SCIP gives you (at minimum) **symbol occurrences with ranges** and a verbose unique symbol string. 
Even if you don’t want to compare raw SCIP symbols in golden tests, you can normalize them into `callee_qname` (module + qualname) while still keeping the original for traceability (SCIP symbols encode tool/language/project/version). 

If you *do* keep raw SCIP symbol strings, the doc’s examples illustrate the “tool language project version …” prefix you’ll see. 

---

## Drop-in golden expectation snippet (Arrow-friendly)

If your harness expects Arrow Tables, this is the minimal pattern:

```python
import pyarrow as pa

call_targets_tbl = pa.Table.from_pylist(expected_call_targets)
arg_to_param_tbl = pa.Table.from_pylist(expected_arg_to_param)
ret_to_call_tbl  = pa.Table.from_pylist(expected_ret_to_call)
```

(Then your existing Arrow comparison harness can schema-check → sort by PK → equality.)

---

### Reference docs you attached

* SCIP parsing + symbol/occurrence basics: 

If you want, I can also provide **one more micro‑fixture after this** that targets “descriptor writes” (i.e., `__set__` via attribute assignment) and `@property.setter`, which is the natural next step once descriptor *reads* are wired correctly.


## Micro‑fixture #4: Descriptor writes (`__set__` via attribute assignment + `@property.setter`)

This fixture is designed to make Stage‑F “call wiring” deterministic for **implicit calls created by attribute assignment**, so your interprocedural flow edges can treat:

* `obj.attr = rhs` as an **implicit call** to either:

  * a **data descriptor** `__set__`, or
  * a **property setter** (`@p.setter`)

…and then emit the same core interproc edges you already target (`ARG_TO_PARAM`, `RET_TO_CALL`). Your CPG doc’s call-wiring section defines these edges at a conceptual level; this fixture pins down concrete expected rows for two canonical “descriptor write” mechanisms.

---

## 1) Corpus file

Create **one** file so SCIP indexing stays tiny and stable:

### `tests/fixtures/cpg_descriptor_writes/write_basic.py`

```python
class LoggingDescriptor:
    def __set__(self, obj: "WithDescriptor", value: int) -> None:
        obj._d = value


class WithDescriptor:
    d: LoggingDescriptor = LoggingDescriptor()
    def __init__(self) -> None:
        self._d = 0


class WithProperty:
    def __init__(self) -> None:
        self._p = 0

    @property
    def p(self) -> int:
        return self._p

    @p.setter
    def p(self, value: int) -> None:
        self._p = value


def run() -> int:
    a = WithDescriptor()
    a.d = 7
    b = WithProperty()
    b.p = 42
    return a._d + b.p
```

### Why this exact corpus?

* `a.d = 7` is a **data-descriptor write** (the class of `d` defines `__set__`).
* `b.p = 42` is a **property setter write**.
* Both are inside `run()` so you get a consistent enclosing function scope and clean spans.
* Type hints make “best effort” inference (SCIP/pyright) less ambiguous without requiring any runtime modeling.

---

## 2) Span anchors (0‑based line/col)

These anchors are what you use to locate the exact syntax nodes (callsite / args / params) in produced tables via joins to your `spans.*` datasets.

### Callsite anchors (Attribute nodes on LHS)

| Construct                 | Line | Col start | Col end | Text  |
| ------------------------- | ---: | --------: | ------: | ----- |
| descriptor write callsite |   26 |         4 |       7 | `a.d` |
| property write callsite   |   28 |         4 |       7 | `b.p` |

### Arg anchors

| Construct         | Line | Col start | Col end | Text |
| ----------------- | ---: | --------: | ------: | ---- |
| recv (descriptor) |   26 |         4 |       5 | `a`  |
| rhs (descriptor)  |   26 |        10 |      11 | `7`  |
| recv (property)   |   28 |         4 |       5 | `b`  |
| rhs (property)    |   28 |        10 |      12 | `42` |

### Callee def + param anchors

**`LoggingDescriptor.__set__` definition:**

* def name `__set__`: line **1**, col **8–15**
* param `obj`: line **1**, col **22–25**
* param `value`: line **1**, col **45–50**

**`WithProperty.p` setter definition:**

* setter def name `p`: line **20**, col **8–9**
* param `self`: line **20**, col **10–14**
* param `value`: line **20**, col **16–21**

---

## 3) Normalization rule this fixture enforces

### 3.1 What counts as the “callsite node” for descriptor writes?

For this fixture, define:

> **callsite_node_id = the syntax node for the LHS `Attribute`** (e.g., `a.d` / `b.p`), not the whole `Assign` statement.

Rationale:

* It is stable across formatting changes that don’t alter the attribute expression.
* It aligns with how you already treat attribute accesses as first-class facts (`core.attribute_accesses` exists explicitly in your best‑in‑class plan).

### 3.2 How to lower these to “call wiring” (best pragmatic choice)

You need two dispatch kinds:

1. **`property_set`**
   Lower `b.p = 42` to an implicit call to the **setter function def** (`@p.setter def p(self, value)`).

2. **`descriptor_set`**
   Lower `a.d = 7` to an implicit call to `LoggingDescriptor.__set__`, but treat the descriptor receiver (`self`) as **implicitly bound**.

Concretely, for `descriptor_set`, this fixture expects you to wire only the *semantically useful* arguments:

* `a` flows to param `obj`
* `7` flows to param `value`

…and you **do not** have to create an ARG edge into the descriptor’s `self` param to pass this fixture.

That gives you deterministic, high-value def-use and interproc flow immediately, without requiring synthetic “descriptor instance” expression nodes.

> If you later want true best‑in‑class descriptor-state tracking, add a P1 enhancement that also wires a synthetic “descriptor instance” node into `self` (but do **not** include it in this fixture’s golden rows yet—otherwise you’ll block incremental rollout).

---

## 4) Golden expectations (semantic rows)

This is the heart of the deliverable: expected rows for the **Stage‑F outputs**.

I’m expressing them in a “semantic projection” form (path + spans + names), because it avoids hardcoding internal hashed IDs while still being byte-for-byte deterministic once you project & sort.

### 4.1 `cpg.call_targets_v1` (projected)

Expected **two** call-target rows:

```python
EXPECTED_CALL_TARGETS = [
    {
        "rel_path": "tests/fixtures/cpg_descriptor_writes/write_basic.py",
        "callsite_start_line_0": 26,
        "callsite_start_col_0": 4,
        "callsite_end_line_0": 26,
        "callsite_end_col_0": 7,
        "dispatch_kind": "descriptor_set",
        "callee_qname": "LoggingDescriptor.__set__",
        "callee_def_start_line_0": 1,
        "callee_def_start_col_0": 8,
        "callee_def_end_line_0": 1,
        "callee_def_end_col_0": 15,
    },
    {
        "rel_path": "tests/fixtures/cpg_descriptor_writes/write_basic.py",
        "callsite_start_line_0": 28,
        "callsite_start_col_0": 4,
        "callsite_end_line_0": 28,
        "callsite_end_col_0": 7,
        "dispatch_kind": "property_set",
        "callee_qname": "WithProperty.p.setter",   # see note below
        "callee_def_start_line_0": 20,
        "callee_def_start_col_0": 8,
        "callee_def_end_line_0": 20,
        "callee_def_end_col_0": 9,
    },
]
```

**Note on `callee_qname` for setters:**
If your system represents setter defs as `WithProperty.p` with an attached decorator fact `decorator_kind="property_setter"`, then in the projection you should normalize to `"WithProperty.p.setter"` so the golden rows stay explicit about *which* `p` def you targeted (getter vs setter).

### 4.2 `cpg.arg_to_param_v1` (projected)

Expected **four** ARG_TO_PARAM edges:

```python
EXPECTED_ARG_TO_PARAM = [
    # a.d = 7  --> LoggingDescriptor.__set__(obj, value)   (self is bound/implicit)
    {
        "rel_path": "tests/fixtures/cpg_descriptor_writes/write_basic.py",
        "callsite_line_0": 26,
        "callsite_col0": 4,
        "callsite_col1": 7,
        "callee_qname": "LoggingDescriptor.__set__",
        "arg_start_line_0": 26,
        "arg_start_col_0": 4,
        "arg_end_line_0": 26,
        "arg_end_col_0": 5,
        "param_name": "obj",
        "param_start_line_0": 1,
        "param_start_col_0": 22,
        "param_end_line_0": 1,
        "param_end_col_0": 25,
    },
    {
        "rel_path": "tests/fixtures/cpg_descriptor_writes/write_basic.py",
        "callsite_line_0": 26,
        "callsite_col0": 4,
        "callsite_col1": 7,
        "callee_qname": "LoggingDescriptor.__set__",
        "arg_start_line_0": 26,
        "arg_start_col_0": 10,
        "arg_end_line_0": 26,
        "arg_end_col_0": 11,
        "param_name": "value",
        "param_start_line_0": 1,
        "param_start_col_0": 45,
        "param_end_line_0": 1,
        "param_end_col_0": 50,
    },

    # b.p = 42  --> WithProperty.p.setter(self, value)
    {
        "rel_path": "tests/fixtures/cpg_descriptor_writes/write_basic.py",
        "callsite_line_0": 28,
        "callsite_col0": 4,
        "callsite_col1": 7,
        "callee_qname": "WithProperty.p.setter",
        "arg_start_line_0": 28,
        "arg_start_col_0": 4,
        "arg_end_line_0": 28,
        "arg_end_col_0": 5,
        "param_name": "self",
        "param_start_line_0": 20,
        "param_start_col_0": 10,
        "param_end_line_0": 20,
        "param_end_col_0": 14,
    },
    {
        "rel_path": "tests/fixtures/cpg_descriptor_writes/write_basic.py",
        "callsite_line_0": 28,
        "callsite_col0": 4,
        "callsite_col1": 7,
        "callee_qname": "WithProperty.p.setter",
        "arg_start_line_0": 28,
        "arg_start_col_0": 10,
        "arg_end_line_0": 28,
        "arg_end_col_0": 12,
        "param_name": "value",
        "param_start_line_0": 20,
        "param_start_col_0": 16,
        "param_end_line_0": 20,
        "param_end_col_0": 21,
    },
]
```

### 4.3 `cpg.ret_to_call_v1`

Expected: **no rows** for these two callsites, because assignment is a statement and you should not manufacture a call-result expression node for setter calls in Stage‑F (keep it minimal and deterministic for now).

---

## 5) Comparison harness snippet (Polars → stable projection → Arrow equality)

This is a representative test skeleton. It assumes you already write Parquet/Arrow datasets per dataset key (as per your build layer conventions).

### `tests/golden/test_descriptor_writes_call_wiring.py` (representative snippet)

```python
from __future__ import annotations

import polars as pl

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _span_key(prefix: str) -> pl.Expr:
    # Creates a stable, sortable key from span columns
    return pl.concat_str(
        [
            pl.col("rel_path"),
            pl.lit(":"),
            pl.col(f"{prefix}_start_line_0").cast(pl.Utf8),
            pl.lit(":"),
            pl.col(f"{prefix}_start_col_0").cast(pl.Utf8),
            pl.lit("-"),
            pl.col(f"{prefix}_end_line_0").cast(pl.Utf8),
            pl.lit(":"),
            pl.col(f"{prefix}_end_col_0").cast(pl.Utf8),
        ]
    )

def assert_frame_equal_sorted(got: pl.DataFrame, exp: pl.DataFrame, sort_cols: list[str]) -> None:
    g = got.sort(sort_cols)
    e = exp.sort(sort_cols)
    if g.shape != e.shape or g.to_dicts() != e.to_dicts():
        raise AssertionError(f"\nGOT:\n{g}\n\nEXPECTED:\n{e}")

# ---------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------

def test_descriptor_writes_call_wiring(dataset_root: str) -> None:
    # Load stage outputs
    call_targets = pl.scan_parquet(f"{dataset_root}/cpg.call_targets_v1/**/*.parquet")
    arg_to_param = pl.scan_parquet(f"{dataset_root}/cpg.arg_to_param_v1/**/*.parquet")

    # Load lookup tables (names may differ in your repo — adjust to your canonical keys)
    spans_nodes = pl.scan_parquet(f"{dataset_root}/spans.syntax_nodes_v1/**/*.parquet")
    goids = pl.scan_parquet(f"{dataset_root}/core.goids_v1/**/*.parquet")
    params = pl.scan_parquet(f"{dataset_root}/core.function_parameters_v1/**/*.parquet")

    # ------------------------------------------------------------
    # call_targets projection
    # ------------------------------------------------------------
    got_ct = (
        call_targets
        .join(spans_nodes, left_on="callsite_node_id", right_on="node_id", how="left")
        .join(goids, left_on="callee_goid_h128", right_on="goid_h128", how="left")
        .filter(pl.col("rel_path") == "tests/fixtures/cpg_descriptor_writes/write_basic.py")
        .select([
            pl.col("rel_path"),
            pl.col("start_line_0").alias("callsite_start_line_0"),
            pl.col("start_col_0").alias("callsite_start_col_0"),
            pl.col("end_line_0").alias("callsite_end_line_0"),
            pl.col("end_col_0").alias("callsite_end_col_0"),
            pl.col("dispatch_kind"),
            pl.col("qname").alias("callee_qname"),
            pl.col("def_start_line_0").alias("callee_def_start_line_0"),
            pl.col("def_start_col_0").alias("callee_def_start_col_0"),
            pl.col("def_end_line_0").alias("callee_def_end_line_0"),
            pl.col("def_end_col_0").alias("callee_def_end_col_0"),
        ])
        .collect()
    )

    exp_ct = pl.DataFrame(EXPECTED_CALL_TARGETS)
    assert_frame_equal_sorted(
        got_ct,
        exp_ct,
        sort_cols=["rel_path", "callsite_start_line_0", "callsite_start_col_0", "dispatch_kind", "callee_qname"],
    )

    # ------------------------------------------------------------
    # arg_to_param projection
    # ------------------------------------------------------------
    got_atp = (
        arg_to_param
        .join(spans_nodes.select(["node_id","rel_path","start_line_0","start_col_0","end_line_0","end_col_0"]),
              left_on="callsite_node_id", right_on="node_id", how="left")
        .rename({
            "start_line_0": "callsite_line_0",
            "start_col_0": "callsite_col0",
            "end_col_0": "callsite_col1",
        })
        .join(goids.select(["goid_h128","qname"]), left_on="callee_goid_h128", right_on="goid_h128", how="left")
        .rename({"qname": "callee_qname"})
        .join(spans_nodes.select(["node_id","start_line_0","start_col_0","end_line_0","end_col_0"]),
              left_on="arg_node_id", right_on="node_id", how="left")
        .rename({
            "start_line_0": "arg_start_line_0",
            "start_col_0": "arg_start_col_0",
            "end_line_0": "arg_end_line_0",
            "end_col_0": "arg_end_col_0",
        })
        .join(params.select(["param_node_id","param_name"]), left_on="param_node_id", right_on="param_node_id", how="left")
        .join(spans_nodes.select(["node_id","start_line_0","start_col_0","end_line_0","end_col_0"]),
              left_on="param_node_id", right_on="node_id", how="left")
        .rename({
            "start_line_0": "param_start_line_0",
            "start_col_0": "param_start_col_0",
            "end_line_0": "param_end_line_0",
            "end_col_0": "param_end_col_0",
        })
        .filter(pl.col("rel_path") == "tests/fixtures/cpg_descriptor_writes/write_basic.py")
        .select([
            "rel_path",
            "callsite_line_0", "callsite_col0", "callsite_col1",
            "callee_qname",
            "arg_start_line_0", "arg_start_col_0", "arg_end_line_0", "arg_end_col_0",
            "param_name",
            "param_start_line_0", "param_start_col_0", "param_end_line_0", "param_end_col_0",
        ])
        .collect()
    )

    exp_atp = pl.DataFrame(EXPECTED_ARG_TO_PARAM)
    assert_frame_equal_sorted(
        got_atp,
        exp_atp,
        sort_cols=["rel_path", "callsite_line_0", "callsite_col0", "callee_qname", "param_name", "arg_start_col_0"],
    )
```

This is intentionally:

* resilient to column-order drift,
* anchored on spans (not internal hash IDs),
* deterministic once spans are deterministic.

---

## 6) SCIP fixture note (recommended for stability)

To keep this micro-corpus deterministic across machines/versions, you should **check in a generated `index.scip`** for this fixture directory (generated by your pinned scip-python/pyright toolchain). That prevents failures from upstream symbol formatting or inference drift.

---

## 7) What this fixture will catch (the intended failure modes)

This micro-fixture is small but “sharp.” It will catch:

1. **Setter target confusion**: choosing the getter `def p(self)` instead of the setter `def p(self, value)` for `b.p = 42`.
2. **Missing implicit calls**: failing to create any call wiring edges for assignment-based descriptor writes.
3. **Wrong param mapping**:

   * mapping `a` to `self` in `__set__` instead of to `obj`
   * mapping `7/42` to the wrong parameter
4. **Over-eager RET_TO_CALL**: manufacturing return-flow edges where the syntax has no call-result use site.

---

If you want, the next “hard but very high value” extension after this fixture is: **augmented assignment** (`obj.attr += 1`) which should produce both a descriptor read (`__get__`) and write (`__set__`) and should be validated as *two* implicit calls with correct arg/param mapping.


## Micro‑fixture: Augmented assignment on a data descriptor (`obj.attr += 1`) must synthesize **two implicit calls** (`__get__`, `__set__`)

This fixture is meant to lock in the *highest-value semantic expectation* for Stage‑F call wiring: an `AugAssign` to an attribute that resolves to a **data descriptor** must be represented as:

1. an implicit **descriptor read** call to `D.__get__(self=<descriptor>, obj=<receiver>, objtype=<type(receiver)>)`
2. an implicit **descriptor write** call to `D.__set__(self=<descriptor>, obj=<receiver>, value=<old_value ⊕ rhs>)`

…and the second call’s `value` argument should be *provably derived* from the first call’s return + the RHS.

> **Testing note (important):** avoid hard-coding raw SCIP symbol strings in goldens because they embed project identity and package/version components; keep those stable or compare via your own `def_key`/qualname normalization layer instead.

---

## 1) Fixture file (single file, ASCII-only, byte offsets == char offsets)

Create:

`tests/fixtures/cpg_stage_f/augassign_descriptor.py`

```python
class D:
    def __get__(self, obj, objtype=None):
        return 0

    def __set__(self, obj, value):
        pass


class C:
    x = D()


def f(c: C) -> None:
    c.x += 1
```

### Span anchors you should use (UTF‑8 byte offsets, 0‑based, end‑exclusive)

These offsets are computed for the exact file content above:

* The statement line is: `    c.x += 1\n`
* Global byte span for the **attribute expression** `c.x`:

  * `attr_span_start_byte = 167`
  * `attr_span_end_byte   = 170`
* Global byte span for the **whole augassign statement** `c.x += 1` (excluding indent/newline):

  * `stmt_span_start_byte = 167`
  * `stmt_span_end_byte   = 175`
* Subspans you’ll use for argument identity:

  * receiver object token `c`: `[167, 168)`
  * attribute name token `x`: `[169, 170)`
  * RHS literal `1`: `[174, 175)`
  * the annotation `C` in `def f(c: C)`: `[151, 152)` (useful for a deterministic `objtype` anchor)

---

## 2) Expected Stage‑F outputs (golden expectations)

### Canonical callsite keys (make the two implicit calls unambiguous)

I strongly recommend your callsite PK include a **call kind** discriminator so GET vs SET do not collide even if spans overlap.

Use these stable IDs in the test:

```python
PATH = "tests/fixtures/cpg_stage_f/augassign_descriptor.py"

CALL_GET = f"{PATH}@167:170#IMPLICIT_DESCRIPTOR_GET"
CALL_SET = f"{PATH}@167:175#IMPLICIT_DESCRIPTOR_SET_AUGASSIGN"
```

### 2.1 Expected `cpg.call_targets` rows (filtered subset)

Your physical table may have different names/columns; the key is that Stage‑F must be able to produce (or be joined into) this semantic view:

```python
expected_call_targets = [
  {
    "callsite_key": CALL_GET,
    "file_path": PATH,
    "span_start_byte": 167,
    "span_end_byte": 170,
    "implicit_kind": "descriptor_get",
    "augop": "+=",
    # compare via def_key/qualname layer (don’t bake full SCIP symbol strings)
    "callee_def_key": f"py:{PATH}#D.__get__",
  },
  {
    "callsite_key": CALL_SET,
    "file_path": PATH,
    "span_start_byte": 167,
    "span_end_byte": 175,
    "implicit_kind": "descriptor_set_augassign",
    "augop": "+=",
    "callee_def_key": f"py:{PATH}#D.__set__",
  },
]
```

**Non-negotiable semantics:**

* exactly **one** target for the GET implicit call → `D.__get__`
* exactly **one** target for the SET implicit call → `D.__set__`

(If later you model `int.__iadd__` / `int.__add__` too, that’s fine, but this golden should continue filtering to `implicit_kind in {descriptor_get, descriptor_set_augassign}`.)

---

### 2.2 Expected `cpg.arg_to_param` rows (core of the augmented assignment correctness)

This is the heart of the fixture: the argument/parameter mapping must reflect descriptor protocol + augassign lowering.

```python
expected_arg_to_param = [
  # --- GET call: D.__get__(self, obj, objtype) ---
  {
    "callsite_key": CALL_GET,
    "arg_index": 0,
    "arg_role": "receiver",
    # descriptor instance is not explicit; anchor to the attribute token `x`
    "arg_span_start_byte": 169,
    "arg_span_end_byte": 170,
    "param_name": "self",
  },
  {
    "callsite_key": CALL_GET,
    "arg_index": 1,
    "arg_role": "positional",
    "arg_span_start_byte": 167,   # `c`
    "arg_span_end_byte": 168,
    "param_name": "obj",
  },
  {
    "callsite_key": CALL_GET,
    "arg_index": 2,
    "arg_role": "positional",
    "arg_expr_kind": "TypeOf(receiver)",
    # deterministic anchor: the annotation `C` in def f(c: C)
    "arg_span_start_byte": 151,
    "arg_span_end_byte": 152,
    "param_name": "objtype",
  },

  # --- SET call: D.__set__(self, obj, value) ---
  {
    "callsite_key": CALL_SET,
    "arg_index": 0,
    "arg_role": "receiver",
    "arg_span_start_byte": 169,
    "arg_span_end_byte": 170,
    "param_name": "self",
  },
  {
    "callsite_key": CALL_SET,
    "arg_index": 1,
    "arg_role": "positional",
    "arg_span_start_byte": 167,   # `c`
    "arg_span_end_byte": 168,
    "param_name": "obj",
  },
  {
    "callsite_key": CALL_SET,
    "arg_index": 2,
    "arg_role": "positional",
    # this is the *key* augmented-assignment requirement:
    # value must be derived from the GET result + RHS
    "arg_expr_kind": "AugAssignValue",
    "augop": "+=",
    "base_read_callsite_key": CALL_GET,   # MUST reference the synthesized GET call
    "rhs_span_start_byte": 174,           # literal `1`
    "rhs_span_end_byte": 175,
    "param_name": "value",
  },
]
```

**Why this representation is “best-in-class” for LLM consumption:**

* It makes the `+=` *explicitly* depend on a property/descriptor read + write.
* It preserves the “sugared” structure (`base_read_callsite_key`, `rhs_span`) so you can reconstruct source-level intent, while still wiring to real callee defs (`D.__get__`, `D.__set__`).
* It avoids inventing a fake AST “binary op node” if you don’t want one; you can still add that later in Stage‑D DDG as a refinement without breaking Stage‑F.

---

### 2.3 Expected `cpg.ret_to_call` rows (GET return is the “old value” for augassign)

```python
expected_ret_to_call = [
  {
    "callsite_key": CALL_GET,
    "callee_def_key": f"py:{PATH}#D.__get__",
    "ret_index": 0,
    # the “value of c.x” is what the read yields; anchor to `c.x`
    "call_result_span_start_byte": 167,
    "call_result_span_end_byte": 170,
  }
]
```

You don’t need a `RET_TO_CALL` for `__set__` (it returns `None`), unless your graph chooses to represent explicit `None` returns.

---

## 3) How to assert this with your Arrow/Polars golden harness (subset‑compare pattern)

Because your full DAG will likely emit other calls/edges (e.g., `D()` in `x = D()`), I recommend **subset golden assertions** for these micro‑fixtures:

* filter to `file_path == PATH`
* filter to `implicit_kind in {"descriptor_get", "descriptor_set_augassign"}`
* normalize symbols → `def_key` if needed (join through your `core.*defs*` table)
* then run your PK-sorting Arrow equality harness

Representative test skeleton (adapt names to your actual dataset keys):

```python
import polars as pl

PATH = "tests/fixtures/cpg_stage_f/augassign_descriptor.py"
CALL_GET = f"{PATH}@167:170#IMPLICIT_DESCRIPTOR_GET"
CALL_SET = f"{PATH}@167:175#IMPLICIT_DESCRIPTOR_SET_AUGASSIGN"

def load_and_normalize_call_targets(out_dir: str) -> pl.DataFrame:
    # 1) load produced
    ct = pl.scan_parquet(f"{out_dir}/cpg.call_targets.parquet")

    # 2) filter to the fixture + implicit calls only
    ct = (
        ct.filter(pl.col("file_path") == PATH)
          .filter(pl.col("implicit_kind").is_in(["descriptor_get", "descriptor_set_augassign"]))
    )

    # 3) if ct already has callee_def_key, you’re done.
    # otherwise: join callee_symbol -> def_key via your defs table:
    # defs = pl.scan_parquet(f"{out_dir}/core.defs.parquet").select(["def_key","scip_symbol"])
    # ct = ct.join(defs, left_on="callee_symbol", right_on="scip_symbol").with_columns(
    #     pl.col("def_key").alias("callee_def_key")
    # )

    return ct.collect()

def assert_augassign_descriptor_goldens(out_dir: str) -> None:
    produced_ct = load_and_normalize_call_targets(out_dir)

    expected_ct = pl.DataFrame(expected_call_targets)

    # now call your existing Arrow harness:
    # assert_arrow_table_equal_by_pk(produced_ct.to_arrow(), expected_ct.to_arrow())
```

---

## 4) Implementation constraint the fixture is enforcing (so engineers don’t “almost” implement it)

When you implement the actual Stage‑F logic, this fixture is forcing you to **lower**:

`c.x += 1`

into a structured pair:

* `t0 = implicit_call(descriptor_get, callee=D.__get__, args=[C.x, c, C])`
* `t1 = implicit_call(descriptor_set, callee=D.__set__, args=[C.x, c, AugAssignValue(base=t0, op="+=", rhs=1)])`

That `AugAssignValue(base=t0, ...)` is the key design trick: it’s compact, deterministic, easy to test, and keeps your DDG free to become more detailed later (you can expand it into explicit binary op/value nodes in Stage‑D without changing call wiring contracts).

---

If you want, I can also extend this same fixture pattern to cover the next two “real world” augmented assignment variants that often break naïve implementations:

* `obj.attr += other` where `other` is not a literal (so rhs is a name/span, not a constant)
* `obj.attr += 1` where the descriptor is resolved *through inheritance* (descriptor defined on base class, accessed on subclass instance)

…but the single-file micro‑fixture above is the tight, high-signal baseline that will catch 90% of implementation mistakes immediately.


Below is a **drop‑in “micro‑fixture extension”** that covers the two augmented‑assignment variants you called out:

1. `obj.attr += other` where RHS is **a name** (span must point at the identifier, not a constant)
2. `obj.attr += 1` where the descriptor is defined on a **base class** and accessed on a **subclass instance** (resolver must walk MRO / inherited attributes)

This is written in the *same fixture style as the earlier augmented‑assignment baseline*: a tiny Python source file + **small, deterministic expected rows** for the *Stage‑F call wiring* outputs:

* `call_targets`
* `arg_to_param`
* `ret_to_call`

I’m assuming your “call wiring” tables use **byte‑offset spans** (0‑based, end‑exclusive) and that **call identity** is derived from `(path, start_byte, end_byte, kind)` (or equivalent). If your exact column names differ, the mapping is straightforward—what matters is that the PK includes the call span + kind, and the callee identity is stable.

---

## Fixture: `fixtures/augassign/augassign_descriptor_variants.py`

**Important:** the byte offsets in the expectations below assume the file content matches *exactly* (including blank lines + indentation + trailing newline).

```python
from __future__ import annotations

from typing import Any

class Num:
    def __init__(self, v: int):
        self.v = v

    def __iadd__(self, other: int) -> Num:
        self.v = self.v + other
        return self

class Field:
    def __init__(self, name: str):
        self.name = name

    def __get__(self, obj: Any, objtype: type | None = None) -> Num:
        return obj.__dict__.get(self.name, Num(0))

    def __set__(self, obj: Any, value: Num) -> None:
        obj.__dict__[self.name] = value

class Box:
    x = Field("x")

def nonliteral_rhs() -> Num:
    b = Box()
    other = 3
    b.x += other
    return b.x

class Base:
    x = Field("x")

class Sub(Base):
    pass

def inherited_descriptor() -> Num:
    s = Sub()
    s.x += 1
    return s.x
```

### Why this exact shape?

* Uses a **real descriptor** (`Field.__get__` / `Field.__set__`) so the augmented assignment must lower into the descriptor protocol.
* Uses **inheritance** (`Sub(Base)`) so the attribute resolution must **look beyond `Sub.__dict__`**.
* Uses RHS as both:

  * an identifier (`other`) → validates *span correctness for nonliteral RHS*
  * a literal (`1`) → validates the “baseline” still works through inheritance
* Avoids an internal `+=` inside `Num.__iadd__` so you don’t accidentally create *another* augmented‑assignment lowering inside the method body.

---

## Canonical span anchors (byte offsets + line/col)

Path used in IDs below:

```text
fixtures/augassign/augassign_descriptor_variants.py
```

### Nonliteral RHS call site: `b.x += other`

* `b.x` attribute span: **start=600, end=603** (line 29, col 4–7)
* `+=` operator span: **start=604, end=606** (line 29, col 8–10)
* RHS identifier `other`: **start=607, end=612** (line 29, col 11–16)
* object token `b`: **start=600, end=601** (line 29, col 4–5)

### Inherited descriptor call site: `s.x += 1`

* `s.x` attribute span: **start=741, end=744** (line 40, col 4–7)
* `+=` operator span: **start=745, end=747** (line 40, col 8–10)
* RHS literal `1`: **start=748, end=749** (line 40, col 11–12)
* object token `s`: **start=741, end=742** (line 40, col 4–5)

### Callee definition name spans (stable “definition anchors”)

These are *name token spans* for the methods—useful as a deterministic `callee_def_id` if you don’t want to depend on SCIP symbol strings in golden tests:

* `Num.__iadd__` name: **start=131, end=139** (line 9)
* `Field.__get__` name: **start=301, end=308** (line 17)
* `Field.__set__` name: **start=422, end=429** (line 20)

---

## Expected Stage‑F outputs

### 0) Normalized identifiers used in expected rows

I recommend encoding IDs exactly like this in the fixture expectations (even if your real tables store the decomposed columns), because it makes the golden rows readable and trivial to expand:

```python
PATH = "fixtures/augassign/augassign_descriptor_variants.py"

def call_id(start, end, kind):
    return f"{PATH}:{start}:{end}:{kind}"

def def_id(start, end):
    return f"{PATH}:{start}:{end}"

# Call IDs (nonliteral RHS)
CALL_B_GET  = call_id(600, 603, "DESCRIPTOR_GET")
CALL_B_IADD = call_id(604, 606, "AUGASSIGN_IADD")
CALL_B_SET  = call_id(600, 603, "DESCRIPTOR_SET")

# Call IDs (inherited descriptor)
CALL_S_GET  = call_id(741, 744, "DESCRIPTOR_GET")
CALL_S_IADD = call_id(745, 747, "AUGASSIGN_IADD")
CALL_S_SET  = call_id(741, 744, "DESCRIPTOR_SET")

# Callee anchors
DEF_GET  = def_id(301, 308)  # Field.__get__
DEF_SET  = def_id(422, 429)  # Field.__set__
DEF_IADD = def_id(131, 139)  # Num.__iadd__
```

If your pipeline already has a proper `symbol_id` (SCIP symbol string or internal symbol key), keep it too—**but the fixture shouldn’t require it** to be deterministic.

---

## 1) Expected `call_targets` rows

These assert two things:

* **augassign lowering produces 3 call sites** (GET → IADD → SET) per statement
* **descriptor call targets resolve through inheritance** (Sub instance still yields Field.**get**/**set**)

```python
EXPECTED_CALL_TARGETS = [
    # nonliteral_rhs: b.x += other
    {"call_id": CALL_B_GET,  "callee_def_id": DEF_GET,  "callee_qname": "Field.__get__",  "resolution": "descriptor"},
    {"call_id": CALL_B_IADD, "callee_def_id": DEF_IADD, "callee_qname": "Num.__iadd__",   "resolution": "operator_dunder"},
    {"call_id": CALL_B_SET,  "callee_def_id": DEF_SET,  "callee_qname": "Field.__set__",  "resolution": "descriptor"},

    # inherited_descriptor: s.x += 1  (descriptor defined on Base, accessed on Sub instance)
    {"call_id": CALL_S_GET,  "callee_def_id": DEF_GET,  "callee_qname": "Field.__get__",  "resolution": "descriptor"},
    {"call_id": CALL_S_IADD, "callee_def_id": DEF_IADD, "callee_qname": "Num.__iadd__",   "resolution": "operator_dunder"},
    {"call_id": CALL_S_SET,  "callee_def_id": DEF_SET,  "callee_qname": "Field.__set__",  "resolution": "descriptor"},
]
```

**Best‑in‑class requirement encoded here:** the resolver must walk from `Sub` → `Base` to find attribute `x` (descriptor instance) and then bind calls to `Field.__get__` / `Field.__set__`.

---

## 2) Expected `arg_to_param` rows

This is where the two “break naïve implementations” requirements are enforced:

### Variant A: RHS is a name

The RHS mapping must reference the **span for the identifier `other` at the call site** (607–612), not the constant assignment `other = 3`, not the parameter name, etc.

### Variant B: descriptor through inheritance

The GET/SET “obj” argument must reference the **Sub instance token `s`** and still resolve to the descriptor methods.

```python
EXPECTED_ARG_TO_PARAM = [
    # ---------- nonliteral_rhs: b.x += other ----------
    # GET: Field.__get__(obj=b, ...)
    {"call_id": CALL_B_GET, "param": "obj",   "arg_kind": "SPAN",        "arg_start": 600, "arg_end": 601, "arg_text": "b"},

    # IADD: Num.__iadd__(self=<get_result>, other=other)
    {"call_id": CALL_B_IADD,"param": "self",  "arg_kind": "CALL_RESULT", "arg_call_id": CALL_B_GET},
    {"call_id": CALL_B_IADD,"param": "other", "arg_kind": "SPAN",        "arg_start": 607, "arg_end": 612, "arg_text": "other"},

    # SET: Field.__set__(obj=b, value=<iadd_result>)
    {"call_id": CALL_B_SET, "param": "obj",   "arg_kind": "SPAN",        "arg_start": 600, "arg_end": 601, "arg_text": "b"},
    {"call_id": CALL_B_SET, "param": "value", "arg_kind": "CALL_RESULT", "arg_call_id": CALL_B_IADD},

    # ---------- inherited_descriptor: s.x += 1 ----------
    # GET: Field.__get__(obj=s, ...)  (descriptor is inherited, but obj arg is still 's')
    {"call_id": CALL_S_GET, "param": "obj",   "arg_kind": "SPAN",        "arg_start": 741, "arg_end": 742, "arg_text": "s"},

    # IADD: Num.__iadd__(self=<get_result>, other=1)
    {"call_id": CALL_S_IADD,"param": "self",  "arg_kind": "CALL_RESULT", "arg_call_id": CALL_S_GET},
    {"call_id": CALL_S_IADD,"param": "other", "arg_kind": "SPAN",        "arg_start": 748, "arg_end": 749, "arg_text": "1"},

    # SET: Field.__set__(obj=s, value=<iadd_result>)
    {"call_id": CALL_S_SET, "param": "obj",   "arg_kind": "SPAN",        "arg_start": 741, "arg_end": 742, "arg_text": "s"},
    {"call_id": CALL_S_SET, "param": "value", "arg_kind": "CALL_RESULT", "arg_call_id": CALL_S_IADD},
]
```

**Notes on conventions (so this remains unambiguous):**

* `CALL_RESULT` is *not* a source span; it’s an edge to another call’s output node. If your graph model uses `ssa_id`, `value_id`, or `expr_id` instead, substitute `arg_call_id` with that ID.
* I’m explicitly **not** requiring you to model the descriptor instance (`Field("x")`) as an explicit `self` argument to `__get__`/`__set__`. If you *do* model it, keep it as an additional row (don’t remove the `obj`/`value` rows).

---

## 3) Expected `ret_to_call` rows

These rows enforce that return flow is wired (needed for DDG welding and later Stage‑F “parameter flow” edges).

```python
EXPECTED_RET_TO_CALL = [
    # Field.__get__ returns the value consumed by IADD
    {"call_id": CALL_B_GET,  "callee_def_id": DEF_GET,  "ret": "return"},
    {"call_id": CALL_S_GET,  "callee_def_id": DEF_GET,  "ret": "return"},

    # Num.__iadd__ returns the value consumed by SET
    {"call_id": CALL_B_IADD, "callee_def_id": DEF_IADD, "ret": "return"},
    {"call_id": CALL_S_IADD, "callee_def_id": DEF_IADD, "ret": "return"},

    # (Optional) Field.__set__ returns None; include only if your schema records void returns
    # {"call_id": CALL_B_SET,  "callee_def_id": DEF_SET,  "ret": "return"},
    # {"call_id": CALL_S_SET,  "callee_def_id": DEF_SET,  "ret": "return"},
]
```

---

## How to use this fixture without pulling in unrelated calls

This file contains other calls (e.g., `Box()` / `Sub()` / `dict.get` / `Num(0)`). You **should not** assert the full call graph for the whole file in this micro‑fixture—assert only the subset relevant to augmented assignment lowering.

A very standard pattern is:

1. Load your produced Arrow table for `call_targets` / `arg_to_param` / `ret_to_call`
2. Filter to `call_kind in {DESCRIPTOR_GET, AUGASSIGN_IADD, DESCRIPTOR_SET}`
3. Filter to `call_start in {600,604,741,745}` (or to the exact `call_id`s)
4. Compare with the expected rows using your existing Arrow schema/PK aware harness

Representative snippet (adapt column names as needed):

```python
import polars as pl
import pyarrow as pa

RELEVANT_CALL_IDS = {
    CALL_B_GET, CALL_B_IADD, CALL_B_SET,
    CALL_S_GET, CALL_S_IADD, CALL_S_SET,
}

def filter_relevant(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(pl.col("call_id").is_in(sorted(RELEVANT_CALL_IDS)))

# produced = pl.from_arrow(pa_table_from_pipeline)
# produced = filter_relevant(produced)
# expected = pl.DataFrame(EXPECTED_CALL_TARGETS)
# compare_with_arrow_harness(expected, produced)  # your “PK-from-schema.metadata” harness
```

---

## What this fixture forces you to implement (so it “fails loudly”)

If this test fails, it very specifically points to one of these concrete implementation gaps:

### A) AugAssign lowering (LHS attribute + `+=`)

You must synthesize the triple:

* `DESCRIPTOR_GET` at LHS attribute span
* `AUGASSIGN_IADD` at operator span
* `DESCRIPTOR_SET` at LHS attribute span

…and connect them by value flow (`get.result → iadd.self`, `iadd.result → set.value`).

### B) RHS span fidelity

Your node extraction for RHS must take the **RHS expression span** from the source AST (here: identifier `other`), not from inferred values/constants.

### C) Descriptor resolution through inheritance

When receiver type is `Sub`, attribute `x` must resolve by searching:

* `Sub.__dict__`
* then base classes in MRO (`Base.__dict__`)
* then fall back if not found

Only then can you decide whether this is a descriptor (`__get__`/`__set__`) vs plain attribute.

---

If you want, I can also provide a **fourth micro‑fixture** that targets the *next* “augassign realism” failure mode after these two:

* `obj.attr += other` where `other` is itself an attribute or subscript (so RHS is a compound span), and
* `obj.attr += 1` where the descriptor is defined on a base class **and overridden by a non‑descriptor** on the subclass (to force correct shadowing rules).


## Micro‑fixture 04: AugAssign realism — compound RHS spans + descriptor shadowing

This fixture is designed to catch two “real world” failure modes that commonly slip through early Stage‑F call wiring implementations:

1. **`obj.attr += other` where `other` is a compound expression** (attribute chain or subscript) — your `ARG_TO_PARAM` must bind the *entire* RHS span, not just the base name.

2. **Descriptor shadowing across inheritance** — base defines a descriptor, subclass overrides with a non‑descriptor, and your resolver must **not** emit implicit descriptor calls for the subclass instance.

The fixture is intentionally small but forces correctness in:

* **implicit descriptor get/set call creation** for `AugAssign(Attribute(...))`,
* **operator call modeling** as an implicit call (here: `Box.__iadd__`),
* **wiring**: `RET_TO_CALL` and `ARG_TO_PARAM` chaining across the implicit call sequence,
* **shadowing** rules (MRO “first hit wins” for name binding).

---

# A) Fixture source file

Create a single file:

`fixtures/cpg_stageF_augassign_realism_04/augassign_realism_04.py`

```python
# augassign_realism_04.py
# Purpose:
#   - A1..A4: LHS is a data-descriptor -> must model __get__ + __set__ around "+="
#   - RHS is compound (attribute chain / subscript) -> ARG_TO_PARAM must bind full RHS span
#   - B1: base has descriptor but subclass shadows with non-descriptor -> MUST NOT model __get__/__set__

from __future__ import annotations


class Box:
    def __init__(self, v: int) -> None:
        self.v = v

    def __iadd__(self, other: int) -> "Box":
        self.v += other
        return self


class Desc:
    """
    Minimal data descriptor (has __get__ and __set__).
    We use Box as the descriptor value to make "+=" unambiguously dispatch to Box.__iadd__.
    """
    def __init__(self, name: str) -> None:
        self._slot = f"_{name}"

    def __get__(self, obj, objtype=None) -> Box:
        return getattr(obj, self._slot)

    def __set__(self, obj, value: Box) -> None:
        setattr(obj, self._slot, value)


class Holder:
    d = Desc("d")

    def __init__(self) -> None:
        self._d = Box(1)


class RHS:
    def __init__(self) -> None:
        self.delta = 2
        self.other = Holder()
        self.arr = [3, 4]
        self.idx = 0


def run() -> None:
    o = Holder()
    rhs = RHS()

    o.d += rhs.delta            # @A1  compound RHS: attribute
    o.d += rhs.arr[rhs.idx]     # @A2  compound RHS: subscript with non-literal index
    o.d += rhs.other._d.v       # @A3  compound RHS: attribute chain (more than 1 dot)
    o.d += rhs.arr[0]           # @A4  compound RHS: subscript with literal index

    class Base:
        d = Desc("d")
        def __init__(self) -> None:
            self._d = Box(10)

    class Shadow(Base):
        d = 0  # non-descriptor override; shadows Base.d entirely

    s = Shadow()
    s.d += 1                    # @B1  MUST NOT emit Desc.__get__/Desc.__set__ calls here
```

### Why these exact constructs?

* `Holder.d = Desc("d")` is statically recognizable as a descriptor instance (RHS is a constructor call; the type defines `__get__`+`__set__`).
* `Box.__iadd__` makes the “operator as call” deterministic and resolvable without guessing builtin semantics.
* `rhs.arr[rhs.idx]` forces your RHS span extraction to handle **nested spans** (subscript whose index is itself an attribute access).
* `Shadow.d = 0` forces correct **shadowing**: you must not “scan base classes for descriptors” after a same‑name subclass binding exists.

---

# B) Expected Stage‑F semantics (what must be emitted)

For each `@A*` line, Stage‑F should model the augmented assignment as this **implicit call chain**:

1. **Descriptor read**: `Desc.__get__(obj=o, objtype=type(o)) -> Box`
2. **In‑place add**: `Box.__iadd__(self=<ret of get>, other=<RHS expr>) -> Box`
3. **Descriptor write**: `Desc.__set__(obj=o, value=<ret of iadd>) -> None`

For `@B1`, because `Shadow.d` is a plain integer on the subclass:

* **No implicit `Desc.__get__` call**
* **No implicit `Desc.__set__` call**
* (Whether you model `int.__iadd__` is up to your operator modeling scope, but **descriptor calls must be absent**.)

---

# C) Golden expectations: the three tables you should assert

Below I’m giving the expectations in a **contract-style, ID-derivable** format that is resilient to column-order drift and doesn’t depend on internal numeric IDs.

## C1) Canonical IDs for the fixture (use in tests)

Use these deterministic IDs in your expected rows:

* `call_id = f"{path}::{anchor}::{call_kind}"`
* `value_id = f"{call_id}::ret"` for call result nodes you synthesize

Where:

* `path = "augassign_realism_04.py"`
* `anchor ∈ {"A1","A2","A3","A4","B1"}`
* `call_kind ∈ {"DESC_GET","IADD","DESC_SET"}`

For span keys, compute byte spans from file bytes (see harness in section D).

---

## C2) Expected `call_targets` rows (subset assertion)

Assert **only** the calls relevant to this fixture (descriptor + iadd). Don’t require “table equality” against all calls in the file if you expect other implicit calls to appear later (e.g., if you add `__getitem__` modeling in the future).

**Expected rows (12 total):**

```python
EXPECTED_CALL_TARGETS = [
  # A1
  {"call_id": "augassign_realism_04.py::A1::DESC_GET", "callee_qname": "Desc.__get__",  "kind": "IMPLICIT", "resolved": True},
  {"call_id": "augassign_realism_04.py::A1::IADD",     "callee_qname": "Box.__iadd__",  "kind": "IMPLICIT", "resolved": True},
  {"call_id": "augassign_realism_04.py::A1::DESC_SET", "callee_qname": "Desc.__set__",  "kind": "IMPLICIT", "resolved": True},

  # A2
  {"call_id": "augassign_realism_04.py::A2::DESC_GET", "callee_qname": "Desc.__get__",  "kind": "IMPLICIT", "resolved": True},
  {"call_id": "augassign_realism_04.py::A2::IADD",     "callee_qname": "Box.__iadd__",  "kind": "IMPLICIT", "resolved": True},
  {"call_id": "augassign_realism_04.py::A2::DESC_SET", "callee_qname": "Desc.__set__",  "kind": "IMPLICIT", "resolved": True},

  # A3
  {"call_id": "augassign_realism_04.py::A3::DESC_GET", "callee_qname": "Desc.__get__",  "kind": "IMPLICIT", "resolved": True},
  {"call_id": "augassign_realism_04.py::A3::IADD",     "callee_qname": "Box.__iadd__",  "kind": "IMPLICIT", "resolved": True},
  {"call_id": "augassign_realism_04.py::A3::DESC_SET", "callee_qname": "Desc.__set__",  "kind": "IMPLICIT", "resolved": True},

  # A4
  {"call_id": "augassign_realism_04.py::A4::DESC_GET", "callee_qname": "Desc.__get__",  "kind": "IMPLICIT", "resolved": True},
  {"call_id": "augassign_realism_04.py::A4::IADD",     "callee_qname": "Box.__iadd__",  "kind": "IMPLICIT", "resolved": True},
  {"call_id": "augassign_realism_04.py::A4::DESC_SET", "callee_qname": "Desc.__set__",  "kind": "IMPLICIT", "resolved": True},
]
```

**Negative assertion (shadowing):**

```python
# must be empty
# (callee_qname endswith "__get__"/"__set__" AND anchor == "B1")
EXPECTED_NO_DESCRIPTOR_CALLS_FOR_B1 = True
```

---

## C3) Expected `arg_to_param` rows (this is where compound RHS is enforced)

The critical checks:

### For each A*:

* `IADD.self` must bind from **`DESC_GET.ret`** (the descriptor read result)
* `IADD.other` must bind from the **full RHS expression span**:

  * A1: `rhs.delta`
  * A2: `rhs.arr[rhs.idx]`
  * A3: `rhs.other._d.v`
  * A4: `rhs.arr[0]`
* `DESC_SET.value` must bind from **`IADD.ret`**

Represented as:

```python
EXPECTED_ARG_TO_PARAM = [
  # A1 chain
  {"call_id": "augassign_realism_04.py::A1::IADD",     "arg_role": "receiver",  "arg_ref": "augassign_realism_04.py::A1::DESC_GET::ret", "param": "self"},
  {"call_id": "augassign_realism_04.py::A1::IADD",     "arg_role": "positional","arg_span": "rhs.delta",                                "param": "other"},
  {"call_id": "augassign_realism_04.py::A1::DESC_SET", "arg_role": "positional","arg_span": "o",                                        "param": "obj"},
  {"call_id": "augassign_realism_04.py::A1::DESC_SET", "arg_role": "positional","arg_ref": "augassign_realism_04.py::A1::IADD::ret",    "param": "value"},

  # A2 chain (RHS = subscript)
  {"call_id": "augassign_realism_04.py::A2::IADD",     "arg_role": "receiver",  "arg_ref": "augassign_realism_04.py::A2::DESC_GET::ret", "param": "self"},
  {"call_id": "augassign_realism_04.py::A2::IADD",     "arg_role": "positional","arg_span": "rhs.arr[rhs.idx]",                          "param": "other"},
  {"call_id": "augassign_realism_04.py::A2::DESC_SET", "arg_role": "positional","arg_span": "o",                                        "param": "obj"},
  {"call_id": "augassign_realism_04.py::A2::DESC_SET", "arg_role": "positional","arg_ref": "augassign_realism_04.py::A2::IADD::ret",    "param": "value"},

  # A3 chain (RHS = attribute chain)
  {"call_id": "augassign_realism_04.py::A3::IADD",     "arg_role": "receiver",  "arg_ref": "augassign_realism_04.py::A3::DESC_GET::ret", "param": "self"},
  {"call_id": "augassign_realism_04.py::A3::IADD",     "arg_role": "positional","arg_span": "rhs.other._d.v",                            "param": "other"},
  {"call_id": "augassign_realism_04.py::A3::DESC_SET", "arg_role": "positional","arg_span": "o",                                        "param": "obj"},
  {"call_id": "augassign_realism_04.py::A3::DESC_SET", "arg_role": "positional","arg_ref": "augassign_realism_04.py::A3::IADD::ret",    "param": "value"},

  # A4 chain (RHS = subscript literal)
  {"call_id": "augassign_realism_04.py::A4::IADD",     "arg_role": "receiver",  "arg_ref": "augassign_realism_04.py::A4::DESC_GET::ret", "param": "self"},
  {"call_id": "augassign_realism_04.py::A4::IADD",     "arg_role": "positional","arg_span": "rhs.arr[0]",                                "param": "other"},
  {"call_id": "augassign_realism_04.py::A4::DESC_SET", "arg_role": "positional","arg_span": "o",                                        "param": "obj"},
  {"call_id": "augassign_realism_04.py::A4::DESC_SET", "arg_role": "positional","arg_ref": "augassign_realism_04.py::A4::IADD::ret",    "param": "value"},
]
```

> Implementation note: your *actual table* likely stores `arg_node_id` (not a literal `arg_span` string). In the test harness, resolve `arg_span` → `(start_byte,end_byte)` → `span_key` → `arg_node_id` via your existing `core.syntax_nodes`/`core.syntax_spans` lookup, then compare by IDs.

---

## C4) Expected `ret_to_call` rows (enforces the welding chain)

```python
EXPECTED_RET_TO_CALL = [
  {"call_id": "augassign_realism_04.py::A1::DESC_GET", "ret_ref": "augassign_realism_04.py::A1::DESC_GET::ret"},
  {"call_id": "augassign_realism_04.py::A1::IADD",     "ret_ref": "augassign_realism_04.py::A1::IADD::ret"},

  {"call_id": "augassign_realism_04.py::A2::DESC_GET", "ret_ref": "augassign_realism_04.py::A2::DESC_GET::ret"},
  {"call_id": "augassign_realism_04.py::A2::IADD",     "ret_ref": "augassign_realism_04.py::A2::IADD::ret"},

  {"call_id": "augassign_realism_04.py::A3::DESC_GET", "ret_ref": "augassign_realism_04.py::A3::DESC_GET::ret"},
  {"call_id": "augassign_realism_04.py::A3::IADD",     "ret_ref": "augassign_realism_04.py::A3::IADD::ret"},

  {"call_id": "augassign_realism_04.py::A4::DESC_GET", "ret_ref": "augassign_realism_04.py::A4::DESC_GET::ret"},
  {"call_id": "augassign_realism_04.py::A4::IADD",     "ret_ref": "augassign_realism_04.py::A4::IADD::ret"},
]
```

No `RET_TO_CALL` row for `DESC_SET` (returns `None`).

---

# D) Minimal “span extraction” helper for this fixture (byte-accurate)

Because the core purpose is “compound RHS span correctness”, your test should compute byte spans from source bytes.

This helper pattern is deterministic and avoids hand‑counting offsets:

```python
def _line_containing_anchor(src: bytes, anchor: str) -> tuple[int, int, bytes]:
    # returns (line_start_byte, line_end_byte, line_bytes_including_comment)
    needle = f"# @{anchor}".encode("utf8")
    hit = src.index(needle)
    line_start = src.rfind(b"\n", 0, hit) + 1
    line_end = src.find(b"\n", hit)
    if line_end == -1:
        line_end = len(src)
    return line_start, line_end, src[line_start:line_end]

def span_in_anchored_line(src: bytes, anchor: str, expr_text: str) -> tuple[int, int]:
    line_start, _, line = _line_containing_anchor(src, anchor)
    rel = line.index(expr_text.encode("utf8"))
    start = line_start + rel
    end = start + len(expr_text.encode("utf8"))
    return start, end
```

Then for A2 you assert:

```python
rhs_start, rhs_end = span_in_anchored_line(src, "A2", "rhs.arr[rhs.idx]")
# Use (rhs_start, rhs_end) as the expected argument span key for IADD.other
```

This is exactly the thing that catches the bug “we accidentally use only `rhs.arr` span, not the full subscript”.

---

# E) What this fixture will catch immediately (common bugs)

### Bug 1: “RHS span truncated”

Symptom: `ARG_TO_PARAM` for `IADD.other` points to `rhs.arr` or `rhs.other` instead of `rhs.arr[rhs.idx]` / `rhs.other._d.v`.

### Bug 2: “No call chain; only **set**”

Symptom: You emit `Desc.__set__` but fail to emit `Desc.__get__` for `+=` targets, which breaks the DDG/CPG semantics.

### Bug 3: “Wrong shadowing / over-eager base descriptor”

Symptom: You emit `Desc.__get__` / `Desc.__set__` for `@B1` because you looked at `Base` and ignored that `Shadow` binds `d` first.

### Bug 4: “No welding of values across implicit calls”

Symptom: `IADD.self` isn’t wired from `DESC_GET.ret`, and/or `DESC_SET.value` isn’t wired from `IADD.ret`.

---

If you want a *fifth* micro‑fixture after this one, the next “CPG usefulness jump” is usually **augassign on subscripts** (`obj[idx] += rhs`) which forces correct `__getitem__` + `__setitem__` + operator call wiring (analogous to descriptor get/set, but for containers).


## Micro‑fixture 05: `augassign_subscript` (obj[idx] += rhs → `__getitem__` + op + `__setitem__`)

### Why this fixture exists (what it catches)

This is the first “container semantics” augassign test that **forces your Stage‑F call wiring** to synthesize **three implicit calls** for a single source statement:

1. `obj.__getitem__(idx)`  (**read**)
2. `<item>.__iadd__(rhs)` *or* `<item>.__add__(rhs)` (**op**)
3. `obj.__setitem__(idx, <result>)` (**write**)

…and to thread dataflow deterministically:

* **RET_TO_CALL**: `getitem.ret → op.receiver`
* **RET_TO_CALL**: `op.ret → setitem.value`
* **ARG_TO_PARAM**: `idx` maps into both `getitem(k)` and `setitem(k)` (same expression span, evaluated once)

This fixture will immediately break naïve implementations that:

* treat `obj[idx] += rhs` like a single op call (missing get/set),
* wire `setitem` to the *getitem receiver* instead of container,
* always pick `__iadd__` even when absent,
* fail to propagate return flows between implicit calls.

---

## Fixture layout (recommended)

```
fixtures/call_wiring/05_augassign_subscript/
  src/
    augassign_subscript.py
  expected/
    core.cpg_call_targets.parquet
    core.cpg_arg_to_param.parquet
    core.cpg_ret_to_call.parquet
  README.md
  index.scip            # optional but recommended for determinism
```

If you already standardized fixture naming/paths in earlier micro‑fixtures, keep those conventions; only the contents below matter.

---

## `src/augassign_subscript.py` (single-file, two cases: iadd path + add-only fallback)

```python
# fixtures/call_wiring/05_augassign_subscript/src/augassign_subscript.py
from __future__ import annotations


class Counter:
    def __init__(self, v: int) -> None:
        self.v = v

    def __iadd__(self, other: int) -> "Counter":
        self.v += other
        return self


class AddOnly:
    def __init__(self, v: int) -> None:
        self.v = v

    def __add__(self, other: int) -> "AddOnly":
        return AddOnly(self.v + other)


class BoxIAdd:
    def __init__(self) -> None:
        self._d: dict[int, Counter] = {0: Counter(1)}

    def __getitem__(self, k: int) -> Counter:
        return self._d[k]

    def __setitem__(self, k: int, v: Counter) -> None:
        self._d[k] = v


class BoxAdd:
    def __init__(self) -> None:
        self._d: dict[int, AddOnly] = {0: AddOnly(1)}

    def __getitem__(self, k: int) -> AddOnly:
        return self._d[k]

    def __setitem__(self, k: int, v: AddOnly) -> None:
        self._d[k] = v


def demo_iadd() -> BoxIAdd:
    b = BoxIAdd()
    i = 0
    rhs = 3
    b[i] += rhs
    return b


def demo_add() -> BoxAdd:
    b = BoxAdd()
    i = 0
    rhs = 3
    b[i] += rhs
    return b
```

### Design notes (intentional choices)

* `i` and `rhs` are **Names**, not literals, to ensure your argument/span plumbing isn’t “constant-only”.
* `BoxAdd` forces the fallback path to `__add__` (since `AddOnly` has no `__iadd__`).
* We keep types explicit and local to avoid dependency on complex inference; this should resolve cleanly in SCIP/pyright.

---

## Contract assumptions this fixture relies on (matches the earlier Stage‑F style you’ve been using)

I’m assuming your call wiring tables already follow the pattern you’ve been asserting elsewhere:

### `core.cpg_call_targets`

Minimal columns needed for this fixture:

* `call_key` (PK component) — stable ID for the callsite you synthesize
* `target_def_key` (PK component) — stable ID for the callee definition
* `confidence` (float/int) — optional but useful
* `reason` (string/enum) — optional but useful (“augassign_subscript”, “method_lookup”, etc.)

**PK recommendation (schema.metadata[b"codeintel.pk"]):**

* `["call_key", "target_def_key"]`

### `core.cpg_arg_to_param`

Minimal columns:

* `call_key` (PK component)
* `arg_ix` (PK component)
* `param_ix`
* `param_name` (optional but makes debugging easier)
* `mapping_kind` (optional)

**PK recommendation:**

* `["call_key", "arg_ix"]`

### `core.cpg_ret_to_call`

Minimal columns:

* `src_call_key` (PK component)
* `dst_call_key` (PK component)
* `dst_arg_ix` (or `dst_arg_role`) — receiver vs value
* `flow_kind` (optional)

**PK recommendation:**

* `["src_call_key", "dst_call_key", "dst_arg_ix"]`

If your existing PKs differ, keep your PKs—just translate the rows below into your column names.

---

## Deterministic call_key naming for implicit calls (fixture-local convention)

To eliminate ambiguity, the fixture expects you to synthesize **exactly these 6 call_keys** (3 per augassign):

For `demo_iadd` statement `b[i] += rhs`:

* `augassign_subscript.py:demo_iadd:b[i]+=rhs#getitem`
* `augassign_subscript.py:demo_iadd:b[i]+=rhs#op`
* `augassign_subscript.py:demo_iadd:b[i]+=rhs#setitem`

For `demo_add` statement `b[i] += rhs`:

* `augassign_subscript.py:demo_add:b[i]+=rhs#getitem`
* `augassign_subscript.py:demo_add:b[i]+=rhs#op`
* `augassign_subscript.py:demo_add:b[i]+=rhs#setitem`

This “source-anchor + suffix” pattern is the same idea you used for descriptor get/set augassign fixtures—just applied to subscripts.

If you already generate call IDs from span keys (start/end byte), you can still *derive* these exact strings as a debug/test column; but the simplest (and most stable) approach is: **make `call_key` a string and treat it as the PK** for CPG call wiring artifacts.

---

## Expected rows

### 1) `expected/core.cpg_call_targets.parquet`

Represented here as JSONL for clarity:

```json
{"call_key":"augassign_subscript.py:demo_iadd:b[i]+=rhs#getitem","target_def_key":"augassign_subscript.py:BoxIAdd.__getitem__","confidence":1.0,"reason":"augassign_subscript:getitem"}
{"call_key":"augassign_subscript.py:demo_iadd:b[i]+=rhs#op","target_def_key":"augassign_subscript.py:Counter.__iadd__","confidence":1.0,"reason":"augassign_subscript:op_iadd"}
{"call_key":"augassign_subscript.py:demo_iadd:b[i]+=rhs#setitem","target_def_key":"augassign_subscript.py:BoxIAdd.__setitem__","confidence":1.0,"reason":"augassign_subscript:setitem"}

{"call_key":"augassign_subscript.py:demo_add:b[i]+=rhs#getitem","target_def_key":"augassign_subscript.py:BoxAdd.__getitem__","confidence":1.0,"reason":"augassign_subscript:getitem"}
{"call_key":"augassign_subscript.py:demo_add:b[i]+=rhs#op","target_def_key":"augassign_subscript.py:AddOnly.__add__","confidence":1.0,"reason":"augassign_subscript:op_add_fallback"}
{"call_key":"augassign_subscript.py:demo_add:b[i]+=rhs#setitem","target_def_key":"augassign_subscript.py:BoxAdd.__setitem__","confidence":1.0,"reason":"augassign_subscript:setitem"}
```

**Key assertions:**

* `demo_iadd#op` targets `Counter.__iadd__`
* `demo_add#op` targets `AddOnly.__add__` (NOT `__iadd__`)

---

### 2) `expected/core.cpg_arg_to_param.parquet`

Assumption: represent receiver as `arg_ix=0` mapped to `self` (`param_ix=0`), then positional args follow.

```json
{"call_key":"augassign_subscript.py:demo_iadd:b[i]+=rhs#getitem","arg_ix":0,"param_ix":0,"param_name":"self","mapping_kind":"method_receiver"}
{"call_key":"augassign_subscript.py:demo_iadd:b[i]+=rhs#getitem","arg_ix":1,"param_ix":1,"param_name":"k","mapping_kind":"positional"}

{"call_key":"augassign_subscript.py:demo_iadd:b[i]+=rhs#op","arg_ix":0,"param_ix":0,"param_name":"self","mapping_kind":"method_receiver"}
{"call_key":"augassign_subscript.py:demo_iadd:b[i]+=rhs#op","arg_ix":1,"param_ix":1,"param_name":"other","mapping_kind":"positional"}

{"call_key":"augassign_subscript.py:demo_iadd:b[i]+=rhs#setitem","arg_ix":0,"param_ix":0,"param_name":"self","mapping_kind":"method_receiver"}
{"call_key":"augassign_subscript.py:demo_iadd:b[i]+=rhs#setitem","arg_ix":1,"param_ix":1,"param_name":"k","mapping_kind":"positional"}
{"call_key":"augassign_subscript.py:demo_iadd:b[i]+=rhs#setitem","arg_ix":2,"param_ix":2,"param_name":"v","mapping_kind":"positional"}

{"call_key":"augassign_subscript.py:demo_add:b[i]+=rhs#getitem","arg_ix":0,"param_ix":0,"param_name":"self","mapping_kind":"method_receiver"}
{"call_key":"augassign_subscript.py:demo_add:b[i]+=rhs#getitem","arg_ix":1,"param_ix":1,"param_name":"k","mapping_kind":"positional"}

{"call_key":"augassign_subscript.py:demo_add:b[i]+=rhs#op","arg_ix":0,"param_ix":0,"param_name":"self","mapping_kind":"method_receiver"}
{"call_key":"augassign_subscript.py:demo_add:b[i]+=rhs#op","arg_ix":1,"param_ix":1,"param_name":"other","mapping_kind":"positional"}

{"call_key":"augassign_subscript.py:demo_add:b[i]+=rhs#setitem","arg_ix":0,"param_ix":0,"param_name":"self","mapping_kind":"method_receiver"}
{"call_key":"augassign_subscript.py:demo_add:b[i]+=rhs#setitem","arg_ix":1,"param_ix":1,"param_name":"k","mapping_kind":"positional"}
{"call_key":"augassign_subscript.py:demo_add:b[i]+=rhs#setitem","arg_ix":2,"param_ix":2,"param_name":"v","mapping_kind":"positional"}
```

**Key assertions:**

* Both `getitem` and `setitem` have an `arg_ix=1 → k`
* Both have `arg_ix=0 → self`
* `setitem` has `arg_ix=2 → v` (the computed result)

---

### 3) `expected/core.cpg_ret_to_call.parquet`

Here we assert the two “weld” edges per augassign:

* `getitem.ret → op.receiver`
* `op.ret → setitem.value`

We encode receiver as `dst_arg_ix=0` and setitem value as `dst_arg_ix=2` under the arg indexing assumption above.

```json
{"src_call_key":"augassign_subscript.py:demo_iadd:b[i]+=rhs#getitem","dst_call_key":"augassign_subscript.py:demo_iadd:b[i]+=rhs#op","dst_arg_ix":0,"flow_kind":"ret_to_receiver"}
{"src_call_key":"augassign_subscript.py:demo_iadd:b[i]+=rhs#op","dst_call_key":"augassign_subscript.py:demo_iadd:b[i]+=rhs#setitem","dst_arg_ix":2,"flow_kind":"ret_to_arg"}

{"src_call_key":"augassign_subscript.py:demo_add:b[i]+=rhs#getitem","dst_call_key":"augassign_subscript.py:demo_add:b[i]+=rhs#op","dst_arg_ix":0,"flow_kind":"ret_to_receiver"}
{"src_call_key":"augassign_subscript.py:demo_add:b[i]+=rhs#op","dst_call_key":"augassign_subscript.py:demo_add:b[i]+=rhs#setitem","dst_arg_ix":2,"flow_kind":"ret_to_arg"}
```

**Key assertions:**

* The op call’s receiver is *not* the container `b`; it’s the **result of getitem**.
* The setitem value arg is *not* the RHS; it’s the **result of the op call**.

---

## How to generate / pin the SCIP fixture (recommended for stability)

If you want this fixture to be deterministic across environments, check in `index.scip` produced with pinned tool versions (whatever you already pinned for earlier fixtures). The workflow is typically:

* run your SCIP indexer on `fixtures/.../src` with a pinned `scip-python` + pinned `pyright` (or whatever drives the index),
* store the resulting `.scip` under the fixture,
* configure your ingestion to accept “prebuilt SCIP” for fixtures (bypassing local tool drift).

This keeps the test focused on your **call wiring logic**, not on toolchain variability.

---

## Implementation expectations (what Stage‑F must do to pass this fixture)

This fixture presumes Stage‑F has a dedicated handler for:

* `AugAssign(target=Subscript(value=..., slice=...), op=Add, value=rhs)` (or equivalent syntax representation)
* and emits **three** synthesized call nodes with the keys above.

Resolution strategy is up to you (SCIP-first, then heuristic), but the test asserts only:

* the **target defs** are correct (`__getitem__`, `__setitem__`, and `__iadd__` vs `__add__`),
* and the **flow wiring** is correct (RET_TO_CALL edges as above),
* and the **arg/param mapping** is correct.

---

If you want, I can also provide a *sixth* micro‑fixture that expands this same subscript augassign pattern to the two common “real world” container variants that break first implementations:

1. `obj[idx] += rhs` where `idx` is a call expression (must be evaluated once; flows to both get/set)
2. `obj[idx] += rhs` where the container’s `__getitem__` returns a `Union[T1, T2]` (forces “multiple possible op targets” handling and confidence scoring)


Below is the **V6 micro‑fixture** extending the subscript‑`augassign` pattern to the two “real world” container variants you called out:

1. `obj[idx()] += rhs` where **idx is a call expression** (must be evaluated once; the same computed value must flow into both GET and SET).

2. `obj[idx] += rhs` where `__getitem__` returns a **Union[T1, T2]** (must yield **multiple possible operator targets** + **confidence scoring**, while staying deterministic).

I’m writing this in the same “golden expectations” style as the prior fixtures: **tiny Python corpus + expected rows** for the call wiring outputs (**CALL_TARGETS**, **ARG_TO_PARAM**, **RET_TO_CALL**). Adjust table names/columns to your exact dataset keys if they differ, but the **semantic contract** is the point (and the spans/anchors make it deterministic).

---

## V6 fixture layout

Create a new folder (keep it frozen; whitespace changes will shift byte offsets):

```
tests/fixtures/cpg_stageF_call_wiring_v6/
  subscript_augassign_idx_call.py
  subscript_augassign_union_getitem.py
  EXPECTED/
    cpg_call_sites.json
    cpg_call_targets.json
    cpg_arg_to_param.json
    cpg_ret_to_call.json
```

You can store expected rows as JSON (then build Arrow in test code), or store them directly as Arrow/Parquet. JSON is easiest for review.

**Span rules assumed (match prior fixtures):**

* `start_byte`/`end_byte` are **0-based byte offsets**, UTF‑8, half‑open `[start,end)`.
* A “call site” is identified by `(path, kind, start_byte, end_byte)`; multiple call sites may share the same span iff `kind` differs.
* For implicit calls in `Subscript` and `AugAssign`, spans are:

  * `subscript.get`: span of the **subscript expression** `obj[idx]`
  * `subscript.set`: span of the **subscript expression** `obj[idx]`
  * `augassign.op`: span of the **whole augassign statement** `obj[idx] += rhs`
  * Explicit calls use the span of the call expression.

---

## File 1: idx is a call expression (must be evaluated once)

### `subscript_augassign_idx_call.py` (exact content)

```python
from __future__ import annotations

class Cell:
    def __iadd__(self, rhs: int) -> "Cell":
        ...

class Box:
    def __getitem__(self, k: int) -> Cell:
        ...

    def __setitem__(self, k: int, v: Cell) -> None:
        ...

def idx() -> int:
    return 1

def main(b: Box) -> None:
    b[idx()] += 5
```

### V6 anchors (byte spans)

These are the only spans you need to validate this case:

* **AugAssign statement** `b[idx()] += 5`

  * `start_byte=299`, `end_byte=312`
  * `kind="augassign.op"`

* **Subscript expression** `b[idx()]`

  * `start_byte=299`, `end_byte=307`
  * `kind="subscript.get"` and `kind="subscript.set"` (same span, different kind)

* **Explicit index call** `idx()`

  * `start_byte=301`, `end_byte=306`
  * `kind="call"`

* Useful arg spans:

  * receiver `b`: `start_byte=299`, `end_byte=300`
  * rhs literal `5`: `start_byte=311`, `end_byte=312`

### Expected call sites (minimal set)

In `EXPECTED/cpg_call_sites.json`:

```json
[
  {"path":"tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_idx_call.py","kind":"call","start_byte":301,"end_byte":306,"is_implicit":false,"callee_hint":"idx"},
  {"path":"tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_idx_call.py","kind":"subscript.get","start_byte":299,"end_byte":307,"is_implicit":true,"callee_hint":"Box.__getitem__"},
  {"path":"tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_idx_call.py","kind":"augassign.op","start_byte":299,"end_byte":312,"is_implicit":true,"callee_hint":"Cell.__iadd__"},
  {"path":"tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_idx_call.py","kind":"subscript.set","start_byte":299,"end_byte":307,"is_implicit":true,"callee_hint":"Box.__setitem__"}
]
```

### Expected CALL_TARGETS

Key intent: **idx() is a single call site**; its value is reused (see ARG_TO_PARAM) for both GET and SET.

In `EXPECTED/cpg_call_targets.json`:

```json
[
  {
    "path":"tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_idx_call.py",
    "call_kind":"call","call_start_byte":301,"call_end_byte":306,
    "target_symbol":"py::tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_idx_call.py#idx",
    "target_kind":"function",
    "resolution":"direct",
    "confidence":1.0
  },
  {
    "path":"tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_idx_call.py",
    "call_kind":"subscript.get","call_start_byte":299,"call_end_byte":307,
    "target_symbol":"py::tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_idx_call.py#Box.__getitem__",
    "target_kind":"method",
    "resolution":"receiver_type",
    "confidence":1.0
  },
  {
    "path":"tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_idx_call.py",
    "call_kind":"augassign.op","call_start_byte":299,"call_end_byte":312,
    "target_symbol":"py::tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_idx_call.py#Cell.__iadd__",
    "target_kind":"method",
    "resolution":"augassign_desugar",
    "confidence":1.0
  },
  {
    "path":"tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_idx_call.py",
    "call_kind":"subscript.set","call_start_byte":299,"call_end_byte":307,
    "target_symbol":"py::tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_idx_call.py#Box.__setitem__",
    "target_kind":"method",
    "resolution":"receiver_type",
    "confidence":1.0
  }
]
```

### Expected ARG_TO_PARAM

This is where we **prove** “idx evaluated once” in a static artifact: both GETITEM and SETITEM consume the **same** `idx()` call result node (not two independently re-parsed “idx()” nodes).

I strongly recommend your ARG mapping table supports “arg is a call result” by referencing the producing call site identity (not just raw spans). If you *only* store arg spans, you can still validate the “same span used twice” property, but you lose the “SSA-ish” guarantee that it’s the *same computed value* object in your DDG later.

In `EXPECTED/cpg_arg_to_param.json`:

```json
[
  // --- subscript.get -> Box.__getitem__(self, k)

  {
    "path":"tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_idx_call.py",
    "call_kind":"subscript.get","call_start_byte":299,"call_end_byte":307,
    "target_symbol":"py::tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_idx_call.py#Box.__getitem__",
    "arg_position":0,"arg_kind":"receiver",
    "arg_span_start":299,"arg_span_end":300,
    "param_name":"self","param_position":0,
    "confidence":1.0
  },
  {
    "path":"tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_idx_call.py",
    "call_kind":"subscript.get","call_start_byte":299,"call_end_byte":307,
    "target_symbol":"py::tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_idx_call.py#Box.__getitem__",
    "arg_position":1,"arg_kind":"call_result",
    "arg_call_kind":"call","arg_call_start":301,"arg_call_end":306,
    "param_name":"k","param_position":1,
    "confidence":1.0
  },

  // --- augassign.op -> Cell.__iadd__(self, rhs)
  // receiver is the *result of subscript.get* (i.e., the loaded element)

  {
    "path":"tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_idx_call.py",
    "call_kind":"augassign.op","call_start_byte":299,"call_end_byte":312,
    "target_symbol":"py::tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_idx_call.py#Cell.__iadd__",
    "arg_position":0,"arg_kind":"call_result",
    "arg_call_kind":"subscript.get","arg_call_start":299,"arg_call_end":307,
    "param_name":"self","param_position":0,
    "confidence":1.0
  },
  {
    "path":"tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_idx_call.py",
    "call_kind":"augassign.op","call_start_byte":299,"call_end_byte":312,
    "target_symbol":"py::tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_idx_call.py#Cell.__iadd__",
    "arg_position":1,"arg_kind":"expr_span",
    "arg_span_start":311,"arg_span_end":312,
    "param_name":"rhs","param_position":1,
    "confidence":1.0
  },

  // --- subscript.set -> Box.__setitem__(self, k, v)
  // k reuses idx() call result; v is the result of augassign.op

  {
    "path":"tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_idx_call.py",
    "call_kind":"subscript.set","call_start_byte":299,"call_end_byte":307,
    "target_symbol":"py::tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_idx_call.py#Box.__setitem__",
    "arg_position":0,"arg_kind":"receiver",
    "arg_span_start":299,"arg_span_end":300,
    "param_name":"self","param_position":0,
    "confidence":1.0
  },
  {
    "path":"tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_idx_call.py",
    "call_kind":"subscript.set","call_start_byte":299,"call_end_byte":307,
    "target_symbol":"py::tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_idx_call.py#Box.__setitem__",
    "arg_position":1,"arg_kind":"call_result",
    "arg_call_kind":"call","arg_call_start":301,"arg_call_end":306,
    "param_name":"k","param_position":1,
    "confidence":1.0
  },
  {
    "path":"tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_idx_call.py",
    "call_kind":"subscript.set","call_start_byte":299,"call_end_byte":307,
    "target_symbol":"py::tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_idx_call.py#Box.__setitem__",
    "arg_position":2,"arg_kind":"call_result",
    "arg_call_kind":"augassign.op","arg_call_start":299,"arg_call_end":312,
    "param_name":"v","param_position":2,
    "confidence":1.0
  }
]
```

### Expected RET_TO_CALL

This is the minimal set that makes Stage‑F “call wiring complete”:

In `EXPECTED/cpg_ret_to_call.json`:

```json
[
  {
    "path":"tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_idx_call.py",
    "call_kind":"call","call_start_byte":301,"call_end_byte":306,
    "target_symbol":"py::tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_idx_call.py#idx",
    "ret_position":0,
    "confidence":1.0
  },
  {
    "path":"tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_idx_call.py",
    "call_kind":"subscript.get","call_start_byte":299,"call_end_byte":307,
    "target_symbol":"py::tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_idx_call.py#Box.__getitem__",
    "ret_position":0,
    "confidence":1.0
  },
  {
    "path":"tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_idx_call.py",
    "call_kind":"augassign.op","call_start_byte":299,"call_end_byte":312,
    "target_symbol":"py::tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_idx_call.py#Cell.__iadd__",
    "ret_position":0,
    "confidence":1.0
  }

  // subscript.set is typically a sink (statement semantics); omit RET_TO_CALL unless you
  // explicitly model the implicit None-return call result node.
]
```

---

## File 2: **getitem** returns Union[T1, T2] (multiple op targets + confidence)

### `subscript_augassign_union_getitem.py` (exact content)

```python
from __future__ import annotations

class A:
    def __iadd__(self, rhs: int) -> "A":
        ...

class B:
    def __iadd__(self, rhs: int) -> "B":
        ...

class Poly:
    def __getitem__(self, k: int) -> A | B:
        ...

    def __setitem__(self, k: int, v: A | B) -> None:
        ...

def main(p: Poly) -> None:
    p[0] += 3
```

### V6 anchors (byte spans)

* **AugAssign statement** `p[0] += 3`

  * `start_byte=328`, `end_byte=337`
  * `kind="augassign.op"`

* **Subscript expression** `p[0]`

  * `start_byte=328`, `end_byte=332`
  * `kind="subscript.get"` and `kind="subscript.set"`

* Useful arg spans:

  * receiver `p`: `start_byte=328`, `end_byte=329`
  * index literal `0`: `start_byte=330`, `end_byte=331`
  * rhs literal `3`: `start_byte=336`, `end_byte=337`

### Expected call sites

`EXPECTED/cpg_call_sites.json` should also contain these rows:

```json
[
  {"path":"tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_union_getitem.py","kind":"subscript.get","start_byte":328,"end_byte":332,"is_implicit":true,"callee_hint":"Poly.__getitem__"},
  {"path":"tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_union_getitem.py","kind":"augassign.op","start_byte":328,"end_byte":337,"is_implicit":true,"callee_hint":"A|B.__iadd__"},
  {"path":"tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_union_getitem.py","kind":"subscript.set","start_byte":328,"end_byte":332,"is_implicit":true,"callee_hint":"Poly.__setitem__"}
]
```

### Expected CALL_TARGETS (the key union assertion)

This is the core of this fixture: **two targets for the same operator call site**, with confidence.

In `EXPECTED/cpg_call_targets.json` (append these rows):

```json
[
  {
    "path":"tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_union_getitem.py",
    "call_kind":"subscript.get","call_start_byte":328,"call_end_byte":332,
    "target_symbol":"py::tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_union_getitem.py#Poly.__getitem__",
    "target_kind":"method",
    "resolution":"receiver_type",
    "confidence":1.0
  },

  // Union-return operator dispatch: TWO viable __iadd__ candidates
  {
    "path":"tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_union_getitem.py",
    "call_kind":"augassign.op","call_start_byte":328,"call_end_byte":337,
    "target_symbol":"py::tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_union_getitem.py#A.__iadd__",
    "target_kind":"method",
    "resolution":"union_dispatch",
    "confidence":0.5
  },
  {
    "path":"tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_union_getitem.py",
    "call_kind":"augassign.op","call_start_byte":328,"call_end_byte":337,
    "target_symbol":"py::tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_union_getitem.py#B.__iadd__",
    "target_kind":"method",
    "resolution":"union_dispatch",
    "confidence":0.5
  },

  {
    "path":"tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_union_getitem.py",
    "call_kind":"subscript.set","call_start_byte":328,"call_end_byte":332,
    "target_symbol":"py::tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_union_getitem.py#Poly.__setitem__",
    "target_kind":"method",
    "resolution":"receiver_type",
    "confidence":1.0
  }
]
```

**Confidence scoring rule (keep deterministic):**

* For `Union[T1…Tn]` with no extra disambiguation evidence: emit all `n` targets with `confidence = 1/n`.
* If you *do* have disambiguation evidence (SCIP symbol resolution, import aliasing, MRO narrowing, literal narrowing, etc.), bump that target and renormalize; but V6 expects the “no extra evidence” default.

### Expected ARG_TO_PARAM (note duplication across two op targets)

In `EXPECTED/cpg_arg_to_param.json` (append these rows):

```json
[
  // subscript.get -> Poly.__getitem__(self, k)
  {
    "path":"tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_union_getitem.py",
    "call_kind":"subscript.get","call_start_byte":328,"call_end_byte":332,
    "target_symbol":"py::tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_union_getitem.py#Poly.__getitem__",
    "arg_position":0,"arg_kind":"receiver",
    "arg_span_start":328,"arg_span_end":329,
    "param_name":"self","param_position":0,
    "confidence":1.0
  },
  {
    "path":"tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_union_getitem.py",
    "call_kind":"subscript.get","call_start_byte":328,"call_end_byte":332,
    "target_symbol":"py::tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_union_getitem.py#Poly.__getitem__",
    "arg_position":1,"arg_kind":"expr_span",
    "arg_span_start":330,"arg_span_end":331,
    "param_name":"k","param_position":1,
    "confidence":1.0
  },

  // augassign.op -> A.__iadd__(self, rhs)
  {
    "path":"tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_union_getitem.py",
    "call_kind":"augassign.op","call_start_byte":328,"call_end_byte":337,
    "target_symbol":"py::tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_union_getitem.py#A.__iadd__",
    "arg_position":0,"arg_kind":"call_result",
    "arg_call_kind":"subscript.get","arg_call_start":328,"arg_call_end":332,
    "param_name":"self","param_position":0,
    "confidence":0.5
  },
  {
    "path":"tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_union_getitem.py",
    "call_kind":"augassign.op","call_start_byte":328,"call_end_byte":337,
    "target_symbol":"py::tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_union_getitem.py#A.__iadd__",
    "arg_position":1,"arg_kind":"expr_span",
    "arg_span_start":336,"arg_span_end":337,
    "param_name":"rhs","param_position":1,
    "confidence":0.5
  },

  // augassign.op -> B.__iadd__(self, rhs)
  {
    "path":"tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_union_getitem.py",
    "call_kind":"augassign.op","call_start_byte":328,"call_end_byte":337,
    "target_symbol":"py::tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_union_getitem.py#B.__iadd__",
    "arg_position":0,"arg_kind":"call_result",
    "arg_call_kind":"subscript.get","arg_call_start":328,"arg_call_end":332,
    "param_name":"self","param_position":0,
    "confidence":0.5
  },
  {
    "path":"tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_union_getitem.py",
    "call_kind":"augassign.op","call_start_byte":328,"call_end_byte":337,
    "target_symbol":"py::tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_union_getitem.py#B.__iadd__",
    "arg_position":1,"arg_kind":"expr_span",
    "arg_span_start":336,"arg_span_end":337,
    "param_name":"rhs","param_position":1,
    "confidence":0.5
  },

  // subscript.set -> Poly.__setitem__(self, k, v)   (v = result of augassign.op)
  {
    "path":"tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_union_getitem.py",
    "call_kind":"subscript.set","call_start_byte":328,"call_end_byte":332,
    "target_symbol":"py::tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_union_getitem.py#Poly.__setitem__",
    "arg_position":0,"arg_kind":"receiver",
    "arg_span_start":328,"arg_span_end":329,
    "param_name":"self","param_position":0,
    "confidence":1.0
  },
  {
    "path":"tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_union_getitem.py",
    "call_kind":"subscript.set","call_start_byte":328,"call_end_byte":332,
    "target_symbol":"py::tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_union_getitem.py#Poly.__setitem__",
    "arg_position":1,"arg_kind":"expr_span",
    "arg_span_start":330,"arg_span_end":331,
    "param_name":"k","param_position":1,
    "confidence":1.0
  },
  {
    "path":"tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_union_getitem.py",
    "call_kind":"subscript.set","call_start_byte":328,"call_end_byte":332,
    "target_symbol":"py::tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_union_getitem.py#Poly.__setitem__",
    "arg_position":2,"arg_kind":"call_result",
    "arg_call_kind":"augassign.op","arg_call_start":328,"arg_call_end":337,
    "param_name":"v","param_position":2,
    "confidence":1.0
  }
]
```

### Expected RET_TO_CALL

In `EXPECTED/cpg_ret_to_call.json` (append these rows):

```json
[
  {
    "path":"tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_union_getitem.py",
    "call_kind":"subscript.get","call_start_byte":328,"call_end_byte":332,
    "target_symbol":"py::tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_union_getitem.py#Poly.__getitem__",
    "ret_position":0,
    "confidence":1.0
  },
  {
    "path":"tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_union_getitem.py",
    "call_kind":"augassign.op","call_start_byte":328,"call_end_byte":337,
    "target_symbol":"py::tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_union_getitem.py#A.__iadd__",
    "ret_position":0,
    "confidence":0.5
  },
  {
    "path":"tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_union_getitem.py",
    "call_kind":"augassign.op","call_start_byte":328,"call_end_byte":337,
    "target_symbol":"py::tests/fixtures/cpg_stageF_call_wiring_v6/subscript_augassign_union_getitem.py#B.__iadd__",
    "ret_position":0,
    "confidence":0.5
  }
]
```

---

## What this fixture will catch immediately (the “failure modes”)

### Variant 1 (`idx()` in the index)

This fails if you:

* create **two** `idx()` call sites (one “for get”, one “for set”) instead of one,
* wire GETITEM key to the `idx()` call but wire SETITEM key to the raw span `idx()` without tying to the computed value node (you’ll later break DDG),
* don’t model the operator call as consuming the GETITEM result and producing the SETITEM value.

### Variant 2 (Union return)

This fails if you:

* collapse union dispatch to a single arbitrary target,
* emit multiple targets but **forget** to also duplicate ARG_TO_PARAM and RET_TO_CALL per target (or at least key them by `target_symbol` so they remain unambiguous),
* output nondeterministic ordering without a stable PK (you want the harness’ sort‑by‑PK to make this deterministic).

---

## Implementation note: the minimal “correct desugar” you should be doing internally (why the fixture is shaped this way)

For `AugAssign(target=Subscript(base, index), op, value)` the stable internal expansion that matches Python’s evaluation constraints is:

1. evaluate `base` once → `tmp_base`
2. evaluate `index` once → `tmp_index`  (**this is what V6 #1 enforces**)
3. `tmp_loaded = CALL subscript.get(tmp_base, tmp_index)` → targets `__getitem__`
4. `tmp_new = CALL augassign.op(tmp_loaded, value)` → targets `__iadd__` (or `__add__` fallback)
5. `CALL subscript.set(tmp_base, tmp_index, tmp_new)` → targets `__setitem__`

Your Stage‑F tables should be the call‑graph encoding of steps (3)(4)(5), and the ARG mapping should encode the value reuse (2) and chaining (3→4→5).

---

If you want, I can now take **this exact V6 fixture** and extend it with **the next two subscript‑augassign pain points** you alluded to earlier (without growing expected rows too much):

* `obj[idx()] += rhs` where `idx()` has side effects and must be evaluated exactly once **and** the RHS is a compound expression (attribute/subscript), so you exercise both sides of evaluation ordering.

* union returns where only one candidate supports `__iadd__` but both support `__add__`, forcing correct fallback modeling + confidence weighting.
