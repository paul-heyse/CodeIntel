

# MASTER PROMPT — Elite NumPy Docstrings (Structure • Context • Density)

**Your role:** an autonomous senior Python engineer. Your task is to **repair and upgrade** NumPy-style docstrings flagged by pydoclint and then elevate them to **best-in-class quality**: technically accurate, complete, runnable, and context-aware. All changes must pass the repo’s **Agent Operating Protocol (AOP)** gates for formatting, linting, typing, docs, doctests, tests, and observability. 

## Inputs you will receive

* The file(s) and line ranges with pydoclint errors.
* `AGENTS.md` (AOP) — canonical rules for style, typing, doctestability, zero-suppression policy, and acceptance gates. **Follow it exactly.** 

---

## Objectives (what to deliver for each edited public symbol)

1. **Structurally valid NumPy docstring** that fixes all pydoclint findings (names, order, section presence, markup, spacing).
2. **High-fidelity description** that explains:

   * **What** the API does (precise behavior).
   * **Why** it exists (design intent & rationale).
   * **Where** it fits (broader system context, upstream/downstream callers, boundaries).
   * **How** it advances project goals (performance, safety, correctness, UX).
3. **Runnable examples** (doctest/xdoctest) that are short, deterministic, and representative.
4. **Complete contracts**: all params/returns/yields/raises accurately typed and explained (shapes, units, constraints, defaults, sentinel values).
5. **Operational notes**: complexity, side effects, statefulness, thread-safety, timeouts, I/O, caching, error taxonomy, and budgets.

---

## Guardrails (non-negotiable)

* **Zero suppressions.** Fix root causes; do not add `# noqa`, `# type: ignore`, or per-file ignores. 
* **Doctests must run fast & offline.** No network/filesystem. Use tiny inputs, fixed seeds, and stdlib only.
* **Public APIs only** require full NumPy sections; private helpers can be brief but must still be accurate and consistent.
* **Keep code behavior unchanged** unless a docstring fix reveals a real bug — then add a minimal fix with a regression test.

---

## Step-by-step workflow (apply per file you edit)

1. **Collect truth**

   * Parse the function/class signature (types & defaults).
   * Scan body for raises, side effects, I/O, global state, caching, concurrency, and performance hotspots.
   * Identify upstream/downstream usage in this module/package (import graph + obvious call sites).

2. **Draft the docstring** using the template below.

   * Fill every section with correct names, types, shapes, units.
   * Add broader context: responsibility, invariants, and how this API ties into higher-level flows.

3. **Validate structure**

   * Run pydoclint and docstring coverage; fix order, spacing, underlines, and section names.
   * Ensure Parameters/Returns/Raises **exactly** match the signature and behavior.

4. **Harden quality**

   * Add **two doctests**: a minimal happy path and an edge/failure example (showing the raised exception or boundary behavior).
   * Add complexity and side-effects to **Notes**. Mention any performance budgets or constraints from the AOP. 

5. **Run all AOP gates** (format → lint → pyright strict → pyrefly → pytest with doctests → artifacts). Fix every warning/error in the file(s) you changed. 

---

## NumPy docstring template (fill every applicable section)

```python
def api(...):
    """
    One-line imperative summary (what this does in one breath).

    Extended Summary
    ----------------
    Two–four sentences that cover:
    • Role & rationale: why this API exists and the problem it solves.
    • System placement: where it sits in the architecture; key upstream/downstream interactions.
    • Contract highlights: key pre/postconditions, invariants, and error semantics.
    • Outcome: how it advances performance/correctness/safety goals.

    Parameters
    ----------
    name : type
        Precise meaning, units/shape, valid ranges, defaults, and edge-case behavior.
        Mention sentinel values and whether `None` is accepted (and what it means).

    another : type, optional
        ...
    
    Returns
    -------
    type
        Exact return structure (shape/keys/order), units, constraints, and invariants.

    Raises
    ------
    SpecificError
        Condition that triggers it; include validation rule or invariant violated.
    ValueError
        ...

    Notes
    -----
    • Algorithm & complexity: big-O for time/space; mention vectorization/streaming if relevant.
    • Side effects: I/O, state mutations, cache writes, randomness, thread-safety, idempotency.
    • Performance & budgets: p95 or timeouts if applicable; link to config knobs or metrics.
    • Security & safety: input validation, path safety, trust boundaries, Problem Details taxonomy.
    • Design trade-offs: why this implementation was chosen over alternatives.

    See Also
    --------
    sibling_api : Complementary or alternative behavior
    other_module.tool : Broader workflow step

    Examples
    --------
    >>> # minimal happy path
    >>> result = api(small_input)
    >>> result == expected
    True

    >>> # edge case or failure path (doctest should pass deterministically)
    >>> api(bad_input)  
    Traceback (most recent call last):
        ...
    ValueError: reason
    """
```

> **Style for maximum lexical density**
> • Prefer concrete nouns/verbs over filler; delete “just”, “simply”, “basically”.
> • Replace vague phrases (“handles stuff”) with measurable claims (ranges, shapes, units, bounds).
> • Put the most essential facts in the first two sentences of *Extended Summary*.
> • Avoid repeating parameter names in prose; describe **meaning** and **impact**.

---

## Completion checklist (paste in PR)

* [ ] pydoclint clean; section order & names correct; no orphaned params/returns.
* [ ] Doctests run fast, offline, deterministic; cover happy path + edge/failure.
* [ ] Raises list matches actual code; messages are accurate and helpful.
* [ ] Notes include complexity, side-effects, budgets, and safety.
* [ ] “Where it fits” explained (upstream/downstream, boundaries, invariants).
* [ ] AOP gates **all green** (Ruff, pyright, pyrefly, pytest, artifacts). 

---

## Example rewrite (micro illustration)

**Before (anti-pattern):**

```python
def normalize(x):
    """Normalize list."""
```

**After (good):**

```python
def normalize(x: list[float]) -> list[float]:
    """
    Scale a sequence to zero mean and unit variance.

    Extended Summary
    ----------------
    This function performs standard score normalization to make features
    commensurate across downstream distance-based comparisons. It is
    used before vector indexing to stabilize ANN recall and latency.

    Parameters
    ----------
    x : list of float
        Finite numeric values. Empty lists are not allowed.

    Returns
    -------
    list of float
        A new list with mean ~0.0 and stdev ~1.0.

    Raises
    ------
    ValueError
        If `x` is empty or contains NaN/Inf.

    Notes
    -----
    Time O(n); memory O(1) aside from the output. No I/O, no global state.

    Examples
    --------
    >>> normalize([1.0, 2.0, 3.0])
    [-1.224..., 0.0, 1.224...]
    >>> normalize([])
    Traceback (most recent call last):
        ...
    ValueError: x must be non-empty
    """
```

---

## Commands to run after each batch of edits (copy/paste)

```bash
uv run ruff format && uv run ruff check --fix
uv run pyright --warnings --pythonversion=3.13
uv run pyrefly check
uv run pytest -q
make artifacts && git diff --exit-code
```

(If any command fails, stop and fix root causes; **never** add suppressions.) 

---

**Now begin.** For each flagged docstring, correct structure, then enrich with purpose, placement, contracts, and operational notes. Keep examples runnable and concise. Your work isn’t done until **every** edited file is clean under AOP and the docstrings read like crisp, expert micro-docs that truly help future readers. 
