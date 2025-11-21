Below is a deep‑dive on **`ast-grep-py`**—the Python bindings for the Rust‑based structural code search/lint/rewriting tool **ast‑grep**—with an emphasis on the **Python API surface**, advanced/niche capabilities, and how to integrate it into an AI agent’s code‑action pipeline.

---

## 0) What `ast-grep-py` is (and when to use it)

* **What it wraps.** `ast-grep-py` exposes ast‑grep’s engine (Tree‑sitter–backed parser + matcher + rewriter) to Python via **PyO3**. You get programmatic construction, search, and transformation of syntax trees from Python. ([Ast Grep][1])
* **Install & versioning.** Published on PyPI as `ast-grep-py` (Python ≥ 3.8). As of **Nov 7, 2025**, latest is **0.39.9**; the project marks itself **“3 – Alpha”** in Trove classifiers. ([PyPI][2])
* **Why the API exists.** YAML rules are intentionally simple; the **API is for cases that are awkward in rules**, e.g., conditional rewrites, content‑dependent decisions, or computing replacement strings from matches. ([Ast Grep][3])
* **Polyglot by design.** Language coverage comes from Tree‑sitter grammars; ast‑grep ships many languages and supports more via custom registration. You pass the **language alias** (e.g., `"python"`, `"typescript"`) to the API. ([Ast Grep][4])

---

## 1) The Python API model at a glance

The Python API is intentionally small and DOM‑like:

```py
from ast_grep_py import SgRoot

root = SgRoot("print('hello')", "python")
node = root.root()
call = node.find(pattern="print($A)")
arg  = call["A"]                 # == call.get_match("A")
text = arg.text()                # => 'hello'
```

* **`SgRoot(src, language)`** parses a string to a syntax tree; **`root()`** returns the **`SgNode`** at the tree root.
* **`SgNode.find(...) / find_all(...)`** accept either **Rule keyword args** (e.g., `pattern=`, `kind=`) or a **Config object** mirroring YAML (for constraints, utilities, etc.). ([Ast Grep][1])
* **Metavariables.** `$A` captures a single node; `$$$ARGS` captures a sequence (0..n) of nodes (arguments/statements/etc.). Use **`get_match()`** (single) or **`get_multiple_matches()`** (multi). ([Ast Grep][5])

> **Key idea:** Treat AST like an HTML DOM. Find with structural selectors (patterns/rules), then traverse/inspect/refine. The docs explicitly compare usage to BeautifulSoup/pyquery. ([Ast Grep][1])

---

## 2) Expressive search: Rules, Config, constraints, and relational filters

### 2.1 Rule kwargs (fast path)

`SgNode.find`/`find_all` accept **atomic**, **relational**, and **composite** rule keys as kwargs:

* **Atomic:** `pattern=`, `kind=`, `regex=`, `nthChild=`, `range=`
* **Relational:** `inside=`, `has=`, `precedes=`, `follows=`
* **Composite:** `all=`, `any=`, `not=`, `matches=` (to reuse named utility rules)

These are the same *Rule Object* fields used by YAML rules. Combine as needed; a node must satisfy **all** fields in the composed rule. ([Ast Grep][6])

**Example (Python rule kwargs):**

```py
node.find(
  pattern="print($A)",
  inside={"kind": "module"},        # relational constraint
)
```

### 2.2 Config object (YAML parity)

For more control, pass a **Config** dict with `rule`, `constraints`, and optional `utils`:

```py
node.find({
  "rule": {"pattern": "print($A)"},
  "constraints": {"A": {"regex": "hello"}}  # filter by capture content
})
```

Config mode unlocks **constraints** (e.g., `regex` over captures) and local **utility rules** reusable via `matches`. ([Ast Grep][1])

### 2.3 Multi‑metavariable semantics (gotcha)

`$$$MULTI` variables are **lazy**: they match the shortest sequence that allows the subsequent token to match (important when you have multiple `$$$` in one pattern). ([Ast Grep][7])

---

## 3) Traversal, inspection & refinement

Once you have an `SgNode`, you can walk the tree and refine matches further:

* **Traversal:** `.parent()`, `.children()`, `.ancestors()`, `.next()/.prev()` etc.
* **Inspection:** `.kind()`, `.text()`, and `range()` → `(start, end)` with 0‑indexed `(line, column, index)` positions for precise patching.
* **Refinement helpers:** `.matches(...)`, `.inside(...)`, `.has(...)`, `.precedes(...)`, `.follows(...)`—useful to post‑filter nodes after a broad `find`.
  All are first‑class in the Python docs with signatures and examples. ([Ast Grep][1])

---

## 4) Rewriting from Python (edits, not templates)

`SgNode` is immutable. To change code:

1. Call **`node.replace(new_text)`** → returns an **Edit** (with `start_pos`, `end_pos`, `inserted_text`).
2. Apply one or many edits by calling **`commit_edits([edit, ...])`** to get a **new source string**.

**Important difference from YAML CLI**: **metavariables are *not* auto‑expanded** in Python. Build the replacement string yourself (e.g., `f"log({node['A'].text()})"`). This is a deliberate API choice so the host language fully controls formatting. ([Ast Grep][1])

---

## 5) Performance tips for an AI agent loop

* **Prefer `find_all` over manual recursion.** Each Python↔Rust hop has FFI cost; bulk matching in Rust is faster than walking the tree in Python and probing each node. ([Ast Grep][8])
* **Batch edits.** Collect many `Edit`s and **commit once** to avoid re‑building intermediate strings repeatedly (API supports `commit_edits([...])`). ([Ast Grep][1])
* **Choose patterns over regex.** `pattern=` lets the engine prune by node kind and structure before text checks; use `regex` as a constraint on captures, not as your primary matcher. ([Ast Grep][6])
* **Re‑parse sparingly.** Keep an `SgRoot` per file buffer and update via edits; avoid re‑creating roots unless the file truly changed.

---

## 6) Language coverage, custom grammars & language injection

* **Built‑in languages.** The docs publish a table of built‑in language aliases (the string you use in CLI and the Python `SgRoot(..., language)` parameter). ([Ast Grep][4])
* **Custom languages.** ast‑grep can **dynamically load Tree‑sitter grammars** and register them in `sgconfig.yml` under `customLanguages` (e.g., for Mojo), specifying the `.so/.dylib` path, file extensions, and optional `expandoChar` when `$VAR` isn’t valid code in that language. This is configured at the **project level**. ([Ast Grep][9])

  * **Python API note:** The official Python page documents selecting a *language name* when constructing `SgRoot`. It does **not** yet document a direct Python function to register grammars at runtime; use project config + CLI registration to make your alias available, then pass the alias in Python. (Community threads point to using the CLI’s custom language mechanism when working from Python.) ([Stack Overflow][10])
* **Language injection** (multi‑language documents). You can declare injected languages (e.g., CSS inside JS tagged templates) in `sgconfig.yml` so ast‑grep will parse the embedded regions correctly during scanning. ([Ast Grep][11])

---

## 7) Searching entire codebases from Python

`ast-grep-py` parses **strings**; for large repositories the recommended approach is to **call the CLI** and consume **JSON** results:

```bash
# JSON output for machine consumption
ast-grep run -p "requests.$FUNC($$$ARGS)" -l python --json .
```

* CLI supports `--json[=<style>]` on both `run` and `scan`. It returns structured matches with file paths, ranges, and optional replacements. Parse these in Python to drive edits with `ast-grep-py`. ([Ast Grep][12])
* This pattern (CLI for breadth, Python API for **content‑aware rewrites**) is explicitly supported in the docs’ “Tooling overview” and “JSON mode” pages. ([Ast Grep][13])

---

## 8) Pattern power‑ups you’ll actually use

* **Pattern syntax** supports single (`$A`) and multi (`$$$ARGS`) metavars, and can be embedded in object patterns (e.g., `{ $$FIELDS }`). The parser only recognizes metavariables when they map to **valid AST nodes**; see the “deep dive” for how meta detection works and edge cases. ([Ast Grep][5])
* **Constraints** apply after pattern match and filter captures by `regex`, `kind`, `nth`, `field`, etc. (great for “match any call, but only keep those where the first argument is a string literal matching X”). ([Ast Grep][14])
* **Relational operators**—`inside/has/precedes/follows`—let you express nesting and ordering, crucial for precise refactors (e.g., “a call inside a test function that *precedes* an assertion”). ([Ast Grep][6])
* **Utility rules** (`utils` + `matches`) help compose complex matchers in modular pieces—these also work when you pass a Config dict from Python. ([Ast Grep][15])

---

## 9) End‑to‑end example: refactor `requests` → `httpx`

**Goal.** Convert simple `requests.get(url, timeout=…)` to `httpx.get(url, timeout=…)`, but only when the name isn’t shadowed locally; keep nuanced control in Python.

```py
from ast_grep_py import SgRoot

code = """
import requests
def fetch(u):
    resp = requests.get(u, timeout=5)
    return resp.text
"""

root = SgRoot(code, "python").root()

# Find all calls of the form: requests.$FUNC($$$ARGS)
calls = root.find_all(pattern="requests.$FUNC($$$ARGS)")
edits = []
for call in calls:
    # Optional: ensure $FUNC is "get" or "post" etc.
    func = call["FUNC"].text()
    if func not in {"get", "post", "put", "delete"}:
        continue

    # Build a replacement string in Python (metavars aren't auto-expanded)
    new_text = f"httpx.{func}({', '.join([n.text() for n in call.get_multiple_matches('ARGS')])})"
    edits.append(call.replace(new_text))

new_src = root.commit_edits(edits)
print(new_src)
```

**Why this shape works well for an agent:**

* Pattern narrows candidates to what you want; Python guard clauses handle **semantic nuances** (e.g., limit which methods to rewrite).
* You control the **replacement string** and can compute it from captured nodes (the Python API doesn’t substitute `$` metavars for you, by design). ([Ast Grep][1])

---

## 10) Debugging & correctness

* **Pattern not matching?** The usual culprit is context: a fragment may not be a legal AST by itself. Use the **object‑pattern** form (`pattern: {context, selector, strictness}`) via Config when needed, or test in the **Playground**. The CLI’s `--debug-query` prints the parser’s view of your pattern. ([Ast Grep][6])
* **Verify multi‑metavar behavior.** When multiple `$$$` are present, remember they’re lazy and will stop before the first following token that can match; adjust pattern layout accordingly. ([Ast Grep][7])

---

## 11) Production notes & ecosystem

* **Status & cadence.** `ast-grep-py` ships frequently; check PyPI for current releases (0.39.9 on Nov 7, 2025). Treat the API as evolving (Alpha). ([PyPI][2])
* **Editor & CI integration.** There’s a VS Code extension, GitHub Action patterns, JSON mode for toolchains, and `ast-grep new` to scaffold `sgconfig.yml`, rules, and tests for lints/codemods. ([Ast Grep][16])

---

## 12) Practical checklist for an AI code‑action agent

1. **Breadth search:** Shell out to `ast-grep run/scan --json` for project‑wide candidates. Parse JSON in Python. ([Ast Grep][12])
2. **Per‑file transform:** Load file text into `SgRoot`; for each candidate `SgNode`, re‑validate with **Config** (constraints/relational), then build **`Edit`s** with context‑aware strings and **commit once**. ([Ast Grep][1])
3. **Edge languages:** If your project uses non‑built‑in languages, **register custom Tree‑sitter grammars** in `sgconfig.yml` (`customLanguages`) and pass that alias to `SgRoot`. For mixed content, tune **language injection** in `sgconfig.yml`. ([Ast Grep][9])
4. **Performance:** Avoid Python‑side tree walks; push matching to Rust via `find_all`. Batch edits. ([Ast Grep][8])
5. **Human‑in‑the‑loop:** When risky, stage a CLI **interactive** session (`-i/--interactive`) or emit diffs for review; when safe, `--update-all` to auto‑apply. (CLI ref.) ([Ast Grep][17])

---

### References (for your internal knowledge graph)

* **Python API doc** (objects, search, traversal, edits, metavariable caveat). ([Ast Grep][1])
* **API usage rationale** (when to use API vs YAML), **performance tip** for FFI. ([Ast Grep][3])
* **Rule Object** (atomic/relational/composite), **pattern syntax**, **constraints**. ([Ast Grep][6])
* **CLI JSON mode** (project‑wide search), **run/scan** reference. ([Ast Grep][18])
* **Languages** (built‑in list), **custom languages** (sgconfig), **language injection**. ([Ast Grep][4])
* **PyPI project page** (version, Python requirement, Alpha status). ([PyPI][2])

---

If you’d like, I can draft a **drop‑in Python module** that wraps the CLI JSON output, hydrates `SgNode` slices in `ast-grep-py`, and provides a uniform “search → validate → rewrite → diff → apply” pipeline tailored to your agent.

[1]: https://ast-grep.github.io/guide/api-usage/py-api.html "Python API | ast-grep"
[2]: https://pypi.org/project/ast-grep-py/ "ast-grep-py · PyPI"
[3]: https://ast-grep.github.io/guide/api-usage.html "API Usage | ast-grep"
[4]: https://ast-grep.github.io/reference/languages.html?utm_source=chatgpt.com "List of Languages with Built-in Support | ast-grep"
[5]: https://ast-grep.github.io/guide/pattern-syntax.html?utm_source=chatgpt.com "Pattern Syntax | ast-grep"
[6]: https://ast-grep.github.io/reference/rule.html?utm_source=chatgpt.com "Rule Object Reference - ast-grep"
[7]: https://ast-grep.github.io/advanced/faq.html?utm_source=chatgpt.com "Frequently Asked Questions - ast-grep"
[8]: https://ast-grep.github.io/guide/api-usage/performance-tip.html?utm_source=chatgpt.com "Performance Tip for napi usage - ast-grep"
[9]: https://ast-grep.github.io/advanced/custom-language.html?utm_source=chatgpt.com "Custom Language Support | ast-grep"
[10]: https://stackoverflow.com/questions/78204918/how-can-i-use-custom-grammar-with-the-ast-grep-python-api?utm_source=chatgpt.com "How can I use custom grammar with the ast-grep Python ..."
[11]: https://ast-grep.github.io/advanced/language-injection.html?utm_source=chatgpt.com "Search Multi-language Documents in ast-grep"
[12]: https://ast-grep.github.io/reference/cli/run.html?utm_source=chatgpt.com "ast-grep run"
[13]: https://ast-grep.github.io/guide/tooling-overview.html?utm_source=chatgpt.com "Command Line Tooling Overview | ast-grep"
[14]: https://ast-grep.github.io/reference/yaml.html?utm_source=chatgpt.com "Configuration Reference | ast-grep"
[15]: https://ast-grep.github.io/guide/rule-config/utility-rule.html?utm_source=chatgpt.com "Reusing Rule as Utility | ast-grep"
[16]: https://ast-grep.github.io/guide/tools/editors.html?utm_source=chatgpt.com "Editor Integration - ast-grep"
[17]: https://ast-grep.github.io/reference/cli/scan.html?utm_source=chatgpt.com "ast-grep scan"
[18]: https://ast-grep.github.io/guide/tools/json.html?utm_source=chatgpt.com "JSON Mode - ast-grep"
