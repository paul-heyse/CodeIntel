# Python Query Pack Library (rpygrep + ast-grep + tree-sitter)

This repository is a **concrete, Python-only “query pack library”** intended to be consumed by an
LLM code agent (or a deterministic service) that needs **best-in-class codebase answers** for:

1) symbol resolution, 2) references, 3) call hierarchy slices, 4) policy/pattern scan,
5) contract extraction, 6) wiring discovery, 7) precedent search, 8) impact slices.

## Design principle

Every pack is executed as a *pipeline*:

1. **Candidate generation (rpygrep)**:
   - fast, scalable, file-selection + shallow evidence (line + submatch spans + context)
   - emits *candidate files* and *candidate spans*.

2. **Structural confirmation (ast-grep)**:
   - runs only on candidate files
   - emits **match spans** and **captures** (metavariables) for extraction.

3. **Fallback / span-anchoring (tree-sitter query)**:
   - used when ast-grep is absent for a pattern, or when you want capture spans in a uniform
     tree-sitter representation for post-processing.

The **wiring packs** explicitly embed rpygrep stages (this is intentional).

## Directory layout

- `manifest.json`:
  Library index. Lists every pack and how it should be executed.

- `rpygrep/presets/*.json`:
  Search presets (interactive vs audit) with safe defaults.

- `tree_sitter/python/*.scm`:
  Tree-sitter query packs (.scm) for Python.

- `ast_grep/python/*.yaml`:
  Ast-grep rule packs in a “Python runner friendly” YAML structure:
  each file contains a `rules:` list; each rule has:
    - `rule_id`
    - `language` (python)
    - `config` (dict compatible with ast-grep-py root.find/find_all)

- `wiring_packs/python/*.json`:
  “Wiring pack specs”: combine rpygrep candidate patterns + ast-grep rule IDs + (optional)
  tree-sitter fallbacks. These are the concrete packs you run for Query Type #6.

- `tools/*`:
  Reference scripts for:
    - compiling tree-sitter query packs and emitting derived contracts
    - linting pack structure
    - validating wiring pack specs

## Output contract (what your program should emit)

All packs are designed to support a unified “match record” with:

- `path`
- `start_byte`, `end_byte` (byte offsets; end exclusive)
- `start_line`, `start_col`, `end_line`, `end_col` (derived)
- `rule_id` or `pattern_index` provenance
- `snippet` (from rpygrep context or direct slice)
- `captures` (from ast-grep metavariables)

Wiring packs additionally emit:

- `entry_kind` ∈ {route, cli, plugin, di, envflag, configkey}
- `entry_key` (e.g. "GET /v1/items", "ENV:FOO", "CLI:mycmd")
- `framework` (fastapi/flask/click/typer/argparse/stdlib)
- `target_symbol_hint` (best-effort textual handler name)
- `hook_span` (span of decorator/call used for wiring)

## How to extend

1. Add a new ast-grep rule file under `ast_grep/python/`.
2. Add a rpygrep candidate pattern group under `rpygrep/patterns/` (optional).
3. Add a wiring pack spec under `wiring_packs/python/` that references both.
4. Update `manifest.json`.

