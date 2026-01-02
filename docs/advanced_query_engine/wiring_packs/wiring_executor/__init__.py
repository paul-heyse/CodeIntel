"""Python wiring-pack executor.

Core entrypoints:
- execute_pack(...)
- execute_packs(...)
- CLI: python -m wiring_executor ...

This package is designed to:
1) Run an rpygrep candidate stage (required)
2) Confirm/structure matches using ast-grep-py (required for these packs)
3) Enrich output (spans, qnames, enclosing symbols) using LibCST

Tree-sitter stages are optional and only used if a pack declares them and the runtime has tree-sitter installed.
"""

from .executor import execute_pack, execute_packs  # noqa: F401
