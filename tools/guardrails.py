"""Static guardrails for migration-sensitive anti-patterns.

This script scans source/test code for banned patterns called out in the
ibis+pandera+sqlglot migration plan. It is intended to be wired into the
quality gate and fail fast when deprecated surfaces reappear.

It also enforces Hamilton build invariants (graph validation) so build-breaking
tag/contract drift is caught without requiring a full test run.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from codeintel.build.hamilton.validate import validate_graph, validation_result_to_json

BASE_DIRS: tuple[str, ...] = ("src", "tests", "tools", "scripts")
_SELF_REL_PATH = "tools/guardrails.py"


@dataclass(frozen=True)
class Guardrail:
    """Guardrail rule with pattern and allowlist prefixes."""

    name: str
    pattern: re.Pattern[str]
    message: str
    include_prefixes: tuple[str, ...] = ()
    allow_prefixes: tuple[str, ...] = ()


GUARDRAILS: tuple[Guardrail, ...] = (
    Guardrail(
        name="normalized_macros",
        pattern=re.compile(
            r"\b(MacroRequirement|require_normalized_macros|requires_normalized_macro)\b"
        ),
        message="Normalized macro compatibility is removed; drop this surface.",
    ),
    Guardrail(
        name="legacy_sql_builder",
        pattern=re.compile(r"\b(SafeTable|SafeColumn|QueryBuilder|codeintel\.storage\.sql)\b"),
        message="Legacy SQL builder usage is forbidden; use DuckDBPolicyBackend or Ibis.",
    ),
    Guardrail(
        name="legacy_macro_helpers",
        pattern=re.compile(r"\b(macro_exists|safe_macro_exists|INGEST_MACRO_TABLES)\b"),
        message="Legacy macro helpers are removed.",
    ),
    Guardrail(
        name="raw_con_execute",
        pattern=re.compile(r"\.con\.execute\("),
        message="Raw con.execute is only allowed inside storage internals.",
        allow_prefixes=(
            "src/codeintel/storage/",
            "tests/",
        ),
    ),
    Guardrail(
        name="legacy_build_context_stack",
        pattern=re.compile(r"\bcodeintel\.build\.(context|context_base|result|protocols)\b"),
        message="Legacy build context stack is removed; use Hamilton BuildEnv/executor patterns.",
    ),
    Guardrail(
        name="hamilton_save_to_decorator",
        pattern=re.compile(
            r"\b(from hamilton\.function_modifiers\.adapters import SaveToDecorator|@SaveToDecorator\b)"
        ),
        message=(
            "Hamilton SaveToDecorator is forbidden; use SaveToObjectMetadataDecorator "
            "(metadata typed as dict[str, object])."
        ),
    ),
    Guardrail(
        name="build_cast_any",
        pattern=re.compile(r"\bcast\(\s*(?:\"Any\"|'Any'|Any)\s*,"),
        message=(
            '`cast("Any", ...)` is forbidden in build/storage modules; use codeintel.core.ibis_typing.'
        ),
        include_prefixes=("src/codeintel/build/", "src/codeintel/storage/"),
    ),
    Guardrail(
        name="compute_result_any",
        pattern=re.compile(r"\b(type\s+)?ComputeResult\s*(?::\s*TypeAlias)?\s*=\s*Any\b"),
        message="ComputeResult = Any is forbidden; use ExecutionResult.",
        include_prefixes=("src/codeintel/build/",),
    ),
)


def iter_candidate_files(repo_root: Path) -> Iterable[Path]:
    """Yield files under the configured base directories.

    Yields
    ------
    Path
        Python files to scan for guardrail violations.
    """
    for base in BASE_DIRS:
        root = repo_root / base
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if path.is_file():
                rel = path.relative_to(repo_root).as_posix()
                if rel == _SELF_REL_PATH:
                    continue
                yield path


def find_violations(repo_root: Path) -> list[str]:
    """Scan for guardrail violations and return human-friendly messages.

    Returns
    -------
    list[str]
        Collected violation messages.
    """
    violations: list[str] = []
    for path in iter_candidate_files(repo_root):
        rel = path.relative_to(repo_root).as_posix()
        text = path.read_text(encoding="utf-8")
        for rule in GUARDRAILS:
            if rule.include_prefixes and not rel.startswith(rule.include_prefixes):
                continue
            if rule.allow_prefixes and rel.startswith(rule.allow_prefixes):
                continue
            for match in rule.pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                violations.append(f"{rel}:{line}: {rule.name}: {rule.message}")
    return violations


def main() -> int:
    """Entry point for the guardrail scanner.

    Returns
    -------
    int
        Zero when clean, non-zero when violations are found.
    """
    repo_root = Path(__file__).resolve().parent.parent
    violations = find_violations(repo_root)
    if violations:
        for line in violations:
            sys.stderr.write(f"{line}\n")
        return 1

    graph_result = validate_graph()
    if graph_result.has_errors:
        sys.stderr.write("Hamilton graph validation failed.\n")
        sys.stderr.write(validation_result_to_json(graph_result, indent=2))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
