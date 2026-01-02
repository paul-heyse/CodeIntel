"""CLI for the advanced query engine."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tools.advanced_query_engine.contracts import QueryBudget, QueryRequest
from tools.advanced_query_engine.service import SearchService


def _load_options(options_path: str | None, options_json: str | None) -> dict[str, object] | None:
    """Load options from a JSON file or JSON string.

    Returns
    -------
    dict[str, object] | None
        Parsed options payload.
    """
    if options_path:
        payload = json.loads(Path(options_path).read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    if options_json:
        payload = json.loads(options_json)
        return payload if isinstance(payload, dict) else None
    return None


def _parse_budget(value: str | None) -> QueryBudget | None:
    """Parse a JSON budget string into QueryBudget.

    Returns
    -------
    QueryBudget | None
        Parsed budget or None when unavailable.
    """
    if value is None:
        return None
    payload = json.loads(value)
    if not isinstance(payload, dict):
        return None
    return QueryBudget(
        max_files=int(payload.get("max_files", 300)),
        max_matches=int(payload.get("max_matches", 2000)),
        max_depth=int(payload.get("max_depth", 2)),
        max_seconds=payload.get("max_seconds"),
        context_lines=int(payload.get("context_lines", 1)),
    )


def main() -> int:
    """Run the CLI entrypoint.

    Returns
    -------
    int
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Advanced query engine")
    parser.add_argument("--repo", required=True, help="Repository root")
    parser.add_argument("--type", required=True, help="Query type")
    parser.add_argument("--text", required=True, help="Query text")
    parser.add_argument("--scope", action="append", help="Scope path (repeatable)")
    parser.add_argument("--options-json", help="Options as JSON string")
    parser.add_argument("--options-file", help="Options JSON file")
    parser.add_argument("--budget-json", help="Budget JSON string")

    args = parser.parse_args()

    repo_root = Path(args.repo).resolve()
    service = SearchService.from_repo(repo_root)
    options = _load_options(args.options_file, args.options_json)
    budget = _parse_budget(args.budget_json)

    request = QueryRequest(
        type=args.type,
        text=args.text,
        repo_root=str(repo_root),
        scope_paths=args.scope,
        budget=budget,
        options=options,
    )

    response = service.run(request)
    sys.stdout.write(json.dumps(response.to_dict(), indent=2, sort_keys=True))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
