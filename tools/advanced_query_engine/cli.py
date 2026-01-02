"""CLI for the advanced query engine."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import msgspec
import orjson

from tools.advanced_query_engine.contracts import (
    JSONValue,
    QueryBudget,
    QueryRequest,
    query_request_schema,
    query_response_schema,
)
from tools.advanced_query_engine.packs.wiring_schema import wiring_pack_schema
from tools.advanced_query_engine.service import RepoServiceOptions, SearchService


def _load_options(
    options_path: str | None, options_json: str | None
) -> dict[str, JSONValue] | None:
    """Load options from a JSON file or JSON string.

    Returns
    -------
    dict[str, object] | None
        Parsed options payload.
    """
    if options_path:
        return msgspec.json.decode(
            Path(options_path).read_bytes(),
            type=dict[str, JSONValue],
        )
    if options_json:
        return msgspec.json.decode(options_json.encode("utf-8"), type=dict[str, JSONValue])
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
    return msgspec.json.decode(value.encode("utf-8"), type=QueryBudget)


def _render_schema(kind: str) -> bytes:
    if kind == "request":
        return orjson.dumps(query_request_schema(), option=orjson.OPT_SORT_KEYS)
    if kind == "response":
        return orjson.dumps(query_response_schema(), option=orjson.OPT_SORT_KEYS)
    return orjson.dumps(wiring_pack_schema(), option=orjson.OPT_SORT_KEYS)


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Advanced query engine")
    parser.add_argument("--repo", help="Repository root")
    parser.add_argument("--type", help="Query type")
    parser.add_argument("--text", help="Query text")
    parser.add_argument("--scope", action="append", help="Scope path (repeatable)")
    parser.add_argument("--options-json", help="Options as JSON string")
    parser.add_argument("--options-file", help="Options JSON file")
    parser.add_argument("--budget-json", help="Budget JSON string")
    parser.add_argument("--persist", action="store_true", help="Persist results to Parquet.")
    parser.add_argument("--persist-path", help="Root directory for persisted results.")
    parser.add_argument(
        "--persist-partition-by",
        action="append",
        help="Partition column for persistence (repeatable).",
    )
    parser.add_argument(
        "--analytics",
        action="store_true",
        help="Run analytics on persisted results.",
    )
    parser.add_argument(
        "--validate-persisted",
        action="store_true",
        help="Validate persisted results with Pandera.",
    )
    parser.add_argument(
        "--analytics-profile",
        action="store_true",
        help="Include Polars profile output in analytics.",
    )
    parser.add_argument(
        "--analytics-chunk-size",
        type=int,
        help="Batch size for analytics streaming.",
    )
    parser.add_argument(
        "--analytics-max-rows",
        type=int,
        help="Override max rows for analytics streaming.",
    )
    parser.add_argument(
        "--schema",
        choices=("request", "response", "wiring_pack"),
        help="Print JSON schema and exit.",
    )
    return parser


def _require_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Ensure required arguments are present.

    Parameters
    ----------
    parser:
        Argument parser for emitting errors.
    args:
        Parsed CLI arguments.
    """
    missing = [name for name in ("repo", "type", "text") if getattr(args, name) is None]
    if missing:
        parser.error(f"Missing required arguments: {', '.join(missing)}")


def _option_overrides(args: argparse.Namespace) -> dict[str, JSONValue]:
    """Return options derived from CLI flags.

    Parameters
    ----------
    args:
        Parsed CLI arguments.

    Returns
    -------
    dict[str, JSONValue]
        Options derived from CLI overrides.
    """
    overrides: dict[str, JSONValue] = {}
    if args.persist:
        overrides["persist"] = True
    if args.persist_path:
        overrides["persist_path"] = args.persist_path
    if args.persist_partition_by:
        overrides["persist_partition_by"] = list(args.persist_partition_by)
    if args.analytics:
        overrides["analytics"] = True
    if args.validate_persisted:
        overrides["validate_persisted"] = True
    if args.analytics_profile:
        overrides["analytics_profile"] = True
    if args.analytics_chunk_size is not None:
        overrides["analytics_chunk_size"] = args.analytics_chunk_size
    if args.analytics_max_rows is not None:
        overrides["analytics_max_rows"] = args.analytics_max_rows
    return overrides


def _merge_cli_options(
    args: argparse.Namespace, options: dict[str, JSONValue] | None
) -> dict[str, JSONValue] | None:
    """Merge options from files/JSON with CLI overrides.

    Parameters
    ----------
    args:
        Parsed CLI arguments.
    options:
        Optional options loaded from files or JSON input.

    Returns
    -------
    dict[str, JSONValue] | None
        Merged options or None if no overrides are present.
    """
    overrides = _option_overrides(args)
    if not overrides and not options:
        return None
    merged: dict[str, JSONValue] = {}
    if options:
        merged.update(options)
    merged.update(overrides)
    return merged


def _build_request(
    *,
    args: argparse.Namespace,
    repo_root: Path,
    budget: QueryBudget | None,
    options: dict[str, JSONValue] | None,
) -> QueryRequest:
    """Build a QueryRequest from CLI inputs.

    Parameters
    ----------
    args:
        Parsed CLI arguments.
    repo_root:
        Repository root path.
    budget:
        Optional query budget.
    options:
        Optional query options.

    Returns
    -------
    QueryRequest
        Constructed query request.
    """
    return QueryRequest(
        type=args.type,
        text=args.text,
        repo_root=str(repo_root),
        scope_paths=args.scope,
        budget=budget,
        options=options,
    )


def main() -> int:
    """Run the CLI entrypoint.

    Returns
    -------
    int
        Process exit code.
    """
    parser = _build_parser()
    args = parser.parse_args()
    if args.schema:
        sys.stdout.buffer.write(_render_schema(args.schema))
        sys.stdout.buffer.write(b"\n")
        return 0
    _require_args(parser, args)

    repo_root = Path(args.repo).resolve()
    enable_persistence = bool(args.persist or args.persist_path)
    service = SearchService.from_repo(
        repo_root,
        RepoServiceOptions(enable_persistence=enable_persistence),
    )
    options = _load_options(args.options_file, args.options_json)
    budget = _parse_budget(args.budget_json)
    options = _merge_cli_options(args, options)
    request = _build_request(args=args, repo_root=repo_root, budget=budget, options=options)

    response = service.run(request)
    payload = response.to_dict()
    output = orjson.dumps(
        payload,
        option=orjson.OPT_SORT_KEYS | orjson.OPT_APPEND_NEWLINE,
    )
    sys.stdout.buffer.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
