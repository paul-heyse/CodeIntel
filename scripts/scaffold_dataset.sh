#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper for creating a dataset scaffold with sensible defaults.

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <dataset_name> [extra scaffold args]" >&2
  exit 2
fi

NAME="$1"
shift

OUTPUT_DIR="${OUTPUT_DIR:-build/dataset_scaffolds}"
KIND="${KIND:-table}"
TABLE_KEY="${TABLE_KEY:-}"

if [ -z "${TABLE_KEY}" ]; then
  if [ "${KIND}" = "view" ]; then
    TABLE_KEY="docs.${NAME}"
  else
    TABLE_KEY="analytics.${NAME}"
  fi
fi

codeintel datasets scaffold "${NAME}" \
  --kind "${KIND}" \
  --table-key "${TABLE_KEY}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"
