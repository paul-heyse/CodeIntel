#!/usr/bin/env bash
set -euo pipefail

uv run codeintel build run --targets=ci_plan --verbose=1
