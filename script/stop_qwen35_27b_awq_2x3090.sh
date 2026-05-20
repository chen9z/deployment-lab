#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_DIR="${SERVICE_DIR:-$ROOT_DIR/qwen3.5-27b/dual_3090_awq_service}"
PID_FILE="$SERVICE_DIR/server.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No PID file found at $PID_FILE"
  exit 0
fi

pid="$(cat "$PID_FILE" 2>/dev/null || true)"
if [[ -z "${pid:-}" ]]; then
  echo "PID file is empty: $PID_FILE" >&2
  exit 1
fi

if kill -0 "$pid" 2>/dev/null; then
  kill "$pid"
  echo "Stopped server PID $pid"
else
  echo "Process $pid is not running"
fi

rm -f "$PID_FILE"
