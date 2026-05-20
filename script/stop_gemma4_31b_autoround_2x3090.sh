#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${COMPOSE_FILE:-$ROOT_DIR/gemma-4-31b/docker-compose.2x3090.autoround.yml}"

(
  cd "$(dirname "$COMPOSE_FILE")"
  docker compose -f "$(basename "$COMPOSE_FILE")" down
)
