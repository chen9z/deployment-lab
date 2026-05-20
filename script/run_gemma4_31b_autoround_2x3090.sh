#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${COMPOSE_FILE:-$ROOT_DIR/gemma-4-31b/docker-compose.2x3090.autoround.yml}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8003}"
READY_TIMEOUT_SEC="${READY_TIMEOUT_SEC:-1200}"
HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"

if [[ ! -f "$COMPOSE_FILE" ]]; then
  echo "Compose file not found: $COMPOSE_FILE" >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker not found in PATH" >&2
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "curl not found in PATH" >&2
  exit 1
fi

echo "Compose file: $COMPOSE_FILE"
echo "Base URL:     $BASE_URL"
echo "GPUs:         NVIDIA device IDs 1,2"
echo "Context:      MAX_MODEL_LEN=${MAX_MODEL_LEN:-98304}, MAX_NUM_SEQS=${MAX_NUM_SEQS:-4}"
echo "Model:        ${MODEL_DIR:-$ROOT_DIR/models/Intel/gemma-4-31B-it-int4-AutoRound}"

(
  cd "$(dirname "$COMPOSE_FILE")"
  docker compose -f "$(basename "$COMPOSE_FILE")" up -d --force-recreate
)

echo
echo "Waiting for readiness ..."
for _ in $(seq 1 $((READY_TIMEOUT_SEC / 5))); do
  if curl -fsS --max-time 5 "$BASE_URL/v1/models" >/dev/null 2>&1; then
    echo "Server is ready at $BASE_URL"
    exit 0
  fi
  if [[ -n "${HOST_IP:-}" ]]; then
    alt_base_url="http://${HOST_IP}:${BASE_URL##*:}"
    if curl -fsS --max-time 5 "$alt_base_url/v1/models" >/dev/null 2>&1; then
      echo "Server is ready at $alt_base_url"
      exit 0
    fi
  fi
  sleep 5
done

echo "Timed out waiting for vLLM readiness." >&2
echo "Recent logs:" >&2
(
  cd "$(dirname "$COMPOSE_FILE")"
  docker compose -f "$(basename "$COMPOSE_FILE")" logs --tail 200 vllm >&2 || true
)
exit 1
