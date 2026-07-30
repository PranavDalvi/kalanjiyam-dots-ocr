#!/bin/bash
# =============================================================================
# Helper Script to build, start, stop, and inspect the DotsOCR Docker Service
# =============================================================================

# Auto-detect docker compose syntax (V1: docker-compose vs V2: docker compose)
if command -v docker-compose &> /dev/null; then
  COMPOSE_CMD="docker-compose"
elif docker compose version &> /dev/null; then
  COMPOSE_CMD="docker compose"
else
  COMPOSE_CMD="docker-compose"
fi

MODEL_HOST_PATH="rednote-hilab/dots.ocr"
ACTION=""

if [[ "$1" == "start" || "$1" == "stop" || "$1" == "logs" || "$1" == "status" ]]; then
  ACTION="$1"
elif [[ "$2" == "start" || "$2" == "stop" || "$2" == "logs" || "$2" == "status" ]]; then
  MODEL_HOST_PATH="$1"
  ACTION="$2"
else
  ACTION="$1"
fi

case "$ACTION" in
  start)
    echo "[Docker Manager] Building and starting DotsOCR container..."
    echo "[Docker Manager] Model Path/Repo: $MODEL_HOST_PATH"
    export HOST_MODEL_PATH="$MODEL_HOST_PATH"
    $COMPOSE_CMD up -d --build
    echo "[Docker Manager] Container started! Endpoint available at http://localhost:8887/ocr"
    ;;

  stop)
    echo "[Docker Manager] Stopping DotsOCR container..."
    $COMPOSE_CMD down
    ;;

  logs)
    echo "[Docker Manager] Fetching logs..."
    $COMPOSE_CMD logs -f
    ;;

  status)
    $COMPOSE_CMD ps
    curl -s http://localhost:8887/gpu-status | python3 -m json.tool
    ;;

  *)
    echo "Usage:"
    echo "  bash run_docker.sh start"
    echo "  bash run_docker.sh stop"
    echo "  bash run_docker.sh logs"
    echo "  bash run_docker.sh status"
    echo "  bash run_docker.sh [CUSTOM_MODEL_PATH] start"
    exit 1
    ;;
esac
