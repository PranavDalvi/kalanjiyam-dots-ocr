#!/bin/bash
# =============================================================================
# Helper Script to build, start, stop, and inspect the Kalanjiyam OCR Service
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
GEMMA_MODEL_HOST_PATH="google/gemma-4-26B-A4B-it"
ACTION=""

if [[ "$1" == "start" || "$1" == "stop" || "$1" == "logs" || "$1" == "status" || "$1" == "start-multi" || "$1" == "stop-multi" || "$1" == "logs-multi" || "$1" == "status-multi" ]]; then
  ACTION="$1"
elif [[ "$2" == "start" || "$2" == "stop" || "$2" == "logs" || "$2" == "status" || "$2" == "start-multi" || "$2" == "stop-multi" || "$2" == "logs-multi" || "$2" == "status-multi" ]]; then
  MODEL_HOST_PATH="$1"
  ACTION="$2"
else
  ACTION="$1"
fi

case "$ACTION" in
  start)
    echo "[Docker Manager] Building and starting Gateway + 2 Workers (DotsOCR on GPU 0, Gemma on GPU 1)..."
    export HOST_MODEL_PATH="$MODEL_HOST_PATH"
    export GEMMA4_MODEL_PATH="$GEMMA_MODEL_HOST_PATH"
    $COMPOSE_CMD up -d --build
    echo ""
    echo "======================================================================"
    echo " [Docker Manager] Unified Service Online at http://localhost:8887"
    echo "----------------------------------------------------------------------"
    echo " • OCR Endpoint:      POST http://localhost:8887/v1/ocr"
    echo " • Metadata Endpoint: POST http://localhost:8887/v1/metadata"
    echo " • Engines Discovery: GET  http://localhost:8887/v1/engines"
    echo " • Health Check:      GET  http://localhost:8887/health"
    echo " • GPU 0: DotsOCR (Port 18887) | GPU 1: Gemma (Port 18888)"
    echo "======================================================================"
    ;;

  stop)
    echo "[Docker Manager] Stopping all OCR containers and Gateway..."
    $COMPOSE_CMD down
    ;;

  logs)
    echo "[Docker Manager] Fetching logs across all containers..."
    $COMPOSE_CMD logs -f
    ;;

  status)
    $COMPOSE_CMD ps
    echo ""
    echo "[Health & Engine Status]:"
    curl -s http://localhost:8887/v1/engines | python3 -m json.tool || true
    echo ""
    echo "[GPU Status]:"
    curl -s http://localhost:8887/gpu-status | python3 -m json.tool || true
    ;;

  start-multi)
    echo "[Docker Manager] Building and starting four pinned DotsOCR GPU workers with a router on port 8887..."
    export HOST_MODEL_PATH="$MODEL_HOST_PATH"
    $COMPOSE_CMD -f docker-compose.multi-gpu.yml up -d --build
    echo "[Docker Manager] Multi-GPU OCR API is available at http://localhost:8887/v1/ocr"
    ;;

  stop-multi)
    echo "[Docker Manager] Stopping multi-GPU DotsOCR services..."
    $COMPOSE_CMD -f docker-compose.multi-gpu.yml down
    ;;

  logs-multi)
    $COMPOSE_CMD -f docker-compose.multi-gpu.yml logs -f
    ;;

  status-multi)
    $COMPOSE_CMD -f docker-compose.multi-gpu.yml ps
    curl -s http://localhost:8887/health | python3 -m json.tool
    ;;

  *)
    echo "Usage:"
    echo "  bash run_docker.sh start         # Start Gateway + DotsOCR (GPU 0) + Gemma (GPU 1)"
    echo "  bash run_docker.sh stop          # Stop all containers"
    echo "  bash run_docker.sh logs          # Tail logs for all containers"
    echo "  bash run_docker.sh status        # Check status and health"
    echo "  bash run_docker.sh start-multi   # 4-GPU DotsOCR dedicated cluster"
    exit 1
    ;;
esac
