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
    echo "[Docker Manager] Building and starting DotsOCR container..."
    echo "[Docker Manager] Model Path/Repo: $MODEL_HOST_PATH"
    echo "[Docker Manager] GPUs: ${GPU_COUNT:-1} | API queue: ${API_MAX_CONCURRENT_REQUESTS:-8} | vLLM base port: ${VLLM_PORT:-8000} | vLLM sequences/GPU: ${VLLM_MAX_NUM_SEQS:-1} | GPU memory target: ${GPU_MEMORY_UTILIZATION:-0.90}"
    export HOST_MODEL_PATH="$MODEL_HOST_PATH"
    $COMPOSE_CMD up -d --build
    echo "[Docker Manager] Container started! Endpoint available at http://localhost:8887/v1/ocr"
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
    echo "  bash run_docker.sh start"
    echo "  bash run_docker.sh stop"
    echo "  bash run_docker.sh logs"
    echo "  bash run_docker.sh status"
    echo "  bash run_docker.sh start-multi"
    echo "  bash run_docker.sh stop-multi"
    echo "  bash run_docker.sh logs-multi"
    echo "  bash run_docker.sh status-multi"
    echo "  bash run_docker.sh [CUSTOM_MODEL_PATH] start"
    echo ""
    echo "Performance tuning examples:"
    echo "  # Recommended A6000 baseline: queue callers, execute one OCR generation"
    echo "  API_MAX_CONCURRENT_REQUESTS=8 VLLM_MAX_NUM_SEQS=1 bash run_docker.sh start"
    echo "  # Use all four A6000 GPUs behind the same http://localhost:8887 API"
    echo "  bash run_docker.sh start-multi"
    echo "  # Controlled scheduler experiment"
    echo "  VLLM_MAX_NUM_BATCHED_TOKENS=8192 bash run_docker.sh start"
    exit 1
    ;;
esac
