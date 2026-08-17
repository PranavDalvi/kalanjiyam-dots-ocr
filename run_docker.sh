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

MODEL_HOST_PATH="${HOST_MODEL_PATH:-rednote-hilab/dots.ocr}"
GEMMA_MODEL_HOST_PATH="${GEMMA4_MODEL_PATH:-google/gemma-4-26B-A4B-it}"
ACTION=""

if [[ "$1" =~ ^(start|stop|restart|rebuild|build|logs|status|health|start-single|stop-single|logs-single|start-multi|stop-multi|logs-multi|status-multi)$ ]]; then
  ACTION="$1"
elif [[ "$2" =~ ^(start|stop|restart|rebuild|build|logs|status|health|start-single|stop-single|logs-single|start-multi|stop-multi|logs-multi|status-multi)$ ]]; then
  MODEL_HOST_PATH="$1"
  ACTION="$2"
else
  ACTION="$1"
fi

export HOST_MODEL_PATH="$MODEL_HOST_PATH"
export GEMMA4_MODEL_PATH="$GEMMA_MODEL_HOST_PATH"

case "$ACTION" in
  build|rebuild)
    echo "[Docker Manager] Building all container images with --no-cache..."
    $COMPOSE_CMD build --no-cache
    echo "[Docker Manager] Build complete."
    ;;

  start)
    echo "[Docker Manager] Building and starting Gateway + 2 Workers (DotsOCR on GPU 0, Gemma on GPU 1)..."
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

  restart)
    echo "[Docker Manager] Restarting all services..."
    $COMPOSE_CMD down
    $COMPOSE_CMD up -d --build
    ;;

  stop)
    echo "[Docker Manager] Stopping all OCR containers and Gateway..."
    $COMPOSE_CMD down
    ;;

  logs)
    echo "[Docker Manager] Fetching logs across all containers..."
    $COMPOSE_CMD logs -f
    ;;

  status|health)
    $COMPOSE_CMD ps
    echo ""
    echo "[Health & Engine Status]:"
    curl -s http://localhost:8887/v1/engines | python3 -m json.tool || true
    echo ""
    echo "[GPU Status]:"
    curl -s http://localhost:8887/gpu-status | python3 -m json.tool || true
    ;;

  start-single)
    echo "[Docker Manager] Building and starting Standalone Single Container on Port 8887..."
    docker build --no-cache -t dotsocr-fastapi-service:latest -f Dockerfile .
    docker rm -f dotsocr_standalone 2>/dev/null || true
    docker run -d \
      --name dotsocr_standalone \
      --gpus all \
      --network host \
      -v ~/.cache/huggingface:/root/.cache/huggingface \
      -v "$(pwd)":/workspace \
      -e MODEL_PATH="$HOST_MODEL_PATH" \
      -e GEMMA4_MODEL_PATH="$GEMMA_MODEL_HOST_PATH" \
      -e FASTAPI_PORT=8887 \
      -e VLLM_PORT=8000 \
      dotsocr-fastapi-service:latest
    echo "[Docker Manager] Standalone container online at http://localhost:8887"
    ;;

  stop-single)
    echo "[Docker Manager] Stopping Standalone Single Container..."
    docker rm -f dotsocr_standalone 2>/dev/null || true
    ;;

  logs-single)
    docker logs -f dotsocr_standalone
    ;;

  start-multi)
    echo "[Docker Manager] Building and starting four pinned DotsOCR GPU workers with a router on port 8887..."
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
    echo "Usage: bash run_docker.sh <command>"
    echo ""
    echo "Commands:"
    echo "  start         # Build & start Gateway + DotsOCR Worker + Gemma Worker (docker-compose)"
    echo "  stop          # Stop all compose containers"
    echo "  restart       # Restart all compose containers"
    echo "  rebuild       # Rebuild all images with --no-cache"
    echo "  logs          # Tail logs for all compose containers"
    echo "  status        # Check status and engine discovery"
    echo "  start-single  # Run standalone single-container service on port 8887"
    echo "  stop-single   # Stop standalone single-container"
    echo "  logs-single   # Tail logs for standalone single-container"
    echo "  start-multi   # 4-GPU DotsOCR dedicated cluster"
    echo "  stop-multi    # Stop 4-GPU DotsOCR dedicated cluster"
    exit 1
    ;;
esac
