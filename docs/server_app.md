# DotsOCR Dynamic GPU Image API Service (`server_app.py`)

A high-performance, image-optimized FastAPI service for **DotsOCR** that manages GPU hardware, automatically selects the GPU with the most free VRAM, enforces strict concurrency limits to prevent deadlocks/OOM, and automatically shuts down the backend after **30 minutes of idleness** to free 100% of GPU VRAM.

---

## Streamlined Image Input API

This service is optimized exclusively for **image file uploads** (`.png`, `.jpg`, `.jpeg`, `.webp`). PDF page extraction is handled client-side for maximum speed and lower server overhead.

---

## Deploying / Syncing to Target Server (`rsync`)

To sync the codebase to your target server (`ganesh@10.129.6.170` at `/home/ganesh/kalanjiyam-dotsocr`):

```bash
rsync -avz --progress \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='*.jsonl' \
  ./ ganesh@10.129.6.170:/home/ganesh/kalanjiyam-dotsocr/
```

### Running Docker on the Server

```bash
ssh ganesh@10.129.6.170
cd /home/ganesh/kalanjiyam-dotsocr

# Start Docker container
bash run_docker.sh start
```

### A6000 performance settings

The default starts an API queue of eight requests but restricts the vLLM engine
to one active OCR sequence. This keeps the single-page generation path fast on
the A6000 while accepting bursts of work.

```bash
# Default recommended A6000 configuration
API_MAX_CONCURRENT_REQUESTS=8 VLLM_MAX_NUM_SEQS=1 bash run_docker.sh start

# Optional controlled scheduler experiment; compare total pages/minute and p95 latency.
VLLM_MAX_NUM_BATCHED_TOKENS=8192 bash run_docker.sh start
```

Do not increase `VLLM_MAX_NUM_SEQS` if it reduces total pages/minute. It changes
the number of active model generations, not merely the size of the HTTP queue.

### Safe four-GPU deployment (same client API)

Use the dedicated multi-GPU deployment rather than `GPU_COUNT=4` in one API
container. It starts one pinned worker per GPU and a router that preserves the
existing `http://<host>:8887/ocr` API URL.

```bash
# Stop the single-GPU service first, then start the isolated workers and router.
bash run_docker.sh stop
bash run_docker.sh start-multi
bash run_docker.sh status-multi
```

The workers use API ports `18887` through `18890` and vLLM ports `18000`
through `18003`; only the HAProxy router exposes port `8887` to the client.
An unhealthy worker is removed from routing without restarting the others.

Each worker checks `MIN_FREE_VRAM_MB` before loading the model (default:
`36000`, roughly 35 GB). If another workload leaves insufficient VRAM on a
GPU, that worker responds unhealthy and HAProxy skips it. The check runs again
after an idle unload and before every subsequent model load. Override it only
when necessary, for example: `MIN_FREE_VRAM_MB=38000 bash run_docker.sh start-multi`.
vLLM retains a separate 1 GB driver/allocation reserve by default; change it
with `GPU_MEMORY_HEADROOM_MB` only if your host requires more headroom.

---

## API Endpoints Reference

### 1. `POST /ocr` or `POST /v1/ocr` (Image-Only OCR Endpoint)
Processes an uploaded image file (`.png`, `.jpg`, `.jpeg`, `.webp`).

* **URL**: `http://<SERVER_IP>:8887/ocr` (or `/v1/ocr`)
* **Method**: `POST`
* **Content-Type**: `multipart/form-data`
* **Form Fields**:
  * `file` (or `image`): Image binary data (required)
  * `engine`: Engine to use: `"dots-ocr"` (default) or `"gemma-4"` (optional)
  * `language`: Language code (optional, e.g. `sa`, `en`, `hi`)
  * `max_tokens`: Max generation tokens (optional, default `4096`)

#### Engine Selection
You can set the default engine at server start:
```bash
python server.py --engine gemma-4
# or via environment variable
DEFAULT_ENGINE=gemma-4 python server_app.py
```
Or specify the engine dynamically per request:
```bash
curl -X POST http://localhost:8887/ocr \
  -F "file=@document.jpg" \
  -F "engine=gemma-4"
```

#### Supported Engines Discovery (`GET /v1/engines`)
Returns all supported engines and the currently active engine:
```bash
curl http://localhost:8887/v1/engines
```
```json
{
  "status": "ok",
  "engines": ["dots-ocr", "gemma-4"],
  "current_engine": "dots-ocr",
  "default_engine": "dots-ocr"
}
```

#### Sample Output Response (`POST /v1/ocr`) (v2.1 Contract)
```json
{
  "contract_version": "2.1",
  "engine": "dots_ocr",
  "model": {
    "name": "dots-ocr",
    "version": "4.0.0"
  },
  "page_confidence": 0.942,
  "engine_latency_ms": 342.5,
  "page_width": 1240,
  "page_height": 1754,
  "blocks": [
    {
      "id": "b1",
      "type": "heading",
      "bbox": [120.0, 40.0, 980.0, 88.0],
      "reading_order": 1,
      "content": "Chapter Title Text",
      "confidence": 0.985,
      "words": [
        {"text": "Chapter", "bbox": [120.0, 40.0, 550.0, 88.0], "confidence": 0.985},
        {"text": "Title", "bbox": [550.0, 40.0, 800.0, 88.0], "confidence": 0.985},
        {"text": "Text", "bbox": [800.0, 40.0, 980.0, 88.0], "confidence": 0.985}
      ]
    },
    {
      "id": "b2",
      "type": "paragraph",
      "bbox": [120.0, 100.0, 980.0, 280.0],
      "reading_order": 2,
      "content": "First line of body text.\nSecond line of body text.",
      "confidence": 0.912,
      "words": [
        {"text": "First", "bbox": [120.0, 100.0, 270.0, 190.0], "confidence": 0.912},
        {"text": "line", "bbox": [270.0, 100.0, 420.0, 190.0], "confidence": 0.912}
      ]
    }
  ]
}
```

---

### 2. `GET /gpu-status`
Query active GPU VRAM, hardware presence, and idle timeout status.

---

### 3. `POST /free-vram`
Manually stop the backend model process and free 100% VRAM.

---

### 4. `GET /health`
Service status check.
