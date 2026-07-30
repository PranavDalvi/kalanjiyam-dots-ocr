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

---

## API Endpoints Reference

### 1. `POST /ocr` (Image-Only OCR Endpoint)
Processes an uploaded image file (`.png`, `.jpg`, `.jpeg`, `.webp`).

* **URL**: `http://<SERVER_IP>:8887/ocr`
* **Method**: `POST`
* **Content-Type**: `multipart/form-data`
* **Form Field**: `file` (or `image`)

#### Sample Output Response
```json
{
  "status": "success",
  "filename": "page_1.jpg",
  "gpu_assigned": 0,
  "results": [
    {
      "bbox": [102, 45, 890, 110],
      "category": "title",
      "text": "# Financial Report 2026"
    },
    {
      "bbox": [102, 130, 890, 650],
      "category": "text",
      "text": "Transcribed text content from the image..."
    }
  ],
  "metrics": {
    "time_taken_seconds": 0.2854,
    "prompt_tokens": 1024,
    "completion_tokens": 180,
    "total_tokens": 1204,
    "generation_speed_tok_per_sec": 630.69
  }
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
