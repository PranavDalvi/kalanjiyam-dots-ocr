import os
import re
import json
import time
from typing import Optional, Dict, Any
import httpx
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

# Worker Backend URLs
DOTSOCR_WORKER_URL = os.getenv("DOTSOCR_WORKER_URL", "http://127.0.0.1:18887").rstrip("/")
GEMMA_WORKER_URL = os.getenv("GEMMA_WORKER_URL", "http://127.0.0.1:18888").rstrip("/")
ENABLE_GEMMA = os.getenv("ENABLE_GEMMA", "false").strip().lower() in ("1", "true", "yes", "on")
GATEWAY_PORT = int(os.getenv("GATEWAY_PORT") or "8887")

app = FastAPI(
    title="Kalanjiyam OCR & Metadata Multi-Engine API Gateway",
    description="High-performance Unified Gateway routing requests to dedicated DotsOCR and Gemma GPU workers.",
    version="1.0.0"
)

# HTTP Client with connection pooling and high timeout for vision model inference
http_client = httpx.AsyncClient(
    timeout=httpx.Timeout(600.0, connect=10.0),
    limits=httpx.Limits(max_keepalive_connections=50, max_connections=200)
)

def log(step: str, message: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [GATEWAY {step}] {message}", flush=True)


def resolve_engine_from_string(val: Optional[str]) -> str:
    if not val:
        return "dots-ocr"
    cleaned = str(val).strip().lower()
    if "gemma" in cleaned or "archival" in cleaned or "metadata" in cleaned:
        if ENABLE_GEMMA:
            return "gemma-4"
        return "dots-ocr"
    return "dots-ocr"


def detect_engine(request: Request, body_bytes: bytes) -> str:
    if not ENABLE_GEMMA:
        return "dots-ocr"

    # 1. Check path
    path = request.url.path.lower()
    if "/metadata" in path:
        return "gemma-4"

    # 2. Check query params
    if "engine" in request.query_params:
        return resolve_engine_from_string(request.query_params.get("engine"))

    # 3. Check custom headers
    if "x-engine" in request.headers:
        return resolve_engine_from_string(request.headers.get("x-engine"))
    if "engine" in request.headers:
        return resolve_engine_from_string(request.headers.get("engine"))

    # 4. Check body content
    content_type = request.headers.get("content-type", "").lower()

    # JSON payload
    if "application/json" in content_type and body_bytes:
        try:
            data = json.loads(body_bytes)
            if isinstance(data, dict) and "engine" in data:
                return resolve_engine_from_string(data["engine"])
        except Exception:
            pass

    # Multipart form-data or URL encoded
    if body_bytes:
        try:
            # Search for 'engine' field in form data using regex (first 4KB of payload)
            prefix = body_bytes[:4096]
            # Match multipart: name="engine"\r\n\r\nvalue
            match = re.search(rb'name="engine"(?:;[^\r\n]*)?\r?\n\r?\n([^\r\n]+)', prefix, re.IGNORECASE)
            if match:
                val = match.group(1).decode("utf-8", errors="ignore").strip()
                return resolve_engine_from_string(val)
            # Match urlencoded: engine=value
            match_url = re.search(rb'(?:^|&)engine=([^&]+)', prefix, re.IGNORECASE)
            if match_url:
                val = match_url.group(1).decode("utf-8", errors="ignore").strip()
                return resolve_engine_from_string(val)
        except Exception:
            pass

    # Default fallback to dots-ocr
    return "dots-ocr"


def get_worker_url(engine: str) -> str:
    if engine == "gemma-4" and ENABLE_GEMMA:
        return GEMMA_WORKER_URL
    return DOTSOCR_WORKER_URL


# Forwarding logic for all endpoints
async def proxy_request(request: Request, target_base_url: str, body_bytes: bytes) -> Response:
    target_url = f"{target_base_url}{request.url.path}"

    # Copy headers except hop-by-hop headers
    forward_headers = {}
    for key, value in request.headers.items():
        if key.lower() not in ("host", "content-length", "connection"):
            forward_headers[key] = value

    try:
        req = http_client.build_request(
            method=request.method,
            url=target_url,
            headers=forward_headers,
            params=request.query_params,
            content=body_bytes
        )
        resp = await http_client.send(req, stream=True)

        response_headers = {}
        for key, value in resp.headers.items():
            if key.lower() not in ("content-length", "transfer-encoding", "connection", "content-encoding"):
                response_headers[key] = value

        return StreamingResponse(
            resp.aiter_raw(),
            status_code=resp.status_code,
            headers=response_headers,
            background=resp.aclose
        )
    except httpx.ConnectError:
        log("ERROR", f"Failed to connect to worker backend at {target_base_url}")
        return JSONResponse(
            status_code=503,
            content={"error": f"Worker backend at {target_base_url} is currently unavailable."}
        )
    except Exception as e:
        log("ERROR", f"Proxy error to {target_url}: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Gateway proxy failed: {str(e)}"}
        )


@app.get("/v1/engines")
@app.get("/engines")
async def get_engines():
    """Aggregates active engines across all backend workers."""
    dots_status = "offline"

    try:
        r = await http_client.get(f"{DOTSOCR_WORKER_URL}/v1/engines", timeout=2.0)
        if r.status_code == 200:
            dots_status = "online"
    except Exception:
        pass

    engines = ["dots-ocr"]
    workers = {
        "dots-ocr": {"url": DOTSOCR_WORKER_URL, "status": dots_status, "gpu": 0}
    }

    if ENABLE_GEMMA:
        gemma_status = "offline"
        try:
            r = await http_client.get(f"{GEMMA_WORKER_URL}/v1/engines", timeout=2.0)
            if r.status_code == 200:
                gemma_status = "online"
        except Exception:
            pass
        engines.append("gemma-4")
        workers["gemma-4"] = {"url": GEMMA_WORKER_URL, "status": gemma_status, "gpu": 1}

    return {
        "status": "ok",
        "engines": engines,
        "workers": workers,
        "default_engine": "dots-ocr",
        "gemma_enabled": ENABLE_GEMMA,
    }


@app.get("/health")
async def health_check(engine: Optional[str] = None):
    """Health check for gateway and workers."""
    target_engine = resolve_engine_from_string(engine) if engine else None

    if engine and ("gemma" in engine.lower() or "archival" in engine.lower() or "metadata" in engine.lower()):
        if not ENABLE_GEMMA:
            return JSONResponse(
                status_code=503,
                content={"status": "disabled", "engine": "gemma-4", "error": "Gemma engine is currently disabled (ENABLE_GEMMA=false)."}
            )

    if target_engine == "dots-ocr":
        try:
            r = await http_client.get(f"{DOTSOCR_WORKER_URL}/health", timeout=3.0)
            return JSONResponse(status_code=r.status_code, content=r.json())
        except Exception as e:
            return JSONResponse(status_code=503, content={"status": "unavailable", "engine": "dots-ocr", "error": str(e)})

    elif target_engine == "gemma-4":
        if not ENABLE_GEMMA:
            return JSONResponse(
                status_code=503,
                content={"status": "disabled", "engine": "gemma-4", "error": "Gemma engine is currently disabled (ENABLE_GEMMA=false)."}
            )
        try:
            r = await http_client.get(f"{GEMMA_WORKER_URL}/health", timeout=3.0)
            return JSONResponse(status_code=r.status_code, content=r.json())
        except Exception as e:
            return JSONResponse(status_code=503, content={"status": "unavailable", "engine": "gemma-4", "error": str(e)})

    else:
        # Check active workers
        dots_healthy = False
        try:
            r = await http_client.get(f"{DOTSOCR_WORKER_URL}/health", timeout=3.0)
            dots_healthy = (r.status_code == 200)
        except Exception:
            pass

        workers_status = {
            "dots-ocr": "healthy" if dots_healthy else "unhealthy"
        }

        if ENABLE_GEMMA:
            gemma_healthy = False
            try:
                r = await http_client.get(f"{GEMMA_WORKER_URL}/health", timeout=3.0)
                gemma_healthy = (r.status_code == 200)
            except Exception:
                pass
            workers_status["gemma-4"] = "healthy" if gemma_healthy else "unhealthy"
            is_healthy = dots_healthy or gemma_healthy
        else:
            is_healthy = dots_healthy

        return JSONResponse(
            status_code=200 if is_healthy else 503,
            content={
                "status": "healthy" if is_healthy else "unavailable",
                "workers": workers_status,
                "gemma_enabled": ENABLE_GEMMA,
            }
        )


@app.get("/gpu-status")
async def gpu_status():
    """Queries GPU status from workers."""
    dots_gpu = {}
    try:
        r = await http_client.get(f"{DOTSOCR_WORKER_URL}/gpu-status", timeout=3.0)
        if r.status_code == 200:
            dots_gpu = r.json()
    except Exception:
        pass

    payload = {
        "gateway_port": GATEWAY_PORT,
        "dots_worker": dots_gpu,
        "gemma_enabled": ENABLE_GEMMA,
    }

    if ENABLE_GEMMA:
        gemma_gpu = {}
        try:
            r = await http_client.get(f"{GEMMA_WORKER_URL}/gpu-status", timeout=3.0)
            if r.status_code == 200:
                gemma_gpu = r.json()
        except Exception:
            pass
        payload["gemma_worker"] = gemma_gpu

    return payload


@app.post("/free-vram")
async def free_vram():
    """Frees VRAM across all active workers."""
    workers = [("dots-ocr", DOTSOCR_WORKER_URL)]
    if ENABLE_GEMMA:
        workers.append(("gemma-4", GEMMA_WORKER_URL))

    results = {}
    for name, url in workers:
        try:
            r = await http_client.post(f"{url}/free-vram", timeout=5.0)
            results[name] = r.json() if r.status_code == 200 else "failed"
        except Exception as e:
            results[name] = f"error: {str(e)}"
    return {"status": "ok", "results": results}


@app.post("/v1/ocr")
async def proxy_ocr(request: Request):
    """Routes OCR requests to the selected engine worker."""
    body_bytes = await request.body()
    engine = detect_engine(request, body_bytes)
    worker_url = get_worker_url(engine)
    log("ROUTE", f"Routing OCR request to {engine} worker ({worker_url})...")
    return await proxy_request(request, worker_url, body_bytes)


@app.post("/v1/metadata")
async def proxy_metadata(request: Request):
    """Routes Archival Metadata Extraction requests directly to Gemma worker."""
    if not ENABLE_GEMMA:
        raise HTTPException(
            status_code=503,
            detail="Gemma / Archival Metadata worker is currently disabled. Set ENABLE_GEMMA=true to enable."
        )
    body_bytes = await request.body()
    log("ROUTE", f"Routing Metadata request to Gemma worker ({GEMMA_WORKER_URL})...")
    return await proxy_request(request, GEMMA_WORKER_URL, body_bytes)


# Catch-all proxy for any other API endpoints
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"])
async def catch_all(request: Request, path: str):
    body_bytes = await request.body()
    engine = detect_engine(request, body_bytes)
    worker_url = get_worker_url(engine)
    return await proxy_request(request, worker_url, body_bytes)


if __name__ == "__main__":
    import uvicorn
    log("STARTUP", f"Starting Kalanjiyam OCR Unified Gateway on port {GATEWAY_PORT}...")
    log("STARTUP", f"  DotsOCR worker target: {DOTSOCR_WORKER_URL} (GPU 0)")
    if ENABLE_GEMMA:
        log("STARTUP", f"  Gemma worker target:   {GEMMA_WORKER_URL} (GPU 1) [ENABLED]")
    else:
        log("STARTUP", f"  Gemma worker:          DISABLED (ENABLE_GEMMA=false)")
    uvicorn.run(app, host="0.0.0.0", port=GATEWAY_PORT, reload=False)
