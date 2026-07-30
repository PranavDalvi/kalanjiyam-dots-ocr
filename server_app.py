import base64
import time
import json
import os
import glob
import subprocess
import signal
import threading
import asyncio
import sys
from typing import Optional
import requests
import io
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from PIL import Image

app = FastAPI(
    title="DotsOCR Dynamic GPU Image API Service",
    description="Streamlined FastAPI service optimized exclusively for image inputs (PNG, JPG, JPEG, WEBP) with dynamic GPU VRAM allocation and step-by-step console logging.",
    version="4.0.0"
)

# Configuration defaults
IDLE_TIMEOUT_SECONDS = int(os.getenv("IDLE_TIMEOUT_SECONDS", "1800"))  # 30 minutes
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "2")) # Concurrency limit (default: 2)
VLLM_PORT = int(os.getenv("VLLM_PORT", "8000"))
VLLM_BASE_URL = f"http://localhost:{VLLM_PORT}/v1/chat/completions"
MODEL_PATH = os.getenv("MODEL_PATH", "rednote-hilab/dots.ocr")
MODEL_NAME = "model"

# Asyncio Semaphore to restrict maximum parallel in-flight OCR requests at API level
request_semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

DOTSOCR_PROMPT = """please output the layout information from the pdf image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. bbox format: [x1, y1, x2, y2]

2. layout categories: the possible categories are ['caption', 'footnote', 'formula', 'list-item', 'page-footer', 'page-header', 'picture', 'section-header', 'table', 'text', 'title'].

3. text extraction & formatting rules:
    - picture: for the 'picture' category, the text field should be omitted.
    - formula: format its text as latex.
    - table: format its text as html.
    - all others (text, title, etc.): format their text as markdown.

4. constraints:
    - the output text must be the original text from the image, with no translation.
    - all layout elements must be sorted according to human reading order.

5. final output: the entire output must be a single json object."""


def log(step: str, message: str):
    """Helper for formatted step-by-step console logging."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{step}] {message}", flush=True)


# =============================================================================
# GPU & PROCESS MANAGEMENT UTILITIES
# =============================================================================

def ensure_model_downloaded() -> str:
    """
    Ensures model weights are downloaded to a local directory, config.json
    has auto_map entries, and vLLM's native registration is installed.
    """
    target_dir = "/root/.cache/weights/DotsOCR"

    # If already downloaded, just verify config
    if os.path.exists(target_dir) and os.path.exists(os.path.join(target_dir, "config.json")):
        _ensure_config_auto_map(target_dir)
        _install_vllm_native_registration(target_dir)
        return target_dir

    # If MODEL_PATH is already a local directory with model files
    if os.path.exists(MODEL_PATH) and os.path.isdir(MODEL_PATH) and os.path.exists(os.path.join(MODEL_PATH, "config.json")):
        _ensure_config_auto_map(MODEL_PATH)
        _install_vllm_native_registration(MODEL_PATH)
        return MODEL_PATH

    log("MODEL DOWNLOAD", f"Pre-downloading model '{MODEL_PATH}' to local folder '{target_dir}'...")
    try:
        from huggingface_hub import snapshot_download
        os.makedirs(target_dir, exist_ok=True)
        snapshot_download(
            repo_id=MODEL_PATH,
            local_dir=target_dir
        )
        log("MODEL DOWNLOAD", "Model weights downloaded successfully to local folder!")
        _ensure_config_auto_map(target_dir)
        _install_vllm_native_registration(target_dir)
        return target_dir
    except Exception as e:
        log("MODEL WARN", f"Snapshot download failed ({e}). Using raw path '{MODEL_PATH}'.")
        return MODEL_PATH


def _ensure_config_auto_map(model_dir: str):
    """
    Ensures config.json has auto_map entries for DotsOCRForCausalLM.
    Without auto_map, vLLM cannot resolve the custom architecture from a local directory.
    """
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        log("CONFIG WARN", f"No config.json found at {config_path}")
        return

    try:
        with open(config_path, "r") as f:
            config = json.load(f)

        log("CONFIG CHECK", f"config.json auto_map: {config.get('auto_map', 'MISSING')}")
        log("CONFIG CHECK", f"config.json architectures: {config.get('architectures', 'MISSING')}")

        needs_update = False

        # Ensure auto_map exists with correct entries
        if "auto_map" not in config:
            config["auto_map"] = {}
            needs_update = True

        required_mappings = {
            "AutoConfig": "configuration_dots.DotsOCRConfig",
            "AutoModel": "modeling_dots_ocr.DotsOCRForCausalLM",
            "AutoModelForCausalLM": "modeling_dots_ocr.DotsOCRForCausalLM",
        }

        for key, value in required_mappings.items():
            if key not in config["auto_map"]:
                config["auto_map"][key] = value
                needs_update = True
                log("CONFIG FIX", f"Added auto_map['{key}'] = '{value}'")

        if needs_update:
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            log("CONFIG FIX", f"Updated config.json at '{config_path}'")
        else:
            log("CONFIG OK", "config.json auto_map already has required entries.")

    except Exception as e:
        log("CONFIG WARN", f"Failed to check/update config.json: {e}")


def _install_vllm_native_registration(model_dir: str):
    """
    Register DotsOCRForCausalLM from the native vLLM implementation file
    (modeling_dots_ocr_vllm.py) by wrapping vLLM's engine spawn function.

    Strategy: Patch vllm/engine/multiprocessing/engine.py to wrap
    run_mp_engine() with model registration. The registration runs inside
    the function body (not at module import time), so vllm is fully
    initialized and there are ZERO circular import issues.

    Without this, vLLM falls back to a generic TransformersModel wrapper
    that crashes on DotsOCR's custom vision module constructors.
    """
    vllm_file = os.path.join(model_dir, "modeling_dots_ocr_vllm.py")
    if not os.path.exists(vllm_file):
        log("REGISTRATION WARN", f"No modeling_dots_ocr_vllm.py found at {vllm_file}")
        return

    # Step 1: Create __init__.py so the model dir is a proper Python package
    init_path = os.path.join(model_dir, "__init__.py")
    if not os.path.exists(init_path):
        with open(init_path, "w") as f:
            f.write("# Auto-generated to enable package-style imports\n")
        log("REGISTRATION", f"Created {init_path}")

    try:
        import shutil

        # Step 2: Clean up any old patches from models/__init__.py
        import vllm.model_executor.models
        models_init_path = vllm.model_executor.models.__file__
        with open(models_init_path, "r") as f:
            models_content = f.read()
        if "DotsOCR" in models_content:
            marker = "\n# ---- DotsOCR"
            idx = models_content.find(marker)
            if idx > 0:
                models_content = models_content[:idx].rstrip() + "\n"
                with open(models_init_path, "w") as f:
                    f.write(models_content)
                log("REGISTRATION", "Removed old patch from models/__init__.py")

        # Step 3: Patch vllm/engine/multiprocessing/engine.py
        import vllm.engine.multiprocessing.engine
        engine_path = vllm.engine.multiprocessing.engine.__file__

        wrapper_block = '''
# ---- DotsOCR Engine Registration (auto-patched) ----
# Module-level registration: runs in BOTH the main HTTP server process
# and spawned worker processes. engine.py is imported AFTER vllm is
# fully initialized, so there are zero circular import issues.
# The @MULTIMODAL_REGISTRY.register_processor decorator fires on import,
# which registers image support in the main process (needed for request parsing).
import sys as _dotsocr_sys
if '/root/.cache/weights' not in _dotsocr_sys.path:
    _dotsocr_sys.path.insert(0, '/root/.cache/weights')
try:
    from DotsOCR.modeling_dots_ocr_vllm import DotsOCRForCausalLM as _DotsOCRCls
    from vllm.model_executor.models import ModelRegistry as _MR
    if "DotsOCRForCausalLM" not in _MR.get_supported_archs():
        _MR.register_model("DotsOCRForCausalLM", _DotsOCRCls)
    print("[DotsOCR-Reg] Native model registered!", flush=True)
except Exception as _dotsocr_err:
    print(f"[DotsOCR-Reg] Module-level FAILED: {_dotsocr_err} (will retry in worker)", flush=True)
    # Fallback: wrap run_mp_engine for spawned process registration
    _dotsocr_orig_run = run_mp_engine
    def _dotsocr_fallback_run(*a, **kw):
        try:
            from DotsOCR.modeling_dots_ocr_vllm import DotsOCRForCausalLM as _C
            from vllm.model_executor.models import ModelRegistry as _R
            if "DotsOCRForCausalLM" not in _R.get_supported_archs():
                _R.register_model("DotsOCRForCausalLM", _C)
                print("[DotsOCR-Reg] Registered in worker (fallback)!", flush=True)
        except Exception as _e2:
            print(f"[DotsOCR-Reg] Fallback FAILED: {_e2}", flush=True)
        return _dotsocr_orig_run(*a, **kw)
    run_mp_engine = _dotsocr_fallback_run
# ---- End DotsOCR Engine Registration ----
'''

        with open(engine_path, "r") as f:
            engine_content = f.read()

        if "DotsOCR Engine Registration" not in engine_content:
            with open(engine_path, "a") as f:
                f.write(wrapper_block)
            # Clear bytecode cache
            pycache_dir = os.path.join(os.path.dirname(engine_path), "__pycache__")
            if os.path.exists(pycache_dir):
                shutil.rmtree(pycache_dir, ignore_errors=True)
            log("REGISTRATION", f"Patched {engine_path} with DotsOCR engine wrapper")
        else:
            log("REGISTRATION", "DotsOCR engine wrapper already present")

    except Exception as e:
        log("REGISTRATION WARN", f"Failed to install registration: {e}")


def get_gpu_info() -> list[dict]:
    """Query nvidia-smi for all GPUs, returning index, memory used, free, and total in MB."""
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits"
        ]
        output = subprocess.check_output(cmd, encoding="utf-8").strip()
        gpus = []
        for line in output.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 4:
                gpus.append({
                    "index": int(parts[0]),
                    "total_mb": int(parts[1]),
                    "used_mb": int(parts[2]),
                    "free_mb": int(parts[3])
                })
        return gpus
    except Exception as e:
        log("GPU WARN", f"nvidia-smi query failed: {e}")
        return []


def select_best_gpu() -> tuple[int, float]:
    """
    Select the GPU index with the most free VRAM and dynamically calculate a safe GPU memory utilization ratio.
    """
    gpus = get_gpu_info()
    if not gpus:
        raise RuntimeError("No NVIDIA GPUs detected on server!")

    sorted_gpus = sorted(gpus, key=lambda g: g["free_mb"], reverse=True)
    best_gpu = sorted_gpus[0]

    total_mb = best_gpu["total_mb"]
    free_mb = best_gpu["free_mb"]

    if free_mb < 4000:
        raise RuntimeError(f"GPU {best_gpu['index']} has only {free_mb} MB free VRAM. Minimum 4GB required for DotsOCR.")

    calculated_util = (free_mb / total_mb) * 0.90
    safe_utilization = round(max(0.25, min(0.85, calculated_util)), 2)

    log("GPU SELECT", f"Picked GPU {best_gpu['index']} ({free_mb} MB free / {total_mb} MB total). Dynamic GPU utilization set to {safe_utilization} (~{int(total_mb * safe_utilization)} MB).")
    return best_gpu["index"], safe_utilization


class GPUProcessManager:
    """Manages lifecycle of vLLM backend: auto-start, health checks, and 30-min idle auto-shutdown."""

    def __init__(self):
        self.process = None
        self.active_gpu = None
        self.last_active_timestamp = time.time()
        self.lock = threading.Lock()
        self.monitor_thread = threading.Thread(target=self._idle_monitor, daemon=True)
        self.monitor_thread.start()

    def touch(self):
        """Update last active timestamp on incoming requests."""
        self.last_active_timestamp = time.time()

    def is_running(self) -> bool:
        """Check if vLLM backend is active and responding to health check."""
        if self.process and self.process.poll() is None:
            try:
                resp = requests.get(f"http://localhost:{VLLM_PORT}/v1/models", timeout=2)
                return resp.status_code == 200
            except Exception:
                return False
        return False

    def _stream_logs(self, proc):
        """Continuously read vLLM subprocess stdout line-by-line to prevent pipe buffer deadlock."""
        for line in iter(proc.stdout.readline, ""):
            if not line:
                break
            print(f"[vLLM] {line.strip()}", flush=True)

    def start_backend(self) -> int:
        """Start vLLM process on the GPU with the most free VRAM if not running."""
        with self.lock:
            if self.is_running():
                return self.active_gpu

            # Ensure model is downloaded locally and config.json has auto_map
            effective_model_path = ensure_model_downloaded()

            log("BACKEND START", "Model backend is offline. Auto-selecting best GPU...")
            gpu_idx, safe_utilization = select_best_gpu()
            self.active_gpu = gpu_idx

            log("BACKEND LAUNCH", f"Launching vLLM process on GPU {gpu_idx} (Path: {effective_model_path}, Memory Util: {safe_utilization})...")

            # Build PYTHONPATH: include model dir AND its parent so:
            # - transformers can resolve auto_map entries
            # - DotsOCR package imports work (from DotsOCR.modeling_dots_ocr_vllm)
            model_parent = os.path.dirname(effective_model_path)
            extra_paths = [effective_model_path, model_parent, "/workspace", "/root/.cache/weights"]
            pythonpath_str = ":".join(extra_paths) + ":" + os.environ.get("PYTHONPATH", "")

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
            env["VLLM_USE_V1"] = "0"
            env["PYTHONPATH"] = pythonpath_str

            cmd = [
                "vllm", "serve", effective_model_path,
                "--tensor-parallel-size", "1",
                "--gpu-memory-utilization", str(safe_utilization),
                "--max-num-seqs", str(MAX_CONCURRENT_REQUESTS),
                "--trust-remote-code",
                "--served-model-name", MODEL_NAME,
                "--port", str(VLLM_PORT)
            ]

            self.process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                preexec_fn=os.setsid if os.name != "nt" else None
            )

            log_reader = threading.Thread(target=self._stream_logs, args=(self.process,), daemon=True)
            log_reader.start()

            log("BACKEND WAIT", "Loading model weights into GPU VRAM (streaming vLLM output below)...")
            start_wait = time.time()
            ready = False

            while time.time() - start_wait < 300:
                if self.is_running():
                    ready = True
                    break
                if self.process.poll() is not None:
                    log("BACKEND ERROR", f"vLLM process exited unexpectedly with code {self.process.poll()}")
                    break
                time.sleep(3)

            if not ready:
                self.stop_backend()
                raise RuntimeError("Failed to start vLLM backend within 300 seconds. Check vLLM log output above.")

            log("BACKEND READY", f"vLLM model backend is ONLINE and READY on GPU {gpu_idx}!")
            self.touch()
            return gpu_idx

    def stop_backend(self):
        """Stop vLLM backend and free 100% VRAM."""
        with self.lock:
            if self.process:
                log("VRAM CLEANUP", f"Stopping vLLM process on GPU {self.active_gpu} to free 100% VRAM...")
                try:
                    if os.name != "nt":
                        os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                    else:
                        self.process.kill()
                except Exception as e:
                    log("VRAM WARN", f"Error stopping process: {e}")

                self.process = None
                self.active_gpu = None
                log("VRAM FREED", "Backend stopped successfully. GPU VRAM is completely free.")

    def _idle_monitor(self):
        """Background thread checking for 30-minute idle timeout."""
        while True:
            time.sleep(30)
            if self.process and self.process.poll() is None:
                idle_time = time.time() - self.last_active_timestamp
                if idle_time > IDLE_TIMEOUT_SECONDS:
                    log("IDLE TIMEOUT", f"No requests for {int(idle_time)}s (limit: {IDLE_TIMEOUT_SECONDS}s). Triggering auto-shutdown...")
                    self.stop_backend()


gpu_manager = GPUProcessManager()


# =============================================================================
# FASTAPI ENDPOINTS
# =============================================================================

def convert_image_bytes_to_base64_uri(image_bytes: bytes) -> str:
    """Fast conversion of raw image bytes to JPEG base64 Data URI."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG", quality=95)
        b64_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64_str}"
    except Exception as err:
        raise ValueError(f"Invalid image format or corrupted image file: {str(err)}")


@app.get("/gpu-status")
def gpu_status():
    """Query current GPU VRAM usage, concurrency limits, and backend status."""
    gpus = get_gpu_info()
    return {
        "gpu_present": len(gpus) > 0,
        "gpu_count": len(gpus),
        "active_gpu": gpu_manager.active_gpu,
        "backend_running": gpu_manager.is_running(),
        "max_concurrent_limit": MAX_CONCURRENT_REQUESTS,
        "idle_seconds": round(time.time() - gpu_manager.last_active_timestamp, 1),
        "idle_timeout_seconds": IDLE_TIMEOUT_SECONDS,
        "gpus": gpus
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "backend_running": gpu_manager.is_running(),
        "active_gpu": gpu_manager.active_gpu,
        "max_concurrent_limit": MAX_CONCURRENT_REQUESTS
    }


@app.post("/ocr")
async def run_ocr(
    file: Optional[UploadFile] = File(None, description="Image file (PNG, JPG, JPEG, WEBP)"),
    image: Optional[UploadFile] = File(None, description="Alternative field name for image file"),
    max_tokens: Optional[int] = Form(None, description="Max generation tokens")
):
    """
    Fast image-only OCR endpoint with dynamic GPU VRAM allocation & step-by-step console logging.
    """
    target_file = file or image
    if not target_file:
        log("REQ ERROR", "Request rejected: No file provided in form-data field 'file' or 'image'")
        raise HTTPException(
            status_code=400,
            detail="No image uploaded. Please provide an image file in form-data field 'file' or 'image'."
        )

    eff_max_tokens = max_tokens if (max_tokens is not None and max_tokens > 0) else 4096

    filename = target_file.filename or "uploaded_image.jpg"
    log("REQ RECEIVED", f"Processing image: '{filename}' (max_tokens: {eff_max_tokens})")

    async with request_semaphore:
        gpu_manager.touch()
        try:
            active_gpu = gpu_manager.start_backend()
        except Exception as e:
            log("REQ ERROR", f"GPU initialization failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize GPU model backend: {str(e)}")

        log("PREPROCESSING", f"Reading image bytes for '{filename}'...")
        file_bytes = await target_file.read()
        file_size_kb = round(len(file_bytes) / 1024, 2)
        log("PREPROCESSING", f"File size: {file_size_kb} KB. Converting image to Base64 Data URI...")

        try:
            img_uri = convert_image_bytes_to_base64_uri(file_bytes)
        except ValueError as val_err:
            log("REQ ERROR", f"Preprocessing failed: {str(val_err)}")
            raise HTTPException(status_code=400, detail=str(val_err))

        start_time = time.perf_counter()

        log("INFERENCE START", f"Sending vision request for '{filename}' to vLLM engine on GPU {active_gpu}...")

        payload = {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": img_uri}},
                        {"type": "text", "text": DOTSOCR_PROMPT}
                    ]
                }
            ],
            "max_tokens": eff_max_tokens,
            "temperature": 0.0
        }

        try:
            response = requests.post(VLLM_BASE_URL, json=payload, timeout=180)
            if response.status_code != 200:
                log("REQ ERROR", f"vLLM backend returned status code {response.status_code}: {response.text}")
                raise HTTPException(status_code=response.status_code, detail=f"vLLM server error: {response.text}")
            res_json = response.json()
        except requests.exceptions.RequestException as req_err:
            log("REQ ERROR", f"HTTP connection to vLLM backend failed: {str(req_err)}")
            raise HTTPException(status_code=502, detail=f"Backend request failed: {str(req_err)}")

        content_text = res_json["choices"][0]["message"]["content"]

        parsed_layout = None
        try:
            parsed_layout = json.loads(content_text)
        except json.JSONDecodeError:
            parsed_layout = content_text

        usage = res_json.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        end_time = time.perf_counter()
        duration_seconds = round(end_time - start_time, 4)
        throughput = round(completion_tokens / duration_seconds, 2) if duration_seconds > 0 else 0.0

        gpu_manager.touch()

        log("INFERENCE SUCCESS", f"Finished '{filename}' in {duration_seconds}s | Tokens: {prompt_tokens} in / {completion_tokens} out | Speed: {throughput} tok/s")

        return {
            "status": "success",
            "filename": filename,
            "gpu_assigned": active_gpu,
            "results": parsed_layout,
            "metrics": {
                "time_taken_seconds": duration_seconds,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "generation_speed_tok_per_sec": throughput
            }
        }


@app.post("/free-vram")
def free_vram_manually():
    """Manually trigger VRAM cleanup by stopping the GPU backend."""
    log("MANUAL FREE", "Received manual request to free VRAM...")
    gpu_manager.stop_backend()
    return {"status": "success", "message": "VRAM freed. Backend stopped."}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("FASTAPI_PORT", "8887"))
    log("SERVER START", f"Starting DotsOCR API server on port {port}...")
    uvicorn.run("server_app:app", host="0.0.0.0", port=port, reload=False)
