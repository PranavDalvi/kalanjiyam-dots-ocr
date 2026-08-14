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
from fastapi.responses import JSONResponse
from PIL import Image

app = FastAPI(
    title="DotsOCR Dynamic GPU Image API Service",
    description="Streamlined FastAPI service optimized exclusively for image inputs (PNG, JPG, JPEG, WEBP) with dynamic GPU VRAM allocation and step-by-step console logging.",
    version="4.0.0"
)

# Configuration defaults
IDLE_TIMEOUT_SECONDS = int(os.getenv("IDLE_TIMEOUT_SECONDS", "1800"))  # 30 minutes
# API admission and engine batching must be controlled independently.  Increasing
# vLLM's sequence count can lower OCR decode throughput on a single A6000, while
# allowing several HTTP requests to wait for the engine remains useful.
API_MAX_CONCURRENT_REQUESTS = int(
    os.getenv("API_MAX_CONCURRENT_REQUESTS", os.getenv("MAX_CONCURRENT_REQUESTS", "8"))
)
VLLM_MAX_NUM_SEQS = int(os.getenv("VLLM_MAX_NUM_SEQS", "1"))
VLLM_MAX_NUM_BATCHED_TOKENS = os.getenv("VLLM_MAX_NUM_BATCHED_TOKENS")
GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.90"))
GPU_COUNT = int(os.getenv("GPU_COUNT", "1"))
PINNED_GPU_ID = os.getenv("PINNED_GPU_ID")
MIN_FREE_VRAM_MB = int(os.getenv("MIN_FREE_VRAM_MB", "36000"))
GPU_MEMORY_HEADROOM_MB = int(os.getenv("GPU_MEMORY_HEADROOM_MB", "1024"))
VLLM_PORT = int(os.getenv("VLLM_PORT", "8000"))
VLLM_BASE_URL = f"http://localhost:{VLLM_PORT}/v1/chat/completions"
MODEL_PATH = os.getenv("MODEL_PATH", "rednote-hilab/dots.ocr")
MODEL_NAME = "model"
DEBUG_BBOX = os.getenv("DEBUG_BBOX", "0").lower() in ("1", "true", "yes")

# Asyncio Semaphore to restrict maximum parallel in-flight OCR requests at API level
request_semaphore = asyncio.Semaphore(API_MAX_CONCURRENT_REQUESTS)

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


def select_best_gpu(excluded_gpu_ids: set[int] | None = None, log_selection: bool = True) -> tuple[int, float]:
    """
    Select the GPU index with the most free VRAM and dynamically calculate a safe GPU memory utilization ratio.
    """
    gpus = get_gpu_info()
    if not gpus:
        raise RuntimeError("No NVIDIA GPUs detected on server!")

    excluded_gpu_ids = excluded_gpu_ids or set()
    if PINNED_GPU_ID is not None:
        pinned_gpu_id = int(PINNED_GPU_ID)
        if pinned_gpu_id in excluded_gpu_ids:
            raise RuntimeError(f"Pinned GPU {pinned_gpu_id} was selected more than once.")
        sorted_gpus = [gpu for gpu in gpus if gpu["index"] == pinned_gpu_id]
        if not sorted_gpus:
            raise RuntimeError(f"Configured PINNED_GPU_ID={pinned_gpu_id} is not available.")
    else:
        sorted_gpus = sorted(
            (gpu for gpu in gpus if gpu["index"] not in excluded_gpu_ids),
            key=lambda g: g["free_mb"], reverse=True,
        )
    eligible_gpus = [gpu for gpu in sorted_gpus if gpu["free_mb"] >= MIN_FREE_VRAM_MB]
    if not eligible_gpus:
        gpu_description = f"GPU {PINNED_GPU_ID}" if PINNED_GPU_ID is not None else "any eligible GPU"
        raise RuntimeError(
            f"No {gpu_description} has the required {MIN_FREE_VRAM_MB} MB free VRAM to load DotsOCR."
        )
    best_gpu = eligible_gpus[0]

    total_mb = best_gpu["total_mb"]
    free_mb = best_gpu["free_mb"]

    # Translate currently free VRAM into vLLM's total-GPU fraction while keeping
    # a small allocation/driver reserve. This avoids wasting 10% of already
    # limited free VRAM on otherwise eligible GPUs.
    calculated_util = (free_mb - GPU_MEMORY_HEADROOM_MB) / total_mb
    safe_utilization = round(max(0.25, min(GPU_MEMORY_UTILIZATION, calculated_util)), 2)

    if log_selection:
        log("GPU SELECT", f"Picked GPU {best_gpu['index']} ({free_mb} MB free / {total_mb} MB total). Dynamic GPU utilization set to {safe_utilization} (~{int(total_mb * safe_utilization)} MB).")
    return best_gpu["index"], safe_utilization


class GPUProcessManager:
    """Runs one single-GPU vLLM worker per selected GPU and routes requests round-robin."""

    def __init__(self):
        self.backends = []
        self.next_backend_index = 0
        self.last_active_timestamp = time.time()
        self.lock = threading.Lock()
        self.monitor_thread = threading.Thread(target=self._idle_monitor, daemon=True)
        self.monitor_thread.start()

    @property
    def active_gpus(self):
        return [backend["gpu_idx"] for backend in self._healthy_backends()]

    def touch(self):
        self.last_active_timestamp = time.time()

    @staticmethod
    def _is_backend_running(backend) -> bool:
        if backend["process"].poll() is not None:
            return False
        try:
            response = requests.get(f"http://localhost:{backend['port']}/v1/models", timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    def _healthy_backends(self):
        """Return only live workers; one failed GPU must not take down the others."""
        return [backend for backend in self.backends if self._is_backend_running(backend)]

    def is_running(self) -> bool:
        return bool(self._healthy_backends())

    def is_available(self) -> bool:
        """A healthy loaded worker or enough free VRAM to load one on demand."""
        if self.is_running():
            return True
        try:
            select_best_gpu(log_selection=False)
            return True
        except RuntimeError:
            return False

    def _next_backend(self):
        healthy_backends = self._healthy_backends()
        if not healthy_backends:
            raise RuntimeError("No healthy vLLM workers are available.")
        backend = healthy_backends[self.next_backend_index % len(healthy_backends)]
        self.next_backend_index = (self.next_backend_index + 1) % len(healthy_backends)
        return backend

    @staticmethod
    def _stream_logs(proc, gpu_idx):
        for line in iter(proc.stdout.readline, ""):
            if not line:
                break
            print(f"[vLLM GPU {gpu_idx}] {line.strip()}", flush=True)

    def _stop_backends(self):
        for backend in self.backends:
            proc = backend["process"]
            log("VRAM CLEANUP", f"Stopping vLLM process on GPU {backend['gpu_idx']}...")
            try:
                if os.name != "nt":
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                else:
                    proc.kill()
            except Exception as e:
                log("VRAM WARN", f"Error stopping GPU {backend['gpu_idx']}: {e}")
        self.backends = []
        self.next_backend_index = 0

    def start_backend(self):
        """Start GPU_COUNT independent single-GPU workers and return one for this request."""
        with self.lock:
            # Do not restart healthy models merely because another GPU failed.
            if self.is_running():
                return self._next_backend()

            if self.backends:
                self._stop_backends()
            effective_model_path = ensure_model_downloaded()
            model_parent = os.path.dirname(effective_model_path)
            extra_paths = [effective_model_path, model_parent, "/workspace", "/root/.cache/weights"]
            pythonpath_str = ":".join(extra_paths) + ":" + os.environ.get("PYTHONPATH", "")
            selected_gpu_ids = set()

            for worker_index in range(GPU_COUNT):
                gpu_idx, safe_utilization = select_best_gpu(selected_gpu_ids)
                selected_gpu_ids.add(gpu_idx)
                port = VLLM_PORT + worker_index
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
                env["VLLM_USE_V1"] = "0"
                env["PYTHONPATH"] = pythonpath_str
                cmd = [
                    "vllm", "serve", effective_model_path,
                    "--tensor-parallel-size", "1",
                    "--gpu-memory-utilization", str(safe_utilization),
                    "--max-num-seqs", str(VLLM_MAX_NUM_SEQS),
                    "--trust-remote-code",
                    "--served-model-name", MODEL_NAME,
                    "--port", str(port),
                ]
                if VLLM_MAX_NUM_BATCHED_TOKENS:
                    cmd.extend(["--max-num-batched-tokens", VLLM_MAX_NUM_BATCHED_TOKENS])
                log("BACKEND LAUNCH", f"Launching worker {worker_index + 1}/{GPU_COUNT} on GPU {gpu_idx}, port {port}, memory target {safe_utilization}.")
                proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, preexec_fn=os.setsid if os.name != "nt" else None)
                backend = {"gpu_idx": gpu_idx, "port": port, "process": proc}
                threading.Thread(target=self._stream_logs, args=(proc, gpu_idx), daemon=True).start()

                start_wait = time.time()
                while time.time() - start_wait < 300:
                    if self._is_backend_running(backend):
                        break
                    if proc.poll() is not None:
                        break
                    time.sleep(3)
                if not self._is_backend_running(backend):
                    log("BACKEND WARN", f"GPU {gpu_idx} worker did not start. Keeping existing workers online and continuing with the remaining GPUs.")
                    continue
                self.backends.append(backend)
                log("BACKEND READY", f"vLLM worker is ready on GPU {gpu_idx}.")

            if not self.backends:
                raise RuntimeError("No vLLM worker could start. Check the worker logs above.")
            self.touch()
            return self._next_backend()

    def stop_backend(self):
        with self.lock:
            self._stop_backends()
            log("VRAM FREED", "All vLLM workers stopped; their VRAM has been released.")

    def _idle_monitor(self):
        while True:
            time.sleep(30)
            if self.backends and time.time() - self.last_active_timestamp > IDLE_TIMEOUT_SECONDS:
                log("IDLE TIMEOUT", f"No requests for {int(time.time() - self.last_active_timestamp)}s (limit: {IDLE_TIMEOUT_SECONDS}s). Triggering auto-shutdown...")
                self.stop_backend()


gpu_manager = GPUProcessManager()


# =============================================================================
# FASTAPI ENDPOINTS
# =============================================================================

ALLOWED_BLOCK_TYPES = {
    "paragraph", "heading", "subheading", "table", "figure",
    "caption", "footnote", "running-header", "page-number",
    "column-header", "equation"
}

CATEGORY_MAP = {
    "title": "heading",
    "section-header": "subheading",
    "text": "paragraph",
    "list-item": "paragraph",
    "formula": "equation",
    "table": "table",
    "picture": "figure",
    "caption": "caption",
    "footnote": "footnote",
    "page-header": "running-header",
    "page-footer": "page-number",
    "column-header": "column-header",
    "heading": "heading",
    "subheading": "subheading",
    "paragraph": "paragraph",
    "equation": "equation",
    "figure": "figure",
    "running-header": "running-header",
    "page-number": "page-number",
}


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


def scale_bbox_to_pixels(raw_bbox: list, page_width: float, page_height: float) -> list[float]:
    """
    Converts Dots.OCR bounding box coordinates to image pixel coordinates [x1, y1, x2, y2].

    Dots.OCR natively outputs bounding boxes normalized to the [0, 1000] scale.
    This function converts [0, 1000] coordinates to [0..page_width, 0..page_height].
    If raw_bbox is already in absolute pixel space (e.g. values > 1050 within image bounds),
    or if coordinates touch/slightly overshoot boundaries, it clamps safely within
    [0, page_width] and [0, page_height]. Malformed bboxes gracefully return [0.0, 0.0, 0.0, 0.0].
    """
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        return [0.0, 0.0, 0.0, 0.0]

    try:
        c1, c2, c3, c4 = float(raw_bbox[0]), float(raw_bbox[1]), float(raw_bbox[2]), float(raw_bbox[3])
    except (ValueError, TypeError):
        return [0.0, 0.0, 0.0, 0.0]

    pw = max(1.0, float(page_width))
    ph = max(1.0, float(page_height))

    # Detect if bbox is already in pixel coordinates (only when coordinates exceed normalized
    # threshold and fit within actual pixel dimensions).
    is_already_pixel = (max(c1, c2, c3, c4) > 1050.0 and max(c1, c2, c3, c4) <= max(pw, ph))

    if is_already_pixel:
        x1, y1, x2, y2 = c1, c2, c3, c4
    else:
        # Standard Dots.OCR normalized [0, 1000] -> pixel coordinates
        x1 = (c1 / 1000.0) * pw
        y1 = (c2 / 1000.0) * ph
        x2 = (c3 / 1000.0) * pw
        y2 = (c4 / 1000.0) * ph

    # Ensure min <= max
    min_x, max_x = min(x1, x2), max(x1, x2)
    min_y, max_y = min(y1, y2), max(y1, y2)

    # Clamp coordinates to image boundaries
    clamped_x1 = max(0.0, min(pw, min_x))
    clamped_y1 = max(0.0, min(ph, min_y))
    clamped_x2 = max(0.0, min(pw, max_x))
    clamped_y2 = max(0.0, min(ph, max_y))

    return [round(clamped_x1, 2), round(clamped_y1, 2), round(clamped_x2, 2), round(clamped_y2, 2)]


def _generate_word_spans(content: str, px_bbox: list[float], block_confidence: float) -> list[dict]:
    """Generates word span objects with bounding boxes and confidence scores."""
    if not content:
        return []

    words_output = []
    x1, y1, x2, y2 = px_bbox
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)

    lines = content.splitlines()
    num_lines = max(1, len(lines))
    lh = bh / num_lines

    for line_idx, line in enumerate(lines):
        line_str = line.strip()
        if not line_str:
            continue
        line_top = y1 + line_idx * lh
        line_bottom = y1 + (line_idx + 1) * lh

        words_in_line = line_str.split()
        if not words_in_line:
            continue

        line_char_len = max(1, len(line_str))
        char_cursor = 0
        for w_text in words_in_line:
            w_idx = line_str.find(w_text, char_cursor)
            if w_idx == -1:
                w_idx = char_cursor
            char_cursor = w_idx + len(w_text)

            c_start_ratio = w_idx / line_char_len
            c_end_ratio = char_cursor / line_char_len

            wx1 = round(x1 + c_start_ratio * bw, 2)
            wx2 = round(x1 + c_end_ratio * bw, 2)
            wy1 = round(line_top, 2)
            wy2 = round(line_bottom, 2)

            words_output.append({
                "text": w_text,
                "bbox": [wx1, wy1, wx2, wy2],
                "confidence": round(float(block_confidence), 3)
            })

    return words_output


def format_kalanjiyam_v2_response(
    parsed_layout,
    image_bytes: bytes,
    filename: str,
    active_gpu: int,
    language: str,
    duration_seconds: float,
    prompt_tokens: int,
    completion_tokens: int,
    throughput: float,
    engine: Optional[str] = None,
) -> dict:
    """Formats DotsOCR outputs into the Kalanjiyam OCR Service Contract (v2.1) JSON response."""
    page_width, page_height = 1000, 1000
    try:
        img = Image.open(io.BytesIO(image_bytes))
        page_width, page_height = img.size
    except Exception:
        pass

    blocks = []
    tsv_lines = []
    text_parts = []
    all_word_confidences = []

    layout_items = parsed_layout
    if isinstance(parsed_layout, dict):
        if "blocks" in parsed_layout and isinstance(parsed_layout["blocks"], list):
            layout_items = parsed_layout["blocks"]
        elif "results" in parsed_layout and isinstance(parsed_layout["results"], list):
            layout_items = parsed_layout["results"]
        elif "layout" in parsed_layout and isinstance(parsed_layout["layout"], list):
            layout_items = parsed_layout["layout"]

    if isinstance(layout_items, list):
        for i, item in enumerate(layout_items, 1):
            if not isinstance(item, dict):
                continue
            raw_cat = str(item.get("category") or item.get("type") or "text").lower().strip()
            block_type = CATEGORY_MAP.get(raw_cat)
            if not block_type:
                block_type = raw_cat if raw_cat in ALLOWED_BLOCK_TYPES else "paragraph"

            raw_bbox = item.get("bbox") or [0, 0, 0, 0]
            px_bbox = scale_bbox_to_pixels(raw_bbox, page_width, page_height)

            content = str(item.get("text") or item.get("content") or "").strip()
            block_conf = float(item.get("confidence", 0.95))

            if content:
                text_parts.append(content)
                tsv_lines.append(f"{px_bbox[0]}\t{px_bbox[1]}\t{px_bbox[2]}\t{px_bbox[3]}\t{content}")

            words_list = []
            if isinstance(item.get("words"), list) and len(item["words"]) > 0:
                for w in item["words"]:
                    if isinstance(w, dict) and "text" in w:
                        w_bbox = w.get("bbox") or [0, 0, 0, 0]
                        px_w_bbox = scale_bbox_to_pixels(w_bbox, page_width, page_height)

                        w_conf = float(w.get("confidence", block_conf))
                        w_obj = {
                            "text": str(w["text"]),
                            "bbox": px_w_bbox,
                            "confidence": round(w_conf, 3)
                        }
                        words_list.append(w_obj)
                        all_word_confidences.append(w_conf)
            else:
                words_list = _generate_word_spans(content, px_bbox, block_conf)
                for w in words_list:
                    all_word_confidences.append(w["confidence"])

            block_obj = {
                "id": f"b{i}",
                "type": block_type,
                "bbox": px_bbox,
                "reading_order": i,
                "content": content,
                "confidence": round(block_conf, 3),
                "words": words_list
            }
            blocks.append(block_obj)

    elif isinstance(parsed_layout, str) and parsed_layout.strip():
        content = parsed_layout.strip()
        text_parts.append(content)
        px_bbox = [0.0, 0.0, float(page_width), float(page_height)]
        words_list = _generate_word_spans(content, px_bbox, 0.95)
        for w in words_list:
            all_word_confidences.append(w["confidence"])
        blocks.append({
            "id": "b1",
            "type": "paragraph",
            "bbox": px_bbox,
            "reading_order": 1,
            "content": content,
            "confidence": 0.95,
            "words": words_list
        })

    plain_text = "\n\n".join(text_parts)
    tsv_text = "\n".join(tsv_lines)

    if all_word_confidences:
        page_confidence = round(sum(all_word_confidences) / len(all_word_confidences), 3)
    elif blocks:
        page_confidence = round(sum(b["confidence"] for b in blocks) / len(blocks), 3)
    else:
        page_confidence = 0.95

    engine_latency_ms = round(duration_seconds * 1000.0, 2)
    selected_engine = (engine or "dots_ocr").strip()

    result_payload = {
        # Kalanjiyam OCR Service Contract (v2.1) required fields
        "contract_version": "2.1",
        "engine": selected_engine,
        "model": {
            "name": "dots-ocr",
            "version": "4.0.0"
        },
        "page_confidence": page_confidence,
        "engine_latency_ms": engine_latency_ms,
        "page_width": int(page_width),
        "page_height": int(page_height),
        "blocks": blocks,

        # Additional standalone / backwards compatibility fields
        "source_type": "scan",
        "coordinate_space": "pixel",
        "text": plain_text,
        "bounding_boxes": tsv_text,
        "status": "success",
        "filename": filename,
        "gpu_assigned": active_gpu,
        "results": parsed_layout,
        "metrics": {
            "time_taken_seconds": duration_seconds,
            "engine_latency_ms": engine_latency_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "generation_speed_tok_per_sec": throughput
        }
    }

    if DEBUG_BBOX:
        log("DEBUG_BBOX", f"[{filename}] Dimensions: {page_width}x{page_height} | Parsed Blocks: {len(blocks)}")
        for b in blocks:
            log("DEBUG_BBOX", f"  [{b['id']}] type={b['type']} bbox={b['bbox']}")
        result_payload["debug_bbox"] = {
            "image_width": int(page_width),
            "image_height": int(page_height),
            "raw_model_response": parsed_layout,
            "parsed_bboxes": [b["bbox"] for b in blocks],
            "final_bboxes": [b["bbox"] for b in blocks]
        }

    return result_payload


@app.get("/v1/engines")
def get_engines():
    """Returns list of active engines for Kalanjiyam service discovery."""
    return {"engines": ["dots-ocr"]}


@app.get("/gpu-status")
def gpu_status():
    """Query current GPU VRAM usage, concurrency limits, and backend status."""
    gpus = get_gpu_info()
    return {
        "gpu_present": len(gpus) > 0,
        "gpu_count": len(gpus),
        "active_gpus": gpu_manager.active_gpus,
        "gpu_count_configured": GPU_COUNT,
        "backend_running": gpu_manager.is_running(),
        "api_max_concurrent_limit": API_MAX_CONCURRENT_REQUESTS,
        "vllm_max_num_seqs": VLLM_MAX_NUM_SEQS,
        "idle_seconds": round(time.time() - gpu_manager.last_active_timestamp, 1),
        "idle_timeout_seconds": IDLE_TIMEOUT_SECONDS,
        "gpus": gpus
    }


@app.get("/health")
def health_check():
    """Health check endpoint."""
    available = gpu_manager.is_available()
    payload = {
        "status": "healthy" if available else "unavailable",
        "backend_running": gpu_manager.is_running(),
        "active_gpus": gpu_manager.active_gpus,
        "gpu_count_configured": GPU_COUNT,
        "api_max_concurrent_limit": API_MAX_CONCURRENT_REQUESTS,
        "vllm_max_num_seqs": VLLM_MAX_NUM_SEQS,
        "min_free_vram_mb": MIN_FREE_VRAM_MB,
    }
    return JSONResponse(status_code=200 if available else 503, content=payload)


@app.post("/ocr")
@app.post("/v1/ocr")
async def run_ocr(
    file: Optional[UploadFile] = File(None, description="Image file (PNG, JPG, JPEG, WEBP)"),
    image: Optional[UploadFile] = File(None, description="Alternative field name for image file"),
    engine: Optional[str] = Form(None, description="OCR engine name"),
    language: Optional[str] = Form(None, description="Language code (e.g. sa, en, hi)"),
    max_tokens: Optional[int] = Form(None, description="Max generation tokens")
):
    """
    Fast image-only OCR endpoint with dynamic GPU VRAM allocation & step-by-step console logging.
    Supports both standalone API and Kalanjiyam OCR Service Contract (v2).
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
    log("REQ RECEIVED", f"Processing image: '{filename}' (engine: {engine or 'dots-ocr'}, language: {language or 'sa'}, max_tokens: {eff_max_tokens})")

    async with request_semaphore:
        gpu_manager.touch()
        try:
            backend = gpu_manager.start_backend()
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

        active_gpu = backend["gpu_idx"]
        backend_url = f"http://localhost:{backend['port']}/v1/chat/completions"
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
            # requests is synchronous; running it in a worker thread keeps the
            # FastAPI event loop free to accept and queue other OCR requests.
            response = await asyncio.to_thread(
                requests.post, backend_url, json=payload, timeout=180
            )
            if response.status_code != 200:
                log("REQ ERROR", f"vLLM backend returned status code {response.status_code}: {response.text}")
                raise HTTPException(status_code=response.status_code, detail=f"vLLM server error: {response.text}")
            res_json = response.json()
        except requests.exceptions.RequestException as req_err:
            log("REQ ERROR", f"HTTP connection to vLLM backend failed: {str(req_err)}")
            raise HTTPException(status_code=502, detail=f"Backend request failed: {str(req_err)}")

        content_text = res_json["choices"][0]["message"]["content"]

        if DEBUG_BBOX:
            log("DEBUG_BBOX", f"Raw Model Output for '{filename}':\n{content_text}")

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

        return format_kalanjiyam_v2_response(
            parsed_layout=parsed_layout,
            image_bytes=file_bytes,
            filename=filename,
            active_gpu=active_gpu,
            language=language or "sa",
            duration_seconds=duration_seconds,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            throughput=throughput,
            engine=engine,
        )


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
