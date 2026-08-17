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
import re
from typing import Optional, List, Dict, Any, Union, Set
import requests
import io
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image

app = FastAPI(
    title="Kalanjiyam OCR & Archival Metadata Extraction API Service",
    description="Streamlined FastAPI service for OCR layout recognition and archival metadata extraction with dynamic GPU VRAM allocation.",
    version="4.1.0"
)

def _env_int(key: str, default: int, fallback_key: Optional[str] = None) -> int:
    val = os.getenv(key, "").strip()
    if not val and fallback_key:
        val = os.getenv(fallback_key, "").strip()
    if not val:
        return default
    try:
        return int(val)
    except ValueError:
        return default

def _env_float(key: str, default: float) -> float:
    val = os.getenv(key, "").strip()
    if not val:
        return default
    try:
        return float(val)
    except ValueError:
        return default

# Configuration defaults
IDLE_TIMEOUT_SECONDS = _env_int("IDLE_TIMEOUT_SECONDS", 1800)  # 30 minutes
# API admission and engine batching must be controlled independently.  Increasing
# vLLM's sequence count can lower OCR decode throughput on a single A6000, while
# allowing several HTTP requests to wait for the engine remains useful.
API_MAX_CONCURRENT_REQUESTS = _env_int(
    "API_MAX_CONCURRENT_REQUESTS", 8, fallback_key="MAX_CONCURRENT_REQUESTS"
)
VLLM_MAX_NUM_SEQS = _env_int("VLLM_MAX_NUM_SEQS", 1)
VLLM_MAX_NUM_BATCHED_TOKENS = os.getenv("VLLM_MAX_NUM_BATCHED_TOKENS")
GPU_MEMORY_UTILIZATION = _env_float("GPU_MEMORY_UTILIZATION", 0.90)
GPU_COUNT = _env_int("GPU_COUNT", 1)
_pinned_raw = os.getenv("PINNED_GPU_ID", "").strip()
PINNED_GPU_ID = int(_pinned_raw) if _pinned_raw.isdigit() else None
MIN_FREE_VRAM_MB = _env_int("MIN_FREE_VRAM_MB", 36000)
GPU_MEMORY_HEADROOM_MB = _env_int("GPU_MEMORY_HEADROOM_MB", 1024)
VLLM_PORT = _env_int("VLLM_PORT", 8000)
VLLM_BASE_URL = f"http://localhost:{VLLM_PORT}/v1/chat/completions"
MODEL_PATH = os.getenv("MODEL_PATH", "")
MODEL_NAME = "model"
DEBUG_BBOX = os.getenv("DEBUG_BBOX", "0").lower() in ("1", "true", "yes")
DEFAULT_ENGINE = os.getenv("DEFAULT_ENGINE", os.getenv("ENGINE", "dots-ocr")).strip().lower()
DEFAULT_METADATA_ENGINE = os.getenv("DEFAULT_METADATA_ENGINE", "gemma-4").strip().lower()

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

GEMMA4_OCR_PROMPT = """please output the layout information from the image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. bbox format: [x1, y1, x2, y2] normalized to 0-1000 scale.

2. layout categories: the possible categories are ['caption', 'footnote', 'formula', 'list-item', 'page-footer', 'page-header', 'picture', 'section-header', 'table', 'text', 'title'].

3. text extraction & formatting rules:
    - picture: for the 'picture' category, the text field should be omitted.
    - formula: format its text as latex.
    - table: format its text as html.
    - all others (text, title, etc.): format their text as markdown.

4. constraints:
    - the output text must be the original text from the image, with no translation.
    - all layout elements must be sorted according to human reading order.

5. final output: the entire output must be a single json object with a "blocks" array or direct array of layout items."""

ENGINE_CONFIGS = {
    "dots-ocr": {
        "engine_id": "dots-ocr",
        "aliases": ["dots-ocr", "dots_ocr", "dotsocr", "dots", "rednote-hilab/dots.ocr"],
        "default_model_path": "rednote-hilab/dots.ocr",
        "local_cache_dir": "/root/.cache/weights/DotsOCR",
        "model_name": "dots-ocr",
        "model_version": "4.0.0",
        "requires_native_registration": True,
        "prompt": DOTSOCR_PROMPT,
        "min_free_vram_mb": MIN_FREE_VRAM_MB,
    },
    "gemma-4": {
        "engine_id": "gemma-4",
        "aliases": [
            "gemma-4",
            "gemma-4-26b",
            "gemma-4-26b-a4b-it",
            "gemma-4-26b-it",
            "gemma4",
            "google/gemma-4-26b-a4b-it",
            "google/gemma-4-26b-a4b",
            "google/gemma-4-26b-it",
            "gemma",
            "gemma-4-26b-a4b",
            "kalanjiyam-archival",
            "kalanjiyam_archival",
            "archival",
            "metadata",
            "kalanjiyam-metadata",
            "gemma-3-27b-it",
            "gemma-3",
        ],
        "default_model_path": os.getenv("GEMMA4_MODEL_PATH", "google/gemma-4-26B-A4B-it"),
        "local_cache_dir": "/root/.cache/weights/gemma-4-26B-A4B-it",
        "model_name": "gemma-4-26b-a4b-it",
        "model_version": "1.0.0",
        "requires_native_registration": False,
        "prompt": GEMMA4_OCR_PROMPT,
        "min_free_vram_mb": _env_int("GEMMA4_MIN_FREE_VRAM_MB", 30000),
    },
    "kalanjiyam-archival": {
        "engine_id": "kalanjiyam-archival",
        "aliases": [
            "kalanjiyam-archival-legacy",
        ],
        "default_model_path": os.getenv("METADATA_MODEL_PATH", os.getenv("GEMMA4_MODEL_PATH", "google/gemma-4-26B-A4B-it")),
        "local_cache_dir": "/root/.cache/weights/gemma-4-26B-A4B-it",
        "model_name": os.getenv("METADATA_MODEL_NAME", "gemma-4-26b-a4b-it"),
        "model_version": "1.0.0",
        "requires_native_registration": False,
        "prompt": "",
        "min_free_vram_mb": _env_int("METADATA_MIN_FREE_VRAM_MB", _env_int("GEMMA4_MIN_FREE_VRAM_MB", 30000)),
    },
}


def resolve_engine(engine_name: Optional[str]) -> str:
    """Resolve user-supplied engine name or alias to canonical engine ID."""
    if not engine_name or not str(engine_name).strip():
        # Fall back to default configured engine
        cleaned_default = DEFAULT_ENGINE.strip().lower()
        if cleaned_default in ENGINE_CONFIGS:
            return cleaned_default
        for eid, conf in ENGINE_CONFIGS.items():
            if cleaned_default in [a.lower() for a in conf.get("aliases", [])]:
                return eid
        return "dots-ocr"

    cleaned = str(engine_name).strip().lower()
    for engine_id, conf in ENGINE_CONFIGS.items():
        if cleaned == engine_id.lower() or cleaned in [a.lower() for a in conf.get("aliases", [])]:
            return engine_id

    # Keyword heuristics
    if "archival" in cleaned or "metadata" in cleaned:
        return "gemma-4"
    if "gemma" in cleaned:
        return "gemma-4"
    if "dots" in cleaned:
        return "dots-ocr"

    # Default fallback
    return "dots-ocr"



def log(step: str, message: str):
    """Helper for formatted step-by-step console logging."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{step}] {message}", flush=True)


# =============================================================================
# GPU & PROCESS MANAGEMENT UTILITIES
# =============================================================================

def ensure_model_downloaded(engine_id: str = "dots-ocr") -> str:
    """
    Ensures model weights are downloaded to a local directory.
    If engine is dots-ocr, config.json auto_map and vLLM native registration are ensured.
    """
    canonical_engine = resolve_engine(engine_id)
    config = ENGINE_CONFIGS.get(canonical_engine, ENGINE_CONFIGS["dots-ocr"])

    target_dir = config["local_cache_dir"]
    if canonical_engine == "dots-ocr":
        model_to_use = os.getenv("DOTSOCR_MODEL_PATH", MODEL_PATH if MODEL_PATH and "gemma" not in MODEL_PATH.lower() else config["default_model_path"])
    elif canonical_engine == "gemma-4":
        model_to_use = os.getenv("GEMMA4_MODEL_PATH", MODEL_PATH if MODEL_PATH and "gemma" in MODEL_PATH.lower() else config["default_model_path"])
    elif canonical_engine == "kalanjiyam-archival":
        model_to_use = os.getenv("METADATA_MODEL_PATH", os.getenv("GEMMA4_MODEL_PATH", config["default_model_path"]))
    else:
        model_to_use = config["default_model_path"]

    # If already a local directory with model files
    if os.path.exists(model_to_use) and os.path.isdir(model_to_use) and os.path.exists(os.path.join(model_to_use, "config.json")):
        if config["requires_native_registration"]:
            _ensure_config_auto_map(model_to_use)
            _install_vllm_native_registration(model_to_use)
        if "gemma" in canonical_engine or "archival" in canonical_engine:
            _ensure_gemma4_config(model_to_use)
        return model_to_use

    # If already downloaded in target cache folder
    if os.path.exists(target_dir) and os.path.exists(os.path.join(target_dir, "config.json")):
        if config["requires_native_registration"]:
            _ensure_config_auto_map(target_dir)
            _install_vllm_native_registration(target_dir)
        if "gemma" in canonical_engine or "archival" in canonical_engine:
            _ensure_gemma4_config(target_dir)
        return target_dir

    log("MODEL DOWNLOAD", f"Pre-downloading model '{model_to_use}' for engine '{canonical_engine}' to local folder '{target_dir}'...")
    try:
        from huggingface_hub import snapshot_download
        os.makedirs(target_dir, exist_ok=True)
        snapshot_download(
            repo_id=model_to_use,
            local_dir=target_dir
        )
        log("MODEL DOWNLOAD", "Model weights downloaded successfully to local folder!")
        if config["requires_native_registration"]:
            _ensure_config_auto_map(target_dir)
            _install_vllm_native_registration(target_dir)
        if "gemma" in canonical_engine or "archival" in canonical_engine:
            _ensure_gemma4_config(target_dir)
        return target_dir
    except Exception as e:
        log("MODEL WARN", f"Snapshot download failed ({e}). Using raw path '{model_to_use}'.")
        return model_to_use


def _ensure_gemma4_config(model_dir: str):
    """
    Ensures config.json has allow_global_per_layer_attribute_access = True
    on top-level config, text_config, and vision_config for Gemma 4 models
    without corrupting typed dicts like id2label/label2id.
    """
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        return
    try:
        with open(config_path, "r") as f:
            cfg = json.load(f)

        def _clean_dict(d: dict):
            for map_key in ("id2label", "label2id"):
                if map_key in d and isinstance(d[map_key], dict):
                    d[map_key].pop("allow_global_per_layer_attribute_access", None)

        cfg["allow_global_per_layer_attribute_access"] = True
        _clean_dict(cfg)

        for sub_key in ("text_config", "vision_config", "language_config"):
            if sub_key in cfg and isinstance(cfg[sub_key], dict):
                cfg[sub_key]["allow_global_per_layer_attribute_access"] = True
                _clean_dict(cfg[sub_key])

        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)
        log("CONFIG FIX", f"Set allow_global_per_layer_attribute_access=True safely in '{config_path}' for Gemma-4.")
    except Exception as e:
        log("CONFIG WARN", f"Failed to patch Gemma-4 config.json: {e}")


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


def select_best_gpu(min_free_vram_mb: int = MIN_FREE_VRAM_MB, excluded_gpu_ids: set[int] | None = None, log_selection: bool = True) -> tuple[int, float]:
    """
    Select the GPU index with the most free VRAM and dynamically calculate a safe GPU memory utilization ratio.
    """
    gpus = get_gpu_info()
    if not gpus:
        raise RuntimeError("No NVIDIA GPUs detected on server!")

    excluded_gpu_ids = excluded_gpu_ids or set()
    if PINNED_GPU_ID is not None:
        pinned_gpu_id = PINNED_GPU_ID
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
    eligible_gpus = [gpu for gpu in sorted_gpus if gpu["free_mb"] >= min_free_vram_mb]
    if not eligible_gpus:
        gpu_description = f"GPU {PINNED_GPU_ID}" if PINNED_GPU_ID is not None else "any eligible GPU"
        raise RuntimeError(
            f"No {gpu_description} has the required {min_free_vram_mb} MB free VRAM to load engine."
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
        self.current_engine = None
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

    def is_available(self, engine_id: str = "dots-ocr") -> bool:
        """A healthy loaded worker for the given engine or enough free VRAM to load one on demand."""
        canonical_engine = resolve_engine(engine_id)
        if self.is_running() and self.current_engine == canonical_engine:
            return True
        config = ENGINE_CONFIGS.get(canonical_engine, ENGINE_CONFIGS["dots-ocr"])
        min_vram = config.get("min_free_vram_mb", MIN_FREE_VRAM_MB)
        try:
            select_best_gpu(min_free_vram_mb=min_vram, log_selection=False)
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
    def _stream_logs(proc, gpu_idx, engine_id):
        for line in iter(proc.stdout.readline, ""):
            if not line:
                break
            print(f"[vLLM GPU {gpu_idx} ({engine_id})] {line.strip()}", flush=True)

    def _stop_backends(self):
        for backend in self.backends:
            proc = backend["process"]
            engine_tag = backend.get("engine", "unknown")
            log("VRAM CLEANUP", f"Stopping vLLM process on GPU {backend['gpu_idx']} (engine: {engine_tag})...")
            try:
                if os.name != "nt":
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                else:
                    proc.kill()
            except Exception as e:
                log("VRAM WARN", f"Error stopping GPU {backend['gpu_idx']}: {e}")
        self.backends = []
        self.current_engine = None
        self.next_backend_index = 0

    def start_backend(self, engine_id: str = "dots-ocr"):
        """Start GPU_COUNT independent single-GPU workers for engine_id and return one for this request."""
        canonical_engine = resolve_engine(engine_id)
        config = ENGINE_CONFIGS.get(canonical_engine, ENGINE_CONFIGS["dots-ocr"])

        with self.lock:
            # If healthy backend already running with the requested engine, return worker
            if self.is_running() and self.current_engine == canonical_engine:
                return self._next_backend()

            # If switching engines or restarting dead workers, clean up first
            if self.backends:
                if self.current_engine and self.current_engine != canonical_engine:
                    log("ENGINE SWITCH", f"Switching active engine from '{self.current_engine}' to '{canonical_engine}'. Stopping current workers to release VRAM...")
                self._stop_backends()

            effective_model_path = ensure_model_downloaded(canonical_engine)
            model_parent = os.path.dirname(effective_model_path)
            extra_paths = [effective_model_path, model_parent, "/workspace", "/root/.cache/weights"]
            pythonpath_str = ":".join(extra_paths) + ":" + os.environ.get("PYTHONPATH", "")
            selected_gpu_ids = set()
            min_vram = config.get("min_free_vram_mb", MIN_FREE_VRAM_MB)

            tp_size = _env_int("TENSOR_PARALLEL_SIZE", _env_int(f"{canonical_engine.upper().replace('-', '_')}_TP_SIZE", 1))

            for worker_index in range(GPU_COUNT):
                worker_gpu_ids = []
                safe_utilization = GPU_MEMORY_UTILIZATION
                for _ in range(tp_size):
                    gpu_idx, safe_utilization = select_best_gpu(
                        min_free_vram_mb=min_vram // tp_size if tp_size > 1 else min_vram,
                        excluded_gpu_ids=selected_gpu_ids
                    )
                    selected_gpu_ids.add(gpu_idx)
                    worker_gpu_ids.append(gpu_idx)

                gpu_str = ",".join(str(g) for g in worker_gpu_ids)
                primary_gpu = worker_gpu_ids[0]
                port = VLLM_PORT + worker_index
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = gpu_str
                env["VLLM_USE_V1"] = "0"
                env["PYTHONPATH"] = pythonpath_str
                env["PYTORCH_CUDA_ALLOC_CONF"] = os.getenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
                cmd = [
                    "vllm", "serve", effective_model_path,
                    "--tensor-parallel-size", str(tp_size),
                    "--gpu-memory-utilization", str(safe_utilization),
                    "--max-num-seqs", str(VLLM_MAX_NUM_SEQS),
                    "--trust-remote-code",
                    "--served-model-name", MODEL_NAME,
                    "--port", str(port),
                ]
                max_model_len = os.getenv("VLLM_MAX_MODEL_LEN", os.getenv(f"{canonical_engine.upper().replace('-', '_')}_MAX_MODEL_LEN", ""))
                if max_model_len:
                    cmd.extend(["--max-model-len", str(max_model_len)])
                elif canonical_engine == "gemma-4":
                    cmd.extend(["--max-model-len", "4096"])

                vllm_dtype = os.getenv("VLLM_DTYPE", "auto")
                if vllm_dtype and vllm_dtype != "auto":
                    cmd.extend(["--dtype", vllm_dtype])

                vllm_quant = os.getenv("VLLM_QUANTIZATION", os.getenv("QUANTIZATION", ""))
                if vllm_quant:
                    cmd.extend(["--quantization", vllm_quant])

                vllm_kv_cache_dtype = os.getenv("VLLM_KV_CACHE_DTYPE", "")
                if vllm_kv_cache_dtype:
                    cmd.extend(["--kv-cache-dtype", vllm_kv_cache_dtype])

                if os.getenv("VLLM_ENFORCE_EAGER", "1" if canonical_engine == "gemma-4" and tp_size == 1 else "0").lower() in ("1", "true", "yes"):
                    cmd.append("--enforce-eager")

                if VLLM_MAX_NUM_BATCHED_TOKENS:
                    cmd.extend(["--max-num-batched-tokens", VLLM_MAX_NUM_BATCHED_TOKENS])
                log("BACKEND LAUNCH", f"Launching {canonical_engine} worker {worker_index + 1}/{GPU_COUNT} on GPU(s) [{gpu_str}] (TP={tp_size}), port {port}, memory target {safe_utilization}.")
                proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, preexec_fn=os.setsid if os.name != "nt" else None)
                backend = {"gpu_idx": primary_gpu, "gpu_ids": worker_gpu_ids, "port": port, "process": proc, "engine": canonical_engine}
                threading.Thread(target=self._stream_logs, args=(proc, primary_gpu, canonical_engine), daemon=True).start()

                start_wait = time.time()
                while time.time() - start_wait < 300:
                    if self._is_backend_running(backend):
                        break
                    if proc.poll() is not None:
                        break
                    time.sleep(3)
                if not self._is_backend_running(backend):
                    log("BACKEND WARN", f"GPU {gpu_idx} worker ({canonical_engine}) did not start. Keeping existing workers online and continuing with the remaining GPUs.")
                    continue
                self.backends.append(backend)
                log("BACKEND READY", f"vLLM worker ({canonical_engine}) is ready on GPU {gpu_idx}.")

            if not self.backends:
                raise RuntimeError(f"No vLLM worker for '{canonical_engine}' could start. Check the worker logs above.")
            self.current_engine = canonical_engine
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


# =============================================================================
# METADATA EXTRACTION & METRICS MODELS (Contract v1.0)
# =============================================================================

class BlockInput(BaseModel):
    id: str = Field(..., description="Unique block ID within the page (e.g. 'b1')")
    type: str = Field(..., description="Layout block type (e.g. 'heading', 'paragraph', 'table')")
    reading_order: Optional[Union[int, float]] = Field(None, description="Reading order index")
    text: str = Field("", description="Text content of the block")


class PageInput(BaseModel):
    page_slug: str = Field(..., description="Page slug identifier (e.g. '61')")
    ocr_confidence: Optional[float] = Field(None, description="Nullable OCR confidence score")
    blocks: List[BlockInput] = Field(default_factory=list, description="Typed layout blocks")


class WindowInput(BaseModel):
    index: int = Field(..., description="Window index (e.g. 3)")
    total: int = Field(..., description="Total windows for this document (e.g. 24)")
    page_slugs: List[str] = Field(default_factory=list, description="List of page slugs in this window")


class MetadataRequest(BaseModel):
    contract_version: str = Field("1.0", description="Contract version (must be '1.0')")
    unit_id: str = Field(..., description="Unique archival document unit ID")
    window: WindowInput = Field(..., description="Window info")
    taxonomy_version: str = Field(..., description="Taxonomy schema revision (e.g. 'client-2026-08')")
    tags: List[str] = Field(..., description="Authoritative list of requested metadata tags")
    language_hint: Optional[List[str]] = Field(None, description="Optional language hints")
    pages: List[PageInput] = Field(..., description="Pages and typed blocks in this window")
    engine: Optional[str] = Field(None, description="Optional extraction engine override")
    max_tokens: Optional[int] = Field(None, description="Max generation tokens (default >= 4500)")


class EvidenceSpan(BaseModel):
    page_slug: str = Field(..., description="Cited page slug")
    block_id: Optional[str] = Field(None, description="Cited block ID (for record source)")
    quote: Optional[str] = Field(None, description="Verbatim quote from source block")


class EntityValue(BaseModel):
    label: str = Field(..., description="Entity name as in record")
    variants: Optional[List[str]] = Field(None, description="Variant names seen in window")
    dates: Optional[str] = Field(None, description="Life or existence dates if stated")
    auth_id: Optional[str] = Field(None, description="Authority file ID if explicitly stated")
    source: str = Field("record", description="Source kind: record, derived, enrichment")
    evidence: Optional[List[EvidenceSpan]] = Field(None, description="Evidence spans")
    note: Optional[str] = Field(None, description="Optional notes")


class ModelInfo(BaseModel):
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")


class TokenUsage(BaseModel):
    prompt_tokens: int = Field(..., description="Prompt tokens consumed")
    completion_tokens: int = Field(..., description="Completion tokens generated")
    total_tokens: int = Field(..., description="Total tokens consumed")


class MetadataResponse(BaseModel):
    contract_version: str = Field("1.0", description="Specification version '1.0'")
    status: str = Field("success", description="Status string: 'success'")
    engine: str = Field(..., description="Extraction engine identifier")
    model: ModelInfo = Field(..., description="Model name and version")
    taxonomy_version: str = Field(..., description="Echoed taxonomy schema version")
    unit_id: str = Field(..., description="Echoed document unit ID")
    window_index: int = Field(..., description="Echoed window index")
    chars_in: int = Field(..., description="Total characters consumed from input window")
    engine_latency_ms: float = Field(..., description="Model processing latency in milliseconds")
    usage: TokenUsage = Field(..., description="Token usage breakdown")
    fields_attempted: int = Field(..., description="Count of tags requested")
    fields_returned: int = Field(..., description="Count of fields extracted")
    fields_declined: int = Field(..., description="Count of tags declined due to lack of evidence")
    fields: Dict[str, Any] = Field(..., description="Extracted metadata fields")


# =============================================================================
# METADATA EXTRACTION UTILITIES
# =============================================================================

def normalize_whitespace(text: str) -> str:
    """Normalize whitespace by collapsing consecutive whitespace chars and stripping."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def verify_evidence_spans(fields: dict, pages: list) -> dict:
    """
    Verifies all evidence spans in extracted fields against the request pages and blocks.
    For every 'record' value, checks that quote appears verbatim (whitespace-normalised)
    in the block text it sent.
    """
    block_lookup = {}
    page_lookup = {}
    for p in pages:
        p_slug = str(p.get("page_slug", "") if isinstance(p, dict) else getattr(p, "page_slug", ""))
        p_blocks = p.get("blocks", []) if isinstance(p, dict) else getattr(p, "blocks", [])
        page_texts = []
        for b in p_blocks:
            b_id = str(b.get("id", "") if isinstance(b, dict) else getattr(b, "id", ""))
            b_text = str(b.get("text", "") if isinstance(b, dict) else getattr(b, "text", ""))
            block_lookup[(p_slug, b_id)] = normalize_whitespace(b_text)
            page_texts.append(b_text)
        page_lookup[p_slug] = normalize_whitespace(" ".join(page_texts))

    span_details = []
    total_record_values = 0
    verified_spans = 0

    for tag, field_obj in fields.items():
        if not isinstance(field_obj, dict):
            continue
        val = field_obj.get("value")
        if val is None:
            continue

        # Single-valued tag
        if not isinstance(val, list):
            source = field_obj.get("source", "record")
            if source == "record":
                total_record_values += 1
                evidence_list = field_obj.get("evidence") or []
                if not evidence_list:
                    span_details.append({
                        "tag": tag,
                        "source": source,
                        "verified": False,
                        "reason": "Missing evidence array for record source"
                    })
                else:
                    field_verified = True
                    for ev in evidence_list:
                        if not isinstance(ev, dict):
                            continue
                        p_slug = str(ev.get("page_slug", ""))
                        b_id = str(ev.get("block_id", ""))
                        quote = normalize_whitespace(str(ev.get("quote", "")))
                        target_text = block_lookup.get((p_slug, b_id)) or page_lookup.get(p_slug, "")
                        is_match = bool(quote and quote in target_text)
                        if not is_match:
                            field_verified = False
                        span_details.append({
                            "tag": tag,
                            "page_slug": p_slug,
                            "block_id": b_id,
                            "quote": quote,
                            "verified": is_match
                        })
                    if field_verified:
                        verified_spans += 1

        # Multi-valued entity list
        else:
            for entity in val:
                if not isinstance(entity, dict):
                    continue
                source = entity.get("source", "record")
                if source == "record":
                    total_record_values += 1
                    evidence_list = entity.get("evidence") or []
                    if not evidence_list:
                        span_details.append({
                            "tag": tag,
                            "label": entity.get("label"),
                            "source": source,
                            "verified": False,
                            "reason": "Missing evidence array for record entity"
                        })
                    else:
                        entity_verified = True
                        for ev in evidence_list:
                            if not isinstance(ev, dict):
                                continue
                            p_slug = str(ev.get("page_slug", ""))
                            b_id = str(ev.get("block_id", ""))
                            quote = normalize_whitespace(str(ev.get("quote", "")))
                            target_text = block_lookup.get((p_slug, b_id)) or page_lookup.get(p_slug, "")
                            is_match = bool(quote and quote in target_text)
                            if not is_match:
                                entity_verified = False
                            span_details.append({
                                "tag": tag,
                                "label": entity.get("label"),
                                "page_slug": p_slug,
                                "block_id": b_id,
                                "quote": quote,
                                "verified": is_match
                            })
                        if entity_verified:
                            verified_spans += 1

    rate = round(verified_spans / total_record_values, 4) if total_record_values > 0 else 1.0
    return {
        "verified_spans_count": verified_spans,
        "total_record_values_count": total_record_values,
        "evidence_verified_rate": rate,
        "span_details": span_details
    }


def compute_window_derived_metrics(
    response_payload: dict,
    request_payload: dict,
    extraction_latency_ms: Optional[float] = None
) -> dict:
    """
    Derives per-window metrics as specified in Section 3 of METADATA_API_Payload_Specification.md.
    """
    fields = response_payload.get("fields", {})
    pages = request_payload.get("pages", [])

    confidences = []
    low_conf_count = 0
    total_evidence_spans = 0

    for tag, field_obj in fields.items():
        if isinstance(field_obj, dict):
            conf = field_obj.get("confidence")
            if isinstance(conf, (int, float)):
                conf_val = float(conf)
                confidences.append(conf_val)
                if conf_val < 0.7:
                    low_conf_count += 1

            ev = field_obj.get("evidence")
            if isinstance(ev, list):
                total_evidence_spans += len(ev)

            val = field_obj.get("value")
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        item_ev = item.get("evidence")
                        if isinstance(item_ev, list):
                            total_evidence_spans += len(item_ev)

    mean_conf = round(sum(confidences) / len(confidences), 3) if confidences else None
    min_conf = round(min(confidences), 3) if confidences else None

    # Source OCR Confidence (Mean of non-null page_confidences)
    page_confs = []
    for p in pages:
        ocr_c = p.get("ocr_confidence") if isinstance(p, dict) else getattr(p, "ocr_confidence", None)
        if ocr_c is not None:
            page_confs.append(float(ocr_c))

    source_ocr_conf = round(sum(page_confs) / len(page_confs), 3) if page_confs else None

    ev_verification = verify_evidence_spans(fields, pages)
    engine_latency = response_payload.get("engine_latency_ms", 0.0)
    wall_latency = extraction_latency_ms if extraction_latency_ms is not None else engine_latency

    return {
        "unit_id": response_payload.get("unit_id"),
        "window_index": response_payload.get("window_index"),
        "engine": response_payload.get("engine"),
        "model": response_payload.get("model"),
        "taxonomy_version": response_payload.get("taxonomy_version"),
        "chars_in": response_payload.get("chars_in"),
        "usage": response_payload.get("usage"),
        "fields_attempted": response_payload.get("fields_attempted"),
        "fields_returned": response_payload.get("fields_returned"),
        "fields_declined": response_payload.get("fields_declined"),
        "mean_field_confidence": mean_conf,
        "min_field_confidence": min_conf,
        "low_confidence_fields_count": low_conf_count,
        "evidence_spans_count": total_evidence_spans,
        "evidence_verified_rate": ev_verification["evidence_verified_rate"],
        "source_ocr_confidence": source_ocr_conf,
        "engine_latency_ms": engine_latency,
        "extraction_latency_ms": wall_latency,
        "verification_details": ev_verification["span_details"]
    }


def aggregate_document_metrics(
    window_responses: list[dict],
    total_pages: Optional[int] = None,
    taxonomy_tags: Optional[list[str]] = None,
    unit_id: Optional[str] = None,
) -> dict:
    """
    Aggregates window responses into Full Document Aggregated Metrics (Section 5).
    """
    if not window_responses:
        return {
            "unit_id": unit_id or "",
            "windows_total": 0,
            "windows_completed": 0,
            "pages_read": 0,
            "pages_total": total_pages or 0,
            "extraction_coverage": 0.0,
            "field_coverage": 0.0,
            "avg_confidence": None,
            "min_confidence": None,
            "fields_below_0_7": 0,
            "evidence_verified_rate": None,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "avg_engine_latency_ms": 0.0,
            "source_ocr_confidence": None,
        }

    first_win = window_responses[0]
    effective_unit_id = unit_id or first_win.get("unit_id", "")
    windows_completed = len(window_responses)
    windows_total = first_win.get("window", {}).get("total", windows_completed) if isinstance(first_win.get("window"), dict) else windows_completed

    all_confidences = []
    low_conf_count = 0
    all_filled_tags = set()
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_engine_latency = 0.0

    for win in window_responses:
        usage = win.get("usage", {})
        total_prompt_tokens += usage.get("prompt_tokens", 0)
        total_completion_tokens += usage.get("completion_tokens", 0)
        total_engine_latency += win.get("engine_latency_ms", 0.0)

        fields = win.get("fields", {})
        for tag, fobj in fields.items():
            if isinstance(fobj, dict) and fobj.get("value") is not None:
                all_filled_tags.add(tag)
                conf = fobj.get("confidence")
                if isinstance(conf, (int, float)):
                    conf_f = float(conf)
                    all_confidences.append(conf_f)
                    if conf_f < 0.7:
                        low_conf_count += 1

    avg_conf = round(sum(all_confidences) / len(all_confidences), 3) if all_confidences else None
    min_conf = round(min(all_confidences), 3) if all_confidences else None
    avg_engine_latency = round(total_engine_latency / windows_completed, 2) if windows_completed > 0 else 0.0

    schema_tag_count = len(taxonomy_tags) if taxonomy_tags else max(1, len(all_filled_tags))
    field_coverage = round(len(all_filled_tags) / schema_tag_count, 4) if schema_tag_count > 0 else 0.0

    pages_read = total_pages or windows_completed
    pages_total_count = total_pages or windows_total
    extraction_coverage = round(pages_read / pages_total_count, 4) if pages_total_count > 0 else 1.0

    return {
        "unit_id": effective_unit_id,
        "windows_total": windows_total,
        "windows_completed": windows_completed,
        "pages_read": pages_read,
        "pages_total": pages_total_count,
        "extraction_coverage": extraction_coverage,
        "field_coverage": field_coverage,
        "avg_confidence": avg_conf,
        "min_confidence": min_conf,
        "fields_below_0_7": low_conf_count,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "avg_engine_latency_ms": avg_engine_latency,
    }


def build_metadata_extraction_prompt(request: MetadataRequest) -> tuple[str, str, int]:
    """
    Constructs the system prompt, user prompt, and calculates total characters consumed.
    """
    tags_list = request.tags or []
    tags_str = ", ".join([f'"{t}"' for t in tags_list])
    lang_hint_str = f"Language hints: {', '.join(request.language_hint)}\n" if request.language_hint else ""

    system_prompt = (
        "You are an archival metadata extraction system following the Kalanjiyam Metadata Extraction Specification (v1.0).\n"
        "Your task is to analyze document text pages provided as typed blocks with IDs, and extract archival description metadata fields ONLY for the requested tags.\n\n"
        "RULES:\n"
        f"1. Target Tags: [{tags_str}]. You MUST NOT return any tag that is not in this list.\n"
        "2. If evidence for a tag is present in the document text, extract it with appropriate field properties.\n"
        "3. If evidence is NOT present, DECLINE the tag by omitting it entirely from the 'fields' object. Never hallucinate or invent values or tags.\n"
        "4. Evidence and Provenance:\n"
        "   - 'record': For facts directly stated in the text. MUST provide 'evidence' citing page_slug, block_id, and verbatim quote.\n"
        "   - 'derived': For synthesised/summarised information (e.g. SCOPE CONTENT). Cite contributing page_slugs without block_id/quote.\n"
        "   - 'enrichment': Sourced from external authority files.\n"
        "5. Field Structure:\n"
        "   - Single-valued tag (e.g. TITLE, DATE, SCOPE CONTENT):\n"
        '     "TAG_NAME": {"value": "...", "confidence": 0.90, "source": "record", "evidence": [{"page_slug": "...", "block_id": "...", "quote": "..."}]}\n'
        "   - Entity list tag (e.g. PERSON NAME, PLACE, SUBJECT):\n"
        '     "PERSON NAME": {"confidence": 0.85, "value": [{"label": "...", "variants": [...], "dates": "...", "source": "record", "evidence": [{"page_slug": "...", "block_id": "...", "quote": "..."}]}]}\n'
        "6. Quotes must match text in the specified block VERBATIM.\n"
        "7. Output ONLY a single valid JSON object of the form: {\"fields\": { ... }}\n"
    )

    page_sections = []
    total_chars = 0

    for page in request.pages:
        slug = str(page.page_slug)
        conf_str = f" (OCR confidence: {page.ocr_confidence})" if page.ocr_confidence is not None else ""
        page_header = f"=== Page Slug: {slug}{conf_str} ==="
        block_lines = []
        for block in page.blocks:
            b_id = str(block.id)
            b_type = str(block.type)
            b_text = str(block.text or "")
            total_chars += len(b_text)
            block_lines.append(f"[Block id={b_id} type={b_type}] {b_text}")

        page_sections.append(page_header + "\n" + "\n".join(block_lines))

    doc_text = "\n\n".join(page_sections)
    win_idx = request.window.index if request.window else 0
    win_tot = request.window.total if request.window else 1
    win_slugs = ", ".join(request.window.page_slugs) if request.window else ""

    user_prompt = (
        f"Unit ID: {request.unit_id}\n"
        f"Window: Index {win_idx} of {win_tot} (Pages: {win_slugs})\n"
        f"Taxonomy Version: {request.taxonomy_version}\n"
        f"{lang_hint_str}"
        f"Requested Tags: {tags_str}\n\n"
        f"Document Content:\n"
        f"{doc_text}\n\n"
        f"Extract metadata fields for the requested tags. Output valid JSON with 'fields' object."
    )

    return system_prompt, user_prompt, total_chars


def format_kalanjiyam_metadata_response(
    raw_output: Any,
    request: MetadataRequest,
    chars_in: int,
    engine_latency_ms: float,
    prompt_tokens: int,
    completion_tokens: int,
    engine: str,
    model_name: str,
    model_version: str,
) -> dict:
    """
    Validates, filters, and formats metadata extraction output to comply strictly with Specification v1.0.
    """
    parsed_json = extract_json_from_model_output(raw_output) if isinstance(raw_output, str) else raw_output
    raw_fields = {}
    if isinstance(parsed_json, dict):
        if "fields" in parsed_json and isinstance(parsed_json["fields"], dict):
            raw_fields = parsed_json["fields"]
        else:
            raw_fields = {k: v for k, v in parsed_json.items() if k not in (
                "contract_version", "status", "engine", "model", "taxonomy_version",
                "unit_id", "window_index", "chars_in", "engine_latency_ms", "usage",
                "fields_attempted", "fields_returned", "fields_declined"
            )}

    validated_fields = {}
    requested_tags_order = request.tags or []

    for tag in requested_tags_order:
        if tag in raw_fields and isinstance(raw_fields[tag], dict):
            field_data = raw_fields[tag]
            val = field_data.get("value")
            if val is None:
                continue

            conf = field_data.get("confidence", 0.85)
            try:
                conf = float(conf)
                conf = max(0.0, min(1.0, conf))
            except (ValueError, TypeError):
                conf = 0.85

            field_obj = {
                "confidence": round(conf, 2),
            }

            if isinstance(val, list):
                cleaned_entities = []
                for entity in val:
                    if isinstance(entity, dict):
                        ent_obj = {
                            "label": str(entity.get("label", "")),
                            "source": str(entity.get("source", "record")),
                        }
                        if "variants" in entity and isinstance(entity["variants"], list):
                            ent_obj["variants"] = entity["variants"]
                        if "dates" in entity and entity["dates"] is not None:
                            ent_obj["dates"] = str(entity["dates"])
                        if "auth_id" in entity and entity["auth_id"] is not None:
                            ent_obj["auth_id"] = str(entity["auth_id"])
                        if "note" in entity and entity["note"] is not None:
                            ent_obj["note"] = str(entity["note"])
                        if "evidence" in entity and isinstance(entity["evidence"], list):
                            ent_obj["evidence"] = entity["evidence"]
                        cleaned_entities.append(ent_obj)
                    else:
                        cleaned_entities.append(entity)
                field_obj["value"] = cleaned_entities
            else:
                field_obj["value"] = val
                if "source" in field_data:
                    field_obj["source"] = str(field_data["source"])
                else:
                    field_obj["source"] = "record"
                if "evidence" in field_data and isinstance(field_data["evidence"], list):
                    field_obj["evidence"] = field_data["evidence"]

            validated_fields[tag] = field_obj

    fields_attempted = len(requested_tags_order)
    fields_returned = len(validated_fields)
    fields_declined = max(0, fields_attempted - fields_returned)
    win_idx = request.window.index if request.window else 0

    return {
        "contract_version": "1.0",
        "status": "success",
        "engine": engine,
        "model": {
            "name": model_name,
            "version": model_version,
        },
        "taxonomy_version": request.taxonomy_version,
        "unit_id": request.unit_id,
        "window_index": win_idx,
        "chars_in": int(chars_in),
        "engine_latency_ms": round(float(engine_latency_ms), 2),
        "usage": {
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "total_tokens": int(prompt_tokens + completion_tokens),
        },
        "fields_attempted": int(fields_attempted),
        "fields_returned": int(fields_returned),
        "fields_declined": int(fields_declined),
        "fields": validated_fields,
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


def extract_json_from_model_output(text: str):
    """
    Attempts to extract JSON from raw model output, handling markdown code fences
    (e.g., ```json ... ```) or embedded JSON objects/arrays.
    """
    if not isinstance(text, str):
        return text
    clean_text = text.strip()
    # 1. Direct parse
    try:
        return json.loads(clean_text)
    except Exception:
        pass

    # 2. Extract from markdown code fence
    if "```" in clean_text:
        import re
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean_text)
        if match:
            code_content = match.group(1).strip()
            try:
                return json.loads(code_content)
            except Exception:
                pass

    # 3. Find first { or [ to last } or ]
    first_brace = clean_text.find("{")
    first_bracket = clean_text.find("[")

    start_idx = -1
    if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
        start_idx = first_brace
        end_idx = clean_text.rfind("}")
    elif first_bracket != -1:
        start_idx = first_bracket
        end_idx = clean_text.rfind("]")
    else:
        end_idx = -1

    if start_idx != -1 and end_idx > start_idx:
        try:
            return json.loads(clean_text[start_idx:end_idx + 1])
        except Exception:
            pass

    return clean_text


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
    model_name: Optional[str] = None,
    model_version: Optional[str] = None,
) -> dict:
    """Formats DotsOCR / Gemma-4 outputs into the Kalanjiyam OCR Service Contract (v2.1) JSON response."""
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

    if not model_name:
        if "gemma" in selected_engine.lower():
            model_name = "gemma-4-26b-a4b-it"
            model_version = model_version or "1.0.0"
        else:
            model_name = "dots-ocr"
            model_version = model_version or "4.0.0"
    else:
        model_version = model_version or "4.0.0"

    result_payload = {
        # Kalanjiyam OCR Service Contract (v2.1) required fields
        "contract_version": "2.1",
        "engine": selected_engine,
        "model": {
            "name": model_name,
            "version": model_version
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


@app.get("/engines")
@app.get("/v1/engines")
def get_engines():
    """Returns list of active engines for Kalanjiyam service discovery."""
    return {
        "status": "ok",
        "engines": list(ENGINE_CONFIGS.keys()),
        "current_engine": gpu_manager.current_engine or DEFAULT_ENGINE,
        "default_engine": DEFAULT_ENGINE,
    }


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
        "current_engine": gpu_manager.current_engine,
        "default_engine": DEFAULT_ENGINE,
        "available_engines": list(ENGINE_CONFIGS.keys()),
        "api_max_concurrent_limit": API_MAX_CONCURRENT_REQUESTS,
        "vllm_max_num_seqs": VLLM_MAX_NUM_SEQS,
        "idle_seconds": round(time.time() - gpu_manager.last_active_timestamp, 1),
        "idle_timeout_seconds": IDLE_TIMEOUT_SECONDS,
        "gpus": gpus
    }


@app.get("/health")
def health_check(engine: Optional[str] = None):
    """Health check endpoint."""
    target_engine = resolve_engine(engine)
    available = gpu_manager.is_available(target_engine)
    payload = {
        "status": "healthy" if available else "unavailable",
        "backend_running": gpu_manager.is_running(),
        "current_engine": gpu_manager.current_engine,
        "checked_engine": target_engine,
        "default_engine": DEFAULT_ENGINE,
        "active_gpus": gpu_manager.active_gpus,
        "gpu_count_configured": GPU_COUNT,
        "api_max_concurrent_limit": API_MAX_CONCURRENT_REQUESTS,
        "vllm_max_num_seqs": VLLM_MAX_NUM_SEQS,
        "min_free_vram_mb": ENGINE_CONFIGS[target_engine]["min_free_vram_mb"],
    }
    return JSONResponse(status_code=200 if available else 503, content=payload)


@app.post("/v1/ocr")
async def run_ocr(
    file: Optional[UploadFile] = File(None, description="Image file (PNG, JPG, JPEG, WEBP)"),
    image: Optional[UploadFile] = File(None, description="Alternative field name for image file"),
    engine: Optional[str] = Form(None, description="OCR engine name ('dots-ocr' or 'gemma-4')"),
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

    target_engine = resolve_engine(engine)
    engine_cfg = ENGINE_CONFIGS[target_engine]
    filename = target_file.filename or "uploaded_image.jpg"
    log("REQ RECEIVED", f"Processing image: '{filename}' (engine: {target_engine}, language: {language or 'sa'}, max_tokens: {eff_max_tokens})")

    async with request_semaphore:
        gpu_manager.touch()
        try:
            backend = gpu_manager.start_backend(engine_id=target_engine)
        except Exception as e:
            log("REQ ERROR", f"GPU initialization failed for engine '{target_engine}': {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize GPU model backend for '{target_engine}': {str(e)}")

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
        log("INFERENCE START", f"Sending vision request for '{filename}' to vLLM engine ({target_engine}) on GPU {active_gpu}...")

        payload = {
            "model": MODEL_NAME,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": img_uri}},
                        {"type": "text", "text": engine_cfg["prompt"]}
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

        parsed_layout = extract_json_from_model_output(content_text)

        usage = res_json.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        end_time = time.perf_counter()
        duration_seconds = round(end_time - start_time, 4)
        throughput = round(completion_tokens / duration_seconds, 2) if duration_seconds > 0 else 0.0

        gpu_manager.touch()

        log("INFERENCE SUCCESS", f"Finished '{filename}' [{target_engine}] in {duration_seconds}s | Tokens: {prompt_tokens} in / {completion_tokens} out | Speed: {throughput} tok/s")

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
            engine=engine if (engine and engine.strip()) else target_engine,
            model_name=engine_cfg.get("model_name"),
            model_version=engine_cfg.get("model_version"),
        )


@app.post("/free-vram")
def free_vram_manually():
    """Manually trigger VRAM cleanup by stopping the GPU backend."""
    log("MANUAL FREE", "Received manual request to free VRAM...")
    gpu_manager.stop_backend()
    return {"status": "success", "message": "VRAM freed. Backend stopped."}


# =============================================================================
# METADATA EXTRACTION & METRICS ENDPOINTS (Contract v1.0)
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Ensure HTTP error payloads strictly follow contract: {"status": "error", "detail": "..."}"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "detail": exc.detail}
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Format validation errors with standard error contract."""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"status": "error", "detail": str(exc)}
    )


@app.post("/v1/metadata", response_model=MetadataResponse)
async def extract_metadata(
    request_data: MetadataRequest,
    engine: Optional[str] = None,
    max_tokens: Optional[int] = None,
):
    """
    Extract archival metadata fields and return single JSON payload with fields and window metrics.
    Complies with Kalanjiyam Metadata Extraction API Response & Metrics Specification (v1.0).
    """
    if not request_data.pages:
        raise HTTPException(status_code=400, detail="Request must contain at least one page.")

    if not request_data.tags:
        raise HTTPException(status_code=400, detail="Request must contain a non-empty list of tags.")

    eff_max_tokens = max_tokens or request_data.max_tokens or 4500
    if eff_max_tokens < 100:
        eff_max_tokens = 4500

    target_engine_name = engine or request_data.engine or DEFAULT_METADATA_ENGINE
    target_engine = resolve_engine(target_engine_name)
    engine_cfg = ENGINE_CONFIGS.get(target_engine, ENGINE_CONFIGS.get("gemma-4", ENGINE_CONFIGS["dots-ocr"]))

    win_idx = request_data.window.index if request_data.window else 0
    win_tot = request_data.window.total if request_data.window else 1
    log("METADATA REQ", f"Processing unit '{request_data.unit_id}' window {win_idx}/{win_tot} (tags: {len(request_data.tags)}, pages: {len(request_data.pages)}, engine: {target_engine})")

    system_prompt, user_prompt, chars_in = build_metadata_extraction_prompt(request_data)

    async with request_semaphore:
        gpu_manager.touch()
        try:
            backend = gpu_manager.start_backend(engine_id=target_engine)
        except Exception as e:
            log("REQ ERROR", f"GPU initialization failed for engine '{target_engine}': {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to initialize GPU model backend for '{target_engine}': {str(e)}")

        start_time = time.perf_counter()
        active_gpu = backend["gpu_idx"]
        backend_url = f"http://localhost:{backend['port']}/v1/chat/completions"

        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": eff_max_tokens,
            "temperature": 0.0
        }

        try:
            response = await asyncio.to_thread(
                requests.post, backend_url, json=payload, timeout=240
            )
            if response.status_code != 200:
                log("REQ ERROR", f"vLLM backend returned status code {response.status_code}: {response.text}")
                raise HTTPException(status_code=response.status_code, detail=f"vLLM server error: {response.text}")
            res_json = response.json()
        except requests.exceptions.RequestException as req_err:
            log("REQ ERROR", f"HTTP connection to vLLM backend failed: {str(req_err)}")
            raise HTTPException(status_code=502, detail=f"Backend request failed: {str(req_err)}")

        choice = res_json.get("choices", [{}])[0]
        finish_reason = choice.get("finish_reason")
        content_text = choice.get("message", {}).get("content", "")

        if finish_reason == "length":
            log("METADATA WARN", f"Generation truncated for unit '{request_data.unit_id}' window {win_idx}")

        end_time = time.perf_counter()
        duration_seconds = max(0.001, end_time - start_time)
        engine_latency_ms = round(duration_seconds * 1000.0, 2)

        usage = res_json.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        gpu_manager.touch()

        formatted_resp = format_kalanjiyam_metadata_response(
            raw_output=content_text,
            request=request_data,
            chars_in=chars_in,
            engine_latency_ms=engine_latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            engine=target_engine,
            model_name=engine_cfg.get("model_name", "gemma-4-26b-a4b-it"),
            model_version=engine_cfg.get("model_version", "1.0.0"),
        )

        log("METADATA SUCCESS", f"Finished unit '{request_data.unit_id}' window {win_idx} in {duration_seconds:.2f}s | Returned {formatted_resp['fields_returned']}/{formatted_resp['fields_attempted']} fields (declined {formatted_resp['fields_declined']}) | Tokens: {prompt_tokens} in / {completion_tokens} out")

        return formatted_resp


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Kalanjiyam DotsOCR & Gemma-4 API Service")
    parser.add_argument("--host", default="0.0.0.0", help="Host address to bind to")
    parser.add_argument("--port", type=int, default=int(os.getenv("FASTAPI_PORT", "8887")), help="Port to run FastAPI service on")
    parser.add_argument("--engine", default=DEFAULT_ENGINE, help="Default OCR engine ('dots-ocr' or 'gemma-4')")
    parser.add_argument("--model-path", default=None, help="Custom model path override")
    parser.add_argument("--tp-size", "--tensor-parallel-size", type=int, default=None, help="Tensor parallel size (e.g. 1 on H100, 2 on 2x A6000)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=None, help="vLLM GPU memory utilization target (0.1 - 1.0)")
    parser.add_argument("--max-model-len", type=int, default=None, help="Maximum context sequence length")
    parser.add_argument("--quantization", default=None, help="Quantization method ('fp8', 'bitsandbytes', 'awq')")
    parser.add_argument("--kv-cache-dtype", default=None, help="KV cache data type (e.g. 'fp8', 'auto')")
    parser.add_argument("--pinned-gpu", default=None, help="Pinned GPU ID (e.g. 0 or 1)")
    parser.add_argument("--enforce-eager", action="store_true", default=False, help="Disable CUDA graphs and enforce eager mode")
    args = parser.parse_args()

    if args.engine:
        DEFAULT_ENGINE = resolve_engine(args.engine)
        os.environ["DEFAULT_ENGINE"] = DEFAULT_ENGINE
    if args.model_path:
        MODEL_PATH = args.model_path
        os.environ["MODEL_PATH"] = args.model_path
    if args.tp_size is not None:
        os.environ["TENSOR_PARALLEL_SIZE"] = str(args.tp_size)
    if args.gpu_memory_utilization is not None:
        os.environ["GPU_MEMORY_UTILIZATION"] = str(args.gpu_memory_utilization)
    if args.max_model_len is not None:
        os.environ["VLLM_MAX_MODEL_LEN"] = str(args.max_model_len)
    if args.quantization:
        os.environ["VLLM_QUANTIZATION"] = str(args.quantization)
    if args.kv_cache_dtype:
        os.environ["VLLM_KV_CACHE_DTYPE"] = str(args.kv_cache_dtype)
    if args.pinned_gpu:
        os.environ["PINNED_GPU_ID"] = str(args.pinned_gpu)
    if args.enforce_eager:
        os.environ["VLLM_ENFORCE_EAGER"] = "1"

    log("SERVER START", f"Starting OCR API server on {args.host}:{args.port} (default engine: {DEFAULT_ENGINE})...")
    uvicorn.run("server_app:app", host=args.host, port=args.port, reload=False)
