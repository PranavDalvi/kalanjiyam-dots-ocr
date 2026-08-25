"""
Kalanjiyam OCR Server Entrypoint.
Provides compatibility for running `python server.py` or referencing `server:app`.
"""
import os
import sys
import argparse
from server_app import (
    app,
    gpu_manager,
    ENGINE_CONFIGS,
    resolve_engine,
    DEFAULT_ENGINE,
    log,
)

if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser(description="Kalanjiyam DotsOCR & Gemma-4 API Service")
    parser.add_argument("--host", default="0.0.0.0", help="Host address to bind to")
    parser.add_argument("--port", type=int, default=int(os.getenv("FASTAPI_PORT", "8887")), help="Port to run FastAPI service on")
    parser.add_argument("--engine", default=os.getenv("DEFAULT_ENGINE", os.getenv("ENGINE", "dots-ocr")), help="Default OCR engine ('dots-ocr' or 'gemma-4')")
    parser.add_argument("--enable-gemma", action="store_true", default=False, help="Enable Gemma-4 engine option")
    parser.add_argument("--disable-gemma", action="store_true", default=False, help="Explicitly disable Gemma-4 engine option")
    parser.add_argument("--model-path", default=None, help="Custom model path override")
    args = parser.parse_args()

    if args.enable_gemma:
        os.environ["ENABLE_GEMMA"] = "1"
    elif args.disable_gemma:
        os.environ["ENABLE_GEMMA"] = "0"

    selected_engine = resolve_engine(args.engine)
    os.environ["DEFAULT_ENGINE"] = selected_engine
    if "gemma" in selected_engine or "archival" in selected_engine:
        os.environ["ENABLE_GEMMA"] = "1"
    if args.model_path:
        os.environ["MODEL_PATH"] = args.model_path

    log("SERVER START", f"Starting OCR API server on {args.host}:{args.port} (default engine: {selected_engine})...")
    uvicorn.run("server_app:app", host=args.host, port=args.port, reload=False)
