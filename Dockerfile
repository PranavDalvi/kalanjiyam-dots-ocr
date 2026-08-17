# Use pre-configured DotsOCR base image containing vLLM, PyTorch & CUDA
FROM rednotehilab/dots.ocr:vllm-openai-v0.9.1

# Install uv from official Astral image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /workspace

# Install git and latest transformers for Gemma 4 (gemma4) architecture support
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

# Install API service dependencies using uv
RUN uv pip install --system \
    fastapi \
    uvicorn \
    pymupdf \
    pillow \
    requests \
    git+https://github.com/huggingface/transformers.git \
    accelerate

# Copy project source code into container
COPY server_app.py /workspace/server_app.py
COPY patch.sh /workspace/patch.sh
COPY instruction_prompts.yml /workspace/instruction_prompts.yml

# Validate the patch script during the image build. The model itself is downloaded
# at runtime, so the script correctly reports that no local config exists yet.
RUN bash /workspace/patch.sh

# Environment defaults
ENV MODEL_PATH=rednote-hilab/dots.ocr
ENV API_MAX_CONCURRENT_REQUESTS=8
ENV VLLM_MAX_NUM_SEQS=1
ENV GPU_MEMORY_UTILIZATION=0.90
ENV GPU_COUNT=1
ENV MIN_FREE_VRAM_MB=36000
ENV GPU_MEMORY_HEADROOM_MB=1024
ENV IDLE_TIMEOUT_SECONDS=1800
ENV FASTAPI_PORT=8887
ENV VLLM_PORT=8000

# Expose ports: 8887 for FastAPI, 8000 for vLLM internal backend
EXPOSE 8887 8000

# Override base image ENTRYPOINT and set command to run server_app.py
ENTRYPOINT ["python3", "server_app.py"]
