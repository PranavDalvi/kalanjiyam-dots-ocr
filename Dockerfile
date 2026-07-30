# Use pre-configured DotsOCR base image containing vLLM, PyTorch & CUDA
FROM rednotehilab/dots.ocr:vllm-openai-v0.9.1

# Set working directory
WORKDIR /workspace

# Install API service dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pymupdf \
    pillow \
    requests

# Copy project source code into container
COPY server_app.py /workspace/server_app.py
COPY patch.sh /workspace/patch.sh
COPY instruction_prompts.yml /workspace/instruction_prompts.yml

# Apply vLLM patch inside container
RUN bash /workspace/patch.sh || true

# Environment defaults
ENV MODEL_PATH=rednote-hilab/dots.ocr
ENV MAX_CONCURRENT_REQUESTS=2
ENV IDLE_TIMEOUT_SECONDS=1800
ENV FASTAPI_PORT=8887
ENV VLLM_PORT=8000

# Expose ports: 8887 for FastAPI, 8000 for vLLM internal backend
EXPOSE 8887 8000

# Override base image ENTRYPOINT and set command to run server_app.py
ENTRYPOINT ["python3", "server_app.py"]
