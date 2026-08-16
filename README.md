# Kalanjiyam DotsOCR & Archival Metadata Extraction Service

A high-performance FastAPI service providing layout-aware OCR (DotsOCR / Gemma-4) and Archival Metadata Extraction with dynamic GPU allocation and concurrency gating.

---

## Key Features

- **Layout-Aware OCR (`POST /v1/ocr`)**: Processes document images into typed layout blocks (headings, paragraphs, tables, equations) with pixel bounding boxes conforming to Kalanjiyam OCR Service Contract (v2.1).
- **Archival Metadata Extraction (`POST /v1/metadata`)**: Stateless per-window metadata extraction returning structured archival description fields (titles, dates, entities, scope content) alongside per-window metrics (`chars_in`, `engine_latency_ms`, `usage`, `fields_attempted`, `fields_returned`, `fields_declined`) using `gemma-4`, conforming to **Metadata Extraction API Specification (v1.0)**.
- **Async Batch Inference Pipeline (`async_infer-lazy-buffer_newer_w_s3.py`)**: High-throughput pipeline streaming images and PDFs from local disk or S3 through LLM backends (vLLM, SGLang, LMDeploy, TRT-LLM).
- Converts PDFs to images lazily per-page using PyMuPDF
- Concurrent inference with configurable concurrency and request rate
- Disk spill buffer when RAM queue is full
- Resume support — skips already-processed IDs on restart
- Supports SGLang, vLLM, LMDeploy, and TRT-LLM backends
- Path-style S3 addressing for VPC Gateway Endpoint compatibility

---

## Installation

```bash
pip install aiohttp aiofiles numpy requests filetype tqdm transformers \
            jinja2 pandas boto3 fsspec s3fs smart_open pillow pymupdf \
            python-magic pyyaml
```

---

## Usage

```bash
python async_infer.py \
  --input-path /path/to/images_or_pdfs \
  --output-file results.jsonl \
  --instruction-path instruction_prompts.yaml \
  --task dotsocr_w_layout \
  --temp-path /tmp \
  --backend vllm-chat \
  --host 0.0.0.0 \
  --port 8000 \
  --max-concurrency 100
```

---

## Arguments

### Input / Output

| Argument | Required | Default | Description |
|---|---|---|---|
| `--input-path` | ✅ | — | Path to input file or directory (PDF, PNG, JPG, JPEG) or `s3://` URI |
| `--output-file` | ✅ | — | Output JSONL file path |
| `--instruction-path` | ✅ | — | YAML file containing task/prompt templates |
| `--task` | ✅ | — | Dot-separated key into the YAML template (e.g. `ocr.books`) |
| `--temp-path` | ✅ | — | Directory for temporary PDF files downloaded from S3 (e.g. `/opt/dlami/nvme/myuser`) *ALWAYS KEEP THE NVME PATH*|
| `--template-fields` | ❌ | — | Record fields to fill into the template (space-separated) |
| `--spill-path` | ❌ | same dir as output | Path for the disk spill JSONL file (*ONLY REQUIRED IF THE OUTPUT PATH IS NOT IN NVME. MAKE SURE THE SPILL PATH IS IN NVME TO AVOID HITTING YOUR STORAGE QUOTA*) |

### Model & Backend

| Argument | Required | Default | Description |
|---|---|---|---|
| `--backend` | ❌ | `sglang` | Inference backend: `sglang`, `sglang-native`, `sglang-oai`, `sglang-oai-chat`, `vllm`, `vllm-chat`, `lmdeploy`, `lmdeploy-chat`, `trt` |
| `--model` | ❌ | auto-detect | Model name or path |
| `--tokenizer` | ❌ | — | Tokenizer name or path |

### Server / Connection

| Argument | Required | Default | Description |
|---|---|---|---|
| `--base-url` | ❌ | — | Full base URL (overrides host/port) |
| `--host` | ❌ | `0.0.0.0` | Server host |
| `--port` | ❌ | backend default | Server port (sglang: 30000, vllm: 8000, lmdeploy: 23333) |

### Generation

| Argument | Required | Default | Description |
|---|---|---|---|
| `--max-new-tokens` | ❌ | — | Max tokens to generate |
| `--extra-request-body` | ❌ | — | Extra JSON fields for the request payload (e.g. `'{"temperature":0.0}'`) |
| `--apply-chat-template` | ❌ | `False` | Apply chat template to prompts |
| `--ignore-eos` | ❌ | `False` | Ignore EOS token during generation |

### Request Control

| Argument | Required | Default | Description |
|---|---|---|---|
| `--request-rate` | ❌ | `inf` | Target requests per second (`inf` = unlimited) |
| `--max-concurrency` | ❌ | `100` | Max concurrent in-flight requests |
| `--buffer-size` | ❌ | `50` | RAM queue size for prefetched requests |
| `--enable-stream` | ❌ | `False` | Enable streaming mode |

### PDF Options

| Argument | Required | Default | Description |
|---|---|---|---|
| `--pdf-dpi` | ❌ | `200` | DPI for PDF-to-image conversion |

### UX

| Argument | Required | Default | Description |
|---|---|---|---|
| `--disable-tqdm` | ❌ | `False` | Disable progress bars |

---

## Template YAML Format

The `--instruction-path` YAML file contains prompt templates. Use dot-separated keys with `--task` to select the right one.

```yaml
ocr:
  books: "Transcribe the text in this image exactly as it appears."
  receipts: "Extract all line items and totals from this receipt image."
```

Use `--task ocr.books` to select the first template.

Use `--template-fields` to inject record fields into the template using positional `{}` placeholders:

```yaml
ocr:
  books: "Transcribe page {} of file {}."
```

```bash
--template-fields page_num file_path
```

---

## S3 Support

S3 paths are supported for `--input-path`:

```bash
--input-path s3://my-bucket/path/to/pdfs/
```

### S3 on EC2

The script uses **path-style S3 URLs** (`s3.amazonaws.com/bucket`) by default, which is required when running on EC2 with a VPC Gateway Endpoint (no NAT gateway). This is handled automatically via `get_s3_client()`.

If you encounter DNS resolution errors for S3, also set this env var to cover any remaining boto3 calls:

```bash
export AWS_S3_ADDRESSING_STYLE=path
```

To make it permanent for your user only (does not affect other users on the cluster):

```bash
echo 'export AWS_S3_ADDRESSING_STYLE=path' >> ~/.bashrc
source ~/.bashrc
```

### Temporary Files (S3 PDFs)

PDFs pulled from S3 are downloaded to `--temp-path` during processing and deleted automatically when the run completes. On a DLAMI EC2 instance, use the NVMe scratch disk:

```bash
--temp-path /opt/dlami/nvme/myuser
```

If the process is killed before cleanup, remove leftover files safely with:

```bash
find /tmp -name "*.pdf" -user $(whoami) -delete
```

Or if using a custom temp path:

```bash
rm /opt/dlami/nvme/myuser/*.pdf
```

---

## Resume / Crash Recovery

The script tracks completed IDs in the output JSONL file. If the run is interrupted, re-running with the same `--output-file` will automatically skip already-processed items.

The disk spill file is deleted and recreated fresh on each run. Use `--spill-path` to control its location.

---

## Output Format

Each line in the output JSONL file is:

```json
{"id": "filename↳page_num", "generated_text": "..."}
```

For single images, `id` is the filename. For PDFs, `id` is `filename↳page_number`.

---

## Full Example

```bash
python async_infer-lazy-buffer_newer_w_s3.py \
  --input-path s3://my-bucket/pdfs/ \
  --output-file /data/ocr_results.jsonl \
  --instruction-path /fsxnew/shyam.pawar/inference_scripts/instruction_prompts.yml \
  --task dotsocr_w_layout \
  --backend vllm-chat \
  --model /fsxnew/opensource-models/weights/DotsOCR \
  --max-concurrency $((NUM_NODES * NUM_GPUS * NUM_REQ_PER_GPU)) \
  --buffer-size $((NUM_NODES * NUM_GPUS * NUM_REQ_PER_GPU)) \
  --temp-path /opt/dlami/nvme/myuser \
  --spill-path /opt/dlami/nvme/myuser/spill.jsonl \
  --extra-request-body '{"temperature": 0.7, "top_p": 0.9, "top_k": 50, "repetition_penalty": 1.2, "min_p": 0.01, "max_tokens": 8192}'
  --host 10.0.129.167 \
  --port 20100
```
