"""
bench_infer.py — Async multimodal batch inference runner.

Supports image files and PDFs as input, converting them to base64-encoded
images on the fly using a process pool. Converted images are queued for
async inference against an OpenAI-compatible or SGLang/vLLM backend.

High-level architecture:
  ┌──────────────┐    ┌───────────────────┐    ┌──────────────────┐
  │  prepare_    │    │    producer()     │    │  consumer_worker │
  │  prompts()   │───▶│  process pool     │───▶│  async inference │
  │  (lazy gen.) │    │  image convert    │    │  per-worker      │
  └──────────────┘    └──────┬────────────┘    └────────┬─────────┘
                             │  RAM queue / disk spill   │
                             └──────────────────────────▶│ aiofiles writer
"""

import asyncio
import base64
import io
import json
import logging as pylogging
import os
import resource
import sys
import time
import yaml
import magic
from pathlib import Path
from argparse import ArgumentParser
from datetime import datetime, timedelta
from collections import deque
import aiohttp
import aiofiles
import numpy as np
import requests
import filetype
from tqdm.asyncio import tqdm as async_tqdm
from tqdm import tqdm
from transformers import logging, AutoTokenizer
from jinja2 import Template
import pandas as pd
import threading
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Tuple
import boto3
from urllib.parse import urlparse
from PIL import Image
import re
import fitz  # PyMuPDF
fitz.TOOLS.mupdf_display_errors(False)
fitz.TOOLS.mupdf_display_warnings(False)
import tempfile

import boto3
from botocore.config import Config
import msgpack
import struct

# ─────────────────────────────────────────────────────────────────────────────
# ERROR LOGGING (pdf s3 path → error)
# ─────────────────────────────────────────────────────────────────────────────

_error_logger = None
_error_log_path = None


def setup_error_logger(log_path: str):
	"""Initialise the error logger that writes to `log_path` as JSONL."""
	global _error_logger, _error_log_path
	_error_log_path = log_path

	_error_logger = pylogging.getLogger("ocr_errors")
	_error_logger.setLevel(pylogging.ERROR)
	_error_logger.propagate = False

	# Remove any existing handlers to avoid duplication on re-init.
	_error_logger.handlers.clear()

	fh = pylogging.FileHandler(log_path, mode="a", encoding="utf-8")
	fh.setLevel(pylogging.ERROR)
	# Raw message only — we build our own JSON structure.
	fh.setFormatter(pylogging.Formatter("%(message)s"))
	_error_logger.addHandler(fh)

	print(f"[INFO] Error log: {log_path}")


def log_error(pdf_s3_path: str, error: str, *, stage: str = "", page: int | None = None):
	"""
	Append one JSONL line to the error log.

	Format per line:
	  {"ts": "...", "pdf_s3_path": "...", "stage": "...", "page": N, "error": "..."}
	"""
	if _error_logger is None:
		return

	entry = {
		"ts": datetime.utcnow().isoformat(timespec="seconds"),
		"pdf_s3_path": pdf_s3_path,
		"stage": stage,
		"error": str(error),
	}
	if page is not None:
		entry["page"] = page

	try:
		_error_logger.error(json.dumps(entry, ensure_ascii=False))
	except Exception:
		pass

# ─────────────────────────────────────────────────────────────────────────────
# S3 CLIENT (lazy singleton)
# ─────────────────────────────────────────────────────────────────────────────

_s3_client = None

def get_s3_client():
	"""
	Return a module-level singleton boto3 S3 client.
	The client is created on first call and reused for all subsequent calls,
	avoiding the overhead of re-authentication on every request.
	"""
	global _s3_client

	if _s3_client is None:
		_s3_client = boto3.client(
			"s3",
			config=Config(
				s3={"addressing_style": "path"},
				retries={"max_attempts": 10},
			),
		)

	return _s3_client


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────────────────────────────────────

# Set of record IDs that have already been written to the output file.
# Populated at startup by scanning the output file, used to skip duplicates.
FINISHED_IDS = set()

# Compiled regex to extract the "id" field from a raw JSONL byte line.
# Avoids full JSON parsing when scanning large output files at startup.
ID_PATTERN = re.compile(rb'"id"\s*:\s*"([^"]+)"')

# Global args namespace, set in run_inference() so helper functions can
# access CLI arguments without passing them through every call.
global args


# ─────────────────────────────────────────────────────────────────────────────
# AIOHTTP SESSION FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def _create_bench_client_session():
	"""
	Create an aiohttp ClientSession tuned for high-throughput inference.

	- Timeout: 6 hours — LLM inference on long documents can be very slow.
	- Read buffer: 10 MB — prevents the TCP buffer from filling up when the
	  server streams tokens faster than the event loop drains them.
	"""
	BENCH_AIOHTTP_TIMEOUT_SECONDS = 6 * 60 * 60  # 6 hours
	BENCH_AIOHTTP_READ_BUFSIZE_BYTES = 10 * 1024**2  # 10 MB

	aiohttp_timeout = aiohttp.ClientTimeout(total=BENCH_AIOHTTP_TIMEOUT_SECONDS)
	return aiohttp.ClientSession(timeout=aiohttp_timeout, read_bufsize=BENCH_AIOHTTP_READ_BUFSIZE_BYTES)


# ─────────────────────────────────────────────────────────────────────────────
# MISC UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def remove_prefix(text, prefix):
	"""Strip a leading prefix from a string (pre-Python-3.9 compatibility)."""
	return text[len(prefix) :] if text.startswith(prefix) else text


def remove_suffix(text, suffix):
	"""Strip a trailing suffix from a string (pre-Python-3.9 compatibility)."""
	return text[: -len(suffix)] if text.endswith(suffix) else text


def detect_mime(base64_str):
	"""
	Detect the MIME type of a base64-encoded binary blob.
	Used to set the correct Content-Type when sending images to the API.
	Falls back to 'application/octet-stream' on any error.
	"""
	try:
		img_bytes = base64.b64decode(base64_str)
		mime = magic.from_buffer(img_bytes, mime=True)
		return mime
	except Exception:
		return "application/octet-stream"


def get_auth_headers():
	"""
	Build authorization headers from the OPENAI_API_KEY environment variable.
	Returns an empty dict if the key is not set (e.g. for local/unauthenticated servers).
	"""
	api_key = os.environ.get("OPENAI_API_KEY")
	if api_key:
		return {"Authorization": f"Bearer {api_key}"}
	else:
		return {}


# ─────────────────────────────────────────────────────────────────────────────
# BACKEND REQUEST FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

async def async_request_trt_llm(request_func_input, pbar = None):
	"""
	Send a streaming generation request to a TensorRT-LLM triton server.

	The endpoint expects a POST to `.../generate_stream` with `accumulate_tokens`
	set to True so that each SSE chunk contains the full text so far (not just
	the delta). We accumulate the last received text as the final output.
	"""
	api_url = request_func_input["api_url"]
	assert api_url.endswith("generate_stream")

	async with _create_bench_client_session() as session:
		payload = {
			"accumulate_tokens": True,
			"text_input": request_func_input["prompt"],
			"stream": True,
			**request_func_input["extra_request_body"],
		}
		if args.ignore_eos:
			del payload["min_length"]
			del payload["end_id"]

		output = {
			"id": request_func_input.get("id"),
			"generated_text": "",
		}

		try:
			async with session.post(url=api_url, json=payload) as response:
				if response.status != 200:
					pass
				
				async for chunk_bytes in response.content:
					chunk_bytes = chunk_bytes.strip()
					if not chunk_bytes:
						continue

					chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data:")
					data = json.loads(chunk)
					# Each chunk carries the full accumulated text so far.
					output["generated_text"] += data["text_output"]
		except Exception as e:
			print(f"[ERROR trt_llm request]: {e}")
			pass

		if pbar:
			pbar.update(1)
		return output


async def async_request_openai_completions(request_func_input, pbar = None):
	"""
	Send a request to an OpenAI-compatible /v1/completions (non-chat) endpoint.

	Supports both streaming and non-streaming modes controlled by args.enable_stream.
	Token deltas are accumulated into a single string for the output.
	"""
	api_url = request_func_input["api_url"]
	assert api_url.endswith("completions"), "OpenAI Completions API URL must end with 'completions'."

	prompt = request_func_input["prompt"]

	async with _create_bench_client_session() as session:
		payload = {
			"model": request_func_input["model"],
			"prompt": prompt,
			"best_of": 1,
			"stream": args.enable_stream,
			"ignore_eos": args.ignore_eos,
			**request_func_input["extra_request_body"],
		}
		headers = get_auth_headers()

		output = {
			"id": request_func_input.get("id"),
			"generated_text": "",
		}

		generated_text = ""
		try:
			async with session.post(url=api_url, json=payload, headers=headers) as response:
				if response.status != 200:
					pass

				async for chunk_bytes in response.content:
					chunk_bytes = chunk_bytes.strip()
					if not chunk_bytes:
						continue

					chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
					if chunk == "[DONE]":
						pass
					else:
						data = json.loads(chunk)
						# Some backends send a final usage-summary chunk with no token text.
						if data["choices"][0]["text"]:
							generated_text += data["choices"][0]["text"]

				output["generated_text"] = generated_text
				output["success"] = True
				output["output_len"] = len(generated_text)

		except Exception as e:
			print(f"[ERROR openai_completions request]: {e}")
			pass

	if pbar:
		pbar.update(1)
	return output


async def async_request_openai_chat_completions(request_func_input, pbar=None, max_retries=3, retry_delay=2, async_session=None):
	"""
	Send a request to an OpenAI-compatible /v1/chat/completions endpoint.

	Supports multimodal inputs: if image_data is provided, each image is
	embedded as a base64 data URI alongside the text prompt in the user message.

	Retries up to max_retries times if the response is empty or the request fails,
	with a fixed retry_delay between attempts.
	"""
	api_url = request_func_input["api_url"]
	assert api_url.endswith("chat/completions"), "OpenAI Chat Completions API URL must end with 'chat/completions'."

	# Build the message payload — interleave image(s) before the text prompt.
	if request_func_input["image_data"]:
		content_items = []
		for img_base64 in request_func_input["image_data"]:
			mime = detect_mime(img_base64)
			content_items.append({
				"type": "image_url",
				"image_url": {"url": f"data:{mime};base64,{img_base64}"}
			})
		content_items.append({
			"type": "text",
			"text": request_func_input["prompt"]
		})
		messages = [{"role": "user", "content": content_items}]
	else:
		messages = [{"role": "user", "content": request_func_input["prompt"]}]

	headers = get_auth_headers()

	for attempt in range(1, max_retries + 1):
		payload = {
			"model": request_func_input["model"],
			"messages": messages,
			"stream": args.enable_stream,
			"ignore_eos": args.ignore_eos,
			**request_func_input["extra_request_body"],
		}

		output = {"id": request_func_input.get("id"), "generated_text": ""}
		generated_text = ""

		try:
			async with async_session.post(url=api_url, json=payload, headers=headers) as response:
				if response.status != 200:
					continue  

				if not args.enable_stream:
					# Non-streaming: parse the complete JSON body at once.
					response_json = await response.json()
					generated_text = response_json["choices"][0]["message"]["content"]
				else:
					# Streaming: accumulate SSE delta content chunks.
					async for chunk_bytes in response.content:
						chunk_bytes = chunk_bytes.strip()
						if not chunk_bytes:
							continue
						chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
						if chunk == "[DONE]":
							continue
						data = json.loads(chunk)
						delta = data.get("choices", [{}])[0].get("delta", {})
						content = delta.get("content", "")
						if content:
							generated_text += content

				output["generated_text"] = generated_text.strip()

				# Only break out of the retry loop if we got a non-empty response.
				if output["generated_text"]:
					if pbar:
						pbar.update(1)
					break  

		except Exception as e:
			print(e)
			pass  

		# Wait before retrying on empty/failed response.
		if not output["generated_text"] and attempt < max_retries:
			await asyncio.sleep(retry_delay)

	return output


async def async_request_sglang_generate(request_func_input, pbar = None):
	"""
	Send a request to SGLang's native /generate endpoint.

	SGLang's native endpoint uses a slightly different payload structure than
	the OpenAI-compatible endpoint: sampling parameters go in a nested dict,
	and image data is passed as a top-level list.
	"""
	api_url = request_func_input["api_url"]
	prompt = request_func_input["prompt"]

	async with _create_bench_client_session() as session:
		payload = {
			"text": prompt,
			"sampling_params": {
				**request_func_input["extra_request_body"],
				"ignore_eos": args.ignore_eos,
			},
			"stream": args.enable_stream,
		}

		# Image data is a list of base64 strings or URLs when present.
		if request_func_input["image_data"]:
			payload["image_data"] = request_func_input["image_data"]

		headers = get_auth_headers()

		output = {
			"id": request_func_input.get("id"),
			"prompt": request_func_input.get("prompt"),
			"generated_text": "",
		}

		generated_text = ""
		try:
			async with session.post(url=api_url, json=payload, headers=headers) as response:
				if response.status != 200:
					pass

				async for chunk_bytes in response.content:
					chunk_bytes = chunk_bytes.strip()
					if not chunk_bytes:
						continue

					chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
					if chunk == "[DONE]":
						pass
					else:
						data = json.loads(chunk)
						# SGLang accumulates tokens in each chunk; take the latest.
						if "text" in data and data["text"]:
							generated_text = data["text"]
				output["generated_text"] = generated_text
		except Exception as e:
			pass

	if pbar:
		pbar.update(1)
	return output


# Map backend names to their async request function.
# Used in infer() to dispatch to the correct function at runtime.
ASYNC_REQUEST_FUNCS = {
	"sglang": async_request_sglang_generate,
	"sglang-native": async_request_sglang_generate,
	"sglang-oai": async_request_openai_completions,
	"sglang-oai-chat": async_request_openai_chat_completions,
	"vllm": async_request_openai_completions,
	"vllm-chat": async_request_openai_chat_completions,
	"lmdeploy": async_request_openai_completions,
	"lmdeploy-chat": async_request_openai_chat_completions,
	"trt": async_request_trt_llm,
}


# ─────────────────────────────────────────────────────────────────────────────
# PDF / IMAGE CONVERSION (process-pool workers)
# ─────────────────────────────────────────────────────────────────────────────

def pdf_to_base64_images(pdf_path, dpi=200):
	"""
	Convert every page of a PDF to a base64-encoded PNG image.

	Uses PyMuPDF (fitz) for rendering. The zoom factor is derived from the
	target DPI relative to fitz's default of 72 DPI.

	Args:
		pdf_path: Path to the PDF file.
		dpi: Render resolution. Higher values produce larger, clearer images.

	Returns:
		List of base64 PNG strings, one per page.
	"""
	try:
		doc = fitz.open(pdf_path)
		base64_images = []
		
		zoom = dpi / 72.0
		mat = fitz.Matrix(zoom, zoom)
		
		for page_num in range(len(doc)):
			page = doc[page_num]
			pix = page.get_pixmap(matrix=mat)
			img_bytes = pix.tobytes("png")
			img_base64 = base64.b64encode(img_bytes).decode('utf-8')
			base64_images.append(img_base64)
		
		doc.close()
		return base64_images
	except Exception as e:
		log_error(str(pdf_path), str(e), stage="pdf_to_base64_images")
		raise RuntimeError(f"Failed to convert PDF to images: {e}") from e
	

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(path):
	"""
	Load a JSONL file into a list of dicts. Silently skips malformed lines.
	"""
	records = []
	with open(path, "r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			try:
				obj = json.loads(line)
			except Exception:
				continue
			records.append(obj)
	return records


def load_parquet(path):
	"""
	Load a Parquet file into a list of dicts using pandas.
	"""
	df = pd.read_parquet(path)
	return df.to_dict(orient="records")


def load_task_template(yaml_path, task_path, default=None):
	"""
	Load a YAML task template by a dot-separated path.

	The YAML file may contain multiple nested templates; `task_path` specifies
	which one to retrieve (e.g. "ocr.default" or "classification.v2.prompt").

	Returns `default` if the path doesn't exist and a default is provided.
	Raises KeyError if the path is missing and no default is given.
	"""
	if not os.path.isfile(yaml_path):
		raise FileNotFoundError(f"YAML file not found: {yaml_path}")

	with open(yaml_path, "r", encoding="utf-8") as f:
		templates = yaml.safe_load(f)

	keys = task_path.split(".")
	current = templates

	for key in keys:
		if not isinstance(current, dict) or key not in current:
			if default is not None:
				return default
			raise KeyError(f"Path '{task_path}' not found in YAML (failed at '{key}').")
		current = current[key]
	return current


def get_by_path(data, path):
	"""
	Retrieve a value from a nested dict/list using a dot-separated path.

	Supports:
	  - Integer indices:  "messages.0.content"
	  - Simple filters:   "messages.[?role==user].content"

	Returns None if any segment of the path is missing.
	"""
	import re

	parts = path.split(".")
	cur = data
	for p in parts:
		# Handle list filter syntax, e.g. [?role==user]
		if p.startswith("[?"):
			m = re.match(r"\[\?([^=]+)==(.+)\]", p)
			if not m:
				raise ValueError(f"Invalid filter syntax: {p}")
			key, val = m.group(1), m.group(2)
			val = val.strip('"\'')
			if isinstance(cur, list):
				cur = next((x for x in cur if x.get(key) == val), None)
			else:
				cur = None
		elif p.isdigit():
			cur = cur[int(p)] if isinstance(cur, list) else None
		else:
			cur = cur.get(p) if isinstance(cur, dict) else None

		if cur is None:
			break
	return cur


def fill_instruction(template, record, template_fields):
	"""
	Render a prompt template by substituting positional fields from a record.

	`template_fields` is a list of dot-separated paths into `record`. Each
	path is resolved via get_by_path() and injected into the template string
	using Python's positional str.format() (i.e. {0}, {1}, …).

	Raises ValueError if any required field is missing from the record.
	"""
	if not template_fields:
		return template

	values = [get_by_path(record, field) for field in template_fields]

	if any(v is None for v in values):
		raise ValueError(f"Missing required field(s) for template_fields: {template_fields}")

	return template.format(*values)


# ─────────────────────────────────────────────────────────────────────────────
# RESUME / DEDUPLICATION
# ─────────────────────────────────────────────────────────────────────────────

def load_finished_ids(output_file):
	"""
	Scan the existing output file and return a set of already-processed IDs.

	Uses a byte-level regex to extract IDs without parsing full JSON — this is
	significantly faster for large output files (millions of records).

	When --skip-pages is set, the page suffix ("↳N") is stripped so that a
	whole document counts as finished if any of its pages are present.
	"""
	if not os.path.exists(output_file):
		return set()

	finished_ids = set()
	with open(output_file, "rb") as f:
		for line in f:
			m = ID_PATTERN.search(line)
			if m:
				if args.skip_pages:
					id_str = m.group(1).decode()
					finished_ids.add(id_str.rsplit("↳", 1)[0])
				else:
					finished_ids.add(m.group(1).decode())

	return finished_ids


# ─────────────────────────────────────────────────────────────────────────────
# S3 / FILESYSTEM HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def is_s3_path(path: str) -> bool:
	"""Return True if `path` is an S3 URI (starts with s3://)."""
	return path.startswith("s3://")


def preflight_s3_check(s3_path: str):
	"""
	Verify that the target S3 bucket and prefix are accessible.

	Performs a head_bucket() check (lightweight, no data transfer) and a
	list_objects_v2() to confirm the prefix is listable. Raises RuntimeError
	with a descriptive message on failure, so users get actionable feedback
	before the job starts rather than discovering the issue mid-run.
	"""
	parsed = urlparse(s3_path)
	bucket = parsed.netloc
	prefix = parsed.path.lstrip("/")

	print(f"[INFO] Preflight check for s3://{bucket}/{prefix or '(root)'}")

	s3_client = get_s3_client()

	try:
		s3_client.head_bucket(Bucket=bucket)
	except Exception as e:
		raise RuntimeError(
			f"Cannot access S3 bucket '{bucket}'. "
			f"Check AWS credentials / permissions.\nError: {e}"
		)

	try:
		s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
	except Exception as e:
		raise RuntimeError(
			f"Cannot list prefix s3://{bucket}/{prefix}.\nError: {e}"
		)

	print("[INFO] S3 access OK")


def iter_paths(input_path: str):
	"""
	Yield all file paths (local or s3://) under the given root directory.

	For S3, pages through list_objects_v2 to enumerate all objects under
	the given prefix. For local paths, walks the directory tree.
	"""
	if is_s3_path(input_path):
		parsed = urlparse(input_path)
		bucket = parsed.netloc
		prefix = parsed.path.lstrip("/")

		s3 = get_s3_client()
		paginator = s3.get_paginator("list_objects_v2")

		for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
			for obj in page.get("Contents", []):
				yield f"s3://{bucket}/{obj['Key']}"
	else:
		for root, _, files in os.walk(input_path):
			for name in files:
				yield os.path.join(root, name)


def detect_file_type(path: str, max_bytes=4096):
	"""
	Sniff the file type of a local or S3 file by reading its first `max_bytes`.

	Uses the `filetype` library for magic-byte detection. Returns the file
	extension string (e.g. "pdf", "png") or None if the type cannot be determined.
	Only reads as many bytes as needed, making it efficient for large S3 objects.
	"""
	try:
		if path.startswith("s3://"):
			parsed = urlparse(path)
			bucket = parsed.netloc
			key = parsed.path.lstrip("/")

			s3 = get_s3_client()
			resp = s3.get_object(
				Bucket=bucket,
				Key=key,
				Range=f"bytes=0-{max_bytes-1}",
			)
			head = resp["Body"].read()
		else:
			with open(path, "rb") as f:
				head = f.read(max_bytes)

		kind = filetype.guess(head)
		return kind.extension if kind else None

	except Exception:
		return None


def detect_file_type_from_ext(path: str):
	"""
	Infer file type from the file extension alone — no I/O required.

	This is the fast path used by default (--no-magic-bytes). Returns the
	lowercase extension string (e.g. "pdf", "png") or None if missing.
	"""
	ext = Path(path).suffix.lstrip(".").lower()
	return ext if ext else None


def detect_file_types_batch(paths: list, max_workers: int = 32):
	"""
	Detect file types for many paths in parallel using a thread pool.

	Used when --magic-bytes is explicitly requested. Each path triggers a
	magic-byte read (S3 Range GET or local disk read), but they execute
	concurrently across `max_workers` threads.

	Returns:
		Dict mapping path → extension string (or None).
	"""
	def _detect_one(p):
		return (p, detect_file_type(p))

	result = {}
	with ThreadPoolExecutor(max_workers=max_workers) as pool:
		for path, ext in pool.map(_detect_one, paths):
			result[path] = ext
	return result


def make_relative(path: str, root: str):
	"""
	Compute the relative path of `path` with respect to `root`, without extension.

	Used to generate stable, human-readable record IDs from file paths.
	For S3 paths, strips the bucket/prefix portion. For local paths, uses
	os.path.relpath(). The file extension is always removed from the result.
	"""
	if is_s3_path(root):
		parsed = urlparse(path)
		rel = parsed.path.lstrip("/")

		root_rel = urlparse(root).path.lstrip("/")
		if rel.startswith(root_rel):
			rel = rel[len(root_rel):].lstrip("/")

		return str(Path(rel).with_suffix(""))
	else:
		rel = os.path.relpath(path, root)
		return str(Path(rel).with_suffix(""))


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT PREPARATION
# ─────────────────────────────────────────────────────────────────────────────

def prepare_prompts(args):
	"""
	Build a (lazy) iterable of prompt records and the count of already-finished ones.

	Handles three input modes:
	  1. Directory (local or S3): streams records for every image/PDF found.
	  2. Single PDF: yields one lazy "expand later" record.
	  3. Single image: yields one fully resolved record.

	For PDFs in directory mode, records are emitted with `pdf_lazy=True` so
	that the producer can expand them into per-page records after counting pages.

	IDs use the relative path with "/" replaced by "↳" so they remain valid
	JSON string values. Page IDs append "↳{page_num}".

	Returns:
	  (iterable_of_records, count_of_already_finished_ids)
	"""
	global FINISHED_IDS
	FINISHED_IDS = load_finished_ids(args.output_file)

	input_path = args.input_path

	if is_s3_path(input_path):
		preflight_s3_check(input_path)

	# valid_extensions = {"pdf", "png", "jpg", "jpeg"}
	valid_extensions = {"pdf"}

	template = load_task_template(
		args.instruction_path, args.task, default={}
	)

	# ── DIRECTORY MODE ──────────────────────────────────────────────────────
	is_dir = is_s3_path(input_path) or os.path.isdir(input_path)

	if is_dir:
		print(f"Streaming directory: {input_path}")

		seen = set()

		def record_stream():
			DETECT_BATCH = 256

			def _detect_batch(paths):
				"""Detect file types for a batch, respecting --no-magic-bytes."""
				if args.no_magic_bytes:
					# Fast path: extension only, zero I/O (default).
					return {p: detect_file_type_from_ext(p) for p in paths}
				elif is_s3_path(input_path):
					# Parallel magic-byte detection for S3.
					return detect_file_types_batch(paths)
				else:
					# Local files: sequential magic-byte is fast enough.
					return {p: detect_file_type(p) for p in paths}

			def _emit(paths, type_map):
				"""Yield prompt records for paths whose detected type is valid."""
				for path in paths:
					ext = type_map.get(path)
					if ext not in valid_extensions:
						continue

					relative_path = make_relative(path, input_path)
					# Replace "/" with "↳" to produce a flat, valid ID string.
					name_sanitized = relative_path.replace("/", "↳")

					if ext == "pdf":
						# Defer page counting to the producer; emit a lazy placeholder.
						yield {
							"id": name_sanitized,
							"prompt": None,
							"template": template,
							"file_path": path,
							"relative_path": relative_path,
							"page_num": None,
							"is_pdf": True,
							"pdf_lazy": True,
							"local_path": None,
						}

					else:
						# Images: resolve the prompt immediately.
						dummy_record = {
							"id": name_sanitized,
							"file_path": relative_path,
						}

						try:
							filled_prompt = fill_instruction(
								template, dummy_record, args.template_fields
							)
						except Exception:
							filled_prompt = (
								template if isinstance(template, str)
								else str(template)
							)

						yield {
							"id": name_sanitized,
							"prompt": filled_prompt,
							"file_path": path,
							"relative_path": relative_path,
							"is_pdf": False,
						}

			buf = []
			for path in iter_paths(input_path):
				if path in seen:
					continue
				seen.add(path)
				buf.append(path)

				if len(buf) >= DETECT_BATCH:
					batch = buf
					buf = []
					type_map = _detect_batch(batch)
					yield from _emit(batch, type_map)

			if buf:
				type_map = _detect_batch(buf)
				yield from _emit(buf, type_map)

		return record_stream(), len(FINISHED_IDS)

	# ── SINGLE FILE MODE ────────────────────────────────────────────────────
	records = []
	ext = detect_file_type_from_ext(input_path) if args.no_magic_bytes else detect_file_type(input_path)

	template = load_task_template(
		args.instruction_path, args.task, default={}
	)

	if ext == "pdf":
		print(f"Processing PDF lazily: {input_path}")
		records.append({
			"id": os.path.basename(input_path),
			"prompt": None,
			"template": template,
			"file_path": input_path,
			"relative_path": os.path.basename(input_path),
			"page_num": None,
			"is_pdf": True,
			"pdf_lazy": True,
			"local_path": None,

		})
		return records, len(FINISHED_IDS)

	if ext in {"png", "jpg", "jpeg"}:
		image_id = os.path.basename(input_path)

		if image_id in FINISHED_IDS:
			print(f"Skipping already processed image: {image_id}")
			return [], len(FINISHED_IDS)

		dummy_record = {"id": image_id}

		try:
			filled_prompt = fill_instruction(
				template, dummy_record, args.template_fields
			)
		except ValueError:
			filled_prompt = (
				template if isinstance(template, str)
				else str(template)
			)

		records.append({
			"id": image_id,
			"prompt": filled_prompt,
			"file_path": input_path,
			"is_pdf": False,
			"local_path": None,

		})

		return records, len(FINISHED_IDS)

	raise ValueError(
		f"Unsupported or unreadable input path: {input_path}"
	)


# ─────────────────────────────────────────────────────────────────────────────
# PROCESS-POOL WORKERS
# These functions run in subprocesses via ProcessPoolExecutor and must be
# module-level (not closures) to be picklable.
# ─────────────────────────────────────────────────────────────────────────────

def convert_image_worker(file_path, is_pdf, page_numbers, dpi, local_path, temp_dir=None):
	"""
	Convert one image file or a batch of PDF pages to base64-encoded JPEGs.

	Runs in a subprocess so that CPU-bound PIL/fitz work doesn't block the
	async event loop.

	Args:
		file_path:    Path to the image or PDF file.
		is_pdf:       True for PDF input, False for standalone images.
		page_numbers: List of 1-based page numbers to render (PDF only).
		dpi:          Render resolution for PDFs.

	Returns:
		Dict mapping page_number → base64 string for PDFs, or {None: base64}
		for standalone images.
	"""
	results = {}
	downloaded_here = False

	try:
		if local_path is None and isinstance(file_path, str) and file_path.startswith("s3://"):
			import boto3
			from urllib.parse import urlparse
			import tempfile
			import os
			from pathlib import Path

			tmp = tempfile.NamedTemporaryFile(
				delete=False,
				suffix=Path(file_path).suffix,
				dir=temp_dir
			)
			parsed = urlparse(file_path)
			from botocore.config import Config as _Cfg
			s3 = boto3.client("s3", config=_Cfg(
				connect_timeout=10, read_timeout=60, retries={"max_attempts": 3}
			))
			resp = s3.get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))
			tmp.write(resp["Body"].read())
			tmp.flush()
			tmp.close()

			if os.path.getsize(tmp.name) == 0:
				os.remove(tmp.name)
				raise ValueError("Downloaded image is 0 bytes")
				
			local_path = tmp.name
			downloaded_here = True

		if is_pdf:
			doc = fitz.open(local_path)
			zoom = dpi / 72.0
			mat = fitz.Matrix(zoom, zoom)
			num_pages = len(doc)
	
			for p in page_numbers:
				try:
					if p <= 0 or p > num_pages:
						continue
					page = doc.load_page(p - 1)  # load_page releases after use
					pix = page.get_pixmap(matrix=mat)
					img_bytes = pix.tobytes("png")
					results[p] = base64.b64encode(img_bytes).decode("utf-8")
					pix = None  # release pixmap memory immediately
				except Exception as e:
					log_error(str(file_path), str(e), stage="convert_page", page=p)
					print(f"[WARN] Page {p} failed: {e} pdf {file_path}")
	
			doc.close()

		else:
			img = Image.open(local_path if local_path else file_path)
			img.load()

			if img.mode != "RGB":
				img = img.convert("RGB")

			buf = io.BytesIO()
			img.save(buf, format="PNG", quality=90)
			buf.seek(0)
			results[None] = base64.b64encode(buf.read()).decode("utf-8")

	except Exception as e:
		log_error(str(file_path), str(e), stage="convert_image_worker")
		print(f"[ERROR] Failed '{file_path}': {e}")
	finally:
		if downloaded_here and not is_pdf:
			import os
			try:
				os.remove(local_path)
			except Exception:
				pass

	return results


def get_pdf_num_pages_worker(file_path, local_path, temp_dir=None):
	"""
	Return the page count of a PDF and the local path used. Runs in a subprocess.
	If local_path is None and file_path is an S3 URI, downloads it to a temp file.
	
	Returns (num_pages, local_path). 
	Returns (0, None) on failure.
	"""
	created_temp = False
	try:
		if local_path is None and isinstance(file_path, str) and file_path.startswith("s3://"):
			import boto3
			from urllib.parse import urlparse
			import tempfile
			import os
			from pathlib import Path

			tmp = tempfile.NamedTemporaryFile(
				delete=False,
				suffix=".pdf",
				dir=temp_dir
			)
			parsed = urlparse(file_path)
			from botocore.config import Config as _Cfg
			s3 = boto3.client("s3", config=_Cfg(
				connect_timeout=10, read_timeout=60, retries={"max_attempts": 3}
			))
			resp = s3.get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))
			tmp.write(resp["Body"].read())
			tmp.flush()
			tmp.close()

			if os.path.getsize(tmp.name) == 0:
				os.remove(tmp.name)
				raise ValueError("Downloaded file is 0 bytes")
				
			local_path = tmp.name
			created_temp = True
		fitz.TOOLS.mupdf_display_errors(False)
		fitz.TOOLS.mupdf_display_warnings(False)

		doc = fitz.open(local_path if local_path else file_path)
		num_pages = len(doc)
		doc.close()
		# return (num_pages, local_path)
		return (num_pages, local_path if created_temp else None)

	except Exception as e:
		log_error(str(file_path), str(e), stage="get_pdf_num_pages")
		print(f"[WARN] Could not read PDF page count '{file_path}': {e}")
		return (0, None)


# ─────────────────────────────────────────────────────────────────────────────
# RATE-LIMITED REQUEST GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

async def get_request(input_requests, request_rate):
	"""
	Async generator that yields requests at a target rate (requests/sec).

	Inter-arrival times are drawn from an exponential distribution so that
	the average rate matches `request_rate` (Poisson process). When
	`request_rate` is `inf`, requests are yielded as fast as possible.
	"""
	input_requests = iter(input_requests)
	for request in input_requests:
		yield request

		if request_rate == float("inf"):
			continue

		interval = np.random.exponential(1.0 / request_rate)
		await asyncio.sleep(interval)


# ─────────────────────────────────────────────────────────────────────────────
# CORE INFERENCE ORCHESTRATOR
# ─────────────────────────────────────────────────────────────────────────────

async def infer(backend, api_url, base_url, model_id, input_requests, already_done, request_rate, max_concurrency, disable_tqdm, extra_request_body):
	"""
	Orchestrate concurrent image conversion and inference.

	Architecture overview:
	  - producer():        reads the input stream, dispatches image conversion
	                       futures to a ProcessPoolExecutor, and pushes ready
	                       requests into a bounded in-memory queue. Overflows
	                       are spilled to disk using msgpack framing.
	  - consumer_worker(): pulls requests from the RAM queue (or disk spill),
	                       calls the backend request function, and appends
	                       results to a write buffer.
	  - flush_writer():    periodically drains the write buffer to the output
	                       JSONL file.

	The RAM queue / disk spill design keeps memory usage bounded even when
	the conversion process produces images faster than the inference backend
	can consume them (e.g. when the server is under heavy load).

	Disk spill uses a binary msgpack format with a 4-byte length prefix per
	record — much faster than JSONL for round-trip serialization.
	"""

	# ── SHARED AIOHTTP SESSION ───────────────────────────────────────────────
	# A single session is shared across all consumer workers to reuse TCP
	# connections and DNS cache entries. Connection limits are set to 0 (no cap)
	# because the semaphore already enforces max_concurrency.
	shared_session = aiohttp.ClientSession(
		connector=aiohttp.TCPConnector(
			limit=0,
			limit_per_host=0,
			keepalive_timeout=30,
			enable_cleanup_closed=True,
			ttl_dns_cache=600,
		),
		timeout=aiohttp.ClientTimeout(total=6 * 3600),
		read_bufsize=10 * 1024**2,
	)
	shutdown_event = asyncio.Event()

	if backend not in ASYNC_REQUEST_FUNCS:
		raise ValueError(f"Unknown backend: {backend}")

	request_func = ASYNC_REQUEST_FUNCS[backend]
	producer_done = asyncio.Event()

	# Semaphore caps the number of in-flight inference requests.
	semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

	# Flush SGLang's KV cache before a new run to avoid stale prefixes
	# interfering with the first few requests.
	if "sglang" in backend:
		requests.post(base_url + "/flush_cache", headers=get_auth_headers())

	# ── BUFFER / EXECUTOR SIZING ─────────────────────────────────────────────
	buffer_size = getattr(args, "buffer_size", 50) or 50
	# Use at least 4 converters; scale with CPU count for heavier workloads.
	num_converters = max(4, multiprocessing.cpu_count()*2)
	process_pool = ProcessPoolExecutor(max_workers=num_converters)

	num_workers = max(1, max_concurrency or 1)

	# In-memory queue for converted images ready for inference.
	ready_queue = asyncio.Queue(maxsize=buffer_size * 2)

	# ── SPILL FILE ───────────────────────────────────────────────────────────
	# When the in-memory queue is full, converted images overflow here.
	# The file is in the same directory as the output file for locality.
	if args.spill_path:
		SPILL_PATH = args.spill_path
	else:
		out = Path(args.output_file)
		SPILL_PATH = str(out.with_name(f".{out.stem}.spill.jsonl"))

	# Always start with a fresh spill file to avoid replaying stale data.
	if os.path.exists(SPILL_PATH):
		os.remove(SPILL_PATH)
	open(SPILL_PATH, "wb").close()

	# ── SHARED COUNTERS ──────────────────────────────────────────────────────
	class Counters:
		conversion_count = 0  # Pages successfully converted
		inference_count = 0   # Responses successfully written
		conversion_buffers = 0
		inference_buffers = 0
		in_flight = 0         # Inference requests currently awaited
		spill_count = 0       # Total records written to disk spill
		spill_done = 0        # Total records drained from disk spill
		total_items = already_done  # Updated live as PDF page counts arrive

	locks = {
		"conversion": asyncio.Lock(),
		"inference": asyncio.Lock(),
		"in_flight": asyncio.Lock(),
		"total_items": asyncio.Lock(),
		"write_buffer": asyncio.Lock(),
	}

	# In-memory write buffer; flushed to disk every ~1 second by the writer task.
	write_buffer = []

	# ── PROGRESS BARS ────────────────────────────────────────────────────────
	conversion_pbar = tqdm(
		total=Counters.total_items,
		initial=already_done,
		desc="Converting",
		unit="img",
		dynamic_ncols=True,
		disable=disable_tqdm,
		position=0,
	)

	inference_pbar = tqdm(
		total=Counters.total_items,
		initial=already_done,
		desc="Inference",
		unit="req",
		dynamic_ncols=True,
		disable=disable_tqdm,
		position=1,
	)

	async def update_inference_stats():
		"""Refresh the inference progress bar postfix with current in-flight count."""
		inference_pbar.set_postfix_str(f"in-flight={Counters.in_flight}", refresh=True)

	# ── DISK SPILL HELPERS ───────────────────────────────────────────────────
	# Records are serialized with msgpack and prefixed with a 4-byte big-endian
	# length so they can be read back sequentially without a line scanner.

	spill_read_offset = 0  # Byte offset into the spill file for the next read
	spill_offset_lock = asyncio.Lock()

	async def spill_to_disk(obj: dict):
		"""Append a single record to the spill file with a length prefix."""
		data = msgpack.packb(obj, use_bin_type=True)
		header = struct.pack(">I", len(data))

		async with aiofiles.open(SPILL_PATH, "ab") as f:
			await f.write(header)
			await f.write(data)

	async def pop_from_disk():
		"""
		Read and consume the next record from the spill file.

		Advances spill_read_offset atomically so concurrent consumers don't
		read the same record. Returns None if there is nothing left to read.
		"""
		nonlocal spill_read_offset

		async with spill_offset_lock:
			offset = spill_read_offset

		async with aiofiles.open(SPILL_PATH, "rb") as f:
			await f.seek(offset)
			header = await f.read(4)
			if not header:
				return None
			size = struct.unpack(">I", header)[0]
			data = await f.read(size)

		async with spill_offset_lock:
			spill_read_offset = offset + 4 + size

		try:
			return msgpack.unpackb(data, raw=False)
		except Exception:
			return None

	# ── REQUEST BUILDER ──────────────────────────────────────────────────────

	async def build_request(req, img_base64):
		"""
		Assemble the request dict that gets passed to an async_request_* function.
		"""
		return {
			"model": model_id,
			"id": req.get("id"),
			"prompt": req.get("prompt"),
			"api_url": api_url,
			"image_data": [img_base64] if img_base64 else [None],
			"extra_request_body": extra_request_body,
		}

	# ══════════════════════════════════════════════════════════════════════════
	# PRODUCER
	# Reads the input stream, converts images in a process pool, and fills the
	# ready queue (with overflow to disk).
	# ══════════════════════════════════════════════════════════════════════════

	def bump_retry(r):
		"""Increment and return the retry counter for a request record."""
		r["_retries"] = r.get("_retries", 0) + 1
		return r["_retries"]

	async def producer():
		nonlocal process_pool

		BATCH_SIZE = 128          # Max pages to convert in a single process-pool call
		MAX_TASK_RETRIES = 3
		BACKOFF_BASE = 1.0
		pdf_local_cache = {}      # s3:// path → local temp file path
		submitted_page_ids = set()  # Tracks which page IDs have been queued to avoid duplicates on pool respawn

		loop = asyncio.get_running_loop()
		request_iter = iter(input_requests)
		request_queue = deque()   # Local deque for buffering before submitting to process pool
		pending = {}              # {future: payload} for futures currently in the process pool
		submit_times = {}         # {future: monotonic submit time} for stall detection
		max_pending = num_converters * 4
		STALL_LIMIT = 30.0       # seconds a single future may run before pool is deemed poisoned

		def drain_pending_futures():
			"""Cancel and clear all tracked futures (used on pool respawn)."""
			for fut in list(pending.keys()):
				try:
					if not fut.done():
						fut.cancel()
					else:
						_ = fut.exception()
				except Exception:
					pass
			pending.clear()
			submit_times.clear()

		async def respawn_pool(reason: str):
			"""Tear down the broken process pool and start a new one."""
			nonlocal process_pool
			print(
				"⚠ Respawning process pool",
				reason,
				"pending",
				len(pending),
				flush=True,
			)
			drain_pending_futures()
			try:
				process_pool.shutdown(wait=False, cancel_futures=True)
			except Exception:
				pass
			process_pool = ProcessPoolExecutor(max_workers=num_converters)
			await asyncio.sleep(0)

		def submit_conversion(func, *args):
			"""Submit a conversion task to the process pool. Returns None if the pool is broken."""
			nonlocal process_pool
			try:
				fut = loop.run_in_executor(process_pool, func, *args)
				pending[fut] = None
				submit_times[fut] = time.monotonic()
				return fut
			except BrokenProcessPool:
				return None

		while True:
			pool_broken = False
			
			# Periodic diagnostics to help track throughput and bottlenecks.
			if int(time.time()) % 10 == 0:
				print(
					"[PRODUCER]",
					"queue", len(request_queue),
					"pending", len(pending),
					"ready", ready_queue.qsize(),
					"spill", Counters.spill_count,
					flush=True,
			)

			# Pull more records from the generator while under capacity.
			while len(request_queue) < max_pending:
				try:
					req = next(request_iter)
					request_queue.append(req)
				except StopIteration:
					break

			# ── SUBMISSION PHASE ─────────────────────────────────────────────
			while request_queue and len(pending) < max_pending:
				req = request_queue.popleft()
				req["_retries"] = req.get("_retries", 0)

				# ── Lazy PDF: get page count first ───────────────────────────
				# PDFs are represented as a single lazy record until we know how
				# many pages they have. The _pagecount_ result expands them into
				# N per-page records that are re-queued for conversion.
				if req.get("is_pdf") and req.get("page_num") is None and req.get("pdf_lazy", True):
					try:
						file_path = req.get("file_path")

						if isinstance(file_path, str) and file_path.startswith("s3://"):
							local_path = pdf_local_cache.get(file_path, None)
						else:
							local_path = file_path

						req["local_path"] = local_path

						fut = submit_conversion(
							get_pdf_num_pages_worker,
							req.get('file_path'),
							local_path,
							args.temp_path
						)
						if fut is None:
							await respawn_pool("submit(page_count)")
							request_queue.appendleft(req)
							continue
						# Tag the future as a page-count job so the completion
						# handler knows to expand it rather than convert it.
						pending[fut] = ("_pagecount_", req)
						continue
					except Exception as e:
						log_error(str(file_path), str(e), stage="producer_lazy_pdf")
						continue

				# ── Standalone image ─────────────────────────────────────────
				if not req.get("is_pdf", False):
					file_path = req.get("file_path")
					if isinstance(file_path, str) and file_path.startswith("s3://"):
						local_path = pdf_local_cache.get(file_path, None)
					else:
						local_path = file_path
						
					req["local_path"] = local_path

					fut = submit_conversion(
						convert_image_worker,
						req.get("file_path"),
						False,
						None,
						args.pdf_dpi,
						req.get("local_path"),
						args.temp_path
					)
					if fut is None:
						await respawn_pool("submit(image)")
						request_queue.appendleft(req)
						continue
					pending[fut] = req
					continue

				# ── PDF page batch ───────────────────────────────────────────
				# Group consecutive pages of the same PDF into one conversion
				# call to amortise the cost of opening the document.
				if req.get("is_pdf") and req.get("page_num") is not None:
					file_path = req.get("local_path", req.get("file_path"))
					batch = [req]

					while request_queue and len(batch) < BATCH_SIZE:
						candidate = request_queue[0]
						candidate_path = candidate.get("local_path", candidate.get("file_path"))
						if (
							candidate_path == file_path
							and candidate.get("is_pdf")
							and candidate.get("page_num") is not None
						):
							batch.append(request_queue.popleft())
						else:
							break

					page_numbers = [r["page_num"] for r in batch]
					fut = submit_conversion(
						convert_image_worker,
						file_path,
						True,
						page_numbers,
						args.pdf_dpi,
						file_path,
						args.temp_path
					)
					if fut is None:
						await respawn_pool("submit(pdf_batch)")
						for r in reversed(batch):
							request_queue.appendleft(r)
						continue
					pending[fut] = batch
					continue

			# ── COMPLETION PHASE ─────────────────────────────────────────────
			if not pending:
				await asyncio.sleep(0.01)
				continue

			# Wait for at least one future to complete, with a timeout to avoid
			# blocking forever if all futures are stalled/dead.
			done, _ = await asyncio.wait(
				list(pending.keys()),
				return_when=asyncio.FIRST_COMPLETED,
				timeout=5.0,
			)
			if not done and pending:
				now_mono = time.monotonic()
				oldest = min((now_mono - submit_times.get(f, now_mono) for f in pending), default=0.0)
				print("[PENDING STALL]", len(pending), "oldest_age", round(oldest, 1), flush=True)

				# A single future exceeding STALL_LIMIT means a worker is hung.
				# The pool can't recover on its own (it's not "broken"), so we
				# forcibly respawn and re-queue retryable payloads.
				if oldest > STALL_LIMIT:
					stuck_items = list(pending.values())
					await respawn_pool(f"stuck_worker_timeout(oldest={oldest:.0f}s)")
					for it in stuck_items:
						if isinstance(it, list):
							for r in it:
								if bump_retry(r) <= MAX_TASK_RETRIES:
									request_queue.appendleft(r)
						elif isinstance(it, dict):
							if bump_retry(it) <= MAX_TASK_RETRIES:
								request_queue.appendleft(it)
						# ("_pagecount_", req) tuples are dropped, as on BrokenProcessPool.
					continue

			# Prune any futures that finished during the timeout gap.
			if not done:
				dead = [f for f in pending if f.done()]
				for f in dead:
					pending.pop(f, None)
					submit_times.pop(f, None)
				continue

			for fut in done:
				item = pending.pop(fut, None)
				submit_times.pop(fut, None)

				if item is None:
					continue

				try:
					result = fut.result()

				except BrokenProcessPool:
					# The entire pool is dead — cancel all pending futures and
					# re-queue retryable items before spawning a new pool.
					drain_pending_futures()
					await respawn_pool("result")

					if isinstance(item, list):
						for r in item:
							if bump_retry(r) <= MAX_TASK_RETRIES:
								request_queue.appendleft(r)
					elif isinstance(item, dict):
						if bump_retry(item) <= MAX_TASK_RETRIES:
							request_queue.appendleft(item)

					# Page-count futures are not retried on pool death because
					# the originating req would need fresh state.
					pool_broken = True
					break

				except Exception as e:
					if isinstance(item, tuple) and item[0] == "_pagecount_":
						log_error(str(item[1].get("file_path")), str(e), stage="producer_future_error_pagecount")
					elif isinstance(item, dict):
						log_error(str(item.get("file_path")), str(e), stage="producer_future_error_image")
					elif isinstance(item, list) and len(item) > 0:
						log_error(str(item[0].get("file_path")), str(e), stage="producer_future_error_batch")
					print(f"[ERROR] Process pool future failed: {e}", flush=True)
					continue

				# ── Handle _pagecount_ results ───────────────────────────────
				# Expand a lazy PDF record into N per-page records and push them
				# back onto the front of the request queue for conversion.
				if isinstance(item, tuple) and item[0] == "_pagecount_":
					_, req = item
					num_pages, new_local_path = result

					original_path = req.get("file_path")
					
					# if new_local_path is not None and original_path.startswith("s3://"):
					# 	pdf_local_cache[original_path] = new_local_path
					# 	local_path = new_local_path
					# else:
					# 	local_path = req.get("local_path", original_path)
					if new_local_path is not None:
						pdf_local_cache[original_path] = new_local_path
						local_path = new_local_path
					else:
						local_path = req.get("local_path", original_path)

					if not num_pages or num_pages <= 0:
						continue

					relative_path = req.get("relative_path", os.path.basename(req.get("file_path", "")))
					template_obj = req.get("template", req.get("prompt"))

					pages = []
					for p in range(1, num_pages + 1):
						page_id = f"{req['id']}↳{p}"
						# Skip pages already in the output file or already queued.
						if page_id in FINISHED_IDS or page_id in submitted_page_ids:
							continue
						submitted_page_ids.add(page_id)

						try:
							filled_prompt = fill_instruction(
								template_obj,
								{"id": page_id, "file_path": relative_path, "page_num": p},
								args.template_fields,
							)
						except Exception:
							filled_prompt = template_obj

						pages.append({
							"id": page_id,
							"prompt": filled_prompt,
							"file_path": local_path,
							"relative_path": relative_path,
							"is_pdf": True,
							"page_num": p,
							"pdf_lazy": False,
						})

					for r in reversed(pages):
						request_queue.appendleft(r)

					# Update the live total so progress bars show accurate counts.
					async with locks["total_items"]:
						Counters.total_items += len(pages) - 1
						conversion_pbar.total = Counters.total_items
						inference_pbar.total = Counters.total_items
						conversion_pbar.refresh()
						inference_pbar.refresh()

					continue

				# ── Handle standalone image results ──────────────────────────
				if isinstance(item, dict):
					img_b64 = result.get(None)
					if img_b64:
						req_input = await build_request(item, img_b64)
						if ready_queue.full():
							# RAM queue full → spill to disk to avoid blocking the producer.
							await spill_to_disk(req_input)
							Counters.spill_count += 1
						else:
							await ready_queue.put(req_input)

					async with locks["conversion"]:
						Counters.conversion_count += 1
						conversion_pbar.update(1)
					continue

				# ── Handle PDF page batch results ────────────────────────────
				for r in item:
					img_b64 = result.get(r["page_num"])
					if not img_b64:
						continue

					req_input = await build_request(r, img_b64)
					if ready_queue.full():
						await spill_to_disk(req_input)
						Counters.spill_count += 1
					else:
						await ready_queue.put(req_input)

					async with locks["conversion"]:
						Counters.conversion_count += 1
						conversion_pbar.update(1)

			# After a BrokenProcessPool event, loop around and retry.
			if pool_broken:
				pool_broken = False
				continue

			# ── TERMINATION CHECK ────────────────────────────────────────────
			# Only exit when both the local deque and the process pool are empty.
			if not pending and not request_queue:
				try:
					req = next(request_iter)
					request_queue.appendleft(req)
				except StopIteration:
					break  # Generator exhausted and all work is done.

		# Signal consumers that no more items will be produced.
		producer_done.set()
		conversion_pbar.close()

		# Remove any temporary local copies of S3 PDFs.
		for original_path, tmp_path in pdf_local_cache.items():
			try:
				os.remove(tmp_path)
			except Exception as e:
				log_error(str(original_path), str(e), stage="cleanup_temp")
				print(f"[WARN] Failed to remove temp file {tmp_path}: {e}")
		pdf_local_cache.clear()
		print(f"\n✓ Conversion complete: {Counters.conversion_count} total", flush=True)


	# ══════════════════════════════════════════════════════════════════════════
	# CONSUMER WORKER
	# Pulls converted images from the ready queue (or spill), calls the backend,
	# and appends results to the write buffer.
	# ══════════════════════════════════════════════════════════════════════════

	async def consumer_worker(worker_id: int):
		last_send = 0.0

		while True:
			item = None
			from_queue = False
			from_spill = False

			# Periodic diagnostics from worker 0.
			if worker_id == 0 and int(time.time()) % 10 == 0:
				print(
					"[CONSUMER]",
					"ready", ready_queue.qsize(),
					"spill", Counters.spill_done, "/", Counters.spill_count,
					"inflight", Counters.in_flight,
					flush=True,
				)

			# 1. Try the in-memory queue first (non-blocking).
			try:
				item = ready_queue.get_nowait()
				from_queue = True
			except asyncio.QueueEmpty:
				pass

			# 2. Try the disk spill if the queue was empty.
			if item is None:
				item = await pop_from_disk()
				if item is not None:
					from_spill = True

			# 3. Nothing available — check if we should exit or wait.
			if item is None:
				if shutdown_event.is_set():
					# Double-check before exiting to catch any last-minute additions.
					print(f"[WORKER {worker_id}] shutdown check", flush=True)
					await asyncio.sleep(0.05)
					try:
						item = ready_queue.get_nowait()
						from_queue = True
					except asyncio.QueueEmpty:
						pass

					if item is None:
						item = await pop_from_disk()
						if item is not None:
							from_spill = True

					if item is None:
						print(f"[WORKER {worker_id}] exit", flush=True)
						break  # Truly nothing left; exit cleanly.
				else:
					# Producer is still running; yield and retry.
					await asyncio.sleep(0.01)
					continue

			# 4. Process the item.
			acquired_semaphore = False
			MAX_RETRIES = 3
			retry_delay = 0.5
			result = None

			try:
				# Honour the requested rate limit with exponential inter-arrival.
				if request_rate != float("inf") and request_rate > 0:
					interval = float(np.random.exponential(1.0 / request_rate))
					now = asyncio.get_event_loop().time()
					sleep_time = max(0.0, last_send + interval - now)
					if sleep_time > 0:
						await asyncio.sleep(sleep_time)
					last_send = asyncio.get_event_loop().time()

				# Acquire semaphore before incrementing in_flight so the counter
				# accurately reflects requests that are actually in progress.
				if semaphore is not None:
					await semaphore.acquire()
					acquired_semaphore = True

				async with locks["in_flight"]:
					Counters.in_flight += 1
				await update_inference_stats()

				# Retry loop for transient inference failures.
				for attempt in range(MAX_RETRIES):
					try:
						result = await request_func(
							request_func_input=item,
							pbar=None,
							async_session=shared_session,
						)
						if result and result.get("generated_text", "").strip():
							break
					except Exception as e:
						if attempt == MAX_RETRIES - 1:
							log_error(item.get("id", "unknown"), str(e), stage="inference_retries_exhausted")
							print(f"[ERROR] Failed after retries: {e}")
						else:
							await asyncio.sleep(retry_delay * (attempt + 1))

				if result and result.get("generated_text", "").strip():
					line = json.dumps(result, ensure_ascii=False) + "\n"
					async with locks["write_buffer"]:
						write_buffer.append(line)
					# Track in memory so the producer can skip already-done IDs.
					FINISHED_IDS.add(result["id"])
					async with locks["inference"]:
						Counters.inference_count += 1
						inference_pbar.update(1)
						await update_inference_stats()

			except Exception as e:
				log_error(item.get("id", "unknown") if item else "unknown", str(e), stage="consumer_worker")
				print(f"[worker-{worker_id}] inference error: {e}", flush=True)

			finally:
				async with locks["in_flight"]:
					Counters.in_flight -= 1
				await update_inference_stats()

				if acquired_semaphore:
					try:
						semaphore.release()
					except Exception:
						pass

				# Notify the queue that this slot has been consumed.
				if from_queue:
					ready_queue.task_done()

				if from_spill:
					Counters.spill_done += 1


	# ══════════════════════════════════════════════════════════════════════════
	# PERIODIC WRITER
	# Drains the in-memory write buffer to disk every second to reduce write
	# syscall overhead while keeping output latency low.
	# ══════════════════════════════════════════════════════════════════════════

	async def flush_writer_periodically(f):
		try:
			while True:
				await asyncio.sleep(1.0)
				if f.closed:
					return

				async with locks["write_buffer"]:
					if write_buffer:
						await f.write("".join(write_buffer))
						await f.flush()
						write_buffer.clear()

		except asyncio.CancelledError:
			# Flush any remaining buffered lines before the task is torn down.
			if not f.closed:
				async with locks["write_buffer"]:
					if write_buffer:
						await f.write("".join(write_buffer))
						await f.flush()
						write_buffer.clear()
			raise


	# ══════════════════════════════════════════════════════════════════════════
	# MAIN ORCHESTRATION
	# Start all tasks, then drain queues in order before shutting down.
	# ══════════════════════════════════════════════════════════════════════════

	async with aiofiles.open(args.output_file, "a", encoding="utf-8") as f:
		flush_task = asyncio.create_task(flush_writer_periodically(f))
		producer_task = asyncio.create_task(producer())
		consumer_tasks = [
			asyncio.create_task(consumer_worker(i)) for i in range(num_workers)
		]

		# Wait for the producer to finish submitting all conversion jobs.
		await producer_task
		print("WAITING ready_queue.join", flush=True)

		# Wait for the in-memory queue to be fully drained by consumers.
		await ready_queue.join()
		print("READY_QUEUE DONE", flush=True)

		# Wait for all spilled items to be consumed.
		print("WAITING spill drain", flush=True)
		while Counters.spill_done < Counters.spill_count:
			print("[SPILL WAIT]", Counters.spill_done, Counters.spill_count, flush=True)
			await asyncio.sleep(0.05)

		# Signal workers to exit after draining remaining work.
		shutdown_event.set()
		await asyncio.gather(*consumer_tasks)

		# Cancel the writer task, which will flush any remaining buffered lines.
		flush_task.cancel()
		try:
			await flush_task
		except asyncio.CancelledError:
			pass

		await shared_session.close()
		await asyncio.sleep(0.25)  # Allow lingering TCP connections to close.

		print(f"✅ All {num_workers} workers exited cleanly", flush=True)

		inference_pbar.close()
		print(
			f"\n✓ Inference complete: "
			f"{Counters.inference_count} total "
			f"({Counters.inference_buffers} buffers)"
		)

	# Shut down the process pool after all async work is done.
	loop = asyncio.get_running_loop()
	await loop.run_in_executor(None, process_pool.shutdown, True)


# ─────────────────────────────────────────────────────────────────────────────
# TOKENIZER / CHAT TEMPLATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def get_chat_template(model_path):
	"""
	Load the Jinja2 chat template from a HuggingFace tokenizer config.

	Returns a compiled Template object if one is found, or None if the model
	doesn't define a chat template (e.g. base models without instruction tuning).
	"""
	try:
		logging.set_verbosity_error()
		tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
		template_str = tokenizer.init_kwargs.get('chat_template', None)

		if template_str:
			return Template(template_str)
		return None
	except Exception as e:
		print(f"Failed to load tokenizer config with error: {e}")
		return None


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM LIMITS
# ─────────────────────────────────────────────────────────────────────────────

def set_ulimit(target_soft_limit=65535):
	"""
	Raise the open-file-descriptor limit to accommodate large numbers of
	concurrent aiohttp connections. Does nothing if the current limit is
	already at or above the target, and prints a warning if the raise fails.
	"""
	resource_type = resource.RLIMIT_NOFILE
	current_soft, current_hard = resource.getrlimit(resource_type)

	if current_soft < target_soft_limit:
		try:
			resource.setrlimit(resource_type, (target_soft_limit, current_hard))
		except ValueError as e:
			print(f"Fail to set RLIMIT_NOFILE: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(args_):
	"""
	Top-level entry point: resolve runtime configuration, load inputs,
	and launch the async inference loop.
	"""
	global args
	args = args_

	# Set up error logger in the same directory as the output file.
	out_p = Path(args.output_file)
	error_log_path = str(out_p.with_name(out_p.stem + ".errors.jsonl"))
	setup_error_logger(error_log_path)

	set_ulimit()

	extra_request_body = {}
	if args.extra_request_body:
		extra_request_body = json.loads(args.extra_request_body)

	# Assign default ports for backends that don't specify one.
	if args.port is None:
		args.port = {
			"sglang": 30000,
			"sglang-native": 30000,
			"sglang-oai": 30000,
			"lmdeploy": 23333,
			"vllm": 8000,
			"trt": 8000,
		}.get(args.backend, 30000)

	model_url = (
		f"{args.base_url}/v1/models"
		if args.base_url
		else f"http://{args.host}:{args.port}/v1/models"
	)

	# Construct the inference API URL based on backend type.
	if args.backend in ["sglang", "sglang-native"]:
		api_url = (
			f"{args.base_url}/generate"
			if args.base_url
			else f"http://{args.host}:{args.port}/generate"
		)
	elif args.backend in ["sglang-oai", "vllm", "lmdeploy"]:
		api_url = (
			f"{args.base_url}/v1/completions"
			if args.base_url
			else f"http://{args.host}:{args.port}/v1/completions"
		)
	elif args.backend in ["sglang-oai-chat", "vllm-chat", "lmdeploy-chat"]:
		api_url = (
			f"{args.base_url}/v1/chat/completions"
			if args.base_url
			else f"http://{args.host}:{args.port}/v1/chat/completions"
		)
	elif args.backend == "trt":
		api_url = (
			f"{args.base_url}/v2/models/ensemble/generate_stream"
			if args.base_url
			else f"http://{args.host}:{args.port}/v2/models/ensemble/generate_stream"
		)
		if args.model is None:
			print("Please provide a model using `--model` when using `trt` backend.")
			sys.exit(1)

	base_url = (
		f"http://{args.host}:{args.port}"
		if args.base_url is None
		else args.base_url
	)

	# Auto-discover model name if not specified.
	if args.model is None:
		try:
			response = requests.get(model_url, headers=get_auth_headers())
			model_list = response.json().get("data", [])
			args.model = model_list[0]["id"] if model_list else None
		except Exception as e:
			print(f"Failed to fetch model from {model_url}. Error: {e}")
			print("Please specify the correct host and port using `--host` and `--port`.")
			sys.exit(1)

	if args.model is None:
		print("No model specified or found. Please provide a model using `--model`.")
		sys.exit(1)

	args.chat_template = get_chat_template(args.model)

	print(f"\nParsed Arguments:")
	for k, v in vars(args).items():
		print(f"{k:20} {v}")
	print("\n")

	backend = args.backend
	model_id = args.model
	input_requests, already_done = prepare_prompts(args)

	return asyncio.run(
		infer(
			backend=backend,
			api_url=api_url,
			base_url=base_url,
			model_id=model_id,
			input_requests=input_requests,
			already_done=already_done,
			request_rate=args.request_rate,
			max_concurrency=args.max_concurrency,
			disable_tqdm=args.disable_tqdm,
			extra_request_body=extra_request_body,
		)
	)


if __name__ == "__main__":
	start_time = time.perf_counter()
	parser = ArgumentParser()

	# ── Input / Output ───────────────────────────────────────────────────────
	parser.add_argument("--input-path", type=str, required=True,
		help="Path to input data file/folder (.jsonl, .parquet, .pdf, image files, or directory)")
	parser.add_argument("--output-file", type=str, required=True,
		help="Output JSONL file name")
	parser.add_argument("--instruction-path", type=str, required=True,
		help="YAML file with instruction/task templates")
	parser.add_argument("--task", type=str, required=True,
		help="Task/template key from the YAML file (dot-separated, e.g. 'ocr.default')")
	parser.add_argument("--template-fields", type=str, nargs="+",
		help="Fields from record to fill into template (dot-separated paths, e.g. 'id file_path page_num')")
	parser.add_argument("--spill-path", type=str, default=None,
		help="Path for disk spill file (default: same dir as output file)")
	parser.add_argument("--temp-path", type=str, required=True, default=None,
		help="Directory for temp PDF files (default: system /tmp)")
	parser.add_argument("--skip-pages", action="store_true", default=False,
		help="When resuming, mark an entire document as done if any of its pages are present")

	# ── Model & Backend ──────────────────────────────────────────────────────
	parser.add_argument("--backend", type=str, choices=list(ASYNC_REQUEST_FUNCS.keys()), default="sglang",
		help="Backend inference engine")
	parser.add_argument("--model", type=str,
		help="Model name or path (default: auto-discover from server)")
	parser.add_argument("--tokenizer", type=str,
		help="Tokenizer name or path (default: use model config)")

	# ── Server / Connection ──────────────────────────────────────────────────
	parser.add_argument("--base-url", type=str, default=None,
		help="Server or API base URL (overrides --host / --port)")
	parser.add_argument("--host", type=str, default="0.0.0.0",
		help="Host address")
	parser.add_argument("--port", type=int,
		help="Port (default depends on backend)")

	# ── Generation Parameters ─────────────────────────────────────────────────
	parser.add_argument("--max-new-tokens", type=int,
		help="Maximum number of new tokens to generate")
	parser.add_argument("--extra-request-body", type=str, metavar='{"key":"value"}',
		help="Extra JSON merged into every request payload (e.g. temperature, top_p)")
	parser.add_argument("--apply-chat-template", action="store_true",
		help="Apply the model's chat template to prompts before sending")
	parser.add_argument("--ignore-eos", action="store_true", default=False,
		help="Ignore EOS tokens (force the model to generate max_new_tokens tokens)")

	# ── Request Control ───────────────────────────────────────────────────────
	parser.add_argument("--request-rate", type=float, default=float("inf"),
		help="Requests per second (default: inf = send as fast as possible)")
	parser.add_argument("--max-concurrency", type=int, default=100,
		help="Max simultaneous in-flight inference requests")
	parser.add_argument("--buffer-size", type=int, default=50,
		help="In-memory prefetch queue size (increase for higher throughput)")
	parser.add_argument("--enable-stream", action="store_true",
		help="Use streaming mode for all inference requests")

	# ── PDF Options ───────────────────────────────────────────────────────────
	parser.add_argument("--pdf-dpi", type=int, default=200,
		help="DPI for PDF-to-image conversion (higher = sharper but larger images)")

	# ── File Detection ────────────────────────────────────────────────────────
	parser.add_argument("--no-magic-bytes", action="store_true", default=False,
		help="Use file extensions for type detection instead of magic-byte sniffing (faster for S3, less accurate)")

	# ── UX ────────────────────────────────────────────────────────────────────
	parser.add_argument("--disable-tqdm", action="store_true",
		help="Disable tqdm progress bars")

	args = parser.parse_args()
	run_inference(args)
	duration = time.perf_counter() - start_time
	print(f"\nInference completed in {timedelta(seconds=int(duration))}.")
