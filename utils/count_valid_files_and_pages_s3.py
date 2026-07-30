#!/usr/bin/env python3
import argparse
import io
import os
import sys
import filetype
import boto3
from tqdm import tqdm
from smart_open import open as sopen
from pypdf import PdfReader
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore
from botocore.config import Config

# ==============================
# MAX-AGGRESSIVE SETTINGS
# ==============================
SCAN_WORKERS = 192      # header inspection threads
PDF_WORKERS = 96        # pdf processing threads

SCAN_SEM = Semaphore(SCAN_WORKERS)
PDF_SEM = Semaphore(PDF_WORKERS)

# Tune boto3 for extreme concurrency
config = Config(
    max_pool_connections=500,
    retries={"max_attempts": 10}
)

print("[INFO] Initializing high-concurrency S3 client")
s3 = boto3.client("s3", config=config)

# Try to use s3fs for fast walking
try:
    import s3fs
    FS = s3fs.S3FileSystem(use_listings_cache=True)
    USE_S3FS = True
    print("[INFO] Using s3fs for fast S3 walk")
except Exception as e:
    USE_S3FS = False
    print(f"[WARN] s3fs unavailable, using boto3 paginator: {e}")


# ----------------------------
# S3 streaming iterators
# ----------------------------
def iter_s3_keys(bucket, prefix=""):
    """Yield S3 object keys one-by-one."""
    print(f"[INFO] Streaming keys from s3://{bucket}/{prefix or '(root)'}")

    if USE_S3FS:
        root_path = f"{bucket}/{prefix}".rstrip("/")
        try:
            for root, _, files in FS.walk(root_path):
                for name in files:
                    key = f"{root}/{name}".replace(f"{bucket}/", "", 1)
                    if not os.path.basename(key).startswith("."):
                        yield key
        except Exception as e:
            print(f"[ERROR] s3fs walk failed: {e}")
            return
    else:
        paginator = s3.get_paginator("list_objects_v2")
        try:
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    if not os.path.basename(key).startswith("."):
                        yield key
        except Exception as e:
            print(f"[ERROR] boto3 walk failed: {e}")
            return


# ----------------------------
# File validation (HEADER READ)
# ----------------------------
def is_valid_file_s3(bucket, key):
    """Detect file type by reading a small header."""
    try:
        with sopen(f"s3://{bucket}/{key}", "rb") as f:
            head = f.read(2048)  # 2KB is enough

        kind = filetype.guess(head)
        if kind and kind.extension in {"pdf", "jpg", "jpeg", "png"}:
            return kind.extension

    except Exception as e:
        print(f"[ERROR] Failed to inspect s3://{bucket}/{key}: {e}")

    return None


def inspect_key(bucket, key):
    """Worker for parallel scanning."""
    with SCAN_SEM:
        ext = is_valid_file_s3(bucket, key)
    return key, ext


# ----------------------------
# PDF processing (STREAMING)
# ----------------------------
def count_pdf_pages_s3(bucket, key):
    """Count pages in a single PDF (streaming)."""
    with PDF_SEM:
        try:
            with sopen(f"s3://{bucket}/{key}", "rb") as f:
                reader = PdfReader(f)  # STREAM instead of f.read()
                return len(reader.pages)
        except Exception as e:
            print(f"[ERROR] Failed reading PDF s3://{bucket}/{key}: {e}")
            return 0

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="MAX-AGGRESSIVE: Stream S3 files, validate types, and count PDF pages.")

    group = parser.add_mutually_exclusive_group(required=True)

    # OPTION A: single s3:// path
    group.add_argument("--s3-path", help="Full S3 path, e.g. s3://my-bucket/path/to/files")

    # OPTION B: bucket + prefix
    group.add_argument("--bucket", help="S3 bucket name (use with --prefix)")
    parser.add_argument("--prefix", default="", help="Prefix inside the bucket (optional, use with --bucket)")

    args = parser.parse_args()


    def parse_s3_path(path: str):
        path = path.replace("s3://", "", 1)
        parts = path.split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        return bucket, prefix


    if args.s3_path:
        bucket, prefix = parse_s3_path(args.s3_path)
    else:
        bucket = args.bucket
        prefix = args.prefix or ""

    print("[INFO] Starting scan (MAX AGGRESSIVE MODE)")
    print(f"[INFO] Bucket: {bucket}")
    print(f"[INFO] Prefix: {prefix or '(root)'}")
    print(f"[INFO] Scan workers: {SCAN_WORKERS}")
    print(f"[INFO] PDF workers: {PDF_WORKERS}")

    total_files = 0
    pdf_keys = []

    # -------- Phase 1: PARALLEL Stream + validate --------
    print("[INFO] Parallel scanning S3 objects...")

    with ThreadPoolExecutor(max_workers=SCAN_WORKERS) as executor:
        futures = {}
        with tqdm(desc="Scanning S3 objects", unit="files", dynamic_ncols=True) as pbar:

            for key in iter_s3_keys(bucket, prefix):
                total_files += 1
                futures[executor.submit(inspect_key, bucket, key)] = key

                # Limit in-flight futures to avoid memory explosion
                if len(futures) >= SCAN_WORKERS * 4:
                    for fut in as_completed(futures):
                        k, ext = fut.result()
                        if ext == "pdf":
                            pdf_keys.append(k)
                        pbar.update(1)
                        pbar.set_postfix({"valid_so_far": f"{len(pdf_keys):,}"})
                        del futures[fut]
                        break

            # Drain remaining futures
            for fut in as_completed(futures):
                k, ext = fut.result()
                if ext == "pdf":
                    pdf_keys.append(k)
                pbar.update(1)

    total_pdfs = len(pdf_keys)

    print(f"[INFO] Files found: {total_files}")
    print(f"[INFO] PDFs found: {total_pdfs}")

    if not pdf_keys:
        print("[INFO] No PDFs found — exiting")
        return

    # -------- Phase 2: PARALLEL PDF page counting --------
    print("[INFO] Counting PDF pages (parallel, streaming)")

    total_pages = 0

    with ThreadPoolExecutor(max_workers=PDF_WORKERS) as executor:
        with tqdm(desc="Counting PDF pages", unit="pdf", dynamic_ncols=True) as pbar:
            futures = {
                executor.submit(count_pdf_pages_s3, bucket, key): key
                for key in pdf_keys
            }

            for future in as_completed(futures):
                try:
                    total_pages += future.result()
                except Exception as e:
                    key = futures[future]
                    print(f"[ERROR] Worker failed for {key}: {e}")
                pbar.update(1)
                pbar.set_postfix({"pages_so_far": f"{total_pages:,}"})

    print(f"[INFO] Files found: {total_files}")
    print(f"[INFO] PDFs found: {total_pdfs}")
    print(f"[INFO] Total PDF pages: {total_pages}")
    print("[INFO] Done")


if __name__ == "__main__":
    main()
