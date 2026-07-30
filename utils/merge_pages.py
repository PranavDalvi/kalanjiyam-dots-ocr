import json
import os
import re
import argparse
import hashlib
from collections import defaultdict
from multiprocessing import Pool

# =========================
# Tuned constants (for your machine)
# =========================

NUM_SHARDS = 64
MERGE_WORKERS = 12
IO_BUFFER = 1024 * 1024  # 1MB buffers

PAGE_RE = re.compile(r"\d+")

# =========================
# Helpers
# =========================

def extract_doc_and_page(raw_id):
    """
    Splits ID like: 'ABC123↳4'
    Returns: (doc_id, page_number)
    """
    if "↳" not in raw_id:
        return raw_id, 0

    doc_id, page = raw_id.split("↳", 1)
    m = PAGE_RE.search(page)
    return doc_id, int(m.group()) if m else 0


def shard_for_doc(doc_id):
    """
    Stable hash → shard index
    """
    h = hashlib.blake2b(doc_id.encode("utf-8"), digest_size=4).hexdigest()
    return int(h, 16) % NUM_SHARDS

# =========================
# Phase 1: Sharding
# =========================

def shard_jsonl(input_file, shard_dir):
    os.makedirs(shard_dir, exist_ok=True)

    shard_paths = [
        os.path.join(shard_dir, f"shard_{i}.jsonl")
        for i in range(NUM_SHARDS)
    ]

    shard_files = [
        open(p, "w", encoding="utf-8", buffering=IO_BUFFER)
        for p in shard_paths
    ]

    with open(input_file, "r", encoding="utf-8", buffering=IO_BUFFER) as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)
            raw_id = record.get("id", "")
            doc_id, _ = extract_doc_and_page(raw_id)

            shard_idx = shard_for_doc(doc_id)
            shard_files[shard_idx].write(line)

    for f in shard_files:
        f.close()

# =========================
# Phase 2: Merge one shard
# =========================

def merge_shard(args):
    shard_path, output_path = args
    documents = defaultdict(list)

    with open(shard_path, "r", encoding="utf-8", buffering=IO_BUFFER) as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)
            raw_id = record.get("id", "")
            text = record.get("text", "")

            doc_id, page = extract_doc_and_page(raw_id)
            documents[doc_id].append((page, text))

    with open(output_path, "w", encoding="utf-8", buffering=IO_BUFFER) as out:
        for doc_id, pages in documents.items():
            pages.sort(key=lambda x: x[0])
            merged_text = "\n\n".join(t for _, t in pages)
            out.write(json.dumps(
                {"id": doc_id, "text": merged_text},
                ensure_ascii=False
            ) + "\n")

# =========================
# Phase 3: Parallel merge + final concat
# =========================

def merge_all_shards(shard_dir, output_file):
    shard_files = [
        os.path.join(shard_dir, f)
        for f in os.listdir(shard_dir)
        if f.startswith("shard_") and f.endswith(".jsonl")
    ]

    shard_files.sort()

    merged_paths = [
        os.path.join(shard_dir, f"merged_{i}.jsonl")
        for i in range(len(shard_files))
    ]

    tasks = list(zip(shard_files, merged_paths))

    with Pool(processes=MERGE_WORKERS, maxtasksperchild=1) as pool:
        pool.map(merge_shard, tasks)

    with open(output_file, "w", encoding="utf-8", buffering=IO_BUFFER) as final:
        for path in merged_paths:
            with open(path, "r", encoding="utf-8", buffering=IO_BUFFER) as f:
                for line in f:
                    final.write(line)

# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser(
        description="Merge paginated JSONL documents using ↳ page markers (high-performance)"
    )
    parser.add_argument("-i", "--input", required=True, help="Input JSONL file")
    parser.add_argument("-o", "--output", required=True, help="Output merged JSONL file")
    parser.add_argument("--workdir", default="shards", help="Temporary shard directory")

    args = parser.parse_args()

    shard_jsonl(args.input, args.workdir)
    merge_all_shards(args.workdir, args.output)

if __name__ == "__main__":
    main()
