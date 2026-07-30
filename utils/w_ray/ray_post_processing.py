#!/usr/bin/env python3

import argparse
import json
import time
import os
import mmap
import glob
import shutil
from typing import Dict, Any, List, Tuple, Optional


import ray
from ray.util import as_completed
from tqdm import tqdm

def parse_args():
	parser = argparse.ArgumentParser(description="Distributed JSONL processor using Ray")
	parser.add_argument("-i", "--input", required=True, help="Input JSONL file")
	parser.add_argument("-o", "--output", required=True, help="Output directory")
	parser.add_argument("--ray-address", default="auto", help="Ray head address")
	parser.add_argument("--lines-per-chunk", type=int, default=0, help="Override auto chunk sizing")
	return parser.parse_args()


def count_lines(path: str) -> int:
	"""Fast line count using mmap."""
	with open(path, "rb") as f:
		mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
		return mm.read().count(b"\n")


def auto_lines_per_chunk(
	total_lines: int,
	total_cpus: int,
	tasks_per_cpu: int = 3,
	min_lines: int = 50_000,
	max_lines: int = 500_000,
) -> int:
	"""Automatically determine chunk size based on cluster capacity."""
	target_tasks = max(total_cpus * tasks_per_cpu, 1)
	raw = max(total_lines // target_tasks, 1)
	return max(min(raw, max_lines), min_lines)


def make_line_chunks(path: str, lines_per_chunk: int) -> List[Tuple[str, int, int]]:
	"""
	Produce (path, start_line, end_line) tuples.
	Lines are 0-based, end_line is exclusive.
	"""
	chunks = []
	start = 0

	with open(path, "r") as f:
		for i, _ in enumerate(f, 1):
			if i % lines_per_chunk == 0:
				chunks.append((path, start, i))
				start = i
		if start < i:
			chunks.append((path, start, i))
	return chunks


def process_line(line: str) -> Optional[Dict[str, Any]]:
    """
    Heavy per-row processing.

    - Keeps the full original JSON record
    - Adds:
        - extracted_text
        - word_count
    - Returns None for invalid rows
    """

    # 1) Parse outer JSON
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return None

    generated_text = record.get("generated_text")

    if not isinstance(generated_text, str):
        return None

    # 2) Parse generated_text JSON
    try:
        items = json.loads(generated_text)
    except Exception:
        return None

    if not isinstance(items, list):
        return None

    # 3) Extract "text" fields
    texts = []

    for item in items:
        if not isinstance(item, dict):
            continue

        # Skip headers / footers
        category = item.get("category")
        if category in {"Page-header", "Page-footer"}:
            continue

        text = item.get("text")
        if isinstance(text, str):
            texts.append(text)

    # 4) Merge text + word count
    if texts:
        merged_text = "\n".join(texts)
        word_count = sum(len(t.split()) for t in texts)
    else:
        merged_text = ""
        word_count = 0

    # 5) Augment original record
    record["extracted_text"] = merged_text
    record["word_count"] = word_count

    return record


@ray.remote(num_cpus=1)
def process_chunk(path: str, start_line: int, end_line: int, out_dir: str) -> str:
	"""
	Process a slice of the JSONL file and write a .part shard.
	"""
	part_dir = os.path.join(out_dir, ".part")
	os.makedirs(part_dir, exist_ok=True)

	out_path = os.path.join(part_dir, f"{start_line}_{end_line}.jsonl")

	with open(path, "r") as infile, open(out_path, "w") as outfile:
		for i, line in enumerate(infile):
			if i < start_line:
				continue
			if i >= end_line:
				break

			result = process_line(line)
			if result is not None:
				outfile.write(json.dumps(result) + "\n")

	return out_path


def merge_parts(output_dir: str, final_name: str = "final.jsonl") -> None:
	"""
	Merge all .part/*.jsonl files into a single final JSONL file
	and remove intermediate shards.
	"""
	part_dir = os.path.join(output_dir, ".part")
	final_path = os.path.join(output_dir, final_name)

	part_files = sorted(glob.glob(os.path.join(part_dir, "*.jsonl")))

	if not part_files:
		raise RuntimeError("No part files found to merge")

	print(f"🔗 Merging {len(part_files)} shards → {final_path}")

	with open(final_path, "w") as outfile:
		for path in part_files:
			with open(path, "r") as infile:
				for line in infile:
					outfile.write(line)

	shutil.rmtree(part_dir)
	print("🧹 Removed intermediate .part directory")



def main():
	start_total = time.perf_counter()

	args = parse_args()

	# ---------------- Phase 0 ----------------
	print(f"\nPhase 0: Initializing Ray")
	t0 = time.perf_counter()

	ray.init(address=args.ray_address)

	t1 = time.perf_counter()
	print(f"Phase 0 complete ({t1 - t0:.2f}s)")

	# ---------------- Phase 1 ----------------
	print(f"\nPhase 1: Counting input lines")
	t0 = time.perf_counter()

	total_lines = count_lines(args.input)
	print(f"   Total lines: {total_lines:,}")

	t1 = time.perf_counter()
	print(f"Phase 1 complete ({t1 - t0:.2f}s)")

	# ---------------- Phase 2 ----------------
	print(f"\nPhase 2: Inspecting cluster resources & selecting chunk size")
	t0 = time.perf_counter()

	cluster_resources = ray.cluster_resources()
	total_cpus = int(cluster_resources.get("CPU", 1))
	print(f"🧠 Cluster CPUs: {total_cpus}")

	if args.lines_per_chunk > 0:
		lines_per_chunk = args.lines_per_chunk
		print(f"📦 Using user-defined lines_per_chunk: {lines_per_chunk:,}")
	else:
		lines_per_chunk = auto_lines_per_chunk(
			total_lines=total_lines,
			total_cpus=total_cpus,
		)
		print(f"📦 Auto-selected lines_per_chunk: {lines_per_chunk:,}")

	chunks = make_line_chunks(args.input, lines_per_chunk)
	print(f"🚀 Prepared {len(chunks)} chunks")

	t1 = time.perf_counter()
	print(f"Phase 2 complete ({t1 - t0:.2f}s)")

	# ---------------- Phase 3 ----------------
	print(f"\nPhase 3: Submitting Ray tasks & processing chunks")
	t0 = time.perf_counter()

	futures = [
		process_chunk.remote(path, start, end, args.output)
		for path, start, end in chunks
	]

	for _ in tqdm(as_completed(futures), total=len(futures)):
		pass

	t1 = time.perf_counter()
	print(f"Phase 3 complete ({t1 - t0:.2f}s)")

	# ---------------- Phase 4 ----------------
	print(f"\nPhase 4: Merging output shards")
	t0 = time.perf_counter()

	input_base = os.path.splitext(os.path.basename(args.input))[0]
	final_name = f"{input_base}_filtered.jsonl"
	merge_parts(args.output, final_name)

	t1 = time.perf_counter()
	print(f"Phase 4 complete ({t1 - t0:.2f}s)")

	# ---------------- Total ----------------
	end_total = time.perf_counter()
	print(f"\n🎉 All phases complete — final output ready")
	print(f"⏱️ Total runtime: {end_total - start_total:.2f}s")

if __name__ == "__main__":
	main()
