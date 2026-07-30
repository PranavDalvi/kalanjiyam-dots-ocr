#!/usr/bin/env python3

import argparse
import json
import os
import mmap
import glob
import shutil
from typing import Dict, Any, List, Tuple

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


def process_line(record: Dict[str, Any]) -> Dict[str, Any]:
	"""
	Return only valid records (generated_text is valid JSON list or dict).
	Invalid records are skipped by returning None.
	"""
	gen = record.get("generated_text")

	if not isinstance(gen, str):
		return None  # skip invalid

	try:
		parsed = json.loads(gen)
		if isinstance(parsed, (list, dict)):
			return record  # valid → keep full record
	except json.JSONDecodeError:
		return None  # skip invalid

	return None


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

			try:
				record = json.loads(line)
			except json.JSONDecodeError:
				outfile.write(json.dumps({
					"id": None,
					"error": "invalid_outer_json"
				}) + "\n")
				continue

			result = process_line(record)
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

def log_phase(phase: int, message: str) -> None:
	print(f"\n📌 Phase {phase}: {message}")
def log_phase_done(phase: int) -> None:
	print(f"✅ Phase {phase} complete")


def main():
	args = parse_args()

	log_phase(0, "Initializing Ray")
	ray.init(address=args.ray_address)
	log_phase_done(0)

	log_phase(1, "Counting input lines")
	total_lines = count_lines(args.input)
	print(f"   Total lines: {total_lines:,}")
	log_phase_done(1)

	log_phase(2, "Inspecting cluster resources & selecting chunk size")
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
	log_phase_done(2)

	log_phase(3, "Submitting Ray tasks & processing chunks")
	futures = [
		process_chunk.remote(path, start, end, args.output)
		for path, start, end in chunks
	]

	for _ in tqdm(as_completed(futures), total=len(futures)):
		pass
	log_phase_done(3)

	log_phase(4, "Merging output shards")
	merge_parts(args.output)
	log_phase_done(4)

	print("\n🎉 All phases complete — final output ready")



if __name__ == "__main__":
	main()
