import json
import os
import sys
import hashlib
import mmap
import argparse
from multiprocessing import Pool, cpu_count
from pathlib import Path
from tqdm import tqdm

SHARDS = max(8, cpu_count() * 2)


# ----------------------------
# Utilities
# ----------------------------

def count_lines(path: Path) -> int:
	with path.open("r+b") as f:
		mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
		return mm.read().count(b"\n")


def split_id(id_str: str):
	doc, page = id_str.rsplit("↳", 1)
	return doc, page


def canonical_doc_id(doc_id: str):
	return doc_id[:-4] if doc_id.endswith(".pdf") else doc_id


def shard_for(key: str):
	h = hashlib.md5(key.encode("utf-8")).hexdigest()
	return int(h, 16) % SHARDS


# ----------------------------
# Phase 1: Map
# ----------------------------

def map_phase(input_path: Path, tmp_dir: Path, total_lines: int):
	tmp_dir.mkdir(exist_ok=True)

	files = [
		open(tmp_dir / f"shard_{i}.jsonl", "w")
		for i in range(SHARDS)
	]

	with input_path.open("r") as f:
		for line in tqdm(f, total=total_lines, desc="Phase 1 | Sharding"):
			obj = json.loads(line)

			doc_id, page = split_id(obj["id"])
			canon_doc = canonical_doc_id(doc_id)
			canon_key = f"{canon_doc}↳{page}"

			shard = shard_for(canon_key)
			files[shard].write(json.dumps(obj, ensure_ascii=False) + "\n")

	for fh in files:
		fh.close()


# ----------------------------
# Phase 2: Reduce
# ----------------------------

def reduce_shard(args):
	shard_file, output_file = args
	seen = set()
	written = 0

	with shard_file.open("r") as f, output_file.open("w") as out:
		for line in f:
			obj = json.loads(line)

			doc_id, page = split_id(obj["id"])
			canon_doc = canonical_doc_id(doc_id)
			canon_id = f"{canon_doc}↳{page}"

			if canon_id in seen:
				continue

			obj["id"] = canon_id
			out.write(json.dumps(obj, ensure_ascii=False) + "\n")
			seen.add(canon_id)
			written += 1

	return written


# ----------------------------
# Driver
# ----------------------------

def main(input_path: Path, output_path: Path):
	tmp_dir = Path("_shards")
	out_dir = Path("_reduced")

	print("▶ Counting lines")
	total_lines = count_lines(input_path)
	print(f"  {total_lines:,} lines")

	print("▶ Phase 1: Map")
	map_phase(input_path, tmp_dir, total_lines)

	print("▶ Phase 2: Reduce")
	out_dir.mkdir(exist_ok=True)

	args = []
	for i in range(SHARDS):
		args.append((
			tmp_dir / f"shard_{i}.jsonl",
			out_dir / f"part_{i}.jsonl",
		))

	with Pool(cpu_count()) as pool:
		results = list(
			tqdm(
				pool.imap_unordered(reduce_shard, args),
				total=len(args),
				desc="Phase 2 | Reducing shards",
			)
		)

	print(f"  Kept {sum(results):,} records")

	print("▶ Phase 3: Merge")
	with output_path.open("w") as out:
		for i in tqdm(range(SHARDS), desc="Merging"):
			part = out_dir / f"part_{i}.jsonl"
			if part.exists():
				with part.open("r") as f:
					for line in f:
						out.write(line)

	print("✔ Done")


# ----------------------------
# CLI
# ----------------------------

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Canonicalize and dedupe JSONL IDs")
	parser.add_argument("-i", "--input", required=True, type=Path, help="Input JSONL file")
	parser.add_argument("-o", "--output", required=True, type=Path, help="Output JSONL file")
	args = parser.parse_args()

	main(args.input, args.output)
