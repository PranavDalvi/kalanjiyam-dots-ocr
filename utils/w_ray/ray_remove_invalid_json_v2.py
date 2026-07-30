#!/usr/bin/env python3

import argparse
import json
import ray
import time
from tqdm import tqdm
from typing import List, Dict, Any


# ----------------------------------------------------------------------
# ARG PARSER
# ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Filter JSONL records where generated_text is valid JSON using Ray Dataset")
    p.add_argument("-i", "--input", required=True, help="Input JSONL file")
    p.add_argument("-o", "--output", required=True, help="Output directory (Ray Dataset will write a single merged JSONL)")
    p.add_argument("--cpus", type=int, default=175, help="Number of CPUs Ray is allowed to use")
    p.add_argument("--batch-size", type=int, default=1024, help="Records per batch")
    return p.parse_args()


# ----------------------------------------------------------------------
# BATCH PROCESSOR (DROP-IN REPLACEMENT POINT)
# ----------------------------------------------------------------------

def filter_valid_generated_text(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Replace this function to change per-record logic.

    Keeps only records where:
      - generated_text exists
      - is a string
      - parses to JSON
      - parsed object is list or dict
    """
    out = []

    for record in batch:
        gen = record.get("generated_text")

        if not isinstance(gen, str):
            continue

        try:
            parsed = json.loads(gen)
            if isinstance(parsed, (list, dict)):
                out.append(record)
        except Exception:
            pass

    return out


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():
    start_total = time.perf_counter()
    args = parse_args()

    # ---------------- Phase 0 ----------------
    print("\n Phase 0: Connecting to Ray cluster")
    t0 = time.perf_counter()

    ray.init(address="auto")

    t1 = time.perf_counter()
    print(f"Phase 0 complete ({t1 - t0:.2f}s)")

    # ---------------- Phase 1 ----------------
    print("\n Phase 1: Reading input dataset")
    t0 = time.perf_counter()

    print("📖 Reading input dataset...")
    ds = ray.data.read_json(args.input)

    total = ds.count()
    print(f"🔢 Total records: {total:,}")

    t1 = time.perf_counter()
    print(f"Phase 1 complete ({t1 - t0:.2f}s)")

    # ---------------- Phase 2 ----------------
    print("\n Phase 2: Filtering valid records")
    t0 = time.perf_counter()

    print("🧠 Filtering valid records...")
    ds = ds.map_batches(
        filter_valid_generated_text,
        batch_size=args.batch_size,
        batch_format="pylist",
        num_cpus=1,
    )

    # Force execution so timing is accurate
    filtered_total = ds.count()
    print(f"✅ Records after filtering: {filtered_total:,}")

    t1 = time.perf_counter()
    print(f"Phase 2 complete ({t1 - t0:.2f}s)")

    # ---------------- Phase 3 ----------------
    print("\n Phase 3: Writing output dataset")
    t0 = time.perf_counter()

    print("✍️ Writing output (single merged JSONL)...")
    ds.write_json(
        args.output,
        try_create_dir=True,
    )

    t1 = time.perf_counter()
    print(f"Phase 3 complete ({t1 - t0:.2f}s)")

    # ---------------- Total ----------------
    end_total = time.perf_counter()
    print("\n🎉 All phases complete")
    print(f"⏱️ Total runtime: {end_total - start_total:.2f}s")



if __name__ == "__main__":
    main()
