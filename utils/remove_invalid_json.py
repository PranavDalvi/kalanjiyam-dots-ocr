#!/usr/bin/env python3
import argparse
import multiprocessing as mp
import json
import os
import sys
import tempfile

MAX_WORKERS = 48
CHUNK_SIZE = 8000


def validate_line(line: str):
    try:
        obj = json.loads(line)
        gen = obj.get("generated_text")
        if gen is None:
            return None

        parsed = json.loads(gen)
        if isinstance(parsed, (list, dict)):
            return line
    except Exception:
        return None

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Filter JSONL lines where generated_text is valid JSON object"
    )
    parser.add_argument("-i", "--input", required=True, help="Input JSONL file")
    parser.add_argument(
        "-o", "--output",
        help="Output JSONL file (omit to modify the input file in-place)"
    )

    args = parser.parse_args()
    input_path = args.input

    if not os.path.isfile(input_path):
        print(f"Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # In-place mode → write to temp file then replace
    if args.output:
        output_path = args.output
        temp_path = None
    else:
        dir_name = os.path.dirname(input_path) or "."
        tmp = tempfile.NamedTemporaryFile(
            dir=dir_name, delete=False, prefix=".filter_tmp_", suffix=".jsonl"
        )
        temp_path = tmp.name
        tmp.close()
        output_path = temp_path

    cpu_count = mp.cpu_count()
    workers = min(MAX_WORKERS, cpu_count)

    try:
        with open(input_path, "r", encoding="utf-8", buffering=1024 * 1024) as infile, \
             open(output_path, "w", encoding="utf-8", buffering=1024 * 1024) as outfile, \
             mp.Pool(processes=workers) as pool:

            for result in pool.imap_unordered(
                validate_line,
                infile,
                chunksize=CHUNK_SIZE,
            ):
                if result:
                    outfile.write(result)

        # Atomic replace if in-place mode
        if temp_path:
            os.replace(temp_path, input_path)

    finally:
        # Cleanup temp file on failure
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


if __name__ == "__main__":
    try:
        mp.set_start_method("fork")
    except RuntimeError:
        pass
    main()

