import os
import json
import csv
import argparse
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from typing import Iterator, Tuple, List

BUFFER_SIZE = 5000


def process_file(filepath: str) -> Iterator[Tuple[str, str, int]]:
    """
    Yield (filename, language, word_count) aggregated per file.
    """
    filename = os.path.basename(filepath)
    lang_counts = defaultdict(int)

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                lang = record.get("identified_language")
                wc = record.get("word_count", 0)
                if lang:
                    lang_counts[lang] += int(wc)
            except json.JSONDecodeError:
                continue

    for lang, wc in lang_counts.items():
        yield filename, lang, wc


def worker(filepath: str):
    # multiprocessing-safe wrapper
    return list(process_file(filepath))


def iter_jsonl_files(directory: str):
    for name in os.listdir(directory):
        if name.endswith(".jsonl"):
            yield os.path.join(directory, name)


def flush_buffer(writer, buffer: List[List]):
    writer.writerows(buffer)
    buffer.clear()


def generate_report(input_dir: str, output_file: str):
    overall_counts = defaultdict(int)
    buffer: List[List] = []

    with open(output_file, "w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(["scope", "filename", "language", "word_count"])

        files = list(iter_jsonl_files(input_dir))

        with Pool(processes=cpu_count()) as pool:
            for results in pool.imap_unordered(worker, files):
                for filename, lang, wc in results:
                    buffer.append(["FILE", filename, lang, wc])
                    overall_counts[lang] += wc

                    if len(buffer) >= BUFFER_SIZE:
                        flush_buffer(writer, buffer)

        # flush remaining FILE rows
        if buffer:
            flush_buffer(writer, buffer)

        # write OVERALL rows (also buffered)
        for lang, wc in sorted(overall_counts.items()):
            buffer.append(["OVERALL", "ALL", lang, wc])
            if len(buffer) >= BUFFER_SIZE:
                flush_buffer(writer, buffer)

        if buffer:
            flush_buffer(writer, buffer)


def main():
    parser = argparse.ArgumentParser(
        description="Generate per-file and overall language word-count report from JSONL files"
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input directory containing .jsonl files"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output CSV report file"
    )

    args = parser.parse_args()
    generate_report(args.input, args.output)
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()

