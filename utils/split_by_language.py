import os
import sys
import re
import argparse
from collections import OrderedDict, defaultdict

def parse_args():
    p = argparse.ArgumentParser(
        description="Split large JSONL files by identified_language (streaming, optimized)"
    )
    p.add_argument(
        "-i", "--input_dir",
        help="Directory containing .jsonl files"
    )
    p.add_argument(
        "-o", "--output-dir",
        default="by_language",
        help="Output directory (default: by_language)"
    )
    p.add_argument(
        "--max-open-files",
        type=int,
        default=64,
        help="Max number of simultaneously open output files (default: 64)"
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of lines buffered per language before writing (default: 1000)"
    )
    p.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan input_dir (uses os.walk)"
    )
    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    LANG_RE = re.compile(rb'"identified_language"\s*:\s*"([^"]+)"')

    files = OrderedDict()
    buffers = defaultdict(list)

    def get_file(lang: bytes):
        if lang not in files:
            if len(files) >= args.max_open_files:
                old_lang, f = files.popitem(last=False)
                if buffers[old_lang]:
                    f.write(b"".join(buffers[old_lang]))
                    buffers[old_lang].clear()
                f.close()

            f = open(
                os.path.join(args.output_dir, lang.decode() + ".jsonl"),
                "ab",
                buffering=1024 * 1024
            )
            files[lang] = f
        else:
            files.move_to_end(lang)

        return files[lang]

    if args.recursive:
        file_iter = (
            os.path.join(root, name)
            for root, _, files_ in os.walk(args.input_dir)
            for name in files_
            if name.endswith(".jsonl")
        )
    else:
        file_iter = (
            os.path.join(args.input_dir, name)
            for name in os.listdir(args.input_dir)
            if name.endswith(".jsonl")
        )

    for path in file_iter:
        with open(path, "rb", buffering=1024 * 1024) as infile:
            for line in infile:
                m = LANG_RE.search(line)
                if not m:
                    continue

                lang = m.group(1)
                buffers[lang].append(line)

                if len(buffers[lang]) >= args.batch_size:
                    f = get_file(lang)
                    f.write(b"".join(buffers[lang]))
                    buffers[lang].clear()

    # Final flush
    for lang, f in files.items():
        if buffers[lang]:
            f.write(b"".join(buffers[lang]))
        f.close()


if __name__ == "__main__":
    main()
