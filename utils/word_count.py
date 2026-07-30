import argparse
import mmap
import re
from pathlib import Path
from tqdm import tqdm

wc_re = re.compile(rb'"word_count"\s*:\s*(\d+)')
# wc_re = re.compile(rb'"total_word_count"\s*:\s*(\d+)')

def process_file(path: Path, update_every=10_000):
    total_words = 0

    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        file_size = mm.size()

        total_lines = mm.read().count(b"\n")

        pbar = tqdm(
            total=file_size,
            desc=path.name,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        )

        last_pos = 0

        for i, m in enumerate(wc_re.finditer(mm), 1):
            total_words += int(m.group(1))

            # Update tqdm occasionally (cheap)
            if i % update_every == 0:
                pos = m.end()
                pbar.update(pos - last_pos)
                last_pos = pos

        # Final update
        pbar.update(file_size - last_pos)
        pbar.close()

    return total_words, total_lines


def main(input_path):
    path = Path(input_path)

    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(path.glob("*.jsonl"))
    else:
        raise RuntimeError("Invalid input path")

    print("\n=== Results ===")
    for file in files:
        words, lines = process_file(file)
        print(f"{file.name}: lines={lines:,}  words={words:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ultra-fast JSONL word_count + line counter (mmap + regex)")
    parser.add_argument("-i", "--input", required=True)
    args = parser.parse_args()

    main(args.input)
