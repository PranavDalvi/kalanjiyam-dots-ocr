import argparse
import mmap
import re
from pathlib import Path
from tqdm import tqdm

wc_re = re.compile(rb'"total_word_count"\s*:\s*(\d+)')


def filter_file(
    in_path: Path,
    out_path: Path,
    threshold: int = 6000,
    update_every: int = 10_000,
):
    kept = 0

    with open(in_path, "rb") as fin, open(out_path, "wb") as fout:
        mm = mmap.mmap(fin.fileno(), 0, access=mmap.ACCESS_READ)
        size = mm.size()

        pbar = tqdm(
            total=size,
            desc=in_path.name,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            leave=False,
        )

        last_pos = 0

        for i, m in enumerate(wc_re.finditer(mm), 1):
            wc = int(m.group(1))
            if wc > threshold:
                # find line boundaries
                start = mm.rfind(b"\n", 0, m.start())
                end = mm.find(b"\n", m.end())

                if start == -1:
                    start = 0
                else:
                    start += 1

                if end == -1:
                    end = size

                fout.write(mm[start:end] + b"\n")
                kept += 1

            if i % update_every == 0:
                pos = m.end()
                pbar.update(pos - last_pos)
                last_pos = pos

        pbar.update(size - last_pos)
        pbar.close()

    return kept


def main(input_path, output_path, threshold):
    in_path = Path(input_path)
    out_path = Path(output_path)

    if in_path.is_file():
        files = [in_path]

        if out_path.is_dir():
            out_path = out_path / in_path.name
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)

        kept = filter_file(in_path, out_path, threshold)
        print(f"{in_path.name}: kept {kept:,} lines")

    elif in_path.is_dir():
        out_path.mkdir(parents=True, exist_ok=True)
        files = sorted(in_path.glob("*.jsonl"))

        if not files:
            raise RuntimeError("No .jsonl files found in directory")

        for f in files:
            out_file = out_path / f.name
            kept = filter_file(f, out_file, threshold)
            print(f"{f.name}: kept {kept:,} lines")

    else:
        raise RuntimeError("Invalid input path")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ultra-fast JSONL filter by word_count (mmap + regex)"
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input JSONL file or directory",
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output JSONL file or directory",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=6000,
        help="word_count threshold (default: 6000)",
    )

    args = parser.parse_args()
    main(args.input, args.output, args.threshold)
