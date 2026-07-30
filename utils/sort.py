import json
import argparse
import tempfile
import os

def extract_sort_key(id_str):
    """
    Split the ID into two parts:
    - prefix: everything before the last ↳
    - suffix: integer after the last ↳
    Returns a tuple: (prefix as string, suffix as int)
    """
    if '↳' not in id_str:
        return (id_str, 0)

    parts = id_str.rsplit('↳', 1)
    prefix = parts[0]
    try:
        suffix = int(parts[1].lstrip('0') or '0')
    except ValueError:
        suffix = float('inf')
    return (prefix, suffix)

def main():
    parser = argparse.ArgumentParser(
        description="Sort JSONL records by custom ID key"
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to input JSONL file"
    )
    parser.add_argument(
        "-o", "--output",
        required=False,
        help="Path to output JSONL file (if omitted, sorts in place)"
    )

    args = parser.parse_args()

    input_path = args.input
    output_path = args.output or input_path  # In-place if no output provided

    # Read and parse lines
    with open(input_path, 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f if line.strip()]

    # Sort by custom key
    records.sort(key=lambda x: extract_sort_key(x.get("id", "")))

    # If doing in-place, write safely via temp file then replace
    if args.output is None:
        fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(input_path))
        os.close(fd)
        write_path = tmp_path
    else:
        write_path = output_path

    with open(write_path, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    # Replace original file if in-place
    if args.output is None:
        os.replace(write_path, input_path)

if __name__ == "__main__":
    main()
