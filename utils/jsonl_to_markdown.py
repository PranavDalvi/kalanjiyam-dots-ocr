import json
import argparse
from pathlib import Path
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(
        description="Group JSONL text by id prefix and write .md files."
    )
    parser.add_argument("-i", "--input", required=True, help="Path to input JSONL file")
    parser.add_argument("-o", "--output", required=True, help="Directory to write markdown files")
    parser.add_argument("--pageless", action="store_true", help="Merge text without page separators (use blank lines instead of <hr>)",)

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    pages = defaultdict(list)

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            obj = json.loads(line)
            file_id = obj["id"]
            text = obj.get("text", "")

            # Split id at ↳
            left, right = file_id.split("↳", 1)
            filename = f"{left}.md"

            # Parse page number if numeric
            try:
                page_no = int(right)
            except ValueError:
                page_no = right

            pages[filename].append((page_no, text))

    # Write grouped + sorted pages
    for filename, items in pages.items():
        items.sort(key=lambda x: x[0])

        if args.pageless:
            merged_text = "\n\n".join(t for _, t in items)
        else:
            merged_text = "\n<hr style=\"border-top: 2px dashed #888;\">\n".join(
                t for _, t in items
            )

        out_path = output_dir / filename
        with out_path.open("w", encoding="utf-8") as f:
            f.write(merged_text)

    print(f"Wrote {len(pages)} files to {output_dir.resolve()}")

if __name__ == "__main__":
    main()
