import json
import argparse

def main(input_file, output_file):
    total_lines = 0
    faulty_generated_text = 0
    written_records = 0

    with open(input_file, "r", encoding="utf-8") as infile, \
         open(output_file, "w", encoding="utf-8") as outfile:

        for line in infile:
            if not line.strip():
                continue

            total_lines += 1

            try:
                record = json.loads(line)
                record_id = record.get("id")
                generated_text = record.get("generated_text")
            except json.JSONDecodeError:
                continue

            # generated_text is a JSON string — first decode it
            try:
                items = json.loads(generated_text)
            except Exception:
                faulty_generated_text += 1
                continue   # skip faulty lines

            # extract only "text" fields
            merged_text = "\n".join(
                item["text"] for item in items
                if isinstance(item, dict) and "text" in item
            )

            output = {
                "id": record_id,
                "text": merged_text
            }

            outfile.write(json.dumps(output, ensure_ascii=False) + "\n")
            written_records += 1

    print("Processing complete")
    print(f"Total non-empty lines processed : {total_lines}")
    print(f"Faulty generated_text entries  : {faulty_generated_text}")
    print(f"Successfully written records  : {written_records}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and merge text fields from OCRed JSONL file")
    parser.add_argument("-i", "--input", required=True, help="Path to input JSONL file")
    parser.add_argument("-o", "--output", required=True, help="Path to output JSONL file")

    args = parser.parse_args()
    main(args.input, args.output)
