import argparse
import json
import re
import markdown2
from bs4 import BeautifulSoup


def remove_all_latex(text: str) -> str:
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.DOTALL)
    text = re.sub(r"\$.*?\$", " ", text)
    return text


def strip_markdown_html(text: str) -> str:
    # Convert markdown to HTML
    html = markdown2.markdown(text)

    # Strip HTML
    plain = BeautifulSoup(html, "html.parser").get_text(" ")

    # Remove LaTeX
    plain = remove_all_latex(plain)

    # Normalize whitespace
    plain = " ".join(plain.split())

    return plain


def clean_text(text: str) -> str:
    return strip_markdown_html(text)


def process_file(input_path: str, output_path: str):
    with open(input_path, "r", encoding="utf-8") as infile, \
         open(output_path, "w", encoding="utf-8") as outfile:

        for line in infile:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)

            cleaned_record = {
                "id": record.get("id"),
                "text": clean_text(record.get("text", ""))
            }

            outfile.write(json.dumps(cleaned_record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Remove Markdown, HTML, and LaTeX from JSONL text field"
    )
    parser.add_argument("-i", "--input", required=True, help="Input JSONL file")
    parser.add_argument("-o", "--output", required=True, help="Output JSONL file")

    args = parser.parse_args()

    process_file(args.input, args.output)


if __name__ == "__main__":
    main()
