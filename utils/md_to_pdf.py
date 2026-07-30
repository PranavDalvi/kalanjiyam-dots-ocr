import argparse
from pathlib import Path
import markdown
from weasyprint import HTML, CSS

# Default table CSS
CSS_STYLE = """
table {
    border-collapse: collapse;
    width: 100%;
}
table, th, td {
    border: 1px solid black;
    padding: 5px;
}
th {
    background-color: #f2f2f2;
}
"""

def convert_md_to_pdf(input_path: Path, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read Markdown and convert to HTML
    md_text = input_path.read_text(encoding="utf-8")
    html_content = markdown.markdown(md_text, extensions=["tables", "fenced_code"])

    # Wrap HTML in basic structure + inject CSS
    full_html = f"""
    <html>
    <head>
    <meta charset="utf-8">
    <style>{CSS_STYLE}</style>
    </head>
    <body>
    {html_content}
    </body>
    </html>
    """

    # Convert HTML → PDF using WeasyPrint
    HTML(string=full_html).write_pdf(str(output_path))

def main():
    parser = argparse.ArgumentParser(description="Convert Markdown (.md) files to PDF with WeasyPrint and default CSS")
    parser.add_argument("-i", "--input", required=True, help="Markdown file or directory containing .md files")
    parser.add_argument("-o", "--output", required=True, help="Output file (single) or output directory")

    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if input_path.is_file():
        if output_path.is_dir() or output_path.suffix.lower() != ".pdf":
            output_path = output_path / f"{input_path.stem}.pdf"

        convert_md_to_pdf(input_path, output_path)
        print(f"✔ Converted {input_path} → {output_path}")

    else:
        output_path.mkdir(parents=True, exist_ok=True)
        md_files = list(input_path.rglob("*.md"))

        if not md_files:
            print("No .md files found.")
            return

        for md in md_files:
            rel = md.relative_to(input_path)
            pdf_out = output_path / rel.with_suffix(".pdf")
            convert_md_to_pdf(md, pdf_out)

        print(f"\nDone — converted {len(md_files)} files.")

if __name__ == "__main__":
    main()
