import json
import os
import argparse
from collections import defaultdict
from markdown import markdown
from weasyprint import HTML

# Parse arguments
parser = argparse.ArgumentParser(description='Convert JSONL markdown to PDFs.')
parser.add_argument('-i', '--input', required=True, help='Input JSONL file')
parser.add_argument('-o', '--output', required=True, help='Output directory for PDFs')
args = parser.parse_args()

jsonl_file = args.input
output_dir = args.output
os.makedirs(output_dir, exist_ok=True)

# Collect pages per file
files_content = defaultdict(list)

with open(jsonl_file, 'r', encoding='utf-8') as f:
    for line in f:
        entry = json.loads(line)
        _id = entry['id']
        text = entry['text']
        filename_part, page_num = _id.split('↳')
        filename = f'{filename_part}.pdf'
        page_num = int(page_num)
        files_content[filename].append((page_num, text))

# CSS to style tables
css_style = """
<style>
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
</style>
"""

# Generate PDFs
for filename, pages in files_content.items():
    pages.sort(key=lambda x: x[0])
    full_html = css_style  # include CSS at the top
    for _, md_text in pages:
        html = markdown(md_text, extensions=['tables'])  # enable markdown tables
        full_html += html + '<p style="page-break-after: always;"></p>'
    
    output_path = os.path.join(output_dir, filename)
    HTML(string=full_html).write_pdf(output_path)
