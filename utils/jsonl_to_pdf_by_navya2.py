import json
import os
import argparse
from markdown import markdown
from weasyprint import HTML

# -----------------------------
# Parse arguments
# -----------------------------
parser = argparse.ArgumentParser(description='Convert JSONL markdown to a single PDF.')
parser.add_argument('-i', '--input', required=True, help='Input JSONL file')
parser.add_argument('-o', '--output', required=True, help='Output directory for PDF')
parser.add_argument('-f', '--fonts', default='noto_fonts_indian_languages', help='Path to fonts directory')
args = parser.parse_args()

jsonl_file = args.input
output_dir = args.output
font_dir = os.path.abspath(args.fonts)

os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Collect all pages globally
# -----------------------------
all_pages = []

with open(jsonl_file, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue

        entry = json.loads(line)
        _id = entry['id']
        text = entry['text']

        parts = _id.split('↳')

        # Dynamic ID parsing
        if len(parts) >= 4:
            # Format: [source, dataset, book_id, page-part]
            page_part = parts[3]
            try:
                page_num = int(page_part.replace("page-no-", ""))
            except ValueError:
                page_num = 0

        elif len(parts) == 2:
            # Format: [filename_part, page_num]
            page_part = parts[1]
            try:
                page_num = int(page_part)
            except ValueError:
                page_num = 0
        else:
            print(f"⚠️ Skipping unrecognized id format: {_id}")
            continue

        all_pages.append((page_num, text))

# Sort pages by page number
all_pages.sort(key=lambda x: x[0])

# -----------------------------
# Enhanced CSS (All Fonts)
# -----------------------------
css_style = f"""
<style>
@font-face {{ font-family: 'NotoSansDevanagari'; src: url('file://{font_dir}/NotoSansDevanagari.ttf'); }}
@font-face {{ font-family: 'NotoSansBengali'; src: url('file://{font_dir}/NotoSansBengali.ttf'); }}
@font-face {{ font-family: 'NotoSansArabic'; src: url('file://{font_dir}/NotoSansArabic.ttf'); }}
@font-face {{ font-family: 'NotoSansGujarati'; src: url('file://{font_dir}/NotoSansGujarati.ttf'); }}
@font-face {{ font-family: 'NotoSansGurmukhi'; src: url('file://{font_dir}/NotoSansGurmukhi.ttf'); }}
@font-face {{ font-family: 'NotoSansKannada'; src: url('file://{font_dir}/NotoSansKannada.ttf'); }}
@font-face {{ font-family: 'NotoSansMalayalam'; src: url('file://{font_dir}/NotoSansMalayalam.ttf'); }}
@font-face {{ font-family: 'NotoSansOriya'; src: url('file://{font_dir}/NotoSansOriya.ttf'); }}
@font-face {{ font-family: 'NotoSansTamil'; src: url('file://{font_dir}/NotoSansTamil.ttf'); }}
@font-face {{ font-family: 'NotoSansTelugu'; src: url('file://{font_dir}/NotoSansTelugu.ttf'); }}
@font-face {{ font-family: 'NotoSansOlChiki'; src: url('file://{font_dir}/NotoSansOlChiki.ttf'); }}
@font-face {{ font-family: 'NotoSansMeeteiMayek'; src: url('file://{font_dir}/NotoSansMeeteiMayek.ttf'); }}

body {{
    font-family: 'NotoSansDevanagari', 'NotoSansBengali', 'NotoSansTamil',
                'NotoSansTelugu', 'NotoSansKannada', 'NotoSansMalayalam',
                'NotoSansGujarati', 'NotoSansGurmukhi', 'NotoSansOriya',
                'NotoSansArabic', 'NotoSansOlChiki', 'NotoSansMeeteiMayek', sans-serif;
    font-size: 14px;
    line-height: 1.6;
    direction: ltr;
}}

.rtl {{
    direction: rtl;
    text-align: right;
    font-family: 'NotoSansArabic', sans-serif;
}}

.rtl-inline {{
    direction: rtl;
    unicode-bidi: bidi-override;
    font-family: 'NotoSansArabic', sans-serif;
    display: inline-block;
}}

table {{
    border-collapse: collapse;
    width: 100%;
}}

table, th, td {{
    border: 1px solid black;
    padding: 5px;
}}

th {{
    background-color: #f2f2f2;
}}
</style>
"""

# -----------------------------
# RTL Detection
# -----------------------------
def contains_rtl_script(text):
    return any(
        ('\u0600' <= char <= '\u06FF') or
        ('\u0750' <= char <= '\u077F') or
        ('\u08A0' <= char <= '\u08FF')
        for char in text
    )

def wrap_rtl_content(html_content):
    if '<table>' in html_content and contains_rtl_script(html_content):
        html_content = html_content.replace('<table>', '<table dir="rtl">')

    lines = html_content.split('\n')
    wrapped_lines = []

    for line in lines:
        if contains_rtl_script(line) and ('<li>' in line or '<p>' in line):
            line = line.replace('<li>', '<li dir="rtl">')
            line = line.replace('<p>', '<p dir="rtl">')
        wrapped_lines.append(line)

    return '\n'.join(wrapped_lines)

# -----------------------------
# Build Full HTML
# -----------------------------
full_html = css_style

for _, md_text in all_pages:
    html_content = markdown(md_text, extensions=['tables', 'extra'])
    html_content = wrap_rtl_content(html_content)

    full_html += f'<div>{html_content}</div>'
    full_html += '<p style="page-break-after: always;"></p>'

# -----------------------------
# Output Filename = Input Filename
# -----------------------------
input_filename = os.path.splitext(os.path.basename(jsonl_file))[0]
output_pdf_path = os.path.join(output_dir, f"{input_filename}.pdf")

HTML(string=full_html, base_url=".").write_pdf(output_pdf_path)

print(f"✅ Done! PDF saved at: {output_pdf_path}")
