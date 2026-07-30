
import json
import os
import argparse
from collections import defaultdict
from markdown import markdown
from weasyprint import HTML, CSS

# Parse arguments
parser = argparse.ArgumentParser(description='Convert JSONL markdown to PDFs.')
parser.add_argument('-i', '--input', required=True, help='Input JSONL file')
parser.add_argument('-o', '--output', required=True, help='Output directory for PDFs')
parser.add_argument('-f', '--fonts', default='noto_fonts_indian_languages', help='Path to fonts directory')
args = parser.parse_args()

jsonl_file = args.input
output_dir = args.output
font_dir = os.path.abspath(args.fonts)
os.makedirs(output_dir, exist_ok=True)

# # Collect pages per file
# files_content = defaultdict(list)

# with open(jsonl_file, 'r', encoding='utf-8') as f:
#     for line in f:
#         if not line.strip(): continue
#         entry = json.loads(line)
#         _id = entry['id']
#         text = entry['text']
#         filename_part, page_num = _id.split('↳')
#         filename = f'{filename_part}.pdf'
#         page_num = int(page_num)
#         files_content[filename].append((page_num, text))

# ------------------ DYNAMIC Collection ------------------
files_content = defaultdict(list)

with open(jsonl_file, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip(): continue
        entry = json.loads(line)
        _id = entry['id']
        text = entry['text']

        parts = _id.split('↳')
        
        # DYNAMIC LOGIC STARTS HERE
        if len(parts) >= 4:
            # Complex Format: [source, dataset, book_id, page-part, ...]
            source, dataset, book_id = parts[0], parts[1], parts[2]
            page_part = parts[3]
            try:
                page_num = int(page_part.replace("page-no-", ""))
            except ValueError:
                page_num = 0 
            filename = f"{source}_{dataset}_{book_id}.pdf"
            
        elif len(parts) == 2:
            # Simple Format: [filename_part, page_num]
            filename_part, page_part = parts
            filename = f"{filename_part}.pdf"
            try:
                page_num = int(page_part)
            except ValueError:
                page_num = 0
        else:
            print(f"⚠️ Skipping unrecognized id format: {_id}")
            continue

        files_content[filename].append((page_num, text))

# Enhanced CSS with all 12 fonts
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

/* RTL support - ONLY for specific RTL blocks */
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

/* Script-specific optimizations */
[lang="hi"], [lang="mr"], [lang="sa"], [lang="ne"], [lang="kok"], [lang="brx"], [lang="doi"], [lang="mai"] {{
    font-family: 'NotoSansDevanagari', sans-serif;
}}

[lang="bn"], [lang="as"] {{
    font-family: 'NotoSansBengali', sans-serif;
}}

[lang="gu"] {{
    font-family: 'NotoSansGujarati', sans-serif;
}}

[lang="pa"] {{
    font-family: 'NotoSansGurmukhi', sans-serif;
}}

[lang="kn"] {{
    font-family: 'NotoSansKannada', sans-serif;
}}

[lang="ml"] {{
    font-family: 'NotoSansMalayalam', sans-serif;
}}

[lang="or"] {{
    font-family: 'NotoSansOriya', sans-serif;
}}

[lang="ta"] {{
    font-family: 'NotoSansTamil', sans-serif;
}}

[lang="te"] {{
    font-family: 'NotoSansTelugu', sans-serif;
}}

[lang="sat"] {{
    font-family: 'NotoSansOlChiki', sans-serif;
}}

[lang="mni"] {{
    font-family: 'NotoSansMeeteiMayek', 'NotoSansBengali', sans-serif;
}}

[lang="ur"], [lang="ks"], [lang="sd"] {{
    font-family: 'NotoSansArabic', sans-serif;
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

def contains_rtl_script(text):
    """Check if text contains RTL script characters"""
    return any(
        ('\u0600' <= char <= '\u06FF') or 
        ('\u0750' <= char <= '\u077F') or 
        ('\u08A0' <= char <= '\u08FF')
        for char in text
    )

def wrap_rtl_content(html_content):
    """
    Wrap RTL content intelligently - only wrap actual RTL text,
    not the entire page structure.
    """
    # If the HTML contains tables with RTL content, wrap table cells
    if '<table>' in html_content:
        # Add dir="rtl" to table if it has RTL text
        if contains_rtl_script(html_content):
            html_content = html_content.replace('<table>', '<table dir="rtl">')
    
    # For lists and paragraphs, use inline wrapping
    lines = html_content.split('\n')
    wrapped_lines = []
    
    for line in lines:
        if contains_rtl_script(line) and ('<li>' in line or '<p>' in line):
            # Wrap RTL list items and paragraphs
            line = line.replace('<li>', '<li dir="rtl">')
            line = line.replace('<p>', '<p dir="rtl">')
        wrapped_lines.append(line)
    
    return '\n'.join(wrapped_lines)

# Generate PDFs
for filename, pages in files_content.items():
    pages.sort(key=lambda x: x[0])
    full_html = css_style
    for _, md_text in pages:
        # Convert markdown to HTML
        html_content = markdown(md_text, extensions=['tables', 'extra'])
        
        # Intelligently wrap RTL content
        html_content = wrap_rtl_content(html_content)
        
        # Add to document (no automatic RTL wrapping of entire page)
        full_html += f'<div>{html_content}</div>'
        full_html += '<p style="page-break-after: always;"></p>'

    output_path = os.path.join(output_dir, filename)
    HTML(string=full_html, base_url=".").write_pdf(output_path)

print(f"Done! Check the '{output_dir}' folder.")
