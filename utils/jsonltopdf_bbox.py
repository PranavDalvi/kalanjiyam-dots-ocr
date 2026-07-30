import json
import os
import re
import argparse
from collections import defaultdict
from markdown import markdown
from weasyprint import HTML
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple
import math

# ------------------ Arguments ------------------

parser = argparse.ArgumentParser(description='Convert JSONL OCR blocks to layout-aware PDFs.')
parser.add_argument('-i', '--input', required=True, help='Input JSONL file')
parser.add_argument('-o', '--output', required=True, help='Output directory for PDFs')
parser.add_argument('-f', '--fonts', default='noto_fonts_indian_languages', help='Path to fonts directory')
parser.add_argument('--workers', type=int, default=8, help='Max parallel workers (use 1 for sequential mode)')
parser.add_argument('--batch-size', type=int, default=5, help='Batch size for parallel processing')
parser.add_argument('--layout-scale', type=float, default=None, help='Optional manual scale override for bbox layout')
parser.add_argument('--strict-bbox', action='store_true', help='Use strict bbox height with clipped overflow')
args = parser.parse_args()

jsonl_file = args.input
output_dir = args.output
font_dir = os.path.abspath(args.fonts)
os.makedirs(output_dir, exist_ok=True)

TARGET_PAGE_WIDTH = 794
TARGET_PAGE_HEIGHT = 1123
DEFAULT_PAGE_WIDTH = TARGET_PAGE_WIDTH
DEFAULT_PAGE_HEIGHT = TARGET_PAGE_HEIGHT
SOURCE_WIDTH_FLOOR = 1123
PAGE_PADDING = 40
HEIGHT_BUFFER = 1.12

# ------------------ Resume / Progress ------------------

progress_file = os.path.join(output_dir, 'progress.json')

if os.path.exists(progress_file):
    with open(progress_file, 'r', encoding='utf-8') as pf:
        completed_files = set(json.load(pf))
else:
    completed_files = set()


def save_progress(done_set):
    with open(progress_file, 'w', encoding='utf-8') as pf:
        json.dump(sorted(done_set), pf, indent=2)


# ------------------ Layout-Aware Helpers ------------------

def contains_rtl_script(text: str) -> bool:
    return any(
        ('\u0600' <= char <= '\u06FF')
        or ('\u0750' <= char <= '\u077F')
        or ('\u08A0' <= char <= '\u08FF')
        for char in text
    )


def parse_blocks(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract OCR blocks from `generated_text`/`text` compatible formats."""
    generated = entry.get('generated_text')

    if isinstance(generated, list):
        return generated

    if isinstance(generated, str):
        try:
            parsed = json.loads(generated)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                return [parsed]
        except json.JSONDecodeError:
            return [{
                'bbox': [40, 40, DEFAULT_PAGE_WIDTH - 40, DEFAULT_PAGE_HEIGHT - 40],
                'category': 'Text',
                'text': generated,
            }]

    text = entry.get('text')
    if isinstance(text, str):
        return [{
            'bbox': [40, 40, DEFAULT_PAGE_WIDTH - 40, DEFAULT_PAGE_HEIGHT - 40],
            'category': 'Text',
            'text': text,
        }]

    return []


def normalize_bbox(raw_bbox: Any) -> Tuple[int, int, int, int]:
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        return 40, 40, DEFAULT_PAGE_WIDTH - 40, DEFAULT_PAGE_HEIGHT - 40

    try:
        x1, y1, x2, y2 = [int(float(v)) for v in raw_bbox]
    except (TypeError, ValueError):
        return 40, 40, DEFAULT_PAGE_WIDTH - 40, DEFAULT_PAGE_HEIGHT - 40

    if x2 <= x1:
        x2 = x1 + 2
    if y2 <= y1:
        y2 = y1 + 2

    return x1, y1, x2, y2


def category_to_class(category: str) -> str:
    cleaned = re.sub(r'[^a-zA-Z0-9]+', '-', (category or 'Text').strip().lower()).strip('-')
    return cleaned or 'text'


def strip_outer_html_body(html: str) -> str:
    if not isinstance(html, str):
        return ''
    body_match = re.search(r'<body[^>]*>(.*)</body>', html, flags=re.IGNORECASE | re.DOTALL)
    if body_match:
        return body_match.group(1).strip()
    html_no_open = re.sub(r'<html[^>]*>', '', html, flags=re.IGNORECASE)
    html_no_both = re.sub(r'</html>', '', html_no_open, flags=re.IGNORECASE)
    return html_no_both.strip()

def estimate_line_count(text: str, bbox_width: float, font_size: float) -> int:
    """
    Estimate how many wrapped lines this text will produce
    based on bbox width and font size.
    Works well with Noto Sans family fonts.
    """
    if not text or not text.strip():
        return 1

    # Approximate average glyph width ratio for Noto Sans
    avg_char_width = font_size * 0.55

    chars_per_line = max(1, int(bbox_width / avg_char_width))

    total_lines = 0

    for paragraph in text.split('\n'):
        paragraph = paragraph.strip()

        if not paragraph:
            total_lines += 1
            continue

        # Estimate wrapping
        estimated = math.ceil(len(paragraph) / chars_per_line)
        total_lines += max(1, estimated)

    return max(1, total_lines)

def block_to_html(block: Dict[str, Any], scale: float, strict_bbox: bool) -> str:
    x1, y1, x2, y2 = normalize_bbox(block.get('bbox'))

    left = x1 * scale
    top = y1 * scale
    width = max(2.0, (x2 - x1) * scale)
    height = max(2.0, (y2 - y1) * scale)

    raw_text = block.get('text', '')
    if not isinstance(raw_text, str):
        raw_text = str(raw_text)

    # HTML content handling
    if raw_text.strip().startswith('<') and (
        '<table' in raw_text.lower() or '<html' in raw_text.lower()
    ):
        content_html = strip_outer_html_body(raw_text)
    else:
        content_html = markdown(raw_text, extensions=['tables', 'extra'])

    rtl_attr = ' dir="rtl"' if contains_rtl_script(raw_text) else ''
    cat_class = category_to_class(str(block.get('category', 'Text')))

    # ------------------------
    # 🔥 Intelligent Font Fitting
    # ------------------------

    bbox_width = width
    bbox_height = height

    # Start with height-based maximum possible size
    max_font_size = bbox_height * 0.85

    # Initial guess
    font_size_guess = max_font_size

    # Estimate how many lines will be needed
    estimated_lines = estimate_line_count(raw_text, bbox_width, font_size_guess)

    line_height_factor = 1.15  # must match CSS

    # Compute final font size to fit vertically
    fitted_font_size = bbox_height / (estimated_lines * line_height_factor)

    # Safety clamping
    font_size = max(6, min(fitted_font_size, max_font_size))

    # Final strict bounding
    size_style = f'height:{bbox_height:.2f}px;overflow:hidden;'

    return (
        f'<div class="ocr-block cat-{cat_class}" '
        f'style="left:{left:.2f}px;'
        f'top:{top:.2f}px;'
        f'width:{bbox_width:.2f}px;'
        f'font-size:{font_size:.2f}px;'
        f'{size_style}"{rtl_attr}>'
        f'{content_html}'
        '</div>'
    )


def compute_page_height(blocks: List[Dict[str, Any]], scale: float) -> int:
    max_y = 0
    for block in blocks:
        _, _, _, y2 = normalize_bbox(block.get('bbox'))
        max_y = max(max_y, y2)
    return max(TARGET_PAGE_HEIGHT, int(round((max_y + PAGE_PADDING) * scale)))


def compute_document_scale(pages: List[Tuple[int, List[Dict[str, Any]]]]) -> float:
    if args.layout_scale is not None:
        return max(0.1, args.layout_scale)

    max_x = 0
    max_y = 0
    for _, blocks in pages:
        for block in blocks:
            _, _, x2, y2 = normalize_bbox(block.get('bbox'))
            max_x = max(max_x, x2)
            max_y = max(max_y, y2)

    src_w = max(SOURCE_WIDTH_FLOOR, max_x + PAGE_PADDING)
    src_h = max(1, max_y + PAGE_PADDING)

    scale_x = TARGET_PAGE_WIDTH / src_w
    scale_y = TARGET_PAGE_HEIGHT / src_h
    return max(0.1, min(scale_x, scale_y, 1.0))


# ------------------ Collect pages per file ------------------

files_content: Dict[str, List[Tuple[int, List[Dict[str, Any]]]]] = defaultdict(list)

with open(jsonl_file, 'r', encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue

        entry = json.loads(line)
        _id = entry['id']
        blocks = parse_blocks(entry)

        parts = _id.split('↳')

        if len(parts) >= 4:
            source, dataset, book_id = parts[0], parts[1], parts[2]
            page_part = parts[3]
            try:
                page_num = int(page_part.replace('page-no-', ''))
            except ValueError:
                page_num = 0
            filename = f'{source}_{dataset}_{book_id}.pdf'

        elif len(parts) == 2:
            filename_part, page_part = parts
            filename = f'{filename_part}.pdf'
            try:
                page_num = int(page_part)
            except ValueError:
                page_num = 0
        else:
            print(f'⚠️ Skipping unrecognized id format: {_id}')
            continue

        files_content[filename].append((page_num, blocks))


# ------------------ CSS ------------------

def build_css(layout_scale: float) -> str:
    return f"""
<style>
@page {{ margin: 0; }}

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

html, body {{
    margin: 0;
    padding: 0;
    font-family: 'NotoSansDevanagari', 'NotoSansBengali', 'NotoSansTamil',
                 'NotoSansTelugu', 'NotoSansKannada', 'NotoSansMalayalam',
                 'NotoSansGujarati', 'NotoSansGurmukhi', 'NotoSansOriya',
                 'NotoSansArabic', 'NotoSansOlChiki', 'NotoSansMeeteiMayek', sans-serif;
    font-size: 12px;
    line-height: 1.2;
}}

.pdf-page {{
    position: relative;
    background: white;
    width: {TARGET_PAGE_WIDTH}px;
    page-break-after: always;
}}

.pdf-page:last-child {{
    page-break-after: auto;
}}

.ocr-block {{
    position: absolute;
    box-sizing: border-box;
    line-height: 1.15;
    white-space: pre-wrap;
    word-break: break-word;
    overflow: visible;
    padding: 0;
    margin: 0;
}}

/* Category-based styling */
.ocr-block.cat-title {{ font-weight: 700; line-height: 1.15; }}
.ocr-block.cat-section-header {{ font-weight: 700; line-height: 1.25; }}
.ocr-block.cat-header {{ font-weight: 600; }}
.ocr-block.cat-footer {{ color: #444; }}
.ocr-block.cat-list-item {{ line-height: 1.35; }}
.ocr-block.cat-caption {{ color: #333; font-style: italic; }}
.ocr-block.cat-table {{ line-height: 1.25; }}
.ocr-block.cat-picture {{ border: 1px dashed #9aa0a6; background: #f8f9fa; color: #5f6368; }}
.ocr-block.cat-page-header {{ color: #222; font-weight: 600; }}
.ocr-block.cat-page-footer {{ color: #555; }}
.ocr-block.cat-footnote {{ color: #666; font-style: italic; }}

.ocr-block table {{ width: 100%; border-collapse: collapse; table-layout: fixed; }}
.ocr-block table, .ocr-block th, .ocr-block td {{ border: 1px solid #222; }}
.ocr-block th, .ocr-block td {{ padding: 3px; vertical-align: top; }}
.ocr-block p, .ocr-block h1, .ocr-block h2, .ocr-block h3, .ocr-block h4 {{ margin: 0; }}
.ocr-block ul, .ocr-block ol {{ margin: 0; padding-left: 16px; }}
</style>
"""


# ------------------ Worker Function ------------------

def generate_single_pdf(task: Tuple[str, List[Tuple[int, List[Dict[str, Any]]]]]) -> str:
    filename, pages = task
    output_path = os.path.join(output_dir, filename)

    if os.path.exists(output_path):
        return filename

    pages.sort(key=lambda x: x[0])
    layout_scale = compute_document_scale(pages)

    full_html = build_css(layout_scale)
    for _, blocks in pages:
        page_height = compute_page_height(blocks, layout_scale)
        full_html += f'<div class="pdf-page" style="height:{page_height}px;"><div style="height:{page_height}px;">'
        for block in blocks:
            full_html += block_to_html(block, layout_scale, args.strict_bbox)
        full_html += '</div></div>'

    HTML(string=full_html, base_url='.').write_pdf(output_path)
    return filename

# # ------------------ updated functions ------------------

# # ------------------ Improved Layout Logic ------------------

# def block_to_html(block: Dict[str, Any], scale: float) -> str:
#     """Renders a block without forcing a rigid height."""
#     raw_text = block.get('text', '')
#     content_html = markdown(raw_text, extensions=['tables', 'extra'])
    
#     # Use category to determine font-weight/style
#     cat_class = category_to_class(str(block.get('category', 'Text')))
    
#     # Use coordinates only for Left and Width. 
#     # Use Top to 'nudge' the block, but don't clip the bottom.
#     x1, y1, x2, y2 = normalize_bbox(block.get('bbox'))
#     left = x1 * scale
#     width = (x2 - x1) * scale
#     top = y1 * scale

#     rtl_attr = ' dir="rtl"' if contains_rtl_script(raw_text) else ''

#     return (
#         f'<div class="ocr-block cat-{cat_class}" '
#         f'style="left:{left:.2f}px; top:{top:.2f}px; width:{width:.2f}px;"{rtl_attr}>'
#         f'{content_html}'
#         '</div>'
#     )

# # ------------------ Updated CSS ------------------

# def build_css(layout_scale: float):
#     return f"""
# <style>
# @page {{ margin: 0; size: A4; }}
# body {{ margin: 0; padding: 0; }}

# .pdf-page {{
#     position: relative;
#     width: {TARGET_PAGE_WIDTH}px;
#     min-height: {TARGET_PAGE_HEIGHT}px;
#     page-break-after: always;
#     overflow: visible; /* Prevents cutting off text */
# }}

# .ocr-block {{
#     position: absolute; /* Keep position but remove height constraints */
#     box-sizing: border-box;
#     word-wrap: break-word;
#     line-height: 1.4;
# }}

# /* Ensure images and tables don't overflow their bbox width */
# .ocr-block img, .ocr-block table {{
#     max-width: 100%;
#     height: auto;
# }}

# .cat-title {{ font-size: 1.8em; font-weight: bold; }}
# .cat-header {{ font-size: 1.4em; font-weight: bold; }}
# /* Default font scaling */
# .ocr-block {{ font-size: {12 * layout_scale}px; }} 
# </style>
# """

# ------------------ Batching + Multiprocessing ------------------

BATCH_SIZE = max(1, args.batch_size)
MAX_WORKERS = max(1, args.workers)

all_items = list(files_content.items())

pending_items = []
for filename, pages in all_items:
    output_path = os.path.join(output_dir, filename)
    if filename in completed_files:
        continue
    if os.path.exists(output_path):
        completed_files.add(filename)
        continue
    pending_items.append((filename, pages))

print(f'Total files: {len(all_items)}')
print(f'Already completed: {len(completed_files)}')
print(f'Pending: {len(pending_items)}')


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


for batch_idx, batch in enumerate(chunks(pending_items, BATCH_SIZE), start=1):
    print(f'\nProcessing batch {batch_idx} with {len(batch)} files...')

    if MAX_WORKERS == 1:
        for task in batch:
            try:
                done_filename = generate_single_pdf(task)
                completed_files.add(done_filename)
                save_progress(completed_files)
                print(f'✅ Done: {done_filename}')
            except Exception as e:
                print(f'❌ Error in one file: {e}')
        continue

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(generate_single_pdf, task) for task in batch]

        for future in as_completed(futures):
            try:
                done_filename = future.result()
                completed_files.add(done_filename)
                save_progress(completed_files)
                print(f'✅ Done: {done_filename}')
            except Exception as e:
                print(f'❌ Error in one file: {e}')

print('\nAll done!')
print(f'Progress saved in: {progress_file}')
print(f"Check the '{output_dir}' folder.")






# import json
# import os
# import re
# import argparse
# from collections import defaultdict
# from markdown import markdown
# from weasyprint import HTML
# from concurrent.futures import ProcessPoolExecutor, as_completed
# from typing import Any, Dict, List, Tuple

# # ------------------ Arguments ------------------

# parser = argparse.ArgumentParser(description='Convert JSONL OCR blocks to layout-aware PDFs.')
# parser.add_argument('-i', '--input', required=True, help='Input JSONL file')
# parser.add_argument('-o', '--output', required=True, help='Output directory for PDFs')
# parser.add_argument('-f', '--fonts', default='noto_fonts_indian_languages', help='Path to fonts directory')
# parser.add_argument('--workers', type=int, default=8, help='Max parallel workers')
# parser.add_argument('--batch-size', type=int, default=5, help='Batch size for parallel processing')
# parser.add_argument('--layout-scale', type=float, default=None, help='Manual scale override')
# args = parser.parse_args()

# jsonl_file = args.input
# output_dir = args.output
# font_dir = os.path.abspath(args.fonts)
# os.makedirs(output_dir, exist_ok=True)

# # Fixed Global Constants
# TARGET_PAGE_WIDTH = 794
# TARGET_PAGE_HEIGHT = 1123
# DEFAULT_PAGE_WIDTH = 794  # Fallback for missing bboxes
# DEFAULT_PAGE_HEIGHT = 1123
# SOURCE_WIDTH_FLOOR = 1123
# PAGE_PADDING = 40

# # ------------------ Helper Functions ------------------

# def contains_rtl_script(text: str) -> bool:
#     return any(('\u0600' <= char <= '\u06FF') or ('\u0750' <= char <= '\u077F') or ('\u08A0' <= char <= '\u08FF') for char in text)

# def parse_blocks(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
#     generated = entry.get('generated_text')
#     if isinstance(generated, list): return generated
#     if isinstance(generated, str):
#         try:
#             parsed = json.loads(generated)
#             return parsed if isinstance(parsed, list) else [parsed]
#         except:
#             return [{'bbox': [40, 40, 750, 200], 'category': 'Text', 'text': generated}]
#     return []

# def normalize_bbox(raw_bbox: Any) -> Tuple[int, int, int, int]:
#     try:
#         x1, y1, x2, y2 = [int(float(v)) for v in raw_bbox]
#         return x1, y1, max(x1+2, x2), max(y1+2, y2)
#     except:
#         return 40, 40, 750, 100

# def category_to_class(category: str) -> str:
#     return re.sub(r'[^a-zA-Z0-9]+', '-', (category or 'Text').strip().lower()).strip('-') or 'text'

# # ------------------ Layout Logic ------------------

# def block_to_html(block: Dict[str, Any], scale: float) -> str:
#     raw_text = block.get('text', '')
#     content_html = markdown(str(raw_text), extensions=['tables', 'extra'])
#     cat_class = category_to_class(str(block.get('category', 'Text')))
    
#     x1, y1, x2, y2 = normalize_bbox(block.get('bbox'))
#     left, top, width = x1 * scale, y1 * scale, (x2 - x1) * scale
#     rtl_attr = ' dir="rtl"' if contains_rtl_script(str(raw_text)) else ''

#     return f'<div class="ocr-block cat-{cat_class}" style="left:{left:.2f}px; top:{top:.2f}px; width:{width:.2f}px;"{rtl_attr}>{content_html}</div>'

# def build_css(layout_scale: float) -> str:
#     return f"""
# <style>
# @page {{ margin: 0; size: A4; }}
# body {{ margin: 0; padding: 0; font-family: sans-serif; }}
# .pdf-page {{ position: relative; width: {TARGET_PAGE_WIDTH}px; min-height: {TARGET_PAGE_HEIGHT}px; page-break-after: always; }}
# .ocr-block {{ position: absolute; box-sizing: border-box; word-wrap: break-word; line-height: 1.4; font-size: {12 * layout_scale}px; }}
# .ocr-block img, .ocr-block table {{ max-width: 100%; height: auto; }}
# .cat-title {{ font-size: 1.8em; font-weight: bold; }}
# .cat-header {{ font-size: 1.4em; font-weight: bold; }}
# </style>
# """

# def compute_document_scale(pages):
#     if args.layout_scale: return args.layout_scale
#     max_x = max_y = 0
#     for _, blocks in pages:
#         for b in blocks:
#             _, _, x2, y2 = normalize_bbox(b.get('bbox'))
#             max_x, max_y = max(max_x, x2), max(max_y, y2)
#     scale_x = TARGET_PAGE_WIDTH / max(SOURCE_WIDTH_FLOOR, max_x + PAGE_PADDING)
#     scale_y = TARGET_PAGE_HEIGHT / max(1, max_y + PAGE_PADDING)
#     return min(scale_x, scale_y, 1.0)

# # ------------------ Core Generation ------------------

# def generate_single_pdf(task):
#     filename, pages = task
#     output_path = os.path.join(output_dir, filename)
#     if os.path.exists(output_path): return filename

#     pages.sort(key=lambda x: x[0])
#     scale = compute_document_scale(pages)
#     full_html = build_css(scale)

#     for _, blocks in pages:
#         max_y = max([normalize_bbox(b.get('bbox'))[3] for b in blocks]) if blocks else 0
#         h = max(TARGET_PAGE_HEIGHT, (max_y + PAGE_PADDING) * scale)
#         full_html += f'<div class="pdf-page" style="height:{h}px;">'
#         for block in blocks:
#             full_html += block_to_html(block, scale)
#         full_html += '</div>'

#     HTML(string=full_html, base_url='.').write_pdf(output_path)
#     return filename

# # ------------------ Main Execution ------------------
# if __name__ == "__main__":
#     files_content = defaultdict(list)
#     with open(jsonl_file, 'r', encoding='utf-8') as f:
#         for line in f:
#             entry = json.loads(line)
#             blocks = parse_blocks(entry)
#             parts = entry['id'].split('↳')
#             filename = f"{parts[0]}_{parts[1]}_{parts[2]}.pdf" if len(parts) >= 4 else f"{parts[0]}.pdf"
#             files_content[filename].append((0, blocks)) # Simple page indexing

#     pending_items = list(files_content.items())
#     with ProcessPoolExecutor(max_workers=args.workers) as executor:
#         futures = [executor.submit(generate_single_pdf, item) for item in pending_items]
#         for f in as_completed(futures):
#             print(f"✅ Done: {f.result()}")
