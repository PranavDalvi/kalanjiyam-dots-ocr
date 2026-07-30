import os
import sys
import logging
import argparse
import filetype
from tqdm import tqdm
from pypdf import PdfReader
from concurrent.futures import ProcessPoolExecutor, as_completed

logger = logging.getLogger("pypdf")
logger.setLevel(logging.ERROR)

def is_valid_file(path):
	try:
		with open(path, "rb") as f:
			head = f.read(4096)
		ext = filetype.guess(head)

		if ext and ext.extension in {"pdf", "jpg", "jpeg", "png"}:
			return ext.extension
		return None
	except Exception:
		return None

def count_pdf_pages(path):
	try:
		pdf_reader = PdfReader(path)
		return len(pdf_reader.pages)
	except Exception as e:
		return 0

def count_pdf_pages_fitz(path):
	try:
		import fitz
		fitz.TOOLS.mupdf_display_errors(False)
		fitz.TOOLS.mupdf_display_warnings(False)

		doc = fitz.open(path)
		num_pages = len(doc)
		doc.close()
		return num_pages
	except Exception as e:
		return 0

def main():
	parser = argparse.ArgumentParser(description="count valid files and pdf pages.")
	parser.add_argument("-i", "--input_path", type=str, required=True, help="input path to file or folder.")
	args = parser.parse_args()

	file_paths = []

	if os.path.isdir(args.input_path):
		all_files = []
		for root, _, files in os.walk(args.input_path):
			for file in files:
				if not file.startswith("."):
					all_files.append(os.path.join(root, file))

		with tqdm(total=len(all_files), desc="checking valid files", dynamic_ncols=True) as pbar:
			for path in all_files:
				ext = is_valid_file(path)
				if ext:
					file_paths.append((path, ext))

				pbar.update(1)
				pbar.set_postfix({"valid_so_far": f"{len(file_paths):,}"})


	else:
		if not os.path.exists(args.input_path):
			print(f"file not found: {args.input_path}")
			return
		ext = is_valid_file(args.input_path)
		if ext:
			file_paths.append((args.input_path, ext))

	pdf_files = [path for path, ext in file_paths if ext == "pdf"]

	print(f"total valid files: {len(file_paths)}")

	# multiprocessing for pdf page counting
	total_pages = 0
	with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
		futures = {executor.submit(count_pdf_pages_fitz, path): path for path in pdf_files}

		with tqdm(as_completed(futures), total=len(futures), desc="Counting PDF pages", dynamic_ncols=True) as pbar:
			for future in pbar:
				total_pages += future.result()
				pbar.set_postfix({"pages_so_far": f"{total_pages:,}"})

	print(f"Total PDF pages: {total_pages}")

if __name__ == "__main__":
	main()


# #!/usr/bin/env python3
# import os
# import sys
# import argparse
# from pathlib import Path
# from tqdm import tqdm
# import filetype
# from pypdf import PdfReader

# from hp_runner import HPRunner

# # -------------------------
# # Worker functions
# # -------------------------
# def is_valid_file_worker(path_tuple):
#     """Worker-safe check if file is valid."""
#     path, _ = path_tuple
#     ext = filetype.guess(path)
#     if ext and ext.extension in {"pdf", "jpg", "jpeg", "png"}:
#         return (path, ext.extension)
#     return None


# def count_pdf_pages_worker(path_tuple):
#     """Count PDF pages; worker-safe."""
#     path, _ = path_tuple
#     try:
#         reader = PdfReader(path)
#         return len(reader.pages)
#     except Exception:
#         return 0


# # -------------------------
# # Main
# # -------------------------
# def main():
#     parser = argparse.ArgumentParser(description="Count valid files and PDF pages using hp_runner.")
#     parser.add_argument("-i", "--input_path", type=str, required=True)
#     args = parser.parse_args()

#     input_path = Path(args.input_path)
#     all_files = []

#     # Collect all files recursively
#     if input_path.is_dir():
#         for root, _, files in os.walk(input_path):
#             for f in files:
#                 if not f.startswith("."):
#                     all_files.append((os.path.join(root, f), None))
#     elif input_path.exists():
#         all_files.append((str(input_path), None))
#     else:
#         print(f"Input path not found: {input_path}", file=sys.stderr)
#         sys.exit(1)

#     print(f"Total files found: {len(all_files)}")

#     # -------------------------
#     # Stage 1: Filter valid files
#     # -------------------------
#     runner1 = HPRunner(
#         fn=is_valid_file_worker,
#         chunk_size=1024,
#         ordered=False,
#         progress_desc="Validating files",
#     )

#     valid_files = []
#     for result in runner1.run(all_files):
#         if result:
#             valid_files.append(result)

#     print(f"Total valid files: {len(valid_files)}")

#     # -------------------------
#     # Stage 2: Count PDF pages
#     # -------------------------
#     pdf_files = [f for f in valid_files if f[1] == "pdf"]
#     runner2 = HPRunner(
#         fn=count_pdf_pages_worker,
#         chunk_size=64,
#         ordered=False,
#         progress_desc="Counting PDF pages",
#     )

#     total_pages = 0
#     for pages in runner2.run(pdf_files):
#         total_pages += pages

#     print(f"Total PDF pages: {total_pages}")


# if __name__ == "__main__":
#     main()
