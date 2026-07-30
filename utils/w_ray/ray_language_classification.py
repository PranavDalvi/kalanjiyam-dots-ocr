import argparse
import json
import os
from typing import List, Optional
from collections import deque
import ray
from tqdm import tqdm
import re
from bs4 import BeautifulSoup
import markdown2
from pybloom_live import BloomFilter


def remove_all_latex(text: str) -> str:
	text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.DOTALL)
	text = re.sub(r"\$.*?\$", " ", text)
	return text


def strip_markdown_html(text: str) -> str:
	try:
		html = markdown2.markdown(text, extras=[])
		plain = BeautifulSoup(html, "html.parser").get_text(" ")
	except (RecursionError, Exception):
		return ""

	plain = remove_all_latex(plain)
	return " ".join(plain.split())


def clean_text(text: str) -> str:
	return strip_markdown_html(text)


def ensure_package(package_name: str, import_name: str = None):
	"""
	Ensure a Python package is installed and importable.

	Args:
		package_name: name to pip install (e.g. 'fasttext')
		import_name: name to import (defaults to package_name)
	"""
	import importlib
	import subprocess
	import sys

	import_name = import_name or package_name

	try:
		importlib.import_module(import_name)
	except ImportError:
		subprocess.check_call(
			[sys.executable, "-m", "pip", "install", package_name]
		)
		importlib.invalidate_caches()
		importlib.import_module(import_name)


def load_completed_ids_bloom(output_path, capacity=100_000_000):
	bloom = BloomFilter(capacity=capacity, error_rate=1e-5)

	if not os.path.exists(output_path):
		return bloom

	with open(output_path, "r", encoding="utf-8") as f:
		for line in f:
			try:
				rec = json.loads(line)
				rid = rec.get("id")
				if rid:
					bloom.add(rid)
			except json.JSONDecodeError:
				continue

	return bloom


# ---------------------------
# Heavy per-row processing
# ---------------------------
def process_line(line: str) -> Optional[dict]:
	try:
		record = json.loads(line)
	except json.JSONDecodeError:
		return None

	generated_text = record.get("generated_text")
	if not isinstance(generated_text, str):
		return None

	try:
		items = json.loads(generated_text)
	except Exception:
		return None

	if not isinstance(items, list):
		return None

	texts = []
	for item in items:
		if not isinstance(item, dict):
			continue
		if item.get("category") in {"Page-header", "Page-footer"}:
			continue
		text = item.get("text")
		if isinstance(text, str):
			texts.append(text)

	merged_text = "\n".join(texts) if texts else ""
	word_count = sum(len(t.split()) for t in texts)

	record["extracted_text"] = merged_text
	record["word_count"] = word_count
	return record


# ---------------------------
# Ray Actor
# ---------------------------
@ray.remote(num_cpus=1)
class LanguageWorker:
	def __init__(self, model_path: str):
		# ensure_package("fasttext")
		# ensure_package("bs4")
		# ensure_package("markdown2")

		import fasttext
		from bs4 import BeautifulSoup
		import markdown2
		self.model = fasttext.load_model(model_path)

	def process_batch(self, lines: List[str]) -> List[dict]:
		out = []
		for line in lines:
			record = process_line(line)
			if record is None:
				continue

			text = clean_text(record.get("extracted_text", ""))
			lang = "unknown"
			if text:
				try:
					labels, _ = self.model.predict(
						text,
						k=1,
						threshold=0.0,
						on_unicode_error="ignore"
					)

					if labels:
						lang = labels[0].replace("__label__", "")
				except Exception:
					pass

			record["identified_language"] = lang
			out.append(record)

		return out


# ---------------------------
# Utilities
# ---------------------------
def resumable_batched_iterator(
	input_path: str,
	batch_size: int,
	completed_ids
):
	batch = []

	with open(input_path, "r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue

			try:
				rec = json.loads(line)
			except json.JSONDecodeError:
				continue

			rid = rec.get("id")
			if not rid:
				continue

			# 🔥 Skip already completed records
			if rid in completed_ids:
				continue

			batch.append(line)

			if len(batch) == batch_size:
				yield batch
				batch = []

		if batch:
			yield batch



def count_lines(path: str) -> int:
	with open(path, "r", encoding="utf-8") as f:
		return sum(1 for _ in f)

def safe_json_dumps(obj) -> str:
	return (
		json.dumps(obj, ensure_ascii=False)
		.encode("utf-8", "surrogatepass")
		.decode("utf-8", "ignore")
	)

# ---------------------------
# Main
# ---------------------------
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input", required=True)
	parser.add_argument("-o", "--output", required=True)
	parser.add_argument("--model", required=True)
	parser.add_argument("--batch-size", type=int, default=500)
	parser.add_argument("--num-actors", type=int, default=8)
	parser.add_argument("--max-in-flight", type=int, default=16)
	args = parser.parse_args()

	input_base = os.path.splitext(os.path.basename(args.input))[0]
	final_name = f"{input_base}_filtered.jsonl"
	os.makedirs(args.output, exist_ok=True)
	out_path = os.path.join(args.output, final_name)

	print("🚀 Connecting to Ray...")
	ray.init(address="auto")


	print("🔁 Loading completed IDs (Bloom filter)...")
	completed_ids = load_completed_ids_bloom(out_path)
	print("   Done.")

	print(f"🧠 Starting {args.num_actors} LanguageWorker actors")
	workers = [
		LanguageWorker.remote(args.model)
		for _ in range(args.num_actors)
	]

	pending = deque()
	worker_idx = 0

	# 🔑 APPEND MODE (required for resume)
	with open(out_path, "a", encoding="utf-8") as outfile, \
		 tqdm(desc="Processing batches", unit="batch") as pbar:

		for batch in resumable_batched_iterator(
			args.input,
			args.batch_size,
			completed_ids
		):
			worker = workers[worker_idx % len(workers)]
			worker_idx += 1

			pending.append(worker.process_batch.remote(batch))

			# BACKPRESSURE
			if len(pending) >= args.max_in_flight:
				done, pending = ray.wait(list(pending), num_returns=1)
				results = ray.get(done[0])

				for record in results:
					outfile.write(safe_json_dumps(record) + "\n")
					completed_ids.add(record["id"])  # 🔒 strengthen resume

				pbar.update(1)

		# Drain remaining
		while pending:
			done, pending = ray.wait(list(pending), num_returns=1)
			results = ray.get(done[0])

			for record in results:
				outfile.write(safe_json_dumps(record) + "\n")
				completed_ids.add(record["id"])

			pbar.update(1)

	print("🎉 Done!")
	print(f"📄 Output written to: {out_path}")


if __name__ == "__main__":
	main()
