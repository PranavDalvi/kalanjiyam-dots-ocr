import re
import zlib
from collections import Counter
from typing import Dict, List, Tuple
import argparse
import os
import json
import multiprocessing as mp
from pathlib import Path
from functools import partial
from tqdm import tqdm
import mmap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import markdown2
from bs4 import BeautifulSoup
import numpy as np

# ==========================================================
# NEW: HTML + Markdown + LATEX CLEANING PIPELINE
# ==========================================================

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


# ==========================================================
# TOKENIZATION & N-GRAM UTILITIES
# ==========================================================

def tokenize(text: str) -> List[str]:
	return re.findall(r"\w+|\S", text)


def ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
	return list(zip(*[tokens[i:] for i in range(n)]))


# ==========================================================
# DETECTORS
# ==========================================================

def repetition_ratio(tokens: List[str]) -> float:
	if not tokens:
		return 0.0
	return 1.0 - (len(set(tokens)) / len(tokens))


def window_median_repetition(
	tokens: List[str], window_size: int = 100, step: int = 50
) -> float:
	if len(tokens) < window_size:
		return repetition_ratio(tokens)

	scores = []
	for i in range(0, len(tokens) - window_size + 1, step):
		w = tokens[i : i + window_size]
		scores.append(repetition_ratio(w))

	return float(np.median(scores)) if scores else repetition_ratio(tokens)


def max_consecutive_repetition(tokens: List[str]) -> int:
	if not tokens:
		return 0

	max_streak = 1
	current = 1

	for i in range(1, len(tokens)):
		if tokens[i] == tokens[i - 1]:
			current += 1
		else:
			current = 1
		max_streak = max(max_streak, current)

	return max_streak


def ngram_echo_score(tokens: List[str]) -> Dict[str, int]:
	bigrams = Counter(ngrams(tokens, 2))
	trigrams = Counter(ngrams(tokens, 3))

	return {
		"max_bigram_count": max(bigrams.values()) if bigrams else 0,
		"max_trigram_count": max(trigrams.values()) if trigrams else 0,
	}


def jaccard_similarity(a: List[str], b: List[str]) -> float:
	set_a, set_b = set(a), set(b)
	if not set_a and not set_b:
		return 0.0
	return len(set_a & set_b) / len(set_a | set_b)


def sliding_window_similarity(
	tokens: List[str], window_size: int = 50, step: int = 25
) -> float:
	if len(tokens) < window_size * 2:
		return 0.0

	sims = []
	for i in range(0, len(tokens) - window_size, step):
		w1 = tokens[i : i + window_size]
		w2 = tokens[i + step : i + step + window_size]
		if len(w2) < window_size:
			break
		sims.append(jaccard_similarity(w1, w2))

	return max(sims) if sims else 0.0


def compression_ratio(text: str) -> float:
	if not text:
		return 1.0
	raw = text.encode("utf-8", errors="ignore")
	compressed = zlib.compress(raw)
	return len(compressed) / len(raw)


def tail_degeneration(tokens: List[str], tail_fraction: float = 0.3) -> float:
	if len(tokens) < 20:
		return 0.0

	split = int(len(tokens) * (1 - tail_fraction))
	head = tokens[:split]
	tail = tokens[split:]

	return repetition_ratio(tail) - repetition_ratio(head)


# def semantic_progress_score(text: str, n_chunks: int = 4) -> float:
#     words = text.split()
#     if len(words) < 100:
#         return 1.0

#     chunk_size = max(50, len(words) // n_chunks)
#     chunks = [
#         " ".join(words[i : i + chunk_size])
#         for i in range(0, len(words), chunk_size)
#     ][:n_chunks]

#     if len(chunks) < 2:
#         return 1.0

#     vectorizer = TfidfVectorizer()
#     X = vectorizer.fit_transform(chunks)

#     sims = []
#     for i in range(len(chunks) - 1):
#         sims.append(cosine_similarity(X[i], X[i + 1])[0][0])

#     return 1.0 - (sum(sims) / len(sims))

def semantic_progress_score(text: str, n_chunks: int = 4) -> float:
	words = text.split()
	if len(words) < 100:
		return 1.0  # too short to meaningfully judge

	chunk_size = max(50, len(words) // n_chunks)
	chunks = [
		" ".join(words[i : i + chunk_size])
		for i in range(0, len(words), chunk_size)
	][:n_chunks]

	# === SAFETY CHECK ===
	if all(len(chunk.strip()) == 0 for chunk in chunks):
		return 1.0  # no content, treat as high semantic progress

	try:
		vectorizer = TfidfVectorizer()
		X = vectorizer.fit_transform(chunks)
		sims = []
		for i in range(len(chunks) - 1):
			sims.append(cosine_similarity(X[i], X[i + 1])[0][0])
		avg_similarity = sum(sims) / len(sims)
		return 1.0 - avg_similarity  # invert similarity -> progress
	except ValueError:
		# Empty vocabulary (e.g., all chunks are stop words)
		return 1.0


# ==========================================================
# UPDATED THRESHOLDS
# ==========================================================

WEIGHTS = {
	"repetition_ratio": 1,
	"max_streak": 2,
	"ngram_echo": 1,
	"window_similarity": 1,
	"compression": 1,
	"tail_degeneration": 1,
	"semantic_progress": 2,
}

THRESHOLDS = {
	"repetition_ratio": 0.65,   # <-- higher
	"max_streak": 8,            # <-- stricter
	"max_bigram": 8,
	"max_trigram": 7,
	"window_similarity": 0.85,  # <-- stricter
	"compression_ratio": 0.32,
	"tail_degeneration": 0.15,
	"semantic_progress": 0.25,
}

DECISION_THRESHOLD = 6


# ==========================================================
# 🔐 ANALYSIS WITH SMART REPETITION GATE
# ==========================================================

def analyze_text(raw_text: str) -> Dict:
	cleaned = clean_text(raw_text)
	tokens = tokenize(cleaned)

	global_rep = repetition_ratio(tokens)
	rep_ratio = window_median_repetition(tokens)

	max_streak = max_consecutive_repetition(tokens)
	ngram_stats = ngram_echo_score(tokens)
	window_sim = sliding_window_similarity(tokens)
	comp_ratio = compression_ratio(cleaned)
	tail_deg = tail_degeneration(tokens)
	sem_prog = semantic_progress_score(cleaned)

	metrics = {
		"global_repetition_ratio": global_rep,
		"window_median_repetition": rep_ratio,
		"max_consecutive_repetition": max_streak,
		"max_bigram_count": ngram_stats["max_bigram_count"],
		"max_trigram_count": ngram_stats["max_trigram_count"],
		"window_similarity": window_sim,
		"compression_ratio": comp_ratio,
		"tail_degeneration": tail_deg,
		"semantic_progress": sem_prog,
	}

	# ======= SEMANTIC GATE (kept as you wanted) =======
	if sem_prog > 0.7 and tail_deg < 0:
		return {
			"cleaned_text_preview": cleaned[:500],
			"metrics": metrics,
			"votes": None,
			"total_weight": sum(WEIGHTS.values()),
			"weight_for_hallucination": 0,
			"hallucinated": False,
			"confidence": 1.0,
			"reason": "Passed semantic gate",
		}

	# ======= NEW: STRUCTURE-AWARE REPETITION RULE =======
	def repetition_is_suspicious():
		return (
			rep_ratio > THRESHOLDS["repetition_ratio"]
			and (
				max_streak >= THRESHOLDS["max_streak"]
				or window_sim >= THRESHOLDS["window_similarity"]
			)
		)

	votes = {}
	total_weight = sum(WEIGHTS.values())
	weight_for_hallucination = 0

	if repetition_is_suspicious():
		votes["repetition_ratio"] = True
		weight_for_hallucination += WEIGHTS["repetition_ratio"]
	else:
		votes["repetition_ratio"] = False

	if max_streak >= THRESHOLDS["max_streak"]:
		votes["max_streak"] = True
		weight_for_hallucination += WEIGHTS["max_streak"]
	else:
		votes["max_streak"] = False

	if (
		ngram_stats["max_bigram_count"] >= THRESHOLDS["max_bigram"]
		or ngram_stats["max_trigram_count"] >= THRESHOLDS["max_trigram"]
	):
		votes["ngram_echo"] = True
		weight_for_hallucination += WEIGHTS["ngram_echo"]
	else:
		votes["ngram_echo"] = False

	if window_sim >= THRESHOLDS["window_similarity"]:
		votes["window_similarity"] = True
		weight_for_hallucination += WEIGHTS["window_similarity"]
	else:
		votes["window_similarity"] = False

	if comp_ratio < THRESHOLDS["compression_ratio"]:
		votes["compression"] = True
		weight_for_hallucination += WEIGHTS["compression"]
	else:
		votes["compression"] = False

	if tail_deg > THRESHOLDS["tail_degeneration"]:
		votes["tail_degeneration"] = True
		weight_for_hallucination += WEIGHTS["tail_degeneration"]
	else:
		votes["tail_degeneration"] = False

	if sem_prog < THRESHOLDS["semantic_progress"]:
		votes["semantic_progress"] = True
		weight_for_hallucination += WEIGHTS["semantic_progress"]
	else:
		votes["semantic_progress"] = False

	hallucinated = weight_for_hallucination >= DECISION_THRESHOLD
	confidence = weight_for_hallucination / total_weight

	return {
		"cleaned_text_preview": cleaned[:500],
		"metrics": metrics,
		"votes": votes,
		"total_weight": total_weight,
		"weight_for_hallucination": weight_for_hallucination,
		"hallucinated": hallucinated,
		"confidence": confidence,
		"reason": "Structure-aware repetition + weighted vote",
	}


# ======================================================
# STREAMING JSONL SPLITTER (YOUR REQUESTED VERSION)
# ======================================================
# Try fast JSON; fall back to stdlib if missing
try:
	import ujson as fastjson
except ImportError:
	fastjson = json

# ------------------------------------------------------
# ---- NUMA helpers ------------------------------------
# ------------------------------------------------------
def pin_to_numa_node(node_id: int):
	"""
	Pin this process to a specific NUMA node.
	Requires Linux.
	"""
	try:
		# Map NUMA nodes to CPU lists (from your lscpu)
		NUMA_MAP = {
			0: list(range(0, 56)) + list(range(112, 168)),
			1: list(range(56, 112)) + list(range(168, 224)),
		}
		os.sched_setaffinity(0, set(NUMA_MAP[node_id]))
	except Exception:
		pass  # silently ignore if not supported


def worker_process(batch, worker_id, numa_node):
	"""
	Each worker:
	- Pins itself to a NUMA node
	- Parses JSON fast
	- Concatenates text fields
	- Runs your existing detector
	- Returns minimal tuples (no file I/O here)
	"""
	pin_to_numa_node(numa_node)

	results = []
	for line_idx, line in batch:
		line = line.rstrip("\n")
		if not line:
			results.append((False, line_idx, line))
			continue

		try:
			record = fastjson.loads(line)
		except Exception:
			results.append((False, line_idx, line))
			continue

		try:
			blocks = fastjson.loads(record.get("generated_text", "[]"))
		except Exception:
			results.append((False, line_idx, line))
			continue

		# ---- ZERO-COPY CONCATENATION (generator join) ----
		if isinstance(blocks, list):
			page_text = "\n".join(
				str(b.get("text", ""))
				for b in blocks
				if isinstance(b, dict) and b.get("text") is not None
			)
		else:
			page_text = record.get("extracted_text", "")

		analysis = analyze_text(page_text)
		results.append((analysis["hallucinated"], line_idx, line))

	return results

def count_lines_mmap(path: Path) -> int:
	with path.open("r+b") as f:
		mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
		return mm.read().count(b"\n")


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--input_path", required=True, help="Path to large input.jsonl")
	args = parser.parse_args()

	input_path = Path(args.input_path)
	input_dir = input_path.parent
	input_base = input_path.stem

	hallucinated_path = input_dir / f"{input_base}_hallucinated.jsonl"
	not_hallucinated_path = input_dir / f"{input_base}_not_hallucinated.jsonl"

	# Counters (updated while writing)
	hallucinated_count = 0
	not_hallucinated_count = 0

	# ------------- TUNING FOR YOUR MACHINE -------------
	TOTAL_CORES = mp.cpu_count()          # 224 on your box
	N_WORKERS = max(2, TOTAL_CORES - 8)   # keep some headroom
	BATCH_SIZE = 50_000                   # lines per batch
	MAX_INFLIGHT = N_WORKERS * 2          # backpressure limit

	print(f"INPUT:  {input_path}")
	print(f"HALLUCINATED → {hallucinated_path}")
	print(f"NOT HALLUCINATED → {not_hallucinated_path}")
	print(f"WORKERS: {N_WORKERS}")
	print(f"BATCH SIZE: {BATCH_SIZE:,}")

	# Assign workers round-robin to NUMA nodes
	worker_numa = [i % 2 for i in range(N_WORKERS)]

	# --------------------------------------------------
	# Open files + pool
	# --------------------------------------------------
	with (
		mp.Pool(processes=N_WORKERS) as pool,
		input_path.open("r", encoding="utf-8") as fin,
		hallucinated_path.open("w", encoding="utf-8") as fout_h,
		not_hallucinated_path.open("w", encoding="utf-8") as fout_nh,
	):

		print("Counting total lines for progress bar...")
		total_lines = count_lines_mmap(input_path)
		progress = tqdm(total=total_lines, unit="lines", desc="Processing")

		batch = []
		pending = []
		line_counter = 0
		next_write_idx = 0
		buffer = {}  # hold out-of-order results

		def submit_batch(batch_data, worker_id):
			return pool.apply_async(
				worker_process,
				(batch_data, worker_id, worker_numa[worker_id]),
			)

		worker_id = 0

		# ------------- STREAM INPUT + DISPATCH -------------
		for line in fin:
			batch.append((line_counter, line))
			line_counter += 1

			if progress is not None:
				progress.update(1)

			if len(batch) >= BATCH_SIZE:
				pending.append(submit_batch(batch, worker_id))
				worker_id = (worker_id + 1) % N_WORKERS
				batch = []

			# ---- BACKPRESSURE: don't overwhelm workers ----
			while len(pending) >= MAX_INFLIGHT:
				fut = pending.pop(0)
				results = fut.get()

				for is_hallucinated, idx, original_line in results:
					buffer[idx] = (is_hallucinated, original_line)

				# Write in-order as much as possible
				while next_write_idx in buffer:
					is_h, text = buffer.pop(next_write_idx)

					if is_h:
						fout_h.write(text + "\n")
						hallucinated_count += 1
					else:
						fout_nh.write(text + "\n")
						not_hallucinated_count += 1

					next_write_idx += 1

		# ---- Flush last partial batch ----
		if batch:
			pending.append(submit_batch(batch, worker_id))

		# ---- Drain remaining futures ----
		for fut in pending:
			results = fut.get()

			for is_hallucinated, idx, original_line in results:
				buffer[idx] = (is_hallucinated, original_line)

			while next_write_idx in buffer:
				is_h, text = buffer.pop(next_write_idx)

				if is_h:
					fout_h.write(text + "\n")
					hallucinated_count += 1
				else:
					fout_nh.write(text + "\n")
					not_hallucinated_count += 1

				next_write_idx += 1

		if progress is not None:
			progress.close()

		print(f"\n✅ DONE. Total processed: {line_counter:,} lines")

		print("\n📊 OUTPUT SUMMARY")
		print(f"Hallucinated lines     : {hallucinated_count:,}")
		print(f"Not hallucinated lines : {not_hallucinated_count:,}")
		print(
			f"Total written          : "
			f"{hallucinated_count + not_hallucinated_count:,}"
		)


if __name__ == "__main__":
	main()
