import re
import zlib
from typing import List, Tuple, Dict
from collections import Counter

import markdown2
from bs4 import BeautifulSoup

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==========================================================
# NEW: HTML + Markdown + LATEX CLEANING PIPELINE
# ==========================================================

def remove_all_latex(text: str) -> str:
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.DOTALL)
    text = re.sub(r"\$.*?\$", " ", text)
    return text


def strip_markdown_html(text: str) -> str:
    html = markdown2.markdown(text)
    plain = BeautifulSoup(html, "html.parser").get_text(" ")
    plain = remove_all_latex(plain)
    plain = " ".join(plain.split())
    return plain


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
    raw = text.encode("utf-8")
    compressed = zlib.compress(raw)
    return len(compressed) / len(raw)


def tail_degeneration(tokens: List[str], tail_fraction: float = 0.3) -> float:
    if len(tokens) < 20:
        return 0.0

    split = int(len(tokens) * (1 - tail_fraction))
    head = tokens[:split]
    tail = tokens[split:]

    return repetition_ratio(tail) - repetition_ratio(head)


def semantic_progress_score(text: str, n_chunks: int = 4) -> float:
    words = text.split()
    if len(words) < 100:
        return 1.0

    chunk_size = max(50, len(words) // n_chunks)
    chunks = [
        " ".join(words[i : i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ][:n_chunks]

    if len(chunks) < 2:
        return 1.0

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(chunks)

    sims = []
    for i in range(len(chunks) - 1):
        sims.append(cosine_similarity(X[i], X[i + 1])[0][0])

    return 1.0 - (sum(sims) / len(sims))


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


# -----------------------------
# Example CLI usage
# -----------------------------
if __name__ == "__main__":
    # sample = """Put any text here for quick testing."""
    sample = """
102
# सुजुनाय बिजाब
**गुबै फरा :**
रा रा रा रा
रौ रौ रौ रौ
ज्ञाम ज्ञाम ज्ञुम ज्नुम
ग्राव ग्राव ग्रौ ग्रौ,
बान्दो बानला सिफायनानै
जुगाम खुलि नागिरदों आं।
थेगदाव बेगदाव जेजोंबो ब्ला ब्ला थार
दैमा दिहुं आं दे बुरलुंबुधुर
- हगार आँनो लामा हगार।
सौफाय सौसि जेखौबो फोजाव लांगोन,
दिनै हाया गाबोनथ' हागोन ?
- हगार आँनो लामा हगार।
सोरबा आंखौ थुज्त्रेथ हरदों,
सोरबा आंखौ बोज्त्रेथ लांदों,
मिथिया आं थांनांगौ थांगासिनल' दं,
लामाया जिखि जाया गावनो थुंगि जासे।
- हगार आंनो लामा हगार,
दुलाराय सोरजियानो दावगाबोला
हे संसार आंबोल' मानो थानो गेन्देला ?
नोंनि खुसियै ? सि सि सि,
नै नोंनि खर'साजोंनो बाज्रुमनोसै
- हगार आंनो लामा हगार।
सोरबा दामदों नुयी थानाव
बोहैनाय सुखांनाय जथा खाम सुफिन
ग्रोब ग्रोब बिजोंनो दावगागोन आं
दैमा दिहुं आंलाय बुरलुंबुधुर
-हगार आंनो लामा हगार।
रा रा रा रा
रौ रौ रौ रौ।
## सुजुनाय बिजाब
103
### सोदोबथिः
- **जुगाम- जुगामि**
  थेगदाउ-बेग्दाउ- हेंथा हेंसि
  शैहाबगोन – सुथि मोननाय, शैनायबादि जाथ’नाय
- **थुंगि जासे – जोबजासे**
  थानाव – थावनियाव, जायगायाव
- **बोरहोनाय – गोसो बोदोर खालामनाय**
  गेलेन्दा – मैला, गाज्ज्रि
### सोंथिः
1. दै बाज्रुम खन्थाइनि खन्थाइगिरिया मा नागिरदों ?
2. दुलाराय सोरजियानो दावगानाय समाव खन्थाइगिरिया गावल' मा जानो लुबैयाखै?
3. खन्थाइगिरिया जुगामि खुलिखौ माबोरै नागिरदों ?
4. खन्थाइगिरिया खाम सिफुं जथानि देंखोजों ग्रोब ग्रोब बहा दावगालांनो नागिरदों लिर।
5. खन्थाइगिरिया मा नागिरनानै, सोरखौ "हगार आंनो लामा हगार" होन्दों लिर।
### बेखेवथिः
* क) बान्दो बानला ............ लामा हगार।
* ख) दिनै हाया गाबोनथ' हागोन ......... लामा हगार।
* ग) दुलाराय सोरजियानो .......... मानो थानो गेन्देला ?
* घ) सोरबा दामदों .......... लामा हगार।
### फरा गियान :
दै बाज्रुमा बान्दो-बानला सिफायनानै, हाजार हेंथा-हेंसि नेवसिनानै दावगालाडे। बुरलुंबुधुर दैमा बादि लैथो थांखिनानै दावगालाडे, बिब्दिनो लिरगिरियाबो

    """
    result = analyze_text(sample)

    print("=== Token-Level Repetition Analysis ===")
    for k, v in result["metrics"].items():
        print(f"{k}: {v}")

    print("\nVotes:", result["votes"])
    print(f"\nFINAL DECISION: {'HALLUCINATED' if result['hallucinated'] else 'NOT hallucinated'}")
    print(f"Confidence: {result['confidence']:.2f}")
