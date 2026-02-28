"""
pdf_retriever.py — Lightweight in-process TF-IDF retriever for PDF chunks.

No external dependencies beyond the Python standard library.

Usage:
    from pdf_retriever import build_index, retrieve

    index  = build_index(cleaned_pages)   # call once after load_pdf + clean_pages
    chunks = retrieve(index, query, top_k=5)
    for chunk in chunks:
        print(chunk["page"], chunk["text"])
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

# ── Stopwords (English) ───────────────────────────────────────────────────────
_STOPWORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "is","are","was","were","be","been","being","have","has","had","do",
    "does","did","will","would","could","should","may","might","shall",
    "this","that","these","those","it","its","we","you","he","she","they",
    "i","me","my","your","our","their","his","her","by","from","as","into",
    "through","about","than","more","also","if","so","up","out","what",
    "which","who","can","not","no","all","any","each",
}

_TOKENISE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKENISE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 2]


# ── Index structure ───────────────────────────────────────────────────────────

def build_index(cleaned_pages: list[tuple[int, list[str]]]) -> dict[str, Any]:
    """
    Build a TF-IDF index from cleaned_pages output of pdf_to_concepts.clean_pages().

    cleaned_pages: [(page_num, [line, ...]), ...]

    Returns a dict with:
      chunks   : list of {"page": int, "text": str, "tokens": Counter}
      idf      : dict token → idf score
      N        : number of chunks
    """
    # Build chunks — one chunk per page (or split large pages into ~300-word blocks)
    raw_chunks: list[dict] = []
    for page_num, lines in cleaned_pages:
        text = " ".join(lines).strip()
        if not text:
            continue
        # Split into ~300-word sub-chunks for better precision
        words = text.split()
        step = 300
        for i in range(0, max(1, len(words)), step):
            block = " ".join(words[i: i + step])
            if len(block.strip()) < 30:
                continue
            raw_chunks.append({"page": page_num, "text": block})

    # Compute TF per chunk
    chunks = []
    for rc in raw_chunks:
        toks = _tokens(rc["text"])
        chunks.append({
            "page":   rc["page"],
            "text":   rc["text"],
            "tokens": Counter(toks),
            "len":    max(1, len(toks)),
        })

    # Compute IDF
    N = len(chunks)
    df: Counter = Counter()
    for chunk in chunks:
        for tok in set(chunk["tokens"].keys()):
            df[tok] += 1

    idf: dict[str, float] = {}
    for tok, freq in df.items():
        idf[tok] = math.log((N + 1) / (freq + 1)) + 1.0   # smoothed IDF

    return {"chunks": chunks, "idf": idf, "N": N}


# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve(
    index: dict[str, Any],
    query: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """
    Return up to `top_k` chunks most relevant to `query`, sorted by TF-IDF score.
    Each result: {"page": int, "text": str, "score": float}
    Returns empty list if index is empty or query has no usable tokens.
    """
    chunks = index.get("chunks", [])
    idf    = index.get("idf", {})
    if not chunks:
        return []

    query_tokens = _tokens(query)
    if not query_tokens:
        return []

    scored: list[tuple[float, int]] = []
    for idx, chunk in enumerate(chunks):
        score = 0.0
        for qt in query_tokens:
            tf   = chunk["tokens"].get(qt, 0) / chunk["len"]
            idf_w = idf.get(qt, 1.0)
            score += tf * idf_w
        if score > 0:
            scored.append((score, idx))

    scored.sort(key=lambda x: -x[0])
    results = []
    for score, idx in scored[:top_k]:
        results.append({
            "page":  chunks[idx]["page"],
            "text":  chunks[idx]["text"],
            "score": round(score, 4),
        })
    return results


def format_retrieved_chunks(chunks: list[dict[str, Any]], max_chars: int = 3000) -> str:
    """
    Format retrieved chunks into a grounding block for the LLM prompt.
    Trims to max_chars total.
    """
    if not chunks:
        return ""
    lines = []
    total = 0
    for c in chunks:
        snippet = f"[Page {c['page']}] {c['text']}"
        if total + len(snippet) > max_chars:
            remaining = max_chars - total
            if remaining > 100:
                lines.append(snippet[:remaining] + "…")
            break
        lines.append(snippet)
        total += len(snippet)
    return "\n\n".join(lines)
