#!/usr/bin/env python3
"""
pdf_to_concepts.py — Convert a lecture PDF into a structured concept map.

Heuristic pipeline (default, no API required)
──────────────────────────────────────────────
  1. load_pdf          Extract raw text page-by-page.
  2. clean_pages       Remove repeated headers/footers, page numbers, collapse
                       whitespace.
  3. tokenize          Classify every line as heading (level 1/2) or body text.
  4. bucket_tokens     Group body lines under their nearest heading → buckets.
  5. process_bucket    Per bucket:
                         a. Pull useful sentences (bullets, definitions, steps).
                         b. Deduplicate with Jaccard similarity.
                         c. Name each as a micro-skill ("Can explain …").
                         d. Extract keywords (freq-based, no stopwords).
                         e. Generate 1–2 short check questions from templates.
  6. build_topic_tree  Assemble Topic → Subtopic → Concept hierarchy.
  7. write_json        Emit concepts.json.
  8. write_markdown    Emit concepts.md.

LLM mode (--llm flag)
──────────────────────
  Replaces step 5 with a call to call_llm(prompt) → structured JSON.
  Edit call_llm() at the top of this file to plug in OpenAI / Ollama / etc.

TODO (video generation, unrelated to this file):
  This project will later generate animated video from manga panels.
  See the manga pipeline in panel_extract.py.
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Optional

# ── PDF backends (prefer pdfplumber, fall back to pypdf) ─────────────────────
try:
    import pdfplumber          # type: ignore
    _PDF_BACKEND = "pdfplumber"
except ImportError:
    pdfplumber = None          # type: ignore
    _PDF_BACKEND = None

try:
    import pypdf               # type: ignore
    if _PDF_BACKEND is None:
        _PDF_BACKEND = "pypdf"
except ImportError:
    pypdf = None               # type: ignore

if _PDF_BACKEND is None:
    sys.exit("ERROR: Install at least one PDF library:\n  pip install pdfplumber")

# ── Optional progress bar ─────────────────────────────────────────────────────
try:
    from tqdm import tqdm      # type: ignore
    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StudyItem:
    """One subtopic entry under a topic in the study plan."""
    name: str
    notes: str = ""
    pages: list[int] = field(default_factory=list)


@dataclass
class StudyTopic:
    """A top-level topic with its list of subtopics."""
    topic: str
    subtopics: list[StudyItem] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# LLM stub  ← PLUG YOUR MODEL IN HERE
# ─────────────────────────────────────────────────────────────────────────────

def call_llm(prompt: str) -> str:
    """Call OpenAI and return the model's reply as a plain string."""
    from openai import OpenAI

    # Reads OPENAI_API_KEY from the environment automatically.
    # Set it once in your shell:
    #   Windows PowerShell : $env:OPENAI_API_KEY = "sk-..."
    #   Windows CMD        : set OPENAI_API_KEY=sk-...
    #   Mac / Linux        : export OPENAI_API_KEY=sk-...
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o-mini",          # change to "gpt-4o" for higher quality
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,              # low temperature → more consistent output
    )
    return response.choices[0].message.content


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 · PDF text extraction
# ─────────────────────────────────────────────────────────────────────────────

def load_pdf(path: str, max_pages: Optional[int] = None) -> list[tuple[int, str]]:
    """
    Return [(page_number_1indexed, raw_text), …].
    pdfplumber is preferred because it handles multi-column layouts and
    preserves whitespace structure better than pypdf.
    """
    pages: list[tuple[int, str]] = []

    if _PDF_BACKEND == "pdfplumber":
        with pdfplumber.open(path) as pdf:
            limit = min(len(pdf.pages), max_pages or len(pdf.pages))
            src = pdf.pages[:limit]
            if _HAS_TQDM:
                src = tqdm(src, desc="Extracting", unit="pg")
            for i, page in enumerate(src, start=1):
                pages.append((i, page.extract_text() or ""))
    else:
        reader = pypdf.PdfReader(path)
        limit = min(len(reader.pages), max_pages or len(reader.pages))
        src = range(limit)
        if _HAS_TQDM:
            src = tqdm(src, desc="Extracting", unit="pg")
        for i in src:
            pages.append((i + 1, reader.pages[i].extract_text() or ""))

    return pages


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 · Text cleaning
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _find_repeated_lines(pages: list[tuple[int, str]], threshold: int) -> set[str]:
    """
    Lines appearing on `threshold` or more pages are almost certainly
    running headers or footers — collect them for removal.
    """
    counts: Counter = Counter()
    for _, text in pages:
        seen: set[str] = set()
        for raw in text.splitlines():
            line = _normalize(raw)
            if line and line not in seen:
                counts[line] += 1
                seen.add(line)
    return {line for line, n in counts.items() if n >= threshold}


_PAGE_NUM_RE = re.compile(
    r"^[\s\-–—]*\d+[\s\-–—]*$"          # bare number
    r"|^page\s+\d+(\s+of\s+\d+)?$"      # "Page 3" / "Page 3 of 40"
    r"|^\d+\s*/\s*\d+$",                 # "3 / 40"
    re.IGNORECASE,
)


def clean_pages(
    pages: list[tuple[int, str]],
) -> list[tuple[int, list[str]]]:
    """
    Returns [(page_num, [cleaned_line, …]), …].

    Removes:
    • Lines repeated on ≥ threshold pages  → headers / footers
    • Lines that are only a page number
    • Empty lines after normalisation
    """
    threshold = max(3, len(pages) // 8)
    repeated = _find_repeated_lines(pages, threshold)

    result: list[tuple[int, list[str]]] = []
    for page_num, text in pages:
        lines: list[str] = []
        for raw in text.splitlines():
            line = _normalize(raw)
            if not line:
                continue
            if line in repeated:
                continue
            if _PAGE_NUM_RE.match(line):
                continue
            lines.append(line)
        result.append((page_num, lines))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 · Heading detection
# ─────────────────────────────────────────────────────────────────────────────

_BULLET_RE = re.compile(r"^[-•·*◦▸▹→➤➢○●►]\s+|^\d+[.)]\s+")


def _classify_line(line: str) -> tuple[bool, int]:
    """
    Return (is_heading, level).
    Level 1 = major topic; level 2 = subtopic; 0 = body text.

    Rules (in priority order):
    1. ALL CAPS ≥ 2 words, < 100 chars, no terminal period → level 1
    2. Chapter/Section/Lecture/Module prefix              → level 1
    3. Numbered "1  Title" or "1.2  Title" (double space) → level 1 or 2
    4. Short slug ending in colon "Binary Search:"        → level 2
    5. Title Case, 2–8 words, < 70 chars, no terminal "." → level 2
    Bullets are always body text, never headings.
    """
    s = line.strip()
    if not s or len(s) < 2:
        return False, 0

    # Bullets are content
    if _BULLET_RE.match(s):
        return False, 0

    words = s.split()
    n = len(words)

    # Rule 1 — ALL CAPS
    alpha = re.sub(r"[^A-Za-z ]", "", s)
    if alpha.isupper() and alpha.strip() and n >= 2 and len(s) < 100 and not s.endswith("."):
        return True, 1

    # Rule 2 — Chapter / Section / … prefix
    if re.match(r"^(chapter|section|part|unit|module|lecture)\s+\d+", s, re.I):
        return True, 1

    # Rule 3 — Numbered heading with double-space separator "1  Intro"
    m = re.match(r"^(\d+)(\.\d+)?\s{2,}\S", s)
    if m:
        return True, 1 if not m.group(2) else 2

    # Rule 4 — Slug ending with colon
    if re.match(r"^[A-Z][A-Za-z0-9 /\-]{1,50}:\s*$", s) and n <= 7:
        return True, 2

    # Rule 5 — Title Case
    cap_ratio = sum(1 for w in words if w and w[0].isupper()) / n
    if cap_ratio >= 0.70 and 2 <= n <= 8 and len(s) < 70 and not s.endswith("."):
        return True, 2

    return False, 0


@dataclass
class _Token:
    text: str
    page: int
    is_heading: bool = False
    heading_level: int = 0


def tokenize(cleaned: list[tuple[int, list[str]]]) -> list[_Token]:
    tokens: list[_Token] = []
    for page_num, lines in cleaned:
        for line in lines:
            is_h, lvl = _classify_line(line)
            tokens.append(_Token(text=line, page=page_num, is_heading=is_h, heading_level=lvl))
    return tokens


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 · Content bucketing
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _Bucket:
    heading: str
    level: int
    parent: str           # level-1 heading that owns this bucket (if level == 2)
    lines: list[str] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)


def bucket_tokens(tokens: list[_Token]) -> list[_Bucket]:
    """
    Walk tokens in order.  Every heading opens a new bucket; body lines
    go into the current bucket.  Empty buckets (no body lines) are dropped.
    """
    buckets: list[_Bucket] = []
    current: Optional[_Bucket] = None
    current_l1 = "General"

    def flush():
        nonlocal current
        if current and current.lines:
            buckets.append(current)
        current = None

    for tok in tokens:
        if tok.is_heading:
            flush()
            if tok.heading_level == 1:
                current_l1 = tok.text
                current = _Bucket(heading=tok.text, level=1, parent="", pages=[tok.page])
            else:
                current = _Bucket(heading=tok.text, level=2, parent=current_l1, pages=[tok.page])
        else:
            if current is None:
                current = _Bucket(heading=current_l1, level=1, parent="", pages=[tok.page])
            current.lines.append(tok.text)
            if tok.page not in current.pages:
                current.pages.append(tok.page)

    flush()
    return buckets


# ─────────────────────────────────────────────────────────────────────────────
# Step 5a · Pull useful sentences from a bucket
# ─────────────────────────────────────────────────────────────────────────────

_DEF_RE = re.compile(
    r"\b(is defined as|refers to|is an?|are an?|means|denotes|"
    r"is called|is known as|can be defined|is the process of)\b",
    re.IGNORECASE,
)
_INST_RE = re.compile(
    r"\b(define|explain|describe|understand|apply|compute|calculate|"
    r"compare|contrast|distinguish|identify|analyse|analyze|evaluate|"
    r"recall|recognize|implement|derive|prove|solve|construct)\b",
    re.IGNORECASE,
)
_STEP_RE = re.compile(
    r"^(step\s+\d+|first|second|third|fourth|finally|next|then|"
    r"lastly|subsequently|afterwards)\b",
    re.IGNORECASE,
)


def _is_useful(line: str, min_len: int = 4) -> bool:
    words = line.split()
    if len(words) < min_len or len(words) > 80:
        return False
    if _DEF_RE.search(line):
        return True
    if _INST_RE.search(line):
        return True
    if _STEP_RE.match(line):
        return True
    # Bullet points with meaningful content
    if _BULLET_RE.match(line) and len(words) >= min_len:
        return True
    return False


def _pull_sentences(bucket: _Bucket, min_len: int) -> list[str]:
    good = [l for l in bucket.lines if _is_useful(l, min_len)]
    # Fallback: if nothing qualifies, use all lines long enough
    if not good:
        good = [l for l in bucket.lines if len(l.split()) >= max(min_len, 6)]
    return good


# ─────────────────────────────────────────────────────────────────────────────
# Step 5b · Deduplication
# ─────────────────────────────────────────────────────────────────────────────

def _jaccard(a: str, b: str) -> float:
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _deduplicate(sentences: list[str], threshold: float = 0.55) -> list[str]:
    unique: list[str] = []
    for s in sentences:
        if all(_jaccard(s, u) < threshold for u in unique):
            unique.append(s)
    return unique


# ─────────────────────────────────────────────────────────────────────────────
# Step 5c · Micro-skill naming
# ─────────────────────────────────────────────────────────────────────────────

_VERB_MAP: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(is defined as|refers to|means|is called|denotes)\b", re.I), "Can explain"),
    (re.compile(r"\b(steps?|process|algorithm|procedure|how to|method)\b", re.I), "Can apply"),
    (re.compile(r"\b(compare|contrast|differ|vs\.?|versus|difference between)\b", re.I), "Can distinguish"),
    (re.compile(r"\b(calculate|compute|derive|solve|find|determine)\b", re.I), "Can compute"),
    (re.compile(r"\b(prove|show that|demonstrate|verify)\b", re.I), "Can prove"),
    (re.compile(r"\b(identify|recognize|classify|categorize)\b", re.I), "Can identify"),
    (re.compile(r"\b(example|instance|application|use case)\b", re.I), "Can apply"),
]
_DEFAULT_VERB = "Can explain"


def _pick_verb(text: str) -> str:
    for pattern, verb in _VERB_MAP:
        if pattern.search(text):
            return verb
    return _DEFAULT_VERB


def _strip_bullet(text: str) -> str:
    return _BULLET_RE.sub("", text).strip()


_SKIP_INTRO_RE = re.compile(
    r"^(understand|define|explain|apply|compute|describe|analyze|"
    r"compare|identify|show|prove|find|calculate|use|know|learn|"
    r"study|introduce|recall)\b",
    re.IGNORECASE,
)


def _topic_phrase(text: str, heading: str) -> str:
    """
    Extract a short noun phrase that names the concept.
    Priority: subject of a "X is/are/means Y" sentence.
    Fallback: first 3–4 content words.
    """
    clean = _strip_bullet(text)

    # "X is/are/refers/means …" — grab X
    m = re.match(
        r"^([A-Za-z][A-Za-z0-9 \-]{1,50?}?)\s+\b(is|are|refers|means|denotes)\b",
        clean,
        re.I,
    )
    if m:
        return m.group(1).strip()

    words = clean.split()
    if _SKIP_INTRO_RE.match(clean):
        words = words[1:]

    phrase = " ".join(words[:4]).rstrip(".,;:")
    if len(phrase.split()) < 2:
        phrase = heading.rstrip(".:")
    return phrase


def _make_name(verb: str, text: str, heading: str) -> str:
    phrase = _topic_phrase(text, heading).lower()
    return f"{verb} {phrase}"


# ─────────────────────────────────────────────────────────────────────────────
# Step 5d · Keyword extraction
# ─────────────────────────────────────────────────────────────────────────────

_STOP = frozenset(
    "a an the is are was were be been being have has had do does did will would "
    "could should may might shall can need ought to of in on at by for with about "
    "against between through during before after above below from up down out off "
    "over under again further then once here there when where why how all both each "
    "few more most other some such no nor not only own same so than too very just "
    "but and or if as it its itself they them their what which who whom this that "
    "these those am into while we our you your i my we".split()
)


def _keywords(text: str, heading: str = "", n: int = 5) -> list[str]:
    raw = re.findall(r"[a-zA-Z][a-zA-Z'\-]{2,}", (text + " " + heading).lower())
    tokens = [w for w in raw if w not in _STOP]
    return [w for w, _ in Counter(tokens).most_common(n)]


# ─────────────────────────────────────────────────────────────────────────────
# Step 5e · Check question generation
# ─────────────────────────────────────────────────────────────────────────────

def _questions(name: str, kws: list[str]) -> list[str]:
    """
    Template-based question generation keyed on the micro-skill verb.
    Returns 1–2 short questions.
    """
    kw  = kws[0] if kws else name.split()[-1]
    kw2 = kws[1] if len(kws) > 1 else None
    lo  = name.lower()

    if "can explain" in lo or "can describe" in lo:
        q1 = f"In your own words, what is {kw}?"
        q2 = f"How does {kw} relate to {kw2}?" if kw2 else f"What is the main purpose of {kw}?"

    elif "can apply" in lo:
        q1 = f"Describe the steps involved in {kw}."
        q2 = f"Give one real-world example of {kw}."

    elif "can compute" in lo or "can calculate" in lo:
        q1 = f"Walk through how you would compute {kw}."
        q2 = f"What inputs does {kw} require, and what does it output?"

    elif "can distinguish" in lo or "can identify" in lo:
        q1 = (
            f"What is the key difference between {kw} and {kw2}?"
            if kw2 else f"How would you recognise {kw} in a problem?"
        )
        q2 = f"Give an example that illustrates {kw}."

    elif "can prove" in lo:
        q1 = f"What is the key insight required to prove {kw}?"
        q2 = f"What assumptions are needed for {kw}?"

    else:
        q1 = f"What is {kw}?"
        q2 = f"Why is {kw} important in this context?"

    return [q1, q2]


# ─────────────────────────────────────────────────────────────────────────────
# LLM-mode extraction (replaces step 5 entirely)
# ─────────────────────────────────────────────────────────────────────────────

def _llm_study_plan(
    cleaned: list[tuple[int, list[str]]],
) -> list[StudyTopic]:
    """
    Send the entire PDF text in ONE call and ask the LLM to return a study
    plan: a list of topics, each with its subtopics from the slides.

    Each page is labelled [Page N] so the model can reference page numbers.
    gpt-4o-mini has a 128k-token context window — easily fits a 40-page PDF.
    """
    page_blocks: list[str] = []
    for page_num, lines in cleaned:
        if lines:
            page_blocks.append(f"[Page {page_num}]\n" + "\n".join(lines))
    full_text = "\n\n".join(page_blocks)

    prompt = f"""You are an expert educator. I will give you the full text of a set of lecture slides.
Your job is to extract a clear, structured study plan from them.

The text below comes from a PDF, with each page labelled [Page N].

---
{full_text}
---

Identify the main topics covered in these slides. For each topic, list the subtopics
or specific concepts that were taught under it. Write a brief note (1–2 sentences)
for each subtopic summarising what it covers. Include the page numbers where it appears
if you can tell.

Return ONLY a valid JSON object in exactly this shape — no other text, no markdown fences:
{{
  "topics": [
    {{
      "topic": "Main topic name",
      "subtopics": [
        {{
          "name": "Subtopic or concept name",
          "notes": "1–2 sentence summary of what this covers.",
          "pages": [1, 2]
        }}
      ]
    }}
  ]
}}"""

    raw = call_llm(prompt)

    try:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        data = json.loads(m.group()) if m else {}
    except (json.JSONDecodeError, AttributeError):
        data = {}

    topics: list[StudyTopic] = []
    for t_data in data.get("topics", []):
        topic = StudyTopic(topic=t_data.get("topic", "Unknown"))
        for st in t_data.get("subtopics", []):
            if not st.get("name"):
                continue
            topic.subtopics.append(StudyItem(
                name=st.get("name", ""),
                notes=st.get("notes", ""),
                pages=st.get("pages", []),
            ))
        if topic.subtopics:
            topics.append(topic)

    return topics


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 · Process one bucket → list[Concept]
# ─────────────────────────────────────────────────────────────────────────────

def process_bucket(
    bucket: _Bucket,
    min_concept_len: int = 4,
    max_concepts: int = 8,
) -> list[StudyItem]:
    """Heuristic fallback: convert one bucket into StudyItem list."""
    sentences = _deduplicate(_pull_sentences(bucket, min_concept_len))[:max_concepts]

    items: list[StudyItem] = []
    for sentence in sentences:
        clean = _strip_bullet(sentence)
        if len(clean.split()) < min_concept_len:
            continue
        phrase = _topic_phrase(clean, bucket.heading)
        notes  = clean[:300]
        if notes and notes[-1] not in ".!?":
            notes += "."
        items.append(StudyItem(
            name=phrase,
            notes=notes,
            pages=sorted(set(bucket.pages)),
        ))

    return items


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 · Build Topic → Subtopic → Concept tree
# ─────────────────────────────────────────────────────────────────────────────

def build_topic_tree(
    buckets: list[_Bucket],
    min_concept_len: int = 4,
    max_concepts: int = 8,
) -> list[StudyTopic]:
    """Heuristic fallback: assemble buckets into a StudyTopic list."""
    topics: dict[str, StudyTopic] = {}

    for bucket in buckets:
        items = process_bucket(bucket, min_concept_len, max_concepts)
        if not items:
            continue

        topic_name = bucket.parent if bucket.level == 2 else bucket.heading
        if topic_name not in topics:
            topics[topic_name] = StudyTopic(topic=topic_name)

        topics[topic_name].subtopics.extend(items)

    return list(topics.values())


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 · Write concepts.json
# ─────────────────────────────────────────────────────────────────────────────

def write_json(topics: list[StudyTopic], source: str, num_pages: int, path: str) -> None:
    payload = {
        "source_pdf": os.path.basename(source),
        "num_pages_used": num_pages,
        "topics": [
            {
                "topic": t.topic,
                "subtopics": [asdict(s) for s in t.subtopics],
            }
            for t in topics
        ],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# Step 8 · Write concepts.md
# ─────────────────────────────────────────────────────────────────────────────

def write_markdown(topics: list[StudyTopic], source: str, num_pages: int, path: str) -> None:
    out: list[str] = [
        f"# Study Plan: {os.path.basename(source)}",
        "",
        f"_Pages processed: {num_pages}_",
        "",
    ]
    for ti, topic in enumerate(topics, 1):
        out.append(f"## {ti}. {topic.topic}")
        out.append("")
        for st in topic.subtopics:
            page_str = f" _(p. {', '.join(str(p) for p in st.pages)})_" if st.pages else ""
            if st.notes:
                out.append(f"- **{st.name}** — {st.notes}{page_str}")
            else:
                out.append(f"- **{st.name}**{page_str}")
        out.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a lecture PDF into a structured concept map."
    )
    parser.add_argument("--input",           required=True,
                        help="Path to the PDF file.")
    parser.add_argument("--outdir",          default="out",
                        help="Output directory (default: out).")
    parser.add_argument("--max_pages",       type=int, default=None,
                        help="Stop after this many pages.")
    parser.add_argument("--min_concept_len", type=int, default=4,
                        help="Minimum word count for a concept sentence (default: 4).")
    parser.add_argument("--max_concepts",    type=int, default=8,
                        help="Max concepts extracted per section (default: 8).")
    parser.add_argument("--llm",             action="store_true",
                        help="Use LLM mode via call_llm() stub.")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        sys.exit(f"ERROR: File not found: {args.input}")

    os.makedirs(args.outdir, exist_ok=True)

    print(f"PDF backend : {_PDF_BACKEND}")
    print(f"LLM mode    : {'ON' if args.llm else 'OFF — using heuristics'}")
    print()

    # 1. Load
    print("[1/6] Extracting text from PDF…")
    pages = load_pdf(args.input, args.max_pages)
    print(f"      {len(pages)} page(s) read.")

    # 2. Clean
    print("[2/6] Cleaning (removing headers/footers, page numbers)…")
    cleaned = clean_pages(pages)
    total_lines = sum(len(ls) for _, ls in cleaned)
    print(f"      {total_lines} lines retained after cleaning.")

    # 3. Tokenise + classify
    print("[3/6] Detecting headings…")
    tokens = tokenize(cleaned)
    n_headings = sum(1 for t in tokens if t.is_heading)
    print(f"      {n_headings} heading(s) found.")

    # 4. Bucket
    print("[4/6] Bucketing content under headings…")
    buckets = bucket_tokens(tokens)
    print(f"      {len(buckets)} content bucket(s).")

    # 5 & 6. Extract + assemble tree
    if args.llm:
        print("[5/6] Extracting study plan (1 LLM call for the full PDF)…")
        topics = _llm_study_plan(cleaned)
    else:
        print("[5/6] Extracting concepts (heuristics)…")
        topics = build_topic_tree(
            buckets,
            min_concept_len=args.min_concept_len,
            max_concepts=args.max_concepts,
        )
    total_subtopics = sum(len(t.subtopics) for t in topics)
    print(f"      {len(topics)} topic(s), {total_subtopics} subtopic(s).")

    # 7 & 8. Write outputs
    print("[6/6] Writing outputs…")
    json_path = os.path.join(args.outdir, "concepts.json")
    md_path   = os.path.join(args.outdir, "concepts.md")
    write_json(topics, args.input, len(pages), json_path)
    write_markdown(topics, args.input, len(pages), md_path)

    # Summary
    sep = "─" * 58
    print(f"\n{sep}")
    print(f"  Pages processed : {len(pages)}")
    print(f"  Topics found    : {len(topics)}")
    print(f"  Subtopics total : {total_subtopics}")
    print(sep)
    print(f"  {os.path.relpath(json_path)}")
    print(f"  {os.path.relpath(md_path)}")
    print(sep)

    if total_subtopics == 0:
        print(
            "\n[WARN] No concepts extracted.\n"
            "  • The PDF may be scanned (image-only) — OCR not supported here.\n"
            "  • Try lowering --min_concept_len (e.g. 3).\n"
            "  • Check that the PDF has selectable text.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
