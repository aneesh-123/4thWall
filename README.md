# AI Khan Academy — PDF → Concept Map

Converts a lecture PDF into a structured, mastery-based concept map without any external API calls.

---

## Install

```bash
pip install -r requirements.txt
```

Python 3.10+ required.

---

## Run

```bash
python pdf_to_concepts.py --input myslides.pdf --outdir out
```

| Flag | Default | Description |
|---|---|---|
| `--input` | *(required)* | Path to the PDF |
| `--outdir` | `out` | Folder to write output files |
| `--max_pages` | all | Stop after N pages |
| `--min_concept_len` | `4` | Min words for a sentence to count as a concept |
| `--max_concepts` | `8` | Max concepts extracted per section |
| `--llm` | off | Use LLM mode (see below) |

---

## Outputs

| File | Description |
|---|---|
| `out/concepts.json` | Full structured concept map (Topics → Subtopics → Concepts) |
| `out/concepts.md` | Human-readable outline with keywords and check questions |

### concepts.json schema

```json
{
  "source_pdf": "myslides.pdf",
  "num_pages_used": 40,
  "topics": [
    {
      "topic": "Sorting Algorithms",
      "subtopics": [
        {
          "subtopic": "Merge Sort",
          "concepts": [
            {
              "name": "Can explain merge sort",
              "description": "Merge sort is a divide-and-conquer algorithm...",
              "keywords": ["merge", "sort", "divide", "conquer", "recursive"],
              "check_questions": [
                "In your own words, what is merge sort?",
                "What is the main purpose of merge sort?"
              ],
              "pages": [4, 5]
            }
          ]
        }
      ]
    }
  ]
}
```

---

## How the heuristics work

```
PDF pages
    │
    ▼
pdfplumber / pypdf  →  raw text per page
    │
    ▼
clean_pages
  • Find lines appearing on ≥ (n_pages / 8) pages → header/footer → remove
  • Remove bare page numbers ("3", "Page 3 of 40", "3 / 40")
  • Collapse whitespace
    │
    ▼
_classify_line  →  heading level 1 / 2 / body
  1. ALL CAPS ≥ 2 words         → topic (level 1)
  2. Chapter/Section/Lecture N  → topic (level 1)
  3. "1  Title" double-space    → level 1 or 2
  4. Short slug ending in ":"   → subtopic (level 2)
  5. Title Case, 2–8 words      → subtopic (level 2)
    │
    ▼
bucket_tokens  →  body lines grouped under nearest heading
    │
    ▼
process_bucket  (per bucket)
  • _pull_sentences: keep bullet points, definition sentences,
    step sequences; fall back to any line ≥ min_concept_len words
  • _deduplicate: drop near-duplicates (Jaccard ≥ 0.55)
  • _pick_verb: match text patterns → "Can explain / apply /
    distinguish / compute / prove / identify"
  • _make_name: extract noun phrase from sentence → micro-skill name
  • _keywords: frequency-count non-stopword tokens
  • _questions: fill template questions keyed on the verb
    │
    ▼
build_topic_tree  →  Topic → Subtopic → Concept hierarchy
    │
    ▼
concepts.json + concepts.md
```

---

## Plugging in an LLM later

Open `pdf_to_concepts.py` and find `call_llm()` near the top.  Replace the body with your API call:

```python
# OpenAI
from openai import OpenAI
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
def call_llm(prompt: str) -> str:
    r = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )
    return r.choices[0].message.content
```

Then run with:
```bash
python pdf_to_concepts.py --input myslides.pdf --llm
```

In LLM mode the heuristic pipeline is bypassed; each section is sent as a structured prompt and the model returns a JSON array of concepts directly.

---

## Tuning knobs

| Problem | Fix |
|---|---|
| Too few concepts | Lower `--min_concept_len` (e.g. `3`) or `--max_concepts 12` |
| Too many noisy concepts | Raise `--min_concept_len` (e.g. `6`) |
| Headings not detected | PDF may use unusual fonts/sizes — check raw text with `--max_pages 1` and look at output |
| No text at all | PDF is likely scanned (image-only) — OCR required (e.g. `ocrmypdf`) |
