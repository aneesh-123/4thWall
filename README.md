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

---

## Persistent Learner Memory (Supermemory)

The server includes a **conversational tutor** endpoint that maintains a persistent
learner profile in [Supermemory](https://supermemory.ai). After each turn the tutor
extracts concept mastery signals, updates the learner model, and conditions the next
response on what the learner knows and struggles with.

### New files

```
4thWall/
├── server.py              ← modified – added POST /chat/turn
├── supermemory_client.py  ← Supermemory SDK wrapper (get/put profile, add event)
├── memory_extract.py      ← LLM-powered turn → structured JSON extraction
├── memory_update.py       ← Deterministic EMA mastery update + profile rebuild
├── demo_memory.py         ← 5-turn demo script (run against live server)
└── tests/
    ├── test_mastery_update.py    ← mastery math + profile mutation tests
    ├── test_schema_validation.py ← validate_extract + heuristic fallback tests
    └── test_memory_client.py     ← MemoryClient with mocked Supermemory SDK
```

### Required environment variables

Create a `.env` file in the repo root (or export these in your shell):

```dotenv
# OpenAI (required for /chat/turn and /api/extract)
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini        # optional, default gpt-4o-mini

# Supermemory (required for /chat/turn)
SUPERMEMORY_API_KEY=sm-...

# Tuning (all optional)
MASTERY_ALPHA=0.2               # EMA learning rate 0-1
OPENAI_CHAT_CONTEXT_TURNS=8     # not used yet; reserved for multi-turn history
```

### Run

```bash
# From the repo root (with venv active)
cd 4thWall
python server.py
```

Server starts at **http://localhost:5000**.

### POST /chat/turn

```bash
curl -s -X POST http://localhost:5000/chat/turn \
  -H "Content-Type: application/json" \
  -d '{
    "message": "How does binary search work?",
    "outcome_signal": null
  }' | python -m json.tool
```

Response:

```json
{
  "learner_id": "a3f9d2c1-...",
  "session_id": "b7e1a4f2-...",
  "assistant": "Great question! Binary search works by...\n\nCheck question: ...",
  "memory_preview": {
    "weak_concepts": [
      {"concept_id": "binary_search", "mastery": 0.2}
    ],
    "misconceptions": [],
    "preferences": {"style": "direct", "verbosity": "medium", "pace": "medium"}
  }
}
```

Pass `learner_id` and `session_id` from the response back on subsequent turns to maintain continuity:

```bash
curl -s -X POST http://localhost:5000/chat/turn \
  -H "Content-Type: application/json" \
  -d '{
    "learner_id": "a3f9d2c1-...",
    "session_id": "b7e1a4f2-...",
    "message": "I understand now — the midpoint is (low+high)//2.",
    "outcome_signal": "correct"
  }' | python -m json.tool
```

After a `correct` turn you will see `binary_search.mastery` rise in `memory_preview`.

### Run the demo

```bash
# With server running in another terminal
python 4thWall/demo_memory.py
```

Runs 5 scripted turns (wrong → improving → correct) and prints the mastery bars
evolving turn-by-turn. You should observe:

1. `binary_search` mastery starts at 0, dips on `incorrect`, climbs on `correct`
2. `preferences.style` changes to `worked_examples` after Turn 4
3. Tutor response in Turn 2 addresses the linear-search misconception
4. Tutor response in Turn 4 contains a step-by-step example

### Run tests

```bash
cd 4thWall
python -m pytest tests/ -v
```

Tests do **not** require any API keys — the Supermemory SDK and OpenAI are fully mocked.

### Mastery model

```
score = 1.0 (correct) | 0.5 (partial) | 0.25 (confused) | 0.0 (incorrect)
mastery_new = mastery_old + α × (score − mastery_old)    α = MASTERY_ALPHA=0.2
mastery clamped to [0.0, 1.0]
```

### Supermemory data layout

| Document tag | Content | One per |
|---|---|---|
| `profile:{learner_id}` | Full profile JSON | learner |
| `event:{learner_id}` | Single turn summary JSON | turn |
| `learner:{learner_id}` | Both (broad container) | — |
