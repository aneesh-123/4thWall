"""
server.py — Flask web server for the PDF → Study Plan tool.

Routes:
  GET  /              Serve the upload UI.
  POST /api/extract   Accept a PDF, call OpenAI, return the study plan.
  POST /chat/turn     Conversational tutor with persistent Supermemory.
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile
import uuid
from datetime import datetime, timezone

# Load .env manually so Windows env-var limits / invalid-arg don't crash the server
def _load_dotenv_safe():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.isfile(env_path):
        return
    try:
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key, value = key.strip(), value.strip().strip('"').strip("'")
                if not key or "\x00" in value:
                    continue
                try:
                    os.environ[key] = value
                except OSError:
                    pass  # Windows can reject long or certain values; skip
    except Exception:
        pass

_load_dotenv_safe()

from flask import Flask, jsonify, render_template, request
from openai import OpenAI

sys.path.insert(0, os.path.dirname(__file__))
from pdf_to_concepts import (
    _llm_study_plan,
    clean_pages,
    load_pdf,
    write_markdown,
)
from supermemory_client import get_memory_client
from memory_extract import (
    build_profile_context_string,
    extract_turn_insights,
)
from memory_update import apply_extract_to_profile, build_memory_preview, update_mastery
from pdf_retriever import build_index, retrieve, format_retrieved_chunks
from recommendation_engine import recommend_next_for_chat_turn, get_recommendations_dict
from grading_engine import grade as grade_submission
from hint_engine import generate_hint, generate_all_hints, stream_hints, validate_hint_request
from test_engine import generate_test, grade_test, get_difficulty_level

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── LLM config ────────────────────────────────────────────────────────────────
_OPENAI_MODEL    = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
_OPENAI_CHAT_CTX = int(os.getenv("OPENAI_CHAT_CONTEXT_TURNS", "8"))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024   # 50 MB limit

# ── In-memory PDF document store ──────────────────────────────────────────────
# Keyed by doc_id (UUID). Holds the study-plan markdown and filename so the
# chat tutor can answer questions specifically about the uploaded document.
# This is process-local; fine for single-server MVP.
_doc_store: dict[str, dict] = {}

# ── Shared-session store ──────────────────────────────────────────────────────
# Maps a short human-readable code → learner_id (UUID).
# All browser tabs that join with the same code share that learner profile,
# so the whole class / study group can build one knowledge graph together.
_session_store: dict[str, str] = {}


def _make_session_code() -> str:
    """Generate a unique 6-char uppercase alphanumeric session code."""
    import random, string
    alphabet = string.ascii_uppercase + string.digits
    while True:
        code = "".join(random.choices(alphabet, k=6))
        if code not in _session_store:
            return code


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/extract", methods=["POST"])
def extract():
    if "pdf" not in request.files:
        return jsonify({"error": "No PDF file provided."}), 400

    pdf_file = request.files["pdf"]

    if not pdf_file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Uploaded file must be a PDF."}), 400

    tmp_pdf = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp_md  = tempfile.NamedTemporaryFile(suffix=".md",  delete=False, mode="w", encoding="utf-8")
    tmp_pdf.close()
    tmp_md.close()

    try:
        pdf_file.save(tmp_pdf.name)

        pages   = load_pdf(tmp_pdf.name)
        cleaned = clean_pages(pages)
        topics  = _llm_study_plan(cleaned)

        write_markdown(topics, pdf_file.filename, len(pages), tmp_md.name)
        with open(tmp_md.name, encoding="utf-8") as f:
            md_content = f.read()

        total_subtopics = sum(len(t.subtopics) for t in topics)

        # Store study plan + TF-IDF index for the chat tutor
        doc_id          = str(uuid.uuid4())
        retrieval_index = build_index(cleaned)   # reuse already-cleaned pages
        _doc_store[doc_id] = {
            "doc_id":          doc_id,
            "filename":        pdf_file.filename,
            "markdown":        md_content,
            "pages":           len(pages),
            "retrieval_index": retrieval_index,
            "sm_uploaded":     False,   # set to True after Supermemory upload
        }
        logger.info("Stored doc_id=%s filename=%s chunks=%d",
                    doc_id, pdf_file.filename,
                    len(retrieval_index.get("chunks", [])))

        # Upload PDF to Supermemory for persistent semantic search (Super RAG).
        # Must happen before the finally-block deletes the temp file.
        try:
            mem_client = get_memory_client()
            mem_client.upload_pdf(doc_id, tmp_pdf.name, pdf_file.filename)
            _doc_store[doc_id]["sm_uploaded"] = True
        except Exception as _upload_exc:
            logger.warning("Supermemory PDF upload skipped: %s", _upload_exc)

        return jsonify({
            "doc_id":          doc_id,
            "markdown":        md_content,
            "pages_processed": len(pages),
            "topics_count":    len(topics),
            "subtopics_count": total_subtopics,
            "sm_uploaded":     _doc_store[doc_id]["sm_uploaded"],
        })

    except NotImplementedError:
        return jsonify({
            "error": (
                "OpenAI API key not configured.\n"
                "Set OPENAI_API_KEY in your environment and restart the server."
            )
        }), 500

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    finally:
        for path in (tmp_pdf.name, tmp_md.name):
            try:
                os.unlink(path)
            except OSError:
                pass


# ── Tutor prompt builder ──────────────────────────────────────────────────────

_TUTOR_SYSTEM_TEMPLATE = """\
You are an expert adaptive tutor. Your goal is to help the learner develop genuine
understanding—not just provide answers.
{doc_block}
LEARNER PROFILE (do NOT reveal this verbatim to the learner):
{profile_block}

INSTRUCTIONS (follow these carefully every turn):
1. {scope_instruction}
2. Adjust explanation **style** to match preferences:
   - worked_examples  → walk through a concrete step-by-step example
   - analogy          → use a memorable real-world analogy
   - socratic         → guide with targeted questions rather than lecturing
   - direct           → concise, factual answer with no extra framing
3. Match **verbosity** (low → ≤3 sentences; medium → ~1 paragraph; high → detailed).
4. If the learner mentions a **weak or prerequisite concept** (mastery < 0.50),
   insert a 1–2 sentence recap before the main answer.
5. If a **known misconception** is relevant, proactively address or pre-empt it
   in 1 sentence; do not label it as a "misconception" to the learner.
6. End EVERY response with exactly ONE targeted check-for-understanding question
   appropriate to the learner's current level—unless the learner explicitly asked
   for a direct answer only.
7. Be warm, encouraging, and specific about what the learner got right or wrong.
"""

_DOC_BLOCK_TEMPLATE = """
DOCUMENT CONTEXT — "{filename}":

STUDY PLAN (outline extracted from the PDF):
{markdown}

--- RETRIEVED PASSAGES (TF-IDF lexical search) ---
{passages}

--- RETRIEVED PASSAGES (Supermemory semantic search) ---
{semantic_passages}

(End of document context — base ALL answers strictly on the above content.
 Cite the page numbers from TF-IDF excerpts where possible.)
"""

_SCOPE_WITH_DOC    = "Answer ONLY questions related to the document above. If the learner asks about something outside it, gently redirect them back to the document."
_SCOPE_WITHOUT_DOC = "Answer questions on any topic the learner brings up."


def _build_tutor_system_prompt(
    profile: dict,
    doc: dict | None = None,
    query: str = "",
    semantic_passages: str = "",
) -> str:
    profile_block = build_profile_context_string(profile) or "(first session — no prior data)"
    if doc:
        # TF-IDF lexical retrieval (local, synchronous)
        passages = ""
        idx = doc.get("retrieval_index")
        if idx and query:
            hits = retrieve(idx, query, top_k=5)
            passages = format_retrieved_chunks(hits, max_chars=2500)
            logger.debug("TF-IDF: %d chunks for query: %s", len(hits), query[:80])
        if not passages:
            passages = "(no TF-IDF passages retrieved)"

        # Supermemory semantic passages (passed in from the caller)
        if not semantic_passages:
            semantic_passages = "(no semantic passages retrieved — PDF may still be indexing)"

        # Study-plan overview (capped; detail comes from retrieved passages)
        md_overview = doc["markdown"][:1200]
        if len(doc["markdown"]) > 1200:
            md_overview += "\n…(full outline available via retrieved passages)"

        doc_block = _DOC_BLOCK_TEMPLATE.format(
            filename=doc["filename"],
            markdown=md_overview,
            passages=passages,
            semantic_passages=semantic_passages,
        )
        scope_instruction = _SCOPE_WITH_DOC
    else:
        doc_block = ""
        scope_instruction = _SCOPE_WITHOUT_DOC
    return _TUTOR_SYSTEM_TEMPLATE.format(
        doc_block=doc_block,
        profile_block=profile_block,
        scope_instruction=scope_instruction,
    )


def _generate_tutor_response(system_prompt: str, user_message: str) -> str:
    """Call OpenAI and return the tutor reply string."""
    client = OpenAI()
    resp = client.chat.completions.create(
        model=_OPENAI_MODEL,
        messages=[
            {"role": "system",  "content": system_prompt},
            {"role": "user",    "content": user_message},
        ],
        temperature=0.7,
    )
    return resp.choices[0].message.content or ""


# ── /chat/turn endpoint ───────────────────────────────────────────────────────

@app.route("/chat/turn", methods=["POST"])
def chat_turn():
    """
    POST /chat/turn
    Request JSON:
      {
        "learner_id":     "string optional",
        "session_id":     "string optional",
        "doc_id":         "string optional — from /api/extract response",
        "message":        "string required",
        "outcome_signal": "correct|partial|confused|incorrect optional"
      }
    Response JSON:
      {
        "learner_id":     "...",
        "session_id":     "...",
        "assistant":      "...",
        "memory_preview": { weak_concepts, misconceptions, preferences }
      }
    """
    body = request.get_json(silent=True) or {}

    message = (body.get("message") or "").strip()
    if not message:
        return jsonify({"error": "message is required"}), 400

    learner_id     = (body.get("learner_id") or "").strip() or str(uuid.uuid4())
    session_id     = (body.get("session_id") or "").strip() or str(uuid.uuid4())
    doc_id         = (body.get("doc_id") or "").strip() or None
    outcome_signal = (body.get("outcome_signal") or "").strip() or None

    # Look up PDF document context if provided
    doc_ctx = _doc_store.get(doc_id) if doc_id else None
    if doc_id and not doc_ctx:
        logger.warning("doc_id=%s not found in store (server may have restarted)", doc_id)

    logger.info(
        "chat/turn learner=%s session=%s doc_id=%s outcome_signal=%s msg_len=%d",
        learner_id, session_id, doc_id, outcome_signal, len(message),
    )

    # ── Step 1: load or initialise profile ───────────────────────────────────
    mem = get_memory_client()
    try:
        profile = mem.get_profile(learner_id)
    except Exception as exc:
        logger.error("Could not load profile for %s: %s", learner_id, exc)
        profile = {"learner_id": learner_id, "version": 1,
                   "preferences": {"style": "direct", "verbosity": "medium", "pace": "medium"},
                   "goals": [], "concepts": {}, "misconceptions": {},
                   "recent_summary": "", "updated_at": ""}

    # ── Step 2a: Supermemory semantic search over the uploaded PDF ─────────
    semantic_passages = ""
    if doc_ctx and doc_ctx.get("sm_uploaded"):
        try:
            sm_hits = mem.search_pdf(doc_id, message, top_k=5)
            if sm_hits:
                semantic_passages = "\n\n".join(
                    f"[Semantic chunk {i+1}]\n{chunk}"
                    for i, chunk in enumerate(sm_hits)
                )
                logger.info("Supermemory PDF search: %d chunks for doc_id=%s", len(sm_hits), doc_id)
        except Exception as _sm_exc:
            logger.warning("Supermemory PDF search failed: %s", _sm_exc)

    # ── Step 2b: build system prompt and generate LLM response ───────────
    system_prompt = _build_tutor_system_prompt(
        profile, doc=doc_ctx, query=message, semantic_passages=semantic_passages
    )
    try:
        assistant_reply = _generate_tutor_response(system_prompt, message)
    except Exception as exc:
        logger.error("Tutor LLM call failed: %s", exc)
        return jsonify({"error": f"LLM error: {exc}"}), 500

    logger.info("Tutor response generated (%d chars)", len(assistant_reply))

    # ── Step 3: extract learning signals from the turn ────────────────────────
    profile_ctx = build_profile_context_string(profile)
    try:
        extract = extract_turn_insights(
            user_message=message,
            tutor_message=assistant_reply,
            profile_context=profile_ctx,
            outcome_signal=outcome_signal,
        )
    except Exception as exc:
        logger.error("Extraction failed: %s", exc)
        extract = {
            "concepts": [], "misconceptions": [], "preferences": [], "goals": [],
            "outcome": {"label": "partial", "confidence": 0.3, "inferred": True},
            "notes": "extraction error",
        }

    logger.info(
        "Extracted: concepts=%s outcome=%s misc=%s prefs=%s",
        [c["concept_id"] for c in extract.get("concepts", [])],
        extract.get("outcome", {}).get("label"),
        [m["misconception_id"] for m in extract.get("misconceptions", [])],
        [(p["key"], p["value"]) for p in extract.get("preferences", [])],
    )

    # ── Step 4: update profile deterministically ──────────────────────────────
    updated_profile, deltas = apply_extract_to_profile(profile, extract)
    if deltas:
        logger.info("Mastery deltas: %s", deltas)

    # Brief recent summary (non-PII — just concept names + outcome)
    concept_ids = [c["concept_id"] for c in extract.get("concepts", [])]
    updated_profile["recent_summary"] = (
        f"Turn covered: {', '.join(concept_ids[:4]) or 'general topic'}. "
        f"Outcome: {extract['outcome']['label']}."
    )[:300]

    # ── Step 5: persist to Supermemory ────────────────────────────────────────
    try:
        mem.put_profile(learner_id, updated_profile)
    except Exception as exc:
        logger.error("put_profile failed: %s", exc)
        # Non-fatal — continue and return result

    event = {
        "ts":                    datetime.now(timezone.utc).isoformat(),
        "session_id":            session_id,
        "user_message_summary":  message[:200],
        "tutor_message_summary": assistant_reply[:200],
        "concepts":              [c["concept_id"] for c in extract.get("concepts", [])],
        "outcome":               extract["outcome"]["label"],
        "inferred_outcome":      extract["outcome"].get("inferred", True),
    }
    try:
        mem.add_event(learner_id, event)
    except Exception as exc:
        logger.error("add_event failed: %s", exc)
        # Non-fatal

    # ── Step 6: get recommendations for next concepts ────────────────────────
    current_concepts = [c["concept_id"] for c in extract.get("concepts", [])]
    try:
        recommendations = recommend_next_for_chat_turn(
            updated_profile,
            current_concept=current_concepts[0] if current_concepts else None,
            top_n=3,
        )
    except Exception as exc:
        logger.error("Recommendation failed: %s", exc)
        recommendations = {"next_concepts": [], "next_template": "lesson"}

    # ── Step 7: build response ────────────────────────────────────────────────
    memory_preview = build_memory_preview(updated_profile)

    return jsonify({
        "learner_id":     learner_id,
        "session_id":     session_id,
        "assistant":      assistant_reply,
        "memory_preview": memory_preview,
        "recommendations": recommendations,
        "doc_found":      doc_ctx is not None,
        "sm_grounded":    bool(semantic_passages),
    })


# ── Session endpoints ────────────────────────────────────────────────────────

@app.route("/api/session/create", methods=["POST"])
def session_create():
    """
    POST /api/session/create
    Creates a new shared session. Returns {code, learner_id}.
    All users who join with the same code share the same learner_id
    and therefore the same persistent memory / knowledge graph.
    """
    code       = _make_session_code()
    learner_id = str(uuid.uuid4())
    _session_store[code] = learner_id
    logger.info("Session created code=%s learner_id=%s", code, learner_id)
    return jsonify({"code": code, "learner_id": learner_id})


@app.route("/api/session/join/<code>")
def session_join(code: str):
    """
    GET /api/session/join/<code>
    Returns {code, learner_id} for an existing session, or 404.
    """
    code = code.upper().strip()
    learner_id = _session_store.get(code)
    if not learner_id:
        return jsonify({"error": f"Session '{code}' not found."}), 404
    logger.info("Session joined code=%s learner_id=%s", code, learner_id)
    return jsonify({"code": code, "learner_id": learner_id})


# ── Quiz endpoints ───────────────────────────────────────────────────────────

_QUIZ_GEN_PROMPT = """\
You are a quiz generator for an adaptive learning system.
Generate exactly {n} multiple-choice questions{topic_clause}.

Each question MUST:
- Test understanding or application, not just trivial recall.
- Have exactly 4 answer options (A–D) where only ONE is definitively correct.
- Include a `concept_id` (snake_case, 1–4 words) naming the core concept tested.
- Include a 1-2 sentence `explanation` of why the correct answer is right.

Return a JSON object in this exact shape:
{{"questions": [
  {{"question": "...", "options": ["...", "...", "...", "..."], "correct_index": 0,
    "concept_id": "snake_case", "explanation": "..."}},
  ...
]}}

CONTENT:
{content}
"""


@app.route("/api/quiz/generate", methods=["POST"])
def quiz_generate():
    """
    POST /api/quiz/generate
    Body: { doc_id?, topic?, n_questions? }
    Returns: { questions: [{question, options, correct_index, concept_id, explanation, id}] }
    """
    body   = request.get_json(silent=True) or {}
    doc_id = (body.get("doc_id") or "").strip() or None
    topic  = (body.get("topic") or "").strip() or None
    n      = max(1, min(int(body.get("n_questions", 5)), 10))

    doc_ctx = _doc_store.get(doc_id) if doc_id else None

    # Build content block for the LLM to write questions about
    if doc_ctx:
        if topic:
            hits    = retrieve(doc_ctx["retrieval_index"], topic, top_k=8)
            content = format_retrieved_chunks(hits, max_chars=4000)
            if not content.strip():
                content = doc_ctx["markdown"][:4000]
        else:
            content = doc_ctx["markdown"][:4000]
    elif topic:
        content = f"Topic: {topic}"
    else:
        return jsonify({"error": "Provide a doc_id or a topic to generate a quiz."}), 400

    topic_clause = f" specifically about '{topic}'" if topic else " based on the content below"
    prompt = _QUIZ_GEN_PROMPT.format(n=n, topic_clause=topic_clause, content=content)

    try:
        client = OpenAI()
        resp = client.chat.completions.create(
            model=_OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or "{}"
    except Exception as exc:
        return jsonify({"error": f"LLM error: {exc}"}), 500

    import json as _json
    try:
        parsed = _json.loads(raw)
        # Model returns {"questions": [...]}, but handle other shapes gracefully
        if isinstance(parsed, list):
            questions = parsed
        else:
            questions = (
                parsed.get("questions")
                or parsed.get("quiz")
                or next(iter(parsed.values()), [])
            )
        if not isinstance(questions, list):
            raise ValueError("no question array found")
    except Exception as exc:
        logger.error("Quiz parse error: %s  raw=%s", exc, raw[:200])
        return jsonify({"error": "Quiz generation failed — could not parse LLM output."}), 500

    for i, q in enumerate(questions):
        q["id"] = i

    logger.info("Quiz generated: %d Qs doc_id=%s topic=%s", len(questions), doc_id, topic)
    return jsonify({"questions": questions, "doc_id": doc_id, "topic": topic})


@app.route("/api/quiz/submit", methods=["POST"])
def quiz_submit():
    """
    POST /api/quiz/submit
    Body: { learner_id, answers: [{id, concept_id, question, options, selected_index, correct_index, explanation}] }
    Returns: { score, total, score_pct, overall_outcome, results, mastery_deltas, memory_preview }
    """
    body       = request.get_json(silent=True) or {}
    learner_id = (body.get("learner_id") or "").strip() or str(uuid.uuid4())
    answers    = body.get("answers", [])

    if not answers:
        return jsonify({"error": "No answers provided."}), 400

    # ── Load profile ──────────────────────────────────────────────────────────
    mem = get_memory_client()
    try:
        profile = mem.get_profile(learner_id)
    except Exception:
        profile = {
            "learner_id": learner_id, "version": 1, "goals": [],
            "preferences": {"style": "direct", "verbosity": "medium", "pace": "medium"},
            "concepts": {}, "misconceptions": {}, "concept_edges": {},
            "recent_summary": "", "updated_at": "",
        }

    # ── Score each answer ─────────────────────────────────────────────────────
    results: list[dict] = []
    # Map concept_id → outcome for the LAST answer on that concept
    concept_outcomes: dict[str, str] = {}
    n_correct = 0

    for ans in answers:
        cid       = (ans.get("concept_id") or "general").lower().replace(" ", "_")
        selected  = int(ans.get("selected_index", -1))
        correct   = int(ans.get("correct_index",  -1))
        is_right  = (selected == correct)
        if is_right:
            n_correct += 1
        concept_outcomes[cid] = "correct" if is_right else "incorrect"
        results.append({
            "id":             ans.get("id"),
            "question":       ans.get("question", ""),
            "options":        ans.get("options", []),
            "concept_id":     cid,
            "selected_index": selected,
            "correct_index":  correct,
            "correct":        is_right,
            "explanation":    ans.get("explanation", ""),
        })

    score_pct = n_correct / len(answers)
    if   score_pct >= 0.8: overall = "correct"
    elif score_pct >= 0.5: overall = "partial"
    elif score_pct >= 0.25: overall = "confused"
    else:                   overall = "incorrect"

    # ── Per-concept mastery update ────────────────────────────────────────────
    concepts_node = profile.setdefault("concepts", {})
    now           = datetime.now(timezone.utc).isoformat()
    mastery_before: dict[str, float] = {}
    mastery_after:  dict[str, float] = {}

    for cid, outcome in concept_outcomes.items():
        entry = concepts_node.get(cid)
        if entry is None:
            entry = {
                "mastery": 0.0, "attempts": 0, "correct_attempts": 0,
                "last_seen": now, "last_outcome": outcome, "error_pattern": "",
            }
            concepts_node[cid] = entry
        mastery_before[cid] = round(entry["mastery"], 3)
        entry["mastery"]          = update_mastery(entry["mastery"], outcome)
        entry["attempts"]        += 1
        entry["correct_attempts"] += (1 if outcome == "correct" else 0)
        entry["last_seen"]        = now
        entry["last_outcome"]     = outcome
        mastery_after[cid]        = round(entry["mastery"], 3)
        logger.info("Quiz mastery %-30s %s  %.3f → %.3f",
                    cid, outcome, mastery_before[cid], mastery_after[cid])

    profile["updated_at"]     = now
    profile["recent_summary"] = (
        f"Quiz: {n_correct}/{len(answers)} correct ({int(score_pct*100)}%). "
        f"Concepts: {', '.join(list(concept_outcomes.keys())[:4])}."
    )[:300]

    # ── Persist ───────────────────────────────────────────────────────────────
    try:
        mem.put_profile(learner_id, profile)
    except Exception as exc:
        logger.error("quiz put_profile failed: %s", exc)

    memory_preview = build_memory_preview(profile)
    logger.info("Quiz submitted learner=%s score=%d/%d overall=%s",
                learner_id, n_correct, len(answers), overall)

    return jsonify({
        "learner_id":      learner_id,
        "score":           n_correct,
        "total":           len(answers),
        "score_pct":       round(score_pct * 100),
        "overall_outcome": overall,
        "results":         results,
        "mastery_deltas":  {
            cid: {"before": mastery_before[cid], "after": mastery_after[cid]}
            for cid in concept_outcomes
        },
        "memory_preview":  memory_preview,
    })


# ── Knowledge-graph endpoint ─────────────────────────────────────────────────

@app.route("/api/knowledge-graph/<learner_id>")
def knowledge_graph(learner_id: str):
    """
    GET /api/knowledge-graph/<learner_id>
    Returns a D3-compatible graph: {nodes: [...], links: [...]}
    """
    mem = get_memory_client()
    try:
        profile = mem.get_profile(learner_id)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    preview = build_memory_preview(profile)
    graph   = preview.get("graph", {"nodes": [], "links": []})
    return jsonify({
        "learner_id": learner_id,
        "nodes":      graph["nodes"],
        "links":      graph["links"],
    })


# ── Grading endpoint ─────────────────────────────────────────────────────────

@app.route("/api/grade", methods=["POST"])
def grade_answer():
    """
    POST /api/grade
    Body: {
      learner_id?: string,
      question_text: string,
      topic: string,
      submission: string,
      mode?: "informal" | "formal" (default: "informal"),
      concept_context?: string
    }
    Returns: transparent grading result with rubric, per-criterion breakdown,
             outcome, sources, and mastery update.
    """
    body = request.get_json(silent=True) or {}

    submission    = (body.get("submission") or "").strip()
    question_text = (body.get("question_text") or "").strip()
    topic         = (body.get("topic") or "").strip()
    mode          = (body.get("mode") or "informal").strip().lower()
    learner_id    = (body.get("learner_id") or "").strip() or None
    concept_ctx   = (body.get("concept_context") or "").strip()

    if not submission:
        return jsonify({"error": "submission is required"}), 400
    if not question_text:
        return jsonify({"error": "question_text is required"}), 400
    if not topic:
        return jsonify({"error": "topic is required"}), 400
    if mode not in ("informal", "formal"):
        mode = "informal"

    # Load mastery context if learner_id provided
    mastery_ctx = ""
    if learner_id:
        mem = get_memory_client()
        try:
            profile = mem.get_profile(learner_id)
            mastery_ctx = build_profile_context_string(profile)
        except Exception:
            pass

    try:
        result = grade_submission(
            submission=submission,
            topic=topic,
            question_text=question_text,
            mode=mode,
            concept_context=concept_ctx,
            mastery_context=mastery_ctx,
        )
    except Exception as exc:
        logger.error("Grade endpoint failed: %s", exc)
        return jsonify({"error": f"Grading error: {exc}"}), 500

    # Update mastery if learner_id provided
    mastery_update = None
    if learner_id:
        outcome = result.get("outcome", "partial")
        concept_id = topic.lower().replace(" ", "_")
        try:
            mem = get_memory_client()
            profile = mem.get_profile(learner_id)
            concepts_node = profile.setdefault("concepts", {})
            entry = concepts_node.get(concept_id)
            now = datetime.now(timezone.utc).isoformat()
            if entry is None:
                entry = {
                    "mastery": 0.0, "attempts": 0, "correct_attempts": 0,
                    "last_seen": now, "last_outcome": outcome, "error_pattern": "",
                }
                concepts_node[concept_id] = entry
            before = entry["mastery"]
            entry["mastery"]          = update_mastery(entry["mastery"], outcome)
            entry["attempts"]        += 1
            entry["correct_attempts"] += (1 if outcome == "correct" else 0)
            entry["last_seen"]        = now
            entry["last_outcome"]     = outcome
            profile["updated_at"]     = now
            mem.put_profile(learner_id, profile)
            mastery_update = {
                concept_id: {"before": round(before, 3), "after": round(entry["mastery"], 3)}
            }
        except Exception as exc:
            logger.error("Grade mastery update failed: %s", exc)

    result["mastery_update"] = mastery_update
    logger.info("Graded: topic=%s mode=%s outcome=%s score=%.1f",
                topic, mode, result.get("outcome"), result.get("overall_score", 0))
    return jsonify(result)


# ── Recommendation endpoint ──────────────────────────────────────────────────

@app.route("/api/recommend", methods=["POST"])
def recommend():
    """
    POST /api/recommend
    Body: { learner_id: string, top_n?: int }
    Returns: ranked study recommendations from the cubic power law engine.
    """
    body       = request.get_json(silent=True) or {}
    learner_id = (body.get("learner_id") or "").strip()
    top_n      = max(1, min(10, int(body.get("top_n", 5))))

    if not learner_id:
        return jsonify({"error": "learner_id is required"}), 400

    mem = get_memory_client()
    try:
        profile = mem.get_profile(learner_id)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    result = get_recommendations_dict(profile, top_n=top_n)
    result["learner_id"] = learner_id
    return jsonify(result)


# ── Adaptive test endpoints ──────────────────────────────────────────────────

@app.route("/api/test/generate", methods=["POST"])
def test_generate():
    """
    POST /api/test/generate
    Body: {
      topic: string,
      learner_id?: string,
      num_questions?: int (default 5, max 10),
      concept_context?: string
    }
    Returns: adaptive test with mixed objective + subjective questions,
             difficulty scaled to learner's mastery.
    """
    body          = request.get_json(silent=True) or {}
    topic         = (body.get("topic") or "").strip()
    learner_id    = (body.get("learner_id") or "").strip() or None
    num_questions = max(1, min(10, int(body.get("num_questions", 5))))
    concept_ctx   = (body.get("concept_context") or "").strip()

    if not topic:
        return jsonify({"error": "topic is required"}), 400

    # Look up mastery if learner_id provided
    mastery = 0.0
    if learner_id:
        mem = get_memory_client()
        try:
            profile = mem.get_profile(learner_id)
            concept_id = topic.lower().replace(" ", "_")
            entry = profile.get("concepts", {}).get(concept_id)
            if entry:
                mastery = entry.get("mastery", 0.0)
        except Exception:
            pass

    try:
        test = generate_test(
            topic=topic,
            num_questions=num_questions,
            mastery=mastery,
            concept_context=concept_ctx,
        )
    except Exception as exc:
        logger.error("Test generation failed: %s", exc)
        return jsonify({"error": f"Test generation error: {exc}"}), 500

    test["learner_id"] = learner_id
    test["learner_mastery"] = round(mastery, 3)

    logger.info("Test generated: topic=%s difficulty=%s n=%d mastery=%.2f",
                topic, test["difficulty"], test["num_questions"], mastery)
    return jsonify(test)


@app.route("/api/test/submit", methods=["POST"])
def test_submit():
    """
    POST /api/test/submit
    Body: {
      test: { test_id, topic, questions: [...] },
      responses: [{ question_id, answer }, ...],
      learner_id?: string
    }
    Returns: grading result with per-question scores, aggregate outcome,
             and mastery update if learner_id provided.
    """
    body       = request.get_json(silent=True) or {}
    test_data  = body.get("test", {})
    responses  = body.get("responses", [])
    learner_id = (body.get("learner_id") or "").strip() or None
    topic      = (test_data.get("topic") or "").strip()

    if not test_data.get("questions"):
        return jsonify({"error": "test with questions is required"}), 400
    if not responses:
        return jsonify({"error": "responses are required"}), 400

    try:
        result = grade_test(test_data, responses, topic=topic)
    except Exception as exc:
        logger.error("Test grading failed: %s", exc)
        return jsonify({"error": f"Test grading error: {exc}"}), 500

    # Update mastery if learner_id provided
    mastery_update = None
    if learner_id and topic:
        concept_id = topic.lower().replace(" ", "_")
        outcome = result.get("outcome", "partial")
        try:
            mem = get_memory_client()
            profile = mem.get_profile(learner_id)
            concepts_node = profile.setdefault("concepts", {})
            now = datetime.now(timezone.utc).isoformat()
            entry = concepts_node.get(concept_id)
            if entry is None:
                entry = {
                    "mastery": 0.0, "attempts": 0, "correct_attempts": 0,
                    "last_seen": now, "last_outcome": outcome, "error_pattern": "",
                }
                concepts_node[concept_id] = entry
            before = entry["mastery"]
            entry["mastery"]          = update_mastery(entry["mastery"], outcome)
            entry["attempts"]        += 1
            entry["correct_attempts"] += (1 if outcome == "correct" else 0)
            entry["last_seen"]        = now
            entry["last_outcome"]     = outcome
            profile["updated_at"]     = now
            mem.put_profile(learner_id, profile)
            mastery_update = {
                concept_id: {"before": round(before, 3), "after": round(entry["mastery"], 3)}
            }
        except Exception as exc:
            logger.error("Test mastery update failed: %s", exc)

    result["mastery_update"] = mastery_update
    result["learner_id"] = learner_id

    logger.info("Test submitted: topic=%s outcome=%s score=%.1f%%",
                topic, result.get("outcome"), result.get("score_pct", 0))
    return jsonify(result)


# ── Hint endpoints ───────────────────────────────────────────────────────────

@app.route("/api/hint", methods=["POST"])
def hint_single():
    """
    POST /api/hint
    Body: {
      question_text: string,
      level?: 1|2|3 (default: 1),
      topic?: string,
      student_answer?: string,
      concept_context?: string,
      previous_hints?: [string]
    }
    Returns: { level, hint_text, can_request_next, topic }
    """
    body = request.get_json(silent=True) or {}

    ok, err = validate_hint_request(body)
    if not ok:
        return jsonify({"error": err}), 400

    result = generate_hint(
        question_text=(body.get("question_text") or "").strip(),
        level=int(body.get("level", 1)),
        topic=(body.get("topic") or "").strip(),
        student_answer=(body.get("student_answer") or "").strip(),
        concept_context=(body.get("concept_context") or "").strip(),
        previous_hints=body.get("previous_hints"),
    )

    logger.info("Hint generated: topic=%s level=%d", result["topic"], result["level"])
    return jsonify(result)


@app.route("/api/hint/all", methods=["POST"])
def hint_all():
    """
    POST /api/hint/all
    Body: { question_text: string, topic?: string, student_answer?: string, concept_context?: string }
    Returns: { hints: [{level, hint_text, can_request_next, topic}, ...] }

    Generates all 3 hint levels at once (non-streaming).
    """
    body = request.get_json(silent=True) or {}

    question_text = (body.get("question_text") or "").strip()
    if not question_text:
        return jsonify({"error": "question_text is required"}), 400

    hints = generate_all_hints(
        question_text=question_text,
        topic=(body.get("topic") or "").strip(),
        student_answer=(body.get("student_answer") or "").strip(),
        concept_context=(body.get("concept_context") or "").strip(),
    )

    logger.info("All hints generated: topic=%s count=%d", body.get("topic", ""), len(hints))
    return jsonify({"hints": hints})


@app.route("/api/hint/stream", methods=["POST"])
def hint_stream():
    """
    POST /api/hint/stream
    Body: { question_text: string, topic?: string, student_answer?: string, concept_context?: string }
    Returns: SSE stream of hint events.

    Each event: data: {"level": N, "hint_text": "...", "can_request_next": bool}\n\n
    Final event: data: {"done": true}\n\n
    """
    from flask import Response

    body = request.get_json(silent=True) or {}

    question_text = (body.get("question_text") or "").strip()
    if not question_text:
        return jsonify({"error": "question_text is required"}), 400

    def event_generator():
        yield from stream_hints(
            question_text=question_text,
            topic=(body.get("topic") or "").strip(),
            student_answer=(body.get("student_answer") or "").strip(),
            concept_context=(body.get("concept_context") or "").strip(),
        )

    logger.info("Hint stream started: topic=%s", body.get("topic", ""))
    return Response(event_generator(), mimetype="text/event-stream")


if __name__ == "__main__":
    print("Starting server → http://localhost:5000")
    app.run(debug=True, port=5000)
