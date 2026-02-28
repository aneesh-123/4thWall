"""
server.py — Flask web server for the PDF → Study Plan tool.

Routes:
  GET  /              Serve the upload UI.
  POST /api/extract   Accept a PDF, call OpenAI, return the study plan.
  POST /chat/turn     Conversational tutor with persistent Supermemory.
"""

import logging
import os
import sys
import tempfile
import uuid
from datetime import datetime, timezone

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file if present

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
from memory_update import apply_extract_to_profile, build_memory_preview
from pdf_retriever import build_index, retrieve, format_retrieved_chunks

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

    # ── Step 6: build response ────────────────────────────────────────────────
    memory_preview = build_memory_preview(updated_profile)

    return jsonify({
        "learner_id":     learner_id,
        "session_id":     session_id,
        "assistant":      assistant_reply,
        "memory_preview": memory_preview,
        "doc_found":      doc_ctx is not None,
        "sm_grounded":    bool(semantic_passages),
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


if __name__ == "__main__":
    print("Starting server → http://localhost:5000")
    app.run(debug=True, port=5000)
