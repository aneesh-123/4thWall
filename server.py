"""
server.py — Flask web server for the PDF → Study Plan tool.

Routes:
  GET  /              Serve the upload UI.
  POST /api/extract   Accept a PDF, call OpenAI, return the study plan.
  POST /chat/turn     Conversational tutor with persistent Supermemory.
"""

import json as _json_mod
import logging
import os
from pathlib import Path
import sys
import tempfile
import uuid
from datetime import datetime, timedelta, timezone

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file if present

from flask import Flask, jsonify, render_template, request, session as flask_session
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
# ── Signed cookie sessions (user identity) ──────────────────────────────────────────
app.secret_key = os.getenv("FLASK_SECRET_KEY", "4thwall-dev-key-change-in-prod")
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.permanent_session_lifetime = timedelta(days=365)

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

# ── Groups persistence ─────────────────────────────────────────────────────────────
# Schema:
#  { "users":  { uid: {"group_ids": [...]} },
#    "groups": { gid: { "name", "owner_user_id", "doc_ids": [],
#                        "doc_meta": { doc_id: {filename, sm_uploaded, concept_ids, pages} },
#                        "created_at" } } }
# Persists in groups_store.json so groups survive server restarts.
# _doc_store (study-plan markdown + TF-IDF index) is still process-local;
# if the server restarts, PDFs must be re-uploaded (UI shows a warning).
_GROUPS_FILE = Path(__file__).parent / "groups_store.json"


def _load_gstore() -> dict:
    if _GROUPS_FILE.exists():
        try:
            return _json_mod.loads(_GROUPS_FILE.read_text(encoding="utf-8"))
        except Exception as _e:
            logger.warning("Could not read groups_store.json: %s", _e)
    return {"users": {}, "groups": {}}


def _save_gstore() -> None:
    try:
        _GROUPS_FILE.write_text(_json_mod.dumps(_gstore, indent=2), encoding="utf-8")
    except Exception as _e:
        logger.error("_save_gstore failed: %s", _e)


_gstore: dict = _load_gstore()


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


# ── PDF processing helper ─────────────────────────────────────────────────────────────
def _process_pdf_file(
    saved_path: str,
    filename: str,
    upload_to_sm: bool = True,
) -> tuple[str, dict]:
    """
    Process an already-saved PDF at *saved_path*, register it in _doc_store,
    and return (doc_id, response_payload_dict).
    """
    tmp_md = tempfile.NamedTemporaryFile(
        suffix=".md", delete=False, mode="w", encoding="utf-8"
    )
    tmp_md.close()
    try:
        pages   = load_pdf(saved_path)
        cleaned = clean_pages(pages)
        topics  = _llm_study_plan(cleaned)

        write_markdown(topics, filename, len(pages), tmp_md.name)
        with open(tmp_md.name, encoding="utf-8") as f:
            md_content = f.read()

        total_subtopics = sum(len(t.subtopics) for t in topics)

        concept_ids: list[str] = []
        _seen: set[str] = set()
        for _t in topics:
            for _s in _t.subtopics:
                _slug = "_".join(
                    p for p in "".join(
                        c if c.isalnum() else "_" for c in _s.name.lower()
                    ).split("_") if p
                )[:48]
                if _slug and _slug not in _seen:
                    concept_ids.append(_slug)
                    _seen.add(_slug)

        doc_id          = str(uuid.uuid4())
        retrieval_index = build_index(cleaned)
        _doc_store[doc_id] = {
            "doc_id":          doc_id,
            "filename":        filename,
            "markdown":        md_content,
            "pages":           len(pages),
            "retrieval_index": retrieval_index,
            "sm_uploaded":     False,
            "concept_ids":     concept_ids,
        }
        logger.info("Stored doc_id=%s filename=%s chunks=%d",
                    doc_id, filename, len(retrieval_index.get("chunks", [])))

        if upload_to_sm:
            try:
                get_memory_client().upload_pdf(doc_id, saved_path, filename)
                _doc_store[doc_id]["sm_uploaded"] = True
            except Exception as _e:
                logger.warning("SM upload skipped for %s: %s", doc_id, _e)

        return doc_id, {
            "doc_id":          doc_id,
            "markdown":        md_content,
            "pages_processed": len(pages),
            "topics_count":    len(topics),
            "subtopics_count": total_subtopics,
            "sm_uploaded":     _doc_store[doc_id]["sm_uploaded"],
            "concept_ids":     concept_ids,
        }
    finally:
        try:
            os.unlink(tmp_md.name)
        except OSError:
            pass


@app.route("/api/extract", methods=["POST"])
def extract():
    if "pdf" not in request.files:
        return jsonify({"error": "No PDF file provided."}), 400
    pdf_file = request.files["pdf"]
    if not pdf_file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Uploaded file must be a PDF."}), 400

    tmp_pdf = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp_pdf.close()
    try:
        pdf_file.save(tmp_pdf.name)
        _, payload = _process_pdf_file(tmp_pdf.name, pdf_file.filename)
        return jsonify(payload)
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
        try:
            os.unlink(tmp_pdf.name)
        except OSError:
            pass


# ── User / Profile / Groups API ───────────────────────────────────────────────────────

@app.route("/api/me")
def api_me():
    """
    GET /api/me
    Returns the current user_id from the signed session cookie, creating one
    on first visit.  The user_id doubles as the learner_id for Supermemory.
    Persists until the browser clears cookies.
    """
    if "user_id" not in flask_session:
        flask_session["user_id"] = str(uuid.uuid4())
        flask_session.permanent = True
    uid = flask_session["user_id"]
    if uid not in _gstore["users"]:
        _gstore["users"][uid] = {"group_ids": []}
        _save_gstore()
    return jsonify({"user_id": uid})


@app.route("/api/groups", methods=["GET"])
def api_groups_list():
    """GET /api/groups — list all groups owned by the current user."""
    uid = flask_session.get("user_id")
    if not uid:
        return jsonify({"error": "Not authenticated — load the page to get a user_id"}), 401
    group_ids = _gstore["users"].get(uid, {}).get("group_ids", [])
    groups = []
    for gid in group_ids:
        g = _gstore["groups"].get(gid)
        if not g:
            continue
        doc_statuses = []
        for did in g.get("doc_ids", []):
            meta = g["doc_meta"].get(did, {})
            doc_statuses.append({
                "doc_id":      did,
                "filename":    meta.get("filename", did),
                "loaded":      did in _doc_store,
                "sm_uploaded": meta.get("sm_uploaded", False),
                "concept_ids": meta.get("concept_ids", []),
                "pages":       meta.get("pages", 0),
            })
        groups.append({
            "group_id":   gid,
            "name":       g["name"],
            "created_at": g["created_at"],
            "doc_count":  len(g.get("doc_ids", [])),
            "docs":       doc_statuses,
        })
    return jsonify({"groups": groups})


@app.route("/api/groups", methods=["POST"])
def api_groups_create():
    """POST /api/groups — create a new group."""
    uid = flask_session.get("user_id")
    if not uid:
        return jsonify({"error": "Not authenticated"}), 401
    body = request.get_json(silent=True) or {}
    name = (body.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Group name is required"}), 400
    gid = str(uuid.uuid4())
    _gstore["groups"][gid] = {
        "name":           name,
        "owner_user_id":  uid,
        "doc_ids":        [],
        "doc_meta":       {},
        "created_at":     datetime.now(timezone.utc).isoformat(),
    }
    _gstore["users"].setdefault(uid, {"group_ids": []})
    _gstore["users"][uid]["group_ids"].append(gid)
    _save_gstore()
    logger.info("Group created gid=%s name=%s uid=%s", gid, name, uid)
    return jsonify({"group_id": gid, "name": name})


@app.route("/api/groups/<gid>/upload", methods=["POST"])
def api_groups_upload(gid: str):
    """
    POST /api/groups/<gid>/upload
    Upload a PDF to a group (max 10). Processes the PDF exactly like /api/extract
    and stores the resulting doc_id + metadata in the group's persistent record.
    """
    uid = flask_session.get("user_id")
    if not uid:
        return jsonify({"error": "Not authenticated"}), 401
    g = _gstore["groups"].get(gid)
    if not g:
        return jsonify({"error": "Group not found"}), 404
    if g["owner_user_id"] != uid:
        return jsonify({"error": "Not your group"}), 403
    if len(g.get("doc_ids", [])) >= 10:
        return jsonify({"error": "Group PDF limit reached (max 10)"}), 400
    if "pdf" not in request.files:
        return jsonify({"error": "No PDF provided"}), 400
    pdf_file = request.files["pdf"]
    if not pdf_file.filename.lower().endswith(".pdf"):
        return jsonify({"error": "Must be a PDF"}), 400

    tmp_pdf = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp_pdf.close()
    try:
        pdf_file.save(tmp_pdf.name)
        doc_id, payload = _process_pdf_file(tmp_pdf.name, pdf_file.filename)
    except NotImplementedError:
        return jsonify({"error": "OpenAI API key not configured."}), 500
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        try:
            os.unlink(tmp_pdf.name)
        except OSError:
            pass

    # Persist to group record
    g["doc_ids"].append(doc_id)
    g["doc_meta"][doc_id] = {
        "filename":    pdf_file.filename,
        "sm_uploaded": payload["sm_uploaded"],
        "concept_ids": payload["concept_ids"],
        "pages":       payload["pages_processed"],
        # Persist markdown so study-plan generation works without a live _doc_store.
        # Store the full markdown (not truncated) so cold-start SM fallback has rich content.
        "markdown":    payload["markdown"],
    }
    # If a group study plan already exists, mark it stale so the UI
    # prompts the user to regenerate and include the new PDF.
    if g.get("group_study_plan"):
        g["plan_needs_update"] = True
    _save_gstore()
    logger.info("Group %s gained doc_id=%s filename=%s", gid, doc_id, pdf_file.filename)
    payload["group_id"]          = gid
    payload["plan_needs_update"] = g.get("plan_needs_update", False)
    return jsonify(payload)


@app.route("/api/groups/<gid>/docs/<doc_id>", methods=["DELETE"])
def api_groups_remove_doc(gid: str, doc_id: str):
    """DELETE /api/groups/<gid>/docs/<doc_id> — remove a PDF from a group."""
    uid = flask_session.get("user_id")
    if not uid:
        return jsonify({"error": "Not authenticated"}), 401
    g = _gstore["groups"].get(gid)
    if not g or g["owner_user_id"] != uid:
        return jsonify({"error": "Not found or not yours"}), 404
    if doc_id in g["doc_ids"]:
        g["doc_ids"].remove(doc_id)
    g["doc_meta"].pop(doc_id, None)
    _doc_store.pop(doc_id, None)
    _save_gstore()
    logger.info("Removed doc_id=%s from group %s", doc_id, gid)
    return jsonify({"ok": True})


@app.route("/api/groups/<gid>/study-plan", methods=["GET"])
def api_group_plan_get(gid: str):
    """GET /api/groups/<gid>/study-plan — return saved group study plan."""
    uid = flask_session.get("user_id")
    if not uid:
        return jsonify({"error": "Not authenticated"}), 401
    g = _gstore["groups"].get(gid)
    if not g or g["owner_user_id"] != uid:
        return jsonify({"error": "Not found"}), 404
    return jsonify({
        "has_plan":          bool(g.get("group_study_plan")),
        "plan":              g.get("group_study_plan", ""),
        "plan_needs_update": g.get("plan_needs_update", False),
        "plan_generated_at": g.get("plan_generated_at", ""),
        "plan_doc_ids":      g.get("plan_doc_ids", []),
    })


@app.route("/api/groups/<gid>/study-plan", methods=["POST"])
def api_group_plan_generate(gid: str):
    """
    POST /api/groups/<gid>/study-plan
    Uses the LLM to synthesise a unified master study plan from all loaded
    per-doc markdowns in the group.  Saves the result to groups_store.json.
    """
    uid = flask_session.get("user_id")
    if not uid:
        return jsonify({"error": "Not authenticated"}), 401
    g = _gstore["groups"].get(gid)
    if not g or g["owner_user_id"] != uid:
        return jsonify({"error": "Not found"}), 404

    # Build per-doc markdown: prefer live _doc_store, fall back to persisted doc_meta markdown.
    doc_entries: list[dict] = []
    for did in g.get("doc_ids", []):
        if did in _doc_store:
            doc_entries.append({"doc_id": did,
                                 "filename": _doc_store[did]["filename"],
                                 "markdown": _doc_store[did]["markdown"]})
        elif g["doc_meta"].get(did, {}).get("markdown"):
            meta = g["doc_meta"][did]
            doc_entries.append({"doc_id": did,
                                 "filename": meta["filename"], "markdown": meta["markdown"]})
    if not doc_entries:
        return jsonify({"error": "No PDFs available for this group — please upload at least one PDF."}), 400

    per_doc_md = "\n\n---\n\n".join(
        f"## Document: {d['filename']}\n\n{d['markdown'][:5000]}"
        for d in doc_entries
    )
    loaded_docs = doc_entries  # keep variable name for len() in prompt below
    _gname = g["name"]
    prompt = (
        f"You are an expert academic study planner. "
        f"Below are study plans for {len(loaded_docs)} document(s) in a learning group "
        f"named \u201c{_gname}\u201d.\n\n"
        "Create a unified, integrated master study plan that:\n"
        "1. Synthesises all topics across all documents\n"
        "2. Groups related concepts together with clear headings\n"
        "3. Highlights connections and dependencies between documents\n"
        "4. Orders topics for optimal learning progression\n"
        "5. Starts with a short \u2018How to use this group\u2019 paragraph\n\n"
        "Output clean, well-structured Markdown. "
        "Use ## for major topic areas and ### for subtopics.\n\n"
        "---\n\n"
        f"{per_doc_md[:7500]}"
    )
    try:
        client = OpenAI()
        resp = client.chat.completions.create(
            model=_OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        plan_md = resp.choices[0].message.content or ""
    except Exception as exc:
        return jsonify({"error": f"LLM error: {exc}"}), 500

    g["group_study_plan"]  = plan_md
    g["plan_needs_update"] = False
    g["plan_generated_at"] = datetime.now(timezone.utc).isoformat()
    g["plan_doc_ids"]      = [d["doc_id"] for d in loaded_docs]
    _save_gstore()
    logger.info("Group study plan generated gid=%s docs=%d chars=%d",
                gid, len(loaded_docs), len(plan_md))
    return jsonify({"plan": plan_md, "plan_needs_update": False})


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
    doc_block_override: str = "",
) -> str:
    profile_block = build_profile_context_string(profile) or "(first session — no prior data)"
    if doc_block_override:
        doc_block = doc_block_override
        scope_instruction = _SCOPE_WITH_DOC
    elif doc:
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
    else:  # no doc and no override
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
    group_id       = (body.get("group_id") or "").strip() or None
    outcome_signal = (body.get("outcome_signal") or "").strip() or None

    # Resolve document context (single doc or group)
    if group_id and group_id in _gstore["groups"]:
        _g         = _gstore["groups"][group_id]
        _all_docs  = [_doc_store[did] for did in _g.get("doc_ids", []) if did in _doc_store]
        doc_ctx    = _all_docs[0] if _all_docs else None
        is_group   = True
    else:
        _g        = None
        _all_docs = [_doc_store[doc_id]] if doc_id and doc_id in _doc_store else []
        doc_ctx   = _all_docs[0] if _all_docs else None
        is_group  = False
        if doc_id and not doc_ctx:
            logger.warning("doc_id=%s not found in store (server may have restarted)", doc_id)

    logger.info(
        "chat/turn learner=%s session=%s group_id=%s doc_id=%s outcome=%s msg_len=%d",
        learner_id, session_id, group_id, doc_id, outcome_signal, len(message),
    )

    # ── Step 1: load or initialise profile ───────────────────────────────────
    # Scope mastery to the specific group so each group has its own mastery profile.
    _eff_lid  = f"{learner_id}|grp|{group_id}" if (group_id and group_id in _gstore["groups"]) else learner_id
    mem = get_memory_client()
    try:
        profile = mem.get_profile(_eff_lid)
    except Exception as exc:
        logger.error("Could not load profile for %s: %s", _eff_lid, exc)
        profile = {"learner_id": _eff_lid, "version": 1,
                   "preferences": {"style": "direct", "verbosity": "medium", "pace": "medium"},
                   "goals": [], "concepts": {}, "misconceptions": {},
                   "recent_summary": "", "updated_at": ""}

    # ── Step 2a: Retrieve passages (TF-IDF + Supermemory) ────────────────────
    semantic_passages = ""
    doc_block_override = ""

    if is_group and _all_docs:
        # Multi-doc: aggregate TF-IDF and SM passages from every loaded doc
        tfidf_sections: list[str] = []
        for _d in _all_docs:
            _idx = _d.get("retrieval_index")
            if _idx:
                _hits = retrieve(_idx, message, top_k=3)
                _chunk_text = format_retrieved_chunks(_hits, max_chars=1200)
                if _chunk_text.strip():
                    tfidf_sections.append(f"\u2500\u2500 {_d['filename']} \u2500\u2500\n{_chunk_text}")
        combined_tfidf = "\n\n".join(tfidf_sections) or "(no TF-IDF passages retrieved)"

        sm_sections: list[str] = []
        for _d in _all_docs:
            if _d.get("sm_uploaded"):
                try:
                    _hits = mem.search_pdf(_d["doc_id"], message, top_k=3)
                    if _hits:
                        for _i, _c in enumerate(_hits):
                            sm_sections.append(f"[{_d['filename']} \u00b7 chunk {_i+1}]\n{_c}")
                except Exception as _sme:
                    logger.warning("SM search failed for %s: %s", _d["doc_id"], _sme)
        combined_sm = "\n\n".join(sm_sections) or "(no semantic passages retrieved)"
        semantic_passages = combined_sm  # for sm_grounded flag

        _gname = _g.get("name", "Group")
        _fnames = ", ".join(d["filename"] for d in _all_docs)
        _outline = _all_docs[0]["markdown"][:700] + ("\u2026" if len(_all_docs[0]["markdown"]) > 700 else "")
        doc_block_override = (
            f'GROUP \"{_gname}\" ({len(_all_docs)} PDFs: {_fnames})\n'
            f"STUDY PLAN OUTLINE (from {_all_docs[0]['filename']}):\n{_outline}\n\n"
            f"--- RETRIEVED PASSAGES (TF-IDF from all PDFs) ---\n{combined_tfidf}\n\n"
            f"--- RETRIEVED PASSAGES (Supermemory from all PDFs) ---\n{combined_sm}\n\n"
            "(Answer based on any of the above PDFs. Cite filenames where helpful.)"
        )
    elif is_group and not _all_docs:
        # _doc_store is cold (server restarted). Retrieve from Supermemory and / or
        # persisted doc_meta markdown so the user never needs to re-upload.
        _gm        = _gstore["groups"][group_id]
        _sm_cold:  list[str] = []
        _md_cold:  list[str] = []
        _fnames_cold: list[str] = []
        for _did in _gm.get("doc_ids", []):
            _meta  = _gm["doc_meta"].get(_did, {})
            _fname = _meta.get("filename", _did)
            _fnames_cold.append(_fname)
            if _meta.get("sm_uploaded"):
                try:
                    _hits = mem.search_pdf(_did, message, top_k=3)
                    for _i, _c in enumerate(_hits):
                        _sm_cold.append(f"[{_fname} \u00b7 chunk {_i+1}]\n{_c}")
                except Exception as _sme:
                    logger.warning("Cold group SM search %s: %s", _did, _sme)
            if _meta.get("markdown"):
                _md_cold.append(f"\u2500\u2500 {_fname} \u2500\u2500\n{_meta['markdown'][:700]}")
        if _sm_cold or _md_cold:
            _gname         = _gm.get("name", "Group")
            _fnames_str    = ", ".join(_fnames_cold)
            combined_sm    = "\n\n".join(_sm_cold) or "(no semantic passages retrieved)"
            combined_md    = "\n\n".join(_md_cold)
            semantic_passages  = combined_sm
            doc_block_override = (
                f'GROUP \"{_gname}\" ({len(_gm.get("doc_ids",[]))} PDFs: {_fnames_str})\n\n'
                f"--- RETRIEVED PASSAGES (Supermemory) ---\n{combined_sm}\n\n"
                + (f"--- STUDY PLAN EXCERPTS ---\n{combined_md}\n\n" if combined_md else "")
                + "(Answer based on the above PDFs. Cite filenames where helpful.)"
            )
            logger.info("Cold group chat fallback gid=%s sm_chunks=%d md_docs=%d",
                        group_id, len(_sm_cold), len(_md_cold))
    elif doc_ctx and doc_ctx.get("sm_uploaded"):
        # Single-doc Supermemory search
        try:
            sm_hits = mem.search_pdf(doc_id, message, top_k=5)
            if sm_hits:
                semantic_passages = "\n\n".join(
                    f"[Semantic chunk {i+1}]\n{chunk}"
                    for i, chunk in enumerate(sm_hits)
                )
                logger.info("SM PDF search: %d chunks doc_id=%s", len(sm_hits), doc_id)
        except Exception as _sm_exc:
            logger.warning("Supermemory PDF search failed: %s", _sm_exc)

    # ── Step 2b: build system prompt and generate LLM response ───────────
    system_prompt = _build_tutor_system_prompt(
        profile, doc=doc_ctx, query=message,
        semantic_passages=semantic_passages,
        doc_block_override=doc_block_override,
    )
    try:
        assistant_reply = _generate_tutor_response(system_prompt, message)
    except Exception as exc:
        logger.error("Tutor LLM call failed: %s", exc)
        return jsonify({"error": f"LLM error: {exc}"}), 500

    logger.info("Tutor response generated (%d chars)", len(assistant_reply))

    # ── Step 3: extract learning signals from the turn ────────────────────────
    # Build concept whitelist from the group/doc so the LLM picks real PDF IDs.
    _chat_allowed: set[str] = set()
    if group_id and group_id in _gstore["groups"]:
        for _did in _gstore["groups"][group_id].get("doc_ids", []):
            if _did in _doc_store:
                _chat_allowed.update(_doc_store[_did].get("concept_ids", []))
            else:
                _chat_allowed.update(
                    _gstore["groups"][group_id]["doc_meta"].get(_did, {}).get("concept_ids", [])
                )
    elif doc_id and doc_ctx:
        _chat_allowed.update(doc_ctx.get("concept_ids", []))

    profile_ctx = build_profile_context_string(profile)
    try:
        extract = extract_turn_insights(
            user_message=message,
            tutor_message=assistant_reply,
            profile_context=profile_ctx,
            outcome_signal=outcome_signal,
            allowed_concept_ids=sorted(_chat_allowed) if _chat_allowed else None,
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

    # ── Step 3b: restrict concepts to in-scope PDF/group slugs ─────────────────
    # Chat turns can invent arbitrary concept IDs; when a group is active we only
    # want to update (or create) concepts that were seeded from the group's PDFs.
    if _chat_allowed:
        extract["concepts"] = [
            c for c in extract.get("concepts", [])
            if c["concept_id"] in _chat_allowed
        ]

    # ── Step 4: update profile deterministically ──────────────────────────────
    # For uninstructed chat (no explicit outcome_signal) do not allow mastery to
    # decrease — asking a question about a topic should never lower mastery.
    updated_profile, deltas = apply_extract_to_profile(
        profile, extract,
        allow_mastery_decrease=(outcome_signal is not None),
    )
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
        mem.put_profile(_eff_lid, updated_profile)
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
        mem.add_event(_eff_lid, event)
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
- Include a `concept_id` (snake_case, 1–4 words) naming the core concept tested.{concept_hint}
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


@app.route("/api/mastery")
def api_get_mastery():
    """
    GET /api/mastery?learner_id=X&group_id=Y
    Returns all_concepts for the (optionally group-scoped) learner profile.
    Used by the frontend to populate the mastery tab after group activation.
    """
    uid        = flask_session.get("user_id") or ""
    learner_id = request.args.get("learner_id", "").strip() or uid
    group_id   = request.args.get("group_id", "").strip() or None
    if not learner_id:
        return jsonify({"all_concepts": []})
    _eff_lid = f"{learner_id}|grp|{group_id}" if (group_id and group_id in _gstore["groups"]) else learner_id
    mem = get_memory_client()
    try:
        profile = mem.get_profile(_eff_lid)
    except Exception:
        profile = {}
    preview = build_memory_preview(profile)
    return jsonify({"all_concepts": preview.get("all_concepts", [])})


@app.route("/api/quiz/generate", methods=["POST"])
def quiz_generate():
    """
    POST /api/quiz/generate
    Body: { doc_id?, topic?, n_questions? }
    Returns: { questions: [{question, options, correct_index, concept_id, explanation, id}] }
    """
    body     = request.get_json(silent=True) or {}
    doc_id   = (body.get("doc_id")   or "").strip() or None
    group_id = (body.get("group_id") or "").strip() or None
    topic    = (body.get("topic")    or "").strip() or None
    n        = max(1, min(int(body.get("n_questions", 5)), 10))

    doc_ctx  = _doc_store.get(doc_id) if doc_id else None

    # Build content block for the LLM to write questions about
    if group_id and group_id in _gstore["groups"]:
        _g = _gstore["groups"][group_id]
        _loaded = [_doc_store[did] for did in _g.get("doc_ids", []) if did in _doc_store]
        if _loaded:
            if topic:
                _secs = []
                for _d in _loaded:
                    _h = retrieve(_d["retrieval_index"], topic, top_k=4)
                    _c = format_retrieved_chunks(_h, max_chars=1000)
                    if _c.strip():
                        _secs.append(f"[{_d['filename']}]\n{_c}")
                content = "\n\n".join(_secs) or "\n\n".join(
                    f"[{d['filename']}]\n{d['markdown'][:2000]}" for d in _loaded
                )
            else:
                content = "\n\n".join(
                    f"[{d['filename']}]\n{d['markdown'][:4000]}" for d in _loaded
                )
        else:
            # _doc_store is cold (server restarted).
            # Priority: (1) Supermemory semantic search, (2) persisted markdown,
            # (3) concept-ID list — at least one will always be available.
            _cold_mem  = get_memory_client()
            _cold_secs: list[str] = []
            _all_cold_cids: list[str] = []
            for _did in _g.get("doc_ids", []):
                _meta  = _g["doc_meta"].get(_did, {})
                _fname = _meta.get("filename", _did)
                _doc_cids = _meta.get("concept_ids", [])
                _all_cold_cids.extend(_dc for _dc in _doc_cids if _dc not in _all_cold_cids)

                _doc_parts: list[str] = []

                # 1) Supermemory: try topic query first, then broader fallbacks
                if _meta.get("sm_uploaded"):
                    _queries = [q for q in [topic, "key concepts", "main topics", _fname] if q]
                    for _q in _queries:
                        try:
                            _sm_hits = _cold_mem.search_pdf(_did, _q, top_k=5)
                            if _sm_hits:
                                _doc_parts.extend(_sm_hits)
                                logger.info("Cold quiz SM ok  did=%s query=%r hits=%d", _did[:8], _q, len(_sm_hits))
                                break  # got content for this doc — stop trying queries
                        except Exception as _ce:
                            logger.warning("Cold quiz SM failed did=%s query=%r: %s", _did[:8], _q, _ce)

                # 2) Persisted markdown
                if _meta.get("markdown") and not _doc_parts:
                    _doc_parts.append(_meta["markdown"][:4000])

                if _doc_parts:
                    _cold_secs.append(f"[{_fname}]\n" + "\n\n".join(_doc_parts))

            if _cold_secs:
                content = "\n\n".join(_cold_secs)
            elif _all_cold_cids:
                # Last resort: no SM content and no markdown (pre-persistence uploads).
                # Generate questions purely from the known concept list — still grounded
                # in the group's actual study topics.
                _cids_str = ", ".join(_all_cold_cids[:50])
                content = (
                    f"This group covers the following concepts: {_cids_str}.\n"
                    f"Generate questions that test understanding of these concepts."
                )
                logger.info("Cold quiz concept-ID fallback gid=%s cids=%d", group_id, len(_all_cold_cids))
            else:
                return jsonify({"error": "No content available for this group. Please re-upload the PDFs."}), 400
    elif doc_ctx:
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
        return jsonify({"error": "Provide a group_id, doc_id, or topic to generate a quiz."}), 400

    # Compute allowed concept IDs BEFORE the LLM call so we can constrain the prompt.
    # Use doc_meta from groups_store (persisted) so this works even after server restart
    # as long as the group has PDFs recorded (even if _doc_store is cold).
    allowed_concept_ids: list[str] = []
    if group_id and group_id in _gstore["groups"]:
        # Prefer live _doc_store; fall back to persisted doc_meta
        for did in _gstore["groups"][group_id].get("doc_ids", []):
            if did in _doc_store:
                allowed_concept_ids.extend(_doc_store[did].get("concept_ids", []))
            else:
                # Persisted concept_ids from groups_store (available without re-upload)
                meta = _gstore["groups"][group_id]["doc_meta"].get(did, {})
                allowed_concept_ids.extend(meta.get("concept_ids", []))
    elif doc_ctx:
        allowed_concept_ids = list(doc_ctx.get("concept_ids", []))
    # Deduplicate (preserve order)
    _aci_seen: set[str] = set()
    allowed_concept_ids = [c for c in allowed_concept_ids
                           if c not in _aci_seen and not _aci_seen.add(c)]

    # Build concept constraint hint so the LLM reuses known concept IDs from the docs.
    concept_hint = ""
    if allowed_concept_ids:
        cids_str = ", ".join(allowed_concept_ids[:40])
        concept_hint = (
            f"\n- The `concept_id` MUST be chosen from this list (pick the closest match): {cids_str}"
        )

    topic_clause = f" specifically about '{topic}'" if topic else " based on the content below"
    prompt = _QUIZ_GEN_PROMPT.format(n=n, topic_clause=topic_clause, content=content, concept_hint=concept_hint)

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

    logger.info("Quiz generated: %d Qs doc_id=%s group_id=%s topic=%s allowed=%d",
                len(questions), doc_id, group_id, topic, len(allowed_concept_ids))
    return jsonify({
        "questions":           questions,
        "doc_id":              doc_id,
        "topic":               topic,
        "allowed_concept_ids": allowed_concept_ids,
    })


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
    group_id   = (body.get("group_id") or "").strip() or None
    # Scope mastery to the group so each group has independent concept mastery.
    _eff_lid   = f"{learner_id}|grp|{group_id}" if (group_id and group_id in _gstore["groups"]) else learner_id
    # Concepts the client considers "in-scope" for this quiz (from the PDF/group).
    # We only CREATE NEW profile entries for concepts on this list.
    # Concepts already in the profile are always updated regardless.
    _raw_allowed     = body.get("allowed_concept_ids", [])
    _whitelist_sent  = "allowed_concept_ids" in body  # distinguishes explicit [] from absent key
    allowed_cids     = set(c.lower().replace(" ", "_") for c in _raw_allowed if c)

    if not answers:
        return jsonify({"error": "No answers provided."}), 400

    # ── Load profile ──────────────────────────────────────────────────────────
    mem = get_memory_client()
    try:
        profile = mem.get_profile(_eff_lid)
    except Exception:
        profile = {
            "learner_id": _eff_lid, "version": 1, "goals": [],
            "preferences": {"style": "direct", "verbosity": "medium", "pace": "medium"},
            "concepts": {}, "misconceptions": {}, "concept_edges": {},
            "recent_summary": "", "updated_at": "",
        }

    # ── Score each answer ─────────────────────────────────────────────────────
    now           = datetime.now(timezone.utc).isoformat()
    results: list[dict] = []
    # Map concept_id → outcome for the LAST answer on that concept
    concept_outcomes: dict[str, str] = {}
    # Misconceptions: wrong answers are evidence of a misconception on that concept
    misc_node = profile.setdefault("misconceptions", {})
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
        # Record wrong answers as misconception evidence
        if not is_right:
            miss_id = f"quiz_{cid}"
            m_e = misc_node.setdefault(miss_id, {
                "concept_id": cid, "count": 0, "last_seen": now, "evidence": []
            })
            m_e["count"]    += 1
            m_e["last_seen"] = now
            ev = (ans.get("question") or "")[:120].strip()
            if ev and ev not in m_e["evidence"]:
                m_e["evidence"] = (m_e["evidence"] + [ev])[-5:]

    score_pct = n_correct / len(answers)
    if   score_pct >= 0.8: overall = "correct"
    elif score_pct >= 0.5: overall = "partial"
    elif score_pct >= 0.25: overall = "confused"
    else:                   overall = "incorrect"

    # ── Per-concept mastery update ────────────────────────────────────────────
    concepts_node = profile.setdefault("concepts", {})
    mastery_before: dict[str, float] = {}
    mastery_after:  dict[str, float] = {}

    for cid, outcome in concept_outcomes.items():
        entry = concepts_node.get(cid)
        if entry is None:
            # Guard: only create new concept entries for in-scope PDF/group concepts.
            # If the whitelist key was absent entirely → allow all (backward-compat, no PDF context).
            # If an explicit list was sent (even empty []) → restrict: only listed concepts.
            if _whitelist_sent and cid not in allowed_cids:
                logger.debug("quiz_submit: skipping new concept '%s' — not in PDF whitelist", cid)
                continue
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
        mem.put_profile(_eff_lid, profile)
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
        # Only include concepts that were actually updated (some may have been skipped
        # by the whitelist guard, so they won't be in mastery_before/after).
        "mastery_deltas":  {
            cid: {"before": mastery_before[cid], "after": mastery_after[cid]}
            for cid in concept_outcomes
            if cid in mastery_before
        },
        "memory_preview":  memory_preview,
    })



# ── Single-doc markdown endpoint ────────────────────────────────────────────────────────────

@app.route("/api/docs/<doc_id>/markdown")
def doc_markdown(doc_id: str):
    """
    GET /api/docs/<doc_id>/markdown
    Returns the stored study-plan markdown for a processed doc.
    Used by the group study plan view in the PDF tab.
    """
    doc = _doc_store.get(doc_id)
    if not doc:
        return jsonify({"error": "Doc not found — please re-upload the PDF."}), 404
    return jsonify({
        "doc_id":      doc_id,
        "filename":    doc["filename"],
        "markdown":    doc["markdown"],
        "concept_ids": doc.get("concept_ids", []),
        "pages":       doc.get("pages", 0),
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
