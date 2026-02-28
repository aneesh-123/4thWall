"""
server.py — Flask web server for the PDF → Study Plan tool.

Routes:
  GET  /              Serve the upload UI.
  POST /api/extract   Accept a PDF, call OpenAI, return the study plan + graph.
"""

import os
import re
import sys
import tempfile
from collections import Counter

from flask import Flask, jsonify, render_template, request

sys.path.insert(0, os.path.dirname(__file__))
from pdf_to_concepts import (
    _llm_study_plan,
    clean_pages,
    load_pdf,
    write_markdown,
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024   # 50 MB limit


# ── Topic-level knowledge graph ───────────────────────────────────────────────

_STOP = frozenset(
    "a an the is are was were be been have has had do does did will would "
    "could should may might shall can of in on at by for with about from "
    "its they them their what which who this that we our you your and or "
    "but not so then when where how all each most other some such too very "
    "just also used use using given well shown often called known".split()
)
_EASY_RE = re.compile(
    r"\b(introduc|overview|basic|fundamental|definition|background|"
    r"motivation|histor|review|recall|simple|primer)\b", re.I
)
_HARD_RE = re.compile(
    r"\b(advanced|complex|optim|proof|prove|theorem|derive|"
    r"formal|analysis|asymptot|worst.case|convergence)\b", re.I
)


def _kws(text: str, n: int = 8) -> list:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z'\-]{2,}", text.lower())
    return [w for w, _ in Counter(
        t for t in tokens if t not in _STOP and len(t) > 2
    ).most_common(n)]


def _diff(text: str, pos: float) -> str:
    if _EASY_RE.search(text):
        return "easy"
    if _HARD_RE.search(text):
        return "hard"
    return "easy" if pos < 0.33 else "medium" if pos < 0.67 else "hard"


def _topic_graph(topics: list) -> dict:
    """Build a topic-level prerequisite graph for the frontend."""
    n = len(topics)
    nodes = []
    for i, t in enumerate(topics):
        full = t.topic + " " + " ".join(st.name + " " + st.notes for st in t.subtopics)
        nodes.append({
            "id": f"t{i}",
            "label": t.topic,
            "subtopics": [{"name": st.name, "notes": st.notes} for st in t.subtopics],
            "difficulty": _diff(full, i / max(n - 1, 1)),
            "keywords": _kws(full),
        })

    # Infer forward edges by keyword Jaccard overlap
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = nodes[i], nodes[j]
            sa = set(a["keywords"]) | {w for w in a["label"].lower().split() if len(w) > 3}
            sb = set(b["keywords"]) | {w for w in b["label"].lower().split() if len(w) > 3}
            if sa and sb and len(sa & sb) / len(sa | sb) >= 0.08:
                edges.append({"from_id": a["id"], "to_id": b["id"]})

    # Fallback: sequential chain when keyword overlap finds nothing
    if not edges and n > 1:
        edges = [{"from_id": nodes[i]["id"], "to_id": nodes[i + 1]["id"]} for i in range(n - 1)]

    # Transitive reduction (skipped gracefully if networkx not installed)
    try:
        import networkx as nx
        G = nx.DiGraph()
        for nd in nodes:
            G.add_node(nd["id"])
        for e in edges:
            G.add_edge(e["from_id"], e["to_id"])
        G = nx.transitive_reduction(G)
        edges = [{"from_id": u, "to_id": v} for u, v in G.edges()]
    except Exception:
        pass

    return {"nodes": nodes, "edges": edges}


# ── Routes ────────────────────────────────────────────────────────────────────

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

        return jsonify({
            "markdown":        md_content,
            "pages_processed": len(pages),
            "topics_count":    len(topics),
            "subtopics_count": total_subtopics,
            "graph":           _topic_graph(topics),
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


if __name__ == "__main__":
    print("Starting server → http://localhost:5000")
    app.run(debug=True, port=5000)
