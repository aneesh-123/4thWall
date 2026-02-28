"""
server.py — Flask web server for the PDF → Study Plan tool.

Routes:
  GET  /              Serve the upload UI.
  POST /api/extract   Accept a PDF, call OpenAI, return the study plan.
"""

import os
import sys
import tempfile

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
