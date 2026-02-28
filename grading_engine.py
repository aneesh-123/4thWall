"""
grading_engine.py — Transparent dual-mode grading with rubric generation.

Two grading modes:
  informal  — generous, concept-focused, accepts colloquial language
  formal    — precise terminology, structurally complete, academic standard

Every grade includes:
  1. The rubric used (criteria, weights, requirements)
  2. Sources referenced
  3. Per-criterion breakdown with specific feedback
  4. An outcome label compatible with memory_update.py (correct/partial/confused/incorrect)

Uses OpenAI (gpt-4o-mini) to match the rest of the codebase.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ── Outcome mapping (matches memory_update.py) ──────────────────────────────

OUTCOME_SCORES: dict[str, float] = {
    "correct":   1.0,
    "partial":   0.5,
    "confused":  0.25,
    "incorrect": 0.0,
}

# ── Rubric generation prompt ─────────────────────────────────────────────────

_RUBRIC_SYSTEM = """\
You are an expert educational assessment designer. Generate a grading rubric
for the given question and topic.

Return ONLY a JSON object (no markdown, no code fences) with this exact schema:
{
  "criteria": [
    {
      "name": "string (criterion name, e.g. 'Core Definition')",
      "weight": 0.0-1.0,
      "formal_requirement": "string (what a formal/academic answer needs)",
      "informal_requirement": "string (what an informal/conceptual answer needs)"
    }
  ],
  "sources": [
    {
      "type": "concept_graph | authoritative | mastery_context",
      "reference": "string (source description)"
    }
  ],
  "total_points": 10
}

Rules:
- Generate 3-5 criteria. Weights must sum to 1.0.
- Each criterion must have BOTH a formal and informal requirement.
- Include at least one source of each type that is relevant.
- Keep descriptions concise (≤100 chars each).
"""

# ── Grading prompts by mode ──────────────────────────────────────────────────

_GRADE_SYSTEM_INFORMAL = """\
You are grading a student's answer in INFORMAL mode.
Focus on whether the student captured the CORE CONCEPT, not exact terminology.
Accept colloquial language, analogies, and approximate definitions.
Be generous with partial credit — if the general direction is right, give credit.

You will be given:
- The student's answer
- A rubric with criteria (use the informal_requirement for each)
- Context about the topic

Return ONLY a JSON object (no markdown, no code fences):
{
  "criteria_scores": [
    {
      "name": "string (must match rubric criterion name)",
      "score": 0-4,
      "max_score": 4,
      "met": true/false,
      "feedback": "string (specific feedback for this criterion, ≤150 chars)"
    }
  ],
  "overall_score": 0.0-10.0,
  "outcome": "correct | partial | confused | incorrect",
  "strength": "string (one thing the student got right, ≤100 chars)",
  "improvement": "string (one thing to improve, ≤100 chars)",
  "misconceptions_detected": ["string (misconception IDs if any, snake_case)"],
  "sources_used": ["string (source references used in grading)"]
}

Outcome mapping:
- correct:   All criteria met, score >= 8/10
- partial:   Most criteria met, score 5-7.9/10
- confused:  Some criteria met but reasoning shows misunderstanding, score 3-4.9/10
- incorrect: Core criteria not met, score < 3/10
"""

_GRADE_SYSTEM_FORMAL = """\
You are grading a student's answer in FORMAL/ACADEMIC mode.
Require precise terminology and structurally complete definitions.
Evaluate whether all required components of the concept are mentioned.
Partial credit requires demonstrating specific sub-concepts.

You will be given:
- The student's answer
- A rubric with criteria (use the formal_requirement for each)
- Context about the topic

Return ONLY a JSON object (no markdown, no code fences):
{
  "criteria_scores": [
    {
      "name": "string (must match rubric criterion name)",
      "score": 0-4,
      "max_score": 4,
      "met": true/false,
      "feedback": "string (specific feedback for this criterion, ≤150 chars)"
    }
  ],
  "overall_score": 0.0-10.0,
  "outcome": "correct | partial | confused | incorrect",
  "strength": "string (one thing the student got right, ≤100 chars)",
  "improvement": "string (one thing to improve, ≤100 chars)",
  "misconceptions_detected": ["string (misconception IDs if any, snake_case)"],
  "sources_used": ["string (source references used in grading)"]
}

Outcome mapping:
- correct:   All criteria met with precise terminology, score >= 8/10
- partial:   Most criteria met, minor terminology gaps, score 5-7.9/10
- confused:  Shows conceptual misunderstanding despite some correct elements, score 3-4.9/10
- incorrect: Core definition wrong or missing, score < 3/10
"""

_VALID_OUTCOMES = {"correct", "partial", "confused", "incorrect"}


# ── Core functions ───────────────────────────────────────────────────────────

def generate_rubric(
    topic: str,
    question_text: str,
    concept_context: str = "",
    mastery_context: str = "",
) -> dict[str, Any]:
    """
    Generate a grading rubric for a question using OpenAI.

    Parameters
    ----------
    topic          : the concept being tested (e.g. "balanced binary trees")
    question_text  : the actual question asked
    concept_context: extracted notes from the concept graph / PDF (if available)
    mastery_context: learner's prior performance on related concepts

    Returns
    -------
    Rubric dict with criteria, sources, and total_points.
    Falls back to a default rubric on LLM failure.
    """
    user_prompt = f"""Topic: {topic}
Question: {question_text}

Concept context (from learning material):
{concept_context or "(no concept context available)"}

Learner context:
{mastery_context or "(no prior mastery data)"}

Generate a grading rubric for this question."""

    try:
        client = OpenAI()
        resp = client.chat.completions.create(
            model=_OPENAI_MODEL,
            messages=[
                {"role": "system", "content": _RUBRIC_SYSTEM},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or "{}"
        rubric = json.loads(raw)
        return _validate_rubric(rubric)

    except Exception as exc:
        logger.warning("Rubric generation failed: %s — using default rubric", exc)
        return _default_rubric(topic, question_text)


def grade_informal(
    submission: str,
    topic: str,
    question_text: str,
    rubric: dict[str, Any],
    concept_context: str = "",
) -> dict[str, Any]:
    """
    Grade a student submission in informal (conceptual) mode.

    Returns a complete grading result with rubric breakdown, outcome,
    feedback, and sources.
    """
    return _grade_with_mode(
        submission=submission,
        topic=topic,
        question_text=question_text,
        rubric=rubric,
        concept_context=concept_context,
        mode="informal",
        system_prompt=_GRADE_SYSTEM_INFORMAL,
    )


def grade_formal(
    submission: str,
    topic: str,
    question_text: str,
    rubric: dict[str, Any],
    concept_context: str = "",
) -> dict[str, Any]:
    """
    Grade a student submission in formal (academic) mode.

    Returns a complete grading result with rubric breakdown, outcome,
    feedback, and sources.
    """
    return _grade_with_mode(
        submission=submission,
        topic=topic,
        question_text=question_text,
        rubric=rubric,
        concept_context=concept_context,
        mode="formal",
        system_prompt=_GRADE_SYSTEM_FORMAL,
    )


def grade(
    submission: str,
    topic: str,
    question_text: str,
    mode: str = "informal",
    concept_context: str = "",
    mastery_context: str = "",
) -> dict[str, Any]:
    """
    Router function: generate rubric, then grade in the specified mode.

    This is the main entry point for grading. It:
    1. Generates a rubric for the question
    2. Grades the submission against that rubric
    3. Returns the complete result with rubric, scores, and outcome

    Parameters
    ----------
    submission      : the student's answer text
    topic           : concept being tested
    question_text   : the question that was asked
    mode            : "informal" (default) or "formal"
    concept_context : extracted notes from PDF / concept graph
    mastery_context : learner's prior mastery summary

    Returns
    -------
    Complete grading result dict.
    """
    rubric = generate_rubric(
        topic=topic,
        question_text=question_text,
        concept_context=concept_context,
        mastery_context=mastery_context,
    )

    if mode == "formal":
        result = grade_formal(
            submission=submission,
            topic=topic,
            question_text=question_text,
            rubric=rubric,
            concept_context=concept_context,
        )
    else:
        result = grade_informal(
            submission=submission,
            topic=topic,
            question_text=question_text,
            rubric=rubric,
            concept_context=concept_context,
        )

    result["rubric"] = rubric
    result["mode"] = mode
    return result


# ── Internal helpers ─────────────────────────────────────────────────────────

def _grade_with_mode(
    submission: str,
    topic: str,
    question_text: str,
    rubric: dict[str, Any],
    concept_context: str,
    mode: str,
    system_prompt: str,
) -> dict[str, Any]:
    """Call OpenAI to grade a submission against a rubric in the given mode."""

    requirement_key = "informal_requirement" if mode == "informal" else "formal_requirement"
    criteria_text = "\n".join(
        f"  - {c['name']} (weight: {c['weight']}): {c.get(requirement_key, c.get('informal_requirement', ''))}"
        for c in rubric.get("criteria", [])
    )
    sources_text = "\n".join(
        f"  - [{s['type']}] {s['reference']}"
        for s in rubric.get("sources", [])
    )

    user_prompt = f"""Topic: {topic}
Question: {question_text}

Student's answer:
{submission[:2000]}

Rubric criteria ({mode} mode):
{criteria_text}

Available sources:
{sources_text}

Concept context:
{concept_context[:1000] if concept_context else "(none)"}

Grade this answer against the rubric. Return JSON only."""

    try:
        client = OpenAI()
        resp = client.chat.completions.create(
            model=_OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or "{}"
        result = json.loads(raw)
        return _validate_grade_result(result)

    except Exception as exc:
        logger.error("Grading LLM call failed: %s", exc)
        return _fallback_grade_result(submission)


def _validate_rubric(rubric: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize a rubric from the LLM."""
    criteria = rubric.get("criteria", [])
    if not criteria:
        return _default_rubric("unknown", "unknown")

    validated_criteria = []
    for c in criteria[:5]:
        validated_criteria.append({
            "name": str(c.get("name", "Criterion"))[:100],
            "weight": max(0.0, min(1.0, float(c.get("weight", 0.25)))),
            "formal_requirement": str(c.get("formal_requirement", ""))[:200],
            "informal_requirement": str(c.get("informal_requirement", ""))[:200],
        })

    # Normalize weights to sum to 1.0
    total_weight = sum(c["weight"] for c in validated_criteria)
    if total_weight > 0:
        for c in validated_criteria:
            c["weight"] = round(c["weight"] / total_weight, 2)

    sources = []
    for s in rubric.get("sources", [])[:5]:
        sources.append({
            "type": str(s.get("type", "authoritative"))[:30],
            "reference": str(s.get("reference", ""))[:200],
        })

    return {
        "criteria": validated_criteria,
        "sources": sources,
        "total_points": int(rubric.get("total_points", 10)),
    }


def _validate_grade_result(result: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize a grading result from the LLM."""
    criteria_scores = []
    for cs in result.get("criteria_scores", []):
        criteria_scores.append({
            "name": str(cs.get("name", ""))[:100],
            "score": max(0, min(4, int(cs.get("score", 0)))),
            "max_score": 4,
            "met": bool(cs.get("met", False)),
            "feedback": str(cs.get("feedback", ""))[:200],
        })

    overall = max(0.0, min(10.0, float(result.get("overall_score", 0.0))))

    outcome = str(result.get("outcome", "partial")).lower().strip()
    if outcome not in _VALID_OUTCOMES:
        # Infer from score
        if overall >= 8.0:
            outcome = "correct"
        elif overall >= 5.0:
            outcome = "partial"
        elif overall >= 3.0:
            outcome = "confused"
        else:
            outcome = "incorrect"

    misconceptions = [
        str(m)[:100] for m in result.get("misconceptions_detected", [])
        if m
    ][:5]

    sources_used = [
        str(s)[:200] for s in result.get("sources_used", [])
        if s
    ][:5]

    return {
        "criteria_scores": criteria_scores,
        "overall_score": round(overall, 1),
        "total_points": 10,
        "outcome": outcome,
        "strength": str(result.get("strength", ""))[:150],
        "improvement": str(result.get("improvement", ""))[:150],
        "misconceptions_detected": misconceptions,
        "sources_used": sources_used,
    }


def _default_rubric(topic: str, question_text: str) -> dict[str, Any]:
    """Fallback rubric when LLM generation fails."""
    return {
        "criteria": [
            {
                "name": "Core Concept",
                "weight": 0.40,
                "formal_requirement": f"Accurate definition of {topic}",
                "informal_requirement": f"Shows understanding of {topic}",
            },
            {
                "name": "Reasoning",
                "weight": 0.35,
                "formal_requirement": "Logical reasoning with correct terminology",
                "informal_requirement": "General reasoning in the right direction",
            },
            {
                "name": "Completeness",
                "weight": 0.25,
                "formal_requirement": "All key components addressed",
                "informal_requirement": "Main idea captured",
            },
        ],
        "sources": [
            {"type": "authoritative", "reference": f"Standard reference for {topic}"},
        ],
        "total_points": 10,
    }


def _fallback_grade_result(submission: str) -> dict[str, Any]:
    """Fallback grading result when LLM call fails."""
    has_content = len(submission.strip()) > 10
    return {
        "criteria_scores": [],
        "overall_score": 5.0 if has_content else 0.0,
        "total_points": 10,
        "outcome": "partial" if has_content else "incorrect",
        "strength": "Answer provided" if has_content else "",
        "improvement": "Could not grade automatically — please review",
        "misconceptions_detected": [],
        "sources_used": [],
    }
