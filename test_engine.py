"""
test_engine.py — Adaptive test generation with mixed question types.

Generates both objective (auto-gradable) and subjective (graded by LLM)
questions, with difficulty scaled to the learner's mastery level.

Difficulty mapping (mastery → difficulty):
  mastery < 0.3    → EASY       (multiple_choice, true_false)
  0.3 ≤ m < 0.6   → MEDIUM     (fill_blank, short_answer)
  0.6 ≤ m < 0.85  → HARD       (explain, code_write)
  mastery ≥ 0.85   → EXTENSION  (design, synthesis)

Question types:
  Objective (auto-gradable): multiple_choice, true_false, fill_blank
  Subjective (LLM-graded):  short_answer, explain, code_write

Uses OpenAI (gpt-4o-mini) to match the rest of the codebase.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ── Difficulty levels ────────────────────────────────────────────────────────

DIFFICULTY_LEVELS = ("EASY", "MEDIUM", "HARD", "EXTENSION")

DIFFICULTY_QUESTION_TYPES: dict[str, list[str]] = {
    "EASY":      ["multiple_choice", "true_false"],
    "MEDIUM":    ["fill_blank", "short_answer"],
    "HARD":      ["explain", "code_write"],
    "EXTENSION": ["explain", "code_write"],  # same types, harder prompts
}

OBJECTIVE_TYPES = {"multiple_choice", "true_false", "fill_blank"}
SUBJECTIVE_TYPES = {"short_answer", "explain", "code_write"}


def get_difficulty_level(mastery: float) -> str:
    """Map mastery score to difficulty level."""
    if mastery < 0.3:
        return "EASY"
    elif mastery < 0.6:
        return "MEDIUM"
    elif mastery < 0.85:
        return "HARD"
    else:
        return "EXTENSION"


# ── Question generation prompts ──────────────────────────────────────────────

_QUESTION_SYSTEM = """\
You are a test question generator for an adaptive learning system.
Generate ONE question of the specified type and difficulty.

Return ONLY a JSON object (no markdown, no code fences) with this schema:

For multiple_choice:
{{"type": "multiple_choice", "text": "question text", "options": ["A", "B", "C", "D"], "correct_index": 0, "explanation": "why correct", "concept_id": "snake_case"}}

For true_false:
{{"type": "true_false", "text": "statement to evaluate", "correct_answer": true, "explanation": "why", "concept_id": "snake_case"}}

For fill_blank:
{{"type": "fill_blank", "text": "sentence with _____ blank", "correct_answer": "expected fill", "accept_variations": ["alt1", "alt2"], "explanation": "why", "concept_id": "snake_case"}}

For short_answer:
{{"type": "short_answer", "text": "question requiring 1-3 sentence answer", "rubric_hint": "key points to cover", "concept_id": "snake_case"}}

For explain:
{{"type": "explain", "text": "explain/compare/contrast prompt", "rubric_hint": "criteria for a good answer", "concept_id": "snake_case"}}

For code_write:
{{"type": "code_write", "text": "coding task description", "rubric_hint": "expected approach and edge cases", "language": "python", "concept_id": "snake_case"}}

Rules:
- Question must match the difficulty level described.
- concept_id must be snake_case (e.g. "binary_search").
- Explanations must be concise (≤150 chars).
- Keep question text clear and unambiguous.
"""


def generate_question(
    topic: str,
    difficulty: str,
    question_type: str,
    concept_context: str = "",
) -> dict[str, Any]:
    """
    Generate a single question using the LLM.

    Parameters
    ----------
    topic           : the concept/topic to test
    difficulty      : EASY | MEDIUM | HARD | EXTENSION
    question_type   : multiple_choice | true_false | fill_blank | short_answer | explain | code_write
    concept_context : relevant learning material context

    Returns a question dict with an auto-generated UUID id.
    """
    difficulty = difficulty.upper()
    if difficulty not in DIFFICULTY_LEVELS:
        difficulty = "MEDIUM"

    user_prompt = f"""Topic: {topic}
Difficulty: {difficulty}
Question type: {question_type}

Context from learning material:
{concept_context[:1500] if concept_context else "(general knowledge)"}

Generate one {question_type} question at {difficulty} difficulty about {topic}."""

    try:
        client = OpenAI()
        resp = client.chat.completions.create(
            model=_OPENAI_MODEL,
            messages=[
                {"role": "system", "content": _QUESTION_SYSTEM},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.7,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or "{}"
        question = json.loads(raw)
    except Exception as exc:
        logger.error("Question generation failed: %s", exc)
        question = _fallback_question(topic, question_type)

    question["id"] = str(uuid.uuid4())[:8]
    question["difficulty"] = difficulty
    question.setdefault("type", question_type)
    question.setdefault("concept_id", topic.lower().replace(" ", "_"))

    return question


def generate_test(
    topic: str,
    num_questions: int = 5,
    mastery: float = 0.0,
    concept_context: str = "",
) -> dict[str, Any]:
    """
    Generate an adaptive test with mixed question types.

    The test adapts to the learner's mastery:
    - Difficulty is set by mastery level
    - Question types are chosen from the difficulty's pool
    - Mix of objective and subjective questions

    Parameters
    ----------
    topic           : concept to test
    num_questions   : how many questions (1-10)
    mastery         : learner's current mastery (0.0-1.0)
    concept_context : relevant material context

    Returns a complete test dict with questions, metadata, and instructions.
    """
    num_questions = max(1, min(10, num_questions))
    mastery = max(0.0, min(1.0, mastery))

    difficulty = get_difficulty_level(mastery)
    available_types = DIFFICULTY_QUESTION_TYPES[difficulty]

    # Distribute questions across available types (round-robin)
    questions: list[dict[str, Any]] = []
    for i in range(num_questions):
        qtype = available_types[i % len(available_types)]
        q = generate_question(
            topic=topic,
            difficulty=difficulty,
            question_type=qtype,
            concept_context=concept_context,
        )
        questions.append(q)

    # Count objective vs subjective
    n_objective = sum(1 for q in questions if q.get("type") in OBJECTIVE_TYPES)
    n_subjective = sum(1 for q in questions if q.get("type") in SUBJECTIVE_TYPES)

    return {
        "test_id": str(uuid.uuid4())[:8],
        "topic": topic,
        "difficulty": difficulty,
        "mastery_at_generation": round(mastery, 3),
        "questions": questions,
        "num_questions": len(questions),
        "num_objective": n_objective,
        "num_subjective": n_subjective,
        "instructions": _instructions_for_difficulty(difficulty),
    }


def grade_test(
    test: dict[str, Any],
    responses: list[dict[str, Any]],
    topic: str = "",
) -> dict[str, Any]:
    """
    Grade a completed test.

    - Objective questions: auto-graded by comparing to answer key.
    - Subjective questions: graded via grading_engine.

    Parameters
    ----------
    test      : the original test dict (from generate_test)
    responses : list of {question_id, answer} dicts
    topic     : topic for grading context

    Returns a grading result with per-question scores and aggregate outcome.
    """
    # Build question lookup by id
    q_map: dict[str, dict] = {q["id"]: q for q in test.get("questions", [])}

    # Build response lookup by question_id
    r_map: dict[str, str] = {
        r.get("question_id", ""): r.get("answer", "")
        for r in responses
    }

    question_scores: list[dict[str, Any]] = []
    total_points = 0
    earned_points = 0.0

    for qid, question in q_map.items():
        answer = r_map.get(qid, "")
        qtype = question.get("type", "")
        total_points += 1

        if qtype in OBJECTIVE_TYPES:
            score_info = _grade_objective(question, answer)
        else:
            score_info = _grade_subjective(question, answer, topic)

        earned_points += score_info["score"]
        question_scores.append({
            "question_id": qid,
            "type": qtype,
            **score_info,
        })

    score_pct = (earned_points / total_points * 100) if total_points > 0 else 0.0

    if score_pct >= 80:
        outcome = "correct"
    elif score_pct >= 50:
        outcome = "partial"
    elif score_pct >= 25:
        outcome = "confused"
    else:
        outcome = "incorrect"

    return {
        "test_id": test.get("test_id", ""),
        "score": round(earned_points, 1),
        "total": total_points,
        "score_pct": round(score_pct, 1),
        "outcome": outcome,
        "question_scores": question_scores,
    }


# ── Internal helpers ─────────────────────────────────────────────────────────

def _grade_objective(question: dict, answer: str) -> dict[str, Any]:
    """Auto-grade an objective question."""
    qtype = question.get("type", "")

    if qtype == "multiple_choice":
        try:
            selected = int(answer)
        except (TypeError, ValueError):
            selected = -1
        correct_idx = question.get("correct_index", -1)
        is_correct = selected == correct_idx
        return {
            "score": 1.0 if is_correct else 0.0,
            "correct": is_correct,
            "correct_answer": correct_idx,
            "feedback": question.get("explanation", ""),
        }

    elif qtype == "true_false":
        user_bool = str(answer).lower().strip() in ("true", "1", "yes")
        correct_bool = bool(question.get("correct_answer", False))
        is_correct = user_bool == correct_bool
        return {
            "score": 1.0 if is_correct else 0.0,
            "correct": is_correct,
            "correct_answer": correct_bool,
            "feedback": question.get("explanation", ""),
        }

    elif qtype == "fill_blank":
        correct = str(question.get("correct_answer", "")).strip().lower()
        variations = [v.strip().lower() for v in question.get("accept_variations", [])]
        all_accepted = [correct] + variations
        user_answer = str(answer).strip().lower()
        is_correct = user_answer in all_accepted
        return {
            "score": 1.0 if is_correct else 0.0,
            "correct": is_correct,
            "correct_answer": question.get("correct_answer", ""),
            "feedback": question.get("explanation", ""),
        }

    return {"score": 0.0, "correct": False, "feedback": "Unknown question type"}


def _grade_subjective(question: dict, answer: str, topic: str) -> dict[str, Any]:
    """Grade a subjective question using grading_engine."""
    if not answer.strip():
        return {
            "score": 0.0,
            "correct": False,
            "feedback": "No answer provided",
            "outcome": "incorrect",
        }

    try:
        from grading_engine import grade as grade_submission

        result = grade_submission(
            submission=answer,
            topic=topic or question.get("concept_id", ""),
            question_text=question.get("text", ""),
            mode="informal",
            concept_context=question.get("rubric_hint", ""),
        )

        # Normalize 0-10 score to 0-1
        raw_score = result.get("overall_score", 0.0)
        normalized = raw_score / 10.0

        return {
            "score": round(normalized, 2),
            "correct": normalized >= 0.8,
            "feedback": result.get("improvement", ""),
            "strength": result.get("strength", ""),
            "outcome": result.get("outcome", "partial"),
            "criteria_scores": result.get("criteria_scores", []),
        }

    except Exception as exc:
        logger.error("Subjective grading failed: %s", exc)
        # Generous fallback: if they wrote something, give partial credit
        return {
            "score": 0.5 if len(answer.strip()) > 20 else 0.0,
            "correct": False,
            "feedback": "Could not grade automatically — please review",
            "outcome": "partial" if len(answer.strip()) > 20 else "incorrect",
        }


def _instructions_for_difficulty(difficulty: str) -> str:
    """Return human-readable test instructions based on difficulty."""
    return {
        "EASY": "Answer the following questions. For multiple choice, select the best option. For true/false, indicate true or false.",
        "MEDIUM": "Complete the following questions. Fill in blanks with the correct term. For short answers, write 1-3 sentences.",
        "HARD": "Answer the following in detail. For explanations, provide thorough reasoning. For coding questions, write working code.",
        "EXTENSION": "These are advanced questions. Demonstrate deep understanding through detailed explanations or well-designed solutions.",
    }.get(difficulty, "Answer all questions to the best of your ability.")


def _fallback_question(topic: str, question_type: str) -> dict[str, Any]:
    """Generate a fallback question when the LLM call fails."""
    if question_type == "multiple_choice":
        return {
            "type": "multiple_choice",
            "text": f"Which of the following best describes {topic}?",
            "options": [
                f"A core concept in {topic}",
                f"An unrelated idea",
                f"A common misconception about {topic}",
                f"None of the above",
            ],
            "correct_index": 0,
            "explanation": f"The first option correctly identifies a core aspect of {topic}.",
            "concept_id": topic.lower().replace(" ", "_"),
        }
    elif question_type == "true_false":
        return {
            "type": "true_false",
            "text": f"{topic} is a fundamental concept in its field.",
            "correct_answer": True,
            "explanation": f"{topic} is indeed a core concept.",
            "concept_id": topic.lower().replace(" ", "_"),
        }
    elif question_type == "fill_blank":
        return {
            "type": "fill_blank",
            "text": f"The key idea behind {topic} is _____.",
            "correct_answer": topic.lower(),
            "accept_variations": [],
            "explanation": f"The blank should be filled with the core concept.",
            "concept_id": topic.lower().replace(" ", "_"),
        }
    else:
        return {
            "type": question_type,
            "text": f"Explain the key concepts of {topic} in your own words.",
            "rubric_hint": f"Should demonstrate understanding of {topic}",
            "concept_id": topic.lower().replace(" ", "_"),
        }
