"""
hint_engine.py — Progressive 3-level hint generation.

Three hint levels (progressive disclosure):
  Level 1 (Nudge):           Guiding question — does NOT reveal the answer.
  Level 2 (Scaffolding):     Concrete conceptual step — names a technique/principle.
  Level 3 (Solution Trace):  High-level walkthrough/pseudocode — still requires student work.

Each level builds on previous hints so the student sees a coherent progression.

Uses OpenAI (gpt-4o-mini) to match the rest of the codebase.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Generator, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

MAX_HINT_LEVEL = 3

# ── System prompts per level ─────────────────────────────────────────────────

_HINT_SYSTEM_L1 = """\
You are a patient tutor giving a Level 1 hint (gentle nudge).

Rules:
- Ask ONE guiding question that points the student in the right direction.
- Do NOT name the answer, technique, or solution directly.
- Do NOT give any part of the solution away.
- Keep it to 1-2 sentences max.
- Be encouraging and warm.

Example: "What data structure lets you look things up in constant time?"
"""

_HINT_SYSTEM_L2 = """\
You are a patient tutor giving a Level 2 hint (scaffolding).

Rules:
- Name the relevant concept, technique, or principle the student should apply.
- Give ONE concrete step or direction they can try.
- Do NOT solve the problem for them or provide final answers.
- Keep it to 2-3 sentences max.
- Reference the student's current attempt if available.

Example: "Try using a hash map to store the values you've already seen. \
That way you can check membership in O(1) for each new element."
"""

_HINT_SYSTEM_L3 = """\
You are a patient tutor giving a Level 3 hint (solution trace).

Rules:
- Break the solution into 3-5 high-level steps (pseudocode or plain English).
- Each step should describe WHAT to do, not give exact code/answers.
- The student still needs to implement/complete the details themselves.
- If the student's attempt has specific errors, point them out.
- Keep it concise — max 5 numbered steps.

Example:
"1. Initialize an empty hash set.
 2. For each element, check if (target - element) is in the set.
 3. If found, return the pair. If not, add the current element.
 4. Handle the edge case of duplicate values."
"""

_HINT_SYSTEMS = {
    1: _HINT_SYSTEM_L1,
    2: _HINT_SYSTEM_L2,
    3: _HINT_SYSTEM_L3,
}


# ── Core functions ───────────────────────────────────────────────────────────

def generate_hint(
    question_text: str,
    level: int,
    topic: str = "",
    student_answer: str = "",
    concept_context: str = "",
    previous_hints: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Generate a single hint at the specified level.

    Parameters
    ----------
    question_text    : the question the student is working on
    level            : hint level (1, 2, or 3)
    topic            : the concept/topic being studied
    student_answer   : the student's current answer attempt (may be empty)
    concept_context  : relevant context from PDF / concept graph
    previous_hints   : hints already shown (for continuity)

    Returns
    -------
    {
      "level": int,
      "hint_text": str,
      "can_request_next": bool,
      "topic": str
    }
    """
    level = max(1, min(MAX_HINT_LEVEL, int(level)))

    system_prompt = _HINT_SYSTEMS[level]

    # Build context for the LLM
    prev_text = ""
    if previous_hints:
        prev_lines = [f"  Level {i+1}: {h}" for i, h in enumerate(previous_hints)]
        prev_text = "\nPrevious hints given:\n" + "\n".join(prev_lines) + "\n"

    user_prompt = f"""Topic: {topic or "(not specified)"}
Question: {question_text}

Student's current answer/attempt:
{student_answer[:1000] if student_answer else "(no attempt yet)"}
{prev_text}
Concept context:
{concept_context[:800] if concept_context else "(none)"}

Generate a Level {level} hint for this student."""

    try:
        client = OpenAI()
        resp = client.chat.completions.create(
            model=_OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.4,
            max_tokens=300,
        )
        hint_text = (resp.choices[0].message.content or "").strip()

        if not hint_text:
            hint_text = _fallback_hint(level, topic)

    except Exception as exc:
        logger.error("Hint generation failed (level %d): %s", level, exc)
        hint_text = _fallback_hint(level, topic)

    return {
        "level": level,
        "hint_text": hint_text,
        "can_request_next": level < MAX_HINT_LEVEL,
        "topic": topic,
    }


def generate_all_hints(
    question_text: str,
    topic: str = "",
    student_answer: str = "",
    concept_context: str = "",
) -> list[dict[str, Any]]:
    """
    Generate all 3 hint levels at once (non-streaming).

    Each level receives the previous hints as context so hints form
    a coherent progression.

    Returns a list of 3 hint dicts, one per level.
    """
    hints: list[dict[str, Any]] = []
    previous: list[str] = []

    for level in range(1, MAX_HINT_LEVEL + 1):
        result = generate_hint(
            question_text=question_text,
            level=level,
            topic=topic,
            student_answer=student_answer,
            concept_context=concept_context,
            previous_hints=previous if previous else None,
        )
        hints.append(result)
        previous.append(result["hint_text"])

    return hints


def stream_hints(
    question_text: str,
    topic: str = "",
    student_answer: str = "",
    concept_context: str = "",
) -> Generator[str, None, None]:
    """
    Generator that yields SSE-formatted hint events for all 3 levels.

    Each yield is a complete SSE "data:" line followed by two newlines.
    The frontend can consume this via EventSource or fetch + ReadableStream.

    Yields:
      data: {"level": 1, "hint_text": "...", "can_request_next": true, "topic": "..."}\n\n
      data: {"level": 2, "hint_text": "...", "can_request_next": true, "topic": "..."}\n\n
      data: {"level": 3, "hint_text": "...", "can_request_next": false, "topic": "..."}\n\n
      data: {"done": true}\n\n
    """
    previous: list[str] = []

    for level in range(1, MAX_HINT_LEVEL + 1):
        result = generate_hint(
            question_text=question_text,
            level=level,
            topic=topic,
            student_answer=student_answer,
            concept_context=concept_context,
            previous_hints=previous if previous else None,
        )
        previous.append(result["hint_text"])
        yield f"data: {json.dumps(result)}\n\n"

    yield f"data: {json.dumps({'done': True})}\n\n"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _fallback_hint(level: int, topic: str) -> str:
    """Provide a generic fallback hint when the LLM call fails."""
    topic_str = topic or "this concept"
    if level == 1:
        return f"Think about the core definition of {topic_str}. What are its key properties?"
    elif level == 2:
        return f"Try breaking {topic_str} into smaller parts. Which specific sub-concept applies here?"
    else:
        return (
            f"Here's a general approach:\n"
            f"1. Identify what {topic_str} requires.\n"
            f"2. Recall the key properties or rules.\n"
            f"3. Apply them step by step to the question.\n"
            f"4. Check your answer against the definition."
        )


def validate_hint_request(data: dict[str, Any]) -> tuple[bool, str]:
    """
    Validate an incoming hint request body.
    Returns (is_valid, error_message).
    """
    question_text = (data.get("question_text") or "").strip()
    if not question_text:
        return False, "question_text is required"

    level = data.get("level", 1)
    try:
        level = int(level)
    except (TypeError, ValueError):
        return False, "level must be an integer"

    if level < 1 or level > MAX_HINT_LEVEL:
        return False, f"level must be between 1 and {MAX_HINT_LEVEL}"

    return True, ""
