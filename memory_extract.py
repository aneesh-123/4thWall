"""
memory_extract.py — LLM-powered extraction of learner state from a chat turn.

Outputs a strictly-validated JSON object describing concepts touched,
outcome, misconceptions, preference signals, and goals.

Schema:
{
  "concepts": [{"concept_id": "str", "confidence": 0-1}],
  "outcome": {"label": "correct|partial|confused|incorrect",
               "confidence": 0-1, "inferred": true|false},
  "misconceptions": [{"misconception_id": "str", "concept_id": "str",
                       "evidence": "str", "confidence": 0-1}],
  "preferences": [{"key": "style|verbosity|pace", "value": "str",
                    "confidence": 0-1}],
  "goals": [{"goal": "str", "confidence": 0-1}],
  "notes": "str optional"
}
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)

_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ── Schema validation ─────────────────────────────────────────────────────────

_OUTCOME_LABELS = {"correct", "partial", "confused", "incorrect"}
_PREF_KEYS      = {"style", "verbosity", "pace"}
_STYLE_VALS     = {"worked_examples", "analogy", "socratic", "direct"}
_VERBOSITY_VALS = {"low", "medium", "high"}
_PACE_VALS      = {"slow", "medium", "fast"}


def _clamp(val: Any, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        return max(lo, min(hi, float(val)))
    except (TypeError, ValueError):
        return 0.5


def validate_extract(data: Any) -> dict[str, Any]:
    """
    Validate and normalise the extractor JSON. Raises ValueError with a
    human-readable message on any structural violation.
    """
    if not isinstance(data, dict):
        raise ValueError("Root must be a JSON object")

    # ── concepts ──────────────────────────────────────────────────────────────
    concepts = []
    for c in data.get("concepts", []):
        cid = str(c.get("concept_id", "")).strip()
        if not cid:
            continue
        concepts.append({"concept_id": cid, "confidence": _clamp(c.get("confidence", 0.5))})

    # ── outcome ───────────────────────────────────────────────────────────────
    oc = data.get("outcome", {})
    label = str(oc.get("label", "partial")).lower()
    if label not in _OUTCOME_LABELS:
        label = "partial"
    outcome = {
        "label":      label,
        "confidence": _clamp(oc.get("confidence", 0.5)),
        "inferred":   bool(oc.get("inferred", True)),
    }

    # ── misconceptions ────────────────────────────────────────────────────────
    misconceptions = []
    for m in data.get("misconceptions", []):
        mid = str(m.get("misconception_id", "")).strip()
        cid = str(m.get("concept_id", "")).strip()
        evidence = str(m.get("evidence", "")).strip()[:300]
        conf = _clamp(m.get("confidence", 0.5))
        if mid and cid:
            misconceptions.append({
                "misconception_id": mid,
                "concept_id": cid,
                "evidence": evidence,
                "confidence": conf,
            })

    # ── preferences ───────────────────────────────────────────────────────────
    preferences = []
    for p in data.get("preferences", []):
        key = str(p.get("key", "")).lower().strip()
        val = str(p.get("value", "")).lower().strip()
        conf = _clamp(p.get("confidence", 0.5))
        if key not in _PREF_KEYS:
            continue
        # loose validation on values
        valid_vals = {"style": _STYLE_VALS, "verbosity": _VERBOSITY_VALS, "pace": _PACE_VALS}[key]
        if val not in valid_vals:
            continue
        preferences.append({"key": key, "value": val, "confidence": conf})

    # ── goals ─────────────────────────────────────────────────────────────────
    goals = []
    for g in data.get("goals", []):
        goal_text = str(g.get("goal", "")).strip()[:200]
        conf = _clamp(g.get("confidence", 0.5))
        if goal_text:
            goals.append({"goal": goal_text, "confidence": conf})

    notes = str(data.get("notes", "")).strip()[:300]

    return {
        "concepts":       concepts,
        "outcome":        outcome,
        "misconceptions": misconceptions,
        "preferences":    preferences,
        "goals":          goals,
        "notes":          notes,
    }


# ── Keyword-heuristic fallback ────────────────────────────────────────────────

_OUTCOME_SIGNAL_WORDS = {
    "correct":   ["correct", "right", "exactly", "precisely", "yes", "got it", "understand now"],
    "partial":   ["kind of", "sort of", "partially", "almost", "close", "i think"],
    "confused":  ["confused", "don't understand", "what do you mean", "lost", "unclear", "not sure"],
    "incorrect": ["wrong", "no", "incorrect", "that's not", "mistaken"],
}


def _heuristic_fallback(user_msg: str, tutor_msg: str) -> dict[str, Any]:
    """
    Very minimal keyword-based fallback used when the LLM extractor fails twice.
    """
    combined = (user_msg + " " + tutor_msg).lower()

    # Detect outcome
    outcome_label = "partial"
    for label, words in _OUTCOME_SIGNAL_WORDS.items():
        if any(w in combined for w in words):
            outcome_label = label
            break

    # Extract naive concept IDs from words ≥ 5 chars that aren't stopwords
    _STOPWORDS = {
        "would", "could", "should", "about", "which", "where", "there",
        "their", "these", "those", "being", "because", "explain", "please",
        "question", "answer", "think", "means", "understand",
    }
    words = re.findall(r"\b[a-z]{5,}\b", combined)
    freq: dict[str, int] = {}
    for w in words:
        if w not in _STOPWORDS:
            freq[w] = freq.get(w, 0) + 1
    top_concepts = [{"concept_id": w, "confidence": 0.3}
                    for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:3]]

    return {
        "concepts":       top_concepts,
        "outcome":        {"label": outcome_label, "confidence": 0.4, "inferred": True},
        "misconceptions": [],
        "preferences":    [],
        "goals":          [],
        "notes":          "heuristic fallback",
    }


# ── LLM extraction ────────────────────────────────────────────────────────────

_EXTRACT_SYSTEM = """\
You are a learning analytics model. Analyse a single tutor–student exchange and
return ONLY a JSON object (no markdown, no code fences) conforming exactly to
this schema:

{
  "concepts": [{"concept_id": "<snake_case_label>", "confidence": 0.0-1.0}],
  "outcome": {"label": "correct|partial|confused|incorrect",
               "confidence": 0.0-1.0, "inferred": true|false},
  "misconceptions": [{"misconception_id": "<snake_case_label>",
                       "concept_id": "<snake_case_label>",
                       "evidence": "<short quote ≤150 chars>",
                       "confidence": 0.0-1.0}],
  "preferences": [{"key": "style|verbosity|pace",
                    "value": "worked_examples|analogy|socratic|direct|low|medium|high|slow|fast",
                    "confidence": 0.0-1.0}],
  "goals": [{"goal": "<string ≤150 chars>", "confidence": 0.0-1.0}],
  "notes": "<optional string ≤200 chars>"
}

Rules:
- concept_id and misconception_id MUST be snake_case (e.g. "binary_search").
- outcome.inferred=false ONLY when the student explicitly states right/wrong.
- Include a preference signal ONLY when the student EXPLICITLY asks for a
  different style/speed (e.g. "can you give me an example?" → style=worked_examples).
- Keep lists short: max 5 concepts, 3 misconceptions, 2 preferences, 3 goals.
- Return valid JSON only. No extra keys.
"""

_REPAIR_SUFFIX = (
    "\n\nThe JSON you returned was invalid. Return ONLY the corrected JSON object, "
    "nothing else; no markdown, no code fences."
)


def extract_turn_insights(
    user_message: str,
    tutor_message: str,
    profile_context: str = "",
    outcome_signal: str | None = None,
    allowed_concept_ids: list[str] | None = None,
) -> dict[str, Any]:
    """
    Call the LLM to extract structured learning signals from one turn.

    Parameters
    ----------
    user_message         : the learner's raw message
    tutor_message        : the tutor's response
    profile_context      : compact string summarising learner state (injected as context)
    outcome_signal       : if the caller already knows the outcome, pass it here
                           (sets inferred=False and skips outcome signals in prompt)
    allowed_concept_ids  : when provided, the LLM MUST choose concept_ids only from
                           this list; prevents invented IDs drifting from PDF slugs.

    Returns a validated dict; falls back to heuristics on LLM/parse failure.
    """
    client = OpenAI()

    # Build concept constraint clause
    concept_constraint = ""
    if allowed_concept_ids:
        cid_list = ", ".join(allowed_concept_ids[:60])  # cap to avoid huge prompts
        concept_constraint = (
            f"\nIMPORTANT: The only valid concept_id values are:\n  {cid_list}\n"
            "Choose concept_ids ONLY from this list. Do not invent new ones.\n"
        )

    # Build user prompt
    outcome_hint = (
        f"\nNote: the explicit outcome signal from the learner is '{outcome_signal}'.\n"
        if outcome_signal else ""
    )
    user_prompt = f"""{outcome_hint}{concept_constraint}
LEARNER CONTEXT (use this to recognise concept names):
{profile_context or "(no prior context)"}

USER MESSAGE:
{user_message[:1000]}

TUTOR MESSAGE:
{tutor_message[:1000]}

Extract the learning signals as described. Return JSON only."""

    raw = ""
    for attempt in range(2):
        try:
            prompt_content = user_prompt if attempt == 0 else (user_prompt + _REPAIR_SUFFIX + f"\n\nYour previous broken output:\n{raw}")
            resp = client.chat.completions.create(
                model=_OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": _EXTRACT_SYSTEM},
                    {"role": "user",   "content": prompt_content},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or ""
            parsed = json.loads(raw)
            validated = validate_extract(parsed)

            # Override outcome if caller supplied an explicit signal
            if outcome_signal and outcome_signal in _OUTCOME_LABELS:
                validated["outcome"]["label"]    = outcome_signal
                validated["outcome"]["inferred"] = False

            logger.debug("extract_turn_insights OK on attempt %d", attempt + 1)
            return validated

        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("extract_turn_insights attempt %d failed: %s", attempt + 1, exc)
        except Exception as exc:
            logger.error("extract_turn_insights LLM call failed: %s", exc)
            break

    logger.warning("extract_turn_insights falling back to heuristics")
    fallback = _heuristic_fallback(user_message, tutor_message)
    if outcome_signal and outcome_signal in _OUTCOME_LABELS:
        fallback["outcome"]["label"]    = outcome_signal
        fallback["outcome"]["inferred"] = False
    return fallback


def build_profile_context_string(profile: dict[str, Any]) -> str:
    """
    Build a short text snippet of the current profile suitable for injection
    into extraction and tutor prompts.
    """
    lines: list[str] = []

    prefs = profile.get("preferences", {})
    if prefs:
        lines.append(f"Preferences: style={prefs.get('style','?')}, "
                     f"verbosity={prefs.get('verbosity','?')}, "
                     f"pace={prefs.get('pace','?')}")

    goals = profile.get("goals", [])
    if goals:
        lines.append(f"Goals: {'; '.join(goals[:3])}")

    concepts = profile.get("concepts", {})
    weak = sorted(
        [(cid, c["mastery"]) for cid, c in concepts.items() if c["mastery"] < 0.5],
        key=lambda x: x[1]
    )[:5]
    if weak:
        lines.append("Weak concepts: " + ", ".join(f"{c}({m:.2f})" for c, m in weak))

    strong = sorted(
        [(cid, c["mastery"]) for cid, c in concepts.items() if c["mastery"] >= 0.5],
        key=lambda x: -x[1]
    )[:3]
    if strong:
        lines.append("Strong concepts: " + ", ".join(f"{c}({m:.2f})" for c, m in strong))

    misconceptions = profile.get("misconceptions", {})
    top_misc = sorted(
        [(mid, m["count"]) for mid, m in misconceptions.items()],
        key=lambda x: -x[1]
    )[:3]
    if top_misc:
        lines.append("Known misconceptions: " + ", ".join(f"{m}(x{n})" for m, n in top_misc))

    recent = profile.get("recent_summary", "")
    if recent:
        lines.append(f"Recent: {recent}")

    return "\n".join(lines)
