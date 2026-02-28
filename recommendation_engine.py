"""
recommendation_engine.py — Cubic power law study recommendations.

Ported from mastery.js (Agastya) and unified with the EMA mastery system
in memory_update.py (Sidharth). This module is the *recommendation layer*;
it reads mastery scores but never writes them. Mastery updates remain in
memory_update.py.

Architecture:
  memory_update.py  → measures mastery  (EMA: m_new = m_old + α(score - m_old))
  recommendation_engine.py → uses mastery to decide what to study next

Cubic power law:
  focusWeight = (1 - mastery)³ + ε

  A subtopic at mastery 0.2 gets ~3.4× more focus than one at 0.5.
  A subtopic at mastery 0.0 gets ~8× more focus than one at 0.5.
  ε prevents fully-mastered topics from getting exactly zero weight.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

EPSILON: float = 0.05
"""Small constant so fully-mastered topics retain minimal recommendation weight."""

MASTERED_THRESHOLD: float = 0.85
"""Default threshold above which a concept is considered 'mastered' and excluded."""

MISCONCEPTION_BOOST: float = 2.0
"""Multiplicative boost for concepts with active misconceptions."""

GOAL_BOOST: float = 1.5
"""Multiplicative boost for concepts aligned to student goals."""

RECENCY_WINDOW_SECONDS: float = 600.0  # 10 minutes
"""If a concept was seen within this window, apply a recency penalty."""

RECENCY_PENALTY: float = 0.5
"""Weight multiplier for recently-seen concepts (within RECENCY_WINDOW_SECONDS)."""

MAX_RECOMMENDATIONS: int = 5
"""Default number of recommendations to return."""


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class ConceptRecommendation:
    """A single study recommendation."""
    concept_id: str
    mastery: float
    difficulty: str           # "easy" | "medium" | "hard"
    focus_pct: float          # 0–100, percentage of total focus
    focus_weight: float       # raw weight before normalization
    reason: str               # human-readable explanation
    misconception_hints: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "concept_id": self.concept_id,
            "mastery": round(self.mastery, 3),
            "difficulty": self.difficulty,
            "focus_pct": round(self.focus_pct, 1),
            "reason": self.reason,
            "misconception_hints": self.misconception_hints,
            "weaknesses": self.weaknesses,
        }


# ── Core functions ────────────────────────────────────────────────────────────

def cubic_focus_weight(mastery: float, epsilon: float = EPSILON) -> float:
    """
    Cubic power law: aggressively prioritize low-mastery concepts.

    Ported from mastery.js:
        focusWeight = (1 - mastery)³ + EPSILON

    Examples:
        mastery=0.0 → weight=1.05
        mastery=0.2 → weight=0.562
        mastery=0.5 → weight=0.175
        mastery=0.8 → weight=0.058
        mastery=1.0 → weight=0.05 (epsilon only)
    """
    return (1.0 - mastery) ** 3 + epsilon


def mastery_to_difficulty(mastery: float) -> str:
    """Map mastery level to recommended question difficulty."""
    if mastery < 0.3:
        return "easy"
    elif mastery < 0.6:
        return "medium"
    else:
        return "hard"


def get_recommendations(
    profile: dict[str, Any],
    *,
    concept_ids: Optional[list[str]] = None,
    exclude: Optional[list[str]] = None,
    mastered_threshold: float = MASTERED_THRESHOLD,
    top_n: int = MAX_RECOMMENDATIONS,
    now: Optional[float] = None,
) -> list[ConceptRecommendation]:
    """
    Generate ranked study recommendations from a learner profile.

    This is the Python equivalent of mastery.js getRecommendations(), enhanced
    with misconception boosting, goal alignment, and recency penalties.

    Parameters
    ----------
    profile : dict
        Learner profile with keys: concepts, misconceptions, goals, preferences.
        Same schema as used by memory_update.py.
    concept_ids : list[str], optional
        Restrict to these concept IDs. If None, uses all concepts in profile.
    exclude : list[str], optional
        Concept IDs to exclude.
    mastered_threshold : float
        Concepts at or above this mastery are excluded (default 0.85).
    top_n : int
        Max number of recommendations (default 5).
    now : float, optional
        Current timestamp (epoch seconds). Defaults to time.time().

    Returns
    -------
    list[ConceptRecommendation]
        Ranked recommendations, highest priority first. Focus percentages sum
        to 100% across the returned list.
    """
    if now is None:
        now = time.time()

    concepts = profile.get("concepts", {})
    misconceptions = profile.get("misconceptions", {})
    goals = profile.get("goals", [])
    exclude_set = set(exclude or [])

    # Build candidate list
    candidates = concept_ids if concept_ids is not None else list(concepts.keys())
    candidates = [c for c in candidates if c not in exclude_set]

    # Build misconception lookup: concept_id → list of misconception details
    misconception_map: dict[str, list[dict]] = {}
    for mid, mdata in misconceptions.items():
        cid = mdata.get("concept_id", "")
        if cid:
            misconception_map.setdefault(cid, []).append({
                "id": mid,
                "count": mdata.get("count", 0),
                "evidence": mdata.get("evidence", []),
            })

    # Build goal concept set (extract concept IDs from goal text or goal objects)
    goal_concepts: set[str] = set()
    for g in goals:
        if isinstance(g, dict):
            gc = g.get("concept", g.get("concept_id", ""))
            if gc:
                goal_concepts.add(gc)
        elif isinstance(g, str):
            # Check if any candidate concept appears in the goal text
            for c in candidates:
                if c.lower() in g.lower():
                    goal_concepts.add(c)

    # Score each candidate
    scored: list[dict[str, Any]] = []

    for cid in candidates:
        entry = concepts.get(cid)
        if entry is None:
            # Unknown concept — treat as zero mastery (new topic)
            mastery = 0.0
            last_seen_str = ""
            weaknesses: list[str] = []
        else:
            mastery = entry.get("mastery", 0.0)
            last_seen_str = entry.get("last_seen", "")
            weaknesses = entry.get("weaknesses", [])

        # Skip mastered concepts
        if mastery >= mastered_threshold:
            continue

        # Base weight: cubic power law
        weight = cubic_focus_weight(mastery)

        # Misconception boost
        concept_misconceptions = misconception_map.get(cid, [])
        has_misconceptions = len(concept_misconceptions) > 0
        if has_misconceptions:
            weight *= (1.0 + MISCONCEPTION_BOOST)

        # Goal boost
        is_goal_aligned = cid in goal_concepts
        if is_goal_aligned:
            weight *= GOAL_BOOST

        # Recency penalty
        recency_applied = False
        if last_seen_str:
            try:
                from datetime import datetime, timezone
                last_seen_dt = datetime.fromisoformat(last_seen_str)
                last_seen_epoch = last_seen_dt.timestamp()
                if (now - last_seen_epoch) < RECENCY_WINDOW_SECONDS:
                    weight *= RECENCY_PENALTY
                    recency_applied = True
            except (ValueError, TypeError):
                pass  # Can't parse timestamp — skip recency

        # Build reason string
        reasons = []
        if mastery < 0.3:
            reasons.append("low mastery")
        elif mastery < 0.6:
            reasons.append("moderate mastery")
        if has_misconceptions:
            misconception_names = [m["id"] for m in concept_misconceptions[:3]]
            reasons.append(f"active misconception(s): {', '.join(misconception_names)}")
        if is_goal_aligned:
            reasons.append("aligned to learning goal")
        if recency_applied:
            reasons.append("recently seen (deprioritized)")
        if not reasons:
            reasons.append("below mastery threshold")

        # Misconception hints for the recommendation
        hints = []
        for m in concept_misconceptions[:3]:
            evidence = m.get("evidence", [])
            if evidence:
                hints.append(f"Review: {evidence[-1]}")
            else:
                hints.append(f"Address misconception: {m['id']}")

        scored.append({
            "concept_id": cid,
            "mastery": mastery,
            "weight": weight,
            "difficulty": mastery_to_difficulty(mastery),
            "reason": "; ".join(reasons),
            "misconception_hints": hints,
            "weaknesses": weaknesses if isinstance(weaknesses, list) else [],
        })

    # Sort by weight descending (highest priority first)
    scored.sort(key=lambda x: -x["weight"])

    # Take top N
    scored = scored[:top_n]

    if not scored:
        return []

    # Normalize to percentages
    total_weight = sum(s["weight"] for s in scored)

    results: list[ConceptRecommendation] = []
    for s in scored:
        pct = (s["weight"] / total_weight * 100.0) if total_weight > 0 else 0.0
        results.append(ConceptRecommendation(
            concept_id=s["concept_id"],
            mastery=s["mastery"],
            difficulty=s["difficulty"],
            focus_pct=pct,
            focus_weight=s["weight"],
            reason=s["reason"],
            misconception_hints=s["misconception_hints"],
            weaknesses=s["weaknesses"],
        ))

    return results


def get_recommendations_dict(
    profile: dict[str, Any],
    **kwargs,
) -> dict[str, Any]:
    """
    Convenience wrapper that returns JSON-serializable dict.

    Output format matches the /recommend API contract:
    {
      "recommendations": [...],
      "all_mastered": false
    }
    """
    recs = get_recommendations(profile, **kwargs)

    if not recs:
        # Check if it's because everything is mastered
        concepts = profile.get("concepts", {})
        threshold = kwargs.get("mastered_threshold", MASTERED_THRESHOLD)
        all_mastered = all(
            v.get("mastery", 0.0) >= threshold
            for v in concepts.values()
        ) if concepts else False

        return {
            "recommendations": [],
            "all_mastered": all_mastered,
        }

    return {
        "recommendations": [r.to_dict() for r in recs],
        "all_mastered": False,
    }


# ── Integration helper: plug into /chat/turn pipeline ─────────────────────────

def recommend_next_for_chat_turn(
    profile: dict[str, Any],
    current_concept: Optional[str] = None,
    top_n: int = 3,
) -> dict[str, Any]:
    """
    Recommend next concepts after a /chat/turn interaction.

    If current_concept is provided, it's excluded from recommendations
    (student just worked on it — suggest something different).

    Returns a dict suitable for embedding in the /chat/turn response:
    {
      "next_concepts": [...],
      "next_template": "lesson" | "hint" | "reflection"
    }
    """
    exclude = [current_concept] if current_concept else None
    recs = get_recommendations(profile, exclude=exclude, top_n=top_n)

    if not recs:
        return {
            "next_concepts": [],
            "next_template": "reflection",
        }

    # Determine next template based on top recommendation
    top = recs[0]
    if top.misconception_hints:
        next_template = "hint"
    elif top.mastery < 0.3:
        next_template = "lesson"
    else:
        next_template = "lesson"

    return {
        "next_concepts": [r.to_dict() for r in recs],
        "next_template": next_template,
    }
